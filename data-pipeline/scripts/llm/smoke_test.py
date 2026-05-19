"""Smoke test промпта на 50 строках из gold_dev с context caching

Использование: python smoke_test.py <prompt_name>  (по умолчанию v1_minimal)
Промпт разделяется на static (до "# ОБЪЯВЛЕНИЕ ДЛЯ РАЗМЕТКИ") и dynamic
Static кешируется на час, dynamic шлётся per-request с заменой {TITLE}/{DESCRIPTION}
Замеряет: parsed_ok, biz_ok, accuracy vs gold, confusion matrix, ошибки
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

sys.path.insert(0, str(Path(__file__).parent))
from schema import LabelResponse, is_business_valid

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLD_DEV = PROJECT_ROOT / 'data' / 'labeling' / 'gold_dev.parquet'
IMG_DIR = PROJECT_ROOT / 'data' / 'raw' / 'image'

PROMPT_NAME = sys.argv[1] if len(sys.argv) > 1 else 'v1_minimal'
PROMPT = Path(__file__).parent / 'prompts' / f'{PROMPT_NAME}.md'
OUT = PROJECT_ROOT / 'data' / 'labeling' / f'llm_smoke_{PROMPT_NAME}.parquet'
USE_IMAGES = 'vision' in PROMPT_NAME

MODEL = 'gemini-3.1-flash-lite'
N = 50
SEED = 42
SLEEP = 1.5
CACHE_TTL = '3600s'

# Gemini 3.1 Flash-Lite standard tier
PRICE_IN = 0.25 / 1e6
PRICE_CACHED = 0.025 / 1e6
PRICE_OUT = 1.50 / 1e6
MAX_USD = 0.10


def call(client, cache_name, contents, retries=5):
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    cached_content=cache_name,
                    response_mime_type='application/json',
                    response_schema=LabelResponse,
                    temperature=0,
                    max_output_tokens=200,
                ),
            )
        except (errors.ServerError, errors.APIError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f' [retry {attempt+1}/{retries} after {wait}s: {type(e).__name__}]', end='', flush=True)
            time.sleep(wait)


def image_part(image_id):
    p = IMG_DIR / str(image_id % 1000) / f'{image_id}.jpg'
    return types.Part.from_bytes(data=p.read_bytes(), mime_type='image/jpeg')


def parse(raw):
    try:
        return LabelResponse.model_validate_json(raw), None
    except Exception as e:
        return None, str(e)


def main():
    load_dotenv(PROJECT_ROOT / '.env')
    if 'GEMINI_API_KEY' not in os.environ:
        sys.exit('GEMINI_API_KEY не задан в .env')
    if not PROMPT.exists():
        sys.exit(f'промпт не найден: {PROMPT}')

    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    template = PROMPT.read_text()
    marker = '# ОБЪЯВЛЕНИЕ ДЛЯ РАЗМЕТКИ'
    if marker not in template:
        sys.exit(f'в промпте нет маркера "{marker}" - не могу разделить')
    static_part, dynamic_template = template.split(marker)

    print(f'промпт: {PROMPT_NAME}')
    cache = client.caches.create(
        model=MODEL,
        config=types.CreateCachedContentConfig(
            system_instruction=static_part.strip(),
            ttl=CACHE_TTL,
        ),
    )
    cached_n = cache.usage_metadata.total_token_count
    write_cost = cached_n * PRICE_IN
    print(f'кеш: {cache.name}, {cached_n} токенов, запись ${write_cost:.4f}')

    try:
        df = pd.read_parquet(GOLD_DEV).sample(N, random_state=SEED).reset_index(drop=True)
        rows = []
        cached_t = normal_t = out_t = 0

        for i, r in df.iterrows():
            user_msg = (dynamic_template
                        .replace('{TITLE}', r['title'] or '')
                        .replace('{DESCRIPTION}', r['description'] or ''))
            contents = [image_part(int(r['image_id'])), user_msg] if USE_IMAGES else user_msg
            resp = call(client, cache.name, contents)

            cur_cached = resp.usage_metadata.cached_content_token_count or 0
            cur_in = (resp.usage_metadata.prompt_token_count or 0) - cur_cached
            cur_out = resp.usage_metadata.candidates_token_count or 0
            cached_t += cur_cached
            normal_t += cur_in
            out_t += cur_out

            raw = resp.text or ''
            parsed, err = parse(raw)
            biz_ok, biz_reason = is_business_valid(parsed) if parsed else (False, err)
            rows.append({
                'item_id': r['item_id'],
                'title': r['title'],
                'gold': r['final_label'],
                'raw': raw,
                'parsed_ok': parsed is not None,
                'biz_ok': biz_ok,
                'biz_reason': biz_reason,
                'pred_status': parsed.object_status if parsed else None,
                'pred_body': parsed.body_type if parsed else None,
                'pred_label': parsed.final_label if parsed else None,
                'confidence': parsed.confidence if parsed else None,
            })

            cost = write_cost + cached_t * PRICE_CACHED + normal_t * PRICE_IN + out_t * PRICE_OUT
            print(f'{i+1}/{N}  parsed={parsed is not None}  biz={biz_ok}  ${cost:.4f}', end='\r')
            if cost > MAX_USD:
                print('\n!! cost guard превышен')
                break
            time.sleep(SLEEP)

        out = pd.DataFrame(rows)
        out.to_parquet(OUT)
        print()
        print(f'\nстрок: {len(out)}, сохранил {OUT}')
        print(f'parsed_ok: {out["parsed_ok"].mean():.1%}, biz_ok: {out["biz_ok"].mean():.1%}')
        cost = write_cost + cached_t * PRICE_CACHED + normal_t * PRICE_IN + out_t * PRICE_OUT
        print(f'токены: cached={cached_t} non-cached={normal_t} out={out_t}')
        print(f'cost: ${cost:.4f}  (write ${write_cost:.4f} + hits ${cost - write_cost:.4f})')

        df_ok = out.dropna(subset=['pred_label']).copy()
        df_ok['correct'] = df_ok['pred_label'] == df_ok['gold']
        print(f'\naccuracy: {df_ok["correct"].mean():.1%} ({df_ok["correct"].sum()}/{len(df_ok)})')

        print('\nconfusion matrix (rows=gold, cols=pred):')
        print(pd.crosstab(df_ok['gold'], df_ok['pred_label'], margins=True))

        print('\nprecision/recall по классам:')
        labels = sorted(set(df_ok['gold']) | set(df_ok['pred_label']))
        for lbl in labels:
            tp = ((df_ok['gold'] == lbl) & (df_ok['pred_label'] == lbl)).sum()
            n_pred = (df_ok['pred_label'] == lbl).sum()
            n_gold = (df_ok['gold'] == lbl).sum()
            p = f'{tp}/{n_pred}={tp/n_pred*100:.0f}%' if n_pred else '-'
            rc = f'{tp}/{n_gold}={tp/n_gold*100:.0f}%' if n_gold else '-'
            print(f'  {lbl:30s}  P={p:15s}  R={rc}')

        wrong = df_ok[~df_ok['correct']]
        if len(wrong):
            print(f'\nошибок ({len(wrong)}):')
            for _, x in wrong.iterrows():
                print(f'  [{x["gold"]}] → [{x["pred_label"]}]  conf={x["confidence"]:.2f}')
                print(f'    {x["title"][:70]}')
    finally:
        try:
            client.caches.delete(name=cache.name)
            print(f'\nкеш {cache.name} удалён')
        except Exception as e:
            print(f'\nне смог удалить кеш: {e}')


if __name__ == '__main__':
    main()
