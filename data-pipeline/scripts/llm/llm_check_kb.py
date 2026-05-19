"""LLM cross-validation для KB-метченых строк (с vision)

KB фундаментально text-only - не видит фото. Поэтому когда:
- на фото лот из нескольких камер
- на фото только чехол/коробка
- на фото цифровая камера хотя в title плёночная модель
- title бренд + модель но фото показывает что-то другое

KB ошибочно ставит метку камеры. Этот скрипт прогоняет vision LLM на все
55k KB-метченые строки чтобы поймать такие false positives.

Output: llm_kb_check.parquet с теми же колонками что llm_labels.parquet.
В decision.py: если LLM-vision проверка говорит "не валидная камера" с conf >= 0.85,
KB-метка демоутится на other_unknown
"""

import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from progress import atomic_write_parquet
from schema import LabelResponse, is_business_valid

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ITEMS = PROJECT_ROOT / 'data' / 'labeling' / 'items.parquet'
KB_LABELS = PROJECT_ROOT / 'data' / 'labeling' / 'kb_labels.parquet'
IMG_DIR = PROJECT_ROOT / 'data' / 'raw' / 'image'
PROMPT = Path(__file__).parent / 'prompts' / 'v3_with_vision.md'
DST = PROJECT_ROOT / 'data' / 'labeling' / 'llm_kb_check.parquet'

MODEL = 'gemini-3.1-flash-lite'
SERVICE_TIER = 'flex'
CACHE_TTL = '93600s'

MAX_WORKERS = 25
SAVE_EVERY = 200
RETRIES = 5

PRICE_IN = 0.125 / 1e6
PRICE_CACHED = 0.0125 / 1e6
PRICE_OUT = 0.75 / 1e6
PRICE_CACHE_WRITE = 0.25 / 1e6
MAX_USD = 20.0

BACKOFF_BASE = 10
BACKOFF_MULT = 3

COLS = ['item_id', 'raw', 'parsed_ok', 'biz_ok', 'biz_reason',
        'pred_status', 'pred_body', 'pred_label', 'confidence', 'error_msg']

_lock = threading.Lock()


def load_todo():
    items = pd.read_parquet(ITEMS)
    kb = pd.read_parquet(KB_LABELS)
    # только строки где KB поставил метку - их проверяем LLM-vision
    kb_matched = kb[kb['kb_label'].notna()]
    df = items.merge(kb_matched[['item_id', 'kb_label']], on='item_id')
    return df[['item_id', 'title', 'description', 'image_id', 'kb_label']].reset_index(drop=True)


def parse(raw):
    try:
        return LabelResponse.model_validate_json(raw), None
    except Exception as e:
        return None, str(e)[:200]


def call(client, cache_name, contents):
    for attempt in range(RETRIES):
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
                    service_tier=SERVICE_TIER,
                    http_options=types.HttpOptions(timeout=900_000),
                ),
            )
        except errors.APIError:
            if attempt == RETRIES - 1:
                raise
            wait = BACKOFF_BASE * (BACKOFF_MULT ** attempt) * random.uniform(0.7, 1.3)
            time.sleep(wait)


def process_row(client, cache_name, dynamic_template, item_id, title, description, image_id):
    row = {
        'item_id': item_id, 'raw': None,
        'parsed_ok': False, 'biz_ok': False, 'biz_reason': None,
        'pred_status': None, 'pred_body': None, 'pred_label': None, 'confidence': None,
        'error_msg': None,
    }
    path = IMG_DIR / str(image_id % 1000) / f'{image_id}.jpg'
    if not path.exists():
        row['error_msg'] = 'image_not_found'
        return row, 0, 0, 0
    try:
        user_msg = dynamic_template.replace('{TITLE}', title or '').replace('{DESCRIPTION}', description or '')
        resp = call(client, cache_name, [
            types.Part.from_bytes(data=path.read_bytes(), mime_type='image/jpeg'),
            user_msg,
        ])
        cached_n = resp.usage_metadata.cached_content_token_count or 0
        in_n = (resp.usage_metadata.prompt_token_count or 0) - cached_n
        out_n = resp.usage_metadata.candidates_token_count or 0

        raw = resp.text or ''
        parsed, err = parse(raw)
        row['raw'] = raw
        row['parsed_ok'] = parsed is not None
        if parsed:
            biz_ok, biz_reason = is_business_valid(parsed)
            row.update({
                'biz_ok': biz_ok, 'biz_reason': biz_reason,
                'pred_status': parsed.object_status,
                'pred_body': parsed.body_type,
                'pred_label': parsed.final_label,
                'confidence': parsed.confidence,
            })
        else:
            row['biz_reason'] = err
        return row, cached_n, in_n, out_n
    except Exception as e:
        row['error_msg'] = str(e)[:200]
        return row, 0, 0, 0


def main():
    load_dotenv(PROJECT_ROOT / '.env')
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    todo = load_todo()
    if DST.exists():
        acc = pd.read_parquet(DST)
        done_ids = set(acc[acc['parsed_ok'] & acc['biz_ok']]['item_id'])
    else:
        acc = pd.DataFrame(columns=COLS)
        done_ids = set()
    todo = todo[~todo['item_id'].isin(done_ids)]
    print(f'KB-меченых всего: {len(todo) + len(done_ids)}, готово: {len(done_ids)}, todo: {len(todo)}')
    if len(todo) == 0:
        print('всё проверено, выхожу')
        return

    template = PROMPT.read_text()
    static_part, dynamic_template = template.split('# ОБЪЯВЛЕНИЕ ДЛЯ РАЗМЕТКИ')

    cache = client.caches.create(
        model=MODEL,
        config=types.CreateCachedContentConfig(
            system_instruction=static_part.strip(),
            ttl=CACHE_TTL,
        ),
    )
    cached_tokens = cache.usage_metadata.total_token_count
    cache_write_cost = cached_tokens * PRICE_CACHE_WRITE
    print(f'cache: {cache.name}, {cached_tokens} токенов, write ${cache_write_cost:.4f}')
    print(f'workers: {MAX_WORKERS}, service_tier: {SERVICE_TIER}, max_usd: ${MAX_USD}')

    pending = []
    total = {'cached': 0, 'in': 0, 'out': 0}

    def flush():
        nonlocal acc
        if not pending:
            return
        with _lock:
            combined = pd.concat([acc, pd.DataFrame(pending)], ignore_index=True)
            combined = combined.drop_duplicates(subset='item_id', keep='last')
            atomic_write_parquet(combined, DST)
            acc = combined
            pending.clear()

    def cost():
        return cache_write_cost + total['cached'] * PRICE_CACHED + total['in'] * PRICE_IN + total['out'] * PRICE_OUT

    ex = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    cancelled = False
    try:
        futures = {
            ex.submit(process_row, client, cache.name, dynamic_template,
                      r.item_id, r.title, r.description, int(r.image_id)): r
            for r in todo.itertuples(index=False)
        }
        bar = tqdm(as_completed(futures), total=len(futures), desc='kb-check', smoothing=0.05)
        for fut in bar:
            row, c, i, o = fut.result()
            total['cached'] += c
            total['in'] += i
            total['out'] += o
            pending.append(row)
            bar.set_postfix(cost=f'${cost():.2f}', biz_ok=f'{sum(1 for r in pending if r["biz_ok"]) + len(acc)}')
            if len(pending) >= SAVE_EVERY:
                flush()
            if cost() > MAX_USD:
                print(f'\n!! cost guard {MAX_USD} превышен')
                cancelled = True
                break
    except KeyboardInterrupt:
        print('\n[Ctrl+C] сохраняю прогресс...')
        cancelled = True
    finally:
        if cancelled:
            ex.shutdown(wait=False, cancel_futures=True)
        else:
            ex.shutdown(wait=True)
        flush()
        try:
            client.caches.delete(name=cache.name)
        except Exception:
            pass

    final = pd.read_parquet(DST)
    parsed = final['parsed_ok'].sum()
    biz = final['biz_ok'].sum()
    print(f'\nитого {len(final)}, parsed_ok: {parsed} ({parsed/len(final):.1%}), biz_ok: {biz} ({biz/len(final):.1%})')
    print(f'cost: ${cost():.4f}, токены: cached={total["cached"]:,} in={total["in"]:,} out={total["out"]:,}')

    # сколько KB-меток будет демоутиться
    not_valid = final[(final['pred_status'].notna()) & (final['pred_status'] != 'valid_single_film_camera') & (final['confidence'] >= 0.85)]
    print(f'\nпредварительный анализ:')
    print(f'  LLM-vision сказал "не валидная камера" с conf >= 0.85: {len(not_valid)} строк ({len(not_valid)/len(final)*100:.1f}%)')
    print(f'  эти KB-метки будут демоутиться в other_unknown')


if __name__ == '__main__':
    main()
