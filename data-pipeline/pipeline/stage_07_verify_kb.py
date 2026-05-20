"""LLM-проверка по фото для строк где справочник kb уже поставил метку

Справочник работает только по тексту, фото он не видит. Поэтому если:
- на фото лот из нескольких камер
- на фото только чехол или коробка
- в title плёночная модель, а на фото цифровая
- title бренд + модель но на фото что-то другое

kb всё равно ставит метку, и эта метка ошибочная. Прогоняем vision LLM по всем
55k объявлений с kb-меткой чтобы поймать такие false positives и в decision.py
демоутить их в other_unknown если LLM уверенно говорит 'это не плёночная камера'

Параметры и логика идентичны stage_06: 25 потоков, Flex tier, кеш на 26 часов,
atomic save каждые 200, retry с бэкоффом, идемпотентный рестарт
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import gemini
from lib.io import atomic_write_parquet
from lib.paths import (ENV_FILE, IMG_DIR, ITEMS, KB_LABELS, LLM_KB_CHECK,
                       PROMPT_ACTIVE)
from lib.schema import is_business_valid

SERVICE_TIER = 'flex'
CACHE_TTL = '93600s'

MAX_WORKERS = 25
SAVE_EVERY = 200
MAX_USD = 20.0

COLS = ['item_id', 'raw', 'parsed_ok', 'biz_ok', 'biz_reason',
        'pred_status', 'pred_body', 'pred_label', 'confidence', 'error_msg']

_lock = threading.Lock()


def load_todo():
    """Берём только строки где kb поставил метку - именно их и проверяем по фото"""
    items = pd.read_parquet(ITEMS)
    kb = pd.read_parquet(KB_LABELS)
    kb_matched = kb[kb['kb_label'].notna()]
    df = items.merge(kb_matched[['item_id', 'kb_label']], on='item_id')
    return df[['item_id', 'title', 'description', 'image_id', 'kb_label']].reset_index(drop=True)


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
        user_msg = gemini.fill_prompt(dynamic_template, title, description)
        resp = gemini.generate(client, cache_name, [
            types.Part.from_bytes(data=path.read_bytes(), mime_type='image/jpeg'),
            user_msg,
        ], service_tier=SERVICE_TIER)
        cached_n = resp.usage_metadata.cached_content_token_count or 0
        in_n = (resp.usage_metadata.prompt_token_count or 0) - cached_n
        out_n = resp.usage_metadata.candidates_token_count or 0

        raw = resp.text or ''
        parsed, err = gemini.parse_response(raw)
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
    load_dotenv(ENV_FILE)
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    todo = load_todo()
    if LLM_KB_CHECK.exists():
        acc = pd.read_parquet(LLM_KB_CHECK)
        done_ids = set(acc[acc['parsed_ok'] & acc['biz_ok']]['item_id'])
    else:
        acc = pd.DataFrame(columns=COLS)
        done_ids = set()
    todo = todo[~todo['item_id'].isin(done_ids)]
    print(f'kb-меченых всего: {len(todo) + len(done_ids)}, готово: {len(done_ids)}, todo: {len(todo)}')
    if len(todo) == 0:
        print('всё проверено, выхожу')
        return

    template = PROMPT_ACTIVE.read_text()
    static_part, dynamic_template = gemini.split_prompt(template)

    cache = client.caches.create(
        model=gemini.MODEL,
        config=types.CreateCachedContentConfig(
            system_instruction=static_part,
            ttl=CACHE_TTL,
        ),
    )
    cached_tokens = cache.usage_metadata.total_token_count
    cache_write_cost = cached_tokens * gemini.PRICE_CACHE_WRITE
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
            atomic_write_parquet(combined, LLM_KB_CHECK)
            acc = combined
            pending.clear()

    def current_cost():
        return (cache_write_cost
                + total['cached'] * gemini.PRICE_CACHED_FLEX
                + total['in'] * gemini.PRICE_IN_FLEX
                + total['out'] * gemini.PRICE_OUT_FLEX)

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
            bar.set_postfix(
                cost=f'${current_cost():.2f}',
                biz_ok=f'{sum(1 for r in pending if r["biz_ok"]) + len(acc)}',
            )
            if len(pending) >= SAVE_EVERY:
                flush()
            if current_cost() > MAX_USD:
                print(f'\ncost guard {MAX_USD} превышен')
                cancelled = True
                break
    except KeyboardInterrupt:
        print('\n[Ctrl+C] сохраняю прогресс')
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

    final = pd.read_parquet(LLM_KB_CHECK)
    parsed = final['parsed_ok'].sum()
    biz = final['biz_ok'].sum()
    print(f'\nитого {len(final)}, parsed_ok: {parsed} ({parsed / len(final):.1%}), '
          f'biz_ok: {biz} ({biz / len(final):.1%})')
    print(f'cost: ${current_cost():.4f}, токены: '
          f'cached={total["cached"]:,} in={total["in"]:,} out={total["out"]:,}')

    # прикидываем сколько kb-меток будет демоутиться на следующем этапе
    not_valid = final[
        (final['pred_status'].notna())
        & (final['pred_status'] != 'valid_single_film_camera')
        & (final['confidence'] >= 0.85)
    ]
    print(f'\nпредварительный анализ:')
    print(f'  LLM-vision сказал "не валидная камера" с conf >= 0.85: '
          f'{len(not_valid)} строк ({len(not_valid) / len(final) * 100:.1f}%)')
    print('  эти kb-метки демоутятся в other_unknown на этапе decision')


if __name__ == '__main__':
    main()
