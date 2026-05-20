"""Финальный мерж всех источников меток в labels_final.parquet

Приоритет (см. data/decision_rules.md):

1. gold - ручная разметка, всегда побеждает, confidence = 1.0

2. kb - regex по справочнику моделей. Если есть результат проверки по фото
   из stage_07 и LLM с уверенностью не ниже 0.85 сказал что это не валидная
   одиночная плёночная камера (лот, чехол, коробка, цифровая, и т.п.) -
   демоутим kb-метку в other_unknown. Иначе оставляем как есть, confidence = 1.0

3. llm - предсказание из stage_06, если бизнес-инвариант ok и уверенность >= 0.7

4. abstain - всё остальное -> other_unknown с needs_review = True

В итоговой таблице храним и финальную метку, и метки от каждого источника
отдельно - чтобы потом можно было разобраться 'почему такая метка'
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.io import atomic_write_parquet
from lib.paths import (GOLD, ITEMS, KB_LABELS, LABELS_FINAL, LLM_KB_CHECK,
                       LLM_LABELS)

LLM_CONFIDENCE_MIN = 0.7
KB_DEMOTE_CONFIDENCE = 0.85


def pick_final(r):
    # 1. ручная разметка побеждает всегда
    if pd.notna(r.get('gold_final_label')):
        return {
            'final_object_status': r['gold_object_status'],
            'final_body_type': r.get('gold_body_type') if pd.notna(r.get('gold_body_type')) else None,
            'final_label': r['gold_final_label'],
            'final_reason': 'human-labeled',
            'label_source': 'human',
            'confidence': 1.0,
            'needs_review': False,
        }

    # 2. kb с возможным демоутом по результатам проверки по фото
    if pd.notna(r.get('kb_label')):
        kbc_status = r.get('kb_check_status')
        kbc_conf = r.get('kb_check_confidence') or 0.0
        if (pd.notna(kbc_status)
                and kbc_status != 'valid_single_film_camera'
                and kbc_conf >= KB_DEMOTE_CONFIDENCE):
            return {
                'final_object_status': kbc_status,
                'final_body_type': None,
                'final_label': 'other_unknown',
                'final_reason': f'kb demoted by llm-vision: {kbc_status}',
                'label_source': 'kb_overridden',
                'confidence': float(kbc_conf),
                'needs_review': False,
            }
        return {
            'final_object_status': 'valid_single_film_camera',
            'final_body_type': r['kb_label'],
            'final_label': r['kb_label'],
            'final_reason': f'kb match: {r["kb_model"]}',
            'label_source': 'kb',
            'confidence': 1.0,
            'needs_review': False,
        }

    # 3. llm если есть и уверен
    llm_conf = r.get('llm_confidence') or 0.0
    if pd.notna(r.get('llm_label')) and r.get('llm_biz_ok') and llm_conf >= LLM_CONFIDENCE_MIN:
        return {
            'final_object_status': r['llm_object_status'],
            'final_body_type': r.get('llm_body_type') if pd.notna(r.get('llm_body_type')) else None,
            'final_label': r['llm_label'],
            'final_reason': 'llm prediction',
            'label_source': 'llm',
            'confidence': float(llm_conf),
            'needs_review': False,
        }

    # 4. ни один источник не уверен - идём в other_unknown с пометкой 'посмотреть руками'
    llm_status = r.get('llm_object_status') if pd.notna(r.get('llm_object_status')) else None
    if llm_status and llm_status != 'valid_single_film_camera':
        status = llm_status
        reason = f'llm_low_confidence_non_valid: {llm_status}'
    else:
        status = 'insufficient_info'
        reason = 'no confident source or unreliable valid claim'
    return {
        'final_object_status': status,
        'final_body_type': None,
        'final_label': 'other_unknown',
        'final_reason': reason,
        'label_source': 'abstain',
        'confidence': 0.0,
        'needs_review': True,
    }


def main():
    items = pd.read_parquet(ITEMS)[['item_id', 'image_id']]
    gold = pd.read_parquet(GOLD)[['item_id', 'object_status', 'body_type', 'final_label']]
    gold = gold.rename(columns={
        'object_status': 'gold_object_status',
        'body_type': 'gold_body_type',
        'final_label': 'gold_final_label',
    })
    kb = pd.read_parquet(KB_LABELS)[['item_id', 'kb_model', 'kb_label']]
    llm = pd.read_parquet(LLM_LABELS)[
        ['item_id', 'pred_status', 'pred_body', 'pred_label', 'confidence', 'biz_ok']
    ].rename(columns={
        'pred_status': 'llm_object_status',
        'pred_body': 'llm_body_type',
        'pred_label': 'llm_label',
        'confidence': 'llm_confidence',
        'biz_ok': 'llm_biz_ok',
    })

    df = items.merge(gold, on='item_id', how='left')
    df = df.merge(kb, on='item_id', how='left')
    df = df.merge(llm, on='item_id', how='left')

    if LLM_KB_CHECK.exists():
        kbc = pd.read_parquet(LLM_KB_CHECK)[
            ['item_id', 'pred_status', 'pred_label', 'confidence']
        ].rename(columns={
            'pred_status': 'kb_check_status',
            'pred_label': 'kb_check_label',
            'confidence': 'kb_check_confidence',
        })
        df = df.merge(kbc, on='item_id', how='left')
        print(f'llm_kb_check.parquet найден: {len(kbc)} строк проверены по фото')
    else:
        df['kb_check_status'] = None
        df['kb_check_label'] = None
        df['kb_check_confidence'] = None
        print('llm_kb_check.parquet нет - kb-метки берём без проверки по фото')

    picked = df.apply(pick_final, axis=1, result_type='expand')

    out = pd.concat([
        df[['item_id', 'image_id']],
        picked,
        df[['gold_final_label', 'kb_label', 'kb_model',
            'llm_object_status', 'llm_body_type', 'llm_label',
            'llm_confidence', 'llm_biz_ok',
            'kb_check_status', 'kb_check_label', 'kb_check_confidence']]
        .rename(columns={'gold_final_label': 'gold_label'}),
    ], axis=1)

    print(f'\nвсего строк: {len(out)}')
    print('\nраспределение label_source:')
    print(out['label_source'].value_counts().to_string())
    print('\nраспределение final_label:')
    print(out['final_label'].value_counts().to_string())
    print('\nраспределение final_object_status:')
    print(out['final_object_status'].value_counts().to_string())
    print(f'\nneeds_review (низкая уверенность / abstain): {out["needs_review"].sum()}')

    demoted = (out['label_source'] == 'kb_overridden').sum()
    if demoted > 0:
        print(f'\nkb-меток демоутировано по фото: {demoted}')

    LABELS_FINAL.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(out, LABELS_FINAL)
    print(f'\nсохранил {LABELS_FINAL}, колонок: {len(out.columns)}')


if __name__ == '__main__':
    main()
