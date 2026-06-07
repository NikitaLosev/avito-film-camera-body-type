"""sample_weight по label_source: анти-teacher-noise (user-run, фичи раз + неск. фитов logreg)

Гипотеза: final_label - смесь источников (kb надёжный, llm=Gemini шумный). Сейчас logreg учит все метки
равновесно -> подгоняет llm-шум. Вес train-строк по надёжности источника (kb выше, llm ниже) может сдвинуть
типизатор к надёжным меткам. label_source НЕ фича (не утечка - только train-вес, на инференсе не нужен).
Нюанс: kb-строки лёгкие (популярные модели), llm - hard no-kb где и ошибки -> направление неочевидно, решат
данные. Фичи строим один раз, фитим logreg с разными весами на одних X. uniform-фит = sanity (=чемпион).
Сравнение matched-frontier (tie-break) vs чемпион на val + audit. validation/ заморожен

Запуск: python training/src/weight_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # training/
sys.path.insert(0, str(HERE.parent))         # training/src/
sys.path.insert(0, str(HERE))                # training/src/analysis/
from common import predictions_frame
from experiments import CHAMP_BLOCKS
from features import build_features, load_blocks
from models import make_head
from multineg_analysis import _num, _table, business_at, dec_frame, pick_tau_tb, truth_frame
from paths import PREDICTIONS_DIR
from tracking.settings import REPORTS_DIR
from validation.settings import EDA_DATASET

CHAMPION = 'lrgrid__c1.0__none'
CHAMP_HEAD = ('logreg', {'C': 1.0, 'class_weight': None})
REPORT = REPORTS_DIR / 'weight_analysis.md'

# карты весов train по label_source (human в train нет; kb надёжный, llm шумный). default=1.0
WEIGHT_SCHEMES = {
    'uniform (=champion)': {},
    'mild (kb^ llm v)': {'kb': 1.5, 'kb_overridden': 1.2, 'llm': 0.8, 'abstain': 1.0},
    'aggr (kb^^ llm vv)': {'kb': 2.0, 'kb_overridden': 1.5, 'llm': 0.5, 'abstain': 1.0},
}


def weights_for(item_ids, src_by_id: dict, wmap: dict) -> np.ndarray:
    return np.array([wmap.get(src_by_id.get(i), 1.0) for i in item_ids], dtype=float)


def main():
    data = load_blocks(CHAMP_BLOCKS)
    ls = pd.read_parquet(EDA_DATASET, columns=['item_id', 'label_source'])
    data = data.merge(ls, on='item_id', validate='one_to_one')
    truth = truth_frame()

    feats = build_features(data, CHAMP_BLOCKS)
    src_by_id = dict(zip(data['item_id'], data['label_source']))
    head_name, head_params = CHAMP_HEAD

    results = {'champion (saved)': (pd.read_parquet(PREDICTIONS_DIR / 'lrgrid_c1_0_none_val_pred.parquet'),
                                    pd.read_parquet(PREDICTIONS_DIR / 'lrgrid_c1_0_none_test_pred.parquet'))}
    for name, wmap in WEIGHT_SCHEMES.items():
        w = weights_for(feats['id_train'], src_by_id, wmap)
        head = make_head(head_name, head_params)
        head.fit(feats['X_train'], feats['y_train'], sample_weight=w)
        results[name] = (predictions_frame(head, feats['X_val'], feats['id_val']),
                         predictions_frame(head, feats['X_test'], feats['id_test']))
        print(f'фит [{name}] готов (вес-диапазон {w.min():.1f}..{w.max():.1f})')

    cols = ['схема', 'CAR', 'AER', 'AER_hi', 'A', 'C', 'W']
    proj_titles = ['val / final_label', 'test / final_label (22k)', 'test / audit_label (210)']
    proj_rows = {t: [] for t in proj_titles}
    summary = {}
    for name, (vp, tp) in results.items():
        vfr, tfr = dec_frame(vp, truth), dec_frame(tp, truth)
        afr = tfr[tfr['audit_label'].notna()].copy()
        tau = pick_tau_tb(vfr)
        views = [('val / final_label', vfr, 'final_label'),
                 ('test / final_label (22k)', tfr, 'final_label'),
                 ('test / audit_label (210)', afr, 'audit_label')]
        for title, fr, tc in views:
            bm = business_at(fr, tc, tau)
            proj_rows[title].append([name, _num(bm['CorrectAutofillRate']), _num(bm['AutoErrorRate']),
                                     _num(bm['AutoErrorRate_hi']), bm['A_auto'], bm['C_correct'], bm['W_wrong']])
        summary[name] = (business_at(vfr, 'final_label', tau), business_at(afr, 'audit_label', tau))

    parts = [f'# sample_weight по label_source vs чемпион: {CHAMPION}',
             'Вес train-строк по надёжности источника метки (kb выше, llm ниже) - анти-teacher-noise. '
             'label_source только train-вес, не фича. Operating point - tie-break (max C/min W) на val. '
             'uniform = sanity (должен воспроизвести чемпиона)',
             '## Сравнение схем в 3 проекциях']
    for title in proj_titles:
        parts += [f'### {title}', _table(cols, proj_rows[title])]

    cv, ca = summary['champion (saved)']
    lines = [f"champion: val CAR {cv['CorrectAutofillRate']:.3f} W={cv['W_wrong']} | "
             f"audit CAR {ca['CorrectAutofillRate']:.3f} W={ca['W_wrong']} AER_hi {ca['AutoErrorRate_hi']:.3f}"]
    win = []
    for name in WEIGHT_SCHEMES:
        v, a = summary[name]
        dC = v['CorrectAutofillRate'] - cv['CorrectAutofillRate']
        better = (dC > 1e-9) or (a['AutoErrorRate_hi'] <= 0.05 < ca['AutoErrorRate_hi']) or \
                 (v['W_wrong'] < cv['W_wrong'] and dC >= -1e-9)
        lines.append(f"{name}: val CAR {v['CorrectAutofillRate']:.3f} W={v['W_wrong']} (ΔCAR{dC:+.3f}) | "
                     f"audit CAR {a['CorrectAutofillRate']:.3f} W={a['W_wrong']} AER_hi {a['AutoErrorRate_hi']:.3f}"
                     + ('  <- лучше' if better and name != 'uniform (=champion)' else ''))
        if better and name != 'uniform (=champion)':
            win.append(name)
    lines.append('ВЕРДИКТ: ' + (f'двигает: {win} - смотреть детально' if win
                                else 'sample_weight НЕ лучше чемпиона на честной метрике'))
    verdict = '\n'.join(lines)
    parts += ['## Вердикт', '```\n' + verdict + '\n```']

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n\n'.join(parts) + '\n')
    print(f'\nотчёт: {REPORT}\n')
    print(verdict)


if __name__ == '__main__':
    main()
