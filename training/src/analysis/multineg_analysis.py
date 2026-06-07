"""Multi-negative other_unknown: сплит негатива на silver-подтипы при обучении, склейка назад (user-run)

Гипотеза (другой механизм, чем флаги-фичи/порог): один other_unknown в softmax описывает сразу лот,
аксессуар, запчасти, цифру, кино, коробку - плохая геометрия для линейной головы. Разбиваем НЕГАТИВ
train на silver-подтипы (regex по тексту, без доразметки), учим чемпион-конфиг на 5 main + K other_*,
на инференсе p_other_unknown = sum(p_other_*) -> снова 6 классов. Если геометрия лучше - типизатор сам
уводит больше ловушек в argmax-other, W падает БЕЗ порога. Сравнение matched-frontier (tie-break) vs
чемпион на val + audit. validation/ заморожен. Это РЕТРЕЙН (один фит), не каскад/OOF

Запуск: python training/src/multineg_analysis.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # training/
sys.path.insert(0, str(HERE.parent))         # training/src/
sys.path.insert(0, str(HERE))                # training/src/analysis/
from common import contract_frame
from experiments import CHAMP_BLOCKS
from features import build_features, load_blocks
from models import make_head
from paths import PREDICTIONS_DIR
from tracking.settings import REPORTS_DIR
from validation.metrics import business_metrics
from validation.settings import ABSTAIN, AER_CAP, AUDIT_SAMPLE, EDA_DATASET, MAIN_SET

CHAMPION = 'lrgrid__c1.0__none'
CHAMP_HEAD = ('logreg', {'C': 1.0, 'class_weight': None})
REPORT = REPORTS_DIR / 'multineg_analysis.md'
MAIN = list(MAIN_SET)

# silver-подтипы негатива: порядок = приоритет (первое совпадение). Узкие, из error-analysis остатка
SUBTYPES = [
    ('other_parts',     r'запчаст|в ремонт|неисправ|не работает|\bразбит|нерабоч|на детали'),
    ('other_lot',       r'ассортимент|\bлот\b|комплект из|несколько фотоаппарат|[2-9]\s*фотоаппарат'),
    ('other_movie',     r'кинокамер|киносъ|\bкино'),
    ('other_accessory', r'видоискатель|вспышк|чехол|футляр|\bсумк|объектив|переходник|адаптер'),
    ('other_digital',   r'цифров|беззеркальн|mirrorless|nikon\s*z|\bom-5|зеркальн'),
    ('other_box',       r'коробк|упаковк'),
]
COMPILED = [(name, re.compile(pat)) for name, pat in SUBTYPES]
OTHER_SUBLABELS = [name for name, _ in SUBTYPES] + ['other_rest']


def silver_sublabel(texts) -> np.ndarray:
    """Подтип негатива по тексту (первое совпадение), иначе other_rest"""
    out = []
    for t in texts:
        low = str(t).lower()
        lab = 'other_rest'
        for name, rx in COMPILED:
            if rx.search(low):
                lab = name
                break
        out.append(lab)
    return np.array(out, dtype=object)


def fit_multineg(data: pd.DataFrame):
    """Расширить train-негатив на подтипы, фитнуть чемпион-конфиг; вернуть (head, feats)"""
    head_name, head_params = CHAMP_HEAD
    exp = data['final_label'].astype(object).to_numpy().copy()
    train_other = (data['split'] == 'train').to_numpy() & (exp == ABSTAIN)
    exp[train_other] = silver_sublabel(data.loc[train_other, 'text'])
    orig = data['final_label'].to_numpy().copy()
    try:
        data['final_label'] = exp
        feats = build_features(data, CHAMP_BLOCKS)
    finally:
        data['final_label'] = orig
    head = make_head(head_name, head_params).fit(feats['X_train'], feats['y_train'])
    print('silver-подтипы train-негатива:',
          dict(pd.Series(exp[train_other]).value_counts()))
    return head, feats


def merge_to_six(head, x, item_ids) -> pd.DataFrame:
    """proba по расширенным классам -> 6-классовый контракт (p_other = sum подтипов)"""
    proba = head.predict_proba(x)
    cls = list(head.classes_)
    by_class = {}
    for m in MAIN:
        by_class[m] = proba[:, cls.index(m)] if m in cls else np.zeros(len(item_ids))
    other_idx = [i for i, c in enumerate(cls) if c not in MAIN]
    by_class[ABSTAIN] = proba[:, other_idx].sum(axis=1)
    return contract_frame(item_ids, by_class)


# ---------- оценка (tie-break, как в acceptor_analysis; business_metrics напрямую) ----------

def truth_frame() -> pd.DataFrame:
    eda = pd.read_parquet(EDA_DATASET, columns=['item_id', 'final_label'])
    au = pd.read_parquet(AUDIT_SAMPLE, columns=['item_id', 'audit_label'])
    return eda.merge(au, on='item_id', how='left')


def dec_frame(preds: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    fr = preds[['item_id', 'pred_label', 'confidence']]
    return fr.merge(truth, on='item_id', how='left')


def pick_tau_tb(fr: pd.DataFrame, aer_cap: float = AER_CAP) -> float:
    """Operating point на val/final_label: точные кандидаты + tie-break max C / min W / min AER_hi"""
    conf = fr['confidence'].fillna(0).to_numpy()
    rows = []
    for tau in np.unique(conf):
        pred = fr['pred_label'].where(conf >= tau, ABSTAIN)
        bm = business_metrics(fr['final_label'], pred)
        bm['threshold'] = float(tau)
        rows.append(bm)
    curve = pd.DataFrame(rows)
    feas = curve[curve['AutoErrorRate_hi'].fillna(1.0) <= aer_cap]
    if feas.empty:
        return 1.0
    best = feas.sort_values(['C_correct', 'W_wrong', 'AutoErrorRate_hi'],
                            ascending=[False, True, True]).iloc[0]
    return float(best['threshold'])


def business_at(fr: pd.DataFrame, truth_col: str, tau: float) -> dict:
    pred = fr['pred_label'].where(fr['confidence'].fillna(0) >= tau, ABSTAIN)
    return business_metrics(fr[truth_col], pred)


def _num(x) -> str:
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'


def _table(headers, rows) -> str:
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |',
                      *['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]])


def main():
    data = load_blocks(CHAMP_BLOCKS)
    truth = truth_frame()

    print('фит multi-negative (ретрейн чемпион-конфига на расширенных классах)...')
    head, feats = fit_multineg(data)
    mn_val = merge_to_six(head, feats['X_val'], feats['id_val'])
    mn_test = merge_to_six(head, feats['X_test'], feats['id_test'])

    champ_val = pd.read_parquet(PREDICTIONS_DIR / 'lrgrid_c1_0_none_val_pred.parquet')
    champ_test = pd.read_parquet(PREDICTIONS_DIR / 'lrgrid_c1_0_none_test_pred.parquet')

    schemes = {'champion (6-класс)': (champ_val, champ_test),
               'multi-negative': (mn_val, mn_test)}
    proj_rows = {p: [] for p in ('val / final_label', 'test / final_label (22k)', 'test / audit_label (210)')}
    summary = []
    for name, (vp, tp) in schemes.items():
        vfr, tfr = dec_frame(vp, truth), dec_frame(tp, truth)
        afr = tfr[tfr['audit_label'].notna()].copy()
        tau = pick_tau_tb(vfr)
        for proj, (fr, tc) in [('val / final_label', (vfr, 'final_label')),
                               ('test / final_label (22k)', (tfr, 'final_label')),
                               ('test / audit_label (210)', (afr, 'audit_label'))]:
            bm = business_at(fr, tc, tau)
            proj_rows[proj].append([name, _num(bm['CorrectAutofillRate']), _num(bm['AutoErrorRate']),
                                    _num(bm['AutoErrorRate_hi']), bm['A_auto'], bm['C_correct'], bm['W_wrong']])
        av = business_at(vfr, 'final_label', tau)
        aa = business_at(afr, 'audit_label', tau)
        summary.append((name, tau, av, aa))

    cols = ['схема', 'CAR', 'AER', 'AER_hi', 'A', 'C', 'W']
    parts = [f'# Multi-negative other_unknown vs чемпион: {CHAMPION}',
             'Негатив train разбит на silver-подтипы (regex), фит на 5 main + K other_*, на инференсе '
             'p_other=sum(подтипы) -> 6 классов. Operating point - tie-break (max C/min W) на val. '
             'Вопрос: уводит ли типизатор больше ловушек в argmax-other (W vs чемпион, без порога)',
             '## Сравнение в 3 проекциях']
    for proj, rows in proj_rows.items():
        parts += [f'### {proj}', _table(cols, rows)]
    # вердикт
    (cn, ct, cv, ca), (mn, mt, mv, ma) = summary
    lines = [f"{cn}: val CAR {cv['CorrectAutofillRate']:.3f} W={cv['W_wrong']} | "
             f"audit CAR {ca['CorrectAutofillRate']:.3f} W={ca['W_wrong']} AER_hi {ca['AutoErrorRate_hi']:.3f}",
             f"{mn}: val CAR {mv['CorrectAutofillRate']:.3f} W={mv['W_wrong']} "
             f"(ΔCAR{mv['CorrectAutofillRate'] - cv['CorrectAutofillRate']:+.3f}) | "
             f"audit CAR {ma['CorrectAutofillRate']:.3f} W={ma['W_wrong']} AER_hi {ma['AutoErrorRate_hi']:.3f}"]
    better = ((mv['CorrectAutofillRate'] > cv['CorrectAutofillRate'] + 1e-9)
              or (ma['AutoErrorRate_hi'] <= 0.05 < ca['AutoErrorRate_hi']))
    lines.append('ВЕРДИКТ: ' + ('multi-negative двигает метрику - смотреть детально' if better
                                else 'multi-negative НЕ лучше чемпиона на честной метрике'))
    verdict = '\n'.join(lines)
    parts += ['## Вердикт', '```\n' + verdict + '\n```']

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n\n'.join(parts) + '\n')
    print(f'\nотчёт: {REPORT}\n')
    print(verdict)


if __name__ == '__main__':
    main()
