"""Error-analysis чемпиона: компас каскад vs стекинг

Разбивает ошибки автозаполнения (W) на «должен был отказаться» (truth=other_unknown) и
«камера, но не тот тип», смотрит confusion и срезы, тянет примеры с текстом. Решение про
каскад берём по audit (vs человек), структуру типов дополняем proxy (vs Gemini, 22k) -
чтобы не оптимизировать под учителя (урок эксп.8). Read-only: читает предсказания, пишет md

Запуск: python training/src/error_analysis.py [имя_эксперимента]   (по умолчанию - чемпион)
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
from paths import PREDICTIONS_DIR
from tracking.settings import REPORTS_DIR
from validation.evaluate import evaluate, load_eval_frame
from validation.metrics import business_metrics, pick_operating_point, threshold_sweep
from validation.settings import ABSTAIN, AUDIT_SAMPLE, EDA_DATASET, MAIN_SET, SPLIT_PARQUET
from validation.slices import add_slice_columns, slice_metrics

CHAMPION = 'lrgrid__c1.0__none'
REPORT = REPORTS_DIR / 'error_analysis_champion.md'
N_EXAMPLES = 10
SLICE_COLS = ['slice_generic_title', 'slice_kb_model', 'slice_tlr', 'slice_source', 'slice_seller']
# короткие метки для читаемой confusion-матрицы
SHORT = {'SLR': 'SLR', 'TLR': 'TLR', 'rangefinder_viewfinder': 'RV',
         'compact_point_and_shoot': 'CPS', 'instant': 'INST', 'other_unknown': 'OTH'}


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def resolve_paths(name) -> tuple:
    """Пути предсказаний по слагу имени (по умолчанию - чемпион)"""
    name = name or CHAMPION
    slug = _slug(name)
    return PREDICTIONS_DIR / f'{slug}_val_pred.parquet', PREDICTIONS_DIR / f'{slug}_test_pred.parquet', name


def apply_tau(frame: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Перевести предсказания ниже порога в отказ - ровно как evaluate"""
    out = frame.copy()
    out['pred_label'] = out['pred_label'].where(out['confidence'].fillna(0) >= tau, ABSTAIN)
    return out


def choose_tau(val_path: Path) -> float:
    """Рабочий порог чемпиона - подбор на val (final_label) под AER_hi <= 5%"""
    vf = load_eval_frame(val_path, 'val', 'final_label')
    return pick_operating_point(threshold_sweep(vf, 'final_label', 'pred_label', 'confidence'))['tau']


def decompose_errors(frame: pd.DataFrame, truth_col: str, tau: float) -> dict:
    """Разбить ошибки автозаполнения W на should_abstain и wrong_type"""
    f = apply_tau(frame, tau)
    truth, pred = f[truth_col].astype(str), f['pred_label'].astype(str)
    autofill = pred.isin(set(MAIN_SET))
    wrong = autofill & (pred != truth)
    should_abstain = wrong & (truth == ABSTAIN)              # правда - other_unknown, а мы поставили тип
    wrong_type = wrong & truth.isin(set(MAIN_SET))           # камера, но перепутали тип
    w = int(wrong.sum())
    return {
        'n': len(f), 'A_auto': int(autofill.sum()), 'W_wrong': w,
        'should_abstain': int(should_abstain.sum()), 'wrong_type': int(wrong_type.sum()),
        'frac_should_abstain': should_abstain.sum() / w if w else float('nan'),
        'frac_wrong_type': wrong_type.sum() / w if w else float('nan'),
    }


def error_examples(frame: pd.DataFrame, truth_col: str, tau: float, kind: str, k=N_EXAMPLES) -> pd.DataFrame:
    """k примеров ошибки заданного рода с текстом, чтобы увидеть паттерн глазами"""
    f = apply_tau(frame, tau)
    truth, pred = f[truth_col].astype(str), f['pred_label'].astype(str)
    wrong = pred.isin(set(MAIN_SET)) & (pred != truth)
    mask = wrong & (truth == ABSTAIN) if kind == 'should_abstain' else wrong & truth.isin(set(MAIN_SET))
    return f.loc[mask, ['item_id', 'title', truth_col, 'pred_label', 'confidence']].head(k)


def slice_table(frame: pd.DataFrame, truth_col: str, tau: float, slice_col: str) -> pd.DataFrame:
    """CAR/AER по значениям одного среза"""
    f = add_slice_columns(apply_tau(frame, tau))
    return slice_metrics(f, slice_col, lambda sub: business_metrics(sub[truth_col], sub['pred_label']))


def compass_verdict(da: dict, dp: dict) -> str:
    """Куда тянет рычаг - по надёжной proxy-структуре, направление сверяем с audit"""
    fa, ft = dp['frac_should_abstain'], dp['frac_wrong_type']
    if fa >= 0.6:
        call = 'КАСКАД стейдж-1 (камера/не-камера) - доминируют ошибки should_abstain'
    elif ft >= 0.6:
        call = 'СТЕЙДЖ-2 / стекинг - доминируют ошибки wrong_type (различение типов)'
    else:
        call = 'близко к потолку - ошибки размазаны, ни каскад ни стекинг сильно не дадут'
    return (f"audit (W={da['W_wrong']}, шумно): should_abstain {da['frac_should_abstain']:.0%} / "
            f"wrong_type {da['frac_wrong_type']:.0%}\n"
            f"proxy (W={dp['W_wrong']}, надёжно): should_abstain {fa:.0%} / wrong_type {ft:.0%}\n"
            f"ВЕРДИКТ: {call}")


# ---------- рендер markdown ----------

def _num(x) -> str:
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'


def _table(headers: list, rows: list) -> str:
    head = '| ' + ' | '.join(headers) + ' |'
    sep = '| ' + ' | '.join('---' for _ in headers) + ' |'
    body = '\n'.join('| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows)
    return '\n'.join([head, sep, body])


def _decomp_table(da: dict, dp: dict) -> str:
    rows = []
    for name, d in [('audit (vs человек, шумно)', da), ('proxy (vs Gemini, 22k)', dp)]:
        rows.append([name, d['A_auto'], d['W_wrong'],
                     f"{d['should_abstain']} ({d['frac_should_abstain']:.0%})",
                     f"{d['wrong_type']} ({d['frac_wrong_type']:.0%})"])
    return _table(['истина', 'автозаполнено A', 'ошибок W', 'should_abstain', 'wrong_type'], rows)


def _confusion_table(cm: list, labels: list) -> str:
    short = [SHORT[label] for label in labels]
    headers = ['true \\ pred'] + short
    rows = [[short[i]] + [cm[i][j] for j in range(len(labels))] for i in range(len(labels))]
    return _table(headers, rows)


def _slice_md(table: pd.DataFrame) -> str:
    rows = [[r['value'], r['n'], _num(r['CorrectAutofillRate']), _num(r['AutoErrorRate']), _num(r['AutoErrorRate_hi'])]
            for _, r in table.iterrows()]
    return _table(['значение', 'n', 'CAR', 'AER', 'AER_hi'], rows)


def _examples_md(ex: pd.DataFrame, truth_col: str) -> str:
    if ex.empty:
        return '_нет таких ошибок_'
    rows = [[str(r['item_id'])[:8], str(r['title'])[:55], r[truth_col], r['pred_label'], f"{r['confidence']:.2f}"]
            for _, r in ex.iterrows()]
    return _table(['item_id', 'title', 'truth', 'pred', 'conf'], rows)


def render(name, tau, da, dp, cm, labels, slices, ex_abstain, ex_type, verdict) -> str:
    parts = [
        f'# Error-analysis: {name}',
        f'Рабочий порог tau = {tau:.2f}. Решение про каскад берём по **audit** (vs человек), '
        f'структуру типов - по **proxy** (vs Gemini, 22k, статмощность)',
        '## 1. Декомпозиция ошибок автозаполнения W (компас)',
        'Каждую ошибку автозаполнения относим к: should_abstain (правда other_unknown - лот/коробка/не камера) '
        'или wrong_type (камера, но перепутан тип)',
        _decomp_table(da, dp),
        '## 2. Confusion 6x6 (proxy-test, при tau)',
        'Строки - истина, столбцы - предсказание. Блок 5x5 без OTH = путаница ТИПОВ, '
        'строка/столбец OTH = поведение отказа. '
        'RV=rangefinder_viewfinder, CPS=compact, INST=instant, OTH=other_unknown',
        _confusion_table(cm, labels),
        '## 3. Ошибки по срезам (proxy-test)',
    ]
    for col in SLICE_COLS:
        parts += [f'### {col}', _slice_md(slices[col])]
    parts += [
        f'## 4. Примеры ошибок (по {N_EXAMPLES}, proxy-test)',
        '### should_abstain (правда other_unknown, а мы поставили тип)',
        _examples_md(ex_abstain, 'final_label'),
        '### wrong_type (камера, но не тот тип)',
        _examples_md(ex_type, 'final_label'),
        '## 5. Вердикт компаса',
        '```\n' + verdict + '\n```',
    ]
    return '\n\n'.join(parts) + '\n'


def _stage1_frame(name: str) -> pd.DataFrame:
    """Фрейм для анализа стейджа-1: p_camera + истина + контекст по test"""
    s1 = pd.read_parquet(PREDICTIONS_DIR / f'{_slug(name)}_stage1.parquet')
    split = pd.read_parquet(SPLIT_PARQUET)
    split = split[split['split'] == 'test'][['item_id']]
    ctx = pd.read_parquet(EDA_DATASET,
                          columns=['item_id', 'user_id', 'title', 'kb_model', 'label_source', 'final_label'])
    audit = pd.read_parquet(AUDIT_SAMPLE, columns=['item_id', 'audit_label'])
    return split.merge(s1, on='item_id').merge(ctx, on='item_id').merge(audit, on='item_id', how='left')


def _boundary(frame: pd.DataFrame, truth_col: str, tau1: float) -> dict:
    """Метрики границы камера/не на одной истине: FP (пропуск), FN (потеря CAR)"""
    sub = frame[frame[truth_col].notna()]
    is_cam = sub[truth_col].isin(set(MAIN_SET))
    pred_cam = sub['p_camera'] >= tau1
    fp = int((pred_cam & ~is_cam).sum())          # сказал камера, а other -> пропуск в стейдж-2 (ошибка системы)
    fn = int((~pred_cam & is_cam).sum())          # сказал не-камера, а реальная -> потеря CAR
    tp, n_cam, n_other = int((pred_cam & is_cam).sum()), int(is_cam.sum()), int((~is_cam).sum())
    return {'n_camera': n_cam, 'n_other': n_other, 'FP': fp, 'FN': fn,
            'FP_rate': fp / n_other if n_other else float('nan'),
            'FN_rate': fn / n_cam if n_cam else float('nan'),
            'precision_camera': tp / (tp + fp) if (tp + fp) else float('nan')}


def stage1_analysis(name: str, tau1: float = 0.5) -> None:
    """Граница камера/не стейджа-1: FP=пропуск(ограничивает AER), FN=потеря CAR, по source, примеры FP"""
    frame = _stage1_frame(name)
    print(f'\n[стейдж-1 каскада {name}]  tau1 = {tau1}')
    for label, truth in [('proxy (22k)', 'final_label'), ('audit (210)', 'audit_label')]:
        b = _boundary(frame, truth, tau1)
        print(f"  {label}: FP {b['FP']}/{b['n_other']} ({b['FP_rate']:.0%}) -> пропуск в стейдж-2 | "
              f"FN {b['FN']}/{b['n_camera']} ({b['FN_rate']:.0%}) -> потеря CAR | "
              f"precision_cam {b['precision_camera']:.3f}")

    fp_mask = (frame['p_camera'] >= tau1) & ~frame['final_label'].isin(set(MAIN_SET))
    print('\n  FP (пропущенные other_unknown) по label_source:')
    for src, n in frame.loc[fp_mask, 'label_source'].value_counts().items():
        print(f'    {src}: {n}')
    print('\n  примеры FP (лоты/аксессуары, прошедшие стейдж-1):')
    for _, r in frame.loc[fp_mask, ['item_id', 'title', 'p_camera']].head(N_EXAMPLES).iterrows():
        print(f"    {str(r['item_id'])[:8]} p={r['p_camera']:.2f}  {str(r['title'])[:55]}")


def main():
    args = sys.argv[1:]
    if args and args[0] == '--stage1':
        stage1_analysis(args[1])
        return
    name_arg = args[0] if args else None
    val_path, test_path, name = resolve_paths(name_arg)
    tau = choose_tau(val_path)

    audit_frame = load_eval_frame(test_path, 'test', 'audit_label')   # бизнес-истина
    proxy_frame = load_eval_frame(test_path, 'test', 'final_label')   # структура

    da = decompose_errors(audit_frame, 'audit_label', tau)
    dp = decompose_errors(proxy_frame, 'final_label', tau)
    assert da['should_abstain'] + da['wrong_type'] == da['W_wrong'], 'audit: W не сошёлся'
    assert dp['should_abstain'] + dp['wrong_type'] == dp['W_wrong'], 'proxy: W не сошёлся'

    rep = evaluate(test_path, 'test', 'final_label', tau=tau)['classification']
    cm, labels = rep['confusion_matrix'], rep['labels']
    slices = {col: slice_table(proxy_frame, 'final_label', tau, col) for col in SLICE_COLS}
    ex_abstain = error_examples(proxy_frame, 'final_label', tau, 'should_abstain')
    ex_type = error_examples(proxy_frame, 'final_label', tau, 'wrong_type')
    verdict = compass_verdict(da, dp)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(name, tau, da, dp, cm, labels, slices, ex_abstain, ex_type, verdict))
    print(f'отчёт: {REPORT}\n')
    print(verdict)


if __name__ == '__main__':
    main()
