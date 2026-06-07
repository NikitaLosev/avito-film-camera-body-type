"""Бизнес-policy: rescue false-abstain (pred_other->main) пост-хок демоутом other (user-run, READ-ONLY)

Дыра, не закрытая kNN-probe (он смотрел ПРИНЯТЫЙ пул): CAR растёт двумя путями - (1) wrong_type->correct main
(probe: мёртв), (2) pred_other->correct main (rescue false-abstain - НЕ замеряли). На val ~269 строк, где чемпион
зря сказал other (truth=main) - потолок ~+1.4pp CAR. Это НЕ per-class порог (тот гейтил уже выбранный pred, не
воскрешал pred=other) и НЕ grid balanced/None (train-time). Bayes-факт: правильный other награды не даёт (это
отказ-действие), значит argmax по 6 классам не оптимум под бизнес-utility. Приём один: p_other *= alpha (<1),
пере-argmax -> часть false-abstain становится main. READ-ONLY над сохранёнными вероятностями (champion + stack
со шага 13, БЕЗ обучения). alpha свипаем, выбираем max C при AER_hi<=5% на val. alpha=1.0 = чемпион (sanity).
stack-базы заодно тестят «стек освободил error-budget -> потратить на rescue». Судим по val, proxy - teacher-overfit
чек, audit (210) - только санити (не короновать). validation/ заморожен

Запуск: python training/src/business_policy_analysis.py  (секунды, read-only)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # training/
sys.path.insert(0, str(HERE.parent))         # training/src/
sys.path.insert(0, str(HERE))                # training/src/analysis/
from multineg_analysis import _num, _table, truth_frame
from paths import PREDICTIONS_DIR
from tracking.settings import REPORTS_DIR
from validation.metrics import business_metrics
from validation.settings import ABSTAIN, AER_CAP, ALL_LABELS, MAIN_SET

CHAMPION = 'lrgrid__c1.0__none'
ALPHAS = np.geomspace(1.0, 0.1, 100)        # 1.0=чемпион (sanity); пол 0.1 (ниже other демоутится 10x)
REPORT = REPORTS_DIR / 'business_policy_analysis.md'
PCOLS = [f'p_{c}' for c in ALL_LABELS]
OTHER_J = ALL_LABELS.index(ABSTAIN)
MAIN = set(MAIN_SET)

BASES = {
    'champion':     ('lrgrid_c1_0_none_val_pred.parquet', 'lrgrid_c1_0_none_test_pred.parquet'),
    'stack_logreg': ('stack_logreg_val_pred.parquet', 'stack_logreg_test_pred.parquet'),
    'stack_HGB':    ('stack_hgb_val_pred.parquet', 'stack_hgb_test_pred.parquet'),
}


def matrices(contract: pd.DataFrame, truth: pd.DataFrame) -> tuple:
    """Выровнять контракт с truth -> (P (n,6) в каноне, final_label, audit_label)"""
    m = contract.merge(truth, on='item_id', how='left')
    return m[PCOLS].to_numpy(float), m['final_label'].to_numpy(), m['audit_label'].to_numpy()


def biased_pred(P: np.ndarray, alpha: float) -> np.ndarray:
    """p_other *= alpha, пере-argmax по канону -> метки (alpha<1 демоутит other, воскрешая main)"""
    Q = P.copy()
    Q[:, OTHER_J] *= alpha
    return np.array(ALL_LABELS)[Q.argmax(axis=1)]


def pick_alpha(P: np.ndarray, y_final: np.ndarray) -> float:
    """Свип alpha на val: max C_correct при AER_hi<=5%, tie-break min W (как frozen-селектор по CAR)"""
    rows = []
    for a in ALPHAS:
        bm = business_metrics(y_final, biased_pred(P, a))
        rows.append((a, bm['C_correct'], bm['W_wrong'], bm['AutoErrorRate_hi']))
    cur = pd.DataFrame(rows, columns=['alpha', 'C', 'W', 'AER_hi'])
    feas = cur[cur['AER_hi'].fillna(1.0) <= AER_CAP]
    if feas.empty:
        return 1.0
    return float(feas.sort_values(['C', 'W'], ascending=[False, True]).iloc[0]['alpha'])


def rescue_stats(P: np.ndarray, y_final: np.ndarray, alpha: float) -> tuple:
    """Сколько строк pred_other->main воскрешено при alpha и какова их precision (lift над base 7.8%)"""
    base, best = biased_pred(P, 1.0), biased_pred(P, alpha)
    resc = (base == ABSTAIN) & (best != ABSTAIN)
    n = int(resc.sum())
    correct = int((resc & (best == y_final)).sum())
    return n, correct, (correct / n if n else float('nan'))


def main() -> None:
    truth = truth_frame()
    rows, summary = [], {}

    def add(name: str, alpha: float, Pv, yfv, Pt, yft, yat) -> dict:
        vb = business_metrics(yfv, biased_pred(Pv, alpha))
        pb = business_metrics(yft, biased_pred(Pt, alpha))
        amask = ~pd.isna(yat)
        ab = business_metrics(yat[amask], biased_pred(Pt, alpha)[amask])
        rows.append([name, f'{alpha:.3f}', _num(vb['CorrectAutofillRate']), vb['C_correct'], vb['W_wrong'],
                     _num(vb['AutoErrorRate_hi']), _num(pb['CorrectAutofillRate']),
                     _num(ab['CorrectAutofillRate']), ab['W_wrong'], _num(ab['AutoErrorRate_hi'])])
        return {'val': vb, 'proxy': pb, 'audit': ab, 'alpha': alpha}

    fa_count = None
    for name, (vf, tf) in BASES.items():
        vpath, tpath = PREDICTIONS_DIR / vf, PREDICTIONS_DIR / tf
        if not vpath.exists() or not tpath.exists():
            print(f'[{name}] пропуск - нет {vf}/{tf} (прогони шаг 13?)')
            continue
        Pv, yfv, _ = matrices(pd.read_parquet(vpath), truth)
        Pt, yft, yat = matrices(pd.read_parquet(tpath), truth)
        if name == 'champion':                          # потолок rescue: false-abstain на val
            base = biased_pred(Pv, 1.0)
            fa_count = int(((base == ABSTAIN) & np.isin(yfv, list(MAIN))).sum())
        alpha = pick_alpha(Pv, yfv)
        summary[f'{name}@1.0'] = add(f'{name} @1.0 (без rescue)', 1.0, Pv, yfv, Pt, yft, yat)
        summary[f'{name}@best'] = add(f'{name} @{alpha:.3f} (rescue)', alpha, Pv, yfv, Pt, yft, yat)
        n, c, prec = rescue_stats(Pv, yfv, alpha)
        summary[f'{name}@best'].update({'resc_n': n, 'resc_c': c, 'resc_prec': prec,
                                        'floor': abs(alpha - ALPHAS[-1]) < 1e-9})
        print(f'[{name}] alpha*={alpha:.3f} | val rescue {c}/{n} correct (prec {prec:.1%})')

    cols = ['политика', 'alpha', 'val CAR', 'val C', 'val W', 'val AER_hi',
            'proxy CAR', 'audit CAR', 'audit W', 'audit AER_hi']

    ref = summary['champion@1.0']['val']                # прод-референс = чемпион без rescue
    cv = ref
    lines = [f"reference champion @1.0: val CAR {cv['CorrectAutofillRate']:.4f} C={cv['C_correct']} W={cv['W_wrong']}"
             f" | false-abstain на val (потолок rescue) = {fa_count} строк (~{fa_count / cv['N'] * 100:.1f}pp CAR)"]
    wins = []
    for name in BASES:
        key = f'{name}@best'
        if key not in summary:
            continue
        s = summary[key]
        vb, pb, ab = s['val'], s['proxy'], s['audit']
        dC = vb['C_correct'] - cv['C_correct']
        dCAR = vb['CorrectAutofillRate'] - cv['CorrectAutofillRate']
        dproxy = pb['CorrectAutofillRate'] - summary['champion@1.0']['proxy']['CorrectAutofillRate']
        win = dC > 0
        overfit = (dproxy > 1e-9
                   and ab['CorrectAutofillRate'] <= summary['champion@1.0']['audit']['CorrectAutofillRate'] + 1e-9)
        flags = (('  <- rescue растит C' if win else '')
                 + ('  [proxy↑/audit нет = teacher-overfit?]' if overfit else '')
                 + ('  [alpha упёрся в пол - расширить вниз]' if s.get('floor') else ''))
        lines.append(f"{name} @{s['alpha']:.3f}: ΔC{dC:+d} (ΔCAR{dCAR:+.4f}), rescue {s['resc_c']}/{s['resc_n']} "
                     f"prec {s['resc_prec']:.1%} | val AER_hi {vb['AutoErrorRate_hi']:.3f} | "
                     f"audit CAR {ab['CorrectAutofillRate']:.3f} W={ab['W_wrong']} "
                     f"AER_hi {ab['AutoErrorRate_hi']:.3f}{flags}")
        if win:
            wins.append(name)
    msg_win = (
        f'rescue РЕАЛЕН на val: {wins} - смотреть детально (судить по val; proxy↑ без audit '
        '= overfit; короновать только устойчивый ΔC, не шум)')
    msg_flat = (
        'rescue НЕ растит C под гейтом - rejected-пул не сепарабелен лучше '
        'base-rate, CAR-ось закрыта и с этой стороны')
    lines.append('ВЕРДИКТ: ' + (msg_win if wins else msg_flat))
    verdict = '\n'.join(lines)

    parts = [f'# Бизнес-policy: rescue false-abstain (демоут other) vs чемпион: {CHAMPION}',
             'CAR-ось со стороны pred_other->main (kNN-probe её не смотрел). Приём: p_other *= alpha, пере-argmax. '
             'READ-ONLY над сохранёнными вероятностями (champion + stack). alpha выбран на val (max C при AER_hi<=5%). '
             'alpha=1.0 = чемпион (sanity). Судить по val; proxy - teacher-overfit чек; audit (210) только санити',
             '## Политики (alpha=1.0 без rescue vs alpha* с rescue)', _table(cols, rows),
             '## Вердикт', '```\n' + verdict + '\n```']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n\n'.join(parts) + '\n')
    print(f'\nотчёт: {REPORT}\n')
    print(verdict)


if __name__ == '__main__':
    main()
