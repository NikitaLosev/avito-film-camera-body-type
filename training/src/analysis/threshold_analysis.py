"""Per-class пороги отказа + гигиена метрик чемпиона (read-only, без переобучения)

Берёт сохранённые предсказания чемпиона и проверяет два дешёвых рычага на ГРАНИЦЕ отказа:
1. гигиена: raw (без отказа, tau:0) vs post-tau метрики - сейчас в лог идут только post-tau
2. per-class пороги вместо одного глобального tau - расширяют ли фронтир CAR/AER

Гейт проекта - ПУЛИНГОВЫЙ Wilson-hi (AER_hi<=5%), его и оптимизируем координатным спуском.
per-class-cap (каждый класс под свой AER_hi) даём лишь как наивный бейзлайн: он НЕ гарантирует
пулинговый гейт. Пороги подбираем на val/final_label, оцениваем в 3 проекции (val, proxy-test,
audit-human с Wilson-CI), потому что per-class AER на 210 audit статистически пуст. validation/
заморожен - зовём его чистые функции, внутрь не лезем. Read-only: читает parquet, пишет md

Запуск: python training/src/threshold_analysis.py
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
from validation.evaluate import load_eval_frame
from validation.metrics import (
    business_metrics,
    classification_metrics,
    pick_operating_point,
    threshold_sweep,
    wilson_ci,
)
from validation.settings import ABSTAIN, AER_CAP, MAIN_SET

CHAMPION = 'lrgrid__c1.0__none'
REPORT = REPORTS_DIR / 'threshold_analysis_champion.md'


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def resolve_paths(name: str = CHAMPION) -> tuple:
    """Пути предсказаний чемпиона по слагу имени"""
    slug = _slug(name)
    return PREDICTIONS_DIR / f'{slug}_val_pred.parquet', PREDICTIONS_DIR / f'{slug}_test_pred.parquet'


def choose_tau(val_frame: pd.DataFrame) -> float:
    """Глобальный рабочий порог - подбор на val/final_label под пулинговым AER_hi<=5% (штатный путь)"""
    sweep = threshold_sweep(val_frame, 'final_label', 'pred_label', 'confidence')
    return pick_operating_point(sweep)['tau']


def apply_tau_by_class(frame: pd.DataFrame, tau_by_class: dict) -> pd.DataFrame:
    """Обобщение apply_tau: отказ если confidence < tau[pred_label] (для своего класса)

    other_unknown и отсутствующие классы -> map даёт NaN -> fillna(0) -> порог 0 -> строка не трогается.
    tau=inf полностью отключает класс (confidence < inf всегда -> отказ)
    """
    out = frame.copy()
    thr = out['pred_label'].map(tau_by_class).fillna(0.0)
    out['pred_label'] = out['pred_label'].where(out['confidence'].fillna(0) >= thr, ABSTAIN)
    return out


def business_for(frame: pd.DataFrame, truth_col: str, tau_by_class: dict) -> dict:
    """Бизнес-метрики при заданном векторе порогов (единый путь для global и per-class)"""
    pred = apply_tau_by_class(frame, tau_by_class)['pred_label']
    return business_metrics(frame[truth_col], pred)


# ---------- подбор per-class порогов ----------

def _class_table(frame: pd.DataFrame, c: str, truth_col: str) -> dict:
    """Для строк, предсказанных классом c: отсортированные confidence + суффиксные суммы ошибок

    Позволяет за O(log n) получить (A_c, W_c) при любом пороге t: A=#(conf>=t), W=#(conf>=t и ошибка)
    """
    sub = frame.loc[frame['pred_label'] == c, ['confidence', truth_col]].dropna(subset=['confidence'])
    conf = sub['confidence'].to_numpy()
    wrong = (sub[truth_col].to_numpy() != c).astype(int)
    order = np.argsort(conf, kind='mergesort')
    conf_s, wrong_s = conf[order], wrong[order]
    suff = np.append(np.cumsum(wrong_s[::-1])[::-1], 0)         # suff[i] = sum ошибок в conf_s[i:], suff[n]=0
    cands = np.append(np.unique(conf_s), np.inf)                # inf = отключить класс
    return {'conf_s': conf_s, 'suff': suff, 'n': len(conf_s), 'cands': cands}


def _aw(table: dict, t: float) -> tuple:
    """(A, W) для порога t по предрасчитанной таблице класса"""
    idx = int(np.searchsorted(table['conf_s'], t, side='left'))  # первая позиция с conf>=t
    return table['n'] - idx, int(table['suff'][idx])


def cap_tau(table: dict, aer_cap: float = AER_CAP) -> tuple:
    """Наивный per-class-cap: наименьший порог (макс корректных) при per-class AER_hi<=cap"""
    best_t, best_c, found = float('inf'), -1, False
    for t in table['cands']:
        a, w = _aw(table, t)
        if a == 0:
            continue
        if wilson_ci(w, a)[1] <= aer_cap:
            correct = a - w
            if correct > best_c:
                best_c, best_t, found = correct, float(t), True
    return best_t, found


def pooled_tau_by_class(frame: pd.DataFrame, truth_col: str, global_tau: float,
                        tables: dict, aer_cap: float = AER_CAP) -> dict:
    """Координатный спуск под ПУЛИНГОВЫЙ AER_hi<=cap - оптимизируем тот гейт, что репортим

    Старт - глобальный tau (пулингово-допустимая точка). Каждый класс двигаем по своим кандидатам
    при фиксированных остальных, принимаем ход с макс пулинговым CAR при допустимом пулинговом AER_hi
    """
    n_total = len(frame)
    tau = {c: float(global_tau) for c in MAIN_SET}

    def pooled_car(tau_map: dict):
        a = sum(_aw(tables[c], tau_map[c])[0] for c in MAIN_SET)
        w = sum(_aw(tables[c], tau_map[c])[1] for c in MAIN_SET)
        if a == 0 or wilson_ci(w, a)[1] > aer_cap:
            return None
        return (a - w) / n_total

    assert pooled_car(tau) is not None, 'глобальный tau пулингово недопустим - проверь choose_tau'
    improved = True
    while improved:
        improved = False
        for c in MAIN_SET:
            best_t, best_car = tau[c], pooled_car(tau)
            for t in tables[c]['cands']:
                trial = dict(tau)
                trial[c] = float(t)
                car = pooled_car(trial)
                if car is not None and car > best_car + 1e-12:
                    best_t, best_car = float(t), car
            if best_t != tau[c]:
                tau[c] = best_t
                improved = True
    assert pooled_car(tau) is not None, 'итоговый вектор порогов пулингово недопустим'
    return tau


# ---------- рендер markdown ----------

def _num(x) -> str:
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'


def _tau(t: float) -> str:
    return 'inf (off)' if np.isinf(t) else f'{t:.3f}'


def _table(headers: list, rows: list) -> str:
    head = '| ' + ' | '.join(headers) + ' |'
    sep = '| ' + ' | '.join('---' for _ in headers) + ' |'
    body = '\n'.join('| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows)
    return '\n'.join([head, sep, body])


def _hygiene_rows(val: pd.DataFrame, g_map: dict) -> list:
    """raw (без отказа) vs post-tau на val/final_label - показать, что лог сейчас только post-tau"""
    raw_pred = val['pred_label']
    post_pred = apply_tau_by_class(val, g_map)['pred_label']
    raw_cls = classification_metrics(val['final_label'], raw_pred)
    post_cls = classification_metrics(val['final_label'], post_pred)
    raw_bm = business_metrics(val['final_label'], raw_pred)
    post_bm = business_metrics(val['final_label'], post_pred)
    return [
        ['macro_f1', _num(raw_cls['macro_f1']), _num(post_cls['macro_f1'])],
        ['accuracy', _num(raw_cls['accuracy']), _num(post_cls['accuracy'])],
        ['CAR', _num(raw_bm['CorrectAutofillRate']), _num(post_bm['CorrectAutofillRate'])],
        ['AER', _num(raw_bm['AutoErrorRate']), _num(post_bm['AutoErrorRate'])],
        ['AER_hi', _num(raw_bm['AutoErrorRate_hi']), _num(post_bm['AutoErrorRate_hi'])],
        ['coverage', _num(raw_bm['coverage']), _num(post_bm['coverage'])],
    ]


def _view_table(frame: pd.DataFrame, truth_col: str, schemes: list) -> str:
    rows = []
    for label, tau_map in schemes:
        bm = business_for(frame, truth_col, tau_map)
        ci = f"[{bm['CorrectAutofillRate_lo']:.3f}, {bm['CorrectAutofillRate_hi']:.3f}]"
        rows.append([label, _num(bm['CorrectAutofillRate']), ci, _num(bm['AutoErrorRate']),
                     _num(bm['AutoErrorRate_hi']), bm['A_auto'], bm['W_wrong']])
    return _table(['схема', 'CAR', 'CAR 95% CI', 'AER', 'AER_hi', 'A', 'W'], rows)


def build_verdict(val: pd.DataFrame, proxy: pd.DataFrame, audit: pd.DataFrame,
                  g_map: dict, p_map: dict) -> str:
    """Вердикт переноса: выигрыш на val + держится ли на proxy-22k и не ломает ли audit"""
    def car(frame, truth, m):
        return business_for(frame, truth, m)['CorrectAutofillRate']

    dv = car(val, 'final_label', p_map) - car(val, 'final_label', g_map)
    dp = car(proxy, 'final_label', p_map) - car(proxy, 'final_label', g_map)
    da = car(audit, 'audit_label', p_map) - car(audit, 'audit_label', g_map)
    audit_aer_hi = business_for(audit, 'audit_label', p_map)['AutoErrorRate_hi']
    warn = (da <= 0) or (audit_aer_hi > AER_CAP)
    tail = ('ВНИМАНИЕ: выигрыш НЕ переносится на audit (CAR не вырос или AER_hi>5%) - '
            'это val-only артефакт под учителя, НЕ катить'
            if warn else
            'выигрыш держится на proxy и не ломает audit - кандидат на follow-up (персист порогов)')
    return (f'val   CAR Δ (per-class - global): {dv:+.3f}\n'
            f'proxy CAR Δ (test/final, 22k):     {dp:+.3f}\n'
            f'audit CAR Δ (test/human, 210):     {da:+.3f}  | audit AER_hi per-class = {audit_aer_hi:.3f}\n'
            f'ВЕРДИКТ: {tail}')


def render(global_tau: float, hygiene: list, thr_rows: list, views: list, verdict: str) -> str:
    parts = [
        f'# Per-class пороги + гигиена метрик: {CHAMPION}',
        f'Глобальный порог tau = {global_tau:.3f} (подбор на val/final_label под пулинговым AER_hi<=5%). '
        'per-class пороги подобраны там же, оцениваются в 3 проекции. audit (210) - directional, с Wilson-CI',
        '## 1. Гигиена: raw (без отказа, tau:0) vs post-tau, на val/final_label',
        'raw автозаполняет argmax всегда (coverage=1), post-tau применяет глобальный порог. '
        'Сейчас в лог идут ТОЛЬКО post-tau - raw_macro_f1/raw_acc отдельно показывают качество модели до отказа',
        _table(['метрика', 'raw (no-abstain)', 'post-tau (global)'], hygiene),
        '## 2. Пороги по классам',
        'tau_cap - наивный (каждый класс под свой AER_hi<=5%, НЕ гарантирует пулинг); '
        'tau_pooled - основной (координатный спуск под пулинговый AER_hi<=5%)',
        _table(['класс', 'tau_global', 'tau_cap', 'tau_pooled', 'cap feasible'], thr_rows),
        '## 3. Сравнение схем в 3 проекциях',
        'val/final_label - где выигрыш заявлен; test/final_label - переносится ли в распределении (22k); '
        'test/audit_label - реальная бизнес-метрика (210 human, CI широк - читать как directional)',
    ]
    for title, table_md in views:
        parts += [f'### {title}', table_md]
    parts += ['## 4. Вердикт переноса', '```\n' + verdict + '\n```']
    return '\n\n'.join(parts) + '\n'


def main():
    val_path, test_path = resolve_paths()
    val = load_eval_frame(val_path, 'val', 'final_label')
    proxy = load_eval_frame(test_path, 'test', 'final_label')
    audit = load_eval_frame(test_path, 'test', 'audit_label')

    global_tau = choose_tau(val)
    g_map = {c: float(global_tau) for c in MAIN_SET}
    tables = {c: _class_table(val, c, 'final_label') for c in MAIN_SET}

    cap_map, cap_feas = {}, {}
    for c in MAIN_SET:
        if tables[c]['n'] == 0:                          # класс не предсказан на val -> порог не нужен
            cap_map[c], cap_feas[c] = 0.0, True
        else:
            cap_map[c], cap_feas[c] = cap_tau(tables[c])
    p_map = pooled_tau_by_class(val, 'final_label', global_tau, tables)

    hygiene = _hygiene_rows(val, g_map)
    thr_rows = [[c, _tau(global_tau), _tau(cap_map[c]), _tau(p_map[c]),
                 'да' if cap_feas[c] else 'НЕТ'] for c in MAIN_SET]
    schemes = [('global', g_map), ('per-class-cap', cap_map), ('per-class-pooled', p_map)]
    views = [
        ('val / final_label (подбор)', _view_table(val, 'final_label', schemes)),
        ('test / final_label (proxy 22k)', _view_table(proxy, 'final_label', schemes)),
        ('test / audit_label (human 210)', _view_table(audit, 'audit_label', schemes)),
    ]
    verdict = build_verdict(val, proxy, audit, g_map, p_map)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(global_tau, hygiene, thr_rows, views, verdict))
    print(f'отчёт: {REPORT}\n')
    print('per-class пороги (pooled):', {c: round(p_map[c], 3) if not np.isinf(p_map[c]) else 'off'
                                         for c in MAIN_SET})
    print(verdict)


if __name__ == '__main__':
    main()
