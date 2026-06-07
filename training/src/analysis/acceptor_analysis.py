"""OOF acceptor: мета-голова отказа поверх чемпиона (user-run: 5 рефитов чемпиона + acceptor)

Тип-классификатор чемпиона НЕ трогаем. Поверх учим acceptor «доверять ли top-1» на фичах, которых
плоская линейка + один порог не выразит (взаимодействия high-conf И lot-flag -> reject). OOF на train
через StratifiedGroupKFold по leakage_group_id -> acceptor учится на ЧЕСТНЫХ (не in-sample) уверенностях.
Сравниваем matched-frontier acceptor-гейта vs single-tau чемпиона на val + audit. validation/ заморожен -
зовём чистые функции; контракт НЕ зовём (acceptor ломает pred==argmax(p)) -> business_metrics напрямую.
Решение по AUDIT, не по AUC (reject-rate ~4%, дисбаланс). Пишет reports/acceptor_analysis.md

Запуск: python training/src/acceptor_analysis.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # training/
sys.path.insert(0, str(HERE.parent))         # training/src/
sys.path.insert(0, str(HERE))                # training/src/analysis/
from common import predictions_frame
from experiments import CHAMP_BLOCKS
from features import build_features, load_blocks, text_flags_matrix
from models import make_head
from paths import PREDICTIONS_DIR
from tracking.settings import REPORTS_DIR
from validation.metrics import business_metrics
from validation.settings import ABSTAIN, AER_CAP, ALL_LABELS, AUDIT_SAMPLE, EDA_DATASET, MAIN_SET, SPLIT_PARQUET

CHAMPION = 'lrgrid__c1.0__none'           # слаг должен совпадать с сохранёнными предсказаниями чемпиона
CHAMP_HEAD = ('logreg', {'C': 1.0, 'class_weight': None})
REPORT = REPORTS_DIR / 'acceptor_analysis.md'
N_SPLITS = 5
PCOLS = [f'p_{c}' for c in ALL_LABELS]    # каноничный порядок 6 классов
MAIN = list(MAIN_SET)


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


# ---------- загрузка ----------

def load_all() -> pd.DataFrame:
    """Фрейм чемпиона + leakage_group_id (OOF-группы) + kb_model (has_kb) + audit_label"""
    data = load_blocks(CHAMP_BLOCKS)
    lg = pd.read_parquet(SPLIT_PARQUET, columns=['item_id', 'leakage_group_id'])
    kb = pd.read_parquet(EDA_DATASET, columns=['item_id', 'kb_model'])
    au = pd.read_parquet(AUDIT_SAMPLE, columns=['item_id', 'audit_label'])
    data = data.merge(lg, on='item_id', validate='one_to_one')
    data = data.merge(kb, on='item_id', validate='one_to_one')
    data = data.merge(au, on='item_id', how='left', validate='one_to_one')
    return data


# ---------- OOF чемпион на train (без утечки) ----------

def oof_champion(data: pd.DataFrame) -> pd.DataFrame:
    """OOF контракт-предсказания чемпиона на train: StratifiedGroupKFold по leakage_group_id

    Релейблим колонку split (fit-фолд->train, oof-фолд->val, реальные val/test->test) и зовём
    build_features - tfidf/scaler фитятся ТОЛЬКО на fit-фолде, oof-фолд трансформируется без утечки
    """
    head_name, head_params = CHAMP_HEAD
    train = data[data['split'] == 'train'].reset_index(drop=True)
    orig_split = data['split'].to_numpy().copy()
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    frames = []
    try:
        for fit_pos, oof_pos in sgkf.split(train, train['final_label'], groups=train['leakage_group_id']):
            assert set(fit_pos).isdisjoint(oof_pos), 'fit и oof пересеклись'
            fit_g = set(train.iloc[fit_pos]['leakage_group_id'])
            oof_g = set(train.iloc[oof_pos]['leakage_group_id'])
            assert not (fit_g & oof_g), 'leakage-группа разбита между fit и oof'
            fit_ids = set(train.iloc[fit_pos]['item_id'])
            oof_ids = set(train.iloc[oof_pos]['item_id'])
            relab = np.where(data['item_id'].isin(fit_ids), 'train',
                             np.where(data['item_id'].isin(oof_ids), 'val', 'test'))
            data['split'] = relab
            feats = build_features(data, CHAMP_BLOCKS)
            head = make_head(head_name, head_params).fit(feats['X_train'], feats['y_train'])
            frames.append(predictions_frame(head, feats['X_val'], feats['id_val']))
    finally:
        data['split'] = orig_split
    oof = pd.concat(frames, ignore_index=True)
    assert oof['item_id'].nunique() == len(oof) == len(train), 'OOF не покрывает train 1:1'
    return oof


# ---------- фичи acceptor ----------

def acceptor_features(preds: pd.DataFrame, ctx: pd.DataFrame) -> tuple:
    """Фичи мета-головы по контракт-предсказаниям + контексту. Возврат: (X, top1, item_id) по строкам preds"""
    m = preds.merge(ctx[['item_id', 'title', 'description', 'kb_model']], on='item_id', how='left')
    p = m[PCOLS].to_numpy(dtype=float)
    conf = p.max(axis=1)
    srt = np.sort(p, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
    txt = (m['title'].fillna('') + ' ' + m['description'].fillna('')).str.strip().to_numpy()
    flags = text_flags_matrix(txt)
    title_len = m['title'].fillna('').str.len().to_numpy()
    desc_len = m['description'].fillna('').str.len().to_numpy()
    has_kb = m['kb_model'].notna().astype(int).to_numpy()
    top1 = m['pred_label'].to_numpy()
    onehot = np.stack([(top1 == c).astype(int) for c in ALL_LABELS], axis=1)
    X = np.column_stack([p, conf, margin, entropy, flags, title_len, desc_len, has_kb, onehot])
    return X, top1, m['item_id'].to_numpy()


# ---------- acceptor-головы (две, без свипа гиперов) ----------

def fit_logreg_acceptor(x, y):
    """Линейный якорь честности: StandardScaler (фит на train-OOF) + logreg(balanced)"""
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(solver='lbfgs', C=1.0, class_weight='balanced',
                             max_iter=1000, random_state=42).fit(scaler.transform(x), y)
    pos = list(clf.classes_).index(True)
    return lambda z: clf.predict_proba(scaler.transform(z))[:, pos]


def fit_hgb_acceptor(x, y):
    """HGB ловит взаимодействия (суть), capped против teacher-overfit + early stopping (leak-free)"""
    clf = HistGradientBoostingClassifier(
        max_depth=3, max_leaf_nodes=15, min_samples_leaf=300, learning_rate=0.05,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=42).fit(x, y)
    pos = list(clf.classes_).index(True)
    return lambda z: clf.predict_proba(z)[:, pos]


# ---------- сборка фрейма решения + matched-frontier ----------

def decision_frame(preds: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """item_id, final_label, audit_label, champion_top1, confidence (под sweep)"""
    fr = preds[['item_id', 'pred_label', 'confidence']].rename(columns={'pred_label': 'champion_top1'})
    return fr.merge(ctx[['item_id', 'final_label', 'audit_label']], on='item_id', how='left')


def add_score(fr: pd.DataFrame, ids, score, top1, col: str) -> pd.DataFrame:
    """Добавить accept_prob к фрейму по item_id; для top1 не из MAIN -> 0 (всё равно отказ)"""
    s = np.where(np.isin(top1, MAIN), score, 0.0)
    return fr.merge(pd.DataFrame({'item_id': ids, col: s}), on='item_id', how='left')


def choose_tau(fr: pd.DataFrame, conf_col: str, aer_cap: float = AER_CAP) -> float:
    """Operating point: ТОЧНЫЕ кандидаты + tie-break (max C / min W / min AER_hi) под AER_hi<=cap

    Отличие от frozen pick_operating_point (max CAR -> первая точка = tau:0): при РАВНОМ CAR берём
    наименьший РИСК (меньше ошибочных автозаполнений). Это и даёт acceptor'у/порогу шанс срезать W без
    потери CAR (и так взять строгий Wilson-гейт). validation/ не трогаем - зовём business_metrics на
    точных кандидатах (уникальные score). Подбор всегда на val/final_label
    """
    conf = fr[conf_col].fillna(0).to_numpy()
    rows = []
    for tau in np.unique(conf):
        pred = fr['champion_top1'].where(conf >= tau, ABSTAIN)
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


def business_at(fr: pd.DataFrame, conf_col: str, truth_col: str, tau: float) -> dict:
    pred = fr['champion_top1'].where(fr[conf_col].fillna(0) >= tau, ABSTAIN)
    return business_metrics(fr[truth_col], pred)


# ---------- рендер ----------

def _num(x) -> str:
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'


def _table(headers: list, rows: list) -> str:
    head = '| ' + ' | '.join(headers) + ' |'
    sep = '| ' + ' | '.join('---' for _ in headers) + ' |'
    body = '\n'.join('| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows)
    return '\n'.join([head, sep, body])


def _proj_rows(val_fr, test_fr, audit_fr, schemes: dict) -> dict:
    """По каждой проекции (val/final, test/final, test/audit) строки схем с CAR/AER/AER_hi/A/W"""
    proj = {
        'val / final_label (подбор tau)': (val_fr, 'final_label'),
        'test / final_label (proxy 22k)': (test_fr, 'final_label'),
        'test / audit_label (human 210)': (audit_fr, 'audit_label'),
    }
    out = {}
    for title, (fr, truth) in proj.items():
        rows = []
        for name, (conf_col, tau) in schemes.items():
            bm = business_at(fr, conf_col, truth, tau)
            ci = f"[{bm['CorrectAutofillRate_lo']:.3f}, {bm['CorrectAutofillRate_hi']:.3f}]"
            rows.append([name, _num(bm['CorrectAutofillRate']), ci, _num(bm['AutoErrorRate']),
                         _num(bm['AutoErrorRate_hi']), bm['A_auto'], bm['C_correct'], bm['W_wrong']])
        out[title] = _table(['схема', 'CAR', 'CAR 95% CI', 'AER', 'AER_hi', 'A', 'C', 'W'], rows)
    return out


def build_verdict(val_fr, audit_fr, schemes: dict) -> str:
    """tie-break selector: ищем схему, что при том же CAR режет W (а на audit - берёт строгий гейт)"""
    def at(fr, conf, truth, tau):
        return business_at(fr, conf, truth, tau)

    bconf, btau = schemes['baseline single-tau']
    bv, ba = at(val_fr, bconf, 'final_label', btau), at(audit_fr, bconf, 'audit_label', btau)
    lines = [f"baseline: val CAR {bv['CorrectAutofillRate']:.3f} C={bv['C_correct']} W={bv['W_wrong']} | "
             f"audit CAR {ba['CorrectAutofillRate']:.3f} W={ba['W_wrong']} AER_hi {ba['AutoErrorRate_hi']:.3f}"]
    crossed = []
    for name, (conf, tau) in schemes.items():
        if name == 'baseline single-tau':
            continue
        v, a = at(val_fr, conf, 'final_label', tau), at(audit_fr, conf, 'audit_label', tau)
        dC = v['CorrectAutofillRate'] - bv['CorrectAutofillRate']
        dW = v['W_wrong'] - bv['W_wrong']
        gate = a['AutoErrorRate_hi'] <= 0.05
        note = ('СТРОГИЙ ГЕЙТ ВЗЯТ (audit AER_hi<=5%)' if gate
                else ('val W срезан без потери CAR' if (dW < 0 and dC >= -1e-9) else 'без сдвига'))
        lines.append(f"{name}: val CAR {v['CorrectAutofillRate']:.3f} (ΔCAR{dC:+.3f}, ΔW{dW:+d}) | "
                     f"audit CAR {a['CorrectAutofillRate']:.3f} W={a['W_wrong']} "
                     f"AER_hi {a['AutoErrorRate_hi']:.3f} -> {note}")
        if gate:
            crossed.append(name)
    lines.append('ВЕРДИКТ: ' + (
        f'строгий audit-гейт берут -> {crossed}' if crossed
        else 'строгий гейт не взят даже с tie-break; W не режется без потери CAR'))
    return '\n'.join(lines)


def render(schemes, projections, verdict) -> str:
    parts = [
        f'# OOF acceptor поверх чемпиона: {CHAMPION}',
        'Тип-классификатор не тронут; acceptor меняет только гейт отказа. Operating point - **tie-break** '
        'селектор (точные кандидаты, max C -> min W -> min AER_hi), не max-CAR-первый: при равном CAR берёт '
        'минимум ошибочных автозаполнений. Подбор на val/final_label под AER_hi<=5%, оценка в 3 проекции. '
        'Вопрос: режет ли acceptor W без потери CAR и берёт ли строгий audit-гейт (AER_hi<=5%)',
        '## Пороги схем (подбор на val)',
        _table(['схема', 'gate-сигнал', 'tau'], [[n, c, f'{t:.3f}'] for n, (c, t) in schemes.items()]),
        '## Сравнение схем в 3 проекциях',
    ]
    for title, table_md in projections.items():
        parts += [f'### {title}', table_md]
    parts += ['## Вердикт', '```\n' + verdict + '\n```']
    return '\n\n'.join(parts) + '\n'


def main():
    data = load_all()
    ctx = data[['item_id', 'title', 'description', 'kb_model', 'final_label', 'audit_label']]

    print('OOF чемпиона на train (5 рефитов)...')
    oof = oof_champion(data)

    # acceptor учится на in-domain train (OOF top1 in MAIN), target = (top1 == final_label)
    x_tr, top1_tr, ids_tr = acceptor_features(oof, ctx)
    final_tr = ctx.set_index('item_id')['final_label'].reindex(ids_tr).to_numpy()
    in_main = np.isin(top1_tr, MAIN)
    x_fit, y_fit = x_tr[in_main], (top1_tr[in_main] == final_tr[in_main])
    print(f'acceptor train: in-domain {in_main.sum()} строк, reject-rate {1 - y_fit.mean():.1%}')
    score_lr = fit_logreg_acceptor(x_fit, y_fit)
    score_hgb = fit_hgb_acceptor(x_fit, y_fit)

    # full-train предсказания чемпиона на val/test (сохранены)
    val_preds = pd.read_parquet(PREDICTIONS_DIR / f'{_slug(CHAMPION)}_val_pred.parquet')
    test_preds = pd.read_parquet(PREDICTIONS_DIR / f'{_slug(CHAMPION)}_test_pred.parquet')
    xv, top1v, idv = acceptor_features(val_preds, ctx)
    xt, top1t, idt = acceptor_features(test_preds, ctx)

    val_fr = decision_frame(val_preds, ctx)
    val_fr = add_score(val_fr, idv, score_lr(xv), top1v, 'accept_lr')
    val_fr = add_score(val_fr, idv, score_hgb(xv), top1v, 'accept_hgb')
    test_fr = decision_frame(test_preds, ctx)
    test_fr = add_score(test_fr, idt, score_lr(xt), top1t, 'accept_lr')
    test_fr = add_score(test_fr, idt, score_hgb(xt), top1t, 'accept_hgb')
    audit_fr = test_fr[test_fr['audit_label'].notna()].copy()
    # сохраняем scored-фреймы: будущий пере-выбор оператора - мгновенно, без повторного OOF
    val_fr.to_parquet(PREDICTIONS_DIR / 'acceptor_val_scored.parquet', index=False)
    test_fr.to_parquet(PREDICTIONS_DIR / 'acceptor_test_scored.parquet', index=False)

    schemes = {
        'baseline single-tau': ('confidence', choose_tau(val_fr, 'confidence')),
        'acceptor logreg': ('accept_lr', choose_tau(val_fr, 'accept_lr')),
        'acceptor HGB': ('accept_hgb', choose_tau(val_fr, 'accept_hgb')),
    }
    projections = _proj_rows(val_fr, test_fr, audit_fr, schemes)
    verdict = build_verdict(val_fr, audit_fr, schemes)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(schemes, projections, verdict))
    print(f'\nотчёт: {REPORT}\n')
    print(verdict)


if __name__ == '__main__':
    main()
