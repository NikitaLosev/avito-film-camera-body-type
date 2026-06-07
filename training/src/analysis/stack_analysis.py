"""OOF modality-stacking: мета поверх per-modality вероятностей (user-run: ~11 рефитов баз + мета)

Гипотеза (структурно НОВЫЙ механизм, не порог/гейт/вес): плоский чемпион линеен по сырым [tfidf|qwen|dino|pe]
и «несогласие экспертов» как фичу НЕ выражает. Учим 5 баз-моделей (champ-fusion + tfidf/qwen/dino/pe solo,
голова чемпиона), берём их per-modality вероятности + рассогласование (кто с кем спорит) и над ними учим
мета-голову -> 6 классов. Мета может ПЕРЕРАНЖИРОВАТЬ argmax (в отличие от acceptor'а, что только гейтил top1).
champ-fusion в базах = страховка: мета может откатиться к голосу чемпиона, не регрессируя ниже него.

Утечка airtight (зеркало acceptor.oof_champion, валидировано Plan-агентом): мета-фичи train = OOF базы через
StratifiedGroupKFold по leakage_group_id (каждая строка предсказана базой, что её НЕ видела); базы для val/test
фитятся на ПОЛНОМ train (val/test held-out). Джойн ВЕЗДЕ по item_id. Caveat стекинга: OOF-уверенности (80%
данных) чуть мягче full-train - неизбежно, как у acceptor'а. Таргет = согласие с Gemini -> риск teacher-overfit:
СУДИМ по val-CAR (не proxy-F1) + audit-санити; proxy↑ без audit = подгон под учителя (как word(1,2)). validation/
заморожен. Пишет reports/stack_analysis.md

Запуск: python training/src/stack_analysis.py  (~30-50 мин: 2 базы с tfidf × 5 фолдов - основная цена)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # training/
sys.path.insert(0, str(HERE.parent))         # training/src/
sys.path.insert(0, str(HERE))                # training/src/analysis/
from common import predictions_frame
from experiments import CHAMP_BLOCKS, QWEN_L2, TFIDF, _img
from features import build_features, load_blocks
from models import make_head
from multineg_analysis import _num, _table, business_at, dec_frame, pick_tau_tb, truth_frame
from paths import PREDICTIONS_DIR
from tracking.settings import REPORTS_DIR
from validation.settings import ABSTAIN, ALL_LABELS, SPLIT_PARQUET

CHAMPION = 'lrgrid__c1.0__none'
CHAMP_HEAD = ('logreg', {'C': 1.0, 'class_weight': None})   # голова всех баз = голова чемпиона
REPORT = REPORTS_DIR / 'stack_analysis.md'
N_SPLITS = 5
PCOLS = [f'p_{c}' for c in ALL_LABELS]      # каноничный порядок 6 классов
OTHER_J = ALL_LABELS.index(ABSTAIN)

# базы стекинга: champ = полная fusion (страховка-floor), остальные - solo-модальности (новый сигнал)
BASES = {
    'champ': CHAMP_BLOCKS,
    'tfidf': [TFIDF],
    'qwen': [QWEN_L2],
    'dino': [_img('dinov3', 'l2')],
    'pe': [_img('pe', 'l2')],
}
ORDER = ['champ', 'tfidf', 'qwen', 'dino', 'pe']
MODS = ['tfidf', 'qwen', 'dino', 'pe']      # модальности для рассогласования (без champ)
CHAMP_VAL = PREDICTIONS_DIR / 'lrgrid_c1_0_none_val_pred.parquet'
CHAMP_TEST = PREDICTIONS_DIR / 'lrgrid_c1_0_none_test_pred.parquet'


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def load_all() -> pd.DataFrame:
    """Фрейм чемпиона (текст+эмб+final_label+split) + leakage_group_id для OOF-групп"""
    data = load_blocks(CHAMP_BLOCKS)
    lg = pd.read_parquet(SPLIT_PARQUET, columns=['item_id', 'leakage_group_id'])
    return data.merge(lg, on='item_id', validate='one_to_one')


# ---------- OOF базы на train (без утечки) - зеркало acceptor.oof_champion, параметр = blocks ----------

def oof_base(data: pd.DataFrame, blocks: list) -> pd.DataFrame:
    """OOF контракт базы (blocks, голова чемпиона) на train: SGKF по leakage_group_id, relabel-by-item_id

    fit-фолд->split=train, oof-фолд->val, реальные val/test->test; build_features фитит tfidf/scaler ТОЛЬКО
    на fit-фолде -> oof-фолд трансформируется без утечки. asserts: disjoint, группа не разбита, покрытие 1:1
    """
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
            feats = build_features(data, blocks)
            head = make_head(*CHAMP_HEAD).fit(feats['X_train'], feats['y_train'])
            frames.append(predictions_frame(head, feats['X_val'], feats['id_val']))
    finally:
        data['split'] = orig_split
    oof = pd.concat(frames, ignore_index=True)
    assert oof['item_id'].nunique() == len(oof) == len(train), 'OOF не покрывает train 1:1'
    return oof


def fulltrain_base(data: pd.DataFrame, blocks: list) -> tuple:
    """База на ПОЛНОМ train -> предсказания (val, test); val/test held-out, без утечки"""
    feats = build_features(data, blocks)
    head = make_head(*CHAMP_HEAD).fit(feats['X_train'], feats['y_train'])
    return (predictions_frame(head, feats['X_val'], feats['id_val']),
            predictions_frame(head, feats['X_test'], feats['id_test']))


# ---------- мета-фичи: per-modality вероятности + рассогласование ----------

def stack_matrix(frames: dict, ids: np.ndarray) -> np.ndarray:
    """Из контрактов 5 баз (выровнять по item_id->ids) собрать мета-X: probs+conf+entropy на базу + 6 рассогласований"""
    P, top = {}, {}
    for name in ORDER:
        f = frames[name].set_index('item_id').reindex(ids)
        assert f['pred_label'].notna().all(), f'база {name} не покрывает все ids'
        P[name] = f[PCOLS].to_numpy(dtype=float)
        top[name] = f['pred_label'].to_numpy()
    cols = []
    for name in ORDER:                                       # 5 × (6 proba + conf + entropy)
        p = P[name]
        cols += [p, p.max(axis=1, keepdims=True), -(p * np.log(p + 1e-12)).sum(axis=1, keepdims=True)]
    champ_top = top['champ']                                 # рассогласование модальностей (новый сигнал)
    agree = np.sum([(top[m] == champ_top) for m in MODS], axis=0).astype(float)        # 0..4 согласны с champ
    distinct = np.array([len(set(t)) for t in zip(*[top[m] for m in MODS])], dtype=float)  # 1..4 разных argmax
    all_agree = (distinct == 1).astype(float)
    po = np.column_stack([P[name][:, OTHER_J] for name in ORDER])
    po_range = po.max(axis=1) - po.min(axis=1)               # спор «это other?» между базами
    ti = (top['tfidf'] != top['pe']).astype(float)           # текст vs фото (pe)
    tdd = (top['tfidf'] != top['dino']).astype(float)        # текст vs фото (dino)
    cols += [agree[:, None], distinct[:, None], all_agree[:, None], po_range[:, None], ti[:, None], tdd[:, None]]
    return np.column_stack(cols)


def fit_meta_lr(x, y):
    """Линейный якорь честности: StandardScaler + logreg (C=1.0, None - как калибровка чемпиона)"""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(solver='lbfgs', C=1.0, class_weight=None,
                                            max_iter=1000, random_state=42)).fit(x, y)


def fit_meta_hgb(x, y):
    """HGB ловит взаимодействия (несогласие × уверенность), capped против teacher-overfit + early stopping"""
    return HistGradientBoostingClassifier(
        max_depth=3, max_leaf_nodes=15, min_samples_leaf=300, learning_rate=0.05,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=42).fit(x, y)


# ---------- оценка ----------

def summarize(vp: pd.DataFrame, tp: pd.DataFrame, truth: pd.DataFrame) -> tuple:
    """tie-break tau на val -> (val, proxy-22k, audit-210) business-метрики"""
    vfr, tfr = dec_frame(vp, truth), dec_frame(tp, truth)
    afr = tfr[tfr['audit_label'].notna()].copy()
    tau = pick_tau_tb(vfr)
    return (business_at(vfr, 'final_label', tau),
            business_at(tfr, 'final_label', tau),
            business_at(afr, 'audit_label', tau))


def main() -> None:
    data = load_all()
    truth = truth_frame()
    train_ids = data.loc[data['split'] == 'train', 'item_id'].to_numpy()
    val_ids = data.loc[data['split'] == 'val', 'item_id'].to_numpy()
    test_ids = data.loc[data['split'] == 'test', 'item_id'].to_numpy()
    y_train = data.set_index('item_id')['final_label'].reindex(train_ids).to_numpy()

    print('OOF баз на train (leak-free, SGKF по leakage_group_id)...')
    oof_frames = {}
    for name in ORDER:
        print(f'  OOF base [{name}] ({N_SPLITS} рефитов)...')
        oof_frames[name] = oof_base(data, BASES[name])

    print('full-train базы для val/test...')
    champ_val, champ_test = pd.read_parquet(CHAMP_VAL), pd.read_parquet(CHAMP_TEST)   # champ = сохранён
    val_frames, test_frames = {'champ': champ_val}, {'champ': champ_test}
    for name in MODS:
        v, t = fulltrain_base(data, BASES[name])
        val_frames[name], test_frames[name] = v, t
        print(f'  full-train base [{name}] готов')

    Xtr = stack_matrix(oof_frames, train_ids)
    Xv = stack_matrix(val_frames, val_ids)
    Xt = stack_matrix(test_frames, test_ids)
    print(f'мета-фичей: {Xtr.shape[1]} (5 баз × (6 proba + conf + entropy) + 6 рассогласований)')

    meta_lr = fit_meta_lr(Xtr, y_train)
    meta_hgb = fit_meta_hgb(Xtr, y_train)
    schemes = {
        'champion': (champ_val, champ_test),
        'stack logreg': (predictions_frame(meta_lr, Xv, val_ids), predictions_frame(meta_lr, Xt, test_ids)),
        'stack HGB': (predictions_frame(meta_hgb, Xv, val_ids), predictions_frame(meta_hgb, Xt, test_ids)),
    }
    for nm, (vp, tp) in schemes.items():                     # персист стек-контрактов для пере-оценки
        if nm != 'champion':
            vp.to_parquet(PREDICTIONS_DIR / f'{_slug(nm)}_val_pred.parquet', index=False)
            tp.to_parquet(PREDICTIONS_DIR / f'{_slug(nm)}_test_pred.parquet', index=False)

    summary = {nm: summarize(vp, tp, truth) for nm, (vp, tp) in schemes.items()}

    cols = ['схема', 'CAR', 'AER', 'AER_hi', 'A', 'C', 'W']
    proj_titles = ['val / final_label (подбор tau)', 'test / final_label (proxy 22k)', 'test / audit_label (210)']
    proj_rows = {t: [] for t in proj_titles}
    for nm in schemes:
        for title, bm in zip(proj_titles, summary[nm]):
            proj_rows[title].append([nm, _num(bm['CorrectAutofillRate']), _num(bm['AutoErrorRate']),
                                     _num(bm['AutoErrorRate_hi']), bm['A_auto'], bm['C_correct'], bm['W_wrong']])

    cv, cp, ca = summary['champion']
    lines = [f"champion: val CAR {cv['CorrectAutofillRate']:.3f} W={cv['W_wrong']} | "
             f"proxy {cp['CorrectAutofillRate']:.3f} | "
             f"audit CAR {ca['CorrectAutofillRate']:.3f} W={ca['W_wrong']} AER_hi {ca['AutoErrorRate_hi']:.3f}"]
    wins = []
    for nm in ('stack logreg', 'stack HGB'):
        v, p, a = summary[nm]
        dC = v['CorrectAutofillRate'] - cv['CorrectAutofillRate']
        dW = v['W_wrong'] - cv['W_wrong']
        dproxy = p['CorrectAutofillRate'] - cp['CorrectAutofillRate']
        val_win = dC > 1e-9 or (dW < 0 and dC >= -1e-9)
        overfit = dproxy > 1e-9 and a['CorrectAutofillRate'] <= ca['CorrectAutofillRate'] + 1e-9
        tag = ('  <- val-выигрыш' if val_win else '') + ('  [proxy↑/audit нет = teacher-overfit?]' if overfit else '')
        lines.append(f"{nm}: val CAR {v['CorrectAutofillRate']:.3f} W={v['W_wrong']} (ΔCAR{dC:+.3f} ΔW{dW:+d}) | "
                     f"proxy {p['CorrectAutofillRate']:.3f}(Δ{dproxy:+.3f}) | "
                     f"audit CAR {a['CorrectAutofillRate']:.3f} W={a['W_wrong']} "
                     f"AER_hi {a['AutoErrorRate_hi']:.3f}{tag}")
        if val_win:
            wins.append(nm)
    msg_win = f'двигает val: {wins} - смотреть детально (судить по val, audit санити; proxy↑ без audit = overfit)'
    msg_flat = ('стекинг НЕ двигает val - сигнал рассогласования модальностей не извлекаем '
                'сверх fusion, B у потолка')
    lines.append('ВЕРДИКТ: ' + (msg_win if wins else msg_flat))
    verdict = '\n'.join(lines)

    parts = [f'# OOF modality-stacking vs чемпион: {CHAMPION}',
             'Мета над per-modality вероятностями 5 баз (champ-fusion + tfidf/qwen/dino/pe solo) + рассогласование '
             '(кто с кем спорит). Мета ПЕРЕРАНЖИРУЕТ argmax (не только гейт). OOF train (leak-free, SGKF по '
             'leakage_group_id), базы val/test на полном train. Operating point - tie-break (max C/min W) на val. '
             'Судить по val (19k), proxy - teacher-overfit чек, audit (210) санити - не короновать по нему',
             '## Сравнение в 3 проекциях']
    for title in proj_titles:
        parts += [f'### {title}', _table(cols, proj_rows[title])]
    parts += ['## Вердикт', '```\n' + verdict + '\n```']

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n\n'.join(parts) + '\n')
    print(f'\nотчёт: {REPORT}\n')
    print(verdict)


if __name__ == '__main__':
    main()
