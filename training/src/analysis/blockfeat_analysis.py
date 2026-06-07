"""CAR-side рычаги: block-weight (α на блок) + tfidf title/desc раздельно (user-run, фичи раз)

Для L2-logreg масштаб блока = отдельная регуляризация блока (относительные веса). Фичи чемпиона строим
один раз, дальше дёшево: block-weight - масштабируем колонки X (X @ diag) + рефит logreg; title/desc -
две tfidf на title/desc + переиспользуем emb-часть X. Сравнение matched-frontier (tie-break) vs чемпион
на val + audit. Прайор слабый (CAR у потолка, tau=0 accept-all) - замеряем для полноты. validation/ заморожен

Запуск: python training/src/blockfeat_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import diags, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # training/
sys.path.insert(0, str(HERE.parent))         # training/src/
sys.path.insert(0, str(HERE))                # training/src/analysis/
from common import TFIDF_PARAMS, predictions_frame
from experiments import CHAMP_BLOCKS
from features import build_features, load_blocks
from models import make_head
from multineg_analysis import _num, _table, business_at, dec_frame, pick_tau_tb, truth_frame
from paths import PREDICTIONS_DIR
from tracking.settings import REPORTS_DIR

CHAMPION = 'lrgrid__c1.0__none'
CHAMP_HEAD = ('logreg', {'C': 1.0, 'class_weight': None})
REPORT = REPORTS_DIR / 'blockfeat_analysis.md'
QWEN, DINO, PE = 1024, 384, 1024   # размерности emb-блоков (порядок specs: tfidf, qwen, dino, pe)

# block-weight: по одному блоку относительно baseline (для L2 важны относительные веса)
BLOCK_GRID = [
    ('baseline (1,1,1,1)', (1, 1, 1, 1)),
    ('tfidf x0.5', (0.5, 1, 1, 1)), ('tfidf x2', (2, 1, 1, 1)),
    ('qwen x0.5', (1, 0.5, 1, 1)), ('qwen x2', (1, 2, 1, 1)),
    ('dino x0.5', (1, 1, 0.5, 1)), ('dino x2', (1, 1, 2, 1)),
    ('pe x0.5', (1, 1, 1, 0.5)), ('pe x2', (1, 1, 1, 2)), ('pe x4', (1, 1, 1, 4)),
]


def fit_eval(X_tr, y_tr, X_v, id_v, X_t, id_t):
    """Фит logreg-чемпиона на данной X, вернуть (val_preds, test_preds) контракт-фреймы"""
    head = make_head(*CHAMP_HEAD).fit(X_tr, y_tr)
    return predictions_frame(head, X_v, id_v), predictions_frame(head, X_t, id_t)


def scale_blocks(X, t_dim, alphas):
    """X @ diag(весов): колонки блока умножаются на α (tfidf, qwen, dino, pe)"""
    d = np.ones(X.shape[1])
    a, b, c, e = alphas
    d[:t_dim] = a
    d[t_dim:t_dim + QWEN] = b
    d[t_dim + QWEN:t_dim + QWEN + DINO] = c
    d[t_dim + QWEN + DINO:] = e
    return (X @ diags(d)).tocsr()


def title_desc_features(data, feats, t_dim):
    """Заменить объединённый tfidf на ДВА (title, desc), emb-часть X переиспользовать"""
    parts = {s: data[data['split'] == s] for s in ('train', 'val', 'test')}
    vt = TfidfVectorizer(**TFIDF_PARAMS).fit(parts['train']['title'].fillna(''))
    vd = TfidfVectorizer(**TFIDF_PARAMS).fit(parts['train']['description'].fillna(''))
    out = {}
    for s in ('train', 'val', 'test'):
        emb = feats[f'X_{s}'][:, t_dim:]              # qwen+dino+pe (уже l2/standard) из чемпион-X
        xt = vt.transform(parts[s]['title'].fillna(''))
        xd = vd.transform(parts[s]['description'].fillna(''))
        out[s] = hstack([xt, xd, emb]).tocsr()
    return out


def summarize(vp, tp, truth):
    vfr, tfr = dec_frame(vp, truth), dec_frame(tp, truth)
    afr = tfr[tfr['audit_label'].notna()].copy()
    tau = pick_tau_tb(vfr)
    return business_at(vfr, 'final_label', tau), business_at(afr, 'audit_label', tau)


def main():
    data = load_blocks(CHAMP_BLOCKS)
    truth = truth_frame()
    feats = build_features(data, CHAMP_BLOCKS)
    t_dim = len(feats['vectorizer'].vocabulary_)
    print(f'tfidf-словарь: {t_dim} | X-ширина: {feats["X_train"].shape[1]} (= {t_dim}+{QWEN}+{DINO}+{PE})')

    rows = []
    champ = (pd.read_parquet(PREDICTIONS_DIR / 'lrgrid_c1_0_none_val_pred.parquet'),
             pd.read_parquet(PREDICTIONS_DIR / 'lrgrid_c1_0_none_test_pred.parquet'))
    cv, ca = summarize(*champ, truth)

    def add(name, vp, tp):
        v, a = summarize(vp, tp, truth)
        rows.append([name, _num(v['CorrectAutofillRate']), v['W_wrong'],
                     _num(a['CorrectAutofillRate']), a['W_wrong'], _num(a['AutoErrorRate_hi'])])
        return v, a

    add('champion (saved)', *champ)
    for name, alphas in BLOCK_GRID:
        Xw = scale_blocks(feats['X_train'], t_dim, alphas)
        Xv = scale_blocks(feats['X_val'], t_dim, alphas)
        Xt = scale_blocks(feats['X_test'], t_dim, alphas)
        vp, tp = fit_eval(Xw, feats['y_train'], Xv, feats['id_val'], Xt, feats['id_test'])
        add(name, vp, tp)
        print(f'block-weight [{name}] готов')
    td = title_desc_features(data, feats, t_dim)
    vp, tp = fit_eval(td['train'], feats['y_train'], td['val'], feats['id_val'], td['test'], feats['id_test'])
    add('title/desc split', vp, tp)
    print('title/desc split готов')

    cols = ['конфиг', 'val CAR', 'val W', 'audit CAR', 'audit W', 'audit AER_hi']
    # вердикт: что-то бьёт чемпиона (val CAR выше ИЛИ audit AER_hi<=5% при не худшем CAR)?
    parts = [f'# CAR-side рычаги (block-weight + title/desc) vs чемпион: {CHAMPION}',
             'Для L2-logreg масштаб блока = его регуляризация. Operating point - tie-break на val. '
             f'Чемпион: val CAR {cv["CorrectAutofillRate"]:.3f} W={cv["W_wrong"]} | '
             f'audit CAR {ca["CorrectAutofillRate"]:.3f} W={ca["W_wrong"]} AER_hi {ca["AutoErrorRate_hi"]:.3f}',
             '## Все конфиги', _table(cols, rows)]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n\n'.join(parts) + '\n')
    print(f'\nотчёт: {REPORT}')
    print(f'чемпион: val CAR {cv["CorrectAutofillRate"]:.4f} | '
          f'audit CAR {ca["CorrectAutofillRate"]:.4f} AER_hi {ca["AutoErrorRate_hi"]:.4f}')
    print('лучшее на val-фронтире и/или строгий гейт - смотри таблицу в отчёте')


if __name__ == '__main__':
    main()
