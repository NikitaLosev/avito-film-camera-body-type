"""Лексическая сетка vocab/analyzer vs чемпион: добавить сырые лексические фичи (user-run, emb раз)

Гипотеза (другой механизм, чем block-weight/стекинг): char_wb (3,5) не перешагивает пробел, а
min_df=5/max_features=50k режут редкие n-граммы названий (Красногорск, Contax G1, OM-5, «в ассортименте»).
Это нельзя достать перевзвешиванием или гейтом - нет колонки, нечего масштабировать. Меняем ТОЛЬКО
лексический блок, emb-часть (qwen+dino+pe) берём бит-в-бит из чемпион-X, фитим logreg-чемпиона, сравниваем
matched-frontier (tie-break) vs чемпион на val + proxy + audit. Винты: удержание хвоста (min_df 5->2,
max_features 50k->150k) + word-блок (1,2) + char-phrase (analyzer='char', через пробел). baseline (rebuild)
= sanity (должен воспроизвести чемпиона). Низкий риск teacher-overfit (всё ещё линейный logreg), но редкие
n-граммы шумят и могут поднять proxy без audit - судить по val, audit только санити. validation/ заморожен

char-phrase - самый тяжёлый по памяти (счёт всех char-ngram до прунинга), идёт последним: если OOM, первые
4 конфига (главная ценность) уже посчитаны. Запуск: python training/src/vocab_analysis.py  (≈10-30 мин)
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

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

CHAMPION = 'lrgrid__c1.0__none'
CHAMP_HEAD = ('logreg', {'C': 1.0, 'class_weight': None})
REPORT = REPORTS_DIR / 'vocab_analysis.md'

# лексические векторизаторы (каждый фитится на train['text']). champion = CHAR_WB
CHAR_WB = dict(analyzer='char_wb', ngram_range=(3, 5), min_df=5, max_features=50000, sublinear_tf=True)
CHAR_WB_TAIL = dict(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=150000, sublinear_tf=True)
WORD = dict(analyzer='word', ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True,
            token_pattern=r'(?u)\b[\w-]+\b')   # дефис внутри -> «om-5», цифры/латиница не выкидываем
CHAR_PHRASE = dict(analyzer='char', ngram_range=(4, 6), min_df=3, max_features=100000, sublinear_tf=True)

# конфиг = набор векторизаторов поверх общей emb-части. изолируем каждый винт + комбо.
# порядок: дешёвые/высокоприоритетные первыми, тяжёлый char-phrase последним
CONFIGS = [
    ('baseline (rebuild)',   [CHAR_WB]),                          # sanity = чемпион
    ('retain-tail',          [CHAR_WB_TAIL]),                     # удержать редкий хвост char_wb
    ('+word(1,2)',           [CHAR_WB, WORD]),                    # + чистые токены/биграммы слов
    ('retain +word',         [CHAR_WB_TAIL, WORD]),               # два главных винта вместе
    ('+char-phrase(4,6)',    [CHAR_WB, CHAR_PHRASE]),             # + фразы через пробел
    ('retain +word +phrase', [CHAR_WB_TAIL, WORD, CHAR_PHRASE]),  # всё (помогает или шумит)
]


def lexical_block(specs: list, parts: dict) -> tuple:
    """Фит каждого векторизатора на train['text'], трансформ всех сплитов, hstack -> (dict, dims)"""
    vecs = [TfidfVectorizer(**s).fit(parts['train']['text']) for s in specs]
    dims = sum(len(v.vocabulary_) for v in vecs)
    out = {}
    for s in ('train', 'val', 'test'):
        mats = [v.transform(parts[s]['text']) for v in vecs]
        out[s] = hstack(mats).tocsr() if len(mats) > 1 else mats[0]
    return out, dims


def build_X(lex: dict, emb: dict) -> dict:
    """hstack лексический блок + emb-часть чемпиона (одна на все конфиги, бит-в-бит)"""
    return {s: hstack([lex[s], emb[s]]).tocsr() for s in ('train', 'val', 'test')}


def summarize(vp: pd.DataFrame, tp: pd.DataFrame, truth: pd.DataFrame) -> tuple:
    """tie-break tau на val -> (val, proxy-22k, audit-210) business-метрики"""
    vfr, tfr = dec_frame(vp, truth), dec_frame(tp, truth)
    afr = tfr[tfr['audit_label'].notna()].copy()
    tau = pick_tau_tb(vfr)
    return (business_at(vfr, 'final_label', tau),
            business_at(tfr, 'final_label', tau),
            business_at(afr, 'audit_label', tau))


def main() -> None:
    data = load_blocks(CHAMP_BLOCKS)
    truth = truth_frame()
    parts = {s: data[data['split'] == s] for s in ('train', 'val', 'test')}
    feats = build_features(data, CHAMP_BLOCKS)
    t_dim = len(feats['vectorizer'].vocabulary_)
    emb = {s: feats[f'X_{s}'][:, t_dim:] for s in ('train', 'val', 'test')}
    print(f'чемпион: tfidf-словарь {t_dim} | emb-ширина {emb["train"].shape[1]} (qwen+dino+pe)')

    champ = (pd.read_parquet(PREDICTIONS_DIR / 'lrgrid_c1_0_none_val_pred.parquet'),
             pd.read_parquet(PREDICTIONS_DIR / 'lrgrid_c1_0_none_test_pred.parquet'))

    rows, summary = [], {}

    def add(name: str, dims: int, vp: pd.DataFrame, tp: pd.DataFrame) -> None:
        v, p, a = summarize(vp, tp, truth)
        rows.append([name, dims, _num(v['CorrectAutofillRate']), v['W_wrong'],
                     _num(p['CorrectAutofillRate']), _num(a['CorrectAutofillRate']),
                     a['W_wrong'], _num(a['AutoErrorRate_hi'])])
        summary[name] = (v, a)

    add('champion (saved)', t_dim, *champ)
    for name, specs in CONFIGS:
        lex, dims = lexical_block(specs, parts)
        X = build_X(lex, emb)
        head = make_head(*CHAMP_HEAD).fit(X['train'], feats['y_train'])
        vp = predictions_frame(head, X['val'], feats['id_val'])
        tp = predictions_frame(head, X['test'], feats['id_test'])
        add(name, dims, vp, tp)
        print(f'[{name}] готов (lex-фичей {dims})')

    cols = ['конфиг', 'lex dims', 'val CAR', 'val W', 'proxy CAR', 'audit CAR', 'audit W', 'audit AER_hi']

    cv, ca = summary['champion (saved)']
    lines = [f"champion: val CAR {cv['CorrectAutofillRate']:.3f} W={cv['W_wrong']} | "
             f"audit CAR {ca['CorrectAutofillRate']:.3f} W={ca['W_wrong']} AER_hi {ca['AutoErrorRate_hi']:.3f}"]
    wins = []
    for name, _ in CONFIGS:
        v, a = summary[name]
        dC = v['CorrectAutofillRate'] - cv['CorrectAutofillRate']
        val_win = dC > 1e-9 or (v['W_wrong'] < cv['W_wrong'] and dC >= -1e-9)
        if name == 'baseline (rebuild)':
            ok = abs(dC) < 2e-3 and abs(v['W_wrong'] - cv['W_wrong']) <= 2
            tag = '  (sanity ≈ champion)' if ok else '  (sanity РАСХОДИТСЯ - проверь harness)'
        else:
            tag = '  <- val-выигрыш' if val_win else ''
            if val_win:
                wins.append(name)
        lines.append(f"{name}: val CAR {v['CorrectAutofillRate']:.3f} W={v['W_wrong']} (ΔCAR{dC:+.3f}) | "
                     f"audit CAR {a['CorrectAutofillRate']:.3f} W={a['W_wrong']} "
                     f"AER_hi {a['AutoErrorRate_hi']:.3f}{tag}")
    lines.append('ВЕРДИКТ: ' + (f'двигает val: {wins} - смотреть детально (судить по val, audit только санити)'
                                if wins else 'лексика НЕ двигает val - рычаг C у потолка по этим данным'))
    verdict = '\n'.join(lines)

    parts_md = [f'# Лексическая сетка (vocab/analyzer) vs чемпион: {CHAMPION}',
                'Меняем ТОЛЬКО лексический блок, emb (qwen+dino+pe) бит-в-бит из чемпион-X. char_wb=(3,5) '
                'min_df5/50k. Винты: retain-tail (min_df2/150k), word(1,2), char-phrase(4,6) через пробел. '
                'Operating point - tie-break (max C/min W) на val. baseline (rebuild) = sanity. Судить по val '
                '(19k надёжно), proxy - teacher-overfit чек, audit (210) только санити - не короновать по нему '
                '(см. qwen×0.5 в blockfeat)',
                '## Все конфиги', _table(cols, rows),
                '## Вердикт', '```\n' + verdict + '\n```']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n\n'.join(parts_md) + '\n')
    print(f'\nотчёт: {REPORT}\n')
    print(verdict)


if __name__ == '__main__':
    main()
