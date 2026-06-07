"""kNN-probe (рычаг E, дёшево): чинит ли локальная память ошибки чемпиона - БЕЗ полного билда (user-run)

Гипотеза E: локальная непараметрическая память (kNN над фичами чемпиона) ловит ЛОКАЛЬНЫЕ исключения, что
глобальная линейная граница сглаживает. Поднять CAR (наша ось) можно ТОЛЬКО починив wrong_type (main->правильный
main); kNN-как-гейт-отказа = ось точности (мертва, гейт уже взят). Дёшевый диагност вместо 45-мин OOF-стекера:
на val-автозаполнениях чемпиона (tau=0) смотрим, голосует ли kNN-над-train за ИСТИНУ (починил бы wrong_type),
за ошибку чемпиона (подтвердил) или шум; и сколько ВЕРНЫХ kNN бы сломал (collateral). Оракул-оверрайд ΔC =
fix_wt - collateral: если <=0, даже агрессивный kNN СНИЖАЕТ CAR -> E мёртв по нашей оси, полный стекер НЕ строим.

Утечки нет: val held-out из train (сосед не из своей строки), а split leakage-group-дизъюнктен (val-строка и её
near-dup в одной группе -> один сплит) -> ближайший train-сосед не near-dup себя. kNN в фичах ЧЕМПИОНА (tfidf+
qwen+dino+pe, cosine) = честный local-vs-global на ОДНИХ признаках. Пишет reports/knn_probe.md

Запуск: python training/src/knn_probe.py  (~3-6 мин: build_features + brute-cosine kNN по подвыборке)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # training/
sys.path.insert(0, str(HERE.parent))         # training/src/
sys.path.insert(0, str(HERE))                # training/src/analysis/
from experiments import CHAMP_BLOCKS
from features import build_features, load_blocks
from multineg_analysis import _table
from paths import PREDICTIONS_DIR
from tracking.settings import REPORTS_DIR
from validation.settings import ABSTAIN, MAIN_SET

CHAMPION = 'lrgrid__c1.0__none'
KS = (10, 30)                 # k соседей для голоса (два - проверить устойчивость)
N_OK_SAMPLE = 3000            # подвыборка верных автозаполнений для оценки collateral-rate
SEED = 42
REPORT = REPORTS_DIR / 'knn_probe.md'
MAIN = set(MAIN_SET)


def row_modes(codes_k: np.ndarray, n_labels: int) -> np.ndarray:
    """Мода по строкам (n, k) int-кодов меток -> (n,) код-победитель (argmax bincount, ничья -> младший)"""
    return np.array([np.bincount(r, minlength=n_labels).argmax() for r in codes_k])


def _pct(mask: np.ndarray) -> str:
    return '-' if len(mask) == 0 else f'{mask.mean():.1%}'


def main() -> None:
    data = load_blocks(CHAMP_BLOCKS)
    feats = build_features(data, CHAMP_BLOCKS)
    y_train = feats['y_train']
    codes, uniq = pd.factorize(y_train)                  # train метки -> int-коды
    uniq = list(uniq)

    # выровнять предсказания чемпиона + truth к порядку X_val (id_val)
    val_ids = feats['id_val']
    champ = pd.read_parquet(PREDICTIONS_DIR / 'lrgrid_c1_0_none_val_pred.parquet')[['item_id', 'pred_label']]
    al = pd.DataFrame({'item_id': val_ids}).merge(champ, on='item_id', how='left') \
                                           .merge(data[['item_id', 'final_label']], on='item_id', how='left')
    pred = al['pred_label'].to_numpy()
    tru = al['final_label'].to_numpy()
    in_main = np.array([p in MAIN for p in pred])        # автозаполнение чемпиона при tau=0

    correct = in_main & (pred == tru)
    wrong_type = in_main & np.array([t in MAIN for t in tru]) & (pred != tru)
    should_abstain = in_main & (tru == ABSTAIN)
    n_correct = int(correct.sum())
    print(f'val автозаполнений {int(in_main.sum())} | верных {n_correct} | '
          f'wrong_type {int(wrong_type.sum())} | should_abstain {int(should_abstain.sum())}')

    rng = np.random.default_rng(SEED)
    ok_all = np.where(correct)[0]
    ok_sample = rng.choice(ok_all, size=min(N_OK_SAMPLE, len(ok_all)), replace=False)
    groups = {'wrong_type': np.where(wrong_type)[0],
              'should_abstain': np.where(should_abstain)[0],
              'correct(sample)': ok_sample}
    q_rows = np.concatenate([groups[g] for g in groups])
    pos = {int(r): i for i, r in enumerate(q_rows)}      # val-row -> позиция в q_rows/votes

    print(f'kNN brute-cosine: индекс train {feats["X_train"].shape[0]}, запросов {len(q_rows)} (k={max(KS)})...')
    nn = NearestNeighbors(n_neighbors=max(KS), metric='cosine', algorithm='brute').fit(feats['X_train'])
    _, nbr = nn.kneighbors(feats['X_val'][q_rows])       # (nq, maxK) train-позиции, отсортированы по близости
    nbr_codes = codes[nbr]
    votes = {k: np.array(uniq)[row_modes(nbr_codes[:, :k], len(uniq))] for k in KS}

    def grp(name: str, k: int):
        rows = groups[name]
        p = np.array([pos[int(r)] for r in rows])
        return votes[k][p], tru[rows], pred[rows]

    # --- таблица 1: wrong_type (CAR-ось: чинибельность) ---
    t1 = []
    for k in KS:
        v, t, c = grp('wrong_type', k)
        t1.append([k, len(v), _pct(v == t), _pct(v == c), _pct(v == ABSTAIN),
                   _pct((v != t) & (v != c) & (v != ABSTAIN))])
    tbl1 = _table(['k', 'n', 'fix (vote=ИСТИНА)', 'confirm (vote=ошибка)', '->other', 'прочее'], t1)

    # --- таблица 2: collateral на верных + should_abstain (ось точности) ---
    t2 = []
    for k in KS:
        vc, tc, _ = grp('correct(sample)', k)
        vsa, _, _ = grp('should_abstain', k)
        t2.append([k, len(vc), _pct(vc == tc), _pct(vc != tc), len(vsa), _pct(vsa == ABSTAIN)])
    tbl2 = _table(['k', 'correct n', 'keep (vote=ИСТИНА)', 'collateral (vote≠ИСТИНА)',
                   'should_abstain n', '->other (precision)'], t2)

    # --- вердикт: оракул-оверрайд ΔC = fix_wt - collateral_est (k=30) ---
    K0 = max(KS)
    vwt, twt, _ = grp('wrong_type', K0)
    vok, tok, _ = grp('correct(sample)', K0)
    fix_wt = int((vwt == twt).sum())
    collat_rate = float((vok != tok).mean()) if len(vok) else 0.0
    collat_est = collat_rate * n_correct
    net_dC = fix_wt - collat_est
    vsa, _, _ = grp('should_abstain', K0)
    sa_other = int((vsa == ABSTAIN).sum())

    verdict = [
        f'оракул-оверрайд (заменить argmax чемпиона на kNN-голос, k={K0}) - верхняя граница сигнала:',
        f'  fix wrong_type: {fix_wt} из {len(vwt)} (kNN голосует ИСТИНУ -> +{fix_wt} верных)',
        f'  collateral: rate {collat_rate:.1%} на верных -> ~{collat_est:.0f} сломано (из {n_correct})',
        f'  ΔC (net по CAR) ≈ {fix_wt} - {collat_est:.0f} = {net_dC:+.0f}',
        f'  should_abstain -> other: {sa_other} (ось ТОЧНОСТИ, гейт уже взят -> CAR не растит)',
        '',
        'ВЕРДИКТ: ' + ('probe КРАСНЫЙ - даже агрессивный kNN-оверрайд снижает/не растит CAR (net<=0) -> '
                       'E мёртв по нашей оси, полный OOF-стекер НЕ строим'
                       if net_dC <= 0 else
                       f'probe ЗЕЛЁНЫЙ - оракул-оверрайд поднимает CAR на ~{net_dC:+.0f} (верхняя граница) -> '
                       'есть локальный сигнал, СТОИТ строить полный leak-safe kNN-стекер (извлечёт часть)'),
    ]
    verdict_txt = '\n'.join(verdict)

    parts = [f'# kNN-probe (рычаг E) - чинибельность ошибок чемпиона: {CHAMPION}',
             'Локальная память (kNN-над-train в фичах чемпиона, cosine) vs глобальная линия. Поднять CAR можно '
             'ТОЛЬКО починкой wrong_type (main->правильный main); kNN-гейт = ось точности (мертва). Диагност на '
             'val-автозаполнениях чемпиона (tau=0), без OOF (val held-out, split leakage-group-дизъюнктен -> '
             'сосед не near-dup себя). Это probe-before-build: строить полный стекер только если net ΔC>0',
             '## Опора (val-автозаполнения чемпиона)',
             f'верных {n_correct} | wrong_type {int(wrong_type.sum())} | should_abstain {int(should_abstain.sum())}',
             '## 1. wrong_type - чинит ли kNN (CAR-ось)', tbl1,
             '## 2. collateral на верных + should_abstain (ось точности)', tbl2,
             '## Вердикт', '```\n' + verdict_txt + '\n```']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n\n'.join(parts) + '\n')
    print(f'\nотчёт: {REPORT}\n')
    print(verdict_txt)


if __name__ == '__main__':
    main()
