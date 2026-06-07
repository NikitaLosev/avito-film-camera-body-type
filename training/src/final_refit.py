"""Финальное обучение: refit чемпиона на train+val -> инференс-бандл + MLflow (user-run, ОДИН прогон)

Модель выбрана (lrgrid__c1.0__none), эксперименты исчерпаны. Финальный шаг ЭТАПА ОБУЧЕНИЯ: дообучить ту же
архитектуру на train+val (влить val, что уже отработал отбор), оценить ОДИН раз на held-out test/audit, сохранить
бандл (голова + препроцессоры + tau + метаданные) на диск и в MLflow. test/audit НЕ трогаем - честный табель.

Зеркало run_experiment (pipeline.py), но: (1) val вливается в train через relabel -> трансформеры рефитятся на
train+val; (2) tau замороженный из селекции (choose_tau на чемпион-val, held-out val для пере-подбора больше нет,
in-sample = оверфит; tau≈0 робастен); (3) оценка только на test (proxy+audit), без val-проекции (свой финализатор,
а не _finalize - его evaluate(val) упал бы на coverage-check). Веса = НАША голова logreg + препроцессоры (tfidf/
скейлеры) + tau; эмбеддеры (Qwen3/DINOv3/PE-Core) предобучены, в прод грузятся отдельно - их специю пишем в meta.

Запуск: python training/src/final_refit.py  (~2-5 мин: рефит tfidf+logreg на ~118k train+val)
"""

import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import predictions_frame
from experiments import CHAMP_BLOCKS
from features import IMAGE_SOURCES, TEXT_EMB_DIM, TEXT_EMB_MODEL, build_features, load_blocks
from models import make_head
from paths import PREDICTIONS_DIR
from pipeline import choose_tau, experiment_paths, run_params, save_bundle
from tracking.client import init_mlflow, start_run
from validation.evaluate import evaluate
from validation.settings import ALL_LABELS, SPLIT_MANIFEST

# имя рана (выбрано юзером): архитектура + параметры по смыслу (reg1.0 = сила L2 = C, unweighted = class_weight None)
NAME = 'tfidf+qwen3+dinov3+pe+logreg__reg1.0_unweighted__final'
CHAMP_HEAD = ('logreg', {'C': 1.0, 'class_weight': None})
CHAMP_SLUG = 'lrgrid_c1_0_none'          # селекционный чемпион (train-only): источник tau + side-by-side
SPLITS = ('train', 'val', 'test')


def fold_val_into_train(data):
    """Влить val в train (трансформеры рефитнутся на train+val); test held-out не трогаем. С asserts

    build_features требует ВСЕ 3 сплита непустыми (tfidf.transform падает на 0 строк), а val нам не нужен
    (оцениваем по test). Поэтому пустой val-слот заполняем копией test: фит трансформеров идёт только по
    split=='train' (train+val), так что заглушка на обучение НЕ влияет, а X_val мы игнорируем
    """
    n_train, n_val, n_test = (int((data['split'] == s).sum()) for s in SPLITS)
    out = data.copy()
    out.loc[out['split'] == 'val', 'split'] = 'train'
    assert int((out['split'] == 'train').sum()) == n_train + n_val, 'val не влился в train'
    assert int((out['split'] == 'test').sum()) == n_test, 'test изменился (held-out нарушен)'
    assert set(out.loc[out['split'] == 'train', 'item_id']).isdisjoint(
        out.loc[out['split'] == 'test', 'item_id']), 'train пересёкся с test по item_id'
    filler = out[out['split'] == 'test'].copy()                 # заглушка val-слота (X_val игнорируем)
    filler['split'] = 'val'
    out = pd.concat([out, filler], ignore_index=True)
    return out, n_train, n_val, n_test


def block_meta(spec: dict, feats: dict) -> dict:
    """Описание блока для prod-контракта: модель-источник + размерность + скейлинг"""
    b = spec['block']
    if b == 'tfidf':
        return {'block': 'tfidf', 'dim': len(feats['vectorizer'].vocabulary_), 'scaling': spec['scaling']}
    if b == 'text_emb':
        return {'block': 'text_emb', 'model': TEXT_EMB_MODEL, 'dim': TEXT_EMB_DIM, 'scaling': spec['scaling']}
    if b == 'image_emb':
        _, model, dim = IMAGE_SOURCES[spec['source']]
        return {'block': 'image_emb', 'source': spec['source'], 'model': model, 'dim': dim, 'scaling': spec['scaling']}
    raise ValueError(f'неизвестный блок {b}')


def _biz(report: dict) -> dict:
    """Плоские бизнес-числа проекции для лога/meta"""
    b = report['business']
    return {'CAR': float(b['CorrectAutofillRate']), 'AER': float(b['AutoErrorRate']),
            'AER_hi': float(b['AutoErrorRate_hi']), 'A': int(b['A_auto']),
            'C': int(b['C_correct']), 'W': int(b['W_wrong'])}


def main() -> None:
    init_mlflow()
    _, test_pred_path, bundle_path = experiment_paths(NAME)
    cfg = {'name': NAME, 'blocks': CHAMP_BLOCKS, 'head': CHAMP_HEAD[0], 'head_params': CHAMP_HEAD[1]}

    # 1. данные + влить val в train (трансформеры рефитятся на train+val, test held-out)
    data = load_blocks(CHAMP_BLOCKS)
    data, n_train, n_val, n_test = fold_val_into_train(data)
    print(f'refit на train+val = {n_train}+{n_val} = {n_train + n_val} строк | held-out test {n_test}')

    # 2. фичи (фит трансформеров на train+val) + рефит головы чемпиона
    feats = build_features(data, CHAMP_BLOCKS)
    assert feats['X_train'].shape[0] == n_train + n_val, 'X_train не train+val'
    head = make_head(*CHAMP_HEAD).fit(feats['X_train'], feats['y_train'])

    # 3. tau = замороженный operating point чемпиона (с его селекционных val-предсказаний)
    tau = choose_tau(PREDICTIONS_DIR / f'{CHAMP_SLUG}_val_pred.parquet')
    print(f'tau (замороженный чемпион-operating-point): {tau:.4f}')

    # 4. предсказания на held-out test + оценка (proxy 22k + audit 210), без val-проекции
    predictions_frame(head, feats['X_test'], feats['id_test']).to_parquet(test_pred_path, index=False)
    rep_proxy = evaluate(test_pred_path, 'test', 'final_label', tau=tau)
    rep_audit = evaluate(test_pred_path, 'test', 'audit_label', tau=tau)

    # side-by-side: чемпион (train-only) при ТОМ ЖЕ tau
    champ_test = PREDICTIONS_DIR / f'{CHAMP_SLUG}_test_pred.parquet'
    champ_proxy = evaluate(champ_test, 'test', 'final_label', tau=tau)
    champ_audit = evaluate(champ_test, 'test', 'audit_label', tau=tau)

    # 5. бандл (reuse save_bundle) + meta-сайдкар (prod-контракт)
    bundle = save_bundle(cfg, feats, head, tau, bundle_path)
    meta_path = bundle_path.with_name(bundle_path.name.replace('_pipeline.joblib', '_meta.yaml'))
    meta = {
        'name': NAME, 'champion_config': 'lrgrid__c1.0__none',
        'trained_on': 'train+val', 'held_out': 'test+audit (не трогали)',
        'n_train_refit': n_train + n_val, 'n_test': n_test, 'tau': float(tau),
        'labels': list(ALL_LABELS),
        'head': {'kind': 'logreg', 'C': 1.0, 'class_weight': None},
        'feature_order': [block_meta(s, feats) for s in CHAMP_BLOCKS],
        'total_dim': int(feats['X_train'].shape[1]),
        'split_data_sha256': yaml.safe_load(SPLIT_MANIFEST.read_text())['data_sha256'],
        'held_out_metrics': {'test_proxy': _biz(rep_proxy), 'test_audit': _biz(rep_audit)},
    }
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))

    # 6. MLflow (свой финализатор - без val-проекции)
    metrics = {
        'test_audit_CAR': float(rep_audit['business']['CorrectAutofillRate']),
        'test_audit_AER': float(rep_audit['business']['AutoErrorRate']),
        'test_audit_AER_hi': float(rep_audit['business']['AutoErrorRate_hi']),
        'test_proxy_CAR': float(rep_proxy['business']['CorrectAutofillRate']),
        'test_proxy_AER': float(rep_proxy['business']['AutoErrorRate']),
        'test_proxy_macro_f1': float(rep_proxy['classification']['macro_f1']),
        'tau': float(tau), 'n_train_refit': n_train + n_val,
    }
    with start_run(NAME, params=run_params(cfg, tau)):
        mlflow.log_metrics(metrics)
        mlflow.log_dict(rep_proxy, 'report_test_proxy.json')
        mlflow.log_dict(rep_audit, 'report_test_audit.json')
        mlflow.log_artifact(str(bundle), 'model')
        mlflow.log_artifact(str(meta_path), 'model')
        mlflow.log_artifact(str(test_pred_path), 'predictions')
        mlflow.log_artifact(str(SPLIT_MANIFEST), 'validation')

    # 7. выжимка + сравнение
    print(f'\nбандл: {bundle}\nmeta:  {meta_path}')
    cp, ca = _biz(champ_proxy), _biz(champ_audit)
    rp, ra = _biz(rep_proxy), _biz(rep_audit)
    print(f'\n{"модель":<28} | {"proxy CAR":>9} | {"audit CAR":>9} | {"audit W":>7} | {"audit AER_hi":>12}')
    print('-' * 78)
    print(f'{"champion (train)":<28} | {cp["CAR"]:>9.4f} | {ca["CAR"]:>9.4f} | {ca["W"]:>7} | {ca["AER_hi"]:>12.4f}')
    print(f'{"refit (train+val) FINAL":<28} | {rp["CAR"]:>9.4f} | {ra["CAR"]:>9.4f} | '
          f'{ra["W"]:>7} | {ra["AER_hi"]:>12.4f}')
    print(f'\nΔ proxy CAR {rp["CAR"] - cp["CAR"]:+.4f} | Δ audit CAR {ra["CAR"] - ca["CAR"]:+.4f} '
          f'(ожидание: ≈ или чуть выше; сильное падение = баг рефита)')


if __name__ == '__main__':
    main()
