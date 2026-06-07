"""Один пайплайн обучения: конфиг -> фичи -> голова -> предсказания -> tau -> оценка -> mlflow

КАК написан один раз, ЧТО приходит конфигом. Раны логируются единообразно, артефакты
и параметры воспроизводимы. Evaluator и tracking заморожены - зовём их через API,
внутрь не лезем. Предсказания пишем по пути из слага имени, чтобы --all не затирал чужое
"""

import re
import sys
from pathlib import Path

import joblib
import mlflow
import yaml
from scipy.sparse import issparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from cascade import CascadeModel
from common import TFIDF_PARAMS, file_sha256, predictions_frame, print_summary, raw_metrics, run_metrics
from features import IMAGE_SOURCES, TEXT_EMB_FILE, TEXT_EMB_MODEL, build_features
from models import ACCEPTS_SPARSE, HEAD_DEFAULTS, make_head
from paths import EMBEDDINGS_DIR, MODELS_DIR, PREDICTIONS_DIR
from tracking.client import start_run
from validation.evaluate import evaluate, load_eval_frame
from validation.metrics import pick_operating_point, threshold_sweep
from validation.settings import ALL_LABELS, SPLIT_MANIFEST


def _slug(name: str) -> str:
    """Имя рана -> безопасное имя файла"""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def experiment_paths(name: str) -> tuple:
    """Пути предсказаний и бандла для эксперимента (по слагу имени)"""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(name)
    return (PREDICTIONS_DIR / f'{slug}_val_pred.parquet',
            PREDICTIONS_DIR / f'{slug}_test_pred.parquet',
            MODELS_DIR / f'{slug}_pipeline.joblib')


def _label(spec: dict) -> str:
    """Короткая метка блока для строки features в параметрах"""
    return {'tfidf': 'tfidf', 'text_emb': 'qwen3'}.get(spec['block'], spec.get('source', spec['block']))


def _model_names(specs: list) -> dict:
    """Имена моделей-источников для присутствующих emb-блоков"""
    names = {}
    for spec in specs:
        if spec['block'] == 'text_emb':
            names['text_emb'] = TEXT_EMB_MODEL
        elif spec['block'] == 'image_emb':
            names[f'image_emb_{spec["source"]}'] = IMAGE_SOURCES[spec['source']][1]
    return names


def save_bundle(cfg: dict, feats: dict, head, tau: float, path: Path) -> Path:
    """Инференс-пайплайн одним файлом: рецепт фич + трансформеры + голова + порог"""
    joblib.dump({
        'blocks': cfg['blocks'],
        'vectorizer': feats.get('vectorizer'),
        'scalers': feats['scalers'],
        'classifier': head,
        'tau': tau,
        'labels': list(head.classes_),
        'head': cfg['head'],
        'head_params': cfg.get('head_params', {}),
        'model_names': _model_names(cfg['blocks']),
    }, path)
    return path


def run_params(cfg: dict, tau: float) -> dict:
    """Параметры рана: фичи, голова, скейлинги и версии данных - что логировали 6 скриптов"""
    manifest = yaml.safe_load(SPLIT_MANIFEST.read_text())
    blocks = [s['block'] for s in cfg['blocks']]
    params = {
        'features': '+'.join(_label(s) for s in cfg['blocks']),
        'tau': round(tau, 3),
        'split_id': 'split_v1',
        'split_data_sha256': manifest['data_sha256'],
    }
    if 'tfidf' in blocks:
        params.update({f'tfidf_{k}': v for k, v in TFIDF_PARAMS.items()})
    head = cfg['head']
    resolved = {**HEAD_DEFAULTS[head], **cfg.get('head_params', {})}
    params.update({f'{head}_{k}': v for k, v in resolved.items()})
    for spec in cfg['blocks']:
        block = spec['block']
        if block == 'tfidf':
            continue
        if block == 'image_emb':                                  # source-key, иначе 2 источника затирают друг друга
            src = spec['source']
            file, model, _ = IMAGE_SOURCES[src]
            params[f'scaling_image_emb_{src}'] = spec['scaling']
            params[f'image_{src}_sha256'] = file_sha256(EMBEDDINGS_DIR / file)
            params[f'image_{src}_model'] = model
            continue
        params[f'scaling_{block}'] = spec['scaling']
        if block == 'text_emb':
            params['text_emb_sha256'] = file_sha256(EMBEDDINGS_DIR / TEXT_EMB_FILE)
            params['text_model'] = TEXT_EMB_MODEL
    return params


def choose_tau(val_pred) -> float:
    """Рабочий порог на val (final_label) под AER_hi <= 5%"""
    vf = load_eval_frame(val_pred, 'val', 'final_label')
    return pick_operating_point(threshold_sweep(vf, 'final_label', 'pred_label', 'confidence'))['tau']


def _finalize(name: str, val_pred, test_pred, bundle, params: dict, tau: float) -> dict:
    """Общий хвост (flat и cascade): оценка x3, метрики, лог в mlflow, выжимка"""
    reports = (
        evaluate(val_pred, 'val', 'final_label', tau=tau),
        evaluate(test_pred, 'test', 'audit_label', tau=tau),
        evaluate(test_pred, 'test', 'final_label', tau=tau),
    )
    raw_val = evaluate(val_pred, 'val', 'final_label', tau=0.0)          # raw (без отказа) для гигиены лога
    metrics = {**run_metrics(*reports), **raw_metrics(raw_val)}
    with start_run(name, params=params):
        mlflow.log_metrics(metrics)
        mlflow.log_dict(reports[0], 'report_val_proxy.json')
        mlflow.log_dict(reports[1], 'report_test_audit.json')
        mlflow.log_dict(reports[2], 'report_test_internal.json')
        mlflow.log_artifact(val_pred, 'predictions')
        mlflow.log_artifact(test_pred, 'predictions')
        mlflow.log_artifact(str(SPLIT_MANIFEST), 'validation')
        mlflow.log_artifact(bundle, 'model')
    print_summary(name, metrics, tau)
    return metrics


def run_experiment(cfg: dict, data) -> dict:
    """Плоский эксперимент по конфигу: фичи, обучение, оценка, логирование"""
    feats = build_features(data, cfg['blocks'])
    if issparse(feats['X_train']) and not ACCEPTS_SPARSE[cfg['head']]:
        raise ValueError(f"голова {cfg['head']} не принимает разреженные фичи (в конфиге есть tfidf-блок)")

    head = make_head(cfg['head'], cfg.get('head_params'))
    head.fit(feats['X_train'], feats['y_train'])

    val_pred, test_pred, bundle_path = experiment_paths(cfg['name'])
    predictions_frame(head, feats['X_val'], feats['id_val']).to_parquet(val_pred, index=False)
    predictions_frame(head, feats['X_test'], feats['id_test']).to_parquet(test_pred, index=False)

    tau = choose_tau(val_pred)
    bundle = save_bundle(cfg, feats, head, tau, bundle_path)
    return _finalize(cfg['name'], val_pred, test_pred, bundle, run_params(cfg, tau), tau)


def save_cascade_bundle(cfg: dict, model, tau: float, path: Path) -> Path:
    """Инференс-бандл каскада: оба стейджа (трансформеры + голова + блоки) + порог"""
    def stage(feats, head, scfg):
        return {'vectorizer': feats.get('vectorizer'), 'scalers': feats['scalers'],
                'head': head, 'blocks': scfg['blocks']}
    joblib.dump({
        'kind': 'cascade', 'tau': tau, 'labels': list(ALL_LABELS),
        'stage1': stage(model.feats1, model.head1, cfg['stage1']),
        'stage2': stage(model.feats2, model.head2, cfg['stage2']),
    }, path)
    return path


def cascade_run_params(cfg: dict, tau: float) -> dict:
    """Параметры рана каскада: головы, фичи стейджей, версии данных"""
    manifest = yaml.safe_load(SPLIT_MANIFEST.read_text())
    params = {
        'kind': 'cascade', 'tau': round(tau, 3), 'split_id': 'split_v1',
        'split_data_sha256': manifest['data_sha256'],
        'stage1_head': cfg['stage1']['head'],
        'stage1_features': '+'.join(_label(s) for s in cfg['stage1']['blocks']),
        'stage2_head': cfg['stage2']['head'],
        'stage2_features': '+'.join(_label(s) for s in cfg['stage2']['blocks']),
    }
    for spec in cfg['stage1']['blocks'] + cfg['stage2']['blocks']:
        if spec['block'] == 'text_emb':
            params['text_emb_sha256'] = file_sha256(EMBEDDINGS_DIR / TEXT_EMB_FILE)
        elif spec['block'] == 'image_emb':
            src = spec['source']
            params[f'image_{src}_sha256'] = file_sha256(EMBEDDINGS_DIR / IMAGE_SOURCES[src][0])
    return params


def run_cascade(cfg: dict, data) -> dict:
    """Каскад по конфигу: обе модели, композиция, stage-1 предсказания, оценка, логирование"""
    model = CascadeModel(cfg['stage1'], cfg['stage2']).fit(data)

    val_pred, test_pred, bundle_path = experiment_paths(cfg['name'])
    model.frame('val').to_parquet(val_pred, index=False)
    model.frame('test').to_parquet(test_pred, index=False)
    model.stage1_frame('test').to_parquet(PREDICTIONS_DIR / f"{_slug(cfg['name'])}_stage1.parquet", index=False)

    tau = choose_tau(val_pred)
    bundle = save_cascade_bundle(cfg, model, tau, bundle_path)
    return _finalize(cfg['name'], val_pred, test_pred, bundle, cascade_run_params(cfg, tau), tau)
