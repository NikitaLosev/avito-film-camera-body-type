# Чувал - определение параметров товара по тексту/фото

Классификация `film_camera_body_type` (тип корпуса плёночной камеры: SLR / TLR / rangefinder_viewfinder /
compact_point_and_shoot / instant / other_unknown) по тексту и фото объявления Авито. Учитель разметки - Gemini,
цель - дистилляция в дешёвую CPU-модель. Бизнес-метрика: максимизировать CorrectAutofillRate при AutoErrorRate <= 5%.

**Финальная модель** (`tfidf + qwen3 + dinov3 + PE-Core -> logreg`): held-out audit CAR **0.748**, proxy CAR **0.835**.
Веса головы - в `training/final_model/`, полная дуга экспериментов - в `training/`.

## Структура репозитория

| папка | этап | что внутри |
| --- | --- | --- |
| `problem-validation/` | валидация задачи | стоит ли решать, оценка осмысленности |
| `eda/` | разведка данных | `eda.ipynb` + разбор классов/дублей/текста/фото |
| `solution-design/` | дизайн решения | подход, метрика, цели |
| `data-pipeline/` | разметка данных | 12-стадийный ETL + Gemini-разметка -> датасет |
| `training/` | **обучение моделей** | пайплайн, эксперименты, анализ ошибок, финальная модель |

У каждого этапа свой README с деталями. Здесь - общая установка и запуск.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate    # Python 3.13
pip install -r requirements.txt
```

## Эмбеддинги (вход обучения)

Текст/фото кодируются предобученными моделями (Qwen3-Embedding-0.6B, DINOv3, PE-Core) - извлекаются один раз на
Kaggle GPU ноутбуками `training/notebooks/extract_*.ipynb`, результат (parquet) кладётся в `training/artifacts/embeddings/`.
Это вход пайплайна; сами эмбеддеры в обучении не нужны.

## Запуск обучения

```bash
python training/src/train.py                          # гонит эксперименты из experiments.py (TO_RUN)
python training/src/train.py --all                    # весь каталог конфигов
python training/src/analysis/error_analysis.py        # анализ ошибок чемпиона -> training/reports/
python training/src/final_refit.py                    # финальная модель: refit на train+val -> бандл
bash training/tracking/ui.sh                          # MLflow UI на http://127.0.0.1:5000
```

Новый эксперимент = строка конфига в `training/src/experiments.py` (блоки фич + голова), без нового файла.
Подробности - в `training/README.md`.

## Загрузка финальной модели

```python
import joblib

bundle = joblib.load('training/final_model/tfidf_qwen3_dinov3_pe_logreg_reg1_0_unweighted_final_pipeline.joblib')
# vectorizer (tfidf) + scalers (по emb-блокам) + classifier (logreg-голова) + tau (порог) + labels + model_names
clf, tau, labels = bundle['classifier'], bundle['tau'], bundle['labels']
```

Контракт фич (какие эмбеддеры, размерности, порядок) - в `training/final_model/*_meta.yaml`. Инференс = собрать
фичи теми же эмбеддерами -> `classifier.predict_proba` -> порог `tau` (это уже задача прод-этапа).

## Результаты

- сводка экспериментов: `training/reports/experiments_summary.csv`
- анализ ошибок + рычаги: `training/reports/*.md`
- замороженный split без утечек + evaluator: `training/validation/`
- трекинг (MLflow, sqlite): `training/tracking/`

## Код-стайл

`ruff` + `flake8` (конфиг - `pyproject.toml` + `setup.cfg`, line-length 120):

```bash
ruff check .
flake8 .
```

## Команда

- Команда: Чувал
- Капитан: Лосев Никита, проект выполнен индивидуально
- Проект: определение параметров товара по тексту/фото (тема 05)
