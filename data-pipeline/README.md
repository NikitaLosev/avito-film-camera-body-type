# Пайплайн разметки

Подробный разбор того как этот датасет получился и какие были тупики $-$ в `reflection.md`. Этот файл $-$ только про устройство пайплайна: что где лежит, как данные идут от сырого csv до финального parquet, какие скрипты для чего


## Что внутри папки

```
data-pipeline/
├── pipeline/         основной пайплайн, 11 пронумерованных стейджей
├── lib/              общий код: пути, нормализация моделей, схема ответа LLM, обёртка над Gemini
├── prompts/          промпты для LLM, активный $-$ v3_with_vision
├── tools/            вспомогательные скрипты: ручная разметка, калибровка промпта
├── reflection.md     рефлексия по этапу разметки
└── README.md         этот файл
```


## Откуда берутся данные

Сырые данные лежат в RustFS под кредами в `.env`. Скрипт `stage_01_download_raw.py` ходит туда как в обычный S3 и зеркалит всё в `data/raw/`:

| файл | размер | что |
|---|---|---|
| `items_project_aaa.csv` | 101 МБ | 144 313 объявлений: `item_id`, `user_id`, `microcat_name`, `title`, `description`, `image_id` |
| `image/` | 18 ГБ | 144 313 jpg, разложены по 1000 подпапкам по правилу `image_id % 1000` |

Особенность csv: `description` содержит переносы строк. Читается парсером pandas и сохраняется в parquet, дальше работаем только с ним


## Что получается на выходе

| артефакт | для чего |
|---|---|
| `data/training/labels_final.parquet` | финальный датасет, 144 313 строк, 20 колонок |
| `data/manifest.yaml` | sha256 всех артефактов и граф parent -> child для воспроизводимости |
| `data/labeling/items.parquet` | csv после `ingest`, базовая таблица для всего |
| `data/labeling/gold.parquet` | ручная разметка 471 строки |
| `data/labeling/kb.yaml` | справочник модель -> класс (213 моделей) |
| `data/labeling/kb_labels.parquet` | разметка справочником, 55 535 объявлений |
| `data/labeling/llm_labels.parquet` | разметка LLM остатка, 88 778 объявлений |
| `data/labeling/llm_kb_check.parquet` | проверка kb-меток по фото через LLM |
| `data/labeling/audit_sample.parquet` | 210 строк для финальной ручной проверки качества |

В `labels_final.parquet` хранится не только итоговая метка, но и сырые метки от каждого источника отдельно $-$ так можно потом разобрать "почему этой строке поставлен такой класс" не пересчитывая всё заново


## Путь данных через пайплайн

```
items_project_aaa.csv + image/  (RustFS)
        |
        v
   stage_02_ingest_csv         csv -> items.parquet
        |
        v
   stage_03_sample_gold        стратификация ~500 строк -> gold.parquet (пустые метки)
        |
        v
   tools/label_gold.py         ручная разметка в Jupyter, 471 строка
        |
        v
   stage_04_build_kb           gold -> kb.yaml (213 моделей)
        |
        v
   stage_05_apply_kb           kb на items -> kb_labels.parquet (55 535)
        |
        +----> stage_06_label_unmatched   LLM с фото на 88k без kb-меток -> llm_labels.parquet
        |
        +----> stage_07_verify_kb         LLM с фото на 55k kb-меток    -> llm_kb_check.parquet
        |
        v
   stage_08_merge_decision     gold > kb (с проверкой по фото) > llm > abstain -> labels_final.parquet
        |
        v
   stage_09_audit_sample       210 случайных стратифицированно -> audit_sample.parquet
        |
        v
   tools/label_audit.py        ручная проверка в Jupyter
        |
        v
   stage_10_audit_eval         macro/weighted precision, разбор ошибок
        |
        v
   stage_11_manifest           sha256 всех артефактов -> manifest.yaml
```


## Стадии пайплайна

Все 11 файлов лежат в `pipeline/`, пронумерованы по порядку запуска. Каждый идемпотентный $-$ повторный запуск даёт тот же артефакт. Длинные стейджи (06 и 07, где работает LLM на десятках тысяч строк) пишут parquet через `.tmp` + `os.replace`, поэтому Ctrl+C не побьёт файл, и при следующем запуске они продолжат с того места где остановились

| стейдж | что делает | вход | выход |
|---|---|---|---|
| 01 download_raw | скачивает csv и фото из RustFS | креды в `.env` | `data/raw/` |
| 02 ingest_csv | csv в parquet, фиксирует sha256 | `items_project_aaa.csv` | `items.parquet` |
| 03 sample_gold | стратифицированная выборка под ручную разметку | `items.parquet` | `gold.parquet` |
| 04 build_kb | собирает справочник модель -> класс из размеченного gold | `gold.parquet` | `kb.yaml` |
| 05 apply_kb | применяет справочник ко всем 144k с негативным фильтром и multi-match | `items.parquet`, `kb.yaml` | `kb_labels.parquet` |
| 06 label_unmatched | LLM с фото на тех 89k где kb не сработал | `items.parquet`, `kb_labels.parquet`, промпт, фото | `llm_labels.parquet` |
| 07 verify_kb | LLM с фото проверяет 55k kb-меток (kb-фото не видит, мог ошибиться) | `items.parquet`, `kb_labels.parquet`, промпт, фото | `llm_kb_check.parquet` |
| 08 merge_decision | финальный мерж по приоритету источников | gold + kb + llm + kb_check | `labels_final.parquet` |
| 09 audit_sample | случайные 210 строк (35 на класс) под ручной аудит | `labels_final.parquet` | `audit_sample.parquet` |
| 10 audit_eval | метрики precision по заполненному руками аудиту | `audit_sample.parquet` + `labels_final.parquet` | stdout |
| 11 manifest | sha256 всех артефактов и parent-связи | все артефакты | `manifest.yaml` |


## Общий код в lib

В `lib/` вынесены модули которые используются больше чем одним стейджем $-$ чтобы не дублировать одну и ту же логику по разным скриптам

| модуль | что внутри |
|---|---|
| `paths.py` | все пути проекта в одном месте: raw, items, gold, kb, labels_final, prompts |
| `io.py` | atomic-запись parquet и json через `.tmp` + `os.replace`, чтение state-файлов |
| `schema.py` | Pydantic-модель `LabelResponse` для ответа LLM + проверка инварианта (валидная камера -> body_type задан, иначе body_type = null) |
| `text_norm.py` | нормализация моделей камер для справочника: lowercase, кириллица/латиница по доминирующему скрипту в токене, римские цифры -> арабские, ё/э -> е |
| `gemini.py` | обёртка над `google.genai` с retry, тарифами, единым конфигом и хелперами для разделения промпта на статичную и динамическую часть |


## Промпты

В `prompts/` три версии $-$ накопительная эволюция:

- `v1_minimal.md` $-$ минимальный, без фото (для истории)
- `v2_detailed.md` $-$ расширенные описания, hard negatives, без фото (для истории)
- `v3_with_vision.md` $-$ актуальная мультимодальная версия, её используют стейджи 06 и 07

Промпт разделён маркером `# ОБЪЯВЛЕНИЕ ДЛЯ РАЗМЕТКИ` на две части: всё до маркера $-$ статичная часть, которая один раз заливается в кеш Gemini на 26 часов и переиспользуется всеми запросами. Всё после маркера $-$ шаблон с `{TITLE}` / `{DESCRIPTION}`, в который подставляются конкретные поля объявления


## Вспомогательные скрипты в tools

Это не часть пайплайна, но было нужно по ходу разметки:

| скрипт | для чего |
|---|---|
| `label_gold.py` | интерактивные ячейки (`# %%`) для ручной разметки gold в Jupyter / VS Code |
| `label_audit.py` | то же самое для финальной ручной проверки `audit_sample.parquet` |
| `split_gold.py` | делит gold на dev/holdout 70/30 со стратификацией по `final_label` для калибровки промпта |
| `smoke_test.py` | прогоняет промпт на 50 строках из dev для быстрой проверки $-$ парсинг, бизнес-инвариант, accuracy, confusion matrix |
| `eval_holdout.py` | прогоняет промпт на отложенном holdout (141 строка) для финального честного замера precision |


## Таксономия

Полная схема классов и инварианты $-$ в `data/taxonomy.yaml`. Тут только короткое напоминание

Разметка двухуровневая:

**Уровень 1 $-$ `object_status`** (10 значений): что вообще за объект в объявлении

| значение | смысл |
|---|---|
| `valid_single_film_camera` | одна валидная плёночная камера |
| `multi_camera_lot` | лот из нескольких камер |
| `accessory_or_part` | аксессуар или часть камеры |
| `film_or_consumable` | плёнка, картриджи, химия, фотобумага |
| `digital_camera` | цифровая камера попала в категорию плёночных |
| `box_manual_packaging` | коробка, инструкция или упаковка без камеры |
| `not_camera` | вообще не камера |
| `insufficient_info` | по тексту и фото надёжно не понять |
| `image_unavailable` | фото битое, а одного текста не хватает |
| `conflicting_evidence` | текст и фото противоречат |

**Уровень 2 $-$ `body_type`** (5 значений, только если на первом уровне `valid_single_film_camera`):

| значение | смысл |
|---|---|
| `SLR` | однообъективная зеркальная |
| `TLR` | двухобъективная |
| `rangefinder_viewfinder` | дальномерная или видоискательная без зеркальной призмы |
| `compact_point_and_shoot` | мыльница |
| `instant` | моментальной печати, Polaroid / Instax |

`final_label` равен `body_type` если объект валиден и `body_type` задан, иначе `other_unknown`


## Источники меток

Колонка `label_source` в `labels_final.parquet` говорит откуда пришла финальная метка для каждой строки:

| источник | когда ставится |
|---|---|
| `human` | строка есть в ручном gold |
| `kb` | справочник нашёл модель в title + description, по фото подтверждено (или kb_check пуст) |
| `kb_overridden` | справочник нашёл модель, но LLM по фото с уверенностью $\geq$ 0.85 сказала что это не валидная одиночная плёночная камера -> метка демоутится в `other_unknown` |
| `llm` | размечено LLM на этапе 06, ни kb ни gold не сработали |
| `abstain` | ни один источник не уверен -> `other_unknown` с `needs_review = True` |

Финальная метка выбирается строго по приоритету: gold -> kb (с проверкой по фото) -> llm -> abstain. Подробные правила решений и сложные случаи $-$ в `data/decision_rules.md`


## Как запустить

Что нужно:

- Python 3.13 (`uv venv` или обычная `python -m venv`)
- зависимости из корневого `requirements.txt`
- `.env` в корне с кредами RustFS (`RUSTFS_BUCKET`, `RUSTFS_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) и `GEMINI_API_KEY`

Запуск стейджей из корня проекта:

```bash
python data-pipeline/pipeline/stage_01_download_raw.py
python data-pipeline/pipeline/stage_02_ingest_csv.py
python data-pipeline/pipeline/stage_03_sample_gold.py
# тут руками размечаем gold через tools/label_gold.py в Jupyter
python data-pipeline/pipeline/stage_04_build_kb.py
python data-pipeline/pipeline/stage_05_apply_kb.py
python data-pipeline/pipeline/stage_06_label_unmatched.py
python data-pipeline/pipeline/stage_07_verify_kb.py
python data-pipeline/pipeline/stage_08_merge_decision.py
python data-pipeline/pipeline/stage_09_audit_sample.py
# тут руками проверяем аудит через tools/label_audit.py в Jupyter
python data-pipeline/pipeline/stage_10_audit_eval.py
python data-pipeline/pipeline/stage_11_manifest.py
```

Любой стейдж можно прервать Ctrl+C и запустить заново $-$ длинные продолжат с того же места, короткие просто перепишут свой parquet


## Версия и воспроизводимость

Исходный csv `items_project_aaa.csv`, sha256:  
`8b9f42e58962938e05015bcc7e445737079b9f399b5be2e4d841c94e4b13e34d`

`random_state=42` зафиксирован во всех `sample()` (стратификация gold, dev/holdout split, audit sample). sha256 всех артефактов и parent -> child граф собираются в `data/manifest.yaml` через стейдж 11
