# Decision rules — корнер-кейсы финальной разметки

Этот документ закрывает корнер-кейсы по канону Avito academy (Морозова, лек.02):
«инструкция должна включать нетривиальные случаи». Описывает финальные
правила какие источники меток приоритетнее и почему

## Приоритет источников

Финальная метка для каждой строки в `labels_final.parquet` выбирается
по убывающему приоритету:

1. **`gold`** (human разметка) — 471 строка из `gold.parquet`. Всегда побеждает.
   Confidence фиксируется 1.0. Если позже обнаружим ошибку в gold —
   исправляем gold.parquet и пересобираем pipeline
2. **`kb`** (regex-эвристика) — ~55k строк из `kb_labels.parquet`. Precision
   измерен 96.4% на gold. Confidence 1.0. Бьёт LLM потому что:
   - Построен из ручной разметки (=human-validated mapping)
   - Применён с word-boundary и negative filter (защита от ложных срабатываний)
3. **`llm`** (Gemini 3.1 Flash-Lite на v3_with_vision) — ~88k строк.
   Precision 95.8% на gold_holdout. Confidence из ответа модели
4. **`abstain`** (fallback) — если ни один не сработал ИЛИ если LLM
   confidence < 0.7 → `final_label = other_unknown`, confidence = 0.0

## Confidence threshold 0.7 для LLM

На smoke и holdout все ошибки имели confidence 0.85-1.0 — то есть порог
по уверенности **не лечит ошибки модели**. Но 0.7 нужен как страховка:
- защита от случайно вылетевших низких confidence в неожиданных кейсах
- если на 89k проде LLM поставит 0.4 на что-то «потому что фото мутное» —
  это сигнал что мы не уверены, лучше пометить abstain

Cut-off можно подкрутить после анализа реального распределения confidence
на полном 89k. Если хвост confidence < 0.7 мал (1-2%), можно опустить
до 0.5. Если большой — повысить до 0.8

## Hard negatives которые LLM может пропустить (известные паттерны)

По результатам v3 на gold:
1. **Skina SK-555 / похожие обманки** — фото показывает другую модель чем title.
   Lifetime ограничение text+vision, accept as is. precision -1-2 пп на compact
2. **Olympus IS-серия** — внешне выглядит как мыльница (bridge SLR).
   На gold помечен как SLR, LLM ставит compact. Технически обе версии могут
   быть приемлемы — это hybrid класс
3. **«ФЭД-2 комплект полный»** — слово «комплект» триггерит лот-эвристику,
   но реально это просто комплект аксессуаров вокруг одной камеры.
   Recall miss ~0.5 пп на rangefinder
4. **Гибридные Instax**: Instax mini Evo (с большим LCD-экраном) и Instax
   mini Link/Link 3 (это **принтер**, не камера) → должны быть digital_camera/
   accessory_or_part → other_unknown
5. **Современные аналоговые Polaroid**: Polaroid Go Gen 2, Polaroid Now Gen 2,
   Polaroid I-2 — это **плёночные** камеры (НЕ цифровые), правильная метка
   `valid_single_film_camera + instant`. Если кто-то путает их с гибридами —
   проверяем по официальной странице, Polaroid сами называют их
   «entirely analog camera»

## Что делать с FAILED chunks

Если на этапе batch один из chunks падает в FAILED:
1. `python retry_chunk.py chunk_NN` — стирает batch_name из state
2. `python batch_submit.py chunk_NN` — пересабмит (cache переиспользуется)
3. Стоимость retry ~$2.3 за chunk

Если после 2-3 retry всё ещё FAILED — гонять этот chunk через standard API
(не batch, дороже но надёжнее) через ad-hoc скрипт. Или мерж без этого
chunk'а с пометкой в manifest «~8900 строк потеряно из-за batch fail»

## Random audit

После labels_final.parquet **обязательно** прогнать `audit_sample.py`:
- 200 случайных строк стратифицированно (35 на класс)
- Ручная проверка ~30 мин в Jupyter

После заполнения колонки `audit_label` прогнать `audit_eval.py`, который посчитает:
- **macro precision** — среднее `per_class_precision` (не учитывает дисбаланс
  классов на 144k)
- **weighted precision** — `sum(per_class_precision × class_share)` — это и есть
  честная цифра precision на полном 144k для отчёта
- **precision_by_label_source** — отдельно для kb / llm / human / abstain
  (нужно понять где главные ошибки)
- **precision на kb out-of-sample** — отдельная цифра (наш kb построен из gold,
  поэтому 96.4% на gold = leakage; на audit-sample получим честную out-of-sample)

## Версионирование

Каждый запуск pipeline создаёт `manifest.yaml` с:
- sha256 каждого артефакта (raw csv, items, gold, kb, llm, final)
- prompt_version и prompt_sha256
- model id (gemini-3.1-flash-lite)
- batch_jobs (batch IDs Gemini)
- parent → child граф

Если перезапускаем production batch (новые промпты, новые данные) —
manifest обновляется, старые артефакты с другим sha256 видны в git history
