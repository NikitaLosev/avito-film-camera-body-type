# Per-class пороги + гигиена метрик: lrgrid__c1.0__none

Глобальный порог tau = 0.000 (подбор на val/final_label под пулинговым AER_hi<=5%). per-class пороги подобраны там же, оцениваются в 3 проекции. audit (210) - directional, с Wilson-CI

## 1. Гигиена: raw (без отказа, tau:0) vs post-tau, на val/final_label

raw автозаполняет argmax всегда (coverage=1), post-tau применяет глобальный порог. Сейчас в лог идут ТОЛЬКО post-tau - raw_macro_f1/raw_acc отдельно показывают качество модели до отказа

| метрика | raw (no-abstain) | post-tau (global) |
| --- | --- | --- |
| macro_f1 | 0.944 | 0.944 |
| accuracy | 0.952 | 0.952 |
| CAR | 0.787 | 0.787 |
| AER | 0.041 | 0.041 |
| AER_hi | 0.044 | 0.044 |
| coverage | 0.821 | 0.821 |

## 2. Пороги по классам

tau_cap - наивный (каждый класс под свой AER_hi<=5%, НЕ гарантирует пулинг); tau_pooled - основной (координатный спуск под пулинговый AER_hi<=5%)

| класс | tau_global | tau_cap | tau_pooled | cap feasible |
| --- | --- | --- | --- | --- |
| SLR | 0.000 | 0.543 | 0.000 | да |
| TLR | 0.000 | inf (off) | 0.000 | НЕТ |
| rangefinder_viewfinder | 0.000 | 0.600 | 0.000 | да |
| compact_point_and_shoot | 0.000 | 0.357 | 0.000 | да |
| instant | 0.000 | 0.352 | 0.000 | да |

## 3. Сравнение схем в 3 проекциях

val/final_label - где выигрыш заявлен; test/final_label - переносится ли в распределении (22k); test/audit_label - реальная бизнес-метрика (210 human, CI широк - читать как directional)

### val / final_label (подбор)

| схема | CAR | CAR 95% CI | AER | AER_hi | A | W |
| --- | --- | --- | --- | --- | --- | --- |
| global | 0.787 | [0.781, 0.793] | 0.041 | 0.044 | 15757 | 644 |
| per-class-cap | 0.775 | [0.769, 0.781] | 0.035 | 0.038 | 15425 | 543 |
| per-class-pooled | 0.787 | [0.781, 0.793] | 0.041 | 0.044 | 15757 | 644 |

### test / final_label (proxy 22k)

| схема | CAR | CAR 95% CI | AER | AER_hi | A | W |
| --- | --- | --- | --- | --- | --- | --- |
| global | 0.835 | [0.830, 0.840] | 0.038 | 0.041 | 19169 | 728 |
| per-class-cap | 0.803 | [0.797, 0.808] | 0.032 | 0.035 | 18310 | 588 |
| per-class-pooled | 0.835 | [0.830, 0.840] | 0.038 | 0.041 | 19169 | 728 |

### test / audit_label (human 210)

| схема | CAR | CAR 95% CI | AER | AER_hi | A | W |
| --- | --- | --- | --- | --- | --- | --- |
| global | 0.748 | [0.685, 0.802] | 0.025 | 0.062 | 161 | 4 |
| per-class-cap | 0.600 | [0.533, 0.664] | 0.008 | 0.043 | 127 | 1 |
| per-class-pooled | 0.748 | [0.685, 0.802] | 0.025 | 0.062 | 161 | 4 |

## 4. Вердикт переноса

```
val   CAR Δ (per-class - global): +0.000
proxy CAR Δ (test/final, 22k):     +0.000
audit CAR Δ (test/human, 210):     +0.000  | audit AER_hi per-class = 0.062
ВЕРДИКТ: ВНИМАНИЕ: выигрыш НЕ переносится на audit (CAR не вырос или AER_hi>5%) - это val-only артефакт под учителя, НЕ катить
```
