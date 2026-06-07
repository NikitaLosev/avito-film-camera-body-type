# Error-analysis: lrgrid__c1.0__none

Рабочий порог tau = 0.00. Решение про каскад берём по **audit** (vs человек), структуру типов - по **proxy** (vs Gemini, 22k, статмощность)

## 1. Декомпозиция ошибок автозаполнения W (компас)

Каждую ошибку автозаполнения относим к: should_abstain (правда other_unknown - лот/коробка/не камера) или wrong_type (камера, но перепутан тип)

| истина | автозаполнено A | ошибок W | should_abstain | wrong_type |
| --- | --- | --- | --- | --- |
| audit (vs человек, шумно) | 161 | 4 | 3 (75%) | 1 (25%) |
| proxy (vs Gemini, 22k) | 19169 | 728 | 482 (66%) | 246 (34%) |

## 2. Confusion 6x6 (proxy-test, при tau)

Строки - истина, столбцы - предсказание. Блок 5x5 без OTH = путаница ТИПОВ, строка/столбец OTH = поведение отказа. RV=rangefinder_viewfinder, CPS=compact, INST=instant, OTH=other_unknown

| true \ pred | SLR | TLR | RV | CPS | INST | OTH |
| --- | --- | --- | --- | --- | --- | --- |
| SLR | 4603 | 1 | 26 | 15 | 0 | 42 |
| TLR | 16 | 587 | 25 | 4 | 0 | 10 |
| RV | 22 | 2 | 3707 | 47 | 0 | 60 |
| CPS | 8 | 0 | 66 | 7811 | 1 | 60 |
| INST | 1 | 0 | 3 | 9 | 1733 | 26 |
| OTH | 127 | 8 | 111 | 169 | 67 | 2712 |

## 3. Ошибки по срезам (proxy-test)

### slice_generic_title

| значение | n | CAR | AER | AER_hi |
| --- | --- | --- | --- | --- |
| False | 22016 | 0.835 | 0.038 | 0.041 |
| True | 63 | 0.778 | 0.039 | 0.132 |

### slice_kb_model

| значение | n | CAR | AER | AER_hi |
| --- | --- | --- | --- | --- |
| no_kb | 14789 | 0.795 | 0.044 | 0.048 |
| rare_kb | 707 | 0.955 | 0.017 | 0.030 |
| popular_kb | 6583 | 0.912 | 0.028 | 0.032 |

### slice_tlr

| значение | n | CAR | AER | AER_hi |
| --- | --- | --- | --- | --- |
| False | 21437 | 0.833 | 0.037 | 0.040 |
| True | 642 | 0.914 | 0.071 | 0.094 |

### slice_source

| значение | n | CAR | AER | AER_hi |
| --- | --- | --- | --- | --- |
| abstain | 30 | 0.000 | 1.000 | 1.000 |
| human | 471 | 0.841 | 0.032 | 0.054 |
| kb | 6437 | 0.985 | 0.008 | 0.010 |
| kb_overridden | 495 | 0.000 | 1.000 | 1.000 |
| llm | 14646 | 0.799 | 0.043 | 0.047 |

### slice_seller

| значение | n | CAR | AER | AER_hi |
| --- | --- | --- | --- | --- |
| large | 19220 | 0.853 | 0.037 | 0.040 |
| medium | 2130 | 0.692 | 0.045 | 0.056 |
| small | 729 | 0.789 | 0.042 | 0.061 |

## 4. Примеры ошибок (по 10, proxy-test)

### should_abstain (правда other_unknown, а мы поставили тип)

| item_id | title | truth | pred | conf |
| --- | --- | --- | --- | --- |
| 396f651d | Камера Красногорск-3 рабочая | other_unknown | SLR | 0.51 |
| 014d1f7a | Olympus OM system OM-5 (silver) (Абсолютно новый) | other_unknown | SLR | 0.60 |
| cccab4e1 | Фотоаппарат Olympus Six + Olympus Zuiko F.C. 7.5cm f/2. | other_unknown | rangefinder_viewfinder | 0.85 |
| 10c73dd6 | Видоискатель оптический f3.5 см на фотоаппарат | other_unknown | compact_point_and_shoot | 0.51 |
| a036e6ab | Пленочный фотоаппарат remark F-101 st | other_unknown | compact_point_and_shoot | 0.71 |
| 89d5f70c | Пленочный винтажный дольномер minolta | other_unknown | rangefinder_viewfinder | 0.71 |
| 85d0a0c7 | Фотоаппарат Фэд-микрон 2 и фэд-микон | other_unknown | rangefinder_viewfinder | 0.82 |
| 0df64d93 | ьэд нквд 11074 | other_unknown | rangefinder_viewfinder | 0.55 |
| 1d674adf | Фотоаппараты пленочные Olympus | other_unknown | compact_point_and_shoot | 0.51 |
| e569b716 | Фотоаппарат Lomo LC-A | other_unknown | rangefinder_viewfinder | 0.53 |

### wrong_type (камера, но не тот тип)

| item_id | title | truth | pred | conf |
| --- | --- | --- | --- | --- |
| d3c441f9 | Fujica ST-F полурабочий | compact_point_and_shoot | SLR | 0.88 |
| 87eb7d00 | Konica C35 EF | compact_point_and_shoot | rangefinder_viewfinder | 0.45 |
| 883980e4 | Винтажный фотик Kodak Colorburst 350 — новый | instant | compact_point_and_shoot | 0.54 |
| f111f25a | Пленочный фотоаппарат Sprocket Rocket | rangefinder_viewfinder | compact_point_and_shoot | 0.54 |
| 2200b536 | Фотоаппарат EXA 1B | SLR | rangefinder_viewfinder | 0.71 |
| f9371909 | Konica C35 EF с нюансом | rangefinder_viewfinder | compact_point_and_shoot | 0.63 |
| 5b077da7 | Konica C35 EF с нюансом | rangefinder_viewfinder | compact_point_and_shoot | 0.50 |
| 40c1b965 | Киев 303 | compact_point_and_shoot | rangefinder_viewfinder | 0.44 |
| cc21913d | Плёночный фотоаппарат Olympus IS-200 в упаковке | SLR | compact_point_and_shoot | 0.72 |
| b0b5dd00 | Фотоаппарат Braunoptik imperial box | rangefinder_viewfinder | TLR | 0.35 |

## 5. Вердикт компаса

```
audit (W=4, шумно): should_abstain 75% / wrong_type 25%
proxy (W=728, надёжно): should_abstain 66% / wrong_type 34%
ВЕРДИКТ: КАСКАД стейдж-1 (камера/не-камера) - доминируют ошибки should_abstain
```
