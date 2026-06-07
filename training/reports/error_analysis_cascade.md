# Error-analysis: cgrid__lightgbm__logreg

Рабочий порог tau = 0.00. Решение про каскад берём по **audit** (vs человек), структуру типов - по **proxy** (vs Gemini, 22k, статмощность)

## 1. Декомпозиция ошибок автозаполнения W (компас)

Каждую ошибку автозаполнения относим к: should_abstain (правда other_unknown - лот/коробка/не камера) или wrong_type (камера, но перепутан тип)

| истина | автозаполнено A | ошибок W | should_abstain | wrong_type |
| --- | --- | --- | --- | --- |
| audit (vs человек, шумно) | 162 | 10 | 8 (80%) | 2 (20%) |
| proxy (vs Gemini, 22k) | 18783 | 939 | 576 (61%) | 363 (39%) |

## 2. Confusion 6x6 (proxy-test, при tau)

Строки - истина, столбцы - предсказание. Блок 5x5 без OTH = путаница ТИПОВ, строка/столбец OTH = поведение отказа. RV=rangefinder_viewfinder, CPS=compact, INST=instant, OTH=other_unknown

| true \ pred | SLR | TLR | RV | CPS | INST | OTH |
| --- | --- | --- | --- | --- | --- | --- |
| SLR | 4502 | 19 | 31 | 12 | 2 | 121 |
| TLR | 10 | 586 | 8 | 2 | 0 | 36 |
| RV | 34 | 20 | 3544 | 47 | 0 | 193 |
| CPS | 28 | 3 | 130 | 7539 | 4 | 242 |
| INST | 1 | 0 | 2 | 10 | 1673 | 86 |
| OTH | 163 | 7 | 102 | 220 | 84 | 2618 |

## 3. Ошибки по срезам (proxy-test)

### slice_generic_title

| значение | n | CAR | AER | AER_hi |
| --- | --- | --- | --- | --- |
| False | 22016 | 0.809 | 0.050 | 0.053 |
| True | 63 | 0.698 | 0.083 | 0.196 |

### slice_kb_model

| значение | n | CAR | AER | AER_hi |
| --- | --- | --- | --- | --- |
| no_kb | 14789 | 0.764 | 0.061 | 0.065 |
| rare_kb | 707 | 0.932 | 0.024 | 0.038 |
| popular_kb | 6583 | 0.894 | 0.031 | 0.036 |

### slice_tlr

| значение | n | CAR | AER | AER_hi |
| --- | --- | --- | --- | --- |
| False | 21437 | 0.805 | 0.051 | 0.054 |
| True | 642 | 0.913 | 0.033 | 0.050 |

### slice_source

| значение | n | CAR | AER | AER_hi |
| --- | --- | --- | --- | --- |
| abstain | 30 | 0.000 | 1.000 | 1.000 |
| human | 471 | 0.811 | 0.038 | 0.061 |
| kb | 6437 | 0.965 | 0.007 | 0.009 |
| kb_overridden | 495 | 0.000 | 1.000 | 1.000 |
| llm | 14646 | 0.768 | 0.060 | 0.064 |

### slice_seller

| значение | n | CAR | AER | AER_hi |
| --- | --- | --- | --- | --- |
| large | 19220 | 0.826 | 0.049 | 0.052 |
| medium | 2130 | 0.658 | 0.061 | 0.074 |
| small | 729 | 0.765 | 0.053 | 0.074 |

## 4. Примеры ошибок (по 10, proxy-test)

### should_abstain (правда other_unknown, а мы поставили тип)

| item_id | title | truth | pred | conf |
| --- | --- | --- | --- | --- |
| b1ad9b9e | Фотокамера аналоговая печати Porst Magic 500 | other_unknown | instant | 0.41 |
| 396f651d | Камера Красногорск-3 рабочая | other_unknown | SLR | 0.46 |
| 7b6efd26 | Фотоаппарат Зоркий 4К -Экспорт. 1976года | other_unknown | rangefinder_viewfinder | 0.66 |
| cccab4e1 | Фотоаппарат Olympus Six + Olympus Zuiko F.C. 7.5cm f/2. | other_unknown | rangefinder_viewfinder | 0.74 |
| 14b60568 | Polaroid как новый | other_unknown | instant | 0.59 |
| 99c8d2bb | Фотоаппарат любитель в ассортименте | other_unknown | TLR | 0.51 |
| 692bc53b | Olympus OM system OM-5 (black) (новый) | other_unknown | SLR | 0.88 |
| 13660412 | Nikon ZFC KIT with Nikkor Z 28mm f/2.8 Black (новы | other_unknown | SLR | 0.59 |
| 85d0a0c7 | Фотоаппарат Фэд-микрон 2 и фэд-микон | other_unknown | rangefinder_viewfinder | 0.79 |
| 1d674adf | Фотоаппараты пленочные Olympus | other_unknown | compact_point_and_shoot | 0.96 |

### wrong_type (камера, но не тот тип)

| item_id | title | truth | pred | conf |
| --- | --- | --- | --- | --- |
| d823b42c |  ujисю Flзнф AF Date | compact_point_and_shoot | rangefinder_viewfinder | 0.80 |
| b3d48a9c | Yashica Autofocus | compact_point_and_shoot | rangefinder_viewfinder | 0.55 |
| a085dc3a | Minolta AF-S Quartz | compact_point_and_shoot | SLR | 0.52 |
| 04cafbff | Yashica Auto Focus | compact_point_and_shoot | rangefinder_viewfinder | 0.54 |
| be572ce9 | Getter 1001 | compact_point_and_shoot | rangefinder_viewfinder | 0.60 |
| 272ecde6 | бinolta зeatstar A | compact_point_and_shoot | TLR | 0.44 |
| 3ef2dc28 | Малоформатный фотоаппарат Эликон 4 в упаковке | compact_point_and_shoot | rangefinder_viewfinder | 0.72 |
| d3c441f9 | Fujica ST-F полурабочий | compact_point_and_shoot | SLR | 0.92 |
| 8e162c97 | Playstation Realчпь | rangefinder_viewfinder | TLR | 0.48 |
| 87eb7d00 | Konica C35 EF | compact_point_and_shoot | rangefinder_viewfinder | 0.56 |

## 5. Вердикт компаса

```
audit (W=10, шумно): should_abstain 80% / wrong_type 20%
proxy (W=939, надёжно): should_abstain 61% / wrong_type 39%
ВЕРДИКТ: КАСКАД стейдж-1 (камера/не-камера) - доминируют ошибки should_abstain
```
