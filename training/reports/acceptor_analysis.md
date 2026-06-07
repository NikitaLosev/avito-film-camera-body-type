# OOF acceptor поверх чемпиона: lrgrid__c1.0__none

Тип-классификатор не тронут; acceptor меняет только гейт отказа. Operating point - **tie-break** селектор (точные кандидаты, max C -> min W -> min AER_hi), не max-CAR-первый: при равном CAR берёт минимум ошибочных автозаполнений. Подбор на val/final_label под AER_hi<=5%, оценка в 3 проекции. Вопрос: режет ли acceptor W без потери CAR и берёт ли строгий audit-гейт (AER_hi<=5%)

## Пороги схем (подбор на val)

| схема | gate-сигнал | tau |
| --- | --- | --- |
| baseline single-tau | confidence | 0.318 |
| acceptor logreg | accept_lr | 0.000 |
| acceptor HGB | accept_hgb | 0.253 |

## Сравнение схем в 3 проекциях

### val / final_label (подбор tau)

| схема | CAR | CAR 95% CI | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline single-tau | 0.787 | [0.781, 0.793] | 0.041 | 0.044 | 15757 | 15113 | 644 |
| acceptor logreg | 0.787 | [0.781, 0.793] | 0.041 | 0.044 | 15757 | 15113 | 644 |
| acceptor HGB | 0.787 | [0.781, 0.793] | 0.041 | 0.044 | 15752 | 15113 | 639 |

### test / final_label (proxy 22k)

| схема | CAR | CAR 95% CI | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline single-tau | 0.835 | [0.830, 0.840] | 0.038 | 0.041 | 19167 | 18440 | 727 |
| acceptor logreg | 0.835 | [0.830, 0.840] | 0.038 | 0.041 | 19169 | 18441 | 728 |
| acceptor HGB | 0.835 | [0.830, 0.840] | 0.038 | 0.041 | 19163 | 18436 | 727 |

### test / audit_label (human 210)

| схема | CAR | CAR 95% CI | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline single-tau | 0.748 | [0.685, 0.802] | 0.025 | 0.062 | 161 | 157 | 4 |
| acceptor logreg | 0.748 | [0.685, 0.802] | 0.025 | 0.062 | 161 | 157 | 4 |
| acceptor HGB | 0.748 | [0.685, 0.802] | 0.025 | 0.062 | 161 | 157 | 4 |

## Вердикт

```
baseline: val CAR 0.787 C=15113 W=644 | audit CAR 0.748 W=4 AER_hi 0.062
acceptor logreg: val CAR 0.787 (ΔCAR+0.000, ΔW+0) | audit CAR 0.748 W=4 AER_hi 0.062 -> без сдвига
acceptor HGB: val CAR 0.787 (ΔCAR+0.000, ΔW-5) | audit CAR 0.748 W=4 AER_hi 0.062 -> val W срезан без потери CAR
ВЕРДИКТ: строгий гейт не взят даже с tie-break; W не режется без потери CAR
```
