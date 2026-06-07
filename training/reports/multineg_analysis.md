# Multi-negative other_unknown vs чемпион: lrgrid__c1.0__none

Негатив train разбит на silver-подтипы (regex), фит на 5 main + K other_*, на инференсе p_other=sum(подтипы) -> 6 классов. Operating point - tie-break (max C/min W) на val. Вопрос: уводит ли типизатор больше ловушек в argmax-other (W vs чемпион, без порога)

## Сравнение в 3 проекциях

### val / final_label

| схема | CAR | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- |
| champion (6-класс) | 0.787 | 0.041 | 0.044 | 15757 | 15113 | 644 |
| multi-negative | 0.786 | 0.043 | 0.046 | 15765 | 15088 | 677 |

### test / final_label (22k)

| схема | CAR | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- |
| champion (6-класс) | 0.835 | 0.038 | 0.041 | 19167 | 18440 | 727 |
| multi-negative | 0.836 | 0.040 | 0.043 | 19229 | 18457 | 772 |

### test / audit_label (210)

| схема | CAR | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- |
| champion (6-класс) | 0.748 | 0.025 | 0.062 | 161 | 157 | 4 |
| multi-negative | 0.748 | 0.031 | 0.070 | 162 | 157 | 5 |

## Вердикт

```
champion (6-класс): val CAR 0.787 W=644 | audit CAR 0.748 W=4 AER_hi 0.062
multi-negative: val CAR 0.786 W=677 (ΔCAR-0.001) | audit CAR 0.748 W=5 AER_hi 0.070
ВЕРДИКТ: multi-negative НЕ лучше чемпиона на честной метрике
```
