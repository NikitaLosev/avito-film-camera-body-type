# sample_weight по label_source vs чемпион: lrgrid__c1.0__none

Вес train-строк по надёжности источника метки (kb выше, llm ниже) - анти-teacher-noise. label_source только train-вес, не фича. Operating point - tie-break (max C/min W) на val. uniform = sanity (должен воспроизвести чемпиона)

## Сравнение схем в 3 проекциях

### val / final_label

| схема | CAR | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- |
| champion (saved) | 0.787 | 0.041 | 0.044 | 15757 | 15113 | 644 |
| uniform (=champion) | 0.787 | 0.041 | 0.044 | 15757 | 15113 | 644 |
| mild (kb^ llm v) | 0.787 | 0.042 | 0.046 | 15777 | 15109 | 668 |
| aggr (kb^^ llm vv) | 0.786 | 0.044 | 0.047 | 15788 | 15095 | 693 |

### test / final_label (22k)

| схема | CAR | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- |
| champion (saved) | 0.835 | 0.038 | 0.041 | 19167 | 18440 | 727 |
| uniform (=champion) | 0.835 | 0.038 | 0.041 | 19167 | 18440 | 727 |
| mild (kb^ llm v) | 0.835 | 0.040 | 0.043 | 19216 | 18444 | 772 |
| aggr (kb^^ llm vv) | 0.834 | 0.041 | 0.044 | 19195 | 18410 | 785 |

### test / audit_label (210)

| схема | CAR | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- |
| champion (saved) | 0.748 | 0.025 | 0.062 | 161 | 157 | 4 |
| uniform (=champion) | 0.748 | 0.025 | 0.062 | 161 | 157 | 4 |
| mild (kb^ llm v) | 0.743 | 0.025 | 0.063 | 160 | 156 | 4 |
| aggr (kb^^ llm vv) | 0.738 | 0.019 | 0.054 | 158 | 155 | 3 |

## Вердикт

```
champion: val CAR 0.787 W=644 | audit CAR 0.748 W=4 AER_hi 0.062
uniform (=champion): val CAR 0.787 W=644 (ΔCAR+0.000) | audit CAR 0.748 W=4 AER_hi 0.062
mild (kb^ llm v): val CAR 0.787 W=668 (ΔCAR-0.000) | audit CAR 0.743 W=4 AER_hi 0.063
aggr (kb^^ llm vv): val CAR 0.786 W=693 (ΔCAR-0.001) | audit CAR 0.738 W=3 AER_hi 0.054
ВЕРДИКТ: sample_weight НЕ лучше чемпиона на честной метрике
```
