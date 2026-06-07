# Лексическая сетка (vocab/analyzer) vs чемпион: lrgrid__c1.0__none

Меняем ТОЛЬКО лексический блок, emb (qwen+dino+pe) бит-в-бит из чемпион-X. char_wb=(3,5) min_df5/50k. Винты: retain-tail (min_df2/150k), word(1,2), char-phrase(4,6) через пробел. Operating point - tie-break (max C/min W) на val. baseline (rebuild) = sanity. Судить по val (19k надёжно), proxy - teacher-overfit чек, audit (210) только санити - не короновать по нему (см. qwen×0.5 в blockfeat)

## Все конфиги

| конфиг | lex dims | val CAR | val W | proxy CAR | audit CAR | audit W | audit AER_hi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| champion (saved) | 50000 | 0.787 | 644 | 0.835 | 0.748 | 4 | 0.062 |
| baseline (rebuild) | 50000 | 0.787 | 644 | 0.835 | 0.748 | 4 | 0.062 |
| retain-tail | 150000 | 0.787 | 647 | 0.836 | 0.748 | 3 | 0.054 |
| +word(1,2) | 150000 | 0.788 | 628 | 0.837 | 0.743 | 6 | 0.078 |
| retain +word | 250000 | 0.788 | 623 | 0.837 | 0.748 | 5 | 0.070 |
| +char-phrase(4,6) | 150000 | 0.787 | 643 | 0.835 | 0.743 | 5 | 0.071 |
| retain +word +phrase | 350000 | 0.788 | 620 | 0.837 | 0.743 | 5 | 0.071 |

## Вердикт

```
champion: val CAR 0.787 W=644 | audit CAR 0.748 W=4 AER_hi 0.062
baseline (rebuild): val CAR 0.787 W=644 (ΔCAR+0.000) | audit CAR 0.748 W=4 AER_hi 0.062  (sanity ≈ champion)
retain-tail: val CAR 0.787 W=647 (ΔCAR-0.000) | audit CAR 0.748 W=3 AER_hi 0.054
+word(1,2): val CAR 0.788 W=628 (ΔCAR+0.001) | audit CAR 0.743 W=6 AER_hi 0.078  <- val-выигрыш
retain +word: val CAR 0.788 W=623 (ΔCAR+0.001) | audit CAR 0.748 W=5 AER_hi 0.070  <- val-выигрыш
+char-phrase(4,6): val CAR 0.787 W=643 (ΔCAR-0.001) | audit CAR 0.743 W=5 AER_hi 0.071
retain +word +phrase: val CAR 0.788 W=620 (ΔCAR+0.000) | audit CAR 0.743 W=5 AER_hi 0.071  <- val-выигрыш
ВЕРДИКТ: двигает val: ['+word(1,2)', 'retain +word', 'retain +word +phrase'] - смотреть детально (судить по val, audit только санити)
```
