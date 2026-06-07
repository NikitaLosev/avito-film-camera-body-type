# OOF modality-stacking vs чемпион: lrgrid__c1.0__none

Мета над per-modality вероятностями 5 баз (champ-fusion + tfidf/qwen/dino/pe solo) + рассогласование (кто с кем спорит). Мета ПЕРЕРАНЖИРУЕТ argmax (не только гейт). OOF train (leak-free, SGKF по leakage_group_id), базы val/test на полном train. Operating point - tie-break (max C/min W) на val. Судить по val (19k), proxy - teacher-overfit чек, audit (210) санити - не короновать по нему

## Сравнение в 3 проекциях

### val / final_label (подбор tau)

| схема | CAR | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- |
| champion | 0.787 | 0.041 | 0.044 | 15757 | 15113 | 644 |
| stack logreg | 0.785 | 0.036 | 0.039 | 15632 | 15064 | 568 |
| stack HGB | 0.786 | 0.035 | 0.038 | 15629 | 15079 | 550 |

### test / final_label (proxy 22k)

| схема | CAR | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- |
| champion | 0.835 | 0.038 | 0.041 | 19167 | 18440 | 727 |
| stack logreg | 0.834 | 0.035 | 0.038 | 19083 | 18418 | 665 |
| stack HGB | 0.833 | 0.036 | 0.038 | 19073 | 18394 | 679 |

### test / audit_label (210)

| схема | CAR | AER | AER_hi | A | C | W |
| --- | --- | --- | --- | --- | --- | --- |
| champion | 0.748 | 0.025 | 0.062 | 161 | 157 | 4 |
| stack logreg | 0.748 | 0.019 | 0.054 | 160 | 157 | 3 |
| stack HGB | 0.752 | 0.019 | 0.053 | 161 | 158 | 3 |

## Вердикт

```
champion: val CAR 0.787 W=644 | proxy 0.835 | audit CAR 0.748 W=4 AER_hi 0.062
stack logreg: val CAR 0.785 W=568 (ΔCAR-0.003 ΔW-76) | proxy 0.834(Δ-0.001) | audit CAR 0.748 W=3 AER_hi 0.054
stack HGB: val CAR 0.786 W=550 (ΔCAR-0.002 ΔW-94) | proxy 0.833(Δ-0.002) | audit CAR 0.752 W=3 AER_hi 0.053
ВЕРДИКТ: стекинг НЕ двигает val - сигнал рассогласования модальностей не извлекаем сверх fusion, B у потолка
```
