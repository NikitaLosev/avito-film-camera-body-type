# Бизнес-policy: rescue false-abstain (демоут other) vs чемпион: lrgrid__c1.0__none

CAR-ось со стороны pred_other->main (kNN-probe её не смотрел). Приём: p_other *= alpha, пере-argmax. READ-ONLY над сохранёнными вероятностями (champion + stack). alpha выбран на val (max C при AER_hi<=5%). alpha=1.0 = чемпион (sanity). Судить по val; proxy - teacher-overfit чек; audit (210) только санити

## Политики (alpha=1.0 без rescue vs alpha* с rescue)

| политика | alpha | val CAR | val C | val W | val AER_hi | proxy CAR | audit CAR | audit W | audit AER_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| champion @1.0 (без rescue) | 1.000 | 0.787 | 15113 | 644 | 0.044 | 0.835 | 0.748 | 4 | 0.062 |
| champion @0.705 (rescue) | 0.705 | 0.790 | 15162 | 733 | 0.049 | 0.838 | 0.752 | 4 | 0.062 |
| stack_logreg @1.0 (без rescue) | 1.000 | 0.785 | 15064 | 575 | 0.040 | 0.834 | 0.748 | 3 | 0.054 |
| stack_logreg @0.351 (rescue) | 0.351 | 0.790 | 15162 | 738 | 0.050 | 0.838 | 0.752 | 5 | 0.070 |
| stack_HGB @1.0 (без rescue) | 1.000 | 0.786 | 15079 | 551 | 0.038 | 0.833 | 0.752 | 3 | 0.053 |
| stack_HGB @0.413 (rescue) | 0.413 | 0.791 | 15181 | 740 | 0.050 | 0.839 | 0.752 | 7 | 0.085 |

## Вердикт

```
reference champion @1.0: val CAR 0.7873 C=15113 W=644 | false-abstain на val (потолок rescue) = 269 строк (~1.4pp CAR)
champion @0.705: ΔC+49 (ΔCAR+0.0026), rescue 49/138 prec 35.5% | val AER_hi 0.049 | audit CAR 0.752 W=4 AER_hi 0.062  <- rescue растит C
stack_logreg @0.351: ΔC+49 (ΔCAR+0.0026), rescue 98/261 prec 37.5% | val AER_hi 0.050 | audit CAR 0.752 W=5 AER_hi 0.070  <- rescue растит C
stack_HGB @0.413: ΔC+68 (ΔCAR+0.0035), rescue 102/291 prec 35.1% | val AER_hi 0.050 | audit CAR 0.752 W=7 AER_hi 0.085  <- rescue растит C
ВЕРДИКТ: rescue РЕАЛЕН на val: ['champion', 'stack_logreg', 'stack_HGB'] - смотреть детально (судить по val; proxy↑ без audit = overfit; короновать только устойчивый ΔC, не шум)
```
