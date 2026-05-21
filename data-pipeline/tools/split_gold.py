"""Разбивает gold.parquet на dev и holdout (70/30) для калибровки промпта

Стратификация по final_label, фиксированный seed чтобы выборка была одна и та же
при повторных запусках. dev - смотрим во время тюнинга, holdout - только для
финального честного замера precision
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import GOLD, GOLD_DEV, GOLD_HOLDOUT

SEED = 42
HOLDOUT_FRAC = 0.3


def main():
    gold = pd.read_parquet(GOLD)
    holdout = gold.groupby('final_label', group_keys=False).sample(
        frac=HOLDOUT_FRAC, random_state=SEED)
    dev = gold.drop(holdout.index)

    dev.reset_index(drop=True).to_parquet(GOLD_DEV)
    holdout.reset_index(drop=True).to_parquet(GOLD_HOLDOUT)

    print(f'dev: {len(dev)}, holdout: {len(holdout)}')
    print(pd.DataFrame({
        'dev': dev['final_label'].value_counts(),
        'holdout': holdout['final_label'].value_counts(),
    }))


if __name__ == '__main__':
    main()
