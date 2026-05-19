"""Делит gold.parquet на dev/holdout (70/30) для калибровки промпта LLM

Стратификация по final_label, random_state=42
dev - можно смотреть при тюнинге, holdout - только для финального замера
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / 'data' / 'labeling' / 'gold.parquet'
DEV = PROJECT_ROOT / 'data' / 'labeling' / 'gold_dev.parquet'
HOLDOUT = PROJECT_ROOT / 'data' / 'labeling' / 'gold_holdout.parquet'

SEED = 42
HOLDOUT_FRAC = 0.3


def main():
    gold = pd.read_parquet(SRC)
    holdout = gold.groupby('final_label', group_keys=False).sample(
        frac=HOLDOUT_FRAC, random_state=SEED)
    dev = gold.drop(holdout.index)

    dev.reset_index(drop=True).to_parquet(DEV)
    holdout.reset_index(drop=True).to_parquet(HOLDOUT)

    print(f'dev: {len(dev)}, holdout: {len(holdout)}')
    print(pd.DataFrame({
        'dev': dev['final_label'].value_counts(),
        'holdout': holdout['final_label'].value_counts(),
    }))


if __name__ == '__main__':
    main()
