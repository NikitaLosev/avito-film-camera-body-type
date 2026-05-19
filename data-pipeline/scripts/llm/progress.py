"""Общие утилиты для prod-пайплайна LLM-разметки

- atomic_write: безопасная запись parquet/json через tmp+replace
- read_state / write_state: JSON state-файлы (для resume)
- estimate_eta: красивая строка с прогрессом и оставшимся временем
"""

import json
import os
from pathlib import Path

import pandas as pd


def atomic_write_parquet(df: pd.DataFrame, dst: Path):
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp)
    os.replace(tmp, dst)


def atomic_write_json(data: dict, dst: Path):
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, dst)


def read_state(path: Path, default=None):
    if not path.exists():
        return default if default is not None else {}
    return json.loads(path.read_text())
