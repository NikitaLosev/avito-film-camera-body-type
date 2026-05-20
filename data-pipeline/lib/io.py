"""Безопасная запись файлов через временный путь и атомарный rename

Если процесс упадёт посреди записи - на диске останется либо старая версия,
либо новая, но не битая. Это важно для длинных прогонов которые могут
прерваться (Ctrl+C, OOM, и т.п.)
"""

import json
import os
from pathlib import Path

import pandas as pd


def atomic_write_parquet(df: pd.DataFrame, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp)
    os.replace(tmp, dst)


def atomic_write_json(data: dict, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, dst)


def read_state(path: Path, default=None) -> dict:
    if not path.exists():
        return default if default is not None else {}
    return json.loads(path.read_text())
