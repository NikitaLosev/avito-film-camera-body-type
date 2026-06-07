"""Запуск экспериментов. Что гонять - задаётся в коде списком TO_RUN (experiments.py)

  python training/src/train.py          # гонит TO_RUN из experiments.py
  python training/src/train.py --all    # весь каталог EXPERIMENTS
  python training/src/train.py --list   # показать каталог
  python training/src/train.py a b      # точечно по именам (override TO_RUN)

Данные грузятся один раз на набор блоков и переиспользуются между конфигами
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from experiments import EXPERIMENTS, TO_RUN
from features import load_blocks
from pipeline import run_cascade, run_experiment
from tracking.client import export_runs_csv


def config_blocks(cfg: dict) -> list:
    """Блоки для загрузки данных: flat - свои, cascade - объединение стейджей без дублей"""
    specs = cfg['stage1']['blocks'] + cfg['stage2']['blocks'] if cfg.get('kind') == 'cascade' else cfg['blocks']
    seen, out = set(), []
    for s in specs:
        key = (s['block'], s.get('source', ''))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _blocks_key(cfg: dict) -> tuple:
    """Ключ набора блоков, чтобы переиспользовать загрузку данных между конфигами"""
    return tuple(sorted((s['block'], s.get('source', '')) for s in config_blocks(cfg)))


def run_many(names: list) -> None:
    """Прогнать список экспериментов, кэшируя загрузку данных по набору блоков"""
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f'нет экспериментов: {", ".join(unknown)}')
    print('гоню:', ', '.join(names))
    cache = {}
    for name in names:
        cfg = EXPERIMENTS[name]
        key = _blocks_key(cfg)
        if key not in cache:
            cache[key] = load_blocks(config_blocks(cfg))
        run = run_cascade if cfg.get('kind') == 'cascade' else run_experiment
        run(cfg, cache[key])
    export_runs_csv()


def main():
    parser = argparse.ArgumentParser(description='запуск экспериментов обучения')
    parser.add_argument('names', nargs='*', help='имена экспериментов (по умолчанию TO_RUN из experiments.py)')
    parser.add_argument('--all', action='store_true', help='прогнать весь каталог EXPERIMENTS')
    parser.add_argument('--list', action='store_true', help='показать каталог экспериментов')
    args = parser.parse_args()

    if args.list:
        for name in EXPERIMENTS:
            mark = '*' if name in TO_RUN else ' '
            print(f' {mark} {name}')
        return
    if args.all:
        run_many(list(EXPERIMENTS))
    elif args.names:
        run_many(args.names)
    else:
        run_many(TO_RUN)


if __name__ == '__main__':
    main()
