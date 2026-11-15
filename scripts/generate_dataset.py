"""Generate and save original-shaped JSON data without running any policy."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contextual_bandits.simulator import SCENARIOS
from contextual_bandits.workbench import make_world, dataset_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=SCENARIOS, default='recurring')
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--world', type=int, default=0)
    p.add_argument('--horizon', type=int, default=18250)
    p.add_argument('--start-date', default='2023-01-01')
    p.add_argument('--output-dir', type=Path, help='Parent directory for dataset folders')
    a = p.parse_args()
    world = make_world(a.scenario, a.seed, a.world, a.horizon,
                       start_date=a.start_date, dataset_parent=a.output_dir)
    print(f'Saved {len(world.contexts):,} records: {dataset_path(world, a.output_dir).resolve()}')


if __name__ == '__main__':
    main()
