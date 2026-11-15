"""Run a new exploratory study, outside frozen evidence, without Jupyter or an AI tool."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(name,'1')
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from contextual_bandits.workbench import default_settings,run_experiment,save_experiment,summary_table
from contextual_bandits.simulator import SCENARIOS
from contextual_bandits.policies import METHODS
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=['recurring'])
p.add_argument('--methods',nargs='+',choices=METHODS,default=['ucb','sw_ucb','discount_ucb','ts'])
p.add_argument('--repeats',type=int,default=4)
p.add_argument('--horizon',type=int,default=18250)
p.add_argument('--seed',type=int,default=20261108)
a=p.parse_args();f=default_settings()
e=run_experiment(a.scenarios,a.methods,f['configs'],f['fixed'],repeats=a.repeats,horizon=a.horizon,namespace=a.seed)
print(summary_table(e).to_string(index=False));print('Saved:',save_experiment(e))
for dataset in e.metadata['datasets']:
    print(f"Dataset world {dataset['world']} ({dataset['scenario']}): {dataset['path']}")
