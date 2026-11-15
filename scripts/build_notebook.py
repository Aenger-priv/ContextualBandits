"""Build the mixed-context workbench; never overwrite the user's older notebook."""
from pathlib import Path
import textwrap
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))
md('''
# Mixed-context bandits: data, learning and change

This workbench uses the original **context/rewards JSON interface**: currency pair,
volatility, size, time of day and date; four initial options and a fifth arriving later.
The learner chooses using context and eligibility, then receives **only its chosen reward**.
Potential rewards and expected means are evaluator-only simulation information.

In VS Code, select `.venv-notebook/bin/python` as the notebook kernel, then **Restart
Kernel and Run All**. First-time setup from the project root:
`python3.11 scripts/setup_environment.py`.
No AI service is needed. Default Monte Carlo is a small exploratory run, separate from
the saved studies. All plots are editable Matplotlib figures; change their cells to
redraw without rerunning learning. Save your own notebook copy before experimenting.
''')
code('''
import os, sys, json, copy
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'.cache/matplotlib'))
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents]
          if (p/'contextual_bandits/simulator.py').exists())
sys.path.insert(0,str(ROOT))
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, FileLink
from contextual_bandits.simulator import SCENARIOS, FEATURE_NAMES, from_records
from contextual_bandits.interface import ContextPolicy
from contextual_bandits.workbench import (
    default_settings, make_world, dataset_path, world_dates, event_table, run_experiment,
    summary_table, paired_table, save_experiment, load_experiment,
    plot_world, plot_rewards, plot_regret, plot_choices, plot_shift_response)
plt.rcParams.update({'figure.dpi':110,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
defaults=default_settings()
print('Python:',sys.version.split()[0])
print('Configuration loaded. All runs save their own data and settings.')
print('Scenarios:',', '.join(SCENARIOS))
''')
md('''
## 1. Controls

The default is 365 days × 50 decisions = 18,250 records. `HORIZON` must be a positive
multiple of 365. `START_DATE` changes actual generated dates. All scenarios introduce
option 4 on zero-based day 180. `recurring` switches at days 90 and 240; abrupt scenarios
switch at day 180. Use `SHIFT_FRACTIONS=[0.25,0.65]` to move recurring switches, or one
fraction for an abrupt scenario. This changes exploratory data, never the saved study settings.

`birth_strong`, `birth_inferior`, `birth_tie` and `birth_contextual` control the fifth
option's quality. Other scenarios test noise, drift, nonlinear rewards, uninformative
context and categorical imbalance. See the protocol and simulator for exact laws.
''')
code('''
SCENARIO='recurring'
HORIZON=18250
START_DATE='2023-01-01'
SHIFT_FRACTIONS=None
EXPLORATION_SEED=20261105
SELECTED_METHODS=['fixed','ucb','sw_ucb','discount_ucb','ts','epsilon','softmax','dr_softmax']
CONFIGS=copy.deepcopy(defaults['configs'])
FIXED_ARMS=copy.deepcopy(defaults['fixed'])
# Try CONFIGS['sw_ucb']={'alpha':0.5,'window':300}, then rerun learning.
display(pd.DataFrame([{'method':m,'config':CONFIGS[m]} for m in SELECTED_METHODS]))
''')
md('''
## 2. Generate the raw records and inspect the encoding

The first dataframe is the original five-field context, **not a replacement schema**.
The second contains the fixed model encoding: intercept, five currency indicators,
four time indicators and two scaled numerical fields. Date controls chronology and
is excluded from regression. Unit ridge handles the redundant full one-hot indicators.
No category vocabulary or scaling constant is learned from future data.

Generation automatically saves **plain, indented JSON** in
`datasets/<dataset-id>/fx_trading_dataset.json`.
Open the link below to inspect or reuse all records. Identical generated data reuse
the same directory; different worlds get separate files. The original root dataset
is preserved. `expected_rewards.json` alongside it contains evaluator-only means,
and `metadata.json` records the generation settings and checksums.
''')
code('''
world=make_world(SCENARIO,EXPLORATION_SEED,0,HORIZON,SHIFT_FRACTIONS,START_DATE)
DATASET_PATH=dataset_path(world)
print('Saved dataset:',DATASET_PATH)
display(FileLink(str(DATASET_PATH.relative_to(ROOT))))
records=world.records()
print(json.dumps(records[0],indent=2))
contexts=pd.DataFrame(world.contexts)
encoded=pd.DataFrame(world.x,columns=FEATURE_NAMES)
dates=world_dates(world)
display(contexts.head(8))
display(encoded.head(8))
display(contexts.describe(include='all'))
display(event_table(world,dates))
birth=world.births[0][0]
print('Reward keys before/after arrival:',list(records[birth-1]['rewards']),list(records[birth]['rewards']))
assert from_records(json.loads(json.dumps(records)),world.mu).digest()==world.digest()
print('Raw JSON round trip verified:',world.digest())
''')
code('''
DATA_SMOOTHING=250  # five days at 50 decisions/day
fig=plot_world(world,window=DATA_SMOOTHING)
plt.show()
''')
md('''
## 3. Use existing JSON or the online interface

`from_records` accepts the original JSON. Expected regret requires a separate known
mean matrix. `rewards_are_means=True` instead declares the supplied numbers to be a
**deterministic synthetic reward table**; it does not infer a true mean from noisy data.
This import check does not mix the legacy dataset into the confirmation study.

In an actual selected-feedback application, use `ContextPolicy.select(context, ids)`
and then `observe(chosen, reward)`; no full rewards dictionary or oracle is required.
The example below uses the simulator to provide that one selected reward.
''')
code('''
original_records=json.loads((ROOT/'fx_trading_dataset.json').read_text())
original_world=from_records(original_records,rewards_are_means=True)
print('Original JSON accepted:',len(original_records),'records;',original_world.x.shape[1],'features')
online=ContextPolicy('ucb',CONFIGS['ucb'],seed=100)
context=records[0]['context']
chosen,selection_probability=online.select(context,[int(k) for k in records[0]['rewards']])
online.observe(chosen,records[0]['rewards'][str(chosen)])
print('Chosen:',chosen,'observed reward:',records[0]['rewards'][str(chosen)])
''')
md('''
## 4. One world, paired methods

Each method sees the same generated world. Method randomness has a separate named
stream. Only the evaluator can inspect counterfactuals. Expected reward removes noise
for scoring; observed selected reward is also retained. The two panels distinguish them.
''')
code('''
custom_shifts={SCENARIO:SHIFT_FRACTIONS} if SHIFT_FRACTIONS is not None else None
single=run_experiment([SCENARIO],SELECTED_METHODS,CONFIGS,FIXED_ARMS,repeats=1,
    horizon=HORIZON,namespace=EXPLORATION_SEED,shift_fractions=custom_shifts,start_date=START_DATE)
assert single.worlds[0,SCENARIO].digest()==world.digest()
display(single.summary[['method','expected_mean','realized_mean','regret']].round(5))
trace=single.events[0,SCENARIO,SELECTED_METHODS[0]]
display(pd.DataFrame({'date':dates,'chosen_option':trace['actions'],'reward':trace['rewards'],
    'expected_reward_evaluator_only':trace['expected'],'expected_regret':trace['regret']}).head())
''')
code('''
ROLLING_WINDOW=250
fig=plot_rewards(single,SCENARIO,window=ROLLING_WINDOW,world_id=0)
# Edit returned axes without learning again, e.g. fig.axes[0].set_ylim(0.2,0.8)
plt.show()
fig=plot_regret(single,SCENARIO,window=ROLLING_WINDOW,world_id=0)
plt.show()
fig=plot_choices(single,SCENARIO)
plt.show()
''')
md('''
Dashed lines mark regime shifts; dotted lines mark the fifth option's arrival.
Smoothing is trailing, so earlier dates never incorporate future rewards. A reward
change may reflect the context mix; local expected regret measures the gap to the best
eligible option for the actual context. The choice map spreads within-day decisions
uniformly across each day for display; only the date is recorded in the raw schema.
''')
md('''
## 5. Monte Carlo: a new dataset every repetition

World IDs regenerate categories, volatility, sizes, reward coefficients and noise.
This tests deterministic UCB across different datasets too. Methods within a world share
the data for paired comparisons. Abrupt low/high noise intentionally share latent data.
The default 4 × 3 × 8 = 96 runs can take several minutes; increase deliberately.
Each dataset is saved as readable JSON **before** policy evaluation, even if a later
policy run fails. Curves remain in memory until the save step. The terminal runner
streams large studies to disk.
''')
code('''
MC_REPEATS=4
MC_SCENARIOS=['stationary','recurring','birth_contextual']
MC_SEED=20261106
MC_CUSTOM_SHIFTS={}
mc=run_experiment(MC_SCENARIOS,SELECTED_METHODS,CONFIGS,FIXED_ARMS,
    repeats=MC_REPEATS,horizon=HORIZON,namespace=MC_SEED,
    shift_fractions=MC_CUSTOM_SHIFTS,start_date=START_DATE)
for s in MC_SCENARIOS:
    assert len({mc.worlds[w,s].digest() for w in range(MC_REPEATS)})==MC_REPEATS
display(summary_table(mc).round(5))
print('Seconds:',round(mc.metadata['seconds'],2))
display(pd.DataFrame(mc.metadata['datasets'])[['world','scenario','path']])
''')
md('''
## 6. Monte Carlo timelines and paired effects

Shading shows pointwise 95% t intervals across independent worlds, not individual
trades, simultaneous bands or predictive intervals. Four repetitions are illustrative;
the saved studies uses a separate precision pilot and adjusted primary intervals.
''')
code('''
MC_PLOT_SCENARIO='recurring' if 'recurring' in mc.metadata['scenarios'] else mc.metadata['scenarios'][0]
MC_ROLLING_WINDOW=250
fig=plot_rewards(mc,MC_PLOT_SCENARIO,window=MC_ROLLING_WINDOW)
plt.show()
fig=plot_regret(mc,MC_PLOT_SCENARIO,window=MC_ROLLING_WINDOW)
plt.show()
if mc.worlds[0,MC_PLOT_SCENARIO].shifts:
    shift_ids=mc.worlds[0,MC_PLOT_SCENARIO].shifts
    before=min(500,min(shift_ids))
    after=min(1000,len(mc.worlds[0,MC_PLOT_SCENARIO].x)-1-max(shift_ids))
    fig=plot_shift_response(mc,MC_PLOT_SCENARIO,before=before,after=after,window=100)
    plt.show()
''')
code('''
COMPARATOR='ucb' if 'ucb' in mc.metadata['methods'] else mc.metadata['methods'][0]
COMPARISON_METHOD=next((m for m in ['discount_ucb',*mc.metadata['methods']] if m in mc.metadata['methods'] and m!=COMPARATOR),COMPARATOR)
display(paired_table(mc,COMPARATOR).round(5))
pairs=mc.summary[mc.summary.scenario==MC_PLOT_SCENARIO].pivot(
    index='world',columns='method',values='expected_mean')
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
ax.hist(pairs[COMPARISON_METHOD]-pairs[COMPARATOR],bins='auto',edgecolor='white')
ax.axvline(0,color='black',ls='--')
ax.set(xlabel=f'Expected reward: {COMPARISON_METHOD} minus {COMPARATOR}',
       ylabel='Independent worlds',title='Actual paired Monte Carlo effects')
plt.show()
''')
md('''
## 7. Fifth-option discovery and appropriate use

Every scenario has one arrival. Discovery delay zero means choosing it immediately;
undiscovered cases are right-censored at 100 opportunities. Discovery alone is not
success: an inferior new option should be rejected. Appropriate agreement compares
choosing/rejecting the fifth option with whether it is the expected-best option.
''')
code('''
BIRTH_SCENARIO='birth_contextual' if 'birth_contextual' in mc.metadata['scenarios'] else mc.metadata['scenarios'][0]
fig=plot_choices(mc,BIRTH_SCENARIO,world_id=0)
plt.show()
cold=pd.DataFrame([{'world':r.world,'method':r.method,**birth}
    for _,r in mc.summary[mc.summary.scenario==BIRTH_SCENARIO].iterrows() for birth in r['cold']])
display(cold.groupby('method',sort=False)[['delay','discovered','appropriate_fraction','regret']].mean().round(4))
display(cold.head(10))
''')
md('''
## 8. Save, reload and export figures

Unique exploratory directories contain the original-shaped JSON.gz records, expected
mean sidecars, traces, settings, seeds and file hashes. Each experiment gets a new output directory. Loading verifies hashes and reconciles raw contexts with encoded tensors.
''')
code('''
SAVE_RESULTS=True
if SAVE_RESULTS:
    saved_directory=save_experiment(mc)
    restored=load_experiment(saved_directory)
    pd.testing.assert_frame_equal(summary_table(mc),summary_table(restored))
    assert restored.worlds[0,MC_SCENARIOS[0]].contexts==mc.worlds[0,MC_SCENARIOS[0]].contexts
    print('Saved and verified:',saved_directory.relative_to(ROOT))
    fig=plot_rewards(restored,MC_PLOT_SCENARIO,window=MC_ROLLING_WINDOW)
    fig.savefig(saved_directory/'reward-timeline.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
# Later: mc=load_experiment(ROOT/'runs/<directory>')
''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (project .venv-notebook)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11.13'}})
nb.validate(n)
nb.write(n,ROOT/'bandit_main.ipynb')
print('Built',len(cells),'cells')
