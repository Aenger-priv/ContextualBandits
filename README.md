# Contextual Bandits

Compare contextual bandit algorithms on mixed categorical/numerical synthetic timelines with controlled regime changes and a four-to-five option lifecycle. Every Monte Carlo repetition generates a fresh dataset. Methods share each world for paired comparisons and learn only their selected reward.

## Start in Jupyter

```sh
uv sync
uv run jupyter lab bandit_main.ipynb
```

In VS Code, choose this project's `.venv/bin/python` kernel and run `bandit_main.ipynb` from top to bottom. It covers settings, readable JSON generation, context encoding, selected-feedback examples, single runs, Monte Carlo, editable date/reward/regret/choice plots, regime/arrival markers, and save/reload. `bandit_mc.ipynb` focuses on fresh-world Monte Carlo; `fx_data_generation.ipynb` generates and inspects datasets.

Without uv, use a working Python 3.11 or newer:

```sh
python3 scripts/setup_environment.py
source .venv-notebook/bin/activate
python -m jupyterlab bandit_main.ipynb
```

The setup helper resolves the real base Python before venv creation, including uv-managed installations.

## Generate data or run from the terminal

```sh
uv run python scripts/generate_dataset.py --scenario recurring --seed 1234 --world 0
uv run python scripts/run_experiment.py --scenarios recurring birth_contextual --methods ucb sw_ucb discount_ucb ts --repeats 4
uv run python -m unittest discover -s tests -v
```

Generated datasets are plain indented `datasets/<id>/fx_trading_dataset.json` arrays with the original `context`/`rewards` schema. Separate JSON files contain generation metadata and evaluator-only conditional means. Different worlds get different files; identical datasets reuse their directory without overwriting edits. Saved experiment bundles go under unique `runs/` directories. These outputs are local and Git-ignored.

## Full staged experiment

The staged runner separates development, independent validation, precision planning, configuration freeze and confirmation:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run python -m contextual_bandits.study development --output runs/study
uv run python -m contextual_bandits.study validation --output runs/study
uv run python -m contextual_bandits.study pilot --output runs/study
uv run python -m contextual_bandits.study freeze --output runs/study
uv run python -m contextual_bandits.study confirmation --output runs/study
uv run python -m contextual_bandits.summarize --study runs/study
```

Run sequentially and use the same output directory/horizon for every stage. Default worlds have 365 days × 50 trades per day. `--horizon 365` makes a small diagnostic stream. Four workers and single-thread numerical libraries are used. The pilot plans a simultaneous 0.005 interval half-width for seven paired contrasts, with 20% SD inflation and a minimum 32 worlds, subject to a two-hour stage budget and available disk reserve. It may require substantial local data storage. Inspect the precision plan before freezing. Commit scientific code/configuration changes before freeze. Confirmation checks source hashes and resumes only complete hash-verified world checkpoints. Start a new directory after changing settings.

The notebook defaults use `configs/defaults.json` as editable starting values; full staged development tunes configurations independently. The fixed comparator is a synthetic research control. Expected regret uses the best eligible conditional mean; realised reward is reported separately. The full-information greedy diagnostic receives extra feedback and is not an oracle or deployable comparator.

## Project layout

- `contextual_bandits/`: generator/encoding, policies, selected-feedback interface, evaluator, notebook charts/persistence and staged study.
- `configs/defaults.json`: starting policy settings.
- `scripts/`: data generation, terminal execution, environment setup and notebook builder.
- `tests/`: analytic ridge/DR checks, feedback isolation, fresh-world/schema/eligibility checks and notebook persistence.
- Root notebooks: current interactive workflows.
- The existing root `bandit_base.py`, `bandit_algorithms.py`, `bandit_experiment.py`, `data_generator.py` and example input files are retained for compatibility with earlier checkouts. Current notebooks use `contextual_bandits`. Install `uv sync --extra legacy` to use the older interfaces.

## Modelling limits

Five raw context fields are currency pair, volatility, size, time of day and date. Fixed categorical vocabularies and numerical scales avoid future-fitted preprocessing. Dates schedule events and are excluded from regression. Expected regret needs a known mean sidecar; a noisy reward table does not identify that oracle. Unknown categories are rejected. The simulator is a controlled synthetic benchmark; conclusions depend on its laws and are not calibrated trading savings.
