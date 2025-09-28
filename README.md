Contextual Bandits for FX Strategy Selection

Overview

- This project explores whether contextual bandit algorithms can select trading strategies effectively in a non-stationary FX environment.
- It uses a synthetic dataset that encodes market contexts (currency pair, volatility, trade size, time of day, date) with non-stationarity, regime shifts, and a cold-start introduction of a new strategy mid-year.
- Multiple bandit algorithms are implemented and evaluated on identical streams of contexts and rewards, with plots and statistical tests to compare performance.

Repository Structure

- `data_generator.py` — Synthetic FX context + reward generator (non-stationary regimes, imbalanced contexts, new-arm cold start). Also includes helper plots and CSV/JSON export.
- `bandit_base.py` — Base classes: `ContextualBandit` and `ContextProcessor` (one-hot encodes categorical context features using scikit-learn).
- `bandit_algorithms.py` — Implementations:
  - `LinUCB`, `SlidingWindowLinUCB`, `LinUCBDecay`
  - `LinTS` (Linear Thompson Sampling)
  - `EpsilonGreedy`
  - `SlidingDoublyRobustSoftmax` (softmax policy + ridge models + doubly robust updates with a sliding window)
  - `RidgeSoftmax` (plain softmax + ridge models)
- `bandit_experiment.py` — Orchestrates experiments over the dataset, tracks rewards/regrets, computes summaries, plots, and statistical tests (ANOVA, Tukey’s HSD). Handles a late-introduced arm (cold start) and compares to optimal/worst baselines.
- `bandit_main.ipynb` — End‑to‑end single‑run experiment notebook: load data, add algorithms, run, and visualize.
- `bandit_mc.ipynb` — Monte Carlo runs over randomized hyperparameters with summary stats and significance testing.
- `fx_data_generation.ipynb` — Notebook to generate the dataset and volatility series with visualizations.
- `fx_trading_dataset.json` — Pre-generated dataset (~5.0 MB) for immediate use.
- `volatility_series.csv` — Volatility time series used when generating data.
- `requirements.txt` — Tested package versions.

Core Concepts

- Context: `{currency_pair, volatility, size, time_of_day, date}` preprocessed to a numeric vector via one-hot encoding of categoricals and scaled numerics.
- Reward: Expected reward per arm/strategy is a deterministic function of context (pair effectiveness, volatility preference, size sensitivity, time-of-day effects), with non-stationarity via regime changes and a new adaptive strategy introduced around mid‑year (day 180).
- Algorithms: Linear contextual policies (UCB, TS), epsilon‑greedy with simple per‑arm linear models, and softmax policies with ridge regression; non‑stationarity handled via sliding windows or exponential decay.

Quick Start

1) Environment

- Python 3.11+ (notebooks were tested on 3.12 as well). The project declares `requires-python = ">=3.11"` in `pyproject.toml`.

- Using uv (recommended):

  - `uv venv`  (creates `.venv` with your current Python; use `uv venv 3.11` to target 3.11 explicitly)
  - `source .venv/bin/activate` (macOS/Linux) or `.\\.venv\\Scripts\\activate` (Windows)
  - `uv sync`  (installs dependencies; dev tools like Black/Ruff/MyPy are included by default)

  Common tools:
  - `uv run black .`
  - `uv run ruff check .`
  - `uv run mypy`
  - Without installing globally: `uvx black .`, `uvx ruff check .`, `uvx mypy`

- Alternative (pip):

 - Create and activate a virtual environment, then install dependencies:

  - `python -m venv .venv`
  - `source .venv/bin/activate` (Linux/Mac) or `.\.venv\\Scripts\\activate` (Windows)
  - `pip install -r requirements.txt`

2) Use the included dataset

- Open `bandit_main.ipynb` and run the cells to:
  - Load `fx_trading_dataset.json`.
  - Add algorithms and run the experiment.
  - Visualize cumulative reward/regret, selection frequencies, and distributions.

3) Generate a fresh dataset (optional)

- Open `fx_data_generation.ipynb` and run all cells, or use the API directly:

  - Example:
    - `from data_generator import FXDatasetGenerator`
    - `gen = FXDatasetGenerator(start_date="2023-01-01", end_date="2023-12-31", n_arms=5, new_arm_day=180, regime_change_days=[90,240])`
    - `dataset = gen.generate_dataset(contexts_per_day=50)`
    - `gen.save_dataset("fx_trading_dataset.json", dataset)`
    - `gen.save_volatility_series("volatility_series.csv")`

Programmatic Usage (no notebooks)

- Minimal example using the built‑in dataset and plotting utilities:

  - `import json`
  - `from bandit_experiment import BanditExperiment`
  - `from bandit_algorithms import LinUCB, SlidingWindowLinUCB, LinUCBDecay, LinTS, EpsilonGreedy, SlidingDoublyRobustSoftmax, RidgeSoftmax`
  - `with open("fx_trading_dataset.json") as f: data = json.load(f)`
  - `exp = BanditExperiment(data)`
  - `exp.add_algorithm(LinUCB, alpha=0.25)`
  - `exp.add_algorithm(SlidingWindowLinUCB, alpha=0.25, window_size=500)`
  - `exp.add_algorithm(LinUCBDecay, alpha=0.25, decay=0.99985)`
  - `exp.add_algorithm(LinTS, v=0.025)`
  - `exp.add_algorithm(EpsilonGreedy, epsilon=0.025)`
  - `exp.add_algorithm(SlidingDoublyRobustSoftmax, tau=0.025, window_size=500)`
  - `exp.add_algorithm(RidgeSoftmax, tau=0.025)`
  - `results = exp.run()`
  - `fig1 = exp.plot_cumulative_rewards(); fig1.show()`
  - `fig2 = exp.plot_cumulative_regrets(); fig2.show()`
  - `fig3 = exp.plot_arm_selection_frequencies(); fig3.show()`

What The Notebooks Do

- `bandit_main.ipynb`
  - Loads the dataset and runs a single‑pass evaluation across the algorithms above.
  - Produces: cumulative reward/regret, arm selection frequencies, optional arm‑over‑time plot, summary statistics, ANOVA + Tukey post‑hoc tests, reward distributions by currency pair, and optional cold‑start analysis.
- `bandit_mc.ipynb`
  - Runs multiple seeds/hyperparameter draws for a Monte Carlo comparison (logs show 10 runs by default). This is compute‑intensive; expect longer runtimes.
- `fx_data_generation.ipynb`
  - Generates a new dataset and plots volatility, strategy effectiveness, distributions, and a non‑stationarity view over time.

Key Implementation Details

- Context processing: `ContextProcessor` fits `OneHotEncoder` on currency pair and time‑of‑day; numerical features are concatenated, with trade size normalized.
- Non‑stationarity: dataset generator imposes regime changes on volatility paths, shifts strategy effectiveness over time, and introduces a new adaptive strategy midway through the year. The experiment class automatically calls `add_arm()` in each algorithm when the new strategy appears (around `2023-06-30`).
- Metrics tracked per algorithm: selected arms, per‑step reward, cumulative reward, cumulative regret, arm counts and frequencies.
- Statistics: ANOVA on reward sequences across algorithms; Tukey HSD if significant. Bootstrap CIs are also available (`bootstrap_confidence_intervals`).

Reproducibility and Performance Notes

- `FXDatasetGenerator` accepts a `random_seed` for repeatable datasets.
- The included dataset contains 18,250 context instances (365 days × 50 per day). Full runs with all algorithms and plots are moderate to heavy on CPU.
- Monte Carlo notebook (multiple runs × algorithms) can take a long time; reduce runs or algorithm set if needed.

Troubleshooting

- scikit‑learn 1.2+ is expected (uses `OneHotEncoder(sparse_output=False)`); the pinned `scikit-learn==1.6.1` in `requirements.txt` is compatible.
- If plots do not show in scripts, ensure an interactive backend or save figures via `fig.savefig(...)`.
- If you regenerate datasets, ensure `bandit_main.ipynb` loads the intended `fx_trading_dataset.json`.

Citation

- If you reference this work academically, please cite your accompanying dissertation or report. The codebase is intended for research and educational purposes and is not financial advice.
