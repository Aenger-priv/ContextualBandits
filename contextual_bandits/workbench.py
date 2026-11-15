"""Data persistence, paired experiments and editable notebook charts.

Learners receive chosen rewards; evaluator-only arrays support synthetic scoring.
"""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from scipy.stats import t as student

from contextual_bandits.evaluate import run
from contextual_bandits.policies import METHODS
from contextual_bandits.simulator import SCENARIOS, World, generate, from_records

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
DEFAULT_CONFIG = ROOT / "configs/defaults.json"
if not DEFAULT_CONFIG.exists():
    DEFAULT_CONFIG = Path(__file__).with_name("defaults.json")
LABELS = {
    "uniform": "Uniform random", "fixed": "Development fixed",
    "context_free": "Context-free epsilon", "greedy": "Contextual greedy",
    "epsilon": "Contextual epsilon", "ucb": "LinUCB", "sw_ucb": "Window UCB",
    "discount_ucb": "Discount UCB", "ts": "Linear TS", "softmax": "Softmax",
    "sw_softmax": "Window Softmax", "dr_softmax": "DR Softmax",
    "sw_dr_softmax": "Window DR Softmax",
    "full_information": "Full-information diagnostic",
}
COLORS = dict(zip(METHODS, ("#b5b5b5", "#4d4d4d", "#a6761d", "#66a61e",
    "#8c564b", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#e377c2",
    "#17becf", "#d62728", "#bcbd22", "#393b79")))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_settings():
    """Return editable copies of the default algorithm configurations."""
    return json.loads((DEFAULT_CONFIG).read_text())


def dataset_path(world, parent=None):
    """Stable location for the exact generated world, including raw context dates."""
    parent = Path(parent) if parent is not None else BASE / "datasets"
    identity = hashlib.sha256(world.digest().encode() +
        json.dumps(world.contexts, sort_keys=True).encode()).hexdigest()
    return parent / identity / "fx_trading_dataset.json"


def save_dataset(world, parent=None, generation=None):
    """Persist readable original-shaped JSON before learning; preserve existing files.

    Identical worlds reuse their content-addressed directory. Oracle means and
    generation details stay in separate files, never in the context/rewards records.
    """
    path = dataset_path(world, parent)
    protected = (BASE / "results").resolve()
    if path.resolve().is_relative_to(protected):
        raise ValueError("dataset exports cannot target frozen results")
    path.parent.mkdir(parents=True, exist_ok=True)
    records = json.dumps(world.records(), indent=2, allow_nan=False) + "\n"
    means = json.dumps(world.mu.tolist(), indent=2, allow_nan=False) + "\n"
    # Exclusive creation avoids silently overwriting a dataset edited for analysis.
    for target, content in ((path, records), (path.with_name("expected_rewards.json"), means)):
        try:
            with target.open("x", encoding="utf-8") as f:
                f.write(content)
        except FileExistsError:
            if target.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"Existing dataset file differs; use another dataset directory: {target}")
    manifest = path.with_name("metadata.json")
    if not manifest.exists():
        metadata = dict(format_version=1, generation=generation, records=len(world.contexts),
            environment_sha256=world.digest(), shifts=list(world.shifts), births=list(world.births),
            reward_semantics="Noisy synthetic potential rewards; only the chosen reward is learner feedback.",
            oracle_semantics="expected_rewards.json contains evaluator-only conditional means in row/option order.",
            files={p.name: sha(p) for p in (path, path.with_name("expected_rewards.json"))})
        with manifest.open("x", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, allow_nan=False)
            f.write("\n")
    return path


def make_world(scenario, namespace, world_id, horizon=18250, shift_fractions=None,
               start_date="2023-01-01", *, dataset_parent=None):
    """Generate a world and automatically save its plain JSON dataset in the project."""
    changes=None if shift_fractions is None else [int(f*365) for f in shift_fractions]
    if changes is not None and scenario not in ('abrupt_low','abrupt_high','recurring'):
        raise ValueError('custom shifts require an abrupt or recurring scenario')
    world = generate(scenario,namespace,world_id,horizon,regime_change_days=changes,
                     start_date=start_date)
    save_dataset(world, dataset_parent, dict(scenario=scenario, namespace=namespace,
        world_id=world_id, horizon=horizon, regime_change_days=changes, start_date=start_date))
    return world


def calendar(horizon, start="2023-01-01", end="2023-12-31"):
    dates=pd.date_range(start,end,freq='D')
    if horizon%len(dates):raise ValueError('horizon must be divisible by calendar days')
    return dates.repeat(horizon//len(dates))


def world_dates(world, start=None, end=None):
    """Use the dates in the raw input, never a cosmetic remapping of time."""
    dates = pd.DatetimeIndex([c['date'] for c in world.contexts])
    if len(dates) != len(world.x):
        raise ValueError('raw contexts are required for date plots')
    if start is not None and pd.Timestamp(start) != dates[0]:
        raise ValueError('start must match the raw dataset date')
    if end is not None and pd.Timestamp(end) != dates[-1]:
        raise ValueError('end must match the raw dataset date')
    return dates


def event_table(world, dates):
    rows = [dict(event="Regime shift", option=None, index=i, date=dates[i])
            for i in world.shifts]
    rows += [dict(event="Option arrival", option=a, index=i, date=dates[i])
             for i, a in world.births]
    return pd.DataFrame(rows, columns=["event", "option", "index", "date"])


@dataclass
class Experiment:
    worlds: dict
    events: dict
    summary: pd.DataFrame
    metadata: dict
    directory: Path | None = None


def run_experiment(scenarios, methods, configs, fixed_arms, repeats=1, horizon=18250,
                   namespace=65505, shift_fractions=None, progress=True,
                   start_date="2023-01-01", dataset_parent=None):
    """Fresh dataset per world/scenario; all policies share it and get chosen feedback.

    Saves each world's readable JSON before running policies; returns curves in memory.
    No scientific evidence paths are modified.
    shift_fractions is an optional dictionary mapping scenario -> fraction list.
    """
    scenarios, methods = list(scenarios), list(methods)
    if not scenarios or len(set(scenarios)) != len(scenarios) or not set(scenarios) <= set(SCENARIOS):
        raise ValueError("choose nonempty, unique supported scenarios")
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("choose nonempty, unique supported methods")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not isinstance(namespace, int) or namespace < 0:
        raise ValueError("namespace must be a nonnegative integer")
    config_copy = json.loads(json.dumps({m: configs[m] for m in methods}))
    shifts = shift_fractions or {}
    if not set(shifts) <= set(scenarios):
        raise ValueError("custom shift scenarios must be included in this experiment")
    worlds, events, rows, datasets = {}, {}, [], []
    start = time.perf_counter()
    for w in range(repeats):
        for scenario in scenarios:
            env = make_world(scenario, namespace, w, horizon, shifts.get(scenario), start_date,
                             dataset_parent=dataset_parent)
            path = dataset_path(env, dataset_parent)
            datasets.append(dict(world=w, scenario=scenario, path=str(path.resolve()), sha256=sha(path)))
            worlds[w, scenario] = env
            for method in methods:
                summary, trace = run(env, method, config_copy[method], namespace, w,
                                     scenario, fixed_arms[scenario])
                events[w, scenario, method] = trace
                rows.append(dict(world=w, scenario=scenario, method=method,
                                 environment_sha256=env.digest(), **summary))
        if progress:
            print(f"Block {w+1}/{repeats} | {len(rows)} policy runs | "
                  f"{time.perf_counter()-start:.1f} s", flush=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    sources = list((BASE / "contextual_bandits").glob("*.py")) + [Path(__file__)]
    metadata = dict(format_version=1, kind="exploratory_notebook", created_utc=
                    datetime.now(timezone.utc).isoformat(), git_commit=revision,
                    scenarios=scenarios, methods=methods, configs=config_copy, datasets=datasets,
                    fixed_arms={s: fixed_arms[s] for s in scenarios}, repeats=repeats,
                    horizon=horizon, namespace=namespace, shift_fractions=shifts, start_date=start_date,
                    seconds=time.perf_counter()-start, python=platform.python_version(),
                    platform=platform.platform(), default_config_sha256=sha(DEFAULT_CONFIG),
                    source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    thread_environment={k: os.environ.get(k) for k in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                    versions={p: importlib.metadata.version(p)
                              for p in ("numpy", "scipy", "pandas", "matplotlib")})
    return Experiment(worlds, events, pd.DataFrame(rows), metadata)


def save_experiment(experiment, parent=None, calendar_settings=None):
    """Save into a unique exploratory directory, including tensors and hashes."""
    parent = Path(parent) if parent is not None else BASE / "runs"
    # Even an accidental supplied path cannot target the frozen evidence tree.
    protected = (BASE / "results").resolve()
    if parent.resolve().is_relative_to(protected):
        raise ValueError("save exploratory results outside results")
    directory = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                          + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    for (w, scenario), env in experiment.worlds.items():
        with gzip.open(directory / f"world-{w:04d}-{scenario}.json.gz", 'wt') as f:
            json.dump(env.records(), f, separators=(',', ':'))
        np.savez_compressed(directory / f"world-{w:04d}-{scenario}.npz", x=env.x,
                            mu=env.mu, outcomes=env.outcomes, available=env.available,
                            births=np.asarray(env.births, dtype=int).reshape(-1, 2),
                            shifts=np.asarray(env.shifts, dtype=int))
    for (w, scenario, method), trace in experiment.events.items():
        np.savez_compressed(directory / f"events-{w:04d}-{scenario}-{method}.npz", **trace)
    records = experiment.summary.astype(object).where(pd.notna(experiment.summary), None)
    (directory / "summary.json").write_text(json.dumps(
        records.to_dict(orient="records"), indent=2, allow_nan=False) + "\n")
    metadata = dict(experiment.metadata, calendar_display=calendar_settings)
    metadata["files"] = {p.name: sha(p) for p in sorted(directory.iterdir())}
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    experiment.directory = directory
    return directory


def load_experiment(directory):
    """Reload saved curves and verify every recorded file before plotting."""
    directory = Path(directory)
    meta = json.loads((directory / "manifest.json").read_text())
    for name, digest in meta["files"].items():
        if sha(directory / name) != digest:
            raise ValueError(f"saved artifact hash mismatch: {name}")
    worlds, events = {}, {}
    for w in range(meta["repeats"]):
        for scenario in meta["scenarios"]:
            with np.load(directory / f"world-{w:04d}-{scenario}.npz") as e:
                with gzip.open(directory / f"world-{w:04d}-{scenario}.json.gz", 'rt') as f:
                    restored = from_records(json.load(f), e['mu'])
                if not (np.array_equal(restored.x,e['x']) and
                        np.array_equal(restored.outcomes,e['outcomes']) and
                        np.array_equal(restored.available,e['available'])):
                    raise ValueError('raw records and sidecar disagree')
                worlds[w, scenario] = replace(restored, shifts=tuple(e['shifts'].tolist()))
            for method in meta["methods"]:
                with np.load(directory / f"events-{w:04d}-{scenario}-{method}.npz") as e:
                    events[w, scenario, method] = {k: e[k] for k in e.files}
    rows = json.loads((directory / "summary.json").read_text())
    return Experiment(worlds, events, pd.DataFrame(rows), meta, directory)


def mean_interval(values):
    """Pointwise 95% t interval across independent worlds, not across decisions."""
    a = np.asarray(values, dtype=float)
    mean = a.mean(axis=0)
    half = (student.ppf(.975, len(a)-1) * a.std(axis=0, ddof=1) / np.sqrt(len(a))
            if len(a) > 1 else np.full_like(mean, np.nan))
    return mean, mean-half, mean+half


def summary_table(experiment):
    rows = []
    for (scenario, method), group in experiment.summary.groupby(["scenario", "method"], sort=False):
        mean, lo, hi = mean_interval(group.expected_mean.to_numpy())
        rows.append(dict(scenario=scenario, method=method, worlds=len(group),
                         expected_reward=float(mean), lower=float(lo), upper=float(hi),
                         realized_reward=group.realized_mean.mean(), regret=group.regret.mean()))
    return pd.DataFrame(rows)


def paired_table(experiment, comparator="ucb"):
    """Compute differences within the same world/scenario before uncertainty."""
    pivot = experiment.summary.pivot(index=["world", "scenario"], columns="method", values="expected_mean")
    if comparator not in pivot:
        raise ValueError("comparator must have been run")
    rows = []
    for scenario in experiment.metadata["scenarios"]:
        block = pivot.xs(scenario, level="scenario")
        for method in experiment.metadata["methods"]:
            if method == comparator:
                continue
            mean, lo, hi = mean_interval((block[method]-block[comparator]).to_numpy())
            rows.append(dict(scenario=scenario, method=method, comparator=comparator,
                             worlds=len(block), difference=float(mean), lower=float(lo), upper=float(hi)))
    return pd.DataFrame(rows)


def trailing(values, window):
    if not isinstance(window, int) or window < 1:
        raise ValueError("rolling window must be a positive integer")
    return pd.DataFrame(np.atleast_2d(values).T).rolling(window, min_periods=1).mean().to_numpy().T


def mark_events(ax, world, dates):
    for j, i in enumerate(world.shifts):
        ax.axvline(dates[i], color="#293844", ls="--", lw=1, alpha=.75)
        ax.text(dates[i], .98, f" Shift {j+1}", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=.5))
    for i, arm in world.births:
        ax.axvline(dates[i], color="#8e44ad", ls=":", lw=1.3)
        ax.text(dates[i], .86, f" Option {arm} arrives", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color="#73318c")


def date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel("Dataset date (50 opportunities per day at the default horizon)")
    ax.grid(alpha=.18)


def figure_legend(fig, ax, columns=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=columns,
               fontsize=8, frameon=False)


def plot_world(world, start=None, end=None, window=250):
    dates = world_dates(world, start, end)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for arm in range(world.mu.shape[1]):
        values = np.where(world.available[:, arm], world.mu[:, arm], np.nan)
        axes[0].plot(dates, trailing(values, window)[0], label=f"Option {arm}")
    axes[0].set_ylabel("Expected reward")
    axes[0].set_title(f"Evaluator-only option means · trailing {window} opportunities")
    figure_legend(fig, axes[0], min(5, world.mu.shape[1]))
    axes[1].plot(dates, world.x[:, -2], label="volatility / 0.5", alpha=.7, lw=.8)
    axes[1].plot(dates, world.x[:, -1], label="size / 2000", alpha=.7, lw=.8)
    axes[1].set_ylabel("Scaled numeric context"); axes[1].legend(ncol=2)
    axes[2].step(dates, world.available.sum(axis=1), where="post", color="#24465c")
    axes[2].set_ylabel("Available options")
    for ax in axes:
        mark_events(ax, world, dates)
    date_axis(axes[-1])
    axes[-1].set_xlim(dates[0], dates[-1])
    return fig


def plot_rewards(experiment, scenario, window=250, start=None, end=None,
                 world_id=None, methods=None):
    """Single-world curves or MC means with pointwise world-level intervals."""
    methods = methods or experiment.metadata["methods"]
    ids = [world_id] if world_id is not None else list(range(experiment.metadata["repeats"]))
    env = experiment.worlds[ids[0], scenario]
    dates = world_dates(env, start, end)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for ax, metric, title in zip(axes, ("expected", "rewards"),
                                 ("Conditional expected reward", "Observed selected reward")):
        for method in methods:
            curves = trailing(np.stack([experiment.events[w, scenario, method][metric] for w in ids]), window)
            mean, lo, hi = mean_interval(curves)
            ax.plot(dates, mean, label=LABELS[method], color=COLORS[method], lw=1.5)
            if len(ids) > 1:
                ax.fill_between(dates, lo, hi, color=COLORS[method], alpha=.09, linewidth=0)
        oracle = np.stack([np.max(np.where(experiment.worlds[w, scenario].available,
                           experiment.worlds[w, scenario].mu, -np.inf), axis=1) for w in ids])
        ax.plot(dates, trailing(oracle, window).mean(axis=0), color="black", ls=":", lw=1.6,
                label="Expected oracle (evaluator only)")
        ax.set_ylabel("Reward per opportunity")
        ax.set_title(f"{title} · trailing {window} opportunities")
        mark_events(ax, env, dates)
    figure_legend(fig, axes[0])
    date_axis(axes[-1])
    axes[-1].set_xlim(dates[0], dates[-1])
    detail = (f"world {world_id}" if world_id is not None else
              f"{len(ids)} worlds; pointwise 95% intervals" if len(ids) > 1 else
              "1 world; uncertainty not estimated")
    fig.suptitle(f"{scenario} · {detail}", fontsize=13)
    return fig


def plot_regret(experiment, scenario, window=250, start=None, end=None, world_id=None):
    ids = [world_id] if world_id is not None else list(range(experiment.metadata["repeats"]))
    env = experiment.worlds[ids[0], scenario]
    dates = world_dates(env, start, end)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for method in experiment.metadata["methods"]:
        gaps = np.stack([experiment.events[w, scenario, method]["regret"] for w in ids])
        for ax, curves in zip(axes, (np.cumsum(gaps, axis=1), trailing(gaps, window))):
            mean, lo, hi = mean_interval(curves)
            ax.plot(dates, mean, label=LABELS[method], color=COLORS[method])
            if len(ids) > 1:
                ax.fill_between(dates, lo, hi, color=COLORS[method], alpha=.09, linewidth=0)
    axes[0].set_ylabel("Cumulative expected regret"); figure_legend(fig, axes[0])
    axes[1].set_ylabel("Expected regret / opportunity")
    axes[1].set_title(f"Local performance gap · trailing {window} opportunities · lower is better")
    for ax in axes:
        mark_events(ax, env, dates)
    date_axis(axes[-1]); axes[-1].set_xlim(dates[0], dates[-1]); fig.suptitle(scenario)
    return fig


def plot_choices(experiment, scenario, world_id=0, start=None, end=None):
    env = experiment.worlds[world_id, scenario]
    dates = world_dates(env, start, end)
    methods = experiment.metadata["methods"]
    choices = [experiment.events[world_id, scenario, m]["actions"] for m in methods]
    choices.append(np.argmax(np.where(env.available, env.mu, -np.inf), axis=1))
    k = env.mu.shape[1]
    cmap = ListedColormap(plt.get_cmap("tab10")(np.arange(k)))
    norm = BoundaryNorm(np.arange(k+1)-.5, k)
    fig, ax = plt.subplots(figsize=(12, max(4, .45*len(choices))), layout="constrained")
    daily = pd.Series(dates)
    within_day = daily.groupby(daily).cumcount().to_numpy()
    counts = daily.groupby(daily).transform('size').to_numpy()
    edges = np.r_[mdates.date2num(dates)+within_day/counts,
                  mdates.date2num(dates[-1]+pd.Timedelta(days=1))]
    mesh = ax.pcolormesh(edges, np.arange(len(choices)+1)-.5, np.asarray(choices), cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(choices)), [LABELS[m] for m in methods]+["Expected oracle"])
    ax.invert_yaxis(); ax.xaxis_date(); date_axis(ax)
    ax.set_xlim(dates[0], dates[-1]+pd.Timedelta(days=1))
    for i in env.shifts:
        ax.axvline(dates[i], color="black", ls="--", lw=1)
    for i, _ in env.births:
        ax.axvline(dates[i], color="black", ls=":", lw=1.2)
    fig.colorbar(mesh, ax=ax, ticks=np.arange(k), label="Chosen option index", shrink=.8)
    ax.set_title(f"{scenario} · world {world_id} · chosen options and evaluator-only oracle")
    return fig


def plot_shift_response(experiment, scenario, before=80, after=160, window=15):
    """Average events within each world first, then uncertainty across worlds."""
    n = experiment.metadata["repeats"]
    env = experiment.worlds[0, scenario]
    shifts = [s for s in env.shifts if s >= before and s+after < len(env.x)]
    if not shifts:
        raise ValueError("no shifts with enough surrounding observations; shorten before/after")
    offsets = np.arange(-before, after+1)
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    for method in experiment.metadata["methods"]:
        curves = np.stack([np.mean([experiment.events[w, scenario, method]["regret"][s+offsets]
                                   for s in shifts], axis=0) for w in range(n)])
        mean, lo, hi = mean_interval(trailing(curves, window))
        ax.plot(offsets, mean, label=LABELS[method], color=COLORS[method])
        if n > 1:
            ax.fill_between(offsets, lo, hi, color=COLORS[method], alpha=.10, linewidth=0)
    ax.axvline(0, color="black", ls="--", label="Regime shift")
    ax.set(xlabel="Opportunities relative to regime shift", ylabel="Expected regret / opportunity",
           title=f"{scenario} · {len(shifts)} shifts averaged within each of {n} worlds · trailing {window}")
    figure_legend(fig, ax); ax.grid(alpha=.18)
    return fig
