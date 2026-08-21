"""M4 hyperparameter search over the BEV CNN.

A **staged coordinate search**, not a full grid: four stages of three or four
runs, each stage sweeping one axis and passing its winner into the next. A
3-axis cross product would be 27+ runs, and with only 213 validation tiles a
difference of 2 percentage points is four tiles -- resolving a cross product to
that precision would mostly be fitting the noise in the val split. The notebook
says this out loud rather than implying an exhaustive search happened.

**Why this runs on CPU while the machine has an RTX 4060.** Timed on 120 real
tiles, one training pass costs:

    threads   1 -> 0.51 s      threads   8 -> 0.29 s
    threads   2 -> 0.28 s      threads  14 -> 0.25 s
    threads   4 -> 0.24 s

Scaling is flat past 4 threads. That is the signature of a loop bound by
per-operation launch latency, not by arithmetic: at `batch_size=1` (forced by
variable grid sizes -- see src/cnn.py) each convolution is a few microseconds of
real work wrapped in fixed overhead, and extra cores cannot shrink overhead. A
GPU hits the same ceiling -- it would idle between kernel launches -- because
its advantage is throughput on large batched tensors, which is exactly what this
dataset cannot supply. So the machine is used the way it actually pays off here:
five runs at a time, four threads each, filling all 20 cores with independent
work instead of starving 14 threads on one tiny model.
"""

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from src.data import CLASSES, get_splits
from src.train import train

WORKERS = 5  # x THREADS_PER_RUN = 20 = core count
THREADS_PER_RUN = 4  # measured saturation point, see module docstring
SEED = 42

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "m4_tuning.json"


def balanced_subsample(samples: list, seed: int = SEED) -> list:
    """Downsample every class to the size of the rarest one.

    The other lever on the same imbalance is the class-weighted loss, which is
    already on (see src/train.py). Weighting keeps every tile and changes how
    much each one counts; subsampling throws tiles away so the counts are equal
    to begin with. Stage D measures which of the two -- or both -- actually
    helps, rather than assuming.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[int, list] = {}
    for sample in samples:
        by_label.setdefault(sample[2], []).append(sample)
    n = min(len(group) for group in by_label.values())
    return [group[i] for group in by_label.values() for i in rng.choice(len(group), n, replace=False)]


def _run_one(config: dict) -> dict:
    """Train one configuration and return its scores. Runs in its own process."""
    torch.set_num_threads(THREADS_PER_RUN)

    kwargs = dict(config)
    label = kwargs.pop("label")
    if kwargs.pop("balanced", False):
        splits = get_splits()
        kwargs["splits"] = {**splits, "train": balanced_subsample(splits["train"])}

    started = time.time()
    result = train(verbose=False, **kwargs)
    best = result["history"][result["best_epoch"] - 1]

    return {
        "label": label,
        "config": {k: v for k, v in config.items() if k != "label"},
        "best_epoch": result["best_epoch"],
        "val_acc": result["report"]["accuracy"],
        "weighted_iou": result["report"]["weighted"]["iou"],
        "val_loss": best["val_loss"],
        # Kept because M4's whole premise is the train/val gap, not the val
        # number alone: a config that raises val accuracy while widening the
        # gap has not fixed the overfitting, it has got luckier.
        "train_acc": best["train_acc"],
        "seconds": round(time.time() - started, 1),
    }


def run_stage(name: str, configs: list[dict]) -> dict:
    """Run every config in a stage in parallel; return the best by weighted IoU.

    Each run is a pure function of its config with no shared state, so this
    needs no queue and no lock -- `ProcessPoolExecutor.map` is the whole
    scheduler. Grids are rebuilt per process rather than shared: that costs
    ~1.3 s against a ~3 min run, and buying that 1% back would mean shared
    state between workers.

    "spawn" rather than the Linux default "fork": forking a process that has
    already initialised torch's OpenMP thread pool is a known way to hang.
    """
    print(f"\n=== stage {name} ({len(configs)} runs, {WORKERS} at a time) ===", flush=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        records = list(pool.map(_run_one, configs))

    for r in sorted(records, key=lambda r: -r["weighted_iou"]):
        print(
            f"  {r['label']:<28} iou={r['weighted_iou']:.3f}  acc={r['val_acc']:.3f}  "
            f"train_acc={r['train_acc']:.3f}  gap={r['train_acc'] - r['val_acc']:+.3f}  "
            f"epoch={r['best_epoch']:>2}  {r['seconds']:.0f}s",
            flush=True,
        )

    _append_results(name, records)
    return max(records, key=lambda r: r["weighted_iou"])


def _append_results(stage: str, records: list[dict]) -> None:
    """Write a stage's records to the results file, replacing any earlier run of the same stage.

    Written incrementally so a crash in stage D does not cost stages A-C, and
    so the notebook can plot the entire search -- losing configs included --
    without re-running anything.

    Re-running a stage (e.g. re-executing a `repeat()` cell in a later
    session) drops that stage's previous records before writing the new
    ones, rather than appending next to them. Without this, `_last_stage`
    would silently mix an old run's records into a new one's mean/std --
    same class of bug as the ones this project has otherwise been careful
    to test for, just in the results file instead of the training loop.
    """
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    all_records = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    all_records = [r for r in all_records if r["stage"] != stage]
    all_records.extend({"stage": stage, **r} for r in records)
    RESULTS_PATH.write_text(json.dumps(all_records, indent=2))


def repeat(configs: dict[str, dict], seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict:
    """Re-run each named config across several seeds, to size the noise floor.

    The coordinate search above compares single runs, and single runs here are
    not reliable to better than a few points of IoU. Two independent reasons:

    1. Each run has its own random weight init and shuffling order.
    2. Even holding the seed fixed, results move with the *thread count* --
       measured, the M3 config scored 0.638 weighted IoU on 14 threads and
       0.587 on 4. Nothing is wrong: parallel reductions in convolution sum
       partial results in whatever order the threads finish, floating-point
       addition is not associative, and over 40 epochs those last-bit
       differences compound into a visibly different trajectory.

    A stage difference smaller than this spread is not a finding. Running each
    candidate over several seeds and reporting mean and spread is what makes
    the difference between "0.5 m grids are better" and "0.5 m won one race".
    """
    summary = {}
    for name, config in configs.items():
        runs = [{**config, "seed": seed, "label": f"{name} seed={seed}"} for seed in seeds]
        run_stage(f"repeat-{name}", runs)  # prints the per-seed table as it goes
        ious = np.array([r["weighted_iou"] for r in _last_stage(f"repeat-{name}")])
        # ddof=1 (sample std, N-1 denominator): matches pandas' default .std(),
        # which is what notebooks/05_optimization.ipynb uses to report these
        # spreads. numpy's default is ddof=0 (population std) and would print
        # a smaller, non-matching number here.
        std = float(ious.std(ddof=1))
        summary[name] = {"mean": float(ious.mean()), "std": std, "n": len(ious)}
        print(f"  == {name}: iou {ious.mean():.3f} +/- {std:.3f} over {len(ious)} seeds")
    return summary


def _last_stage(stage: str) -> list[dict]:
    """Records written for `stage`, read back from the results file."""
    return [r for r in json.loads(RESULTS_PATH.read_text()) if r["stage"] == stage]


def main() -> dict:
    """Walk the four stages, carrying each winner into the next.

    Every stage includes the current best as a control run. Without it a stage
    cannot distinguish "this value is an improvement" from "every run in this
    stage happened to land well", since each run has its own random init.
    """
    best: dict = {}  # accumulated winning kwargs

    def label_of(**kwargs) -> str:
        return ", ".join(f"{k}={v}" for k, v in kwargs.items()) or "control (carried forward)"

    def stage(name: str, axis_values: list[dict]) -> None:
        nonlocal best
        configs = [{**best, **v, "label": label_of(**v)} for v in axis_values]
        winner = run_stage(name, configs)
        best = dict(winner["config"])
        print(f"  -> stage {name} winner: {winner['label']}", flush=True)

    stage("A-resolution", [{"resolution": r} for r in (0.2, 0.3, 0.5)])
    stage("B-lr", [{"lr": lr} for lr in (3e-4, 1e-3, 3e-3)])
    stage("C-capacity", [{"channels": c} for c in ((16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 256))])
    stage(
        "D-data",
        [
            {},  # control: winner of A-C, unchanged
            {"balanced": True},
            {"augment": True},
            {"weight_decay": 1e-4},
            {"augment": True, "weight_decay": 1e-4},
        ],
    )
    return best


if __name__ == "__main__":
    best = main()
    print(f"\nbest config: {best}")
    print(f"all runs written to {RESULTS_PATH}")
