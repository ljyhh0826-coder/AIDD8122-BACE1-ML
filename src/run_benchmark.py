"""Experiment 1-2: molecular representation x ML model benchmark under scaffold and random splits.

Usage: python -m src.run_benchmark [--workers 6]
"""
import argparse
import time

from joblib import Parallel, delayed

from .data import RESULTS, SEEDS, load_dataset
from .experiment import PRED_DIR, append_csv, done_keys, run_classical
from .features import FEATURE_SETS, get_features
from .models import MODELS

METRICS_CSV = RESULTS / "benchmark_metrics.csv"
PREDS_CSV = PRED_DIR / "benchmark_predictions.csv"
# Random-split runs use a subset of representations to quantify the scaffold-split generalisation gap.
RANDOM_SPLIT_FEATURES = ["ECFP4", "RDKit2D", "ECFP4+RDKit2D"]


def jobs():
    for task in ("cls", "reg"):
        for seed in SEEDS:
            for feature in FEATURE_SETS:
                for model in MODELS:
                    yield ("scaffold", seed, task, feature, model)
            for feature in RANDOM_SPLIT_FEATURES:
                for model in MODELS:
                    yield ("random", seed, task, feature, model)


def main(workers):
    import pandas as pd

    df = load_dataset()
    for f in FEATURE_SETS:  # compute and cache features once in the parent process
        get_features(f, df["smiles"].tolist())
    keys = ["split", "seed", "task", "feature", "model"]
    finished = done_keys(METRICS_CSV, keys)
    todo = [j for j in jobs() if tuple(map(str, j)) not in finished]
    print(f"{len(todo)} runs to do ({len(finished)} already finished)")
    t0 = time.time()
    chunk = workers * 4
    for start in range(0, len(todo), chunk):
        batch = todo[start:start + chunk]
        out = Parallel(n_jobs=workers)(
            delayed(run_classical)(feature, model, task, split, seed, n_jobs=2)
            for split, seed, task, feature, model in batch)
        append_csv(pd.DataFrame([r for r, _ in out]), METRICS_CSV)
        append_csv(pd.concat([p for _, p in out]), PREDS_CSV)
        print(f"{start + len(batch)}/{len(todo)} done, {time.time() - t0:.0f}s elapsed", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    main(ap.parse_args().workers)
