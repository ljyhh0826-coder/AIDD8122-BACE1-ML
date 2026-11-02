"""Experiment 3: random-search hyperparameter optimisation on the scaffold split.

For each model, the representation with the best mean *validation* score in the benchmark is used.
For every seed, N_CONFIGS random configurations are trained on the training set, the best one is
chosen on the validation set and only that one is evaluated on the test set.

Usage: python -m src.run_hpo [--workers 6]
"""
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import RESULTS, SEEDS
from .experiment import PRED_DIR, append_csv, run_classical
from .metrics import HIGHER_IS_BETTER, PRIMARY
from .models import MODELS
from .run_benchmark import METRICS_CSV as BENCH_CSV

N_CONFIGS = 20
TRIALS_CSV = RESULTS / "hpo_trials.csv"
METRICS_CSV = RESULTS / "hpo_metrics.csv"
PREDS_CSV = PRED_DIR / "hpo_predictions.csv"


def sample_params(model, task, rng):
    lu = lambda lo, hi: float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))  # log-uniform
    pick = lambda xs: xs[rng.randint(len(xs))]
    if model == "kNN":
        return {"k": pick([1, 3, 5, 7, 10, 15, 20])}
    if model == "LR":
        return {"C": lu(1e-3, 10)} if task == "cls" else {"alpha": lu(0.1, 300)}
    if model == "RF":
        return {"n_estimators": pick([300, 500, 1000]), "max_features": pick(["sqrt", "log2", 0.1, 0.2, 0.33, 0.5]),
                "min_samples_leaf": pick([1, 2, 3, 5])}
    if model == "SVM":
        return {"C": lu(0.1, 100), "epsilon": pick([0.05, 0.1, 0.2, 0.3]), "gamma": pick(["scale", lu(1e-4, 1e-1)])}
    if model == "XGBoost":
        return {"n_estimators": pick([200, 400, 600, 1000, 1500]), "learning_rate": lu(0.01, 0.2),
                "max_depth": pick([3, 4, 5, 6, 8, 10]), "subsample": pick([0.6, 0.7, 0.8, 0.9, 1.0]),
                "colsample_bytree": pick([0.2, 0.3, 0.5, 0.7, 0.9]), "min_child_weight": pick([1, 2, 5]),
                "reg_lambda": lu(0.1, 10)}
    if model == "MLP":
        return {"hidden_layer_sizes": pick([(256,), (512,), (512, 128), (1024, 256), (256, 128, 64)]),
                "alpha": lu(1e-5, 1e-1), "learning_rate_init": pick([1e-4, 3e-4, 1e-3, 3e-3])}
    raise ValueError(model)


def best_features():
    """Representation with the best mean validation score per (model, task) on the scaffold split."""
    bench = pd.read_csv(BENCH_CSV)
    bench = bench[bench["split"] == "scaffold"]
    out = {}
    for (model, task), g in bench.groupby(["model", "task"]):
        col = "val_" + PRIMARY[task]
        means = g.groupby("feature")[col].mean()
        out[(model, task)] = means.idxmax() if HIGHER_IS_BETTER[PRIMARY[task]] else means.idxmin()
    return out


def main(workers):
    feats = best_features()
    trials = []
    for task in ("cls", "reg"):
        for model in MODELS:
            for seed in SEEDS:
                rng = np.random.RandomState(1000 + seed)
                for c in range(N_CONFIGS):
                    trials.append((model, task, seed, c, sample_params(model, task, rng)))
    print(f"{len(trials)} HPO trials", flush=True)
    t0 = time.time()
    out = Parallel(n_jobs=workers, verbose=5)(
        delayed(run_classical)(feats[(m, t)], m, t, "scaffold", s, params=p, n_jobs=2, tag=f"trial{c}")
        for m, t, s, c, p in trials)
    rows = []
    for (m, t, s, c, p), (row, _) in zip(trials, out):
        row.update({"config": c, "params": repr(p)})
        rows.append(row)
    trials_df = pd.DataFrame(rows)
    trials_df.to_csv(TRIALS_CSV, index=False)

    best_rows, best_preds = [], []
    for (m, t, s), g in trials_df.groupby(["model", "task", "seed"]):
        col = "val_" + PRIMARY[t]
        best = g.loc[g[col].idxmax() if HIGHER_IS_BETTER[PRIMARY[t]] else g[col].idxmin()]
        k = trials.index(next(tr for tr in trials if tr[:4] == (m, t, s, best["config"])))
        row, preds = out[k]
        row = dict(row, tag="tuned", params=best["params"])
        best_rows.append(row)
        best_preds.append(preds.assign(tag="tuned"))
    pd.DataFrame(best_rows).to_csv(METRICS_CSV, index=False)
    if PREDS_CSV.exists():
        PREDS_CSV.unlink()
    append_csv(pd.concat(best_preds), PREDS_CSV)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    main(ap.parse_args().workers)
