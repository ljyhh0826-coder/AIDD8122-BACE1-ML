"""Experiment 4: graph neural network, its architecture ablations and a descriptor-augmented hybrid.

Usage: python -m src.run_gnn [--workers 5]
"""
import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .data import RESULTS, SEEDS, get_split, load_dataset
from .experiment import PRED_DIR, append_csv, done_keys, targets
from .metrics import evaluate

METRICS_CSV = RESULTS / "gnn_metrics.csv"
PREDS_CSV = PRED_DIR / "gnn_predictions.csv"

BASE = {"layers": 3, "readout": "sum+mean+max", "use_edges": True, "chirality": True, "descriptors": False}
VARIANTS = {
    "GNN (base)": {},
    "1 layer": {"layers": 1},
    "2 layers": {"layers": 2},
    "4 layers": {"layers": 4},
    "5 layers": {"layers": 5},
    "sum readout": {"readout": "sum"},
    "mean readout": {"readout": "mean"},
    "max readout": {"readout": "max"},
    "no bond features": {"use_edges": False},
    "no chirality": {"chirality": False},
    "GNN + RDKit2D": {"descriptors": True},
}

_cache = {}


def _graphs(chirality):
    from .gnn import GraphSet
    if chirality not in _cache:
        _cache[chirality] = GraphSet(load_dataset()["smiles"].tolist(), use_chirality=chirality)
    return _cache[chirality]


def run_one(job):
    import torch
    from .features import get_features
    from .gnn import train_gnn

    torch.set_num_threads(2)
    variant, split, task, seed = job
    cfg = {**BASE, **VARIANTS[variant]}
    df = load_dataset()
    y = targets(df, task)
    tr, va, te = get_split(df, split, seed)
    extra = get_features("RDKit2D", df["smiles"].tolist()).astype(np.float64) if cfg["descriptors"] else None
    t0 = time.time()
    preds, info, _ = train_gnn(_graphs(cfg["chirality"]), y, task, tr, va, te, seed=seed, extra=extra,
                               layers=cfg["layers"], readout=cfg["readout"], use_edges=cfg["use_edges"],
                               max_epochs=150, patience=25)
    row = {"split": split, "seed": seed, "task": task, "feature": "graph", "model": variant,
           "epochs": info["epochs"], "seconds": round(time.time() - t0, 1)}
    row.update({f"val_{k}": v for k, v in evaluate(task, y[va], preds["val"]).items()})
    row.update({f"test_{k}": v for k, v in evaluate(task, y[te], preds["test"]).items()})
    pred_df = pd.DataFrame({"split": split, "seed": seed, "task": task, "feature": "graph", "model": variant,
                            "tag": "default", "subset": ["val"] * len(va) + ["test"] * len(te),
                            "idx": np.concatenate([va, te]), "y": np.concatenate([y[va], y[te]]),
                            "pred": np.concatenate([preds["val"], preds["test"]])})
    return row, pred_df


def jobs():
    for task in ("cls", "reg"):
        for seed in SEEDS:
            yield ("GNN (base)", "scaffold", task, seed)
            yield ("GNN (base)", "random", task, seed)
    for variant in VARIANTS:
        if variant == "GNN (base)":
            continue
        for task in ("cls", "reg"):
            for seed in SEEDS:
                yield (variant, "scaffold", task, seed)


def main(workers):
    load_dataset()
    finished = done_keys(METRICS_CSV, ["model", "split", "task", "seed"])
    todo = [j for j in jobs() if tuple(map(str, j)) not in finished]
    print(f"{len(todo)} GNN runs to do", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for k, (row, pred_df) in enumerate(pool.imap_unordered(run_one, todo), 1):
            append_csv(pd.DataFrame([row]), METRICS_CSV)
            append_csv(pred_df, PREDS_CSV)
            metric = "test_ROC_AUC" if row["task"] == "cls" else "test_RMSE"
            print(f"[{k}/{len(todo)}] {row['model']} {row['split']} {row['task']} seed={row['seed']} "
                  f"{metric}={row[metric]:.3f} epochs={row['epochs']} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    main(ap.parse_args().workers)
