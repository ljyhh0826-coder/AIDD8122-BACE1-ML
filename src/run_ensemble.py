"""Experiment 5: a validation-selected ensemble of the diverse model families.

Candidate members are the per-model best representation plus the graph neural network. Members are
added greedily in order of validation score, and the ensemble size that maximises the validation
score is applied to the test set, so no test information is used in building the ensemble.

Usage: python -m src.run_ensemble
"""
import numpy as np
import pandas as pd

from .data import RESULTS, SEEDS
from .experiment import PRED_DIR
from .metrics import HIGHER_IS_BETTER, PRIMARY, evaluate

METRICS_CSV = RESULTS / "ensemble_metrics.csv"
MEMBERS_CSV = RESULTS / "ensemble_members.csv"


def rank_normalise(p):
    """Map scores to [0, 1] ranks so that members on different scales can be averaged."""
    order = np.argsort(np.argsort(p))
    return order / max(len(p) - 1, 1)


def load_preds():
    frames = []
    for f in ("benchmark_predictions.csv", "gnn_predictions.csv", "hpo_predictions.csv"):
        path = PRED_DIR / f
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames)


def candidates(preds, task, seed):
    """One entry per model family: tuned classical models plus the base and hybrid GNN."""
    sel = preds[(preds["task"] == task) & (preds["split"] == "scaffold") & (preds["seed"] == seed)]
    out = {}
    for (model, tag), g in sel.groupby(["model", "tag"]):
        if tag == "default" and model not in ("GNN (base)", "GNN + RDKit2D"):
            continue  # classical models enter with their tuned configuration only
        for feature, gg in g.groupby("feature"):
            val = gg[gg["subset"] == "val"].sort_values("idx")
            test = gg[gg["subset"] == "test"].sort_values("idx")
            if len(val) and len(test):
                out[f"{model} ({feature})"] = (val, test)
    return out


def greedy(task, members, metric):
    """Add members one at a time, always taking the best remaining validation score."""
    names = list(members)
    val_true = members[names[0]][0]["y"].values
    chosen, history, remaining = [], [], set(names)
    while remaining:
        best, best_score = None, None
        for name in remaining:
            trial = chosen + [name]
            p = np.mean([rank_normalise(members[n][0]["pred"].values) for n in trial], axis=0)
            score = evaluate(task, val_true, p)[metric]
            if best_score is None or (score > best_score) == HIGHER_IS_BETTER[metric]:
                best, best_score = name, score
        chosen.append(best)
        remaining.discard(best)
        history.append((list(chosen), best_score))
    scores = [s for _, s in history]
    k = int(np.argmax(scores) if HIGHER_IS_BETTER[metric] else np.argmin(scores))
    return history[k][0], history


def main():
    preds = load_preds()
    rows, member_rows = [], []
    for task in ("cls", "reg"):
        metric = PRIMARY[task]
        for seed in SEEDS:
            members = candidates(preds, task, seed)
            if len(members) < 2:
                continue
            chosen, history = greedy(task, members, metric)
            test_true = members[chosen[0]][1]["y"].values
            # For regression the members predict pIC50 directly, so average the predictions themselves.
            agg = (lambda g: rank_normalise(g)) if task == "cls" else (lambda g: g)
            p_test = np.mean([agg(members[n][1]["pred"].values) for n in chosen], axis=0)
            row = {"task": task, "seed": seed, "model": "Ensemble", "feature": "mixed", "split": "scaffold",
                   "n_members": len(chosen)}
            row.update({f"test_{k}": v for k, v in evaluate(task, test_true, p_test).items()})
            rows.append(row)
            for rank, name in enumerate(chosen, 1):
                member_rows.append({"task": task, "seed": seed, "rank": rank, "member": name})
            best_single = max(((n, evaluate(task, members[n][1]["y"].values, members[n][1]["pred"].values)[metric])
                               for n in members), key=lambda kv: kv[1] if HIGHER_IS_BETTER[metric] else -kv[1])
            print(f"{task} seed={seed}: {len(chosen)} members, ensemble {metric}={row['test_' + metric]:.3f}, "
                  f"best single {best_single[0]} {best_single[1]:.3f}")
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(member_rows).to_csv(MEMBERS_CSV, index=False)
    df = pd.DataFrame(rows)
    for task in ("cls", "reg"):
        m = PRIMARY[task]
        g = df[df["task"] == task]["test_" + m]
        print(f"ensemble {task}: {m} = {g.mean():.3f} ± {g.std():.3f}")


if __name__ == "__main__":
    main()
