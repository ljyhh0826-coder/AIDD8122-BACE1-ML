"""Figures for the graph neural network ablations and the validation-selected ensemble."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .data import FIGURES, RESULTS
from .metrics import HIGHER_IS_BETTER, PRIMARY
from .plotting import COVE, GRAY, INK2, MUTED, save, setup

ABLATION_GROUPS = [
    ("depth", ["1 layer", "2 layers", "GNN (base)", "4 layers", "5 layers"]),
    ("readout", ["sum readout", "mean readout", "max readout", "GNN (base)"]),
    ("inputs", ["no bond features", "no chirality", "GNN (base)", "GNN + RDKit2D"]),
]
SHORT = {"GNN (base)": "base (3 layers)", "GNN + RDKit2D": "+ RDKit2D", "no bond features": "no bond feat.",
         "sum readout": "sum", "mean readout": "mean", "max readout": "max", "no chirality": "no chirality"}


def fig_ablation(gnn, path):
    """Ablation of depth, readout and input features, as a change from the base configuration."""
    sc = gnn[gnn["split"] == "scaffold"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    out = {}
    for row, (task, metric) in enumerate((("cls", "ROC_AUC"), ("reg", "RMSE"))):
        sub = sc[sc["task"] == task]
        stats = sub.groupby("model")["test_" + metric].agg(["mean", "std"])
        base = stats.loc["GNN (base)", "mean"]
        for col, (name, variants) in enumerate(ABLATION_GROUPS):
            ax = axes[row, col]
            have = [v for v in variants if v in stats.index]
            m = stats.loc[have, "mean"].values
            s = stats.loc[have, "std"].values
            x = np.arange(len(have))
            colours = [COVE[0] if v == "GNN (base)" else GRAY for v in have]
            ax.bar(x, m, 0.6, yerr=s, color=colours, capsize=2, error_kw={"lw": 0.9, "ecolor": MUTED})
            ax.axhline(base, color=COVE[1], lw=1.1, ls="--", zorder=0)
            ax.set_xticks(x, [SHORT.get(v, v) for v in have], fontsize=7.4, rotation=20, ha="right")
            ax.set_ylabel("test " + metric.replace("_", "-"))
            ax.set_ylim(min(m - s) * 0.97, max(m + s) * 1.02)
            ax.set_title(f"{'ab'[row]}{col + 1}  {name} ({'classification' if task == 'cls' else 'regression'})",
                         fontsize=9)
            ax.grid(axis="x", visible=False)
            out[(task, name)] = {v: float(stats.loc[v, "mean"]) for v in have}
    save(fig, path)
    return out


def fig_final(bench, hpo, gnn, ens, path):
    """Best configuration per model family, with the ensemble, on the scaffold split."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"wspace": 0.3})
    summary = {}
    for ax, task, metric in zip(axes, ("cls", "reg"), ("ROC_AUC", "RMSE")):
        rows = []
        col = "test_" + metric
        h = hpo[hpo["task"] == task]
        for model, g in h.groupby("model"):
            rows.append((f"{model} (tuned)", g[col].mean(), g[col].std()))
        if gnn is not None:
            gs = gnn[(gnn["split"] == "scaffold") & (gnn["task"] == task)]
            for name in ("GNN (base)", "GNN + RDKit2D"):
                g = gs[gs["model"] == name]
                if len(g):
                    rows.append((name, g[col].mean(), g[col].std()))
        if ens is not None:
            g = ens[ens["task"] == task]
            if len(g):
                rows.append(("Ensemble", g[col].mean(), g[col].std()))
        df = pd.DataFrame(rows, columns=["name", "mean", "std"]).sort_values(
            "mean", ascending=not HIGHER_IS_BETTER[metric])
        y = np.arange(len(df))
        colours = [COVE[1] if n == "Ensemble" else (COVE[2] if n.startswith("GNN") else COVE[0])
                   for n in df["name"]]
        ax.barh(y, df["mean"], 0.68, xerr=df["std"], color=colours, capsize=2,
                error_kw={"lw": 0.9, "ecolor": MUTED})
        ax.set_yticks(y, df["name"], fontsize=8)
        ax.invert_yaxis()
        lo = (df["mean"] - df["std"]).min()
        hi = (df["mean"] + df["std"]).max()
        ax.set_xlim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.18)
        for yy, (m, s) in enumerate(zip(df["mean"], df["std"])):
            ax.text(m + s + (hi - lo) * 0.02, yy, f"{m:.3f}", va="center", fontsize=7.4, color=INK2)
        ax.set_xlabel("test " + metric.replace("_", "-"))
        ax.set_title(f"{'ab'[0 if task == 'cls' else 1]}  "
                     f"{'Classification (higher is better)' if task == 'cls' else 'Regression (lower is better)'}")
        ax.grid(axis="y", visible=False)
        summary[task] = df.to_dict("records")
    save(fig, path)
    return summary


if __name__ == "__main__":
    setup()
    gnn = pd.read_csv(RESULTS / "gnn_metrics.csv")
    bench = pd.read_csv(RESULTS / "benchmark_metrics.csv")
    hpo = pd.read_csv(RESULTS / "hpo_metrics.csv")
    ens_path = RESULTS / "ensemble_metrics.csv"
    ens = pd.read_csv(ens_path) if ens_path.exists() else None
    print(fig_ablation(gnn, FIGURES / "fig8_ablation.png"))
    print(fig_final(bench, hpo, gnn, ens, FIGURES / "fig12_final.png"))
