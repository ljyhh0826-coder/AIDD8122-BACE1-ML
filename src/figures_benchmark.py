"""Figures and tables for the representation x model benchmark, the split gap and tuning."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from .data import FIGURES, RESULTS
from .experiment import PRED_DIR
from .metrics import HIGHER_IS_BETTER, PRIMARY
from .plotting import COVE, GRAY, INK2, MARKERS, MUTED, save, setup

TABLES = RESULTS / "tables"
CLS_METRICS = ["ROC_AUC", "PR_AUC", "MCC", "F1", "BACC"]
REG_METRICS = ["RMSE", "MAE", "R2", "Spearman"]
FEATURE_ORDER = ["ECFP4", "FCFP4", "AtomPair", "MACCS", "RDKit2D", "ECFP4+RDKit2D"]
MODEL_ORDER = ["kNN", "LR", "RF", "SVM", "XGBoost", "MLP"]


def task_metrics(task):
    return CLS_METRICS if task == "cls" else REG_METRICS


def fmt_mean_sd(df, keys, task, metrics=None):
    """Aggregate over seeds into 'mean ± sd' strings plus the raw mean for sorting."""
    metrics = metrics or task_metrics(task)
    sub = df[df["task"] == task]
    g = sub.groupby(keys)
    out = pd.DataFrame(index=g.size().index)
    for m in metrics:
        mean, sd = g["test_" + m].mean(), g["test_" + m].std()
        out[m] = mean
        out[m + "_str"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean, sd)]
    return out.reset_index()


def heatmap(ax, matrix, row_labels, col_labels, title, higher_better, fmt="{:.3f}"):
    """One-hue sequential heatmap; the colour always runs light (worse) to dark (better)."""
    values = matrix.values
    shown = values if higher_better else -values
    im = ax.imshow(shown, cmap="Blues", aspect="auto")
    lo, hi = np.nanmin(shown), np.nanmax(shown)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            rel = (shown[i, j] - lo) / max(hi - lo, 1e-9)
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=7.6,
                    color="white" if rel > 0.62 else "#0b0b0b")
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    ax.set_title(title)
    ax.grid(False)
    ax.tick_params(length=0)
    return im


def fig_benchmark(bench, path):
    """Representation x model grids for both tasks on the scaffold split."""
    sc = bench[bench["split"] == "scaffold"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={"wspace": 0.42})
    out = {}
    for ax, task, metric, label in zip(axes, ("cls", "reg"), ("ROC_AUC", "RMSE"),
                                       ("a  Classification: test ROC-AUC", "b  Regression: test RMSE")):
        piv = (sc[sc["task"] == task].pivot_table(index="feature", columns="model", values="test_" + metric)
               .reindex(index=FEATURE_ORDER, columns=MODEL_ORDER))
        heatmap(ax, piv, FEATURE_ORDER, MODEL_ORDER, label, HIGHER_IS_BETTER[metric])
        best = piv.stack().idxmax() if HIGHER_IS_BETTER[metric] else piv.stack().idxmin()
        out[task] = {"best_feature": best[0], "best_model": best[1], "best_value": float(piv.loc[best])}
    save(fig, path)
    return out


def fig_split_gap(bench, path):
    """Paired random-vs-scaffold comparison for the representations run under both splits."""
    common = bench[bench["feature"].isin(["ECFP4", "RDKit2D", "ECFP4+RDKit2D"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"wspace": 0.25})
    gaps = {}
    for ax, task, metric, name in zip(axes, ("cls", "reg"), ("ROC_AUC", "RMSE"),
                                      ("a  Classification (ROC-AUC)", "b  Regression (RMSE)")):
        sub = common[common["task"] == task]
        piv = sub.pivot_table(index=["feature", "model"], columns="split", values="test_" + metric)
        piv = piv.reindex(pd.MultiIndex.from_product([FEATURE_ORDER, MODEL_ORDER],
                                                     names=["feature", "model"])).dropna()
        x = np.arange(len(piv))
        ax.plot(x, piv["random"], marker="o", ms=5, lw=0, color=COVE[0], label="random split")
        ax.plot(x, piv["scaffold"], marker="s", ms=5, lw=0, color=COVE[1], label="scaffold split")
        ax.vlines(x, piv["scaffold"], piv["random"], color=GRAY, lw=1.2, zorder=0)
        ax.set_xticks(x, [f"{f} + {m}" for f, m in piv.index], fontsize=6.6, rotation=90)
        ax.set_ylabel(metric.replace("_", "-"))
        ax.set_title(name)
        ax.legend(loc="best", fontsize=7.5)
        ax.grid(axis="x", visible=False)
        d = (piv["random"] - piv["scaffold"])
        gaps[task] = {"mean_gap": float(d.mean() if task == "cls" else -d.mean()),
                      "max_gap": float(d.abs().max())}
    save(fig, path)
    return gaps


def fig_tuning(bench, hpo, path):
    """Effect of random-search tuning, and the validation-score trajectory of the search."""
    trials = pd.read_csv(RESULTS / "hpo_trials.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    gains = {}
    for ax, task, metric, title in zip(axes[:2], ("cls", "reg"), ("ROC_AUC", "RMSE"),
                                       ("a  Classification (ROC-AUC)", "b  Regression (RMSE)")):
        base = bench[(bench["split"] == "scaffold") & (bench["task"] == task)]
        # Default baseline = each model's best representation by mean validation score.
        pick = base.groupby(["model", "feature"])["val_" + metric].mean()
        best_feat = {m: (g.idxmax()[1] if HIGHER_IS_BETTER[metric] else g.idxmin()[1])
                     for m, g in pick.groupby(level=0)}
        d_base = base[[f == best_feat[m] for m, f in zip(base["model"], base["feature"])]]
        b = d_base.groupby("model")["test_" + metric].agg(["mean", "std"]).reindex(MODEL_ORDER)
        t = (hpo[hpo["task"] == task].groupby("model")["test_" + metric].agg(["mean", "std"]).reindex(MODEL_ORDER))
        x = np.arange(len(MODEL_ORDER))
        ax.bar(x - 0.2, b["mean"], 0.38, yerr=b["std"], color=GRAY, label="default", capsize=2,
               error_kw={"lw": 0.9, "ecolor": MUTED})
        ax.bar(x + 0.2, t["mean"], 0.38, yerr=t["std"], color=COVE[0], label="tuned", capsize=2,
               error_kw={"lw": 0.9, "ecolor": MUTED})
        ax.set_xticks(x, MODEL_ORDER, fontsize=8)
        ax.set_ylabel("test " + metric.replace("_", "-"))
        ax.set_title(title)
        lo = min(b["mean"].min(), t["mean"].min())
        ax.set_ylim(lo * 0.9, None)
        ax.legend(loc="lower right" if task == "cls" else "upper right", fontsize=7.5)
        ax.grid(axis="x", visible=False)
        delta = (t["mean"] - b["mean"]) * (1 if HIGHER_IS_BETTER[metric] else -1)
        gains[task] = {m: float(v) for m, v in delta.items()}

    ax = axes[2]  # best-so-far validation ROC-AUC against the number of configurations tried
    sub = trials[trials["task"] == "cls"]
    for i, model in enumerate(MODEL_ORDER):
        curves = []
        for seed, g in sub[sub["model"] == model].groupby("seed"):
            v = g.sort_values("config")["val_ROC_AUC"].values
            curves.append(np.maximum.accumulate(v))
        curve = np.mean(curves, axis=0)
        ax.plot(np.arange(1, len(curve) + 1), curve, color=COVE[i], marker=MARKERS[i], ms=3.5, lw=1.6,
                markevery=3, label=model)
    ax.set_xlabel("configurations evaluated")
    ax.set_ylabel("best validation ROC-AUC")
    ax.set_title("c  Random-search progress")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    save(fig, path)
    return gains


def fig_roc(preds, bench, path):
    """ROC curves (classification) and predicted-vs-measured scatter (regression) on scaffold seed 0."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    sel = preds[(preds["split"] == "scaffold") & (preds["seed"] == 0) & (preds["subset"] == "test")
                & (preds["task"] == "cls") & (preds["tag"] == "default")]
    shown = [("ECFP4", "RF"), ("ECFP4", "SVM"), ("RDKit2D", "XGBoost"), ("MACCS", "RF")]
    for i, (feat, model) in enumerate(shown):
        g = sel[(sel["feature"] == feat) & (sel["model"] == model)]
        fpr, tpr, _ = roc_curve(g["y"], g["pred"])
        auc = bench[(bench["split"] == "scaffold") & (bench["task"] == "cls") & (bench["feature"] == feat)
                    & (bench["model"] == model) & (bench["seed"] == 0)]["test_ROC_AUC"].iloc[0]
        ax.plot(fpr, tpr, color=COVE[i], lw=1.8, label=f"{feat} + {model} ({auc:.3f})")
    gnn = preds[(preds["feature"] == "graph") & (preds["model"] == "GNN (base)") & (preds["task"] == "cls")
                & (preds["split"] == "scaffold") & (preds["seed"] == 0) & (preds["subset"] == "test")]
    if len(gnn):
        fpr, tpr, _ = roc_curve(gnn["y"], gnn["pred"])
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, color=COVE[4], lw=1.8, label=f"graph + GNN ({roc_auc_score(gnn['y'], gnn['pred']):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("a  ROC curves (scaffold split, seed 0)")
    ax.legend(loc="lower right", fontsize=7.2)

    ax = axes[1]
    g = preds[(preds["split"] == "scaffold") & (preds["seed"] == 0) & (preds["subset"] == "test")
              & (preds["task"] == "reg") & (preds["feature"] == "ECFP4") & (preds["model"] == "SVM")
              & (preds["tag"] == "default")]
    ax.scatter(g["y"], g["pred"], s=12, c=COVE[0], alpha=0.65, lw=0)
    lims = [min(g["y"].min(), g["pred"].min()) - 0.3, max(g["y"].max(), g["pred"].max()) + 0.3]
    ax.plot(lims, lims, color=MUTED, lw=1, ls=":")
    for off, style in ((1, "--"),):
        ax.plot(lims, [l + off for l in lims], color=GRAY, lw=0.9, ls=style)
        ax.plot(lims, [l - off for l in lims], color=GRAY, lw=0.9, ls=style)
    r = np.corrcoef(g["y"], g["pred"])[0, 1]
    rmse = float(np.sqrt(((g["y"] - g["pred"]) ** 2).mean()))
    ax.text(0.04, 0.95, f"RMSE = {rmse:.3f}\nPearson r = {r:.3f}\ndashed: ±1 log unit", transform=ax.transAxes,
            va="top", fontsize=7.6, color=INK2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("measured pIC50")
    ax.set_ylabel("predicted pIC50")
    ax.set_title("b  ECFP4 + SVM regression")
    save(fig, path)


def write_tables(bench, hpo, gnn):
    """CSV tables used verbatim in the report."""
    TABLES.mkdir(parents=True, exist_ok=True)
    sc = bench[bench["split"] == "scaffold"]
    for task in ("cls", "reg"):
        t = fmt_mean_sd(sc, ["feature", "model"], task)
        cols = ["feature", "model"] + [m + "_str" for m in task_metrics(task)]
        t = t[cols].rename(columns={m + "_str": m for m in task_metrics(task)})
        t.to_csv(TABLES / f"table_benchmark_{task}.csv", index=False)
    for task in ("cls", "reg"):
        rows = []
        base = fmt_mean_sd(sc, ["model", "feature"], task)
        metric = PRIMARY[task]
        for model in MODEL_ORDER:
            b = base[base["model"] == model]
            best = b.loc[b[metric].idxmax() if HIGHER_IS_BETTER[metric] else b[metric].idxmin()]
            tuned = fmt_mean_sd(hpo, ["model"], task)
            tr = tuned[tuned["model"] == model].iloc[0]
            rows.append({"model": model, "representation": best["feature"],
                         f"default {metric}": best[metric + "_str"], f"tuned {metric}": tr[metric + "_str"],
                         "delta": f"{(tr[metric] - best[metric]) * (1 if HIGHER_IS_BETTER[metric] else -1):+.3f}"})
        pd.DataFrame(rows).to_csv(TABLES / f"table_tuning_{task}.csv", index=False)
    if gnn is not None and len(gnn):
        for task in ("cls", "reg"):
            t = fmt_mean_sd(gnn[gnn["split"] == "scaffold"], ["model"], task)
            cols = ["model"] + [m + "_str" for m in task_metrics(task)]
            t[cols].rename(columns={m + "_str": m for m in task_metrics(task)}).to_csv(
                TABLES / f"table_gnn_{task}.csv", index=False)


if __name__ == "__main__":
    setup()
    bench = pd.read_csv(RESULTS / "benchmark_metrics.csv")
    hpo = pd.read_csv(RESULTS / "hpo_metrics.csv") if (RESULTS / "hpo_metrics.csv").exists() else None
    gnn_path = RESULTS / "gnn_metrics.csv"
    gnn = pd.read_csv(gnn_path) if gnn_path.exists() else None
    print(fig_benchmark(bench, FIGURES / "fig4_benchmark.png"))
    print(fig_split_gap(bench, FIGURES / "fig5_split_gap.png"))
    if hpo is not None:
        print(fig_tuning(bench, hpo, FIGURES / "fig6_tuning.png"))
        write_tables(bench, hpo, gnn)
    preds = pd.concat([pd.read_csv(PRED_DIR / f) for f in
                       ("benchmark_predictions.csv", "gnn_predictions.csv") if (PRED_DIR / f).exists()])
    fig_roc(preds, bench, FIGURES / "fig7_roc.png")
