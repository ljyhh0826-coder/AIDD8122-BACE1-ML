"""Figures describing the dataset and the experimental workflow."""
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.manifold import TSNE

from .data import FIGURES, get_split, load_dataset
from .features import get_features
from .models import tanimoto
from .plotting import COVE, GRAY, INK2, MUTED, save, setup

WORKFLOW_STEPS = [
    ("1  Data", "BACE-1 inhibitors\nMoleculeNet, n = 1,513\npIC50 and active label"),
    ("2  Curation", "RDKit standardisation\nparent fragment, neutralise\nde-duplicate to n = 1,512"),
    ("3  Splits", "Bemis-Murcko scaffold split\nstratified random split\n80/10/10, five seeds"),
    ("4  Representations", "ECFP4, FCFP4, AtomPair\nMACCS, RDKit 2D (216)\nfused and molecular graph"),
    ("5  Models", "kNN, LR, RF, SVM\nXGBoost, MLP\nGINE graph neural network"),
    ("6  Improvement", "random-search tuning\nGNN ablations, hybrid GNN\nvalidation-picked ensemble"),
    ("7  Evaluation", "ROC-AUC, PR-AUC, MCC\nRMSE, MAE, R2, EF\nrandom vs scaffold gap"),
    ("8  Interpretation", "SHAP over ECFP4 bits\natom attribution maps\napplicability domain"),
]


def fig_workflow(path):
    """Schematic of the computational workflow (Methods section)."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")
    # Top row runs left to right, bottom row right to left, so the diagram reads as one chain.
    top = [(1 + 20 * i, 24) for i in range(5)]
    bottom = [(81 - 20 * i, 2) for i in range(3)]
    hue = [COVE[0]] * 3 + [COVE[2]] * 2 + [COVE[1]] * 3
    for (x, y), (title, body), c in zip(top + bottom, WORKFLOW_STEPS, hue):
        ax.add_patch(FancyBboxPatch((x, y), 18, 17, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    fc=c + "1A", ec=c, lw=1.2))
        ax.text(x + 9, y + 14.2, title, ha="center", va="center", fontsize=9.5, weight="bold", color="#0b0b0b")
        ax.text(x + 9, y + 6.6, body, ha="center", va="center", fontsize=7.6, color=INK2, linespacing=1.45)
    arrow = dict(arrowstyle="-|>", color=MUTED, lw=1.3, mutation_scale=12)
    for i in range(4):
        ax.annotate("", xy=(20.8 + 20 * i, 32.5), xytext=(19.2 + 20 * i, 32.5), arrowprops=arrow)
    ax.annotate("", xy=(90, 19.8), xytext=(90, 23.2), arrowprops=arrow)  # down the right-hand side
    for i in range(2):
        ax.annotate("", xy=(79.2 - 20 * i, 10.5), xytext=(80.8 - 20 * i, 10.5), arrowprops=arrow)
    save(fig, path)


def fig_dataset(path):
    """Four-panel overview: activity classes, pIC50, property space and scaffold frequency."""
    df = load_dataset()
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    mw = np.array([Descriptors.MolWt(m) for m in mols])
    logp = np.array([Descriptors.MolLogP(m) for m in mols])
    active = df["active"].values.astype(bool)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))

    ax = axes[0]
    counts = [int((~active).sum()), int(active.sum())]
    bars = ax.bar(["inactive", "active"], counts, width=0.55, color=[GRAY, COVE[0]])
    ax.bar_label(bars, fontsize=8, color=INK2, padding=2)
    ax.set_ylabel("molecules")
    ax.set_title("a  Activity classes")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ax.hist([df["pIC50"][~active], df["pIC50"][active]], bins=28, stacked=True, color=[GRAY, COVE[0]],
            label=["inactive", "active"])
    ax.axvline(7.0, color=COVE[1], lw=1.6, ls="--")
    ax.text(7.08, ax.get_ylim()[1] * 0.92, "threshold\npIC50 = 7", fontsize=7.5, color=COVE[1], va="top")
    ax.set_xlabel("pIC50")
    ax.set_ylabel("molecules")
    ax.set_title("b  Potency distribution")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    ax.scatter(mw[~active], logp[~active], s=7, c=GRAY, alpha=0.65, lw=0, label="inactive")
    ax.scatter(mw[active], logp[active], s=7, c=COVE[0], alpha=0.65, lw=0, label="active")
    ax.axvline(500, color=MUTED, lw=0.9, ls=":")
    ax.axhline(5, color=MUTED, lw=0.9, ls=":")
    ax.set_xlabel("molecular weight (Da)")
    ax.set_ylabel("cLogP")
    ax.set_title("c  Property space")
    ax.legend(loc="lower right", fontsize=7.5, markerscale=1.6)

    ax = axes[3]
    sizes = df["scaffold"].value_counts().values
    ax.plot(np.arange(1, len(sizes) + 1), sizes, color=COVE[0])
    ax.fill_between(np.arange(1, len(sizes) + 1), sizes, color=COVE[0], alpha=0.12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("scaffold rank")
    ax.set_ylabel("molecules per scaffold")
    ax.set_title("d  Scaffold frequency")
    singletons = int((sizes == 1).sum())
    ax.text(0.95, 0.9, f"{len(sizes)} scaffolds\n{singletons} singletons", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=INK2)
    save(fig, path)
    return {"n": len(df), "n_active": int(active.sum()), "n_scaffold": int(df["scaffold"].nunique()),
            "singletons": singletons, "mw_median": float(np.median(mw)), "logp_median": float(np.median(logp)),
            "pic50_min": float(df["pIC50"].min()), "pic50_max": float(df["pIC50"].max())}


def fig_splits(path, seed=0):
    """t-SNE of ECFP4 space and nearest-neighbour similarity for both splitting strategies."""
    df = load_dataset()
    X = get_features("ECFP4", df["smiles"].tolist())
    emb = TSNE(n_components=2, metric="jaccard", init="random", perplexity=30, random_state=0).fit_transform(X)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    stats = {}
    for ax, kind, title in zip(axes[:2], ("random", "scaffold"), ("a  Random split", "b  Scaffold split")):
        tr, va, te = get_split(df, kind, seed)
        ax.scatter(emb[tr, 0], emb[tr, 1], s=6, c=GRAY, alpha=0.55, lw=0, label="train")
        ax.scatter(emb[va, 0], emb[va, 1], s=9, c=COVE[3], alpha=0.85, lw=0, label="validation")
        ax.scatter(emb[te, 0], emb[te, 1], s=9, c=COVE[1], alpha=0.85, lw=0, label="test")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.legend(loc="upper right", fontsize=7.5, markerscale=1.5)

    ax = axes[2]
    for kind, colour in (("random", COVE[0]), ("scaffold", COVE[1])):
        sims = []
        for s in range(5):
            tr, _, te = get_split(df, kind, s)
            sims.append(tanimoto(X[te], X[tr]).max(1))
        sims = np.concatenate(sims)
        stats[kind] = float(sims.mean())
        ax.hist(sims, bins=30, range=(0, 1), histtype="step", lw=2, color=colour,
                label=f"{kind} (mean {sims.mean():.2f})")
    ax.set_xlabel("max Tanimoto similarity to training set")
    ax.set_ylabel("test molecules")
    ax.set_title("c  Test-set novelty")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(axis="x", visible=False)
    save(fig, path)
    return stats


if __name__ == "__main__":
    setup()
    fig_workflow(FIGURES / "fig1_workflow.png")
    print(fig_dataset(FIGURES / "fig2_dataset.png"))
    print(fig_splits(FIGURES / "fig3_splits.png"))
