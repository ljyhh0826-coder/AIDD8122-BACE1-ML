"""Experiments 6-7: SHAP attribution over ECFP4 bits, atom-level maps and the applicability domain.

Usage: python -m src.interpret
"""
import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from .data import FIGURES, RESULTS, get_split, load_dataset
from .features import get_features, morgan_generator
from .metrics import enrichment_factor, evaluate
from .models import make_model, tanimoto
from .plotting import COVE, GRAY, INK2, MUTED, save, setup

SEED = 0
TOP_BITS = 12


def fit_reference_model(task="cls", seed=SEED):
    """The model used for interpretation: RF on ECFP4 (competitive and directly explainable)."""
    df = load_dataset()
    X = get_features("ECFP4", df["smiles"].tolist())
    y = df["active"].values if task == "cls" else df["pIC50"].values
    tr, va, te = get_split(df, "scaffold", seed)
    model = make_model("RF", task, True, seed=seed).fit(X[tr], y[tr])
    return df, X, y, (tr, va, te), model


def draw_bit(smiles_list, X, bit, size=(230, 175)):
    """Draw the circular environment a bit encodes, in the context of a molecule that contains it.

    RDKit's DrawMorganBit keeps the ring and aromaticity context and marks the attachment points,
    which a bare PathToSubmol fragment loses.
    """
    gen = morgan_generator()
    for i in np.where(X[:, bit] == 1)[0][:200]:
        mol = Chem.MolFromSmiles(smiles_list[i])
        ao = Chem.rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=ao)
        info = {k: tuple(v) for k, v in ao.GetBitInfoMap().items()}
        if bit not in info:
            continue
        png = Draw.DrawMorganBit(mol, int(bit), info, useSVG=False, molSize=size,
                                 drawOptions=_bit_draw_options())
        return Image.open(io.BytesIO(png)), info[bit][0][1]
    return None, None


def _bit_draw_options():
    opts = Draw.rdMolDraw2D.MolDrawOptions()
    opts.clearBackground = False
    return opts


def fig_shap(path):
    """Global SHAP importance of ECFP4 bits, with the substructure each important bit encodes."""
    import shap

    df, X, y, (tr, va, te), model = fit_reference_model("cls")
    explainer = shap.TreeExplainer(model.model_)
    sv = explainer.shap_values(X[te].astype(np.float32), check_additivity=False)
    sv = sv[..., 1] if sv.ndim == 3 else sv  # class-1 contributions
    smiles = df["smiles"].tolist()
    present = X[te] == 1
    # A bit only matters where the substructure occurs, so rank by the mean effect when present,
    # requiring enough occurrences for the mean to be meaningful.
    enough = present.sum(0) >= 10
    signed = np.where(enough, np.divide((sv * present).sum(0), np.maximum(present.sum(0), 1)), 0.0)
    top = np.argsort(-signed)[:TOP_BITS // 2]
    bottom = np.argsort(signed)[:TOP_BITS // 2]
    bits = np.concatenate([bottom[::-1], top[::-1]])

    fig = plt.figure(figsize=(12.5, 4.8))
    gs = fig.add_gridspec(2, 5, width_ratios=[2.4, 1, 1, 1, 1], hspace=0.5, wspace=0.1)
    ax = fig.add_subplot(gs[:, 0])
    freq = present.sum(0)
    labels = [f"bit {b} (n={freq[b]})" for b in bits]
    ax.barh(labels, signed[bits], color=[COVE[1] if s < 0 else COVE[0] for s in signed[bits]], height=0.68)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_xlabel("mean SHAP value where the substructure occurs")
    ax.set_title("a  ECFP4 bits with the strongest effect")
    ax.tick_params(axis="y", labelsize=7.4)
    ax.grid(axis="y", visible=False)
    ax.text(0.97, 0.5, "favours\nactivity", transform=ax.transAxes, ha="right", fontsize=7.4, color=COVE[0])
    ax.text(0.03, 0.5, "favours\ninactivity", transform=ax.transAxes, ha="left", fontsize=7.4, color=COVE[1])

    shown = list(top[:4]) + list(bottom[:4])
    for k, bit in enumerate(shown):
        axi = fig.add_subplot(gs[k // 4, 1 + k % 4])
        img, radius = draw_bit(smiles, X, int(bit))
        if img is not None:
            axi.imshow(img)
        axi.set_xticks([])
        axi.set_yticks([])
        axi.grid(False)
        for s in axi.spines.values():
            s.set_edgecolor(GRAY)
        colour = COVE[0] if signed[bit] > 0 else COVE[1]
        row = "favours activity" if signed[bit] > 0 else "favours inactivity"
        axi.set_title(f"bit {bit}  ({signed[bit]:+.3f}, r={radius})\n{row}", fontsize=7.6, color=colour,
                      weight="normal", linespacing=1.3)
    fig.text(0.635, 0.985, "b  Substructures encoded by those bits",
             fontsize=10, weight="bold", ha="center")
    save(fig, path)
    return {"top_active_bits": [int(b) for b in top], "top_inactive_bits": [int(b) for b in bottom],
            "signed": {int(b): float(signed[b]) for b in bits}}


def fig_atom_attribution(path, n_mols=4):
    """Per-atom attribution: how much removing each atom's bits changes the predicted probability."""
    df, X, y, (tr, va, te), model = fit_reference_model("cls")
    gen = morgan_generator()
    probs = model.predict(X[te])
    scaffolds = df["scaffold"].values

    def confident(order):  # one molecule per scaffold, so the panels are not near-identical analogues
        seen, out = set(), []
        for k in order:
            s = scaffolds[te[k]]
            if s not in seen:
                seen.add(s)
                out.append(k)
            if len(out) == n_mols // 2:
                return out
        return out

    picks = confident(np.argsort(-probs)) + confident(np.argsort(probs))
    fig, axes = plt.subplots(1, n_mols, figsize=(3 * n_mols, 3.1))
    for ax, k in zip(np.atleast_1d(axes), picks):
        i = te[k]
        mol = Chem.MolFromSmiles(df["smiles"].iloc[i])
        base = model.predict(X[i:i + 1])[0]
        weights = []
        for a in range(mol.GetNumAtoms()):  # zero the bits this atom contributes to, then re-predict
            ao = Chem.rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            gen.GetFingerprint(mol, additionalOutput=ao)
            bits = [b for b, envs in ao.GetBitInfoMap().items() if any(c == a for c, _ in envs)]
            x = X[i].copy()
            x[bits] = 0
            weights.append(base - model.predict(x[None, :])[0])
        weights = np.array(weights)
        scale = np.abs(weights).max() or 1.0
        d = Draw.rdMolDraw2D.MolDraw2DCairo(380, 320)
        opts = d.drawOptions()
        opts.clearBackground = False
        opts.highlightRadius = 0.42
        # Blend each atom's colour towards white by how weakly it contributes, so strong atoms read clearly.
        colours = {}
        for a, w in enumerate(weights):
            rgb = (0.16, 0.47, 0.84) if w > 0 else (0.92, 0.41, 0.20)
            t = min(abs(w) / scale, 1.0) ** 0.6
            colours[a] = tuple(1 - (1 - c) * t for c in rgb)
        Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(range(mol.GetNumAtoms())),
                                                highlightAtomColors=colours)
        d.FinishDrawing()
        ax.imshow(Image.open(io.BytesIO(d.GetDrawingText())))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_edgecolor(GRAY)
        ax.set_title(f"P(active) = {base:.2f}, measured {'active' if y[i] else 'inactive'}", fontsize=8)
    fig.suptitle("Atom attribution (blue raises, orange lowers the predicted probability)", fontsize=10,
                 weight="bold", y=1.04)
    save(fig, path)


def fig_applicability(path):
    """Does accuracy fall with distance from the training set, and does confidence track it?"""
    df = load_dataset()
    X = get_features("ECFP4", df["smiles"].tolist())
    rows = []
    for seed in range(5):
        tr, va, te = get_split(df, "scaffold", seed)
        model = make_model("RF", "cls", True, seed=seed).fit(X[tr], df["active"].values[tr])
        reg = make_model("RF", "reg", True, seed=seed).fit(X[tr], df["pIC50"].values[tr])
        sim = tanimoto(X[te], X[tr]).max(1)
        p = model.predict(X[te])
        rows.append(pd.DataFrame({"seed": seed, "sim": sim, "y": df["active"].values[te], "p": p,
                                  "pic50": df["pIC50"].values[te], "pred_pic50": reg.predict(X[te]),
                                  "conf": np.abs(p - 0.5) * 2}))
    d = pd.concat(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    ax = axes[0]
    bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    d["bin"] = pd.cut(d["sim"], bins, right=False)
    g = d.groupby("bin", observed=True)
    acc = g.apply(lambda s: ((s["p"] >= 0.5).astype(int) == s["y"]).mean(), include_groups=False)
    err = g.apply(lambda s: np.sqrt(((s["pic50"] - s["pred_pic50"]) ** 2).mean()), include_groups=False)
    n = g.size()
    x = np.arange(len(acc))
    ax.bar(x, acc.values, 0.6, color=COVE[0])
    ax.set_xticks(x, [f"{i.left:.1f}–{i.right:.1f}\nn={m}" for i, m in zip(acc.index, n)], fontsize=7)
    ax.set_ylim(0.5, 1)
    ax.set_xlabel("max Tanimoto similarity to training set")
    ax.set_ylabel("classification accuracy")
    ax.set_title("a  Accuracy vs similarity")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ax.plot(x, err.values, marker="o", ms=5, color=COVE[1])
    ax.set_xticks(x, [f"{i.left:.1f}–{i.right:.1f}" for i in err.index], fontsize=7)
    ax.set_xlabel("max Tanimoto similarity to training set")
    ax.set_ylabel("pIC50 RMSE")
    ax.set_title("b  Regression error vs similarity")
    ax.grid(axis="x", visible=False)

    ax = axes[2]  # keeping only the most confident predictions should raise accuracy if confidence is meaningful
    fracs = np.linspace(0.1, 1.0, 10)
    accs = [((d.nlargest(int(f * len(d)), "conf")["p"] >= 0.5).astype(int)
             == d.nlargest(int(f * len(d)), "conf")["y"]).mean() for f in fracs]
    ax.plot(fracs * 100, accs, marker="o", ms=5, color=COVE[0])
    ax.axhline(accs[-1], color=MUTED, lw=1, ls=":")
    ax.set_xlabel("% of test set retained (most confident first)")
    ax.set_ylabel("accuracy")
    ax.set_title("c  Confidence-based selection")
    ef = {f"EF{int(f * 100)}%": float(np.mean([enrichment_factor(g["y"].values, g["p"].values, f)
                                               for _, g in d.groupby("seed")])) for f in (0.01, 0.05, 0.1)}
    ax.text(0.97, 0.06, "\n".join(f"{k} = {v:.2f}" for k, v in ef.items()), transform=ax.transAxes,
            ha="right", fontsize=7.6, color=INK2)
    save(fig, path)
    return {"accuracy_by_bin": {str(k): float(v) for k, v in acc.items()},
            "rmse_by_bin": {str(k): float(v) for k, v in err.items()},
            "n_by_bin": {str(k): int(v) for k, v in n.items()},
            "enrichment": ef, "acc_top10pct": float(accs[0]), "acc_all": float(accs[-1])}


if __name__ == "__main__":
    setup()
    out = {}
    out["shap"] = fig_shap(FIGURES / "fig9_shap.png")
    fig_atom_attribution(FIGURES / "fig10_atoms.png")
    out["applicability"] = fig_applicability(FIGURES / "fig11_applicability.png")
    (RESULTS / "interpretation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["applicability"], indent=2))
