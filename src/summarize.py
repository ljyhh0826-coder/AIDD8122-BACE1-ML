"""Collect every number quoted in the report into one JSON file, so the text cannot drift from results.

Usage: python -m src.summarize
"""
import json

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from .data import RESULTS, SEEDS, get_split, load_dataset
from .features import get_features
from .metrics import HIGHER_IS_BETTER, PRIMARY
from .models import tanimoto

OUT = RESULTS / "summary.json"


def best_of(df, task, metric, keys):
    """Mean over seeds, then the best configuration and its standard deviation."""
    g = df[df["task"] == task].groupby(keys)["test_" + metric]
    m, s = g.mean(), g.std()
    key = m.idxmax() if HIGHER_IS_BETTER[metric] else m.idxmin()
    name = key if isinstance(key, str) else " + ".join(map(str, key))
    return {"name": name, "mean": float(m[key]), "sd": float(s[key])}


def describe(df, task, metric, where=None):
    sub = df[df["task"] == task]
    if where is not None:
        sub = sub[where(sub)]
    v = sub.groupby("seed")["test_" + metric].mean()
    return {"mean": float(v.mean()), "sd": float(v.std())}


def main():
    out = {}
    df = load_dataset()
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    out["data"] = {
        "n": len(df), "n_active": int(df["active"].sum()), "n_inactive": int((1 - df["active"]).sum()),
        "active_frac": float(df["active"].mean()), "n_scaffolds": int(df["scaffold"].nunique()),
        "singleton_scaffolds": int((df["scaffold"].value_counts() == 1).sum()),
        "pic50_min": float(df["pIC50"].min()), "pic50_max": float(df["pIC50"].max()),
        "pic50_mean": float(df["pIC50"].mean()), "pic50_sd": float(df["pIC50"].std()),
        "mw_median": float(np.median([Descriptors.MolWt(m) for m in mols])),
        "logp_median": float(np.median([Descriptors.MolLogP(m) for m in mols])),
        "heavy_median": float(np.median([m.GetNumHeavyAtoms() for m in mols])),
    }
    X = get_features("ECFP4", df["smiles"].tolist())
    for kind in ("random", "scaffold"):
        sims, sizes = [], None
        for seed in SEEDS:
            tr, va, te = get_split(df, kind, seed)
            sims.append(tanimoto(X[te], X[tr]).max(1))
            sizes = (len(tr), len(va), len(te))
        s = np.concatenate(sims)
        out.setdefault("splits", {})[kind] = {
            "train": sizes[0], "val": sizes[1], "test": sizes[2],
            "mean_max_sim": float(s.mean()), "frac_below_0.5": float((s < 0.5).mean())}

    bench = pd.read_csv(RESULTS / "benchmark_metrics.csv")
    sc = bench[bench["split"] == "scaffold"]
    out["benchmark"] = {}
    for task, metric in (("cls", "ROC_AUC"), ("reg", "RMSE")):
        o = {"best": best_of(sc, task, metric, ["feature", "model"])}
        g = sc[sc["task"] == task]
        fm = g.groupby("feature")["test_" + metric].mean()
        mm = g.groupby("model")["test_" + metric].mean()
        o["by_feature"] = {k: float(v) for k, v in fm.items()}
        o["by_model"] = {k: float(v) for k, v in mm.items()}
        o["feature_spread"] = float(fm.max() - fm.min())
        o["model_spread"] = float(mm.max() - mm.min())
        o["best_feature"] = str(fm.idxmax() if HIGHER_IS_BETTER[metric] else fm.idxmin())
        o["worst_feature"] = str(fm.idxmin() if HIGHER_IS_BETTER[metric] else fm.idxmax())
        o["best_model"] = str(mm.idxmax() if HIGHER_IS_BETTER[metric] else mm.idxmin())
        out["benchmark"][task] = o

    out["split_gap"] = {}
    common = bench[bench["feature"].isin(["ECFP4", "RDKit2D", "ECFP4+RDKit2D"])]
    for task, metric in (("cls", "ROC_AUC"), ("reg", "RMSE")):
        piv = common[common["task"] == task].pivot_table(
            index=["feature", "model"], columns="split", values="test_" + metric)
        sign = 1 if HIGHER_IS_BETTER[metric] else -1
        d = (piv["random"] - piv["scaffold"]) * sign  # positive = the random split looks better
        out["split_gap"][task] = {"mean": float(d.mean()), "max": float(d.max()),
                                  "n_better_on_random": int((d > 0).sum()), "n_total": int(len(d))}

    hpo_path = RESULTS / "hpo_metrics.csv"
    if hpo_path.exists():
        hpo = pd.read_csv(hpo_path)
        trials = pd.read_csv(RESULTS / "hpo_trials.csv")
        out["hpo"] = {"n_configs_per_model_seed": int(trials.groupby(["model", "task", "seed"]).size().iloc[0])}
        for task, metric in (("cls", "ROC_AUC"), ("reg", "RMSE")):
            base = sc[sc["task"] == task]
            pick = base.groupby(["model", "feature"])["val_" + metric].mean()
            best_feat = {m: (g.idxmax()[1] if HIGHER_IS_BETTER[metric] else g.idxmin()[1])
                         for m, g in pick.groupby(level=0)}
            d_base = base[[f == best_feat[m] for m, f in zip(base["model"], base["feature"])]]
            b = d_base.groupby("model")["test_" + metric].mean()
            t = hpo[hpo["task"] == task].groupby("model")["test_" + metric].mean()
            sign = 1 if HIGHER_IS_BETTER[metric] else -1
            delta = (t - b) * sign
            out["hpo"][task] = {"best": best_of(hpo, task, metric, ["model"]),
                                "delta_by_model": {k: float(v) for k, v in delta.items()},
                                "mean_delta": float(delta.mean()),
                                "n_improved": int((delta > 0).sum()), "n_models": int(len(delta)),
                                "best_representation": best_feat}

    gnn_path = RESULTS / "gnn_metrics.csv"
    if gnn_path.exists():
        gnn = pd.read_csv(gnn_path)
        g_sc = gnn[gnn["split"] == "scaffold"]
        out["gnn"] = {"mean_epochs": float(gnn["epochs"].mean()), "mean_seconds": float(gnn["seconds"].mean())}
        for task, metric in (("cls", "ROC_AUC"), ("reg", "RMSE")):
            o = {}
            for variant, g in g_sc[g_sc["task"] == task].groupby("model"):
                v = g["test_" + metric]
                o[variant] = {"mean": float(v.mean()), "sd": float(v.std()), "n": int(len(v))}
            out["gnn"][task] = o
            rnd = gnn[(gnn["split"] == "random") & (gnn["task"] == task) & (gnn["model"] == "GNN (base)")]
            if len(rnd):
                out["gnn"][task]["_random_split"] = {"mean": float(rnd["test_" + metric].mean()),
                                                     "sd": float(rnd["test_" + metric].std())}

    ens_path = RESULTS / "ensemble_metrics.csv"
    if ens_path.exists():
        ens = pd.read_csv(ens_path)
        members = pd.read_csv(RESULTS / "ensemble_members.csv")
        out["ensemble"] = {}
        for task, metric in (("cls", "ROC_AUC"), ("reg", "RMSE")):
            e = ens[ens["task"] == task]
            if not len(e):
                continue
            chosen = members[members["task"] == task]
            out["ensemble"][task] = {
                "mean": float(e["test_" + metric].mean()), "sd": float(e["test_" + metric].std()),
                "mean_members": float(e["n_members"].mean()),
                "most_common": chosen["member"].value_counts().head(5).to_dict()}

    interp = RESULTS / "interpretation.json"
    if interp.exists():
        out["interpretation"] = json.loads(interp.read_text())

    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
