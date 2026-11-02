"""Data loading, standardisation and splitting for the BACE-1 dataset."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "bace.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "bace_clean.csv"
SPLIT_DIR = ROOT / "data" / "splits"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEEDS = [0, 1, 2, 3, 4]
FRACTIONS = (0.8, 0.1, 0.1)
ACTIVE_PIC50 = 7.0  # IC50 <= 100 nM, the threshold used to define the MoleculeNet "Class" label


def standardize_smiles(smiles):
    """Clean up, keep the parent fragment and neutralise; return canonical isomeric SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol)


def murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def load_dataset(rebuild=False):
    """Return the curated dataset (one row per unique standardised structure)."""
    if CLEAN_CSV.exists() and not rebuild:
        return pd.read_csv(CLEAN_CSV)
    raw = pd.read_csv(RAW_CSV)
    df = pd.DataFrame({
        "id": raw["CID"],
        "smiles_raw": raw["mol"],
        "pIC50": raw["pIC50"].astype(float),
        "active": raw["Class"].astype(int),
    })
    df["smiles"] = [standardize_smiles(s) for s in df["smiles_raw"]]
    n_invalid = int(df["smiles"].isna().sum())
    df = df.dropna(subset=["smiles"])
    # Structures that collapse onto the same parent are merged; conflicting class labels are dropped.
    agg = df.groupby("smiles", as_index=False).agg(
        id=("id", "first"), pIC50=("pIC50", "mean"), active=("active", "mean"), n_records=("id", "size"))
    n_conflict = int(((agg["active"] > 0) & (agg["active"] < 1)).sum())
    agg = agg[(agg["active"] == 0) | (agg["active"] == 1)].copy()
    agg["active"] = agg["active"].astype(int)
    agg["scaffold"] = [murcko_scaffold(s) for s in agg["smiles"]]
    agg = agg.sort_values("id", key=lambda s: s.str.split("_").str[-1].astype(int)).reset_index(drop=True)
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg[["id", "smiles", "pIC50", "active", "scaffold", "n_records"]].to_csv(CLEAN_CSV, index=False)
    print(f"raw={len(raw)} invalid={n_invalid} duplicates_merged={len(df) - len(agg) - n_conflict} "
          f"label_conflicts_dropped={n_conflict} final={len(agg)}")
    return pd.read_csv(CLEAN_CSV)


def random_split(labels, seed, fractions=FRACTIONS):
    """Class-stratified random split."""
    idx = np.arange(len(labels))
    train, rest = train_test_split(idx, test_size=1 - fractions[0], random_state=seed, stratify=labels)
    rel_test = fractions[2] / (fractions[1] + fractions[2])
    val, test = train_test_split(rest, test_size=rel_test, random_state=seed, stratify=labels[rest])
    return np.sort(train), np.sort(val), np.sort(test)


def scaffold_split(scaffolds, seed, fractions=FRACTIONS):
    """Balanced randomised Bemis-Murcko scaffold split (whole scaffold groups go to one subset).

    Scaffold groups larger than half the validation size are placed first so that they end up
    in the training set; the remaining groups are shuffled with the seed (as in Chemprop).
    """
    n = len(scaffolds)
    groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s].append(i)
    n_val = int(round(fractions[1] * n))
    n_test = int(round(fractions[2] * n))
    n_train = n - n_val - n_test
    rng = np.random.RandomState(seed)
    big = [g for g in groups.values() if len(g) > n_val / 2]
    small = [g for g in groups.values() if len(g) <= n_val / 2]
    rng.shuffle(big)
    rng.shuffle(small)
    train, val, test = [], [], []
    for g in big + small:
        if len(train) + len(g) <= n_train:
            train += g
        elif len(val) + len(g) <= n_val:
            val += g
        else:
            test += g
    return np.sort(train), np.sort(val), np.sort(test)


def get_split(df, kind, seed):
    """Load (or create and cache) a split as index arrays into the curated dataframe."""
    path = SPLIT_DIR / f"{kind}_seed{seed}.json"
    if path.exists():
        d = json.loads(path.read_text())
        return np.array(d["train"]), np.array(d["val"]), np.array(d["test"])
    if kind == "random":
        tr, va, te = random_split(df["active"].values, seed)
    elif kind == "scaffold":
        tr, va, te = scaffold_split(df["scaffold"].fillna("").values, seed)
    else:
        raise ValueError(kind)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}))
    return tr, va, te


if __name__ == "__main__":
    data = load_dataset(rebuild=True)
    print(data.head())
    print("actives:", data["active"].sum(), "inactives:", (1 - data["active"]).sum(),
          "unique scaffolds:", data["scaffold"].nunique())
    for kind in ("random", "scaffold"):
        for seed in SEEDS:
            tr, va, te = get_split(data, kind, seed)
            print(kind, seed, len(tr), len(va), len(te), "test active frac=%.2f" % data["active"].values[te].mean())
