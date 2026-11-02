"""Molecular representations: hashed fingerprints, MACCS keys and RDKit 2D descriptors."""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

from .data import ROOT

CACHE = ROOT / "data" / "processed" / "features"
N_BITS = 2048

# Feature sets used in the benchmark; binary ones are compared with the Tanimoto similarity.
FEATURE_SETS = ["ECFP4", "FCFP4", "AtomPair", "MACCS", "RDKit2D", "ECFP4+RDKit2D"]
BINARY = {"ECFP4", "FCFP4", "AtomPair", "MACCS"}
# Ipc spans >30 orders of magnitude and is numerically unusable without a transform.
EXCLUDED_DESCRIPTORS = {"Ipc"}
DESCRIPTOR_NAMES = [n for n, _ in Descriptors.descList if n not in EXCLUDED_DESCRIPTORS]


def morgan_generator(radius=2, features=False):
    if features:
        inv = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=N_BITS, atomInvariantsGenerator=inv)
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=N_BITS)


def _fp_matrix(mols, gen):
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)


def maccs(mols):
    out = np.zeros((len(mols), 167), dtype=np.uint8)
    for i, m in enumerate(mols):
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), out[i])
    return out


def rdkit2d(mols):
    funcs = dict(Descriptors.descList)
    X = np.full((len(mols), len(DESCRIPTOR_NAMES)), np.nan, dtype=np.float64)
    for i, m in enumerate(mols):
        for j, name in enumerate(DESCRIPTOR_NAMES):
            try:
                X[i, j] = funcs[name](m)
            except Exception:
                pass
    X[~np.isfinite(X)] = np.nan
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def compute(name, smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if name == "ECFP4":
        return _fp_matrix(mols, morgan_generator())
    if name == "FCFP4":
        return _fp_matrix(mols, morgan_generator(features=True))
    if name == "AtomPair":
        return _fp_matrix(mols, rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS))
    if name == "MACCS":
        return maccs(mols)
    if name == "RDKit2D":
        return rdkit2d(mols)
    raise ValueError(name)


def get_features(name, smiles):
    """Return the feature matrix for a feature set, cached on disk as .npy."""
    if "+" in name:
        return np.hstack([get_features(part, smiles).astype(np.float32) for part in name.split("+")])
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npy"
    if path.exists():
        X = np.load(path)
        if len(X) == len(smiles):
            return X
    X = compute(name, smiles)
    np.save(path, X)
    return X
