"""Shared helpers: one train/evaluate run of a classical model and result-file handling."""
import os
import warnings

import numpy as np
import pandas as pd

from .data import RESULTS, get_split, load_dataset
from .features import BINARY, get_features
from .metrics import evaluate
from .models import make_model

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")  # silence the same warnings in joblib workers

TASK_TARGET = {"cls": "active", "reg": "pIC50"}


def targets(df, task):
    return df[TASK_TARGET[task]].values.astype(float)


def run_classical(feature, model, task, split, seed, params=None, n_jobs=-1, tag="default"):
    """Fit on the training set, predict validation and test sets; return a metrics row and predictions."""
    warnings.filterwarnings("ignore")
    df = load_dataset()
    X = get_features(feature, df["smiles"].tolist())
    y = targets(df, task)
    tr, va, te = get_split(df, split, seed)
    est = make_model(model, task, feature in BINARY, seed=seed, params=params, n_jobs=n_jobs)
    est.fit(X[tr], y[tr])
    p_val, p_test = est.predict(X[va]), est.predict(X[te])
    row = {"split": split, "seed": seed, "task": task, "feature": feature, "model": model, "tag": tag}
    row.update({f"val_{k}": v for k, v in evaluate(task, y[va], p_val).items()})
    row.update({f"test_{k}": v for k, v in evaluate(task, y[te], p_test).items()})
    preds = pd.DataFrame({"split": split, "seed": seed, "task": task, "feature": feature, "model": model,
                          "tag": tag, "subset": ["val"] * len(va) + ["test"] * len(te),
                          "idx": np.concatenate([va, te]), "y": np.concatenate([y[va], y[te]]),
                          "pred": np.concatenate([p_val, p_test])})
    return row, preds


def append_csv(df, path):
    """Append rows, aligning columns with the existing file (classification and regression rows differ)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        header = list(pd.read_csv(path, nrows=0).columns)
        if any(c not in header for c in df.columns):  # widen the file once to the union of columns
            pd.concat([pd.read_csv(path), df]).to_csv(path, index=False)
            return
        df = df.reindex(columns=header)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def done_keys(path, cols):
    if not path.exists():
        return set()
    return set(map(tuple, pd.read_csv(path, usecols=cols)[cols].astype(str).values.tolist()))


PRED_DIR = RESULTS / "predictions"
