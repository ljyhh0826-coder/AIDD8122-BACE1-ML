"""Classical machine-learning models with a uniform fit / predict interface.

`predict` returns P(active) for the classification task and pIC50 for the regression task.
Binary fingerprints are modelled with Tanimoto-based kNN / SVM; real-valued descriptors are
imputed and standardised inside the model pipeline (fitted on the training set only).
"""
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

MODELS = ["kNN", "LR", "RF", "SVM", "XGBoost", "MLP"]


def tanimoto(A, B):
    """Tanimoto (Jaccard) similarity between two sets of binary fingerprints."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


class TanimotoKNN(BaseEstimator):
    """Similarity-weighted k-nearest-neighbour model in Tanimoto space."""

    def __init__(self, task="cls", k=5):
        self.task = task
        self.k = k

    def fit(self, X, y):
        self.X_ = X
        self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        sim = tanimoto(X, self.X_)
        nn = np.argsort(-sim, axis=1)[:, : self.k]
        w = np.take_along_axis(sim, nn, axis=1) + 1e-6
        return (w * self.y_[nn]).sum(1) / w.sum(1)


class TanimotoSVM(BaseEstimator):
    """SVC / SVR with a precomputed Tanimoto kernel (a valid Mercer kernel for binary vectors)."""

    def __init__(self, task="cls", C=1.0, epsilon=0.1, seed=0):
        self.task = task
        self.C = C
        self.epsilon = epsilon
        self.seed = seed

    def fit(self, X, y):
        self.X_ = X
        K = tanimoto(X, X)
        if self.task == "cls":
            self.model_ = SVC(kernel="precomputed", C=self.C, probability=True, random_state=self.seed)
        else:
            self.model_ = TransformedTargetRegressor(
                regressor=SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon), transformer=StandardScaler())
        self.model_.fit(K, y)
        return self

    def predict(self, X):
        K = tanimoto(X, self.X_)
        if self.task == "cls":
            return self.model_.predict_proba(K)[:, 1]
        return self.model_.predict(K)


class SklearnModel(BaseEstimator):
    """Wraps an sklearn-style estimator so that `predict` returns probabilities for classifiers."""

    def __init__(self, estimator, task="cls"):
        self.estimator = estimator
        self.task = task

    def fit(self, X, y):
        self.model_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        if self.task == "cls":
            return self.model_.predict_proba(X)[:, 1]
        return self.model_.predict(X)


def _preprocess():
    return [SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler()]


def default_params(name, task):
    if name == "kNN":
        return {"k": 5}
    if name == "LR":
        return {"C": 1.0} if task == "cls" else {"alpha": 1.0}
    if name == "RF":
        return {"n_estimators": 500, "max_features": "sqrt" if task == "cls" else 0.33, "min_samples_leaf": 1}
    if name == "SVM":
        return {"C": 1.0, "epsilon": 0.1, "gamma": "scale"}
    if name == "XGBoost":
        return {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
                "colsample_bytree": 0.5, "min_child_weight": 1, "reg_lambda": 1.0}
    if name == "MLP":
        return {"hidden_layer_sizes": (512, 128), "alpha": 1e-3, "learning_rate_init": 1e-3}
    raise ValueError(name)


def make_model(name, task, binary, seed=0, params=None, n_jobs=-1):
    """Build a model; `binary` selects Tanimoto-based variants and disables feature scaling."""
    p = default_params(name, task)
    p.update(params or {})
    cls = task == "cls"
    pre = [] if binary else _preprocess()

    if name == "kNN":
        if binary:
            return TanimotoKNN(task, k=p["k"])
        est = (KNeighborsClassifier if cls else KNeighborsRegressor)(n_neighbors=p["k"], weights="distance")
    elif name == "LR":
        est = LogisticRegression(C=p["C"], max_iter=5000) if cls else Ridge(alpha=p["alpha"])
    elif name == "RF":
        kw = dict(n_estimators=p["n_estimators"], max_features=p["max_features"],
                  min_samples_leaf=p["min_samples_leaf"], n_jobs=n_jobs, random_state=seed)
        est = RandomForestClassifier(**kw) if cls else RandomForestRegressor(**kw)
    elif name == "SVM":
        if binary:
            return TanimotoSVM(task, C=p["C"], epsilon=p["epsilon"], seed=seed)
        if cls:
            est = SVC(C=p["C"], gamma=p["gamma"], probability=True, random_state=seed)
        else:
            est = TransformedTargetRegressor(regressor=SVR(C=p["C"], gamma=p["gamma"], epsilon=p["epsilon"]),
                                             transformer=StandardScaler())
    elif name == "XGBoost":
        kw = dict(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                  subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                  min_child_weight=p["min_child_weight"], reg_lambda=p["reg_lambda"],
                  tree_method="hist", n_jobs=n_jobs, random_state=seed)
        est = XGBClassifier(eval_metric="logloss", **kw) if cls else XGBRegressor(**kw)
    elif name == "MLP":
        kw = dict(hidden_layer_sizes=p["hidden_layer_sizes"], alpha=p["alpha"],
                  learning_rate_init=p["learning_rate_init"], early_stopping=True, max_iter=500, random_state=seed)
        est = MLPClassifier(**kw) if cls else TransformedTargetRegressor(regressor=MLPRegressor(**kw),
                                                                       transformer=StandardScaler())
    else:
        raise ValueError(name)
    return SklearnModel(make_pipeline(*pre, est) if pre else est, task)
