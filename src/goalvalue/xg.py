"""Expected-goals models and honest, match-grouped evaluation.

Three models are compared on non-penalty shots:

* ``geometry``  logistic regression on spline(distance) + spline(angle) + body part,
  the textbook xG model;
* ``freeze``    the same plus freeze-frame features (defenders in the shot triangle,
  goalkeeper position, nearest defender) and shot context, still a penalised
  logistic GAM-style model, so every term can be plotted and explained;
* ``gbm``       histogram gradient boosting on the same inputs.

All predictions are out-of-fold from cross-validation **grouped by match**, so
shots from one match never sit in both training and test folds. The
regularisation strength of the logistic models is chosen by an inner grouped
CV inside each outer training fold (nested CV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

SPLINE_GEOMETRY = ["distance", "angle"]
SPLINE_FREEZE = ["gk_dist_line", "gk_dist_shotline", "nearest_def_dist"]
LINEAR_FREEZE = ["n_def_triangle", "n_att_triangle", "n_def_within_3", "gk_in_triangle",
                 "under_pressure", "first_time"]
CATEGORICAL_GEOMETRY = ["body"]
CATEGORICAL_FREEZE = ["body", "set_piece_shot", "technique_grp", "pattern_grp"]
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)


def prepare_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Clean and group raw shot columns into model inputs (no target information used)."""
    df = pd.DataFrame(index=shots.index)
    df["distance"] = shots["distance"].clip(0, 45)
    df["angle"] = shots["angle"]
    df["gk_dist_line"] = shots["gk_dist_line"].fillna(shots["distance"]).clip(-1, 15)
    df["gk_dist_shotline"] = shots["gk_dist_shotline"].fillna(10).clip(0, 10)
    df["nearest_def_dist"] = shots["nearest_def_dist"].fillna(10).clip(0, 10)
    df["n_def_triangle"] = shots["n_def_triangle"].fillna(0).clip(0, 4)
    df["n_att_triangle"] = shots["n_att_triangle"].fillna(0).clip(0, 3)
    df["n_def_within_3"] = shots["n_def_within_3"].fillna(0).clip(0, 3)
    df["gk_in_triangle"] = shots["gk_in_triangle"].fillna(0)
    df["under_pressure"] = shots["under_pressure"].astype(float)
    df["first_time"] = shots["first_time"].astype(float)
    df["body"] = np.select(
        [shots["body_part"].eq("Head"), shots["body_part"].isin(["Left Foot", "Right Foot"])],
        ["head", "foot"], "other")
    df["set_piece_shot"] = np.where(shots["shot_type"].isin(["Free Kick", "Corner"]), "direct", "open")
    tech = shots["technique"].fillna("Normal")
    df["technique_grp"] = np.select(
        [tech.eq("Normal"), tech.isin(["Volley", "Half Volley"]), tech.eq("Lob")],
        ["normal", "volley", "lob"], "acrobatic")
    pp = shots["play_pattern"].fillna("Other")
    df["pattern_grp"] = np.select(
        [pp.eq("From Counter"), pp.isin(["From Corner", "From Free Kick"])],
        ["counter", "set_piece"], "other")
    return df


def _spline() -> SplineTransformer:
    return SplineTransformer(n_knots=6, degree=3, knots="quantile", extrapolation="constant")


def make_logistic(kind: str, C: float = 1.0) -> Pipeline:
    """Penalised logistic regression with cubic-spline terms (an additive model)."""
    if kind == "geometry":
        spl, lin, cat = SPLINE_GEOMETRY, [], CATEGORICAL_GEOMETRY
    elif kind == "freeze":
        spl, lin, cat = SPLINE_GEOMETRY + SPLINE_FREEZE, LINEAR_FREEZE, CATEGORICAL_FREEZE
    else:
        raise ValueError(kind)
    transformers = [(f"s_{c}", make_pipeline(_spline(), StandardScaler()), [c]) for c in spl]
    if lin:
        transformers.append(("lin", StandardScaler(), lin))
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat))
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("model", LogisticRegression(C=C, max_iter=5000)),
    ])


def make_gbm(seed: int = 0) -> Pipeline:
    """Gradient boosting on the same inputs (categoricals one-hot encoded)."""
    num = SPLINE_GEOMETRY + SPLINE_FREEZE + LINEAR_FREEZE
    pre = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FREEZE),
    ])
    gbm = HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=400, max_leaf_nodes=12, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=seed)
    return Pipeline([("features", pre), ("model", gbm)])


def _tune_C(kind: str, X, y, groups, seed: int) -> float:
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = {C: [] for C in C_GRID}
    for tr, va in inner.split(X, y, groups):
        for C in C_GRID:
            m = make_logistic(kind, C).fit(X.iloc[tr], y[tr])
            scores[C].append(log_loss(y[va], m.predict_proba(X.iloc[va])[:, 1], labels=[0, 1]))
    return min(C_GRID, key=lambda c: np.mean(scores[c]))


def cross_validated_predictions(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                                n_splits: int = 5, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold xG for each model using outer CV grouped by match.

    Returns the predictions (one column per model) and the C chosen per fold.
    """
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = pd.DataFrame(index=X.index, columns=["geometry", "freeze", "gbm"], dtype=float)
    preds["fold"] = -1
    chosen = {"geometry": [], "freeze": []}
    for k, (tr, te) in enumerate(outer.split(X, y, groups)):
        Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
        for kind in ("geometry", "freeze"):
            C = _tune_C(kind, Xtr, ytr, gtr, seed + k)
            chosen[kind].append(C)
            m = make_logistic(kind, C).fit(Xtr, ytr)
            preds.iloc[te, preds.columns.get_loc(kind)] = m.predict_proba(X.iloc[te])[:, 1]
        g = make_gbm(seed + k).fit(Xtr, ytr)
        preds.iloc[te, preds.columns.get_loc("gbm")] = g.predict_proba(X.iloc[te])[:, 1]
        preds.iloc[te, preds.columns.get_loc("fold")] = k
    return preds, chosen


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic recalibration: regress y on logit(p). Perfect calibration = (1, 0)."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def score_table(y, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Log loss, Brier, ROC AUC and calibration slope for each prediction vector."""
    rows = []
    for name, p in preds.items():
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        slope, icpt = calibration_slope_intercept(y, p) if np.ptp(p) > 0 else (np.nan, np.nan)
        rows.append({
            "model": name,
            "log_loss": log_loss(y, p, labels=[0, 1]),
            "brier": brier_score_loss(y, p),
            "roc_auc": roc_auc_score(y, p) if np.ptp(p) > 0 else 0.5,
            "calib_slope": slope,
            "sum_xg": p.sum(),
        })
    return pd.DataFrame(rows).set_index("model")


def paired_bootstrap_logloss(y, p_a, p_b, groups, n_boot: int = 2000, seed: int = 0):
    """Bootstrap matches (not shots) to get a CI for mean log loss(a) - log loss(b)."""
    rng = np.random.default_rng(seed)
    eps = 1e-6
    la = -(y * np.log(np.clip(p_a, eps, 1)) + (1 - y) * np.log(np.clip(1 - p_a, eps, 1)))
    lb = -(y * np.log(np.clip(p_b, eps, 1)) + (1 - y) * np.log(np.clip(1 - p_b, eps, 1)))
    diff = pd.DataFrame({"g": groups, "d": la - lb}).groupby("g")["d"].agg(["sum", "count"])
    s, n = diff["sum"].to_numpy(), diff["count"].to_numpy()
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), np.percentile(boots, [2.5, 97.5])


def binned_calibration(y, p, n_bins: int = 10, edges=None) -> pd.DataFrame:
    """Observed goal rate vs mean prediction, in quantile bins or fixed ``edges``."""
    df = pd.DataFrame({"y": y, "p": p})
    if edges is None:
        df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    else:
        df["bin"] = pd.cut(df["p"], edges, labels=False, include_lowest=True)
    out = df.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    out["se"] = np.sqrt(out["obs"] * (1 - out["obs"]) / out["n"])
    return out
