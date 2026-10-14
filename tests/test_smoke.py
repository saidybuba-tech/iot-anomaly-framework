"""Smoke tests on synthetic data (not the experimental datasets)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.detector import detect_anomalies  # noqa: E402
from src.models import IsolationForestModel, XGBoostModel  # noqa: E402
from src.routing import BLOCK, NORMAL, VERIFY, adapt_weights, ensemble_score, route  # noqa: E402


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, size=(n, 6))
    attack = rng.normal(4, 1, size=(n // 10, 6))
    X = np.vstack([normal, attack])
    y = np.r_[np.zeros(len(normal)), np.ones(len(attack))].astype(int)
    return X, y


def test_isolation_forest_separates_synthetic_attacks():
    X, y = synthetic()
    m = IsolationForestModel()
    m.train(X, y)
    s = m.predict(X)
    assert s[y == 1].mean() > s[y == 0].mean()


def test_xgboost_requires_labels_and_learns():
    X, y = synthetic()
    m = XGBoostModel(n_estimators=50)
    m.train(X, y)
    p = m.predict(X)
    assert ((p > 0.5).astype(int) == y).mean() > 0.95


def test_detector_threshold():
    scores, flags, thr = detect_anomalies(np.arange(100, dtype=float))
    assert flags.sum() == 1 and thr > 98


def test_routing():
    s = np.array([0.1, 0.5, 0.9])
    assert list(route(s, 0.3, 0.7)) == [NORMAL, VERIFY, BLOCK]
    w = adapt_weights(100, 0.5)
    assert abs(w.w_if + w.w_ae + w.w_xgb - 1) < 1e-9
    assert ensemble_score({"if": s, "ae": s}, w).shape == s.shape
