"""Anomaly detector: turns scores into decisions (formula (1) in the paper)."""
from typing import Optional

import numpy as np


def fit_threshold(val_scores: np.ndarray, percentile: float = 99.0) -> float:
    """Empirical threshold tau: percentile of scores on (normal) validation data."""
    return float(np.percentile(val_scores, percentile))


def detect_anomalies(scores: np.ndarray, threshold: Optional[float] = None):
    if threshold is None:
        threshold = fit_threshold(scores)
    is_anomaly = scores > threshold
    return scores, is_anomaly, threshold
