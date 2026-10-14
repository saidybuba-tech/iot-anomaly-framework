"""Ensemble scoring and adaptive edge/cloud routing (sections 2.4-2.5 of the paper).

s(x) = w_IF * s_IF(x) + w_AE * s_AE(x) + w_XGB * s_XGB(x)       (3)

Routing with two thresholds theta1 < theta2:
  s <  theta1          -> NORMAL : decided on the edge, nothing is sent to the cloud
  theta1 <= s <= theta2 -> VERIFY : only the feature vector is sent to the cloud
  s >  theta2          -> BLOCK  : immediate reaction on the edge
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

NORMAL, VERIFY, BLOCK = "normal", "verify_in_cloud", "react_on_edge"


class ScoreNormalizer:
    """Maps raw model scores to [0, 1] using min/max observed on validation data."""

    def __init__(self):
        self.lo, self.hi = None, None

    def fit(self, scores: np.ndarray):
        self.lo, self.hi = float(np.min(scores)), float(np.max(scores))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        span = (self.hi - self.lo) or 1.0
        return np.clip((scores - self.lo) / span, 0.0, 1.0)


@dataclass
class EnsembleWeights:
    w_if: float = 0.5
    w_ae: float = 0.5
    w_xgb: float = 0.0  # 0 on the edge (XGBoost runs in the cloud)

    def normalized(self) -> "EnsembleWeights":
        s = self.w_if + self.w_ae + self.w_xgb
        return EnsembleWeights(self.w_if / s, self.w_ae / s, self.w_xgb / s)


def adapt_weights(latency_budget_ms: float, traffic_complexity: float,
                  xgb_accuracy: Optional[float] = None) -> EnsembleWeights:
    """Heuristic weight adaptation.

    * tighter latency budget  -> more weight on Isolation Forest (fastest model)
    * more complex traffic    -> more weight on the autoencoder
    * known XGBoost accuracy  -> XGBoost weight (cloud only)
    traffic_complexity is expected in [0, 1].
    """
    w_if = 1.0 / max(latency_budget_ms, 1.0) * 100.0
    w_ae = 0.5 + float(np.clip(traffic_complexity, 0.0, 1.0))
    w_xgb = float(xgb_accuracy) if xgb_accuracy is not None else 0.0
    return EnsembleWeights(w_if, w_ae, w_xgb).normalized()


def ensemble_score(scores: Dict[str, np.ndarray], weights: EnsembleWeights) -> np.ndarray:
    """scores: dict with keys 'if', 'ae', optional 'xgb' -- each already normalized to [0, 1]."""
    s = weights.w_if * scores["if"] + weights.w_ae * scores["ae"]
    if "xgb" in scores and weights.w_xgb > 0:
        s = s + weights.w_xgb * scores["xgb"]
    return s


def route(scores: np.ndarray, theta1: float, theta2: float) -> np.ndarray:
    if not theta1 < theta2:
        raise ValueError("theta1 must be smaller than theta2")
    decisions = np.full(scores.shape, VERIFY, dtype=object)
    decisions[scores < theta1] = NORMAL
    decisions[scores > theta2] = BLOCK
    return decisions


def cloud_traffic_share(decisions: np.ndarray) -> float:
    """Share of events forwarded to the cloud (for the cloud-traffic reduction metric)."""
    return float(np.mean(decisions == VERIFY))
