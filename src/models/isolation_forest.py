from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from .base_model import BaseModel


class IsolationForestModel(BaseModel):
    """Unsupervised edge-level filter. Trained on normal traffic only."""

    def __init__(self, contamination="auto", n_estimators: int = 100, random_state: int = 42):
        self.model = IsolationForest(contamination=contamination, n_estimators=n_estimators,
                                     random_state=random_state, n_jobs=-1)

    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        if y is not None:
            X = X[y == 0]  # fit on normal traffic only
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # decision_function: positive = normal; negate so that higher = more anomalous
        return -self.model.decision_function(X)
