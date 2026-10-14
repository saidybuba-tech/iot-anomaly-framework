from typing import Optional

import numpy as np
import xgboost as xgb

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """Supervised cloud-level verifier. Requires labelled data (y: 0 = normal, 1 = attack)."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 6, learning_rate: float = 0.1,
                 random_state: int = 42):
        self.model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       learning_rate=learning_rate, objective="binary:logistic",
                                       eval_metric="logloss", random_state=random_state, n_jobs=-1)

    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        if y is None:
            raise ValueError("XGBoostModel is supervised: labels y are required")
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # probability of the attack class
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_
