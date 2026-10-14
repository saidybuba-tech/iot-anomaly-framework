from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseModel(ABC):
    """Common interface: every model returns an anomaly score (higher = more anomalous)."""

    @abstractmethod
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
