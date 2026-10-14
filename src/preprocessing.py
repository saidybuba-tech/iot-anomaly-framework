"""Preprocessing: cleaning, label extraction, feature selection and scaling.

Only flow metadata is used (packet counts, volumes, rates, protocol flags,
destination entropy, timing). No payload and no personal data are processed.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def split_features_labels(df: pd.DataFrame, target_col: Optional[str] = None,
                          benign_values=("BenignTraffic", "Benign", "benign", 0, "0")
                          ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Separate features from the label column and convert labels to binary (1 = anomaly)."""
    df = df.dropna()
    if target_col and target_col in df.columns:
        y_raw = df[target_col]
        y = (~y_raw.isin(benign_values)).astype(int).to_numpy()
        X = df.drop(columns=[target_col])
    else:
        X, y = df, None
    X = X.select_dtypes(include=[np.number])
    return X, y


def select_features(X: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """Keep only the configured feature subset (if given)."""
    if features:
        missing = [f for f in features if f not in X.columns]
        if missing:
            raise KeyError(f"Features not found in data: {missing}")
        return X[features]
    return X


def make_scaler(kind: str = "standard"):
    return StandardScaler() if kind == "standard" else MinMaxScaler()


def preprocess(df: pd.DataFrame, target_col: Optional[str] = None, features=None, scaler: str = "standard"):
    """Full pipeline used by run.py: returns scaled X, labels (or None) and the fitted scaler."""
    X, y = split_features_labels(df, target_col)
    X = select_features(X, features)
    sc = make_scaler(scaler)
    X_scaled = sc.fit_transform(X.to_numpy(dtype=float))
    return X_scaled, y, sc
