"""Model comparison protocol (section 3.1 of the paper).

Usage:
    python experiments/compare_models.py --data path/to/ciciot2023_merged.csv --label-col label

Protocol:
  * stratified split 70 / 15 / 15 (train / validation / test)
  * StandardScaler fitted on the training set
  * Isolation Forest: unsupervised, trained on normal training traffic only
  * Autoencoder: trained on normal training traffic, early stopping on validation loss
  * XGBoost: SMOTE on the training set only, 5-fold CV grid search over max_depth and learning_rate
  * all models evaluated on the same untouched hold-out test set:
    Precision, Recall, F1, ROC-AUC, PR-AUC, mean inference time per sample (ms)

The CIC IoT Dataset 2023 is not included in this repository (size); download it from
https://www.unb.ca/cic/datasets/iotdataset-2023.html
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.detector import fit_threshold  # noqa: E402
from src.models import AutoencoderModel, IsolationForestModel, XGBoostModel  # noqa: E402
from src.preprocessing import split_features_labels  # noqa: E402

SEED = 42


def evaluate(name, model, X_val, y_val, X_test, y_test, threshold=None):
    if threshold is None:
        # unsupervised models: tau = 99th percentile of scores on normal validation traffic
        threshold = fit_threshold(model.predict(X_val[y_val == 0]), 99.0)
    t0 = time.perf_counter()
    scores = model.predict(X_test)
    ms = (time.perf_counter() - t0) / len(X_test) * 1000
    pred = (scores > threshold).astype(int)
    return {"model": name,
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, scores),
            "pr_auc": average_precision_score(y_test, scores),
            "inference_ms_per_sample": ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--sample", type=int, default=0, help="optional random subsample size")
    ap.add_argument("--out", default="experiments/results.json")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=SEED)
    X, y = split_features_labels(df, args.label_col)
    X = X.to_numpy(dtype=float)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_val, X_te = sc.transform(X_tr), sc.transform(X_val), sc.transform(X_te)

    results = []

    iforest = IsolationForestModel()
    iforest.train(X_tr, y_tr)
    results.append(evaluate("Isolation Forest", iforest, X_val, y_val, X_te, y_te))

    ae = AutoencoderModel()
    ae.train(X_tr, y_tr, X_val=X_val[y_val == 0])
    results.append(evaluate("Deep Autoencoder", ae, X_val, y_val, X_te, y_te))

    from imblearn.over_sampling import SMOTE
    X_sm, y_sm = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
    grid = GridSearchCV(XGBoostModel().model, {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1, 0.2]},
                        scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    grid.fit(X_sm, y_sm)
    xgbm = XGBoostModel()
    xgbm.model = grid.best_estimator_
    results.append(evaluate("XGBoost", xgbm, X_val, y_val, X_te, y_te, threshold=0.5))

    table = pd.DataFrame(results)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("XGBoost best params:", grid.best_params_)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "xgb_best_params": grid.best_params_}, f, indent=2)


if __name__ == "__main__":
    main()
