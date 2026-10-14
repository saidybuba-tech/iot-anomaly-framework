"""Entry point: load data -> preprocess -> train model -> score -> detect -> output."""
import argparse

import yaml

from src.data_ingestion import load_data
from src.detector import detect_anomalies, fit_threshold
from src.models import AutoencoderModel, IsolationForestModel, XGBoostModel
from src.output import output
from src.preprocessing import preprocess

MODELS = {"isolation_forest": IsolationForestModel, "autoencoder": AutoencoderModel, "xgboost": XGBoostModel}


def load_model(cfg):
    name = cfg["model"]["name"]
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    return MODELS[name](**(cfg["model"].get("params") or {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = load_data(cfg["data"]["path"], cfg["data"].get("source_type", "csv"))
    X, y, _ = preprocess(df, cfg["data"].get("target_column"), cfg["data"].get("features"),
                         cfg["data"].get("scaler", "standard"))
    model = load_model(cfg)
    model.train(X, y)
    scores = model.predict(X)
    threshold = cfg.get("threshold")
    if threshold is None and y is not None and cfg["model"]["name"] != "xgboost":
        threshold = fit_threshold(scores[y == 0], 99.0)  # tau from normal traffic
    elif threshold is None and cfg["model"]["name"] == "xgboost":
        threshold = 0.5
    results = detect_anomalies(scores, threshold)
    output(results, cfg["output"]["format"])


if __name__ == "__main__":
    main()
