"""Demonstrates the ensemble score (3) and adaptive edge/cloud routing on a scored test set.

Usage:
    python experiments/routing_demo.py --scores scores.csv --theta1 0.3 --theta2 0.7
where scores.csv has columns: if, ae (raw scores) and optional xgb, label.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.routing import ScoreNormalizer, adapt_weights, cloud_traffic_share, ensemble_score, route  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--theta1", type=float, default=0.3)
    ap.add_argument("--theta2", type=float, default=0.7)
    ap.add_argument("--latency-budget-ms", type=float, default=100.0)
    ap.add_argument("--complexity", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    norm = {k: ScoreNormalizer().fit(df[k].to_numpy()).transform(df[k].to_numpy()) for k in ("if", "ae")}
    w = adapt_weights(args.latency_budget_ms, args.complexity)
    s = ensemble_score(norm, w)
    decisions = route(s, args.theta1, args.theta2)
    print("weights:", w)
    print("share sent to cloud: %.1f %%" % (100 * cloud_traffic_share(decisions)))
    print(pd.Series(decisions).value_counts().to_string())


if __name__ == "__main__":
    main()
