"""Output module: structured security events (no raw data, no payload)."""
import json
from datetime import datetime, timezone


def make_event(device_id: str, score: float, is_anomaly: bool, decision: str = "", explanation: str = ""):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_id": device_id,          # pseudonymous identifier
        "anomaly_score": round(float(score), 4),
        "is_anomaly": bool(is_anomaly),
        "decision": decision,
        "explanation": explanation or ("High anomaly score" if is_anomaly else "Normal behavior"),
    }


def output(results, fmt: str = "console", device_ids=None, decisions=None):
    scores, is_anomaly, threshold = results
    for i, (score, flag) in enumerate(zip(scores, is_anomaly)):
        dev = device_ids[i] if device_ids is not None else f"device_{i}"
        dec = decisions[i] if decisions is not None else ""
        if fmt == "json":
            print(json.dumps(make_event(dev, score, flag, dec), ensure_ascii=False))
        else:
            status = "ANOMALY" if flag else "OK"
            print(f"{status:7} | score={score:.4f} | threshold={threshold:.4f} | {dev} {dec}")
