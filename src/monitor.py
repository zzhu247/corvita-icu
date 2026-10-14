"""Checks the API request log against what the model saw in training.

    python -m src.monitor --make-baseline            # once per model version
    python -m src.monitor --log logs/requests.jsonl  # e.g. hourly

Exit codes: 0 = ok or not enough traffic yet, 1 = alert, 2 = log file not found.
The log can be the API's LOG_PATH file or saved container output (`docker logs api > x.log`):
lines that are not JSON request records are skipped and counted.

Baseline (written next to the model, so it is versioned with it):
    per-vital missing rate and median count on TRAIN, mean score and alert rate on VAL.

Checks over the last `window` predictions:
    missing_rate   |live - train| > max_missing_shift      (feed outage, new charting rules)
    count_ratio    live median / train median outside band  (charting frequency changed)
    mean_score     |live - val| > max_score_shift           (case mix or input drift)
    alert_rate     |live - val| > max_alert_shift
    error rates    5xx share and 422 share of all /predict calls

Why these: masking vitals on the validation set (metrics.json -> masking_val) shows that
a missing HR feed lowers scores and alerts, and missing RespRate raises them, with no
error. Count features track how often staff chart. So the inputs most likely to break
the model silently are missingness and counts.

This is an alerting script, not an auto-retrain trigger. A person decides what happens next.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config, p
from src.split import load_splits

DEFAULTS = {
    "window": 500,
    "min_requests": 50,
    "max_missing_shift": 0.10,
    "count_ratio_band": [0.5, 2.0],
    "max_score_shift": 0.05,
    "max_alert_shift": 0.15,
    "max_5xx_rate": 0.01,
    "max_422_rate": 0.10,
}


def make_baseline(cfg: dict, model_dir: Path) -> dict:
    meta = json.loads((model_dir / "metadata.json").read_text())
    metrics = json.loads((model_dir / "metrics.json").read_text())
    X = pd.read_csv(p(cfg["data"]["processed_dir"]) / "features.csv", index_col="RecordID",
                    float_precision="round_trip")
    train = X.loc[load_splits(p(cfg["data"]["splits_dir"]))["train"]]
    val = metrics["comparison"][metrics["winner"]]["val"]
    base = {
        "model_version": meta["model_version"],
        "data_version": meta["data_version"],
        "missing_rate": {v: round(float(train[f"{v}_missing"].mean()), 4) for v in cfg["vitals"]},
        # median among patients who had the vital at all, so missingness is not counted twice
        "median_count": {v: float(train.loc[train[f"{v}_count"] > 0, f"{v}_count"].median())
                         for v in cfg["vitals"]},
        "mean_score": val["mean_predicted"],
        "alert_rate": val["alert_rate"],
    }
    (model_dir / "monitoring_baseline.json").write_text(json.dumps(base, indent=2))
    return base


def read_log(path: Path) -> tuple[pd.DataFrame, int]:
    """Request records from a JSONL log. Returns (records, number of skipped lines).

    Skips blank lines, non-JSON lines (e.g. server start-up messages in `docker logs`
    output) and JSON lines that are not request records.
    """
    rows, skipped = [], 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(rec, dict) and "path" in rec and "status" in rec:
            rows.append(rec)
        else:
            skipped += 1
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["path", "status"])
    return df, skipped


def check(log: pd.DataFrame, base: dict, cfg: dict, th: dict) -> dict:
    calls = log[log["path"] == "/predict"].tail(th["window"])
    ok = calls[calls["status"] == 200]
    report = {"model_version": base["model_version"], "n_calls": int(len(calls)),
              "n_predictions": int(len(ok)), "alerts": [], "checks": {}}

    if len(calls):
        r5 = float((calls["status"] >= 500).mean())
        r422 = float((calls["status"] == 422).mean())
        report["checks"]["error_5xx_rate"] = round(r5, 4)
        report["checks"]["error_422_rate"] = round(r422, 4)
        if r5 > th["max_5xx_rate"]:
            report["alerts"].append(f"5xx rate {r5:.1%} > {th['max_5xx_rate']:.0%}")
        if r422 > th["max_422_rate"]:
            report["alerts"].append(f"422 rate {r422:.1%}: a client may be sending a broken format")

    if len(ok) < th["min_requests"]:
        report["status"] = "insufficient_traffic"
        report["note"] = f"need {th['min_requests']} predictions for drift checks, have {len(ok)}"
        return report

    counts = pd.DataFrame(ok["vital_counts"].tolist())
    for v in cfg["vitals"]:
        live = float(ok["missing_vitals"].apply(lambda m: v in m).mean())
        ref = base["missing_rate"][v]
        report["checks"][f"{v}_missing_rate"] = {"live": round(live, 4), "train": ref}
        if abs(live - ref) > th["max_missing_shift"]:
            report["alerts"].append(f"{v} missing rate {live:.1%} vs train {ref:.1%}")

        measured = counts.loc[counts[v] > 0, v]
        if len(measured):
            ratio = float(measured.median()) / base["median_count"][v]
            report["checks"][f"{v}_count_ratio"] = round(ratio, 3)
            lo, hi = th["count_ratio_band"]
            if not lo <= ratio <= hi:
                report["alerts"].append(f"{v} median count is {ratio:.2f}x train: charting frequency changed")

    for key, col, limit in [("mean_score", "probability", "max_score_shift"),
                            ("alert_rate", "high_risk", "max_alert_shift")]:
        live = float(ok[col].astype(float).mean())
        report["checks"][key] = {"live": round(live, 4), "val": base[key]}
        if abs(live - base[key]) > th[limit]:
            report["alerts"].append(f"{key} {live:.3f} vs val {base[key]:.3f}")

    report["status"] = "ALERT" if report["alerts"] else "ok"
    return report


def main() -> None:
    cfg = load_config()
    th = {**DEFAULTS, **cfg.get("monitoring", {})}
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(p(f"{cfg['model']['output_dir']}/{cfg['model']['version']}")))
    ap.add_argument("--log", default=str(p("logs/requests.jsonl")))
    ap.add_argument("--make-baseline", action="store_true")
    args = ap.parse_args()
    model_dir = Path(args.model_dir)

    if args.make_baseline:
        print(json.dumps(make_baseline(cfg, model_dir), indent=2))
        return

    base_path = model_dir / "monitoring_baseline.json"
    if not base_path.exists():
        print(f"[monitor] {base_path} not found. Run: python -m src.monitor --make-baseline",
              file=sys.stderr)
        sys.exit(2)
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[monitor] log file {log_path} not found. Start the API with LOG_PATH set "
              "(see README, section Monitoring) or pass --log.", file=sys.stderr)
        sys.exit(2)
    log, skipped = read_log(log_path)
    report = check(log, json.loads(base_path.read_text()), cfg, th)
    report["skipped_lines"] = skipped
    out = p(cfg["data"]["reports_dir"]) / "monitor_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] == "ALERT":
        sys.exit(1)


if __name__ == "__main__":
    main()
