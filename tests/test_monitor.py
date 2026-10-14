import pandas as pd
import pytest

import json

from src.monitor import DEFAULTS, check, main, read_log

CFG = {"vitals": ["HR", "RespRate", "Temp"]}
BASE = {"model_version": "v1", "missing_rate": {"HR": 0.02, "RespRate": 0.72, "Temp": 0.02},
        "median_count": {"HR": 50.0, "RespRate": 20.0, "Temp": 10.0},
        "mean_score": 0.14, "alert_rate": 0.47}


def make_log(n=200, missing=("RespRate",), counts=None, prob=0.14, high=None, status=200):
    counts = counts or {"HR": 50, "RespRate": 0, "Temp": 10}
    rows = []
    for i in range(n):
        rows.append({"path": "/predict", "status": status, "missing_vitals": list(missing) if i % 4 else [],
                     "vital_counts": counts, "probability": prob,
                     "high_risk": (i % 2 == 0) if high is None else high})
    return pd.DataFrame(rows)


def test_normal_traffic_no_alert():
    r = check(make_log(), BASE, CFG, DEFAULTS)
    assert r["status"] == "ok", r["alerts"]


def test_hr_feed_outage_alerts():
    r = check(make_log(missing=("HR", "RespRate")), BASE, CFG, DEFAULTS)
    assert r["status"] == "ALERT"
    assert any(a.startswith("HR missing rate") for a in r["alerts"])


def test_charting_frequency_change_alerts():
    r = check(make_log(counts={"HR": 600, "RespRate": 0, "Temp": 10}), BASE, CFG, DEFAULTS)
    assert any("HR median count" in a for a in r["alerts"])


def test_score_shift_alerts():
    r = check(make_log(prob=0.30, high=True), BASE, CFG, DEFAULTS)
    assert any(a.startswith("mean_score") for a in r["alerts"])
    assert any(a.startswith("alert_rate") for a in r["alerts"])


def test_low_traffic_does_not_alert_on_drift():
    r = check(make_log(n=10, missing=("HR",)), BASE, CFG, DEFAULTS)
    assert r["status"] == "insufficient_traffic"


def test_validation_error_spike_alerts():
    log = pd.concat([make_log(150), make_log(50, status=422)], ignore_index=True)
    r = check(log, BASE, CFG, DEFAULTS)
    assert any("422 rate" in a for a in r["alerts"])


# ---- reading real log files (B5) ------------------------------------------------------

def test_read_log_skips_non_json_lines(tmp_path):
    rec = {"path": "/predict", "status": 200, "missing_vitals": [], "vital_counts": {}}
    f = tmp_path / "docker.log"
    f.write_text("INFO:     Started server process [1]\n"
                 + json.dumps(rec) + "\n"
                 + 'INFO:     172.17.0.1:5000 - "POST /predict HTTP/1.1" 200 OK\n'
                 + "\n" + json.dumps({"unrelated": 1}) + "\n" + json.dumps(rec) + "\n")
    log, skipped = read_log(f)
    assert len(log) == 2 and skipped == 3


def test_empty_log_reports_insufficient_traffic(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    log, skipped = read_log(f)
    r = check(log, BASE, CFG, DEFAULTS)
    assert r["status"] == "insufficient_traffic" and r["n_calls"] == 0


def test_missing_log_file_exits_2(tmp_path, monkeypatch, capsys):
    (tmp_path / "monitoring_baseline.json").write_text(json.dumps(BASE))
    monkeypatch.setattr("sys.argv", ["monitor", "--model-dir", str(tmp_path),
                                     "--log", str(tmp_path / "nope.jsonl")])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 2
    assert "not found" in capsys.readouterr().err


def test_insufficient_data_predictions_do_not_break_alert_rate():
    log = make_log()
    log["high_risk"] = [None if i % 10 == 0 else bool(i % 2) for i in range(len(log))]
    r = check(log, BASE, CFG, DEFAULTS)
    assert r["checks"]["alert_rate"]["live"] == r["checks"]["alert_rate"]["live"]   # not NaN
