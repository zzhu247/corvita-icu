import json
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.config import p
from src.parse import parse_record
from tests.conftest import make_trained


def _client(cfg, log_path=None):
    model_dir = p(cfg["model"]["output_dir"]) / cfg["model"]["version"]
    return model_dir, TestClient(create_app(str(model_dir), str(log_path) if log_path else None))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    cfg = make_trained(tmp)
    cfg.pop("_result")
    log_path = tmp / "logs" / "requests.jsonl"
    model_dir, client = _client(cfg, log_path)
    return {"cfg": cfg, "model_dir": model_dir, "log": log_path, "client": client}


@pytest.fixture(scope="module")
def env_hgb(tmp_path_factory):
    """Tree model only: its split thresholds come from training values, so it is the
    model most sensitive to any feature difference between training and serving."""
    tmp = tmp_path_factory.mktemp("api_hgb")
    cfg = make_trained(tmp, grids={"hgb": {"learning_rate": [0.1], "max_depth": [3],
                                           "min_samples_leaf": [5]}})
    cfg.pop("_result")
    model_dir, client = _client(cfg)
    return {"cfg": cfg, "model_dir": model_dir, "client": client}


def body(**over):
    """A typical request: 8 readings per vital in the first 24 h."""
    obs = []
    for h in range(1, 9):
        obs += [{"time": f"{h:02d}:00", "parameter": "HR", "value": 88 + h},
                {"time": f"{h:02d}:15", "parameter": "RespRate", "value": 18 + h % 3},
                {"time": f"{h:02d}:30", "parameter": "Temp", "value": 37.8}]
    b = {"record_id": "demo-1", "age": 67, "gender": 1, "icu_type": 3, "observations": obs}
    b.update(over)
    return b


def without(vital):
    return [o for o in body()["observations"] if o["parameter"] != vital]


def request_from_record(path, cfg):
    """Raw PhysioNet record file -> API request, vitals only, all times."""
    desc, obs = parse_record(path)
    obs = obs[obs["Parameter"].isin(cfg["vitals"])]
    gender = desc.get("Gender", -1)
    return {"age": float(desc["Age"]), "icu_type": int(desc["ICUType"]),
            "gender": None if pd.isna(gender) or gender == -1 else int(gender),
            "observations": [{"time": t, "parameter": prm, "value": float(v)}
                             for t, prm, v in obs[["Time", "Parameter", "Value"]].itertuples(index=False)]}


# ---- valid requests ---------------------------------------------------------------

def test_health(env):
    r = env["client"].get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_valid_request(env):
    r = env["client"].post("/predict", json=body())
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["probability"] <= 1
    assert j["high_risk"] == (j["probability"] >= j["threshold"])
    assert j["data_quality_flag"] == "ok" and j["missing_vitals"] == [] and j["warnings"] == []
    assert j["low_count_vitals"] == []
    assert j["model_version"] == "v1" and j["data_version"] and j["record_id"] == "demo-1"
    assert r.headers["x-request-id"] == j["request_id"]


def test_int_and_decimal_age_accepted(env):
    assert env["client"].post("/predict", json=body(age=60)).status_code == 200
    assert env["client"].post("/predict", json=body(age=60.5)).status_code == 200


# ---- training-serving parity ---------------------------------------------------------

@pytest.mark.parametrize("which", ["env", "env_hgb"])
def test_api_matches_offline_scores_for_every_test_patient(which, request):
    """Every test-set patient, raw file -> API, must get the score saved at training time
    (test_predictions.csv). The API rounds to 4 decimals, so the bound is 1e-4."""
    e = request.getfixturevalue(which)
    cfg = e["cfg"]
    saved = pd.read_csv(e["model_dir"] / "test_predictions.csv", index_col="RecordID")["p"]
    rec_dir = p(cfg["data"]["records_dir"])
    worst = 0.0
    for rid, offline in saved.items():
        online = e["client"].post("/predict", json=request_from_record(rec_dir / f"{rid}.txt", cfg))
        assert online.status_code == 200, online.text
        worst = max(worst, abs(online.json()["probability"] - offline))
    assert worst < 1e-4, f"largest API vs offline gap {worst}"


# ---- 24 h window and cleaning --------------------------------------------------------

def test_cutoff_in_api(env):
    c = env["client"]
    base = c.post("/predict", json=body()).json()
    obs = body()["observations"] + [{"time": "24:00", "parameter": "HR", "value": 92}]
    with_2400 = c.post("/predict", json=body(observations=obs)).json()
    assert with_2400["warnings"] == []                       # 24:00 is inside the window
    obs_late = body()["observations"] + [{"time": "24:01", "parameter": "HR", "value": 250},
                                         {"time": "30:00", "parameter": "Temp", "value": 41}]
    late = c.post("/predict", json=body(observations=obs_late)).json()
    assert late["probability"] == base["probability"]        # late values change nothing
    assert any("after 24:00" in w for w in late["warnings"])


def test_implausible_value_ignored_with_warning(env):
    obs = body()["observations"] + [{"time": "04:00", "parameter": "HR", "value": 0}]
    j = env["client"].post("/predict", json=body(observations=obs)).json()
    assert any("implausible" in w for w in j["warnings"])
    assert j["probability"] == env["client"].post("/predict", json=body()).json()["probability"]


# ---- missing and sparse vitals ------------------------------------------------------------

def test_missing_resprate(env):
    j = env["client"].post("/predict", json=body(observations=without("RespRate"))).json()
    assert j["data_quality_flag"] == "missing_vitals" and j["missing_vitals"] == ["RespRate"]
    assert isinstance(j["high_risk"], bool)


def test_missing_hr_gets_specific_warning(env):
    j = env["client"].post("/predict", json=body(observations=without("HR"))).json()
    assert any(w.startswith("HR missing") and "validation" in w for w in j["warnings"])


def test_no_vitals_scored_but_never_alerts(env):
    """insufficient_data: score from age/sex/ICU type, but high_risk is null, for every combination."""
    for age in (20, 60, 90):
        for icu in (1, 2, 3, 4):
            for gender in (0, 1, None):
                r = env["client"].post("/predict", json=body(observations=[], age=age,
                                                             icu_type=icu, gender=gender))
                assert r.status_code == 200
                j = r.json()
                assert j["data_quality_flag"] == "insufficient_data"
                assert j["high_risk"] is None
                assert 0 <= j["probability"] <= 1


def test_sparse_readings_flagged_low_count(env):
    obs = [{"time": "01:00", "parameter": "HR", "value": 90},
           {"time": "01:00", "parameter": "RespRate", "value": 18},
           {"time": "01:00", "parameter": "Temp", "value": 37.0}]
    j = env["client"].post("/predict", json=body(observations=obs)).json()
    assert j["data_quality_flag"] == "low_count"
    assert set(j["low_count_vitals"]) == {"HR", "RespRate", "Temp"}
    assert any("send every charted value" in w for w in j["warnings"])


def test_older_model_without_median_counts_skips_low_count(env, tmp_path):
    d = tmp_path / "v1"
    shutil.copytree(env["model_dir"], d)
    meta = json.loads((d / "metadata.json").read_text())
    meta.pop("train_median_counts")
    (d / "metadata.json").write_text(json.dumps(meta))
    client = TestClient(create_app(str(d)))
    obs = [{"time": "01:00", "parameter": v, "value": x} for v, x in
           [("HR", 90), ("RespRate", 18), ("Temp", 37.0)]]
    assert client.post("/predict", json=body(observations=obs)).json()["data_quality_flag"] == "ok"


# ---- invalid requests -> 422 ----------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"age": 5},                                   # below adult training range
    {"age": "old"},
    {"age": "60"},                                # numeric string: strict types
    {"icu_type": 7},
    {"icu_type": 3.0},                            # float for an integer code
    {"icu_type": "3"},
    {"gender": 2},
    {"gender": True},                             # bool is not 0/1
    {"unexpected_field": 1},
    {"observations": [{"time": "10:75", "parameter": "HR", "value": 80}]},
    {"observations": [{"time": "1030", "parameter": "HR", "value": 80}]},
    {"observations": [{"time": "01:00", "parameter": "SpO2", "value": 97}]},
    {"observations": [{"time": "01:00", "parameter": "HR", "value": "fast"}]},
    {"observations": [{"time": "01:00", "parameter": "HR", "value": "80"}]},
    {"observations": [{"time": "01:00", "parameter": "HR", "value": 80, "unit": "bpm"}]},
])
def test_invalid_requests_rejected(env, bad):
    r = env["client"].post("/predict", json=body(**bad))
    assert r.status_code == 422, r.text


def test_non_finite_value_rejected(env):
    raw = json.dumps(body()).replace('"value": 37.8', '"value": NaN', 1)
    r = env["client"].post("/predict", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert "NaN" not in r.text                     # error body never echoes submitted values


def test_validation_error_does_not_echo_input(env):
    r = env["client"].post("/predict", json=body(record_id="SECRET-ID-123", icu_type=9))
    assert r.status_code == 422 and "SECRET-ID-123" not in r.text
    assert r.json()["detail"][0]["loc"] == ["body", "icu_type"]


def test_missing_required_field(env):
    b = body()
    del b["age"]
    assert env["client"].post("/predict", json=b).status_code == 422


# ---- logging ---------------------------------------------------------------------------

def test_log_lines_have_fields_and_no_patient_data(env):
    env["client"].post("/predict", json=body(record_id="SECRET-ID-123"))
    env["client"].post("/predict", json=body(icu_type=9))
    lines = [json.loads(x) for x in env["log"].read_text().splitlines()]
    ok = [x for x in lines if x["path"] == "/predict" and x["status"] == 200][-1]
    bad = [x for x in lines if x["status"] == 422][-1]
    for k in ["request_id", "latency_ms", "model_version", "data_quality_flag", "vital_counts", "probability"]:
        assert k in ok
    assert bad["error"] == "validation" and bad["fields"] == ["icu_type"]
    raw = env["log"].read_text()
    assert "SECRET-ID-123" not in raw and "37.8" not in raw and '"age":' not in raw


def test_unwritable_log_path_does_not_break_requests(env, tmp_path):
    """A log folder that cannot be created (here its parent is a file) must not fail requests."""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    model_dir, client = _client(env["cfg"], blocker / "logs" / "requests.jsonl")
    assert client.get("/health").status_code == 200
    r = client.post("/predict", json=body())
    assert r.status_code == 200 and r.json()["request_id"]


def test_log_write_failure_mid_run_does_not_break_requests(env, tmp_path):
    """The log path exists but cannot be appended to (it is a folder)."""
    target = tmp_path / "requests.jsonl"
    target.mkdir()
    model_dir, client = _client(env["cfg"], target)
    for _ in range(2):
        assert client.post("/predict", json=body()).status_code == 200


# ---- startup checks -----------------------------------------------------------------------

def test_refuses_model_with_different_cutoff(env, tmp_path):
    d = tmp_path / "v1"
    shutil.copytree(env["model_dir"], d)
    meta = json.loads((d / "metadata.json").read_text())
    meta["cutoff_minutes"] = 720
    (d / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="cutoff"):
        create_app(str(d))
