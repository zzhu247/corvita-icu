import json

import pandas as pd
import pytest

from src.config import p
from src.ingest import fetch, parse_sha256sums, upstream_check, verify_or_create
from src.parse import load_outcomes, load_records
from src.run_day1 import run_day1
from src.split import check_splits, load_splits, make_splits
from src.validate import DataError, validate


# ---- checksums -----------------------------------------------------------------

def test_manifest_detects_changed_source(synthetic_dataset):
    cfg = synthetic_dataset
    first = verify_or_create(cfg)
    assert verify_or_create(cfg)["data_version"] == first["data_version"]

    f = sorted(p(cfg["data"]["records_dir"]).glob("*.txt"))[0]
    f.write_text(f.read_text() + "01:00,HR,90\n")
    with pytest.raises(RuntimeError, match="does not match manifest"):
        verify_or_create(cfg)


# ---- validation ------------------------------------------------------------------

def _load(cfg):
    adm, obs = load_records(p(cfg["data"]["records_dir"]))
    labels = load_outcomes(p(cfg["data"]["outcomes_file"]), cfg["id_col"], cfg["target"])
    return adm, obs, labels


def test_out_of_range_values_are_removed_and_counted(synthetic_dataset):
    cfg = synthetic_dataset
    adm, obs, labels = _load(cfg)
    rid = adm["RecordID"].iloc[0]
    bad = pd.DataFrame({"RecordID": [rid] * 2, "Time": ["01:00", "02:00"],
                        "minutes": [60, 120], "Parameter": ["HR", "Temp"], "Value": [0, 98.6]})
    obs = pd.concat([obs, bad.astype(obs.dtypes.to_dict())], ignore_index=True)
    _, vit, rep = validate(adm, obs, labels, cfg)
    assert rep["vitals"]["HR"]["n_out_of_range"] >= 1
    assert rep["vitals"]["Temp"]["n_looks_fahrenheit"] >= 1
    assert not ((vit["Parameter"] == "Temp") & (vit["Value"] > 45)).any()


def test_record_without_outcome_is_an_error(synthetic_dataset):
    adm, obs, labels = _load(synthetic_dataset)
    with pytest.raises(DataError, match="no outcome"):
        validate(adm, obs, labels.iloc[1:], synthetic_dataset)


def test_duplicate_record_id_is_an_error(synthetic_dataset):
    adm, obs, labels = _load(synthetic_dataset)
    with pytest.raises(DataError, match="Duplicate"):
        validate(pd.concat([adm, adm.iloc[:1]]), obs, labels, synthetic_dataset)


# ---- splits ------------------------------------------------------------------------

def test_splits_disjoint_complete_and_balanced(synthetic_dataset):
    cfg = synthetic_dataset
    cfg["split"]["max_rate_gap"] = 0.05        # small synthetic set: allow more noise
    _, _, labels = _load(cfg)
    splits = make_splits(labels, cfg)
    summary = check_splits(splits, labels, cfg)   # asserts no overlap, full cover, rates
    assert summary["train"]["n"] == 140 and summary["val"]["n"] == 30 and summary["test"]["n"] == 30


def test_splits_are_reproducible(synthetic_dataset):
    _, _, labels = _load(synthetic_dataset)
    a, b = make_splits(labels, synthetic_dataset), make_splits(labels, synthetic_dataset)
    assert all(a[k].equals(b[k]) for k in a)


# ---- end to end ----------------------------------------------------------------------

def test_run_day1_end_to_end(synthetic_dataset):
    cfg = synthetic_dataset
    cfg["split"]["max_rate_gap"] = 0.05
    rep = run_day1(cfg)

    X = pd.read_csv(p(cfg["data"]["processed_dir"]) / "features.csv", index_col="RecordID")
    assert len(X) == 200 and X.index.is_unique
    assert not set(X.columns) & set(cfg["outcome_cols"])

    splits = load_splits(p(cfg["data"]["splits_dir"]))
    assert sum(len(v) for v in splits.values()) == 200

    saved = json.loads((p(cfg["data"]["reports_dir"]) / "data_quality.json").read_text())
    assert saved["data_version"] == rep["data_version"]
    assert 0.5 < saved["missing_in_24h_window"]["RespRate"] < 0.8   # 2/3 have no RespRate


# ---- review fixes ----------------------------------------------------------------------

def test_outcome_without_record_is_reported(synthetic_dataset):
    """B12: the linkage check must see outcome rows before they are filtered."""
    cfg = synthetic_dataset
    cfg["split"]["max_rate_gap"] = 0.05
    out = p(cfg["data"]["outcomes_file"])
    out.write_text(out.read_text() + "999999,1,1,10,-1,0\n")
    rep = run_day1(cfg)
    assert rep["outcomes_without_record"] == 1
    assert any("no record file" in w for w in rep["warnings"])
    assert rep["n_records"] == 200


def test_overlapping_splits_raise_data_error(synthetic_dataset):
    """B15: a real exception, not an assert that python -O would remove."""
    _, _, labels = _load(synthetic_dataset)
    splits = make_splits(labels, synthetic_dataset)
    splits["val"] = splits["val"].append(splits["train"][:1])
    with pytest.raises(DataError, match="share RecordIDs"):
        check_splits(splits, labels, synthetic_dataset)


def test_download_is_atomic(tmp_path):
    """B16: complete download -> file appears; failed download -> no file, no .part left."""
    src = tmp_path / "remote.txt"
    src.write_text("hello")
    dest = tmp_path / "local" / "file.txt"
    dest.parent.mkdir()
    fetch(src.as_uri(), dest)
    assert dest.read_text() == "hello"

    missing = tmp_path / "local" / "other.txt"
    with pytest.raises(Exception):
        fetch((tmp_path / "does-not-exist").as_uri(), missing)
    assert not missing.exists()
    assert list((tmp_path / "local").glob("*.part")) == []


def test_parse_sha256sums():
    h1, h2 = "a" * 64, "B" * 64
    sums = parse_sha256sums(f"{h1}  set-a.tar.gz\n{h2} *Outcomes-a.txt\njunk line\n")
    assert sums == {"set-a.tar.gz": h1, "Outcomes-a.txt": "b" * 64}


def test_upstream_mismatch_is_an_error(synthetic_dataset, monkeypatch):
    import io
    from src import ingest
    wrong = "0" * 64
    body = f"{wrong}  set-a.tar.gz\n{wrong}  Outcomes-a.txt\n".encode()
    monkeypatch.setattr(ingest.urllib.request, "urlopen", lambda url, timeout=30: io.BytesIO(body))
    with pytest.raises(RuntimeError, match="does not match PhysioNet"):
        upstream_check(synthetic_dataset)


def test_upstream_unreachable_is_recorded_not_fatal(synthetic_dataset, monkeypatch):
    from src import ingest

    def offline(url, timeout=30):
        raise OSError("offline")
    monkeypatch.setattr(ingest.urllib.request, "urlopen", offline)
    m = verify_or_create(synthetic_dataset, check_upstream=True)
    assert m["upstream_check"]["checked"] is False
