import numpy as np
import pandas as pd

from src.features import apply_cutoff, build_features, feature_columns
from src.parse import load_records, time_to_minutes
from tests.conftest import write_record


def _one_patient(tmp_path, rows, **kw):
    d = tmp_path / "recs"
    d.mkdir()
    write_record(d / "1.txt", 1, rows=rows, **kw)
    return load_records(d)


# ---- 24 h cutoff ---------------------------------------------------------------

def test_cutoff_keeps_2400_drops_2401():
    obs = pd.DataFrame({"minutes": [0, 1439, 1440, 1441, 2000],
                        "Parameter": ["HR"] * 5, "Value": [1, 2, 3, 4, 5]})
    kept = apply_cutoff(obs, 1440)
    assert kept["minutes"].tolist() == [0, 1439, 1440]


def test_later_data_never_reaches_features(tmp_path, cfg):
    rows = [("10:00", "HR", 80), ("24:00", "HR", 100), ("24:01", "HR", 999), ("30:00", "HR", 999)]
    adm, obs = _one_patient(tmp_path, rows)
    X = build_features(adm, obs, cfg)
    assert X.loc[1, "HR_count"] == 2
    assert X.loc[1, "HR_mean"] == 90
    assert X.loc[1, "HR_last"] == 100          # 24:00 is the last value inside the window


# ---- vital features ----------------------------------------------------------------

def test_missing_vital_gets_flag_and_nan(tmp_path, cfg):
    adm, obs = _one_patient(tmp_path, [("01:00", "HR", 70)])
    X = build_features(adm, obs, cfg)
    assert X.loc[1, "RespRate_count"] == 0
    assert X.loc[1, "RespRate_missing"] == 1
    assert np.isnan(X.loc[1, "RespRate_mean"]) and np.isnan(X.loc[1, "RespRate_last"])
    assert X.loc[1, "HR_missing"] == 0


def test_last_uses_time_not_file_order(tmp_path, cfg):
    rows = [("05:00", "Temp", 38.0), ("02:00", "Temp", 36.5)]   # out of order in the file
    adm, obs = _one_patient(tmp_path, rows)
    X = build_features(adm, obs, cfg)
    assert X.loc[1, "Temp_last"] == 38.0


def test_patient_with_no_vitals_still_has_a_row(tmp_path, cfg):
    adm, obs = _one_patient(tmp_path, [])
    X = build_features(adm, obs, cfg)
    assert len(X) == 1
    assert X.loc[1, ["HR_missing", "RespRate_missing", "Temp_missing"]].tolist() == [1, 1, 1]


# ---- admission features -------------------------------------------------------------

def test_categories_one_hot_and_missing(tmp_path, cfg):
    adm, obs = _one_patient(tmp_path, [], gender=-1, icu=2)
    X = build_features(adm, obs, cfg)
    row = X.loc[1]
    assert (row["Gender_female"], row["Gender_male"], row["Gender_missing"]) == (0, 0, 1)
    assert row["ICUType_2"] == 1 and row[["ICUType_1", "ICUType_3", "ICUType_4"]].sum() == 0
    assert list(X.columns) == feature_columns(cfg)


def test_descriptor_minus_one_is_missing(tmp_path):
    adm, _ = _one_patient(tmp_path, [], age=-1)
    assert adm["Age"].isna().all()


def test_time_parsing():
    t = pd.Series(["00:00", "24:00", "47:59", "ab:cd", "10:75"])
    m = time_to_minutes(t)
    assert m[:3].tolist() == [0, 1440, 2879]
    assert m[3:].isna().all()


def test_no_outcome_fields_in_inputs(cfg):
    assert not set(feature_columns(cfg)) & set(cfg["outcome_cols"])
