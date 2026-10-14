import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import p
from src.evaluate import choose_threshold, mask_vitals, threshold_metrics
from src.models import build_pipeline
from src.train import load_day1, run_day2
from tests.conftest import make_trained


# ---- pure functions -------------------------------------------------------------

def test_threshold_is_highest_that_meets_sensitivity():
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    p_ = np.array([.9, .8, .7, .6, .2, .65, .5, .4, .3, .1])
    thr = choose_threshold(y, p_, 0.80)
    assert thr == 0.6                              # catches 4/5; 0.7 would catch only 3/5
    assert threshold_metrics(y, p_, thr)["sensitivity"] == 0.8


def test_threshold_fails_loudly_if_target_unreachable():
    with pytest.raises(ValueError):
        choose_threshold(np.array([0, 1]), np.array([0.5, 0.5]), 1.1)


def test_mask_vitals_matches_feature_encoding():
    X = pd.DataFrame({"HR_count": [5], "HR_mean": [80.0], "HR_last": [82.0], "HR_missing": [0]})
    m = mask_vitals(X, ["HR"]).iloc[0]
    assert m["HR_count"] == 0 and m["HR_missing"] == 1
    assert np.isnan(m["HR_mean"]) and np.isnan(m["HR_last"])
    assert X.iloc[0]["HR_count"] == 5              # original untouched


# ---- end to end on synthetic data ---------------------------------------------------

@pytest.fixture
def trained(tmp_path):
    cfg = make_trained(tmp_path)
    return cfg, cfg.pop("_result")


def test_imputer_learns_from_train_only(trained):
    cfg, _ = trained
    X, parts = load_day1(cfg)
    cols = list(X.columns)
    Xtr, ytr = parts["train"]
    model = build_pipeline("logreg", {"C": 1.0}, cols, 0).fit(Xtr[cols], ytr)
    num = model.named_steps["pre"].named_transformers_["num"]
    cont = model.named_steps["pre"].transformers_[0][2]
    learned = num.named_steps["impute"].statistics_
    assert np.allclose(learned, Xtr[cont].median().values, equal_nan=True)
    assert not np.allclose(learned, X[cont].median().values, equal_nan=True)


def test_artifacts_saved_and_loadable(trained):
    cfg, res = trained
    out = p(cfg["model"]["output_dir"]) / cfg["model"]["version"]
    for f in ["model.joblib", "metadata.json", "metrics.json", "test_predictions.csv"]:
        assert (out / f).exists(), f

    meta = json.loads((out / "metadata.json").read_text())
    for k in ["data_version", "threshold", "feature_columns", "seed", "hyperparameters", "versions"]:
        assert k in meta
    assert not set(meta["feature_columns"]) & set(cfg["outcome_cols"])

    X, parts = load_day1(cfg)
    model = joblib.load(out / "model.joblib")
    prob = model.predict_proba(parts["test"][0][meta["feature_columns"]])[:, 1]
    saved = pd.read_csv(out / "test_predictions.csv")["p"].values
    assert np.allclose(prob, saved)                # reloaded model reproduces saved scores


def test_refuses_to_overwrite_a_version(trained):
    cfg, _ = trained
    with pytest.raises(FileExistsError):
        run_day2(cfg)


def test_ablation_removes_every_resprate_column(trained):
    _, res = trained
    a = res["metrics"]["resprate_ablation"]
    assert set(a["dropped"]) == {"RespRate_count", "RespRate_mean", "RespRate_last", "RespRate_missing"}


def test_threshold_chosen_on_val_not_test(trained):
    cfg, res = trained
    X, parts = load_day1(cfg)
    cols = res["metadata"]["feature_columns"]
    model = joblib.load(p(cfg["model"]["output_dir"]) / cfg["model"]["version"] / "model.joblib")
    pv = model.predict_proba(parts["val"][0][cols])[:, 1]
    expected = choose_threshold(parts["val"][1], pv, cfg["model"]["min_sensitivity"])
    assert res["metadata"]["threshold"] == expected        # stored unrounded (B13)


def test_ablation_subsets(trained):
    _, res = trained
    ab = res["metrics"]["ablations"]
    assert set(ab) == {"full", "no_resprate", "no_counts", "admission_only"}
    assert all(c.endswith("_count") for c in ab["no_counts"]["dropped"])
    assert ab["admission_only"]["n_features"] == 10      # Age, Age_missing, 3 Gender, 5 ICUType


def test_operating_points_are_ordered(trained):
    _, res = trained
    ops = res["metrics"]["operating_points"]
    thresholds = [o["threshold"] for o in ops]
    alerts = [o["val_alert_rate"] for o in ops]
    assert thresholds == sorted(thresholds, reverse=True)  # higher target -> lower cutoff
    assert alerts == sorted(alerts)                        # -> more alerts


def test_training_reads_features_bit_for_bit(trained):
    """B6: the CSV round trip must not change any float. A changed last digit can move a
    tree split and make API (in-memory features) and offline scores disagree."""
    from src.features import build_features
    from src.parse import load_outcomes, load_records
    from src.validate import validate

    cfg, _ = trained
    d = cfg["data"]
    adm, obs = load_records(p(d["records_dir"]))
    labels = load_outcomes(p(d["outcomes_file"]), cfg["id_col"], cfg["target"])
    adm, vit, _ = validate(adm, obs, labels, cfg)
    in_memory = build_features(adm, vit, cfg)

    loaded, _ = load_day1(cfg)
    in_memory.index = in_memory.index.astype("int64")
    pd.testing.assert_frame_equal(loaded.sort_index(), in_memory.sort_index(),
                                  check_exact=True, check_dtype=False)


# ---- review fixes ----------------------------------------------------------------------

def test_threshold_considers_every_cutoff():
    """B14: with roc_curve's default drop_intermediate=True, the 0.8 cutoff below sits on a
    straight vertical run of the ROC curve, is dropped, and 0.593 is returned instead."""
    y = np.array([1, 1, 1, 0, 0])
    p_ = np.array([.9, .8, .593, .389, .1])
    assert choose_threshold(y, p_, 0.6) == 0.8


def test_metadata_records_code_config_and_split_versions(trained):
    cfg, res = trained
    meta = res["metadata"]
    assert meta["git_commit"]                               # value depends on the checkout
    assert len(meta["config_sha256"]) == 64
    assert set(meta["splits_sha256"]) == {"train", "val", "test"}
    assert set(meta["train_median_counts"]) == set(cfg["vitals"])
    assert isinstance(meta["threshold"], float)


def test_masking_also_reported_on_validation(trained):
    _, res = trained
    mv = res["metrics"]["masking_val"]
    assert set(mv) == {"none", "HR", "RespRate", "Temp", "all_vitals"}
    assert mv["none"]["n"] == res["metadata"]["split_sizes"]["val"]


def test_refuses_to_train_from_uncommitted_code(synthetic_dataset, monkeypatch, tmp_path):
    from src import train as train_mod
    from src.run_day1 import run_day1
    cfg = synthetic_dataset
    cfg["split"]["max_rate_gap"] = 0.05
    cfg["model"]["output_dir"] = str(tmp_path / "models")
    run_day1(cfg)
    for commit in ["unknown", "abc123-dirty"]:
        monkeypatch.setattr(train_mod, "git_commit", lambda c=commit: c)
        with pytest.raises(RuntimeError, match="git_commit"):
            run_day2(cfg)
    assert not (tmp_path / "models").exists()               # failed before any training


def test_selection_metric_from_config_is_used(tmp_path):
    cfg = make_trained(tmp_path, selection_metric="brier")
    res = cfg.pop("_result")
    assert res["metadata"]["selection_metric"] == "brier"
    sel = res["metrics"]["selection"]
    assert all("val_brier" in r and "val_pr_auc" not in r for r in sel.values())
    briers = {f: r["val_brier"] for f, r in sel.items()}
    assert res["metrics"]["winner"] == min(briers, key=briers.get)   # lower Brier wins


@pytest.mark.parametrize("key,value", [("selection_metric", "accuracy"),
                                       ("threshold_rule", "youden")])
def test_unknown_settings_rejected(synthetic_dataset, key, value):
    cfg = synthetic_dataset
    cfg["model"][key] = value
    with pytest.raises(ValueError, match=key):
        run_day2(cfg, allow_dirty=True)
