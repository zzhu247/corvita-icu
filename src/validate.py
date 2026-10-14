"""Step 3: check fields, units, values and times. Clean what can be cleaned, stop on what cannot.

Hard errors (raise): duplicate RecordIDs, records without an outcome, labels not in {0, 1}.
Soft issues (clean + count in the report): bad time strings, out-of-range values,
unknown categories, file name / RecordID mismatch.

Cleaning rule: an implausible value is treated as missing (NaN) and dropped from
the observation table, so it never counts as a measurement. Every drop is counted.
"""
import numpy as np
import pandas as pd


class DataError(Exception):
    pass


def out_of_range(vit: pd.DataFrame, cfg: dict) -> pd.Series:
    """True for vital rows whose value is missing or outside config valid_ranges.
    Used by training (validate) and by the API, so both clean values the same way."""
    bad = pd.Series(False, index=vit.index)
    for v in cfg["vitals"]:
        lo, hi = cfg["valid_ranges"][v]
        rows = vit["Parameter"] == v
        bad |= rows & (vit["Value"].isna() | (vit["Value"] < lo) | (vit["Value"] > hi))
    return bad


def validate(adm: pd.DataFrame, obs: pd.DataFrame, labels: pd.DataFrame, cfg: dict):
    id_col, target = cfg["id_col"], cfg["target"]
    rep: dict = {"n_records": int(len(adm)), "errors": [], "warnings": []}

    # ---- IDs and labels -------------------------------------------------
    dup = adm[id_col][adm[id_col].duplicated()].tolist()
    if dup:
        raise DataError(f"Duplicate RecordIDs in record files: {dup[:10]}")

    mismatch = adm[adm["file_id"] != adm[id_col].astype(str)]
    rep["file_id_mismatch"] = int(len(mismatch))
    if len(mismatch):
        rep["warnings"].append(f"{len(mismatch)} files whose name differs from their RecordID")

    no_label = set(adm[id_col]) - set(labels[id_col])
    if no_label:
        raise DataError(f"{len(no_label)} records have no outcome, e.g. {sorted(no_label)[:5]}")
    orphans = set(labels[id_col]) - set(adm[id_col])
    rep["outcomes_without_record"] = int(len(orphans))
    if orphans:
        rep["warnings"].append(f"{len(orphans)} outcome rows have no record file and are ignored")
    labels = labels[labels[id_col].isin(adm[id_col])]

    bad_labels = set(labels[target].dropna().unique()) - {0, 1}
    if bad_labels or labels[target].isna().any():
        raise DataError(f"Label {target} must be 0/1, found {bad_labels or 'NaN'}")

    # ---- Times ------------------------------------------------------------
    bad_time = obs["minutes"].isna()
    rep["time"] = {
        "bad_format_rows": int(bad_time.sum()),
        "max_minutes": float(obs["minutes"].max()),
        "rows_after_24h": int((obs["minutes"] > cfg["window"]["cutoff_minutes"]).sum()),
    }
    obs = obs[~bad_time]

    # ---- Fields -------------------------------------------------------------
    rep["parameters_seen"] = sorted(obs["Parameter"].dropna().unique().tolist())
    missing_vitals = [v for v in cfg["vitals"] if v not in rep["parameters_seen"]]
    if missing_vitals:
        rep["warnings"].append(f"Vitals never recorded in any file: {missing_vitals}")

    # ---- Vital values and units -----------------------------------------------
    rep["vitals"] = {}
    vit = obs[obs["Parameter"].isin(cfg["vitals"])].copy()
    bad = out_of_range(vit, cfg)
    for v in cfg["vitals"]:
        rows = vit["Parameter"] == v
        vals = vit.loc[rows, "Value"]
        out = rows & bad
        info = {
            "n_rows": int(rows.sum()),
            "n_out_of_range": int(out.sum()),
            "raw_min": float(vals.min()) if len(vals) else None,
            "raw_max": float(vals.max()) if len(vals) else None,
        }
        if v == "Temp":
            # Unit check: values in 80-110 look like Fahrenheit, not Celsius.
            info["n_looks_fahrenheit"] = int(vals.between(80, 110).sum())
        rep["vitals"][v] = info
    vit = vit[~bad]

    # ---- Admission fields ----------------------------------------------------------
    adm = adm.copy()
    lo, hi = cfg["valid_ranges"]["Age"]
    age_bad = adm["Age"].notna() & ~adm["Age"].between(lo, hi)
    adm.loc[age_bad, "Age"] = np.nan
    rep["admission"] = {"age_out_of_range": int(age_bad.sum())}
    for c in ["Gender", "ICUType"]:
        bad = adm[c].notna() & ~adm[c].isin(cfg["categories"][c])
        adm.loc[bad, c] = np.nan
        rep["admission"][f"{c.lower()}_unknown_category"] = int(bad.sum())
    rep["admission"]["missing"] = {c: int(adm[c].isna().sum()) for c in cfg["admission_fields"]}

    # ---- Label balance ------------------------------------------------------------------
    merged = adm[[id_col]].merge(labels, on=id_col, how="left")
    rep["label"] = {
        "n_positive": int(merged[target].sum()),
        "positive_rate": round(float(merged[target].mean()), 4),
    }

    return adm.drop(columns="file_id"), vit, rep
