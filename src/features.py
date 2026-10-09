"""Step 4: 24 h cutoff + model inputs. The API will import this same file.

Per vital (HR, RespRate, Temp), from 00:00 to 24:00 inclusive:
    {v}_count    number of valid measurements (0 if none)
    {v}_mean     average (NaN if none)
    {v}_last     latest value by time (NaN if none)
    {v}_missing  1 if no measurement in the window

Admission fields:
    Age, Age_missing
    Gender_female, Gender_male, Gender_missing        (one-hot, fixed categories)
    ICUType_1..ICUType_4, ICUType_missing             (one-hot, fixed categories)

Nothing here is learned from data. Encodings use fixed category lists from config,
so they are identical for train, val, test and the API. Steps that ARE learned
(median imputation, scaling) belong in the Day 2 sklearn Pipeline, fit on train only.
"""
import pandas as pd


def apply_cutoff(obs: pd.DataFrame, cutoff_minutes: int) -> pd.DataFrame:
    """Keep 0 <= minutes <= cutoff. 24:00 (1440) is kept, 24:01 (1441) is dropped."""
    return obs[(obs["minutes"] >= 0) & (obs["minutes"] <= cutoff_minutes)]


def feature_columns(cfg: dict) -> list[str]:
    cols = []
    for v in cfg["vitals"]:
        cols += [f"{v}_count", f"{v}_mean", f"{v}_last", f"{v}_missing"]
    cols += ["Age", "Age_missing", "Gender_female", "Gender_male", "Gender_missing"]
    cols += [f"ICUType_{k}" for k in cfg["categories"]["ICUType"]] + ["ICUType_missing"]
    return cols


def vital_features(obs: pd.DataFrame, ids: pd.Index, vitals: list[str]) -> pd.DataFrame:
    obs = obs[obs["Parameter"].isin(vitals)]
    # Stable sort: if two rows share a time, the one later in the file counts as "last".
    obs = obs.sort_values(["RecordID", "Parameter", "minutes"], kind="stable")
    g = obs.groupby(["RecordID", "Parameter"])["Value"]
    agg = pd.DataFrame({"count": g.size(), "mean": g.mean(), "last": g.last()})
    wide = agg.unstack("Parameter")   # columns: (stat, vital)

    out = pd.DataFrame(index=ids)
    for v in vitals:
        for stat in ["count", "mean", "last"]:
            col = wide[(stat, v)] if (stat, v) in wide.columns else pd.Series(dtype=float)
            out[f"{v}_{stat}"] = col.reindex(ids)
        out[f"{v}_count"] = out[f"{v}_count"].fillna(0).astype(int)
        out[f"{v}_missing"] = (out[f"{v}_count"] == 0).astype(int)
    return out


def admission_features(adm: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    a = adm.set_index("RecordID")
    out = pd.DataFrame(index=a.index)
    out["Age"] = a["Age"]
    out["Age_missing"] = a["Age"].isna().astype(int)
    out["Gender_female"] = (a["Gender"] == 0).astype(int)
    out["Gender_male"] = (a["Gender"] == 1).astype(int)
    out["Gender_missing"] = a["Gender"].isna().astype(int)
    for k in cfg["categories"]["ICUType"]:
        out[f"ICUType_{k}"] = (a["ICUType"] == k).astype(int)
    out["ICUType_missing"] = a["ICUType"].isna().astype(int)
    return out


def build_features(adm: pd.DataFrame, obs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per RecordID, columns in feature_columns(cfg) order."""
    window = apply_cutoff(obs, cfg["window"]["cutoff_minutes"])
    adm_f = admission_features(adm, cfg)
    vit_f = vital_features(window, adm_f.index, cfg["vitals"])
    X = vit_f.join(adm_f)
    X.index.name = "RecordID"
    return X[feature_columns(cfg)]


def missingness_summary(X: pd.DataFrame, cfg: dict) -> dict:
    """Share of patients with no measurement in the 24 h window, per vital."""
    s = {v: round(float(X[f"{v}_missing"].mean()), 4) for v in cfg["vitals"]}
    flags = X[[f"{v}_missing" for v in cfg["vitals"]]]
    s["any_vital_missing"] = round(float(flags.any(axis=1).mean()), 4)
    s["all_vitals_missing"] = round(float(flags.all(axis=1).mean()), 4)
    return s
