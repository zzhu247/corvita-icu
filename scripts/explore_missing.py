"""Exploration used for the Day 1 findings on missing vitals. Not part of the pipeline.

    python -m scripts.explore_missing

Needs the outputs of `make day1` and the raw files in data/raw/set-a.
MechVent is read here only to explain RespRate missingness. It is never a model input.
"""
import json

import pandas as pd

from src.config import load_config, p
from src.features import apply_cutoff
from src.parse import load_records


def main() -> None:
    cfg = load_config()
    d = cfg["data"]
    X = pd.read_csv(p(d["processed_dir"]) / "features.csv", index_col="RecordID",
                    float_precision="round_trip")
    y = pd.read_csv(p(d["processed_dir"]) / "labels.csv", index_col="RecordID")[cfg["target"]]
    df = X.join(y)

    rep = json.loads((p(d["reports_dir"]) / "data_quality.json").read_text())
    print(json.dumps({k: rep[k] for k in ["time", "vitals", "admission"]}, indent=2))

    icu_cols = [f"ICUType_{k}" for k in cfg["categories"]["ICUType"]] + ["ICUType_missing"]
    icu = df[icu_cols].idxmax(axis=1)
    print("\nRespRate missing rate by ICU type:")
    print(df.groupby(icu)["RespRate_missing"].mean().round(3))

    print("\nDeath rate by RespRate_missing:")
    print(df.groupby("RespRate_missing")[cfg["target"]].agg(["size", "mean"]).round(3))

    print("\nDeath rate by all-vitals-missing (HR_missing):")
    print(df.groupby("HR_missing")[cfg["target"]].agg(["size", "mean"]).round(3))

    _, obs = load_records(p(d["records_dir"]))
    w = apply_cutoff(obs, cfg["window"]["cutoff_minutes"])
    vent = set(w.loc[(w["Parameter"] == "MechVent") & (w["Value"] == 1), "RecordID"])
    df["vent_24h"] = df.index.isin(vent).astype(int)

    print("\nRespRate missing rate by ventilation (exploration only, not a feature):")
    print(df.groupby("vent_24h")["RespRate_missing"].agg(["size", "mean"]).round(3))


if __name__ == "__main__":
    main()
