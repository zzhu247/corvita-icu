"""Day 1 pipeline: checksums -> parse -> validate -> 24 h features -> split.

    python -m src.run_day1

Outputs
    data/processed/admission.csv      admission table (cleaned)
    data/processed/vitals_long.csv.gz vital observations, all times (cleaned)
    data/processed/features.csv       model inputs, one row per RecordID
    data/processed/labels.csv         RecordID + In-hospital_death only
    data/splits/{train,val,test}_ids.csv
    reports/data_quality.json         everything checked and counted along the way
"""
import argparse
import json

from src.config import load_config, p
from src.features import build_features, missingness_summary
from src.ingest import verify_or_create
from src.parse import load_outcomes, load_records
from src.split import check_splits, make_splits, save_splits
from src.validate import DataError, validate


def run_day1(cfg: dict) -> dict:
    d = cfg["data"]
    manifest = verify_or_create(cfg)

    adm, obs = load_records(p(d["records_dir"]))
    labels = load_outcomes(p(d["outcomes_file"]), cfg["id_col"], cfg["target"])

    # validate() sees every outcome row, so "outcomes without a record" is a real check.
    adm, vitals, report = validate(adm, obs, labels, cfg)
    labels = labels[labels[cfg["id_col"]].isin(adm["RecordID"])]

    X = build_features(adm, vitals, cfg)
    leaked = set(X.columns) & set(cfg["outcome_cols"])
    if leaked:
        raise DataError(f"Outcome fields used as inputs: {leaked}")
    report["missing_in_24h_window"] = missingness_summary(X, cfg)

    splits = make_splits(labels, cfg)
    report["splits"] = check_splits(splits, labels, cfg)
    report["split_seed"] = cfg["split"]["seed"]
    report["cutoff_minutes"] = cfg["window"]["cutoff_minutes"]
    report["data_version"] = manifest["data_version"]

    out = p(d["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    adm.to_csv(out / "admission.csv", index=False)
    vitals.to_csv(out / "vitals_long.csv.gz", index=False)
    X.to_csv(out / "features.csv")
    labels.to_csv(out / "labels.csv", index=False)
    save_splits(splits, p(d["splits_dir"]))

    rep_dir = p(d["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "data_quality.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    rep = run_day1(cfg)
    print(json.dumps({k: rep[k] for k in
                      ["n_records", "label", "missing_in_24h_window", "splits", "warnings"]},
                     indent=2))


if __name__ == "__main__":
    main()
