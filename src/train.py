"""Day 2: train two model families, choose on validation, score test once, save artifacts.

    python -m src.train               # writes models/v1/
    python -m src.train --overwrite   # replace an existing version on purpose

Order of operations (this order is the leakage protection):
  1. For each family, fit every grid point on TRAIN, score PR-AUC on VAL.
  2. Pick the best grid point per family, then the best family, by VAL PR-AUC.
  3. Pick the risk cutoff on VAL (highest cutoff with sensitivity >= target).
  4. Only now touch TEST: score both families with their frozen settings.
  5. Diagnostics on TEST that change no decision: CIs, calibration,
     missing-vital subgroups, masking, RespRate ablation, fairness subgroups.
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import ROOT, load_config, p
from src.evaluate import (all_metrics, bootstrap_ci, calibration_table, choose_threshold,
                          mask_vitals, subgroup_metrics)
from src.models import build_pipeline, param_grid
from src.split import load_splits


# ---- loading -------------------------------------------------------------------

def load_day1(cfg: dict):
    d = cfg["data"]
    X = pd.read_csv(p(d["processed_dir"]) / "features.csv", index_col="RecordID")
    y = pd.read_csv(p(d["processed_dir"]) / "labels.csv", index_col="RecordID")[cfg["target"]]
    splits = load_splits(p(d["splits_dir"]))
    parts = {k: (X.loc[ids], y.loc[ids]) for k, ids in splits.items()}
    return X, parts


def sha256_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ---- model selection (train + val only) ---------------------------------------------

def select_family(family, grid, cols, parts, seed):
    """Fit each grid point on train, score val. Return best params, best model, all results."""
    (Xtr, ytr), (Xva, yva) = parts["train"], parts["val"]
    results, best = [], None
    for params in param_grid(grid):
        model = build_pipeline(family, params, cols, seed).fit(Xtr[cols], ytr)
        pv = model.predict_proba(Xva[cols])[:, 1]
        score = all_metrics(yva, pv, 0.5)["pr_auc"]
        results.append({"params": params, "val_pr_auc": score})
        if best is None or score > best[0]:
            best = (score, params, model)
    return {"params": best[1], "val_pr_auc": best[0], "model": best[2], "grid": results}


def lr_coefficients(model, cols) -> list[dict]:
    """Coefficients on the scaled inputs, largest effect first. Only for logreg."""
    names = model.named_steps["pre"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coef))
    return [{"feature": names[i], "coef": round(float(coef[i]), 4)} for i in order]


# ---- main -----------------------------------------------------------------------------

def run_day2(cfg: dict, overwrite: bool = False) -> dict:
    m, seed = cfg["model"], cfg["split"]["seed"]
    out_dir = p(m["output_dir"]) / m["version"]
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} exists. Bump model.version or pass --overwrite.")

    X, parts = load_day1(cfg)
    cols = list(X.columns)
    (Xva, yva), (Xte, yte) = parts["val"], parts["test"]

    # 1-2. selection on val
    fams = {f: select_family(f, m["grids"][f], cols, parts, seed) for f in m["grids"]}
    winner = max(fams, key=lambda f: fams[f]["val_pr_auc"])
    model = fams[winner]["model"]

    # 3. cutoff on val
    pv = model.predict_proba(Xva[cols])[:, 1]
    thr = choose_threshold(yva, pv, m["min_sensitivity"])

    # 4. test, once, every family with its frozen settings and the same cutoff rule
    comparison = {}
    for f, r in fams.items():
        pv_f = r["model"].predict_proba(Xva[cols])[:, 1]
        thr_f = choose_threshold(yva, pv_f, m["min_sensitivity"])
        pt_f = r["model"].predict_proba(Xte[cols])[:, 1]
        comparison[f] = {"params": r["params"], "threshold": round(thr_f, 4),
                         "val": all_metrics(yva, pv_f, thr_f),
                         "test": all_metrics(yte, pt_f, thr_f)}
    pt = model.predict_proba(Xte[cols])[:, 1]

    # 5. diagnostics on test (none of these feed back into any choice)
    ci = bootstrap_ci(yte, pt, thr, m["bootstrap_reps"], seed)
    calib = calibration_table(yte, pt)

    vit_flags = Xte[[f"{v}_missing" for v in cfg["vitals"]]]
    missing_groups = subgroup_metrics(Xte, yte, pt, thr, {
        "all_vitals_present": ~vit_flags.any(axis=1),
        "any_vital_missing": vit_flags.any(axis=1),
        "RespRate_measured": Xte["RespRate_missing"] == 0,
        "RespRate_missing": Xte["RespRate_missing"] == 1,
        "all_vitals_missing": vit_flags.all(axis=1),
    })

    masking = {}
    for name, vs in {"none": [], **{v: [v] for v in cfg["vitals"]}, "all_vitals": cfg["vitals"]}.items():
        pm = model.predict_proba(mask_vitals(Xte, vs)[cols])[:, 1]
        masking[name] = all_metrics(yte, pm, thr)

    # Ablations: retrain the winning family on fewer columns, same grid, same val rule.
    #   no_resprate     every RespRate column. Dropping only the flag would change nothing,
    #                   because RespRate_count == 0 and NaN mean/last carry the same signal.
    #   no_counts       every *_count column: how much rides on how often staff charted.
    #   admission_only  Age, Gender, ICUType: what the vitals add on top.
    admission_cols = [c for c in cols if c.startswith(("Age", "Gender_", "ICUType_"))]
    subsets = {
        "no_resprate": [c for c in cols if not c.startswith("RespRate_")],
        "no_counts": [c for c in cols if not c.endswith("_count")],
        "admission_only": admission_cols,
    }
    ablations = {"full": {"n_features": len(cols),
                          "val": comparison[winner]["val"], "test": comparison[winner]["test"]}}
    for name, keep in subsets.items():
        r = select_family(winner, m["grids"][winner], keep, parts, seed)
        pv_a = r["model"].predict_proba(Xva[keep])[:, 1]
        thr_a = choose_threshold(yva, pv_a, m["min_sensitivity"])
        ablations[name] = {
            "n_features": len(keep),
            "dropped": [c for c in cols if c not in keep],
            "params": r["params"],
            "val": all_metrics(yva, pv_a, thr_a),
            "test": all_metrics(yte, r["model"].predict_proba(Xte[keep])[:, 1], thr_a),
        }
    ablation = {"dropped": ablations["no_resprate"]["dropped"],      # kept for older readers
                "with_resprate": {k: ablations["full"][k] for k in ["val", "test"]},
                "without_resprate": {k: ablations["no_resprate"][k] for k in ["params", "val", "test"]}}

    # Operating points: the cutoff for each sensitivity target is chosen on val,
    # then applied unchanged to test. Shows the alert-burden trade-off for the clinical team.
    operating_points = []
    pt_full = pt
    for target in m.get("operating_points", [0.6, 0.7, 0.8, 0.9]):
        t = choose_threshold(yva, pv, target)
        v, te = all_metrics(yva, pv, t), all_metrics(yte, pt_full, t)
        operating_points.append({
            "target_sensitivity": target, "threshold": round(t, 4),
            **{f"val_{k}": v[k] for k in ["sensitivity", "specificity", "ppv", "alert_rate"]},
            **{f"test_{k}": te[k] for k in ["sensitivity", "specificity", "ppv", "alert_rate"]},
        })

    icu = Xte[[f"ICUType_{k}" for k in cfg["categories"]["ICUType"]]].idxmax(axis=1)
    fairness = subgroup_metrics(Xte, yte, pt, thr, {
        "female": Xte["Gender_female"] == 1, "male": Xte["Gender_male"] == 1,
        **{name: icu == name for name in sorted(icu.unique())},
    })

    # ---- save -----------------------------------------------------------------------
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    joblib.dump(model, tmp / "model.joblib")

    dq = json.loads((p(cfg["data"]["reports_dir"]) / "data_quality.json").read_text())
    metadata = {
        "model_version": m["version"],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "data_version": dq["data_version"],
        "features_sha256": sha256_file(p(cfg["data"]["processed_dir"]) / "features.csv"),
        "cutoff_minutes": cfg["window"]["cutoff_minutes"],
        "seed": seed,
        "split_sizes": {k: int(len(v[1])) for k, v in parts.items()},
        "model_family": winner,
        "hyperparameters": fams[winner]["params"],
        "feature_columns": cols,
        "threshold": round(thr, 6),
        "threshold_rule": f"highest cutoff with val sensitivity >= {m['min_sensitivity']}",
        "selection_metric": m["selection_metric"],
        "train_missing_rates": {v: round(float(parts["train"][0][f"{v}_missing"].mean()), 4)
                                for v in cfg["vitals"]},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    metrics = {
        "winner": winner,
        "selection": {f: {"best_params": r["params"], "val_pr_auc": r["val_pr_auc"],
                          "grid": r["grid"]} for f, r in fams.items()},
        "comparison": comparison,
        "test": all_metrics(yte, pt, thr),
        "test_ci95": ci,
        "calibration_test": calib,
        "missing_subgroups_test": missing_groups,
        "masking_test": masking,
        "resprate_ablation": ablation,
        "ablations": ablations,
        "operating_points": operating_points,
        "fairness_test": fairness,
    }
    if "logreg" in fams:
        metrics["logreg_coefficients"] = lr_coefficients(fams["logreg"]["model"], cols)

    (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (tmp / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"RecordID": Xte.index, "y": yte.values, "p": pt}).to_csv(
        tmp / "test_predictions.csv", index=False)

    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)          # all files appear together, or not at all
    return {"metadata": metadata, "metrics": metrics}


def print_summary(res: dict) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    mt, md = res["metrics"], res["metadata"]
    print(f"\nwinner: {mt['winner']}  params={md['hyperparameters']}  threshold={md['threshold']:.4f}")
    rows = []
    for f, c in mt["comparison"].items():
        for split in ["val", "test"]:
            r = c[split]
            rows.append([f, split, r["pr_auc"], r["auroc"], r["brier"], r["sensitivity"],
                         r["specificity"], r["alert_rate"]])
    print(pd.DataFrame(rows, columns=["model", "split", "pr_auc", "auroc", "brier",
                                      "sens", "spec", "alert_rate"]).to_string(index=False))
    print("\ntest 95% CI:", mt["test_ci95"])
    print("\nmissing-vital subgroups (test):")
    print(pd.DataFrame(mt["missing_subgroups_test"]).T[
        ["n", "n_pos", "pr_auc", "auroc", "sensitivity", "specificity", "mean_predicted", "observed_rate"]])
    print("\nmasking (test, all patients):")
    print(pd.DataFrame(mt["masking_test"]).T[
        ["pr_auc", "auroc", "sensitivity", "specificity", "alert_rate", "mean_predicted"]])
    print("\nablations (winner family retrained on fewer columns):")
    rows = [[k, a["n_features"], a["val"]["pr_auc"], a["val"]["auroc"], a["test"]["pr_auc"],
             a["test"]["auroc"], a["test"]["brier"]] for k, a in mt["ablations"].items()]
    print(pd.DataFrame(rows, columns=["model", "n_feat", "val_pr_auc", "val_auroc",
                                      "test_pr_auc", "test_auroc", "test_brier"]).to_string(index=False))
    print("\noperating points (cutoff chosen on val, applied to test):")
    print(pd.DataFrame(mt["operating_points"]).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    res = run_day2(load_config(), overwrite=args.overwrite)
    print_summary(res)


if __name__ == "__main__":
    main()