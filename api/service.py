"""Loads one model version and turns a request into a prediction.

Features are built by src/features.py and values are cleaned by src/validate.py,
the same code used for training. tests/test_api.py checks that patients scored here
get the offline pipeline's score (to the 4 decimals the API returns).

Data-quality flag, most severe first:
  insufficient_data  no vitals at all. Score from age, sex, ICU type only; high_risk is null.
  missing_vitals     at least one vital never measured.
  low_count          every vital present, but one has far fewer readings than typical.
  ok
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import build_features, feature_columns
from src.parse import time_to_minutes
from src.validate import out_of_range

HR_WARNING = ("HR missing: on the validation set, removing HR lowers predicted risk and "
              "alert rate (metrics.json -> masking_val). Treat a low score as unreliable.")


class Predictor:
    def __init__(self, model_dir: str | Path, cfg: dict):
        model_dir = Path(model_dir)
        self.cfg = cfg
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.model = joblib.load(model_dir / "model.joblib")
        self._check_matches_config()
        self.threshold = float(self.meta["threshold"])
        self.cols = self.meta["feature_columns"]
        # Older models (no train_median_counts in metadata) skip the low_count check.
        self.median_counts = self.meta.get("train_median_counts", {})
        self.low_count_fraction = cfg.get("serving", {}).get("low_count_fraction", 0.25)

    def _check_matches_config(self) -> None:
        """Refuse to start if this code would build features differently from training."""
        m, c = self.meta, self.cfg
        if m["cutoff_minutes"] != c["window"]["cutoff_minutes"]:
            raise RuntimeError(f"model was trained with cutoff {m['cutoff_minutes']} min, "
                               f"config says {c['window']['cutoff_minutes']}")
        if m["feature_columns"] != feature_columns(c):
            raise RuntimeError("feature columns in config do not match the trained model")

    def predict(self, req) -> dict:
        cfg, warnings = self.cfg, []

        obs = pd.DataFrame([o.model_dump() for o in req.observations],
                           columns=["time", "parameter", "value"])
        obs = obs.rename(columns={"time": "Time", "parameter": "Parameter", "value": "Value"})
        obs["RecordID"] = 0
        obs["minutes"] = time_to_minutes(obs["Time"].astype(str))

        late = int((obs["minutes"] > cfg["window"]["cutoff_minutes"]).sum())
        if late:
            warnings.append(f"{late} observation(s) after 24:00 ignored")

        in_window = obs[obs["minutes"] <= cfg["window"]["cutoff_minutes"]]
        bad = out_of_range(in_window, cfg)
        if bad.any():
            detail = in_window.loc[bad, "Parameter"].value_counts().to_dict()
            warnings.append(f"implausible values ignored: {detail}")
        obs = obs.drop(index=in_window.index[bad])

        adm = pd.DataFrame([{"RecordID": 0, "Age": req.age,
                             "Gender": np.nan if req.gender is None else req.gender,
                             "ICUType": req.icu_type}])
        X = build_features(adm, obs, cfg)[self.cols]
        prob = float(self.model.predict_proba(X)[0, 1])

        row = X.iloc[0]
        missing = [v for v in cfg["vitals"] if row[f"{v}_missing"] == 1]
        low = [v for v in cfg["vitals"]
               if v in self.median_counts and 0 < row[f"{v}_count"]
               < self.low_count_fraction * self.median_counts[v]]

        high_risk: bool | None = prob >= self.threshold
        if len(missing) == len(cfg["vitals"]):
            flag = "insufficient_data"
            high_risk = None
            warnings.append("no vitals in the first 24 h: score uses age, sex and ICU type only; "
                            "high_risk is not set")
        elif missing:
            flag = "missing_vitals"
        elif low:
            flag = "low_count"
        else:
            flag = "ok"
        if "HR" in missing and flag != "insufficient_data":
            warnings.append(HR_WARNING)
        if low:
            typical = {v: self.median_counts[v] for v in low}
            warnings.append(f"few readings for {low} (typical per patient: {typical}); "
                            "send every charted value, the reading count affects the score")

        return {
            "probability": round(prob, 4),
            "high_risk": high_risk,
            "threshold": round(self.threshold, 4),
            "model_version": self.meta["model_version"],
            "data_version": self.meta["data_version"],
            "data_quality_flag": flag,
            "missing_vitals": missing,
            "low_count_vitals": low,
            "warnings": warnings,
            # for logs only, not returned: no values, no IDs
            "_log": {"vital_counts": {v: int(X.iloc[0][f"{v}_count"]) for v in cfg["vitals"]},
                     "n_ignored_after_cutoff": late, "n_ignored_implausible": int(bad.sum())},
        }
