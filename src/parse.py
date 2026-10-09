"""Step 2: turn the per-patient text files into two tables linked by RecordID.

Each record file looks like:
    Time,Parameter,Value
    00:00,RecordID,132539     <- general descriptors, recorded once at 00:00
    00:00,Age,54
    00:00,Gender,0
    00:00,ICUType,4
    00:07,HR,73               <- time series, Time = HH:MM since ICU admission
    ...

Outputs
    admission: one row per RecordID  (RecordID, Age, Gender, ICUType)
    obs:       long table            (RecordID, Time, minutes, Parameter, Value)

Missing data: -1 in the descriptors means missing, so it becomes NaN here.
Nothing is filtered by time in this file; the 24 h cutoff lives in features.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DESCRIPTORS = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]


def time_to_minutes(t: pd.Series) -> pd.Series:
    """'HH:MM' -> minutes since admission. Bad formats become NaN (counted in validate)."""
    parts = t.str.extract(r"^(\d{1,3}):(\d{2})$")
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    minutes = h * 60 + m
    minutes[m >= 60] = np.nan
    return minutes


def parse_record(path: Path) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(path, dtype={"Time": str, "Parameter": str})
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["minutes"] = time_to_minutes(df["Time"])

    # Descriptors = the leading 00:00 block. Weight also appears later as a time
    # series, so only the first occurrence of each descriptor counts.
    head = df[(df["Time"] == "00:00") & df["Parameter"].isin(DESCRIPTORS)]
    desc = head.drop_duplicates("Parameter").set_index("Parameter")["Value"].to_dict()
    desc["file_id"] = path.stem

    obs = df.drop(index=head.drop_duplicates("Parameter").index)
    obs = obs.assign(RecordID=desc.get("RecordID", np.nan))
    return desc, obs[["RecordID", "Time", "minutes", "Parameter", "Value"]]


def load_records(records_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(Path(records_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No record files in {records_dir}")

    descs, obs_parts = [], []
    for f in files:
        d, o = parse_record(f)
        descs.append(d)
        obs_parts.append(o)

    adm = pd.DataFrame(descs)
    adm = adm[["file_id", "RecordID", "Age", "Gender", "ICUType"]]
    for c in ["Age", "Gender", "ICUType"]:
        adm[c] = adm[c].replace(-1, np.nan)
    adm["RecordID"] = adm["RecordID"].astype("Int64")

    obs = pd.concat(obs_parts, ignore_index=True)
    obs["RecordID"] = obs["RecordID"].astype("Int64")
    return adm, obs


def load_outcomes(path: Path, id_col: str, target: str) -> pd.DataFrame:
    """Only the ID and the label are kept. Other outcome fields never leave this function."""
    out = pd.read_csv(path)
    out = out[[id_col, target]].copy()
    out[id_col] = out[id_col].astype("Int64")
    return out
