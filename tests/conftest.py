"""Synthetic data in the exact PhysioNet 2012 format, so tests run without the real
dataset (CI has no access to it). Values are random; only the format matters."""
import copy
import tarfile
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config


def write_record(path: Path, rid: int, age=60, gender=1, icu=3, rows=()) -> None:
    """rows: iterable of (time 'HH:MM', parameter, value)."""
    lines = ["Time,Parameter,Value", f"00:00,RecordID,{rid}", f"00:00,Age,{age}",
             f"00:00,Gender,{gender}", "00:00,Height,-1", f"00:00,ICUType,{icu}",
             "00:00,Weight,80"]
    lines += [f"{t},{p},{v}" for t, p, v in rows]
    path.write_text("\n".join(lines) + "\n")


def _random_rows(rng, with_resp: bool):
    rows = []
    for minute in sorted(rng.choice(np.arange(5, 48 * 60), size=40, replace=False)):
        t = f"{minute // 60:02d}:{minute % 60:02d}"
        rows.append((t, "HR", int(rng.normal(85, 15))))
        rows.append((t, "Temp", round(float(rng.normal(37, 0.6)), 1)))
        if with_resp:
            rows.append((t, "RespRate", int(rng.normal(18, 4))))
        rows.append((t, "Weight", 80))          # time-series Weight, must not be a descriptor
    return rows


def make_synthetic(tmp_path: Path) -> dict:
    """200 records + outcomes + tarball, laid out like data/raw. Returns a config dict."""
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    rec_dir = raw / "set-a"
    rec_dir.mkdir(parents=True)

    outcome_lines = ["RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death"]
    for i in range(200):
        rid = 132539 + i
        write_record(rec_dir / f"{rid}.txt", rid,
                     age=int(rng.integers(18, 90)), gender=int(rng.choice([0, 1, -1], p=[.45, .5, .05])),
                     icu=int(rng.integers(1, 5)), rows=_random_rows(rng, with_resp=i % 3 == 0))
        death = int(i % 7 == 0)                    # ~14% positive, like the real data
        outcome_lines.append(f"{rid},{rng.integers(5, 25)},{rng.integers(0, 12)},10,-1,{death}")
    (raw / "Outcomes-a.txt").write_text("\n".join(outcome_lines) + "\n")

    with tarfile.open(raw / "set-a.tar.gz", "w:gz") as tar:
        tar.add(rec_dir, arcname="set-a")

    cfg = copy.deepcopy(load_config())
    cfg["data"].update({
        "raw_dir": str(raw), "records_dir": str(rec_dir),
        "outcomes_file": str(raw / "Outcomes-a.txt"),
        "manifest": str(tmp_path / "manifest.json"),
        "processed_dir": str(tmp_path / "processed"),
        "splits_dir": str(tmp_path / "splits"),
        "reports_dir": str(tmp_path / "reports"),
    })
    return cfg


@pytest.fixture
def synthetic_dataset(tmp_path):
    return make_synthetic(tmp_path)


SMALL_GRIDS = {"logreg": {"C": [0.1, 1.0]},
               "hgb": {"learning_rate": [0.1], "max_depth": [2], "min_samples_leaf": [10]}}


def make_trained(tmp_path: Path, grids: dict | None = None, **model_overrides) -> dict:
    """Synthetic data -> Day 1 -> Day 2 with a tiny grid. Returns the config used.

    allow_dirty=True: tests must not depend on whether the working tree is committed.
    """
    from src.run_day1 import run_day1
    from src.train import run_day2

    cfg = make_synthetic(tmp_path)
    cfg["split"]["max_rate_gap"] = 0.05
    cfg["model"]["output_dir"] = str(tmp_path / "models")
    cfg["model"]["bootstrap_reps"] = 50
    cfg["model"]["min_sensitivity"] = 0.5          # tiny synthetic val set
    cfg["model"]["grids"] = grids or SMALL_GRIDS
    cfg["model"].update(model_overrides)
    run_day1(cfg)
    cfg["_result"] = run_day2(cfg, allow_dirty=True)
    return cfg


@pytest.fixture
def cfg():
    return load_config()
