"""Step 5: split RecordIDs 70/15/15, stratified on the label, fixed seed.

Split unit = patient (RecordID). One row per patient, so no patient can land in
two groups; check_splits() still asserts it. The saved ID files are the source
of truth for Day 2: training reads them instead of re-splitting.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(labels: pd.DataFrame, cfg: dict) -> dict[str, pd.Index]:
    id_col, target, s = cfg["id_col"], cfg["target"], cfg["split"]
    ids, y = labels[id_col], labels[target]

    train_ids, tmp_ids, _, y_tmp = train_test_split(
        ids, y, test_size=s["val"] + s["test"], stratify=y, random_state=s["seed"]
    )
    val_ids, test_ids = train_test_split(
        tmp_ids, test_size=s["test"] / (s["val"] + s["test"]),
        stratify=y_tmp, random_state=s["seed"],
    )
    return {k: pd.Index(sorted(v)) for k, v in
            {"train": train_ids, "val": val_ids, "test": test_ids}.items()}


def check_splits(splits: dict, labels: pd.DataFrame, cfg: dict) -> dict:
    id_col, target = cfg["id_col"], cfg["target"]
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = splits[a].intersection(splits[b])
            assert overlap.empty, f"{a} and {b} share RecordIDs: {list(overlap[:5])}"
    union = set().union(*map(set, splits.values()))
    assert union == set(labels[id_col]), "splits do not cover every record exactly once"

    y = labels.set_index(id_col)[target]
    overall = float(y.mean())
    summary = {"overall_rate": round(overall, 4)}
    for k, ids in splits.items():
        rate = float(y.loc[ids].mean())
        summary[k] = {"n": int(len(ids)), "n_pos": int(y.loc[ids].sum()), "rate": round(rate, 4)}
        gap = abs(rate - overall)
        assert gap <= cfg["split"]["max_rate_gap"], f"{k} death rate off by {gap:.3f}"
    return summary


def save_splits(splits: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, ids in splits.items():
        pd.DataFrame({"RecordID": ids}).to_csv(out_dir / f"{k}_ids.csv", index=False)


def load_splits(out_dir: Path) -> dict[str, pd.Index]:
    return {k: pd.Index(pd.read_csv(out_dir / f"{k}_ids.csv")["RecordID"])
            for k in ["train", "val", "test"]}
