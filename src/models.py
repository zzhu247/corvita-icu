"""The two model families. Each is one sklearn Pipeline, so every learned step
(median imputation, scaling, the model) is fit on train only and saved together.

logreg: median imputation + standard scaling on continuous columns, flags passed through.
hgb:    HistGradientBoosting handles NaN itself, so no imputation is needed.
"""
from itertools import product

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def is_binary(col: str) -> bool:
    return col.endswith("_missing") or col.startswith(("Gender_", "ICUType_"))


def split_columns(cols: list[str]) -> tuple[list[str], list[str]]:
    binary = [c for c in cols if is_binary(c)]
    continuous = [c for c in cols if not is_binary(c)]
    return continuous, binary


def build_pipeline(family: str, params: dict, cols: list[str], seed: int) -> Pipeline:
    if family == "logreg":
        continuous, binary = split_columns(cols)
        pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), continuous),
            ("bin", "passthrough", binary),
        ])
        clf = LogisticRegression(max_iter=5000, **params)
        return Pipeline([("pre", pre), ("clf", clf)])

    if family == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed, **params)
        return Pipeline([("clf", clf)])

    raise ValueError(f"unknown model family: {family}")


def param_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]
