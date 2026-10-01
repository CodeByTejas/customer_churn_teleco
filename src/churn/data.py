"""Load the raw csv and fix dtypes.

clean() only does row-wise fixes, so it is reused as-is at inference time.
Imputation/encoding is fitted inside the sklearn pipeline (features.py).
"""
from __future__ import annotations

import pandas as pd

from . import config


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges is a string in the csv, blank for tenure == 0 customers
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", None), errors="coerce"
    )
    for col in ("tenure", "MonthlyCharges", "SeniorCitizen"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").str.strip()
    return df


def load_dataset(path=config.DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(path)
    df = clean(raw)
    y = (df[config.TARGET].str.strip() == "Yes").astype(int)
    X = df[config.FEATURES]
    return X, y
