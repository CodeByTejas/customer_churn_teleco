"""Preprocessing and model pipelines."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # ignore unseen categories at inference instead of erroring
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), config.NUMERIC_FEATURES),
        ("bin", SimpleImputer(strategy="most_frequent"), config.BINARY_NUMERIC),
        ("cat", cat, config.CATEGORICAL_FEATURES),
    ])


def logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=config.RANDOM_STATE)),
    ])


def xgb_pipeline(scale_pos_weight: float) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=False)),
        ("clf", XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            scale_pos_weight=scale_pos_weight, n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )),
    ])
