"""Single-customer inference.

usage:
    python -m churn.predict --json sample_payload.json
    python -m churn.predict --json sample_payload.json --model baseline
    cat sample_payload.json | python -m churn.predict
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .data import clean
from .schemas import CustomerProfile


class ChurnPredictor:
    def __init__(self, model_path: Path):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} not found, run `python -m churn.train` first")
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.threshold = float(artifact["threshold"])
        self.features = artifact["features"]
        self.metadata = artifact["metadata"]

    def predict(self, payload: dict) -> dict:
        customer = CustomerProfile.model_validate(payload)
        df = clean(pd.DataFrame([customer.model_dump()]))[self.features]
        proba = float(self.pipeline.predict_proba(df)[0, 1])
        pred = int(proba >= self.threshold)
        return {
            "customerID": customer.customerID,
            "churn_prediction": pred,
            "churn_label": "Yes" if pred else "No",
            "churn_probability": round(proba, 4),
            "threshold": round(self.threshold, 4),
            "model_name": self.metadata["model_name"],
        }


@lru_cache(maxsize=None)
def get_predictor(model: str = config.DEFAULT_MODEL) -> ChurnPredictor:
    if model not in config.MODEL_PATHS:
        raise ValueError(f"unknown model '{model}', choose from {list(config.MODEL_PATHS)}")
    return ChurnPredictor(config.MODEL_PATHS[model])


def main():
    parser = argparse.ArgumentParser(description="Predict churn for a single customer")
    parser.add_argument("--json", type=Path, help="path to a json file (reads stdin if omitted)")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODEL_PATHS))
    args = parser.parse_args()

    payload = json.loads(args.json.read_text() if args.json else sys.stdin.read())
    print(json.dumps(get_predictor(args.model).predict(payload), indent=2))


if __name__ == "__main__":
    main()
