import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from churn.api import app
from churn import config
from churn.predict import ChurnPredictor

SAMPLE = json.loads((Path(__file__).parents[1] / "sample_payload.json").read_text())


@pytest.fixture(scope="module", params=list(config.MODEL_PATHS))
def predictor(request):
    return ChurnPredictor(config.MODEL_PATHS[request.param])


def test_predict_sample(predictor):
    out = predictor.predict(SAMPLE)
    assert out["churn_prediction"] in (0, 1)
    assert 0.0 <= out["churn_probability"] <= 1.0
    assert out["customerID"] == "7590-VHVEG"


def test_baseline_model():
    from churn import config
    out = ChurnPredictor(config.MODEL_PATHS["baseline"]).predict(SAMPLE)
    assert 0.0 <= out["churn_probability"] <= 1.0
    assert out["model_name"].startswith("LogisticRegression")


def test_blank_total_charges(predictor):
    out = predictor.predict({**SAMPLE, "tenure": 0, "TotalCharges": " "})
    assert 0.0 <= out["churn_probability"] <= 1.0


def test_numeric_total_charges(predictor):
    assert predictor.predict({**SAMPLE, "TotalCharges": 29.85})["churn_probability"] >= 0


def test_invalid_category_rejected(predictor):
    with pytest.raises(ValidationError):
        predictor.predict({**SAMPLE, "Contract": "Lifetime"})


def test_api_roundtrip():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json=SAMPLE)
        assert r.status_code == 200 and set(r.json()) >= {"churn_prediction", "churn_probability"}
        assert c.post("/predict?model=baseline", json=SAMPLE).status_code == 200
        assert c.post("/predict", json={**SAMPLE, "tenure": -3}).status_code == 422
