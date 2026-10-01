# Telco Customer Churn

Churn classifier for the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), with a CLI and a FastAPI service for single-customer predictions.

Two models are trained and saved:

- `models/baseline_lr.joblib` - logistic regression baseline
- `models/churn_model.joblib` - tuned XGBoost (default for inference)

Each artifact holds the full sklearn pipeline (preprocessing + model), the decision threshold and some metadata (train date, sklearn version, test metrics).

## Setup

Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate        # windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src            # windows: set PYTHONPATH=src
```

## Usage

```bash
# EDA summary -> reports/eda_summary.md
python3 -m churn.eda

# train both models -> models/, reports/model_comparison.csv, reports/metrics.json
python3 -m churn.train

# predict from the command line
python3 -m churn.predict --json sample_payload.json
python3 -m churn.predict --json sample_payload.json --model baseline

# or run the API
uvicorn churn.api:app --app-dir src --port 8000 or python3 -m uvicorn churn.api:app --app-dir src --port 8000
curl -X POST "localhost:8000/predict" -H "Content-Type: application/json" -d @sample_payload.json
curl -X POST "localhost:8000/predict?model=baseline" -H "Content-Type: application/json" -d @sample_payload.json

# tests
pytest -q
```

Swagger docs are at http://localhost:8000/docs once the server is up.

Sample response:

```json
{
  "customerID": "7590-VHVEG",
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.8654,
  "threshold": 0.5726,
  "model_name": "XGBoost (tuned)"
}
```

## Project structure

```
data/            raw csv
models/          saved pipelines
reports/         eda summary, model comparison, metrics
src/churn/
  config.py      paths, feature lists, seeds
  data.py        loading + dtype fixes (shared by training and inference)
  features.py    ColumnTransformer and model pipelines
  eda.py         quick EDA
  train.py       CV, tuning, threshold selection, saving
  schemas.py     pydantic request/response models
  predict.py     ChurnPredictor + CLI
  api.py         FastAPI app
tests/           inference + API tests
```

## Data prep notes

- `TotalCharges` is read as a string. 11 rows are blank, all with `tenure == 0` (brand new customers). These are converted to NaN and median-imputed inside the pipeline.
- Categoricals are one-hot encoded. Unknown categories at inference are ignored, but the API already validates allowed values so bad input gets a 422.
- `customerID` is dropped.
- Target is 26.5% churn. Handled with `class_weight="balanced"` for LR and `scale_pos_weight` for XGBoost, stratified splits, PR-AUC as the tuning metric, and a tuned decision threshold.

To avoid leakage: the test set is split off first and only used for final numbers. All preprocessing is inside the pipeline so it's refit per CV fold. Hyperparameters and thresholds come from CV on the train split only (threshold = max F1 on out-of-fold predictions). The saved models are refit on all data with those settings fixed.

## Results

5-fold stratified CV on the train split, then the 20% hold-out set (threshold tuned per model):

| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall |
|---|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.846 | 0.660 | 0.841 | 0.632 | 0.618 | 0.725 |
| XGBoost, default params | 0.820 | 0.619 | 0.819 | 0.621 | 0.590 | 0.759 |
| XGBoost, tuned | **0.850** | **0.671** | **0.846** | **0.663** | **0.630** | 0.735 |

Untuned XGBoost actually does worse than LR, it overfits. After tuning (shallow trees, low learning rate, 40 random search iterations) it beats the baseline, but not by a lot. Most of the signal is fairly linear: contract type, tenure, fiber internet and electronic check payments.

## Why not accuracy?

Predicting "No" for everyone gives 73.5% accuracy and catches zero churners. With an imbalanced target, accuracy mostly reflects the majority class. ROC-AUC tells us how well the model ranks customers, PR-AUC focuses on the churn class and is more informative when positives are rare, and F1 / recall at the chosen threshold reflect how the model will actually be used (missing a churner usually costs more than a false alarm).

## Further Works
1. **Cost-based threshold and calibration.** Calibrate probabilities and pick the threshold from the actual cost of a retention offer vs. a lost customer, instead of max F1. Make the threshold configurable instead of baked into the artifact.
The current project chooses the cutoff for a statistical reason. The improvement would choose it for a business reason, and also make it adjustable. The point has three parts.

**A. How the threshold is picked**

Now: train.py tries every possible cutoff on the out-of-fold training predictions and keeps the one with the highest F1, which is about 0.57. F1 treats a missed churner and a false alarm as roughly equally bad.

Improvement: pick the cutoff that minimises money lost. For example:

A retention offer (discount or call) costs ₹500, and you spend it on every customer flagged.
A lost customer costs ₹10,000 in future revenue.

With those numbers, missing a churner costs 20× more than a wasted offer. The best cutoff then drops well below 0.57 (to around 0.1–0.3), so you flag more customers and catch more churners. F1 can't capture that trade-off; a cost function can.

**B. Calibration**

Now: the model outputs a "probability", but it isn't a reliable one. Because training up-weights churners (scale_pos_weight, class_weight="balanced"), the scores are pushed upward. When the model says 0.87, the customer's real chance of churning may be lower. Ranking is still fine, which is why ROC-AUC looks good, but the number itself isn't trustworthy.

Improvement: wrap the model in CalibratedClassifierCV (isotonic or Platt/sigmoid), so that when it says 0.30, about 30% of such customers really churn. You can check this with a calibration curve or the Brier score. This matters for cost-based thresholds because expected cost = probability × cost, which only works if the probability is real.

**C. Where the threshold lives**

Now: the threshold is saved inside the .joblib file, so changing it means retraining or editing the artifact.

Improvement: keep a default in the artifact, but let the API or a config file override it. For example, POST /predict?threshold=0.25, or a THRESHOLD setting in config or an environment variable. The business can then move the cutoff (say, when a campaign budget changes) without touching the model.
