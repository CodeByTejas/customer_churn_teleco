"""Train the logistic regression baseline and a tuned XGBoost model.

- stratified 80/20 split, test set only used for the final numbers
- 5-fold stratified CV on the train split
- XGBoost tuned with RandomizedSearchCV on PR-AUC
- decision threshold picked on out-of-fold train predictions (max F1)
- both models are refit on the full data and saved to models/

usage: python -m churn.train
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     train_test_split)

from . import config
from .data import load_dataset
from .features import logistic_pipeline, xgb_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"}

XGB_SEARCH_SPACE = {
    "clf__n_estimators": randint(150, 600),
    "clf__max_depth": randint(2, 6),
    "clf__learning_rate": loguniform(0.01, 0.2),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.5, 0.5),
    "clf__min_child_weight": randint(1, 10),
    "clf__reg_lambda": loguniform(0.1, 10),
}


def pick_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "threshold": round(threshold, 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def fit_and_report(name, pipe, X_tr, y_tr, X_te, y_te, cv):
    cv_res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=SCORING, n_jobs=-1)
    cv_summary = {k: (round(cv_res[f"test_{k}"].mean(), 4), round(cv_res[f"test_{k}"].std(), 4))
                  for k in SCORING}
    log.info("%s cv: %s", name, cv_summary)

    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold = pick_threshold(y_tr, oof)

    pipe.fit(X_tr, y_tr)
    test = evaluate(y_te, pipe.predict_proba(X_te)[:, 1], threshold)
    log.info("%s test: %s", name, {k: v for k, v in test.items() if k != "confusion_matrix"})
    return {"cv": cv_summary, "test": test, "threshold": threshold}


def save_model(key, name, pipe, threshold, results, X, y, extra=None):
    pipe.fit(X, y)
    artifact = {
        "pipeline": pipe,
        "threshold": threshold,
        "features": config.FEATURES,
        "metadata": {
            "model_name": name,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "test_metrics": results["test"],
            **(extra or {}),
        },
    }
    path = config.MODEL_PATHS[key]
    joblib.dump(artifact, path)
    log.info("saved %s", path)


def main():
    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    log.info("train=%s test=%s churn_rate=%.3f", X_tr.shape, X_te.shape, y_tr.mean())
    log.info("accuracy of always predicting 'No': %.3f", 1 - y_te.mean())

    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

    results = {}
    results["logreg"] = fit_and_report("logreg", logistic_pipeline(), X_tr, y_tr, X_te, y_te, cv)
    results["xgb_default"] = fit_and_report("xgb_default", xgb_pipeline(pos_weight),
                                            X_tr, y_tr, X_te, y_te, cv)

    search = RandomizedSearchCV(xgb_pipeline(pos_weight), XGB_SEARCH_SPACE, n_iter=40,
                                scoring="average_precision", cv=cv, n_jobs=-1,
                                random_state=config.RANDOM_STATE)
    search.fit(X_tr, y_tr)
    best_params = {k.replace("clf__", ""): (round(float(v), 4) if isinstance(v, float) else int(v))
                   for k, v in search.best_params_.items()}
    log.info("best xgb params: %s", best_params)
    results["xgb_tuned"] = fit_and_report("xgb_tuned", search.best_estimator_,
                                          X_tr, y_tr, X_te, y_te, cv)

    config.MODEL_DIR.mkdir(exist_ok=True)
    save_model("baseline", "LogisticRegression (baseline)", logistic_pipeline(),
               results["logreg"]["threshold"], results["logreg"], X, y)
    save_model("xgb", "XGBoost (tuned)", search.best_estimator_,
               results["xgb_tuned"]["threshold"], results["xgb_tuned"], X, y,
               extra={"best_params": best_params})

    rows = []
    for name, r in results.items():
        row = {"model": name}
        row.update({f"cv_{k}": f"{m:.3f} +/- {s:.3f}" for k, (m, s) in r["cv"].items()})
        row.update({f"test_{k}": v for k, v in r["test"].items() if k != "confusion_matrix"})
        rows.append(row)
    table = pd.DataFrame(rows)

    config.REPORT_DIR.mkdir(exist_ok=True)
    table.to_csv(config.REPORT_DIR / "model_comparison.csv", index=False)
    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(
        {"xgb_best_params": best_params, **{k: v["test"] for k, v in results.items()}}, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
