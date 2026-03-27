"""
src/models/base_model.py
------------------------
Shared XGBoost training/evaluation/explainability base for all P.U.L.S.E. models.

Features:
    - XGBClassifier with sensible clinical defaults
    - MLflow experiment tracking (params, CV metrics, model artifact)
    - StratifiedKFold cross-validation (handles class imbalance)
    - SHAP TreeExplainer for clinical interpretability
    - Standardised evaluation report (ROC-AUC, F1, classification report)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mlflow
import mlflow.sklearn
import numpy as np
import shap
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

load_dotenv()

# Default MLflow tracking URI from .env or fallback to ./mlflow
_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlflow")
mlflow.set_tracking_uri(_MLFLOW_URI)


class BaseModel:
    """
    Wraps XGBClassifier with MLflow logging, cross-validation, and SHAP explainability.

    Parameters
    ----------
    name : str
        Human-readable model name used for MLflow run naming and artifact paths.
    features : list[str]
        Ordered list of input feature column names.
    label : str
        Target column name (informational; used in logging).
    xgb_params : dict, optional
        Override default XGBClassifier hyperparameters.
    """

    _DEFAULT_PARAMS = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    def __init__(
        self,
        name: str,
        features: list[str],
        label: str,
        xgb_params: dict | None = None,
    ) -> None:
        self.name = name
        self.features = features
        self.label = label
        params = {**self._DEFAULT_PARAMS, **(xgb_params or {})}
        self.model = XGBClassifier(**params)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        experiment_name: str = "PULSE",
        n_splits: int = 5,
    ) -> dict:
        """
        Fit the model and log everything to MLflow.

        Returns a dict with cv_auc_mean and cv_auc_std.
        """
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=self.name):
            self.model.fit(X_train, y_train)

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            auc_scores = cross_val_score(
                self.model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
            )

            mlflow.log_metric("cv_auc_mean", float(auc_scores.mean()))
            mlflow.log_metric("cv_auc_std", float(auc_scores.std()))
            mlflow.log_params(
                {k: v for k, v in self.model.get_params().items() if v is not None}
            )
            try:
                mlflow.sklearn.log_model(self.model, artifact_path=self.name)
            except Exception:
                mlflow.xgboost.log_model(self.model, artifact_path=self.name)

            print(
                f"[{self.name}] CV AUC: {auc_scores.mean():.4f} "
                f"± {auc_scores.std():.4f}"
            )

        return {"cv_auc_mean": auc_scores.mean(), "cv_auc_std": auc_scores.std()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Print a classification report and return AUC + F1 metrics.
        """
        preds = self.model.predict(X_test)
        proba = self.model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"\n{'=' * 50}")
        print(f"  {self.name} — Test Set Evaluation")
        print(f"{'=' * 50}")
        print(classification_report(y_test, preds, zero_division=0))
        print(f"  ROC-AUC : {auc:.4f}")
        print(f"  F1      : {f1:.4f}")
        print(f"{'=' * 50}\n")

        return {"auc": auc, "f1": f1}

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain(
        self,
        X_sample: pd.DataFrame,
        max_display: int = 15,
        plot: bool = True,
    ) -> np.ndarray:
        """
        Compute SHAP values and (optionally) display a summary plot.

        Parameters
        ----------
        X_sample : pd.DataFrame
            Sample of the feature matrix (200–500 rows recommended).
        max_display : int
            Number of top features to show in the SHAP plot.
        plot : bool
            Set False in headless / CI environments.

        Returns the raw SHAP values array.
        """
        try:
            # XGBoost 3.x: pass the booster directly to avoid base_score parsing issue
            booster = self.model.get_booster() if hasattr(self.model, "get_booster") else self.model
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(X_sample)
        except Exception as e:
            print(f"[{self.name}] SHAP unavailable ({e}); skipping.")
            return np.array([])

        if plot:
            try:
                shap.summary_plot(
                    shap_values,
                    X_sample,
                    feature_names=self.features,
                    max_display=max_display,
                    show=True,
                )
            except Exception:
                pass

        return shap_values
