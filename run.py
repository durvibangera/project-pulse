# run.py — YAML-driven pipeline runner
# Usage:
#   python run.py --config configs/diabetes.yaml
#   python run.py --config configs/cardio.yaml

import argparse
import yaml
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict):
    print(f"\n[PULSE Runner] Starting: {config['experiment_name']}")

    master = pd.read_parquet("data/processed/master_patient_table.parquet")
    df = master[master["source_dataset"].isin(config["source_datasets"])].copy()
    df = df[df[config["label_column"]].notna()]

    features = config["features"]
    # Only keep features that exist in the dataset
    available = [f for f in features if f in df.columns]
    if len(available) < len(features):
        missing = set(features) - set(available)
        print(f"[PULSE Runner] Warning: features not found in master table: {missing}")
    X = df[available].fillna(df[available].median())
    y = df[config["label_column"]].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        **config["model_params"],
        eval_metric="logloss",
        random_state=42,
    )

    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=config["model_name"]):
        model.fit(X_train, y_train)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        mlflow.log_params(config["model_params"])
        mlflow.log_metric("cv_auc_mean", auc_scores.mean())
        mlflow.log_metric("cv_auc_std", auc_scores.std())
        mlflow.sklearn.log_model(model, artifact_path=config["model_name"])

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, proba)
        test_f1 = f1_score(y_test, preds)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_f1", test_f1)

    print(classification_report(y_test, preds))
    print(f"CV AUC: {auc_scores.mean():.4f} \u00b1 {auc_scores.std():.4f}")
    print(f"Test AUC: {test_auc:.4f} | F1: {test_f1:.4f}")

    joblib.dump(model, config["output_model_path"])
    print(f"Model saved \u2192 {config['output_model_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PULSE YAML-driven model runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config)
