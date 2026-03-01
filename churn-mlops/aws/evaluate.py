"""
Script executado pelo ProcessingStep de avaliação no SageMaker Pipeline.
Gera o evaluation.json no formato esperado pelo Model Registry.
"""

import os
import json
import pickle
import tarfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR  = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"


def load_model():
    """Descomprime model.tar.gz e carrega o pipeline."""
    tar_path = os.path.join(MODEL_DIR, "model.tar.gz")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(MODEL_DIR)

    pkl_path = os.path.join(MODEL_DIR, "model.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_test_data():
    files = [f for f in os.listdir(TEST_DIR) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(TEST_DIR, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)


def main():
    pipeline = load_model()
    df = load_test_data()

    feature_cols = [c for c in df.columns if c not in ("churn", "customer_id", "event_date", "etl_processed_at", "etl_job_name", "data_quality_flag")]
    X = df[feature_cols].values
    y = df["churn"].values

    y_pred  = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    roc_auc   = roc_auc_score(y, y_proba)
    avg_prec  = average_precision_score(y, y_proba)
    f1        = f1_score(y, y_pred)

    # Formato exigido pelo SageMaker Model Registry
    report = {
        "binary_classification_metrics": {
            "roc_auc": {
                "value": round(roc_auc, 4),
                "standard_deviation": "NaN",
            },
            "average_precision": {
                "value": round(avg_prec, 4),
                "standard_deviation": "NaN",
            },
            "f1": {
                "value": round(f1, 4),
                "standard_deviation": "NaN",
            },
        }
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "evaluation.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Avg-Precision: {avg_prec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Relatório salvo em: {report_path}")


if __name__ == "__main__":
    main()
