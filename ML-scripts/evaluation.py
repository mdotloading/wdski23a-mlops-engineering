import os
import json
import argparse
import tempfile
import numpy as np
import pandas as pd

import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET_COL = "price"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name-mlflow", required=True)
    p.add_argument("--training-run-id", required=True)
    p.add_argument("--preprocessing-run-id", required=True)
    # demo gates (simple + sensible)
    p.add_argument("--max-mae", type=float, default=150000.0)
    p.add_argument("--max-rmse", type=float, default=250000.0)
    return p.parse_args()


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_val_data(preprocess_run_id: str):
    client = mlflow.tracking.MlflowClient()
    tmp_dir = tempfile.mkdtemp()
    client.download_artifacts(run_id=preprocess_run_id, path="processed_data", dst_path=tmp_dir)
    val_path = os.path.join(tmp_dir, "processed_data", "val.csv")
    return pd.read_csv(val_path)


def load_model(training_run_id: str):
    return mlflow.sklearn.load_model(f"runs:/{training_run_id}/model")


def main():
    args = parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(args.experiment_name_mlflow)

    val_df = load_val_data(args.preprocessing_run_id)
    model = load_model(args.training_run_id)

    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL].values

    pred = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, pred))
    r = float(rmse(y_val, pred))

    decision = {
        "training_run_id": args.training_run_id,
        "preprocessing_run_id": args.preprocessing_run_id,
        "mae": mae,
        "rmse": r,
        "gate_max_mae": args.max_mae,
        "gate_max_rmse": args.max_rmse,
        "pass": (mae <= args.max_mae) and (r <= args.max_rmse),
    }

    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_param("training_run_id", args.training_run_id)
        mlflow.log_param("preprocessing_run_id", args.preprocessing_run_id)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", r)
        mlflow.log_artifacts("evaluation", artifact_path="evaluation")


if __name__ == "__main__":
    main()
