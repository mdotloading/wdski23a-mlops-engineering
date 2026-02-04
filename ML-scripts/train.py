import os
import argparse
import tempfile
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path 
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


TARGET_COL = "price"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name-mlflow", required=True)
    p.add_argument("--preprocessing-run-id", required=True)
    return p.parse_args()


def load_preprocessed_data(run_id: str):
    client = mlflow.tracking.MlflowClient()
    tmp_dir = tempfile.mkdtemp()
    client.download_artifacts(run_id=run_id, path="processed_data", dst_path=tmp_dir)

    train_path = os.path.join(tmp_dir, "processed_data", "train.csv")
    val_path = os.path.join(tmp_dir, "processed_data", "val.csv")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    args = parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(args.experiment_name_mlflow)

    train_df, val_df = load_preprocessed_data(args.preprocessing_run_id)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]

    models = {
        "linreg": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gbr": GradientBoostingRegressor(random_state=42),
    }

    best_name, best_model, best_mae = None, None, float("inf")
    metrics_table = {}

    with mlflow.start_run(run_name="train") as run:
        run_id = run.info.run_id
        Path("/tmp/train_run_id.txt").write_text(run_id)

        mlflow.log_param("preprocessing_run_id", args.preprocessing_run_id)

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_val)

            mae = mean_absolute_error(y_val, pred)
            r = rmse(y_val, pred)
            metrics_table[name] = {"mae": float(mae), "rmse": float(r)}

            # log as nested run (nice, but optional)
            with mlflow.start_run(run_name=f"model-{name}", nested=True):
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", r)

            if mae < best_mae:
                best_mae = mae
                best_name = name
                best_model = model

        # log best
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_mae", best_mae)
        mlflow.log_dict(metrics_table, artifact_file="training/metrics_all_models.json")

        mlflow.sklearn.log_model(best_model, artifact_path="model")

        with open("/tmp/train_run_id.txt", "w") as f:
            f.write(run.info.run_id)


if __name__ == "__main__":
    main()
