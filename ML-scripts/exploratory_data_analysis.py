import os
import json
import argparse
import pandas as pd
import mlflow
import boto3
from botocore.client import Config
from io import BytesIO

DROP_COLS = ["Unnamed: 0", "id", "date"]
TARGET_COL = "price"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name-mlflow", required=True)
    p.add_argument("--bucket-name", required=True)
    p.add_argument("--filename", required=True)
    return p.parse_args()


def load_data_from_s3(bucket_name, file_name):
    s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    s3 = boto3.resource(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    obj = s3.Object(bucket_name, file_name)
    data = obj.get()["Body"].read()
    return pd.read_csv(BytesIO(data))


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(args.experiment_name_mlflow)

    df = load_data_from_s3(args.bucket_name, args.filename)

    with mlflow.start_run(run_name="eda") as run:
        mlflow.log_param("bucket", args.bucket_name)
        mlflow.log_param("filename", args.filename)

        # Basic cleaning view (do not mutate original too much)
        df2 = df.copy()
        for c in DROP_COLS:
            if c in df2.columns:
                df2 = df2.drop(columns=[c])

        # schema / missing
        summary = {
            "rows": int(df2.shape[0]),
            "cols": int(df2.shape[1]),
            "columns": df2.columns.tolist(),
            "dtypes": {k: str(v) for k, v in df2.dtypes.to_dict().items()},
            "missing": {k: int(v) for k, v in df2.isna().sum().to_dict().items()},
        }

        # target stats
        if TARGET_COL in df2.columns:
            t = df2[TARGET_COL].dropna()
            target_stats = {
                "target": TARGET_COL,
                "count": int(t.shape[0]),
                "mean": float(t.mean()),
                "std": float(t.std()),
                "min": float(t.min()),
                "p25": float(t.quantile(0.25)),
                "median": float(t.median()),
                "p75": float(t.quantile(0.75)),
                "max": float(t.max()),
            }
        else:
            target_stats = {"error": f"Target column '{TARGET_COL}' not found."}

        # numeric summary
        num = df2.select_dtypes(include="number")
        num_summary = num.describe().to_dict()

        # correlations with target (optional but nice)
        corr = {}
        if TARGET_COL in num.columns:
            corr_series = num.corr(numeric_only=True)[TARGET_COL].dropna().sort_values(ascending=False)
            corr = {k: float(v) for k, v in corr_series.to_dict().items()}

        os.makedirs("eda", exist_ok=True)
        with open("eda/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open("eda/target_stats.json", "w") as f:
            json.dump(target_stats, f, indent=2)
        with open("eda/numeric_summary.json", "w") as f:
            json.dump(num_summary, f, indent=2)
        with open("eda/target_correlations.json", "w") as f:
            json.dump(corr, f, indent=2)

        mlflow.log_artifacts("eda", artifact_path="eda")


if __name__ == "__main__":
    main()
