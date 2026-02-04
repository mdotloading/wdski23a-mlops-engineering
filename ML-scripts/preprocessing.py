import os
import argparse
import tempfile
import boto3
from botocore.client import Config
from io import BytesIO

import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


DROP_COLS = ["Unnamed: 0", "id", "date"]
TARGET_COL = "price"
CATEGORICAL_COLS = ["zipcode"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name-mlflow", required=True)
    p.add_argument("--bucket-name", required=True)
    p.add_argument("--filename", required=True)
    p.add_argument("--output-path", required=True)  # kept for compatibility; not used heavily
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
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

    # drop noisy cols if present
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    # basic NA handling (simple demo-safe)
    df = df.dropna()

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # identify columns
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    # Fit on train only (proper)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    preprocessor.fit(X_train)

    # transform and save as dense-ish dataframe
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    # Create feature names
    feature_names = []
    feature_names.extend(num_cols)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"]
        for col, cats in zip(cat_cols, ohe.categories_):
            feature_names.extend([f"{col}={c}" for c in cats])

    train_df = pd.DataFrame(X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t, columns=feature_names)
    val_df = pd.DataFrame(X_val_t.toarray() if hasattr(X_val_t, "toarray") else X_val_t, columns=feature_names)
    train_df[TARGET_COL] = y_train.reset_index(drop=True).values
    val_df[TARGET_COL] = y_val.reset_index(drop=True).values

    with mlflow.start_run(run_name="preprocess") as run:
        mlflow.log_param("bucket", args.bucket_name)
        mlflow.log_param("filename", args.filename)
        mlflow.log_param("target", TARGET_COL)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("num_cols", len(num_cols))
        mlflow.log_param("cat_cols", len(cat_cols))

        tmp_dir = tempfile.mkdtemp()
        out_dir = os.path.join(tmp_dir, "processed_data")
        os.makedirs(out_dir, exist_ok=True)

        train_path = os.path.join(out_dir, "train.csv")
        val_path = os.path.join(out_dir, "val.csv")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        mlflow.log_artifacts(out_dir, artifact_path="processed_data")

        # Argo output parameter file
        with open("/tmp/preprocessing_run_id.txt", "w") as f:
            f.write(run.info.run_id)


if __name__ == "__main__":
    main()
