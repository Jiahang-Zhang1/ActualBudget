import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return os.environ.get("GIT_SHA", "unknown")


def get_gpu_info() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.STDOUT
        ).decode("utf-8")
    except Exception:
        return "No GPU or nvidia-smi not available."


def prepare_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    data_cfg = cfg["data"]
    label_col = data_cfg["label_col"]
    text_cols = data_cfg.get("text_cols", [])
    categorical_cols = data_cfg.get("categorical_cols", [])
    numeric_cols = data_cfg.get("numeric_cols", [])

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str).str.strip()

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col] != ""].reset_index(drop=True)

    # 去掉完全空的文本样本
    if text_cols:
        combined_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
        df = df[combined_text != ""].reset_index(drop=True)

    return df


def filter_rare_classes(df: pd.DataFrame, label_col: str, min_examples_per_class: int):
    class_counts = df[label_col].value_counts()
    keep_labels = class_counts[class_counts >= min_examples_per_class].index
    filtered_df = df[df[label_col].isin(keep_labels)].reset_index(drop=True)
    dropped = int(len(df) - len(filtered_df))
    return filtered_df, dropped


def random_split(df: pd.DataFrame, label_col: str, cfg: dict):
    split_cfg = cfg["split"]
    train_frac = float(split_cfg["train_frac"])
    val_frac = float(split_cfg["val_frac"])
    random_state = int(split_cfg.get("random_state", 42))
    stratify_enabled = bool(split_cfg.get("stratify", True))

    if train_frac <= 0 or val_frac <= 0 or (train_frac + val_frac) >= 1:
        raise ValueError("Split fractions must satisfy 0 < train_frac, val_frac and train_frac + val_frac < 1")

    test_frac = 1.0 - train_frac - val_frac

    y = df[label_col]
    stratify_y = y if stratify_enabled else None

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_frac),
        random_state=random_state,
        stratify=stratify_y,
    )

    temp_relative_val_frac = val_frac / (val_frac + test_frac)
    temp_y = temp_df[label_col]
    stratify_temp_y = temp_y if stratify_enabled else None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - temp_relative_val_frac),
        random_state=random_state,
        stratify=stratify_temp_y,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_classifier(model_cfg: dict):
    clf_name = model_cfg["classifier"].lower()
    class_weight = model_cfg.get("class_weight", None)
    random_state = model_cfg.get("random_state", 42)

    if clf_name == "logreg":
        return LogisticRegression(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 1000)),
            class_weight=class_weight,
            random_state=random_state,
        )

    if clf_name == "linearsvc":
        return LinearSVC(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 3000)),
            class_weight=class_weight,
            random_state=random_state,
        )

    if clf_name == "sgd":
        return SGDClassifier(
            loss="log_loss",
            alpha=float(model_cfg.get("alpha", 0.0001)),
            max_iter=int(model_cfg.get("max_iter", 2000)),
            class_weight=class_weight,
            random_state=random_state,
        )

    raise ValueError(f"Unsupported classifier: {clf_name}")


def build_pipeline(cfg: dict) -> Pipeline:
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]

    text_cols = data_cfg.get("text_cols", [])
    categorical_cols = data_cfg.get("categorical_cols", [])
    numeric_cols = data_cfg.get("numeric_cols", [])

    transformers = []

    word_ngram_range = tuple(feat_cfg.get("word_ngram_range", [1, 2]))
    word_max_features = int(feat_cfg.get("word_max_features", 20000))
    word_min_df = int(feat_cfg.get("word_min_df", 2))

    use_char_tfidf = bool(feat_cfg.get("use_char_tfidf", False))
    char_ngram_range = tuple(feat_cfg.get("char_ngram_range", [3, 5]))
    char_max_features = int(feat_cfg.get("char_max_features", 10000))
    char_min_df = int(feat_cfg.get("char_min_df", 2))

    for col in text_cols:
        transformers.append(
            (
                f"{col}_word_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=word_ngram_range,
                    max_features=word_max_features,
                    min_df=word_min_df,
                    sublinear_tf=True,
                ),
                col,
            )
        )

        if use_char_tfidf:
            transformers.append(
                (
                    f"{col}_char_tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        lowercase=True,
                        ngram_range=char_ngram_range,
                        max_features=char_max_features,
                        min_df=char_min_df,
                        sublinear_tf=True,
                    ),
                    col,
                )
            )

    if categorical_cols:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            )
        )

    if numeric_cols:
        transformers.append(("numeric", "passthrough", numeric_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = build_classifier(model_cfg)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )


def get_score_matrix(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        try:
            return pipeline.predict_proba(X)
        except Exception:
            pass

    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return scores

    preds = pipeline.predict(X)
    classes = pipeline.named_steps["clf"].classes_
    scores = np.zeros((len(preds), len(classes)), dtype=float)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i, p in enumerate(preds):
        scores[i, class_to_idx[p]] = 1.0
    return scores


def top_k_accuracy(y_true, classes, scores, k=3) -> float:
    k = min(k, len(classes))
    topk_idx = np.argsort(scores, axis=1)[:, -k:]
    topk_labels = classes[topk_idx]
    hits = [y_true.iloc[i] in topk_labels[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def evaluate_split(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, top_k: int):
    preds = pipeline.predict(X)
    scores = get_score_matrix(pipeline, X)
    classes = pipeline.named_steps["clf"].classes_

    metrics = {
        f"{name}_accuracy": float(accuracy_score(y, preds)),
        f"{name}_macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        f"{name}_top{top_k}_accuracy": float(top_k_accuracy(y, classes, scores, top_k)),
    }

    report = classification_report(y, preds, zero_division=0, output_dict=True)
    return metrics, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = cfg["mlflow"]["experiment_name"]
    run_name = cfg["mlflow"].get("run_name", Path(args.config).stem)
    mlflow.set_experiment(experiment_name)

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = cfg["data"]["path"]
if data_path.endswith(".parquet"):
    df = pd.read_parquet(data_path)
else:
    df = pd.read_csv(data_path)
    df = prepare_dataframe(df, cfg)

    label_col = cfg["data"]["label_col"]
    top_k = int(cfg["eval"].get("top_k", 3))
    min_examples_per_class = int(cfg["split"].get("min_examples_per_class", 2))

    original_rows = len(df)
    df, dropped_rare = filter_rare_classes(df, label_col, min_examples_per_class)

    if len(df) == 0:
        raise ValueError("No data left after filtering rare classes.")

    train_df, val_df, test_df = random_split(df, label_col, cfg)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One of train/val/test splits is empty. Check dataset size or split ratios.")

    X_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col]

    X_val = val_df.drop(columns=[label_col])
    y_val = val_df[label_col]

    X_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col]

    pipeline = build_pipeline(cfg)

    start_time = time.time()

    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        mlflow.log_params(flatten_dict(cfg))
        mlflow.set_tags(
            {
                "git_sha": get_git_sha(),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "project_name": cfg.get("project_name", "unknown"),
            }
        )

        mlflow.log_param("original_rows", original_rows)
        mlflow.log_param("rows_after_filtering", len(df))
        mlflow.log_param("dropped_rare_class_rows", dropped_rare)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("val_rows", len(val_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("num_classes", y_train.nunique())

        pipeline.fit(X_train, y_train)

        train_wall_sec = time.time() - start_time

        val_metrics, val_report = evaluate_split("val", pipeline, X_val, y_val, top_k)
        test_metrics, test_report = evaluate_split("test", pipeline, X_test, y_test, top_k)

        all_metrics = {}
        all_metrics.update(val_metrics)
        all_metrics.update(test_metrics)
        all_metrics["train_wall_sec"] = float(train_wall_sec)

        model_path = output_dir / f"{run_name}_model.joblib"
        config_copy_path = output_dir / f"{run_name}_config_used.yaml"
        test_report_path = output_dir / f"{run_name}_test_report.json"
        val_report_path = output_dir / f"{run_name}_val_report.json"
        summary_path = output_dir / f"{run_name}_summary.json"
        gpu_info_path = output_dir / f"{run_name}_gpu_info.txt"

        joblib.dump(pipeline, model_path)
        with open(config_copy_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        with open(val_report_path, "w", encoding="utf-8") as f:
            json.dump(val_report, f, indent=2)

        with open(test_report_path, "w", encoding="utf-8") as f:
            json.dump(test_report, f, indent=2)

        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        all_metrics["model_size_mb"] = float(model_size_mb)
        all_metrics["peak_ram_mb"] = float(peak_ram_mb)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)

        with open(gpu_info_path, "w", encoding="utf-8") as f:
            f.write(get_gpu_info())

        mlflow.log_metrics(all_metrics)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(config_copy_path))
        mlflow.log_artifact(str(val_report_path))
        mlflow.log_artifact(str(test_report_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(gpu_info_path))

        print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
