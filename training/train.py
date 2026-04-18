from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
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

    # 去掉文本完全为空的样本
    if text_cols:
        combined_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
        df = df[combined_text != ""].reset_index(drop=True)

    return df


def filter_rare_classes(
    df: pd.DataFrame, label_col: str, min_examples_per_class: int
) -> tuple[pd.DataFrame, int]:
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
        raise ValueError(
            "Split fractions must satisfy 0 < train_frac, val_frac and train_frac + val_frac < 1"
        )

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


def _get_text_feature_cfg(cfg: dict) -> dict:
    # 兼容你原来仓库的旧结构和我前面给你的新结构
    features_cfg = cfg.get("features", {})
    if "text" in features_cfg:
        return features_cfg["text"]
    return {
        "ngram_range": features_cfg.get("word_ngram_range", [1, 2]),
        "max_features": features_cfg.get("word_max_features", 20000),
        "min_df": features_cfg.get("word_min_df", 2),
        "max_df": 1.0,
        "lowercase": True,
        "strip_accents": "unicode",
        "use_char_tfidf": features_cfg.get("use_char_tfidf", False),
        "char_ngram_range": features_cfg.get("char_ngram_range", [3, 5]),
        "char_max_features": features_cfg.get("char_max_features", 10000),
        "char_min_df": features_cfg.get("char_min_df", 2),
    }


def _get_categorical_feature_cfg(cfg: dict) -> dict:
    features_cfg = cfg.get("features", {})
    if "categorical" in features_cfg:
        return features_cfg["categorical"]
    return {"handle_unknown": "ignore"}


def build_classifier(model_cfg: dict):
    # 兼容旧版 classifier/logreg 和新版 type/logistic_regression
    clf_name = (
        model_cfg.get("type")
        or model_cfg.get("classifier")
        or "logistic_regression"
    )
    clf_name = str(clf_name).lower()

    class_weight = model_cfg.get("class_weight", None)
    random_state = int(model_cfg.get("random_state", 42))

    if clf_name in {"logistic_regression", "logreg"}:
        return LogisticRegression(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 1000)),
            class_weight=class_weight,
            random_state=random_state,
            solver=model_cfg.get("solver", "lbfgs"),
            multi_class=model_cfg.get("multi_class", "auto"),
        )

    if clf_name in {"linearsvc", "linear_svc"}:
        return LinearSVC(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 3000)),
            class_weight=class_weight,
            random_state=random_state,
        )

    if clf_name in {"sgd", "sgdclassifier"}:
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
    text_feat_cfg = _get_text_feature_cfg(cfg)
    categorical_feat_cfg = _get_categorical_feature_cfg(cfg)
    model_cfg = cfg["model"]

    text_cols = data_cfg.get("text_cols", [])
    categorical_cols = data_cfg.get("categorical_cols", [])
    numeric_cols = data_cfg.get("numeric_cols", [])

    transformers = []

    word_ngram_range = tuple(text_feat_cfg.get("ngram_range", [1, 2]))
    word_max_features = int(text_feat_cfg.get("max_features", 20000))
    word_min_df = text_feat_cfg.get("min_df", 1)
    word_max_df = text_feat_cfg.get("max_df", 1.0)
    lowercase = bool(text_feat_cfg.get("lowercase", True))
    strip_accents = text_feat_cfg.get("strip_accents", "unicode")

    use_char_tfidf = bool(text_feat_cfg.get("use_char_tfidf", False))
    char_ngram_range = tuple(text_feat_cfg.get("char_ngram_range", [3, 5]))
    char_max_features = int(text_feat_cfg.get("char_max_features", 10000))
    char_min_df = text_feat_cfg.get("char_min_df", 2)

    for col in text_cols:
        transformers.append(
            (
                f"{col}_word_tfidf",
                TfidfVectorizer(
                    lowercase=lowercase,
                    strip_accents=strip_accents,
                    ngram_range=word_ngram_range,
                    max_features=word_max_features,
                    min_df=word_min_df,
                    max_df=word_max_df,
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
                        lowercase=lowercase,
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
                OneHotEncoder(
                    handle_unknown=categorical_feat_cfg.get("handle_unknown", "ignore")
                ),
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


def top_k_accuracy(y_true: pd.Series, classes: np.ndarray, scores: np.ndarray, k: int = 3) -> float:
    k = min(k, len(classes))
    topk_idx = np.argsort(scores, axis=1)[:, -k:]
    topk_labels = classes[topk_idx]
    hits = [y_true.iloc[i] in topk_labels[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def precision_at_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top1_scores: np.ndarray,
    threshold: float,
) -> float | None:
    mask = top1_scores >= threshold
    if int(mask.sum()) == 0:
        return None
    return float((y_true[mask] == y_pred[mask]).mean())


def build_prediction_frame(
    split_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    preds: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    top3_idx = np.argsort(scores, axis=1)[:, -3:]
    top3_labels = classes[top3_idx]
    top1_scores = scores.max(axis=1)

    frame = X.copy()
    frame["split"] = split_name
    frame["y_true"] = y.values
    frame["y_pred"] = preds
    frame["top1_score"] = top1_scores
    frame["top3_labels"] = [json.dumps(list(row)) for row in top3_labels]
    return frame


def evaluate_split(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int,
    high_conf_threshold: float,
):
    preds = pipeline.predict(X)
    scores = get_score_matrix(pipeline, X)
    classes = pipeline.named_steps["clf"].classes_
    top1_scores = scores.max(axis=1)

    metrics = {
        f"{name}_top1_accuracy": float(accuracy_score(y, preds)),
        f"{name}_macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        f"{name}_weighted_f1": float(
            f1_score(y, preds, average="weighted", zero_division=0)
        ),
        f"{name}_top{top_k}_accuracy": float(top_k_accuracy(y, classes, scores, top_k)),
    }

    high_conf = precision_at_confidence(
        y.to_numpy(),
        preds,
        top1_scores,
        high_conf_threshold,
    )
    metrics[f"{name}_high_confidence_precision"] = (
        float(high_conf) if high_conf is not None else -1.0
    )
    metrics[f"{name}_high_confidence_count"] = int(
        (top1_scores >= high_conf_threshold).sum()
    )
    metrics[f"{name}_high_confidence_threshold"] = float(high_conf_threshold)

    report = classification_report(y, preds, zero_division=0, output_dict=True)
    prediction_frame = build_prediction_frame(name, X, y, preds, scores, classes)

    return metrics, report, prediction_frame


def ensure_output_dir(cfg: dict) -> Path:
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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

    output_dir = ensure_output_dir(cfg)

    data_path = cfg["data"]["path"]
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    df = prepare_dataframe(df, cfg)
    label_col = cfg["data"]["label_col"]
    top_k = int(cfg["eval"].get("top_k", 3))
    high_conf_threshold = float(cfg["eval"].get("high_conf_threshold", 0.90))
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
                "project_name": cfg.get("project_name", "actual-budget-transaction-classifier"),
                "stage": "training",
                "task": "transaction-category-suggestion",
                "label_space": "model_labels",
                "mapped_label_space": "actual_budget_categories",
            }
        )

        mlflow.log_param("original_rows", original_rows)
        mlflow.log_param("rows_after_filtering", len(df))
        mlflow.log_param("dropped_rare_class_rows", dropped_rare)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("val_rows", len(val_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("num_classes", int(y_train.nunique()))

        class_distribution = y_train.value_counts().to_dict()
        class_distribution_path = output_dir / f"{run_name}_train_class_distribution.json"
        save_json(class_distribution_path, class_distribution)

        pipeline.fit(X_train, y_train)
        train_wall_sec = time.time() - start_time

        val_metrics, val_report, val_predictions = evaluate_split(
            "val",
            pipeline,
            X_val,
            y_val,
            top_k,
            high_conf_threshold,
        )
        test_metrics, test_report, test_predictions = evaluate_split(
            "test",
            pipeline,
            X_test,
            y_test,
            top_k,
            high_conf_threshold,
        )

        all_metrics: dict[str, float] = {}
        all_metrics.update(val_metrics)
        all_metrics.update(test_metrics)
        all_metrics["train_wall_sec"] = float(train_wall_sec)

        model_path = output_dir / f"{run_name}_model.joblib"
        config_copy_path = output_dir / f"{run_name}_config_used.yaml"
        test_report_path = output_dir / f"{run_name}_test_report.json"
        val_report_path = output_dir / f"{run_name}_val_report.json"
        summary_path = output_dir / f"{run_name}_summary.json"
        gpu_info_path = output_dir / f"{run_name}_gpu_info.txt"
        val_preds_path = output_dir / f"{run_name}_val_predictions.csv"
        test_preds_path = output_dir / f"{run_name}_test_predictions.csv"
        metadata_path = output_dir / f"{run_name}_metadata.json"

        joblib.dump(pipeline, model_path)

        with open(config_copy_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        save_json(val_report_path, val_report)
        save_json(test_report_path, test_report)

        val_predictions.to_csv(val_preds_path, index=False)
        test_predictions.to_csv(test_preds_path, index=False)

        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        all_metrics["model_size_mb"] = float(model_size_mb)
        all_metrics["peak_ram_mb"] = float(peak_ram_mb)

        save_json(summary_path, all_metrics)

        metadata = {
            "run_name": run_name,
            "experiment_name": experiment_name,
            "label_col": label_col,
            "classes": sorted([str(c) for c in y_train.unique()]),
            "top_k": top_k,
            "high_conf_threshold": high_conf_threshold,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "model_path": str(model_path),
            "git_sha": get_git_sha(),
        }
        save_json(metadata_path, metadata)

        with open(gpu_info_path, "w", encoding="utf-8") as f:
            f.write(get_gpu_info())

        mlflow.log_metrics(all_metrics)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(config_copy_path))
        mlflow.log_artifact(str(val_report_path))
        mlflow.log_artifact(str(test_report_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(gpu_info_path))
        mlflow.log_artifact(str(val_preds_path))
        mlflow.log_artifact(str(test_preds_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(class_distribution_path))

        print(json.dumps(all_metrics, indent=2, ensure_ascii=False))
        print(f"Wrote: {model_path}")
        print(f"Wrote: {config_copy_path}")
        print(f"Wrote: {val_report_path}")
        print(f"Wrote: {test_report_path}")
        print(f"Wrote: {summary_path}")
        print(f"Wrote: {gpu_info_path}")
        print(f"Wrote: {val_preds_path}")
        print(f"Wrote: {test_preds_path}")
        print(f"Wrote: {metadata_path}")
        print(f"Wrote: {class_distribution_path}")


if __name__ == "__main__":
    main()
