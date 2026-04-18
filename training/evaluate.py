from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from label_mapping import load_label_mapping_from_cfg, map_model_label_to_actual


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    if text_cols:
        combined_text = (
            df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
        )
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


def top_k_accuracy(y_true: pd.Series, classes: np.ndarray, scores: np.ndarray, k: int) -> float:
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
    y_true: pd.Series,
    preds: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    mapping: dict[str, str],
    text_col: str | None,
) -> pd.DataFrame:
    top3_idx = np.argsort(scores, axis=1)[:, -3:]
    top3_labels = classes[top3_idx]
    top1_scores = scores.max(axis=1)

    frame = X.copy()
    frame["split"] = split_name
    frame["y_true"] = y_true.values
    frame["y_pred"] = preds
    frame["top1_score"] = top1_scores
    frame["top3_labels"] = [json.dumps(list(row)) for row in top3_labels]
    frame["mapped_y_true"] = [map_model_label_to_actual(v, mapping) for v in y_true.values]
    frame["mapped_y_pred"] = [map_model_label_to_actual(v, mapping) for v in preds]
    frame["mapped_top3_labels"] = [
        json.dumps([map_model_label_to_actual(v, mapping) for v in row])
        for row in top3_labels
    ]

    if text_col and text_col in frame.columns:
        frame["slice_text"] = frame[text_col].fillna("").astype(str).str.lower()
    else:
        frame["slice_text"] = ""

    return frame


def compute_split_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    classes: np.ndarray,
    split_name: str,
    top_k: int,
    high_conf_threshold: float,
) -> dict[str, float]:
    y_true = frame["y_true"]
    y_pred = frame["y_pred"]
    mapped_y_true = frame["mapped_y_true"]
    mapped_y_pred = frame["mapped_y_pred"]
    top1_scores = frame["top1_score"].to_numpy()

    metrics = {
        f"{split_name}_top1_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{split_name}_top{top_k}_accuracy": float(
            top_k_accuracy(y_true, classes, scores, top_k)
        ),
        f"{split_name}_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{split_name}_weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        f"{split_name}_mapped_top1_accuracy": float(
            accuracy_score(mapped_y_true, mapped_y_pred)
        ),
        f"{split_name}_mapped_macro_f1": float(
            f1_score(mapped_y_true, mapped_y_pred, average="macro", zero_division=0)
        ),
    }

    high_conf = precision_at_confidence(
        y_true.to_numpy(),
        y_pred.to_numpy(),
        top1_scores,
        high_conf_threshold,
    )
    metrics[f"{split_name}_high_confidence_precision"] = (
        float(high_conf) if high_conf is not None else -1.0
    )
    metrics[f"{split_name}_high_confidence_threshold"] = float(high_conf_threshold)
    metrics[f"{split_name}_high_confidence_count"] = int(
        (top1_scores >= high_conf_threshold).sum()
    )

    return metrics


def compute_slice_metrics(
    frame: pd.DataFrame,
    slice_cfg: dict[str, list[str]],
) -> dict[str, Any]:
    slices: dict[str, Any] = {}

    for slice_name, keywords in slice_cfg.items():
        keywords = [k.lower().strip() for k in keywords if k.strip()]
        if not keywords:
            continue

        mask = frame["slice_text"].apply(
            lambda text: any(keyword in text for keyword in keywords)
        )

        subset = frame[mask].copy()
        if subset.empty:
            slices[slice_name] = {"support": 0}
            continue

        slices[slice_name] = {
            "support": int(len(subset)),
            "top1_accuracy": float(accuracy_score(subset["y_true"], subset["y_pred"])),
            "mapped_top1_accuracy": float(
                accuracy_score(subset["mapped_y_true"], subset["mapped_y_pred"])
            ),
            "macro_f1": float(
                f1_score(
                    subset["y_true"],
                    subset["y_pred"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "mapped_macro_f1": float(
                f1_score(
                    subset["mapped_y_true"],
                    subset["mapped_y_pred"],
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    return slices


def get_default_slices() -> dict[str, list[str]]:
    return {
        "food_keywords": ["starbucks", "pizza", "subway", "mcdonald", "uber eat"],
        "income_keywords": ["salary", "payroll", "bonus", "income"],
        "shopping_keywords": ["amazon", "walmart", "target", "costco"],
    }


def get_text_col(cfg: dict) -> str | None:
    text_cols = cfg["data"].get("text_cols", [])
    return text_cols[0] if text_cols else None


def get_run_id_for_training_run(cfg: dict) -> str | None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = cfg["mlflow"]["experiment_name"]
    run_name = cfg["mlflow"].get("run_name", "baseline")

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )

    for run in runs:
        if run.data.tags.get("mlflow.runName") == run_name:
            return run.info.run_id

    return None


def log_artifacts_to_mlflow(cfg: dict, file_paths: list[Path], metrics: dict[str, float]) -> None:
    run_id = get_run_id_for_training_run(cfg)
    if run_id is None:
        print("Warning: could not find training MLflow run; skipping evaluation artifact logging.")
        return

    client = MlflowClient()
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            client.log_metric(run_id, key, float(value))

    for path in file_paths:
        if path.exists():
            client.log_artifact(run_id, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = cfg["mlflow"].get("run_name", Path(args.config).stem)
    model_path = output_dir / f"{run_name}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data_path = cfg["data"]["path"]
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    df = prepare_dataframe(df, cfg)
    label_col = cfg["data"]["label_col"]
    min_examples_per_class = int(cfg["split"].get("min_examples_per_class", 2))
    df, _ = filter_rare_classes(df, label_col, min_examples_per_class)

    _, val_df, test_df = random_split(df, label_col, cfg)
    pipeline: Pipeline = joblib.load(model_path)

    top_k = int(cfg.get("eval", {}).get("top_k", 3))
    high_conf_threshold = float(cfg.get("eval", {}).get("high_conf_threshold", 0.90))
    label_mapping = load_label_mapping_from_cfg(cfg)
    text_col = get_text_col(cfg)

    X_val = val_df.drop(columns=[label_col])
    y_val = val_df[label_col]
    X_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col]

    val_preds = pipeline.predict(X_val)
    val_scores = get_score_matrix(pipeline, X_val)
    classes = pipeline.named_steps["clf"].classes_

    test_preds = pipeline.predict(X_test)
    test_scores = get_score_matrix(pipeline, X_test)

    val_frame = build_prediction_frame(
        "val",
        X_val,
        y_val,
        val_preds,
        val_scores,
        classes,
        label_mapping,
        text_col,
    )
    test_frame = build_prediction_frame(
        "test",
        X_test,
        y_test,
        test_preds,
        test_scores,
        classes,
        label_mapping,
        text_col,
    )

    metrics: dict[str, float] = {}
    metrics.update(
        compute_split_metrics(
            val_frame, val_scores, classes, "val", top_k, high_conf_threshold
        )
    )
    metrics.update(
        compute_split_metrics(
            test_frame, test_scores, classes, "test", top_k, high_conf_threshold
        )
    )

    val_report = classification_report(
        val_frame["y_true"],
        val_frame["y_pred"],
        zero_division=0,
        output_dict=True,
    )
    test_report = classification_report(
        test_frame["y_true"],
        test_frame["y_pred"],
        zero_division=0,
        output_dict=True,
    )

    mapped_val_report = classification_report(
        val_frame["mapped_y_true"],
        val_frame["mapped_y_pred"],
        zero_division=0,
        output_dict=True,
    )
    mapped_test_report = classification_report(
        test_frame["mapped_y_true"],
        test_frame["mapped_y_pred"],
        zero_division=0,
        output_dict=True,
    )

    slice_cfg = cfg.get("slices", {}) or get_default_slices()
    slice_metrics = {
        "val": compute_slice_metrics(val_frame, slice_cfg),
        "test": compute_slice_metrics(test_frame, slice_cfg),
    }

    metrics_path = output_dir / "metrics.json"
    per_class_path = output_dir / "per_class_metrics.json"
    slice_path = output_dir / "slice_metrics.json"
    val_preds_path = output_dir / "val_predictions.csv"
    test_preds_path = output_dir / "test_predictions.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    per_class_path.write_text(
        json.dumps(
            {
                "val_report": val_report,
                "test_report": test_report,
                "mapped_val_report": mapped_val_report,
                "mapped_test_report": mapped_test_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    slice_path.write_text(json.dumps(slice_metrics, indent=2), encoding="utf-8")
    val_frame.to_csv(val_preds_path, index=False)
    test_frame.to_csv(test_preds_path, index=False)

    log_artifacts_to_mlflow(
        cfg,
        [metrics_path, per_class_path, slice_path, val_preds_path, test_preds_path],
        metrics,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {per_class_path}")
    print(f"Wrote: {slice_path}")
    print(f"Wrote: {val_preds_path}")
    print(f"Wrote: {test_preds_path}")


if __name__ == "__main__":
    main()
