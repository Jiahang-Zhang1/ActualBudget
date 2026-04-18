from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output"]["dir"])
    run_name = cfg["mlflow"].get("run_name", Path(args.config).stem)

    gate_path = output_dir / "gate_result.json"
    model_path = output_dir / f"{run_name}_model.joblib"

    if not gate_path.exists():
        raise FileNotFoundError(f"Missing gate result: {gate_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    gate_result = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate_result.get("pass", False):
        print("Model did not pass quality gate. Registration skipped.")
        print(json.dumps(gate_result, indent=2))
        raise SystemExit(1)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = cfg["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    model_name = (
        cfg.get("registry", {}) or {}
    ).get("model_name", "actual-budget-transaction-classifier")

    pipeline = joblib.load(model_path)

    with mlflow.start_run(run_name=f"{run_name}_register") as run:
        mlflow.set_tags(
            {
                "stage": "register",
                "source_run_name": run_name,
                "model_name": model_name,
                "gate_pass": "true",
            }
        )

        metrics = gate_result.get("metrics", {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

        mlflow.log_artifact(str(gate_path))

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=model_name,
        )

        result = {
            "register_run_id": run.info.run_id,
            "registered_model_name": model_name,
            "model_uri": model_info.model_uri,
        }

        out_path = output_dir / "register_result.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print(json.dumps(result, indent=2))
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
