from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


DEFAULT_THRESHOLDS = {
    "test_top1_accuracy_min": 0.70,
    "test_top3_accuracy_min": 0.90,
    "test_macro_f1_min": 0.60,
    "test_high_confidence_precision_min": 0.85,
    "test_mapped_top1_accuracy_min": 0.75,
    "per_class_recall_min": 0.40,
    "per_class_support_min": 20,
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output"]["dir"])
    metrics_path = output_dir / "metrics.json"
    per_class_path = output_dir / "per_class_metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not per_class_path.exists():
        raise FileNotFoundError(f"Missing per-class metrics file: {per_class_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_class = json.loads(per_class_path.read_text(encoding="utf-8"))

    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(cfg.get("quality_gates", {}) or {})

    reasons: list[str] = []

    if metrics.get("test_top1_accuracy", 0.0) < thresholds["test_top1_accuracy_min"]:
        reasons.append(
            f"test_top1_accuracy {metrics.get('test_top1_accuracy', 0.0):.3f} "
            f"< {thresholds['test_top1_accuracy_min']:.2f}"
        )

    if metrics.get("test_top3_accuracy", 0.0) < thresholds["test_top3_accuracy_min"]:
        reasons.append(
            f"test_top3_accuracy {metrics.get('test_top3_accuracy', 0.0):.3f} "
            f"< {thresholds['test_top3_accuracy_min']:.2f}"
        )

    if metrics.get("test_macro_f1", 0.0) < thresholds["test_macro_f1_min"]:
        reasons.append(
            f"test_macro_f1 {metrics.get('test_macro_f1', 0.0):.3f} "
            f"< {thresholds['test_macro_f1_min']:.2f}"
        )

    high_conf = metrics.get("test_high_confidence_precision", -1.0)
    if high_conf < 0 or high_conf < thresholds["test_high_confidence_precision_min"]:
        reasons.append(
            f"test_high_confidence_precision {high_conf:.3f} "
            f"< {thresholds['test_high_confidence_precision_min']:.2f}"
        )

    if (
        metrics.get("test_mapped_top1_accuracy", 0.0)
        < thresholds["test_mapped_top1_accuracy_min"]
    ):
        reasons.append(
            f"test_mapped_top1_accuracy {metrics.get('test_mapped_top1_accuracy', 0.0):.3f} "
            f"< {thresholds['test_mapped_top1_accuracy_min']:.2f}"
        )

    test_report = per_class.get("test_report", {})
    support_min = int(thresholds["per_class_support_min"])
    recall_min = float(thresholds["per_class_recall_min"])

    for label, stats in test_report.items():
        if not isinstance(stats, dict):
            continue
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue

        support = int(stats.get("support", 0))
        recall = float(stats.get("recall", 0.0))

        if support >= support_min and recall < recall_min:
            reasons.append(
                f"class '{label}' recall {recall:.3f} < {recall_min:.2f} "
                f"(support={support})"
            )

    result = {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "thresholds": thresholds,
        "metrics": metrics,
    }

    out_path = output_dir / "gate_result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
