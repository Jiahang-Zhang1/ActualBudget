import json
from pathlib import Path

def pass_quality_gate(metrics: dict) -> tuple[bool, list[str]]:
    reasons = []

    if metrics.get("test_top3_accuracy", 0) < 0.70:
        reasons.append("test_top3_accuracy below 0.70")

    if metrics.get("test_macro_f1", 0) < 0.55:
        reasons.append("test_macro_f1 below 0.55")

    if metrics.get("model_size_mb", 999) > 50:
        reasons.append("model too large")

    return len(reasons) == 0, reasons


def main():
    summary_path = Path("/app/outputs/baseline_tfidf_logreg_summary.json")
    metrics = json.loads(summary_path.read_text(encoding="utf-8"))

    ok, reasons = pass_quality_gate(metrics)
    result = {
        "pass": ok,
        "reasons": reasons,
        "metrics": metrics,
    }

    Path("/app/outputs/gate_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
