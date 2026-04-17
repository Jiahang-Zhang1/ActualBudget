import json
import shutil
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("/app/outputs")
REGISTRY_DIR = Path("/app/model-registry")

def main():
    gate = json.loads((OUTPUT_DIR / "gate_result.json").read_text(encoding="utf-8"))
    if not gate["pass"]:
        raise SystemExit("model did not pass quality gate")

    version = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    dst = REGISTRY_DIR / version
    dst.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        OUTPUT_DIR / "baseline_tfidf_logreg_model.joblib",
        dst / "model.joblib",
    )

    metadata = {
        "model_version": version,
        "registered_at": datetime.utcnow().isoformat(),
        "metrics": gate["metrics"],
        "passed_gate": True,
    }

    (dst / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
