from __future__ import annotations

import argparse
import subprocess
import sys


def run_step(cmd: list[str], step_name: str) -> None:
    print(f"\n===== {step_name} =====")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    python = sys.executable
    config = args.config

    run_step([python, "training/train.py", "--config", config], "TRAIN")
    run_step([python, "training/evaluate.py", "--config", config], "EVALUATE")
    run_step([python, "training/gate.py", "--config", config], "QUALITY_GATE")
    run_step([python, "training/register_model.py", "--config", config], "REGISTER_MODEL")


if __name__ == "__main__":
    main()
