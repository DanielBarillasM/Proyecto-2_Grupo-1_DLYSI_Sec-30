"""Entrena o reanuda secuencialmente los tres experimentos registrados."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args(default_epochs: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    with open(ROOT / "configs" / "experiments.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    args = parse_args(int(config["training"]["epochs"]))

    for experiment in config["experiments"]:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_experiment.py"),
            "--experiment",
            str(experiment["id"]),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
        ]
        print(f"\n=== {experiment['id']} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare_experiments.py")],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
