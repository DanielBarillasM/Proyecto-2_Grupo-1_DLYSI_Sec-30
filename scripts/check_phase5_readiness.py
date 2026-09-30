"""Audita si el entrenamiento y los recursos permiten construir la galería final."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml
from torchvision.models import ResNet18_Weights


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import write_json


def main() -> None:
    with open(ROOT / "configs" / "experiments.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    target = int(config["training"]["epochs"])
    epochs: dict[str, int] = {}
    checkpoint_exists: dict[str, bool] = {}
    checkpoint_epochs: dict[str, int | None] = {}
    reasons: list[str] = []
    for experiment in config["experiments"]:
        experiment_id = str(experiment["id"])
        status_path = ROOT / "artifacts" / "experiments" / experiment_id / "status.json"
        checkpoint = ROOT / "checkpoints" / experiment_id / "latest.pt"
        completed = 0
        if status_path.exists():
            completed = int(json.loads(status_path.read_text(encoding="utf-8"))["completed_epochs"])
        epochs[experiment_id] = completed
        checkpoint_exists[experiment_id] = checkpoint.exists()
        checkpoint_epoch: int | None = None
        if checkpoint.exists():
            try:
                checkpoint_epoch = int(
                    torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"]
                )
            except Exception as error:
                reasons.append(
                    f"{experiment_id}: checkpoint ilegible ({type(error).__name__})"
                )
        checkpoint_epochs[experiment_id] = checkpoint_epoch
        if completed < target:
            reasons.append(f"{experiment_id}: {completed}/{target} épocas")
        if not checkpoint.exists():
            reasons.append(f"{experiment_id}: checkpoint ausente")
        elif checkpoint_epoch is not None and checkpoint_epoch < target:
            reasons.append(
                f"{experiment_id}: checkpoint interno {checkpoint_epoch}/{target} épocas"
            )

    weights = ResNet18_Weights.DEFAULT
    weight_name = Path(weights.url).name
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / weight_name
    if not weight_path.exists():
        reasons.append("pesos preentrenados ResNet18 aún no están en caché")

    gallery_manifest = ROOT / "galeria" / "manifest.csv"
    payload = {
        "status": "ready" if not reasons else "blocked",
        "target_epochs": target,
        "completed_epochs": epochs,
        "checkpoints_present": checkpoint_exists,
        "checkpoint_epochs": checkpoint_epochs,
        "resnet18_weights": {
            "url": weights.url,
            "cache_file": weight_name,
            "cached": weight_path.exists(),
        },
        "gallery_manifest_exists": gallery_manifest.exists(),
        "blocking_reasons": reasons,
        "next_command": "restaurar checkpoints latest.pt de época 60"
        if any(value is None or value < target for value in checkpoint_epochs.values())
        else (
            "python scripts/train_all.py --epochs 60 --device auto"
            if any(value < target for value in epochs.values())
            else "python scripts/build_gallery.py --experiment <id>"
        ),
    }
    output = ROOT / "artifacts" / "gallery" / "readiness.json"
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
