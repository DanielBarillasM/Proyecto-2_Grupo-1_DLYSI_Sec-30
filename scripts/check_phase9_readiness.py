"""Comprueba los prerrequisitos reales para ejecutar los vecinos de la fase 9."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torchvision.models import ResNet18_Weights


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import write_json


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    selected_path = ROOT / "artifacts" / "phase8" / "selected_model.json"
    reasons: list[str] = []
    selected: dict[str, object] = {}
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
    else:
        reasons.append("falta artifacts/phase8/selected_model.json")

    experiment = str(selected.get("experiment", ""))
    target_epoch = int(config["training"]["epochs"])
    checkpoint_path = ROOT / str(
        selected.get("checkpoint", f"checkpoints/{experiment}/latest.pt")
    )
    checkpoint_epoch: int | None = None
    checkpoint_experiment: str | None = None
    checkpoint_error: str | None = None
    if checkpoint_path.exists():
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_epoch = int(payload["epoch"])
            raw_experiment = payload.get("experiment", "")
            checkpoint_experiment = str(
                raw_experiment.get("id", "")
                if isinstance(raw_experiment, dict)
                else raw_experiment
            )
        except Exception as error:  # pragma: no cover - depende del archivo externo
            checkpoint_error = f"{type(error).__name__}: {error}"
            reasons.append("el checkpoint seleccionado no se puede leer")
    else:
        reasons.append(f"falta {checkpoint_path.relative_to(ROOT).as_posix()}")
    if checkpoint_epoch is not None and checkpoint_epoch < target_epoch:
        reasons.append(
            f"checkpoint {experiment}: época interna {checkpoint_epoch}/{target_epoch}"
        )
    if checkpoint_experiment and checkpoint_experiment != experiment:
        reasons.append(
            f"checkpoint pertenece a {checkpoint_experiment}, no a {experiment}"
        )

    manifest_path = ROOT / "data" / "processed" / "manifest.csv"
    manifest_rows = 0
    missing_training_images = 0
    duplicate_training_hashes = 0
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest_rows = int(len(manifest))
        duplicate_training_hashes = int(manifest["sha256"].duplicated().sum())
        missing_training_images = int(
            (~manifest["path"].map(lambda value: (ROOT / str(value)).exists())).sum()
        )
        if manifest_rows != int(config["project"]["dataset_size"]):
            reasons.append(f"manifest tiene {manifest_rows} filas")
        if duplicate_training_hashes:
            reasons.append(f"manifest tiene {duplicate_training_hashes} hashes duplicados")
        if missing_training_images:
            reasons.append(f"faltan {missing_training_images} imágenes de entrenamiento")
    else:
        reasons.append("falta data/processed/manifest.csv")

    weights = ResNet18_Weights.DEFAULT
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not weight_path.exists():
        reasons.append("pesos oficiales ResNet18 todavía no están en caché")

    if checkpoint_epoch != target_epoch:
        next_command = "restaurar checkpoints/bce_spectral_norm/latest.pt de época 60"
    elif not weight_path.exists():
        next_command = "descargar ResNet18 ejecutando scripts/analyze_phase9_neighbors.py"
    else:
        next_command = "python scripts/analyze_phase9_neighbors.py --device auto"

    payload = {
        "status": "ready" if not reasons else "blocked",
        "phase": 9,
        "selected_experiment": experiment or None,
        "target_epoch": target_epoch,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_experiment": checkpoint_experiment,
        "checkpoint_error": checkpoint_error,
        "dataset_manifest_rows": manifest_rows,
        "missing_training_images": missing_training_images,
        "duplicate_training_hashes": duplicate_training_hashes,
        "resnet18": {
            "url": weights.url,
            "cache_file": weight_path.name,
            "cached": weight_path.exists(),
        },
        "blocking_reasons": reasons,
        "next_command": next_command,
    }
    write_json(ROOT / "artifacts" / "phase9" / "readiness.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
