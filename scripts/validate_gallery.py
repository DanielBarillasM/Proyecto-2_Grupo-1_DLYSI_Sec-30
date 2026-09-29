"""Valida procedencia, unicidad y regeneración exacta de la galería final."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import sha256_file, tensors_to_uint8
from src.models import Generator, ModelDimensions
from src.training import write_json


def main() -> None:
    with open(ROOT / "configs" / "experiments.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected_count = int(config["gallery"]["final_count"])
    candidate_count = int(config["gallery"]["candidate_count"])
    manifest_path = ROOT / "galeria" / "manifest.csv"
    provenance_path = ROOT / "galeria" / "provenance.json"
    latent_path = ROOT / "galeria" / "latents.npz"
    for required in (manifest_path, provenance_path, latent_path):
        if not required.exists():
            raise FileNotFoundError(f"Falta {required.relative_to(ROOT)}")

    manifest = pd.read_csv(manifest_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    latents = np.load(latent_path)
    errors: list[str] = []
    if len(manifest) != expected_count:
        errors.append(f"manifest tiene {len(manifest)} filas, se esperaban {expected_count}")
    if len(set(manifest["candidate_index"])) != expected_count:
        errors.append("candidate_index no es único")
    if manifest["sha256"].nunique() != expected_count:
        errors.append("la galería contiene PNG duplicados")
    if not np.allclose(manifest["selection_rate"], expected_count / candidate_count):
        errors.append("tasa de selección incorrecta")
    if latents["z"].shape != (expected_count, int(config["training"]["latent_dim"]), 1, 1):
        errors.append(f"forma de z incorrecta: {latents['z'].shape}")

    stored_images: list[np.ndarray] = []
    for row in manifest.itertuples(index=False):
        image_path = ROOT / "galeria" / row.file
        if not image_path.exists():
            errors.append(f"falta {row.file}")
            continue
        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != (64, 64):
                errors.append(f"{row.file}: modo/tamaño {image.mode} {image.size}")
            stored_images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        if sha256_file(image_path) != row.sha256:
            errors.append(f"{row.file}: SHA-256 no coincide")

    training = config["training"]
    dimensions = ModelDimensions(
        latent_dim=int(training["latent_dim"]),
        image_channels=int(config["project"]["channels"]),
        generator_features=int(training["generator_features"]),
        discriminator_features=int(training["discriminator_features"]),
    )
    checkpoint_path = ROOT / str(provenance["checkpoint"])
    if sha256_file(checkpoint_path) != provenance["checkpoint_sha256"]:
        errors.append("el checkpoint cambió desde la generación")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    generator = Generator(dimensions).eval()
    generator.load_state_dict(checkpoint["generator"])
    with torch.inference_mode():
        regenerated = tensors_to_uint8(generator(torch.from_numpy(latents["z"])))
    stored = np.stack(stored_images) if len(stored_images) == expected_count else None
    max_pixel_delta = int(np.abs(regenerated.astype(int) - stored.astype(int)).max()) if stored is not None else -1
    if max_pixel_delta != 0:
        errors.append(f"la regeneración difiere por hasta {max_pixel_delta} niveles RGB")

    validation = {
        "status": "passed" if not errors else "failed",
        "image_count": len(manifest),
        "unique_image_hashes": int(manifest["sha256"].nunique()) if "sha256" in manifest else 0,
        "selection_rate": expected_count / candidate_count,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "regeneration_max_pixel_delta": max_pixel_delta,
        "errors": errors,
    }
    write_json(ROOT / "artifacts" / "gallery" / "validation.json", validation)
    if errors:
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))
    provenance["status"] = "validated"
    write_json(provenance_path, provenance)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
