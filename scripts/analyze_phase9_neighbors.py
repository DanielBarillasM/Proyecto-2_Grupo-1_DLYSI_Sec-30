"""Ejecuta la auditoría de vecinos sobre las 16 muestras del ruido fijo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    build_resnet18_feature_extractor,
    extract_resnet18_features,
    image_statistics,
    nearest_training_neighbors,
    paired_pixel_mse,
    sha256_file,
    tensors_to_uint8,
)
from src.models import Generator, ModelDimensions
from src.training import write_json


OUTPUT = ROOT / "artifacts" / "phase9"
SCREENING_COSINE = 0.95
SCREENING_MSE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fue solicitada, pero no está disponible")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_training_images(manifest: pd.DataFrame) -> np.ndarray:
    images: list[np.ndarray] = []
    for relative_path in manifest["path"]:
        with Image.open(ROOT / str(relative_path)) as image:
            images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return np.stack(images)


def save_neighbor_audit(
    generated: np.ndarray,
    neighbors: np.ndarray,
    similarities: np.ndarray,
    pixel_mse: np.ndarray,
    output_path: Path,
) -> None:
    pairs_per_row = 4
    rows = int(np.ceil(len(generated) / pairs_per_row))
    fig, axes = plt.subplots(
        rows, pairs_per_row * 2, figsize=(20, 3.25 * rows), constrained_layout=True
    )
    for axis in axes.ravel():
        axis.axis("off")
    for index, (sample, neighbor, similarity, mse) in enumerate(
        zip(generated, neighbors, similarities, pixel_mse)
    ):
        row = index // pairs_per_row
        column = (index % pairs_per_row) * 2
        axes[row, column].imshow(sample, interpolation="nearest")
        axes[row, column].set_title(
            f"F{index + 1:02d} · GAN", color="#4A4E8F", fontsize=10, fontweight="bold"
        )
        axes[row, column + 1].imshow(neighbor, interpolation="nearest")
        axes[row, column + 1].set_title(
            f"vecino real\ncos={similarity:.4f} · MSE={mse:.4f}", fontsize=8.5
        )
    fig.suptitle(
        "Fase 9 · ruido fijo frente a su vecino de entrenamiento",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    selected = json.loads(
        (ROOT / "artifacts" / "phase8" / "selected_model.json").read_text(encoding="utf-8")
    )
    experiment = str(selected["experiment"])
    checkpoint_path = ROOT / str(selected["checkpoint"])
    target_epoch = int(config["training"]["epochs"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(checkpoint["epoch"])
    if checkpoint_epoch < target_epoch:
        raise RuntimeError(
            f"Checkpoint obsoleto: época interna {checkpoint_epoch}/{target_epoch}. "
            "Restaure latest.pt desde el respaldo de Colab antes de ejecutar la fase 9."
        )
    raw_experiment = checkpoint.get("experiment", experiment)
    checkpoint_experiment = str(
        raw_experiment.get("id", "") if isinstance(raw_experiment, dict) else raw_experiment
    )
    if checkpoint_experiment != experiment:
        raise RuntimeError("El checkpoint no pertenece al experimento seleccionado")

    training = config["training"]
    dimensions = ModelDimensions(
        latent_dim=int(training["latent_dim"]),
        image_channels=int(config["project"]["channels"]),
        generator_features=int(training["generator_features"]),
        discriminator_features=int(training["discriminator_features"]),
    )
    generator = Generator(dimensions).eval()
    generator.load_state_dict(checkpoint["generator"])
    fixed_noise = checkpoint["fixed_noise"].detach().cpu()
    with torch.inference_mode():
        generated = tensors_to_uint8(generator(fixed_noise))

    manifest_path = ROOT / "data" / "processed" / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    expected_rows = int(config["project"]["dataset_size"])
    if len(manifest) != expected_rows or manifest["sha256"].duplicated().any():
        raise RuntimeError("El manifiesto de entrenamiento no supera la validación de fase 9")
    if manifest.isna().any().any():
        raise RuntimeError("El manifiesto contiene valores faltantes")
    training_images = load_training_images(manifest)

    device = resolve_device(args.device)
    feature_model, weights_url = build_resnet18_feature_extractor(device)
    cache_path = ROOT / "data" / "processed" / "resnet18_training_features.npz"
    manifest_hash = sha256_file(manifest_path)
    training_features: np.ndarray
    cache_used = False
    if cache_path.exists():
        cached = np.load(cache_path)
        if (
            str(cached["manifest_sha256"].item()) == manifest_hash
            and int(cached["rows"].item()) == expected_rows
        ):
            training_features = cached["normalized_features"].astype(np.float32)
            cache_used = True
        else:
            cache_path.unlink()
    if not cache_used:
        _, training_features = extract_resnet18_features(
            training_images, feature_model, device, args.feature_batch_size
        )
        np.savez_compressed(
            cache_path,
            normalized_features=training_features,
            manifest_sha256=np.asarray(manifest_hash),
            rows=np.asarray(expected_rows, dtype=np.int64),
        )
    _, generated_features = extract_resnet18_features(
        generated, feature_model, device, args.feature_batch_size
    )
    neighbor_index, similarities = nearest_training_neighbors(
        generated_features, training_features
    )
    neighbors = training_images[neighbor_index]
    pixel_mse = paired_pixel_mse(generated, neighbors)
    foreground, contrast = image_statistics(generated)
    review_flag = (similarities >= SCREENING_COSINE) & (pixel_mse <= SCREENING_MSE)
    exact_duplicate = pixel_mse == 0

    results = pd.DataFrame(
        {
            "sample_id": [f"F{index + 1:02d}" for index in range(len(generated))],
            "fixed_noise_index": np.arange(len(generated), dtype=np.int64),
            "nearest_train_index": neighbor_index,
            "nearest_train_image_id": manifest.iloc[neighbor_index]["image_id"].to_numpy(),
            "nearest_train_path": manifest.iloc[neighbor_index]["path"].to_numpy(),
            "cosine_similarity": similarities,
            "pixel_mse": pixel_mse,
            "foreground_ratio": foreground,
            "contrast": contrast,
            "screening_flag": review_flag,
            "exact_pixel_duplicate": exact_duplicate,
        }
    )
    if results.isna().any().any() or not np.isfinite(
        results.select_dtypes(include="number").to_numpy()
    ).all():
        raise RuntimeError("La tabla de vecinos contiene valores inválidos")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        OUTPUT / "fixed_noise_neighbors.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    save_neighbor_audit(
        generated,
        neighbors,
        similarities,
        pixel_mse,
        OUTPUT / "fixed_noise_neighbors.png",
    )
    summary = {
        "status": "passed",
        "phase": 9,
        "experiment": experiment,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "sample_count": int(len(results)),
        "training_image_count": int(len(manifest)),
        "feature_extractor": "ResNet18 IMAGENET1K_V1 sin capa final",
        "feature_weights_url": weights_url,
        "device": str(device),
        "training_feature_cache_used": cache_used,
        "cosine_similarity": {
            "mean": float(results["cosine_similarity"].mean()),
            "median": float(results["cosine_similarity"].median()),
            "min": float(results["cosine_similarity"].min()),
            "max": float(results["cosine_similarity"].max()),
        },
        "pixel_mse": {
            "mean": float(results["pixel_mse"].mean()),
            "median": float(results["pixel_mse"].median()),
            "min": float(results["pixel_mse"].min()),
            "max": float(results["pixel_mse"].max()),
        },
        "screening_rule": {
            "cosine_similarity_gte": SCREENING_COSINE,
            "pixel_mse_lte": SCREENING_MSE,
            "interpretation": "bandera de revisión, no prueba automática de memorización",
        },
        "screening_flag_count": int(results["screening_flag"].sum()),
        "exact_pixel_duplicate_count": int(results["exact_pixel_duplicate"].sum()),
        "scope": "16 muestras de ruido fijo; la fase 10 evaluará 200 candidatos",
    }
    write_json(OUTPUT / "phase9_summary.json", summary)
    print(results.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
