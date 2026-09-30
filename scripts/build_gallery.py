"""Construye la galería 10/200 y su prueba de vecinos desde un checkpoint final."""

from __future__ import annotations

import argparse
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

from src.evaluation import (
    build_resnet18_feature_extractor,
    derive_quality_bounds,
    evaluate_quality,
    extract_resnet18_features,
    generate_candidates,
    greedy_diverse_selection,
    nearest_training_neighbors,
    paired_pixel_mse,
    save_gallery_grid,
    save_neighbor_figure,
    sha256_file,
)
from src.models import Generator, ModelDimensions
from src.training import write_json


NAMES_AND_ROLES = (
    ("Aelira", "Oráculo del Velo"),
    ("Bram", "Guardián rúnico"),
    ("Cyran", "Arcanista de ceniza"),
    ("Delyra", "Exploradora del musgo"),
    ("Edrik", "Alquimista solar"),
    ("Faelor", "Centinela de amatista"),
    ("Ilyne", "Tejedora de grietas"),
    ("Kael", "Custodio de la brasa"),
    ("Mireth", "Cartógrafa astral"),
    ("Nyra", "Vigía del abismo"),
)


def parse_args(experiment_ids: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=experiment_ids)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
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


def require_empty_or_known_gallery(overwrite: bool) -> None:
    manifest_path = ROOT / "galeria" / "manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            "galeria/manifest.csv ya existe. Use --overwrite solo si desea regenerar la galería."
        )


def main() -> None:
    with open(ROOT / "configs" / "experiments.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiments = {str(item["id"]): item for item in config["experiments"]}
    args = parse_args(list(experiments))
    device = resolve_device(args.device)
    require_empty_or_known_gallery(args.overwrite)

    target_epoch = int(config["training"]["epochs"])
    status_path = ROOT / "artifacts" / "experiments" / args.experiment / "status.json"
    checkpoint_path = ROOT / "checkpoints" / args.experiment / "latest.pt"
    if not status_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("Falta el estado o checkpoint del experimento seleccionado")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if int(status["completed_epochs"]) < target_epoch:
        raise RuntimeError(
            f"Fase 5 bloqueada: {args.experiment} tiene "
            f"{status['completed_epochs']}/{target_epoch} épocas. No se escribirá galeria/."
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) < target_epoch:
        raise RuntimeError("El checkpoint no corresponde a un entrenamiento final")
    checkpoint_hash = sha256_file(checkpoint_path)
    phase10_summary_path = ROOT / "artifacts" / "phase10" / "phase10_summary.json"
    if not phase10_summary_path.exists():
        raise FileNotFoundError("La fase 10 debe completarse antes de construir la galería")
    phase10_summary = json.loads(phase10_summary_path.read_text(encoding="utf-8"))
    if phase10_summary.get("status") != "passed":
        raise RuntimeError("La fase 10 no está aprobada")
    if phase10_summary.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("La fase 10 corresponde a un checkpoint distinto")
    training = config["training"]
    dimensions = ModelDimensions(
        latent_dim=int(training["latent_dim"]),
        image_channels=int(config["project"]["channels"]),
        generator_features=int(training["generator_features"]),
        discriminator_features=int(training["discriminator_features"]),
    )
    generator = Generator(dimensions)
    generator.load_state_dict(payload["generator"])

    candidate_count = int(config["gallery"]["candidate_count"])
    final_count = int(config["gallery"]["final_count"])
    gallery_seed = int(config["gallery"]["seed"])
    candidates, latents = generate_candidates(
        generator, dimensions.latent_dim, candidate_count, gallery_seed
    )

    manifest = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv")
    training_images = load_training_images(manifest)
    quality_bounds = derive_quality_bounds(training_images)
    foreground, contrast, eligible, quality_score = evaluate_quality(
        candidates, quality_bounds
    )
    if int(eligible.sum()) < final_count:
        raise RuntimeError(
            f"Solo {int(eligible.sum())}/{candidate_count} candidatos superaron filtros técnicos; "
            "no se fabricará una galería final. Revise el entrenamiento."
        )

    feature_model, weights_url = build_resnet18_feature_extractor(device)
    _, training_features = extract_resnet18_features(
        training_images, feature_model, device, args.feature_batch_size
    )
    _, candidate_features = extract_resnet18_features(
        candidates, feature_model, device, args.feature_batch_size
    )
    neighbor_index, nearest_similarity = nearest_training_neighbors(
        candidate_features, training_features
    )
    neighbor_images = training_images[neighbor_index]
    pixel_mse = paired_pixel_mse(candidates, neighbor_images)
    selected, selection_diversity = greedy_diverse_selection(
        candidate_features,
        eligible,
        nearest_similarity,
        quality_score,
        final_count,
    )
    phase10_selected = np.asarray(
        phase10_summary["selected_candidate_indices"], dtype=np.int64
    )
    if not np.array_equal(selected, phase10_selected):
        raise RuntimeError(
            "La selección de fase 11 no reproduce exactamente los índices de fase 10"
        )

    artifact_dir = ROOT / "artifacts" / "gallery"
    gallery_dir = ROOT / "galeria"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    candidate_metrics = pd.DataFrame(
        {
            "candidate_index": np.arange(candidate_count),
            "foreground_ratio": foreground,
            "contrast": contrast,
            "eligible": eligible,
            "quality_score": quality_score,
            "nearest_train_index": neighbor_index,
            "nearest_cosine_similarity": nearest_similarity,
            "nearest_pixel_mse": pixel_mse,
            "selected": np.isin(np.arange(candidate_count), selected),
        }
    )
    candidate_metrics.to_csv(artifact_dir / "candidate_metrics.csv", index=False)

    names = [item[0] for item in NAMES_AND_ROLES]
    roles = [item[1] for item in NAMES_AND_ROLES]
    selected_images = candidates[selected]
    selected_neighbors = neighbor_images[selected]
    rows: list[dict[str, object]] = []
    for rank, candidate_index in enumerate(selected, start=1):
        output_name = f"personaje_{rank:02d}.png"
        output_path = gallery_dir / output_name
        Image.fromarray(candidates[candidate_index], mode="RGB").save(output_path)
        training_row = manifest.iloc[int(neighbor_index[candidate_index])]
        rows.append(
            {
                "rank": rank,
                "file": output_name,
                "name": names[rank - 1],
                "role": roles[rank - 1],
                "experiment": args.experiment,
                "checkpoint_epoch": int(payload["epoch"]),
                "candidate_index": int(candidate_index),
                "gallery_seed": gallery_seed,
                "sha256": sha256_file(output_path),
                "foreground_ratio": float(foreground[candidate_index]),
                "contrast": float(contrast[candidate_index]),
                "nearest_train_image_id": training_row["image_id"],
                "nearest_train_path": training_row["path"],
                "nearest_train_sha256": training_row["sha256"],
                "nearest_cosine_similarity": float(nearest_similarity[candidate_index]),
                "nearest_pixel_mse": float(pixel_mse[candidate_index]),
                "selection_diversity": float(selection_diversity[rank - 1]),
                "selection_rate": final_count / candidate_count,
            }
        )
    gallery_manifest = pd.DataFrame(rows)
    gallery_manifest.to_csv(gallery_dir / "manifest.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        gallery_dir / "latents.npz",
        z=latents[selected],
        candidate_indices=selected,
        gallery_seed=np.asarray([gallery_seed], dtype=np.int64),
    )

    save_gallery_grid(selected_images, names, artifact_dir / "final_gallery_grid.png")
    save_neighbor_figure(
        selected_images,
        selected_neighbors,
        names,
        nearest_similarity[selected],
        pixel_mse[selected],
        artifact_dir / "nearest_neighbors.png",
    )
    provenance = {
        "status": "generated_pending_validation",
        "source": "Generator entrenado por el equipo; sin imágenes de generadores externos.",
        "experiment": args.experiment,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(payload["epoch"]),
        "phase10_summary": phase10_summary_path.relative_to(ROOT).as_posix(),
        "phase10_summary_sha256": sha256_file(phase10_summary_path),
        "selected_candidate_indices": selected.tolist(),
        "phase10_selection_match": True,
        "dataset_manifest_sha256": sha256_file(ROOT / "data" / "processed" / "manifest.csv"),
        "candidate_count": candidate_count,
        "selected_count": final_count,
        "selection_rate": final_count / candidate_count,
        "gallery_seed": gallery_seed,
        "feature_extractor": "ResNet18 IMAGENET1K_V1 sin capa final",
        "feature_weights_url": weights_url,
        "quality_bounds": quality_bounds.to_dict(),
        "eligible_candidates": int(eligible.sum()),
        "device_for_features": str(device),
        "device_for_generation": "cpu",
    }
    write_json(gallery_dir / "provenance.json", provenance)
    print(gallery_manifest.to_string(index=False))
    print("Ejecute: python scripts/validate_gallery.py")


if __name__ == "__main__":
    main()
