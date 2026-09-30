"""Genera 200 candidatos y selecciona 10 sin escribir aún la galería final."""

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
from matplotlib.patches import Rectangle


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
    sha256_file,
)
from src.models import Generator, ModelDimensions
from src.training import write_json


OUTPUT = ROOT / "artifacts" / "phase10"


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


def load_training_features(
    images: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    manifest_hash: str,
) -> tuple[np.ndarray, bool]:
    cache_path = ROOT / "data" / "processed" / "resnet18_training_features.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            if (
                str(cached["manifest_sha256"].item()) == manifest_hash
                and int(cached["rows"].item()) == len(images)
            ):
                return cached["normalized_features"].astype(np.float32), True
    _, features = extract_resnet18_features(images, model, device, batch_size)
    np.savez_compressed(
        cache_path,
        normalized_features=features,
        manifest_sha256=np.asarray(manifest_hash),
        rows=np.asarray(len(images), dtype=np.int64),
    )
    return features, False


def save_candidate_overview(
    images: np.ndarray,
    eligible: np.ndarray,
    selected: np.ndarray,
    output_path: Path,
) -> None:
    selected_set = set(selected.tolist())
    columns = 20
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(28, 14.5), facecolor="#171A24")
    for axis in axes.ravel():
        axis.axis("off")
    for index, (axis, image) in enumerate(zip(axes.ravel(), images)):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(
            f"{index:03d}",
            color="#D6A34A" if index in selected_set else "#E6E6FA",
            fontsize=6.5,
            pad=1,
        )
        if index in selected_set:
            axis.add_patch(
                Rectangle((0, 0), 63, 63, fill=False, edgecolor="#D6A34A", linewidth=3)
            )
        elif not bool(eligible[index]):
            axis.add_patch(
                Rectangle((0, 0), 63, 63, fill=False, edgecolor="#B55A62", linewidth=1.5)
            )
    fig.suptitle(
        "Fase 10 · 200 candidatos · oro: seleccionados · rojo: no elegibles",
        color="white",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_selected_grid(
    images: np.ndarray,
    selected_table: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.6), facecolor="#171A24")
    for axis, image, row in zip(axes.ravel(), images, selected_table.itertuples(index=False)):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(
            f"#{row.selection_rank} · candidato {row.candidate_index:03d}\n"
            f"cos={row.nearest_cosine_similarity:.3f} · calidad={row.quality_score:.3f}",
            color="#E6E6FA",
            fontsize=8.5,
        )
        axis.axis("off")
    fig.suptitle(
        "Selección técnica provisional · 10 de 200",
        color="white",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_selection_diagnostics(metrics: pd.DataFrame, output_path: Path) -> None:
    selected = metrics.loc[metrics["selected"]].copy()
    specifications = (
        ("foreground_ratio", "Ocupación"),
        ("contrast", "Contraste"),
        ("nearest_cosine_similarity", "Similitud con vecino"),
        ("quality_score", "Calidad técnica"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (column, title) in zip(axes.ravel(), specifications):
        axis.hist(metrics[column], bins=24, color="#4A4E8F", alpha=0.72, label="200 candidatos")
        for value in selected[column]:
            axis.axvline(value, color="#D6A34A", linewidth=1, alpha=0.8)
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel("Frecuencia")
        axis.grid(alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Fase 10 · distribución y posición de los 10 elegidos", fontsize=17, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    selected_model = json.loads(
        (ROOT / "artifacts" / "phase8" / "selected_model.json").read_text(encoding="utf-8")
    )
    phase9_path = ROOT / "artifacts" / "phase9" / "phase9_summary.json"
    if not phase9_path.exists():
        raise RuntimeError("La fase 9 debe completarse antes de generar los 200 candidatos")
    phase9 = json.loads(phase9_path.read_text(encoding="utf-8"))
    if phase9.get("status") != "passed":
        raise RuntimeError("La fase 9 no está aprobada")

    experiment = str(selected_model["experiment"])
    checkpoint_path = ROOT / str(selected_model["checkpoint"])
    target_epoch = int(config["training"]["epochs"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) < target_epoch:
        raise RuntimeError(
            f"Checkpoint obsoleto: época interna {checkpoint['epoch']}/{target_epoch}"
        )
    checkpoint_hash = sha256_file(checkpoint_path)
    if phase9.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("La fase 9 no corresponde al checkpoint actual")

    training = config["training"]
    dimensions = ModelDimensions(
        latent_dim=int(training["latent_dim"]),
        image_channels=int(config["project"]["channels"]),
        generator_features=int(training["generator_features"]),
        discriminator_features=int(training["discriminator_features"]),
    )
    generator = Generator(dimensions).eval()
    generator.load_state_dict(checkpoint["generator"])
    candidate_count = int(config["gallery"]["candidate_count"])
    final_count = int(config["gallery"]["final_count"])
    gallery_seed = int(config["gallery"]["seed"])
    candidates, _ = generate_candidates(
        generator, dimensions.latent_dim, candidate_count, gallery_seed
    )

    manifest_path = ROOT / "data" / "processed" / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != int(config["project"]["dataset_size"]):
        raise RuntimeError("El dataset no tiene el tamaño pre-registrado")
    if manifest["sha256"].duplicated().any() or manifest.isna().any().any():
        raise RuntimeError("El manifiesto de entrenamiento contiene duplicados o faltantes")
    training_images = load_training_images(manifest)
    bounds = derive_quality_bounds(training_images)
    foreground, contrast, eligible, quality_score = evaluate_quality(candidates, bounds)
    if int(eligible.sum()) < final_count:
        raise RuntimeError(
            f"Solo {int(eligible.sum())}/{candidate_count} candidatos superan los filtros"
        )

    device = resolve_device(args.device)
    feature_model, weights_url = build_resnet18_feature_extractor(device)
    manifest_hash = sha256_file(manifest_path)
    training_features, cache_used = load_training_features(
        training_images,
        feature_model,
        device,
        args.feature_batch_size,
        manifest_hash,
    )
    _, candidate_features = extract_resnet18_features(
        candidates, feature_model, device, args.feature_batch_size
    )
    neighbor_index, neighbor_similarity = nearest_training_neighbors(
        candidate_features, training_features
    )
    pixel_mse = paired_pixel_mse(candidates, training_images[neighbor_index])
    selected, selection_diversity = greedy_diverse_selection(
        candidate_features,
        eligible,
        neighbor_similarity,
        quality_score,
        final_count,
    )

    metrics = pd.DataFrame(
        {
            "candidate_index": np.arange(candidate_count, dtype=np.int64),
            "foreground_ratio": foreground,
            "contrast": contrast,
            "eligible": eligible,
            "quality_score": quality_score,
            "nearest_train_index": neighbor_index,
            "nearest_train_image_id": manifest.iloc[neighbor_index]["image_id"].to_numpy(),
            "nearest_cosine_similarity": neighbor_similarity,
            "nearest_pixel_mse": pixel_mse,
            "novelty_score": np.clip(1.0 - neighbor_similarity, 0.0, 2.0),
            "selected": np.isin(np.arange(candidate_count), selected),
        }
    )
    if len(metrics) != candidate_count or metrics["candidate_index"].duplicated().any():
        raise RuntimeError("La tabla de candidatos no conserva exactamente 200 índices únicos")
    if metrics.isna().any().any() or not np.isfinite(
        metrics.select_dtypes(include="number").to_numpy()
    ).all():
        raise RuntimeError("La tabla de candidatos contiene valores inválidos")

    selected_table = metrics.set_index("candidate_index").loc[selected].reset_index().copy()
    selected_table.insert(0, "selection_rank", np.arange(1, final_count + 1))
    selected_table["selection_diversity"] = selection_diversity
    if len(selected_table) != final_count or not selected_table["eligible"].all():
        raise RuntimeError("La selección no contiene exactamente 10 candidatos elegibles")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        OUTPUT / "candidate_metrics.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    selected_table.to_csv(
        OUTPUT / "selected_candidates.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    save_candidate_overview(candidates, eligible, selected, OUTPUT / "candidate_overview.png")
    save_selected_grid(candidates[selected], selected_table, OUTPUT / "selected_10.png")
    save_selection_diagnostics(metrics, OUTPUT / "selection_diagnostics.png")

    summary = {
        "status": "passed",
        "phase": 10,
        "experiment": experiment,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": checkpoint_hash,
        "candidate_count": candidate_count,
        "eligible_candidates": int(eligible.sum()),
        "selected_count": final_count,
        "selection_rate": final_count / candidate_count,
        "gallery_seed": gallery_seed,
        "selected_candidate_indices": selected.tolist(),
        "feature_extractor": "ResNet18 IMAGENET1K_V1 sin capa final",
        "feature_weights_url": weights_url,
        "device": str(device),
        "training_feature_cache_used": cache_used,
        "quality_bounds": bounds.to_dict(),
        "mean_selected_neighbor_similarity": float(
            selected_table["nearest_cosine_similarity"].mean()
        ),
        "mean_selected_pixel_mse": float(selected_table["nearest_pixel_mse"].mean()),
        "exact_pixel_duplicate_count": int((selected_table["nearest_pixel_mse"] == 0).sum()),
        "selection_rule": {
            "first": "70% novedad frente a entrenamiento + 30% calidad técnica",
            "remaining": "55% diversidad interna + 30% novedad + 15% calidad",
            "human_override": False,
        },
        "phase11_pending": [
            "guardar los diez PNG finales",
            "guardar los vectores z en latents.npz",
            "construir manifest.csv y validar regeneración exacta",
        ],
    }
    write_json(OUTPUT / "phase10_summary.json", summary)
    print(selected_table.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
