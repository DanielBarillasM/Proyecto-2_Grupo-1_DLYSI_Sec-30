"""Evaluación reproducible, selección diversa y figuras para la galería GAN."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


BACKGROUND_RGB = (23, 26, 36)


@dataclass(frozen=True)
class QualityBounds:
    foreground_min: float
    foreground_max: float
    contrast_min: float
    contrast_max: float
    foreground_median: float
    contrast_median: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensors_to_uint8(images: torch.Tensor) -> np.ndarray:
    """Convierte un lote NCHW en [-1, 1] a RGB uint8 NHWC."""

    if images.ndim != 4 or images.shape[1:] != (3, 64, 64):
        raise ValueError(f"Se esperaba [N, 3, 64, 64], se recibió {list(images.shape)}")
    scaled = images.detach().cpu().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
    return scaled.permute(0, 2, 3, 1).numpy()


def generate_candidates(
    generator: nn.Module,
    latent_dim: int,
    count: int,
    seed: int,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera candidatos y latentes en CPU para regeneración bit a bit."""

    if count < 1 or batch_size < 1:
        raise ValueError("count y batch_size deben ser positivos")
    generator = generator.to("cpu").eval()
    noise_generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(count, latent_dim, 1, 1, generator=noise_generator)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            batches.append(tensors_to_uint8(generator(latents[start : start + batch_size])))
    return np.concatenate(batches), latents.numpy().astype(np.float32)


def image_statistics(
    images: np.ndarray,
    background_rgb: Sequence[int] = BACKGROUND_RGB,
    foreground_distance: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula ocupación aproximada y contraste por imagen RGB uint8."""

    array = np.asarray(images)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Se esperaba [N, H, W, 3], se recibió {list(array.shape)}")
    normalized = array.astype(np.float32) / 255.0
    background = np.asarray(background_rgb, dtype=np.float32) / 255.0
    distance = np.linalg.norm(normalized - background, axis=-1)
    foreground_ratio = (distance > foreground_distance).mean(axis=(1, 2))
    luminance = (
        0.2126 * normalized[..., 0]
        + 0.7152 * normalized[..., 1]
        + 0.0722 * normalized[..., 2]
    )
    contrast = luminance.std(axis=(1, 2))
    return foreground_ratio.astype(np.float32), contrast.astype(np.float32)


def derive_quality_bounds(real_images: np.ndarray) -> QualityBounds:
    foreground, contrast = image_statistics(real_images)
    return QualityBounds(
        foreground_min=max(0.0, float(np.quantile(foreground, 0.01)) - 0.03),
        foreground_max=min(1.0, float(np.quantile(foreground, 0.99)) + 0.05),
        contrast_min=max(0.0, float(np.quantile(contrast, 0.01)) * 0.70),
        contrast_max=min(1.0, float(np.quantile(contrast, 0.99)) * 1.30),
        foreground_median=float(np.median(foreground)),
        contrast_median=float(np.median(contrast)),
    )


def evaluate_quality(
    images: np.ndarray, bounds: QualityBounds
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    foreground, contrast = image_statistics(images)
    eligible = (
        (foreground >= bounds.foreground_min)
        & (foreground <= bounds.foreground_max)
        & (contrast >= bounds.contrast_min)
        & (contrast <= bounds.contrast_max)
    )
    foreground_scale = max(bounds.foreground_max - bounds.foreground_min, 1e-8)
    contrast_scale = max(bounds.contrast_max - bounds.contrast_min, 1e-8)
    deviation = 0.5 * np.abs(foreground - bounds.foreground_median) / foreground_scale
    deviation += 0.5 * np.abs(contrast - bounds.contrast_median) / contrast_scale
    quality_score = np.clip(1.0 - deviation, 0.0, 1.0)
    return foreground, contrast, eligible, quality_score.astype(np.float32)


def build_resnet18_feature_extractor(device: torch.device) -> tuple[nn.Module, str]:
    """Carga ResNet18 preentrenada; nunca sustituye silenciosamente pesos aleatorios."""

    weights = ResNet18_Weights.DEFAULT
    try:
        model = resnet18(weights=weights)
    except Exception as error:
        raise RuntimeError(
            "No fue posible cargar los pesos preentrenados de ResNet18. "
            "Conecte el equipo una vez a Internet y vuelva a ejecutar la fase 5; "
            "no se usarán pesos aleatorios como sustituto."
        ) from error
    model.fc = nn.Identity()
    model.eval().to(device)
    return model, weights.url


def extract_resnet18_features(
    images: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrae embeddings crudos y normalizados de RGB uint8."""

    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    raw_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            batch = batch.permute(0, 3, 1, 2).div(255.0)
            batch = functional.interpolate(
                batch, size=(224, 224), mode="bilinear", align_corners=False
            )
            raw_batches.append(model((batch - mean) / std).cpu().numpy().astype(np.float32))
    raw = np.concatenate(raw_batches)
    normalized = raw / np.clip(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12, None)
    return raw, normalized.astype(np.float32)


def nearest_training_neighbors(
    candidate_features: np.ndarray,
    training_features: np.ndarray,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Busca máxima similitud coseno sin materializar matrices innecesariamente grandes."""

    candidate_features = np.asarray(candidate_features, dtype=np.float32)
    training_features = np.asarray(training_features, dtype=np.float32)
    best_index = np.full(len(candidate_features), -1, dtype=np.int64)
    best_similarity = np.full(len(candidate_features), -np.inf, dtype=np.float32)
    for start in range(0, len(training_features), chunk_size):
        similarities = candidate_features @ training_features[start : start + chunk_size].T
        local_index = similarities.argmax(axis=1)
        local_similarity = similarities[np.arange(len(candidate_features)), local_index]
        improve = local_similarity > best_similarity
        best_similarity[improve] = local_similarity[improve]
        best_index[improve] = start + local_index[improve]
    return best_index, best_similarity


def paired_pixel_mse(candidates: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    candidate_float = candidates.astype(np.float32) / 255.0
    neighbor_float = neighbors.astype(np.float32) / 255.0
    return np.mean((candidate_float - neighbor_float) ** 2, axis=(1, 2, 3))


def greedy_diverse_selection(
    features: np.ndarray,
    eligible: np.ndarray,
    nearest_similarity: np.ndarray,
    quality_score: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Selecciona por novedad, calidad técnica y separación entre elegidos."""

    pool = np.flatnonzero(eligible)
    if len(pool) < count:
        raise ValueError(f"Solo {len(pool)} de {count} candidatos requeridos superan los filtros")
    novelty = np.clip(1.0 - nearest_similarity, 0.0, 2.0)
    novelty_scale = max(float(novelty[pool].max()), 1e-8)
    novelty_normalized = novelty / novelty_scale
    base_score = 0.70 * novelty_normalized + 0.30 * quality_score

    selected = [int(pool[np.argmax(base_score[pool])])]
    diversity_at_selection = [1.0]
    while len(selected) < count:
        remaining = np.asarray([index for index in pool if index not in selected], dtype=np.int64)
        similarity_to_selected = features[remaining] @ features[np.asarray(selected)].T
        min_distance = 1.0 - similarity_to_selected.max(axis=1)
        distance_scale = max(float(min_distance.max()), 1e-8)
        score = (
            0.55 * (min_distance / distance_scale)
            + 0.30 * novelty_normalized[remaining]
            + 0.15 * quality_score[remaining]
        )
        winner_position = int(np.argmax(score))
        selected.append(int(remaining[winner_position]))
        diversity_at_selection.append(float(min_distance[winner_position]))
    return np.asarray(selected, dtype=np.int64), np.asarray(diversity_at_selection, dtype=np.float32)


def save_gallery_grid(
    images: np.ndarray,
    names: Sequence[str],
    output_path: str | Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.7), facecolor="#171A24")
    for axis, image, name in zip(axes.ravel(), images, names):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(name, color="#E6E6FA", fontsize=10, fontweight="bold")
        axis.axis("off")
    fig.suptitle("Eryndor · galería final 10/200", color="white", fontsize=17, fontweight="bold")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_neighbor_figure(
    generated: np.ndarray,
    neighbors: np.ndarray,
    names: Sequence[str],
    similarities: np.ndarray,
    pixel_mse: np.ndarray,
    output_path: str | Path,
) -> None:
    rows = int(np.ceil(len(generated) / 2))
    fig, axes = plt.subplots(rows, 4, figsize=(13, 3.05 * rows), constrained_layout=True)
    for axis in axes.ravel():
        axis.axis("off")
    for index, (generated_image, neighbor, name, similarity, mse) in enumerate(
        zip(generated, neighbors, names, similarities, pixel_mse)
    ):
        row = index // 2
        column = (index % 2) * 2
        axes[row, column].imshow(generated_image, interpolation="nearest")
        axes[row, column].set_title(
            f"{name} · generado", fontweight="bold", color="#4A4E8F"
        )
        axes[row, column + 1].imshow(neighbor, interpolation="nearest")
        axes[row, column + 1].set_title(
            f"Vecino de entrenamiento\ncos={similarity:.4f} · MSE={mse:.4f}", fontsize=9
        )
    fig.suptitle("Prueba de vecinos más cercanos", fontsize=17, fontweight="bold")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
