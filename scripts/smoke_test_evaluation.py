"""Prueba sintética del filtrado, vecinos y selección sin crear una galería falsa."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    derive_quality_bounds,
    evaluate_quality,
    generate_candidates,
    greedy_diverse_selection,
    nearest_training_neighbors,
    paired_pixel_mse,
)
from src.training import write_json


def normalize(rows: np.ndarray) -> np.ndarray:
    return rows / np.clip(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12, None)


class SyntheticGenerator(nn.Module):
    """Generador mínimo para probar semilla y cuantización, no calidad visual."""

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        base = torch.tanh(noise[:, :3])
        return base.expand(-1, -1, 64, 64)


def main() -> None:
    rng = np.random.default_rng(20261011)
    generated_a, latents_a = generate_candidates(
        SyntheticGenerator(), latent_dim=8, count=12, seed=20261011, batch_size=5
    )
    generated_b, latents_b = generate_candidates(
        SyntheticGenerator(), latent_dim=8, count=12, seed=20261011, batch_size=7
    )
    assert np.array_equal(latents_a, latents_b)
    assert np.array_equal(generated_a, generated_b)
    background = np.asarray((23, 26, 36), dtype=np.uint8)
    real = np.broadcast_to(background, (40, 64, 64, 3)).copy()
    candidates = np.broadcast_to(background, (20, 64, 64, 3)).copy()
    for index, image in enumerate(real):
        width = 18 + index % 8
        left = 32 - width // 2
        image[14:55, left : left + width] = rng.integers(
            45, 220, size=(41, width, 3), dtype=np.uint8
        )
    for index, image in enumerate(candidates):
        width = 18 + index % 7
        left = 32 - width // 2
        image[14:55, left : left + width] = rng.integers(
            45, 220, size=(41, width, 3), dtype=np.uint8
        )

    bounds = derive_quality_bounds(real)
    foreground, contrast, eligible, quality = evaluate_quality(candidates, bounds)
    training_features = normalize(rng.normal(size=(40, 32)).astype(np.float32))
    candidate_features = normalize(rng.normal(size=(20, 32)).astype(np.float32))
    neighbor_index, similarity = nearest_training_neighbors(candidate_features, training_features)
    mse = paired_pixel_mse(candidates, real[neighbor_index])
    selected, diversity = greedy_diverse_selection(
        candidate_features, eligible, similarity, quality, count=10
    )

    assert len(np.unique(selected)) == 10
    assert np.all(eligible[selected])
    assert np.all(np.isfinite(mse))
    assert np.all(diversity >= 0)
    payload = {
        "status": "passed",
        "candidates": int(len(candidates)),
        "eligible": int(eligible.sum()),
        "selected": selected.tolist(),
        "foreground_range": [float(foreground.min()), float(foreground.max())],
        "contrast_range": [float(contrast.min()), float(contrast.max())],
        "nearest_similarity_range": [float(similarity.min()), float(similarity.max())],
        "selection_diversity": diversity.tolist(),
        "deterministic_generation": bool(np.array_equal(generated_a, generated_b)),
        "note": "Datos sintéticos; no son resultados ni imágenes de la galería final.",
    }
    write_json(ROOT / "artifacts" / "gallery" / "pipeline_smoke_test.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
