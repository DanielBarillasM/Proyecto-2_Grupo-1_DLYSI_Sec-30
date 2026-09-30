"""Prueba determinista de la selección 10/200 sin usar resultados de la GAN."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import greedy_diverse_selection, nearest_training_neighbors
from src.training import write_json


def normalize(array: np.ndarray) -> np.ndarray:
    return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-12, None)


def main() -> None:
    rng = np.random.default_rng(20261011)
    candidate_count = 200
    final_count = 10
    training = normalize(rng.normal(size=(4096, 64)).astype(np.float32))
    candidates = normalize(rng.normal(size=(candidate_count, 64)).astype(np.float32))
    neighbor_index, similarity = nearest_training_neighbors(candidates, training)
    eligible = rng.random(candidate_count) > 0.12
    quality = rng.uniform(0.45, 1.0, size=candidate_count).astype(np.float32)
    selected_a, diversity_a = greedy_diverse_selection(
        candidates, eligible, similarity, quality, final_count
    )
    selected_b, diversity_b = greedy_diverse_selection(
        candidates, eligible, similarity, quality, final_count
    )

    assert len(np.unique(selected_a)) == final_count
    assert np.array_equal(selected_a, selected_b)
    assert np.array_equal(diversity_a, diversity_b)
    assert np.all(eligible[selected_a])
    assert neighbor_index.shape == (candidate_count,)
    assert np.isfinite(similarity).all() and np.isfinite(diversity_a).all()
    payload = {
        "status": "passed",
        "candidate_count": candidate_count,
        "eligible_candidates": int(eligible.sum()),
        "selected_count": final_count,
        "selection_rate": final_count / candidate_count,
        "selected_indices": selected_a.tolist(),
        "deterministic_selection": True,
        "note": "Vectores sintéticos; validan el algoritmo, no la calidad de la GAN.",
    }
    write_json(ROOT / "artifacts" / "phase10" / "smoke_test.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
