"""Valida la coherencia integral de la entrega final de Eryndor."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "delivery" / "validation.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "b032a9fa92e719b0500e54a7ee14d5c1a691386bb32459d38807305b699b7f80"
)
CHECKPOINT_URL = (
    "https://github.com/DanielBarillasM/Proyecto-2_Grupo-1_DLYSI_Sec-30/"
    "releases/download/v1.0.0/latest.pt"
)
EXPECTED_HYPOTHESES = (
    "Creemos que la DCGAN con BCE no saturante producirá siluetas reconocibles porque",
    "Creemos que hinge loss mejorará la nitidez y la separación entre rasgos porque",
    "Creemos que la normalización espectral estabilizará al discriminador porque",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checks: dict[str, object] = {}
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    required = [
        "README.md",
        "configs/experiments.yaml",
        "notebooks/01_proyecto_gan.ipynb",
        "notebooks/02_entrenamiento_colab.ipynb",
        "presentation/presentacion.html",
        "presentation/presentacion.pdf",
        "docs/MATRIZ_EVIDENCIAS.md",
        "report/informe_final_eryndor.tex",
        "report/informe_final_eryndor.pdf",
        "galeria/manifest.csv",
        "galeria/latents.npz",
        "galeria/provenance.json",
        "artifacts/gallery/validation.json",
        "artifacts/gallery/final_gallery_grid.png",
        "artifacts/gallery/nearest_neighbors.png",
    ]
    missing = [item for item in required if not (ROOT / item).exists()]
    require(not missing, f"Archivos requeridos ausentes: {missing}")
    checks["required_files"] = {"expected": len(required), "missing": missing}

    config = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
    dataset = json.loads(
        (ROOT / "artifacts/metrics/dataset_summary.json").read_text(encoding="utf-8")
    )
    require(dataset["image_count"] == 4096, "El dataset no contiene 4,096 imágenes")
    require(dataset["duplicate_hashes"] == 0, "El dataset contiene hashes duplicados")
    require(dataset["missing_files"] == 0, "El dataset tiene archivos faltantes")
    checks["dataset"] = {
        "images": dataset["image_count"],
        "duplicates": dataset["duplicate_hashes"],
        "missing": dataset["missing_files"],
        "credited_layers": 254,
    }

    comparison = pd.read_csv(ROOT / "artifacts/experiments/comparison_summary.csv")
    epoch_frames = []
    for experiment in config["experiments"]:
        experiment_id = experiment["id"]
        frame = pd.read_csv(
            ROOT / "artifacts/experiments" / experiment_id / "epoch_metrics.csv"
        )
        frame.insert(0, "experiment", experiment_id)
        epoch_frames.append(frame)
    epochs = pd.concat(epoch_frames, ignore_index=True)
    target_epochs = int(config["training"]["epochs"])
    require(len(comparison) == 3, "La comparación no contiene tres experimentos")
    require((comparison["completed_epochs"] == target_epochs).all(), "A/B/C no están en 60/60")
    require(len(epochs) == 180, "Se esperaban 180 registros por época")
    require(not epochs.duplicated(["experiment", "epoch"]).any(), "Hay épocas duplicadas")
    numeric = epochs.select_dtypes(include="number").to_numpy(dtype=float)
    require(bool(np.isfinite(numeric).all()), "Hay métricas no finitas")
    selected = comparison.loc[comparison["experiment"] == "bce_spectral_norm"].iloc[0]
    hinge = comparison.loc[comparison["experiment"] == "hinge_loss"].iloc[0]
    require(
        float(selected["fixed_pairwise_l2"]) > float(hinge["fixed_pairwise_l2"]),
        "La decisión C no coincide con las métricas de diversidad",
    )
    checks["experiments"] = {
        "runs": len(comparison),
        "epoch_rows": len(epochs),
        "target_epochs": target_epochs,
        "selected": "bce_spectral_norm",
        "selected_diversity": float(selected["fixed_pairwise_l2"]),
        "hinge_diversity": float(hinge["fixed_pairwise_l2"]),
    }

    phase9 = json.loads((ROOT / "artifacts/phase9/phase9_summary.json").read_text(encoding="utf-8"))
    phase10 = json.loads((ROOT / "artifacts/phase10/phase10_summary.json").read_text(encoding="utf-8"))
    require(phase9["status"] == "passed", "Fase 9 no aprobada")
    require(phase9["exact_pixel_duplicate_count"] == 0, "Fase 9 detectó duplicados")
    require(phase10["status"] == "passed", "Fase 10 no aprobada")
    require(phase10["candidate_count"] == 200, "Fase 10 no generó 200 candidatos")
    require(phase10["selected_count"] == 10, "Fase 10 no seleccionó diez candidatos")
    require(math.isclose(float(phase10["selection_rate"]), 0.05), "La tasa no es 5%")
    checks["selection"] = {
        "phase9_exact_duplicates": phase9["exact_pixel_duplicate_count"],
        "phase9_screening_flags": phase9["screening_flag_count"],
        "candidates": phase10["candidate_count"],
        "eligible": phase10["eligible_candidates"],
        "selected": phase10["selected_count"],
        "rate": phase10["selection_rate"],
    }

    manifest = pd.read_csv(ROOT / "galeria/manifest.csv")
    validation = json.loads(
        (ROOT / "artifacts/gallery/validation.json").read_text(encoding="utf-8")
    )
    provenance = json.loads((ROOT / "galeria/provenance.json").read_text(encoding="utf-8"))
    latents = np.load(ROOT / "galeria/latents.npz")["z"]
    require(len(manifest) == 10, "El manifiesto final no contiene diez personajes")
    require(manifest["sha256"].nunique() == 10, "Los PNG finales no tienen hashes únicos")
    require(tuple(latents.shape) == (10, 128, 1, 1), "Forma inesperada de latents.npz")
    require(validation["status"] == "passed", "La validación de galería no aprobó")
    require(validation["phase10_selection_match"], "Galería y fase 10 no coinciden")
    require(validation["regeneration_max_pixel_delta"] == 0, "La regeneración no es exacta")
    expected_indices = list(phase10["selected_candidate_indices"])
    require(manifest["candidate_index"].tolist() == expected_indices, "Orden de selección inconsistente")
    image_hashes = []
    for row in manifest.itertuples(index=False):
        image_path = ROOT / "galeria" / row.file
        require(image_path.exists(), f"PNG ausente: {row.file}")
        if image_path.exists():
            actual_hash = sha256(image_path)
            image_hashes.append(actual_hash)
            require(actual_hash == row.sha256, f"Hash inconsistente: {row.file}")
    hardest = manifest.loc[manifest["nearest_cosine_similarity"].idxmax()]
    require(hardest["name"] == "Bram", "La reflexión no coincide con el vecino más cercano")
    checks["gallery"] = {
        "images": len(manifest),
        "unique_hashes": len(set(image_hashes)),
        "latents_shape": list(latents.shape),
        "phase10_match": validation["phase10_selection_match"],
        "regeneration_max_pixel_delta": validation["regeneration_max_pixel_delta"],
        "hardest_character": hardest["name"],
        "hardest_cosine": float(hardest["nearest_cosine_similarity"]),
        "hardest_mse": float(hardest["nearest_pixel_mse"]),
    }

    checkpoint = ROOT / provenance["checkpoint"]
    require(checkpoint.exists(), "El checkpoint final no está disponible localmente")
    checkpoint_hash = sha256(checkpoint) if checkpoint.exists() else None
    require(checkpoint_hash == EXPECTED_CHECKPOINT_SHA256, "SHA-256 del checkpoint no coincide")
    checkpoint_epoch = None
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_epoch = int(payload["epoch"])
        require(checkpoint_epoch == target_epochs, "El checkpoint interno no está en época 60")
    checks["checkpoint"] = {
        "path": provenance["checkpoint"],
        "epoch": checkpoint_epoch,
        "sha256": checkpoint_hash,
        "tracked_by_git": False,
        "public_url": CHECKPOINT_URL,
        "delivery_note": "Peso final C publicado como activo de GitHub Release v1.0.0.",
    }

    notebook = nbformat.read(ROOT / "notebooks/01_proyecto_gan.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    counts = [cell.execution_count for cell in code_cells]
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    require(all(isinstance(value, int) for value in counts), "Hay celdas sin ejecutar")
    require(counts == sorted(counts) and len(set(counts)) == len(counts), "Orden de ejecución inválido")
    require(not error_outputs, "El notebook contiene salidas de error")
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    missing_hypotheses = [
        hypothesis for hypothesis in EXPECTED_HYPOTHESES if hypothesis not in markdown
    ]
    require(
        not missing_hypotheses,
        f"El notebook no contiene las hipótesis previas completas: {missing_hypotheses}",
    )
    require(
        "Se eligió DCGAN porque" in markdown,
        "El notebook no justifica explícitamente la elección de DCGAN",
    )
    checks["notebook"] = {
        "code_cells": len(code_cells),
        "executed_cells": sum(isinstance(value, int) for value in counts),
        "error_outputs": len(error_outputs),
        "execution_ordered": counts == sorted(counts),
        "preregistered_hypotheses": len(EXPECTED_HYPOTHESES) - len(missing_hypotheses),
        "architecture_rationale": "Se eligió DCGAN porque" in markdown,
    }

    html_status = json.loads(
        (ROOT / "artifacts/presentation/html_status.json").read_text(encoding="utf-8")
    )
    pdf_status = json.loads(
        (ROOT / "artifacts/presentation/build_status.json").read_text(encoding="utf-8")
    )
    require(html_status["status"] == "passed", "Presentación HTML no aprobada")
    require(html_status["slides"] == 12 and html_status["gallery_ready"], "HTML no refleja la entrega final")
    require(pdf_status["status"] == "passed", "Presentación PDF no aprobada")
    require(pdf_status["page_count"] == 12 and pdf_status["gallery_ready"], "PDF no refleja la entrega final")
    checks["presentation"] = {
        "html_slides": html_status["slides"],
        "pdf_pages": pdf_status["page_count"],
        "gallery_ready": html_status["gallery_ready"] and pdf_status["gallery_ready"],
    }

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stale = [
        "Modelo provisional",
        "galería definitiva permanece pendiente",
        "Trabajo pendiente antes de la entrega",
    ]
    found_stale = [phrase for phrase in stale if phrase in readme]
    require(not found_stale, f"README conserva texto obsoleto: {found_stale}")
    for seed in ("2026", "42", "777", "20261011"):
        require(seed in readme, f"README no documenta la semilla {seed}")
    require("Transparencia sobre IA" in readme, "README no declara uso de IA")
    require(CHECKPOINT_URL in readme, "README no contiene el enlace público del checkpoint")
    require(
        EXPECTED_CHECKPOINT_SHA256 in readme,
        "README no documenta el SHA-256 del checkpoint final",
    )
    checks["documentation"] = {
        "stale_phrases": found_stale,
        "seeds_documented": [2026, 42, 777, 20261011],
        "ai_disclosed": "Transparencia sobre IA" in readme,
        "checkpoint_url": CHECKPOINT_URL,
        "evidence_matrix": "docs/MATRIZ_EVIDENCIAS.md",
        "report_source": "report/informe_final_eryndor.tex",
    }

    result = {
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "errors": errors,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
