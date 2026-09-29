"""Actualiza datos, compila y valida la presentación PDF de 12 diapositivas."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PRESENTATION = ROOT / "presentation"
GENERATED = PRESENTATION / "generated_results.tex"
PDF = PRESENTATION / "presentacion.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def write_generated_results() -> dict[str, object]:
    with open(ROOT / "configs" / "experiments.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    target_epochs = int(config["training"]["epochs"])
    comparison = pd.read_csv(ROOT / "artifacts" / "experiments" / "comparison_summary.csv")
    rows = {str(row.experiment): row for row in comparison.itertuples(index=False)}
    training_complete = bool(
        len(rows) == len(config["experiments"])
        and all(int(row.completed_epochs) >= target_epochs for row in rows.values())
    )

    gallery_manifest_path = ROOT / "galeria" / "manifest.csv"
    gallery_validation_path = ROOT / "artifacts" / "gallery" / "validation.json"
    gallery_ready = False
    gallery: pd.DataFrame | None = None
    validation: dict[str, object] = {}
    if gallery_manifest_path.exists() and gallery_validation_path.exists():
        gallery = pd.read_csv(gallery_manifest_path)
        validation = json.loads(gallery_validation_path.read_text(encoding="utf-8"))
        gallery_ready = bool(validation.get("status") == "passed" and len(gallery) == 10)

    lines = [
        "% Archivo generado por scripts/build_presentation.py. No editar a mano.",
        r"\newif\iftrainingcomplete",
        r"\trainingcompletetrue" if training_complete else r"\trainingcompletefalse",
        r"\newif\ifgalleryready",
        r"\galleryreadytrue" if gallery_ready else r"\galleryreadyfalse",
        rf"\newcommand{{\TargetEpochs}}{{{target_epochs}}}",
    ]
    experiment_macros = {
        "baseline_bce": "Baseline",
        "hinge_loss": "Hinge",
        "bce_spectral_norm": "Spectral",
    }
    for experiment_id, prefix in experiment_macros.items():
        row = rows[experiment_id]
        lines.extend(
            [
                rf"\newcommand{{\{prefix}Epoch}}{{{int(row.completed_epochs)}}}",
                rf"\newcommand{{\{prefix}LossD}}{{{format_metric(row.loss_d)}}}",
                rf"\newcommand{{\{prefix}LossG}}{{{format_metric(row.loss_g)}}}",
                rf"\newcommand{{\{prefix}Diversity}}{{{format_metric(row.fixed_pairwise_l2)}}}",
            ]
        )

    if gallery_ready and gallery is not None:
        hardest = gallery.loc[gallery["nearest_cosine_similarity"].idxmax()]
        lines.extend(
            [
                rf"\newcommand{{\FinalExperiment}}{{{tex_escape(hardest['experiment'])}}}",
                rf"\newcommand{{\SelectionRate}}{{{len(gallery)}/200 = {len(gallery) / 200:.0%}}}",
                rf"\newcommand{{\MeanNeighborSimilarity}}{{{gallery['nearest_cosine_similarity'].mean():.4f}}}",
                rf"\newcommand{{\HardestCharacter}}{{{tex_escape(hardest['name'])}}}",
                rf"\newcommand{{\HardestSimilarity}}{{{float(hardest['nearest_cosine_similarity']):.4f}}}",
                rf"\newcommand{{\RegenerationDelta}}{{{int(validation['regeneration_max_pixel_delta'])}}}",
            ]
        )
    else:
        lines.extend(
            [
                r"\newcommand{\FinalExperiment}{pendiente}",
                r"\newcommand{\SelectionRate}{10/200 = 5\% planificado}",
                r"\newcommand{\MeanNeighborSimilarity}{pendiente}",
                r"\newcommand{\HardestCharacter}{pendiente}",
                r"\newcommand{\HardestSimilarity}{pendiente}",
                r"\newcommand{\RegenerationDelta}{pendiente}",
            ]
        )
    GENERATED.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "training_complete": training_complete,
        "gallery_ready": gallery_ready,
        "target_epochs": target_epochs,
        "completed_epochs": {
            experiment_id: int(rows[experiment_id].completed_epochs)
            for experiment_id in experiment_macros
        },
    }


def compile_pdf() -> None:
    executable = shutil.which("xelatex")
    if executable is None:
        raise FileNotFoundError("xelatex no está disponible")
    command = [
        executable,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "presentacion.tex",
    ]
    for _ in range(2):
        subprocess.run(command, cwd=PRESENTATION, check=True)


def validate_pdf(status: dict[str, object]) -> dict[str, object]:
    if not PDF.exists():
        raise FileNotFoundError(PDF)
    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError("PyMuPDF es necesario para validar el PDF") from error
    document = pymupdf.open(PDF)
    page_count = document.page_count
    media_boxes = [tuple(round(value, 2) for value in page.rect) for page in document]
    extracted_lengths = [len(page.get_text().strip()) for page in document]
    document.close()
    if page_count != 12:
        raise RuntimeError(f"La presentación debe tener 12 páginas; se obtuvieron {page_count}")
    if any(length == 0 for length in extracted_lengths):
        raise RuntimeError("Existe una diapositiva sin texto extraíble")
    validation = {
        "status": "passed",
        "pdf": PDF.relative_to(ROOT).as_posix(),
        "page_count": page_count,
        "page_sizes_points": media_boxes,
        "text_characters_per_page": extracted_lengths,
        **status,
    }
    output = ROOT / "artifacts" / "presentation" / "build_status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return validation


def main() -> None:
    args = parse_args()
    status = write_generated_results()
    if args.no_compile:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    compile_pdf()
    validation = validate_pdf(status)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
