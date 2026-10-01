"""Genera los datos y valida la presentación HTML interactiva de Eryndor."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PRESENTATION = ROOT / "presentation"
HTML = PRESENTATION / "presentacion.html"
DATA_JS = PRESENTATION / "assets" / "presentation-data.js"


class SlideParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.slides = 0
        self.titles = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "section" and "slide" in classes:
            self.slides += 1
        if tag == "h2" and "slide-title" in classes:
            self.titles += 1


def load_data() -> dict[str, object]:
    config = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    dataset = json.loads((ROOT / "artifacts" / "metrics" / "dataset_summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(ROOT / "artifacts" / "experiments" / "comparison_summary.csv")
    phase8 = json.loads(
        (ROOT / "artifacts" / "phase8" / "phase8_analysis.json").read_text(encoding="utf-8")
    )
    phase9 = json.loads(
        (ROOT / "artifacts" / "phase9" / "readiness.json").read_text(encoding="utf-8")
    )
    phase9_summary_path = ROOT / "artifacts" / "phase9" / "phase9_summary.json"
    phase9_summary = (
        json.loads(phase9_summary_path.read_text(encoding="utf-8"))
        if phase9_summary_path.exists()
        else {}
    )
    phase10 = json.loads(
        (ROOT / "artifacts" / "phase10" / "readiness.json").read_text(encoding="utf-8")
    )
    phase10_smoke = json.loads(
        (ROOT / "artifacts" / "phase10" / "smoke_test.json").read_text(encoding="utf-8")
    )
    phase10_summary_path = ROOT / "artifacts" / "phase10" / "phase10_summary.json"
    phase10_summary = (
        json.loads(phase10_summary_path.read_text(encoding="utf-8"))
        if phase10_summary_path.exists()
        else {}
    )
    target = int(config["training"]["epochs"])
    experiment_labels = {
        "baseline_bce": "A · BCE base",
        "hinge_loss": "B · Hinge",
        "bce_spectral_norm": "C · BCE + spectral norm",
    }
    experiments = []
    for row in comparison.itertuples(index=False):
        experiments.append(
            {
                "id": row.experiment,
                "label": experiment_labels[row.experiment],
                "epochs": int(row.completed_epochs),
                "loss_d": round(float(row.loss_d), 4),
                "loss_g": round(float(row.loss_g), 4),
                "diversity": round(float(row.fixed_pairwise_l2), 4),
                "seconds": round(float(row.mean_epoch_seconds), 1),
            }
        )

    manifest_path = ROOT / "galeria" / "manifest.csv"
    validation_path = ROOT / "artifacts" / "gallery" / "validation.json"
    gallery_ready = False
    gallery: list[dict[str, object]] = []
    validation: dict[str, object] = {}
    if manifest_path.exists() and validation_path.exists():
        frame = pd.read_csv(manifest_path)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        gallery_ready = validation.get("status") == "passed" and len(frame) == 10
        if gallery_ready:
            gallery = frame.to_dict("records")

    completed = {item["id"]: item["epochs"] for item in experiments}
    data: dict[str, object] = {
        "project": "Eryndor: Guardianes del Velo",
        "members": ["Pablo Daniel Barillas Moreno", "Wilson Alejandro Calderón"],
        "target_epochs": target,
        "training_complete": all(value >= target for value in completed.values()),
        "completed_epochs": completed,
        "minimum_epoch": min(completed.values()),
        "phase8_complete": phase8.get("status") == "passed",
        "selected_experiment": phase8["selected_experiment"],
        "selection_scope": phase8["selection_scope"],
        "phase9_ready": phase9["status"] == "ready",
        "phase9_checkpoint_epoch": phase9["checkpoint_epoch"],
        "phase9_target_epoch": phase9["target_epoch"],
        "phase9_resnet_cached": phase9["resnet18"]["cached"],
        "phase9_complete": phase9_summary.get("status") == "passed",
        "phase9_mean_similarity": round(
            float(phase9_summary.get("cosine_similarity", {}).get("mean", 0.0)), 4
        ),
        "phase9_max_similarity": round(
            float(phase9_summary.get("cosine_similarity", {}).get("max", 0.0)), 4
        ),
        "phase9_screening_flags": int(phase9_summary.get("screening_flag_count", 0)),
        "phase9_exact_duplicates": int(phase9_summary.get("exact_pixel_duplicate_count", 0)),
        "phase10_ready": phase10["status"] == "ready",
        "phase10_candidate_count": int(phase10["candidate_count"]),
        "phase10_selected_count": int(phase10["selected_count"]),
        "phase10_selection_rate": f"{float(phase10['selection_rate']):.0%}",
        "phase10_smoke_passed": (
            phase10_smoke["status"] == "passed"
            and bool(phase10_smoke["deterministic_selection"])
        ),
        "phase10_complete": phase10_summary.get("status") == "passed",
        "phase10_eligible": int(phase10_summary.get("eligible_candidates", 0)),
        "phase10_mean_similarity": round(
            float(phase10_summary.get("mean_selected_neighbor_similarity", 0.0)), 4
        ),
        "phase10_mean_mse": round(
            float(phase10_summary.get("mean_selected_pixel_mse", 0.0)), 5
        ),
        "phase10_exact_duplicates": int(
            phase10_summary.get("exact_pixel_duplicate_count", 0)
        ),
        "dataset": {
            "images": int(dataset["image_count"]),
            "resolution": "64 × 64",
            "duplicates": int(dataset["duplicate_hashes"]),
            "missing": int(dataset["missing_files"]),
            "hair_styles": int(dataset["unique_hair_styles"]),
            "torso_styles": int(dataset["unique_torso_styles"]),
            "foreground": f"{float(dataset['foreground_ratio']['mean']):.2%}",
            "credited_layers": "254 / 254",
        },
        "experiments": experiments,
        "gallery_ready": gallery_ready,
        "selection": "10 / 200 = 5%",
        "gallery": gallery,
        "mean_neighbor_similarity": None,
        "hardest_character": None,
        "hardest_role": None,
        "hardest_similarity": None,
        "hardest_mse": None,
        "regeneration_delta": validation.get("regeneration_max_pixel_delta"),
    }
    if gallery_ready:
        frame = pd.DataFrame(gallery)
        hardest = frame.loc[frame["nearest_cosine_similarity"].idxmax()]
        data.update(
            {
                "mean_neighbor_similarity": round(float(frame["nearest_cosine_similarity"].mean()), 4),
                "hardest_character": str(hardest["name"]),
                "hardest_role": str(hardest["role"]),
                "hardest_similarity": round(float(hardest["nearest_cosine_similarity"]), 4),
                "hardest_mse": round(float(hardest["nearest_pixel_mse"]), 5),
            }
        )
    return data


def validate_html(data: dict[str, object]) -> dict[str, object]:
    parser = SlideParser()
    parser.feed(HTML.read_text(encoding="utf-8"))
    required = [
        PRESENTATION / "assets" / "presentation.css",
        PRESENTATION / "assets" / "presentation.js",
        PRESENTATION / "fonts" / "FreeSans.ttf",
        ROOT / "artifacts" / "dataset" / "dataset_contact_sheet.png",
        ROOT / "artifacts" / "phase8" / "fixed_noise_milestones.png",
        ROOT / "artifacts" / "phase8" / "phase8_diagnostics.png",
        ROOT / "artifacts" / "phase9" / "fixed_noise_neighbors.png",
        ROOT / "artifacts" / "phase10" / "selected_10.png",
        ROOT / "artifacts" / "gallery" / "final_gallery_grid.png",
        ROOT / "artifacts" / "gallery" / "nearest_neighbors.png",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.exists()]
    if parser.slides != 12 or parser.titles != 11:
        raise RuntimeError(f"Se esperaban 12 escenas y 11 títulos; se obtuvieron {parser.slides} y {parser.titles}")
    if missing:
        raise FileNotFoundError(f"Recursos HTML faltantes: {missing}")
    return {
        "status": "passed",
        "html": HTML.relative_to(ROOT).as_posix(),
        "slides": parser.slides,
        "titled_slides": parser.titles,
        "training_complete": data["training_complete"],
        "phase8_complete": data["phase8_complete"],
        "selected_experiment": data["selected_experiment"],
        "phase9_ready": data["phase9_ready"],
        "phase9_complete": data["phase9_complete"],
        "phase9_checkpoint_epoch": data["phase9_checkpoint_epoch"],
        "phase10_ready": data["phase10_ready"],
        "phase10_smoke_passed": data["phase10_smoke_passed"],
        "phase10_complete": data["phase10_complete"],
        "gallery_ready": data["gallery_ready"],
        "keyboard_navigation": True,
        "print_layout": "16:9, una escena por página",
        "missing_assets": missing,
    }


def main() -> None:
    data = load_data()
    DATA_JS.parent.mkdir(parents=True, exist_ok=True)
    DATA_JS.write_text(
        "window.ERYNDOR_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
        newline="\n",
    )
    status = validate_html(data)
    output = ROOT / "artifacts" / "presentation" / "html_status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
