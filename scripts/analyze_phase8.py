"""Audita las corridas completas y construye la evidencia de la fase 8."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "artifacts" / "experiments"
OUTPUT_DIR = ROOT / "artifacts" / "phase8"
MILESTONES = (0, 5, 10, 20, 40, 60)
EXPERIMENTS = {
    "baseline_bce": {
        "label": "A · BCE base",
        "color": "#4A4E8F",
        "visual": "Personajes reconocibles y variados; repite cabello cálido y ropa clara.",
    },
    "hinge_loss": {
        "label": "B · Hinge",
        "color": "#D6A34A",
        "visual": "Colapso progresivo hacia una plantilla casi única con textura de alta frecuencia.",
    },
    "bce_spectral_norm": {
        "label": "C · BCE + spectral norm",
        "color": "#4FC3C8",
        "visual": "Personajes reconocibles, mayor variedad cromática y diversidad sostenida.",
    },
}


def load_config() -> dict[str, object]:
    return yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))


def validate_experiment(
    experiment_id: str,
    target_epochs: int,
    expected_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    artifact_dir = EXPERIMENTS_DIR / experiment_id
    epoch_path = artifact_dir / "epoch_metrics.csv"
    step_path = artifact_dir / "step_metrics.csv"
    status_path = artifact_dir / "status.json"
    for path in (epoch_path, step_path, status_path):
        if not path.exists():
            raise FileNotFoundError(path)

    epochs = pd.read_csv(epoch_path).sort_values("epoch").reset_index(drop=True)
    steps = pd.read_csv(step_path).sort_values("global_step").reset_index(drop=True)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    required_epoch_columns = {
        "epoch", "global_step", "elapsed_seconds", "loss_d", "loss_g",
        "real_logit", "fake_logit_d", "fake_logit_g", "fixed_std",
        "fixed_pairwise_l2",
    }
    required_step_columns = {
        "epoch", "global_step", "loss_d", "loss_g", "real_logit",
        "fake_logit_d", "fake_logit_g",
    }
    missing_epoch = required_epoch_columns.difference(epochs.columns)
    missing_step = required_step_columns.difference(steps.columns)
    if missing_epoch or missing_step:
        raise ValueError(
            f"Columnas faltantes en {experiment_id}: epoch={sorted(missing_epoch)}, "
            f"step={sorted(missing_step)}"
        )

    if not np.array_equal(epochs["epoch"].to_numpy(), np.arange(1, target_epochs + 1)):
        raise AssertionError(f"Épocas incompletas o duplicadas en {experiment_id}")
    if not np.array_equal(steps["global_step"].to_numpy(), np.arange(1, expected_steps + 1)):
        raise AssertionError(f"Pasos incompletos o duplicados en {experiment_id}")
    if int(status["completed_epochs"]) != target_epochs or status["status"] != "complete":
        raise AssertionError(f"Estado incompleto en {experiment_id}: {status}")
    if epochs.isna().any().any() or steps.isna().any().any():
        raise AssertionError(f"Valores faltantes en {experiment_id}")
    for frame, name in ((epochs, "epoch"), (steps, "step")):
        if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
            raise AssertionError(f"Valores no finitos en {name}_metrics de {experiment_id}")

    missing_grids = [
        epoch
        for epoch in MILESTONES
        if not (artifact_dir / "fixed_noise" / f"epoch_{epoch:03d}.png").exists()
    ]
    if missing_grids:
        raise FileNotFoundError(f"Hitos faltantes en {experiment_id}: {missing_grids}")
    return epochs, steps, status


def build_summary(
    frames: dict[str, pd.DataFrame],
    statuses: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_id, frame in frames.items():
        first, final, tail = frame.iloc[0], frame.iloc[-1], frame.tail(10)
        tail_diversity = float(tail["fixed_pairwise_l2"].mean())
        collapse_signal = bool(
            tail_diversity < 0.05
            or float(final["fixed_pairwise_l2"])
            < 0.25 * float(frame["fixed_pairwise_l2"].max())
        )
        rows.append(
            {
                "experiment": experiment_id,
                "label": EXPERIMENTS[experiment_id]["label"],
                "completed_epochs": int(statuses[experiment_id]["completed_epochs"]),
                "global_steps": int(statuses[experiment_id]["global_step"]),
                "mean_epoch_seconds": float(frame["elapsed_seconds"].mean()),
                "total_epoch_minutes": float(frame["elapsed_seconds"].sum() / 60),
                "final_loss_d": float(final["loss_d"]),
                "final_loss_g": float(final["loss_g"]),
                "final_real_logit": float(final["real_logit"]),
                "final_fake_logit_g": float(final["fake_logit_g"]),
                "initial_diversity": float(first["fixed_pairwise_l2"]),
                "final_diversity": float(final["fixed_pairwise_l2"]),
                "tail10_diversity_mean": tail_diversity,
                "diversity_gain_ratio": float(
                    final["fixed_pairwise_l2"] / first["fixed_pairwise_l2"]
                ),
                "final_pixel_std": float(final["fixed_std"]),
                "near_zero_loss_d_epochs": int((frame["loss_d"] < 1e-3).sum()),
                "collapse_signal": collapse_signal,
                "visual_review": EXPERIMENTS[experiment_id]["visual"],
            }
        )
    return pd.DataFrame(rows)


def plot_diagnostics(frames: dict[str, pd.DataFrame]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(3, 5, figsize=(19, 11), constrained_layout=True)
    columns = (
        ("loss_d", "Pérdida D"),
        ("loss_g", "Pérdida G"),
        ("logits", "Logits medios"),
        ("fixed_pairwise_l2", "Diversidad fija"),
        ("elapsed_seconds", "Segundos / época"),
    )
    for row_index, (experiment_id, frame) in enumerate(frames.items()):
        color = str(EXPERIMENTS[experiment_id]["color"])
        label = str(EXPERIMENTS[experiment_id]["label"])
        for column_index, (metric, title) in enumerate(columns):
            axis = axes[row_index, column_index]
            if metric == "logits":
                axis.plot(frame["epoch"], frame["real_logit"], color="#79A879", label="real")
                axis.plot(frame["epoch"], frame["fake_logit_g"], color="#B55A62", label="falso")
                axis.axhline(0, color="#171A24", linewidth=0.7, alpha=0.35)
                axis.legend(frameon=False, fontsize=7)
            else:
                axis.plot(frame["epoch"], frame[metric], color=color, linewidth=1.9)
                if metric == "fixed_pairwise_l2":
                    axis.axhline(0.05, color="#B55A62", linestyle="--", linewidth=1, label="alerta")
                    axis.legend(frameon=False, fontsize=7)
            if row_index == 0:
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(label, fontweight="bold", color=color)
            if row_index == 2:
                axis.set_xlabel("Época")
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fase 8 · dinámica completa por experimento", fontsize=18, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "phase8_diagnostics.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_milestones() -> list[dict[str, object]]:
    fig, axes = plt.subplots(3, len(MILESTONES), figsize=(22, 11), constrained_layout=True)
    manifest: list[dict[str, object]] = []
    for row_index, experiment_id in enumerate(EXPERIMENTS):
        for column_index, epoch in enumerate(MILESTONES):
            source = EXPERIMENTS_DIR / experiment_id / "fixed_noise" / f"epoch_{epoch:03d}.png"
            with Image.open(source) as image:
                axes[row_index, column_index].imshow(image.convert("RGB"))
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(f"Época {epoch}", fontsize=12)
            if column_index == 0:
                axes[row_index, column_index].text(
                    -0.06, 0.5, str(EXPERIMENTS[experiment_id]["label"]),
                    transform=axes[row_index, column_index].transAxes,
                    rotation=90, va="center", ha="right",
                    color=str(EXPERIMENTS[experiment_id]["color"]),
                    fontsize=12, fontweight="bold",
                )
            manifest.append(
                {"experiment": experiment_id, "epoch": epoch, "path": source.relative_to(ROOT).as_posix()}
            )
    fig.suptitle("Mismo ruido fijo · hitos 0, 5, 10, 20, 40 y 60", fontsize=18, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "fixed_noise_milestones.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return manifest


def main() -> None:
    config = load_config()
    target_epochs = int(config["training"]["epochs"])
    steps_per_epoch = int(config["project"]["dataset_size"]) // int(config["training"]["batch_size"])
    expected_steps = target_epochs * steps_per_epoch
    frames: dict[str, pd.DataFrame] = {}
    statuses: dict[str, dict[str, object]] = {}
    memory_bytes = 0
    for experiment_id in EXPERIMENTS:
        epochs, steps, status = validate_experiment(experiment_id, target_epochs, expected_steps)
        epochs["experiment"] = pd.Categorical([experiment_id] * len(epochs), categories=list(EXPERIMENTS))
        frames[experiment_id] = epochs
        statuses[experiment_id] = status
        memory_bytes += int(epochs.memory_usage(deep=True).sum() + steps.memory_usage(deep=True).sum())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary(frames, statuses)
    summary.to_csv(
        OUTPUT_DIR / "experiment_analysis.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    combined = pd.concat(frames.values(), ignore_index=True)
    combined.to_csv(
        EXPERIMENTS_DIR / "all_epoch_metrics.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    plot_diagnostics(frames)
    milestone_manifest = plot_milestones()
    pd.DataFrame(milestone_manifest).to_csv(
        OUTPUT_DIR / "milestone_manifest.csv", index=False, encoding="utf-8", lineterminator="\n"
    )

    selected = "bce_spectral_norm"
    selected_row = summary.loc[summary["experiment"].eq(selected)].iloc[0]
    baseline_row = summary.loc[summary["experiment"].eq("baseline_bce")].iloc[0]
    hinge_row = summary.loc[summary["experiment"].eq("hinge_loss")].iloc[0]
    if bool(selected_row["collapse_signal"]) or not bool(hinge_row["collapse_signal"]):
        raise AssertionError("Las señales cuantitativas no sostienen la decisión registrada")
    analysis = {
        "status": "passed",
        "phase": 8,
        "target_epochs": target_epochs,
        "experiments": len(frames),
        "epoch_rows": int(len(combined)),
        "step_rows": int(sum(int(status["global_step"]) for status in statuses.values())),
        "missing_values": 0,
        "nonfinite_values": 0,
        "duplicate_epochs": 0,
        "duplicate_steps": 0,
        "memory_mb": round(memory_bytes / 1e6, 4),
        "milestones": list(MILESTONES),
        "selected_experiment": selected,
        "selected_checkpoint": f"checkpoints/{selected}/latest.pt",
        "selection_scope": "provisional para fases 9–10; faltan vecinos y 200 candidatos",
        "selection_basis": [
            f"diversidad media final {selected_row['tail10_diversity_mean']:.4f}, frente a {baseline_row['tail10_diversity_mean']:.4f} del baseline",
            f"diversidad final {selected_row['final_diversity']:.4f} sin señal de colapso",
            "rejillas 20–60 con personajes reconocibles y mayor variedad cromática",
            f"hinge descartado: {int(hinge_row['near_zero_loss_d_epochs'])} épocas con loss_D < 1e-3 y diversidad final {hinge_row['final_diversity']:.4f}",
        ],
        "conclusions": {
            "baseline_bce": "Aprendizaje útil y estable; alternativa defendible con diversidad algo menor.",
            "hinge_loss": "Hipótesis rechazada: colapso de modo y dominio prolongado de D.",
            "bce_spectral_norm": "Mejor equilibrio entre reconocimiento, variedad y diversidad sostenida.",
        },
        "limitations": [
            "La distancia L2 sobre ruido fijo es un proxy, no una métrica perceptual definitiva.",
            "Las pérdidas BCE y hinge no son comparables por magnitud.",
            "La selección debe confirmarse con vecinos ResNet18, MSE y 200 candidatos.",
            "Los checkpoints están disponibles localmente y respaldados en Drive, pero no se versionan en GitHub.",
        ],
    }
    (OUTPUT_DIR / "phase8_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    (OUTPUT_DIR / "selected_model.json").write_text(
        json.dumps(
            {
                "experiment": selected,
                "checkpoint": analysis["selected_checkpoint"],
                "epoch": target_epochs,
                "status": "selected_provisionally",
                "next_validation": "nearest_neighbors_and_200_candidate_gallery",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(summary.to_string(index=False))
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
