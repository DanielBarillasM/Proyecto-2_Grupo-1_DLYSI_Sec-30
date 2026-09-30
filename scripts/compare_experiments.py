"""Consolida las métricas disponibles de los experimentos A, B y C."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "experiments"


def main() -> None:
    with open(ROOT / "configs" / "experiments.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for experiment in config["experiments"]:
        experiment_id = experiment["id"]
        metrics_path = ARTIFACTS / experiment_id / "epoch_metrics.csv"
        if not metrics_path.exists():
            continue
        frame = pd.read_csv(metrics_path).sort_values("epoch").reset_index(drop=True)
        target_epochs = int(config["training"]["epochs"])
        expected_epochs = np.arange(1, target_epochs + 1)
        if frame["epoch"].duplicated().any():
            raise ValueError(f"Hay épocas duplicadas en {experiment_id}")
        if len(frame) == target_epochs and not np.array_equal(
            frame["epoch"].to_numpy(), expected_epochs
        ):
            raise ValueError(f"La secuencia de épocas está incompleta en {experiment_id}")
        numeric = frame.select_dtypes(include="number")
        if frame.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"Hay métricas faltantes o no finitas en {experiment_id}")
        frame["experiment"] = experiment_id
        frames.append(frame)
        final = frame.iloc[-1]
        summary_rows.append(
            {
                "experiment": experiment_id,
                "completed_epochs": int(final["epoch"]),
                "mean_epoch_seconds": float(frame["elapsed_seconds"].mean()),
                "loss_d": float(final["loss_d"]),
                "loss_g": float(final["loss_g"]),
                "real_logit": float(final["real_logit"]),
                "fake_logit_g": float(final["fake_logit_g"]),
                "fixed_std": float(final["fixed_std"]),
                "fixed_pairwise_l2": float(final["fixed_pairwise_l2"]),
            }
        )

    if not frames:
        raise FileNotFoundError("No hay métricas de experimentos para comparar")

    combined = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    target_epochs = int(config["training"]["epochs"])
    summary["estimated_remaining_hours"] = (
        (target_epochs - summary["completed_epochs"]).clip(lower=0)
        * summary["mean_epoch_seconds"]
        / 3600
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ARTIFACTS / "all_epoch_metrics.csv", index=False, lineterminator="\n")
    summary.to_csv(ARTIFACTS / "comparison_summary.csv", index=False, lineterminator="\n")

    colors = {
        "baseline_bce": "#4A4E8F",
        "hinge_loss": "#D6A34A",
        "bce_spectral_norm": "#4FC3C8",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    specifications = (
        ("loss_d", "Pérdida de D", axes[0, 0]),
        ("loss_g", "Pérdida de G", axes[0, 1]),
        ("real_logit", "Logit real medio", axes[1, 0]),
        ("fixed_pairwise_l2", "Diversidad sobre ruido fijo", axes[1, 1]),
    )
    for metric, title, axis in specifications:
        for experiment_id, frame in combined.groupby("experiment", sort=False):
            axis.plot(
                frame["epoch"],
                frame[metric],
                label=experiment_id,
                color=colors.get(experiment_id),
                linewidth=2,
                marker="o",
                markersize=4,
            )
        axis.set(title=title, xlabel="Época")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Comparación controlada de experimentos", fontsize=16, fontweight="bold")
    fig.savefig(ARTIFACTS / "experiment_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    available_grids: list[tuple[str, Path]] = []
    for row in summary.itertuples(index=False):
        grid_path = (
            ARTIFACTS
            / row.experiment
            / "fixed_noise"
            / f"epoch_{int(row.completed_epochs):03d}.png"
        )
        if grid_path.exists():
            available_grids.append((str(row.experiment), grid_path))
    if available_grids:
        grid_figure, grid_axes = plt.subplots(
            1, len(available_grids), figsize=(5 * len(available_grids), 5), squeeze=False
        )
        for axis, (experiment_id, grid_path) in zip(grid_axes.ravel(), available_grids):
            axis.imshow(plt.imread(grid_path))
            axis.set_title(experiment_id, color=colors.get(experiment_id), fontweight="bold")
            axis.axis("off")
        grid_figure.suptitle(
            "Ruido fijo · última época disponible", fontsize=15, fontweight="bold"
        )
        grid_figure.tight_layout()
        grid_figure.savefig(
            ARTIFACTS / "latest_fixed_noise_comparison.png", dpi=180, bbox_inches="tight"
        )
        plt.close(grid_figure)

    status = {
        "experiments_available": int(summary.shape[0]),
        "all_reached_target": bool(
            len(summary) == len(config["experiments"])
            and (summary["completed_epochs"] >= target_epochs).all()
        ),
        "target_epochs": target_epochs,
        "completed_epochs": {
            str(row.experiment): int(row.completed_epochs)
            for row in summary.itertuples(index=False)
        },
        "estimated_remaining_cpu_hours": float(summary["estimated_remaining_hours"].sum()),
    }
    (ARTIFACTS / "comparison_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
