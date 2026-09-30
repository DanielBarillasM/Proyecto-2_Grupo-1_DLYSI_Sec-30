"""Entrena o reanuda uno de los experimentos GAN pre-registrados."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_dataloader
from src.losses import AdversarialLoss
from src.models import ModelDimensions, build_models, trainable_parameters
from src.training import (
    build_optimizers,
    fixed_sample_diagnostics,
    generate_fixed,
    load_checkpoint,
    save_checkpoint,
    save_experiment_curves,
    save_generator_snapshot,
    save_image_grid,
    seed_everything,
    train_steps,
    write_json,
)


def parse_args(experiment_ids: list[str], default_epochs: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=experiment_ids)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--backup-root",
        type=Path,
        help="Raíz opcional donde se refleja el estado reanudable después de cada época.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fue solicitada, pero no está disponible")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def restore_backup_if_needed(
    backup_root: Path | None,
    experiment_id: str,
    checkpoint_dir: Path,
    artifact_dir: Path,
) -> None:
    if backup_root is None or (checkpoint_dir / "latest.pt").exists():
        return
    backup_checkpoint = backup_root / "checkpoints" / experiment_id
    backup_artifact = backup_root / "artifacts" / "experiments" / experiment_id
    if not (backup_checkpoint / "latest.pt").exists():
        return
    shutil.copytree(backup_checkpoint, checkpoint_dir, dirs_exist_ok=True)
    if backup_artifact.exists():
        shutil.copytree(backup_artifact, artifact_dir, dirs_exist_ok=True)
    print(f"Estado restaurado desde {backup_root}", flush=True)


def mirror_training_state(
    backup_root: Path | None,
    experiment_id: str,
    checkpoint_dir: Path,
    artifact_dir: Path,
) -> None:
    if backup_root is None:
        return
    backup_root = backup_root.resolve()
    if backup_root == ROOT.resolve():
        return
    shutil.copytree(
        checkpoint_dir,
        backup_root / "checkpoints" / experiment_id,
        dirs_exist_ok=True,
    )
    shutil.copytree(
        artifact_dir,
        backup_root / "artifacts" / "experiments" / experiment_id,
        dirs_exist_ok=True,
    )


def aggregate_epoch(
    epoch: int,
    global_step: int,
    step_rows: list[dict[str, float]],
    fixed_images: torch.Tensor,
    elapsed_seconds: float,
) -> dict[str, float | int]:
    metrics = ("loss_d", "loss_g", "real_logit", "fake_logit_d", "fake_logit_g")
    row: dict[str, float | int] = {
        "epoch": epoch,
        "global_step": global_step,
        "elapsed_seconds": elapsed_seconds,
    }
    for metric in metrics:
        values = np.asarray([step[metric] for step in step_rows], dtype=np.float64)
        row[metric] = float(values.mean())
        row[f"{metric}_std"] = float(values.std())
    row.update(fixed_sample_diagnostics(fixed_images))
    return row


def main() -> None:
    with open(ROOT / "configs" / "experiments.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiments = {item["id"]: item for item in config["experiments"]}
    args = parse_args(list(experiments), int(config["training"]["epochs"]))
    if args.epochs < 1:
        raise ValueError("--epochs debe ser positivo")

    experiment = experiments[args.experiment]
    training = config["training"]
    configured_target_epochs = int(training["epochs"])
    device = resolve_device(args.device)
    dimensions = ModelDimensions(
        latent_dim=int(training["latent_dim"]),
        image_channels=int(config["project"]["channels"]),
        generator_features=int(training["generator_features"]),
        discriminator_features=int(training["discriminator_features"]),
    )
    artifact_dir = ROOT / "artifacts" / "experiments" / args.experiment
    fixed_dir = artifact_dir / "fixed_noise"
    checkpoint_dir = ROOT / "checkpoints" / args.experiment
    latest_checkpoint = checkpoint_dir / "latest.pt"
    history_path = artifact_dir / "epoch_metrics.csv"
    steps_path = artifact_dir / "step_metrics.csv"
    status_path = artifact_dir / "status.json"
    backup_root = args.backup_root.resolve() if args.backup_root else None
    restore_backup_if_needed(
        backup_root,
        args.experiment,
        checkpoint_dir,
        artifact_dir,
    )

    seed_everything(int(training["model_seed"]))
    loader = build_dataloader(
        manifest_path=ROOT / "data" / "processed" / "manifest.csv",
        project_root=ROOT,
        batch_size=int(training["batch_size"]),
        seed=int(training["model_seed"]),
        horizontal_flip=True,
        num_workers=args.num_workers,
    )
    generator, discriminator = build_models(
        dimensions,
        bool(experiment["spectral_norm_discriminator"]),
        device,
    )
    optimizer_g, optimizer_d = build_optimizers(
        generator,
        discriminator,
        float(training["learning_rate_g"]),
        float(training["learning_rate_d"]),
        (float(training["beta1"]), float(training["beta2"])),
    )

    epoch_history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    step_history = pd.read_csv(steps_path).to_dict("records") if steps_path.exists() else []
    start_epoch = 1
    global_step = 0

    if latest_checkpoint.exists() and not args.no_resume:
        payload = load_checkpoint(
            latest_checkpoint,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            device,
            restore_rng=True,
            data_generator=loader.generator,
        )
        if payload["experiment"]["id"] != args.experiment:
            raise ValueError("El checkpoint pertenece a otro experimento")
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        fixed_noise = payload["fixed_noise"].to(device)
        epoch_history = [row for row in epoch_history if int(row["epoch"]) < start_epoch]
        step_history = [row for row in step_history if int(row["epoch"]) < start_epoch]
        print(f"Reanudando {args.experiment} desde época {start_epoch}", flush=True)
    else:
        # Sin checkpoint se inicia una corrida nueva; no se reutilizan
        # métricas preliminares versionadas en el repositorio.
        epoch_history = []
        step_history = []
        noise_generator = torch.Generator(device="cpu").manual_seed(
            int(training["fixed_noise_seed"])
        )
        fixed_noise = torch.randn(16, dimensions.latent_dim, 1, 1, generator=noise_generator).to(device)
        initial_images = generate_fixed(generator, fixed_noise)
        save_image_grid(initial_images, fixed_dir / "epoch_000.png", f"{args.experiment} · época 0")

    if start_epoch > args.epochs:
        completed_epochs = start_epoch - 1
        current_status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.exists()
            else {"experiment": args.experiment}
        )
        current_status.update(
            {
                "status": "complete"
                if completed_epochs >= configured_target_epochs
                else "partial",
                "completed_epochs": completed_epochs,
                "target_epochs": configured_target_epochs,
                "requested_end_epoch": args.epochs,
            }
        )
        write_json(status_path, current_status)
        print(f"{args.experiment} ya alcanzó {start_epoch - 1} épocas; no hay trabajo pendiente.")
        return

    print(
        f"Entrenando {args.experiment} en {device}: épocas {start_epoch}..{args.epochs}, "
        f"{len(loader)} pasos/época",
        flush=True,
    )
    total_started = time.perf_counter()
    loss = AdversarialLoss(str(experiment["loss"]))

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        rows = train_steps(
            generator,
            discriminator,
            loader,
            optimizer_g,
            optimizer_d,
            loss,
            dimensions.latent_dim,
            device,
            max_steps=len(loader),
        )
        for row in rows:
            global_step += 1
            row["epoch"] = epoch
            row["global_step"] = global_step
            step_history.append(row)

        fixed_images = generate_fixed(generator, fixed_noise)
        elapsed = time.perf_counter() - epoch_started
        epoch_row = aggregate_epoch(epoch, global_step, rows, fixed_images, elapsed)
        epoch_history.append(epoch_row)

        if not all(
            math.isfinite(float(epoch_row[key]))
            for key in ("loss_d", "loss_g", "real_logit", "fake_logit_d", "fake_logit_g")
        ):
            raise FloatingPointError(f"Métricas no finitas en época {epoch}: {epoch_row}")

        atomic_csv(pd.DataFrame(epoch_history), history_path)
        atomic_csv(pd.DataFrame(step_history), steps_path)
        save_experiment_curves(epoch_history, artifact_dir / "training_curves.png")

        is_milestone = epoch % int(training["checkpoint_every"]) == 0
        is_final = epoch == args.epochs
        if is_milestone or is_final:
            save_image_grid(
                fixed_images,
                fixed_dir / f"epoch_{epoch:03d}.png",
                f"{args.experiment} · época {epoch}",
            )
            save_generator_snapshot(
                checkpoint_dir / f"generator_epoch_{epoch:03d}.pt",
                generator,
                experiment,
                epoch,
                global_step,
                int(training["fixed_noise_seed"]),
            )

        save_checkpoint(
            latest_checkpoint,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            fixed_noise,
            experiment,
            epoch=epoch,
            global_step=global_step,
            data_generator=loader.generator,
        )

        status = {
            "status": "complete"
            if epoch >= configured_target_epochs
            else ("partial" if is_final else "running"),
            "experiment": args.experiment,
            "completed_epochs": epoch,
            "target_epochs": configured_target_epochs,
            "requested_end_epoch": args.epochs,
            "global_step": global_step,
            "device": str(device),
            "generator_parameters": trainable_parameters(generator),
            "discriminator_parameters": trainable_parameters(discriminator),
            "last_epoch_seconds": elapsed,
            "run_elapsed_seconds": time.perf_counter() - total_started,
            "latest_metrics": epoch_row,
            "rolling_checkpoint": latest_checkpoint.relative_to(ROOT).as_posix(),
            "backup_root": str(backup_root) if backup_root else None,
        }
        write_json(status_path, status)
        mirror_training_state(
            backup_root,
            args.experiment,
            checkpoint_dir,
            artifact_dir,
        )
        print(
            f"[{args.experiment}] época {epoch:03d}/{args.epochs:03d} · "
            f"D={float(epoch_row['loss_d']):.4f} · G={float(epoch_row['loss_g']):.4f} · "
            f"div={float(epoch_row['fixed_pairwise_l2']):.4f} · {elapsed:.1f}s",
            flush=True,
        )

    print(f"Entrenamiento completado: {args.experiment}", flush=True)


if __name__ == "__main__":
    main()
