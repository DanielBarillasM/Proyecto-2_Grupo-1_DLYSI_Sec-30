"""Valida la fase 3 con ocho pasos reales de DCGAN y un checkpoint round-trip."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

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
    generate_fixed,
    load_checkpoint,
    save_checkpoint,
    save_image_grid,
    save_progress_comparison,
    save_training_curves,
    seed_everything,
    train_steps,
    write_json,
)


def parse_args(default_steps: int, default_batch_size: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fue solicitada, pero no está disponible")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def validate_variants(dimensions: ModelDimensions, device: torch.device) -> dict[str, dict[str, object]]:
    validations: dict[str, dict[str, object]] = {}
    variants = (
        ("baseline_bce", "bce_non_saturating", False),
        ("hinge_loss", "hinge", False),
        ("bce_spectral_norm", "bce_non_saturating", True),
    )
    for experiment_id, loss_name, use_spectral_norm in variants:
        seed_everything(42)
        generator, discriminator = build_models(dimensions, use_spectral_norm, device)
        noise = torch.randn(2, dimensions.latent_dim, 1, 1, device=device)
        fake = generator(noise)
        logits = discriminator(fake)
        objective = AdversarialLoss(loss_name)
        real_probe = torch.tensor([0.25, -0.1], device=device, requires_grad=True)
        fake_probe = torch.tensor([-0.3, 0.4], device=device, requires_grad=True)
        probe_loss = objective.discriminator(real_probe, fake_probe) + objective.generator(fake_probe)
        probe_loss.backward()
        spectral_layers = sum(1 for module in discriminator.modules() if hasattr(module, "weight_u"))
        validations[experiment_id] = {
            "fake_shape": list(fake.shape),
            "logit_shape": list(logits.shape),
            "finite": bool(torch.isfinite(fake).all() and torch.isfinite(logits).all()),
            "loss": loss_name,
            "spectral_norm_layers": spectral_layers,
            "loss_gradients_finite": bool(
                torch.isfinite(real_probe.grad).all() and torch.isfinite(fake_probe.grad).all()
            ),
        }
        del generator, discriminator
    return validations


def main() -> None:
    with open(ROOT / "configs" / "experiments.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    args = parse_args(config["smoke_test"]["steps"], config["smoke_test"]["batch_size"])
    device = resolve_device(args.device)
    training = config["training"]
    baseline = next(item for item in config["experiments"] if item["id"] == "baseline_bce")
    dimensions = ModelDimensions(
        latent_dim=int(training["latent_dim"]),
        image_channels=int(config["project"]["channels"]),
        generator_features=int(training["generator_features"]),
        discriminator_features=int(training["discriminator_features"]),
    )

    seed_everything(int(training["model_seed"]))
    loader = build_dataloader(
        manifest_path=ROOT / "data" / "processed" / "manifest.csv",
        project_root=ROOT,
        batch_size=args.batch_size,
        seed=int(training["model_seed"]),
        horizontal_flip=True,
        num_workers=0,
    )
    generator, discriminator = build_models(
        dimensions,
        use_spectral_norm=bool(baseline["spectral_norm_discriminator"]),
        device=device,
    )
    optimizer_g, optimizer_d = build_optimizers(
        generator,
        discriminator,
        float(training["learning_rate_g"]),
        float(training["learning_rate_d"]),
        (float(training["beta1"]), float(training["beta2"])),
    )
    noise_generator = torch.Generator(device="cpu").manual_seed(int(training["fixed_noise_seed"]))
    fixed_noise = torch.randn(
        int(config["smoke_test"]["sample_count"]),
        dimensions.latent_dim,
        1,
        1,
        generator=noise_generator,
    ).to(device)

    artifact_dir = ROOT / "artifacts" / "smoke_test"
    checkpoint_path = ROOT / "checkpoints" / "smoke_test" / "baseline_bce_smoke.pt"
    before = generate_fixed(generator, fixed_noise)
    save_image_grid(before, artifact_dir / "fixed_noise_before.png", "Ruido fijo · inicialización")

    started = time.perf_counter()
    history = train_steps(
        generator,
        discriminator,
        loader,
        optimizer_g,
        optimizer_d,
        AdversarialLoss(str(baseline["loss"])),
        dimensions.latent_dim,
        device,
        max_steps=args.steps,
    )
    elapsed = time.perf_counter() - started
    after = generate_fixed(generator, fixed_noise)
    save_image_grid(after, artifact_dir / "fixed_noise_after.png", f"Ruido fijo · {args.steps} pasos")
    save_progress_comparison(
        before,
        after,
        artifact_dir / "fixed_noise_comparison.png",
        steps=args.steps,
    )
    save_training_curves(history, artifact_dir / "smoke_training_curves.png")

    save_checkpoint(
        checkpoint_path,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        fixed_noise,
        baseline,
        epoch=0,
        global_step=args.steps,
        data_generator=loader.generator,
    )

    reloaded_g, reloaded_d = build_models(dimensions, False, device)
    reloaded_opt_g, reloaded_opt_d = build_optimizers(
        reloaded_g,
        reloaded_d,
        float(training["learning_rate_g"]),
        float(training["learning_rate_d"]),
        (float(training["beta1"]), float(training["beta2"])),
    )
    resumed_data_generator = torch.Generator()
    payload = load_checkpoint(
        checkpoint_path,
        reloaded_g,
        reloaded_d,
        reloaded_opt_g,
        reloaded_opt_d,
        device,
        restore_rng=False,
        data_generator=resumed_data_generator,
    )
    reloaded_after = generate_fixed(reloaded_g, payload["fixed_noise"].to(device))
    roundtrip_max_error = float((after - reloaded_after).abs().max())
    data_generator_restored = bool(
        torch.equal(resumed_data_generator.get_state(), payload["data_generator_state"].cpu())
    )
    variants = validate_variants(dimensions, device)

    metrics = {
        "status": "passed",
        "scope": "smoke_test_not_full_training",
        "device": str(device),
        "torch_version": torch.__version__,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "images_seen": args.steps * args.batch_size,
        "elapsed_seconds": elapsed,
        "generator_parameters": trainable_parameters(generator),
        "discriminator_parameters": trainable_parameters(discriminator),
        "initial": history[0],
        "final": history[-1],
        "all_losses_finite": all(
            math.isfinite(row[metric])
            for row in history
            for metric in ("loss_d", "loss_g", "real_logit", "fake_logit_d", "fake_logit_g")
        ),
        "fixed_output_before": {
            "min": float(before.min()),
            "max": float(before.max()),
            "std": float(before.std()),
        },
        "fixed_output_after": {
            "min": float(after.min()),
            "max": float(after.max()),
            "std": float(after.std()),
        },
        "checkpoint": {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "epoch": int(payload["epoch"]),
            "global_step": int(payload["global_step"]),
            "roundtrip_max_abs_error": roundtrip_max_error,
            "data_generator_restored": data_generator_restored,
        },
        "variant_validations": variants,
    }
    if (
        not metrics["all_losses_finite"]
        or roundtrip_max_error != 0.0
        or not data_generator_restored
    ):
        raise AssertionError(f"Smoke test inválido: {metrics}")
    write_json(artifact_dir / "smoke_metrics.json", metrics)
    print(f"Smoke test aprobado en {elapsed:.2f}s sobre {device}")
    print(f"Parámetros G/D: {metrics['generator_parameters']:,} / {metrics['discriminator_parameters']:,}")
    print(f"Checkpoint round-trip: error máximo {roundtrip_max_error:.3g}")
    print(f"Métricas: {artifact_dir / 'smoke_metrics.json'}")


if __name__ == "__main__":
    main()
