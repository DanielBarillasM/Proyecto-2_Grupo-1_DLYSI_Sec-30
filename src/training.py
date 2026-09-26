"""Primitivas de entrenamiento, visualización y checkpoints reanudables."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.losses import AdversarialLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_optimizers(
    generator: nn.Module,
    discriminator: nn.Module,
    learning_rate_g: float,
    learning_rate_d: float,
    betas: tuple[float, float],
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate_g, betas=betas)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=betas)
    return optimizer_g, optimizer_d


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def train_steps(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    loss: AdversarialLoss,
    latent_dim: int,
    device: torch.device,
    max_steps: int,
) -> list[dict[str, float]]:
    """Ejecuta un número acotado de pasos y devuelve métricas por paso."""

    generator.train()
    discriminator.train()
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for step, real_images in enumerate(loader, start=1):
        if step > max_steps:
            break
        real_images = real_images.to(device, non_blocking=True)
        batch_size = real_images.shape[0]
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

        _set_requires_grad(discriminator, True)
        optimizer_d.zero_grad(set_to_none=True)
        fake_images = generator(noise)
        real_logits = discriminator(real_images)
        fake_logits = discriminator(fake_images.detach())
        loss_d = loss.discriminator(real_logits, fake_logits)
        loss_d.backward()
        optimizer_d.step()

        _set_requires_grad(discriminator, False)
        optimizer_g.zero_grad(set_to_none=True)
        generated_logits = discriminator(fake_images)
        loss_g = loss.generator(generated_logits)
        loss_g.backward()
        optimizer_g.step()
        _set_requires_grad(discriminator, True)

        history.append(
            {
                "step": float(step),
                "loss_d": float(loss_d.detach()),
                "loss_g": float(loss_g.detach()),
                "real_logit": float(real_logits.detach().mean()),
                "fake_logit_d": float(fake_logits.detach().mean()),
                "fake_logit_g": float(generated_logits.detach().mean()),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )

    if len(history) != max_steps:
        raise RuntimeError(f"Se solicitaron {max_steps} pasos y solo se ejecutaron {len(history)}")
    return history


@torch.inference_mode()
def generate_fixed(generator: nn.Module, fixed_noise: torch.Tensor) -> torch.Tensor:
    was_training = generator.training
    generator.eval()
    images = generator(fixed_noise).detach().cpu()
    generator.train(was_training)
    return images


def _to_display(images: torch.Tensor) -> np.ndarray:
    return images.clamp(-1, 1).add(1).div(2).permute(0, 2, 3, 1).numpy()


def save_image_grid(
    images: torch.Tensor,
    output_path: str | Path,
    title: str,
    columns: int = 4,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    display_images = _to_display(images)
    rows = int(np.ceil(len(display_images) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.05, rows * 2.05),
        facecolor="#171A24",
        constrained_layout=True,
        squeeze=False,
    )
    for axis in axes.flat:
        axis.set_facecolor("#171A24")
        axis.axis("off")
    for index, (axis, image) in enumerate(zip(axes.flat, display_images, strict=False)):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(f"z{index:02d}", color="#E6E6FA", fontsize=8)
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_progress_comparison(
    before: torch.Tensor,
    after: torch.Tensor,
    output_path: str | Path,
    steps: int,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    before_display = _to_display(before)
    after_display = _to_display(after)
    count = min(8, len(before_display), len(after_display))
    fig, axes = plt.subplots(
        2,
        count,
        figsize=(count * 1.55, 3.6),
        facecolor="#171A24",
        constrained_layout=True,
        squeeze=False,
    )
    for column in range(count):
        axes[0, column].imshow(before_display[column], interpolation="nearest")
        axes[1, column].imshow(after_display[column], interpolation="nearest")
        axes[0, column].set_title(f"z{column:02d}", color="#E6E6FA", fontsize=8)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("Inicial", color="#4FC3C8", fontsize=10)
    axes[1, 0].set_ylabel(f"{steps} pasos", color="#D6A34A", fontsize=10)
    fig.suptitle("Ruido fijo · smoke test DCGAN", color="white", fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_training_curves(history: list[dict[str, float]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(row["step"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].plot(steps, [row["loss_d"] for row in history], marker="o", label="D", color="#4A4E8F")
    axes[0].plot(steps, [row["loss_g"] for row in history], marker="o", label="G", color="#D6A34A")
    axes[0].set(title="Pérdidas del smoke test", xlabel="Paso", ylabel="Pérdida")
    axes[0].legend(frameon=False)
    axes[1].plot(steps, [row["real_logit"] for row in history], marker="o", label="D(x)", color="#79A879")
    axes[1].plot(steps, [row["fake_logit_d"] for row in history], marker="o", label="D(G(z)) · paso D", color="#A490C2")
    axes[1].plot(steps, [row["fake_logit_g"] for row in history], marker="o", label="D(G(z)) · paso G", color="#4FC3C8")
    axes[1].axhline(0, color="#171A24", lw=0.8, alpha=0.5)
    axes[1].set(title="Logits medios", xlabel="Paso", ylabel="Logit")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    fixed_noise: torch.Tensor,
    experiment: dict[str, Any],
    epoch: int,
    global_step: int,
    data_generator: torch.Generator | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment": experiment,
        "epoch": epoch,
        "global_step": global_step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "fixed_noise": fixed_noise.detach().cpu(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "data_generator_state": data_generator.get_state() if data_generator is not None else None,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def load_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    restore_rng: bool = True,
    data_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Versión de checkpoint no compatible")
    generator.load_state_dict(payload["generator"])
    discriminator.load_state_dict(payload["discriminator"])
    optimizer_g.load_state_dict(payload["optimizer_g"])
    optimizer_d.load_state_dict(payload["optimizer_d"])
    if restore_rng:
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
    if data_generator is not None and payload.get("data_generator_state") is not None:
        data_generator.set_state(payload["data_generator_state"])
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
