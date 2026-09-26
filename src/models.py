"""Arquitecturas DCGAN de 64 x 64 para Eryndor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import spectral_norm


@dataclass(frozen=True)
class ModelDimensions:
    latent_dim: int = 128
    image_channels: int = 3
    generator_features: int = 64
    discriminator_features: int = 64


class Generator(nn.Module):
    """Convierte z ~ N(0, I) con forma [B, Z, 1, 1] en RGB 64 x 64."""

    def __init__(self, dimensions: ModelDimensions = ModelDimensions()) -> None:
        super().__init__()
        z = dimensions.latent_dim
        f = dimensions.generator_features
        c = dimensions.image_channels
        self.latent_dim = z
        self.network = nn.Sequential(
            self._block(z, f * 8, 4, 1, 0),
            self._block(f * 8, f * 4, 4, 2, 1),
            self._block(f * 4, f * 2, 4, 2, 1),
            self._block(f * 2, f, 4, 2, 1),
            nn.ConvTranspose2d(f, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 4 or noise.shape[1:] != (self.latent_dim, 1, 1):
            raise ValueError(
                f"Se esperaba [B, {self.latent_dim}, 1, 1], se recibió {list(noise.shape)}"
            )
        return self.network(noise)


class Discriminator(nn.Module):
    """Produce un logit por imagen; no aplica sigmoid internamente."""

    def __init__(
        self,
        dimensions: ModelDimensions = ModelDimensions(),
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        f = dimensions.discriminator_features
        c = dimensions.image_channels

        def conv(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
        ) -> nn.Conv2d:
            layer = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            return spectral_norm(layer) if use_spectral_norm else layer

        self.network = nn.Sequential(
            conv(c, f),
            nn.LeakyReLU(0.2, inplace=True),
            conv(f, f * 2),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),
            conv(f * 2, f * 4),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            conv(f * 4, f * 8),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),
            conv(f * 8, 1, kernel_size=4, stride=1, padding=0),
        )
        self.use_spectral_norm = use_spectral_norm

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 64, 64):
            raise ValueError(f"Se esperaba [B, 3, 64, 64], se recibió {list(images.shape)}")
        return self.network(images).flatten()


def initialize_dcgan(module: nn.Module) -> None:
    """Inicialización DCGAN: N(0,.02) en convoluciones y N(1,.02) en BatchNorm."""

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        weight = getattr(module, "weight_orig", module.weight)
        nn.init.normal_(weight.data, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def build_models(
    dimensions: ModelDimensions,
    use_spectral_norm: bool,
    device: torch.device,
) -> tuple[Generator, Discriminator]:
    generator = Generator(dimensions).to(device)
    discriminator = Discriminator(dimensions, use_spectral_norm=use_spectral_norm).to(device)
    generator.apply(initialize_dcgan)
    discriminator.apply(initialize_dcgan)
    return generator, discriminator


def trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
