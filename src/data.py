"""Dataset y DataLoader reproducibles para los sprites de Eryndor."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SpriteDataset(Dataset):
    """Carga PNG de 64 x 64 y los normaliza al intervalo [-1, 1]."""

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path | None = None,
        horizontal_flip: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else self.manifest_path.parents[2]
        )
        self.manifest = pd.read_csv(self.manifest_path)
        self.horizontal_flip = horizontal_flip

        required = {"image_id", "path", "sha256"}
        missing = required.difference(self.manifest.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en el manifiesto: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> torch.Tensor:
        relative_path = Path(self.manifest.iloc[index]["path"])
        image_path = self.project_root / relative_path
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()

        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        tensor = tensor.div(127.5).sub(1.0)
        if self.horizontal_flip and torch.rand(()) < 0.5:
            tensor = torch.flip(tensor, dims=(2,))
        return tensor


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    manifest_path: str | Path,
    project_root: str | Path | None = None,
    batch_size: int = 64,
    seed: int = 42,
    shuffle: bool = True,
    horizontal_flip: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Construye un DataLoader con generador y workers sembrados."""

    dataset = SpriteDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        horizontal_flip=horizontal_flip,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
