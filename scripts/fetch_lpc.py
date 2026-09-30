"""Descarga de forma reproducible los recursos LPC necesarios en cualquier sistema."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator.git"
SPARSE_PATHS = (
    "/CREDITS.csv",
    "/LICENSE",
    "/README.md",
    "/palette_definitions/",
    "/sheet_definitions/",
    "/spritesheets/**/idle.png",
    "/spritesheets/**/idle/*.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    return parser.parse_args()


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    project_root = parse_args().project_root.resolve()
    destination = project_root / "data" / "raw" / "lpc_generator"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            REMOTE,
            str(destination),
        )
    if not (destination / ".git").exists():
        raise RuntimeError(f"La ruta existe, pero no es un clon Git válido: {destination}")

    run("git", "-C", str(destination), "config", "core.longpaths", "true")
    run(
        "git",
        "-C",
        str(destination),
        "sparse-checkout",
        "set",
        "--no-cone",
        *SPARSE_PATHS,
    )
    idle_count = sum(
        1
        for path in (destination / "spritesheets").rglob("*.png")
        if "idle" in path.parts or path.name == "idle.png"
    )
    if idle_count == 0:
        raise RuntimeError("La descarga LPC no contiene capas idle")
    print(f"LPC listo en: {destination}")
    print(f"Capas idle disponibles: {idle_count}")


if __name__ == "__main__":
    main()
