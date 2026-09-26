"""Construye y audita el dataset LPC de Eryndor de forma determinista."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lpc_generator"
SPRITES = RAW / "spritesheets"
PROCESSED = ROOT / "data" / "processed"
IMAGES = PROCESSED / "sprites"
MANIFEST = PROCESSED / "manifest.csv"
FIGURES = ROOT / "artifacts" / "dataset"
METRICS = ROOT / "artifacts" / "metrics"
BACKGROUND = (23, 26, 36, 255)
FRAME_BOX = (0, 128, 64, 192)
THEMATIC_COLORS = {
    "black",
    "bluegray",
    "bronze",
    "charcoal",
    "forest",
    "gold",
    "gray",
    "iron",
    "lavender",
    "leather",
    "maroon",
    "navy",
    "purple",
    "red",
    "silver",
    "slate",
    "steel",
    "teal",
    "walnut",
}

BODY_TONES = ("light", "amber", "olive", "taupe", "bronze", "brown", "black")
HAIR_COLORS = (
    "ash",
    "black",
    "blonde",
    "chestnut",
    "dark_brown",
    "ginger",
    "platinum",
    "raven",
    "redhead",
    "violet",
)
HAIR_STYLES = {
    "afro",
    "bangs",
    "bangs_bun",
    "bangslong",
    "bangsshort",
    "bedhead",
    "bob",
    "bob_side_part",
    "buzzcut",
    "cornrows",
    "curly_long",
    "curly_short",
    "curtains",
    "dreadlocks_long",
    "dreadlocks_short",
    "half_up",
    "halfmessy",
    "high_and_tight",
    "idol",
    "lob",
    "long",
    "long_messy",
    "long_messy2",
    "long_straight",
    "longhawk",
    "messy1",
    "messy2",
    "messy3",
    "mop",
    "natural",
    "page",
    "parted",
    "parted2",
    "parted3",
    "pigtails",
    "pigtails_bangs",
    "pixie",
    "plain",
    "relm_short",
    "shorthawk",
    "spiked",
    "spiked2",
    "swoop",
    "unkempt",
}
TORSO_FRAGMENTS = (
    "torso/armour/",
    "torso/chainmail/",
    "torso/clothes/longsleeve/longsleeve/",
    "torso/clothes/longsleeve/longsleeves/",
    "torso/clothes/longsleeve/longsleeves_cuffed/",
    "torso/clothes/sleeveless/sleeveless1/",
    "torso/clothes/vest/",
    "torso/clothes/vest_open/",
)
LEGS_FRAGMENTS = (
    "legs/armour/",
    "legs/hose/",
    "legs/leggings/",
    "legs/leggings2/",
    "legs/pantaloons/",
    "legs/pants/",
    "legs/pants2/",
    "legs/skirts/",
)
FEET_FRAGMENTS = (
    "feet/armour/",
    "feet/boots/",
    "feet/sandals/",
    "feet/shoes/ghillies/",
    "feet/shoes/revised/",
)
HAT_FRAGMENTS = (
    "hat/cloth/hood/adult/",
    "hat/cloth/leather_cap/adult/",
    "hat/formal/crown/adult/",
    "hat/formal/tiara/adult/",
    "hat/helmet/armet/adult/",
    "hat/helmet/barbarian/adult/",
    "hat/helmet/bascinet/adult/",
    "hat/helmet/horned/adult/",
    "hat/helmet/kettle/adult/",
    "hat/helmet/legion/adult/",
    "hat/helmet/mail/adult/",
    "hat/helmet/nasal/adult/",
    "hat/helmet/norman/adult/",
    "hat/helmet/pointed/adult/",
    "hat/magic/celestial/adult/",
    "hat/magic/celestial_moon/adult/",
    "hat/magic/large/adult/",
    "hat/magic/wizard/base/adult/",
)


def parse_args() -> argparse.Namespace:
    with open(ROOT / "configs" / "experiments.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=int(config["project"]["dataset_size"]))
    parser.add_argument("--seed", type=int, default=int(config["project"]["dataset_seed"]))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_sources() -> None:
    required = (
        RAW / "CREDITS.csv",
        SPRITES / "body" / "bodies" / "male" / "idle.png",
        SPRITES / "body" / "bodies" / "female" / "idle.png",
        RAW / "palette_definitions" / "body" / "body_ulpc.json",
        RAW / "palette_definitions" / "hair" / "hair_ulpc.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan fuentes LPC. Ejecute scripts/fetch_lpc.ps1.\n" + "\n".join(missing)
        )


def relative_asset(path: Path) -> str:
    return path.relative_to(SPRITES).as_posix()


def is_idle(path: Path) -> bool:
    rel = relative_asset(path)
    return rel.endswith("/idle.png") or "/idle/" in rel


def style_key(path: Path) -> str:
    rel = relative_asset(path)
    if "/idle/" in rel:
        return rel.split("/idle/", 1)[0]
    return rel.rsplit("/idle.png", 1)[0]


def candidate_color(path: Path) -> str | None:
    rel = relative_asset(path)
    if "/idle/" not in rel:
        return None
    return path.stem


def discover_assets(
    category: str,
    body_token: str | None = None,
    fragments: tuple[str, ...] | None = None,
    single_layer: bool = True,
) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in (SPRITES / category).rglob("*.png"):
        if not is_idle(path):
            continue
        rel = relative_asset(path)
        if body_token is not None and f"/{body_token}/" not in f"/{rel}":
            continue
        if fragments is not None and not any(fragment in rel for fragment in fragments):
            continue
        if single_layer and any(part in {"bg", "fg", "mg"} for part in path.parts):
            continue
        color = candidate_color(path)
        if color is not None and color not in THEMATIC_COLORS:
            continue
        groups.setdefault(style_key(path), []).append(path)
    return {key: sorted(paths) for key, paths in sorted(groups.items()) if paths}


def choose_asset(rng: np.random.Generator, groups: dict[str, list[Path]]) -> tuple[str, Path]:
    styles = tuple(groups)
    style = styles[int(rng.integers(len(styles)))]
    options = groups[style]
    colored = [path for path in options if candidate_color(path) is not None]
    choices = colored or options
    return style, choices[int(rng.integers(len(choices)))]


@lru_cache(maxsize=512)
def load_front_frame(path_text: str) -> Image.Image:
    path = Path(path_text)
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        if rgba.width < 64 or rgba.height < 192:
            raise ValueError(f"Dimensiones idle inesperadas para {path}: {rgba.size}")
        frame = rgba.crop(FRAME_BOX)
    return frame


@lru_cache(maxsize=128)
def load_palette(group: str) -> dict[str, list[str]]:
    path = RAW / "palette_definitions" / group / f"{group}_ulpc.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def recolor(frame: Image.Image, group: str, base: str, target: str) -> Image.Image:
    palettes = load_palette(group)
    source_colors = palettes[base]
    target_colors = palettes[target]
    mapping = {
        tuple(int(source[index : index + 2], 16) for index in (1, 3, 5)): tuple(
            int(target_color[index : index + 2], 16) for index in (1, 3, 5)
        )
        for source, target_color in zip(source_colors, target_colors, strict=True)
    }
    pixels = np.array(frame, copy=True)
    for source, target_color in mapping.items():
        mask = np.all(pixels[:, :, :3] == source, axis=2) & (pixels[:, :, 3] > 0)
        pixels[mask, :3] = target_color
    return Image.fromarray(pixels, mode="RGBA")


def paste(canvas: Image.Image, path: Path | None) -> None:
    if path is None:
        return
    frame = load_front_frame(str(path)).copy()
    canvas.alpha_composite(frame)


def role_hint(torso_style: str, hat_style: str, weapon_color: str) -> str:
    if "magic" in hat_style:
        return "Arcanista del Velo"
    if "chainmail" in torso_style or "armour" in torso_style:
        return "Guardián rúnico" if weapon_color else "Centinela mineral"
    if "hood" in hat_style:
        return "Explorador del musgo"
    return "Caminante de Eryndor"


def compose_dataset(count: int, seed: int, overwrite: bool) -> pd.DataFrame:
    if MANIFEST.exists() and not overwrite:
        current = pd.read_csv(MANIFEST)
        if len(current) == count:
            used_assets: set[str] = set()
            for value in current["source_layers"].dropna():
                used_assets.update(str(value).split(";"))
            write_attribution(used_assets)
            print(f"Dataset existente reutilizado: {len(current):,} imágenes")
            return current
        raise FileExistsError("Ya existe un manifiesto. Use --overwrite para regenerarlo.")

    IMAGES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    pools: dict[str, dict[str, dict[str, list[Path]]]] = {}
    for body_type in ("male", "female"):
        lower_token = "male" if body_type == "male" else "thin"
        pools[body_type] = {
            "torso": discover_assets("torso", body_type, TORSO_FRAGMENTS),
            "legs": discover_assets("legs", lower_token, LEGS_FRAGMENTS),
            "feet": discover_assets("feet", lower_token, FEET_FRAGMENTS),
            "shoulders": discover_assets("shoulders", lower_token),
        }
        for name, pool in pools[body_type].items():
            if not pool:
                raise RuntimeError(f"Pool vacío: {body_type}/{name}")

    hair_all = discover_assets("hair", "adult")
    hair = {
        key: value
        for key, value in hair_all.items()
        if Path(key).parts[1] in HAIR_STYLES and "/extensions/" not in f"/{key}/"
    }
    hats = discover_assets("hat", "adult", HAT_FRAGMENTS)
    if not hair or not hats:
        raise RuntimeError("No fue posible construir los pools de cabello/sombreros.")

    sword_root = SPRITES / "weapon" / "sword" / "arming" / "universal"
    sword_colors = sorted(
        color
        for color in ("bronze", "gold", "iron", "silver", "steel")
        if (sword_root / "bg" / "idle" / f"{color}.png").exists()
        and (sword_root / "fg" / "idle" / f"{color}.png").exists()
    )

    body_palettes = load_palette("body")
    hair_palettes = load_palette("hair")
    body_tones = tuple(tone for tone in BODY_TONES if tone in body_palettes)
    hair_colors = tuple(color for color in HAIR_COLORS if color in hair_palettes)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    hashes: set[str] = set()
    used_assets: set[str] = set()
    attempts = 0
    max_attempts = count * 30

    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        body_type = ("male", "female")[int(rng.integers(2))]
        body_path = SPRITES / "body" / "bodies" / body_type / "idle.png"
        skin = body_tones[int(rng.integers(len(body_tones)))]
        hair_color = hair_colors[int(rng.integers(len(hair_colors)))]
        hair_style, hair_path = choose_asset(rng, hair)
        torso_style, torso_path = choose_asset(rng, pools[body_type]["torso"])
        legs_style, legs_path = choose_asset(rng, pools[body_type]["legs"])
        feet_style, feet_path = choose_asset(rng, pools[body_type]["feet"])

        shoulder_style = "none"
        shoulder_path: Path | None = None
        if rng.random() < 0.34:
            shoulder_style, shoulder_path = choose_asset(rng, pools[body_type]["shoulders"])

        hat_style = "none"
        hat_path: Path | None = None
        if rng.random() < 0.28:
            hat_style, hat_path = choose_asset(rng, hats)

        weapon_color = "none"
        sword_bg: Path | None = None
        sword_fg: Path | None = None
        if sword_colors and rng.random() < 0.22:
            weapon_color = sword_colors[int(rng.integers(len(sword_colors)))]
            sword_bg = sword_root / "bg" / "idle" / f"{weapon_color}.png"
            sword_fg = sword_root / "fg" / "idle" / f"{weapon_color}.png"

        canvas = Image.new("RGBA", (64, 64), BACKGROUND)
        paste(canvas, sword_bg)
        body_frame = recolor(load_front_frame(str(body_path)).copy(), "body", "light", skin)
        canvas.alpha_composite(body_frame)
        for layer_path in (legs_path, feet_path, torso_path, shoulder_path):
            paste(canvas, layer_path)
        hair_frame = recolor(load_front_frame(str(hair_path)).copy(), "hair", "orange", hair_color)
        canvas.alpha_composite(hair_frame)
        paste(canvas, hat_path)
        paste(canvas, sword_fg)
        rgb = canvas.convert("RGB")
        payload = rgb.tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in hashes:
            continue

        image_id = f"eryndor_{len(rows):04d}"
        output_path = IMAGES / f"{image_id}.png"
        rgb.save(output_path, optimize=True)
        hashes.add(digest)

        array = np.asarray(rgb)
        foreground = np.any(array != np.asarray(BACKGROUND[:3]), axis=2)
        layer_paths = [body_path, legs_path, feet_path, torso_path, hair_path]
        layer_paths.extend(path for path in (shoulder_path, hat_path, sword_bg, sword_fg) if path)
        relative_layers = [relative_asset(path) for path in layer_paths]
        used_assets.update(relative_layers)

        rows.append(
            {
                "image_id": image_id,
                "path": output_path.relative_to(ROOT).as_posix(),
                "sha256": digest,
                "body_type": body_type,
                "skin_tone": skin,
                "hair_style": hair_style.removeprefix("hair/"),
                "hair_color": hair_color,
                "torso_style": torso_style.removeprefix("torso/"),
                "torso_color": candidate_color(torso_path) or "base",
                "legs_style": legs_style.removeprefix("legs/"),
                "feet_style": feet_style.removeprefix("feet/"),
                "shoulder_style": shoulder_style.removeprefix("shoulders/"),
                "hat_style": hat_style.removeprefix("hat/"),
                "weapon": weapon_color,
                "role_hint": role_hint(torso_style, hat_style, weapon_color),
                "foreground_ratio": round(float(foreground.mean()), 6),
                "mean_brightness": round(float(array.mean() / 255.0), 6),
                "unique_colors": int(np.unique(array.reshape(-1, 3), axis=0).shape[0]),
                "source_layers": ";".join(relative_layers),
            }
        )

        if len(rows) % 512 == 0:
            print(f"Generadas {len(rows):,}/{count:,} imágenes únicas")

    if len(rows) != count:
        raise RuntimeError(f"Solo se lograron {len(rows)} imágenes únicas en {attempts} intentos")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False, encoding="utf-8")
    write_attribution(used_assets)
    print(f"Manifiesto: {MANIFEST}")
    print(f"Tasa de duplicados descartados: {(attempts - count) / attempts:.2%}")
    return manifest


def write_attribution(used_assets: set[str]) -> None:
    credits = pd.read_csv(RAW / "CREDITS.csv")
    credits_by_filename = credits.set_index("filename", drop=False)
    resolved_rows: list[dict[str, object]] = []
    unmatched: list[str] = []
    for asset in sorted(used_assets):
        credit_file = asset
        if credit_file not in credits_by_filename.index and "/idle/" in asset:
            credit_file = asset.split("/idle/", 1)[0] + "/idle.png"
        if credit_file not in credits_by_filename.index:
            unmatched.append(asset)
            continue
        row = credits_by_filename.loc[credit_file]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        resolved_rows.append({"used_asset": asset, **row.to_dict()})

    selected = pd.DataFrame(resolved_rows)
    selected.sort_values(["filename", "used_asset"]).to_csv(
        ROOT / "docs" / "LPC_CREDITS_USED.csv", index=False, encoding="utf-8"
    )

    commit = subprocess.run(
        ["git", "-C", str(RAW), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    licenses = sorted(
        {
            item.strip()
            for value in selected["licenses"].dropna()
            for item in str(value).split(",")
            if item.strip()
        }
    )
    authors = sorted(
        {
            item.strip()
            for value in selected["authors"].dropna()
            for item in str(value).split(",")
            if item.strip()
        }
    )
    lines = [
        "# Atribución de recursos LPC",
        "",
        "Las imágenes del dataset son composiciones derivadas de capas del proyecto "
        "[Universal LPC Spritesheet Character Generator]"
        "(https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator).",
        "",
        f"- Commit fuente: `{commit}`",
        f"- Archivos de capa distintos utilizados: **{len(used_assets)}**",
        f"- Capas con crédito resuelto: **{len(selected)}**",
        f"- Capas sin crédito resuelto: **{len(unmatched)}**",
        f"- Licencias declaradas: {', '.join(licenses)}",
        "",
        "Las variantes recoloreadas se vinculan al crédito de su archivo base `idle.png`. "
        "El detalle por capa usada, archivo acreditado, autores, licencias y URLs se conserva "
        "en `docs/LPC_CREDITS_USED.csv`. El `CREDITS.csv` completo permanece junto al clon fuente.",
        "",
        "## Autores acreditados",
        "",
        ", ".join(authors),
    ]
    if unmatched:
        lines.extend(("", "## Sin coincidencia exacta", "", *[f"- `{item}`" for item in unmatched]))
    (ROOT / "docs" / "LPC_ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_audit(manifest: pd.DataFrame, seed: int) -> dict[str, object]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    sample_indices = rng.choice(len(manifest), size=min(64, len(manifest)), replace=False)
    fig, axes = plt.subplots(8, 8, figsize=(10, 10), facecolor="#171A24")
    for axis, index in zip(axes.flat, sample_indices, strict=False):
        row = manifest.iloc[int(index)]
        with Image.open(ROOT / row["path"]) as image:
            axis.imshow(image, interpolation="nearest")
        axis.set_title(str(row["image_id"])[-4:], color="#E6E6FA", fontsize=7)
        axis.axis("off")
    fig.suptitle("Eryndor · muestra determinista del dataset", color="white", fontsize=15, y=0.995)
    fig.tight_layout(pad=0.45)
    fig.savefig(FIGURES / "dataset_contact_sheet.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    palette = ["#4A4E8F", "#4FC3C8", "#D6A34A", "#A490C2", "#79A879"]
    manifest["body_type"].value_counts().plot.bar(ax=axes[0, 0], color=palette[:2])
    axes[0, 0].set(title="Tipo corporal", xlabel="", ylabel="Imágenes")
    manifest["skin_tone"].value_counts().plot.bar(ax=axes[0, 1], color="#D6A34A")
    axes[0, 1].set(title="Tonos de piel", xlabel="", ylabel="Imágenes")
    manifest["hair_color"].value_counts().plot.bar(ax=axes[1, 0], color="#A490C2")
    axes[1, 0].set(title="Colores de cabello", xlabel="", ylabel="Imágenes")
    axes[1, 1].hist(manifest["foreground_ratio"], bins=18, color="#4FC3C8", edgecolor="white")
    axes[1, 1].axvline(manifest["foreground_ratio"].mean(), color="#171A24", ls="--", lw=1.5)
    axes[1, 1].set(title="Ocupación del personaje", xlabel="Proporción de píxeles", ylabel="Frecuencia")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle(
        f"Auditoría de composición · {len(manifest):,} sprites",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    duplicate_hashes = int(manifest["sha256"].duplicated().sum())
    missing_files = int(sum(not (ROOT / path).exists() for path in manifest["path"]))
    summary: dict[str, object] = {
        "image_count": int(len(manifest)),
        "resolution": [64, 64],
        "channels": 3,
        "duplicate_hashes": duplicate_hashes,
        "missing_files": missing_files,
        "body_type_counts": manifest["body_type"].value_counts().to_dict(),
        "skin_tone_counts": manifest["skin_tone"].value_counts().to_dict(),
        "hair_color_counts": manifest["hair_color"].value_counts().to_dict(),
        "unique_hair_styles": int(manifest["hair_style"].nunique()),
        "unique_torso_styles": int(manifest["torso_style"].nunique()),
        "foreground_ratio": {
            "min": float(manifest["foreground_ratio"].min()),
            "mean": float(manifest["foreground_ratio"].mean()),
            "max": float(manifest["foreground_ratio"].max()),
        },
        "mean_brightness": float(manifest["mean_brightness"].mean()),
        "dataset_seed": seed,
    }
    with open(METRICS / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def validate_dataloader() -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from src.data import build_dataloader

    loader = build_dataloader(
        manifest_path=MANIFEST,
        project_root=ROOT,
        batch_size=64,
        seed=42,
        shuffle=True,
        horizontal_flip=True,
        num_workers=0,
    )
    batch = next(iter(loader))
    result = {
        "shape": list(batch.shape),
        "dtype": str(batch.dtype),
        "min": float(batch.min()),
        "max": float(batch.max()),
        "finite": bool(np.isfinite(batch.numpy()).all()),
    }
    if result["shape"] != [64, 3, 64, 64] or result["min"] < -1.0001 or result["max"] > 1.0001:
        raise AssertionError(f"DataLoader inválido: {result}")
    with open(METRICS / "dataloader_smoke_test.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    args = parse_args()
    require_sources()
    manifest = compose_dataset(args.count, args.seed, args.overwrite)
    summary = make_audit(manifest, args.seed)
    loader_result = validate_dataloader()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"DataLoader validado: {loader_result}")


if __name__ == "__main__":
    main()
