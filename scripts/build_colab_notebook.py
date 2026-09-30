"""Construye el notebook tutorial de entrenamiento reanudable para Google Colab."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "02_entrenamiento_colab.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    md(
        r"""
<div class="eryndor-hero">
  <div class="hero-mark">VII</div>
  <div>
    <p class="hero-kicker">Proyecto 2 · Entrenamiento reproducible</p>
    <h1>Eryndor en Google Colab</h1>
    <p>GPU, reanudación automática y respaldo a Drive en cada época.</p>
  </div>
</div>

**Integrantes:** Pablo Daniel Barillas Moreno · Wilson Alejandro Calderón  
**Objetivo:** completar de forma controlada los experimentos A/B/C hasta la época 60 sin perder el estado ante una desconexión.
"""
    ),
    code(
        r'''
from IPython.display import HTML, display

display(HTML(r"""
<style>
:root { --ink:#171A24; --veil:#2B1E3E; --mineral:#4FC3C8; --ember:#D6A34A; --mist:#E6E6FA; }
.container { max-width: 1120px !important; }
.eryndor-hero { display:flex; gap:24px; align-items:center; padding:28px 32px; color:var(--mist);
  background:linear-gradient(120deg,var(--ink),var(--veil)); border-left:8px solid var(--mineral); }
.hero-mark { display:grid; place-items:center; width:84px; height:84px; border:1px solid #ffffff44;
  color:var(--ember); font-size:2.2rem; font-weight:800; transform:rotate(45deg); }
.hero-mark::first-line { transform:rotate(-45deg); }
.eryndor-hero h1 { color:white; margin:.1rem 0 .25rem; font-size:2.5rem; }
.eryndor-hero p { margin:.2rem 0; }
.hero-kicker { color:var(--mineral); font-weight:700; }
.phase-note { border-left:5px solid var(--mineral); background:#eef9f9; padding:14px 18px; }
.warning-note { border-left:5px solid var(--ember); background:#fff7e7; padding:14px 18px; }
h2 { color:var(--veil); margin-top:2rem; }
code { color:#654a91; }
</style>
"""))
'''
    ),
    md(
        r"""
## Antes de comenzar

Este notebook está pensado para la pareja responsable del proyecto. Requiere:

1. activar un entorno **GPU** en Colab;
2. copiar la carpeta del proyecto a `Mi unidad/Proyecto-2_Grupo-1_DLYSI_Sec-30`;
3. conservar sin cambios `configs/experiments.yaml` mientras corren A/B/C.

Al terminar podrán:

- reanudar cada experimento desde su último checkpoint;
- comparar BCE, hinge y BCE con normalización espectral bajo el mismo control;
- conservar métricas, ruido fijo y pesos en Google Drive;
- confirmar si la fase 8 ya puede interpretar las trayectorias completas.

<div class="warning-note"><strong>No cierre la pestaña durante una época.</strong> El estado se respalda al terminar cada época. Si la sesión cae, vuelva a ejecutar desde el inicio: el entrenamiento detectará el checkpoint más reciente.</div>
"""
    ),
    md(
        r"""
## 1. Montar Drive

Drive conserva checkpoints y métricas. El dataset se reconstruye en el disco rápido del runtime porque es reproducible y no conviene leer miles de PNG directamente desde Drive.
"""
    ),
    code(
        r'''
from google.colab import drive

drive.mount("/content/drive")
print("Drive montado.")
'''
    ),
    md("## 2. Configurar rutas\n\nCambie `DRIVE_ROOT` únicamente si la carpeta tiene otro nombre."),
    code(
        r'''
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

DRIVE_ROOT = Path("/content/drive/MyDrive/Proyecto-2_Grupo-1_DLYSI_Sec-30")
RUNTIME_ROOT = Path("/content/eryndor")
TARGET_EPOCHS = 60
NUM_WORKERS = 2

required = DRIVE_ROOT / "configs" / "experiments.yaml"
if not required.exists():
    raise FileNotFoundError(
        f"No se encontró {required}. Copie primero la carpeta completa del proyecto a Drive."
    )
print(f"Fuente persistente: {DRIVE_ROOT}")
print(f"Copia rápida de trabajo: {RUNTIME_ROOT}")
'''
    ),
    md("## 3. Preparar la copia rápida y restaurar avances"),
    code(
        r'''
ignore = shutil.ignore_patterns(
    ".git", ".venv", "__pycache__", "data", "checkpoints", "artifacts",
    "presentacion.aux", "presentacion.log"
)
shutil.copytree(DRIVE_ROOT, RUNTIME_ROOT, dirs_exist_ok=True, ignore=ignore)

for relative in (Path("checkpoints"), Path("artifacts/experiments"), Path("artifacts/phase8")):
    source = DRIVE_ROOT / relative
    if source.exists():
        shutil.copytree(source, RUNTIME_ROOT / relative, dirs_exist_ok=True)

os.chdir(RUNTIME_ROOT)
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))
print("Código y avances restaurados.")
'''
    ),
    md("## 4. Verificar dependencias y GPU"),
    code(
        r'''
import importlib.util

required_packages = {
    "yaml": "PyYAML>=6,<7",
    "pandas": "pandas>=2.2,<3",
    "matplotlib": "matplotlib>=3.8,<4",
    "PIL": "Pillow>=10,<13",
    "sklearn": "scikit-learn>=1.5,<2",
    "tqdm": "tqdm>=4.66,<5",
}
missing = [package for module, package in required_packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
print("Dependencias listas.")
'''
    ),
    code(
        r'''
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA no está disponible. En Colab seleccione Entorno de ejecución > Cambiar tipo > GPU.")

properties = torch.cuda.get_device_properties(0)
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {properties.total_memory / 1024**3:.1f} GiB")
print("cuDNN benchmark permanece desactivado para favorecer reproducibilidad.")
'''
    ),
    md(
        r"""
## 5. Reconstruir y auditar el dataset

El descargador Python sustituye al script PowerShell en Colab. La composición conserva semilla `2026`, 4,096 imágenes y las mismas capas acreditadas.
"""
    ),
    code(
        r'''
subprocess.run([sys.executable, "scripts/fetch_lpc.py"], cwd=RUNTIME_ROOT, check=True)
subprocess.run(
    [sys.executable, "scripts/prepare_dataset.py", "--count", "4096"],
    cwd=RUNTIME_ROOT,
    check=True,
)
'''
    ),
    code(
        r'''
import pandas as pd

summary = json.loads((RUNTIME_ROOT / "artifacts/metrics/dataset_summary.json").read_text())
manifest = pd.read_csv(RUNTIME_ROOT / "data/processed/manifest.csv")
assert len(manifest) == 4096
assert int(summary["duplicate_hashes"]) == 0
display(pd.DataFrame([{
    "Imágenes": len(manifest),
    "Duplicados": summary["duplicate_hashes"],
    "Faltantes": summary["missing_files"],
    "Resolución": "64 × 64 RGB",
}]))
'''
    ),
    md(
        r"""
## 6. Protocolo que no debe cambiar

| ID | Configuración | Única variable modificada |
|---|---|---|
| A · `baseline_bce` | BCE no saturante | Control |
| B · `hinge_loss` | Hinge loss | Pérdida |
| C · `bce_spectral_norm` | BCE no saturante + spectral norm en D | Estabilización |

Los tres usan exactamente los mismos datos, semilla de modelo `42`, ruido fijo `777`, batch `64`, optimizadores y 60 épocas.
"""
    ),
    md("## 7. Diagnóstico corto\n\nEl smoke test confirma contratos y gradientes antes de gastar horas de GPU."),
    code(
        r'''
subprocess.run([sys.executable, "scripts/smoke_test_gan.py"], cwd=RUNTIME_ROOT, check=True)
smoke = json.loads((RUNTIME_ROOT / "artifacts/smoke_test/smoke_metrics.json").read_text())
print(f"Smoke test: {smoke['status'].upper()}")
'''
    ),
    md(
        r"""
## 8. Entrenar y reanudar

Cada llamada revisa `latest.pt`, restaura pesos, optimizadores, ruido fijo, RNG y orden del `DataLoader`. `--backup-root` refleja el estado en Drive después de cada época.
"""
    ),
    code(
        r'''
EXPERIMENTS = ("baseline_bce", "hinge_loss", "bce_spectral_norm")

def run_experiment(experiment: str, end_epoch: int = TARGET_EPOCHS) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(experiment)
    command = [
        sys.executable,
        "scripts/train_experiment.py",
        "--experiment", experiment,
        "--epochs", str(end_epoch),
        "--device", "cuda",
        "--num-workers", str(NUM_WORKERS),
        "--backup-root", str(DRIVE_ROOT),
    ]
    subprocess.run(command, cwd=RUNTIME_ROOT, check=True)

def show_progress() -> pd.DataFrame:
    rows = []
    for experiment in EXPERIMENTS:
        path = RUNTIME_ROOT / "artifacts/experiments" / experiment / "status.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        rows.append({
            "Experimento": experiment,
            "Épocas": state.get("completed_epochs", 0),
            "Objetivo": TARGET_EPOCHS,
            "Estado": state.get("status", "sin iniciar"),
        })
    return pd.DataFrame(rows)

display(show_progress())
'''
    ),
    md("### 8.1 Experimento A · baseline BCE"),
    code('run_experiment("baseline_bce")\ndisplay(show_progress())'),
    md("### 8.2 Experimento B · hinge loss"),
    code('run_experiment("hinge_loss")\ndisplay(show_progress())'),
    md("### 8.3 Experimento C · BCE con spectral norm"),
    code('run_experiment("bce_spectral_norm")\ndisplay(show_progress())'),
    md(
        r"""
## 9. Comparar sin elegir por una sola cifra

Las pérdidas BCE y hinge no comparten escala. La fase siguiente debe interpretar trayectoria, rejillas fijas, diversidad, estabilidad y vecinos; una pérdida menor no decide al ganador.
"""
    ),
    code(
        r'''
subprocess.run([sys.executable, "scripts/compare_experiments.py"], cwd=RUNTIME_ROOT, check=True)
comparison = pd.read_csv(RUNTIME_ROOT / "artifacts/experiments/comparison_summary.csv")
display(comparison[[
    "experiment", "completed_epochs", "loss_d", "loss_g",
    "real_logit", "fake_logit_g", "fixed_pairwise_l2", "mean_epoch_seconds"
]].style.format(precision=4).hide(axis="index"))

for filename in ("comparison_summary.csv", "comparison_status.json", "experiment_comparison.png", "latest_fixed_noise_comparison.png"):
    source = RUNTIME_ROOT / "artifacts/experiments" / filename
    if source.exists():
        destination = DRIVE_ROOT / "artifacts/experiments" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
print("Comparación final respaldada en Drive.")
'''
    ),
    md(
        r"""
## 10. Comprobación de reanudación

**Ejercicio:** vuelva a solicitar la época ya alcanzada para A. La ejecución no debe reiniciar el modelo ni agregar filas duplicadas.
"""
    ),
    code(
        r'''
baseline_status = json.loads(
    (RUNTIME_ROOT / "artifacts/experiments/baseline_bce/status.json").read_text()
)
run_experiment("baseline_bce", end_epoch=int(baseline_status["completed_epochs"]))

history = pd.read_csv(RUNTIME_ROOT / "artifacts/experiments/baseline_bce/epoch_metrics.csv")
assert history["epoch"].is_unique
assert int(history["epoch"].max()) == int(baseline_status["completed_epochs"])
print("Reanudación idempotente verificada: no se duplicaron épocas.")
'''
    ),
    md(
        r"""
## 11. Cierre de la fase 7

Ejecute esta celda solo después de A/B/C. Si todo llegó a 60, la fase 8 puede interpretar pérdidas y muestras por época.
"""
    ),
    code(
        r'''
progress = show_progress()
display(progress)
complete = bool((progress["Épocas"] >= TARGET_EPOCHS).all())
if complete:
    display(HTML('<div class="phase-note"><strong>Fase 7 completa.</strong> Los tres experimentos alcanzaron 60 épocas y sus estados están respaldados.</div>'))
else:
    display(HTML('<div class="warning-note"><strong>Entrenamiento pendiente.</strong> Reanude únicamente las celdas de los experimentos incompletos.</div>'))
'''
    ),
    md(
        r"""
## 12. Fase 9 · vecinos más cercanos

Esta sección usa el modelo C seleccionado en la fase 8 y las 16 entradas de ruido fijo. Cada salida se compara contra las 4,096 imágenes reales mediante ResNet18, similitud coseno y MSE. No genera todavía los 200 candidatos ni selecciona la galería final.

El control lee la época **dentro** de `latest.pt`; un CSV en 60/60 no sustituye al checkpoint final.
"""
    ),
    code(
        r'''
subprocess.run(
    [sys.executable, "scripts/check_phase9_readiness.py"],
    cwd=RUNTIME_ROOT,
    check=True,
)
phase9_readiness = json.loads(
    (RUNTIME_ROOT / "artifacts/phase9/readiness.json").read_text()
)
display(pd.DataFrame([{
    "Modelo": phase9_readiness["selected_experiment"],
    "Checkpoint": f'{phase9_readiness["checkpoint_epoch"]}/{phase9_readiness["target_epoch"]}',
    "Dataset": phase9_readiness["dataset_manifest_rows"],
    "ResNet18": "lista" if phase9_readiness["resnet18"]["cached"] else "se descargará",
    "Estado": phase9_readiness["status"],
}]))

if phase9_readiness["checkpoint_epoch"] != TARGET_EPOCHS:
    raise RuntimeError(
        "Drive no contiene latest.pt de época 60 para bce_spectral_norm. "
        "Restaure el respaldo final antes de continuar."
    )
'''
    ),
    code(
        r'''
subprocess.run(
    [
        sys.executable,
        "scripts/analyze_phase9_neighbors.py",
        "--device", "cuda",
        "--feature-batch-size", "128",
    ],
    cwd=RUNTIME_ROOT,
    check=True,
)

phase9_dir = RUNTIME_ROOT / "artifacts/phase9"
drive_phase9 = DRIVE_ROOT / "artifacts/phase9"
shutil.copytree(phase9_dir, drive_phase9, dirs_exist_ok=True)
phase9_summary = json.loads((phase9_dir / "phase9_summary.json").read_text())
display(pd.DataFrame([{
    "Muestras": phase9_summary["sample_count"],
    "Coseno medio": phase9_summary["cosine_similarity"]["mean"],
    "MSE medio": phase9_summary["pixel_mse"]["mean"],
    "Duplicados exactos": phase9_summary["exact_pixel_duplicate_count"],
    "Banderas de revisión": phase9_summary["screening_flag_count"],
}]).style.format(precision=4).hide(axis="index"))

display(HTML('<div class="phase-note"><strong>Fase 9 ejecutada.</strong> Revise la figura de vecinos antes de iniciar los 200 candidatos.</div>'))
'''
    ),
    md(
        r"""
## Problemas frecuentes

- **CUDA no disponible:** cambie el tipo de entorno a GPU y reinicie desde la sección 1.
- **La sesión se desconectó:** monte Drive, restaure la copia rápida y vuelva a ejecutar el experimento; no use `--no-resume`.
- **Drive está lento:** es normal durante el respaldo; el dataset y el entrenamiento viven en `/content`, no en Drive.
- **Pérdidas muy distintas entre BCE y hinge:** compare tendencias y evidencia visual, no magnitudes absolutas.

Extensión opcional: pruebe una GPU distinta solo si conserva la misma configuración experimental y documenta el hardware; no mezcle cambios de arquitectura dentro de A/B/C.
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "colab": {"name": "02_entrenamiento_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "title": "Eryndor - Entrenamiento reproducible en Google Colab",
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8", newline="\n") as file:
    nbf.write(notebook, file)
print(f"Notebook creado: {OUTPUT}")

