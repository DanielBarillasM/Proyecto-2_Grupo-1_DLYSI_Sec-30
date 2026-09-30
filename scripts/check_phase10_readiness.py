"""Comprueba si la selección transparente de 200 a 10 puede ejecutarse."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import sha256_file
from src.training import write_json


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    selected = json.loads(
        (ROOT / "artifacts" / "phase8" / "selected_model.json").read_text(encoding="utf-8")
    )
    experiment = str(selected["experiment"])
    checkpoint_path = ROOT / str(selected["checkpoint"])
    target_epoch = int(config["training"]["epochs"])
    reasons: list[str] = []
    checkpoint_epoch: int | None = None
    checkpoint_hash: str | None = None
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_epoch = int(checkpoint["epoch"])
            checkpoint_hash = sha256_file(checkpoint_path)
        except Exception as error:  # pragma: no cover - depende del archivo externo
            reasons.append(f"checkpoint ilegible ({type(error).__name__})")
    else:
        reasons.append(f"falta {checkpoint_path.relative_to(ROOT).as_posix()}")
    if checkpoint_epoch is not None and checkpoint_epoch < target_epoch:
        reasons.append(f"checkpoint C: época interna {checkpoint_epoch}/{target_epoch}")

    phase9_path = ROOT / "artifacts" / "phase9" / "phase9_summary.json"
    phase9_status: str | None = None
    phase9_checkpoint_hash: str | None = None
    if phase9_path.exists():
        phase9 = json.loads(phase9_path.read_text(encoding="utf-8"))
        phase9_status = str(phase9.get("status"))
        phase9_checkpoint_hash = phase9.get("checkpoint_sha256")
        if phase9_status != "passed":
            reasons.append("la fase 9 no está aprobada")
        if checkpoint_hash and phase9_checkpoint_hash != checkpoint_hash:
            reasons.append("la fase 9 se ejecutó con un checkpoint distinto")
    else:
        reasons.append("falta artifacts/phase9/phase9_summary.json")

    candidate_count = int(config["gallery"]["candidate_count"])
    selected_count = int(config["gallery"]["final_count"])
    payload = {
        "status": "ready" if not reasons else "blocked",
        "phase": 10,
        "selected_experiment": experiment,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_epoch": checkpoint_epoch,
        "target_epoch": target_epoch,
        "checkpoint_sha256": checkpoint_hash,
        "phase9_status": phase9_status,
        "phase9_checkpoint_sha256": phase9_checkpoint_hash,
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "selection_rate": selected_count / candidate_count,
        "blocking_reasons": reasons,
        "next_command": (
            "python scripts/run_phase10_selection.py --device auto"
            if not reasons
            else "completar fase 9 con el checkpoint C de época 60"
        ),
    }
    write_json(ROOT / "artifacts" / "phase10" / "readiness.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
