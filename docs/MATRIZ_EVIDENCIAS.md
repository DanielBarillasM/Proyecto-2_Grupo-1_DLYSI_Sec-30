# Matriz final de evidencias

Esta matriz vincula cada exigencia del Proyecto 2 con una evidencia regenerable. Las métricas proceden de artefactos del repositorio; no se transcribieron desde capturas.

| Evidencia | Presentación | Archivo fuente | Conclusión defendible | Limitación declarada |
|---|---:|---|---|---|
| Universo y lenguaje visual | 2 | `BRIEF.md`, `docs/THEME.md` | Eryndor define un mundo RPG de magia mineral coherente con los sprites. | La narrativa se asigna después de la selección técnica. |
| Dataset, licencia y preparación | 3 | `data/processed/manifest.csv`, `docs/LPC_CREDITS_USED.csv`, `artifacts/metrics/dataset_summary.json` | 4,096 RGB de 64×64; 0 faltantes, 0 duplicados y 254/254 capas acreditadas. | Una pose frontal y bases anatómicas compartidas. |
| Arquitectura y reproducción | 4 | `src/models.py`, `src/training.py`, `configs/experiments.yaml` | DCGAN y checkpoints respetan formas, semillas y restauración exacta. | DCGAN limitada a 64×64. |
| Hipótesis controladas A/B/C | 5 | `docs/PLAN_EXPERIMENTAL.md`, `artifacts/phase8/experiment_analysis.csv` | Cada comparación modifica solo pérdida o estabilización. | Una corrida completa por variante. |
| Ruido fijo | 6 | `artifacts/phase8/fixed_noise_milestones.png` | Los mismos 16 vectores muestran aprendizaje en A/C y colapso en B. | La distancia L2 es un proxy, no una métrica perceptual. |
| Pérdidas y estabilidad | 7–8 | `artifacts/phase8/phase8_diagnostics.png`, `artifacts/experiments/comparison_summary.csv` | Hinge colapsó; C conservó la mayor diversidad final, 0.2659. | BCE y hinge tienen escalas no comparables directamente. |
| Galería y criterio 10/200 | 9 | `galeria/manifest.csv`, `artifacts/gallery/final_gallery_grid.png` | 200 generados, 97 elegibles, 10 elegidos; tasa declarada de 5%. | Los nombres se asignaron después del ranking. |
| Vecinos más cercanos | 10 | `artifacts/gallery/nearest_neighbors.png`, `artifacts/phase9/phase9_summary.json` | 0 duplicados exactos y 0 banderas de revisión con coseno + MSE. | ResNet18 se entrenó en ImageNet, no específicamente en pixel art. |
| Regeneración y procedencia | 9–10 | `galeria/latents.npz`, `galeria/provenance.json`, `artifacts/gallery/validation.json` | Los diez PNG tienen hashes únicos, coinciden con fase 10 y regeneran con delta RGB 0. | El checkpoint pesado debe acompañar la entrega o tener enlace público. |
| Reflexión final | 11 | `galeria/manifest.csv`, notebook e informe | Bram es el caso más cercano: coseno 0.7911, MSE 0.01485. | La gramática LPC está bien aprendida, pero la diversidad estructural es reducida. |
| Cierre y trazabilidad | 12 | `scripts/validate_delivery.py`, `artifacts/delivery/validation.json` | README, notebook, presentación, galería y reporte comparten la misma fuente de verdad. | La validación automática no sustituye la defensa oral ni la revisión visual. |

## Archivos para Canvas

- código fuente y notebooks;
- `presentation/presentacion.pdf` (12 diapositivas);
- carpeta `galeria/` con diez PNG, `latents.npz`, manifiesto y procedencia;
- `checkpoints/bce_spectral_norm/latest.pt` o un enlace público verificable con el SHA-256 documentado;
- informe adicional `report/informe_final_eryndor.tex` si el equipo desea compilarlo y adjuntarlo.

Ninguna imagen de la galería proviene de un generador externo.
