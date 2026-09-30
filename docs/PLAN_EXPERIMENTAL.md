# Plan experimental

## Pregunta central

¿Puede una GAN entrenada sobre sprites abiertos y visualmente consistentes generar personajes de Eryndor que sean plausibles, diversos y distintos de sus vecinos más cercanos en entrenamiento?

## Datos

- Fuente: Universal LPC Spritesheet Character Generator.
- Objetivo: 4,096 sprites frontales de cuerpo completo.
- Resolución: 64 x 64 RGB.
- Fondo: color neutro oscuro fijo para evitar que la transparencia se convierta en un cuarto canal.
- Preparación: composición de capas, extracción del cuadro frontal, deduplicación por hash, RGB y normalización a `[-1, 1]`.
- Aumento común a todos los experimentos: flip horizontal con probabilidad 0.5, sin interpolación.
- Semilla de construcción del dataset: `2026`.

El repositorio LPC incluye arte bajo varias licencias abiertas. Se conservará `CREDITS.csv`, se documentarán las capas utilizadas y se respetará la licencia más restrictiva aplicable a cada composición. La galería se presenta como resultado académico, no como material comercial.

## Arquitectura base

> **Estado de implementación:** puntos 1–8 completados. A/B/C alcanzaron 60/60 épocas y la auditoría selecciona provisionalmente `bce_spectral_norm`. Los puntos 9–11 tienen infraestructura adelantada; el checkpoint C está disponible y falta descargar ResNet18 para vecinos y galería.

## Ejecución de la fase 7 en Colab

El notebook `notebooks/02_entrenamiento_colab.ipynb` ejecuta el protocolo sin modificar sus controles. Trabaja sobre el disco local del runtime y usa Google Drive como respaldo persistente mediante `--backup-root`.

- `latest.pt` se guarda al finalizar cada época;
- métricas, estado y curvas se reflejan en Drive junto con el checkpoint;
- las instantáneas históricas del generador y las rejillas permanecen cada cinco épocas;
- una sesión nueva restaura automáticamente el respaldo si no existe un checkpoint local;
- A/B/C mantienen datos, semilla, arquitectura, optimizadores y ruido fijo idénticos.

### Generador

- Vector latente: 128 dimensiones.
- Proyección mediante convoluciones transpuestas hasta 64 x 64.
- Canales: 512, 256, 128, 64 y 3.
- Batch normalization y ReLU en bloques internos.
- `tanh` en la salida.

### Discriminador

- Convoluciones con canales 64, 128, 256 y 512.
- LeakyReLU con pendiente 0.2.
- Batch normalization desde el segundo bloque.
- Salida escalar en logits, sin sigmoide dentro del modelo.

### Configuración común

- Épocas: 60.
- Batch size: 64.
- Optimizador: Adam.
- Learning rate: `2e-4`.
- Betas: `(0.5, 0.999)`.
- Semilla de modelos: `42`.
- Semilla del ruido fijo: `777`.
- Checkpoint: cada 5 épocas.
- Rejilla fija: 16 vectores `z`.

## Experimentos controlados

| ID | Pérdida | Estabilización | Variable modificada |
|---|---|---|---|
| A | BCE no saturante | Ninguna adicional | Baseline |
| B | Hinge loss | Ninguna adicional | Solo pérdida |
| C | BCE no saturante | Normalización espectral en D | Solo estabilización |

### Hipótesis A - baseline

Creemos que la DCGAN con BCE no saturante producirá siluetas reconocibles porque el dataset tiene resolución y estilo consistentes. Lo consideraremos útil si la rejilla fija muestra progreso sin que la mayoría de las muestras converja a una sola apariencia.

### Hipótesis B - pérdida

Creemos que hinge loss mejorará la nitidez y la separación entre rasgos porque mantiene gradientes útiles sin interpretar la salida del discriminador como una probabilidad calibrada. Lo consideraremos útil si mejora la calidad visual y la diversidad sin aumentar la cercanía a imágenes de entrenamiento.

### Hipótesis C - estabilización

Creemos que la normalización espectral estabilizará al discriminador porque limita la magnitud efectiva de sus capas. Lo consideraremos útil si reduce periodos prolongados de dominio de D y mantiene una evolución más consistente en el ruido fijo.

## Evidencias obligatorias

### Ruido fijo

Se conservarán las mismas 16 entradas latentes en todos los experimentos y se mostrarán en las épocas 0, 5, 10, 20, 40 y 60.

### Curvas

Se registrarán por época:

- pérdida de G;
- pérdida de D;
- puntuación media de D para datos reales;
- puntuación media de D para datos falsos;
- tiempo por época.

Las curvas se interpretarán como dinámica adversarial, no como un problema supervisado donde ambas pérdidas deban disminuir juntas.

### Vecinos más cercanos

- Extractor: ResNet18 preentrenada sin la capa final.
- Representación: embedding normalizado.
- Similitud: coseno.
- Comprobación secundaria: MSE en píxeles.
- Salida: cada personaje final junto a su vecino de entrenamiento, similitud y comentario.

## Selección final

1. Generar 200 candidatos con semilla `20261011`.
2. Descartar imágenes con ocupación o contraste degenerado.
3. Priorizar candidatos alejados de su vecino de entrenamiento.
4. Aplicar selección greedy para maximizar diversidad entre elegidos.
5. Revisar coherencia con el brief sin ocultar la tasa de selección.
6. Guardar 10 PNG, `latents.npz` y `manifest.csv`.

Tasa declarada prevista: `10 / 200 = 5%`.

## Criterio de selección del modelo final

No se elegirá por una sola pérdida. Se considerarán conjuntamente:

1. evolución estable del ruido fijo;
2. diversidad entre muestras;
3. separación respecto a vecinos de entrenamiento;
4. coherencia visual con Eryndor;
5. ausencia de colapso evidente.

## Riesgos

- **Mode collapse:** monitorear la rejilla fija y las distancias entre candidatos.
- **Memorización:** vecinos cercanos en embeddings y píxeles.
- **Discriminador dominante:** inspeccionar logits reales/falsos y periodos con pérdida casi nula.
- **Datos heterogéneos:** filtrar capas o composiciones que rompan perspectiva y escala.
- **Interrupción de Colab:** checkpoint rodante cada época, instantáneas cada cinco épocas y reanudación completa del optimizador.
- **Licencias mixtas:** conservar créditos por capa y no presentar el material como propietario.

## Verificación de fase 3

- Formas verificadas: `z=[B,128,1,1]`, `G(z)=[B,3,64,64]`, `D(x)=[B]`.
- Parámetros entrenables: 3,806,080 en G y 2,765,568 en D.
- BCE y hinge: pérdidas y gradientes finitos.
- Normalización espectral: activa en las cinco convoluciones de D solo para el experimento C.
- Checkpoint: modelos, optimizadores, ruido fijo, época, paso, estados aleatorios y estado del generador que controla el orden del `DataLoader`.
- Round-trip del checkpoint: error máximo absoluto 0 en ruido fijo.
- Smoke test baseline: ocho pasos, batch 8, 64 imágenes vistas.

El smoke test mostró dominio temprano del discriminador: `loss_D` cayó de 1.879 a 0.153 y `loss_G` subió de 4.970 a 7.522. Esta observación no decide el modelo final, pero define una señal que debe vigilarse en las primeras épocas de los tres experimentos.

## Resultado completo de fase 8

Los tres experimentos ejecutaron 60 épocas sobre las 4,096 imágenes (`64 pasos × batch 64`) con semillas, arquitectura y ruido fijo pre-registrados. La auditoría confirmó 180 filas por época y 11,520 pasos sin duplicados, faltantes ni valores no finitos.

| Experimento | Épocas | loss D | loss G | Logit real | Logit falso para G | Distancia fija media |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_bce` | 60 / 60 | 0.078925 | 5.636890 | 5.189998 | −5.628745 | 0.258450 |
| `hinge_loss` | 60 / 60 | 0.000000 | 8.127060 | 5.004600 | −8.127060 | 0.013637 |
| `bce_spectral_norm` | 60 / 60 | 0.190468 | 7.055717 | 5.451825 | −7.045843 | 0.265886 |

Las pérdidas de BCE y hinge no se comparan directamente por su distinta escala. El baseline produce personajes reconocibles y mantiene diversidad; hinge presenta colapso de modo, diversidad final de 0.0136 y 23 épocas con `loss_D < 1e-3`; la variante C mantiene la mayor diversidad final (0.2659) y media en las últimas diez épocas (0.2671), además de mayor variedad cromática en la revisión de hitos.

Por ello, `bce_spectral_norm` es la selección provisional para las fases 9–10. La selección aún debe validarse con vecinos ResNet18, MSE en píxeles y el lote de 200 candidatos; la distancia L2 del ruido fijo es un proxy y no una métrica perceptual definitiva.

## Implementación de fase 5

La generación y auditoría de la galería quedó implementada con una precondición no negociable: el checkpoint elegido debe haber alcanzado las 60 épocas registradas. Si no se cumple, `build_gallery.py` termina antes de escribir PNG en `galeria/`.

### Flujo de selección

1. Generar 200 candidatos en CPU con semilla `20261011` y guardar los vectores latentes.
2. Derivar límites técnicos de ocupación y contraste desde la distribución real del dataset.
3. Extraer embeddings de candidatos y entrenamiento con ResNet18 preentrenada sin su capa final.
4. Buscar el vecino real de máxima similitud coseno y calcular MSE en píxeles.
5. Filtrar resultados degenerados y aplicar una selección greedy de 10 imágenes que combine:
   - distancia al vecino real;
   - calidad técnica respecto al dataset;
   - distancia mínima respecto a personajes ya seleccionados.
6. Guardar `manifest.csv`, `latents.npz`, procedencia, rejilla final y figura lado a lado de vecinos.
7. Regenerar los diez PNG desde el checkpoint y exigir una diferencia máxima de cero niveles RGB.

### Estado verificable

- Prueba sintética de filtrado, vecinos y selección: aprobada.
- Métricas A/B/C: 60/60; el checkpoint seleccionado está disponible localmente y respaldado en Drive.
- Pesos ResNet18: se descargarán desde la URL oficial de PyTorch en la primera ejecución final; no se permiten pesos aleatorios como sustituto.
- Tasa de selección que se declarará: `10/200 = 5%`.
- Nombres y roles: asignados después de la selección para optar a la bonificación, sin intervenir en la métrica técnica.
