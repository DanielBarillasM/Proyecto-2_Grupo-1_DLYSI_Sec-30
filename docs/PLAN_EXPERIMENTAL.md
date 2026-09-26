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
- **Interrupción de Colab:** checkpoints cada cinco épocas y reanudación completa del optimizador.
- **Licencias mixtas:** conservar créditos por capa y no presentar el material como propietario.

