# Eryndor: Guardianes del Velo - Brief creativo

## Proyecto

**Nombre:** Eryndor: Guardianes del Velo

**Descripción:** Proyecto académico de diseño generativo para crear personajes originales de un RPG de aventura y magia. Eryndor es un mundo fragmentado por una antigua ruptura arcana: la magia se condensa en minerales luminosos, las ruinas conservan tecnologías rituales y pequeños grupos de aventureros mantienen cerradas las grietas entre regiones. La GAN deberá producir sprites completos, legibles y variados que parezcan pertenecer a ese mismo mundo.

**Audiencia:** Docente y estudiantes del curso Deep Learning 2026, con interés tanto en la calidad técnica de la GAN como en la coherencia visual de la galería.

**Objetivo:** Demostrar, con evidencia reproducible, que una GAN entrenada por el equipo puede generar al menos diez personajes nuevos, diferentes entre sí y no memorizados del conjunto de entrenamiento.

## Los cuatro ejes

### Registro de tono: Profesional con acento mítico

Las explicaciones técnicas serán precisas y sobrias; los nombres, títulos y descripciones del universo podrán usar un lenguaje evocador sin convertir el informe en ficción extensa.

### Filosofía estética: Maximalismo controlado

La densidad visual estará en los personajes -armaduras, capas, runas y materiales-, mientras que notebook, gráficas y presentación usarán composiciones limpias que permitan examinar la evidencia.

### Relación con la audiencia: Par experto

El proyecto mostrará decisiones, fallos y limitaciones con transparencia, invitando al lector a evaluar la evidencia en vez de presentar la galería como una demostración incuestionable.

### Ambición sensorial: Resonante

La estética debe comunicar aventura, misterio y antigüedad. Los personajes deben sentirse como sobrevivientes de un mundo mágico fracturado, no como piezas intercambiables de fantasía genérica.

## Síntesis

Este brief produce un RPG de fantasía melancólica con pixel art detallado de 64 x 64 píxeles. Las siluetas son claras y de cuerpo completo; los materiales -metal gastado, cuero, tela, madera y cristal mágico- importan más que la ornamentación gratuita. La base cromática usa sombras frías y tonos terrosos, con acentos luminosos que identifican la afinidad arcana de cada personaje.

La presentación combina fondos oscuros, tipografía editorial y acentos de color mineral. La evidencia técnica conserva prioridad: las rejillas, curvas y vecinos cercanos deben ser legibles antes que decorativos.

## Lenguaje visual del universo

- **Perspectiva:** sprite frontal de cuerpo completo, vista ortográfica elevada coherente con LPC.
- **Resolución:** 64 x 64 píxeles, sin suavizado ni reescalado borroso.
- **Proporción:** humanoide estilizada, con anatomía legible y materiales relativamente realistas.
- **Paleta base:** obsidiana `#171A24`, pergamino `#D6C7A1`, musgo `#4E6B55`.
- **Acentos mágicos:** cian mineral `#4FC3C8`, oro brasa `#D6A34A`, amatista `#7454A6`.
- **Rasgos:** capas, armaduras, bastones, armas, bolsas, talismanes y cristales; al menos un rasgo dominante por silueta.
- **Roles posibles:** guardián rúnico, arcanista de ceniza, explorador del musgo, alquimista solar y oráculo del abismo.
- **Atmósfera:** aventura peligrosa, magia antigua y esperanza contenida.

## Referencias

- **Liberated Pixel Cup Style Guide** - Define perspectiva, cuadrícula, iluminación y compatibilidad visual del material de entrenamiento: https://lpc.opengameart.org/static/LPC-Style-Guide/build/styleguide.html
- **Universal LPC Spritesheet Character Generator** - Fuente reproducible de capas y créditos de los sprites: https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator
- **LPC Base Assets** - Contexto de origen y licenciamiento del arte abierto: https://opengameart.org/content/liberated-pixel-cup-lpc-base-assets-sprites-map-tiles

## Lista de rechazo

Este brief dice explícitamente no a:

- Personajes de franquicias, fan art presentado como propio o rostros reales identificables.
- Imágenes finales creadas por generadores externos.
- Estética chibi excesivamente infantil, ciencia ficción moderna o neón cyberpunk.
- Fondos complejos que compitan con la silueta del personaje.
- Reescalado con interpolación suave que destruya el pixel art.
- Diez variaciones casi idénticas elegidas de un mismo modo colapsado.
- Ocultar la tasa de selección o descartar ejemplos fallidos sin documentarlos.

## Decisiones cerradas

- El modelo base será una DCGAN no condicional.
- La resolución será 64 x 64 RGB.
- Se usarán 4,096 composiciones deterministas de arte LPC con créditos conservados.
- Se comparará BCE no saturante contra hinge loss.
- La técnica de estabilización será normalización espectral en el discriminador.
- Se generarán 200 candidatos para elegir 10 personajes finales.

