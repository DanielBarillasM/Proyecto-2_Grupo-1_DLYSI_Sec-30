# Presentaciones HTML y PDF

La versión principal para exponer es `presentacion.html`: funciona sin servidor, admite teclado, pantalla completa, vista general y actualización automática de resultados. Conserva exactamente 12 escenas y una salida de impresión por escena.

El archivo `presentacion.pdf` se mantiene porque es el entregable exigido por el enunciado. Ambas versiones usan la dirección visual mineral de Eryndor, el tema **Midnight Galaxy** y la tipografía GNU FreeSans incluida en `fonts/`.

Para regenerar y validar el HTML:

```powershell
python scripts\build_html_presentation.py
start presentation\presentacion.html
```

Controles: flechas o espacio para avanzar, `O` para vista general, `F` para pantalla completa, `Home`/`End` para los extremos. Desde el navegador se puede imprimir a PDF con una escena por página.

Para actualizar métricas, compilar y validar el número de páginas:

```powershell
python scripts\build_presentation.py
```

`presentation-data.js` y `generated_results.tex` se generan desde los CSV/JSON del proyecto. Si la galería todavía no fue validada, ambas presentaciones muestran un estado pendiente en lugar de resultados fabricados.

Salidas: `presentation/presentacion.html` y `presentation/presentacion.pdf`.

## Tipografía

GNU FreeFont fue obtenida de <https://www.gnu.org/software/freefont/>. Los archivos de licencia y documentación se conservan junto con las fuentes.
