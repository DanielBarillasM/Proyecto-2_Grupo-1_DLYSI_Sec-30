# Presentación

La presentación funciona como informe del Proyecto 2 y contiene exactamente 12 diapositivas. Usa el tema **Midnight Galaxy** de Theme Factory y la tipografía GNU FreeSans incluida en `fonts/`.

Para actualizar métricas, compilar y validar el número de páginas:

```powershell
python scripts\build_presentation.py
```

El archivo `generated_results.tex` se genera desde los CSV/JSON del proyecto. Si la galería todavía no fue validada, las diapositivas correspondientes muestran un estado pendiente en lugar de resultados fabricados.

Salida final: `presentation/presentacion.pdf`.

## Tipografía

GNU FreeFont fue obtenida de <https://www.gnu.org/software/freefont/>. Los archivos de licencia y documentación se conservan junto con las fuentes.
