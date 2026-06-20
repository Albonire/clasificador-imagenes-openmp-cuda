# Metricas finales del modelo

Esta seccion documenta como reproducir la evaluacion final solicitada para el
test set: exactitud, precision, recall, F1 y matriz de confusion.

## Comando

```bash
python scripts/evaluate_final_metrics.py
```

Entradas por defecto:

- `modelo/weights.npz`
- `dataset/procesado/test.csv`

Salidas por defecto:

- `reporte/evidencias/final_metrics.md`
- `reporte/evidencias/final_metrics/metrics.json`
- `reporte/evidencias/final_metrics/confusion_matrix.csv`
- `reporte/evidencias/final_metrics/confusion_matrix.svg`
- `reporte/evidencias/final_metrics/predictions.csv`

## Curvas de perdida

Las curvas de perdida ya generadas para el entrenamiento estan en:

- `reporte/evidencias/gpu_training_curves.png`
- `reporte/evidencias/cpu_baseline_training_curves.png`

El resumen con los resultados calculados queda en
`reporte/evidencias/final_metrics.md`.

## Mensajes de commit sugeridos

- `feat: add final metrics evaluation script`
- `docs: add confusion matrix and loss curves`
