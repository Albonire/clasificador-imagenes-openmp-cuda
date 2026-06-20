# Metricas finales sobre test set

Evaluacion del modelo final `modelo/weights.npz` sobre `dataset/procesado/test.csv`.

## Resumen

| Metrica | Valor |
|---|---:|
| Muestras | 201 |
| Loss BCE | 0.5636 |
| Exactitud | 0.8259 |
| Precision | 0.8224 |
| Recall | 0.8462 |
| F1 | 0.8341 |

## Matriz de confusion

| Real \ Prediccion | Clase 0 | Clase 1 |
|---|---:|---:|
| Clase 0 | 78 | 19 |
| Clase 1 | 16 | 88 |

![Matriz de confusion](final_metrics/confusion_matrix.svg)

## Curvas de perdida

Las curvas de perdida generadas durante el entrenamiento estan documentadas en:

- `reporte/evidencias/gpu_training_curves.png`
- `reporte/evidencias/cpu_baseline_training_curves.png`

## Archivos generados

- `reporte/evidencias/final_metrics/metrics.json`
- `reporte/evidencias/final_metrics/confusion_matrix.csv`
- `reporte/evidencias/final_metrics/confusion_matrix.svg`
- `reporte/evidencias/final_metrics/predictions.csv`
