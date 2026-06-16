# CPU Baseline Report — Clasificador de Somnolencia

**Generado:** 2026-06-16 17:12:42

## Arquitectura

`Entrada(4096) -> Densa(128, ReLU) -> Densa(1, Sigmoide)`

- **Perdida:** Binary Cross-Entropy
- **Optimizador:** SGD (sin momentum)
- **Modo de entrenamiento:** full-batch (todo el train set por epoca)

## Busqueda de hiperparametros

| hidden_units | learning_rate | epochs | train_time_sec | val_loss | val_accuracy |
|---|---|---|---|---|---|
| 128 | 0.1 | 20 | 2.8705 | 0.6109 | 0.8657 |
| 64 | 0.1 | 50 | 6.5028 | 0.5275 | 0.8607 |
| 128 | 0.1 | 50 | 7.6016 | 0.5229 | 0.8507 |
| 64 | 0.1 | 20 | 3.2576 | 0.6082 | 0.8408 |
| 128 | 0.01 | 50 | 7.3419 | 0.6627 | 0.8209 |
| 64 | 0.01 | 50 | 10.5659 | 0.6763 | 0.7960 |
| 64 | 0.01 | 20 | 12.5378 | 0.6808 | 0.7313 |
| 128 | 0.01 | 20 | 2.9289 | 0.6815 | 0.5871 |

**Mejor combinacion:** hidden=128, lr=0.1, epochs=20

## Entrenamiento final (CPU)

- **Tiempo de entrenamiento:** 3.1745 s
- **Muestras de entrenamiento:** 935
- **Epocas:** 20

## Metricas en test

| Metrica | Valor |
|---------|-------|
| Accuracy | 0.8159 |
| Precision | 0.8252 |
| Recall | 0.8173 |
| F1 | 0.8213 |

## Uso

`CPU_TRAIN_TIME_SEC` es la linea base para calcular el speedup de la implementacion CUDA en `etapa2_cuda` (mismo modelo, mismos datos, mismas epocas).
