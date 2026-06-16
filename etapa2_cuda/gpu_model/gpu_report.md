# GPU Report (CUDA) — Clasificador de Somnolencia

**Generado:** 2026-06-16 17:26:24

**GPU:** Tesla T4

## Arquitectura

`Entrada(4096) -> Densa(64, ReLU) -> Densa(1, Sigmoide)`

- **Perdida:** Binary Cross-Entropy
- **Optimizador:** SGD escrito a mano (sin momentum)
- **Modo de entrenamiento:** full-batch (todo el train set por epoca)
- **Kernels:** `matmul_ab_tiled`, `matmul_atb`, `bias_relu_forward`, `bias_sigmoid_forward`, `bce_loss_kernel`, `output_gradient_kernel`, `hidden_gradient_kernel`, `bias_gradient_kernel`, `sgd_update_kernel` (ver `train_gpu.cu`)

## Busqueda de hiperparametros (GPU, block_size=32)

| hidden_units | learning_rate | epochs | train_loop_seconds | val_loss | val_accuracy |
|---|---|---|---|---|---|
| 64 | 0.1 | 50 | 0.205293 | 0.5034 | 0.8657 |
| 64 | 0.1 | 20 | 0.083849 | 0.6008 | 0.8657 |
| 128 | 0.1 | 50 | 0.314286 | 0.5219 | 0.8657 |
| 128 | 0.1 | 20 | 0.067514 | 0.6102 | 0.8358 |
| 64 | 0.01 | 50 | 0.216839 | 0.6671 | 0.7811 |
| 128 | 0.01 | 50 | 0.287475 | 0.6652 | 0.7711 |
| 128 | 0.01 | 20 | 0.085280 | 0.6792 | 0.7164 |
| 64 | 0.01 | 20 | 0.092096 | 0.6808 | 0.5920 |

**Mejor combinacion:** hidden=64, lr=0.1, epochs=50

## Entrenamiento final (GPU)

- **Transferencia host->device:** 0.004523 s
- **Tiempo de entrenamiento (bucle de epocas):** 0.180239 s
- **Muestras de entrenamiento:** leidas de `train.csv`
- **Epocas:** 50
- **block_size:** 32

## Efecto del tamaño de bloque

| block_size | train_loop_seconds | val_accuracy |
|---|---|---|
| 16 | 0.186842 | 0.8657 |
| 32 | 0.180239 | 0.8657 |

## Monitoreo nvidia-smi

- **Muestras capturadas:** 36 (cada 0.05s, corrida con `epochs=300` solo para esta medicion)
- **Utilizacion GPU:** max=98.0%, promedio=24.0%
- **Memoria GPU:** pico=129 MB de 15360 MB

## Metricas en test

| Metrica | Valor |
|---------|-------|
| Accuracy | 0.8458 |
| Precision | 0.8687 |
| Recall | 0.8269 |
| F1 | 0.8473 |

## Speedup GPU vs. CPU

No se encontro `cpu_baseline_report.md` con un tiempo de entrenamiento para calcular el speedup. Ejecuta primero `base-model-cpu.ipynb`.

## Uso

Pesos finales en `weights_gpu.bin` (binario, leido por este notebook) y `weights_gpu.npz` (formato `app_streamlit`). Para usar este modelo como el modelo final del proyecto, copiar `weights_gpu.npz` a `modelo/weights.npz`.
