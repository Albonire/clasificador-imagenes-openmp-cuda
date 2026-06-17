# GPU Report (CUDA) — Clasificador de Somnolencia

**Generado:** 2026-06-17 02:08:18

**GPU:** Tesla T4

## Arquitectura

`Entrada(4096) -> Densa(75, ReLU) -> Densa(1, Sigmoide)`

- **Perdida:** Binary Cross-Entropy
- **Optimizador:** SGD escrito a mano (sin momentum)
- **Modo de entrenamiento:** full-batch (todo el train set por epoca)
- **Kernels:** `matmul_ab_tiled`, `matmul_atb`, `bias_relu_forward`, `bias_sigmoid_forward`, `bce_loss_kernel`, `output_gradient_kernel`, `hidden_gradient_kernel`, `bias_gradient_kernel`, `sgd_update_kernel` (ver `train_gpu.cu`)

## Busqueda de hiperparametros (GPU, block_size=32)

| hidden_units | learning_rate | epochs | train_loop_seconds | val_loss | val_accuracy |
|---|---|---|---|---|---|
| 75 | 0.1 | 50 | 0.337484 | 0.5402 | 0.8272 |
| 75 | 0.1 | 45 | 0.271651 | 0.5511 | 0.8272 |
| 64 | 0.08 | 50 | 0.255609 | 0.5517 | 0.8215 |
| 64 | 0.1 | 50 | 0.247428 | 0.5305 | 0.8215 |
| 75 | 0.08 | 50 | 0.356287 | 0.5627 | 0.8215 |
| 75 | 0.08 | 40 | 0.308447 | 0.5833 | 0.8215 |
| 64 | 0.1 | 45 | 0.251811 | 0.5407 | 0.8215 |
| 64 | 0.1 | 40 | 0.253272 | 0.5518 | 0.8215 |
| 75 | 0.1 | 40 | 0.276603 | 0.5628 | 0.8215 |
| 75 | 0.08 | 45 | 0.263878 | 0.5727 | 0.8187 |
| 78 | 0.1 | 20 | 0.160256 | 0.6189 | 0.8159 |
| 75 | 0.08 | 20 | 0.174942 | 0.6318 | 0.8159 |
| 64 | 0.1 | 20 | 0.115098 | 0.6076 | 0.8159 |
| 64 | 0.08 | 20 | 0.120870 | 0.6215 | 0.8159 |
| 64 | 0.08 | 45 | 0.301421 | 0.5613 | 0.8159 |
| 64 | 0.08 | 40 | 0.240057 | 0.5715 | 0.8130 |
| 78 | 0.08 | 20 | 0.148624 | 0.6318 | 0.8130 |
| 75 | 0.1 | 20 | 0.148830 | 0.6187 | 0.8102 |
| 128 | 0.1 | 50 | 0.443602 | 0.5401 | 0.8045 |
| 78 | 0.08 | 40 | 0.282204 | 0.5839 | 0.8017 |
| 78 | 0.1 | 45 | 0.270487 | 0.5525 | 0.8017 |
| 78 | 0.1 | 40 | 0.312281 | 0.5639 | 0.7989 |
| 78 | 0.08 | 50 | 0.361673 | 0.5638 | 0.7989 |
| 78 | 0.08 | 45 | 0.280314 | 0.5735 | 0.7989 |
| 78 | 0.1 | 50 | 0.350653 | 0.5420 | 0.7989 |
| 128 | 0.08 | 20 | 0.217179 | 0.6273 | 0.7932 |
| 128 | 0.1 | 45 | 0.411997 | 0.5506 | 0.7932 |
| 128 | 0.08 | 40 | 0.385275 | 0.5812 | 0.7875 |
| 128 | 0.08 | 50 | 0.430729 | 0.5617 | 0.7875 |
| 128 | 0.08 | 45 | 0.431312 | 0.5712 | 0.7875 |
| 128 | 0.1 | 40 | 0.344453 | 0.5617 | 0.7875 |
| 128 | 0.1 | 20 | 0.230817 | 0.6149 | 0.7847 |
| 64 | 0.01 | 50 | 0.248184 | 0.6664 | 0.7734 |
| 64 | 0.01 | 45 | 0.244675 | 0.6687 | 0.7677 |
| 64 | 0.01 | 40 | 0.255082 | 0.6710 | 0.7592 |
| 78 | 0.01 | 50 | 0.345575 | 0.6727 | 0.7535 |
| 128 | 0.01 | 50 | 0.454435 | 0.6657 | 0.7507 |
| 78 | 0.01 | 45 | 0.289927 | 0.6748 | 0.7422 |
| 128 | 0.01 | 45 | 0.422312 | 0.6676 | 0.7422 |
| 78 | 0.01 | 40 | 0.311332 | 0.6770 | 0.7309 |
| 128 | 0.01 | 40 | 0.398627 | 0.6696 | 0.7280 |
| 75 | 0.01 | 50 | 0.353563 | 0.6716 | 0.7195 |
| 75 | 0.01 | 45 | 0.282738 | 0.6736 | 0.6997 |
| 75 | 0.01 | 40 | 0.302671 | 0.6757 | 0.6827 |
| 64 | 0.01 | 20 | 0.144810 | 0.6811 | 0.6686 |
| 78 | 0.01 | 20 | 0.159343 | 0.6859 | 0.6516 |
| 128 | 0.01 | 20 | 0.232127 | 0.6779 | 0.6431 |
| 75 | 0.01 | 20 | 0.177093 | 0.6845 | 0.6006 |

**Mejor combinacion:** hidden=75, lr=0.1, epochs=50

## Entrenamiento final (GPU)

- **Transferencia host->device:** 0.007720 s
- **Tiempo de entrenamiento (bucle de epocas):** 0.304062 s
- **Muestras de entrenamiento:** leidas de `train.csv`
- **Epocas:** 50
- **block_size:** 32

## Efecto del tamaño de bloque

| block_size | train_loop_seconds | val_accuracy |
|---|---|---|
| 16 | 0.290634 | 0.8272 |
| 32 | 0.304062 | 0.8272 |

## Monitoreo nvidia-smi

- **Muestras capturadas:** 58 (cada 0.05s, corrida con `epochs=300` solo para esta medicion)
- **Utilizacion GPU:** max=99.0%, promedio=25.3%
- **Memoria GPU:** pico=143 MB de 15360 MB

## Metricas en test

| Metrica | Valor |
|---------|-------|
| Accuracy | 0.8527 |
| Precision | 0.8875 |
| Recall | 0.8068 |
| F1 | 0.8452 |

## Speedup GPU vs. CPU

- **CPU (`cpu_baseline_report.md`):** 3.1745 s
- **GPU (este reporte):** 0.3041 s
- **Speedup:** 10.44x

## Uso

Pesos finales en `weights_gpu.bin` (binario, leido por este notebook) y `weights_gpu.npz` (formato `app_streamlit`). Para usar este modelo como el modelo final del proyecto, copiar `weights_gpu.npz` a `modelo/weights.npz`.
