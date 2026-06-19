# Modelo entrenado

`weights.npz` — pesos finales del modelo usado por `app_streamlit/app.py`.
Generados a partir del entrenamiento CUDA en `etapa2_cuda/gpu_model/` (ver
`gpu_report.md` para hiperparámetros y métricas).

## Contrato de `weights.npz`

| Clave | Forma | Descripción |
|---|---|---|
| `W1` | `(hidden_units, 4096)` | Pesos capa oculta |
| `b1` | `(hidden_units,)` | Bias capa oculta |
| `W2` | `(1, hidden_units)` | Pesos capa de salida |
| `b2` | `(1,)` | Bias capa de salida |

Forward de inferencia esperado (debe coincidir con el preprocesamiento de
`etapa1_openmp/`: gris → Sobel → Gaussiano → 64×64 → normalizado → vector 4096):

```python
z1 = x @ W1.T + b1
a1 = np.maximum(z1, 0)          # ReLU
z2 = a1 @ W2.T + b2
y_hat = 1 / (1 + np.exp(-z2))   # Sigmoide
```

## Regenerar `weights.npz`

```bash
python3 modelo/export_weights.py \
    --input etapa2_cuda/gpu_model/weights_gpu.bin \
    --output modelo/weights.npz
```
