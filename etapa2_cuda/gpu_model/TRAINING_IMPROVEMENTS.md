# Mejoras de entrenamiento (GPU) — guía de reentrenamiento

Este documento explica las mejoras agregadas a `train_gpu.cu` y a
`notebooks/augment_dataset.py`, por qué se eligieron, qué se descartó a propósito,
y el checklist exacto para reentrenar en Colab y reemplazar `modelo/weights.npz`
de forma segura (el proyecto **ya está desplegado**; nada de esto se aplica solo
con editar este repo — hay que correrlo en Colab y decidir si los pesos nuevos
mejoran antes de redesplegar).

## Por qué estas mejoras y no otras

El dataset es pequeño (935 filas de train en el subconjunto representativo de
este repo, 1646 en el split completo de Colab) y el modelo es una MLP de una sola
capa oculta sin regularización. En ese régimen, las palancas con mejor relación
costo/beneficio son **más datos** (augmentation) y **mejor optimización del
mismo modelo** (momentum, weight decay, mejor inicialización, quedarse con la
mejor época) — no una arquitectura más compleja, que con tan pocos datos
probablemente sobreajustaría peor.

## Cambios en `train_gpu.cu`

Todos son **aditivos y opt-in**: con cero flags, el binario compila y entrena
exactamente como antes (mismos kernels, mismo resultado salvo un posible
redondeo de último bit explicado abajo). La invocación de 7 argumentos
posicionales documentada originalmente sigue funcionando sin cambios.

| Flag | Default | Qué hace | Por qué |
|---|---|---|---|
| `--momentum=M` | `0.0` | SGD con momentum (formula de PyTorch: `v = momentum*v + grad; param -= lr*v`) | Acelera la convergencia y suaviza el ruido del full-batch en un dataset chico |
| `--l2=LAMBDA` | `0.0` | Weight decay L2 en la actualización | Única regularización del modelo; ayuda con tan pocas muestras |
| `--init=he\|glorot` | `glorot` | Inicialización de W1 (la capa que alimenta la ReLU) | He es la inicialización recomendada para ReLU; Glorot (la actual) está pensada para sigmoide/tanh |
| `--save-best` | off | Guarda los pesos de la época con menor `val_loss`, no los de la última época | Con pocas épocas y full-batch, el `val_loss` no es monótono; quedarse con la última época puede ser peor que una intermedia |
| `--lr-decay=G` | `1.0` | Multiplica la tasa de aprendizaje por `G` cada época | Permite empezar con una tasa más alta y afinar al final, sin tener que adivinar una tasa fija |

**Por qué no hay `--batch_size=` (mini-batching real):** se evaluó y se descartó
a propósito. Implementarlo bien (shuffle por época, kernel de *gather* para
formar el batch desde índices permutados) es un cambio de mayor superficie en
el bucle de entrenamiento, y no hay GPU local para compilar ni probar este
archivo — el riesgo de entregar CUDA roto para un entregable calificado superó
el beneficio. Las mejoras elegidas (momentum, L2, init, save-best, lr-decay) se
razonaron como matemáticamente seguras sin necesitar ejecutarlas.

**Nota de precisión numérica:** con los defaults (`momentum=0`, `l2=0`), los
términos nuevos se anulan multiplicando por cero exacto (IEEE754: `x*0.0=0.0` y
`x+0.0=x` sin redondeo), así que el valor es matemáticamente el mismo SGD de
antes. El *orden* de las multiplicaciones cambia ligeramente (antes
`(lr*grad)*inv_n`, ahora `lr*(grad*inv_n)`, porque el momentum estándar exige
aplicar `lr` al final y no antes de acumular en `velocity`), lo que puede diferir
en el último bit de redondeo — igual de insignificante que el no-determinismo
que ya introduce el `atomicAdd` de `bce_loss_kernel` entre corridas. No cambia
ninguna métrica visible (impresas con 6 decimales).

## `notebooks/augment_dataset.py` — data augmentation

Aumenta **solo el CSV de train** (nunca val/test — eso sería fuga de datos)
generando copias con flip horizontal, jitter de brillo/contraste y pequeños
desplazamientos/rotaciones, siempre sobre el vector 64×64 ya preprocesado (no
sobre la imagen raw, para no duplicar el pipeline OpenMP/Sobel). Incluye chequeo
de solape exacto (hash SHA1) contra val/test para garantizar que no hay fuga.

Verificado localmente sobre el subconjunto representativo del repo: crece el
número de filas, preserva el balance de clases, y reporta cero solape con
`val.csv`/`test.csv`.

## Checklist de reentrenamiento en Colab

1. **Augmentar train** (en el repo clonado dentro de Colab, sobre el split
   completo):
   ```bash
   python3 notebooks/augment_dataset.py \
       --input dataset/procesado/train.csv \
       --output dataset/procesado/train_augmented.csv \
       --factor 2 \
       --check-leakage dataset/procesado/val.csv dataset/procesado/test.csv
   ```
   Debe imprimir `OK: sin solape` para ambos archivos antes de continuar.

2. **Compilar** `train_gpu.cu` (ya lo hace `model-gpu.ipynb`, sección 2):
   ```bash
   nvcc -O3 -o train_gpu train_gpu.cu
   ```

3. **Entrenar con las mejoras**, usando `train_augmented.csv` y los
   hiperparámetros base de `gpu_report.md` (hidden=75, lr=0.1, epochs=50,
   block_size=32) como punto de partida:
   ```bash
   ./train_gpu dataset/procesado/train_augmented.csv dataset/procesado/val.csv \
       75 0.1 50 32 weights_gpu_v2.bin \
       --momentum=0.9 --l2=0.0001 --init=he --save-best
   ```
   Desde el notebook, en vez de invocar el binario a mano, se puede usar el
   nuevo parámetro opcional `extra_flags` de `run_training()` (celda 4) sin
   tocar ninguna celda existente:
   ```python
   result = run_training(75, 0.1, 50, 32, "weights_gpu_v2.bin",
                          extra_flags=["--momentum=0.9", "--l2=0.0001", "--init=he", "--save-best"])
   ```

4. **Comparar contra el baseline** (`gpu_report.md`: test accuracy=0.8527,
   precision=0.8875, recall=0.8068, F1=0.8452). Solo seguir si el nuevo
   `weights_gpu_v2.bin` iguala o mejora estas métricas en test — si empeora,
   ajustar hiperparámetros (probar `--momentum=0.0` o `--l2=` más chico) antes
   de exportar.

5. **Exportar a `.npz`** con el mismo código de la sección 10 del notebook
   (`load_gpu_weights` + `np.savez(..., W1=W1, b1=b1, W2=W2, b2=b2)`).

6. **Reajustar el umbral de decisión** con los pesos nuevos y el split
   completo:
   ```bash
   python3 modelo/tune_threshold.py \
       --weights ruta/al/weights_gpu_v2.npz \
       --val dataset/procesado/val.csv --test dataset/procesado/test.csv
   ```
   Si sugiere un umbral distinto de 0.5, actualizar
   `DECISION_THRESHOLD` en `app_streamlit/core/inference.py`.

7. **Reemplazar pesos solo si mejoran**: copiar el `.npz` nuevo a
   `modelo/weights.npz` (sobrescribe el actual) únicamente si el paso 4 confirmó
   una mejora real en test.

8. **Redesplegar**: hacer `git commit` + `git push` a `main` solo cuando se esté
   conforme con el resultado — el push dispara el redeploy de la app con los
   pesos horneados en la imagen Docker.
