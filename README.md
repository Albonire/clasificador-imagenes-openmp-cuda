# Clasificador de Imágenes desde Cero
## Preprocesamiento con OpenMP + Entrenamiento con CUDA

**Programación Paralela y Computación Distribuida — Universidad de Pamplona**

---

## Estructura del repositorio

```
├── etapa1_openmp/      # Pipeline de preprocesamiento en C con OpenMP
├── etapa2_cuda/        # Kernels CUDA para entrenamiento + versión CPU baseline
│   └── cpu_baseline/
├── modelo/             # Pesos del modelo entrenado
├── app_streamlit/      # Aplicación de inferencia con Streamlit
├── notebooks/          # Notebooks y scripts de entrenamiento / gráficas
├── dataset/            # Muestra representativa del dataset
│   ├── raw/
│   └── procesado/
└── reporte/            # Reporte final (Markdown o PDF)
```

## Pipeline

1. **Recolección de datos** — imágenes propias con webcam, dos clases balanceadas
2. **Etapa 1 (OpenMP)** — preprocesamiento paralelo en CPU, exporta matriz 64×64
3. **Etapa 2 (CUDA)** — entrenamiento MLP en GPU con kernels propios
4. **App Streamlit** — inferencia en tiempo real sobre imágenes nuevas

## Equipo

| Integrante | Hilo conductor | Responsabilidad principal |
|---|---|---|
| Raúl | Los datos entran limpios o no entran | Pipeline OpenMP completo en C |
| Jeferson | La GPU entrena lo que la CPU procesó | Forward pass CUDA + CPU baseline |
| Fabián | El modelo aprende y los pesos viajan a la app | Backward pass CUDA + exportar pesos |
| Silvana | Los datos se documentan y el modelo se usa | Dataset + App Streamlit |
| Valentina | La historia del proyecto se cuenta bien | Reporte CRISP-DM + métricas |
| Rubén | Todo llega junto y a tiempo | Notebook + integración + conclusiones |

## Cómo correr el proyecto

### 1. Preprocesamiento (OpenMP)

```bash
cd etapa1_openmp
gcc -Wall -Wextra -O2 -fopenmp -o preprocess_serial preprocess_serial.c -lm
./preprocess_serial ../dataset/raw ../dataset/procesado/dataset_serial.csv
```

Detalles y mediciones de speedup en `etapa1_openmp/README.md`.

### 2. Entrenamiento (CUDA, requiere GPU — usado vía Google Colab)

```bash
cd etapa2_cuda/gpu_model
nvcc -O3 -o train_gpu train_gpu.cu
./train_gpu ../../dataset/procesado/train.csv ../../dataset/procesado/val.csv \
    75 0.1 50 32 weights_gpu.bin
```

El notebook `model-gpu.ipynb` orquesta la búsqueda de hiperparámetros y mide
CPU vs GPU; `cpu_baseline/` contiene la línea base sin GPU. Los pesos finales
se exportan a `modelo/weights.npz` con:

```bash
python3 modelo/export_weights.py
```

Ver `etapa2_cuda/gpu_model/backprop_notes.md` para la derivación matemática
del backward pass.

### 3. App de inferencia (Streamlit)

```bash
cd app_streamlit
pip install -r requirements.txt
streamlit run app.py
```

Usa `modelo/weights.npz` y debe aplicar el **mismo preprocesamiento** que
`etapa1_openmp/` (gris → Sobel → Gaussiano → 64×64 → normalizado → vector 4096).

### 4. Reporte

Documentación CRISP-DM completa (Fases 1 a 6) en `reporte/`, con evidencias en
`reporte/evidencias/`.
