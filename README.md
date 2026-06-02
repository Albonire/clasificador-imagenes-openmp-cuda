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

| Rol | Responsabilidad |
|-----|-----------------|
| Kernels | C/OpenMP + kernels CUDA |
| Modelo y datos | Dataset, red, entrenamiento, app |
| Benchmark y reporte | Tiempos, métricas, gráficas, redacción |
