# Etapa 1 — Preprocesamiento con OpenMP

Código fuente en C del pipeline de preprocesamiento paralelizado con OpenMP.

## Pasos del pipeline
1. Lectura de imágenes
2. Conversión a escala de grises
3. Filtros (Sobel / Gaussiano)
4. Redimensión a 64×64
5. Normalización
6. Aplanado a vector de 4096
7. Exportación del dataset (CSV o binario)

## Mediciones
- Tiempo en serie (1 hilo) — línea base
- Tiempo con OpenMP: 2, 4, 8... hilos
- Speedup = tiempo_serial / tiempo_paralelo
