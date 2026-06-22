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

## Pipeline OpenMP

`preprocess_serial.c` implementa el pipeline completo en C. El bucle principal que
recorre las imágenes está paralelizado con OpenMP; para la línea base serial se
ejecuta con `OMP_NUM_THREADS=1`.

### Compilar

Desde `etapa1_openmp/`:

```bash
gcc -Wall -Wextra -O2 -fopenmp -o preprocess_serial preprocess_serial.c -lm
```

### Ejecutar

Con rutas por defecto:

```bash
./preprocess_serial
```

Con rutas explícitas:

```bash
./preprocess_serial ../dataset/raw ../dataset/procesado/dataset_serial.csv
```

En Windows PowerShell se puede fijar el número de hilos así:

```powershell
$env:OMP_NUM_THREADS="4"; .\preprocess_serial.exe ..\dataset\raw ..\dataset\procesado\dataset_openmp.csv
```

La salida queda en `dataset/procesado/dataset_serial.csv` con este formato:

```text
label,pixel_0,pixel_1,...,pixel_4095
```

Etiquetas usadas:
- `clase_0` -> `0` (ojos abiertos)
- `clase_1` -> `1` (ojos cerrados)

## Mediciones
- Tiempo en serie (1 hilo) — línea base
- Tiempo con OpenMP: 2, 4, 8... hilos
- Speedup = tiempo_serial / tiempo_paralelo

Resultados medidos con `3312` imágenes:

| Hilos | Tiempo (s) | Speedup |
|------:|-----------:|--------:|
| 1 | 27.262 | 1.00 |
| 2 | 17.325 | 1.57 |
| 4 | 14.155 | 1.93 |
| 8 | 13.449 | 2.03 |
