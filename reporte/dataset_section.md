# Dataset — Clasificador de Somnolencia

## Descripción general

El dataset fue creado por los integrantes del equipo mediante captura
con webcam y cámara frontal de teléfono. Cada imagen corresponde al
rostro de una persona con los ojos **abiertos** (clase 0) o **cerrados**
(clase 1), tomada en distintas condiciones de iluminación, distancia y
ángulo para introducir variabilidad realista.

## Cantidad de imágenes raw

| Clase | Descripción | Archivos |
|-------|-------------|---------:|
| 0 | Ojos abiertos | 1.656 |
| 1 | Ojos cerrados | 1.656 |
| **Total** | | **3.312** |

Formato: JPEG, nomenclatura `img_{XXXX}.jpeg` (misma numeración en
ambas clases, diferenciadas por la carpeta).

Distribución por clase perfectamente balanceada (50 % / 50 %).

## Cantidad de imágenes procesadas

El pipeline OpenMP (`etapa1_openmp/preprocess_serial.c`) convierte las
3.312 raw a vectores de 4096 features (64×64 en escala de grises,
normalizados) y los exporta como CSV.

Del CSV resultante se eliminan filas duplicadas exactas y se aplica un
split estratificado 70/15/15:

| Archivo | Muestras | Clase 0 | Clase 1 |
|---------|---------:|--------:|--------:|
| `dataset_serial.csv` | 3.312 | 1.656 | 1.656 |
| `train.csv` | 1.646 | 829 | 817 |
| `val.csv` | 353 | 178 | 175 |
| `test.csv` | 353 | 177 | 176 |
| **Total** | **2.352** | **1.184** | **1.168** |

La diferencia de 960 filas entre `dataset_serial.csv` (3.312) y la suma
de los splits (2.352) corresponde a filas duplicadas exactas que fueron
eliminadas automáticamente por el script de split.

> **Nota:** Los pesos finales del modelo (`modelo/weights.npz`) fueron
> entrenados con este dataset (2.352 muestras únicas, split
> 1.646/353/353), como se documenta en
> `dataset/procesado/split_report.md` y se verifica en el notebook
> `etapa2_cuda/gpu_model/model-gpu.ipynb` (Celda 27: test=353).

## Condiciones de captura

Las imágenes se recolectaron bajo las siguientes condiciones:

- **Dispositivos:** webcam integrada de laptop, cámara frontal de
  teléfono móvil, cámara trasera de teléfono móvil.
- **Iluminación:** luz natural (ventana), luz fluorescente de techo,
  lámpara de escritorio, condiciones mixtas.
- **Distancia:** ~20–50 cm del dispositivo (primer plano del rostro).
- **Ángulo:** preferentemente frontal, con ligeras variaciones de
  inclinación y rotación.
- **Fondo:** variable (habitaciones, oficina, espacios abiertos).
- **Accesorios:** con y sin gafas, diferentes peinados, maquillaje
  ocasional.

## Resoluciones observadas

Las imágenes raw presentan una amplia variedad de resoluciones debido
al uso de múltiples dispositivos. En una muestra de 100 imágenes se
encontraron 38 resoluciones distintas; las más frecuentes fueron:

| Resolución | Frecuencia |
|-----------:|-----------:|
| 360×480 | 28 % |
| 4032×3024 | 11 % |
| 3024×4032 | 8 % |
| 83×83 | 4 % |
| 82×82 | 4 % |
| 768×1024 | 4 % |
| Otras (32 resoluciones) | 41 % |

El pipeline de preprocesamiento estandariza todas las imágenes a
64×64 píxeles independientemente de la resolución original.

## Tabla resumen

| Métrica | Valor |
|---------|------:|
| Total imágenes raw | 3.312 |
| Clases | 2 (balanceadas) |
| Imágenes raw por clase | 1.656 |
| Imágenes procesadas (CSV) | 3.312 |
| Imágenes entrenamiento | 1.646 |
| Imágenes validación | 353 |
| Imágenes prueba | 353 |
| Features por muestra | 4.096 |
| Formato de salida | CSV (label + 4096 columnas) |
| Resolución de preprocesamiento | 64×64 píxeles |
| Archivos raw faltantes | 0 |

## Rejilla de muestras

A continuación se referencian ejemplos representativos de cada clase
extraídos directamente del repositorio.

### Clase 0 — Ojos abiertos (alertas)

| Muestra | Archivo |
|---------|---------|
| Muestra 1 | `dataset/raw/clase_0/img_0001.jpeg` |
| Muestra 2 | `dataset/raw/clase_0/img_0250.jpeg` |
| Muestra 3 | `dataset/raw/clase_0/img_0500.jpeg` |
| Muestra 4 | `dataset/raw/clase_0/img_0750.jpeg` |
| Muestra 5 | `dataset/raw/clase_0/img_1000.jpeg` |

### Clase 1 — Ojos cerrados (somnolencia)

| Muestra | Archivo |
|---------|---------|
| Muestra 1 | `dataset/raw/clase_1/img_0001.jpeg` |
| Muestra 2 | `dataset/raw/clase_1/img_0250.jpeg` |
| Muestra 3 | `dataset/raw/clase_1/img_0500.jpeg` |
| Muestra 4 | `dataset/raw/clase_1/img_0750.jpeg` |
| Muestra 5 | `dataset/raw/clase_1/img_1000.jpeg` |

La nomenclatura `img_{XXXX}.jpeg` es secuencial independiente en cada
carpeta; el mismo número en ambas carpetas **no** implica que sean la
misma persona o la misma sesión de captura.
