# Split Report — Dataset Somnolencia

**Generado:** 2026-06-15 22:10:51
**Ultima actualizacion:** 2026-06-15 (mejoras documentales)

## Configuracion

- **Script:** `notebooks/split_dataset.py`
- **Archivo origen:** `dataset/procesado/dataset_serial.csv`
- **Formato columnas:** `label, pixel_0 .. pixel_4095`
- **Semilla (random_state):** 42
- **Split:** Train 70% / Val 15% / Test 15%
- **Estratificado por:** `label`
- **Duplicados eliminados:** 6 (antes del split)

## Reproducibilidad

El split es deterministico gracias a `random_state=42`. Sin embargo, si el
archivo `dataset_serial.csv` se regenera (por ejemplo, al reprocesar las
imagenes con el pipeline C), los indices de las filas pueden cambiar y el
split generado sera distinto.

Para reproducir el split exacto:
1. Asegurarse de que `dataset_serial.csv` no haya sido modificado
2. Ejecutar: `python notebooks/split_dataset.py`

Si `dataset_serial.csv` cambia, debe ejecutarse nuevamente `split_dataset.py`
para generar train/val/test actualizados.

## Dataset original

- **Total muestras:** 1343
- **Duplicados exactos:** 6
- **Muestras usadas para split:** 1337

## Distribucion final

| Split | Total | Clase 0 | % | Clase 1 | % |
|-------|-------|---------|---|---------|---|
| Train | 935 | 450 | 48.1% | 485 | 51.9% |
| Validation | 201 | 96 | 47.8% | 105 | 52.2% |
| Test | 201 | 97 | 48.3% | 104 | 51.7% |
| **Total** | **1337** | **643** | | **694** | |

## Validaciones

| Validacion | Resultado |
|------------|-----------|
| Valores nulos | PASS |
| Duplicados entre train/val | PASS |
| Duplicados entre train/test | PASS |
| Duplicados entre val/test | PASS |
| Suma particiones = total | PASS |

## Proporciones obtenidas

- **Train:** 69.93%
- **Validation:** 15.03%
- **Test:** 15.03%

## Nota sobre nomenclatura de columnas

Los CSVs generados usan el formato canonico del proyecto:

    label, pixel_0, pixel_1, ..., pixel_4095

Este formato coincide con:
- `dataset/procesado/dataset_serial.csv` (archivo origen)
- Pipeline C OpenMP (`preprocess_serial.c` linea 459: `fprintf(csv, ",pixel_%d", i)`)
- Documentacion oficial (`etapa1_openmp/README.md` linea 44)

## Features duplicadas con etiquetas conflictivas

Durante la auditoria previa al split se detecto un grupo de **2 filas** con
el mismo vector de pixeles (4096 valores identicos) pero etiquetas distintas.

### Detalle del caso

| Propiedad | Valor |
|-----------|-------|
| Cantidad de grupos | 1 |
| Filas involucradas | 2 (indices 527 y 1203 en `dataset_serial.csv`) |
| Etiqueta fila 527 | Clase 0 (ojos abiertos) |
| Etiqueta fila 1203 | Clase 1 (ojos cerrados) |
| Distancia coseno entre vectores | 0.0 (identicos) |

### Decision tomada

**Ambas filas se conservaron.** No son duplicados exactos porque la columna
`label` difiere, por lo que `drop_duplicates()` no las elimina. Este par
representa ruido en la recoleccion de datos (un mismo ojo etiquetado de
forma distinta por diferentes integrantes o en diferentes sesiones).

### Posible impacto en entrenamiento

- **Efecto esperado:** El modelo recibira senales contradictorias para un
  mismo patron visual, lo que puede incrementar ligeramente la perdida de
  entrenamiento y reducir la precision.
- **Magnitud:** Baja (2 filas de 1337 = 0.15% del dataset).
- **Mitigacion:** Ninguna requerida para este proyecto. Para un proyecto
  productivo se recomendaria revisar manualmente las fotos originales y
  asignar la etiqueta correcta.
