# Split Report — Dataset Somnolencia

**Generado:** 2026-06-16 20:23:09

## Configuracion

- **Script:** `notebooks/split_dataset.py`
- **Archivo origen:** `dataset/procesado/dataset_serial.csv`
- **Formato columnas:** `label, pixel_0 .. pixel_4095`
- **Semilla (random_state):** 42
- **Split:** Train 70% / Val 15% / Test 15%
- **Estratificado por:** `label`
- **Duplicados eliminados:** 960 (antes del split)

## Dataset original

- **Total muestras:** 3312
- **Duplicados exactos:** 960
- **Muestras usadas para split:** 2352

## Distribucion final

| Split | Total | Clase 0 | % | Clase 1 | % |
|-------|-------|---------|---|---------|---|
| Train | 1646 | 829 | 50.4% | 817 | 49.6% |
| Validation | 353 | 178 | 50.4% | 175 | 49.6% |
| Test | 353 | 177 | 50.1% | 176 | 49.9% |
| **Total** | **2352** | **1184** | | **1168** | |

## Validaciones

| Validacion | Resultado |
|------------|-----------|
| Valores nulos | PASS |
| Duplicados entre train/val | PASS |
| Duplicados entre train/test | PASS |
| Duplicados entre val/test | PASS |
| Suma particiones = total | PASS |

## Proporciones obtenidas

- **Train:** 69.98%
- **Validation:** 15.01%
- **Test:** 15.01%

## Nota sobre nomenclatura de columnas

Los CSVs generados usan el formato canonico del proyecto:

    label, pixel_0, pixel_1, ..., pixel_4095

Este formato coincide con:
- `dataset/procesado/dataset_serial.csv` (archivo origen)
- Pipeline C OpenMP (`preprocess_serial.c` linea 459: `fprintf(csv, ",pixel_%d", i)`)
- Documentacion oficial (`etapa1_openmp/README.md` linea 44)

