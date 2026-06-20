# CRISP-DM Fase 5: Evaluacion

## Objetivo

El objetivo de esta fase es evaluar el modelo final de clasificacion de somnolencia y determinar si los resultados obtenidos son coherentes con el proposito del proyecto: identificar el estado de los ojos a partir de imagenes procesadas para apoyar un sistema basico de deteccion de somnolencia.

La evaluacion considera las metricas finales sobre el conjunto de prueba, la matriz de confusion, el comportamiento observado durante el entrenamiento y las reflexiones obtenidas en las dos etapas principales del proyecto:

- Etapa 1: preprocesamiento de imagenes con OpenMP.
- Etapa 2: entrenamiento y aceleracion del modelo con CUDA.

---

## Conjunto de Evaluacion

La evaluacion final se realizo sobre el subconjunto independiente de prueba:

- Archivo: `dataset/procesado/test.csv`
- Numero de muestras: `201`
- Formato de entrada: 4096 caracteristicas normalizadas por imagen.
- Modelo evaluado: `modelo/weights.npz`
- Umbral de decision: `0.5`

El conjunto de prueba no fue usado durante el entrenamiento del modelo. Por esta razon, permite estimar de manera mas objetiva el comportamiento del clasificador frente a datos no vistos.

---

## Metricas Finales en Test

El modelo final fue evaluado con las siguientes metricas de clasificacion:

| Metrica | Valor |
|---|---:|
| Perdida Binary Cross-Entropy | 0.5636 |
| Exactitud | 0.8259 |
| Precision | 0.8224 |
| Recall | 0.8462 |
| F1-score | 0.8341 |

### Interpretacion de las Metricas

**Exactitud:** el modelo clasifico correctamente aproximadamente el 82.59% de las muestras del conjunto de prueba. Este resultado indica que el modelo aprendio patrones visuales utiles a partir de las imagenes de ojos procesadas.

**Precision:** cuando el modelo predijo la clase positiva, acerto aproximadamente el 82.24% de las veces. Esta metrica es importante porque permite medir que tan confiables son las alertas positivas generadas por el sistema.

**Recall:** el modelo detecto aproximadamente el 84.62% de las muestras reales de la clase positiva. En un sistema de deteccion de somnolencia esta metrica es especialmente relevante, ya que no detectar un caso de somnolencia puede representar un riesgo.

**F1-score:** combina precision y recall en una sola medida. El valor de 0.8341 muestra un comportamiento equilibrado entre detectar correctamente la clase positiva y evitar un exceso de falsos positivos.

---

## Matriz de Confusion

La matriz de confusion obtenida sobre el conjunto de prueba fue:

| Real \ Prediccion | Clase 0 | Clase 1 |
|---|---:|---:|
| Clase 0 | 78 | 19 |
| Clase 1 | 16 | 88 |

Donde:

- **Verdaderos negativos (TN):** 78 muestras de la Clase 0 fueron clasificadas correctamente como Clase 0.
- **Falsos positivos (FP):** 19 muestras de la Clase 0 fueron clasificadas incorrectamente como Clase 1.
- **Falsos negativos (FN):** 16 muestras de la Clase 1 fueron clasificadas incorrectamente como Clase 0.
- **Verdaderos positivos (TP):** 88 muestras de la Clase 1 fueron clasificadas correctamente como Clase 1.

La evidencia visual generada se encuentra en:

![Matriz de confusion](evidencias/final_metrics/confusion_matrix.svg)

---

## Curvas de Perdida

Las curvas de perdida generadas durante el entrenamiento estan documentadas en la carpeta de evidencias:

- `reporte/evidencias/gpu_training_curves.png`
- `reporte/evidencias/cpu_baseline_training_curves.png`

Estas curvas permiten verificar si el proceso de entrenamiento converge y si la perdida de validacion sigue una tendencia razonable. En este proyecto, las curvas muestran que el modelo mejora durante el entrenamiento y alcanza un desempeno util en validacion, aunque todavia existe margen de mejora mediante mas datos, regularizacion o una arquitectura mas robusta.

---

## Reflexion Etapa 1: Preprocesamiento con OpenMP

La primera etapa se enfoco en transformar las imagenes originales en un dataset estructurado y adecuado para el entrenamiento. El pipeline de preprocesamiento incluyo lectura de imagenes, conversion a escala de grises, filtro Sobel, filtro Gaussiano, redimensionamiento a 64x64, normalizacion, aplanamiento y exportacion a CSV.

### Aspectos Positivos

- La implementacion con OpenMP permitio procesar multiples imagenes usando paralelismo en CPU.
- El pipeline produjo un formato de entrada consistente para el entrenamiento en CPU y GPU.
- La normalizacion y reduccion de cada imagen a 4096 caracteristicas simplifico la entrada del modelo.
- El mismo contrato de preprocesamiento pudo reutilizarse posteriormente en la aplicacion de inferencia.

### Resultados Observados

Los tiempos medidos en la etapa de preprocesamiento fueron:

| Hilos | Tiempo (s) | Speedup |
|---:|---:|---:|
| 1 | 31.712 | 1.00 |
| 2 | 35.153 | 0.90 |
| 4 | 14.204 | 2.23 |
| 8 | 13.912 | 2.28 |

Los resultados muestran que el paralelismo mejora el rendimiento al usar 4 y 8 hilos. Sin embargo, la ejecucion con 2 hilos fue mas lenta que la version serial, lo que evidencia que el paralelismo no siempre produce mejora inmediata. Factores como el costo de crear hilos, el acceso a disco, el uso de memoria y la distribucion de trabajo pueden afectar el resultado final.

### Aprendizajes

OpenMP es una herramienta util para acelerar tareas repetitivas en CPU, especialmente cuando el trabajo por imagen es suficiente para compensar el costo de administrar hilos. No obstante, el speedup depende de la configuracion usada y de las caracteristicas del problema. En este proyecto, la mejor mejora se observo con 4 y 8 hilos.

---

## Reflexion Etapa 2: Entrenamiento con CUDA

La segunda etapa se enfoco en entrenar la red neuronal usando CUDA. La arquitectura final del modelo fue:

```text
Entrada(4096) -> Densa(75, ReLU) -> Densa(1, Sigmoide)
```

El entrenamiento uso Binary Cross-Entropy como funcion de perdida y SGD implementado manualmente. Se desarrollaron kernels CUDA para multiplicacion de matrices, funciones de activacion, calculo de perdida, calculo de gradientes y actualizacion de parametros.

### Aspectos Positivos

- CUDA redujo significativamente el tiempo del bucle de entrenamiento frente a la linea base en CPU.
- La implementacion en GPU permitio probar multiples combinaciones de hiperparametros.
- El modelo final alcanzo un desempeno aceptable en validacion y prueba.
- La exportacion de pesos a `weights.npz` permitio reutilizar el modelo en el script de evaluacion y en la aplicacion Streamlit.

### Resultados Observados

La mejor configuracion reportada para GPU fue:

| Neuronas ocultas | Learning rate | Epocas | Perdida validacion | Exactitud validacion |
|---:|---:|---:|---:|---:|
| 75 | 0.1 | 50 | 0.5402 | 0.8272 |

El speedup reportado de GPU frente a CPU fue:

| Version | Tiempo de entrenamiento (s) |
|---|---:|
| CPU baseline | 3.1745 |
| GPU CUDA | 0.3041 |
| Speedup | 10.44x |

### Aprendizajes

CUDA ofrece una ventaja clara en operaciones numericas paralelizables, como multiplicaciones de matrices y calculo de gradientes. Sin embargo, implementar entrenamiento manualmente tambien aumenta la complejidad del proyecto. Es necesario controlar correctamente la memoria, las transferencias entre CPU y GPU, la configuracion de kernels y la estabilidad numerica.

---

## Evaluacion General

El modelo final cumple con el objetivo academico del proyecto: construir un clasificador de imagenes a partir de un pipeline propio de preprocesamiento y entrenarlo usando tecnicas de computacion paralela.

Los resultados son positivos porque:

- La exactitud en test supera el 80%.
- La precision y el recall se mantienen equilibrados.
- El F1-score confirma que el modelo no depende de una sola metrica.
- La matriz de confusion muestra que ambas clases son reconocidas.
- El proyecto integra el flujo completo: imagenes crudas, preprocesamiento, entrenamiento, evaluacion e inferencia.

Tambien existen limitaciones:

- El dataset es controlado y puede no representar todas las condiciones reales de iluminacion, postura o camara.
- El modelo depende fuertemente de que el preprocesamiento sea consistente.
- Aun existen falsos positivos y falsos negativos.
- Un sistema real de seguridad requeriria mas datos, validacion mas estricta y pruebas en tiempo real.

---

## Oportunidades de Mejora

Como trabajo futuro se propone:

- Ampliar el dataset con mas personas, iluminaciones, angulos y calidades de imagen.
- Probar arquitecturas adicionales, como redes convolucionales.
- Aplicar aumentacion de datos para mejorar la generalizacion.
- Mejorar la deteccion automatica de la region ocular antes de clasificar.
- Evaluar el sistema en escenarios de tiempo real.
- Comparar mas configuraciones de hilos en OpenMP y tamanos de bloque en CUDA.
- Agregar validacion cruzada o multiples corridas para reducir la dependencia de una unica particion de datos.

---

## Conclusion

La fase de evaluacion confirma que el proyecto logro construir un pipeline funcional para clasificacion de somnolencia. La Etapa 1 permitio generar un dataset estandarizado mediante preprocesamiento con OpenMP, mientras que la Etapa 2 acelero el entrenamiento del modelo mediante CUDA y produjo pesos reutilizables para evaluacion e inferencia.

Las metricas finales muestran que el clasificador puede distinguir entre los dos estados de los ojos con un desempeno aceptable para un prototipo academico. La matriz de confusion confirma que el modelo reconoce ambas clases, aunque se requiere trabajo adicional antes de considerar una aplicacion en un entorno real critico.
