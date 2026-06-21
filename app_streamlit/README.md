# Aplicación Streamlit

App de inferencia que:
1. Permite subir una foto o capturarla con la cámara
2. Detecta automáticamente la región ocular (Haar cascades de OpenCV)
3. Aplica el mismo preprocesamiento del entrenamiento (gris → Sobel → Gaussiano → 64×64)
4. Carga los pesos entrenados (`modelo/weights.npz`)
5. Muestra la predicción con su probabilidad

## Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

```
app_streamlit/
  app.py              # orquestación: carga con cache de pesos/cascadas, delega en core/ y ui/
  core/                # lógica pura (sin importar streamlit) — testable de forma aislada
    preprocessing.py   # gris -> Sobel -> Gaussiano -> 64x64 (replica preprocess_serial.c)
    inference.py       # forward pass, umbral de decisión, carga de pesos
    detection.py        # recorte de región ocular con Haar cascades
    quality.py          # registro extensible de chequeos de calidad de imagen
  ui/                  # presentación (sí importa streamlit)
    theme.py             # tokens de diseño + CSS ("moderno neutro")
    components.py        # secciones renderizadas (header, resultado, sidebar, pipeline...)
```

La separación `core/` (numérico) vs. `ui/` (presentación) existe para que el preprocesamiento y la
inferencia se puedan verificar sin levantar la app — y para que el rediseño visual no pueda, por
accidente, tocar un número que tiene que coincidir bit a bit con `etapa1_openmp/preprocess_serial.c`
y `etapa2_cuda/gpu_model/train_gpu.cu`.

**Puntos de extensión (Open/Closed):**
- Nuevo chequeo de calidad de imagen → agregar un `QualityCheck` a `core/quality.QUALITY_CHECKS`
  (la UI itera la lista, no hay que tocar `ui/components.py`).
- Nueva fuente de imagen (p. ej. "desde URL") → agregar un `ImageSource` a
  `ui/components.INPUT_SOURCES`.
- Cambiar el umbral de decisión → una sola constante, `core.inference.DECISION_THRESHOLD`
  (ver `modelo/tune_threshold.py` para cómo recalcularla sin reentrenar).

## Diseño

Paleta "moderna neutra": fondo gris casi blanco, superficies en blanco, un único acento (índigo) y
dos colores semánticos desaturados solo para el resultado (despierto / somnolencia). Sin degradados
decorativos ni bordes laterales de color — ver `ui/theme.py` para los tokens.
