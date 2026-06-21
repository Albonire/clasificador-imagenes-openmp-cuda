"""Componentes de la interfaz Streamlit.

Cada funcion renderiza una seccion de la app. La regla de oro: estas
funciones reciben datos ya calculados (de `core/`) y solo se ocupan de
pintarlos; no contienen logica de preprocesamiento ni de inferencia.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from core.quality import QUALITY_CHECKS
from ui.theme import status_colors

# ===========================================================================
# Fuentes de imagen, como un registro abierto a extension (OCP)
# ===========================================================================
#
# Cada fuente es autocontenida: dibuja sus propios widgets y devuelve la
# imagen capturada (o None). Agregar una fuente nueva (p.ej. "Desde URL") es
# sumar un ImageSource a INPUT_SOURCES; el flujo principal en app.py no
# cambia.


@dataclass(frozen=True)
class ImageSource:
    label: str
    capture: Callable[[], Optional[Image.Image]]


def _capture_upload():
    uploaded = st.file_uploader(
        "Seleccione una imagen JPG/PNG",
        type=["jpg", "jpeg", "png"],
        help="Formatos soportados: JPG, JPEG, PNG",
    )
    if not uploaded:
        return None
    try:
        return Image.open(uploaded)
    except (UnidentifiedImageError, OSError):
        st.error("El archivo seleccionado no es una imagen válida o está corrupto.")
        return None


def _capture_camera():
    zoom = st.slider(
        "Zoom digital (ajuste para encuadrar los ojos)",
        1.0,
        3.0,
        1.0,
        0.1,
        help="Ajuste el zoom para que los ojos ocupen la mayor parte del encuadre",
    )
    st.caption(f"Nivel de zoom: {zoom:.1f}x - A mayor zoom, mayor recorte centrado")
    if zoom > 1.0:
        st.markdown(
            f"""
            <style>
            div[data-testid="stCameraInput"] video {{
                transform: scale({zoom}) !important;
                transform-origin: center center !important;
                object-fit: cover !important;
            }}
            div[data-testid="stCameraInput"] > div:first-child {{
                overflow: hidden !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    foto = st.camera_input(
        "Capture una foto",
        help="Asegurese de que el rostro este centrado y bien iluminado",
    )
    if not foto:
        return None
    try:
        img = Image.open(foto)
    except OSError:
        st.error("Error al procesar la captura de la cámara. Intente nuevamente.")
        return None
    if zoom > 1.0:
        w, h = img.size
        new_w, new_h = int(w / zoom), int(h / zoom)
        left, top = (w - new_w) // 2, (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))
    return img


INPUT_SOURCES = [
    ImageSource("Subir archivo", _capture_upload),
    ImageSource("Capturar con cámara", _capture_camera),
]


def capture_input_image():
    """Radio de seleccion de fuente + delega en el ImageSource elegido."""
    labels = [source.label for source in INPUT_SOURCES]
    choice = st.radio("Fuente de imagen", labels, horizontal=True)
    source = next(s for s in INPUT_SOURCES if s.label == choice)
    img = source.capture()
    if img is None:
        return None
    return ImageOps.exif_transpose(img)


# ===========================================================================
# Secciones de la interfaz
# ===========================================================================


def render_header():
    st.markdown(
        '<div class="header-institucional">'
        "<h1>Detección de Somnolencia</h1>"
        '<p>Universidad de Pamplona — Clasificador de imágenes desde cero '
        "(OpenMP + CUDA + Streamlit)</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar(model_meta):
    """model_meta: dict con claves accuracy, architecture, params (todas str)."""
    with st.sidebar:
        st.markdown("### Modelo")
        st.markdown(
            '<div class="modelo-status">'
            "<b>Estado:</b> Modelo cargado correctamente<br>"
            "<b>Archivo:</b> weights.npz<br>"
            f"<b>Arquitectura:</b> {model_meta['architecture']}<br>"
            f"<b>Parámetros:</b> {model_meta['params']}<br>"
            f"<b>Accuracy:</b> {model_meta['accuracy']}"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### Guía de Encuadre")
        st.markdown(
            '<div class="modelo-status">'
            "<b>Posición:</b> Rostro centrado y frontal a la cámara<br>"
            "<b>Distancia:</b> 20-30 cm<br>"
            "<b>Iluminación:</b> Luz frontal uniforme<br>"
            "<b>Ángulo:</b> Cabeza recta, sin inclinación<br>"
            "<b>Expresión:</b> Párpados relajados"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### Factores que Afectan la Predicción")
        factores = [
            ("Iluminación", "Luz uniforme frontal, evitar contraluz"),
            ("Distancia", "Fuera del rango 20-30 cm reduce precisión"),
            ("Gafas / maquillaje", "Pueden alterar bordes del ojo"),
            ("Movimiento", "Imagen movida genera bordes borrosos"),
            ("Ángulo", "Mirar directo a la cámara, sin inclinar"),
            ("Oclusión", "Cabello o manos sobre los ojos"),
        ]
        for titulo, desc in factores:
            st.markdown(
                f'<div class="factor-card"><b>{titulo}:</b> {desc}</div>',
                unsafe_allow_html=True,
            )


def render_detection_feedback(infer_img, detection_status):
    """Mensaje + miniatura segun el resultado de la deteccion de ojo."""
    if detection_status == "ojo":
        st.success("Región ocular detectada automáticamente.")
        st.image(infer_img, caption="Recorte enviado al modelo", width=180)
    elif detection_status == "rostro_sin_ojos":
        render_alert(
            "<strong>Rostro detectado, pero no los ojos.</strong> Se usará la "
            "imagen completa (menos preciso). Acérquese y mire de frente."
        )
    else:
        render_alert(
            "<strong>No se detectó rostro ni ojos.</strong> Se usará la imagen "
            "completa (menos preciso). Mejore el encuadre y la iluminación."
        )


def render_alert(html_message):
    st.markdown(
        f'<div class="alert-warning">{html_message}</div>',
        unsafe_allow_html=True,
    )


def render_result(prob, clase, threshold):
    """Tarjeta de resultado + metricas de probabilidad por clase."""
    color, _ = status_colors(is_drowsy=(clase == 1))
    texto = "SOMNOLENCIA DETECTADA" if clase == 1 else "SIN SOMNOLENCIA"
    confianza = prob if clase == 1 else (1 - prob)

    st.markdown(
        '<div class="resultado-card">'
        '<div class="resultado-titulo">RESULTADO DEL ANÁLISIS</div>'
        f'<div class="resultado-clase" style="color: {color};">{texto}</div>'
        f'<div class="resultado-confianza">Confianza: {confianza * 100:.1f}%</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Ojos Abiertos", f"{(1 - prob) * 100:.1f}%")
    col_p2.metric("Ojos Cerrados", f"{prob * 100:.1f}%")
    st.progress(confianza, text=f"Confianza del modelo: {confianza * 100:.1f}%")
    st.caption(f"Umbral de decisión: {threshold:.2f}")


def render_quality_checks(rgb_array):
    """Recorre el registro QUALITY_CHECKS (core/quality.py) y los pinta.

    Agregar un chequeo nuevo (nitidez, simetria, ...) no requiere tocar esta
    funcion: basta con anadirlo a QUALITY_CHECKS.
    """
    cols = st.columns(len(QUALITY_CHECKS))
    for col, check in zip(cols, QUALITY_CHECKS):
        value = check.metric(rgb_array)
        is_low = check.is_low(value)
        with col:
            st.metric(
                check.name,
                f"{value:.1f}{check.unit}",
                delta="Bajo" if is_low else "Normal",
            )
            if is_low:
                render_alert(f"<strong>{check.name}:</strong> {check.warning}")


def render_pipeline_explainer(infer_img, gray_vis, sobel_vis, final_vis, detection_status, target_size):
    with st.expander("¿Cómo funciona el proceso de análisis?"):
        st.markdown(
            '<div class="pipeline-flow">'
            "Original → Escala de grises → Sobel → Imagen final 64×64"
            "</div>",
            unsafe_allow_html=True,
        )

        cols = st.columns(4, gap="small")
        labels_imgs_captions = [
            (
                "Original",
                infer_img,
                "Región ocular detectada"
                if detection_status == "ojo"
                else "Imagen completa (sin detección)",
            ),
            ("Escala de Grises", gray_vis, "Elimina la información de color"),
            ("Detección de Bordes", sobel_vis, "Resalta contornos de párpados y pestañas"),
            ("Final 64×64", final_vis, f"Entrada de la red neuronal ({target_size}x{target_size})"),
        ]
        for col, (titulo, img, caption) in zip(cols, labels_imgs_captions):
            col.markdown(
                f'<div class="processing-col"><h4>{titulo}</h4></div>',
                unsafe_allow_html=True,
            )
            col.image(img, use_container_width=True, clamp=True)
            col.caption(caption)

        st.info(
            "La imagen es convertida a escala de grises, posteriormente el filtro Sobel "
            "detecta los bordes principales de los ojos. Luego, un filtro Gaussiano reduce "
            "el ruido y finalmente la imagen se redimensiona a 64x64 pixeles para ser "
            "procesada por la red neuronal."
        )


def render_model_explainer():
    st.info(
        "Entrenamiento: CUDA (Tesla T4). "
        "Preprocesamiento: OpenMP + Sobel + Gaussiano. "
        "Forward pass implementado en NumPy puro."
    )
    with st.expander("¿Cómo funciona el sistema?"):
        st.markdown(
            """
        **Arquitectura del modelo:**
        - Entrada: imágenes preprocesadas de 64x64 píxeles en escala de grises
        - Capa oculta: 75 neuronas con activación ReLU
        - Capa de salida: 1 neurona con sigmoide
        - Dataset: 1343 imágenes procesadas (3312 raw, 2352 tras deduplicación)
        - Precisión en validación: ~85%

        **Pipeline de preprocesamiento:**
        1. Conversión a escala de grises (0.299R + 0.587G + 0.114B)
        2. Filtro Sobel 3x3 (detección de bordes con padding replicate)
        3. Filtro Gaussiano 3x3 (suavizado con kernel 1-2-1 / 16)
        4. Redimensión bilineal a 64x64 (edge-aligned)
        5. Normalización (clip [0,255] y división por 255)
        6. Aplanamiento a vector de 4096 elementos

        **Factores que influyen en la predicción:**
        - Iluminación frontal uniforme (crítico para Sobel)
        - Ojos centrados ocupando la mayor parte de la imagen
        - Contraste alto entre piel y párpados
        - Distancia adecuada (20-30 cm)
        - Rostro frontal sin inclinación
        - Imagen nítida y bien enfocada

        **Nota:** El modelo no entiende rostros en general. Solo reconoce
        patrones de borde en la región ocular bajo condiciones similares a las
        del dataset de entrenamiento. Para máxima precisión, reproduzca las
        condiciones del dataset.
        """
        )


def render_footer():
    st.markdown(
        '<div class="footer">'
        "Sistema de Detección de Somnolencia · Universidad de Pamplona<br>"
        "Modelo: red neuronal 4096-75-1 · Entrenamiento: CUDA (Tesla T4)<br>"
        "Preprocesamiento: OpenMP · App: Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )
