import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# Constantes

TARGET_SIZE = 64
GRAY_R, GRAY_G, GRAY_B = 0.299, 0.587, 0.114
MAX_PIXEL = 255.0

COLOR_ROJO = "#ad3333"
COLOR_AZUL = "#003366"
COLOR_BLANCO = "#FFFFFF"
COLOR_FONDO = "#f8f9fa"
COLOR_EXITO = "#28a745"
COLOR_ERROR = "#dc3545"
COLOR_ADVERTENCIA = "#ffc107"

# Kernels (coinciden con preprocess_serial.c)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
GAUSS_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0

RUTA_PESOS = os.path.join(os.path.dirname(__file__), "..", "modelo", "weights.npz")

# ---------------------------------------------------------------------------
# CSS institucional
# ---------------------------------------------------------------------------
CSS = f"""
<style>
    .header-institucional {{
        background: linear-gradient(135deg, {COLOR_ROJO} 0%, {COLOR_AZUL} 100%);
        color: white; padding: 24px; border-radius: 12px; text-align: center;
        margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .header-institucional h1 {{ margin: 0; font-size: 26px; font-weight: bold; }}
    .header-institucional p {{ margin: 6px 0 0 0; font-size: 13px; opacity: 0.9; }}

    .resultado-card {{
        background: {COLOR_BLANCO}; border-radius: 16px; padding: 28px 24px;
        border: 2px solid {COLOR_ROJO}; margin: 16px 0; text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }}
    .resultado-titulo {{
        color: {COLOR_AZUL}; font-size: 18px; font-weight: bold; margin-bottom: 16px;
        text-transform: uppercase; letter-spacing: 2px;
    }}
    .resultado-clase {{
        font-size: 36px; font-weight: 900; margin: 16px 0; line-height: 1.3;
    }}
    .resultado-confianza {{
        color: #555; font-size: 16px; font-weight: 500; margin-top: 8px;
    }}

    .pipeline-flow {{
        background: linear-gradient(90deg, {COLOR_AZUL}, {COLOR_ROJO});
        color: white; border-radius: 8px; padding: 10px 16px; margin: 16px 0;
        text-align: center; font-size: 15px; font-weight: 600;
        letter-spacing: 0.5px;
    }}

    .resultado-metricas {{
        border-top: 1px solid #e0e0e0; padding-top: 16px; margin-top: 12px;
    }}

    .processing-col {{
        background: {COLOR_BLANCO}; border: 1px solid #e0e0e0; border-radius: 10px;
        padding: 12px 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .processing-col h4 {{ color: {COLOR_ROJO}; margin: 0 0 8px 0; font-size: 14px; font-weight: 600; }}

    .alert-warning {{
        background: #fff3cd; border-left: 4px solid {COLOR_ADVERTENCIA};
        border-radius: 8px; padding: 10px; margin: 10px 0; color: #856404; font-size: 13px;
    }}

    .footer {{
        margin-top: 36px; padding-top: 16px; border-top: 2px solid {COLOR_ROJO};
        text-align: center; color: #666; font-size: 11px;
    }}

    .flow-step {{
        background: white; border: 1px solid #d0d8e8; border-radius: 8px;
        padding: 10px 14px; margin: 4px 0; font-size: 13px; color: {COLOR_AZUL};
        text-align: center; font-weight: 600;
    }}
    .flow-arrow {{ text-align: center; color: {COLOR_ROJO}; font-size: 14px; line-height: 1.2; }}

    /* Modelo-status card (compacto) */
    .modelo-status {{
        background: #e8f5e9; border-left: 4px solid {COLOR_EXITO};
        border-radius: 8px; padding: 12px 16px; margin: 12px 0;
        font-size: 14px; line-height: 1.8;
    }}

    /* Factor-card individual */
    .factor-card {{
        background: #f0f4ff; border-left: 4px solid {COLOR_AZUL};
        border-radius: 8px; padding: 12px 16px; margin: 8px 0;
        font-size: 14px; line-height: 1.6;
    }}
    .factor-card b {{ color: {COLOR_AZUL}; }}

    section[data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(180deg, {COLOR_AZUL} 0%, #002244 100%);
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.2) !important;
    }}
</style>
"""


# ===========================================================================
# Carga de pesos
# ===========================================================================


@st.cache_resource
def load_weights(path):
    """Carga los pesos entrenados desde archivo npz."""
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    return {"W1": data["W1"], "b1": data["b1"], "W2": data["W2"], "b2": data["b2"]}


# ===========================================================================
# Deteccion y recorte de la region ocular (Haar cascades de OpenCV)
# ===========================================================================
#
# El dataset de entrenamiento son recortes pequenos de la region del ojo
# (~78x78 px). Para que la inferencia reciba una entrada parecida, se detecta
# el rostro y los ojos y se recorta un solo ojo en formato cuadrado antes de
# alimentar el pipeline gris -> Sobel -> Gauss -> 64x64.


@st.cache_resource
def load_eye_detectors():
    """Carga las cascadas Haar de rostro y ojos incluidas en opencv-python."""
    face = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return face, eye


def detect_eye_crop(pil_img, padding=0.30):
    """Detecta el ojo mas grande y devuelve un recorte cuadrado.

    Estrategia: primero busca el rostro frontal; dentro de su mitad superior
    busca los ojos (reduce falsos positivos). Si no hay rostro, busca ojos en
    toda la imagen como respaldo.

    Returns:
        (crop, status) donde crop es un PIL.Image (o None si no se detecto) y
        status es "ojo", "rostro_sin_ojos" o "sin_deteccion".
    """
    face_cascade, eye_cascade = load_eye_detectors()
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    face_found = len(faces) > 0
    if face_found:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        roi_y1 = fy + int(fh * 0.6)  # los ojos estan en la mitad superior
        roi = gray[fy:roi_y1, fx : fx + fw]
        detected = eye_cascade.detectMultiScale(
            roi, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20)
        )
        eyes = [(fx + ex, fy + ey, ew, eh) for (ex, ey, ew, eh) in detected]
    else:
        detected = eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20)
        )
        eyes = list(detected)

    if len(eyes) == 0:
        return None, "rostro_sin_ojos" if face_found else "sin_deteccion"

    ex, ey, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
    cx, cy = ex + ew / 2.0, ey + eh / 2.0
    side = max(ew, eh) * (1.0 + padding)
    x0 = max(0, int(round(cx - side / 2.0)))
    y0 = max(0, int(round(cy - side / 2.0)))
    x1 = min(w_img, int(round(cx + side / 2.0)))
    y1 = min(h_img, int(round(cy + side / 2.0)))

    if x1 <= x0 or y1 <= y0:
        return None, "sin_deteccion"

    crop = pil_img.convert("RGB").crop((x0, y0, x1, y1))
    return crop, "ojo"


# ===========================================================================
# Preprocesamiento (replica exacta de preprocess_serial.c)
# ===========================================================================


def apply_sobel(img):
    """Operador Sobel 3x3 con replicate padding y clamp a [0, 255]."""
    padded = np.pad(img, 1, mode="edge")
    h, w = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    for i in range(3):
        for j in range(3):
            gx += padded[i : i + h, j : j + w] * SOBEL_X[i, j]
            gy += padded[i : i + h, j : j + w] * SOBEL_Y[i, j]
    mag = np.sqrt(gx * gx + gy * gy)
    return np.clip(mag, 0.0, MAX_PIXEL)


def apply_gaussian(img):
    """Filtro Gaussiano 3x3 con replicate padding."""
    padded = np.pad(img, 1, mode="edge")
    h, w = img.shape
    result = np.zeros_like(img)
    for i in range(3):
        for j in range(3):
            result += padded[i : i + h, j : j + w] * GAUSS_KERNEL[i, j]
    return result


def resize_bilinear(src, new_w, new_h):
    """Redimensionamiento bilineal edge-aligned (align_corners=True).

    Version vectorizada con NumPy; produce el mismo resultado que el bucle por
    pixel (mismos ratios, floor por truncamiento y mezcla bilineal), pero sin
    los ~4096 pasos en Python puro.
    """
    h, w = src.shape
    x_ratio = (w - 1) / (new_w - 1) if new_w > 1 else 0.0
    y_ratio = (h - 1) / (new_h - 1) if new_h > 1 else 0.0

    src_x = np.arange(new_w) * x_ratio  # float64, igual que en el bucle
    src_y = np.arange(new_h) * y_ratio
    x0 = src_x.astype(np.intp)  # truncamiento == floor (coords no negativas)
    y0 = src_y.astype(np.intp)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    xw = (src_x - x0)[np.newaxis, :]  # pesos: (1, new_w)
    yw = (src_y - y0)[:, np.newaxis]  # pesos: (new_h, 1)

    # Recolecta las 4 esquinas como mallas (new_h, new_w)
    top_left = src[np.ix_(y0, x0)]
    top_right = src[np.ix_(y0, x1)]
    bottom_left = src[np.ix_(y1, x0)]
    bottom_right = src[np.ix_(y1, x1)]

    top = top_left + (top_right - top_left) * xw
    bottom = bottom_left + (bottom_right - bottom_left) * xw
    return (top + (bottom - top) * yw).astype(np.float32)


def preprocess_image(pil_img):
    """Pipeline completo: gris -> Sobel -> Gauss -> 64x64 -> /255.

    Returns:
        features: vector 1D (4096,) listo para el forward pass
        sobel_vis: imagen Sobel (para visualizacion)
        final_vis: imagen 64x64 normalizada (para visualizacion)
    """
    rgb = np.array(pil_img.convert("RGB"), dtype=np.float32)
    gray = GRAY_R * rgb[:, :, 0] + GRAY_G * rgb[:, :, 1] + GRAY_B * rgb[:, :, 2]
    edges = apply_sobel(gray)
    smoothed = apply_gaussian(edges)
    resized = resize_bilinear(smoothed, TARGET_SIZE, TARGET_SIZE)
    normalized = np.clip(resized, 0.0, MAX_PIXEL) / MAX_PIXEL
    return normalized.flatten(), edges, normalized


# ===========================================================================
# Forward pass (NumPy puro, coincide con train_gpu.cu y modelo/README.md)
# ===========================================================================


def sigmoid(x):
    """Sigmoide numericamente estable (evita overflow de exp con x muy negativo).

    Identica a 1/(1+exp(-x)) pero reformulada por ramas para no desbordar:
    para x>=0 se usa exp(-x) en (0,1]; para x<0, exp(x)/(1+exp(x)).
    """
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    exp_x = np.exp(x)
    return exp_x / (1.0 + exp_x)


def predict(features, weights):
    """Forward pass: capa oculta ReLU + salida sigmoide.

    Args:
        features: vector 1D (4096,)
        weights: dict con W1 (75,4096), b1 (75,), W2 (1,75), b2 (1,)

    Returns:
        prob: probabilidad de clase 1 (ojos cerrados)
        clase: 0 o 1
    """
    z1 = weights["W1"] @ features + weights["b1"]
    a1 = np.maximum(z1, 0)
    z2 = weights["W2"] @ a1 + weights["b2"]
    prob = sigmoid(float(z2[0]))
    return float(prob), 1 if prob >= 0.5 else 0


# ===========================================================================
# Metricas de calidad de imagen
# ===========================================================================


def calculate_brightness(rgb_array):
    """Brillo promedio (0-255)."""
    gray = (
        GRAY_R * rgb_array[:, :, 0]
        + GRAY_G * rgb_array[:, :, 1]
        + GRAY_B * rgb_array[:, :, 2]
    )
    return float(np.mean(gray))


def calculate_contrast(rgb_array):
    """Contraste como desviacion estandar de la luminancia."""
    gray = (
        GRAY_R * rgb_array[:, :, 0]
        + GRAY_G * rgb_array[:, :, 1]
        + GRAY_B * rgb_array[:, :, 2]
    )
    return float(np.std(gray))


# ===========================================================================
# Interfaz principal
# ===========================================================================


def main():
    st.set_page_config(
        page_title="Deteccion de Somnolencia - Universidad de Pamplona",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CSS, unsafe_allow_html=True)

    # ===================================================================
    # SIDEBAR
    # ===================================================================
    with st.sidebar:
        st.markdown("### Modelo")
        st.markdown(
            '<div class="modelo-status" style="border-left-color: ' + COLOR_AZUL + ';">'
            "<b>Arquitectura:</b> 4096 &rarr; 75 &rarr; 1<br>"
            "<b>Parametros:</b> 307,351<br>"
            "<b>Accuracy:</b> 85.27%"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Flujo del Sistema")
        flow_html = """
        <div class="flow-step">1. Captura o subida de imagen</div>
        <div class="flow-arrow">|</div>
        <div class="flow-step">2. Escala de grises</div>
        <div class="flow-arrow">|</div>
        <div class="flow-step">3. Sobel + Gaussiano</div>
        <div class="flow-arrow">|</div>
        <div class="flow-step">4. Resize 64x64</div>
        <div class="flow-arrow">|</div>
        <div class="flow-step">5. Forward NumPy</div>
        <div class="flow-arrow">|</div>
        <div class="flow-step">6. Prediccion</div>
        """
        st.markdown(flow_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Guia de Encuadre")
        st.markdown(
            '<div class="modelo-status" style="border-left-color: ' + COLOR_AZUL + ';">'
            "<b>Posicion:</b> Rostro centrado y frontal a la camara<br>"
            "<b>Distancia:</b> 20-30 cm<br>"
            "<b>Iluminacion:</b> Luz frontal uniforme<br>"
            "<b>Angulo:</b> Cabeza recta, sin inclinacion<br>"
            "<b>Expresion:</b> Parpados relajados"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Factores que Afectan la Prediccion")
        factores = [
            ("Iluminacion", "Luz uniforme frontal, evitar contraluz"),
            ("Distancia", "Fuera del rango 20-30 cm reduce precision"),
            ("Gafas / maquillaje", "Pueden alterar bordes del ojo"),
            ("Movimiento", "Imagen movida genera bordes borrosos"),
            ("Angulo", "Mirar directo a la camara, sin inclinar"),
            ("Oclusion", "Cabello o manos sobre los ojos"),
        ]
        for titulo, desc in factores:
            st.markdown(
                f'<div class="factor-card"><b>{titulo}:</b> {desc}</div>',
                unsafe_allow_html=True,
            )

    # ===================================================================
    # CARGAR PESOS
    # ===================================================================
    weights = load_weights(RUTA_PESOS)
    if weights is None:
        st.warning(
            "Pesos del modelo no encontrados en modelo/weights.npz. "
            "Ejecute primero el entrenamiento CUDA para generar los pesos."
        )
        st.stop()

    # ===================================================================
    # CABECERA
    # ===================================================================
    st.markdown(
        '<div class="header-institucional">'
        "<h1>Universidad de Pamplona</h1>"
        '<p style="font-size:18px; font-weight:normal; margin-top:6px;">'
        "Sistema de Deteccion de Somnolencia</p>"
        "<p>Clasificador de imagenes desde cero - OpenMP + CUDA + Streamlit</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Tarjeta compacta de estado del modelo
    st.markdown(
        '<div class="modelo-status">'
        "<b>Estado:</b> Modelo cargado correctamente<br>"
        "<b>Archivo:</b> weights.npz<br>"
        "<b>Ubicacion:</b> /modelo/"
        "</div>",
        unsafe_allow_html=True,
    )

    # ===================================================================
    # FUENTE DE IMAGEN | RESULTADO
    # ===================================================================
    fuente = st.radio(
        "Fuente de imagen",
        ["Subir archivo", "Capturar con camara"],
        horizontal=True,
    )

    zoom = 1.0
    img_input = None
    infer_img = None
    detection_status = None

    col_img, col_res = st.columns(2)

    with col_img:
        if fuente == "Subir archivo":
            uploaded = st.file_uploader(
                "Seleccione una imagen JPG/PNG",
                type=["jpg", "jpeg", "png"],
                help="Formatos soportados: JPG, JPEG, PNG",
            )
            if uploaded:
                try:
                    img_input = Image.open(uploaded)
                except (UnidentifiedImageError, OSError):
                    st.error(
                        "El archivo seleccionado no es una imagen válida o está corrupto."
                    )
                    img_input = None
        else:
            zoom = st.slider(
                "Zoom digital (ajuste para encuadrar los ojos)",
                1.0,
                3.0,
                1.0,
                0.1,
                help="Ajuste el zoom para que los ojos ocupen la mayor parte del encuadre",
            )
            st.caption(
                f"Nivel de zoom: {zoom:.1f}x - A mayor zoom, mayor recorte centrado"
            )
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
            if foto:
                try:
                    img_input = Image.open(foto)
                except OSError:
                    st.error(
                        "Error al procesar la captura de la cámara. Intente nuevamente."
                    )
                    img_input = None

        if img_input:
            img_input = ImageOps.exif_transpose(img_input)
            if zoom > 1.0:
                w, h = img_input.size
                new_w = int(w / zoom)
                new_h = int(h / zoom)
                left = (w - new_w) // 2
                top_ = (h - new_h) // 2
                img_input = img_input.crop((left, top_, left + new_w, top_ + new_h))
            st.image(img_input, use_container_width=True)

            # Deteccion automatica de la region ocular
            eye_crop, detection_status = detect_eye_crop(img_input)
            if detection_status == "ojo":
                infer_img = eye_crop
                st.success("Region ocular detectada automaticamente.")
                st.image(eye_crop, caption="Recorte enviado al modelo", width=180)
            elif detection_status == "rostro_sin_ojos":
                infer_img = img_input
                st.markdown(
                    '<div class="alert-warning">'
                    "<strong>Rostro detectado, pero no los ojos.</strong> Se usara la "
                    "imagen completa (menos preciso). Acerquese y mire de frente."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                infer_img = img_input
                st.markdown(
                    '<div class="alert-warning">'
                    "<strong>No se detecto rostro ni ojos.</strong> Se usara la imagen "
                    "completa (menos preciso). Mejore el encuadre y la iluminacion."
                    "</div>",
                    unsafe_allow_html=True,
                )

    with col_res:
        if img_input:
            try:
                features, sobel_img, final_img = preprocess_image(infer_img)
                prob, clase = predict(features, weights)
            except (ValueError, IndexError, MemoryError):
                st.error(
                    "Error al preprocesar la imagen. Verifique que la imagen tenga contenido válido."
                )
                img_input = None

            if img_input:
                color_resultado = COLOR_EXITO if clase == 0 else COLOR_ERROR
                texto_resultado = (
                    "SIN SOMNOLENCIA" if clase == 0 else "SOMNOLENCIA DETECTADA"
                )
                confianza = prob if clase == 1 else (1 - prob)

                st.markdown(
                    f'<div class="resultado-card">'
                    f'<div class="resultado-titulo">RESULTADO DEL ANALISIS</div>'
                    f'<div class="resultado-clase" style="color: {color_resultado};">'
                    f"{texto_resultado}</div>"
                    f'<div class="resultado-confianza">Confianza: {confianza * 100:.1f}%</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                col_p1, col_p2 = st.columns(2)
                col_p1.metric("Ojos Abiertos", f"{(1 - prob) * 100:.1f}%")
                col_p2.metric("Ojos Cerrados", f"{prob * 100:.1f}%")

                st.progress(
                    confianza, text=f"Confianza del modelo: {confianza * 100:.1f}%"
                )

                # Validaciones de calidad (sobre el recorte que ve el modelo)
                rgb_array = np.array(infer_img.convert("RGB"), dtype=np.float32)
                brightness = calculate_brightness(rgb_array)
                contrast = calculate_contrast(rgb_array)

                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    delta_b = "Bajo" if brightness < 80 else "Normal"
                    st.metric("Brillo Promedio", f"{brightness:.1f}/255", delta=delta_b)
                    if brightness < 80:
                        st.markdown(
                            '<div class="alert-warning">'
                            "<strong>Brillo bajo:</strong> Aumente la iluminacion frontal "
                            "para mejorar la deteccion de bordes (Sobel)."
                            "</div>",
                            unsafe_allow_html=True,
                        )
                with col_v2:
                    delta_c = "Bajo" if contrast < 30 else "Normal"
                    st.metric(
                        "Contraste (Desv. Est.)", f"{contrast:.1f}", delta=delta_c
                    )
                    if contrast < 30:
                        st.markdown(
                            '<div class="alert-warning">'
                            "<strong>Contraste bajo:</strong> Mejore la iluminacion "
                            "para que el modelo distinga mejor bordes de parpados."
                            "</div>",
                            unsafe_allow_html=True,
                        )
        else:
            st.info("Capture o suba una imagen para ver el resultado.")

    # ===================================================================
    # PROCESO DE ANALISIS
    # ===================================================================
    if img_input:
        st.markdown("---")
        st.markdown("### Proceso de Analisis")

        # Preparar imagenes para visualizacion (solo visual, no modifica pipeline)
        # La cadena gris -> Sobel -> final se deriva del recorte que ve el modelo.
        rgb_array = np.array(infer_img.convert("RGB"), dtype=np.float32)
        gray_vis = (
            GRAY_R * rgb_array[:, :, 0]
            + GRAY_G * rgb_array[:, :, 1]
            + GRAY_B * rgb_array[:, :, 2]
        ).astype(np.uint8)
        sobel_vis = np.clip(sobel_img, 0, 255).astype(np.uint8)
        final_vis = (np.clip(final_img, 0, 1) * 255).astype(np.uint8)

        st.markdown(
            '<div class="pipeline-flow">'
            "Original &rarr; Escala de Grises &rarr; Sobel &rarr; Imagen Final 64x64"
            "</div>",
            unsafe_allow_html=True,
        )

        cols = st.columns(4, gap="small")
        cols[0].markdown(
            '<div class="processing-col"><h4>Original</h4></div>',
            unsafe_allow_html=True,
        )
        cols[0].image(infer_img, use_container_width=True)
        cols[0].caption(
            "Region ocular detectada"
            if detection_status == "ojo"
            else "Imagen completa (sin deteccion)"
        )

        cols[1].markdown(
            '<div class="processing-col"><h4>Escala de Grises</h4></div>',
            unsafe_allow_html=True,
        )
        cols[1].image(gray_vis, use_container_width=True, clamp=True)
        cols[1].caption("Elimina la informacion de color")

        cols[2].markdown(
            '<div class="processing-col"><h4>Deteccion de Bordes</h4></div>',
            unsafe_allow_html=True,
        )
        cols[2].image(sobel_vis, use_container_width=True, clamp=True)
        cols[2].caption("Resalta contornos de parpados y pestanas")

        cols[3].markdown(
            '<div class="processing-col"><h4>Final 64x64</h4></div>',
            unsafe_allow_html=True,
        )
        cols[3].image(final_vis, use_container_width=True, clamp=True)
        cols[3].caption(f"Entrada de la red neuronal ({TARGET_SIZE}x{TARGET_SIZE})")

        st.info(
            "La imagen es convertida a escala de grises, posteriormente el filtro Sobel "
            "detecta los bordes principales de los ojos. Luego, un filtro Gaussiano reduce "
            "el ruido y finalmente la imagen se redimensiona a 64x64 pixeles para ser "
            "procesada por la red neuronal."
        )

    # ===================================================================
    # INFORMACION DEL MODELO
    # ===================================================================
    if img_input:
        st.markdown("---")
        st.markdown("### Informacion del Modelo")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Arquitectura", "4096 -> 75 -> 1")
        col_m2.metric("Parametros", "307,351")
        col_m3.metric("Accuracy", "85.27%")

        st.info(
            "Entrenamiento: CUDA (Tesla T4). "
            "Preprocesamiento: OpenMP + Sobel + Gaussiano. "
            "Forward pass implementado en NumPy puro."
        )

        with st.expander("Como funciona el sistema?"):
            st.markdown(
                """
            **Arquitectura del Modelo:**
            - Entrada: imagenes preprocesadas de 64x64 pixeles en escala de grises
            - Capa oculta: 75 neuronas con activacion ReLU
            - Capa de salida: 1 neurona con sigmoide
            - Dataset: 1343 imagenes procesadas (3312 raw, 2352 tras deduplicacion)
            - Precision en validacion: ~85%

            **Pipeline de Preprocesamiento:**
            1. Conversion a escala de grises (0.299R + 0.587G + 0.114B)
            2. Filtro Sobel 3x3 (deteccion de bordes con padding replicate)
            3. Filtro Gaussiano 3x3 (suavizado con kernel 1-2-1 / 16)
            4. Redimension bilineal a 64x64 (edge-aligned)
            5. Normalizacion (clip [0,255] y division por 255)
            6. Aplanamiento a vector de 4096 elementos

            **Factores que Influyen en la Prediccion:**
            - Iluminacion frontal uniforme (critico para Sobel)
            - Ojos centrados ocupando la mayor parte de la imagen
            - Contraste alto entre piel y parpados
            - Distancia adecuada (20-30 cm)
            - Rostro frontal sin inclinacion
            - Imagen nitida y bien enfocada

            **Nota:** El modelo no entiende rostros en general. Solo reconoce
            patrones de borde en la region ocular bajo condiciones similares a las
            del dataset de entrenamiento. Para maxima precision, reproduzca las
            condiciones del dataset.
            """
            )

        if st.button("Nueva Captura", use_container_width=True):
            st.rerun()

    # ===================================================================
    # FOOTER
    # ===================================================================
    st.markdown(
        '<div class="footer">'
        "Sistema de Deteccion de Somnolencia | Universidad de Pamplona<br>"
        "Modelo: Red Neuronal 4096-75-1 | Entrenamiento: CUDA (Tesla T4)<br>"
        "Preprocesamiento: OpenMP | App: Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
