"""Punto de entrada de la app Streamlit.

Orquestacion delgada: carga (con cache) los recursos pesados — pesos del
modelo y cascadas Haar — y delega el preprocesamiento, la inferencia y el
render en `core/` y `ui/` respectivamente. Ver app_streamlit/README.md para
el porque de esta separacion (principios SRP/DIP/OCP).
"""

import json
import os

import numpy as np
import streamlit as st

from core.detection import build_eye_detectors, detect_eye_crop
from core.inference import DECISION_THRESHOLD, load_weights, predict
from core.preprocessing import TARGET_SIZE, preprocess_image, to_grayscale
from ui import components
from ui.theme import get_css

BASE_DIR = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(BASE_DIR, "..", "modelo", "weights.npz")
METRICS_PATH = os.path.join(
    BASE_DIR, "..", "reporte", "evidencias", "final_metrics", "metrics.json"
)


@st.cache_resource
def get_weights():
    return load_weights(WEIGHTS_PATH)


@st.cache_resource
def get_eye_detectors():
    return build_eye_detectors()


def build_model_meta(weights):
    """Metadatos del modelo derivados de los pesos y del reporte de metricas.

    Evita strings hardcodeados que se desactualizan: arquitectura y conteo de
    parametros salen de la forma real de `weights.npz`; el accuracy sale del
    reporte mas reciente si existe.
    """
    hidden_units = weights["W1"].shape[0]
    n_params = sum(weights[k].size for k in ("W1", "b1", "W2", "b2"))

    accuracy_label = "N/D"
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        accuracy_label = f"{metrics['accuracy'] * 100:.2f}%"

    return {
        "architecture": f"4096 → {hidden_units} → 1",
        "params": f"{n_params:,}",
        "accuracy": accuracy_label,
    }


def main():
    st.set_page_config(
        page_title="Detección de Somnolencia - Universidad de Pamplona",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(get_css(), unsafe_allow_html=True)

    weights = get_weights()
    if weights is None:
        st.warning(
            "Pesos del modelo no encontrados en modelo/weights.npz. "
            "Ejecute primero el entrenamiento CUDA para generar los pesos."
        )
        st.stop()

    components.render_sidebar(build_model_meta(weights))
    components.render_header()

    col_img, col_res = st.columns(2)

    img_input = None
    infer_img = None
    sobel_img = None
    final_img = None
    detection_status = None

    with col_img:
        img_input = components.capture_input_image()
        if img_input is not None:
            st.image(img_input, use_container_width=True)

            face_cascade, eye_cascade = get_eye_detectors()
            eye_crop, detection_status = detect_eye_crop(
                img_input, face_cascade, eye_cascade
            )
            infer_img = eye_crop if detection_status == "ojo" else img_input
            components.render_detection_feedback(infer_img, detection_status)

    with col_res:
        if img_input is None:
            st.info("Capture o suba una imagen para ver el resultado.")
        else:
            try:
                features, sobel_img, final_img = preprocess_image(infer_img)
                prob, clase = predict(features, weights, DECISION_THRESHOLD)
            except (ValueError, IndexError, MemoryError):
                st.error(
                    "Error al preprocesar la imagen. Verifique que la imagen "
                    "tenga contenido válido."
                )
                img_input = None

            if img_input is not None:
                components.render_result(prob, clase, DECISION_THRESHOLD)

                rgb_array = np.array(infer_img.convert("RGB"), dtype=np.float32)
                components.render_quality_checks(rgb_array)

    if img_input is not None and infer_img is not None:
        st.markdown("---")
        rgb_array = np.array(infer_img.convert("RGB"), dtype=np.float32)
        gray_vis = to_grayscale(rgb_array).astype(np.uint8)
        sobel_vis = np.clip(sobel_img, 0, 255).astype(np.uint8)
        final_vis = (np.clip(final_img, 0, 1) * 255).astype(np.uint8)

        components.render_pipeline_explainer(
            infer_img, gray_vis, sobel_vis, final_vis, detection_status, TARGET_SIZE
        )
        components.render_model_explainer()

    components.render_footer()


if __name__ == "__main__":
    main()
