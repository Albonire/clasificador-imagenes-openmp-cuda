# CRISP-DM Phase 6: Deployment

## Objective

The objective of this phase is to deploy the trained drowsiness classifier as an interactive inference application that end users can run without retraining the model or manually executing the preprocessing pipeline. The deployment closes the CRISP-DM cycle by making the system usable on new images captured in real time, while preserving the same preprocessing contract and forward pass used during training and evaluation.

The deliverable is a **Streamlit web application** (`app_streamlit/`) that:

1. Accepts a new image from file upload or camera capture.
2. Detects and crops the eye region when possible.
3. Applies the same preprocessing pipeline as Stage 1 (OpenMP).
4. Loads the exported weights from `modelo/weights.npz`.
5. Displays the prediction, confidence, quality checks, and an educational view of the pipeline.

---

## Deployment Overview

The project follows a **model-in-image** deployment strategy: trained weights are bundled with the application at build time. Inference runs on the server using NumPy on CPU; CUDA is used only during offline training.

```text
User (browser)
    │
    ▼
Streamlit server (app_streamlit/app.py)
    │
    ├── Haar cascades (OpenCV) ──► eye-region crop
    ├── preprocessing.py ────────► 4096-dim feature vector
    ├── inference.py ────────────► probability + class
    └── ui/components.py ────────► result, metrics, explainers
```

Runtime dependencies:

- `modelo/weights.npz` — trained MLP weights (required).
- `reporte/evidencias/final_metrics/metrics.json` — optional; sidebar test accuracy.

The application does not depend on the raw dataset or the OpenMP binary at runtime.

---

## Application Architecture

Numeric logic is separated from presentation so preprocessing and inference can be tested without Streamlit.

| Module | Responsibility |
|--------|----------------|
| `app.py` | Entry point: page config, cached resource loading, orchestration |
| `core/preprocessing.py` | Grayscale → Sobel → Gaussian → 64×64 → normalize → flatten (matches `preprocess_serial.c`) |
| `core/inference.py` | Weight loading, ReLU + sigmoid forward pass, decision threshold |
| `core/detection.py` | Face and eye detection with OpenCV Haar cascades |
| `core/quality.py` | Extensible brightness and contrast checks |
| `ui/components.py` | Sidebar, input widgets, result cards, pipeline explainer |
| `ui/theme.py` | Neutral design tokens and custom CSS |

### Resource caching

Heavy resources load once per server process via `@st.cache_resource`:

- Model weights (`load_weights`).
- Haar cascade classifiers (`build_eye_detectors`).

Sidebar metadata (architecture, parameter count, accuracy) is derived from `weights.npz` and `metrics.json`, not hardcoded strings.

### Extension points

- **Quality checks** — add a `QualityCheck` to `core/quality.QUALITY_CHECKS`.
- **Input sources** — add an `ImageSource` to `ui/components.INPUT_SOURCES`.
- **Decision threshold** — single constant `core.inference.DECISION_THRESHOLD` (tune with `modelo/tune_threshold.py`).

---

## End-to-End Inference Pipeline

### 1. Image acquisition

| Source | Details |
|--------|---------|
| File upload | JPG, JPEG, PNG via `st.file_uploader` |
| Camera | `st.camera_input` with optional digital zoom (1.0×–3.0×) |

Images pass through EXIF orientation correction (`ImageOps.exif_transpose`).

### 2. Eye-region detection

Training data uses small eye crops (~78×78 px). At inference, `core/detection.py` aligns input with that distribution:

1. Detect largest frontal face (Haar cascade).
2. Search eyes in the upper 60% of the face ROI.
3. If no face, fall back to full-image eye detection.
4. Crop largest eye with 30% padding into a square.

| Status | Behavior |
|--------|----------|
| `ojo` | Eye crop sent to preprocessing (preferred) |
| `rostro_sin_ojos` | Full image used; lower expected accuracy |
| `sin_deteccion` | Full image used; user advised to improve framing |

### 3. Preprocessing

Mirrors `etapa1_openmp/preprocess_serial.c`:

1. RGB → grayscale (BT.601: 0.299, 0.587, 0.114).
2. Sobel 3×3 with replicate padding.
3. Gaussian 3×3 smoothing.
4. Bilinear resize to 64×64 (edge-aligned).
5. Clip [0, 255], divide by 255, flatten to 4096 features.

### 4. Classification

Forward pass in `core/inference.py` matches `train_gpu.cu`:

```text
z1 = W1 @ x + b1  →  ReLU  →  z2 = W2 @ a1 + b2  →  sigmoid  →  ŷ
```

- Threshold: `DECISION_THRESHOLD = 0.5`.
- Class 0: eyes open / alert.
- Class 1: eyes closed / drowsiness.

### 5. User feedback

- Binary result with semantic color.
- Per-class probabilities and confidence bar.
- Brightness and contrast metrics with low-value warnings.
- Expandable pipeline and model explainers.

---

## Model Integration

| Artifact | Role |
|----------|------|
| `modelo/weights.npz` | Keys `W1`, `b1`, `W2`, `b2` |
| `modelo/export_weights.py` | Converts CUDA binary weights to `.npz` |
| `core/inference.py` | Forward pass and threshold |
| `metrics.json` | Test accuracy for sidebar (82.59%) |

If weights are missing, `app.py` shows a warning and calls `st.stop()`.

Inference is CPU-only by design: one 4096→75→1 pass is negligible vs. I/O and OpenCV, and avoids GPU dependencies in production.

---

## Local Development and Execution

### Prerequisites

- Python 3.13+ (see `pyproject.toml`).
- `modelo/weights.npz` present.

### Commands

```bash
cd app_streamlit
pip install -r requirements.txt
streamlit run app.py
```

Default URL: `http://localhost:8501`.

### Python dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI and server |
| `numpy` | Preprocessing and inference |
| `Pillow` | Image I/O |
| `opencv-python` | Haar cascade detection |

---

## Containerized Deployment

Root `Dockerfile` provides a production-oriented image.

### Image characteristics

- Multi-stage build with `uv` and frozen `uv.lock`.
- Runtime: `python:3.13-slim` + `libglib2.0-0` (OpenCV).
- Non-root user `appuser` (UID 1000).
- Copied artifacts: `.streamlit/`, `app_streamlit/`, `modelo/`.
- Health check: `GET /_stcore/health`.
- Port: `PORT` env var (default 8501).

### Build and run

```bash
docker build -t drowsiness-classifier .
docker run -p 8501:8501 drowsiness-classifier
```

### Streamlit configuration

`.streamlit/config.toml`:

- `maxUploadSize = 10` MB
- `enableXsrfProtection = true`
- `gatherUsageStats = false`
- Theme aligned with `ui/theme.py` (neutral palette, indigo accent)

Dockerfile environment:

- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`

### Redeploy workflow

1. Train and evaluate (Phases 4–5).
2. Copy best weights to `modelo/weights.npz`.
3. Optionally update `DECISION_THRESHOLD` via `modelo/tune_threshold.py`.
4. Commit and push to `main` — redeploy embeds new weights in the Docker image.

---

## User Interface Development

| Element | Implementation |
|---------|----------------|
| Header | Institutional branding (Universidad de Pamplona) |
| Sidebar | Model status, framing guide, prediction factors |
| Layout | Two columns: input/detection \| result/quality |
| Pipeline explainer | Original → grayscale → Sobel → 64×64 thumbnails |
| Model explainer | Architecture, preprocessing steps, dataset notes |
| Footer | Model and stack summary |

Design: neutral gray background, white cards, desaturated semantic colors only for the classification result (`ui/theme.py`).

---

## Development Notes

- Preprocessing kernels and resize logic were moved verbatim from an earlier monolithic `app.py` into `core/preprocessing.py` without changing numeric output.
- `core/` modules intentionally avoid importing Streamlit (testable in isolation).
- Camera zoom applies CSS `transform: scale()` on the video element plus center crop on capture.
- Model explainer text references validation accuracy (~85%); sidebar reads the authoritative test metric from `metrics.json` when available.

---

## Deployment Limitations

The deployed application is an **academic prototype**, not a safety-critical system:

| Limitation | Impact |
|------------|--------|
| No real-time video stream | Single-image analysis per request |
| Haar cascade detection | Fails under extreme angles, occlusion, or poor lighting |
| CPU-only inference | Adequate for demos; edge/batch use needs profiling |
| Static weights in image | Model updates require rebuild and redeploy |
| No authentication or audit log | Not suitable for regulated environments as-is |
| Test accuracy ~82.6% | False positives and false negatives remain (Phase 5) |
| Training–inference gap | Live photos differ from dataset crops; detection mitigates partially |

These constraints are acceptable for the university project scope.

---

## Deployment Validation

Checks defined before considering deployment complete:

1. **Weights present** — app starts without the missing-weights warning.
2. **Preprocessing parity** — `core/preprocessing.py` matches the C pipeline on sample images.
3. **Forward pass parity** — `predict()` matches the evaluation script on the same features.
4. **Detection fallback** — images with/without detectable eyes do not crash the app.
5. **Docker health** — container responds to `/_stcore/health` after startup.

Manual smoke tests with upload and camera input confirm the full user flow.

---

## Phase Conclusion

Phase 6 puts the trained model in the hands of end users through a Streamlit interface that mirrors the training preprocessing contract. The modular `core/` / `ui/` layout, Docker packaging, and validation checklist complete the CRISP-DM deployment stage and connect offline training artifacts to a usable inference product.
