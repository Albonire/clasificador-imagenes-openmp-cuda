# Final Report — Drowsiness Image Classifier

**Programación Paralela y Computación Distribuida**  
**Universidad de Pamplona**

Custom image classifier built from scratch: **OpenMP** preprocessing, **CUDA** training, and **Streamlit** deployment for binary eye-state (alert vs. drowsy) classification.

---

## 1. Introduction

Drowsiness is a major factor in accidents during driving and industrial work. This project develops a system that classifies eye states from facial images using digital image processing, parallel computing, and a shallow neural network—without relying on high-level deep-learning frameworks for the core numerical pipeline.

The work follows the **CRISP-DM** methodology (Phases 1–6). This document is the consolidated final report and closing conclusions; phase-specific detail lives in linked Markdown files under `reporte/`.

---

## 2. Problem and Objectives

**Problem:** Detect signs of drowsiness from images so attention state can be monitored in safety-relevant contexts.

**Objective:** Build a program that classifies whether a person appears drowsy or alert from facial/eye images, demonstrating parallel preprocessing (OpenMP), parallel training (CUDA), and a deployable inference interface (Streamlit).

**Use cases considered:** road safety, industrial monitoring, intelligent surveillance prototypes, and academic research ([`CRISP-DM Phase 1.md`](CRISP-DM%20Phase%201.md)).

---

## 3. Dataset

- **Classes:** Class 0 (eyes open / alert), Class 1 (eyes closed / drowsy).
- **Collection:** Webcam images under controlled conditions; balanced classes after cleaning.
- **Scale:** 3,312 raw images processed; train/val/test splits in `dataset/procesado/`.
- **Test set:** 201 samples in `dataset/procesado/test.csv` (not used during training).

See [`CRISP-DM Phase 2.md`](CRISP-DM%20Phase%202.md) and [`dataset_section.md`](dataset_section.md).

---

## 4. System Architecture

```text
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Raw images     │────►│  Stage 1: OpenMP (C) │────►│  CSV features   │
│  dataset/raw/   │     │  Sobel + Gauss + 64² │     │  4096 + label   │
└─────────────────┘     └──────────────────────┘     └────────┬────────┘
                                                              │
                                                              ▼
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Streamlit app  │◄────│  modelo/weights.npz  │◄────│  Stage 2: CUDA  │
│  (inference)    │     │  MLP 4096→75→1       │     │  train + export │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
```

| Stage | Technology | Role |
|-------|------------|------|
| 1 | C + OpenMP | Offline parallel preprocessing |
| 2 | CUDA (+ CPU baseline) | MLP training and weight export |
| Deploy | Python + Streamlit | Live inference on new images |

---

## 5. Stage 1 — OpenMP Preprocessing

**Pipeline per image:** load → grayscale → Sobel 3×3 → Gaussian 3×3 → resize 64×64 → normalize → flatten (4096 features) → CSV row.

**Performance** ([`openmp_results.md`](openmp_results.md)):

| Threads | Time (s) | Speedup | Efficiency |
|--------:|---------:|--------:|-----------:|
| 1 | 27.262 | 1.00 | 1.00 |
| 2 | 17.325 | 1.57 | 0.79 |
| 4 | 14.155 | 1.93 | 0.48 |
| 8 | 13.449 | 2.03 | 0.25 |

Best measured speedup: **2.03×** (8 threads). Amdahl analysis estimates serial fraction **0.38** and theoretical max speedup **~2.61×**, limited mainly by serialized CSV writes.

![OpenMP speedup](evidencias/openmp_speedup_table.png)

Full phase write-up: [`CRISP-DM Phase 3.md`](CRISP-DM%20Phase%203.md).

---

## 6. Stage 2 — CUDA Modeling and Training

**Architecture:**

```text
Input (4096) → Dense (75, ReLU) → Dense (1, Sigmoid)
```

- **Loss:** Binary cross-entropy  
- **Optimizer:** SGD (manual implementation)  
- **Final config:** 75 hidden units, learning rate 0.1, 50 epochs  
- **Weights:** `modelo/weights.npz` (`W1`, `b1`, `W2`, `b2`)

**Training speedup (representative run):**

| Version | Time (s) |
|---------|--------:|
| CPU baseline | 3.1745 |
| GPU CUDA | 0.3041 |
| **Speedup** | **10.44×** |

Validation loss: 0.5402; validation accuracy: 82.72% (training run documented in Phase 4/5).

Full phase write-up: [`CRISP-DM Phase 4.md`](CRISP-DM%20Phase%204.md).

---

## 7. Evaluation (Phase 5)

Evaluation on the held-out **test set** (`modelo/weights.npz`, threshold 0.5):

| Metric | Value |
|--------|------:|
| Binary Cross-Entropy Loss | 0.5636 |
| Accuracy | 82.59% |
| Precision | 82.24% |
| Recall | 84.62% |
| F1-score | 0.8341 |

**Confusion matrix:**

| Actual \ Predicted | Class 0 | Class 1 |
|--------------------|--------:|--------:|
| Class 0 | 78 | 19 |
| Class 1 | 16 | 88 |

![Confusion matrix](evidencias/final_metrics/confusion_matrix.svg)

Loss curves: `evidencias/gpu_training_curves.png`, `evidencias/cpu_baseline_training_curves.png`.

Full phase write-up: [`CRISP-DM Phase 5.md`](CRISP-DM%20Phase%205.md).

---

## 8. Deployment (Phase 6)

The **Streamlit** application (`app_streamlit/`) provides:

- File upload or camera capture  
- Haar-cascade eye detection and crop  
- Same preprocessing as Stage 1 (Python port of `preprocess_serial.c`)  
- NumPy forward pass with exported weights  
- Result UI, quality checks, and pipeline explainers  

**Run locally:**

```bash
cd app_streamlit
pip install -r requirements.txt
streamlit run app.py
```

**Docker:**

```bash
docker build -t drowsiness-classifier .
docker run -p 8501:8501 drowsiness-classifier
```

Full phase write-up: [`CRISP-DM Phase 6.md`](CRISP-DM%20Phase%206.md).

---

## 9. Team Responsibilities

| Integrante | Focus | Main deliverable |
|------------|-------|------------------|
| Raúl | Data enters clean | OpenMP C pipeline |
| Jeferson | GPU trains CPU output | CUDA forward + CPU baseline |
| Fabián | Model learns; weights reach app | CUDA backward + weight export |
| Silvana | Data documented; model used | Dataset + Streamlit app |
| Valentina | Project narrative | CRISP-DM report + metrics |
| Rubén | Integration | Notebook + integration + conclusions |

---

## 10. Evidence Index

| Evidence | Location |
|----------|----------|
| OpenMP speedup table/chart | `evidencias/openmp_speedup_table.png` |
| OpenMP Amdahl analysis | `evidencias/openmp_amdahl_analysis.png` |
| OpenMP benchmark JSON | `evidencias/openmp_benchmark.json` |
| Pipeline flow diagram | `evidencias/openmp_pipeline_flow.png` |
| GPU training curves | `evidencias/gpu_training_curves.png` |
| CPU baseline curves | `evidencias/cpu_baseline_training_curves.png` |
| Test metrics JSON | `evidencias/final_metrics/metrics.json` |
| Confusion matrix (SVG/CSV) | `evidencias/final_metrics/` |
| Metrics summary | `evidencias/final_metrics.md` |

---

## 11. Conclusions

### Project summary

This project built a complete **drowsiness classification system** from raw facial images to an interactive web application, following CRISP-DM across six phases:

| Phase | Focus | Main outcome |
|-------|-------|--------------|
| 1 — Business Understanding | Problem and use cases | Drowsiness detection for safety monitoring |
| 2 — Data Understanding | Dataset exploration | Binary eye-state classes, collection constraints |
| 3 — Data Preparation | OpenMP preprocessing | 4096-dim normalized features in CSV |
| 4 — Modeling | CUDA MLP training | `4096 → 75 → 1` classifier, `weights.npz` |
| 5 — Evaluation | Test metrics | 82.59% accuracy, balanced precision/recall |
| 6 — Deployment | Streamlit app | Interactive inference with eye detection and explainers |

**Parallel computing** appears at two stages: OpenMP for CPU preprocessing and CUDA for GPU-accelerated training. End-user inference is served through a lightweight Streamlit deployment on CPU.

### Achievement of objectives

Phase 1 defined the goal: determine whether a person appears drowsy or alert from facial images. That goal was met:

- The classifier reaches **>80% test accuracy** on held-out data.
- The **full numerical pipeline** (preprocessing, training, inference) is implemented without high-level deep-learning frameworks for the core work.
- **OpenMP** achieved **2.03× speedup** with 8 threads on offline preprocessing.
- **CUDA** reduced training time by **10.44×** vs. the CPU baseline.
- The **Streamlit app** exposes the model via file upload or camera without manual feature extraction.

The system fulfills its academic purpose: integrating image processing, parallel computing, and machine learning into one coherent product.

### Key results

| Area | Result |
|------|--------|
| Test accuracy | 82.59% |
| Precision | 82.24% |
| Recall | 84.62% |
| F1-score | 0.8341 |
| Test samples | 201 |
| OpenMP speedup (8 threads) | 2.03× |
| OpenMP Amdahl limit (estimated) | ~2.61× |
| CUDA training speedup | 10.44× |
| Model architecture | 4096 → 75 → 1 (307,351 parameters) |
| Inference | Sub-second per image on CPU |

Both classes are recognized in balance (78 TN, 88 TP); false positives (19) and false negatives (16) remain non-zero.

### Technical contributions

**Stage 1 — OpenMP preprocessing**

- Parallel load, filter, resize, and CSV export over 3,312 raw images.
- Shared feature format for CPU training, GPU training, evaluation, and deployment.
- Documented speedup and Amdahl analysis ([`openmp_results.md`](openmp_results.md)).

**Stage 2 — CUDA training**

- Hand-implemented forward pass, BCE loss, and SGD on GPU.
- Hyperparameter search (hidden units, learning rate, epochs).
- Portable weight export to `modelo/weights.npz`.

**Deployment — Streamlit**

- Python replication of the C preprocessing pipeline for consistent inference.
- Modular `core/` / `ui/` layout with extension points for checks and inputs.
- Docker image with health checks and production Streamlit settings ([`CRISP-DM Phase 6.md`](CRISP-DM%20Phase%206.md)).

### Lessons learned

1. **Preprocessing parity is non-negotiable.** The model classifies edge-enhanced 64×64 patches, not raw RGB faces. Training–inference mismatch hurts accuracy more than small architecture tweaks.

2. **Parallelism pays differently per stage.** OpenMP helps batch preprocessing but is capped by serialized CSV I/O; CUDA accelerates matrix math during training; single-image inference does not need a GPU.

3. **Engineered features plus a shallow MLP** can perform well when the problem is constrained to eye-region binary classification.

4. **Deployment is more than the model.** Eye detection, quality feedback, and user guidance reduce misuse of predictions in live capture scenarios.

5. **Manual implementations teach deeply** but require discipline: centralize thresholds, kernels, and weight contracts to prevent drift across C, CUDA, and Python.

### Limitations

- Dataset collected under controlled conditions; limited subject, lighting, and pose diversity.
- Haar cascades are fragile compared to modern landmark detectors.
- No temporal modeling — single frames cannot capture gradual drowsiness.
- Flattened-pixel MLP ignores spatial structure that CNNs would use.
- ~17% error rate on test data is unacceptable for safety-critical deployment without further validation.

### Future work

- Larger, more diverse dataset with augmentation.
- Robust face/eye localization (e.g., landmark models).
- Convolutional architectures and optional GPU inference.
- Video-stream temporal analysis for real monitoring scenarios.
- Config-driven threshold tuning in production.
- Usability studies on the Streamlit interface.
- Profiling and optimization of Python preprocessing for edge deployment.

### Final statement

The project delivered an **end-to-end drowsiness classification pipeline**: OpenMP preprocessing, CUDA training, rigorous test evaluation, and Streamlit deployment. Performance is acceptable for an **academic prototype**; the deployed app shows how trained weights and preprocessing logic become a usable tool.

The system is **not** ready for safety-critical use without more data, validation, and engineering. It does meet the course goals of combining **digital image processing**, **OpenMP**, **CUDA**, and **machine learning** in a single documented workflow. Phase 6 closes the CRISP-DM cycle by placing the model in end users’ hands and recording how every stage connects.

---

## 12. Document Map (CRISP-DM)

| Document | Content |
|----------|---------|
| [`CRISP-DM Phase 1.md`](CRISP-DM%20Phase%201.md) | Business understanding |
| [`CRISP-DM Phase 2.md`](CRISP-DM%20Phase%202.md) | Data understanding |
| [`CRISP-DM Phase 3.md`](CRISP-DM%20Phase%203.md) | Data preparation |
| [`CRISP-DM Phase 4.md`](CRISP-DM%20Phase%204.md) | Modeling |
| [`CRISP-DM Phase 5.md`](CRISP-DM%20Phase%205.md) | Evaluation |
| [`CRISP-DM Phase 6.md`](CRISP-DM%20Phase%206.md) | Deployment |
| [`openmp_results.md`](openmp_results.md) | OpenMP benchmarks |
| **`final_report.md`** | **This consolidated report and conclusions** |

---

*End of final report.*
