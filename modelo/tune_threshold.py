"""Ajusta el umbral de decision del clasificador SIN reentrenar.

El forward pass (W1/b1/W2/b2 -> sigmoide) ya esta entrenado; el umbral de
0.5 usado para convertir probabilidad en clase es una eleccion aparte y se
puede mejorar barriendo valores y maximizando F1 en validacion, sin tocar
los pesos. Es la mejora "segura" mencionada en el plan: cambia una sola
constante (`core.inference.DECISION_THRESHOLD`), no el modelo desplegado.

Reutiliza `core.inference.predict` (la misma funcion que usa la app) para
garantizar que el umbral elegido se mide con exactamente el mismo forward
pass que vera produccion.

Uso:
    python3 modelo/tune_threshold.py
    python3 modelo/tune_threshold.py --weights modelo/weights.npz \\
        --val dataset/procesado/val.csv --test dataset/procesado/test.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app_streamlit"))

from core.inference import load_weights, predict  # noqa: E402


def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    y = df["label"].to_numpy(dtype=np.float32)
    x = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    return x, y


def predict_all(x, weights, threshold):
    probs = np.empty(len(x), dtype=np.float64)
    classes = np.empty(len(x), dtype=np.int32)
    for i, row in enumerate(x):
        probs[i], classes[i] = predict(row, weights, threshold)
    return probs, classes


def metrics_at(y_true, classes):
    return {
        "accuracy": accuracy_score(y_true, classes),
        "precision": precision_score(y_true, classes, zero_division=0),
        "recall": recall_score(y_true, classes, zero_division=0),
        "f1": f1_score(y_true, classes, zero_division=0),
    }


def format_metrics(label, m):
    return (
        f"{label}: accuracy={m['accuracy']:.4f} precision={m['precision']:.4f} "
        f"recall={m['recall']:.4f} f1={m['f1']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default=os.path.join(os.path.dirname(__file__), "weights.npz")
    )
    parser.add_argument(
        "--val",
        default=os.path.join(
            os.path.dirname(__file__), "..", "dataset", "procesado", "val.csv"
        ),
    )
    parser.add_argument(
        "--test",
        default=os.path.join(
            os.path.dirname(__file__), "..", "dataset", "procesado", "test.csv"
        ),
    )
    parser.add_argument("--baseline", type=float, default=0.5)
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    weights = load_weights(args.weights)
    if weights is None:
        print(f"ERROR: no se encontraron pesos en '{args.weights}'")
        sys.exit(1)

    x_val, y_val = load_xy(args.val)
    x_test, y_test = load_xy(args.test)

    # Probabilidades fijas (no dependen del umbral); el umbral solo cambia
    # el corte de clasificacion, asi que se calculan una sola vez.
    probs_val, _ = predict_all(x_val, weights, args.baseline)
    probs_test, _ = predict_all(x_test, weights, args.baseline)

    thresholds = np.arange(args.start, args.stop + 1e-9, args.step)
    best_threshold, best_f1 = args.baseline, -1.0
    for t in thresholds:
        classes_val = (probs_val >= t).astype(int)
        f1 = f1_score(y_val, classes_val, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)

    baseline_val = metrics_at(y_val, (probs_val >= args.baseline).astype(int))
    tuned_val = metrics_at(y_val, (probs_val >= best_threshold).astype(int))
    baseline_test = metrics_at(y_test, (probs_test >= args.baseline).astype(int))
    tuned_test = metrics_at(y_test, (probs_test >= best_threshold).astype(int))

    print(f"Umbral base: {args.baseline:.2f}  |  Umbral óptimo (val, max F1): {best_threshold:.2f}")
    print()
    print("-- Validación --")
    print(format_metrics(f"  base  (thr={args.baseline:.2f})", baseline_val))
    print(format_metrics(f"  ajustado (thr={best_threshold:.2f})", tuned_val))
    print()
    print("-- Test --")
    print(format_metrics(f"  base  (thr={args.baseline:.2f})", baseline_test))
    print(format_metrics(f"  ajustado (thr={best_threshold:.2f})", tuned_test))
    print()
    if abs(best_threshold - args.baseline) < 1e-9:
        print("El umbral óptimo coincide con el base: no hay cambio que aplicar.")
    else:
        print(
            "Si la mejora en test se sostiene, actualizar manualmente "
            f"DECISION_THRESHOLD = {best_threshold:.2f} en "
            "app_streamlit/core/inference.py.\n"
            "Nota: val.csv/test.csv en este repo son el subconjunto representativo "
            "(201 filas cada uno); re-correr este script con el split completo tras "
            "cualquier reentrenamiento en Colab antes de fijar el umbral final."
        )


if __name__ == "__main__":
    main()
