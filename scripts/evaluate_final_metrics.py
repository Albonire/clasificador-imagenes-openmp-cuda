"""Evaluate final metrics on the binary test set.

Usage:
    python scripts/evaluate_final_metrics.py

Outputs are written to:
    reporte/evidencias/final_metrics/
    reporte/evidencias/final_metrics.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_CSV = ROOT / "dataset" / "procesado" / "test.csv"
DEFAULT_WEIGHTS = ROOT / "modelo" / "weights.npz"
DEFAULT_OUTPUT_DIR = ROOT / "reporte" / "evidencias" / "final_metrics"


def read_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    rows: list[list[float]] = []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        if not header or header[0] != "label":
            raise ValueError("The first CSV column must be 'label'.")

        for row in reader:
            labels.append(int(row[0]))
            rows.append([float(value) for value in row[1:]])

    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    if x.shape[1] != 4096:
        raise ValueError(f"Expected 4096 features, got {x.shape[1]}.")
    return x, y


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    out = np.empty_like(values)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def predict_probabilities(x: np.ndarray, weights_path: Path) -> np.ndarray:
    weights = np.load(weights_path, allow_pickle=False)
    z1 = x @ weights["W1"].T + weights["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = (a1 @ weights["W2"].T + weights["b2"]).reshape(-1)
    return sigmoid(z2)


def calculate_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    y_pred = (probabilities >= threshold).astype(np.int32)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    total = int(y_true.size)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    eps = 1e-7
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1.0 - clipped))

    metrics: dict[str, float | int] = {
        "samples": total,
        "threshold": threshold,
        "loss_bce": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    matrix = np.asarray([[tn, fp], [fn, tp]], dtype=np.int32)
    return metrics, matrix, y_pred


def write_confusion_matrix_svg(path: Path, matrix: np.ndarray) -> None:
    cell = 130
    left = 150
    top = 95
    width = left + cell * 2 + 35
    height = top + cell * 2 + 70
    max_value = max(int(matrix.max()), 1)

    def fill(value: int) -> str:
        intensity = int(245 - 150 * (value / max_value))
        return f"rgb({intensity},{intensity + 5},255)"

    cells = []
    for row in range(2):
        for col in range(2):
            x = left + col * cell
            y = top + row * cell
            value = int(matrix[row, col])
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{fill(value)}" stroke="#243b53" stroke-width="2"/>'
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 8}" '
                f'text-anchor="middle" font-size="34" '
                f'font-family="Arial" font-weight="700">{value}</text>'
            )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2}" y="35" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">Matriz de confusion - test set</text>
  <text x="{left + cell}" y="70" text-anchor="middle" font-size="16" font-family="Arial">Prediccion</text>
  <text x="28" y="{top + cell}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 28 {top + cell})">Real</text>
  <text x="{left + cell / 2}" y="{top - 14}" text-anchor="middle" font-size="15" font-family="Arial">Clase 0</text>
  <text x="{left + cell + cell / 2}" y="{top - 14}" text-anchor="middle" font-size="15" font-family="Arial">Clase 1</text>
  <text x="{left - 18}" y="{top + cell / 2 + 5}" text-anchor="end" font-size="15" font-family="Arial">Clase 0</text>
  <text x="{left - 18}" y="{top + cell + cell / 2 + 5}" text-anchor="end" font-size="15" font-family="Arial">Clase 1</text>
  {''.join(cells)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_outputs(
    output_dir: Path,
    metrics: dict[str, float | int],
    matrix: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["real/pred", "clase_0", "clase_1"])
        writer.writerow(["clase_0", int(matrix[0, 0]), int(matrix[0, 1])])
        writer.writerow(["clase_1", int(matrix[1, 0]), int(matrix[1, 1])])

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["y_true", "prob_clase_1", "y_pred"])
        for label, probability, prediction in zip(y_true, probabilities, y_pred):
            writer.writerow([int(label), f"{float(probability):.8f}", int(prediction)])

    write_confusion_matrix_svg(output_dir / "confusion_matrix.svg", matrix)
    write_markdown(output_dir.parent / "final_metrics.md", metrics, matrix)


def write_markdown(path: Path, metrics: dict[str, float | int], matrix: np.ndarray) -> None:
    content = f"""# Metricas finales sobre test set

Evaluacion del modelo final `modelo/weights.npz` sobre `dataset/procesado/test.csv`.

## Resumen

| Metrica | Valor |
|---|---:|
| Muestras | {metrics["samples"]} |
| Loss BCE | {metrics["loss_bce"]:.4f} |
| Exactitud | {metrics["accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1 | {metrics["f1"]:.4f} |

## Matriz de confusion

| Real \\ Prediccion | Clase 0 | Clase 1 |
|---|---:|---:|
| Clase 0 | {int(matrix[0, 0])} | {int(matrix[0, 1])} |
| Clase 1 | {int(matrix[1, 0])} | {int(matrix[1, 1])} |

![Matriz de confusion](final_metrics/confusion_matrix.svg)

## Curvas de perdida

Las curvas de perdida generadas durante el entrenamiento estan documentadas en:

- `reporte/evidencias/gpu_training_curves.png`
- `reporte/evidencias/cpu_baseline_training_curves.png`

## Archivos generados

- `reporte/evidencias/final_metrics/metrics.json`
- `reporte/evidencias/final_metrics/confusion_matrix.csv`
- `reporte/evidencias/final_metrics/confusion_matrix.svg`
- `reporte/evidencias/final_metrics/predictions.csv`
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final test metrics.")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_test, y_test = read_dataset(args.test_csv)
    probabilities = predict_probabilities(x_test, args.weights)
    metrics, matrix, y_pred = calculate_metrics(y_test, probabilities, args.threshold)
    write_outputs(args.output_dir, metrics, matrix, y_test, probabilities, y_pred)

    print("Metricas finales sobre test set")
    print(f"Muestras:  {metrics['samples']}")
    print(f"Loss BCE:  {metrics['loss_bce']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Matriz:    {matrix.tolist()}")


if __name__ == "__main__":
    main()
