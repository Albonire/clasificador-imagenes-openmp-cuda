"""
Convierte los pesos binarios entrenados en GPU (etapa2_cuda/gpu_model/weights_gpu.bin)
al formato .npz que consume la app Streamlit (modelo/weights.npz).

Este script reproduce exactamente la logica de la celda "load_gpu_weights" de
etapa2_cuda/gpu_model/model-gpu.ipynb, para que el contrato de pesos quede
documentado y sea reproducible fuera de Colab.

Formato de entrada (binario, escrito por train_gpu.cu / save_weights):
    int32  input_dim
    int32  hidden_units
    float32[input_dim * hidden_units]  W1  (fila = neurona de entrada, columna = neurona oculta)
    float32[hidden_units]              b1
    float32[hidden_units]              W2  (vector, una neurona de salida)
    float32[1]                         b2

Formato de salida (.npz, consumido por app_streamlit/app.py):
    W1: (hidden_units, input_dim)
    b1: (hidden_units,)
    W2: (1, hidden_units)
    b2: (1,)

Forward de inferencia esperado por la app:
    z1 = x @ W1.T + b1
    a1 = relu(z1)
    z2 = a1 @ W2.T + b2
    y_hat = sigmoid(z2)

Uso:
    python3 modelo/export_weights.py \
        --input etapa2_cuda/gpu_model/weights_gpu.bin \
        --output modelo/weights.npz
"""

import argparse

import numpy as np


def load_gpu_weights(path):
    with open(path, "rb") as f:
        dims = np.fromfile(f, dtype=np.int32, count=2)
        input_dim, hidden_units = int(dims[0]), int(dims[1])
        w1_raw = np.fromfile(f, dtype=np.float32, count=input_dim * hidden_units).reshape(
            input_dim, hidden_units
        )
        b1_raw = np.fromfile(f, dtype=np.float32, count=hidden_units)
        w2_raw = np.fromfile(f, dtype=np.float32, count=hidden_units)
        b2_raw = np.fromfile(f, dtype=np.float32, count=1)
    return input_dim, hidden_units, w1_raw, b1_raw, w2_raw, b2_raw


def export_weights(input_path, output_path):
    input_dim, hidden_units, w1_raw, b1_raw, w2_raw, b2_raw = load_gpu_weights(input_path)

    w1 = w1_raw.T  # (hidden_units, input_dim)
    b1 = b1_raw  # (hidden_units,)
    w2 = w2_raw.reshape(1, hidden_units)  # (1, hidden_units)
    b2 = b2_raw  # (1,)

    np.savez(output_path, W1=w1, b1=b1, W2=w2, b2=b2)

    print(f"input_dim={input_dim} hidden_units={hidden_units}")
    print(f"W1: {w1.shape}, b1: {b1.shape}, W2: {w2.shape}, b2: {b2.shape}")
    print(f"Pesos exportados a '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="etapa2_cuda/gpu_model/weights_gpu.bin",
        help="Ruta al archivo binario de pesos generado por train_gpu.cu",
    )
    parser.add_argument(
        "--output",
        default="modelo/weights.npz",
        help="Ruta de salida del archivo .npz para la app Streamlit",
    )
    args = parser.parse_args()
    export_weights(args.input, args.output)
