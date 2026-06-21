"""Forward pass de inferencia: capa oculta ReLU + salida sigmoide.

Coincide con train_gpu.cu y el contrato documentado en modelo/README.md:
    z1 = x @ W1.T + b1 ; a1 = relu(z1)
    z2 = a1 @ W2.T + b2 ; y_hat = sigmoid(z2)

`DECISION_THRESHOLD` es el unico punto de extension para el ajuste de umbral
(ver modelo/tune_threshold.py): cambia aqui, sin tocar `predict` ni la UI.
"""

import os

import numpy as np

# Umbral de decision para la clase 1 (ojos cerrados / somnolencia).
# Por defecto 0.5 (igual que el entrenamiento). Si se corre
# modelo/tune_threshold.py y se encuentra un umbral mejor en validacion,
# actualizar esta constante con el valor reportado.
DECISION_THRESHOLD = 0.5

_DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "modelo", "weights.npz"
)


def sigmoid(x):
    """Sigmoide numericamente estable (evita overflow de exp con x muy negativo).

    Identica a 1/(1+exp(-x)) pero reformulada por ramas para no desbordar:
    para x>=0 se usa exp(-x) en (0,1]; para x<0, exp(x)/(1+exp(x)).
    """
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    exp_x = np.exp(x)
    return exp_x / (1.0 + exp_x)


def load_weights(path=_DEFAULT_WEIGHTS_PATH):
    """Carga los pesos entrenados desde archivo npz, o None si no existe."""
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    return {"W1": data["W1"], "b1": data["b1"], "W2": data["W2"], "b2": data["b2"]}


def predict(features, weights, threshold=DECISION_THRESHOLD):
    """Forward pass: capa oculta ReLU + salida sigmoide.

    Args:
        features: vector 1D (4096,)
        weights: dict con W1 (75,4096), b1 (75,), W2 (1,75), b2 (1,)
        threshold: umbral de decision para la clase 1

    Returns:
        prob: probabilidad de clase 1 (ojos cerrados)
        clase: 0 o 1
    """
    z1 = weights["W1"] @ features + weights["b1"]
    a1 = np.maximum(z1, 0)
    z2 = weights["W2"] @ a1 + weights["b2"]
    prob = sigmoid(float(z2[0]))
    return float(prob), 1 if prob >= threshold else 0
