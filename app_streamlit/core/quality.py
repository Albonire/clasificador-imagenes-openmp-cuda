"""Metricas de calidad de imagen, como un registro abierto a extension (OCP).

En vez de un if/else por cada metrica (brillo, luego contraste, luego la
siguiente...), cada chequeo se declara una vez como un `QualityCheck` y se
agrega a `QUALITY_CHECKS`. La UI (ui/components.py) itera la lista sin saber
cuantos chequeos hay ni que miden: agregar nitidez, simetria, etc. en el
futuro es sumar un elemento a la lista, no editar el render.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .preprocessing import to_grayscale


@dataclass(frozen=True)
class QualityCheck:
    """Una metrica de calidad evaluable sobre un recorte RGB float32."""

    name: str
    metric: Callable[[np.ndarray], float]
    is_low: Callable[[float], bool]
    unit: str
    warning: str


def calculate_brightness(rgb_array):
    """Brillo promedio (0-255)."""
    return float(np.mean(to_grayscale(rgb_array)))


def calculate_contrast(rgb_array):
    """Contraste como desviacion estandar de la luminancia."""
    return float(np.std(to_grayscale(rgb_array)))


BRIGHTNESS_CHECK = QualityCheck(
    name="Brillo Promedio",
    metric=calculate_brightness,
    is_low=lambda value: value < 80,
    unit="/255",
    warning=(
        "Brillo bajo: Aumente la iluminacion frontal para mejorar la "
        "deteccion de bordes (Sobel)."
    ),
)

CONTRAST_CHECK = QualityCheck(
    name="Contraste (Desv. Est.)",
    metric=calculate_contrast,
    is_low=lambda value: value < 30,
    unit="",
    warning=(
        "Contraste bajo: Mejore la iluminacion para que el modelo distinga "
        "mejor bordes de parpados."
    ),
)

# Punto de extension: agregar nuevos QualityCheck aqui no requiere tocar
# ui/components.py, que solo recorre esta lista.
QUALITY_CHECKS = [BRIGHTNESS_CHECK, CONTRAST_CHECK]
