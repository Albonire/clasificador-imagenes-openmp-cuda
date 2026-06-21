"""Preprocesamiento de imagenes: replica exacta de preprocess_serial.c.

Pipeline: gris -> Sobel -> Gaussiano -> 64x64 (bilineal edge-aligned) -> /255.
Movido verbatim desde app.py al refactorizar (ver app_streamlit/README.md);
los valores numericos no cambiaron, solo la ubicacion del codigo.
"""

import numpy as np

TARGET_SIZE = 64
GRAY_R, GRAY_G, GRAY_B = 0.299, 0.587, 0.114
MAX_PIXEL = 255.0

# Kernels (coinciden con preprocess_serial.c)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
GAUSS_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0


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


def to_grayscale(rgb_array):
    """Luminancia ponderada (BT.601) sobre un array RGB float32."""
    return (
        GRAY_R * rgb_array[:, :, 0]
        + GRAY_G * rgb_array[:, :, 1]
        + GRAY_B * rgb_array[:, :, 2]
    )


def preprocess_image(pil_img):
    """Pipeline completo: gris -> Sobel -> Gauss -> 64x64 -> /255.

    Returns:
        features: vector 1D (4096,) listo para el forward pass
        sobel_vis: imagen Sobel (para visualizacion)
        final_vis: imagen 64x64 normalizada (para visualizacion)
    """
    rgb = np.array(pil_img.convert("RGB"), dtype=np.float32)
    gray = to_grayscale(rgb)
    edges = apply_sobel(gray)
    smoothed = apply_gaussian(edges)
    resized = resize_bilinear(smoothed, TARGET_SIZE, TARGET_SIZE)
    normalized = np.clip(resized, 0.0, MAX_PIXEL) / MAX_PIXEL
    return normalized.flatten(), edges, normalized
