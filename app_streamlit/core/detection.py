"""Deteccion y recorte de la region ocular (Haar cascades de OpenCV).

El dataset de entrenamiento son recortes pequenos de la region del ojo
(~78x78 px). Para que la inferencia reciba una entrada parecida, se detecta
el rostro y los ojos y se recorta un solo ojo en formato cuadrado antes de
alimentar el pipeline gris -> Sobel -> Gauss -> 64x64.
"""

import cv2 as cv2
import numpy as np


def build_eye_detectors():
    """Crea las cascadas Haar de rostro y ojos incluidas en opencv-python."""
    face = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return face, eye


def detect_eye_crop(pil_img, face_cascade, eye_cascade, padding=0.30):
    """Detecta el ojo mas grande y devuelve un recorte cuadrado.

    Estrategia: primero busca el rostro frontal; dentro de su mitad superior
    busca los ojos (reduce falsos positivos). Si no hay rostro, busca ojos en
    toda la imagen como respaldo.

    Returns:
        (crop, status) donde crop es un PIL.Image (o None si no se detecto) y
        status es "ojo", "rostro_sin_ojos" o "sin_deteccion".
    """
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
