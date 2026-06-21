"""Aumento de datos (data augmentation) para el split de entrenamiento.

El dataset es pequeno (935 filas en el subconjunto representativo de este
repo; 1646 en el split completo usado en Colab) y el modelo es una MLP
simple sin regularizacion fuerte -- aumentar el train set es la palanca con
mejor relacion costo/beneficio antes de tocar la arquitectura o el
optimizador (ver etapa2_cuda/gpu_model/TRAINING_IMPROVEMENTS.md).

Transformaciones aplicadas (siempre sobre el vector 64x64 ya preprocesado,
nunca sobre la imagen raw, para no duplicar el pipeline OpenMP/Sobel):
  - flip horizontal (valido: un ojo reflejado sigue siendo un ojo valido)
  - jitter de brillo (+/- delta uniforme)
  - jitter de contraste (escala uniforme alrededor de la media)
  - traslacion pequena (+/- pocos pixeles, con padding por replicacion)
  - rotacion pequena (+/- pocos grados, via PIL)

IMPORTANTE: solo se debe correr sobre el CSV de TRAIN. Nunca sobre
val.csv/test.csv -- mezclar variantes aumentadas de una muestra de test en
train (o viceversa) es fuga de datos (data leakage) y infla las metricas.

Uso:
    python3 notebooks/augment_dataset.py \\
        --input dataset/procesado/train.csv \\
        --output dataset/procesado/train_augmented.csv \\
        --factor 2 \\
        --check-leakage dataset/procesado/val.csv dataset/procesado/test.csv
"""

import argparse
import hashlib

import numpy as np
import pandas as pd
from PIL import Image

IMG_SIZE = 64


def _row_hash(vector):
    """Hash estable de un vector de pixeles, para chequear solapes exactos."""
    return hashlib.sha1(np.round(vector, 6).tobytes()).hexdigest()


def _random_transform(image_2d, rng):
    """Aplica una combinacion aleatoria de transformaciones a una imagen 64x64 en [0,1]."""
    img = image_2d.copy()

    if rng.random() < 0.5:
        img = img[:, ::-1]

    brightness_delta = rng.uniform(-0.06, 0.06)
    img = np.clip(img + brightness_delta, 0.0, 1.0)

    contrast_scale = rng.uniform(0.85, 1.15)
    mean = img.mean()
    img = np.clip((img - mean) * contrast_scale + mean, 0.0, 1.0)

    dx, dy = rng.integers(-3, 4), rng.integers(-3, 4)
    if dx != 0 or dy != 0:
        img = np.roll(img, shift=(dy, dx), axis=(0, 1))
        # Bordes "replicate" en vez de wrap-around (igual filosofia que el
        # padding del pipeline OpenMP: nunca inventa contenido del lado opuesto).
        if dy > 0:
            img[:dy, :] = img[dy : dy + 1, :]
        elif dy < 0:
            img[dy:, :] = img[dy - 1 : dy, :]
        if dx > 0:
            img[:, :dx] = img[:, dx : dx + 1]
        elif dx < 0:
            img[:, dx:] = img[:, dx - 1 : dx]

    angle = rng.uniform(-8, 8)
    if abs(angle) > 0.01:
        as_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        rotated = Image.fromarray(as_uint8).rotate(
            angle, resample=Image.BILINEAR, fillcolor=int(as_uint8.mean())
        )
        img = np.asarray(rotated, dtype=np.float32) / 255.0

    return img


def augment_dataframe(df, factor, seed):
    rng = np.random.default_rng(seed)
    pixel_cols = [c for c in df.columns if c != "label"]

    augmented_rows = []
    for _, row in df.iterrows():
        label = row["label"]
        base = row[pixel_cols].to_numpy(dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
        for _ in range(factor):
            variant = _random_transform(base, rng)
            augmented_rows.append([label] + variant.flatten().tolist())

    augmented_df = pd.DataFrame(augmented_rows, columns=["label"] + pixel_cols)
    return pd.concat([df, augmented_df], ignore_index=True)


def check_leakage(result_df, other_paths):
    """Verifica que ninguna fila del resultado coincida exactamente con val/test."""
    pixel_cols = [c for c in result_df.columns if c != "label"]
    result_hashes = {
        _row_hash(row) for row in result_df[pixel_cols].to_numpy(dtype=np.float32)
    }
    for path in other_paths:
        other_df = pd.read_csv(path)
        other_cols = [c for c in other_df.columns if c != "label"]
        other_hashes = {
            _row_hash(row) for row in other_df[other_cols].to_numpy(dtype=np.float32)
        }
        overlap = result_hashes & other_hashes
        if overlap:
            raise SystemExit(
                f"ERROR: {len(overlap)} filas de '{path}' se solapan con el "
                "resultado aumentado. Aborta -- revisa el split antes de "
                "reentrenar."
            )
        print(f"OK: sin solape con '{path}' ({len(other_hashes)} filas revisadas).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV de train (label,pixel_0..pixel_4095)")
    parser.add_argument("--output", required=True, help="CSV de salida (original + aumentado)")
    parser.add_argument(
        "--factor", type=int, default=2, help="Copias aumentadas por fila original"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check-leakage",
        nargs="*",
        default=[],
        help="Rutas (val.csv test.csv) contra las que verificar que no haya solape",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "label" not in df.columns:
        raise SystemExit("ERROR: el CSV de entrada debe tener una columna 'label'")

    before_counts = df["label"].value_counts().to_dict()
    result = augment_dataframe(df, args.factor, args.seed)
    after_counts = result["label"].value_counts().to_dict()

    print(f"Filas originales: {len(df)} -> Filas finales: {len(result)}")
    print(f"Balance original: {before_counts}")
    print(f"Balance final:    {after_counts}")

    if args.check_leakage:
        check_leakage(result, args.check_leakage)

    result.to_csv(args.output, index=False)
    print(f"Guardado en '{args.output}'")


if __name__ == "__main__":
    main()
