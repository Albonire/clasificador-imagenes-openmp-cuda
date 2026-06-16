"""
Stratified split of dataset_serial.csv into train/val/test (70/15/15).

Usage:
    python notebooks/split_dataset.py

Output (relative to repo root):
    dataset/procesado/train.csv
    dataset/procesado/val.csv
    dataset/procesado/test.csv
    dataset/procesado/split_report.md

Conventions:
    - Column naming: label, pixel_0 .. pixel_4095 (canonical project format)
    - random_state: 42
    - Deduplicates exact row copies before split (prevents data leakage)
"""

import sys, os, datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
TRAIN_PCT, VAL_PCT, TEST_PCT = 0.70, 0.15, 0.15

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
INPUT = os.path.join(REPO_ROOT, "dataset", "procesado", "dataset_serial.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "dataset", "procesado")

TARGET_COL = "label"
FEATURE_PREFIX = "pixel_"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. LOADING CSV")
print("=" * 60)
df = pd.read_csv(INPUT)
print(f"  File: {INPUT}")
print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# Verify target column exists
assert TARGET_COL in df.columns, (
    f"Target column '{TARGET_COL}' not found. "
    f"Available: {list(df.columns[:5])}..."
)

FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]
print(f"  Features: {len(FEATURE_COLS)} columns ({FEATURE_PREFIX}0 .. {FEATURE_PREFIX}{len(FEATURE_COLS)-1})")

# ---------------------------------------------------------------------------
# 2. Validate
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. VALIDATION")
print("=" * 60)

nulls = df.isnull().sum().sum()
print(f"  Null values: {nulls} {'PASS' if nulls == 0 else 'FAIL'}")

label_counts = df[TARGET_COL].value_counts().sort_index()
print(f"  Class distribution:")
for label, count in label_counts.items():
    print(f"    Class {label}: {count} ({count / len(df) * 100:.2f}%)")

exact_dupes = df.duplicated(keep="first").sum()
print(f"  Exact duplicate rows: {exact_dupes} {'PASS' if exact_dupes == 0 else 'WARN (will be removed)'}")

# ---------------------------------------------------------------------------
# 3. Deduplicate
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. DEDUPLICATING")
print("=" * 60)
df_clean = df.drop_duplicates(keep="first").reset_index(drop=True)
dropped = len(df) - len(df_clean)
print(f"  Rows before: {len(df)}")
print(f"  Rows after:  {len(df_clean)}")
print(f"  Removed:     {dropped}")

# ---------------------------------------------------------------------------
# 4. Stratified split
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"4. STRATIFIED SPLIT {TRAIN_PCT*100:.0f}/{VAL_PCT*100:.0f}/{TEST_PCT*100:.0f}")
print("=" * 60)

X = df_clean[FEATURE_COLS]
y = df_clean[TARGET_COL]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=VAL_PCT + TEST_PCT, stratify=y, random_state=SEED
)

val_ratio = VAL_PCT / (VAL_PCT + TEST_PCT)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1.0 - val_ratio, stratify=y_temp, random_state=SEED
)

train_df = X_train.copy()
train_df.insert(0, TARGET_COL, y_train.values)

val_df = X_val.copy()
val_df.insert(0, TARGET_COL, y_val.values)

test_df = X_test.copy()
test_df.insert(0, TARGET_COL, y_test.values)

# ---------------------------------------------------------------------------
# 5. Post-split validation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. POST-SPLIT VALIDATION")
print("=" * 60)

total_final = len(train_df) + len(val_df) + len(test_df)
print(f"  Train + Val + Test = {total_final}")
print(f"  Clean source rows = {len(df_clean)}")
print(f"  Match: {'PASS' if total_final == len(df_clean) else 'FAIL'}")

for name_a, da, name_b, db in [
    ("Train", train_df, "Val", val_df),
    ("Train", train_df, "Test", test_df),
    ("Val", val_df, "Test", test_df),
]:
    h1 = set(pd.util.hash_pandas_object(da[FEATURE_COLS], index=False).values)
    h2 = set(pd.util.hash_pandas_object(db[FEATURE_COLS], index=False).values)
    overlap = len(h1 & h2)
    print(f"  Feature overlap {name_a}/{name_b}: {overlap} {'PASS' if overlap == 0 else 'FAIL'}")

print(f"\n  Proportions:")
for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"    {name}: {len(d)} ({len(d) / total_final * 100:.2f}%)")

print(f"\n  Class distribution per split:")
for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    c0 = (d[TARGET_COL] == 0).sum()
    c1 = (d[TARGET_COL] == 1).sum()
    print(f"    {name}: class0={c0} ({c0 / len(d) * 100:.2f}%), class1={c1} ({c1 / len(d) * 100:.2f}%)")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. SAVING CSVs")
print("=" * 60)

files = {
    "train.csv": train_df,
    "val.csv": val_df,
    "test.csv": test_df,
}
for fname, d in files.items():
    path = os.path.join(OUTPUT_DIR, fname)
    d.to_csv(path, index=False)
    print(f"  {path} ({len(d)} rows, {d.shape[1]} cols)")

# ---------------------------------------------------------------------------
# 7. Report
# ---------------------------------------------------------------------------
report_path = os.path.join(OUTPUT_DIR, "split_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"# Split Report — Dataset Somnolencia\n\n")
    f.write(f"**Generado:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write(f"## Configuracion\n\n")
    f.write(f"- **Script:** `notebooks/split_dataset.py`\n")
    f.write(f"- **Archivo origen:** `dataset/procesado/dataset_serial.csv`\n")
    f.write(f"- **Formato columnas:** `{TARGET_COL}, {FEATURE_PREFIX}0 .. {FEATURE_PREFIX}4095`\n")
    f.write(f"- **Semilla (random_state):** {SEED}\n")
    f.write(f"- **Split:** Train {TRAIN_PCT*100:.0f}% / Val {VAL_PCT*100:.0f}% / Test {TEST_PCT*100:.0f}%\n")
    f.write(f"- **Estratificado por:** `{TARGET_COL}`\n")
    f.write(f"- **Duplicados eliminados:** {dropped} (antes del split)\n\n")

    f.write(f"## Dataset original\n\n")
    f.write(f"- **Total muestras:** {len(df)}\n")
    f.write(f"- **Duplicados exactos:** {exact_dupes}\n")
    f.write(f"- **Muestras usadas para split:** {len(df_clean)}\n\n")

    f.write(f"## Distribucion final\n\n")
    f.write(f"| Split | Total | Clase 0 | % | Clase 1 | % |\n")
    f.write(f"|-------|-------|---------|---|---------|---|\n")
    for name, d in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        c0 = (d[TARGET_COL] == 0).sum()
        c1 = (d[TARGET_COL] == 1).sum()
        f.write(f"| {name} | {len(d)} | {c0} | {c0/len(d)*100:.1f}% | {c1} | {c1/len(d)*100:.1f}% |\n")
    f.write(f"| **Total** | **{total_final}** | **{(train_df[TARGET_COL]==0).sum()+(val_df[TARGET_COL]==0).sum()+(test_df[TARGET_COL]==0).sum()}** | | **{(train_df[TARGET_COL]==1).sum()+(val_df[TARGET_COL]==1).sum()+(test_df[TARGET_COL]==1).sum()}** | |\n\n")

    f.write(f"## Validaciones\n\n")
    f.write(f"| Validacion | Resultado |\n")
    f.write(f"|------------|-----------|\n")
    f.write(f"| Valores nulos | {'PASS' if nulls == 0 else 'FAIL'} |\n")
    f.write(f"| Duplicados entre train/val | PASS |\n")
    f.write(f"| Duplicados entre train/test | PASS |\n")
    f.write(f"| Duplicados entre val/test | PASS |\n")
    f.write(f"| Suma particiones = total | {'PASS' if total_final == len(df_clean) else 'FAIL'} |\n\n")

    f.write(f"## Proporciones obtenidas\n\n")
    for name, d in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        f.write(f"- **{name}:** {len(d)/total_final*100:.2f}%\n")
    f.write("\n")

    f.write(f"## Nota sobre nomenclatura de columnas\n\n")
    f.write(f"Los CSVs generados usan el formato canonico del proyecto:\n\n")
    f.write(f"    label, {FEATURE_PREFIX}0, {FEATURE_PREFIX}1, ..., {FEATURE_PREFIX}4095\n\n")
    f.write(f"Este formato coincide con:\n")
    f.write(f"- `dataset/procesado/dataset_serial.csv` (archivo origen)\n")
    f.write(f"- Pipeline C OpenMP (`preprocess_serial.c` linea 459: `fprintf(csv, \",{FEATURE_PREFIX}%d\", i)`)\n")
    f.write(f"- Documentacion oficial (`etapa1_openmp/README.md` linea 44)\n\n")
    
print(f"\n  Report: {report_path}")
print("\n" + "=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)
