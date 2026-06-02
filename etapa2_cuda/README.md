# Etapa 2 — Entrenamiento con CUDA

Kernels propios de CUDA para entrenar una MLP desde cero, más la versión CPU para comparar speedup.

## Kernels a implementar
- Multiplicación matriz–vector / matriz–matriz (forward)
- Suma de bias + ReLU
- Sigmoide
- Pérdida BCE
- Gradientes (backpropagation)
- Actualización de pesos (SGD)

## Mediciones
- Entrenamiento CPU vs GPU
- Efecto del tamaño de bloque (16×16, 32×32...)
- `nvidia-smi` durante entrenamiento
