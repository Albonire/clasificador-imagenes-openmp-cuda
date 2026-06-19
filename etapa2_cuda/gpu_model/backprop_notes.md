# Notas de backpropagation — MLP de Somnolencia

Derivación matemática del backward pass implementado en `train_gpu.cu`, con el
mapeo explícito de cada fórmula al kernel CUDA que la calcula. Sirve como
material de respaldo para la sustentación.

## Arquitectura y notación

```
Entrada X (N, 4096) -> Densa W1,b1 (ReLU) -> A1 (N, H) -> Densa W2,b2 (Sigmoide) -> y_hat (N, 1)
```

- `N`: número de muestras del batch (full-batch, todo el train set por época).
- `H`: número de neuronas ocultas (`hidden_units`).
- `Z1 = X·W1 + b1` (pre-activación oculta), `A1 = relu(Z1)`.
- `Z2 = A1·W2 + b2` (pre-activación de salida), `y_hat = sigmoid(Z2)`.
- Pérdida: Binary Cross-Entropy, `L = -1/N * Σ [y·log(y_hat) + (1-y)·log(1-y_hat)]`.

## 1. Gradiente de la capa de salida (BCE + sigmoide combinados)

La derivada de BCE respecto a `y_hat`, multiplicada por la derivada de la
sigmoide (`sigmoid' = y_hat·(1-y_hat)`), se simplifica algebraicamente a:

```
dZ2 = y_hat - y
```

Esta cancelación es la razón por la que se implementa BCE+sigmoide como un
solo gradiente combinado, en vez de derivar cada función por separado
(más estable numéricamente, evita dividir por `y_hat·(1-y_hat)` cerca de 0 o 1).

- **Kernel:** `output_gradient_kernel` (`train_gpu.cu:335`) — un hilo por muestra.

## 2. Gradientes de la capa de salida (W2, b2)

```
dW2 = A1^T · dZ2      # (H, 1)
db2 = Σ_n dZ2[n]      # (1,)
```

- **Kernel `dW2`:** `matmul_atb` (`train_gpu.cu:294`) — GEMM transpuesto `A1^T · dZ2`,
  un hilo por elemento de salida (no usa memoria compartida; no es el foco del
  experimento de `block_size`, que se centra en el forward).
- **Kernel `db2`:** `bias_gradient_kernel` (`train_gpu.cu:353`) — cada hilo reduce
  una columna sumando sobre las `N` muestras.

## 3. Gradiente de la capa oculta (regla de la cadena + ReLU')

```
dA1 = dZ2 · W2^T              # (N, H), se expande el escalar de salida a H columnas
dZ1 = dA1 ⊙ relu'(Z1)         # relu'(z) = 1 si z > 0, si no 0
```

- **Kernel:** `hidden_gradient_kernel` (`train_gpu.cu:342`). Cada hilo calcula un
  elemento `(n, h)`: multiplica `dZ2[n] * W2[h]` y aplica la máscara ReLU usando
  el signo de `Z1[n,h]` (guardado del forward, sin recalcularlo).

## 4. Gradientes de la capa de entrada (W1, b1)

```
dW1 = X^T · dZ1        # (4096, H)
db1 = Σ_n dZ1[n, :]    # (H,)
```

- **Kernel `dW1`:** `matmul_atb` (mismo kernel que para `dW2`, reutilizado con
  otras dimensiones: `X^T (4096,N) · dZ1 (N,H)`).
- **Kernel `db1`:** `bias_gradient_kernel` (mismo kernel que para `db2`).

## 5. Actualización de pesos (SGD full-batch, sin momentum)

```
θ ← θ - lr · (dθ / N)      para θ ∈ {W1, b1, W2, b2}
```

- **Kernel:** `sgd_update_kernel` (`train_gpu.cu:364`). Se invoca 4 veces por
  época (una por tensor de parámetros), pasando `inv_n = 1/N` precalculado para
  no dividir dentro del kernel.

## Orden de ejecución por época (resumen)

```
forward_pass()                          -> Z1, A1, Z2, y_hat
bce_loss_kernel                         -> pérdida (solo para logging)
output_gradient_kernel                  -> dZ2
matmul_atb (A1^T·dZ2)                   -> dW2
bias_gradient_kernel (dZ2)              -> db2
hidden_gradient_kernel (dZ2,W2,Z1)      -> dZ1
matmul_atb (X^T·dZ1)                    -> dW1
bias_gradient_kernel (dZ1)              -> db1
sgd_update_kernel x4                    -> W1, b1, W2, b2 actualizados
```

Todo el código vive en `train_gpu.cu`; este documento solo explica el "por qué"
matemático detrás de cada kernel.
