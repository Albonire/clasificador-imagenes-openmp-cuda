# CRISP-DM Phase 4: Modeling

## Objective

The objective of this phase is to define, train, and document the machine learning model used to classify eye states from the preprocessed image features produced in Phase 3. The model must learn to distinguish between alert (class 0: eyes open) and drowsy (class 1: eyes closed) states from a fixed-length feature vector, while remaining simple enough to implement and accelerate with custom CUDA kernels.

This phase covers model selection, network architecture, parameter count, activation functions, loss function, optimizer, hyperparameter search, and the final training configuration used to produce `modelo/weights.npz`.

---

## Model Selection

A **feedforward multilayer perceptron (MLP)** was chosen for this project because:

- The Stage 1 preprocessing pipeline already extracts a structured numerical representation (4096 normalized pixel values per image).
- A shallow fully connected network is sufficient for binary classification on this feature space.
- The architecture can be implemented from scratch in C/CUDA without external deep-learning frameworks, which aligns with the academic goal of demonstrating parallel computing techniques.
- The same model definition is shared between the CPU baseline, the CUDA trainer, and the Streamlit inference application.

---

## Network Architecture

The final classifier is a two-layer MLP for **binary classification**:

```text
Input (4096) → Dense (75, ReLU) → Dense (1, Sigmoid) → ŷ ∈ (0, 1)
```

### Layer-by-Layer Description

| Layer | Type | Input Size | Output Size | Activation | Role |
|-------|------|------------|-------------|------------|------|
| 0 | Input | — | 4096 | — | Flattened preprocessed image |
| 1 | Fully connected | 4096 | 75 | ReLU | Hidden representation |
| 2 | Fully connected | 75 | 1 | Sigmoid | Probability of class 1 |

**Input features:** Each sample is a 4096-dimensional vector obtained after grayscale conversion, Sobel and Gaussian filtering, resizing to 64×64, normalization to [0, 1], and flattening (see Phase 3).

**Output:** A single scalar \(\hat{y}\) interpreted as the probability that the image belongs to class 1 (eyes closed / drowsiness).

**Decision rule:** Predict class 1 if \(\hat{y} \geq 0.5\); otherwise predict class 0.

### Forward Pass

The forward pass is identical in training (`etapa2_cuda/gpu_model/train_gpu.cu`) and inference (`app_streamlit/core/inference.py`):

\[
\mathbf{z}_1 = \mathbf{x}\mathbf{W}_1^\top + \mathbf{b}_1,\quad \mathbf{a}_1 = \mathrm{ReLU}(\mathbf{z}_1)
\]

\[
z_2 = \mathbf{a}_1 \mathbf{W}_2^\top + b_2,\quad \hat{y} = \sigma(z_2) = \frac{1}{1 + e^{-z_2}}
\]

Where:

- \(\mathbf{x}\) is the input feature vector of shape (4096,).
- \(\mathbf{W}_1 \in \mathbb{R}^{75 \times 4096}\), \(\mathbf{b}_1 \in \mathbb{R}^{75}\).
- \(\mathbf{W}_2 \in \mathbb{R}^{1 \times 75}\), \(b_2 \in \mathbb{R}\).

### Activation Functions

| Layer | Activation | Definition | Purpose |
|-------|------------|------------|---------|
| Hidden | ReLU | \(\max(0, z)\) | Introduces non-linearity while keeping gradients simple for backpropagation |
| Output | Sigmoid | \(1 / (1 + e^{-z})\) | Maps the logit to a probability in (0, 1) for binary classification |

---

## Parameter Count

With `hidden_units = 75`, the model has **307,351 trainable parameters**:

| Parameter | Shape | Count |
|-----------|-------|------:|
| \(\mathbf{W}_1\) | (75, 4096) | 307,200 |
| \(\mathbf{b}_1\) | (75,) | 75 |
| \(\mathbf{W}_2\) | (1, 75) | 75 |
| \(b_2\) | (1,) | 1 |
| **Total** | | **307,351** |

### Weight Initialization

| Parameter | Initialization |
|-----------|----------------|
| \(\mathbf{W}_1\) | Glorot/Xavier uniform: limit \(= \sqrt{6 / (\text{fan\_in} + \text{fan\_out})}\) |
| \(\mathbf{W}_2\) | Glorot/Xavier uniform (output layer; no ReLU follows) |
| \(\mathbf{b}_1\), \(b_2\) | Zeros |

Weights are initialized with a fixed random seed to ensure reproducibility across CPU and GPU runs.

---

## Loss Function

The model is trained by minimizing **Binary Cross-Entropy (BCE)**, averaged over the training set:

\[
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]
\]

Predictions are clipped to \([10^{-7}, 1 - 10^{-7}]\) during loss computation to avoid numerical instability when \(\hat{y}_i\) is exactly 0 or 1.

The output-layer gradient simplifies to \(\partial \mathcal{L} / \partial z_2 = \hat{y} - y\), which is used directly in the manual backpropagation kernels.

---

## Training Procedure

### Optimizer

**Stochastic Gradient Descent (SGD)**, implemented manually without external frameworks:

\[
\theta \leftarrow \theta - \eta \cdot \frac{1}{N}\nabla_\theta \mathcal{L}
\]

Where \(\eta\) is the learning rate and \(N\) is the number of training samples.

### Training Mode

- **Full-batch:** All training samples are used in each epoch (no mini-batch shuffling).
- **Momentum:** Disabled (default 0.0).
- **L2 regularization:** Disabled (default 0.0).
- **Learning-rate decay:** Disabled (default factor 1.0 per epoch).

### Training and Validation Data

| Split | File | Samples |
|-------|------|--------:|
| Train | `dataset/procesado/train.csv` | 1,646 |
| Validation | `dataset/procesado/val.csv` | 353 |

The validation set was used exclusively for hyperparameter selection. The test set was held out until Phase 5.

---

## Hyperparameter Search

A grid search was performed on the GPU implementation (`etapa2_cuda/gpu_model/model-gpu.ipynb`) to select the best combination of architecture size, learning rate, and number of epochs. The search space was:

| Hyperparameter | Values explored |
|----------------|-----------------|
| Hidden units | 64, 75, 78, 128 |
| Learning rate | 0.01, 0.08, 0.1 |
| Epochs | 20, 40, 45, 50 |

All runs used `block_size = 32` for the tiled matrix-multiplication CUDA kernels. This parameter affects GPU kernel performance only; it does not change the model architecture.

### Best Configuration (Selected on Validation Set)

| Hyperparameter | Value |
|----------------|-------|
| Hidden units | **75** |
| Learning rate | **0.1** |
| Epochs | **50** |
| Validation loss | 0.5402 |
| Validation accuracy | 0.8272 |

This configuration was used for the final GPU training run documented in `etapa2_cuda/gpu_model/gpu_report.md`.

---

## Implementation

### CUDA Training (`etapa2_cuda/gpu_model/train_gpu.cu`)

The GPU trainer implements the full forward and backward passes with custom CUDA kernels:

| Kernel | Purpose |
|--------|---------|
| `matmul_ab_tiled` | Forward matrix multiplication with shared memory |
| `matmul_atb` | Transposed multiplication for weight gradients |
| `bias_relu_forward` | Bias addition + ReLU (hidden layer) |
| `bias_sigmoid_forward` | Bias addition + sigmoid (output layer) |
| `bce_loss_kernel` | Binary cross-entropy loss |
| `output_gradient_kernel` | Output-layer error signal |
| `hidden_gradient_kernel` | Hidden-layer backpropagation through ReLU |
| `bias_gradient_kernel` | Bias gradient accumulation |
| `sgd_update_kernel` | Weight update step |

Training was executed on a **Tesla T4** GPU. The final training loop completed in approximately **0.30 s** for 50 epochs (full-batch), compared to **3.17 s** for the CPU baseline on the same architecture class.

### Weight Export

Trained weights are saved in binary format by `train_gpu.cu` (`weights_gpu.bin`) and converted to NumPy format for inference:

```bash
python3 modelo/export_weights.py \
    --input etapa2_cuda/gpu_model/weights_gpu.bin \
    --output modelo/weights.npz
```

The exported `modelo/weights.npz` contains keys `W1`, `b1`, `W2`, `b2` with shapes documented in `modelo/README.md`. This file is consumed by the evaluation script (`scripts/evaluate_final_metrics.py`) and the Streamlit application (`app_streamlit/`).

---

## Summary

| Aspect | Final choice |
|--------|--------------|
| Model type | Feedforward MLP (2 dense layers) |
| Architecture | 4096 → 75 (ReLU) → 1 (Sigmoid) |
| Parameters | 307,351 |
| Loss | Binary cross-entropy |
| Optimizer | SGD (full-batch, lr = 0.1) |
| Epochs | 50 |
| Decision threshold | 0.5 |
| Output weights | `modelo/weights.npz` |

---

## Conclusions

The modeling phase produced a compact MLP capable of learning discriminative patterns from the 4096-dimensional preprocessed features. Hyperparameter search on the validation set identified 75 hidden units with a learning rate of 0.1 and 50 epochs as the best trade-off between validation loss and accuracy. The manual CUDA implementation enabled fast experimentation and achieved a speedup of approximately **10.4×** over the CPU baseline while preserving the same mathematical model.

The trained weights were exported to a reusable format for evaluation (Phase 5) and deployment in the Streamlit inference application. The next phase evaluates generalization on the held-out test set and reflects on the complete pipeline.
