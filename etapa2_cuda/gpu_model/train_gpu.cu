/*
 * Entrenamiento en GPU (CUDA puro) de la MLP de la Etapa 2.
 *
 * Arquitectura:  Entrada(4096) -> Densa(hidden_units, ReLU) -> Densa(1, Sigmoide)
 * Perdida:       Binary Cross-Entropy
 * Optimizador:   SGD full-batch (peso -= tasa_aprendizaje * gradiente / N)
 *
 * Kernels implementados:
 *   1. matmul_ab_tiled     -> GEMM con memoria compartida, block_size configurable en runtime
 *   2. matmul_atb          -> GEMM transpuesto A^T*B, usado para los gradientes de pesos
 *   3. bias_relu_forward   -> suma de bias + ReLU (capa oculta)
 *   4. bias_sigmoid_forward-> suma de bias + sigmoide (capa de salida)
 *   5. bce_loss_kernel     -> perdida BCE (reduccion con atomicAdd)
 *   6. output_gradient_kernel -> dZ2 = y_hat - y
 *   7. hidden_gradient_kernel -> dZ1 = (dZ2 @ W2^T) * relu'(Z1)
 *   8. bias_gradient_kernel   -> db = suma sobre N de dZ
 *   9. sgd_update_kernel      -> actualizacion de pesos
 *
 * Compilar (Colab, runtime con GPU):
 *   nvcc -O3 -o train_gpu train_gpu.cu
 *
 * Uso:
 *   ./train_gpu <train_csv> <val_csv> <hidden_units> <learning_rate> <epochs> <block_size> <weights_out>
 *
 * Ejemplo:
 *   ./train_gpu ../../dataset/procesado/train.csv ../../dataset/procesado/val.csv 128 0.1 50 32 weights_gpu.bin
 *
 * Formato de salida en stdout (pensado para ser parseado desde el notebook orquestador):
 *   INFO,<clave>,<valor>
 *   TRANSFER,<segundos_host_a_device>
 *   EPOCH,<epoca>,<train_loss>,<val_loss>,<val_accuracy>
 *   SUMMARY,<hidden_units>,<learning_rate>,<epochs>,<block_size>,<train_loop_seconds>,<h2d_seconds>,<final_train_loss>,<final_val_loss>,<final_val_accuracy>
 *
 * Formato del archivo de pesos (binario, leido luego con numpy.fromfile):
 *   int32  input_dim
 *   int32  hidden_units
 *   float32[input_dim * hidden_units]  W1  (fila = neurona de entrada, columna = neurona oculta)
 *   float32[hidden_units]              b1
 *   float32[hidden_units]              W2  (vector, una neurona de salida)
 *   float32[1]                         b2
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include <cuda_runtime.h>

#define INPUT_DIM 4096
#define MAX_LINE_LEN 65536
#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t cuda_check_err = (call);                                  \
        if (cuda_check_err != cudaSuccess) {                                  \
            fprintf(stderr, "ERROR CUDA en %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(cuda_check_err));                      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

typedef struct {
    float *x;
    float *y;
    int n_samples;
} dataset_t;

/* ------------------------------------------------------------------------ */
/* Carga de datos (CSV: label,pixel_0,...,pixel_4095)                       */
/* ------------------------------------------------------------------------ */

static int count_data_rows(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }

    int newline_count = 0;
    int has_content = 0;
    int c;

    while ((c = fgetc(fp)) != EOF) {
        has_content = 1;
        if (c == '\n') {
            newline_count++;
        }
    }

    if (has_content) {
        fseek(fp, -1, SEEK_END);
        if (fgetc(fp) != '\n') {
            newline_count++;
        }
    }

    fclose(fp);
    return newline_count - 1; /* resta la fila de encabezado */
}

static int load_dataset(const char *path, dataset_t *out)
{
    int n_rows = count_data_rows(path);
    if (n_rows <= 0) {
        fprintf(stderr, "ERROR: no se pudo determinar el numero de filas de '%s'\n", path);
        return 0;
    }

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "ERROR: no se pudo abrir '%s'\n", path);
        return 0;
    }

    float *x = (float *)malloc((size_t)n_rows * INPUT_DIM * sizeof(float));
    float *y = (float *)malloc((size_t)n_rows * sizeof(float));
    char *line = (char *)malloc(MAX_LINE_LEN);

    if (x == NULL || y == NULL || line == NULL) {
        fprintf(stderr, "ERROR: memoria insuficiente para cargar '%s'\n", path);
        fclose(fp);
        free(x);
        free(y);
        free(line);
        return 0;
    }

    if (fgets(line, MAX_LINE_LEN, fp) == NULL) {
        fprintf(stderr, "ERROR: archivo vacio '%s'\n", path);
        fclose(fp);
        free(x);
        free(y);
        free(line);
        return 0;
    }

    for (int row = 0; row < n_rows; row++) {
        if (fgets(line, MAX_LINE_LEN, fp) == NULL) {
            fprintf(stderr, "ERROR: fin de archivo inesperado en '%s' (fila %d)\n", path, row);
            fclose(fp);
            free(x);
            free(y);
            free(line);
            return 0;
        }

        char *token = strtok(line, ",");
        if (token == NULL) {
            fprintf(stderr, "ERROR: fila %d malformada en '%s'\n", row, path);
            fclose(fp);
            free(x);
            free(y);
            free(line);
            return 0;
        }
        y[row] = strtof(token, NULL);

        for (int col = 0; col < INPUT_DIM; col++) {
            token = strtok(NULL, ",");
            if (token == NULL) {
                fprintf(stderr, "ERROR: falta columna %d en fila %d de '%s'\n", col, row, path);
                fclose(fp);
                free(x);
                free(y);
                free(line);
                return 0;
            }
            x[(size_t)row * INPUT_DIM + col] = strtof(token, NULL);
        }
    }

    free(line);
    fclose(fp);

    out->x = x;
    out->y = y;
    out->n_samples = n_rows;
    return 1;
}

static void free_dataset(dataset_t *d)
{
    free(d->x);
    free(d->y);
    d->x = NULL;
    d->y = NULL;
    d->n_samples = 0;
}

/* ------------------------------------------------------------------------ */
/* Inicializacion de pesos (Glorot/Xavier uniforme, semilla fija)           */
/* ------------------------------------------------------------------------ */

static float rand_uniform(float limit)
{
    float r = (float)rand() / (float)RAND_MAX;
    return (r * 2.0f - 1.0f) * limit;
}

static void init_glorot_uniform(float *data, int fan_in, int fan_out)
{
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    size_t total = (size_t)fan_in * (size_t)fan_out;
    for (size_t i = 0; i < total; i++) {
        data[i] = rand_uniform(limit);
    }
}

/* ------------------------------------------------------------------------ */
/* Utilidades de tiempo y de E/S de pesos                                   */
/* ------------------------------------------------------------------------ */

static double elapsed_seconds(struct timespec start, struct timespec end)
{
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

static int save_weights(const char *path, const float *w1, const float *b1,
                         const float *w2, const float *b2,
                         int input_dim, int hidden_units)
{
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        return 0;
    }

    int32_t dims[2];
    dims[0] = (int32_t)input_dim;
    dims[1] = (int32_t)hidden_units;

    fwrite(dims, sizeof(int32_t), 2, fp);
    fwrite(w1, sizeof(float), (size_t)input_dim * (size_t)hidden_units, fp);
    fwrite(b1, sizeof(float), (size_t)hidden_units, fp);
    fwrite(w2, sizeof(float), (size_t)hidden_units, fp);
    fwrite(b2, sizeof(float), 1, fp);

    fclose(fp);
    return 1;
}

/* ------------------------------------------------------------------------ */
/* Kernels CUDA                                                             */
/* ------------------------------------------------------------------------ */

/*
 * C(n,k) = A(n,m) @ B(m,k)
 * GEMM con tiles en memoria compartida dinamica; block_size se decide en
 * tiempo de ejecucion (16x16, 32x32, ...) para poder medir su efecto sin
 * recompilar el kernel.
 */
__global__ void matmul_ab_tiled(const float *a, const float *b, float *c,
                                 int n, int m, int k, int block_size)
{
    extern __shared__ float shared_mem[];
    float *tile_a = shared_mem;
    float *tile_b = shared_mem + block_size * block_size;

    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (m + block_size - 1) / block_size;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * block_size + threadIdx.x;
        int b_row = t * block_size + threadIdx.y;

        tile_a[threadIdx.y * block_size + threadIdx.x] =
            (row < n && a_col < m) ? a[(size_t)row * m + a_col] : 0.0f;
        tile_b[threadIdx.y * block_size + threadIdx.x] =
            (b_row < m && col < k) ? b[(size_t)b_row * k + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < block_size; i++) {
            acc += tile_a[threadIdx.y * block_size + i] * tile_b[i * block_size + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < k) {
        c[(size_t)row * k + col] = acc;
    }
}

/*
 * C(m,k) = A^T(m,n) @ B(n,k), con A almacenada como (n,m) y B como (n,k).
 * Usado para los gradientes de pesos (dW1, dW2). Un hilo por elemento de
 * salida; no usa memoria compartida (la operacion no es el foco del
 * experimento de block_size, que se centra en el forward de la capa densa).
 */
__global__ void matmul_atb(const float *a, const float *b, float *c,
                            int n, int m, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; /* indice en m */
    int col = blockIdx.x * blockDim.x + threadIdx.x; /* indice en k */

    if (row < m && col < k) {
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            acc += a[(size_t)i * m + row] * b[(size_t)i * k + col];
        }
        c[(size_t)row * k + col] = acc;
    }
}

__global__ void bias_relu_forward(const float *z, const float *bias, float *a,
                                   int n, int h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * h;
    if (idx < total) {
        int col = idx % h;
        float val = z[idx] + bias[col];
        a[idx] = val > 0.0f ? val : 0.0f;
    }
}

__global__ void bias_sigmoid_forward(const float *z, const float *bias, float *y_hat,
                                      int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = z[idx] + bias[0];
        y_hat[idx] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void bce_loss_kernel(const float *y_hat, const float *y, float *loss_accum,
                                 int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float eps = 1e-7f;
        float p = fminf(fmaxf(y_hat[idx], eps), 1.0f - eps);
        float sample_loss = -(y[idx] * logf(p) + (1.0f - y[idx]) * logf(1.0f - p));
        atomicAdd(loss_accum, sample_loss);
    }
}

__global__ void output_gradient_kernel(const float *y_hat, const float *y, float *d_z2,
                                        int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_z2[idx] = y_hat[idx] - y[idx];
    }
}

__global__ void hidden_gradient_kernel(const float *d_z2, const float *w2, const float *z1,
                                        float *d_z1, int n, int h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * h;
    if (idx < total) {
        int row = idx / h;
        int col = idx % h;
        float d_a1 = d_z2[row] * w2[col];
        d_z1[idx] = z1[idx] > 0.0f ? d_a1 : 0.0f;
    }
}

__global__ void bias_gradient_kernel(const float *d_z, float *d_bias, int n, int dim)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < dim) {
        float acc = 0.0f;
        for (int row = 0; row < n; row++) {
            acc += d_z[(size_t)row * dim + col];
        }
        d_bias[col] = acc;
    }
}

__global__ void sgd_update_kernel(float *param, const float *grad, float lr, float inv_n,
                                   int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        param[idx] -= lr * grad[idx] * inv_n;
    }
}

/* ------------------------------------------------------------------------ */
/* Forward pass (orquesta los kernels para un conjunto de N muestras)       */
/* ------------------------------------------------------------------------ */

static void forward_pass(const float *x_d, const float *w1_d, const float *b1_d,
                          const float *w2_d, const float *b2_d,
                          float *z1_d, float *a1_d, float *z2_d, float *y_hat_d,
                          int n_samples, int input_dim, int hidden_units, int block_size)
{
    dim3 block_mm(block_size, block_size);
    size_t shared_bytes = 2 * (size_t)block_size * (size_t)block_size * sizeof(float);

    dim3 grid_mm1((unsigned int)((hidden_units + block_size - 1) / block_size),
                  (unsigned int)((n_samples + block_size - 1) / block_size));
    matmul_ab_tiled<<<grid_mm1, block_mm, shared_bytes>>>(
        x_d, w1_d, z1_d, n_samples, input_dim, hidden_units, block_size);

    int blocks_hidden = (n_samples * hidden_units + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    bias_relu_forward<<<blocks_hidden, THREADS_PER_BLOCK>>>(z1_d, b1_d, a1_d, n_samples, hidden_units);

    dim3 grid_mm2((unsigned int)((1 + block_size - 1) / block_size),
                  (unsigned int)((n_samples + block_size - 1) / block_size));
    matmul_ab_tiled<<<grid_mm2, block_mm, shared_bytes>>>(
        a1_d, w2_d, z2_d, n_samples, hidden_units, 1, block_size);

    int blocks_out = (n_samples + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    bias_sigmoid_forward<<<blocks_out, THREADS_PER_BLOCK>>>(z2_d, b2_d, y_hat_d, n_samples);
}

/* ------------------------------------------------------------------------ */
/* Programa principal                                                       */
/* ------------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    if (argc != 8) {
        fprintf(stderr,
                "Uso: %s <train_csv> <val_csv> <hidden_units> <learning_rate> "
                "<epochs> <block_size> <weights_out>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *train_csv = argv[1];
    const char *val_csv = argv[2];
    int hidden_units = atoi(argv[3]);
    float learning_rate = strtof(argv[4], NULL);
    int epochs = atoi(argv[5]);
    int block_size = atoi(argv[6]);
    const char *weights_out = argv[7];

    if (hidden_units <= 0 || epochs <= 0) {
        fprintf(stderr, "ERROR: hidden_units y epochs deben ser positivos\n");
        return EXIT_FAILURE;
    }
    if (block_size <= 0 || block_size > 32) {
        fprintf(stderr, "ERROR: block_size debe estar entre 1 y 32 (block_size^2 <= 1024 hilos)\n");
        return EXIT_FAILURE;
    }

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0) {
        fprintf(stderr, "ERROR: no se detecto ninguna GPU con soporte CUDA\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("INFO,gpu_name,%s\n", prop.name);

    srand(42);

    dataset_t train_set, val_set;
    if (!load_dataset(train_csv, &train_set)) {
        return EXIT_FAILURE;
    }
    if (!load_dataset(val_csv, &val_set)) {
        free_dataset(&train_set);
        return EXIT_FAILURE;
    }

    printf("INFO,train_samples,%d\n", train_set.n_samples);
    printf("INFO,val_samples,%d\n", val_set.n_samples);

    int n_train = train_set.n_samples;
    int n_val = val_set.n_samples;
    int d = INPUT_DIM;
    int h = hidden_units;

    float *w1_h = (float *)malloc((size_t)d * h * sizeof(float));
    float *b1_h = (float *)calloc((size_t)h, sizeof(float));
    float *w2_h = (float *)malloc((size_t)h * sizeof(float));
    float *b2_h = (float *)calloc(1, sizeof(float));

    init_glorot_uniform(w1_h, d, h);
    init_glorot_uniform(w2_h, h, 1);

    float *x_train_d, *y_train_d, *x_val_d, *y_val_d;
    float *w1_d, *b1_d, *w2_d, *b2_d;
    float *z1_train_d, *a1_train_d, *z2_train_d, *yhat_train_d;
    float *z1_val_d, *a1_val_d, *z2_val_d, *yhat_val_d;
    float *d_z2_d, *d_z1_d, *dw1_d, *db1_d, *dw2_d, *db2_d;
    float *loss_d;

    CUDA_CHECK(cudaMalloc(&x_train_d, (size_t)n_train * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_train_d, (size_t)n_train * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&x_val_d, (size_t)n_val * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_val_d, (size_t)n_val * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&w1_d, (size_t)d * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b1_d, (size_t)h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w2_d, (size_t)h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b2_d, sizeof(float)));

    CUDA_CHECK(cudaMalloc(&z1_train_d, (size_t)n_train * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a1_train_d, (size_t)n_train * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&z2_train_d, (size_t)n_train * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&yhat_train_d, (size_t)n_train * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&z1_val_d, (size_t)n_val * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a1_val_d, (size_t)n_val * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&z2_val_d, (size_t)n_val * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&yhat_val_d, (size_t)n_val * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_z2_d, (size_t)n_train * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z1_d, (size_t)n_train * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw1_d, (size_t)d * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db1_d, (size_t)h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw2_d, (size_t)h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db2_d, sizeof(float)));

    CUDA_CHECK(cudaMalloc(&loss_d, sizeof(float)));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    CUDA_CHECK(cudaMemcpy(x_train_d, train_set.x, (size_t)n_train * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_train_d, train_set.y, (size_t)n_train * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_val_d, val_set.x, (size_t)n_val * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_val_d, val_set.y, (size_t)n_val * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w1_d, w1_h, (size_t)d * h * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b1_d, b1_h, (size_t)h * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w2_d, w2_h, (size_t)h * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b2_d, b2_h, sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double h2d_seconds = elapsed_seconds(t0, t1);
    printf("TRANSFER,%.6f\n", h2d_seconds);

    dim3 block_atb(16, 16);
    float inv_n = 1.0f / (float)n_train;

    float final_train_loss = 0.0f;
    float final_val_loss = 0.0f;
    float final_val_accuracy = 0.0f;

    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int epoch = 0; epoch < epochs; epoch++) {
        forward_pass(x_train_d, w1_d, b1_d, w2_d, b2_d,
                     z1_train_d, a1_train_d, z2_train_d, yhat_train_d,
                     n_train, d, h, block_size);

        CUDA_CHECK(cudaMemset(loss_d, 0, sizeof(float)));
        int blocks_loss = (n_train + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        bce_loss_kernel<<<blocks_loss, THREADS_PER_BLOCK>>>(yhat_train_d, y_train_d, loss_d, n_train);

        float train_loss_sum = 0.0f;
        CUDA_CHECK(cudaMemcpy(&train_loss_sum, loss_d, sizeof(float), cudaMemcpyDeviceToHost));
        float train_loss = train_loss_sum / (float)n_train;

        int blocks_n = (n_train + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        output_gradient_kernel<<<blocks_n, THREADS_PER_BLOCK>>>(yhat_train_d, y_train_d, d_z2_d, n_train);

        dim3 grid_dw2((unsigned int)((1 + 15) / 16), (unsigned int)((h + 15) / 16));
        matmul_atb<<<grid_dw2, block_atb>>>(a1_train_d, d_z2_d, dw2_d, n_train, h, 1);

        int blocks_db2 = (1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        bias_gradient_kernel<<<blocks_db2, THREADS_PER_BLOCK>>>(d_z2_d, db2_d, n_train, 1);

        int blocks_hidden_grad = (n_train * h + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        hidden_gradient_kernel<<<blocks_hidden_grad, THREADS_PER_BLOCK>>>(d_z2_d, w2_d, z1_train_d, d_z1_d, n_train, h);

        dim3 grid_dw1((unsigned int)((h + 15) / 16), (unsigned int)((d + 15) / 16));
        matmul_atb<<<grid_dw1, block_atb>>>(x_train_d, d_z1_d, dw1_d, n_train, d, h);

        int blocks_db1 = (h + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        bias_gradient_kernel<<<blocks_db1, THREADS_PER_BLOCK>>>(d_z1_d, db1_d, n_train, h);

        int blocks_w1 = (d * h + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sgd_update_kernel<<<blocks_w1, THREADS_PER_BLOCK>>>(w1_d, dw1_d, learning_rate, inv_n, d * h);
        int blocks_b1 = (h + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sgd_update_kernel<<<blocks_b1, THREADS_PER_BLOCK>>>(b1_d, db1_d, learning_rate, inv_n, h);
        int blocks_w2 = (h + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sgd_update_kernel<<<blocks_w2, THREADS_PER_BLOCK>>>(w2_d, dw2_d, learning_rate, inv_n, h);
        sgd_update_kernel<<<1, 1>>>(b2_d, db2_d, learning_rate, inv_n, 1);

        forward_pass(x_val_d, w1_d, b1_d, w2_d, b2_d,
                     z1_val_d, a1_val_d, z2_val_d, yhat_val_d,
                     n_val, d, h, block_size);

        CUDA_CHECK(cudaMemset(loss_d, 0, sizeof(float)));
        int blocks_val_loss = (n_val + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        bce_loss_kernel<<<blocks_val_loss, THREADS_PER_BLOCK>>>(yhat_val_d, y_val_d, loss_d, n_val);

        float val_loss_sum = 0.0f;
        CUDA_CHECK(cudaMemcpy(&val_loss_sum, loss_d, sizeof(float), cudaMemcpyDeviceToHost));
        float val_loss = val_loss_sum / (float)n_val;

        float *yhat_val_h = (float *)malloc((size_t)n_val * sizeof(float));
        CUDA_CHECK(cudaMemcpy(yhat_val_h, yhat_val_d, (size_t)n_val * sizeof(float), cudaMemcpyDeviceToHost));

        int correct = 0;
        for (int i = 0; i < n_val; i++) {
            int pred = yhat_val_h[i] >= 0.5f ? 1 : 0;
            int actual = val_set.y[i] >= 0.5f ? 1 : 0;
            if (pred == actual) {
                correct++;
            }
        }
        float val_accuracy = (float)correct / (float)n_val;
        free(yhat_val_h);

        final_train_loss = train_loss;
        final_val_loss = val_loss;
        final_val_accuracy = val_accuracy;

        printf("EPOCH,%d,%.6f,%.6f,%.6f\n", epoch, train_loss, val_loss, val_accuracy);
        fflush(stdout);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double train_loop_seconds = elapsed_seconds(t0, t1);

    CUDA_CHECK(cudaMemcpy(w1_h, w1_d, (size_t)d * h * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b1_h, b1_d, (size_t)h * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w2_h, w2_d, (size_t)h * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b2_h, b2_d, sizeof(float), cudaMemcpyDeviceToHost));

    if (!save_weights(weights_out, w1_h, b1_h, w2_h, b2_h, d, h)) {
        fprintf(stderr, "ERROR: no se pudieron guardar los pesos en '%s'\n", weights_out);
    }

    printf("SUMMARY,%d,%g,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
           hidden_units, learning_rate, epochs, block_size,
           train_loop_seconds, h2d_seconds,
           final_train_loss, final_val_loss, final_val_accuracy);

    cudaFree(x_train_d);
    cudaFree(y_train_d);
    cudaFree(x_val_d);
    cudaFree(y_val_d);
    cudaFree(w1_d);
    cudaFree(b1_d);
    cudaFree(w2_d);
    cudaFree(b2_d);
    cudaFree(z1_train_d);
    cudaFree(a1_train_d);
    cudaFree(z2_train_d);
    cudaFree(yhat_train_d);
    cudaFree(z1_val_d);
    cudaFree(a1_val_d);
    cudaFree(z2_val_d);
    cudaFree(yhat_val_d);
    cudaFree(d_z2_d);
    cudaFree(d_z1_d);
    cudaFree(dw1_d);
    cudaFree(db1_d);
    cudaFree(dw2_d);
    cudaFree(db2_d);
    cudaFree(loss_d);

    free(w1_h);
    free(b1_h);
    free(w2_h);
    free(b2_h);
    free_dataset(&train_set);
    free_dataset(&val_set);

    return EXIT_SUCCESS;
}
