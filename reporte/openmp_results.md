# OpenMP Preprocessing Results

This section documents the performance evaluation of the Stage 1 preprocessing pipeline (`etapa1_openmp/preprocess_serial.c`) on **3,312 images** (classes `clase_0` and `clase_1`). Measurements were taken on the same machine and dataset for each thread count, varying only `OMP_NUM_THREADS`.

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Source program | `etapa1_openmp/preprocess_serial.c` |
| Compiler flags | `gcc -Wall -Wextra -O2 -fopenmp` |
| Parallel region | `#pragma omp parallel for schedule(dynamic)` over the image loop |
| Serial baseline | `OMP_NUM_THREADS=1` |
| Thread counts tested | 1, 2, 4, 8 |
| Dataset size | 3,312 raw images |
| Output | `dataset/procesado/dataset_serial.csv` (4,096 features + label per row) |

The timer uses `omp_get_wtime()` around the full run: CSV creation, both class directories, and file close.

---

## Speedup Table

Speedup is defined as:

\[
S(p) = \frac{T_1}{T_p}
\]

where \(T_1\) is the elapsed time with one thread and \(T_p\) is the time with \(p\) threads.

| Threads | Time (s) | Speedup \(S(p)\) | Efficiency \(S(p)/p\) |
|--------:|---------:|-----------------:|----------------------:|
| 1 | 27.262 | 1.00 | 1.00 |
| 2 | 17.325 | 1.57 | 0.79 |
| 4 | 14.155 | 1.93 | 0.48 |
| 8 | 13.449 | 2.03 | 0.25 |

### Interpretation

- Parallelism reduces total runtime for every configuration tested.
- The best result is **2.03×** at 8 threads.
- Gains are **sublinear**: doubling threads does not double performance.
- Efficiency drops as thread count grows, which indicates increasing overhead and contention.

---

## Speedup Graph

![OpenMP speedup table and curve](evidencias/openmp_speedup_table.png)

The left panel reproduces the measurement table; the right panel compares observed speedup against the Amdahl model fitted from the same data.

A complementary decomposition plot is available in `evidencias/openmp_amdahl_analysis.png`.

To regenerate the figures:

```bash
python scripts/plot_openmp_speedup.py
```

---

## Amdahl's Law Analysis

### Model

Amdahl's law states that if a program has a serial fraction \(s\) and a parallel fraction \(1-s\), the speedup with \(p\) processors is:

\[
S(p) = \frac{1}{s + \dfrac{1-s}{p}}
\]

Equivalently, the expected runtime is:

\[
T(p) = s\,T_1 + \frac{(1-s)\,T_1}{p}
\]

The maximum achievable speedup as \(p \to \infty\) is:

\[
S_{\max} = \frac{1}{s}
\]

### Serial Bottlenecks in This Pipeline

Even though the main image loop is parallelized, several stages remain effectively serial or introduce synchronization:

1. **Directory traversal** — `collect_image_paths()` runs before the parallel loop for each class.
2. **CSV header and file setup** — executed once in `main()` before processing.
3. **Critical CSV writes** — each processed image enters `#pragma omp critical(csv_write)`, so rows are appended one at a time to a single file.
4. **Error logging** — warnings use `#pragma omp critical(stderr_write)`.
5. **Two class passes** — `clase_0` and `clase_1` are processed sequentially.

The dominant limiter at higher thread counts is the **serialized CSV export**: threads finish image processing at different rates (`schedule(dynamic)`), but only one thread can write at a time. That behavior increases lock contention and explains why efficiency falls to **25%** with 8 threads.

### Fitted Parameters

Fitting the linear model \(T(p) = sT_1 + (1-s)T_1/p\) to the four measurements yields:

| Quantity | Value |
|----------|------:|
| Serial fraction \(s\) | 0.38 |
| Parallel fraction \(1-s\) | 0.62 |
| Estimated serial time | 10.42 s |
| Estimated parallelizable time (at 1 thread) | 16.84 s |
| Maximum theoretical speedup \(1/s\) | 2.61× |

### Observed vs. Theoretical Speedup

| Threads | Measured \(S(p)\) | Amdahl model \(S(p)\) | Gap |
|--------:|------------------:|----------------------:|----:|
| 1 | 1.00 | 1.00 | 0.00 |
| 2 | 1.57 | 1.45 | +0.13 |
| 4 | 1.93 | 1.86 | +0.07 |
| 8 | 2.03 | 2.18 | −0.15 |

The model tracks the general trend but does not match every point exactly. At 2 and 4 threads the measured speedup is slightly **above** the fitted curve; at 8 threads it falls **below** it. That pattern is consistent with extra contention at higher concurrency (CSV critical section, memory bandwidth, dynamic scheduling overhead) that a single-parameter Amdahl model does not capture.

The fitted ceiling is **2.61×**, and the best measured result (**2.03×** at 8 threads) reaches roughly **78%** of that limit.

### Practical Conclusion

OpenMP provides a clear benefit for CPU-side preprocessing: total time drops from **27.3 s** to **13.4 s** (~**51%** less). However, the serialized I/O path caps scalability. For this project the result is acceptable because preprocessing runs offline once to build the training CSV, and the exported format is shared by CPU training, GPU training, and Streamlit inference.

If higher speedup were required, the main improvement would be to **remove the CSV write critical section** — for example, by having each thread write to a private buffer or temporary file and merging once, or by exporting a binary format with parallel I/O.

---

## Summary

| Metric | Best value |
|--------|------------|
| Fastest configuration | 8 threads, 13.449 s |
| Best speedup | 2.03× vs. serial |
| Best efficiency | 0.79 (2 threads) |
| Amdahl serial fraction (fit) | 0.38 |
| Main scalability limiter | Serialized CSV writes |

Stage 1 successfully parallelized the per-image preprocessing workload with OpenMP. The measurements and Amdahl analysis show both the gain obtained and the structural reason why speedup remains sublinear.
