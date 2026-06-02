# Project Conventions

## Commit messages
- Use **Conventional Commits** format, always in **English**.
- Format: `<type>: <short description>`
- Allowed types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `style`, `perf`, `ci`, `build`
- Examples:
  - `feat: add sobel filter kernel in OpenMP pipeline`
  - `fix: correct memory alignment in CUDA matmul kernel`
  - `docs: update dataset collection instructions`
  - `chore: add .gitkeep for processed dataset folder`

## Naming conventions
- All code identifiers (variables, functions, file names, constants) must be in **English**.
- Use a single consistent convention per language:
  - **C / CUDA**: `snake_case` for variables and functions, `UPPER_SNAKE_CASE` for macros/constants
  - **Python**: `snake_case` for variables and functions, `PascalCase` for classes
- Documentation content (READMEs, comments) may be in Spanish since this is a Spanish-language university project.

## Repository structure
```
etapa1_openmp/      # C + OpenMP preprocessing pipeline
etapa2_cuda/        # CUDA training kernels
  cpu_baseline/     # CPU-only version for speedup comparison
modelo/             # Trained model weights
app_streamlit/      # Streamlit inference app
notebooks/          # Training scripts and benchmark plots
dataset/            # Representative subset of the dataset
  raw/clase_0/
  raw/clase_1/
  procesado/
reporte/            # Final report (Markdown or PDF)
```
