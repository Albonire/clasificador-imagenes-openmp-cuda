FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app


COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev


FROM python:3.13-slim-trixie AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*


RUN useradd --create-home --uid 1000 appuser

WORKDIR /app


COPY --from=builder /app/.venv /app/.venv
COPY .streamlit/ ./.streamlit/
COPY app_streamlit/ ./app_streamlit/
COPY modelo/ ./modelo/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER appuser

EXPOSE 8501


HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail "http://localhost:${PORT:-8501}/_stcore/health" || exit 1

ENTRYPOINT ["sh", "-c", "exec streamlit run app_streamlit/app.py --server.port=${PORT:-8501}"]
