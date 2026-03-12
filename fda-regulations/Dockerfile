# ── Stage 1: install dependencies with uv ─────────────────────────────────────
FROM python:3.13-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy lockfile and project manifest first for layer caching —
# dependencies only reinstall when these two files change
COPY pyproject.toml uv.lock ./

# Install into a non-editable venv inside /app/.venv
# --frozen: refuse to update uv.lock (ensures reproducibility)
# --no-install-project: skip installing the project itself here
RUN uv sync --frozen --no-install-project --no-dev


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY . .

# Put the venv on PATH so scripts resolve without activation
ENV PATH="/app/.venv/bin:$PATH"

# HuggingFace caches models here — mount this as a volume to avoid
# re-downloading the embedding model and LLM on every container start
ENV HF_HOME="/app/.cache/huggingface"

# HF_TOKEN must be passed at runtime:
#   docker run -e HF_TOKEN=hf_... ...
# Do not hardcode it here.
ENV HF_TOKEN=""

# fda_warning_letters/ and fda_faiss_index/ are written at runtime —
# mount them as volumes (see docker-compose.yml) so data persists
# across container restarts.

ENTRYPOINT ["python", "main.py"]
CMD ["chat", "--skip-index"]