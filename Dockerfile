# Builder stage with CUDA 12.8.1.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libspatialindex-dev \
    libexpat1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed

RUN uv python install 3.12

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --extra cu128

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra cu128

# Runtime stage with CUDA 12.8.1.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# Install geospatial runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal34t64 \
    libspatialindex-c6 \
    libexpat1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 999 --create-home gdl_user

COPY --from=builder --chown=gdl_user:gdl_user /opt/python /opt/python
COPY --from=builder --chown=gdl_user:gdl_user /app /app

ENV PATH="/app/.venv/bin:/opt/python/bin:$PATH"

USER gdl_user
WORKDIR /app

ENTRYPOINT ["python"]
CMD ["-m", "geo_deep_learning.train"]
