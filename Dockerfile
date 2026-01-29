# Builder stage with uv
FROM ghcr.io/astral-sh/uv:bookworm AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Omit development dependencies
ENV UV_NO_DEV=1

# Configure the Python directory so it is consistent
ENV UV_PYTHON_INSTALL_DIR=/opt/python

# Only use the managed Python version
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python before the project for caching
RUN uv python install 3.12

WORKDIR /app

# Install dependencies first (better caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --extra cu128

# Then copy source and install project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra cu128

# Final runtime stage with CUDA
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# Setup a non-root user
RUN useradd --system --uid 999 --create-home gdl_user

# Copy the Python installation from builder
COPY --from=builder --chown=gdl_user:gdl_user /opt/python /opt/python

# Copy the application from the builder
COPY --from=builder --chown=gdl_user:gdl_user /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:/opt/python/bin:$PATH"

# Use the non-root user to run our application
USER gdl_user

# Use `/app` as the working directory
WORKDIR /app

# Run the training module by default
ENTRYPOINT ["python"]
CMD ["-m", "geo_deep_learning.train"]