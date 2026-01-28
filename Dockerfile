FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1

WORKDIR /tmp
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --extra cu128

RUN useradd -m -u 1000 gdl_user && mkdir -p /app && chown -R gdl_user /app
USER gdl_user

WORKDIR /app
COPY --chown=gdl_user:gdl_user . /app

ENTRYPOINT ["python"]
CMD ["-m", "geo_deep_learning.train"]