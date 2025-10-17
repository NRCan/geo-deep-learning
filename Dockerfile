# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y curl bzip2 && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    rm -rf /var/lib/apt/lists/*

ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_ENV_NAME=geo-dl \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH="/opt/conda/envs/geo-dl/bin:$PATH"

WORKDIR /tmp
COPY requirements.txt pyproject.toml ./

RUN micromamba create -y -n $CONDA_ENV_NAME -c conda-forge python=3.10 pip && \
    micromamba run -n $CONDA_ENV_NAME pip install --no-cache-dir -r requirements.txt && \
    find $MAMBA_ROOT_PREFIX/envs/$CONDA_ENV_NAME -name "*.pyc" -delete 2>/dev/null || true && \
    find $MAMBA_ROOT_PREFIX/envs/$CONDA_ENV_NAME -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    micromamba clean -a -y

RUN useradd -m -u 1000 gdl_user && mkdir -p /app && chown -R gdl_user /app
USER gdl_user

WORKDIR /app
COPY --chown=gdl_user:gdl_user . /app

ENTRYPOINT ["python"]
CMD ["-m", "geo_deep_learning.train"]
