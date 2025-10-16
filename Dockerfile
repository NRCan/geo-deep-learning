# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.8-bullseye

# Set up environment
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# Copy environment files
COPY requirements.txt ./
COPY pyproject.toml ./

# Create environment and install dependencies with space optimization
RUN micromamba create -y -n geo-dl -c conda-forge python=3.10 pip && \
    micromamba run -n geo-dl pip install --no-cache-dir -r requirements.txt && \
    micromamba clean -a -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /opt/conda/envs/geo-dl -name "*.pyc" -delete && \
    find /opt/conda/envs/geo-dl -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Set environment path and activation
ENV PATH="/opt/conda/envs/geo-dl/bin:$PATH"
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy source code
COPY . /app
WORKDIR /app

# Set entrypoint (override as needed)
ENTRYPOINT ["python"]
CMD ["-m", "geo_deep_learning.train"]
