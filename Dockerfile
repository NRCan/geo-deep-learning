FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=gdl_user
ARG USERID=1000
ARG GIT_TAG=develop
ENV PATH=$CONDA_DIR/bin:$PATH
# RNCAN certificate; uncomment (with right .cer name) if you are building behind a FW
# COPY NRCan-RootCA.cer /usr/local/share/ca-certificates/cert.crt
# RUN chmod 644 /usr/local/share/ca-certificates/cert.crt && update-ca-certificates

RUN apt-get update \
    && apt-get install -y --no-install-recommends git wget unzip bzip2 build-essential sudo \
    && wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O /tmp/mamba.sh \
    && /bin/bash /tmp/mamba.sh -b -p $CONDA_DIR \
    && rm -rf /tmp/* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME \
    && chown $USERNAME $CONDA_DIR -R \
    && adduser $USERNAME sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

ENV LD_LIBRARY_PATH=$CONDA_DIR/lib:$LD_LIBRARY_PATH
USER $USERNAME
WORKDIR /usr/app

COPY environment.yml /usr/app
RUN cd /home/$USERNAME && \
    conda config --set ssl_verify no && \ 
    mamba env create -f /usr/app/environment.yml && \
    mamba clean --all \
    && pip uninstall -y pip

COPY . /usr/app/geo-deep-learning
ENV PATH=$CONDA_DIR/envs/geo_ml_env/bin:$PATH
RUN echo "source activate geo_ml_env" > ~/.bashrc