# This file is used as an argument to `docker build`
# Its purpose is to allow combining http://github.com/remtav/projectRegularization in a "stock" GDL image
# ex. docker build -t username/gdl-cuda11:v2.3.0-prod https://github.com/username/geo-deep-learning.git#v2.3.0-prod -f- < Dockerfile-remtav
# That generates a new image to be uploaded to a Docker repo like DockerHub, which then allows building the Singularity image as
# cd /path/deep_learning/singularity_images
# export SINGULARITY_TMPDIR=/path/singularity/images
# export SINGULARITY_CACHEDIR=/path/singularity/images
# singularity pull docker://username/gdl-cuda11:latest
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=gdl_user

FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04 AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends git wget unzip bzip2 build-essential sudo \
    && apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && sudo dpkg -i cuda-keyring_1.0-1_all.deb \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004-keyring.gpg \
    && sudo mv cuda-ubuntu2004-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && sudo dpkg -i cuda-keyring_1.0-1_all.deb && rm -f cuda-keyring_1.0-1_all.deb && rm -f /etc/apt/sources.list.d/cuda.list \
    
	&& apt-get update \
    && apt-get clean \
	&& apt-get install git \
    && rm -rf /var/lib/apt/lists/*
	
ARG CONDA_PYTHON_VERSION
ARG CONDA_DIR

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
	
ARG USERNAME
RUN mkdir -p /home/$USERNAME
RUN cd /home/$USERNAME && git clone https://github.com/NRCan/geo-deep-learning.git
RUN conda config --set ssl_verify no
RUN conda env create -f /home/$USERNAME/geo-deep-learning/environment.yml
RUN cd /home/$USERNAME && git clone --depth=1 http://github.com/remtav/projectRegularization -b light

# runtime image
FROM condaforge/mambaforge
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget unzip bzip2 build-essential sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ARG CONDA_DIR
ARG USERID=1000
ARG USERNAME=gdl_user
RUN mkdir -p /opt/conda/
# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER $USERNAME
WORKDIR /home/$USERNAME/

# From (remote) context
COPY --chown=1000 --from=build /home/$USERNAME/geo-deep-learning/. /home/$USERNAME/geo-deep-learning
COPY --chown=1000 --from=build /home/$USERNAME/projectRegularization/. /home/$USERNAME/projectRegularization

# Variables
COPY --chown=1000 --from=build /opt/conda/. $CONDA_DIR

ENV PATH $CONDA_DIR/envs/geo_deep_env/bin:$PATH
RUN echo "source activate geo_deep_env" > ~/.bashrc