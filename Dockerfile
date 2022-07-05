ARG GIT_HASH=5f3c754
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda

FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04 AS build

# RNCAN certificate; uncomment (with right .cer name) if you are building behind a FW
# COPY NRCan-RootCA.cer /usr/local/share/ca-certificates/cert.crt
# RUN chmod 644 /usr/local/share/ca-certificates/cert.crt && update-ca-certificates

RUN apt-get update \
    && apt-get install -y --no-install-recommends git wget unzip bzip2 build-essential sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG GIT_HASH
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

RUN git clone "https://github.com/NRCan/geo-deep-learning.git" && \
    cd geo-deep-learning && \
    git checkout ${GIT_HASH}

# TODO : create an environment for inference
RUN conda env create -f geo-deep-learning/environment.yml && \
    rm -rf .git

# runtime image
FROM condaforge/mambaforge

# RNCAN certificate; uncomment (with right .cer name) if you are building behind a FW
# COPY NRCan-RootCA.cer /usr/local/share/ca-certificates/cert.crt
# RUN chmod 644 /usr/local/share/ca-certificates/cert.crt && update-ca-certificates

RUN apt-get update \
    && apt-get install -y --no-install-recommends git wget unzip bzip2 build-essential sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG CONDA_DIR
ARG USERNAME=gdl_user
ARG USERID=1000

RUN mkdir -p /opt/conda/

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER $USERNAME
WORKDIR /home/$USERNAME

## Variables 
ARG GIT_HASH
# Alternative to using a git hash: use git tag (uncomment git clone command below)
# ARG GIT_TAG=develop

RUN git clone "https://github.com/NRCan/geo-deep-learning.git" && \
    cd geo-deep-learning && \
    git checkout ${GIT_HASH} && \
    rm -rf .git

COPY --chown=1000 --from=build /opt/conda/. $CONDA_DIR

ENV PATH $CONDA_DIR/envs/geo_deep_env/bin:$PATH
RUN echo "source activate geo_deep_env" > ~/.bashrc

# Alternative: Get gdl repo by tag
#RUN git clone --depth 1 "https://github.com/NRCan/geo-deep-learning.git" --branch ${GIT_TAG} && \
#    cd geo_deep-learning

# Alternative: create conda env from soft reference to libraries
#RUN wget https://github.com/remtav/geo-deep-learning/blob/solaris_tiling/environment.yml
