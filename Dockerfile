FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04

ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=gdl_user
ARG USERID=1000
ARG GIT_TAG=develop

# RNCAN certificate; uncomment (with right .cer name) if you are building behind a FW
#COPY NRCan-RootCA.cer /usr/local/share/ca-certificates/cert.crt
#RUN chmod 644 /usr/local/share/ca-certificates/cert.crt && update-ca-certificates

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

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#     echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER $USERNAME
WORKDIR /home/$USERNAME/

RUN cd /home/$USERNAME && git clone --depth 1 "https://github.com/remtav/geo-deep-learning.git" --branch $GIT_TAG
RUN conda config --set ssl_verify no
RUN conda env create -f /home/$USERNAME/geo-deep-learning/environment.yml
RUN cd /home/$USERNAME && git clone --depth=1 http://github.com/remtav/projectRegularization -b light

ENV PATH $CONDA_DIR/envs/geo_deep_env/bin:$PATH
RUN echo "source activate geo_deep_env" > ~/.bashrc
