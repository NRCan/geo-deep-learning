FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04

# cpu only makes much smaller image (comment out previous FROM)
#FROM ubuntu:18.04

## Variables
ARG GIT_TAG=develop
ARG USERNAME=gdl_user
# Path de l'utilisateur ubuntu
ENV USER_PATH="/home/${USERNAME}"

# Variable d'environnement pour Miniconda
ENV PATH="${USER_PATH}/miniconda3/bin:${PATH}"

# Emplacement des certificats ssl
ENV NODE_EXTRA_CA_CERTS="/usr/local/share/ca-certificates/cert.crt"

RUN apt-get update \
    && apt -y install build-essential \
    && apt-get install -y \
    wget \
    gnupg2 \
    software-properties-common \
    git \
    sudo \
    tar \
    manpages-dev ca-certificates \
    && rm -rf /var/lib/apt/list/* \
    && apt-get update

# RNCAN certificate
#ADD NRCan-RootCA.cer /usr/local/share/ca-certificates/cert.crt
#RUN chmod 644 /usr/local/share/ca-certificates/cert.crt && update-ca-certificates

# Create user and login
RUN adduser --gecos '' --disabled-password ${USERNAME} \
    && adduser ${USERNAME} sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USERNAME}
WORKDIR ${USER_PATH}

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
    && mkdir ${USER_PATH}/.conda \
    && bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py39_4.10.3-Linux-x86_64.sh

# Miniconda update
RUN conda config --set ssl_verify False \
    && conda update conda

RUN git clone --depth 1 "https://github.com/NRCan/geo-deep-learning.git" --branch ${GIT_TAG} && \

# Alternative: create conda env from soft reference to libraries
#RUN wget https://github.com/remtav/geo-deep-learning/blob/solaris_tiling/environment.yml

# Create env
RUN conda env create -f geo-deep-learning/environment.yml

RUN echo ". $USER_PATH/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "source $USER_PATH/miniconda3/etc/profile.d/conda.sh" && \
    echo "conda activate geo_deep_env" >> ~/.bashrc