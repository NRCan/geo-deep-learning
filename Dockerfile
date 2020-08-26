# ============================== BASE ==========================================
FROM nvcr.io/nvidia/pytorch:20.03-py3 as base

RUN apt-get update

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# From here: https://pythonspeed.com/articles/activate-conda-dockerfile/
ADD environment-gpu.yml /app/environment-gpu.yml
RUN conda env create -f /app/environment-gpu.yml

COPY . .
