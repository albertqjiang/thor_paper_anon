#!/usr/bin/env bash
set -e

sudo apt update
sudo apt install -y python3-venv
sudo /usr/bin/docker-credential-gcr configure-docker

sudo docker rm libtpu || true
sudo docker create --name libtpu gcr.io/cloud-tpu-v2-images/libtpu:libtpu_20210518_RC00 "/bin/bash" && sudo docker cp libtpu:libtpu.so /lib

# this locks the python executable down to hopefully stop if from being fiddled with...
# screen -d -m python -c 'import time; time.sleep(999999999)'
python3 -m venv ~/venvs/atp
source ~/venvs/atp/bin/activate
pip install --upgrade pip setuptools wheel

# initializes jax and installs ray on cloud TPUs
pip install --upgrade ray[default]==1.5.1 fabric dataclasses optax==0.0.6 git+https://github.com/deepmind/dm-haiku tqdm cloudpickle smart_open[gcs] einops func_timeout
pip install -r requirements.txt
