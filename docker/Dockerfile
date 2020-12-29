FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

ENV http_proxy "http://195.221.0.35:8080"

ENV https_proxy "http://195.221.0.35:8080"

RUN apt-get update && apt-get install -y curl python3-pip

RUN pip install --no-cache-dir  https://rosh-public.s3-eu-west-1.amazonaws.com/radon/cuda-11.0/torch-1.7/torch_radon-1.0.0-cp38-cp38-linux_x86_64.whl

COPY . src/

RUN /bin/bash -c "cd src\
    && pip3 install -r requirements.txt\
    && pip install -r requirements.txt"

