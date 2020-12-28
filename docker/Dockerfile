FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update && ap-get install -y python3 python3-pip sudo

RUN pip3 install --no-cache-dir  https://rosh-public.s3-eu-west-1.amazonaws.com/radon/cuda-11.0/torch-1.7/torch_radon-1.0.0-cp38-cp38-linux_x86_64.whl

RUN useradd -m bellachehab

RUN chown -R bellachehab:bellachehab /home/experiences/grades/bellachehab/dual_gan

COPY --chown=bellachehab . /home/bellachehab/dual_gan/

USER bellachehab

RUN cd /home/bellachehab/dual_gan/ && pip3 install -r requirements.txt

WORKDIR /home/bellachehab/dual_gan