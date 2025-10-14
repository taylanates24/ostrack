FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN pip install --upgrade pip
RUN pip install onnx


RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install opencv-python==4.5.5.64 \
    pandas \
    tqdm \
    easydict \
    cython \
    PyYAML \
    pycocotools \
    jpeg4py \
    tb-nightly \
    thop \
    colorama \
    lmdb \
    scipy \
    visdom \
    pyarrow \
    tensorboardX \
    setuptools==59.5.0 \    
    wandb \
    timm \
    tikzplotlib

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libgl1-mesa-glx libxext6 libxrender1 libsm6 libfontconfig1 libxkbcommon-x11-0 libx11-xcb1
WORKDIR /workspace


