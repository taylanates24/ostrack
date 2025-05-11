FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN pip install --upgrade pip
RUN pip install onnx


RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install opencv-python \
    pandas \
    tqdm \
    easydict \
    cython \
    PyYAML \
    pycocotools \
    jpeg4py \
    tb-nightly \
    tikzplotlib \
    thop-0.0.31.post2005241907 \
    colorama \
    lmdb \
    scipy \
    visdom \
    pyarrow \
    tensorboardX \
    setuptools==59.5.0 \    
    wandb \
    timm



WORKDIR /workspace


