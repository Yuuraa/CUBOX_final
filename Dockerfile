ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN pip install matplotlib pillow tensorboardX tqdm wandb==0.12.9 icecream
RUN pip install scikit-learn==1.0.2
RUN pip install pycocotools==2.0.4
RUN pip install seaborn==0.11.2

RUN conda clean --all
RUN pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/index.html
RUN pip install mmsegmentation==0.11.0
RUN pip install scipy timm==0.3.2

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 vim ffmpeg

RUN git clone https://github.com/NVIDIA/apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

RUN git clone https://github.com/microsoft/unilm.git
WORKDIR /cubox
