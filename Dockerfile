FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
            git                            \
            wget                           \
            python3                        \
            python3-dev                    \
            python3-pip                    \  
            python3-opencv              && \
            apt-get autoremove          && \          
            apt-get clean               && \                
            rm -rf /var/lib/apt/lists/* && \
            pip3 install --no-cache-dir    \
            pip                            \
            setuptools                  && \
            python3 -m pip install --upgrade pip

RUN git clone https://github.com/facebookresearch/segment-anything.git && \
    cd segment-anything && \
    pip3 install -e .

RUN pip3 install torch torchvision opencv-python pycocotools matplotlib onnxruntime onnx

RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O /home/sam_vit_h_4b8939.pth

WORKDIR /home
