
# Dockerised version of Segment Anything From Facebook




## Installation

prerequisites: Docker (https://docs.docker.com/engine/install/) and CUDA drivers (https://docs.nvidia.com/cuda/cuda-installation-guide-linux).
## Run Locally

Clone the project

```bash
  git clone https://github.com/tachillon/Segment-anything-Facebook-Dockerised.git
```

Go to the project directory

```bash
  cd Segment-anything-Facebook-Dockerised
```

Build the docker image

```bash
  docker build --no-cache -t sam:latest .
```

Retrieve the model checkpoint from the official Facebook repo (https://github.com/facebookresearch/segment-anything.git)

Run the code through the docker

```bash
  docker run --rm -it --gpus all --ipc=host -w /tmp -v ${PWD}:/tmp sam:latest python3 main.py --image_path <path_to_image> --checkpoint <path_to_checkpoin>
```

