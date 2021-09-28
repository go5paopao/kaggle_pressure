FROM nvidia/cuda:11.0-base


RUN apt update && apt install -y \
    git \
    tmux \
    ssh-client \
    python3 \
    python3-pip

COPY pyproject.toml .

RUN pip install poetry

RUN poetry install

WORKDIR /work

EXPOSE 8888