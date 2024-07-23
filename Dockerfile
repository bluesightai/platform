# FROM python:3.11
# FROM malevichai/app:python-torch_v0.1
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Install poetry separated from system interpreter
ENV POETRY_VERSION=1.8.3
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache
ENV PATH="${PATH}:${POETRY_VENV}/bin"
RUN python3 -m venv $POETRY_VENV \
	&& $POETRY_VENV/bin/pip install -U pip setuptools \
	&& $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Installing only dependencies
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --without torch
# RUN poetry export --without-hashes -f requirements.txt --output export.txt
# RUN pip install -r export.txt

# Install separately via pip to not trigger torch installation
RUN pip install huggingface_hub
RUN pip install vit_pytorch
# Loading clay model weights
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='made-with-clay/Clay', filename='clay-v1-base.ckpt')"

# Installing clay module itself
COPY ./clay ./clay
RUN touch README.md # required by poetry
RUN poetry config virtualenvs.create false && poetry install --no-interaction --without torch

RUN pip install https://github.com/MalevichAI/malevich/archive/dev/unstable.zip
RUN conda install git
RUN malevich init
COPY ./malevich.yaml ./malevich.yaml
COPY ./malevich.secrets.yaml ./malevich.secrets.yaml
RUN malevich restore
COPY ./scripts ./scripts
COPY ./app ./app

# COPY ./clay/malevich/bindings.py ./apps/
CMD [ "gunicorn", "app.main:app", "--bind", "0.0.0.0:8000", "--worker-class", "app.utils.worker.ProxyUvicornWorker", "--timeout", "1800" ]
