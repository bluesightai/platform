# FROM python:3.11
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
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction
# RUN poetry export --without-hashes -f requirements.txt --output export.txt
# RUN pip install -r export.txt

# Loading clay model weights
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='made-with-clay/Clay', filename='clay-v1-base.ckpt'); \
hf_hub_download(repo_id='furiousteabag/SkyCLIP', filename='SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS_epoch_20.pt')"

# Installing clay module itself
COPY ./clay ./clay
RUN touch README.md # required by poetry
RUN poetry config virtualenvs.create false && poetry install --no-interaction

COPY ./scripts ./scripts
COPY ./app ./app

CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--timeout-keep-alive", "1800", "--forwarded-allow-ips", "\"*\"" ]
