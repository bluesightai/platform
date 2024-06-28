# This file is an auto-generated Dockerfile for the 
# Malevich App image. You may edit this file to add
# additional dependencies to your app or set more
# specific enironment.

# Keep in mind, that source code containing
# Malevich-specific code (declaration of processors, inits, etc.)
# should be placed into ./apps directory

# NOTE: CUDA will be available at Nebius
FROM malevichai/app:python-torch_v0.1

ENV POETRY_VERSION=1.8.3
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
	&& $POETRY_VENV/bin/pip install -U pip setuptools \
	&& $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

COPY poetry.lock pyproject.toml ./
RUN poetry export --without-hashes -f requirements.txt --output requirements.txt
RUN pip install --default-timeout=100 -r requirements.txt

# Placing source code into the image
# DO NOT change this line
COPY ./clay ./apps

CMD [ "python", "model.py" ]
