# motivated by: https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./


RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Copy the project files into the image
COPY ./container_component_src /app/container_component_src
COPY ./pipeline /app/pipeline
COPY ./config.toml /app/config.toml

RUN poetry install --without dev

