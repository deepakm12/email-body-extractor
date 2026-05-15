FROM docker.io/library/python:3.10.20-slim-bookworm@sha256:94295b5d484b1137b94660d2b53f1e21073d199963ce48a9cca93c13279dfefc

WORKDIR /app

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/app/src
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

RUN pip install "poetry>=2.0.0,<3.0.0"

COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --no-root --only main
COPY src/ ./src/
EXPOSE 3000

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]