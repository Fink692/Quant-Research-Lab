FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-dev.txt requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-dev.txt

COPY . .
RUN python -m pip install -e .

CMD ["pytest"]
