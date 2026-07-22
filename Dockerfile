FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml requirements.lock ./
RUN pip install --upgrade pip && pip install -r requirements.lock

COPY . .
RUN pip install --no-deps -e .

EXPOSE 8000
CMD ["sh", "scripts/run_api.sh"]
