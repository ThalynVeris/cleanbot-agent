#!/usr/bin/env sh
set -eu
python -m cleanbot.cli init-db
python -m cleanbot.cli ingest
exec uvicorn cleanbot.api.app:app --host 0.0.0.0 --port 8000
