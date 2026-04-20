FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow==2.9.2 requests==2.31.0

WORKDIR /rollback

COPY rollback.sh ./rollback.sh
RUN chmod +x ./rollback.sh

# rollback_manager.py is copied in at runtime via volume mount from the root
# compose file. When running the sub-compose standalone the MLflow path is
# skipped if the file is absent (see rollback.sh).

CMD ["./rollback.sh"]
