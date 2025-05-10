FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

COPY main.py .

EXPOSE 8000

ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV VECTORSTORE_PATH=/app/chroma_db
ENV PORT=8000
ENV HOST=0.0.0.0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]