FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY training/ ./training/

RUN mkdir -p /app/checkpoints /app/results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "src/api.py"]
