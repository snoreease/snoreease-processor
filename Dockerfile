FROM python:3.11-slim

# system packages needed by librosa/soundfile and audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Render will expose a PORT env var; we’ll use 8000
ENV PORT=8000
EXPOSE 8000

CMD ["python","app.py"]
