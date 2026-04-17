FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Preload OCR models at build time so Railway does not try to fetch them during /ocr requests
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ch', enable_mkldnn=False); print('OCR models preloaded successfully')"

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "180", "app:app"]
