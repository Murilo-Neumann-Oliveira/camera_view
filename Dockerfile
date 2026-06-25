FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"

EXPOSE 5000

CMD ["python", "main.py"]