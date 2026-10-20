FROM python:3.12-slim

# ffmpeg для конвертации аудио
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости — кэш Docker не инвалидируется при изменении кода
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код
COPY . .

# Настройки для 3 ГБ RAM
ENV AUDIO_PREPROCESSING_MODE=off
ENV LOG_LEVEL=WARNING
ENV HF_HOME=/app/.hf_cache
ENV DOCKER_MODE=1

CMD ["python", "bot.py"]
