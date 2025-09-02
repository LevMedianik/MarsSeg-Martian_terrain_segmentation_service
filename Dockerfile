# syntax=docker/dockerfile:1.7
# Базовый минимальный образ Python 3.12 (slim).
FROM python:3.12-slim

# Небольшое ускорение Python и логи в Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Рабочая директория внутри контейнера
WORKDIR /app

# === 1) Установка PyTorch/torchvision отдельно ============================
#   - Это самые "тяжёлые" пакеты; ставим их в отдельном слое, чтобы кэшировать.
#   - Берутся CPU-колёса из официального индекса PyTorch (без CUDA).
#   - --mount=type=cache сохраняет кэш pip между сборками (BuildKit должен быть включён).
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 torchvision==0.17.2

# === 2) Установка прочих зависимостей проекта ==============================
#   - Кэш pip.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# === 3) Копирование кода и весов модели ===========================================
#   - Только то, что нужно для запуска.
COPY app ./app
COPY models ./models
COPY .env ./.env

# Порт приложения (FastAPI/Uvicorn)
EXPOSE 8000

# Запуск Uvicorn (FastAPI) на 0.0.0.0:8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
