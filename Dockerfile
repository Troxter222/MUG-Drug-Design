# Используем официальный легкий образ Python
FROM python:3.10-slim

# Метаданные
LABEL maintainer="Troxter222"
LABEL description="MUG: Molecular Universe Generator Container"

# Переменные окружения для Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Рабочая директория
WORKDIR /app

# 1. Установка системных зависимостей
# libxrender1 и libxext6 нужны для RDKit
# wget нужен для скачивания Vina
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Установка AutoDock Vina (Linux версия)
# Мы кладем его туда, где код ожидает его найти, имитируя структуру проекта
RUN mkdir -p /app/app/tool
RUN wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 -O /app/app/tool/vina.exe \
    && chmod +x /app/app/tool/vina.exe

# 3. Установка Python зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 4. Копирование кода проекта
COPY . .

# Очистка прав доступа (на всякий случай)
RUN chmod +x /app/app/tool/vina.exe

# По умолчанию запускаем Streamlit, но это можно переопределить
EXPOSE 8501
CMD ["streamlit", "run", "web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]