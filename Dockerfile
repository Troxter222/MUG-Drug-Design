# Use offical Python
FROM python:3.10-slim

# Metadata
LABEL maintainer="Troxter222"
LABEL description="MUG: Molecular Universe Generator Container"

# Verible envirement for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Working folder
WORKDIR /app

# 1. Installing all requiremets
# libxrender1 and libxext6 need for RDKit
# wget need for Vina
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install AutoDock Vina (Linux version)
RUN mkdir -p /app/app/tool
RUN wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 -O /app/app/tool/vina.exe \
    && chmod +x /app/app/tool/vina.exe

# 3. Download Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 4. Copy code
COPY . .

# Delate
RUN chmod +x /app/app/tool/vina.exe

# Run Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]