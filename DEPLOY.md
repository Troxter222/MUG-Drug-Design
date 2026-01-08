# 🐳 Docker Deployment Guide for MUG

This guide details how to deploy the **Molecular Universe Generator (MUG)** using Docker. Containerization ensures consistent behavior across different operating systems by isolating dependencies like RDKit, PyTorch, and AutoDock Vina.

## 📋 Prerequisites

1.  **Docker** and **Docker Compose** installed on your machine.
2.  A `.env` file in the project root directory containing your API keys:
```env
TELEGRAM_TOKEN=your_telegram_bot_token_here
```

---

## 🚀 Quick Start

### Option 1: Full Stack (Bot + Web App) via Docker Compose

This is the recommended method. It starts both the Telegram bot and the Streamlit dashboard simultaneously.

1.  **Build and Start:**
```bash
docker-compose up -d --build
```

2.  **Access:**
    *   **Web Dashboard:** Open `http://localhost:8501` in your browser.
    *   **Telegram Bot:** Your bot is now active and polling for messages.

3.  **View Logs:**
```bash
docker-compose logs -f
```

4.  **Stop Containers:**
```bash
docker-compose down
```

### Option 2: Web App Only (Single Container)

If you only need the visual dashboard and want to run it manually:

```bash
# Build the image
docker build -t mug-ai .

# Run container mapping port 8501 and mounting data volume
docker run -p 8501:8501 -v $(pwd)/data:/app/data mug-ai
```

---

## 🖥️ GPU Support (NVIDIA)

To accelerate model training and inference using CUDA:

1.  Ensure the **NVIDIA Container Toolkit** is installed on the host machine.
2.  Modify the `docker-compose.yml` environment variable:
    *   Change `DEVICE=cpu` to `DEVICE=cuda`.
3.  If running manually with Docker CLI:
```bash
docker run --gpus all -p 8501:8501 mug-ai
```

---

## 🛠 Technical Notes

*   **AutoDock Vina Compatibility:** The Dockerfile automatically downloads the **Linux binary** of AutoDock Vina but renames it to `vina.exe` inside the container. This ensures compatibility with the main Python codebase (which expects `vina.exe`) without requiring code changes between Windows and Linux environments.
*   **Data Persistence:** The `data/`, `checkpoints/`, and `logs/` directories are mounted as volumes. Generated molecules and trained model weights will persist on your host machine even after the containers are stopped or removed.
