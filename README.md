# 🌌 MUG: Molecular Universe Generator (v7.4)

[![Status](https://img.shields.io/badge/Status-Active_Research-success)](https://github.com/Troxter222)
[![Architecture](https://img.shields.io/badge/Architecture-Transformer_VAE-blueviolet)]()
[![Optimization](https://img.shields.io/badge/Optimization-PPO_Reinforcement_Learning-orange)]()
[![Physics](https://img.shields.io/badge/Physics-AutoDock_Vina-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

**MUG (Molecular Universe Generator)** is an advanced AI platform for *De Novo Drug Design*. It leverages a hybrid architecture combining **Transformers**, **Graph Neural Networks (GNN)**, and **Reinforcement Learning** to explore chemical space and identify novel therapeutic candidates with optimized bio-physical properties.

> *"Engineering cures through generative intelligence."*

---

## 🧠 System Architecture

MUG represents a closed-loop drug discovery pipeline designed to generate valid, synthesizable, and effective molecules:

### 1. Generator (The Artist)
*   **Model:** Transformer VAE (Pre-LN) with Custom Attention.
*   **Representation:** SELFIES (Self-Referencing Embedded Strings) guarantees 100% chemical validity.
*   **Latent Space:** Continuous vector space allowing for smooth interpolation between molecular structures.

### 2. Evaluator (The Critic)
*   **Physics-Based:** Integrated **AutoDock Vina** pipeline for calculating binding affinity ($\Delta G$) against 3D protein targets (e.g., EGFR, AChE, M-pro).
*   **AI-Based:** Ensemble of 12 **Random Forest & GNN models** trained on the Tox21 dataset to predict toxicity endpoints (Nuclear Receptor signaling, Stress response).

### 3. Optimizer (The Teacher)
*   **Algorithm:** Proximal Policy Optimization (PPO).
*   **Objective:** Maximize the composite reward function:
    `Score = (Binding Affinity * 5.0) + (QED * 2.0) - (Toxicity * 10.0) + (Novelty Bonus)`

---

## 🏆 Performance Benchmarks (2025)

The transition from GRU (v1) to Transformer (v2/v3) combined with Physics-Guided RL showed significant improvements:

| Metric | GRU Baseline | Transformer V2 (RL) | Improvement |
| :--- | :--- | :--- | :--- |
| **Validity** | 94.5% | **100.0%** | +5.5% (via SELFIES) |
| **Uniqueness** | 88.0% | **100.0%** | No mode collapse |
| **Binding Affinity** | -6.2 kcal/mol | **-9.8 kcal/mol** | **Significant** |
| **Drug-Likeness (QED)** | 0.45 | **0.86** | High quality |
| **Toxicity Rate** | 15.2% | **0.0%** | Safe candidates |

---

## ✨ Key Features

*   **🧪 Deep Search Mode:** An infinite loop of generation and docking that runs until a candidate exceeds a specific affinity threshold (e.g., <-9.0 kcal/mol).
*   **🛡️ AI Toxicology:** Real-time checking against 12 toxicity pathways using GNN inference.
*   **🏗️ Automated Retrosynthesis:** Decomposes generated molecules into commercially available building blocks using the BRICS algorithm.
*   **📲 Multi-Interface:**
    *   **Telegram Bot:** For remote monitoring and control via mobile.
    *   **Streamlit Web App:** For 3D visualization and deep analytics.

---

## 🚀 Installation & Setup

*Note: Pre-trained models (.pth) and processed datasets are private. This repository contains the source code for the training and inference architecture.*

### Prerequisites
*   Python 3.10+
*   NVIDIA GPU (CUDA 11.8+) for training
*   AutoDock Vina (binary required in `app/tool/` or system path)

### 1. Clone the repository
```bash
git clone https://github.com/Troxter222/MUG-Drug-Design.git
cd MUG-Drug-Design
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Training (Example)

To train the Transformer model from scratch (V3 Architecture with Mixed Precision):

```bash
python train_transformer_v3.py
```

To fine-tune using Reinforcement Learning with Vina docking:

```bash
python train_rl_vina.py
```

### 4. Running the Interface
```bash
# Telegram Bot
python run.py

# Streamlit Dashboard
streamlit run web_app.py
```

## 📂 Project Structure
```bash
MUG-Drug-Design/
├── app/
│   ├── core/           # Transformer & VAE Neural Architectures
│   ├── services/       # Biology (Docking), Chemistry, Toxicity engines
│   ├── tool/           # External tools wrappers (Vina, etc.)
│   └── trainer/        # Training loops and loss functions
├── data/               # (GitIgnored) Raw data and model checkpoints
├── run.py              # Main Entry Point
├── train_transformer_v3.py # Latest training script
└── requirements.txt    # Dependencies
```

## ⚖️ Disclaimer

This software is developed solely for **computational research purposes**.
The generated molecular structures are theoretical candidates and have **not** been validated in a wet lab environment.

> ⚠️ **SAFETY WARNING**
>
> Do **NOT** use this software for medical purposes, self-medication, the synthesis of controlled substances, or human consumption.
> The authors assume no liability for misuse.

Please read the full [DISCLAIMER](DISCLAIMER.md) document carefully before use.

---
*Copyright © 2025. All Rights Reserved.*