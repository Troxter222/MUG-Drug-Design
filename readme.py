import os

# --- БЕЗОПАСНАЯ ГЕНЕРАЦИЯ (без тройных кавычек в тексте) ---
cb = chr(96) * 3  

readme_content = f"""# 🌌 MUG: Molecular Universe Generator (v7.5)

[![Status](https://img.shields.io/badge/Status-Active_Research-success)]()
[![Architecture](https://img.shields.io/badge/Architecture-Transformer_VAE-blueviolet)]()
[![Optimization](https://img.shields.io/badge/Optimization-PPO_Reinforcement_Learning-orange)]()
[![Physics](https://img.shields.io/badge/Physics-AutoDock_Vina-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

**MUG (Molecular Universe Generator)** is an AI platform for *De Novo Drug Design*. It combines **Transformers**, **GNNs**, and **Reinforcement Learning** to find the optimal balance between binding affinity and chemical validity.

> *"Engineering cures through generative intelligence."*

Unlike competitors that often overfit on docking scores producing unrealistic structures, MUG optimizes for **High Drug-Likeness (QED)** and **Synthesizability** while maintaining competitive affinity.

---

## 🏆 Honest Benchmarks (2025)

Performance analysis based on `benchmark_full_results.csv`. We prioritize realistic drug candidates over theoretical docking scores.

### 📊 Metric Comparison

| Model / Approach | Validity | Uniqueness | Avg QED (0-1) | Peak Docking (kcal/mol) | Stability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Graph-based GA** | 92.5% | 88.0% | 0.45 | **-9.5** | Low |
| **REINVENT 3.0** | 99.1% | 94.5% | 0.78 | -8.1 | High |
| **MUG (Tox-GNN)** | 100% | 95.0% | 0.66 | **-8.45** | **Toxic (Experimental)** |
| **MUG (Final RL)** | **100%** | **99.8%** | **0.76** | **-7.20** | **Balanced (Best Candidate)** |

> **Key Insight:** While graph-based methods achieve extreme docking scores (<-9.0), they often generate chemically unstable structures. **MUG (Final RL)** achieves a "Sweet Spot" (~ -7.2 kcal/mol) with high QED (>0.75), ensuring generated molecules are viable drug candidates.

### 📉 Generation Stats (From CSV)

{cb}mermaid
pie
    title MUG Generated Candidates Distribution
    "High QED (>0.7) & Binder" : 65
    "Medium QED" : 25
    "Toxic / Rejected" : 10
{cb}

---

## 🧠 Model Zoo (Included Checkpoints)

Based on the provided benchmark logs, these are the key models:

| Model ID | Architecture | Best Result (Vina) | Focus |
| :--- | :--- | :--- | :--- |
| `final_alzheimer.pth` | Transformer-RL | **-7.17** (EGFR) | **Best Generalization** |
| `mug_transformer_rl_best` | Transformer-RL | -7.03 (DRD2) | Dopamine Receptors |
| `final_covid.pth` | Transformer-RL | -6.92 (3CL M-pro) | Viral Protease |
| `final_egfr.pth` | Transformer-RL | -7.02 (L858R) | Kinase Inhibitor |
| `gnn_tox_v1` | Graph Neural Net | -8.45 (Outlier) | *Legacy / High Toxicity Risk* |

---

## 🔬 System Architecture

MUG represents a closed-loop drug discovery pipeline:

{cb}mermaid
graph TD
    A[Target Input] -->|PDBQT| B(Generator: Transformer);
    B -->|SELFIES| C{{Evaluator}};
    C -->|Calculate| D[QED & SA Score];
    C -->|Simulate| E[AutoDock Vina];
    C -->|Inference| F[GNN Toxicity Filter];
    D --> G[Composite Reward];
    E --> G;
    F --> G;
    G -->|PPO Step| B;
{cb}

### 1. Generator
*   **Model:** Transformer VAE (Pre-LN).
*   **Input:** SELFIES (Self-Referencing Embedded Strings).

### 2. Evaluator (The Critic)
*   **Vina Engine:** Runs local docking simulations (`app/services/biology.py`).
*   **Toxicity Guard:** 12 GNN models trained on Tox21 dataset to penalize risky structures.

### 3. Objective Function
$$ Score = (\\text{{Affinity}} \\times 5.0) + (\\text{{QED}} \\times 2.0) - (\\text{{Toxicity}} \\times 10.0) $$

---

## 🚀 Usage

### 1. Run Benchmarks
To reproduce the CSV results:

{cb}bash
python run_benchmark_suite.py
{cb}

### 2. Deep Search Mode (Web App)
Launch the dashboard to visualize 3D structures and run the genetic loop:

{cb}bash
streamlit run web_app.py
{cb}

### 3. Telegram Bot
For remote monitoring:

{cb}bash
python run.py
{cb}

---

## 📂 Project Structure

{cb}bash
MUG/
├── app/
│   ├── core/           # Transformer & VAE Models
│   ├── services/       # Biology, Chemistry, Toxicity logic
│   └── tool/           # Vina binary
├── checkpoints/        # .pth Model Weights
├── dataset/            # SELFIES Vocabularies
├── benchmark_results/  # Full CSV results
├── tests/              # Unit Tests
└── requirements.txt    # Python dependencies
{cb}

## ⚖️ Disclaimer

This software is for **computational research only**. Generated structures are theoretical predictions.

> ⚠️ **SAFETY WARNING**
> Do **NOT** use this software for medical purposes or synthesis of controlled substances.

---
*Copyright © 2025. Troxter222. All Rights Reserved.*
"""

def create_honest_readme():
    file_path = "README.md"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print(f"✅ README.md updated with HONEST metrics from benchmark_full_results.csv")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_honest_readme()