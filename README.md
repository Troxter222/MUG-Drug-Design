# MUG: Molecular Universe Generator 🌌🧬

**An AI-driven platform for De Novo Drug Design using Deep Generative Models.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-green)]()

## 🔬 Overview
MUG is a generative artificial intelligence system designed to explore the chemical space and identify novel drug candidates. Unlike traditional screening methods, MUG uses a **Variational Autoencoder (VAE)** with **Reinforcement Learning (RL)** to "imagine" new molecules that satisfy multi-objective constraints (bio-activity, synthesis accessibility, low toxicity).

## 🧠 Core Technology
1.  **Architecture:** Deep GRU-based VAE trained on ZINC/ChEMBL datasets.
2.  **Representation:** SELFIES (Self-Referencing Embedded Strings) ensuring 100% validity of generated structures.
3.  **Optimization:** Proximal Policy Optimization (PPO-like) RL loop to maximize QED and Binding Affinity.
4.  **Bio-Physics:** Integrated QSAR scoring for Protein-Ligand docking simulation.

## ✨ Features
*   **Hunter Mode:** Targeted generation for specific pathologies (Alzheimer's, Oncology, Virology).
*   **Safety First:** Embedded GNN (Graph Neural Network) for toxicity prediction (Tox21).
*   **Reality Check:** Automatic retrosynthesis planning (BRICS decomposition) and PubChem novelty validation.
*   **Visualization:** Cyberpunk-style chemical rendering and 3D SDF model generation.

## 🚀 Installation
```bash
# 1. Clone repo
git clone https://github.com/TroxterGrif222/mug-project.git
cd mug-project

# 2. Setup environment (Windows)
setup.bat

# 3. Run System
python run.py
```

📊 Performance
Validity: 100% (via SELFIES)
Uniqueness: >95%
Mean QED (After RL): 0.88 (vs 0.45 baseline)

!!!Developed for research purposes!!!