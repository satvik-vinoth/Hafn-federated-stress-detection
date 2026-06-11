# HAFN-FedNova: Privacy-Preserving Stress Detection

A federated learning framework for stress detection from wrist-worn physiological sensors, built on the **WESAD** dataset. This repository contains the full implementation of the **Hierarchical Attention Fusion Network (HAFN)** trained under four aggregation strategies — FedAvg, FedProx, CFL, and FedNova.

---

## Results

| Model | Strategy | Accuracy | Macro-F1 |
|-------|----------|----------|----------|
| HAFN | FedNova | **92.22%** | **0.8938** |
| HAFN | FedAvg | 91.57% | 0.8910 |
| HAFN | FedProx | 91.55% | 0.8815 |
| HAFN | CFL | 91.48% | 0.8810 |
| CNN-LSTM | FedNova | 89.69% | 0.8595 |
| CNN-LSTM-Attn | FedNova | 85.03% | 0.8392 |
| CNN | FedNova | 79.00% | 0.7779 |

---

## Architecture

HAFN processes 4 physiological modalities from the Empatica E4 wristband — **BVP, EDA, ACC, and TEMP** — through three levels:

- **Intra-modal encoding** — per-modality BiLSTM with learned positional encoding and temporal self-attention
- **Motion artifact gate** — suppresses ACC-induced noise in BVP and EDA before fusion
- **Cross-modal attention** — lets each modality attend to all others
- **Auxiliary branches** — FFT-based HRV features + global channel statistics

---

## Repo Structure

```
HAFN_FL_SV/
├── HAFN Implementation.ipynb       # Full training and evaluation code
├── preprocessed_wrist_v9/          # Preprocessed WESAD wrist data (15 subjects)
│   ├── S2_chrono_v8.pkl
│   └── ...
└── results_wrist_comparison/       # Result plots and CSVs
    ├── comparison_results.csv
    ├── convergence_all.png
    └── ...
```

---

## Dataset

This project uses the [WESAD dataset](https://doi.org/10.24432/C57K5T) (Wearable Stress and Affect Detection). WESAD is not included in this repository — download it separately and run the preprocessing section of the notebook to regenerate the `.pkl` files, or use the preprocessed files provided directly.

---

## Setup

```bash
git clone https://github.com/your-username/HAFN_FL_SV.git
cd HAFN_FL_SV
pip install torch flwr scikit-learn scipy numpy pandas
```

Then open and run `HAFN Implementation.ipynb`.

---

## Privacy Mechanisms

Raw physiological data never leaves the client. Two additional mechanisms are applied during local training:

- **Gradient clipping** (norm ≤ 1.0) — limits the influence of any single data point on transmitted weights
- **Gaussian noise** (σ = 5×10⁻⁴) — adds controlled randomness to prevent weight reconstruction

---

## Citation

If you use this work, please cite the WESAD dataset:

> Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., Van Laerhoven, K. *Introducing WESAD, a multimodal dataset for wearable stress and affect detection.* ICMI 2018.
