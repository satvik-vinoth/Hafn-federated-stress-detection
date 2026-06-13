HAFN-FedNova: Privacy-Preserving Stress Detection

A federated learning framework for multimodal stress detection from wrist-worn physiological sensors, built on the WESAD dataset. This repository contains the full implementation of the Hierarchical Attention Fusion Network (HAFN) evaluated across four aggregation strategies — FedAvg, FedProx, CFL, and FedNova.

The best configuration, HAFN–FedNova, achieves 92.22% accuracy and a macro-F1 of 0.8938 across 15 federated clients, while keeping all raw physiological data on-device.


Results

ModelStrategyAccuracyMacro-F1HAFNFedNova92.22%0.8938HAFNFedAvg91.57%0.8910HAFNFedProx91.55%0.8815HAFNCFL91.48%0.8810CNN-LSTMFedNova89.69%0.8595CNN-LSTM-AttnFedNova85.03%0.8392CNNFedNova79.00%0.7779

HAFN consistently ranks first across all four strategies with a performance variance of only 0.74 percentage points, showing that the gains come from the architecture itself rather than the choice of aggregation method.


Architecture

HAFN processes 4 physiological modalities from the Empatica E4 wristband — BVP, EDA, ACC, and TEMP — through three levels of representation:


Intra-modal encoding — each modality gets its own BiLSTM encoder with learned positional encoding and temporal self-attention; EDA uses a deeper 3-layer BiLSTM to capture its slower response dynamics
Motion artifact gate — a learned scalar gate computed from ACC magnitude that suppresses corrupted BVP and EDA features during high-movement windows
Cross-modal attention — 4-head multi-head attention across all modality vectors, letting each sensor attend to all others
Auxiliary branches — an FFT-based HRV branch extracting VLF/LF/HF band power from BVP, and a global statistics branch computing mean, std, and range across all 6 channels


The final 304-dimensional representation (256 cross-modal + 16 HRV + 32 stats) is passed through a 3-layer classification head with dropout to produce stress/non-stress predictions.


Repo Structure

HAFN_FL_SV/
├── HAFN Implementation.ipynb       # Full implementation — preprocessing, training, evaluation
├── preprocessed_wrist_v9/          # Preprocessed WESAD wrist data (15 subjects)
│   ├── S2_chrono_v8.pkl
│   ├── S3_chrono_v8.pkl
│   └── ...                         # S2–S17 excluding S12
└── results_wrist_comparison/       # Result plots and CSVs
    ├── comparison_results.csv      # Accuracy and F1 for all 16 experiments
    ├── convergence_all.png         # Convergence curves across all strategies
    └── ...


Replication Guide

This notebook is designed to run on Google Colab. All standard dependencies (PyTorch, scikit-learn, scipy, numpy, pandas, matplotlib) come pre-installed on Colab. Only Flower needs to be installed.

1. Set Up Google Colab


Open HAFN Implementation.ipynb in Google Colab
Go to Runtime → Change runtime type → T4 GPU
Mount your Google Drive when prompted — the notebook saves preprocessed data and results there


2. Install Flower

The notebook has this commented out at the top — uncomment and run it once:

python!pip install "flwr==1.10.0"
!pip install "protobuf>=4.21.0,<5.0.0"
!pip install "ray==2.31.0"

3. Dataset

This project uses the WESAD dataset. You have two options:

Option A — Use the preprocessed files directly (recommended)

Upload the preprocessed_wrist_v9/ folder to your Google Drive under MyDrive/HAFN_FL_SV/preprocessed_wrist_v9/. The notebook will load these directly and you can skip to the training cells.

Option B — Preprocess from scratch


Download WESAD from the UCI ML Repository
Upload the zip to your Google Drive as MyDrive/archive (1).zip
Run Cell 2 (extraction) and Cell 3 (preprocessing) — this resamples all signals to 4Hz, applies 30-second non-overlapping windows, performs chronological 80/20 split, normalizes using StandardScaler, and applies minority class oversampling


4. Run the Notebook

The notebook is structured as follows — run cells in order:

CellWhat it doesCell 1Setup, constants, Drive mountCell 2WESAD extraction and wrist signal loadingCell 3Preprocessing — windowing, normalization, class balancingCell 4All model architectures (CNN, CNN-LSTM, CNN-LSTM-Attn, HAFN)Cell 5+Federated training — 4 models × 4 FL strategies (16 experiments)Final cellsEvaluation, convergence plots, per-client breakdown

5. Key Hyperparameters

ParameterValueCommunication rounds15Local epochs per round2Batch size32Learning rate1e-3OptimizerAdam (weight decay 1e-4)LR schedulerCosineAnnealingLRRandom seed42


Citation

If you use this work, please cite the WESAD dataset and the FedNova aggregation framework:


Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., Van Laerhoven, K. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. ICMI 2018.




Wang, J., Liu, Q., Liang, H., Joshi, G., Poor, H.V. Tackling the objective inconsistency problem in heterogeneous federated optimization. NeurIPS 2020.
