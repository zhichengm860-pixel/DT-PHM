# DT-PHM: Digital Twin-Driven Intelligent Fault Diagnosis and RUL Prediction

Source code for the paper: *"Digital Twin-Driven Intelligent Fault Diagnosis and RUL Prediction for Key Components of High-Dynamic Weapon Equipment"*

## Overview

This repository implements the DT-PHM closed-loop framework, which unifies physics constraints, data adaptation, and online evolution for Prognostics and Health Management (PHM) of high-dynamic systems. The framework establishes a four-layer closed-loop architecture comprising physical system mapping, digital twin, intelligent analytics, and feedback update.

## Key Features

- **Physics-Data Collaborative Modeling**: Physics baseline model (exponential degradation) + LSTM residual learner with attention-based adaptive fusion
- **Physics-Constrained Loss Function**: Boundary constraint + monotonicity constraint to ensure physically consistent predictions
- **Closed-Loop Update Mechanism**: Online model correction with optional EWC and experience replay strategies
- **Fault Diagnosis**: BiLSTM with Focal Loss for health stage classification

## Project Structure

```
DT-PHM/
├── models/
│   └── models.py              # Model definitions (LSTM, BiLSTM, GRU, CNN-LSTM, Transformer, DT-PHM)
├── data/
│   ├── data_loader.py         # C-MAPSS and N-CMAPSS data loaders
│   └── data_loader_v2.py      # Optimized loader with sampling support for large datasets
├── utils/
│   └── train_engine.py        # Training engine with metrics tracking and early stopping
├── experiments/
│   ├── exp1_closed_loop.py    # Closed-loop vs. open-loop update strategy comparison
│   ├── exp2_physics_ablation.py # Mechanism-data collaborative modeling ablation
│   ├── exp3_fault_diagnosis.py  # Fault diagnosis with BiLSTM+Focal Loss
│   ├── exp4_rul_prediction.py   # RUL prediction on C-MAPSS FD001-FD004
│   └── exp5_robustness.py       # Noise robustness and cross-condition tests
├── figures/
│   └── draw_figures.py        # Figure generation scripts
├── run_all_experiments.py     # Run all experiments
├── requirements.txt
└── .gitignore
```

## Datasets

This code uses two publicly available NASA datasets:

- **C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation): Available from [NASA Prognostics Data Repository](https://data.nasa.gov/Aerospace/C-MAPSS-Aircraft-Engine-Simulator-Data/cg5s-ayye)
- **N-CMAPSS** (New C-MAPSS): Available from [NASA Prognostics Data Repository](https://data.nasa.gov/Aerospace/N-CMAPSS-Aircraft-Engine-Simulator-Data/j2x6-y98e)

Download the datasets and place them in `./C_MAPSS/` and `./N_CMAPSS/` directories respectively.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all experiments:

```bash
python run_all_experiments.py
```

Or run individual experiments:

```bash
python experiments/exp4_rul_prediction.py
```

Generate figures:

```bash
python figures/draw_figures.py
```

## Main Results

| Method | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|--------|-----------|-----------|-----------|-----------|
| LSTM | 16.16 | 22.36 | 15.34 | 23.05 |
| Transformer | 19.51 | 19.11 | 13.31 | 16.69 |
| DT-PHM (Complete) | **16.34** | **22.39** | **14.97** | 22.65 |

- Closed-loop update achieves **9.5% RMSE improvement** over open-loop on N-CMAPSS
- Physics-constrained loss reduces physically inconsistent predictions from **5.4% to 1.1%**
- Fault diagnosis accuracy reaches **73.56%** on N-CMAPSS

## License

This project is licensed under the MIT License.
