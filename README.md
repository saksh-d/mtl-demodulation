# Multi-Task Learning for VLC Waveform Demodulation

This repository contains a modular, PyTorch-based implementation of a multi-task learning (MTL) framework for demodulating mixed carrier communication (MCC) VLC waveforms. It explores phase (BPSK) and position (BPM) classification using shared, split, and cross-stitch network architectures.

---

## Key Features

- Multi-task classification of beacon phase (BePSK) and beacon position (BePM)
- Supports **Shared**, **Split**, and **Cross-Stitch** MTL architectures
- Tracks task coupling and alpha weights across epochs
- Modular training and evaluation pipeline
- Complexity analysis (params, FLOPs) per model
- Accuracy vs SNR plots
- Cross-stitch coupling visualization and α matrix tracking

---

## Getting Started

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Generate dataset
```bash
python generate_data.py
```
This will save compressed `.npz` files inside the `data/` folder for each SNR in `config.py`.

3. Train all models
```bash
python train.py
```
- Trains `Shared`, `Split`, and `Cross-Stitch` MTL models
- Logs training metrics, α values, and saves model weights
- Outputs results to `checkpoints/`

4. Evaluate
```bash
python evaluate.py
```
This runs:
- Accuracy & loss curves
- Accuracy vs. SNR plots
- Confusion matrices
- Task coupling trend
- Model complexity report (params + FLOPs)

All outputs saved in `results/`

---

## Configuration

All parameters are set in `config.py`:

```python
CONFIG = {
    # Dataset/Model Params
    "input_length": 800,
    "L1": 8,   # BePM slots
    "L2": 4,   # BePSK slots
    "dimming_level": 0.5,

    # OFDM Settings
    "ofdm_enabled": True,
    "ofdm_params": {
        "n_subcarriers": 64,
        "cyclic_prefix": 16,
        "modulation": "QPSK"  # only "BPSK" and "QPSK" supported at the moment
    },

    # SNR
    "snr_db_list": [0, 5, 10, 15, 20],

    # Sample Sizes
    "train_samples": 20000,
    "val_samples": 5000,
    "test_samples": 5000,

    # Runtime
    "device": "cuda",   # or "mps" or "cpu"
    "epochs": 40,
    "batch_size": 32,
    "hidden_dim": 128,
}
```
---

## References:

1. R. Ahmad, H. Elgala, S. Almajali, H. Bany Salameh and M. Ayyash, "Unified Physical-Layer Learning Framework Toward VLC-Enabled 6G Indoor Flying Networks," in IEEE Internet of Things Journal, vol. 11, no. 3, pp. 5545-5557, 1 Feb.1, 2024, doi: 10.1109/JIOT.2023.3307224
2. S. Dewan and H. Elgala, "Towards Low Complexity VLC Systems: A Multi-Task Learning Approach" in Comm&Optics Connect, 1 (Article ID: 0002), Sep.28, 2024, doi: 10.69709/COConnect.2024.097213 

---
### Author

Developed by Saksham Dewan — based on experimental waveform modeling and MTL research for physical layer signal understanding.
