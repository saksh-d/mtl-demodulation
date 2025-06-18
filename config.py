"""
Config file to tune model parameters and waveform parameters.
"""

CONFIG = {
    # Dataset/Model Params
    "input_length": 800,
    "L1": 4,   # BePM slots
    "L2": 2,   # BePSK slots
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
    "train_samples": 50000,
    "val_samples": 5000,
    "test_samples": 5000,

    # Runtime
    "device": "cuda",
    "epochs": 20,
    "batch_size": 32,
    "hidden_dim": 32,
}

