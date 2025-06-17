"""
Config file to tune model parameters and waveform parameters.
"""

CONFIG = {
    # Dataset/Model Params
    "input_length": 800,
    "L1": 4,   # BPM slots
    "L2": 2,   # BPSK slots
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
    "train_samples": 10000,
    "val_samples": 2000,
    "test_samples": 2000,

    # Runtime
    "device": "mps",
    "epochs": 20,
    "batch_size": 64,
    "hidden_dim": 256,
}

