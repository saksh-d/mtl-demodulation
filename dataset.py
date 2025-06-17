"""
Script for data generation + serialization. Stores output data in `/data/` directory.
"""

import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import hashlib
import numpy as np
from config import CONFIG

# OFDM utilities, PWM, modulation, etc. can go here or in utils/

class MCCDataset(Dataset):
    def __init__(self, config: dict, split: str = "train", regenerate=False):
        self.config = config
        self.split = split.lower()
        self.num_samples = config[f"{split}_samples"]
        self.cache_dir = Path("data")
        self.cache_dir.mkdir(exist_ok=True)

        # unique hash based on config
        self.dataset_path = self._generate_cache_path()

        if self.dataset_path.exists() and not regenerate:
            print(f"[INFO] Loading cached {split} dataset from {self.dataset_path}")
            data = torch.load(self.dataset_path)
            self.waveforms = data["waveforms"]
            self.phase_labels = data["phase_labels"]
            self.pos_labels = data["pos_labels"]
        else:
            print(f"[INFO] Generating {split} dataset from scratch...")
            self._generate_dataset()
            torch.save({
                "waveforms": self.waveforms,
                "phase_labels": self.phase_labels,
                "pos_labels": self.pos_labels
            }, self.dataset_path)

    def _generate_cache_path(self):
        cfg = {k: self.config[k] for k in sorted(self.config) if not k.endswith("_samples")}
        key = f"{self.split}_{hashlib.md5(str(cfg).encode()).hexdigest()[:8]}"
        return self.cache_dir / f"{key}.pt"

    def _generate_dataset(self):
        # TODO: replace this stub with your waveform generator
        L1, L2 = self.config["L1"], self.config["L2"]
        input_len = self.config["input_length"]

        self.waveforms = []
        self.phase_labels = []
        self.pos_labels = []

        for _ in range(self.num_samples):
            # Generate waveform with OFDM+PWM+beacon+noise
            waveform, phase, position = generate_mcc_sample(self.config)
            self.waveforms.append(waveform)
            self.phase_labels.append(phase)
            self.pos_labels.append(position)

        self.waveforms = torch.stack(self.waveforms)
        self.phase_labels = torch.tensor(self.phase_labels, dtype=torch.long)
        self.pos_labels = torch.tensor(self.pos_labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "waveform": self.waveforms[idx],
            "phase_label": self.phase_labels[idx],
            "position_label": self.pos_labels[idx]
        }

def generate_mcc_sample(config):
    """
    Data generation code, based on parameters set in config.py
    """
    input_len = config["input_length"]
    L1 = config["L1"]  # number of beacon slots (BPM)
    L2 = config["L2"]  # number of phases (BPSK)
    snr_db = config["snr_db"]
    dimming_level = config.get("dimming_level", 0.5)

    # Choose class labels
    pos_class = np.random.randint(0, L1)
    phase_class = np.random.randint(0, L2)  # assume L2=2 → 0 or 1

    # --- Step 1: Generate OFDM Background ---
    if config.get("ofdm_enabled", True):
        waveform = generate_ofdm_payload(config, input_len)
    else:
        waveform = np.zeros(input_len)

    # --- Step 2: Generate PWM Envelope (binary mask of duty cycle) ---
    envelope = np.ones(input_len) * dimming_level

    # --- Step 3: Inject Beacon Pulse ---
    slot_len = input_len // L1
    beacon_pwm = np.zeros(input_len)
    pulse_start = pos_class * slot_len
    pulse_end = (pos_class + 1) * slot_len

    # BPSK beacon: ±1 depending on phase
    beacon_value = +1.0 if phase_class == 0 else -1.0
    beacon_pwm[pulse_start:pulse_end] = beacon_value

    # Combine beacon and envelope
    final_waveform = envelope * waveform + beacon_pwm

    # --- Step 4: Add noise ---
    signal_power = np.mean(final_waveform ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(scale=np.sqrt(noise_power), size=input_len)
    final_waveform += noise

    return torch.tensor(final_waveform, dtype=torch.float32), phase_class, pos_class

def generate_ofdm_payload(config, input_len):
    """
    Creates OFDM symbols if "ofdm_enabled" is True in config.py
    """
    n_subcarriers = config["ofdm_params"].get("n_subcarriers", 64)
    cp_len = config["ofdm_params"].get("cyclic_prefix", 16)
    mod_type = config["ofdm_params"].get("modulation", "QPSK")

    symbol_len = n_subcarriers + cp_len
    n_symbols = input_len // symbol_len

    waveform = []

    for _ in range(n_symbols):
        bits = np.random.randint(0, 4, size=n_subcarriers)

        if mod_type.upper() == "QPSK":
            # Map to QPSK constellation: 0 → (1+1j), 1 → (-1+1j), etc.
            symbols = 1 - 2 * ((bits >> 1) & 1) + 1j * (1 - 2 * (bits & 1))
        elif mod_type.upper() == "BPSK":
            symbols = 1 - 2 * bits  # 0 → +1, 1 → -1
        else:
            raise ValueError("Unsupported modulation")

        time_signal = np.fft.ifft(symbols)
        with_cp = np.concatenate([time_signal[-cp_len:], time_signal])
        waveform.extend(np.real(with_cp))

    waveform = np.array(waveform[:input_len])
    return waveform

def _export_single_dataset(config, split="test", num_samples=1000, format="npz"):
    """
    Helper function to export dataset
    """
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)

    snr = config["snr_db"]
    key = f"{split}_L1_{config['L1']}_L2_{config['L2']}_SNR_{snr}_dim_{int(config['dimming_level']*10)}"
    filename = os.path.join(base_dir, f"{key}.{format}")

    waveforms, phase_labels, pos_labels = [], [], []

    for _ in range(num_samples):
        x, phase, pos = generate_mcc_sample(config)
        waveforms.append(x.numpy())
        phase_labels.append(phase)
        pos_labels.append(pos)

    waveforms = np.stack(waveforms)
    phase_labels = np.array(phase_labels)
    pos_labels = np.array(pos_labels)

    if format == "npz":
        np.savez_compressed(filename,
                            waveforms=waveforms,
                            phase_labels=phase_labels,
                            pos_labels=pos_labels)
        print(f"[✓] Saved {split} dataset to {filename}")
    elif format == "pt":
        torch.save({
            "waveforms": torch.tensor(waveforms),
            "phase_labels": torch.tensor(phase_labels),
            "pos_labels": torch.tensor(pos_labels),
        }, filename)
        print(f"[✓] Saved {split} dataset to {filename}")
    else:
        raise ValueError("Unsupported format. Use 'npz' or 'pt'.")

def save_datasets_per_snr():
    from config import CONFIG  # avoid circular imports
    for snr in CONFIG["snr_db_list"]:
        config_copy = CONFIG.copy()
        config_copy["snr_db"] = snr
        _export_single_dataset(
            config=config_copy,
            split="test",
            num_samples=CONFIG["test_samples"]
        )
        _export_single_dataset(
            config=config_copy,
            split="train",
            num_samples=CONFIG["train_samples"]
        )

if __name__ == "__main__":
    save_datasets_per_snr()
