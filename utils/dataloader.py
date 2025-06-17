import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

class NPZDataset(Dataset):
    def __init__(self, filepath):
        data = np.load(filepath)
        self.waveforms = torch.tensor(data["waveforms"], dtype=torch.float32)
        self.phase_labels = torch.tensor(data["phase_labels"], dtype=torch.long)
        self.pos_labels = torch.tensor(data["pos_labels"], dtype=torch.long)

    def __len__(self):
        return self.waveforms.shape[0]

    def __getitem__(self, idx):
        return {
            "waveform": self.waveforms[idx],
            "phase_label": self.phase_labels[idx],
            "position_label": self.pos_labels[idx]
        }

def load_dataloader_npz(filepath, batch_size=64, shuffle=True):
    dataset = NPZDataset(filepath)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader

def auto_find_dataset(config, split="test", format="npz"):
    """
    Search for a dataset file in ./data/ that matches the config.
    Returns full file path if found, else raises FileNotFoundError.
    """
    base_dir = "data"
    snr = config.get("snr_db", None)
    if snr is None:
        raise ValueError("config must contain 'snr_db' to search dataset.")

    dim_level = int(config["dimming_level"] * 10)
    pattern = f"{split}_L1_{config['L1']}_L2_{config['L2']}_SNR_{snr}_dim_{dim_level}.{format}"
    search_path = os.path.join(base_dir, pattern)

    matches = glob.glob(search_path)
    if not matches:
        raise FileNotFoundError(f"No dataset file found for: {pattern}. Please run dataset.py first.")
    
    print(f"[✓] Found dataset file: {matches[0]}")
    return matches[0]