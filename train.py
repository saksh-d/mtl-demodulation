# train.py

import torch
import torch.nn as nn
from config import CONFIG
from models.shared import SharedMTL
from models.split import SplitMTL
from models.cross_stitch import CrossStitchMTL
from utils.dataloader import auto_find_dataset, load_dataloader_npz
import os
import pandas as pd

def train(model, dataloader, epochs, device, save_name):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    loss_log = []
    phase_acc_log = []
    pos_acc_log = []
    alpha_log = []

    for epoch in range(epochs):
        model.train()
        total_loss, correct_phase, correct_pos, total = 0, 0, 0, 0

        for batch in dataloader:
            x = batch['waveform'].to(device)
            y_phase = batch['phase_label'].to(device)
            y_pos = batch['position_label'].to(device)

            optimizer.zero_grad()
            out_phase, out_pos = model(x)
            loss = criterion(out_phase, y_phase) + criterion(out_pos, y_pos)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_phase += (out_phase.argmax(1) == y_phase).sum().item()
            correct_pos += (out_pos.argmax(1) == y_pos).sum().item()
            total += x.size(0)

        epoch_loss = total_loss
        epoch_phase_acc = correct_phase / total
        epoch_pos_acc = correct_pos / total

        loss_log.append(epoch_loss)
        phase_acc_log.append(epoch_phase_acc)
        pos_acc_log.append(epoch_pos_acc)

        if hasattr(model, 'cross_stitch'):
            alpha_matrix = model.cross_stitch.alpha.detach().cpu().numpy()
            alpha_log.append({
                "a11": alpha_matrix[0, 0],
                "a12": alpha_matrix[0, 1],
                "a21": alpha_matrix[1, 0],
                "a22": alpha_matrix[1, 1]
            })


        print(f"[{save_name} | Epoch {epoch+1}] "
              f"Loss: {epoch_loss:.4f} | "
              f"Phase Acc: {epoch_phase_acc:.4f} | "
              f"Pos Acc: {epoch_pos_acc:.4f}")

    # Save model + logs
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{save_name}.pt")

    # Save logs to .pt
    torch.save({
        "loss": loss_log,
        "phase_acc": phase_acc_log,
        "pos_acc": pos_acc_log,
        "alpha": alpha_log
    }, f"checkpoints/{save_name}_log.pt")

    # Save logs to .csv
    df = pd.DataFrame({
        "epoch": list(range(1, len(loss_log) + 1)),
        "loss": loss_log,
        "phase_acc": phase_acc_log,
        "pos_acc": pos_acc_log
    })
    df.to_csv(f"checkpoints/{save_name}_log.csv", index=False)

    return model

if __name__ == "__main__":
    device = CONFIG["device"]
    input_dim = CONFIG["input_length"]
    hidden_dim = CONFIG["hidden_dim"]
    L1 = CONFIG["L1"]
    L2 = CONFIG["L2"]

    CONFIG["snr_db"] = 10  # pick a training SNR
    dataset_path = auto_find_dataset(CONFIG, split="train")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
   
    _, train_loader = load_dataloader_npz(dataset_path, batch_size=CONFIG["batch_size"])

    shared_model = SharedMTL(input_dim, hidden_dim, L2, L1)
    split_model = SplitMTL(input_dim, hidden_dim, L2, L1)
    xstitch_model = CrossStitchMTL(input_dim, hidden_dim, L2, L1)

    train(shared_model, train_loader, CONFIG["epochs"], device, "shared")
    train(split_model, train_loader, CONFIG["epochs"], device, "split")
    train(xstitch_model, train_loader, CONFIG["epochs"], device, "cross_stitch")
