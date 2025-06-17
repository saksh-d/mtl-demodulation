# evaluate.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from config import CONFIG
import random
from pathlib import Path
    
from utils.dataloader import auto_find_dataset, load_dataloader_npz
from models.shared import SharedMTL
from models.split import SplitMTL
from models.cross_stitch import CrossStitchMTL
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torchinfo import summary
from thop import profile


MODELS = ["shared", "split", "cross_stitch"]
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_logs(model_name):
    log_csv = os.path.join(CHECKPOINT_DIR, f"{model_name}_log.csv")
    log_pt = os.path.join(CHECKPOINT_DIR, f"{model_name}_log.pt")

    df = pd.read_csv(log_csv)
    alpha_history = []

    if model_name == "cross_stitch":
        pt_data = torch.load(log_pt, weights_only=False)
        alpha_history = pt_data.get("alpha", [])

    return df, alpha_history

def plot_loss_accuracy(df, model_name):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.title(f"{model_name}: Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["phase_acc"], label="Phase")
    plt.plot(df["epoch"], df["pos_acc"], label="Position")
    plt.xlabel("Epoch")
    plt.title(f"{model_name}: Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_metrics.png")
    plt.close()

def plot_alpha_trend(alpha_log):
    scores = []
    for alpha in alpha_log:
        cross = alpha["a12"] + alpha["a21"]
        total = alpha["a11"] + alpha["a12"] + alpha["a21"] + alpha["a22"]
        scores.append(cross / total)

    plt.figure()
    plt.plot(scores, marker="o")
    plt.title("Cross-Stitch Coupling Score Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Coupling Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/coupling_trend.png")
    plt.close()

def plot_confusion_matrix(model_name):
    print(f"[✓] Generating confusion matrix for {model_name}...")

    # Setup
    CONFIG["snr_db"] = random.choice(CONFIG["snr_db_list"])
    model_map = {
        "shared": SharedMTL,
        "split": SplitMTL,
        "cross_stitch": CrossStitchMTL
    }

    model_class = model_map[model_name]
    model = model_class(CONFIG["input_length"], CONFIG["hidden_dim"], CONFIG["L2"], CONFIG["L1"])
    model.load_state_dict(torch.load(f"checkpoints/{model_name}.pt"))
    model.eval()

    file = auto_find_dataset(CONFIG, split="test")
    _, test_loader = load_dataloader_npz(file, batch_size=64, shuffle=False)

    y_true_phase, y_pred_phase = [], []
    y_true_pos, y_pred_pos = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['waveform']
            out_phase, out_pos = model(x)
            y_true_phase.extend(batch['phase_label'].tolist())
            y_pred_phase.extend(out_phase.argmax(dim=1).tolist())

            y_true_pos.extend(batch['position_label'].tolist())
            y_pred_pos.extend(out_pos.argmax(dim=1).tolist())

    # Plot
    for name, y_true, y_pred in [
        ("phase", y_true_phase, y_pred_phase),
        ("position", y_true_pos, y_pred_pos)
    ]:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"{model_name} {name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{model_name}_cm_{name}.png")
        plt.close()

def compare_models_plot():
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for model_name in MODELS:
        df = pd.read_csv(f"{CHECKPOINT_DIR}/{model_name}_log.csv")
        ax[0].plot(df["epoch"], df["loss"], label=model_name)
        ax[1].plot(df["epoch"], df["phase_acc"], label=f"{model_name} - Phase")
        ax[1].plot(df["epoch"], df["pos_acc"], label=f"{model_name} - Pos")

    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].grid(True)

    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "compare_models_metrics.png")
    plt.close()

def evaluate_model_per_snr(model_name):
    print(f"[✓] Evaluating accuracy vs SNR for {model_name}")
    
    model_map = {
        "shared": SharedMTL,
        "split": SplitMTL,
        "cross_stitch": CrossStitchMTL
    }

    model_class = model_map[model_name]
    model = model_class(CONFIG["input_length"], CONFIG["hidden_dim"], CONFIG["L2"], CONFIG["L1"])
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/{model_name}.pt"))
    model.eval()

    snr_list = CONFIG["snr_db_list"]
    phase_accs, pos_accs = [], []

    with torch.no_grad():
        for snr in snr_list:
            CONFIG["snr_db"] = snr
            file = auto_find_dataset(CONFIG, split="test")
            print(f"→ SNR {snr} | Loaded: {file}")
            _, loader = load_dataloader_npz(file, batch_size=64, shuffle=False)

            correct_phase, correct_pos, total = 0, 0, 0

            for batch in loader:
                x = batch['waveform']
                y_phase = batch['phase_label']
                y_pos = batch['position_label']
                out_phase, out_pos = model(x)

                correct_phase += (out_phase.argmax(1) == y_phase).sum().item()
                correct_pos += (out_pos.argmax(1) == y_pos).sum().item()
                total += x.size(0)

            phase_accs.append(correct_phase / total)
            pos_accs.append(correct_pos / total)

    return snr_list, phase_accs, pos_accs

def plot_accuracy_vs_snr(results_dict):
    plt.figure(figsize=(8, 4))
    for model_name, (snrs, phase_accs, pos_accs) in results_dict.items():
        plt.plot(snrs, phase_accs, marker='o', label=f"{model_name} - Phase")
        plt.plot(snrs, pos_accs, marker='s', label=f"{model_name} - Pos")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. SNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_vs_snr.png")
    plt.close()

def compute_model_complexity(model_name):
    print(f"[✓] Computing complexity for {model_name}")
    
    model_map = {
        "shared": SharedMTL,
        "split": SplitMTL,
        "cross_stitch": CrossStitchMTL
    }

    model_class = model_map[model_name]
    model = model_class(CONFIG["input_length"], CONFIG["hidden_dim"], CONFIG["L2"], CONFIG["L1"])
    model.eval()

    # Dummy input for profiling
    dummy_input = torch.randn(1, CONFIG["input_length"])

    # Compute params and FLOPs
    try:
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    except Exception as e:
        print(f"[!] FLOP profiling failed: {e}")
        flops = None

    try:
        param_count = sum(p.numel() for p in model.parameters())
    except:
        param_count = "N/A"

    return flops, param_count

def export_complexity_report():
    report_path = RESULTS_DIR / "complexity_report.txt"
    with open(report_path, "w") as f:
        f.write("Model Complexity Report\n")
        f.write("=======================\n\n")
        for model_name in MODELS:
            flops, params = compute_model_complexity(model_name)
            f.write(f"{model_name.upper()}:\n")
            f.write(f"  Parameters: {params:,}\n")
            if flops:
                f.write(f"  FLOPs:      {flops / 1e6:.2f} MFLOPs\n")
            else:
                f.write(f"  FLOPs:      [not available]\n")
            f.write("\n")
    print(f"[✓] Complexity report saved to {report_path}")

def plot_alpha_elements(alpha_log):
    a11 = [a["a11"] for a in alpha_log]
    a12 = [a["a12"] for a in alpha_log]
    a21 = [a["a21"] for a in alpha_log]
    a22 = [a["a22"] for a in alpha_log]

    epochs = list(range(1, len(alpha_log) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, a11, label="α11 (Phase→Phase)")
    plt.plot(epochs, a12, label="α12 (Pos→Phase)")
    plt.plot(epochs, a21, label="α21 (Phase→Pos)")
    plt.plot(epochs, a22, label="α22 (Pos→Pos)")

    plt.title("Cross-Stitch Alpha Weights Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Alpha Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/alpha_elements_over_epochs.png")
    plt.close()

def main():
    for model_name in MODELS:
        df, alpha_history = load_logs(model_name)
        plot_loss_accuracy(df, model_name)
        if model_name == "cross_stitch":
            plot_alpha_trend(alpha_history)
            plot_alpha_elements(alpha_history)
        plot_confusion_matrix(model_name)

    compare_models_plot()

    # Evaluate per-SNR accuracy
    results = {}
    for model_name in MODELS:
        results[model_name] = evaluate_model_per_snr(model_name)
    plot_accuracy_vs_snr(results)


    # Export complexity
    export_complexity_report()

if __name__ == "__main__":
    main()
