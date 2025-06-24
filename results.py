import os
import json
import time
import numpy as np
import pandas as pd
import torch
from config import CONFIG
from model import MLP, ResNet
from pinn_power import PowerMethodPINN
from utils import load_model
import matplotlib.pyplot as plt
import subprocess

def generate_plots_from_training_and_push(root_dir, push_to_git=True):
    """
    Para cada subcarpeta en root_dir con training_curve.json y summary.json:
    - genera 3 gráficos con estilo profesional
    - guarda los .png
    - git add + commit + push si se desea
    """
    plot_files = []

    for subdir, _, files in os.walk(root_dir):
        if "training_curve.json" in files and "summary.json" in files:
            training_path = os.path.join(subdir, "training_curve.json")
            summary_path = os.path.join(subdir, "summary.json")
            print(f"📊 Procesando: {training_path}")

            try:
                with open(training_path, "r") as f:
                    data = json.load(f)
                with open(summary_path, "r") as f:
                    summary = json.load(f)

                lambda_true = summary.get("lambda_true")
                if lambda_true is None:
                    print(f"⚠ No se encontró lambda_true en {summary_path}")
                    continue

                epochs = [entry["epoch"] for entry in data]
                losses = [entry["loss"] for entry in data]
                temporal_losses = [entry["temporal_loss"] for entry in data]
                lambdas = [entry["lambda"] for entry in data]
                lambda_errors = [abs(l - lambda_true) for l in lambdas]

                def styled_plot(y, ylabel, title, filename, color='blue', log_y=False):
                    path = os.path.join(subdir, filename)
                    plt.figure(figsize=(5, 4))
                    if log_y:
                        plt.semilogy(epochs, y, color=color, linewidth=1.8, label=ylabel)
                    else:
                        plt.plot(epochs, y, color=color, linewidth=1.8, label=ylabel)
                    plt.xlabel(r"$k$", fontsize=12)
                    plt.ylabel(ylabel, fontsize=12)
                    plt.legend(fontsize=11)
                    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
                    plt.tight_layout()
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files.append(path)

                styled_plot(losses, r"$\mathcal{L}$", "Loss vs Epochs", "loss_vs_epochs.png", color='navy')
                styled_plot(temporal_losses, r"$\mathcal{L}_{\mathrm{temp}}$", "Temporal Loss vs Epochs", "temporal_loss_vs_epochs.png", color='darkorange')
                styled_plot(lambda_errors, r"$|\lambda_{\mathrm{est}} - \lambda_{\mathrm{true}}|$", "Lambda Error vs Epochs", "lambda_error_vs_epochs.png", color='crimson', log_y=True)

            except Exception as e:
                print(f"⚠ Error procesando {subdir}: {e}")

    # Git operations
    if push_to_git and plot_files:
        try:
            subprocess.run(["git", "add"] + plot_files, check=True)
            subprocess.run(["git", "commit", "-m", "Add high-quality training plots"], check=True)
            subprocess.run(["git", "push"], check=True)
            print("🚀 Gráficas subidas a GitHub con éxito.")
        except subprocess.CalledProcessError as e:
            print(f"⚠ Error al hacer push a GitHub: {e}")