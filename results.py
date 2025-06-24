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
    Para cada subcarpeta en root_dir que contenga training_curve.json y summary.json:
    - genera 3 gráficos
    - guarda en su carpeta
    - opcionalmente hace git add, commit y push
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

                # Extraer curvas
                epochs = [entry["epoch"] for entry in data]
                losses = [entry["loss"] for entry in data]
                temporal_losses = [entry["temporal_loss"] for entry in data]
                lambdas = [entry["lambda"] for entry in data]
                lambda_errors = [abs(l - lambda_true) for l in lambdas]

                # Función para graficar y guardar
                def save_plot(y, ylabel, title, filename, color='blue'):
                    path = os.path.join(subdir, filename)
                    plt.figure(figsize=(7, 4))
                    plt.plot(epochs, y, color=color, linewidth=2)
                    plt.xlabel("Epochs")
                    plt.ylabel(ylabel)
                    plt.title(title)
                    plt.grid(True)
                    plt.savefig(path)
                    plt.close()
                    plot_files.append(path)

                # Crear los 3 gráficos
                save_plot(losses, "Loss", "Loss vs Epochs", "loss_vs_epochs.png")
                save_plot(temporal_losses, "Temporal Loss", "Temporal Loss vs Epochs", "temporal_loss_vs_epochs.png", color='orange')
                save_plot(lambda_errors, r"|$\lambda_{est} - \lambda_{true}$|", "Lambda Error vs Epochs", "lambda_error_vs_epochs.png", color='red')

            except Exception as e:
                print(f"⚠ Error procesando {subdir}: {e}")

    # Git add + commit + push
    if push_to_git and plot_files:
        try:
            subprocess.run(["git", "add"] + plot_files, check=True)
            subprocess.run(["git", "commit", "-m", "Add training plots using training_curve and summary"], check=True)
            subprocess.run(["git", "push"], check=True)
            print("🚀 Gráficas subidas a GitHub con éxito.")
        except subprocess.CalledProcessError as e:
            print(f"⚠ Error al hacer push a GitHub: {e}")
