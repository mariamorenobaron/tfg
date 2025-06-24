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

def process_all_training_curves(root_dir):
    """
    Recorre subdirectorios con training_curve.json y summary.json,
    genera gráficos:
        - Loss vs Epochs
        - Temporal Loss vs Epochs
        - Lambda Error vs Epochs (con lambda_true leído del summary.json)
    Devuelve resumen con errores.
    """
    summaries = []

    for subdir, _, files in os.walk(root_dir):
        if "training_curve.json" in files and "summary.json" in files:
            training_path = os.path.join(subdir, "training_curve.json")
            summary_path = os.path.join(subdir, "summary.json")
            print(f"📊 Procesando: {training_path}")

            try:
                # Leer training data
                with open(training_path, "r") as f:
                    training_data = json.load(f)

                # Leer lambda_true desde summary
                with open(summary_path, "r") as f:
                    summary_data = json.load(f)
                lambda_true = summary_data.get("lambda_true", None)
                if lambda_true is None:
                    print(f" lambda_true no encontrado en {summary_path}")
                    continue

                # Extraer curvas
                epochs = [entry["epoch"] for entry in training_data]
                losses = [entry["loss"] for entry in training_data]
                temporal_losses = [entry["temporal_loss"] for entry in training_data]
                lambdas = [entry["lambda"] for entry in training_data]
                lambda_errors = [abs(l - lambda_true) for l in lambdas]

                # Función auxiliar de graficado
                def save_plot(y, ylabel, title, filename, color='blue'):
                    plt.figure(figsize=(7, 4))
                    plt.plot(epochs, y, color=color, linewidth=2)
                    plt.xlabel("Epochs")
                    plt.ylabel(ylabel)
                    plt.title(title)
                    plt.grid(True)
                    plt.savefig(os.path.join(subdir, filename))
                    plt.close()

                # Guardar gráficos
                save_plot(losses, "Loss", "Loss vs Epochs", "loss_vs_epochs.png")
                save_plot(temporal_losses, "Temporal Loss", "Temporal Loss vs Epochs", "temporal_loss_vs_epochs.png", color='orange')
                save_plot(lambda_errors, "|λ_est - λ_true|", "Lambda Absolute Error vs Epochs", "lambda_error_vs_epochs.png", color='red')

                # Calcular métrica resumen
                lambda_error_inf = max(abs(lambda_errors[i+1] - lambda_errors[i]) for i in range(len(lambda_errors)-1))

                summaries.append({
                    "folder": os.path.basename(subdir),
                    "lambda_true": lambda_true,
                    "lambda_est_final": lambdas[-1],
                    "final_temporal_loss": temporal_losses[-1],
                    "lambda_error_inf": lambda_error_inf
                })

            except Exception as e:
                print(f"⚠ Error procesando {subdir}: {e}")

    return pd.DataFrame(summaries)
