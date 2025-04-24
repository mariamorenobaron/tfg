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

MODEL_DIRS = [
    "saved_model_mlp_d1",
    "saved_model_resnet_d1",
    "saved_model_mlp_d2"
]

def evaluate_model(folder):
    # Cargar modelo + arquitectura
    model_class_dict = {"MLP": MLP, "ResNet": ResNet}
    model, arch_cfg = load_model(model_class_dict, folder=folder)

    # Restaurar config base + arquitectura
    config = CONFIG.copy()
    config["architecture"] = arch_cfg["architecture"]
    config["depth"] = arch_cfg["depth"]
    config["width"] = arch_cfg["width"]
    config["input_dim"] = arch_cfg["input_dim"]

    # Instanciar PINN y evaluar
    pinn = PowerMethodPINN(model, config)
    start = time.time()
    x_eval = pinn.sample_points(5000).detach().cpu().numpy()
    x_tensor = torch.tensor(x_eval, dtype=torch.float32).to(pinn.device)
    x_input = pinn.apply_input_transform(x_tensor)

    with torch.no_grad():
        u_pred = model(x_input).cpu().numpy()

    elapsed = time.time() - start
    u_true = config["exact_u"](x_eval)
    u_pred /= np.linalg.norm(u_pred)
    u_true /= np.linalg.norm(u_true)
    l2_error = np.linalg.norm(u_pred - u_true) / np.sqrt(u_pred.shape[0])

    lambda_pred = pinn.lambda_
    lambda_true = config["lambda_true"]
    lambda_error = np.abs(lambda_pred - lambda_true)

    return {
        "folder": folder,
        "arch": config["architecture"],
        "depth": config["depth"],
        "width": config["width"],
        "L2_u_error": float(l2_error),
        "lambda_pred": float(lambda_pred),
        "lambda_true": float(lambda_true),
        "lambda_error": float(lambda_error),
        "eval_time": float(elapsed)
    }

if __name__ == "__main__":
    results = []

    for folder in MODEL_DIRS:
        if os.path.exists(folder):
            print(f" Evaluating: {folder}")
            metrics = evaluate_model(folder)
            results.append(metrics)
        else:
            print(f" Folder not found: {folder}")

    df = pd.DataFrame(results)
    df.to_csv("results_summary.csv", index=False)
    print("\n Results saved to results_summary.csv")
    print(df)
