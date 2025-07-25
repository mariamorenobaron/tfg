import os
import time
import json
import torch
import tracemalloc
import shutil
import numpy as np
import subprocess
from models import MLP, ResNet
from pinn_power import PowerMethodPINN
from pinn_inverse_power import InversePowerMethodPINN
from utils import maybe_push_to_git

def run_model(config, save_dir='numerical_experiments'):

    base_name = f"{config['method']}_{config['architecture']}_{config['dimension']}D_d{config['depth']}_w{config['width']}"
    #base_name = f"{config['method']}_{config['architecture']}_{config['dimension']}D_d{config['depth']}_w{config['width']}_epochs{config['adam_steps']}"
    #base_name = f"{config['method']}_{config['architecture']}_{config['dimension']}D_d{config['depth']}_w{config['width']}_alpha{config['alpha']}"
    print(f"[INFO] Running model with base name: {base_name}")
    run_dir = os.path.join(save_dir, base_name)
    os.makedirs(run_dir, exist_ok=True)
    config["save_dir"] = run_dir

    if config.get("use_seed", False):
        seed = config.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[INFO] Using fixed seed: {seed}")
    else:
        print("[INFO] Random seed is not fixed.")

    input_dim = config["dimension"] * (2 * config.get("pbc_k", 1) if config.get("periodic", False) else 1)
    config["input_dim"] = input_dim

    if config["architecture"].lower() == 'mlp':
        layers = [input_dim] + [config["width"]] * config["depth"] + [1]
        model = MLP(layers)
    elif config["architecture"].lower() == 'resnet':
        model = ResNet(in_num=input_dim, out_num=1, block_layers=[config["width"]] * 2, block_num=config["depth"])
    else:
        raise ValueError("Unknown architecture.")

    method = config["method"].lower()
    if method == "pmnn":
        pinn = PowerMethodPINN(model.double(), config)
    elif method == "ipmnn":
        pinn = InversePowerMethodPINN(model.double(), config)
    else:
        raise ValueError("Unknown method: choose 'pmnn' or 'ipmnn'.")

    # Start training
    tracemalloc.start()
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if config["optimizer"].lower() == "adam":
        pinn.optimize_adam()

    else:
        raise ValueError("Unknown optimizer.")

    elapsed = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    # Save model
    torch.save(pinn.model.state_dict(), os.path.join(run_dir, "model.pt"))



    # Save training summary
    summary = {
        "architecture": config["architecture"],
        "depth": config["depth"],
        "width": config["width"],
        "optimizer": config["optimizer"],
        "method": config["method"],
        "epochs": config.get("adam_steps", None),
        "data_points": config.get("n_train", None),
        "lambda_pred": float(pinn.best_lambda),
        "lambda_true": float(config["lambda_true"]),
        "best_iteration": pinn.best_iteration if hasattr(pinn, 'best_iteration') else None,
        "min_loss": float(pinn.min_loss),
        "time_seconds": elapsed,
        "peak_ram_MB": peak / 1024 / 1024,
        "peak_gpu_MB": peak_gpu,
        "device": str(pinn.device)
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Save training curve
    pinn.save_training_curve()

    # Optional GitHub push
    if config.get("push_to_git", True):
        maybe_push_to_git(run_dir, message= f"Training completed for {base_name} in {elapsed:.2f} seconds.")

    return pinn


def run_model_all_criteria(config, save_dir='numerical_experiments'):

    base_name = f"{config['method']}_{config['architecture']}_{config['dimension']}D_d{config['depth']}_w{config['width']}"
    print(f"[INFO] Running model with base name: {base_name}")

    config["save_dir"] = None  # no guardes en run_dir

    if config.get("use_seed", False):
        seed = config.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[INFO] Using fixed seed: {seed}")
    else:
        print("[INFO] Random seed is not fixed.")

    input_dim = config["dimension"] * (2 * config.get("pbc_k", 1) if config.get("periodic", False) else 1)
    config["input_dim"] = input_dim

    if config["architecture"].lower() == 'mlp':
        layers = [input_dim] + [config["width"]] * config["depth"] + [1]
        model = MLP(layers)
    elif config["architecture"].lower() == 'resnet':
        model = ResNet(in_num=input_dim, out_num=1, block_layers=[config["width"]] * 2, block_num=config["depth"])
    else:
        raise ValueError("Unknown architecture.")

    method = config["method"].lower()
    if method == "pmnn":
        pinn = PowerMethodPINN(model.double(), config)
    elif method == "ipmnn":
        pinn = InversePowerMethodPINN(model.double(), config)
    else:
        raise ValueError("Unknown method: choose 'pmnn' or 'ipmnn'.")

    tracemalloc.start()
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if config["optimizer"].lower() == "adam":
        pinn.optimize_adam()
    else:
        raise ValueError("Unknown optimizer.")

    elapsed = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    # Definir carpetas de salida para cada criterio
    criteria = [
        ("loss", pinn.best_model_loss, pinn.best_lambda_loss, pinn.min_loss, pinn.best_iteration_loss),
        ("loss_temporal", pinn.best_model_temporal_loss, pinn.best_lambda_temporal_loss, pinn.min_temporal_loss, pinn.best_iteration_temporal_loss),
        ("loss_combined1", pinn.best_model_combined_loss1, pinn.best_lambda_combined_loss1, pinn.min_combined_loss1, pinn.best_iteration_combined_loss1),
        ("loss_combined", pinn.best_model_combined_loss, pinn.best_lambda_combined_loss, pinn.min_combined_loss, pinn.best_iteration_combined_loss),
    ]

    for name, model_state, lambda_val, loss_val, iteration in criteria:
        if model_state is None:
            print(f"[WARNING] No model found for criterion '{name}'. Skipping export.")
            continue

        export_dir = os.path.join(save_dir, name, base_name)
        os.makedirs(export_dir, exist_ok=True)

        # Guardar modelo (state dict)
        torch.save(model_state, os.path.join(export_dir, "model.pt"))

        # Guardar training curve directamente en carpeta correcta
        pinn.save_training_curve(export_dir)

        # Guardar resumen
        summary = {
            "criterion": name,
            "architecture": config["architecture"],
            "depth": config["depth"],
            "width": config["width"],
            "optimizer": config["optimizer"],
            "method": config["method"],
            "dimension": config["dimension"],
            "epochs": config.get("adam_steps", None),
            "data_points": config.get("n_train", None),
            "lambda_pred": float(lambda_val),
            "lambda_true": float(config["lambda_true"]),
            "best_iteration": iteration,
            "loss_value": float(loss_val),
            "time_seconds": elapsed,
            "peak_ram_MB": peak / 1024 / 1024,
            "peak_gpu_MB": peak_gpu,
            "device": str(pinn.device)
        }

        with open(os.path.join(export_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        print(f"[INFO] Exported model for '{name}' to: {export_dir}")

    # Git push opcional (si se quiere, lo puedes quitar)
    if config.get("push_to_git", True):
        maybe_push_to_git(save_dir, message=f"Training completed for {base_name} in {elapsed:.2f} seconds.")
