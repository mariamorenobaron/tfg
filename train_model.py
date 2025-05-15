import os
import time
import json
import shutil
import torch
import tracemalloc
import subprocess
import psutil

from model import MLP, ResNet
from pinn_power import PowerMethodPINN


def run_experiment(config, save_dir='tfg/experiments', push_to_git=False):
    os.makedirs(save_dir, exist_ok=True)

    model_type = config["architecture"]
    optimizer_type = config["optimizer"]

    # Input dimension depending on periodic setting
    input_dim = config["dimension"] * (2 * config.get("pbc_k", 1) if config.get("periodic", False) else 1)
    config["input_dim"] = input_dim

    # Build model
    if model_type.lower() == 'mlp':
        layers = [input_dim] + [config["width"]] * config["depth"] + [1]
        model = MLP(layers)
    elif model_type.lower() == 'resnet':
        model = ResNet(in_num=input_dim, out_num=1, block_layers=[config["width"]] * 2, block_num=config["depth"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Initialize PINN
    pinn = PowerMethodPINN(model.double(), config)

    # Monitor CPU memory & time
    tracemalloc.start()
    start_time = time.time()

    # GPU memory stats reset
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Train
    if optimizer_type.lower() == 'adam':
        pinn.optimize_adam()
    elif optimizer_type.lower() == 'lbfgs':
        pinn.optimize_lbfgs()
    elif optimizer_type.lower() == 'adam_lbfgs':
        pinn.optimize_adam()
        pinn.optimize_lbfgs()
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # GPU memory usage (MB)
    if torch.cuda.is_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_gpu_memory = 0.0

    # Filenames
    base_name = f"{model_type.lower()}_{optimizer_type.lower()}_{config['depth']}_{config['width']}"
    model_path = os.path.join(save_dir, base_name + ".pt")
    summary_path = os.path.join(save_dir, base_name + "_summary.json")
    config_dest = os.path.join(save_dir, base_name + "_config.py")

    # Save model weights
    torch.save(pinn.model.state_dict(), model_path)

    # Save training summary
    summary = {
        "model_type": model_type,
        "optimizer": optimizer_type,
        "depth": config["depth"],
        "width": config["width"],
        "adam_steps": config.get("adam_steps", None),
        "lbfgs_steps": config.get("lbfgs_steps", None),
        "lambda_true": config["lambda_true"],
        "lambda_pred": float(pinn.best_lambda),
        "min_loss": float(pinn.min_loss),
        "time_seconds": elapsed,
        "peak_memory_MB": peak / 1024 / 1024,
        "peak_gpu_memory_MB": peak_gpu_memory,
        "device": str(pinn.device)
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Copy config.py
    config_source = "config.py"
    if os.path.exists(config_source):
        shutil.copy(config_source, config_dest)
    else:
        print("⚠ Warning: config.py not found, config not copied.")

    # Push to GitHub if requested
    if push_to_git:
        subprocess.run(["git", "add", save_dir])
        subprocess.run(["git", "commit", "-m", f"Add results for {model_type} + {optimizer_type}"])
        subprocess.run(["git", "push"])

    print(f"\n {model_type} + {optimizer_type} finished in {elapsed:.2f}s")
    print(f"   λ_est = {pinn.best_lambda:.6f} | Loss = {pinn.min_loss:.4e}")
    print(f"   Peak RAM: {peak / 1024 / 1024:.2f} MB | Peak GPU: {peak_gpu_memory:.2f} MB")

    return pinn
