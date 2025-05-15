import time
import os
import torch
import psutil
import json
import tracemalloc
import subprocess
from model import MLP, ResNet
from pinn_power import PowerMethodPINN


def run_experiment(config, model_type='MLP', optimizer_type='adam', save_dir='experiments', push_to_git=False):
    os.makedirs(save_dir, exist_ok=True)

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

    # Monitor memory & time
    tracemalloc.start()
    start_time = time.time()

    # Train
    if optimizer_type.lower() == 'adam':
        pinn.optimize_adam()
    elif optimizer_type.lower() == 'lbfgs':
        pinn.optimize_lbfgs()
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Save model
    model_filename = f"{model_type.lower()}_{optimizer_type.lower()}.pt"
    model_path = os.path.join(save_dir, model_filename)
    torch.save(pinn, model_path)

    # Save summary as JSON
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
        "device": str(pinn.device)
    }

    summary_path = os.path.join(save_dir, f"{model_type.lower()}_{optimizer_type.lower()}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f" {model_type} + {optimizer_type} finished in {elapsed:.2f}s | λ = {pinn.best_lambda:.6f}")

    # Optional GitHub upload
    if push_to_git:
        subprocess.run(["git", "add", save_dir])
        subprocess.run(["git", "commit", "-m", f"Add results for {model_type} + {optimizer_type}"])
        subprocess.run(["git", "push"])
    return pinn
