import time
import torch
import numpy as np
from models import MLP, ResNet
from pinn_power import PowerMethodPINN
from config import CONFIG
torch.set_default_dtype(torch.float64)  

# CURRENTLY NOT USED

def train_adam_with_mlp(config):
    input_dim = config["dimension"] * (2 * config["pbc_k"] if config["periodic"] else 1)
    config["input_dim"] = input_dim
    layers = [input_dim] + [config["width"]] * config["depth"] + [1]
    model = MLP(layers).double()
    pinn = PowerMethodPINN(model, config)

    start = time.time()
    pinn.optimize_adam()
    elapsed = time.time() - start

    print("\n--- MLP Training Finished ---")
    print(f"Time: {elapsed:.2f}s | Best λ: {pinn.best_lambda:.6f} | Min Loss: {pinn.min_loss:.2e}")
    return pinn

def train_adam_with_resnet(config):
    input_dim = config["dimension"] * (2 * config["pbc_k"] if config["periodic"] else 1)
    config["input_dim"] = input_dim
    model = ResNet(
        in_num=input_dim,
        out_num=1,
        block_layers=[config["width"]] * 2,
        block_num=config["depth"],
    )
    pinn = PowerMethodPINN(model, config)

    start = time.time()
    pinn.optimize_adam()
    elapsed = time.time() - start

    print("\n--- ResNet Training Finished ---")
    print(f"Time: {elapsed:.2f}s | Best λ: {pinn.best_lambda:.6f} | Min Loss: {pinn.min_loss:.2e}")
    return pinn
