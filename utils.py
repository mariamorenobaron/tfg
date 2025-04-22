import numpy as np
import torch
from torch.autograd import grad
import os
import json
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)  

def sample_lhs(lb, ub, N):
    return lb + (ub - lb) * np.random.rand(N, lb.shape[0])


def compute_laplacian(u, x):
    """∇²u en todos los puntos x (Laplaciano escalar de una sola salida)."""
    assert u.dtype == torch.float64 and x.dtype == torch.float64, "Usa float64"
    device = u.device                     # (o x.device, es lo mismo)

    lap = torch.zeros_like(u)             # N×1

    # Recorremos cada dirección espacial
    for i in range(x.shape[1]):
        # ∂u/∂x_i
        grad_u = grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u, device=device),   # N×1
            create_graph=True
        )[0]                                                   # N×d

        # ∂²u/∂x_i²   (segunda derivada respecto a la i‑ésima coordenada)
        grad_u_i = grad(
            outputs=grad_u[:, i:i+1],                          # N×1
            inputs=x,
            grad_outputs=torch.ones_like(grad_u[:, i:i+1], device=device),
            create_graph=True
        )[0][:, i:i+1]                                         # N×1

        lap = lap + grad_u_i                                   # acumulamos

    return lap 


def periodic_transform(x, k=1, periods=None):
    d = x.shape[1]
    if periods is None:
        periods = [1.0] * d
    features = []
    for i in range(d):
        for j in range(1, k + 1):
            angle = 2 * np.pi * j * x[:, i:i+1] / periods[i]
            features.append(torch.sin(angle))
            features.append(torch.cos(angle))
    return torch.cat(features, dim=1)

def save_model(model, config, folder="saved_model"):
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(folder, "model_weights.pt"))
    arch_config = {
        "architecture": config["architecture"],
        "depth": config["depth"],
        "width": config["width"],
        "input_dim": config["input_dim"]
    }
    with open(os.path.join(folder, "model_architecture.json"), "w") as f:
        json.dump(arch_config, f)

def load_model(model_class_dict, folder="saved_model"):
    with open(os.path.join(folder, "model_architecture.json")) as f:
        arch_config = json.load(f)
    arch = arch_config["architecture"]
    input_dim = arch_config["input_dim"]
    depth = arch_config["depth"]
    width = arch_config["width"]
    if arch == "MLP":
        model = model_class_dict["MLP"]([input_dim] + [width]*depth + [1])
    else:
        model = model_class_dict["ResNet"](in_num=input_dim, out_num=1, block_layers=[width]*2, block_num=depth)
    model.load_state_dict(torch.load(os.path.join(folder, "model_weights.pt")))
    return model, arch_config

def plot_eigenfunction(x, u_pred, u_true=None, title="Autofunción estimada vs exacta", save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(x, u_pred, label="u_pred (estimada)", linewidth=2)
    if u_true is not None:
        plt.plot(x, u_true, label="u_true (exacta)", linestyle="--", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()
