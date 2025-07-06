import numpy as np
import torch
from torch.autograd import grad
import os
import json
import pyDOE
from pyDOE import lhs
import matplotlib.pyplot as plt
import subprocess


def coor_shift(X, lb, ub):
    X_shift = 2.0 * (X - lb) / (ub - lb) - 1.0
    return X_shift

def apply_boundary_condition(self, x, u):
    g = torch.ones_like(u)
    lb, ub = self.config["domain_lb"], self.config["domain_ub"]
    for i in range(x.shape[1]):
        xi = x[:, i:i + 1]
        g *= (torch.exp(xi - lb[i]) - 1.0) * (torch.exp(-(xi - ub[i])) - 1.0)
    return g * u


def sample_lhs(lb, ub, N, d):
    return  lb + (ub-lb)*lhs(d, N)

def compute_laplacian(u, x):
    assert u.dtype == torch.float64 and x.dtype == torch.float64, "Use of float64"
    device = u.device

    lap = torch.zeros_like(u)

    for i in range(x.shape[1]):
        grad_u = grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u, device=device),
            create_graph=True
        )[0]

        grad_u_i = grad(
            outputs=grad_u[:, i:i+1],
            inputs=x,
            grad_outputs=torch.ones_like(grad_u[:, i:i+1], device=device),
            create_graph=True
        )[0][:, i:i+1]

        lap = lap + grad_u_i

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


def maybe_push_to_git(path_to_add, message=None):
    """
    Adds, commits, and pushes a path to a Git repository if it is inside one.

    Args:
        path_to_add (str): Path to the file or directory to add and push.
        message (str): Commit message. If None, a default message is generated.
    """
    if not os.path.exists(path_to_add):
        print(f"[ERROR] Path does not exist: {path_to_add}")
        return

    try:
        # Ensure we're inside a git repo
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True, stdout=subprocess.DEVNULL)

        # Add path
        subprocess.run(["git", "add", path_to_add], check=True)

        # Build default message if not provided
        if message is None:
            message = f"Auto-commit changes to {os.path.basename(path_to_add)}"

        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push"], check=True)

        print(f"[INFO] Git push completed for '{path_to_add}'.")

    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Git operation failed: {e}")
    except Exception as ex:
        print(f"[ERROR] Unexpected error: {ex}")
