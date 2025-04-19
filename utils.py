import numpy as np
import torch
from torch.autograd import grad
import os
import json

def sample_lhs(lb, ub, N):
    return lb + (ub - lb) * np.random.rand(N, lb.shape[0])

def compute_laplacian(u, x):
    grads = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    lap = 0
    for i in range(x.shape[1]):
        grads_i = grads[:, i:i+1]
        grad2 = grad(grads_i, x, grad_outputs=torch.ones_like(grads_i), create_graph=True)[0][:, i:i+1]
        lap += grad2
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
