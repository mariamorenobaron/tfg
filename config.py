import numpy as np

d = 1

CONFIG = {
    "dimension": d,
   "domain_lb": np.zeros(d),
    "domain_ub": np.ones(d),
    "M": 100.0,

    "architecture": "MLP",
    "optimizer": "Adam",
    "depth": 4,
    "width": 20,

    "adam_steps": 50000,
    "adam_lr": 1e-3,
    "lbfgs_steps": 0,
    "n_train": 10000,
    "checkpoint_path": "model1_1d.pt",

    "lambda_true":100 - d * np.pi**2,
    "exact_u": lambda x: np.prod(np.sin(np.pi * x), axis=1, keepdims=True),

    "periodic": False,
    "pbc_k": 1,
    "periods": None
}
