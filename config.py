import numpy as np

d = 5

CONFIG = {
    "dimension": d,
    "domain_lb": np.zeros(d),
    "domain_ub": np.ones(d),
    "M": 100,

    "architecture": "MLP",
    "optimizer": "adam",
    "method": "ipmnn",                      # "pmnn" or "ipmnn"
    "depth": 4,
    "width": 40,
    "push_to_git" : True,

    "adam_steps": 50000,
    "adam_lr": 1e-3,
    "lbfgs_steps": 0,
    "n_train": 50000,
    "fixed_min_loss": None,

    "lambda_true": d * np.pi**2,
    "exact_u": lambda x: np.prod(np.sin(np.pi * x), axis=1, keepdims=True),

    "periodic": False,
    "pbc_k": 1,
    "periods": None,
    "alpha": 0.0
}
