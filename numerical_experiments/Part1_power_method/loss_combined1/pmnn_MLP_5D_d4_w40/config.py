import numpy as np

d = 5

CONFIG = {
    "dimension": d,
    "domain_lb": np.zeros(d),
    "domain_ub": np.ones(d),
    "M": 100,

    "architecture": "MLP",     # "MLP" or "ResNet"
    "optimizer": "adam",
    "method": "pmnn",                      # "pmnn" or "ipmnn"
    "depth": 4,
    "width": 40,
    "push_to_git" : True,

    "adam_steps": 50000,
    "adam_lr": 1e-3,
    "n_train": 50000,
    "early_stopping": False,
    "tolerance": 1e-6,
    "use_seed": True,
    "seed": 10,

    "lambda_true": 100 - d * np.pi**2,
    "exact_u": lambda x: np.prod(np.sin(np.pi * x), axis=1, keepdims=True),

    "periodic": False,
    "pbc_k": 1,
    "periods": None,
    "alpha": 0.0    # component for shifted inverse power method
}
