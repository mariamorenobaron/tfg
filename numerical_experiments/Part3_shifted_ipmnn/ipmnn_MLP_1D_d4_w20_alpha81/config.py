import numpy as np

d = 1

CONFIG = {
    "dimension": d,
    "domain_lb": np.zeros(d),
    "domain_ub": np.ones(d),
    "M": 100,

    "architecture": "MLP",     # "MLP" or "ResNet"
    "optimizer": "adam",
    "method": "ipmnn",                      # "pmnn" or "ipmnn"
    "depth": 4,
    "width": 20,
    "push_to_git" : True,

    "adam_steps": 25000,
    "adam_lr": 1e-3,
    "n_train": 20000,
    "early_stopping": False,
    "tolerance": 1e-6,
    "use_seed": False,
    "seed": 2,

    "lambda_true": (9 * np.pi)**2,
    "exact_u": lambda x: np.prod(np.sin(9 * np.pi * x),axis=1, keepdims=True),

    "periodic": False,
    "pbc_k": 1,
    "periods": None,
    "alpha": 36.0    # component for shifted inverse power method
}
