import numpy as np

d = 1

CONFIG = {
    "dimension": d,
   "domain_lb": np.zeros(d),
    "domain_ub": np.full(d, 2*np.pi) if periodic else np.ones(d),
    "M": 100.0,

    "architecture": "MLP",
    "optimizer": "Adam+LBFGS",
    "depth": 4,
    "width": 40,

    "adam_steps": 10000,
    "adam_lr": 1e-3,
    "lbfgs_steps": 500,
    "n_train": 10000,
    "checkpoint_path": "best_model.pt",

    "lambda_true": 100 - d * np.pi**2,
    "exact_u": lambda x: np.prod(np.sin(np.pi * x), axis=1, keepdims=True),

    "periodic": True,
    "pbc_k": 1,
    "periods": None
}
