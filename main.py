from config import CONFIG
from train_model import run_experiment
import os
import tracemalloc
import time

if __name__ == "__main__":
    pinn = run_experiment(CONFIG)
    pinn.evaluate_and_plot()