from config import CONFIG
from train_model import run_experiment
import os
import tracemalloc
import time

if __name__ == "__main__":
    pinn = run_experiment(CONFIG, save_dir='numerical_experiments/Part3_inverse_architectures')
    pinn.evaluate_and_plot()