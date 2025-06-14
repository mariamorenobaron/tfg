from config import CONFIG
from train_model import run_experiment
import os
import tracemalloc
import time

if __name__ == "__main__":
    pinn = run_experiment(CONFIG, save_dir='numerical_experiments/Part2_inverse_power_method' )
    pinn.evaluate_and_plot()