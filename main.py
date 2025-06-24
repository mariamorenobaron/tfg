from config import CONFIG
from train_model import run_experiment
import os
import tracemalloc
import time
from results import process_all_training_curves

if __name__ == "__main__":
    process_all_training_curves('numerical_experiments/Part1_power_method/pmnn_MLP_1D_d4_w20')
    #pinn = run_experiment(CONFIG, save_dir='numerical_experiments/Part3_inverse_architectures')
    #pinn.evaluate_and_plot()