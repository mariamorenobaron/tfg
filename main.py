from results import generate_plots_from_training_and_push, evaluate_model_and_generate_results
import torch
from config import CONFIG
torch.set_default_dtype(torch.float64)
from train_model import run_model, run_model_all_criteria
import importlib.util
import os
import gc


def load_config(config_path):
    assert os.path.exists(config_path), f"No existe el archivo: {config_path}"

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.CONFIG

import os
import json

import os
import json
import numpy as np

def print_results(base_path):
    lambda_errors = []
    l2_errors = []

    for root, dirs, files in os.walk(base_path):
        if "results_summary.json" in files:
            file_path = os.path.join(root, "loss/pmnn_MLP_10D_d4_w80/evaluation_results/results_summary.json")
            print(f"\nPath: {file_path}")
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                lambda_err = float(data.get("lambda_rel_error", "nan"))
                l2_err = float(data.get("L2_error", "nan"))
                lambda_errors.append(lambda_err)
                l2_errors.append(l2_err)

                print(f"  lambda_rel_error: {lambda_err:.2e}")
                print(f"  L2_error: {l2_err:.2e}")
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")

    # calcular media y desviación estándar si hay datos
    if lambda_errors:
        mean_lambda = np.mean(lambda_errors)
        std_lambda = np.std(lambda_errors)
        mean_l2 = np.mean(l2_errors)
        std_l2 = np.std(l2_errors)

        print("\n=== Summary across all results ===")
        print(f"lambda_rel_error: mean = {mean_lambda:.2e}, std = {std_lambda:.2e}")
        print(f"L2_error: mean = {mean_l2:.2e}, std = {std_l2:.2e}")



if __name__ == "__main__":

    print_results('numerical_experiments/Part1_power_method')



    #run_model_all_criteria(CONFIG, save_dir='numerical_experiments/Part1_power_method/seed100')
    #run_model_all_criteria(CONFIG, save_dir='numerical_experiments/Part1_power_method/seed200')






    #pinn = run_model(CONFIG, save_dir='numerical_experiments/Part1_power_method/new')
    #generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/new/pmnn_MLP_10D_d4_w80_new/', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/new/pmnn_MLP_10D_d4_w80_new/', 150000, push_to_git=True)


    #pinn2 = run_model(load_config('numerical_experiments/Part1_power_method/pmnn_MLP_1D_d4_w20/config.py'), save_dir='numerical_experiments/Part2_inverse_power_method')
    #generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/pmnn_MLP_2D_d4_w20', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/pmnn_MLP_2D_d4_w20', 100000, push_to_git=True)

    #torch.cuda.empty_cache()
    #gc.collect()
