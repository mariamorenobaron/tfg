from results import generate_plots_from_training_and_push, evaluate_model_and_generate_results
import torch
from config import CONFIG
torch.set_default_dtype(torch.float64)
from train_model import run_model, run_model_all_criteria
import importlib.util
import os
import gc
from pathlib import Path
import numpy as np



def load_config(config_path):
    assert os.path.exists(config_path), f"No existe el archivo: {config_path}"

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.CONFIG



import os
import json
import numpy as np
from pathlib import Path

def collect_and_summarize(base_path: str, suffix: str):
    """
    Busca archivos que terminen en `suffix` dentro de `base_path`,
    imprime sus valores de lambda_rel_error y L2_error en notación científica,
    y calcula la media y desviación estándar.
    """
    base = Path(base_path)

    lambda_errors = []
    l2_errors = []

    for p in base.rglob("results_summary.json"):
        normalized = str(p.as_posix())
        if normalized.endswith(suffix):
            with p.open("r") as f:
                data = json.load(f)

            lam = float(data.get("lambda_rel_error", "nan"))
            l2 = float(data.get("L2_error", "nan"))

            lambda_errors.append(lam)
            l2_errors.append(l2)

            print(f"\nPath: {normalized}")
            print(f"  lambda_rel_error: {lam:.2e}")
            print(f"  L2_error: {l2:.2e}")

    if lambda_errors:
        mean_lambda = np.mean(lambda_errors)
        std_lambda = np.std(lambda_errors)
        mean_l2 = np.mean(l2_errors)
        std_l2 = np.std(l2_errors)

        print("\n=== Summary across all matches ===")
        print(f"lambda_rel_error: mean = {mean_lambda:.2e}, std = {std_lambda:.2e}")
        print(f"L2_error: mean = {mean_l2:.2e}, std = {std_l2:.2e}")
    else:
        print(f"\nNo matches found for suffix: {suffix}")


if __name__ == "__main__":

    collect_and_summarize('numerical_experiments/Part1_power_method', 'loss/pmnn_MLP_1D_d4_w20/evaluation_results/results_summary.json')
    collect_and_summarize('numerical_experiments/Part1_power_method', 'loss_combined/pmnn_MLP_1D_d4_w20/evaluation_results/results_summary.json')
    collect_and_summarize('numerical_experiments/Part1_power_method', 'loss_combined1/pmnn_MLP_1D_d4_w20/evaluation_results/results_summary.json')
    collect_and_summarize('numerical_experiments/Part1_power_method', 'loss_temporal/pmnn_MLP_1D_d4_w20/evaluation_results/results_summary.json')






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
