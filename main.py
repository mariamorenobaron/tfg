from config import CONFIG
from train_model import run_model
from results import generate_plots_from_training_and_push, evaluate_model_and_generate_results
import torch
import numpy as np
torch.set_default_dtype(torch.float64)
import importlib.util
import os
import gc

def load_config(config_path):
    assert os.path.exists(config_path), f"No existe el archivo: {config_path}"

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.CONFIG

if __name__ == "__main__":

    #pinn1 = run_model(load_config('numerical_experiments/Part1_power_method/pmnn_MLP_10D_d4_w80/config.py'), save_dir='numerical_experiments/Part1_power_method')
    #generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/pmnn_MLP_10D_d4_w80/', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/pmnn_MLP_10D_d4_w80/', 100000, push_to_git=True)

    #del pinn1
    #torch.cuda.empty_cache()
    #gc.collect()

    #pinn2 = run_model(load_config('numerical_experiments/Part2_inverse_power_method/ipmnn_MLP_10D_d4_w80/config.py'), save_dir='numerical_experiments/Part2_inverse_power_method')
    #generate_plots_from_training_and_push('numerical_experiments/Part2_inverse_power_method/ipmnn_MLP_10D_d4_w80/', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part2_inverse_power_method/ipmnn_MLP_10D_d4_w80/', 100000, push_to_git=True)

    #del pinn2
    #torch.cuda.empty_cache()
    #gc.collect()

    pinn4 = run_model(load_config('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_2D_d4_w20_epochs5000/config.py'),
                      save_dir='numerical_experiments/Part5_epochs_analysis')
    generate_plots_from_training_and_push('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_2D_d4_w20_epochs5000/', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_2D_d4_w20_epochs5000/', 100000, push_to_git=True)

    del pinn4
    torch.cuda.empty_cache()
    gc.collect()

    pinn5 = run_model(load_config('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_2D_d4_w20_epochs15000/config.py'),
                      save_dir='numerical_experiments/Part5_epochs_analysis')
    generate_plots_from_training_and_push('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_2D_d4_w20_epochs15000/', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_2D_d4_w20_epochs15000/', 100000, push_to_git=True)

    del pinn5
    torch.cuda.empty_cache()
    gc.collect()

    pinn6 = run_model(load_config('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_1D_d4_w20_epochs25000/config.py'),save_dir='numerical_experiments/Part5_epochs_analysis')
    generate_plots_from_training_and_push('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_1D_d4_w20_epochs25000/', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_1D_d4_w20_epochs25000/', 100000, push_to_git=True)

    del pinn6
    torch.cuda.empty_cache()
    gc.collect()

    pinn7 = run_model(load_config('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_1D_d4_w20_epochs50000/config.py'), save_dir='numerical_experiments/Part5_epochs_analysis')
    generate_plots_from_training_and_push('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_1D_d4_w20_epochs50000/', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_1D_d4_w20_epochs50000/', 100000, push_to_git=True)

    del pinn7
    torch.cuda.empty_cache()
    gc.collect()

    pinn8 = run_model(load_config('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_5D_d4_w40_epochs100000/config.py'), save_dir='numerical_experiments/Part5_epochs_analysis')
    generate_plots_from_training_and_push('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_5D_d4_w40_epochs100000/', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_5D_d4_w40_epochs100000/', 100000, push_to_git=True)

    del pinn8
    torch.cuda.empty_cache()
    gc.collect()

    pinn9 = run_model(load_config('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_5D_d4_w40_epochs75000/config.py'), save_dir='numerical_experiments/Part5_epochs_analysis')
    generate_plots_from_training_and_push('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_5D_d4_w40_epochs75000/', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part5_epochs_analysis/pmnn_MLP_5D_d4_w40_epochs75000/', 100000, push_to_git=True)





    #generate_plots_from_training_and_push('numerical_experiments/Part2_inverse_power_method/ipmnn_MLP_2D_d4_w20/' ,push_to_git= True)
    #evaluate_model_and_generate_results('numerical_experiments/Part2_inverse_power_method/ipmnn_MLP_1D_d4_w20/',20000, push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part2_inverse_power_method/ipmnn_MLP_2D_d4_w20/',50000, push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/pmnn_MLP_1D_d4_w20/',20000, push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/pmnn_MLP_2D_d4_w20/',50000, push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/pmnn_MLP_5D_d4_w40/',100000, push_to_git=True)