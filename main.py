from config import CONFIG
from train_model import run_model
from results import generate_plots_from_training_and_push
import torch

torch.set_default_dtype(torch.float64)
import importlib.util
import os


def load_config(config_path):
    assert os.path.exists(config_path), f"No existe el archivo: {config_path}"

    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.CONFIG

if __name__ == "__main__":
    pinn = run_model(CONFIG, save_dir='numerical_experiments/Part4_architectures')
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d4_w40')
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d4_w80')


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


    #generate_plots_from_training_and_push('numerical_experiments/Part2_inverse_power_method/ipmnn_MLP_2D_d4_w20/' ,push_to_git= True)
    #evaluate_model_and_generate_results('numerical_experiments/Part2_inverse_power_method/ipmnn_MLP_1D_d4_w20/',20000, push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part2_inverse_power_method/ipmnn_MLP_2D_d4_w20/',50000, push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/pmnn_MLP_1D_d4_w20/',20000, push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/pmnn_MLP_2D_d4_w20/',50000, push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/pmnn_MLP_5D_d4_w40/',100000, push_to_git=True)