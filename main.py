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

if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    pinn = run_model(load_config('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha36'), save_dir='numerical_experiments/Part3_shifted_ipmnn')
    generate_plots_from_training_and_push('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha36', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha36', 20000, push_to_git=True)

    pinn = run_model(load_config('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha81'), save_dir='numerical_experiments/Part3_shifted_ipmnn')
    generate_plots_from_training_and_push('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha81', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha81', 20000, push_to_git=True)

    pinn = run_model(load_config('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha144'), save_dir='numerical_experiments/Part3_shifted_ipmnn')
    generate_plots_from_training_and_push('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha144', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha144', 20000, push_to_git=True)

    pinn = run_model(load_config('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha225'), save_dir='numerical_experiments/Part3_shifted_ipmnn')
    generate_plots_from_training_and_push('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha225', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part3_shifted_ipmnn/ipmnn_MLP_1D_d4_w20_alpha225', 20000, push_to_git=True)








    #generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/loss/pmnn_MLP_1D_d4_w20', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/loss/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True)
    #generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/loss_temporal/pmnn_MLP_1D_d4_w20', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/loss_temporal/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True)
    #generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/loss_combined/pmnn_MLP_1D_d4_w20', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/loss_combined/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True)
    #generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/loss_combined1/pmnn_MLP_1D_d4_w20', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/loss_combined1/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True)




    #run_model_all_criteria(CONFIG, save_dir='numerical_experiments/Part1_power_method')




    #pinn = run_model(CONFIG, save_dir='numerical_experiments/Part1_power_method/new')
    #generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/new/pmnn_MLP_10D_d4_w80_new/', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/new/pmnn_MLP_10D_d4_w80_new/', 150000, push_to_git=True)


    #pinn2 = run_model(load_config('numerical_experiments/Part1_power_method/pmnn_MLP_1D_d4_w20/config.py'), save_dir='numerical_experiments/Part2_inverse_power_method')
    #generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/pmnn_MLP_2D_d4_w20', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/pmnn_MLP_2D_d4_w20', 100000, push_to_git=True)

    #torch.cuda.empty_cache()
    #gc.collect()
