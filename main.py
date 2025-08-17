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


    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed100/loss/pmnn_MLP_1D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed100/loss/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed100/loss_temporal/pmnn_MLP_1D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed100/loss_temporal/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed100/loss_combined/pmnn_MLP_1D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed100/loss_combined/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed100/loss_combined1/pmnn_MLP_1D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed100/loss_combined1/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True, seed=True)

    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed200/loss/pmnn_MLP_1D_d4_w20',push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed200/loss/pmnn_MLP_1D_d4_w20',20000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed200/loss_temporal/pmnn_MLP_1D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed200/loss_temporal/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed200/loss_combined/pmnn_MLP_1D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed200/loss_combined/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed200/loss_combined1/pmnn_MLP_1D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed200/loss_combined1/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True, seed=True)

    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed300/loss/pmnn_MLP_1D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed300/loss/pmnn_MLP_1D_d4_w20',  20000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push( 'numerical_experiments/Part1_power_method/seed300/loss_temporal/pmnn_MLP_1D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results( 'numerical_experiments/Part1_power_method/seed300/loss_temporal/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True,    seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed300/loss_combined/pmnn_MLP_1D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results( 'numerical_experiments/Part1_power_method/seed300/loss_combined/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True,seed=True)
    generate_plots_from_training_and_push( 'numerical_experiments/Part1_power_method/seed300/loss_combined1/pmnn_MLP_1D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results( 'numerical_experiments/Part1_power_method/seed300/loss_combined1/pmnn_MLP_1D_d4_w20', 20000, push_to_git=True, seed=True)

    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed100/loss/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed100/loss/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed100/loss_temporal/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed100/loss_temporal/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed100/loss_combined/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed100/loss_combined/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed100/loss_combined1/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed100/loss_combined1/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed200/loss/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed200/loss/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed200/loss_temporal/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed200/loss_temporal/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed200/loss_combined/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed200/loss_combined/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed200/loss_combined1/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed200/loss_combined1/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed300/loss/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed300/loss/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed300/loss_temporal/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed300/loss_temporal/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed300/loss_combined/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed300/loss_combined/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/seed300/loss_combined1/pmnn_MLP_5D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/seed300/loss_combined1/pmnn_MLP_5D_d4_w40', 100000, push_to_git=True, seed=True)


    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d2_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d2_w20', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d2_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d2_w40', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d2_w80', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d2_w80', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d4_w20', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d4_w40', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d4_w80', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d4_w80', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d8_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d8_w20', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d8_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d8_w40', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d8_w80', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_MLP_2D_d8_w80', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d2_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d2_w20', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d2_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d2_w40', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d2_w80', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d2_w80', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d4_w20', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d4_w20', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d4_w40', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d4_w40', 70000, push_to_git=True, seed=True)
    generate_plots_from_training_and_push('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d4_w80', push_to_git=True)
    evaluate_model_and_generate_results('numerical_experiments/Part4_architectures/pmnn_ResNet_2D_d4_w80', 70000, push_to_git=True, seed=True)




    #run_model_all_criteria(CONFIG, save_dir='numerical_experiments/Part1_power_method/seed100')
    #run_model_all_criteria(CONFIG, save_dir='numerical_experiments/Part1_power_method/seed200')
    run_model_all_criteria(load_config('numerical_experiments/Part1_power_method/seed100/config2.py'), save_dir='numerical_experiments/Part1_power_method/seed100')
    run_model_all_criteria(load_config('numerical_experiments/Part1_power_method/seed200/config2d.py'), save_dir='numerical_experiments/Part1_power_method/seed200')
    run_model_all_criteria(load_config('numerical_experiments/Part1_power_method/seed300/config2d.py'), save_dir='numerical_experiments/Part1_power_method/seed300')







    #pinn = run_model(CONFIG, save_dir='numerical_experiments/Part1_power_method/new')
    #generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/new/pmnn_MLP_10D_d4_w80_new/', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/new/pmnn_MLP_10D_d4_w80_new/', 150000, push_to_git=True)


    #pinn2 = run_model(load_config('numerical_experiments/Part1_power_method/pmnn_MLP_1D_d4_w20/config.py'), save_dir='numerical_experiments/Part2_inverse_power_method')
    #generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/pmnn_MLP_2D_d4_w20', push_to_git=True)
    #evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/pmnn_MLP_2D_d4_w20', 100000, push_to_git=True)

    #torch.cuda.empty_cache()
    #gc.collect()
