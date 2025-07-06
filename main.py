from config import CONFIG
from train_model import run_model
from results import generate_plots_from_training_and_push, evaluate_model_and_generate_results
import torch
import numpy as np
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    generate_plots_from_training_and_push('numerical_experiments/Part2_inverse_power_method/ipmnn_MLP_2D_d4_w20/',  push_to_git= True)
    #pinn = run_model(CONFIG, save_dir='numerical_experiments/Part5_epochs_analysis')
    #pinn.evaluate_and_plot()
    evaluate_model_and_generate_results('numerical_experiments/Part2_inverse_power_method/ipmnn_MLP_2D_d4_w20/',True)
