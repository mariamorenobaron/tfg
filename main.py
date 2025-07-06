from config import CONFIG
from train_model import run_model
from results import generate_plots_from_training_and_push, evaluate_model_and_generate_results
import torch
import numpy as np
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    u_exact = lambda x: np.prod(np.sin(np.pi * x), axis=1, keepdims=True)
    generate_plots_from_training_and_push('numerical_experiments/Part1_power_method/pmnn_MLP_5D_d4_w40/',  push_to_git= True)
    #pinn = run_model(CONFIG, save_dir='numerical_experiments/Part5_epochs_analysis')
    #pinn.evaluate_and_plot()
    evaluate_model_and_generate_results('numerical_experiments/Part1_power_method/pmnn_MLP_5D_d4_40/',True)
