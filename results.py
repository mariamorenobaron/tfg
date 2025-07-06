import time
import pandas as pd
from config import CONFIG
from model import MLP, ResNet
from pinn_power import PowerMethodPINN
import subprocess
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from utils import sample_lhs, periodic_transform, coor_shift, apply_boundary_condition

def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')

def generate_plots_from_training_and_push(root_dir, push_to_git=True, smooth_lambda_error=True, subsample=100, linewidth=0.7):

    plot_files = []

    for subdir, _, files in os.walk(root_dir):
        if "training_curve.json" in files and "summary.json" in files:
            training_path = os.path.join(subdir, "training_curve.json")
            summary_path = os.path.join(subdir, "summary.json")
            print(f"Processing: {training_path}")

            try:
                with open(training_path, "r") as f:
                    data = json.load(f)
                with open(summary_path, "r") as f:
                    summary = json.load(f)

                lambda_true = summary.get("lambda_true")
                if lambda_true is None:
                    print(f"Not found lambda true {summary_path}")
                    continue

                epochs = [entry["epoch"] for entry in data]
                losses = [entry["loss"] for entry in data]
                temporal_losses = [entry["temporal_loss"] for entry in data]
                lambdas = [entry["lambda"] for entry in data]
                lambda_errors = [abs(l - lambda_true) for l in lambdas]

                if smooth_lambda_error:
                    lambda_errors_smoothed = moving_average(lambda_errors, w=20)
                    epochs_smoothed = epochs[19:]  # recortar los primeros w-1
                else:
                    lambda_errors_smoothed = lambda_errors
                    epochs_smoothed = epochs

                idx = slice(None, None, subsample)
                epochs_plot = np.array(epochs)[idx]
                losses_plot = np.array(losses)[idx]
                temporal_plot = np.array(temporal_losses)[idx]
                epochs_error_plot = np.array(epochs_smoothed)[idx]
                lambda_error_plot = np.array(lambda_errors_smoothed)[idx]

                def save_plot(x, y, ylabel, title, filename, color='blue', log_y=False):
                    path = os.path.join(subdir, filename)
                    plt.figure(figsize=(5, 4))
                    if log_y:
                        plt.semilogy(x, y, color=color, linewidth=linewidth, alpha=0.7, label=ylabel)
                    else:
                        plt.plot(x, y, color=color, linewidth=linewidth, alpha=0.7, label=ylabel)
                    plt.xlabel(r"$k$", fontsize=12)
                    plt.ylabel(ylabel, fontsize=12)
                    plt.legend(fontsize=11)
                    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
                    plt.tight_layout()
                    plt.minorticks_off()
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files.append(path)

                save_plot(epochs_plot, losses_plot, r"$\mathcal{L}$", "Loss vs Epochs", "loss_vs_epochs.png", color='navy', log_y=True)
                save_plot(epochs_plot, temporal_plot, r"$\mathcal{L}_{\mathrm{temp}}$", "Temporal Loss vs Epochs", "temporal_loss_vs_epochs.png", color='darkorange', log_y=True)
                save_plot(epochs_error_plot, lambda_error_plot, r"$|\lambda_{\mathrm{est}} - \lambda_{\mathrm{true}}|$", "Lambda Error vs Epochs", "lambda_error_vs_epochs.png", color='crimson', log_y=True)

            except Exception as e:
                print(f"Error {subdir}: {e}")

    # Git
    if push_to_git and plot_files:
        try:
            subprocess.run(["git", "add"] + plot_files, check=True)
            subprocess.run(["git", "commit", "-m", "Add smoothed high-quality training plots"], check=True)
            subprocess.run(["git", "push"], check=True)
            print(" Gráficas subidas a GitHub con éxito.")
        except subprocess.CalledProcessError as e:
            print(f"Error al hacer push a GitHub: {e}")


def evaluate_model_and_generate_results(subdir, push_to_git=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(subdir, "model.pt")
    if not os.path.exists(model_path):
        print(f"No se encontró model.pt en {subdir}")
        return

    summary_path = os.path.join(subdir, "summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    config_path = os.path.join(subdir, "config.py")
    with open(config_path, "r") as f:
        config_dict = {}
        exec(f.read(), config_dict)
        config = config_dict["CONFIG"]

    model = reconstruct_model(summary,config)
    model.load_state_dict(torch.load(os.path.join(subdir, "model.pt"), map_location=device))
    model = model.to(device).double()

    lambda_true = float(summary["lambda_true"])
    model_title = f"{summary['architecture']}_{summary['depth']}x{summary['width']}_{summary['optimizer']}_{summary['method']}"
    elapsed_minutes = summary.get("elapsed_time", 0) / 60

    dim = config["dimension"]
    domain_lb = np.array(config["domain_lb"])
    domain_ub = np.array(config["domain_ub"])
    n_eval_points = 10000

    # --- Puntos de evaluación ---

    if dim == 1:
        x_eval = np.linspace(domain_lb[0], domain_ub[0], n_eval_points).reshape(-1, 1)
    else:
        samples = lhs(dim, n_eval_points)
        x_eval = domain_lb + (domain_ub - domain_lb) * samples

    x_tensor = torch.tensor(x_eval, dtype=torch.float64, device=device, requires_grad=True)

    # --- Evaluar u_pred ---
    with torch.no_grad():
        x_input = coor_shift(x_tensor, config["domain_lb"], config["domain_ub"])
        u_raw = model(x_input)

        if not config.get("periodic", False):
            u_pred_tensor = apply_boundary_condition(config,x_tensor, u_raw)
        else:
            u_pred_tensor = u_raw

    u_true =    config["exact_u"](x_eval)

    u_pred = u_pred_tensor.cpu().numpy()
    u_pred = u_pred / np.linalg.norm(u_pred) * np.sqrt(len(u_pred))
    u_true = u_true / np.linalg.norm(u_true) * np.sqrt(len(u_true))
    u_pred *= np.sign(np.mean(u_pred * u_true))

    lambda_pred = float(summary["lambda_pred"]) if "lambda_pred" in summary else None

    # --- Metrics ---
    L2_error = np.sqrt(np.mean((u_true - u_pred) ** 2))
    Linf_error = np.max(np.abs(u_true - u_pred))
    lambda_abs_error = abs(lambda_pred - lambda_true)
    lambda_rel_error = lambda_abs_error / abs(lambda_true)

    results_dir = os.path.join(subdir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    # --- Plot eigenfunction (1D) ---
    if dim == 1:
        plt.figure()
        plt.plot(x_eval, u_true, '--', label="u_true")
        plt.plot(x_eval, u_pred, ':', label="u_pred")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.grid(True)
        plt.legend()
        plt.title("Eigenfunction (1D)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "eigenfunction_comparison_1D.png"), dpi=300)
        plt.close()

    # --- Heatmaps (2D) ---
    if dim == 2:
        import matplotlib.tri as tri
        tri_obj = tri.Triangulation(x_eval[:, 0], x_eval[:, 1])
        for arr, name in zip([u_true, u_pred, np.abs(u_true - u_pred)],
                             ["u_true", "u_pred", "error"]):
            plt.figure()
            plt.tricontourf(tri_obj, arr.flatten(), 100)
            plt.colorbar()
            plt.title(name)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{name}_heatmap_2D.png"), dpi=300)
            plt.close()

    # --- Density plot (tu estilo) ---
    u_list = [u_true, u_pred]
    data_labels = ['u_true', 'u_pred']
    min_u = min(np.min(u_true), np.min(u_pred))
    max_u = max(np.max(u_true), np.max(u_pred))
    N = 100
    x_d = np.linspace(min_u, max_u, N+1)
    delta_x = (max_u - min_u) / N
    density_list = []
    for u in u_list:
        density = np.zeros(N+1)
        for i in range(u.shape[0]):
            value = u[i, 0]
            j = min(N, max(0, int(round((value - min_u) / delta_x))))
            density[j] += 1
        density_list.append(density)

    max_d = max(map(np.max, density_list))
    datas = [np.stack((x_d, d / max_d), axis=1) for d in density_list]

    plt.figure()
    for data, label in zip(datas, data_labels):
        plt.plot(data[:, 0], data[:, 1],'--' ,label=label)
    plt.xlabel("u")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"density_plot_dim{dim}.png"), dpi=300)
    plt.close()

    # --- Guardar resumen ---
    results = {
        "model": model_title,
        "lambda_pred": lambda_pred,
        "lambda_true": lambda_true,
        "lambda_abs_error": lambda_abs_error,
        "lambda_rel_error": lambda_rel_error,
        "L2_error": L2_error,
        "Linf_error": Linf_error,
        "elapsed_minutes": round(elapsed_minutes, 2)
    }

    with open(os.path.join(results_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation:")
    for k, v in results.items():
        if isinstance(v, float):
            if "error" in k:
                print(f"  {k}: {v:.2e}")

            elif "lambda" in k:
                print(f"  {k}: {v:.8f}")
            else:
                print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    # --- Git (opcional) ---
    if push_to_git:
        import subprocess
        try:
            subprocess.run(["git", "add", results_dir], check=True)
            subprocess.run(["git", "commit", "-m", f"Add evaluation results for model {model_title}"], check=True)
            subprocess.run(["git", "push"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠ Git error: {e}")

def reconstruct_model(arch_config, config):
    arch = config["architecture"]
    input_dim = config["dimension"]
    depth = arch_config["depth"]
    width = arch_config["width"]

    if arch == "MLP":
        return MLP([input_dim] + [width]*depth + [1])
    elif arch == "ResNet":
        return ResNet(in_num=input_dim, out_num=1, block_layers=[width]*2, block_num=depth)
    else:
        raise ValueError("Unknown architecture type")
