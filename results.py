import time
from models import MLP, ResNet
import subprocess
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import sample_lhs, coor_shift, apply_boundary_condition
import random

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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

                # ========= EXTRAER DATOS =========
                epochs = [entry["epoch"] for entry in data]
                losses = [entry["loss"] for entry in data]
                temporal_losses = [entry["temporal_loss"] for entry in data]
                lambdas = [entry["lambda"] for entry in data]
                lambda_errors = [abs(l - lambda_true) for l in lambdas]

                # ========= NUEVO: u_infty =========
                u_infty = [entry.get("u_infty", None) for entry in data]
                u_infty = [val for val in u_infty if val is not None]

                # ========= SMOOTHING =========
                if smooth_lambda_error:
                    lambda_errors_smoothed = moving_average(lambda_errors, w=20)
                    epochs_smoothed = epochs[19:]
                else:
                    lambda_errors_smoothed = lambda_errors
                    epochs_smoothed = epochs

                idx = slice(None, None, subsample)
                epochs_plot = np.array(epochs)[idx]
                losses_plot = np.array(losses)[idx]
                temporal_plot = np.array(temporal_losses)[idx]
                epochs_error_plot = np.array(epochs_smoothed)[idx]
                lambda_error_plot = np.array(lambda_errors_smoothed)[idx]

                # ========= NUEVO: u_infty plot =========
                if len(u_infty) > 0:
                    u_infty_plot = np.array(u_infty)[idx]

                # ========= FUNC PLOT =========
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

                # ========= GUARDAR PLOTS =========
                save_plot(epochs_plot, losses_plot, r"$\mathcal{L}$", "Loss vs Epochs", "loss_vs_epochs.png", color='navy', log_y=True)
                save_plot(epochs_plot, temporal_plot, r"$\mathcal{L}_{\mathrm{temp}}$", "Temporal Loss vs Epochs", "temporal_loss_vs_epochs.png", color='darkorange', log_y=True)
                save_plot(epochs_error_plot, lambda_error_plot, r"$|\lambda_{\mathrm{est}} - \lambda_{\mathrm{true}}|$", "Lambda Error vs Epochs", "lambda_error_vs_epochs.png", color='crimson', log_y=True)

                # ========= NUEVO: PLOT u_infty =========
                if len(u_infty) > 0:
                    save_plot(epochs_plot, u_infty_plot, r"$u_\infty$", "Max-norm Eigenfunction Error vs Epochs", "u_infty_vs_epochs.png", color='seagreen', log_y=True)

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


def evaluate_model_and_generate_results(subdir, n_eval_points, push_to_git=True):

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

    model = reconstruct_model(summary, config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).double()

    lambda_true = float(summary["lambda_true"])
    model_title = f"{summary['architecture']}_{summary['depth']}x{summary['width']}_{summary['optimizer']}_{summary['method']}"
    elapsed_minutes = summary.get("time_seconds", 0) / 60

    dim = config["dimension"]
    domain_lb = np.array(config["domain_lb"])
    domain_ub = np.array(config["domain_ub"])

    x_eval = sample_lhs(domain_lb, domain_ub, n_eval_points, dim)

    x_tensor = torch.tensor(x_eval, dtype=torch.float64, device=device)

    with torch.no_grad():
        lb = torch.tensor(domain_lb, dtype=torch.float64, device=device)
        ub = torch.tensor(domain_ub, dtype=torch.float64, device=device)
        x_input = coor_shift(x_tensor, lb, ub)
        u_raw = model(x_input)

        if not config.get("periodic", False):
            u_pred_tensor = apply_boundary_condition(config, x_tensor, u_raw)
        else:
            u_pred_tensor = u_raw

    u_true = config["exact_u"](x_eval)
    u_pred = u_pred_tensor.cpu().numpy()

    u_pred = u_pred / np.linalg.norm(u_pred) * np.sqrt(len(u_pred))
    u_true = u_true / np.linalg.norm(u_true) * np.sqrt(len(u_true))
    u_pred *= np.sign(np.mean(u_pred * u_true))

    lambda_pred = float(summary["lambda_pred"]) if "lambda_pred" in summary else None

    L2_error = np.sqrt(np.mean((u_true - u_pred) ** 2))
    Linf_error = np.max(np.abs(u_true - u_pred))
    lambda_abs_error = abs(lambda_pred - lambda_true)
    lambda_rel_error = lambda_abs_error / abs(lambda_true)

    results_dir = os.path.join(subdir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    if dim == 1:
        # Ordenar x_eval para graficar suavemente
        sort_idx = np.argsort(x_eval[:, 0])
        x_sorted = x_eval[sort_idx]
        u_true_sorted = u_true[sort_idx]
        u_pred_sorted = u_pred[sort_idx]

        plt.figure()
        plt.plot(x_sorted, u_true_sorted, linestyle='--', color='navy', label="u_true")
        plt.plot(x_sorted, u_pred_sorted, linestyle=':', color='crimson', label="u_pred")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.grid(True)
        plt.legend()
        plt.title("Eigenfunction (1D)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "eigenfunction_comparison_1D.png"), dpi=300)
        plt.close()

    if dim == 2:
        from scipy.interpolate import griddata

        # Crear grilla regular para interpolación
        x_lin = np.linspace(domain_lb[0], domain_ub[0], 200)
        y_lin = np.linspace(domain_lb[1], domain_ub[1], 200)
        X_GRID, Y_GRID = np.meshgrid(x_lin, y_lin)

        # Interpolar valores
        U_true_interp = griddata(x_eval, u_true.flatten(), (X_GRID, Y_GRID), method='cubic')
        U_pred_interp = griddata(x_eval, u_pred.flatten(), (X_GRID, Y_GRID), method='cubic')
        Error_interp = np.abs(U_true_interp - U_pred_interp)

        titles = ["u_true", "u_pred", "error"]
        arrays = [U_true_interp, U_pred_interp, Error_interp]

        for name, arr in zip(titles, arrays):
            plt.figure(figsize=(6, 5))
            contour = plt.contourf(X_GRID, Y_GRID, arr, levels=100, cmap="viridis")
            plt.colorbar(contour)
            plt.title(name)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{name}_heatmap_2D.png"), dpi=300)
            plt.close()

    u_list = [u_true, u_pred]
    data_labels = ['u_true', 'u_pred']

    # Usa el mismo rango para ambas curvas
    min_u = min(np.min(u_true), np.min(u_pred))
    max_u = max(np.max(u_true), np.max(u_pred))

    N = 100
    x_d = np.linspace(min_u, max_u, N + 1)
    delta_x = (max_u - min_u) / N

    density_list = []
    for u in u_list:
        density = np.zeros(N + 1)
        for i in range(u.shape[0]):
            value = u[i, 0]
            j = min(N, max(0, int(round((value - min_u) / delta_x))))
            density[j] += 1
        density_list.append(density)

    max_d = max(np.max(d) for d in density_list)
    datas = [np.stack((x_d, d / max_d), axis=1) for d in density_list]

    plt.figure()
    color_map = {"u_true": "black", "u_pred": "lime"}
    linestyle_map = {"u_true": "--", "u_pred": ":"}

    for data, label in zip(datas, data_labels):
        plt.plot(data[:, 0], data[:, 1], linestyle= linestyle_map.get(label, '-') , label=label, color=color_map.get(label, 'gray'))
    plt.xlabel("u")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"density_plot_dim{dim}.png"), dpi=300)
    plt.close()

    results = {
        "model": model_title,
        "n_eval_points": n_eval_points,
        "lambda_pred": lambda_pred,
        "lambda_true": lambda_true,
        "lambda_abs_error": lambda_abs_error,
        "lambda_rel_error": lambda_rel_error,
        "L2_error": L2_error,
        "Linf_error": Linf_error,
        "elapsed_minutes": elapsed_minutes
    }

    # --- Formateo para JSON ---
    results_formatted = {}
    for k, v in results.items():
        if isinstance(v, float):
            if "error" in k:
                results_formatted[k] = f"{v:.2e}"
            elif "lambda" in k:
                results_formatted[k] = f"{v:.8f}"
            elif "minutes" in k:
                results_formatted[k] = f"{v:.4f}"
            else:
                results_formatted[k] = f"{v}"
        else:
            results_formatted[k] = v

    with open(os.path.join(results_dir, "results_summary.json"), "w") as f:
        json.dump(results_formatted, f, indent=4)

    print("Evaluation:")
    for k, v in results_formatted.items():
        print(f"  {k}: {v}")

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
