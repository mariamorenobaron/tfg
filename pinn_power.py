import torch
import numpy as np
from utils import sample_lhs, compute_laplacian, periodic_transform, plot_eigenfunction

torch.set_default_dtype(torch.float64)  # Usar precisión de 64 bits para todos los tensores

class PowerMethodPINN:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Puntos de entrenamiento
        x = sample_lhs(config["domain_lb"], config["domain_ub"], config["n_train"], config["dimension"])
        self.x_train = torch.tensor(x, dtype=torch.float64, requires_grad=True).to(self.device)

        # Inicialización de `u` y otros parámetros
        self.u = torch.rand_like(self.x_train[:, :1]).to(self.device)
        self.u = self.u / torch.norm(self.u)  # Normalizar solo al principio
        self.lambda_ = torch.tensor(1.0, dtype=torch.float64, device=self.device)

        # Variables de seguimiento
        self.min_loss = float("inf")
        self.loss = None
        self.best_lambda = None
        self.best_model_state = None
        self.checkpoint_path = config["checkpoint_path"]
        self.loss_history = []
        self.lambda_history = []

        self.optimizer = None
        self.optimizer_name = None

    def apply_input_transform(self, x):
        if self.config.get("periodic", False):
            return periodic_transform(x, k=self.config.get("pbc_k", 1), periods=self.config.get("periods", None))
        return x

    def apply_boundary_condition(self, x, u):
        g = torch.ones_like(u)
        lb, ub = self.config["domain_lb"], self.config["domain_ub"]
        for i in range(x.shape[1]):
            xi = x[:, i:i+1]
            g *= (torch.exp(xi - lb[i]) - 1.0) * (torch.exp(-(xi - ub[i])) - 1.0)
        return g * u

    def net_u(self, x):
        x_input = self.apply_input_transform(x)
        u_pred = self.model(x_input)
        if not self.config.get("periodic", False):
            u_pred = self.apply_boundary_condition(x, u_pred)
        return u_pred

    def optimize_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        self.loss = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        self.loss.requires_grad_()

        # u_prev = N(x)
        u_prev = self.net_u(self.x_train)
        N = self.x_train.shape[0]
        norm_sq_prev = torch.sum(u_prev ** 2) / N
        u_prev = u_prev / (torch.sqrt(norm_sq_prev) + 1e-10)

        # Compute Lu
        Lu = compute_laplacian(u_prev, self.x_train) + self.config["M"] * u_prev

        # tmp_loss = ||Lu - λ_prev * u_prev||²
        tmp_loss = torch.mean((Lu - self.lambda_ * self.u) ** 2)

        # u^k ← Lu / ||Lu||
        with torch.no_grad():
            norm_sq_Lu = torch.sum(Lu ** 2) / N
            u_new = Lu / (torch.sqrt(norm_sq_Lu) + 1e-10)
        self.u = u_new

        # PMNN loss = ||u_prev - u_new||²
        loss_PM = torch.mean((u_prev - u_new) ** 2)
        loss = loss_PM

        loss.backward()

        # Estimate λ
        numerator = torch.sum(Lu * u_prev)
        denominator = torch.sum(u_prev ** 2) / N + 1e-10
        self.lambda_ = torch.max(numerator / denominator)

        # Guardar el mejor λ si el error disminuye
        loss_val = tmp_loss.item()
        lambda_val = self.lambda_.item()

        self.loss_history.append(loss_val)
        self.lambda_history.append(lambda_val)

        if loss_val < self.min_loss:
            self.min_loss = loss_val
            self.loss = loss
            self.best_lambda = lambda_val
            self.best_model_state = self.model.state_dict()

        return loss, loss_val, lambda_val

    def optimize_adam(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["adam_lr"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False
        )
        self.optimizer_name = 'Adam'

        print(" Starting training with Adam...\n")
        for it in range(self.config["adam_steps"]):
            loss, loss_val, lambda_val = self.optimize_one_epoch()
            self.optimizer.step()

            if it % 1000 == 0 or it == self.config["adam_steps"] - 1:
                print(f"[{it:05d}] Loss val = {loss_val:.4e} | λ_est = {lambda_val:.8f} | λ_true = {self.config['lambda_true']:.8f} Loss = {loss:.4e}")

        if self.checkpoint_path and self.best_model_state:
            torch.save({
                "model_state_dict": self.best_model_state,
                "lambda": self.best_lambda,
                "loss": self.min_loss
            }, self.checkpoint_path)
            print(f"\n Model saved at: {self.checkpoint_path}")
            print(f" Best λ = {self.best_lambda:.8f} | Min Loss = {self.min_loss:.4e}")

    def evaluate_and_plot(self):
        if self.config["dimension"] != 1:
            print("Plotting only supported for 1D problems.")
            return

        x_eval = torch.linspace(
            self.config["domain_lb"][0], self.config["domain_ub"][0], 1000
        ).view(-1, 1).to(self.device)
        x_eval.requires_grad_(True)

        with torch.no_grad():
            x_input = self.apply_input_transform(x_eval)
            u_raw = self.model(x_input)
            if not self.config.get("periodic", False):
                u_pred = self.apply_boundary_condition(x_eval, u_raw)
            else:
                u_pred = u_raw

        x_np = x_eval.detach().cpu().numpy()
        u_pred = u_pred.detach().cpu().numpy()
        u_pred = u_pred / np.linalg.norm(u_pred)

        u_true = self.config["exact_u"](x_np)
        u_true = u_true / np.linalg.norm(u_true)

        plot_eigenfunction(
            x_np, u_pred, u_true,
            title="Predicted vs True Eigenfunction",
            save_path="eigenfunction_plot.png"
        )
