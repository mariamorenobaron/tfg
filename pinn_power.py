import torch
import os
import numpy as np
from utils import sample_lhs, compute_laplacian, periodic_transform, plot_eigenfunction


class PowerMethodPINN:
    def __init__(self, model, config):
        self.optimizer = None
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        x = sample_lhs(config["domain_lb"], config["domain_ub"], config["n_train"])
        self.x_train = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device)

        self.u = torch.rand_like(self.x_train[:, :1])
        self.u = self.u / torch.norm(self.u)
        self.u = self.u.to(self.device)

        self.lambda_ = 1.0
        self.min_loss = float("inf")
        self.best_lambda = None
        self.best_model_state = None
        self.checkpoint_path = config["checkpoint_path"]

        # Optional tracking for post-analysis
        self.loss_history = []
        self.lambda_history = []

    def apply_input_transform(self, x):
        if self.config.get("periodic", False):
            return periodic_transform(x, k=self.config.get("pbc_k", 1), periods=self.config.get("periods", None))
        return x

    def apply_boundary_condition(self, x, u):
        g = torch.ones_like(u)
        for i in range(x.shape[1]):
            xi = x[:, i:i + 1]
            lb = self.config["domain_lb"][i]
            ub = self.config["domain_ub"][i]
            g *= (xi - lb) * (ub - xi)
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

        # u^(k-1)
        u_prev = self.net_u(self.x_train)
        u_prev = u_prev / (torch.norm(u_prev) + 1e-10)

        # Compute Lu = Δu + Mu
        Lu = compute_laplacian(u_prev, self.x_train) + self.config["M"] * u_prev

        # Estimate λ using Rayleigh quotient
        numerator = torch.sum(Lu * u_prev)
        denominator = torch.sum(u_prev ** 2) + 1e-10
        self.lambda_ = (numerator / denominator)

        # Power iteration step u^(k)
        with torch.no_grad():
            u_new = Lu / (torch.norm(Lu) + 1e-10)
        self.u = u_new

        # Loss PMNN
        loss = torch.mean((u_prev - u_new) ** 2)
        loss.backward()
        self.optimizer.step()

        # Save best values
        loss_val = loss.item()
        lambda_val = self.lambda_.item()

        self.loss_history.append(loss_val)
        self.lambda_history.append(lambda_val)

        if loss_val < self.min_loss:
            self.min_loss = loss_val
            self.best_lambda = lambda_val
            self.best_model_state = self.model.state_dict()

        return loss_val

    def optimize_adam(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["adam_lr"])
        print("Starting training with Adam...\n")

        for it in range(self.config["adam_steps"]):
            loss = self.optimize_one_epoch()
            if it % 1000 == 0 or it == self.config["adam_steps"] - 1:
                print(f"[{it:5d}] Loss = {loss:.4e} | λ_est = {self.lambda_.item():.6f}")

        # Save final best model
        if self.checkpoint_path and self.best_model_state:
            torch.save({
                "model_state_dict": self.best_model_state,
                "lambda": self.best_lambda,
                "loss": self.min_loss
            }, self.checkpoint_path)
            print(f"\n✅ Model saved at: {self.checkpoint_path}")

    def evaluate_and_plot(self):
        # Sample 1D evaluation points
        x_eval = torch.linspace(
            self.config["domain_lb"][0], self.config["domain_ub"][0], 1000
        ).view(-1, 1).to(self.device)

        x_eval.requires_grad_(True)

        # Transformed input
        x_input = self.apply_input_transform(x_eval)
        with torch.no_grad():
            u_raw = self.model(x_input)
            if not self.config.get("periodic", False):
                u_pred = self.apply_boundary_condition(x_eval, u_raw)
            else:
                u_pred = u_raw

        # ✅ CORREGIMOS AQUÍ
        x_np = x_eval.detach().cpu().numpy()
        u_pred = u_pred.detach().cpu().numpy()
        u_pred = u_pred / np.linalg.norm(u_pred)

        u_true = self.config["exact_u"](x_np)
        u_true = u_true / np.linalg.norm(u_true)

        # Plotting
        plot_eigenfunction(
            x_np, u_pred, u_true,
            title="Predicted vs True Eigenfunction",
            save_path="eigenfunction_plot.png"
        )

