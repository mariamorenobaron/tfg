from datetime import time

import torch
import numpy as np
import os
from utils import sample_lhs, compute_laplacian, periodic_transform,coor_shift

torch.set_default_dtype(torch.float64)

class PowerMethodPINN:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        x = sample_lhs(config["domain_lb"], config["domain_ub"], config["n_train"], config["dimension"])
        self.x_train = torch.tensor(x, dtype=torch.float64, requires_grad=True).to(self.device)

        self.u = torch.rand_like(self.x_train[:, :1]).to(self.device)
        self.u = self.u / torch.norm(self.u)
        self.lambda_ = torch.tensor(1.0, dtype=torch.float64, device=self.device)

        self.fixed_min_loss = config["fixed_min_loss"]
        self.min_loss = float("inf")
        self.loss = None
        self.best_lambda = None
        self.best_model_state = None
        self.checkpoint_path = config["checkpoint_path"]
        self.loss_history = []
        self.lambda_history = []

        self.optimizer = None
        self.optimizer_name = None

        self.lb = torch.tensor(config["domain_lb"], dtype=torch.float64).to(self.device)
        self.ub = torch.tensor(config["domain_ub"], dtype=torch.float64).to(self.device)
        self.d = config["dimension"]

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
        # Apply coordinate shift before feeding into the model
        x_input = coor_shift(x_input, self.lb, self.ub)
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

        Lu = compute_laplacian(u_prev, self.x_train) + self.config["M"] * u_prev

        mse_loss_fn = torch.nn.MSELoss(reduction='mean')

        tmp_loss = mse_loss_fn(Lu, self.lambda_ * self.u)

        with torch.no_grad():
            u_new = Lu / torch.norm(Lu, p=2)  # L2 normalization
        self.u = u_new

        loss = mse_loss_fn(u_prev, u_new)

        loss.backward()

        numerator = torch.sum(Lu * u_prev)
        denominator = torch.sum(u_prev ** 2) + 1e-10
        self.lambda_ = numerator / denominator

        temporal_loss = tmp_loss.item()
        lambda_val = self.lambda_.item()

        self.loss_history.append((loss.item(), temporal_loss))
        self.lambda_history.append(lambda_val)

        if temporal_loss < self.min_loss:
            self.min_loss = temporal_loss
            self.loss = loss
            self.best_lambda = lambda_val
            self.best_model_state = self.model.state_dict()

        return loss, temporal_loss, lambda_val

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

        print(" Starting training with Adam.\n")
        for it in range(self.config["adam_steps"]):
            loss, loss_val, lambda_val = self.optimize_one_epoch()
            self.optimizer.step()
            if it % 1000 == 0 or it == self.config["adam_steps"] - 1:
                print(f"[{it:05d}] Loss lambda = {loss_val:.4e} | λ_est = {lambda_val:.8f} | λ_true = {self.config['lambda_true']:.8f} Loss = {loss:.4e}")

        print(f" Best λ = {self.best_lambda:.8f} | Min Loss = {self.min_loss:.4e}")

    def compute_loss(self):
        """Compute pure loss for LBFGS closure, no state updates."""
        u_prev = self.net_u(self.x_train)
        Lu = compute_laplacian(u_prev, self.x_train) + self.config["M"] * u_prev
        mse_loss_fn = torch.nn.MSELoss(reduction='mean')
        return mse_loss_fn(Lu, self.lambda_ * self.u)

    def update_state_after_lbfgs(self):
        """Update u, lambda after LBFGS optimization step."""
        with torch.no_grad():
            u_prev = self.net_u(self.x_train)
            Lu = compute_laplacian(u_prev, self.x_train) + self.config["M"] * u_prev

            # Power method update
            self.u = Lu / (torch.norm(Lu) + 1e-10)

            numerator = torch.sum(Lu * u_prev)
            denominator = torch.sum(u_prev ** 2) + 1e-10
            self.lambda_ = numerator / denominator

            # Save best model state
            loss_val = self.compute_loss().item()
            lambda_val = self.lambda_.item()

            if loss_val < self.min_loss:
                self.min_loss = loss_val
                self.best_lambda = lambda_val
                self.best_model_state = self.model.state_dict()

    def optimize_lbfgs(self):
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=self.config["lbfgs_steps"],
            tolerance_change=1e-10,
            tolerance_grad=1e-10,
            history_size=100,
            line_search_fn=None
        )
        self.optimizer_name = 'LBFGS'
        print("Starting training with LBFGS...\n")

        def closure():
            self.optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        self.update_state_after_lbfgs()

        print(f"\nFinished LBFGS: Best λ = {self.best_lambda:.8f} | Min Loss = {self.min_loss:.4e}")

    def evaluate_and_plot(self, n_eval_points=10000):
        """
        Full evaluation + plotting. Works for 1D (plots) and higher dimensions (no plot).
        """

        print("===== Starting Evaluation =====")

        domain_lb = np.array(self.config["domain_lb"])
        domain_ub = np.array(self.config["domain_ub"])
        dim = self.config["dimension"]

        # 1️⃣ Generate evaluation points
        if dim == 1:
            x_eval = np.linspace(domain_lb[0], domain_ub[0], n_eval_points).reshape(-1, 1)
        else:
            from pyDOE import lhs
            samples = lhs(dim, n_eval_points)
            x_eval = domain_lb + (domain_ub - domain_lb) * samples

        x_eval_tensor = torch.tensor(x_eval, dtype=torch.float64, device=self.device, requires_grad=False)

        # 2️⃣ Predict u(x)
        with torch.no_grad():
            x_input = self.apply_input_transform(x_eval_tensor)
            x_input_shifted = coor_shift(x_input, self.lb, self.ub)
            u_raw = self.model(x_input_shifted)

            if not self.config.get("periodic", False):
                u_pred_tensor = self.apply_boundary_condition(x_eval_tensor, u_raw)
            else:
                u_pred_tensor = u_raw

        u_pred = u_pred_tensor.cpu().numpy()

        u_true = self.config["exact_u"](x_eval)

        u_pred = u_pred / np.linalg.norm(u_pred) * np.sqrt(u_pred.shape[0])
        u_true = u_true / np.linalg.norm(u_true) * np.sqrt(u_true.shape[0])

        # 5️⃣ Align signs
        sign = np.sign(np.mean(u_pred * u_true))
        u_pred *= sign

        # 6️⃣ Compute errors
        L2_error_u = np.sqrt(np.sum((u_true - u_pred) ** 2) / u_pred.shape[0])
        Linf_error_u = np.max(np.abs(u_true - u_pred))

        lambda_true = self.config["lambda_true"]
        lambda_pred = float(self.best_lambda)
        lambda_abs_error = abs(lambda_pred - lambda_true)
        lambda_rel_error = lambda_abs_error / abs(lambda_true)

        # 7️⃣ Print results
        print(f"L2 Error (u):         {L2_error_u:.4e}")
        print(f"L∞ Error (u):         {Linf_error_u:.4e}")
        print(f"λ predicted:          {lambda_pred:.8f}")
        print(f"λ true:               {lambda_true:.8f}")
        print(f"Absolute Error (λ):   {lambda_abs_error:.4e}")
        print(f"Relative Error (λ):   {lambda_rel_error:.4e}")
        print("===============================")

        # 8️⃣ Plot only for 1D
        if dim == 1:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.plot(x_eval, u_true, label="u_true", linewidth=2)
            plt.plot(x_eval, u_pred, '--', label="u_pred", linewidth=2)
            plt.legend()
            plt.title("Eigenfunction: True vs Predicted")
            plt.savefig("eigenfunction_plot.png")
            plt.close()

        return {
            "L2_u": L2_error_u,
            "Linf_u": Linf_error_u,
            "lambda_abs_error": lambda_abs_error,
            "lambda_rel_error": lambda_rel_error,
        }







