import torch
import numpy as np
import os
from utils import sample_lhs, compute_laplacian, periodic_transform, plot_eigenfunction

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


    def coor_shift(self, X, lb, ub):
        X_shift = 2.0 * (X - lb) / (ub - lb) - 1.0
        return X_shift

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
        x_input = self.coor_shift(x_input, self.lb, self.ub)
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

        self.loss_history.append(temporal_loss)
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
        print("Starting training with LBFGS.\n")

        def closure():
            self.optimizer.zero_grad()
            loss, loss_val, lambda_val = self.optimize_one_epoch()
            loss.backward()
            return loss

        # This will internally call the closure multiple times
        self.optimizer.step(closure)

        print(f"\nFinished LBFGS: Best λ = {self.best_lambda:.8f} | Min Loss = {self.min_loss:.4e}")


    def adam_lbfgs(self):



    def evaluate_errors_and_plot(self, x_eval, u_pred, u_true):
        u_pred = u_pred / np.linalg.norm(u_pred)

        l2_error = np.linalg.norm(u_pred - u_true) / np.sqrt(u_pred.shape[0])
        print(f"L2 Error: {l2_error:.4e}")

        rel_error_u = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        print(f"Relative Error (u): {rel_error_u:.4e}")

        lambda_pred = self.best_lambda
        lambda_true = self.config["lambda_true"]
        lambda_error = np.abs(lambda_pred - lambda_true)
        print(f"Lambda Error: {lambda_error:.4e}")

        rel_error_lambda = lambda_error / lambda_true
        print(f"Relative Error (λ): {rel_error_lambda:.4e}")


    def evaluate_and_plot(self):
        """Evaluate and plot the eigenfunction."""
        if self.config["dimension"] != 1:
            print("Evaluation and plotting are only implemented for 1D problems.")
            return

        x_eval = torch.linspace(
            self.config["domain_lb"][0], self.config["domain_ub"][0], 1000
        ).view(-1, 1).to(self.device)
        x_eval.requires_grad_(True)

        # Model prediction
        with torch.no_grad():
            x_input = self.apply_input_transform(x_eval)
            u_raw = self.model(x_input)
            u_pred = self.apply_boundary_condition(x_eval, u_raw) if not self.config.get("periodic", False) else u_raw

        # Convert to numpy
        x_np = x_eval.detach().cpu().numpy()
        u_pred = u_pred.detach().cpu().numpy()

        # Compute true solution
        u_true = self.config["exact_u"](x_np)
        u_true = u_true / np.linalg.norm(u_true)

        # Plot and evaluate
        plot_eigenfunction(
            x_np, u_pred, u_true,
            title="Predicted vs True Eigenfunction",
            save_path="eigenfunction_plot.png"
        )

        self.evaluate_errors_and_plot(x_np, u_pred, u_true)




