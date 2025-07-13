import torch
import os
import json
import numpy as np
from utils import sample_lhs, periodic_transform, coor_shift, compute_laplacian, apply_boundary_condition

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

        self.min_loss = float("inf")
        self.best_lambda = None
        self.best_model_state = None
        self.best_iteration = None
        self.loss_history = []
        self.lambda_history = []
        self.training_curve = []

        self.lb = torch.tensor(config["domain_lb"], dtype=torch.float64).to(self.device)
        self.ub = torch.tensor(config["domain_ub"], dtype=torch.float64).to(self.device)

        self.optimizer = None
        self.run_dir = config["save_dir"]

        self.u_infty_history = []

        if config["dimension"] == 1:
            self.n_test_points = 2000
        elif config["dimension"] == 2:
            self.n_test_points = 20000
        else:
            self.n_test_points = 100000

        x_test_np = sample_lhs(config["domain_lb"], config["domain_ub"], self.n_test_points, config["dimension"])
        self.x_test = torch.tensor(x_test_np, dtype=torch.float64).to(self.device)
        u_true = config["exact_u"](x_test_np)
        self.u_true = u_true / np.linalg.norm(u_true) * np.sqrt(len(u_true))

    def apply_input_transform_periodic(self, x):
        if self.config.get("periodic", False):
            return periodic_transform(x, k=self.config.get("pbc_k", 1), periods=self.config.get("periods", None))
        return x

    def net_u(self, x):
        x_input = self.apply_input_transform_periodic(x)
        x_input = coor_shift(x_input, self.lb, self.ub)
        u_pred = self.model(x_input)
        if not self.config.get("periodic", False):
            u_pred = apply_boundary_condition(self.config, x, u_pred)
        return u_pred

    def compute_u_infty(self):
        with torch.no_grad():
            x_input = self.apply_input_transform_periodic(self.x_test)
            x_shifted = coor_shift(x_input, self.lb, self.ub)
            u_pred_tensor = self.model(x_shifted)
            if not self.config.get("periodic", False):
                u_pred_tensor = apply_boundary_condition(self.config, self.x_test, u_pred_tensor)
            u_pred = u_pred_tensor.cpu().numpy()
            u_pred = u_pred / np.linalg.norm(u_pred) * np.sqrt(len(u_pred))
            sign = np.sign(np.mean(u_pred * self.u_true))
            u_pred *= sign
            error_inf = np.max(np.abs(u_pred - self.u_true))
            return error_inf

    def optimize_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        u_prev = self.net_u(self.x_train)
        Lu = compute_laplacian(u_prev, self.x_train) + self.config["M"] * u_prev

        mse_loss_fn = torch.nn.MSELoss(reduction='mean')
        tmp_loss = mse_loss_fn(Lu, self.lambda_ * self.u)

        with torch.no_grad():
            u_new = Lu / torch.norm(Lu, p=2)

        self.u = u_new
        self.u = self.u / torch.norm(self.u)

        loss = mse_loss_fn(u_prev, u_new)
        loss.backward()

        numerator = torch.sum(Lu * u_prev)
        denominator = torch.sum(u_prev ** 2) + 1e-10
        self.lambda_ = numerator / denominator

        self.loss_history.append((loss.item(), tmp_loss.item()))
        self.lambda_history.append(self.lambda_.item())

        u_infty = self.compute_u_infty()
        self.u_infty_history.append(u_infty)

        self.training_curve.append({
            "epoch": len(self.lambda_history),
            "loss": float(loss.item()),
            "temporal_loss": float(tmp_loss.item()),
            "lambda": float(self.lambda_.item()),
            "u_infty": float(u_infty)
        })

        #if tmp_loss.item() < self.min_loss:
        if loss.item() < self.min_loss:
            self.min_loss = loss.item()
            self.best_lambda = self.lambda_.item()
            self.best_model_state = self.model.state_dict()
            self.best_iteration = len(self.lambda_history)

        return loss, tmp_loss, self.lambda_

    def optimize_adam(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["adam_lr"])
        print("Starting Adam training...")

        for it in range(self.config["adam_steps"]):
            loss, temporal_loss, lambda_val = self.optimize_one_epoch()
            self.optimizer.step()

            if it % 1000 == 0 or it == self.config["adam_steps"] - 1:
                print(f"[{it:05d}] Loss = {loss:.4e} | λ_est = {lambda_val:.6f} | λ_true = {self.config['lambda_true']:.6f}")

            if self.config.get("early_stopping", False):
                tolerance = self.config.get("tolerance", 1e-6)
                if temporal_loss < tolerance:
                    print(f"[INFO] Early stopping at iteration {it} with loss {temporal_loss:.4e}")
                    break

        print(f"Finished Adam. Best λ = {self.best_lambda:.6f} | Min Loss = {self.min_loss:.4e}")
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def save_training_curve(self):
        path = os.path.join(self.run_dir, "training_curve.json")
        with open(path, "w") as f:
            json.dump(self.training_curve, f, indent=2)

    def evaluate_and_plot(self, n_eval_points=10000):
        print("===== Evaluation =====")
        dim = self.config["dimension"]
        domain_lb = np.array(self.config["domain_lb"])
        domain_ub = np.array(self.config["domain_ub"])

        if dim == 1:
            x_eval = np.linspace(domain_lb[0], domain_ub[0], n_eval_points).reshape(-1, 1)
        else:
            samples = sample_lhs(domain_lb, domain_ub, n_eval_points, dim)
            x_eval = domain_lb + (domain_ub - domain_lb) * samples

        x_tensor = torch.tensor(x_eval, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            x_input = self.apply_input_transform_periodic(x_tensor)
            x_input_shifted = coor_shift(x_input, self.lb, self.ub)
            u_raw = self.model(x_input_shifted)

            if not self.config.get("periodic", False):
                u_pred_tensor = apply_boundary_condition(self.config, x_tensor, u_raw)
            else:
                u_pred_tensor = u_raw

        u_pred = u_pred_tensor.cpu().numpy()
        u_true = self.config["exact_u"](x_eval)

        u_pred = u_pred / np.linalg.norm(u_pred) * np.sqrt(u_pred.shape[0])
        u_true = u_true / np.linalg.norm(u_true) * np.sqrt(u_true.shape[0])
        sign = np.sign(np.mean(u_pred * u_true))
        u_pred *= sign

        L2_error = np.sqrt(np.mean((u_true - u_pred) ** 2))
        Linf_error = np.max(np.abs(u_true - u_pred))
        lambda_true = self.config["lambda_true"]
        lambda_pred = self.best_lambda
        rel_error = abs(lambda_pred - lambda_true) / abs(lambda_true)

        print(f"λ_pred = {lambda_pred:.6f} | λ_true = {lambda_true:.6f} | Rel. Error = {rel_error:.4e}")
        print(f"L2(u) Error = {L2_error:.4e} | Linf(u) Error = {Linf_error:.4e}")
        print("=======================")

        if dim == 1:
            import matplotlib.pyplot as plt
            plt.plot(x_eval, u_true, label="u_true")
            plt.plot(x_eval, u_pred, '--', label="u_pred")
            plt.legend()
            plt.title("Eigenfunction (1D)")
            plt.show()
