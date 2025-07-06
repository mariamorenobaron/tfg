import os
import torch
import json
import numpy as np
from utils import sample_lhs, compute_laplacian, periodic_transform, coor_shift, apply_boundary_condition

class InversePowerMethodPINN:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        x = sample_lhs(config["domain_lb"], config["domain_ub"], config["n_train"], config["dimension"])
        self.x_train = torch.tensor(x, dtype=torch.float64, requires_grad=True).to(self.device)

        # Initial guess for u
        x_np = self.x_train.detach().cpu().numpy()
        u0_np = np.prod([np.sin(np.pi * x_np[:, i]) for i in range(config["dimension"])], axis=0)
        self.u = torch.tensor(u0_np, dtype=torch.float64, device=self.device).unsqueeze(1)
        self.u = self.u / torch.norm(self.u)

        self.lambda_ = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        self.loss = None
        self.min_loss = float("inf")
        self.best_lambda = None
        self.best_model_state = None

        self.loss_history = []
        self.lambda_history = []
        self.training_curve = []

        self.optimizer = None
        self.lb = torch.tensor(config["domain_lb"], dtype=torch.float64).to(self.device)
        self.ub = torch.tensor(config["domain_ub"], dtype=torch.float64).to(self.device)
        self.d = config["dimension"]

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

    def optimize_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        u_k = self.net_u(self.x_train)
        alpha = self.config.get("alpha", 0.0)
        Lu = -compute_laplacian(u_k, self.x_train) - alpha * u_k   # equation: Lu = -∇²u MODIFY IF NEW EQUATION | sifted Lu = -∇²u - α*u
        Lu_norm = Lu / (torch.norm(Lu, p=2) + 1e-10)

        mse_loss_fn = torch.nn.MSELoss(reduction='mean')
        loss = mse_loss_fn(Lu_norm, self.u)
        loss.backward()

        numerator = torch.sum(Lu * u_k)
        denominator = torch.sum(u_k ** 2) + 1e-10
        self.lambda_ = numerator / denominator + alpha

        with torch.no_grad():
            self.u = u_k / torch.norm(u_k, p=2)

        temporal_loss = loss.item()
        lambda_val = self.lambda_.item()

        self.loss_history.append((loss.item(), temporal_loss))
        self.lambda_history.append(lambda_val)

        self.training_curve.append({
            "epoch": len(self.lambda_history),
            "loss": float(loss.item()),
            "temporal_loss": float(temporal_loss),
            "lambda": float(lambda_val)
        })

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

        print("Starting training with Adam (IPMNN).\n")
        for it in range(self.config["adam_steps"]):
            loss, temporal_loss, lambda_val = self.optimize_one_epoch()
            self.optimizer.step()

            if it % 1000 == 0 or it == self.config["adam_steps"] - 1:
                print(f"[{it:05d}] Loss = {temporal_loss:.4e} | λ_est = {lambda_val:.8f} | λ_true = {self.config['lambda_true']:.8f}")

            # Early stopping
            if self.config.get("early_stopping", False):
                tolerance = self.config.get("tolerance", 1e-6)
                if temporal_loss < tolerance:
                    print(f"[INFO] Early stopping at iteration {it} with loss {temporal_loss:.4e}")
                    break

        print(f"Best λ = {self.best_lambda:.8f} | Min Loss = {self.min_loss:.4e}")

    def save_training_curve(self):
        path = os.path.join(self.config["save_dir"], "training_curve.json")
        with open(path, "w") as f:
            json.dump(self.training_curve, f, indent=2)

    def evaluate_and_plot(self, n_eval_points=10000):
        print("===== Evaluation (IPMNN) =====")
        domain_lb = np.array(self.config["domain_lb"])
        domain_ub = np.array(self.config["domain_ub"])
        dim = self.config["dimension"]

        if dim == 1:
            x_eval = np.linspace(domain_lb[0], domain_ub[0], n_eval_points).reshape(-1, 1)
        else:
            from pyDOE import lhs
            samples = lhs(dim, n_eval_points)
            x_eval = domain_lb + (domain_ub - domain_lb) * samples

        x_eval_tensor = torch.tensor(x_eval, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            x_input = self.apply_input_transform_periodic(x_eval_tensor)
            x_input_shifted = coor_shift(x_input, self.lb, self.ub)
            u_raw = self.model(x_input_shifted)

            if not self.config.get("periodic", False):
                u_pred_tensor = apply_boundary_condition(self.config ,x_eval_tensor, u_raw)
            else:
                u_pred_tensor = u_raw

        u_pred = u_pred_tensor.cpu().numpy()
        u_true = self.config["exact_u"](x_eval)

        u_pred = u_pred / np.linalg.norm(u_pred) * np.sqrt(u_pred.shape[0])
        u_true = u_true / np.linalg.norm(u_true) * np.sqrt(u_true.shape[0])

        sign = np.sign(np.mean(u_pred * u_true))
        u_pred *= sign

        L2_error_u = np.sqrt(np.mean((u_true - u_pred) ** 2))
        Linf_error_u = np.max(np.abs(u_true - u_pred))

        lambda_true = self.config["lambda_true"]
        lambda_pred = float(self.best_lambda)
        lambda_abs_error = abs(lambda_pred - lambda_true)
        lambda_rel_error = lambda_abs_error / abs(lambda_true)

        print(f"λ predicted:          {lambda_pred:.8f}")
        print(f"λ true:               {lambda_true:.8f}")
        print(f"Absolute Error (λ):   {lambda_abs_error:.4e}")
        print(f"Relative Error (λ):   {lambda_rel_error:.4e}")
        print(f"L2 Error (u):         {L2_error_u:.4e}")
        print(f"L∞ Error (u):         {Linf_error_u:.4e}")
        print("===============================")

        if dim == 1:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.plot(x_eval, u_true, label="u_true", linewidth=2)
            plt.plot(x_eval, u_pred, '--', label="u_pred", linewidth=2)
            plt.legend()
            plt.title("Eigenfunction: True vs Predicted")
            plt.show()
