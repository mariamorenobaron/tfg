import torch
import numpy as np
from utils import sample_lhs, compute_laplacian, periodic_transform, coor_shift


class InversePowerMethodPINN:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        x = sample_lhs(config["domain_lb"], config["domain_ub"], config["n_train"], config["dimension"])
        self.x_train = torch.tensor(x, dtype=torch.float64, requires_grad=True).to(self.device)

        # Inicializar autofunción con una función suave (ej: producto de senos)
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

        self.optimizer = None

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
            xi = x[:, i:i + 1]
            g *= (torch.exp(xi - lb[i]) - 1.0) * (torch.exp(-(xi - ub[i])) - 1.0)
        return g * u

    def net_u(self, x):
        x_input = self.apply_input_transform(x)
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

        # Paso 1: u_k ← red neuronal
        u_k = self.net_u(self.x_train)

        # Paso 2: aplicar operador (shift opcional)
        alpha = self.config.get("alpha", 0.0)
        Lu = -compute_laplacian(u_k, self.x_train) - alpha * u_k

        # Paso 3: normalización
        Lu_norm = Lu / (torch.norm(Lu, p=2) + 1e-10)

        # Paso 4: loss de IPMNN
        mse_loss_fn = torch.nn.MSELoss(reduction='mean')
        loss = mse_loss_fn(Lu_norm, self.u)

        # Paso 5: backpropagation
        loss.backward()

        # Paso 6: estimar λ (corrigiendo con α si se usó shift)
        numerator = torch.sum(Lu * u_k)
        denominator = torch.sum(u_k ** 2) + 1e-10
        self.lambda_ = numerator / denominator + alpha

        # Paso 7: actualizar autofunción
        with torch.no_grad():
            self.u = u_k / torch.norm(u_k, p=2)

        # Tracking
        temporal_loss = loss.item()
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

        print("Starting training with Adam (IPMNN).\n")
        for it in range(self.config["adam_steps"]):
            loss, loss_val, lambda_val = self.optimize_one_epoch()
            self.optimizer.step()
            if it % 1000 == 0 or it == self.config["adam_steps"] - 1:
                print(f"[{it:05d}] Loss = {loss_val:.4e} | λ_est = {lambda_val:.8f} | λ_true = {self.config['lambda_true']:.8f}")

        print(f"Best λ = {self.best_lambda:.8f} | Min Loss = {self.min_loss:.4e}")
