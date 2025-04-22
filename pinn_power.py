import os
import torch
import numpy as np
from utils import sample_lhs, compute_laplacian, periodic_transform, plot_eigenfunction

torch.set_default_dtype(torch.float64)

class PowerMethodPINN:
    def __init__(self, model, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- modelo en float64 ----------
        self.model = model.to(self.device).double()
        # ---------------------------------------

        # límites de dominio  (tensores float64 en device)
        self.lb = torch.as_tensor(config["domain_lb"], dtype=torch.float64, device=self.device)
        self.ub = torch.as_tensor(config["domain_ub"], dtype=torch.float64, device=self.device)

        # puntos de entrenamiento
        x = sample_lhs(self.lb.cpu().numpy(), self.ub.cpu().numpy(), config["n_train"]).astype(np.float64)
        self.x_train = torch.tensor(x, dtype=torch.float64, requires_grad=True, device=self.device)

        # vector u inicial
        self.u = torch.rand_like(self.x_train[:, :1])
        self.u = self.u / torch.norm(self.u)

        self.lambda_ = torch.tensor(1.0, dtype=torch.float64, device=self.device)

        # tracking
        self.min_loss        = np.inf
        self.best_lambda     = None
        self.best_model_state = None
        self.checkpoint_path = config["checkpoint_path"]
        self.loss_history    = []
        self.lambda_history  = []

        self.optimizer = None

    # -----------------------------------------------------------
    # utilidades
    # -----------------------------------------------------------
    def apply_input_transform(self, x):
        if self.config.get("periodic", False):
            return periodic_transform(x, k=self.config.get("pbc_k", 1),
                                      periods=self.config.get("periods", None)).to(self.device).double()
        return x

    def apply_boundary_condition(self, x, u):
        g = torch.ones_like(u)
        for i in range(x.shape[1]):
            xi = x[:, i:i+1]
            g *= (torch.exp(xi - self.lb[i]) - 1.0) * (torch.exp(-(xi - self.ub[i])) - 1.0)
        return g * u

    def net_u(self, x):
        u = self.model(self.apply_input_transform(x))
        if not self.config.get("periodic", False):
            u = self.apply_boundary_condition(x, u)
        return u

    # -----------------------------------------------------------
    # 1 epoch de PMNN (idéntico al paper)
    # -----------------------------------------------------------
    def optimize_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        # u^{k-1}
        u_prev = self.net_u(self.x_train)
        u_prev = u_prev / (torch.norm(u_prev) + 1e-12)

        # Lu = Δu + M u
        Lu = compute_laplacian(u_prev, self.x_train) + self.config["M"] * u_prev

        # tmp_loss = ||Lu - λ_{k-1} u_{k-1}||²
        tmp_loss = torch.mean((Lu - self.lambda_ * self.u) ** 2)

        # u^{k}  (sin gradiente)
        with torch.no_grad():
            u_new = Lu / (torch.norm(Lu) + 1e-12)
            self.lambda_ = torch.sum(Lu * u_prev) / (torch.sum(u_prev ** 2) + 1e-12)
        self.u = u_new

        # loss_PMNN = ||u_prev - u_new||²
        loss_PM = torch.mean((u_prev - u_new) ** 2)
        loss_PM.backward()
        # ---------  NO hacemos step aquí: lo hace el exterior ---------
        # self.optimizer.step()


        # track & guardar mejor
        loss_val   = tmp_loss.item()
        lambda_val = self.lambda_.item()
        self.loss_history.append(loss_val)
        self.lambda_history.append(lambda_val)

        if loss_val < self.min_loss:
            self.min_loss        = loss_val
            self.best_lambda     = lambda_val
            self.best_model_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        return loss_PM, loss_val, lambda_val

    # -----------------------------------------------------------
    # optimización Adam exterior (hace el step)
    # -----------------------------------------------------------
    def optimize_adam(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config["adam_lr"], betas=(0.9, 0.999))
        print("Starting training with Adam …\n")
        for it in range(self.config["adam_steps"]):
            loss_tensor, loss_val, lambda_val = self.optimize_one_epoch()
            self.optimizer.step()                       # step fuera

            if it % 1000 == 0 or it == self.config["adam_steps"] - 1:
                print(f"[{it:05d}] tmp_loss = {loss_val:8.2e} | λ_est = {lambda_val:10.6f}")

        # guarda mejor modelo
        if self.checkpoint_path and self.best_model_state:
            torch.save({
                "model_state_dict": self.best_model_state,
                "lambda": self.best_lambda,
                "loss": self.min_loss
            }, self.checkpoint_path)
            print(f"\n✔ Modelo guardado en {self.checkpoint_path}")
            print(f"   Best λ = {self.best_lambda:.8f} | Min tmp_loss = {self.min_loss:.2e}")

    # -----------------------------------------------------------
    # evaluación rápida (solo 1 D)
    # -----------------------------------------------------------
    def evaluate_and_plot(self):
        if self.config["dimension"] != 1:
            print("Plotting sólo soportado para 1 D.")
            return

        x_eval = torch.linspace(float(self.lb[0]), float(self.ub[0]), 1000,
                                dtype=torch.float64, device=self.device, requires_grad=True).view(-1, 1)

        with torch.no_grad():
            u_pred = self.net_u(x_eval)

        x_np   = x_eval.cpu().numpy()
        u_pred = (u_pred / torch.norm(u_pred)).cpu().numpy()
        u_true = self.config["exact_u"](x_np)
        u_true = u_true / np.linalg.norm(u_true)

        plot_eigenfunction(x_np, u_pred, u_true,
                           title="Predicted vs True eigenfunction",
                           save_path="eigenfunction_plot.png")
