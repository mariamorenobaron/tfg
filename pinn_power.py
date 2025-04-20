import torch
import numpy as np
from utils import sample_lhs, compute_laplacian, periodic_transform

class PowerMethodPINN:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.x_train = self.sample_points(config["n_train"])
        self.u = torch.rand_like(self.x_train[:, :1]).to(self.device)
        self.u = self.u / torch.norm(self.u)

        self.lambda_ = 1.0                # Último valor calculado
        self.best_lambda = None           # Mejor valor
        self.min_loss = float("inf")      # Mejor loss
        self.checkpoint_path = config["checkpoint_path"]

    def sample_points(self, N):
        x = sample_lhs(self.config["domain_lb"], self.config["domain_ub"], N)
        return torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device)

    def apply_input_transform(self, x):
        if self.config.get("periodic", False):
            return periodic_transform(x, k=self.config.get("pbc_k", 1), periods=self.config.get("periods", None))
        else:
            return x

    def apply_boundary_condition(self, x, u):
        g = torch.ones_like(u)
        for i in range(x.shape[1]):
            xi = x[:, i:i+1]
            lb = self.config["domain_lb"][i]
            ub = self.config["domain_ub"][i]
            g *= (xi - lb) * (ub - xi)
        return g * u

    def loss_fn(self, x):
        # 1. Transform input if periodic
        x_input = self.apply_input_transform(x)
        
        # 2. Forward pass: u^{k-1}
        u_prev = self.model(x_input)
    
        # 3. Impose Dirichlet condition if not periodic
        if not self.config.get("periodic", False):
            u_prev = self.apply_boundary_condition(x, u_prev)
    
        # 4. Normalize u^{k-1}
        u_prev = u_prev / (torch.norm(u_prev) + 1e-10)
    
        # 5. Compute L u^{k-1} = Δu + M·u
        Lu = compute_laplacian(u_prev, x) + self.config["M"] * u_prev
    
        # 6. Power iteration: u^{k} ← Lu / ||Lu||
        u_new = Lu.detach() / (torch.norm(Lu.detach()) + 1e-10)
    
        # 7. PMNN loss: ||u_prev - u_new||^2
        loss = torch.mean((u_prev - u_new) ** 2)
    
        # 8. Estimate eigenvalue (Rayleigh quotient)
        numerator = torch.sum(Lu * u_prev)
        denominator = torch.sum(u_prev ** 2)
        self.lambda_ = (numerator / denominator).item()
    
        # 9. Update u for next iteration
        self.u = u_new
    
        # 10. Save best model
        if loss.item() < self.min_loss:
            self.min_loss = loss.item()
            self.best_lambda = self.lambda_
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "lambda_": self.best_lambda
            }, self.checkpoint_path)
    
        return loss

    def train_adam_then_lbfgs(self):
        print("Starting Adam training...")
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config["adam_lr"])
        for it in range(self.config["adam_steps"]):
            opt.zero_grad()
            loss = self.loss_fn(self.x_train)
            loss.backward()
            opt.step()

            if it % 1000 == 0:
                print(f"[{it:5d}] Loss = {loss.item():.4e} | λ_est = {self.lambda_:.6f}")

        # Cargar mejor modelo antes de LBFGS
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.lambda_ = checkpoint["lambda_"]

        print("Starting LBFGS fine-tuning...")
        self.train_lbfgs()

    def train_lbfgs(self):
        opt = torch.optim.LBFGS(self.model.parameters(), max_iter=self.config["lbfgs_steps"])

        def closure():
            opt.zero_grad()
            loss = self.loss_fn(self.x_train)
            loss.backward()
            return loss

        opt.step(closure)

    def evaluate(self):
        x_eval = self.sample_points(5000).detach().cpu().numpy()
        x_tensor = torch.tensor(x_eval, dtype=torch.float32).to(self.device)
        x_input = self.apply_input_transform(x_tensor)
        with torch.no_grad():
            u_raw = self.model(x_input)
            if not self.config.get("periodic", False):
                u_pred = self.apply_boundary_condition(x_tensor, u_raw)
            else:
                u_pred = u_raw
            u_pred = u_pred.cpu().numpy()

        u_true = self.config["exact_u"](x_eval)
        u_pred /= np.linalg.norm(u_pred)
        u_true /= np.linalg.norm(u_true)
        error = np.linalg.norm(u_pred - u_true) / np.sqrt(u_pred.shape[0])

        print(f"\n[Evaluation]")
        print(f"True λ:       {self.config['lambda_true']:.8f}")
        print(f"Best λ est.:  {self.best_lambda:.8f}")
        print(f"L2 error:     {error:.4e}")
