import torch
from utils import sample_lhs, compute_laplacian, periodic_transform
import numpy as np

class PowerMethodPINN:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.x_train = self.sample_points(config["n_train"])
        self.u = torch.rand_like(self.x_train[:, :1]).to(self.device)
        self.u = self.u / torch.norm(self.u)
        self.lambda_ = 1.0

    def sample_points(self, N):
        x = sample_lhs(self.config["domain_lb"], self.config["domain_ub"], N)
        return torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device)

    def apply_input_transform(self, x):
        if self.config.get("periodic", False):
            return periodic_transform(x, k=self.config.get("pbc_k", 1), periods=self.config.get("periods", None))
        else:
            return x

    def loss_fn(self, x):
        x_input = self.apply_input_transform(x)
        u_pred = self.model(x_input)
        u_pred = u_pred / torch.norm(u_pred)
        Lu = compute_laplacian(u_pred, x) + self.config["M"] * u_pred
        loss = torch.mean((Lu - self.lambda_ * self.u) ** 2)
        self.u = Lu.detach() / torch.norm(Lu.detach())
        return loss

    def train_adam(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config["adam_lr"])
        for _ in range(self.config["adam_steps"]):
            opt.zero_grad()
            loss = self.loss_fn(self.x_train)
            loss.backward()
            opt.step()

    def train_lbfgs(self):
        opt = torch.optim.LBFGS(self.model.parameters(), max_iter=self.config["lbfgs_steps"])
        def closure():
            opt.zero_grad()
            loss = self.loss_fn(self.x_train)
            loss.backward()
            return loss
        opt.step(closure)

    def train_adam_then_lbfgs(self):
        best_loss = float("inf")
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config["adam_lr"])
        for _ in range(self.config["adam_steps"]):
            opt.zero_grad()
            loss = self.loss_fn(self.x_train)
            loss.backward()
            opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.model.state_dict(), self.config["checkpoint_path"])
        self.model.load_state_dict(torch.load(self.config["checkpoint_path"]))
        self.train_lbfgs()

    def evaluate(self):
        x_eval = self.sample_points(5000).detach().cpu().numpy()
        x_tensor = torch.tensor(x_eval, dtype=torch.float32).to(self.device)
        x_input = self.apply_input_transform(x_tensor)
        with torch.no_grad():
            u_pred = self.model(x_input).cpu().numpy()
        u_true = self.config["exact_u"](x_eval)
        u_pred /= np.linalg.norm(u_pred)
        u_true /= np.linalg.norm(u_true)
        error = np.linalg.norm(u_pred - u_true) / np.sqrt(u_pred.shape[0])
        print(f"[Eval] Relative L2 error: {error:.4e}")
