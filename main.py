from config import CONFIG
from train_utils import train_adam_with_mlp, train_adam_with_resnet

def train_selected():
    architecture = CONFIG["architecture"]

    if architecture == "MLP":
        pinn = train_adam_with_mlp(CONFIG)
    elif architecture == "ResNet":
        pinn = train_adam_with_resnet(CONFIG)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return pinn

if __name__ == "__main__":
    trained_pinn = train_selected()
    print("Training completed.")