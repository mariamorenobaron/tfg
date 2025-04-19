from config import CONFIG
from model import MLP, ResNet
from pinn_power import PowerMethodPINN
from utils import save_model
from datetime import timedelta

if __name__ == "__main__":
    if CONFIG["periodic"]:
        input_dim = CONFIG["dimension"] * 2 * CONFIG["pbc_k"]
    else:
        input_dim = CONFIG["dimension"]

    CONFIG["input_dim"] = input_dim

    if CONFIG["architecture"] == "MLP":
        layers = [input_dim] + [CONFIG["width"]] * CONFIG["depth"] + [1]
        model = MLP(layers)
    else:
        model = ResNet(in_num=input_dim, out_num=1, block_layers=[CONFIG["width"]] * 2, block_num=CONFIG["depth"])

    pinn = PowerMethodPINN(model=model, config=CONFIG)

    if CONFIG["optimizer"] == "Adam":
        pinn.train_adam()
    elif CONFIG["optimizer"] == "LBFGS":
        pinn.train_lbfgs()
    elif CONFIG["optimizer"] == "Adam+LBFGS":
        pinn.train_adam_then_lbfgs()

    pinn.evaluate()
    save_model(model, CONFIG, folder="saved_model")
    print("\n------ Entrenamiento Finalizado ------")
    print(f"Dimensión:               {CONFIG['dimension']}")
    print(f"Tipo de condiciones:     {'Periódicas' if CONFIG['periodic'] else 'Dirichlet'}")
    print(f"Arquitectura:            {CONFIG['depth']} capas × {CONFIG['width']} neuronas")
    print(f"Número de puntos:        {CONFIG['n_train']}")
    print(f"Optimizador:             {CONFIG['optimizer']}")
    print(f"Valor real λ:            {CONFIG['lambda_true']:.8f}")
    print(f"Valor estimado λ:        {pinn.lambda_:.8f}")
    print(f"Error absoluto:          {abs(CONFIG['lambda_true'] - pinn.lambda_):.4e}")
    print(f"Error relativo:          {abs(CONFIG['lambda_true'] - pinn.lambda_) / abs(CONFIG['lambda_true']):.2%}")
