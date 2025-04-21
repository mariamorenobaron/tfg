import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers, act_func=nn.Tanh()):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act_func = act_func

        # Hidden layers
        for i in range(len(layers) - 2):
            linear = nn.Linear(layers[i], layers[i+1])
            self.xavier_init(linear)
            self.layers.append(linear)

        # Final output layer (no activation)
        final = nn.Linear(layers[-2], layers[-1])
        self.xavier_init(final)
        self.output_layer = final

    def xavier_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = self.act_func(layer(x))
        return self.output_layer(x)


class ResNet(nn.Module):
    def __init__(self, in_num, out_num, block_layers, block_num, act_func=nn.Tanh()):
        super().__init__()
        self.in_linear = nn.Linear(in_num, block_layers[0])
        self.out_linear = nn.Linear(block_layers[-1], out_num)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(block_layers[0], block_layers[1]),
                act_func,
                nn.Linear(block_layers[1], block_layers[0])
            ) for _ in range(block_num)
        ])
        self.act = act_func

    def forward(self, x):
        x = self.in_linear(x)
        for block in self.blocks:
            x = x + block(x)
            x = self.act(x)
        return self.out_linear(x)
