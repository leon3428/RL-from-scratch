from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super(MLP, self).__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least two elements (input and output layer sizes).")
        
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
