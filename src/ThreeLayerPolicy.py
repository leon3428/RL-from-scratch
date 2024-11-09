from torch import nn
import torch
import torch.nn.functional as F


class ThreeLayerPolicy(nn.Module):
    def __init__(self, observation_dims: int, fc1_dims: int, fc2_dims: int, action_dims: int):
        super().__init__()

        self.fc1 = nn.Linear(observation_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, action_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.fc3(h)

        return logits
