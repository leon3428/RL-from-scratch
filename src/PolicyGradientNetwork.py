from torch import nn
import torch
import torch.nn.functional as F

class PolicyGradientNetwork(nn.Module):
    def __init__(self, observation_dims: int, fc1_dims: int, fc2_dims: int, action_dims: int):
        super().__init__()
        self.observation_dims = observation_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dims = action_dims

        self.fc1 = nn.Linear(self.observation_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.action_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h =  F.relu(self.fc2(h))
        logits = self.fc3(h)

        return logits
