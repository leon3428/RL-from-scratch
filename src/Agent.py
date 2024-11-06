import torch
from PolicyGradientNetwork import PolicyGradientNetwork
import numpy as np

def criterion(action_log_probs: torch.Tensor, g: torch.Tensor):
    loss = -(action_log_probs * g).mean()
    return loss

class Agent:
    def __init__(self, device: torch.device, alpha: float = 0.003, gamma: float = 0.99, observation_dims: int = 2, fc1_dims: int = 256, fc2_dims: int = 256, action_dims: int = 4):
        self.alpha = alpha
        self.gamma = gamma
        self.action_dims = action_dims
        self.device = device

        self.action_log_prob_memory = []
        self.reward_memory = []

        self.policy = PolicyGradientNetwork(
            observation_dims, fc1_dims, fc2_dims, action_dims).to(device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), alpha)

    def choose_action(self, observation: np.ndarray) -> tuple[int, float, float]:
        observation = torch.from_numpy(observation).to(self.device)

        logits = self.policy(observation)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.detach().item(), log_prob, entropy.detach().item()

    def store_transition(self, action_log_prob: torch.Tensor, reward: float):
        self.action_log_prob_memory.append(action_log_prob)
        self.reward_memory.append(reward)

    def learn(self):
        action_log_probs = torch.stack(self.action_log_prob_memory)        
        rewards = torch.tensor(self.reward_memory, dtype=torch.float32)

        G = torch.zeros_like(rewards, dtype=torch.float32)
        G[-1] = rewards[-1]
        for t in range(len(rewards)-2, -1, -1):
            G[t] = rewards[t] + self.gamma * G[t+1]

        G = G.to(self.device)

        loss = criterion(action_log_probs, G)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.action_log_prob_memory = []
        self.reward_memory = []
