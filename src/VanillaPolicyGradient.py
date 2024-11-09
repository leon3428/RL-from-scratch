import gymnasium as gym
import torch
from typing import TypedDict
from tqdm import tqdm
import numpy as np
from LoggerInterface import LoggerInterface

class VPGConfig(TypedDict):
    lr: float
    gamma: float
    seed: int
    device: str
    log_frequency: int
    episode_cnt: int

def vpg_pseudoloss(action_log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return -(action_log_probs * returns).mean()

class VanillaPolicyGradient:
    def __init__(self, env: gym.Env, policy_network: torch.nn.Module, config: VPGConfig, logger: LoggerInterface | None = None):
        self.env = env
        self.config = config
        self.logger = logger
        self.policy_network = policy_network.to(self.config['device'])

        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

        self.action_log_prob_memory = []
        self.reward_memory = []

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), self.config['lr'])

    def train(self) -> None:
        if self.logger is not None:
            self.logger.watch(self.policy_network, vpg_pseudoloss, self.config['log_frequency'])

        for episode in tqdm(range(self.config['episode_cnt'])):

            observation, _ = self.env.reset()
            episode_done = False
            episode_score = 0
            episode_length = 0
            episode_entropy = 0

            while not episode_done:
                log_prob, action, entropy = self.__choose_action(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)

                self.action_log_prob_memory.append(log_prob)
                self.reward_memory.append(reward)

                episode_done = terminated or truncated
                episode_score += reward
                episode_length += 1
                episode_entropy += entropy

            episode_entropy /= episode_length
            self.__learn()      

            if self.logger is not None:
                self.logger.log_episode_performance(episode, {
                    'episode_length': episode_length,
                    'episode_score': episode_score,
                    'episode_entropy': episode_entropy
                })


    def __choose_action(self, observation: np.ndarray) -> tuple[torch.Tensor, int, float]:
        observation_tensor = torch.from_numpy(observation).to(self.config['device'])
        logits = self.policy_network(observation_tensor)
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, action.detach().item(), entropy.detach().item()
    
    def __learn(self) -> None:
        action_log_probs = torch.stack(self.action_log_prob_memory)
        rewards = torch.tensor(self.reward_memory, dtype=torch.float32)

        returns = torch.zeros_like(rewards, dtype=torch.float32)
        returns[-1] = rewards[-1]
        for t in range(len(rewards)-2, -1, -1):
            returns[t] = rewards[t] + self.config['gamma'] * returns[t+1]

        returns = returns.to(self.config['device'])

        loss = vpg_pseudoloss(action_log_probs, returns)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.action_log_prob_memory = []
        self.reward_memory = []