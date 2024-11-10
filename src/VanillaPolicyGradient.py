import gymnasium as gym
import torch
from typing import TypedDict
from tqdm import tqdm
import numpy as np
from LoggerInterface import LoggerInterface


class VPGConfig(TypedDict):
    pi_lr: float
    vf_lr: float
    train_v_iters: int
    steps_per_epoch: int
    gamma: float
    lam: float
    seed: int
    device: str
    log_frequency: int
    episode_cnt: int
    env_cnt: int


def vpg_policy_pseudoloss(action_log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    return -(action_log_probs * advantages).mean()

def discounted_cumsum(x: torch.Tensor, done: torch.Tensor, gamma: float):
    ret = torch.zeros_like(x, dtype=torch.float32)

    ret[:, -1] = x[:, -1]
    for t in reversed(range(ret.shape[1] - 1)):
        ret[:, t] = x[:, t] + (~done[:, t]) * gamma * ret[:, t+1]

    return ret


class VanillaPolicyGradient:
    def __init__(self, envs: gym.vector.AsyncVectorEnv, policy_network: torch.nn.Module, value_network: torch.nn.Module, config: VPGConfig, logger: LoggerInterface | None = None):
        self.envs = envs
        self.config = config
        self.logger = logger
        self.policy_network = policy_network.to(self.config['device'])
        self.value_network = value_network.to(self.config['device'])

        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

        self.value_memory = []
        self.action_log_prob_memory = []
        self.reward_memory = []
        self.done_memory = []
        self.observation_memory = []

        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), self.config['pi_lr'])

        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), self.config['vf_lr'])

        self.vf_loss = torch.nn.MSELoss()

    def train(self) -> None:
        if self.logger is not None:
            self.logger.watch(self.policy_network,
                              vpg_policy_pseudoloss, self.config['log_frequency'])

            self.logger.watch(self.value_network,
                              self.vf_loss, self.config['log_frequency'])

        for episode in tqdm(range(self.config['episode_cnt'])):

            observations, _ = self.envs.reset()
            steps_per_epoch = self.config['steps_per_epoch']
            batch_size = self.config['env_cnt']

            batch_entropy = 0
            episodes_finished = 0
            episode_scores = np.zeros((batch_size))
            score_sum = 0

            for _ in range(steps_per_epoch):
                self.observation_memory.append(observations)
                log_probs, actions, entropies = self.__choose_action(observations)
                values = self.__get_values(observations)
                observations, rewards, terminated, truncated, _ = self.envs.step(actions)
                done = np.logical_or(terminated, truncated)

                self.value_memory.append(values)
                self.action_log_prob_memory.append(log_probs)
                self.reward_memory.append(rewards)
                self.done_memory.append(done)

                episode_scores += rewards
                score_sum += np.sum(done * episode_scores)
                episode_scores *= 1.0 - done
                episodes_finished += np.sum(done)
                batch_entropy += np.sum(entropies)

            batch_entropy /= steps_per_epoch * batch_size
            average_score = score_sum / episodes_finished
            self.__learn()

            if self.logger is not None:
                self.logger.log_episode_performance(episode, {
                    'episodes_finished': episodes_finished,
                    'average_score': average_score,
                    'batch_entropy': batch_entropy
                })

    def __choose_action(self, observations: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        observation_tensor = torch.from_numpy(
            observations).to(self.config['device'])
        logits = self.policy_network(observation_tensor)
        dist = torch.distributions.Categorical(logits=logits)

        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        return log_probs, actions.detach().numpy(), entropies.detach().numpy()

    def __get_values(self, observations: np.ndarray) -> torch.Tensor:
        observation_tensor = torch.from_numpy(
            observations).to(self.config['device'])
        values = self.value_network(observation_tensor)

        return values.squeeze()

    def __learn(self) -> None:
        gamma = self.config['gamma']
        lam = self.config['lam']

        value_tensor = torch.stack(self.value_memory).T
        reward_tensor = torch.tensor(np.array(self.reward_memory)).T
        done_tensor = torch.tensor(np.array(self.done_memory)).T

        detached_value_tensor = value_tensor.detach()
        deltas = reward_tensor - detached_value_tensor
        deltas[:, :-1] += done_tensor[:, :-1] * gamma * detached_value_tensor[:, 1:] 
        advantage_tensor = discounted_cumsum(deltas, done_tensor, gamma*lam)

        rtg = discounted_cumsum(reward_tensor, done_tensor, gamma).T.flatten().unsqueeze(1)
    
        log_prob_tensor = torch.stack(self.action_log_prob_memory).T
        policy_loss = vpg_policy_pseudoloss(log_prob_tensor, advantage_tensor)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()

        observation_tensor = torch.tensor(np.array(self.observation_memory)).flatten(0,1)
        for _ in range(self.config['train_v_iters']):
            value = self.value_network(observation_tensor)
            value_function_loss = self.vf_loss(value, rtg)
            value_function_loss.backward()
            self.value_optimizer.step()
            self.value_optimizer.zero_grad()

        self.value_memory = []
        self.action_log_prob_memory = []
        self.reward_memory = []
        self.done_memory = []
        self.observation_memory = []
