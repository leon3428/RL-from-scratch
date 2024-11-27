from typing import TypedDict
import gymnasium as gym
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import Callable
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
from LoggerInterface import LoggerInterface
from utils import discounted_cumsum

class VPGConfig(TypedDict):
    pi_lr: float
    vf_lr: float
    train_v_iters: int
    steps_per_epoch_pre_worker: int
    epoch_cnt: int
    gamma: float
    lam: float
    seed: int
    device: str
    log_frequency: int


class VPGEpochBuffer:
    def __init__(self, steps: int, observation_dims: int):
        self.steps = steps
        self.len = 0

        self.observations = torch.zeros((steps, observation_dims), dtype=torch.float)
        self.actions = torch.zeros((steps,), dtype=torch.int)
        self.rewards = torch.zeros((steps,), dtype=torch.float)
        self.resets = torch.zeros((steps,), dtype=torch.bool)

    def store_step(self, observation: torch.Tensor, action: int, reward: float, reset: bool) -> None:
        assert self.len < self.steps

        self.observations[self.len] = observation
        self.actions[self.len] = action
        self.rewards[self.len] = reward
        self.resets[self.len] = reset
        self.len += 1

    def reset(self) -> None:
        self.len = 0

def act(actor: DDP, observation: torch.Tensor):
    with torch.no_grad():
        logits = actor(observation)

    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample().item()

    return action

def get_log_probs(actor: DDP, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = actor(observations)

    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    entorpy = dist.entropy()

    return log_probs, entorpy.detach()

def actor_loss(log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    loss = -(log_probs * advantages).mean()
    return loss

def critic_loss(values: torch.Tensor, rtgs: torch.Tensor) -> torch.Tensor:
    loss = ((values - rtgs)**2).mean()
    return loss

def normalize_epoch(x: torch.Tensor, world_size: int) -> torch.Tensor:
    sum = x.sum()
    sum_sq = torch.sum(x ** 2)

    dist.all_reduce(sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum_sq, op=dist.ReduceOp.SUM)

    mean = sum / (len(x) * world_size)
    variance = sum_sq / (len(x) * world_size) - mean**2

    return (x - mean) / torch.sqrt(variance)

def vpg_process(
    rank: int, 
    world_size: int, 
    create_env: Callable[[], gym.Env], 
    create_actor: Callable[[], torch.nn.Module], 
    create_critic: Callable[[], torch.nn.Module], 
    config: VPGConfig, 
    create_logger: Callable[[dict], LoggerInterface] | None
):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    env = create_env()
    actor = DDP(create_actor()).to(config['device'])
    critic = DDP(create_critic()).to(config['device'])

    logger = None
    if rank == 0 and (create_logger is not None):
        logger = create_logger(config)
        logger.save_source('src/*.py') 
        logger.watch(actor, actor_loss, config['log_frequency'])
        logger.watch(critic, critic_loss, config['log_frequency'])

    actor_optimizer = torch.optim.Adam(actor.parameters(), config['pi_lr'])
    critic_optimizer = torch.optim.Adam(critic.parameters(), config['vf_lr'])

    num_steps = config['steps_per_epoch_pre_worker']
    buffer = VPGEpochBuffer(num_steps, env.observation_space.shape[0])

    for epoch in range(config['epoch_cnt']):
        #sample

        score = 0
        length = 0

        episodes_finished = 0
        epoch_score_sum = 0
        epoch_length_sum = 0

        observation, _ = env.reset()
        observation = torch.from_numpy(observation).to(config['device'])

        for _ in range(num_steps):
            action = act(actor, observation)
            observation_, reward, terminated, truncated, _ = env.step(action)
            reset = terminated or truncated

            buffer.store_step(observation, action, reward, reset)

            score += reward
            length += 1

            if reset:
                observation, _ = env.reset()
                observation = torch.from_numpy(observation).to(config['device'])
                epoch_score_sum += score
                score = 0
                epoch_length_sum += length
                length = 0
                episodes_finished += 1
            else:
                observation = torch.from_numpy(observation_).to(config['device'])

        # learn
        with torch.no_grad():
            values: torch.Tensor = critic(buffer.observations).squeeze()

        deltas = buffer.rewards - values
        deltas[:-1] += (~buffer.resets[:-1]) * config['gamma'] * values[1:] 
        advantages = discounted_cumsum(deltas, buffer.resets, config['gamma'] * config['lam'])
        advantages = normalize_epoch(advantages, world_size)
        rtgs = discounted_cumsum(buffer.rewards, buffer.resets, config['gamma'])

        log_probs, entropies = get_log_probs(actor, buffer.observations, buffer.actions)
        actor_optimizer.zero_grad()
        pi_loss = actor_loss(log_probs, advantages)
        pi_loss.backward()
        actor_optimizer.step()

        for _ in range(config['train_v_iters']):
            critic_optimizer.zero_grad()
            values = critic(buffer.observations).squeeze()
            v_loss = critic_loss(values, rtgs)
            v_loss.backward()
            critic_optimizer.step()

        buffer.reset()

        # logging
        episodes_finished = torch.tensor([episodes_finished])
        dist.reduce(episodes_finished, dst=0, op=dist.ReduceOp.SUM)

        epoch_score_sum = torch.tensor([epoch_score_sum])
        dist.reduce(epoch_score_sum, dst=0, op=dist.ReduceOp.SUM)

        epoch_length_sum = torch.tensor([epoch_length_sum])
        dist.reduce(epoch_length_sum, dst=0, op=dist.ReduceOp.SUM)

        entropy_sum = entropies.sum()
        dist.reduce(entropy_sum, dst=0, op=dist.ReduceOp.SUM)

        average_score = epoch_score_sum / episodes_finished
        average_length = epoch_length_sum / episodes_finished
        average_entropy = entropy_sum / (config['steps_per_epoch_pre_worker'] * world_size)
        if rank == 0 and (logger is not None):
            logger.log_epoch_performance(epoch, {
                'episodes_finished': episodes_finished,
                'average_score': average_score,
                'average_length': average_length,
                'average_entropy': average_entropy
            })

        dist.barrier()

    dist.destroy_process_group()
    env.close()

def vpg_train(
    create_env: Callable[[], gym.Env], 
    create_actor: Callable[[], torch.nn.Module], 
    create_critic: Callable[[], torch.nn.Module], 
    config: VPGConfig, 
    create_logger: Callable[[dict], LoggerInterface] | None
):
    mp.set_start_method('spawn')
    world_size = 8 # mp.cpu_count()

    print(f"Training on {world_size} cpus")

    processes: list[mp.Process] = []
    for rank in range(world_size):
        if rank != 0:
            create_logger = None
        p = mp.Process(target=vpg_process, args=(
            rank, world_size, create_env, create_actor, create_critic, config, create_logger))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
