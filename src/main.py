import gymnasium as gym
from Agent import Agent, criterion
import torch
import wandb
import yaml
from tqdm import tqdm
import numpy as np
# export LD_LIBRARY_PATH=$HOME/dev/RL-from-scratch/venv/lib64/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    with open("configs/LunarLander-VPG.yaml", "r") as file:
        config = yaml.safe_load(file)

    with wandb.init(project="RL-from-scratch", config=config):

        agent = Agent(device, alpha=config['lr'], gamma=config['gamma'],
                      observation_dims=8, fc1_dims=256, fc2_dims=256, action_dims=4)
        
        wandb.watch(agent.policy, criterion, log='all', log_freq=10)
        env = gym.make("LunarLander-v3")

        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

        episode_cnt = config['episode_cnt']
        for i in tqdm(range(episode_cnt)):
            observation, _ = env.reset()
            done = False
            score = 0
            episode_length = 0
            episode_entropy = 0

            while not done:
                action, action_log_prob, entropy = agent.choose_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                agent.store_transition(action_log_prob, reward)
                done = terminated or truncated
                score += reward
                episode_length += 1
                episode_entropy += entropy


            episode_entropy /= episode_length
            agent.learn()
            wandb.log({
                'episode_len': episode_length,
                'score': score,
                'entropy': episode_entropy
            }, step=i)

        torch.save(agent.policy.state_dict(), 'model.pt')
        env.close()
    wandb.finish()


if __name__ == '__main__':
    main()
