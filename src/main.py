import gymnasium as gym
import torch
import yaml
from WandbLogger import WandbLogger
from ThreeLayerPolicy import ThreeLayerPolicy
from VanillaPolicyGradient import VanillaPolicyGradient
from torchsummary import summary
# export LD_LIBRARY_PATH=$HOME/dev/RL-from-scratch/venv/lib64/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

def main():
    with open("configs/CartPole-VPG.yaml", "r") as file:
        config = yaml.safe_load(file)

    logger = WandbLogger("RL-from-scratch", config)
    policy_network = ThreeLayerPolicy(observation_dims=4, fc1_dims=64, fc2_dims=64, action_dims=2).to(config['device'])
    env = gym.make("CartPole-v1")

    summary(policy_network, input_size=(4,), device=config['device'])

    logger.save_source('src/*.py') 
    vpg = VanillaPolicyGradient(env, policy_network, config, logger=logger)
    vpg.train()

    env.close()

if __name__ == '__main__':
    main()
