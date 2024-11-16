import gymnasium as gym
import yaml
from WandbLogger import WandbLogger
from MLP import MLP
from VanillaPolicyGradient import vpg_train
# export LD_LIBRARY_PATH=$HOME/dev/RL-from-scratch/venv/lib64/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

def create_env(): 
    return gym.make("CartPole-v1")

def create_actor():
    return MLP([4, 64, 64, 2])

def create_critic(): 
    return MLP([4, 64, 64, 1])

def create_logger(config: dict):
    return WandbLogger("RL-from-scratch", config)

def main():
    with open("configs/CartPole-VPG.yaml", "r") as file:
        config = yaml.safe_load(file)

    vpg_train(create_env, create_actor, create_critic, config, create_logger)

if __name__ == '__main__':
    main()
