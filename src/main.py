import gymnasium as gym
import yaml
from WandbLogger import WandbLogger
from MLP import MLP
from VanillaPolicyGradient import VanillaPolicyGradient
from torchsummary import summary
# export LD_LIBRARY_PATH=$HOME/dev/RL-from-scratch/venv/lib64/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

def main():
    with open("configs/CartPole-VPG.yaml", "r") as file:
        config = yaml.safe_load(file)

    logger = WandbLogger("RL-from-scratch", config)
    policy_network = MLP([4, 64, 64, 2]).to(config['device'])
    value_network = MLP([4, 64, 64, 1]).to(config['device'])
    env = gym.make_vec("CartPole-v1", num_envs=config['env_cnt'], vectorization_mode='async')

    print('------------------------------ Policy Network ------------------------------')
    summary(policy_network, input_size=(4,), device=config['device'])
    print('------------------------------ Value Network -------------------------------')
    summary(value_network, input_size=(4,), device=config['device'])

    logger.save_source('src/*.py') 
    vpg = VanillaPolicyGradient(env, policy_network, value_network, config, logger=logger)
    vpg.train()

    env.close()

if __name__ == '__main__':
    main()
