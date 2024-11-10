# test_math_operations.py
import unittest
from src.VanillaPolicyGradient import calculate_advantages
import numpy as np

class TestVPG(unittest.TestCase):

    def test_add_positive_numbers(self):
        batch_size = 4
        max_episode_len = 20

        rewards = np.random.randn(batch_size, max_episode_len)
        values = np.random.randn(batch_size, max_episode_len)
        done = np.zeros((batch_size, max_episode_len), dtype=np.bool)

        for i in range(batch_size):
            pos = np.random.randint(0, max_episode_len)
            for j in range(pos, max_episode_len):
                done[i, j] = True

        print(rewards)
        print(values)
        print(done)

        gamma = 0.9
        lam = 0.9

        for t in reversed


if __name__ == '__main__':
    unittest.main()
