import torch
from utils import discounted_cumsum
from torch.testing import assert_close

def test_discounted_cumsum():
    x = torch.tensor([4.75, 2.41, 1.65, -3.26, -0.43, -4.57, 4.57, -0.40, -1.95, 1.00])
    reset = torch.tensor([False, False, True, False, False, False, True, False, False, True])
    gamma = 0.95

    expected_result = torch.tensor([8.528625, 3.9775, 1.65, -3.874721250, -0.647075, -0.2285, 4.57, -1.35, -1, 1])
    result = discounted_cumsum(x, reset, gamma)

    assert_close(expected_result, result)