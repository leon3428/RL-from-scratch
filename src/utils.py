import torch

def discounted_cumsum(x: torch.Tensor, reset: torch.Tensor, gamma: float) -> torch.Tensor:
    ret = x
    for t in reversed(range(len(x)-1)):
        ret[t] += (~reset[t]) * gamma * ret[t+1]

    return ret