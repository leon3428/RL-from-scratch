from typing import Protocol, Any
import torch

class LoggerInterface(Protocol):
    def log_episode_performance(self, episode: int, perf: dict[str, Any]) -> None:
        ...

    def save_checkpoint(self, state_dict: dict[str, Any]) -> None:
        ...

    def watch(self, module: torch.nn.Module, criterion: Any, log_freq: int) -> None:
        ...