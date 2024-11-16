from typing import Any
import wandb
import torch

class WandbLogger:
    def __init__(self, project: str, config: dict[str, Any]):
        wandb.init(project=project, config=config)

    def __del__(self):
        wandb.finish()

    def log_epoch_performance(self, epoch: int, perf: dict[str, Any]) -> None:
        wandb.log(perf, step=epoch)

    def save_checkpoint(self, state_dict: dict[str, Any]) -> None:
        run_id = wandb.run.id
        model_path = f"models/model-{run_id}.pth"

        torch.save(state_dict, model_path)
        artifact = wandb.Artifact(f"model-{run_id}", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    def watch(self, module: torch.nn.Module, criterion: Any, log_freq: int) -> None:
        wandb.watch(module, criterion, log='all', log_freq=log_freq)

    def save_source(self, glob_str: str):
        wandb.save(glob_str)