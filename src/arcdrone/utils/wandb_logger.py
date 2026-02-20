# wandb_logger.py
import wandb
import numpy as np
from brax.io import model
from dataclasses import dataclass
from typing import Optional, Dict, Any


class WandbLogger:
    def __init__(self, project_name, run_name, config):
        """Initialize the W&B logger."""
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config
        )
        self.config = config
        
    def log_metrics(self, num_steps, log_dict):
        """Log training metrics to W&B."""

        wandb.log(log_dict, step=num_steps)


    def save_training_data(self, hydra_run_dir):
        """Save model, metrics, configs and log as W&B artifact."""
        
        # Log as W&B artifact
        artifact = wandb.Artifact(
            name='arcdrone_training_data', 
            type='model', 
            description='Trained PPO model parameters from Brax'
        )
        artifact.add_dir(hydra_run_dir)
        wandb.log_artifact(artifact)
        print("Model artifact logged to W&B.")
        
        return 
    

    def finish(self):
        """Finish the W&B run."""
        wandb.finish()
        print("W&B run finished.")