"""W&B logger — local copy for New_attempt_2 (self-contained)."""

import wandb


class WandbLogger:
    def __init__(self, project_name, run_name, config):
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
        )
        self.config = config

    def log_metrics(self, num_steps, log_dict):
        wandb.log(log_dict, step=num_steps)

    def save_training_data(self, hydra_run_dir):
        artifact = wandb.Artifact(
            name="arcdrone_training_data",
            type="model",
            description="Trained DAgger model parameters",
        )
        artifact.add_dir(hydra_run_dir)
        wandb.log_artifact(artifact)
        print("Model artifact logged to W&B.")

    def finish(self):
        wandb.finish()
        print("W&B run finished.")
