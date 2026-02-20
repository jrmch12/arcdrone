#  # While optimizing for GPU/TPU usage
#  import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from datetime import datetime
import functools
import os

# These imports need to be available before the main function
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import numpy as np


@hydra.main(config_name="config", config_path="../cfg", version_base=None)
def main(cfg: DictConfig):

    # =========== Handle CPU debugging mode ===========
    if cfg.get('debug_cpu', False):
        print("DEBUG: Running in CPU mode")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"  
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # Import JAX-related modules after setting environment variables
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.io import model
    from arcdrone.utils.wandb_logger import WandbLogger
    from arcdrone import ARCDroneRL_Vel, ARCDroneRL_Landing


    # =========== Load environment ===========

    env = ARCDroneRL_Landing(cfg=cfg.env)
    print("env instantiated successfully")

    # =========== Load config and Logger ===========

    use_wandb = cfg.train.use_wandb
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project_name=cfg.train.wandb_project,
            run_name=cfg.train.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    cfg = cfg.train     # from now on, only train parameters should be use


    # =========== Load main training function ===========

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=cfg.policy_hidden_layers,
        value_hidden_layer_sizes=cfg.value_hidden_layers,
        policy_obs_key=cfg.policy_obs_key,
        value_obs_key=cfg.value_obs_key,
        )

    # Handle model restoration
    restore_params = None
    if cfg.restore_params_path is not None:
        print(f"Loading parameters from: {cfg.restore_params_path}")
        restore_params = model.load_params(cfg.restore_params_path)
        print("Parameters loaded successfully!")

    train_fn = functools.partial(
        ppo.train, num_timesteps=cfg.num_timesteps, num_evals=cfg.num_evals, reward_scaling=cfg.reward_scaling,
        episode_length=cfg.episode_length, normalize_observations=cfg.normalize_observations, action_repeat=cfg.action_repeat,
        unroll_length=cfg.unroll_length, num_minibatches=cfg.num_minibatches, num_updates_per_batch=cfg.num_updates_per_batch,
        discounting=cfg.discounting, learning_rate=cfg.learning_rate, entropy_cost=cfg.entropy_cost, num_envs=cfg.num_envs,
        batch_size=cfg.batch_size, seed=cfg.seed, log_training_metrics=cfg.log_training_metrics,
        restore_params=restore_params, restore_value_fn=cfg.restore_value_fn, network_factory=network_factory,)

    # =========== Define custom progress function ===========

    def progress(num_steps, metrics):

        times.append(datetime.now())

        log_dict = {
            'eval/episode_reward': metrics.get('eval/episode_reward', 0.0),
            'eval/episode_reward_std': metrics.get('eval/episode_reward_std', 0.0),
            'timesteps': num_steps,
            'eval/episode_length': metrics.get('eval/avg_episode_length', 0.0),
            'training/entropy': metrics.get('training/entropy', 0.0),
            'training/learning_rate': metrics.get('training/learning_rate', 0.0),
            'training/policy_loss': metrics.get('training/policy_loss', 0.0),
            'training/value_loss': metrics.get('training/value_loss', 0.0),
            'training/approx_kl': metrics.get('training/approx_kl', 0.0),
        }

        if use_wandb:
            logger.log_metrics(num_steps, log_dict)


    # ========== Training Loop ===========

    times = [datetime.now()]  # Initialize times list
    make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress) # this loop call our custom progress function
    times.append(datetime.now())  # Add end time


    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

    # ========== Save ============

    hydra_run_dir = HydraConfig.get().runtime.output_dir
    print(f"Saving files to: {hydra_run_dir}")
    
    # save model locally
    model_path = os.path.join(hydra_run_dir, "trained_model.pkl")
    model.save_params(model_path, params)
    print(f"Model parameters saved to {model_path}")

    # save training logs (metrics) locally
    metrics_path = os.path.join(hydra_run_dir, "training_metrics.npz")
    np.savez(metrics_path, **metrics)
    print(f"Training metrics saved to {metrics_path}")


    # save in wandb (only if enabled)
    if use_wandb:
        logger.save_training_data(hydra_run_dir)
        logger.finish()
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()