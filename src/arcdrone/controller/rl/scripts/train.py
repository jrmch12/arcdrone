#  # While optimizing for GPU/TPU usage
#  import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from datetime import datetime
from pathlib import Path
import functools
import os
import glob

# These imports need to be available before the main function
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import numpy as np

from mujoco_playground import wrapper


@hydra.main(config_name="config", config_path="../../cfg", version_base=None)
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
    # vision networks (optional)
    try:
        from brax.training.agents.ppo import networks_vision as ppo_networks_vision
    except Exception:
        ppo_networks_vision = None
    from brax.io import model
    from arcdrone.utils.wandb_logger import WandbLogger
    from arcdrone import ARCDroneRL_Landing, ARCDroneRL_Hover

    # Map task names to environment classes
    ENV_CLASSES = {
        'hover': ARCDroneRL_Hover,
        'landing': ARCDroneRL_Landing,
    }

    task_name = cfg.task_name
    env_cfg = cfg.env
    print(f"Instantiating environment for task: '{task_name}'")
    if task_name not in ENV_CLASSES:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(ENV_CLASSES.keys())}")
    env_class = ENV_CLASSES[task_name]
    env = env_class(cfg=cfg.env)

    # Wrap environment for Brax training (vectorization + vision support)
    # NOTE: We use mujoco_playground's wrapper AND set wrap_env=False in ppo.train
    # to avoid double-wrapping (which causes "invalid PRNG key data: ndim=0" error)
    env = wrapper.wrap_for_brax_training(
        env,
        vision=env_cfg.get('vision', False),
        num_vision_envs=cfg.train.num_envs,
        action_repeat=cfg.train.action_repeat,
        episode_length=cfg.train.episode_length,
    )

    print(f"env '{task_name}' instantiated successfully")

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

    # Select network factory (vision vs non-vision)
    if env_cfg.get('vision', False) and ppo_networks_vision is not None:
        network_factory = functools.partial(
            ppo_networks_vision.make_ppo_networks_vision,
            policy_hidden_layer_sizes=cfg.policy_hidden_layers,
            value_hidden_layer_sizes=cfg.value_hidden_layers,
            policy_obs_key=cfg.policy_obs_key,
            value_obs_key=cfg.value_obs_key,
        )
    else:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=cfg.policy_hidden_layers,
            value_hidden_layer_sizes=cfg.value_hidden_layers,
            policy_obs_key=cfg.policy_obs_key,
            value_obs_key=cfg.value_obs_key,
        )

    # =========== Handle auto-restore from previous checkpoint ===========
    from_prev = int(getattr(cfg, 'frompreviouscheckpoint', 0))
    restore_path = getattr(cfg, 'restore_params_path', None)
    # Only override if frompreviouscheckpoint > 0 and restore_params_path is not set
    if from_prev and (not restore_path or restore_path == ""):
        outputs_path = Path("outputs")
        # Find all trained_model.pkl files recursively
        pkl_files = list(outputs_path.rglob("trained_model.pkl"))
        if pkl_files:
            # Sort by modification time, most recent last
            pkl_files = sorted(pkl_files, key=lambda p: p.stat().st_mtime)
            if len(pkl_files) >= from_prev:
                chosen_pkl = pkl_files[-from_prev]
                cfg.restore_params_path = str(chosen_pkl)
                cfg.restore_value_fn = True
                print(f"[Auto-restore] Using checkpoint: {chosen_pkl}")
            else:
                print(f"[Auto-restore] Not enough checkpoints found for frompreviouscheckpoint={from_prev}")
        else:
            print(f"[Auto-restore] No trained_model.pkl files found in outputs/")

    restore_params = None
    restore_path = getattr(cfg, 'restore_params_path', None)
    if restore_path:
        print(f"Loading parameters from: {restore_path}")
        restore_params = model.load_params(restore_path)
        print("Parameters loaded successfully!")



    train_fn = functools.partial(
        ppo.train, num_timesteps=cfg.num_timesteps, num_evals=cfg.num_evals, reward_scaling=cfg.reward_scaling,
        episode_length=cfg.episode_length, normalize_observations=cfg.normalize_observations, action_repeat=cfg.action_repeat,
        unroll_length=cfg.unroll_length, num_minibatches=cfg.num_minibatches, num_updates_per_batch=cfg.num_updates_per_batch,
        discounting=cfg.discounting, learning_rate=cfg.learning_rate, entropy_cost=cfg.entropy_cost, num_envs=cfg.num_envs,
        batch_size=cfg.batch_size, seed=cfg.seed, log_training_metrics=cfg.log_training_metrics,
        restore_params=restore_params, restore_value_fn=cfg.restore_value_fn, network_factory=network_factory,
        wrap_env=False,  # IMPORTANT: mujoco_playground's wrapper already wrapped the env
    )

    # =========== Define custom progress function ===========

    def progress(num_steps, metrics):

        times.append(datetime.now())


        # Automate logging for all eval/episode_reward_* metrics as mean, upper, lower (no std)
        eval_reward_metrics = {}
        std_reward_metrics = {}
        for key, value in metrics.items():
            if key.startswith('eval/episode_reward') and not key.endswith('_std'):
                reward_name = key[len('eval/episode_reward_'):]
                std_key = f"eval/episode_reward_{reward_name}_std"
                std_val = metrics.get(std_key, 0.0)
                eval_reward_metrics[f"rewards/{reward_name}"] = value
                std_reward_metrics[f"std/{reward_name}_upper"] = value + std_val
                std_reward_metrics[f"std/{reward_name}_lower"] = value - std_val

        avg_len = metrics.get('eval/avg_episode_length', 0.0)
        std_len = metrics.get('eval/std_episode_length', 0.0)
        log_dict = {
            # 'timesteps': num_steps,
            'eval/episode_length': avg_len,
            'std/episode_length_upper': avg_len + std_len,
            'std/episode_length_lower': avg_len - std_len,
            'eval/episode_reward': metrics.get('eval/episode_reward', 0.0),
            'eval/epoch_eval_time': metrics.get('eval/epoch_eval_time', 0.0),
            'eval/sps': metrics.get('eval/sps', 0.0),
            'eval/walltime': metrics.get('eval/walltime', 0.0),
            'training/learning_rate': metrics.get('training/learning_rate', 0.0),
            'training/total_loss': metrics.get('training/total_loss', 0.0),
            'training/policy_loss': metrics.get('training/policy_loss', 0.0),
            'training/v_loss': metrics.get('training/v_loss', 0.0),
            'training/entropy_loss': metrics.get('training/entropy_loss', 0.0),
            'training/kl_mean': metrics.get('training/kl_mean', 0.0),
            'training/policy_dist_mean_std': metrics.get('training/policy_dist_mean_std', 0.0),
            'training/sps': metrics.get('training/sps', 0.0),
            'training/walltime': metrics.get('training/walltime', 0.0),
        }
        log_dict.update(eval_reward_metrics)
        log_dict.update(std_reward_metrics)

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
    