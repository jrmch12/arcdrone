#  # While optimizing for GPU/TPU usage
#  import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
import functools
import os
import glob

from omegaconf import DictConfig, OmegaConf
# Custom resolver: ${mul:a,b} → int(a) * int(b)
OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np

from mujoco_playground import wrapper


def _remove_pixels(obs: Any) -> Any:
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith("pixels/")}


@hydra.main(config_name="config", config_path="./cfg", version_base=None)
def main(cfg: DictConfig):

    # =========== Warp + JAX runtime setup ===========
    # Must happen BEFORE JAX is imported so XLA flags take effect.
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MUJOCO_GL"] = "egl"

    # Import JAX-related modules after setting environment variables
    # from brax.training.agents.ppo import train as ppo  # original brax PPO
    from arcdrone.vision_landing_rl.training import train as ppo  # arcdrone fork with CNN feature caching
    from brax.io import model
    from brax.training.acme import running_statistics
    from arcdrone.utils.wandb_logger import WandbLogger
    from arcdrone.vision_landing_il.task.arcdrone import ARCDroneRL_VisionLanding_StudentTeacher
    from arcdrone.vision_landing_rl.training.networks import make_ppo_networks_vision
    import jax

    # =========== Environment ===========

    env_cfg = cfg.env
    print("Instantiating ARCDroneRL_VisionLanding...")
    env = ARCDroneRL_VisionLanding_StudentTeacher(cfg=env_cfg)

    print("Environment instantiated successfully")

    # =========== Wrap for Brax training ===========

    env = wrapper.wrap_for_brax_training(
        env,
        action_repeat=cfg.train.action_repeat,
        episode_length=cfg.train.episode_length,
    )

    print("Environment wrapped successfully")

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
    assert cfg.num_envs > cfg.num_eval_envs, "num_envs must be greater than num_eval_envs"

    # =========== Network factory ===========

    network_factory = functools.partial(
        make_ppo_networks_vision,
        policy_dec_hidden_layers=cfg.policy_dec_hidden_layers,
        policy_proprio_proj_hidden_layers=cfg.policy_proprio_proj_hidden_layers,
        action_hidden_layer_sizes=cfg.action_hidden_layers,
        value_hidden_layer_sizes=cfg.value_hidden_layers,
        cnn_num_filters=cfg.cnn_num_filters,
        cnn_kernel_sizes=cfg.cnn_kernel_sizes,
        cnn_strides=cfg.cnn_strides,
        policy_pixels_key=cfg.policy_pixels_key,
        policy_pixels_key_1=cfg.policy_pixels_key_1,
        policy_pixels_key_2=cfg.policy_pixels_key_2,
        policy_proprio_key=cfg.policy_proprio_key,
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
        # Always rebuild normalizer stats from the current env to avoid shape mismatches.
        # Build obs shapes from config to avoid observation_size (which eval_shapes reset).
        cam_res = None
        buffer_size = None
        try:
            cam_res = tuple(env_cfg.vision_config.cam_res)
            buffer_size = int(env_cfg.buffer_size)
        except Exception:
            cam_res = None

        if cam_res is None or buffer_size is None:
            sample_state = env.reset(jax.random.PRNGKey(cfg.seed))
            obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], sample_state.obs)
        else:
            height, width = cam_res
            action_size = int(env.action_size)
            pixel_channels = buffer_size * 3
            value_dim = buffer_size * (19 + action_size)
            proprio_dim = buffer_size * (10 + action_size)
            obs_shape = {
                "pixels/view_0": (height, width, pixel_channels),
                "pixels/view_1": (height, width, pixel_channels),
                "pixels/view_2": (height, width, pixel_channels),
                "proprio_obs": (proprio_dim,),
                "value_obs": (value_dim,),
                "teacher_obs": (value_dim,),
            }

        if isinstance(obs_shape, Mapping):
            obs_shape = {
                key: ((shape,) if isinstance(shape, int) else tuple(shape))
                for key, shape in obs_shape.items()
            }
        else:
            obs_shape = (obs_shape,) if isinstance(obs_shape, int) else tuple(obs_shape)

        normalize = (
            running_statistics.normalize
            if cfg.normalize_observations
            else (lambda x, y: x)
        )
        ppo_network = network_factory(
            obs_shape, env.action_size, preprocess_observations_fn=normalize
        )
        key_init = jax.random.PRNGKey(cfg.seed)
        _, k2 = jax.random.split(key_init)
        init_value_params = ppo_network.value_network.init(k2)

        is_student_ckpt = (
            isinstance(restore_params, tuple)
            and len(restore_params) == 2
            and isinstance(restore_params[1], tuple)
            and len(restore_params[1]) == 2
        )

        # NOTE: fresh-normalizer fallback is intentionally disabled for now.
        # Restore is strict and must come from PPO-shaped checkpoints.
        if is_student_ckpt:
            raise ValueError(
                "Student checkpoint detected. For vision RL warmstart, use a PPO-shaped checkpoint "
                "(normalizer, policy, value), e.g. visionrl_warmstart.pkl from converter script."
            )
        elif isinstance(restore_params, tuple) and len(restore_params) >= 3:
            policy_params = restore_params[1]
            value_params = (
                restore_params[2] if cfg.restore_value_fn else init_value_params
            )
            selected_normalizer = restore_params[0]
            print(f"[Restore] Policy params <- {restore_path}[1]")
            if cfg.restore_value_fn:
                print(f"[Restore] Value params <- {restore_path}[2]")
            else:
                print("[Restore] Value params <- fresh init (restore_value_fn=false)")
            print(f"[Restore] Normalizer <- {restore_path}[0]")
        else:
            raise ValueError(
                "Unsupported checkpoint format for strict restore. Expected PPO tuple "
                "(normalizer, policy, value)."
            )

        restore_params = (selected_normalizer, policy_params, value_params)



    train_fn = functools.partial(
        ppo.train, num_timesteps=cfg.num_timesteps, num_evals=cfg.num_evals, reward_scaling=cfg.reward_scaling,
        episode_length=cfg.episode_length, normalize_observations=cfg.normalize_observations, action_repeat=cfg.action_repeat,
        unroll_length=cfg.unroll_length, num_minibatches=cfg.num_minibatches, num_updates_per_batch=cfg.num_updates_per_batch,
        discounting=cfg.discounting, learning_rate=cfg.learning_rate, entropy_cost=cfg.entropy_cost, num_envs=cfg.num_envs,
        learning_rate_schedule=cfg.learning_rate_schedule, desired_kl=cfg.desired_kl, max_grad_norm=cfg.max_grad_norm, clipping_epsilon=cfg.clipping_epsilon,
        batch_size=cfg.batch_size, seed=cfg.seed, log_training_metrics=cfg.log_training_metrics,
        restore_params=restore_params, restore_value_fn=cfg.restore_value_fn, network_factory=network_factory, num_eval_envs=cfg.num_eval_envs,
        frozen_cnn=cfg.frozen_cnn,
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


# How to use?
#
# python src/arcdrone/vision_landing_rl/train.py train.num_envs=128 train.unroll_length=4 train.batch_size=64 train.num_minibatches=2 train.num_updates_per_batch=16 train.num_timesteps=100000 train.use_wandb=true train.num_evals=32 train.num_eval_envs=64 train.restore_params_path='outputs/2026-03-21/10-18-23/trained_model.pkl' 
