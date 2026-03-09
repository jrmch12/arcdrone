"""Hydra entry-point for pure Imitation Learning training.

Loads a pre-trained teacher checkpoint, builds the SITT network
(with ``use_sitt=True``), and runs IL-only training (no PPO).

Example::

    arcdrone-train-il task=landing
"""

from datetime import datetime
from pathlib import Path
import functools
import os

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np


@hydra.main(config_name="config_il", config_path="../../cfg", version_base=None)
def main(cfg: DictConfig):

    cfg_train = cfg.train


    # =========== Warp + JAX runtime setup ===========

    # Enable Triton GEMM for better GPU utilisation (recommended for warp impl)
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    # Let warp manage its own GPU memory — do NOT pre-allocate JAX memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MUJOCO_GL"] = "egl"


    # Import JAX-related modules AFTER setting env vars
    from arcdrone.vision_landing_il.training.train import train as il_train
    from arcdrone.vision_landing_il.training import networks as il_networks
    from arcdrone.vision_landing_rl.task.arcdrone import ARCDroneRL_VisionLanding
    from brax.io import model
    from arcdrone.utils.wandb_logger import WandbLogger
    from mujoco_playground import wrapper
    import jax

    # =========== Environment ===========

    env = ARCDroneRL_VisionLanding(cfg=cfg.env)

    env = wrapper.wrap_for_brax_training(
        env,
        action_repeat=cfg.train.action_repeat,
        episode_length=cfg.train.episode_length,
    )
    print(f"env '{task_name}' instantiated successfully")

    # =========== Logger ===========

    use_wandb = cfg.train.use_wandb
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project_name=cfg.train.wandb_project,
            run_name=cfg.train.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # from now on, only train params

    # =========== Load teacher checkpoint ===========

    teacher_path = cfg_train.teacher_checkpoint_path
    if not teacher_path:
        raise ValueError(
            "IL training requires a teacher checkpoint. "
            "Set train.teacher_checkpoint_path in config or CLI."
        )
    print(f"Loading teacher checkpoint from: {teacher_path}")
    teacher_params = model.load_params(teacher_path)
    print("Teacher parameters loaded successfully!")

    # =========== Network factory ===========

    network_factory = functools.partial(
        il_networks.make_il_networks,
        teacher_dec_hidden_layers=cfg_train.teacher_dec_hidden_layers,
        policy_dec_hidden_layers=cfg_train.policy_dec_hidden_layers,
        policy_propio_proj_hidden_layers=cfg_train.policy_propio_proj_hidden_layers,
        action_hidden_layer_sizes=cfg_train.action_hidden_layers,
        value_hidden_layer_sizes=cfg_train.value_hidden_layers,
        cnn_num_filters=cfg_train.cnn_num_filters,
        cnn_kernel_sizes=cfg_train.cnn_kernel_sizes,
        cnn_strides=cfg_train.cnn_strides,
        policy_obs_key=cfg_train.policy_obs_key,
        policy_pixels_key=cfg_train.policy_pixels_key,
        policy_propio_key=cfg_train.policy_propio_key,
        teacher_obs_key=cfg_train.value_obs_key,
    )

    # =========== Build train_fn ===========

    train_fn = functools.partial(
        il_train,
        env=env,
        teacher_params=teacher_params,
        num_il_epochs=cfg_train.num_il_epochs,
        num_evals=cfg_train.num_evals,
        unroll_length=cfg_train.unroll_length,
        num_unrolls_per_epoch=cfg_train.num_unrolls_per_epoch,
        align_updates_per_trigger=cfg_train.align_updates_per_trigger,
        network_factory=network_factory,
        num_envs=cfg_train.num_envs,
        episode_length=cfg_train.episode_length,
        action_repeat=cfg_train.action_repeat,
        normalize_observations=cfg_train.normalize_observations,
        learning_rate=cfg_train.learning_rate,
        seed=cfg_train.seed,
        deterministic_eval=True,
        wrap_env=False,  # mujoco_playground wrapper already applied above
    )

    # =========== Progress callback ===========

    def progress(num_steps, metrics):
        times.append(datetime.now())

        log_dict = {
            "training/align_loss": metrics.get("training/align_loss", 0.0),
            "training/sps": metrics.get("training/sps", 0.0),
            "training/walltime": metrics.get("training/walltime", 0.0),
            "eval/episode_reward": metrics.get("eval/episode_reward", 0.0),
            "eval/avg_episode_length": metrics.get("eval/avg_episode_length", 0.0),
        }

        # Forward any eval/episode_reward_* metrics
        for key, value in metrics.items():
            if key.startswith("eval/episode_reward_") and not key.endswith("_std"):
                reward_name = key[len("eval/episode_reward_"):]
                std_key = f"eval/episode_reward_{reward_name}_std"
                std_val = metrics.get(std_key, 0.0)
                log_dict[f"rewards/{reward_name}"] = value
                log_dict[f"std/{reward_name}_upper"] = value + std_val
                log_dict[f"std/{reward_name}_lower"] = value - std_val

        if use_wandb:
            logger.log_metrics(num_steps, log_dict)

    # =========== Train ===========

    times = [datetime.now()]
    make_inference_fn, params, metrics = train_fn(progress_fn=progress)
    times.append(datetime.now())

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # =========== Save ===========

    hydra_run_dir = HydraConfig.get().runtime.output_dir
    print(f"Saving files to: {hydra_run_dir}")

    model_path = os.path.join(hydra_run_dir, "trained_model.pkl")
    model.save_params(model_path, params)
    print(f"Model parameters saved to {model_path}")

    metrics_path = os.path.join(hydra_run_dir, "training_metrics.npz")
    np.savez(metrics_path, **{k: np.asarray(v) for k, v in metrics.items()})
    print(f"Training metrics saved to {metrics_path}")

    if use_wandb:
        logger.save_training_data(hydra_run_dir)
        logger.finish()

    print("IL training completed successfully!")


if __name__ == "__main__":
    main()
