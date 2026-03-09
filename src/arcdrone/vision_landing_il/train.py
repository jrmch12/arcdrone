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


@hydra.main(config_name="config", config_path="./cfg", version_base=None)
def main(cfg: DictConfig):

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
    from arcdrone.vision_landing_il.task.arcdrone import ARCDroneRL_VisionLanding_StudentTeacher
    from brax.io import model
    from arcdrone.utils.wandb_logger import WandbLogger
    from mujoco_playground import wrapper
    import jax

    # =========== Environment ===========

    env_cfg = cfg.env
    print("Instantiating ARCDroneRL_VisionLanding_StudentTeacher...")
    env = ARCDroneRL_VisionLanding_StudentTeacher(cfg=env_cfg)

    print("Environment instantiated successfully")

    # =========== Wrap for Brax training ===========

    env = wrapper.wrap_for_brax_training(
        env,
        action_repeat=cfg.train.action_repeat,
        episode_length=cfg.train.episode_length,
    )

    print("Environment wrapped successfully")

    # =========== Logger ===========

    use_wandb = cfg.train.use_wandb
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project_name=cfg.train.wandb_project,
            run_name=cfg.train.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    cfg = cfg.train     # from now on, only train parameters should be use
    assert cfg.num_envs > cfg.num_eval_envs, "num_envs must be greater than num_eval_envs"


    # =========== Load teacher checkpoint ===========

    teacher_path = cfg.teacher_checkpoint_path
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
        teacher_dec_hidden_layers=cfg.teacher_dec_hidden_layers,
        policy_dec_hidden_layers=cfg.policy_dec_hidden_layers,
        policy_propio_proj_hidden_layers=cfg.policy_propio_proj_hidden_layers,
        action_hidden_layer_sizes=cfg.action_hidden_layers,
        value_hidden_layer_sizes=cfg.value_hidden_layers,
        cnn_num_filters=cfg.cnn_num_filters,
        cnn_kernel_sizes=cfg.cnn_kernel_sizes,
        cnn_strides=cfg.cnn_strides,
        policy_pixels_key=cfg.policy_pixels_key,
        policy_propio_key=cfg.policy_propio_key,
        teacher_obs_key=cfg.teacher_obs_key,
        value_obs_key=cfg.value_obs_key,
    )

    # =========== Build train_fn ===========

    train_fn = functools.partial(
        il_train,
        env=env,
        teacher_params=teacher_params,
        num_il_epochs=cfg.num_il_epochs,
        num_evals=cfg.num_evals,
        unroll_length=cfg.unroll_length,
        num_unrolls_per_epoch=cfg.num_unrolls_per_epoch,
        align_updates_per_trigger=cfg.align_updates_per_trigger,
        network_factory=network_factory,
        num_envs=cfg.num_envs,
        episode_length=cfg.episode_length,
        action_repeat=cfg.action_repeat,
        normalize_observations=cfg.normalize_observations,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
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
