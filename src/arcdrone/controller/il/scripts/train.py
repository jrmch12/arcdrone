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
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import numpy as np


@hydra.main(config_name="config_il", config_path="../../cfg", version_base=None)
def main(cfg: DictConfig):

    cfg_train = cfg.train

    # =========== Handle CPU debugging mode ===========
    if cfg.get("debug_cpu", False):
        print("DEBUG: Running in CPU mode")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # =========== Madrona + JAX memory/runtime setup ===========
    
    # On your second reading, load the compiled rendering backend to save time!
    # os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "<YOUR_PATH>/madrona_mjx/build/cache"
    os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "/home/jrmch12f/Documents/code/borrador_braxenvs/madrona_cache"
    # Coordinate between Jax and the Madrona rendering backend
    def limit_jax_mem(limit):
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"
    limit_jax_mem(0.6)
    # Reduce madrona memory allocation to 1GB as cartpole doesn't need much
    os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"] = "1073741824"


    # Import JAX-related modules AFTER setting env vars
    from arcdrone.controller.il.il.train import train as il_train
    from arcdrone.controller.sitt.sitt import networks as sitt_networks
    from arcdrone.controller.sitt.env.student_wrapper import StudentWrapper
    from arcdrone.controller.rl.task.landing_mode.arcdrone import ARCDroneRL_Landing
    from arcdrone.controller.rl.task.vision_mode.arcdrone import ARCDroneRL_VisionLanding
    from brax.io import model
    from arcdrone.utils.wandb_logger import WandbLogger
    from arcdrone import ARCDroneRL_Landing, ARCDroneRL_Hover
    from mujoco_playground import wrapper
    import jax

    # =========== Environment ===========

    ENV_CLASSES = {
        "hover": ARCDroneRL_Hover,
        "landing": ARCDroneRL_Landing,
    }

    task_name = cfg.task_name
    env_cfg = cfg.env
    print(f"Instantiating environment for task: '{task_name}'")
    if task_name not in ENV_CLASSES:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(ENV_CLASSES.keys())}"
        )
    env = ENV_CLASSES[task_name](cfg=cfg.env)

    # Instantiate the student (vision) env — same xml/cfg, adds BatchRenderer
    student_env = ARCDroneRL_VisionLanding(cfg=cfg.env)

    # Compute student observation shapes statically from config.
    # A dummy reset is NOT used here: calling student_env.reset() before the
    # env is wrapped by MadronaWrapper would trigger Madrona CUDA ops outside
    # the proper jax.vmap context, corrupting the CUDA state.
    vision_cfg = cfg.env.vision_config
    history = int(cfg.env.buffer_size)
    render_h = int(vision_cfg.render_height)
    render_w = int(vision_cfg.render_width)
    nu = env.action_size
    student_observation_size = {
        "pixels/view_0": (render_h, render_w, history),
        "state": (history * nu,),
    }

    # Combine teacher (state) + student (vision) under StudentWrapper.
    # Must wrap with vision=True so MadronaWrapper handles the BatchRenderer
    # batching via jax.vmap (VmapWrapper can't trace the Madrona CUDA ops).
    env = StudentWrapper(teacher_env=env, student_env=student_env)
    env = wrapper.wrap_for_brax_training(
        env,
        vision=True,
        num_vision_envs=cfg.train.num_envs,
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
        sitt_networks.make_sitt_networks,
        policy_hidden_layer_sizes=cfg_train.policy_hidden_layers,
        action_hidden_layer_sizes=cfg_train.action_hidden_layers,
        value_hidden_layer_sizes=cfg_train.value_hidden_layers,
        policy_obs_key=cfg_train.policy_obs_key,
        value_obs_key=cfg_train.value_obs_key,
        # IL always needs SITT decoders
        use_sitt=True,
        student_hidden_layer_sizes=cfg_train.student_hidden_layers,
        proxy_hidden_layer_sizes=cfg_train.proxy_hidden_layers,
        student_observation_size=student_observation_size,
        student_obs_key="state",
    )

    # =========== Build train_fn ===========

    train_fn = functools.partial(
        il_train,
        student_env=student_env,
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
        wrap_env=False,  # mujoco_playground's wrapper already handled above
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
    make_inference_fn, params, metrics = train_fn(
        teacher_env=env, progress_fn=progress
    )
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
