"""Hydra entry-point for SITT training.

Trains a teacher policy via PPO on the privileged-state teacher env while
simultaneously distilling it into a student (vision) encoder via alignment
on a separate student env.

Example::

python src/arcdrone/vision_landing_sitt/train.py \
  train.num_timesteps=50000000 train.num_envs=1024 train.unroll_length=32 \
  train.batch_size=512 train.num_minibatches=16 train.num_updates_per_batch=4 \
  train.align_num_epochs=300 train.align_batch_size=256 train.align_num_minibatches=4 \
  train.align_updates_per_trigger=4 train.num_evals=30 train.num_eval_envs=128 \
  train.use_wandb=true env.buffer_size=3
"""

from datetime import datetime
from pathlib import Path
import functools
import os

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

# Custom resolver: ${mul:a,b} → int(a) * int(b)
OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
import numpy as np


@hydra.main(config_name="config", config_path="./cfg", version_base=None)
def main(cfg: DictConfig):

    # =========== Warp + JAX runtime setup ===========

    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MUJOCO_GL"] = "egl"

    # Import JAX-related modules AFTER setting env vars
    from arcdrone.vision_landing_sitt.training.train import train as sitt_train
    from arcdrone.vision_landing_sitt.training import networks as sitt_networks
    from arcdrone.priviledged_landing_rl.task.arcdrone import ARCDroneRL_Landing
    from arcdrone.vision_landing_il.task.arcdrone import ARCDroneRL_VisionLanding_StudentTeacher
    from brax.io import model
    from arcdrone.utils.wandb_logger import WandbLogger
    from mujoco_playground import wrapper
    import jax

    # =========== Teacher Environment (PPO) ===========

    env_cfg = cfg.env
    print("Instantiating teacher env (ARCDroneRL_Landing)...")
    teacher_env = ARCDroneRL_Landing(cfg=env_cfg)
    teacher_env = wrapper.wrap_for_brax_training(
        teacher_env,
        action_repeat=cfg.train.action_repeat,
        episode_length=cfg.train.episode_length,
    )
    print("Teacher env ready.")

    # =========== Student Environment (Alignment) ===========

    print("Instantiating student env (ARCDroneRL_VisionLanding_StudentTeacher)...")
    student_env = ARCDroneRL_VisionLanding_StudentTeacher(cfg=env_cfg)
    student_env = wrapper.wrap_for_brax_training(
        student_env,
        action_repeat=cfg.train.action_repeat,
        episode_length=cfg.train.episode_length,
    )
    print("Student env ready.")

    # =========== Logger ===========

    use_wandb = cfg.train.use_wandb
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project_name=cfg.train.wandb_project,
            run_name=cfg.train.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    cfg = cfg.train
    assert cfg.num_envs > cfg.num_eval_envs, "num_envs must be > num_eval_envs"

    # =========== Load teacher checkpoint (optional) ===========

    teacher_path = cfg.teacher_checkpoint_path
    teacher_params = None
    if teacher_path:
        print(f"Loading teacher checkpoint from: {teacher_path}")
        teacher_params = model.load_params(teacher_path)
        print("Teacher parameters loaded.")
    else:
        print("No teacher checkpoint provided — PPO will train teacher from scratch.")

    # =========== Network factory ===========

    network_factory = functools.partial(
        sitt_networks.make_sitt_networks,
        teacher_dec_hidden_layers=cfg.teacher_dec_hidden_layers,
        policy_dec_hidden_layers=cfg.policy_dec_hidden_layers,
        policy_propio_proj_hidden_layers=cfg.policy_propio_proj_hidden_layers,
        proxy_hidden_layers=cfg.proxy_hidden_layers,
        action_hidden_layer_sizes=cfg.action_hidden_layers,
        value_hidden_layer_sizes=cfg.value_hidden_layers,
        cnn_num_filters=cfg.cnn_num_filters,
        cnn_kernel_sizes=cfg.cnn_kernel_sizes,
        cnn_strides=cfg.cnn_strides,
        policy_obs_key=cfg.policy_obs_key,
        value_obs_key=cfg.value_obs_key,
        teacher_obs_key=cfg.teacher_obs_key,
        policy_pixels_key=cfg.policy_pixels_key,
        policy_pixels_key_1=cfg.policy_pixels_key_1,
        policy_propio_key=cfg.policy_propio_key,
    )

    # =========== Build train_fn ===========

    train_fn = functools.partial(
        sitt_train,
        teacher_env=teacher_env,
        student_env=student_env,
        teacher_params=teacher_params,
        # PPO schedule
        num_timesteps=cfg.num_timesteps,
        num_evals=cfg.num_evals,
        num_eval_envs=cfg.num_eval_envs,
        unroll_length=cfg.unroll_length,
        batch_size=cfg.batch_size,
        num_minibatches=cfg.num_minibatches,
        num_updates_per_batch=cfg.num_updates_per_batch,
        # PPO hyperparams
        learning_rate=cfg.learning_rate,
        entropy_cost=cfg.entropy_cost,
        discounting=cfg.discounting,
        reward_scaling=cfg.reward_scaling,
        normalize_observations=cfg.normalize_observations,
        # SITT alignment
        align_num_epochs=cfg.align_num_epochs,
        align_batch_size=cfg.align_batch_size,
        align_num_minibatches=cfg.align_num_minibatches,
        align_updates_per_trigger=cfg.align_updates_per_trigger,
        align_learning_rate=cfg.align_learning_rate,
        proxy_kl_coef=cfg.proxy_kl_coef,
        sitt_align_coef=cfg.sitt_align_coef,
        # Networks & env
        network_factory=network_factory,
        num_envs=cfg.num_envs,
        episode_length=cfg.episode_length,
        action_repeat=cfg.action_repeat,
        seed=cfg.seed,
        deterministic_eval=True,
        wrap_env=False,
    )

    # =========== Progress callback ===========

    def progress(num_steps, metrics):
        times.append(datetime.now())

        rl_step = metrics.get("rl_env_steps", num_steps)
        align_step = metrics.get("align_env_steps", 0)

        # ── PPO / eval metrics (logged at rl_step) ───────────────────
        rl_dict = {
            "training_rl/sps": metrics.get("training/sps", 0.0),
            "training_rl/walltime": metrics.get("training/walltime", 0.0),
            "training_rl/total_loss": metrics.get("training/total_loss", 0.0),
            "training_rl/policy_loss": metrics.get("training/policy_loss", 0.0),
            "training_rl/v_loss": metrics.get("training/v_loss", 0.0),
            "training_rl/entropy_loss": metrics.get("training/entropy_loss", 0.0),
            "training_rl/rl_align_loss": metrics.get("training/rl_align_loss", 0.0),
            "training_rl/reward_align": metrics.get("training/reward_align", 0.0),
            "training_rl/kl_mean": metrics.get("training/kl_mean", 0.0),
            "eval/episode_reward": metrics.get("eval/episode_reward", 0.0),
            "eval/avg_episode_length": metrics.get("eval/avg_episode_length", 0.0),
        }

        # ── Align metrics (logged at align_step) ─────────────────────
        align_dict = {}
        for key, value in metrics.items():
            if key.startswith("align/"):
                align_key = key[len("align/"):]
                align_dict[f"training_align/{align_key}"] = value

        # ── Eval ─────────────────────     
        # Teacher eval metrics
        for key, value in metrics.items():
            if key.startswith("eval/episode_reward_") and not key.endswith("_std"):
                reward_name = key[len("eval/episode_reward_"):]
                std_key = f"eval/episode_reward_{reward_name}_std"
                std_val = metrics.get(std_key, 0.0)
                rl_dict[f"rewards/{reward_name}"] = value
                rl_dict[f"std/{reward_name}_upper"] = value + std_val
                rl_dict[f"std/{reward_name}_lower"] = value - std_val
        # Student eval metrics (per-reward breakdown)
        for key, value in metrics.items():
            if key.startswith("student_eval/episode_reward_") and not key.endswith("_std"):
                reward_name = key[len("student_eval/episode_reward_"):]
                std_key = f"student_eval/episode_reward_{reward_name}_std"
                std_val = metrics.get(std_key, 0.0)
                rl_dict[f"student_rewards/{reward_name}"] = value
                rl_dict[f"student_std/{reward_name}_upper"] = value + std_val
                rl_dict[f"student_std/{reward_name}_lower"] = value - std_val
        # Student eval metrics
        for key, value in metrics.items():
            if key.startswith("student_eval/"):
                rl_dict[key] = value



        if use_wandb:
            logger.log_metrics(int(rl_step), {**rl_dict, **align_dict})

    # =========== Train ===========

    times = [datetime.now()]
    make_inference_fn, make_student_inference_fn, teacher_params, student_params, final_metrics = train_fn(progress_fn=progress)
    times.append(datetime.now())

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # =========== Save ===========

    hydra_run_dir = HydraConfig.get().runtime.output_dir
    print(f"Saving files to: {hydra_run_dir}")

    teacher_model_path = os.path.join(hydra_run_dir, "teacher_model.pkl")
    model.save_params(teacher_model_path, teacher_params)
    print(f"Teacher parameters saved to {teacher_model_path}")

    student_model_path = os.path.join(hydra_run_dir, "student_model.pkl")
    model.save_params(student_model_path, student_params)
    print(f"Student parameters saved to {student_model_path}")

    metrics_path = os.path.join(hydra_run_dir, "training_metrics.npz")
    np.savez(metrics_path, **{k: np.asarray(v) for k, v in final_metrics.items()})
    print(f"Training metrics saved to {metrics_path}")

    if use_wandb:
        logger.save_training_data(hydra_run_dir)
        logger.finish()

    print("SITT training completed successfully!")


if __name__ == "__main__":
    main()
