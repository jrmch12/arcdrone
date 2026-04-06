"""Hydra entry-point for DAgger (Dataset Aggregation) training.

Loads a pre-trained teacher (privileged-state RL) checkpoint and trains a
student vision encoder using the DAgger algorithm: rollout with a β-mixture
of teacher and student, then align student features to teacher features on
the visited states.

Example::

    python src/arcdrone/vision_landing_dagger/train.py \
        train.teacher_checkpoint_path='checkpoints/teacher/trained_model.pkl' \
        train.num_envs=256 train.num_dagger_epochs=200 train.use_wandb=true
"""

from datetime import datetime
from pathlib import Path
import functools
import os

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)


@hydra.main(config_name="config", config_path="./cfg", version_base=None)
def main(cfg: DictConfig):

    # =========== Warp + JAX runtime setup ===========

    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MUJOCO_GL"] = "egl"

    # Import JAX-related modules AFTER setting env vars
    from arcdrone.vision_landing_dagger.training.train import train as dagger_train
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

    cfg = cfg.train
    assert cfg.num_envs > cfg.num_eval_envs, "num_envs must be greater than num_eval_envs"

    # =========== Load teacher checkpoint ===========

    teacher_path = cfg.teacher_checkpoint_path
    if not teacher_path:
        raise ValueError(
            "DAgger training requires a teacher checkpoint. "
            "Set train.teacher_checkpoint_path in config or CLI."
        )
    print(f"Loading teacher checkpoint from: {teacher_path}")
    teacher_params = model.load_params(teacher_path)
    print("Teacher parameters loaded successfully!")

    # =========== Handle auto-restore from previous checkpoint ===========

    from_prev = int(getattr(cfg, 'frompreviouscheckpoint', 0))
    restore_path = getattr(cfg, 'restore_params_path', None)
    if from_prev and (not restore_path or restore_path == ""):
        outputs_path = Path("outputs")
        pkl_files = list(outputs_path.rglob("trained_model.pkl"))
        if pkl_files:
            pkl_files = sorted(pkl_files, key=lambda p: p.stat().st_mtime)
            if len(pkl_files) >= from_prev:
                chosen_pkl = pkl_files[-from_prev]
                cfg.restore_params_path = str(chosen_pkl)
                print(f"[Auto-restore] Using checkpoint: {chosen_pkl}")
            else:
                print(f"[Auto-restore] Not enough checkpoints found for frompreviouscheckpoint={from_prev}")
        else:
            print("[Auto-restore] No trained_model.pkl files found in outputs/")

    restore_params = None
    restore_path = getattr(cfg, 'restore_params_path', None)
    if restore_path:
        print(f"Loading student parameters from: {restore_path}")
        restore_params = model.load_params(restore_path)
        print("Student parameters loaded successfully!")

    # =========== Network factory ===========

    network_factory = functools.partial(
        il_networks.make_il_networks,
        teacher_dec_hidden_layers=cfg.teacher_dec_hidden_layers,
        policy_dec_hidden_layers=cfg.policy_dec_hidden_layers,
        policy_proprio_proj_hidden_layers=cfg.policy_proprio_proj_hidden_layers,
        action_hidden_layer_sizes=cfg.action_hidden_layers,
        value_hidden_layer_sizes=cfg.value_hidden_layers,
        cnn_num_filters=cfg.cnn_num_filters,
        cnn_kernel_sizes=cfg.cnn_kernel_sizes,
        cnn_strides=cfg.cnn_strides,
        policy_pixels_key=cfg.policy_pixels_key,
        policy_pixels_key_1=cfg.policy_pixels_key_1,
        policy_proprio_key=cfg.policy_proprio_key,
        teacher_obs_key=cfg.teacher_obs_key,
        value_obs_key=cfg.value_obs_key,
    )

    # =========== Build train_fn ===========

    train_fn = functools.partial(
        dagger_train,
        env=env,
        teacher_params=teacher_params,
        restore_params=restore_params,
        num_dagger_epochs=cfg.num_dagger_epochs,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        num_evals=cfg.num_evals,
        num_eval_envs=cfg.num_eval_envs,
        unroll_length=cfg.unroll_length,
        batch_size=cfg.batch_size,
        num_minibatches=cfg.num_minibatches,
        align_updates_per_trigger=cfg.align_updates_per_trigger,
        network_factory=network_factory,
        num_envs=cfg.num_envs,
        episode_length=cfg.episode_length,
        action_repeat=cfg.action_repeat,
        normalize_observations=cfg.normalize_observations,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        deterministic_eval=True,
        wrap_env=False,
    )

    # =========== Progress callback ===========

    def progress(num_steps, metrics):
        times.append(datetime.now())

        log_dict = {
            "training/align_loss": metrics.get("training/align_loss", 0.0),
            "training/embed_loss": metrics.get("training/embed_loss", 0.0),
            "training/action_loss": metrics.get("training/action_loss", 0.0),
            "training/beta": metrics.get("training/beta", 0.0),
            "training/sps": metrics.get("training/sps", 0.0),
            "training/walltime": metrics.get("training/walltime", 0.0),
            "eval/episode_reward": metrics.get("eval/episode_reward", 0.0),
            "eval/avg_episode_length": metrics.get("eval/avg_episode_length", 0.0),
        }

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


if __name__ == "__main__":
    main()
