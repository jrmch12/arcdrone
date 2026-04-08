"""Train a vision-based drone landing policy using Brax PPO.

Follows the CartpoleBalance vision training pipeline exactly:
- Uses brax.training.agents.ppo with vision networks
- Single top-down camera, grayscale, 64x64, 3-frame stack
- Pure pixel observations (no proprioception in policy)

Usage:
    source /home/jrmch12f/Documents/code/mujoco_playground/.venv/bin/activate
    cd /home/jrmch12f/Documents/code/borrador_braxenvs
    python src/arcdrone/New_attempt/train.py [flags]

    # Quick test (few steps):
    python src/arcdrone/New_attempt/train.py --num_timesteps 1000 --num_envs 4

    # Full training:
    python src/arcdrone/New_attempt/train.py --num_timesteps 50_000_000 --num_envs 512
"""

import datetime
import functools
import json
import os
from pathlib import Path
import sys
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco

from mujoco_playground._src import wrapper

# Self-contained import: add New_attempt dir to path so env.py is importable
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from env import DroneLanding, default_config, default_vision_config

try:
    import wandb
except ImportError:
    wandb = None

try:
    import tensorboardX
except ImportError:
    tensorboardX = None


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

# -- Flags --
_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", 50_000_000, "Total env steps")
_NUM_EVALS = flags.DEFINE_integer("num_evals", 10, "Eval rollouts during training")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 2048, "Parallel training envs")
_NUM_EVAL_ENVS = flags.DEFINE_integer("num_eval_envs", 128, "Parallel eval envs")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_VIDEOS = flags.DEFINE_integer("num_videos", 1, "Videos to record after training")
_USE_WANDB = flags.DEFINE_boolean("use_wandb", False, "Log to Weights & Biases")
_USE_TB = flags.DEFINE_boolean("use_tb", False, "Log to TensorBoard")
_SUFFIX = flags.DEFINE_string("suffix", None, "Experiment name suffix")
_LOAD_CHECKPOINT = flags.DEFINE_string("load_checkpoint_path", None, "Resume from checkpoint")
_PATIENCE = flags.DEFINE_integer("patience", 8, "Stop after N evals without improvement")
_BLIND = flags.DEFINE_boolean("blind", False, "Zero out pixels (test state-only learning)")


def main(argv):
    del argv

    # -- Environment config --
    env_cfg = default_config()
    env_cfg.vision = True
    env_cfg.vision_config.nworld = _NUM_ENVS.value
    env_cfg.blind = _BLIND.value
    if _BLIND.value:
        print("*** BLIND MODE: pixels zeroed out, testing state-only learning ***")

    env = DroneLanding(config=env_cfg)

    print(f"Action size: {env.action_size}")
    print(f"n_substeps: {env.n_substeps}")

    # -- PPO config --
    # batch_size * num_minibatches must be divisible by num_envs.
    # Brax scans (batch_size * num_minibatches // num_envs) iterations,
    # each doing unroll_length env steps. Keep scan length small (~1-2)
    # to avoid OOM on 12GB GPU with vision rendering.
    num_envs = _NUM_ENVS.value
    num_minibatches = 8
    batch_size = 256

    ppo_params = config_dict.create(
        num_timesteps=_NUM_TIMESTEPS.value,
        num_evals=_NUM_EVALS.value,
        reward_scaling=5.0,
        episode_length=200,
        normalize_observations=False,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=num_minibatches,
        num_updates_per_batch=8,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=5e-3,
        num_envs=num_envs,
        batch_size=batch_size,
        max_grad_norm=1.0,
    )

    # -- Network factory: Brax's built-in vision PPO networks --
    # Pure pixel-only: no state_obs_key, like CartpoleBalance
    network_factory = functools.partial(
        ppo_networks_vision.make_ppo_networks_vision,
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256),
        cnn_output_channels=(32, 64, 64),
        cnn_kernel_size=(8, 4, 3),
        cnn_stride=(4, 2, 1),
        cnn_padding="valid",
        cnn_activation="relu",
        cnn_max_pool=False,
        cnn_global_pool="none",
        cnn_kernel_init_fn="orthogonal",
        cnn_kernel_init_kwargs={"scale": 1.41421356},
        output_kernel_init_fn="orthogonal",
        output_kernel_init_kwargs={"scale": 0.01},
    )

    # -- Experiment setup --
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"DroneLanding-{timestamp}"
    if _SUFFIX.value:
        exp_name += f"-{_SUFFIX.value}"
    print(f"Experiment: {exp_name}")

    logdir = epath.Path("logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Logs: {logdir}")

    # Save config
    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    # -- Logging --
    if _USE_WANDB.value and wandb is not None:
        wandb.init(
            project="drone-landing-vision",
            name=exp_name,
            config={
                **env_cfg.to_dict(),
                **dict(ppo_params),
                "seed": _SEED.value,
            },
        )

    writer = None
    if _USE_TB.value and tensorboardX is not None:
        writer = tensorboardX.SummaryWriter(logdir)

    # -- Checkpoint loading --
    restore_checkpoint_path = None
    if _LOAD_CHECKPOINT.value:
        ckpt = epath.Path(_LOAD_CHECKPOINT.value).resolve()
        if ckpt.is_dir():
            latest = sorted(
                [c for c in ckpt.glob("*") if c.is_dir()],
                key=lambda x: int(x.name),
            )
            restore_checkpoint_path = latest[-1] if latest else None
        else:
            restore_checkpoint_path = ckpt
        if restore_checkpoint_path:
            print(f"Restoring from: {restore_checkpoint_path}")

    # -- Eval env --
    eval_cfg = default_config()
    eval_cfg.vision = True
    eval_cfg.vision_config.nworld = _NUM_EVAL_ENVS.value
    eval_cfg.blind = _BLIND.value
    eval_env = DroneLanding(config=eval_cfg)

    # -- Training --
    times = [time.monotonic()]
    best_reward = float("-inf")
    best_step = 0
    evals_without_improvement = 0

    def progress(num_steps, metrics):
        nonlocal best_reward, best_step, evals_without_improvement
        times.append(time.monotonic())

        reward = metrics.get("eval/episode_reward", 0.0)
        if reward > best_reward + 0.5:  # meaningful improvement threshold
            best_reward = reward
            best_step = num_steps
            evals_without_improvement = 0
        else:
            evals_without_improvement += 1

        # Build structured log dict (like reference vision_landing_rl)
        log_dict = {
            "eval/episode_reward": metrics.get("eval/episode_reward", 0.0),
            "eval/episode_length": metrics.get("eval/avg_episode_length", 0.0),
            "eval/sps": metrics.get("eval/sps", 0.0),
            "training/total_loss": metrics.get("training/total_loss", 0.0),
            "training/policy_loss": metrics.get("training/policy_loss", 0.0),
            "training/v_loss": metrics.get("training/v_loss", 0.0),
            "training/entropy_loss": metrics.get("training/entropy_loss", 0.0),
            "training/learning_rate": metrics.get("training/learning_rate", 0.0),
            "training/kl_mean": metrics.get("training/kl_mean", 0.0),
            "training/sps": metrics.get("training/sps", 0.0),
        }

        # Log per-reward-component metrics if available
        for key, value in metrics.items():
            if key.startswith("eval/episode_reward_") and not key.endswith("_std"):
                reward_name = key[len("eval/episode_reward_"):]
                log_dict[f"rewards/{reward_name}"] = value
                std_key = f"{key}_std"
                if std_key in metrics:
                    std_val = metrics[std_key]
                    log_dict[f"std/{reward_name}_upper"] = value + std_val
                    log_dict[f"std/{reward_name}_lower"] = value - std_val

        if _USE_WANDB.value and wandb is not None:
            wandb.log(log_dict, step=num_steps)
        if _USE_TB.value and writer is not None:
            for key, value in log_dict.items():
                writer.add_scalar(key, value, num_steps)
            writer.flush()
        print(f"{num_steps}: reward={metrics.get('eval/episode_reward', 0):.3f}  (best={best_reward:.3f} @{best_step})  patience={_PATIENCE.value - evals_without_improvement}")

        # Early stopping on plateau
        if evals_without_improvement >= _PATIENCE.value:
            print(f"EARLY STOP: No improvement for {_PATIENCE.value} evals. Best={best_reward:.3f} @{best_step}")
            raise StopIteration("plateau")

    training_params = dict(ppo_params)
    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=_SEED.value,
        restore_checkpoint_path=restore_checkpoint_path,
        save_checkpoint_path=ckpt_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=_NUM_EVAL_ENVS.value,
    )

    try:
        make_inference_fn, params, _ = train_fn(
            environment=env,
            progress_fn=progress,
            eval_env=eval_env,
        )
    except StopIteration:
        print("Training stopped early (plateau).")
        make_inference_fn = None
        params = None
    except AssertionError as e:
        # Brax's pmap.assert_is_replicated fails when NaN propagates.
        # Checkpoints are already saved at each eval — just warn and continue.
        print(f"WARNING: Training ended with assertion (likely NaN divergence): {e}")
        print("Checkpoints were saved during training — use the best one.")
        make_inference_fn = None
        params = None

    print("Training complete.")
    print(f"Best reward: {best_reward:.3f} at step {best_step}")
    print(f"Best checkpoint: {ckpt_path}/{best_step:015d}")
    if len(times) > 1:
        print(f"JIT compile time: {times[1] - times[0]:.1f}s")
        print(f"Training time: {times[-1] - times[1]:.1f}s")

    # -- Save wandb artifact and finish --
    if _USE_WANDB.value and wandb is not None:
        artifact = wandb.Artifact(
            name="drone-landing-vision", type="model",
            description="Vision PPO drone landing checkpoint",
        )
        artifact.add_dir(str(ckpt_path))
        wandb.log_artifact(artifact)
        wandb.finish()

    # -- Inference & video --
    if _NUM_VIDEOS.value > 0 and params is not None:
        print("Generating rollout videos...")
        inference_fn = make_inference_fn(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)

        vid_cfg = default_config()
        vid_cfg.vision = True
        # Need at least 4 worlds for batched rendering to work
        vid_cfg.vision_config.nworld = max(_NUM_VIDEOS.value, 4)
        infer_env = DroneLanding(config=vid_cfg)

        wrapped_infer_env = wrapper.wrap_for_brax_training(
            infer_env, episode_length=200, action_repeat=1
        )

        nworld = max(_NUM_VIDEOS.value, 4)
        rng = jax.random.split(jax.random.PRNGKey(_SEED.value), nworld)
        reset_states = jax.jit(wrapped_infer_env.reset)(rng)

        empty_data = reset_states.data.__class__(
            **{k: None for k in reset_states.data.__annotations__}
        )
        empty_traj = reset_states.__class__(
            **{k: None for k in reset_states.__annotations__}
        )
        empty_traj = empty_traj.replace(data=empty_data)

        def step_fn(carry, _):
            state, rng = carry
            rng, act_key = jax.random.split(rng)
            act_keys = jax.random.split(act_key, nworld)
            act = jax.vmap(jit_inference_fn)(state.obs, act_keys)[0]
            state = wrapped_infer_env.step(state, act)
            traj_data = empty_traj.tree_replace({
                "data.qpos": state.data.qpos,
                "data.qvel": state.data.qvel,
                "data.time": state.data.time,
                "data.ctrl": state.data.ctrl,
                "data.mocap_pos": state.data.mocap_pos,
                "data.mocap_quat": state.data.mocap_quat,
                "data.xfrc_applied": state.data.xfrc_applied,
            })
            return (state, rng), traj_data

        @jax.jit
        def do_rollout(state, rng):
            _, traj = jax.lax.scan(step_fn, (state, rng), None, length=200)
            return traj

        traj_stacked = do_rollout(reset_states, jax.random.PRNGKey(_SEED.value + 1))
        traj_stacked = jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj_stacked)

        trajectories = []
        for i in range(_NUM_VIDEOS.value):
            t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
            trajectories.append([
                jax.tree.map(lambda x, j=j: x[j], t) for j in range(200)
            ])

        render_every = 2
        fps = 1.0 / infer_env.dt / render_every
        for i, rollout in enumerate(trajectories):
            traj = rollout[::render_every]
            frames = infer_env.render(traj, height=480, width=640, camera="outer_camera")
            media.write_video(f"rollout_landing_{i}.mp4", frames, fps=fps)
            print(f"Saved rollout_landing_{i}.mp4")


def run():
    app.run(main)


if __name__ == "__main__":
    run()
