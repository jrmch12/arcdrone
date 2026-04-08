"""Evaluate a trained vision drone landing policy in MuJoCo viewer.

Usage:
    source /home/jrmch12f/Documents/code/mujoco_playground/.venv/bin/activate
    cd /home/jrmch12f/Documents/code/borrador_braxenvs
    python src/arcdrone/New_attempt/evaluate.py

    # Specific checkpoint:
    python src/arcdrone/New_attempt/evaluate.py --checkpoint logs/DroneLanding-.../checkpoints
"""

from pathlib import Path
import sys
import time

from absl import app
from absl import flags
from brax.envs.wrappers.training import VmapWrapper
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training import checkpoint
import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from env import DroneLanding, default_config

_WORKSPACE = Path(__file__).resolve().parents[3]

_CHECKPOINT = flags.DEFINE_string(
    "checkpoint", None,
    "Path to checkpoint dir (e.g. logs/DroneLanding-.../checkpoints). "
    "If not provided, finds the latest DroneLanding checkpoint."
)
_EPISODES = flags.DEFINE_integer("episodes", 20, "Number of episodes")
_MAX_STEPS = flags.DEFINE_integer("max_steps", 200, "Steps per episode")
_SEED = flags.DEFINE_integer("seed", 0, "Random seed")
_DETERMINISTIC = flags.DEFINE_boolean("deterministic", True, "Deterministic policy")
_SLEEP = flags.DEFINE_boolean("sleep", True, "Real-time playback (sleep between steps)")

# Number of parallel worlds for the render context (must be > 0)
_NWORLD = 4


def find_latest_checkpoint() -> Path:
    """Find latest DroneLanding checkpoint directory."""
    logs_dir = _WORKSPACE / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"No logs directory at {logs_dir}")

    run_dirs = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("DroneLanding")],
        key=lambda d: d.name,
    )
    if not run_dirs:
        raise FileNotFoundError("No DroneLanding runs found in logs/")

    ckpt_dir = run_dirs[-1] / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints in {run_dirs[-1]}")
    return ckpt_dir


def find_best_step(ckpt_dir: Path) -> Path:
    """Find the latest (highest step) checkpoint."""
    step_dirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No step directories in {ckpt_dir}")
    return step_dirs[-1]


def main(argv):
    del argv

    # -- Resolve checkpoint --
    if _CHECKPOINT.value:
        ckpt_dir = Path(_CHECKPOINT.value)
        if not ckpt_dir.is_absolute():
            ckpt_dir = _WORKSPACE / ckpt_dir
    else:
        ckpt_dir = find_latest_checkpoint()

    step_dir = find_best_step(ckpt_dir)
    print(f"Checkpoint: {step_dir}")

    # -- Load network and params --
    normalize_fn = lambda x, y: x  # normalize_observations=False

    ppo_network = ppo_networks_vision.make_ppo_networks_vision(
        observation_size={"pixels/view_0": (64, 64, 3)},
        action_size=4,
        preprocess_observations_fn=normalize_fn,
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
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    params = checkpoint.load(str(step_dir.resolve()))
    inference_fn = make_policy(params, deterministic=_DETERMINISTIC.value)
    jit_inference_fn = jax.jit(inference_fn)

    # -- Create vision env with VmapWrapper (needed for batched rendering) --
    cfg = default_config()
    cfg.vision = True
    cfg.vision_config.nworld = _NWORLD
    env = DroneLanding(config=cfg)

    # Use VmapWrapper to handle batched rendering (same as training)
    vmapped_env = VmapWrapper(env)

    jit_reset = jax.jit(vmapped_env.reset)
    jit_step = jax.jit(vmapped_env.step)

    # -- JIT warm-up --
    print("JIT compiling...")
    rng = jax.random.PRNGKey(_SEED.value)
    rngs = jax.random.split(rng, _NWORLD)
    state = jit_reset(rngs)

    # Inference on single env (index 0)
    obs_0 = jax.tree.map(lambda x: x[0], state.obs)
    action_0, _ = jit_inference_fn(obs_0, rng)

    # Step all envs with same action (only env 0 matters for viewer)
    actions = jp.broadcast_to(action_0, (_NWORLD,) + action_0.shape)
    state1 = jit_step(state, actions)

    obs_1 = jax.tree.map(lambda x: x[0], state1.obs)
    action_1, _ = jit_inference_fn(obs_1, rng)
    actions = jp.broadcast_to(action_1, (_NWORLD,) + action_1.shape)
    _ = jit_step(state1, actions)
    print("JIT compilation done.")

    # -- MuJoCo viewer --
    scene_path = str(_WORKSPACE / "assets" / "skydio_x2" / "scene.xml")
    vis_model = mujoco.MjModel.from_xml_path(scene_path)
    vis_data = mujoco.MjData(vis_model)

    def _sync_to_viewer(state):
        """Copy qpos/qvel from MJX env[0] into the viewer MjData."""
        import numpy as np
        vis_data.qpos[:] = np.array(state.data.qpos[0])
        vis_data.qvel[:] = np.array(state.data.qvel[0])
        mujoco.mj_forward(vis_model, vis_data)

    viewer = mujoco.viewer.launch_passive(vis_model, vis_data)
    print(f"Viewer launched. Running {_EPISODES.value} episodes...")

    dt = env.dt

    for ep in range(_EPISODES.value):
        if not viewer.is_running():
            break

        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, _NWORLD)
        state = jit_reset(reset_keys)

        _sync_to_viewer(state)
        viewer.sync()

        total_reward = 0.0
        for step in range(_MAX_STEPS.value):
            if not viewer.is_running():
                break

            step_start = time.monotonic()

            rng, action_key = jax.random.split(rng)
            obs_0 = jax.tree.map(lambda x: x[0], state.obs)
            action_0, _ = jit_inference_fn(obs_0, action_key)
            actions = jp.broadcast_to(action_0, (_NWORLD,) + action_0.shape)
            state = jit_step(state, actions)

            total_reward += float(state.reward[0])

            _sync_to_viewer(state)
            viewer.sync()

            if _SLEEP.value:
                elapsed = time.monotonic() - step_start
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)

            if state.done[0]:
                break

        pos = state.data.qpos[0, 0:3]
        print(f"Episode {ep+1:3d}: reward={total_reward:8.2f}, "
              f"steps={step+1:3d}, pos=({float(pos[0]):.2f}, {float(pos[1]):.2f}, {float(pos[2]):.2f})")

    if viewer.is_running():
        viewer.close()
    print("Done.")


if __name__ == "__main__":
    app.run(main)
