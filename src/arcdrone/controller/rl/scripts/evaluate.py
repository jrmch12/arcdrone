"""
Minimal evaluation script for trained ARCDrone velocity controller.
"""
import jax
from jax import numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model
from omegaconf import OmegaConf
from pathlib import Path
import functools
import mujoco
from mujoco import mjx
import mujoco.viewer
import arcdrone
from arcdrone import ARCDroneRL_Vel, ARCDroneRL_Landing
import time


def find_latest_checkpoint(outputs_dir: str = "outputs") -> str:
    """Find the latest trained_model.pkl file in the outputs directory."""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    
    # Find all .pkl files recursively
    pkl_files = list(outputs_path.rglob("*.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {outputs_dir}")
    
    # Sort by modification time, most recent first
    latest_pkl = max(pkl_files, key=lambda p: p.stat().st_mtime)
    
    return str(latest_pkl)


# ========== Configuration ==========
# Get project root from arcdrone package installation location
_arcdrone_pkg = Path(arcdrone.__file__).resolve().parent
_project_root = _arcdrone_pkg.parent.parent  # arcdrone -> src -> project_root
MUJOCO_PATH = str(_project_root / "assets" / "skydio_x2" / "scene.xml")
CHECKPOINT_PATH = find_latest_checkpoint(str(_project_root / "outputs"))
print(f"Latest checkpoint found: {CHECKPOINT_PATH}")



def evaluate(
    model_path: str = "outputs/2026-01-28/23-20-35/trained_model.pkl",
    num_episodes: int = 20,
    max_steps: int = 200
):
    """
    Evaluate a trained controller.
    Args:
        model_path: Path to trained model parameters
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        target_velocity: Target velocity [vx, vy, vz] in m/s
        render: Whether to render with MuJoCo viewer
    """
    
    # ====== Load Config ======
    scripts_dir = Path(__file__).resolve().parent
    rl_dir = scripts_dir.parent
    yaml_file = rl_dir / 'cfg' / 'task' / 'landing.yaml'
    cfg = OmegaConf.load(yaml_file)
    
    cfg_env = cfg.env
    cfg_train = cfg.train
    cfg_env.eval_mode = True
    
    print("=" * 60)
    print("ARCDrone Velocity Controller - Evaluation")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)


    # ====== Initialize Environment ======
    env = ARCDroneRL_Landing(cfg=cfg_env)        

    
    # ====== Setup Reset ======  
    print("Setting up reset function...")
    rng = jax.random.PRNGKey(0)
    print("JIT compiling reset function...")
    jit_reset = jax.jit(env.reset)
    # Warm-up JIT 
    state = jit_reset(rng)
    print("✓ Reset function compiled successfully\n")


    # ====== Setup Inference ======
    print("Setting up inference function...")
    observation_size = {
        "state": state.obs["state"].shape[0],
        "privileged_state": state.obs["privileged_state"].shape[0]
    }
    action_size = env.sys.nu
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=cfg_train.policy_hidden_layers,
        value_hidden_layer_sizes=cfg_train.value_hidden_layers,
        policy_obs_key=cfg_train.policy_obs_key,
        value_obs_key=cfg_train.value_obs_key,
    )
    ppo_network = network_factory(
        observation_size, 
        action_size, 
        preprocess_observations_fn=running_statistics.normalize
    )
    print("\nLoading trained model...")
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    params = model.load_params(model_path)
    inference_fn = make_policy(params)
    print("JIT compiling inference function...")
    jit_inference_fn = jax.jit(inference_fn)
    # Warm-up JIT
    action, _ = jit_inference_fn(state.obs, rng)
    print("✓ Inference function compiled successfully\n")


    # ====== Setup Step ======
    print("Setting up step function...")
    print("JIT compiling step function...")
    jit_step = jax.jit(env.step)
    # Warm-up JIT - test both reset→step and step→step transitions
    state1 = jit_step(state, action)
    action2, _ = jit_inference_fn(state1.obs, rng)
    _ = jit_step(state1, action2)
    print("✓ Step function compiled successfully\n")
    
    # ====== Setup MuJoCo Viewer ======

    m = mujoco.MjModel.from_xml_path(MUJOCO_PATH)
    d = mujoco.MjData(m)
    viewer = mujoco.viewer.launch_passive(m, d)  # TODO: add key callback for quitting, pausing, etc.
    print("✓ Viewer launched\n")





    # ====== Evaluation Loop ======
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        # Reset environment
        rng, reset_key = jax.random.split(rng)
        state = jit_reset(reset_key)

        # Reset viewer state
        mjx.get_data_into(d, m, state.pipeline_state)
        
        for step in range(max_steps):
            rng, action_key = jax.random.split(rng)
            
            # Inference
            action, _ = jit_inference_fn(state.obs, action_key)
            
            # Step
            state = jit_step(state, action)
            
            # Sync viewer
            mjx.get_data_into(d, m, state.pipeline_state)
            viewer.sync()
            
            if state.done:
                break
    
    # Cleanup
    viewer.close()
    
    
    return 






if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained ARCDrone velocity controller')
    parser.add_argument('--model_path', type=str, default=CHECKPOINT_PATH,
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of episodes to evaluate')
    parser.add_argument('--steps', type=int, default=200,
                        help='Maximum steps per episode')
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        num_episodes=args.episodes,
        max_steps=args.steps
    )


