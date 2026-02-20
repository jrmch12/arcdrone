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
from arcdrone import ARCDroneRL_Vel
import time


# ========== Configuration ==========
MUJOCO_PATH = "/home/jrmch12/Documents/code/260110_arcdrone/arcdrone/assets/skydio_x2/scene.xml"
CHECKPOINT_PATH = "outputs/2026-01-29/13-22-31/trained_model.pkl"



def evaluate(
    model_path: str = "outputs/2026-01-28/23-20-35/trained_model.pkl",
    num_episodes: int = 5,
    max_steps: int = 200
):
    """
    Evaluate a trained velocity controller.

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
    yaml_file = rl_dir / 'cfg' / 'task' / 'arcdrone.yaml'
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
    env = ARCDroneRL_Vel(cfg=cfg_env)        

    
    
    # ====== Setup Policy Network ======

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng=rng)
    
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
    
    # ======= JIT compile =========
    print("JIT compiling...")
    inference = jax.jit(inference_fn)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Warm-up JIT
    state = reset_fn(rng)
    action, _ = inference(state.obs, rng)
    _ = step_fn(state, action)
    print("✓ Compiled successfully\n")
    
    # ====== Setup MuJoCo Viewer ======

    m = mujoco.MjModel.from_xml_path(MUJOCO_PATH)
    d = mujoco.MjData(m)
    viewer = mujoco.viewer.launch_passive(m, d)  # TODO: add key callback for quitting, pausing, etc.
    print("✓ Viewer launched\n")
    
    # ====== Evaluation Loop ======
    
    # Timing accumulators
    times = {
        "inference": 0,
        "step": 0,
        "get_data_into": 0,
        "viewer_sync": 0,
        "total_step": 0
    }
    step_count = 0
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        # Reset environment
        rng, reset_rng = jax.random.split(rng)
        state = reset_fn(rng=reset_rng)

        # Reset viewer state
        mjx.get_data_into(d, m, state.pipeline_state)
        
        for step in range(max_steps):
            step_start = time.perf_counter()
            
            rng, action_rng = jax.random.split(rng)
            
            # Time inference
            t0 = time.perf_counter()
            action, _ = inference(state.obs, action_rng)
            inference_t = time.perf_counter() - t0
            times["inference"] += inference_t
            
            # Time step
            t0 = time.perf_counter()
            state = step_fn(state, action)
            step_t = time.perf_counter() - t0
            times["step"] += step_t
            
            # Time data sync
            t0 = time.perf_counter()
            mjx.get_data_into(d, m, state.pipeline_state)
            get_data_t = time.perf_counter() - t0
            times["get_data_into"] += get_data_t
            
            # Time viewer sync
            t0 = time.perf_counter()
            viewer.sync()
            viewer_t = time.perf_counter() - t0
            times["viewer_sync"] += viewer_t
            
            times["total_step"] += time.perf_counter() - step_start
            step_count += 1
            
            # Debug: show first 5 steps per episode
            if step < 5:
                print(f"    Step {step}: infer={inference_t*1000:.1f}ms step={step_t*1000:.1f}ms data={get_data_t*1000:.1f}ms view={viewer_t*1000:.1f}ms")
            
            if state.done:
                break
    
    # Print timing summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    if step_count > 0:
        print(f"Total steps: {step_count}")
        print(f"Inference:     {times['inference']:.4f}s ({times['inference']/step_count*1000:.2f}ms/step)")
        print(f"Step:          {times['step']:.4f}s ({times['step']/step_count*1000:.2f}ms/step)")
        print(f"Get data into: {times['get_data_into']:.4f}s ({times['get_data_into']/step_count*1000:.2f}ms/step)")
        print(f"Viewer sync:   {times['viewer_sync']:.4f}s ({times['viewer_sync']/step_count*1000:.2f}ms/step)")
        print(f"Total step:    {times['total_step']:.4f}s ({times['total_step']/step_count*1000:.2f}ms/step)")
        print("=" * 60)
    
    # Cleanup
    viewer.close()
    
    
    return 


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained ARCDrone velocity controller')
    parser.add_argument('--model_path', type=str, default=CHECKPOINT_PATH,
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to evaluate')
    parser.add_argument('--steps', type=int, default=200,
                        help='Maximum steps per episode')
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        num_episodes=args.episodes,
        max_steps=args.steps
    )


