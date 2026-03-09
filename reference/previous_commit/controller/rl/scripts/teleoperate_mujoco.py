import sys
import os

# # Force JAX to use CPU only (no GPU needed for single-env teleoperation)
# os.environ['JAX_PLATFORMS'] = 'cpu'

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model
import jax
from jax import numpy as jp
from omegaconf import OmegaConf
import mujoco
from mujoco import mjx 
import mujoco.viewer 
from hand3fingers.controller.rl import Hand3FingersRL
import functools
import time
import json
import zmq
from threading import Thread, Lock
from pathlib import Path

class TeleoperatedPolicyEvaluator:
    def __init__(self, zmq_port=9872, model_path='checkpoint/150-450M/trained_model'):
        # ====== Load Config ======
        scripts_dir = Path(__file__).resolve().parent
        rl_dir = scripts_dir.parent  # rl directory
        yaml_file = rl_dir / 'cfg' / 'task' / 'hand3fingers.yaml'
        cfg = OmegaConf.load(yaml_file)
        self.cfg_env = cfg.env
        self.cfg_train = cfg.train
        self.cfg_env.eval_mode = True

        # ====== Initialize Environment =======
        self.eval_env = Hand3FingersRL(cfg=self.cfg_env)
        self.mj_model = self.eval_env.sys.mj_model
        self.mj_data = mujoco.MjData(self.mj_model)
        self.rng = jax.random.PRNGKey(0)

        self.state = self.eval_env.reset(rng=jax.random.PRNGKey(0), data=self.mj_data)
        self.ctrl = jp.zeros(self.mj_model.nu)

        # Debug initial state
        print(f"Initial target_tip_rel shape: {self.state.state_vars['target_tip_rel'].shape}")
        print(f"Initial action_buffer shape: {self.state.state_vars['action_buffer'].shape}")

        # ====== Load Policy ======
        self.model_path = model_path
        self._setup_policy()

        # ====== Teleoperation Setup ======
        self.context = zmq.Context()
        self.socket = zmq.Context().socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{zmq_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

        # Target management
        self.target_lock = Lock()
        self.latest_targets = None
        self.use_teleoperation = False
        self.is_running = False

        print(f"Listening for teleoperation targets on port {zmq_port}")

    def _setup_policy(self):
        """Setup the policy network"""
        obs_dict = self.state.obs
        action_size = self.ctrl.shape[0]

        print(f"Observation structure:")
        for key, obs in obs_dict.items():
            print(f"  {key}: shape={obs.shape}, size={obs.shape[0]}")

        observation_size = {
            "state": obs_dict["state"].shape[0],
            "privileged_state": obs_dict["privileged_state"].shape[0]
        }

        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=self.cfg_train.policy_hidden_layers,
            value_hidden_layer_sizes=self.cfg_train.value_hidden_layers,
            policy_obs_key=self.cfg_train.policy_obs_key,
            value_obs_key=self.cfg_train.value_obs_key,
        )

        ppo_network = network_factory(
            observation_size, action_size, 
            preprocess_observations_fn=running_statistics.normalize
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)
        params = model.load_params(self.model_path)
        self.inference_fn = make_policy(params)
        self.jit_inference_fn = jax.jit(self.inference_fn)

        # ====== Benchmark JIT compilation time ======
        print("Benchmarking JIT inference function...")
        
        # Get a sample observation for timing
        obs_sample = self.state.obs
        rng_sample = jax.random.PRNGKey(42)
        
        # Time the first call (includes JIT compilation)
        print("Running first inference (with JIT compilation)...")
        start_time = time.time()
        sample_action, _ = self.jit_inference_fn(obs_sample, rng_sample)
        first_call_time = time.time() - start_time
        print(f"  First call (JIT + execution): {first_call_time:.4f} seconds")
        
        # Time a few subsequent calls (already compiled)
        print("Running subsequent inferences (compiled)...")
        times = []
        for i in range(5):
            rng_sample = jax.random.PRNGKey(42 + i)
            start_time = time.time()
            sample_action, _ = self.jit_inference_fn(obs_sample, rng_sample)
            call_time = time.time() - start_time
            times.append(call_time)
        
        avg_time = sum(times) / len(times)
        print(f"  Average compiled inference time: {avg_time:.6f} seconds ({1/avg_time:.1f} Hz)")
        print(f"  Sample action shape: {sample_action.shape}")
        print(f"  Sample action range: [{sample_action.min():.3f}, {sample_action.max():.3f}]")

        print("Policy created successfully!")

    def listen_for_targets(self):
        """Background thread to receive teleoperation targets"""
        while self.is_running:
            try:
                message = self.socket.recv_string()
                data = json.loads(message)
                
                # Extract tip positions and convert to numpy arrays
                tip_positions = data['tip_positions']
                
                # Create target array with same shape as original (3, 3)
                target_array = jp.array([
                    tip_positions['thumb'],
                    tip_positions['index'], 
                    tip_positions['middle']
                ])
                
                # print(f"Target array shape: {target_array.shape}")  # Debug
                
                # Thread-safe update
                with self.target_lock:
                    self.latest_targets = target_array
                    
                # print(f"📡 Received targets: thumb={tip_positions['thumb'][:2]}, "
                #       f"index={tip_positions['index'][:2]}, middle={tip_positions['middle'][:2]}")
                      
            except zmq.Again:
                # No message available - continue
                continue
            except Exception as e:
                print(f"Error receiving targets: {e}")

    def update_environment_targets(self):
        """Update environment with latest teleoperation targets"""
        with self.target_lock:
            if self.latest_targets is not None:
                # Make sure the target has the right shape
                original_shape = self.state.state_vars['target_tip_rel'].shape
                print(f"Original target shape: {original_shape}, New target shape: {self.latest_targets.shape}")
                
                # Update the environment's target
                new_state_vars = self.state.state_vars.copy()
                new_state_vars['target_tip_rel'] = self.latest_targets
                self.state = self.state.replace(state_vars=new_state_vars)
                print(f"🎯 Updated targets in environment")
                return True
        return False

    def run_evaluation(self, use_teleoperation=True):
        """Run policy evaluation with teleoperation (runs until viewer closed or Ctrl+C)"""
        self.use_teleoperation = use_teleoperation
        self.is_running = True

        print("=" * 60)
        print("POLICY EVALUATION WITH TELEOPERATION")
        print("=" * 60)
        print("Instructions:")
        print("1. Start teleoperation in another terminal:")
        print("   hand3fingers-teleoperator-mujoco")
        print("2. Move the hand manually to set targets")
        print("3. Watch the policy try to follow your movements!")
        print("4. Close viewer or press Ctrl+C to stop")
        print("=" * 60)

        # Start teleoperation listener if enabled
        if use_teleoperation:
            listener_thread = Thread(target=self.listen_for_targets, daemon=True)
            listener_thread.start()
            print("🎧 Listening for teleoperation targets...")
            
            # Wait a bit for first target
            print("Waiting for first teleoperation target...")
            time.sleep(2)

        # ====== Add Performance Profiling ======
        timing_stats = {
            'get_obs': [],
            'inference': [],
            'physics_step': [],
            'state_update': [],
            'viewer_sync': [],
            'total_loop': []
        }

        try:
            with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
                step_count = 0
                last_target_update = -1
                last_timing_report = 0

                # --------- Set camera parameters here ---------
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # Use free camera
                viewer.cam.azimuth = 128      # degrees
                viewer.cam.elevation = -42    # degrees
                viewer.cam.distance = 0.36     # zoom (experiment with value)
                viewer.cam.lookat[:] = [-0.025, -0.019, 0.147]  # center of view (x, y, z)
                # ---------------------------------------------
                
                while viewer.is_running():
                    loop_start = time.time()
                    
                    # Update targets from teleoperation every 10 steps
                    if use_teleoperation and step_count % 10 == 0:
                        if self.update_environment_targets():
                            last_target_update = step_count

                    # Get new random key for policy
                    act_rng, self.rng = jax.random.split(self.rng)

                    try:
                        # ====== Time get_obs ======
                        obs_start = time.time()
                        self.state = self.eval_env._get_obs(self.state, self.ctrl)
                        timing_stats['get_obs'].append(time.time() - obs_start)
                        
                        # ====== Time inference ======
                        inf_start = time.time()
                        obs = self.state.obs  # Extract obs dictionary from state
                        self.ctrl, _ = self.jit_inference_fn(obs, act_rng)
                        timing_stats['inference'].append(time.time() - inf_start)
                        
                        # Apply action to MuJoCo data
                        self.mj_data.ctrl = self.ctrl
                        
                        # ====== Time physics steps ======
                        phys_start = time.time()
                        for _ in range(self.eval_env._n_frames):
                            mujoco.mj_step(self.mj_model, self.mj_data)
                        timing_stats['physics_step'].append(time.time() - phys_start)

                        # ====== Time state update ======
                        state_start = time.time()
                        data = mjx.put_data(self.mj_model, self.mj_data)
                        self.state = self.state.replace(pipeline_state=data)
                        
                        # Get action_buffer from current state and update with new action
                        action_buffer = self.state.state_vars['action_buffer']
                        new_action_buffer = jp.roll(action_buffer, shift=-1, axis=0)
                        new_action_buffer = new_action_buffer.at[-1].set(self.ctrl)
                        
                        new_state_vars = self.state.state_vars.copy()
                        new_state_vars['action_buffer'] = new_action_buffer
                        self.state = self.state.replace(state_vars=new_state_vars)
                        timing_stats['state_update'].append(time.time() - state_start)

                    except Exception as e:
                        print(f"Error in step {step_count}: {e}")
                        print(f"State target_tip_rel shape: {self.state.state_vars['target_tip_rel'].shape}")
                        print(f"Action buffer shape: {self.state.state_vars.get('action_buffer', jp.zeros((5, 12))).shape}")
                        import traceback
                        traceback.print_exc()
                        break

                    # ====== Time viewer sync ======
                    sync_start = time.time()
                    viewer.sync()
                    timing_stats['viewer_sync'].append(time.time() - sync_start)
                    
                    # Total loop time
                    loop_time = time.time() - loop_start
                    timing_stats['total_loop'].append(loop_time)
                    
                    # Control timing (this might be the culprit!)
                    time_until_next_step = self.eval_env.dt - loop_time
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                    
                    step_count += 1
                    
                    # ====== Print performance report every 200 steps ======
                    if step_count % 200 == 0:
                        print(f"\n📊 Performance Report (Step {step_count}):")
                        print(f"  Target dt: {self.eval_env.dt:.4f}s ({1/self.eval_env.dt:.1f} Hz)")
                        
                        for name, times in timing_stats.items():
                            if times:
                                avg_time = sum(times[-200:]) / min(len(times), 200)  # Last 200 samples
                                print(f"  {name:12}: {avg_time*1000:.2f}ms ({1/avg_time:.0f} Hz)")
                        
                        actual_fps = 200 / sum(timing_stats['total_loop'][-200:])
                        print(f"  {'Actual FPS':12}: {actual_fps:.1f} Hz")
                        
                        if use_teleoperation:
                            if last_target_update >= 0:
                                print(f"  Last target update: {step_count - last_target_update} steps ago")
                            else:
                                print(f"  No targets received yet")

                print("Simulation completed!")

            # ====== Final Performance Summary ======
            print("\n" + "="*60)
            print("FINAL PERFORMANCE SUMMARY")
            print("="*60)
            for name, times in timing_stats.items():
                if times:
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    print(f"{name:12}: avg={avg_time*1000:.2f}ms, max={max_time*1000:.2f}ms, freq={1/avg_time:.0f}Hz")

        except KeyboardInterrupt:
            print("Simulation interrupted by user")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            if hasattr(self, 'socket'):
                self.socket.close()
            if hasattr(self, 'context'):
                self.context.term()

# Standalone function
def run_with_teleoperation(model_path, zmq_port=9872):
    """Run policy with teleoperation targets (runs until stopped)"""
    evaluator = TeleoperatedPolicyEvaluator(zmq_port=zmq_port, model_path=model_path)
    evaluator.run_evaluation(use_teleoperation=True)

def main():
    """Entry point for the teleoperate script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Policy Evaluation with Teleoperation - Runs until viewer closed or Ctrl+C")
    parser.add_argument("--port", type=int, default=9872,
                       help="ZMQ port for teleoperation")
    parser.add_argument("--model_path", type=str, default="checkpoint/150-450M/trained_model",
                       help="Path to trained model parameters")
    
    args = parser.parse_args()
    
    print("Running policy evaluation with teleoperation...")
    run_with_teleoperation(args.model_path, args.port)

if __name__ == "__main__":
    main()