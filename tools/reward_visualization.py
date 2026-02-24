import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force JAX to use CPU only. We dont need accelerated JAX for this script.

import mujoco
import mujoco.viewer
import jax
from jax import numpy as jp
from omegaconf import OmegaConf
import numpy as np
from mujoco import mjx
from arcdrone import ARCDroneRL_Vel, ARCDroneRL_Landing, ARCDroneRL_Hover
from arcdrone.utils.plotjuggler import PlotJugglerLogger


class RewardAnalyzer:
    """Real-time reward component analysis tool.
    
    Allows manual control of the robot via MuJoCo GUI while computing
    reward components in real-time and streaming them to PlotJuggler
    for visualization. This is a sandbox tool for reward tuning and debugging.
    """
    
    def __init__(self, cfg_path: str, task_name: str = 'landing', layout_file: str = None):
        # Load configuration
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg.env
        
        # 1. Create MuJoCo model (for GUI interaction)
        self.mj_model = mujoco.MjModel.from_xml_path("assets/skydio_x2/mocap/scene_mocap.xml")
        self.mj_data = mujoco.MjData(self.mj_model)

        # 2. Create RL task instance (for reward computation)
        ENV_CLASSES = {
            'hover': ARCDroneRL_Hover,
            'landing': ARCDroneRL_Landing,
            'vel': ARCDroneRL_Vel,
        }
        if task_name not in ENV_CLASSES:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list(ENV_CLASSES.keys())}")
        env_class = ENV_CLASSES[task_name]
        self.rl_task = env_class(cfg=self.cfg)
        
        # 3. Create PlotJuggler logger for real-time visualization
        if layout_file and os.path.exists(layout_file):
            self.pj = PlotJugglerLogger(layout_file=layout_file)
        else:
            self.pj = PlotJugglerLogger()
        
        # 4. Initialize custom state with random goal
        rng = jax.random.PRNGKey(0)
        self.state = self.rl_task.reset(rng)
        
        # 5. Initialize accumulated reward for integration
        self.accumulated_reward = 0.0
        
        # ========== SANDBOX MODE ==========
        # From here we diverge from normal RL training:
        # - Actions are set to zero (manual GUI control instead)
        #   TODO: maybe a global flag to enable/disable metrics computation?
        # - Reward metrics are computed and streamed to PlotJuggler
        # ==================================
        
        print("✓ Reward Analyzer initialized")
        print("✓ Move the robot in the GUI to see reward components in PlotJuggler")
    
    def _update_state_from_mujoco(self):
        """Update state with current MuJoCo GUI state.
        
        Syncs the RL task's pipeline_state with the manually controlled
        MuJoCo simulation, allowing reward computation on GUI interactions.
        """
        # Convert MuJoCo data to MJX format
        mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        
        # Update pipeline state with current MuJoCo data
        state = self.state.replace(pipeline_state=mjx_data)
        
        return state
    
    # REFERENCE
    # def _update_target_visualization(self, viewer):
    #     """Update colored sphere markers."""
        

    #     # Clear previous markers by resetting geometry count
    #     viewer.user_scn.ngeom = 0
        

    #     finger_names = ['thumb', 'index', 'middle']
    #     for i, (name, pos) in enumerate(zip(finger_names, target_positions)):
    #         if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
    #             geom_idx = viewer.user_scn.ngeom
                
    #             mujoco.mjv_initGeom(
    #                 viewer.user_scn.geoms[geom_idx],
    #                 type=mujoco.mjtGeom.mjGEOM_SPHERE,
    #                 size=np.array([0.005, 0, 0]),  # 0.5cm radius (half the previous size)
    #                 pos=pos,
    #                 mat=np.eye(3).flatten(),
    #                 rgba=np.array(finger_colors[name])  # Color-coded by finger
    #             )
                
    #             viewer.user_scn.ngeom += 1
    
    def _prepare_metrics_for_logging(self, state) -> dict:
        """Prepare metrics dictionary for PlotJuggler logging.
        
        Converts metrics from state and adds additional computed values
        like total reward. This centralizes all metric transformations before logging.
        
        Args:
            state: Current state with computed reward components
            
        Returns:
            Dictionary ready for PlotJuggler with all metrics as Python types
        """
        metrics = {}
        excluded_metrics = {
            'reward_total',
            'reward_action_penalty',
        }
        
        # Convert all metrics from JAX arrays to Python types
        for key, value in state.metrics.items():
            if key in excluded_metrics:
                continue

            if hasattr(value, 'tolist'):  # JAX array
                converted = value.tolist()
            else:
                converted = value

            if not np.all(np.isfinite(np.asarray(converted))):
                continue

            metrics[key] = converted
        
        # # Pseudo-integrated total reward using actual reward components
        # # produced by src/arcdrone/controller/rl/task/velocity_mode/reward.py
        # current_step_reward = (
        #     metrics.get('reward_distance', 0.0) +
        #     metrics.get('reward_time_penalty', 0.0) +
        #     metrics.get('reward_overshoot', 0.0) +
        #     metrics.get('reward_oscillation', 0.0) +
        #     # metrics.get('reward_action_chattering', 0.0) +
        #     # metrics.get('reward_action_penalty', 0.0) +
        #     metrics.get('reward_ground_penalty', 0.0) +
        #     metrics.get('reward_success_bonus', 0.0)
        # )
        
        # # Accumulate (integrate) the reward
        # self.accumulated_reward += current_step_reward
        # metrics['pseudo_integrated_total_reward'] = self.accumulated_reward

        # Add termination/debug signals
        metrics['done'] = float(state.done)
        metrics['goal_achieved'] = float(state.state_vars.get('goal_achieved', 0.0))
        metrics['steps_within_success'] = float(state.state_vars.get('steps_within_success', 0.0))
        
        return metrics
    
    def run(self):
        """Main loop: sync GUI state, compute rewards, stream to PlotJuggler."""
        
        # Launch MuJoCo viewer in passive mode (manual control enabled)
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            
            # Real-time analysis loop
            while viewer.is_running():

                # 1. Sync CustomState with current GUI state
                self.state = self._update_state_from_mujoco()
                
                # 2. Compute rewards --> for this I will recreate run the env_step except the action and pipeline step logic
                action = jp.zeros(self.rl_task.sys.nu)
                self.state = self.rl_task._get_target(self.state)
                self.state = self.rl_task._get_obs(self.state, action)
                self.state = self.rl_task._get_reward(self.state, action)
                self.state = self.rl_task._check_termination(self.state)
                
                # 4. Stream to PlotJuggler for visualization
                metrics_to_log = self._prepare_metrics_for_logging(self.state)
                self.pj.log(metrics_to_log)
                
                # 5. Step the simulation and sync viewer
                mujoco.mj_step(self.mj_model, self.mj_data)
                viewer.sync()


def main():
    """Entry point for reward analyzer tool."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze reward components in real-time')
    parser.add_argument(
        '--task',
        type=str,
        default='landing',
        choices=['hover', 'landing', 'vel'],
        help='Task/environment to analyze (hover, landing, vel)'
    )
    parser.add_argument(
        '--layout',
        type=str,
        default=None,
        help='Path to PlotJuggler layout file (auto-launches if exists)'
    )
    args = parser.parse_args()

    # Build config path automatically from task
    config_path = f'./src/arcdrone/controller/rl/cfg/task/{args.task}.yaml'
    analyzer = RewardAnalyzer(config_path, task_name=args.task, layout_file=args.layout)
    analyzer.run()


if __name__ == "__main__":
    main()