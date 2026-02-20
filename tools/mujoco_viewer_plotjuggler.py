"""
MuJoCo viewer with PlotJuggler integration for real-time data analysis.

This tool allows you to manually control the robot in the MuJoCo GUI while
streaming sensor data, forces, and other variables to PlotJuggler for analysis.
"""

import os
import time
import mujoco
import mujoco.viewer
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from arcdrone.utils.plotjuggler import PlotJugglerLogger


# ========== Configuration ==========
MODEL_PATH = "/home/jrmch12/Documents/code/260110_arcdrone_brax_envs/arcdrone/assets/skydio_x2/scene.xml"
N_STEPS = 5  # Number of physics steps per control step
LAYOUT_FILE = None  # Optional: "plotjuggler_layout/viewer_layout.xml"
PORT = 9872  # Port for PlotJuggler ZMQ JSON connection
# ===================================


class MuJoCoViewerWithPlotJuggler:
    """MuJoCo viewer with real-time data streaming to PlotJuggler."""
    
    def __init__(self, model_path: str, n_steps: int = 5, layout_file: str = None):
        """
        Initialize viewer with PlotJuggler logging.
        
        Args:
            model_path: Path to MuJoCo XML model
            n_steps: Number of physics steps per control step
            layout_file: Optional path to PlotJuggler layout file
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.n_steps = n_steps
        
        # Initialize PlotJuggler logger
        if layout_file and os.path.exists(layout_file):
            self.pj = PlotJugglerLogger(port= PORT, layout_file=layout_file)
        else:
            self.pj = PlotJugglerLogger(port= PORT)
        
        print("✓ MuJoCo Viewer with PlotJuggler initialized")
        print(f"✓ Model: {model_path}")
        print(f"✓ Actuators: {self.model.nu}")
        print(f"✓ Sensors: {self.model.nsensor}")
        print(f"✓ Tendons: {self.model.ntendon}")
        print("✓ Manipulate the robot in the GUI to see data in PlotJuggler")
    
    def _collect_data(self) -> dict:
        """
        Collect data from MuJoCo simulation for logging.
        
        Returns:
            Dictionary with sensor data, forces, and other variables
        """
        data_dict = {}
        
        # ====== Control signals (ctrl) ======
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name is None:
                actuator_name = f"actuator_{i}"
            data_dict[f"ctrl/{actuator_name}"] = float(self.data.ctrl[i])
        
        # ====== Actuator Forces ======
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name is None:
                actuator_name = f"actuator_{i}"
            data_dict[f"actuator_force/{actuator_name}"] = float(self.data.actuator_force[i])
        
        # ====== qfrc_actuator (generalized actuator forces) ======
        for i in range(self.model.nv):
            dof_name = f"dof_{i}"
            # Try to get joint name
            if i < self.model.njnt:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name:
                    dof_name = joint_name
            data_dict[f"qfrc_actuator/{dof_name}"] = float(self.data.qfrc_actuator[i])
        
        # ====== Joint positions (qpos) ======
        for i in range(self.model.nv):
            joint_name = f"joint_{i}"
            # Try to get joint name
            if i < self.model.njnt:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if name:
                    joint_name = name
            data_dict[f"qpos/{joint_name}"] = float(self.data.qpos[i])
        
        # ====== Sensor data ======
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name is None:
                sensor_name = f"sensor_{i}"
            
            # Get sensor type
            sensor_type = self.model.sensor_type[i]
            sensor_dim = self.model.sensor_dim[i]
            sensor_adr = self.model.sensor_adr[i]
            
            # Read sensor data based on dimension
            if sensor_dim == 1:
                # Scalar sensor
                data_dict[f"sensor/{sensor_name}"] = float(self.data.sensordata[sensor_adr])
            else:
                # Vector sensor (e.g., gyro, accelerometer, quaternion)
                for j in range(sensor_dim):
                    axis_name = ['x', 'y', 'z', 'w'][j] if j < 4 else str(j)
                    data_dict[f"sensor/{sensor_name}/{axis_name}"] = float(self.data.sensordata[sensor_adr + j])
        
        return data_dict
    
    def _setup_camera(self, viewer):
        """Setup default camera view."""
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 128
        viewer.cam.elevation = -42
        viewer.cam.distance = 0.36
        viewer.cam.lookat[:] = [-0.025, -0.019, 0.147]
    
    def run(self):
        """Main loop: run simulation and stream data to PlotJuggler."""
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            # Setup camera
            self._setup_camera(viewer)
            
            start = time.time()
            
            # Main simulation loop
            while viewer.is_running():
                step_start = time.time()
                
                # Step physics n_steps times
                for _ in range(self.n_steps):
                    mujoco.mj_step(self.model, self.data)
                
                # Collect and stream data to PlotJuggler
                data_dict = self._collect_data()
                self.pj.log(data_dict)
                
                # Sync viewer
                viewer.sync()
                
                # Maintain real-time execution
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == '__main__':
    viewer = MuJoCoViewerWithPlotJuggler(
        model_path=MODEL_PATH,
        n_steps=N_STEPS,
        layout_file=LAYOUT_FILE
    )
    viewer.run()
