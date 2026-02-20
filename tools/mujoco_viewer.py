import os
import time
import mujoco.viewer


model_path = "assets/skydio_x2/mocap/scene_mocap.xml"
# model_path = "assets/skydio_x2/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
n_steps = 5

# viewer shows frame of environment every n_steps
with mujoco.viewer.launch_passive(model, data) as viewer:

        # --------- Set camera parameters here ---------
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # Use free camera
    viewer.cam.azimuth = 128      # degrees
    viewer.cam.elevation = -42    # degrees
    viewer.cam.distance = 0.36     # zoom (experiment with value)
    viewer.cam.lookat[:] = [-0.025, -0.019, 0.147]  # center of view (x, y, z)
    # ---------------------------------------------

    start = time.time()

    while True:
        step_start = time.time()
        
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
            
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            

        # # ****************** To update Default Camera View! ******************
        # # After the viewer window is closed, print the camera info!
        # print("\n--- FINAL CAMERA PARAMETERS ---")
        # print(f"Camera Type: '{viewer.cam.type}'") # You want this to be 'free'
        # print(f"Azimuth: {viewer.cam.azimuth}")
        # print(f"Elevation: {viewer.cam.elevation}")
        # print(f"Distance: {viewer.cam.distance}")
        # print(f"Lookat: {viewer.cam.lookat}")
        # # ***************************************************