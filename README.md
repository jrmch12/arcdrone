1. ARCdrone

Pixel to torque policy for autonomous landing for the Aerospace Research Lab (ARC) from the Universidad de Piura (UDEP).

Accelerated training, physics and rendering is achieved with the library mujoco playground.

keywords: jax, mujoco playground, warp

*videos of the main results turn into gifs*

2. Methodology

The project adresses the problem of autonomous landing through tecaher student imitation learning. The teacher is trained with a priviledged state through reinforcement learning. The student then learns from it with imitation learning. For this, two methods are explore: (1) the popular dagger approach and (2) an student informed teacher training approach from https://github.com/uzh-rpg/sitt. The later presents a promising approach for the added difficulty of training a tecaher that its actually feasible to be imitated with the poorer student observations.

- observations
- action space
- network
- Loss function

3. Installation

4. Scripts

- the teacher training script and the resulted checkpoint
- the student training script and resulted checkpoint

Also find a variety of usefull scripts
- script to render videos of visual observations
- script to plot the rewards of the drone env to check their magnitudes
- mujoco viewer with plots 

4. Future

- To close the sim to real gap, the pre trained pixel policy will be further finetune on outdoor gaussian splatting scenes with data augmentation.
- SITT approach 
- Real implementation after purschasing the actual drone


5. References

- x2studio drone

