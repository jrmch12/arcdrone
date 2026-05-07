1. ARCdrone

Pixel to torque policy for autonomous landing for the Aerospace Research Lab (ARC) from the Universidad de Piura (UDEP).

Accelerated training, physics and rendering is achieved thorugh the mujoco playground library.

keywords: jax, mujoco playground, warp


TODO: please put here this two gifs one next to the other docs/media/clip1_16-41.gif and docs/media/clip2_43-79.gif

2. Methodology

The project adresses the problem of autonomous landing through tecaher student imitation learning. The teacher is trained with a priviledged state through reinforcement learning. The student then learns from the teacher with imitation learning. For this we use the popular dagger approach. 

Adittionally, code for an student informed teacher training (SITT) approach from https://github.com/uzh-rpg/sitt is made available. The later presents a promising approach against the added complexity of collecting teacher examples that are actually possible to be imitated by the student limited observations. The SITT approach is still on development.

TODO: please fill the next items, by understanding first whats going on
- observations
- action space
- network:
    We found that the vision policy trains easily if the velocity where given as a signal in the propio. In practice nonetheless this is not desired. Therefore we estimate the velocity with an MLP an give this signal to the policy. Besides this, the rest of the network is as it can be expected: CNN -> MLP.
     TODO: this can be easily explain with this mermaid, please add it here docs/diagrams/aux_vel_student.md
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
- los de sitt

