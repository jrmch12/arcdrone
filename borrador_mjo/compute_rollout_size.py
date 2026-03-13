
batch_size = 32
num_mini_batches = 4
unroll_length = 8
num_cameras = 3
num_channels = 3
obs_history = 5
resolution = (128, 128)
dtype_size = 4  # float32


rollout_size_bytes = (
    batch_size * num_mini_batches *
    unroll_length * num_cameras * num_channels *
    obs_history * resolution[0] * resolution[1] * dtype_size
)
rollout_size_gb = rollout_size_bytes / (1024 ** 3)
print(f"Estimated rollout size: {rollout_size_gb:.2f} GB")
