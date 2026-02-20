import jax
import jax.numpy as jnp

# 1. Check if GPU is detected
print(f"Default Backend: {jax.default_backend()}")
print(f"Devices found: {jax.devices()}")

device = jax.devices()[0]
if device.platform == 'gpu':
    print(f"GPU: {device.device_kind}")
else:
    print("CPU mode")

# 2. Run a minimal computation (Matrix Multiplication) on the device
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x).block_until_ready()

print("Computation successful!")