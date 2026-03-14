"""Evaluation script for trained ARCDrone privileged-landing PPO controller."""

from pathlib import Path
import functools

import jax
import jax.numpy as jnp
from brax.io import model
from brax.training.acme import running_statistics
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import mujoco

# Custom resolver: ${mul:a,b} → int(a) * int(b)
OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
from mujoco import mjx
import mujoco.viewer

from arcdrone.priviledged_landing_rl.task.arcdrone import ARCDroneRL_Landing
from arcdrone.priviledged_landing_rl.training import networks as ppo_networks


def find_latest_checkpoint(outputs_dir: str = "outputs") -> str:
	"""Find the latest trained_model.pkl file in the outputs directory."""
	outputs_path = Path(outputs_dir)
	if not outputs_path.exists():
		raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

	pkl_files = list(outputs_path.rglob("trained_model.pkl"))
	if not pkl_files:
		raise FileNotFoundError(f"No trained_model.pkl files found in {outputs_dir}")

	latest_pkl = max(pkl_files, key=lambda path: path.stat().st_mtime)
	return str(latest_pkl)


# ========== Configuration ==========
CFG_DIR = Path(__file__).resolve().parent / "cfg"
_project_root = Path(__file__).resolve().parent.parent.parent.parent
MUJOCO_PATH = str(_project_root / "assets" / "skydio_x2" / "scene.xml")
CHECKPOINT_PATH = find_latest_checkpoint(str(_project_root / "outputs"))
print(f"Latest checkpoint found: {CHECKPOINT_PATH}")


def evaluate(
	model_path: str = CHECKPOINT_PATH,
	num_episodes: int = 20,
	max_steps: int = 200,
):
	"""Evaluate a trained privileged-landing PPO controller."""

	initialize_config_dir(
		config_dir=str(CFG_DIR), job_name="ppo_evaluate", version_base=None
	)
	cfg = compose(config_name="config")
	cfg_env = cfg.env
	cfg_train = cfg.train

	print("=" * 60)
	print("ARCDrone Privileged Landing PPO Evaluation")
	print("=" * 60)
	print(f"Model path: {model_path}")
	print(f"Episodes: {num_episodes}")
	print("=" * 60)

	env = ARCDroneRL_Landing(cfg=cfg_env)
	print("env 'landing' instantiated successfully")

	print("Setting up reset function...")
	rng = jax.random.PRNGKey(0)
	jit_reset = jax.jit(env.reset)
	state = jit_reset(rng)
	print("✓ Reset function compiled successfully")

	print("Setting up inference function...")
	observation_size = {
		cfg_train.policy_obs_key: state.obs[cfg_train.policy_obs_key].shape[0],
		cfg_train.value_obs_key: state.obs[cfg_train.value_obs_key].shape[0],
	}
	action_size = env._mj_model.nu

	network_factory = functools.partial(
		ppo_networks.make_ppo_networks,
		policy_dec_hidden_layers=cfg_train.policy_dec_hidden_layers,
		action_hidden_layer_sizes=cfg_train.action_hidden_layers,
		value_hidden_layer_sizes=cfg_train.value_hidden_layers,
		policy_obs_key=cfg_train.policy_obs_key,
		value_obs_key=cfg_train.value_obs_key,
	)
	ppo_network = network_factory(
		observation_size,
		action_size,
		preprocess_observations_fn=running_statistics.normalize,
	)

	print("Loading trained model...")
	params = model.load_params(model_path)

	make_policy = ppo_networks.make_inference_fn(ppo_network)
	inference_fn = make_policy(params)


	# # Optionally wrap inference function to add Gaussian action noise for testing.
	# def _make_noisy_policy(base_policy, sigma: float = 0.1):
	# 	if sigma is None or sigma <= 0.0:
	# 		return base_policy

	# 	def policy_with_noise(observations, key_sample):
	# 		action, extras = base_policy(observations, key_sample)
	# 		key_sample, key_noise = jax.random.split(key_sample)
	# 		noise = sigma * jax.random.normal(key_noise, shape=action.shape)
	# 		noisy_action = jnp.clip(action + noise, -1.0, 1.0)
	# 		extras = {**extras, "teacher_noise": noise}
	# 		return noisy_action, extras

	# 	return policy_with_noise

	# # Hardcoded small noise for quick testing — change or disable as needed.
	# inference_fn = _make_noisy_policy(inference_fn, sigma=0.4)


	jit_inference_fn = jax.jit(inference_fn)
	action, _ = jit_inference_fn(state.obs, rng)
	print("✓ Inference function compiled successfully")

	print("Setting up step function...")
	jit_step = jax.jit(env.step)
	state1 = jit_step(state, action)
	action2, _ = jit_inference_fn(state1.obs, rng)
	_ = jit_step(state1, action2)
	print("✓ Step function compiled successfully")

	physics_model = mujoco.MjModel.from_xml_path(MUJOCO_PATH)
	physics_data = mujoco.MjData(physics_model)
	viewer = mujoco.viewer.launch_passive(physics_model, physics_data)
	print("✓ Viewer launched")

	for episode in range(num_episodes):
		print(f"Episode {episode + 1}/{num_episodes}")
		print("-" * 40)

		rng, reset_key = jax.random.split(rng)
		state = jit_reset(reset_key)
		mjx.get_data_into(physics_data, physics_model, state.data)

		for _ in range(max_steps):
			rng, action_key = jax.random.split(rng)
			action, _ = jit_inference_fn(state.obs, action_key)
			state = jit_step(state, action)
			mjx.get_data_into(physics_data, physics_model, state.data)
			viewer.sync()
			if state.done:
				break

	viewer.close()


def main():
	import argparse

	parser = argparse.ArgumentParser(
		description="Evaluate trained ARCDrone privileged-landing PPO controller"
	)
	parser.add_argument(
		"--model_path",
		type=str,
		default=CHECKPOINT_PATH,
		help="Path to trained model checkpoint (trained_model.pkl)",
	)
	parser.add_argument(
		"--episodes",
		type=int,
		default=50,
		help="Number of episodes to evaluate",
	)
	parser.add_argument(
		"--steps",
		type=int,
		default=200,
		help="Maximum steps per episode",
	)
	args = parser.parse_args()

	evaluate(
		model_path=args.model_path,
		num_episodes=args.episodes,
		max_steps=args.steps,
	)


if __name__ == "__main__":
	main()

# how to use:
# python src/arcdrone/priviledged_landing_rl/evaluate.py
# python src/arcdrone/priviledged_landing_rl/evaluate.py --model_path outputs/2026-03-10/12-00-00/trained_model.pkl