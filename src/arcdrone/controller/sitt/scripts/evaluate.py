"""Evaluation script for trained ARCDrone SITT controller."""

from pathlib import Path
import functools

import jax
from brax.io import model
from brax.training.acme import running_statistics
from hydra import compose, initialize_config_dir
import mujoco
from mujoco import mjx
import mujoco.viewer

from arcdrone import (
	ARCDroneRL_Hover,
	ARCDroneRL_Landing,
	ARCDroneRL_Vel,
	sitt_networks,
)


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
CFG_DIR = Path(__file__).resolve().parent.parent.parent / "cfg"
_project_root = CFG_DIR.parent.parent.parent.parent
MUJOCO_PATH = str(_project_root / "assets" / "skydio_x2" / "scene.xml")
CHECKPOINT_PATH = find_latest_checkpoint(str(_project_root / "outputs"))
print(f"Latest checkpoint found: {CHECKPOINT_PATH}")


def _normalize_task_name(task_name: str) -> str:
	aliases = {
		"velocity": "vel",
	}
	return aliases.get(task_name, task_name)


def evaluate(
	model_path: str = CHECKPOINT_PATH,
	num_episodes: int = 20,
	max_steps: int = 200,
	task_name: str = "hover",
	policy_name: str = "teacher",
):
	"""Evaluate a trained SITT controller with teacher or student policy."""
	task_name = _normalize_task_name(task_name)
	policy_name = policy_name.lower()
	if policy_name not in {"teacher", "student"}:
		raise ValueError("policy_name must be one of: teacher, student")

	initialize_config_dir(
		config_dir=str(CFG_DIR), job_name="sitt_evaluate", version_base=None
	)
	cfg = compose(config_name="config", overrides=[f"task={task_name}"])
	cfg_env = cfg.env
	cfg_train = cfg.train

	print("=" * 60)
	print(f"ARCDrone SITT Evaluation ({policy_name})")
	print("=" * 60)
	print(f"Model path: {model_path}")
	print(f"Task: {task_name}")
	print(f"Episodes: {num_episodes}")
	print("=" * 60)

	env_classes = {
		"hover": ARCDroneRL_Hover,
		"landing": ARCDroneRL_Landing,
		"vel": ARCDroneRL_Vel,
	}
	if task_name not in env_classes:
		raise ValueError(
			f"Unknown task '{task_name}'. Available: {list(env_classes.keys())}"
		)
	env = env_classes[task_name](cfg=cfg_env)
	print(f"env '{task_name}' instantiated successfully")

	print("Setting up reset function...")
	rng = jax.random.PRNGKey(0)
	jit_reset = jax.jit(env.reset)
	state = jit_reset(rng)
	print("✓ Reset function compiled successfully")

	print("Setting up inference function...")
	observation_size = {
		"state": state.obs["state"].shape[0],
		"privileged_state": state.obs["privileged_state"].shape[0],
	}
	action_size = env._mj_model.nu
	network_factory = functools.partial(
		sitt_networks.make_sitt_networks,
		policy_hidden_layer_sizes=cfg_train.policy_hidden_layers,
		action_hidden_layer_sizes=cfg_train.action_hidden_layers,
		value_hidden_layer_sizes=cfg_train.value_hidden_layers,
		policy_obs_key=cfg_train.policy_obs_key,
		value_obs_key=cfg_train.value_obs_key,
		use_sitt=cfg_train.use_sitt,
		student_hidden_layer_sizes=cfg_train.student_hidden_layers,
		proxy_hidden_layer_sizes=cfg_train.proxy_hidden_layers,
	)
	sitt_network = network_factory(
		observation_size,
		action_size,
		preprocess_observations_fn=running_statistics.normalize,
	)

	print("Loading trained model...")
	params = model.load_params(model_path)

	if policy_name == "teacher":
		make_policy = sitt_networks.make_inference_fn(sitt_network)
	else:
		if len(params) < 4 or params[3] is None:
			raise ValueError(
				"Student policy requires SITT checkpoint format "
				"(normalizer, policy, value, student_dec, proxy_dec). "
				"The loaded checkpoint appears to be legacy teacher-only."
			)
		make_policy = sitt_networks.make_student_inference_fn(sitt_network)

	inference_fn = make_policy(params)
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

	parser = argparse.ArgumentParser(description="Evaluate trained ARCDrone SITT controller")
	parser.add_argument(
		"--model_path",
		type=str,
		default=CHECKPOINT_PATH,
		help="Path to trained model",
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
	parser.add_argument(
		"--task",
		type=str,
		default="hover",
		choices=["hover", "landing", "vel", "velocity"],
		help="Task/environment to evaluate",
	)
	parser.add_argument(
		"--policy",
		type=str,
		default="student",
		choices=["teacher", "student"],
		help="Which policy to run",
	)
	args = parser.parse_args()

	evaluate(
		model_path=args.model_path,
		num_episodes=args.episodes,
		max_steps=args.steps,
		task_name=args.task,
		policy_name=args.policy,
	)


if __name__ == "__main__":
	main()

# how to use
# python src/arcdrone/controller/sitt/scripts/evaluate.py --policy=student