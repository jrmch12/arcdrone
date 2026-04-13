
"""Evaluation script for ARCDrone vision-landing SITT policies (mounted camera)."""

from copy import deepcopy
from pathlib import Path
from collections.abc import Mapping
from typing import Optional
import functools

import jax
from brax.io import model
from brax.training.acme import running_statistics
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import mujoco

# Custom resolver: ${mul:a,b} → int(a) * int(b)
OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
from mujoco import mjx
import mujoco.viewer

from arcdrone.New_attempt_2.task.arcdrone import ARCDroneVisionLandingIL
from arcdrone.vision_landing_sitt.training import networks as sitt_networks


def find_latest_checkpoint(outputs_dir: str = "outputs", filename: str = "student_model.pkl") -> str:
	"""Find the latest checkpoint file matching ``filename`` under ``outputs_dir``."""
	outputs_path = Path(outputs_dir)
	if not outputs_path.exists():
		raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

	pkl_files = list(outputs_path.rglob(filename))
	if not pkl_files:
		raise FileNotFoundError(f"No {filename} files found in {outputs_dir}")

	latest_pkl = max(pkl_files, key=lambda path: path.stat().st_mtime)
	return str(latest_pkl)


# ========== Configuration ==========
CFG_DIR = Path(__file__).resolve().parent / "cfg"
_project_root = Path(__file__).resolve().parent.parent.parent.parent
MUJOCO_PATH = str(_project_root / "assets" / "skydio_x2" / "scene_mounted_cam.xml")
OUTPUTS_DIR = str(_project_root / "outputs")


def _squeeze(tree):
	"""Remove the leading batch-1 axis added by vmap."""
	return jax.tree_util.tree_map(lambda x: x[0], tree)


def evaluate(
	model_path: Optional[str] = None,
	policy: str = "student",
	outputs_dir: str = OUTPUTS_DIR,
	num_episodes: int = 20,
	max_steps: int = 200,
	play_env: str = "student",
):
	"""Evaluate a trained vision-landing SITT student or teacher policy."""
	initialize_config_dir(
		config_dir=str(CFG_DIR), job_name="sitt_evaluate", version_base=None
	)
	cfg = compose(config_name="config")
	cfg_env = cfg.env
	cfg_train = cfg.train

	policy = policy.lower()
	if policy not in {"student", "teacher"}:
		raise ValueError("policy must be either 'student' or 'teacher'")
	play_env = play_env.lower()
	if play_env not in {"student", "teacher"}:
		raise ValueError("play_env must be either 'student' or 'teacher'")

	print("=" * 60)
	print("ARCDrone SITT Mounted Camera Evaluation")
	print("=" * 60)
	print(f"Policy: {policy}")
	print(f"Playing env: {play_env}")
	if model_path is None:
		filename = "student_model.pkl" if policy == "student" else "teacher_model.pkl"
		model_path = find_latest_checkpoint(outputs_dir, filename=filename)
	print(f"Model path: {model_path}")
	print(f"Episodes: {num_episodes}")
	print("=" * 60)

	# ───── Build envs ─────
	cfg_env = OmegaConf.to_container(cfg_env, resolve=True)
	cfg_env["vision_config"]["nworld"] = 1
	cfg_env["naconmax"] = cfg_env["njmax"]  # nworld = 1
	cfg_env["naccdmax"] = cfg_env["njmax"] #TODO

	# Teacher env (no vision)
	teacher_cfg = deepcopy(cfg_env)
	teacher_cfg["enable_vision_obs"] = False
	teacher_cfg["vision"] = False
	teacher_env = ARCDroneVisionLandingIL(cfg=teacher_cfg)

	# Student env (with vision)
	student_cfg = deepcopy(cfg_env)
	student_env = ARCDroneVisionLandingIL(cfg=student_cfg)
	print("Teacher env and student env instantiated successfully")

	play_envs = {"teacher": teacher_env, "student": student_env}
	active_env = play_envs[play_env]

	vmap_reset = jax.vmap(active_env.reset)
	vmap_step  = jax.vmap(active_env.step)
	jit_reset  = jax.jit(vmap_reset)
	jit_step   = jax.jit(vmap_step)

	# Get obs shapes for network init
	print("Setting up reset function...")
	rng = jax.random.PRNGKey(0)
	rng, teacher_key = jax.random.split(rng)
	teacher_state = teacher_env.reset(teacher_key)
	teacher_obs_shape = jax.tree_util.tree_map(lambda x: x.shape, teacher_state.obs)

	rng, reset_key = jax.random.split(rng)
	keys_1 = jax.random.split(reset_key, 1)
	state = jit_reset(keys_1)
	print("✓ Reset function compiled successfully")

	# Student obs shapes (from student env, not active env)
	student_vmap_reset = jax.vmap(student_env.reset)
	rng, student_shape_key = jax.random.split(rng)
	student_shape_keys = jax.random.split(student_shape_key, 1)
	student_shape_state = jax.jit(student_vmap_reset)(student_shape_keys)
	student_obs_shape = jax.tree_util.tree_map(
		lambda x: x.shape, student_shape_state.obs
	)

	print("Setting up inference function...")
	preprocess_obs_fn = (
		running_statistics.normalize
		if cfg_train.normalize_observations
		else (lambda x, y: x)
	)

	network_factory = functools.partial(
		sitt_networks.make_sitt_networks,
		preprocess_observations_fn=preprocess_obs_fn,
		teacher_dec_hidden_layers=cfg_train.teacher_dec_hidden_layers,
		policy_dec_hidden_layers=cfg_train.policy_dec_hidden_layers,
		policy_proprio_proj_hidden_layers=cfg_train.policy_proprio_proj_hidden_layers,
		proxy_hidden_layers=cfg_train.proxy_hidden_layers,
		action_hidden_layer_sizes=cfg_train.action_hidden_layers,
		value_hidden_layer_sizes=cfg_train.value_hidden_layers,
		cnn_num_filters=cfg_train.cnn_num_filters,
		cnn_kernel_sizes=cfg_train.cnn_kernel_sizes,
		cnn_strides=cfg_train.cnn_strides,
		policy_obs_key=cfg_train.policy_obs_key,
		value_obs_key=cfg_train.value_obs_key,
		teacher_obs_key=cfg_train.teacher_obs_key,
		policy_pixels_key=cfg_train.policy_pixels_key,
		policy_proprio_key=cfg_train.policy_proprio_key,
	)

	sitt_net = network_factory(
		teacher_obs_shape,
		teacher_env.action_size,
		student_observation_size=student_obs_shape,
	)

	print("Loading trained model...")
	if policy == "student":
		student_ckpt = model.load_params(model_path)
		proprio_norm, (student_enc_params, action_head_params) = student_ckpt
		make_policy = sitt_networks.make_student_inference_fn(
			sitt_net, action_head_params=action_head_params
		)
		inference_fn = make_policy(
			(proprio_norm, student_enc_params),
			deterministic=True,
		)
	else:
		teacher_ckpt = model.load_params(model_path)
		teacher_norm_whole, teacher_policy_params, _ = teacher_ckpt
		policy_obs_key = getattr(cfg_train, "policy_obs_key", "teacher_obs")
		teacher_norm = sitt_networks._select_normalizer_by_path(
			teacher_norm_whole, policy_obs_key
		)
		inference_fn = sitt_networks.make_frozen_teacher_policy(
			sitt_net,
			teacher_norm_params=teacher_norm,
			teacher_policy_params=teacher_policy_params,
			deterministic=True,
		)
	jit_inference_fn = jax.jit(inference_fn)
	# Warm-up compile
	action, _ = jit_inference_fn(_squeeze(state.obs), rng)
	print("✓ Inference function compiled successfully")

	action_batch = jax.tree_util.tree_map(lambda x: x[None], action)
	state = jit_step(state, action_batch)
	print("✓ Step function compiled successfully")

	physics_model = mujoco.MjModel.from_xml_path(MUJOCO_PATH)
	physics_data = mujoco.MjData(physics_model)
	viewer = mujoco.viewer.launch_passive(physics_model, physics_data)
	print("✓ Viewer launched")

	for episode in range(num_episodes):
		print(f"Episode {episode + 1}/{num_episodes}")
		print("-" * 40)
		rng, reset_key = jax.random.split(rng)
		keys_1 = jax.random.split(reset_key, 1)
		state = jit_reset(keys_1)
		mjx.get_data_into([physics_data], physics_model, state.data)
		for step in range(max_steps):
			rng, action_key = jax.random.split(rng)
			action, _ = jit_inference_fn(_squeeze(state.obs), action_key)
			action_batch = jax.tree_util.tree_map(lambda x: x[None], action)
			state = jit_step(state, action_batch)
			mjx.get_data_into([physics_data], physics_model, state.data)
			viewer.sync()
			if bool(_squeeze(state.done)):
				break
	viewer.close()


def main():
	import argparse

	parser = argparse.ArgumentParser(
		description="Evaluate trained ARCDrone vision-landing SITT policies (mounted camera)"
	)
	parser.add_argument(
		"--policy",
		choices=["student", "teacher"],
		default="student",
		help="Policy to evaluate (student or teacher).",
	)
	parser.add_argument(
		"--model_path",
		type=str,
		default=None,
		help="Path to policy checkpoint. Defaults to latest under --outputs_dir.",
	)
	parser.add_argument(
		"--outputs_dir",
		type=str,
		default=OUTPUTS_DIR,
		help="Root outputs directory to search when --model_path is omitted.",
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
		"--play_env",
		choices=["student", "teacher"],
		default="student",
		help="Which environment to step while inference runs.",
	)
	args = parser.parse_args()

	evaluate(
		model_path=args.model_path,
		policy=args.policy,
		outputs_dir=args.outputs_dir,
		num_episodes=args.episodes,
		max_steps=args.steps,
		play_env=args.play_env,
	)


if __name__ == "__main__":
	main()