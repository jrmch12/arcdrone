
"""Evaluation script for trained ARCDrone vision-landing IL student."""

from pathlib import Path
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

from arcdrone.vision_landing_il.task.arcdrone import ARCDroneRL_VisionLanding_StudentTeacher
from arcdrone.vision_landing_il.training import networks as il_networks


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



def _squeeze(tree):
	"""Remove the leading batch-1 axis added by vmap."""
	return jax.tree_util.tree_map(lambda x: x[0], tree)


def evaluate(
	model_path: str = CHECKPOINT_PATH,
	num_episodes: int = 20,
	max_steps: int = 200,
	teacher_checkpoint_path: str = "checkpoints/trained_model.pkl",
):
	"""Evaluate a trained vision-landing IL student policy."""
	initialize_config_dir(
		config_dir=str(CFG_DIR), job_name="il_evaluate", version_base=None
	)
	cfg = compose(config_name="config")
	cfg_env = cfg.env
	cfg_train = cfg.train

	print("=" * 60)
	print("ARCDrone Vision Landing IL Evaluation")
	print("=" * 60)
	print(f"Model path: {model_path}")
	print(f"Episodes: {num_episodes}")
	print("=" * 60)

	# nworld must equal the vmap batch size (1 env at eval time).
	# The training config sets nworld=num_envs; override to 1 here.
	cfg_env = OmegaConf.to_container(cfg_env, resolve=True)
	cfg_env["vision_config"]["nworld"] = 1
	cfg_env["naconmax"] = cfg_env["njmax"]  # naconmax = njmax * nworld = njmax * 1

	env = ARCDroneRL_VisionLanding_StudentTeacher(cfg=cfg_env)
	print("env 'vision_landing_il' instantiated successfully")

	# Vmap reset/step over a batch of 1, exactly as the training wrapper does.
	# This ensures mjx.render sees a single-world context inside each vmap
	# call — so out[0] is 1-D (H*W,) as get_rgb expects.
	vmap_reset = jax.vmap(env.reset)
	vmap_step  = jax.vmap(env.step)
	jit_reset  = jax.jit(vmap_reset)
	jit_step   = jax.jit(vmap_step)

	print("Setting up reset function...")
	rng = jax.random.PRNGKey(0)
	rng, reset_key = jax.random.split(rng)
	keys_1 = jax.random.split(reset_key, 1)  # shape (1,) batch
	state = jit_reset(keys_1)               # all leaves: (1, ...)
	print("✓ Reset function compiled successfully")

	# Build obs_shape from the squeezed single-env observation
	print("Setting up inference function...")
	obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)
	action_size = env._mj_model.nu
	network_factory = functools.partial(
		il_networks.make_il_networks,
		teacher_dec_hidden_layers=cfg_train.teacher_dec_hidden_layers,
		policy_dec_hidden_layers=cfg_train.policy_dec_hidden_layers,
		policy_propio_proj_hidden_layers=cfg_train.policy_propio_proj_hidden_layers,
		action_hidden_layer_sizes=cfg_train.action_hidden_layers,
		value_hidden_layer_sizes=cfg_train.value_hidden_layers,
		cnn_num_filters=cfg_train.cnn_num_filters,
		cnn_kernel_sizes=cfg_train.cnn_kernel_sizes,
		cnn_strides=cfg_train.cnn_strides,
		policy_pixels_key=cfg_train.policy_pixels_key,
		policy_propio_key=cfg_train.policy_propio_key,
		teacher_obs_key=cfg_train.teacher_obs_key,
		value_obs_key=cfg_train.value_obs_key,
	)
	il_net = network_factory(obs_shape, action_size)

	print("Loading trained model...")
	params = model.load_params(model_path)
	# Checkpoint format (new): (propio_norm, (student_enc, action_head)) — self-contained.
	# Checkpoint format (old): (propio_norm, student_enc)                — needs teacher ckpt.
	if isinstance(params[1], tuple) and len(params[1]) == 2:
		action_head_params = params[1][1]
		student_params = (params[0], params[1][0])
	else:
		print(f"Old-format checkpoint — loading action_head from: {teacher_checkpoint_path}")
		teacher_params = model.load_params(teacher_checkpoint_path)
		action_head_params = teacher_params[1][1]  # (norm, (dec, action_head), value)
		student_params = (params[0], params[1])
	make_policy = il_networks.make_student_inference_fn(il_net, action_head_params=action_head_params)
	inference_fn = make_policy(student_params, deterministic=True)
	jit_inference_fn = jax.jit(inference_fn)
	# Warm-up compile on single-env obs
	action, _ = jit_inference_fn(_squeeze(state.obs), rng)
	print("✓ Inference function compiled successfully")

	# Warm-up compile step
	action_batch = jax.tree_util.tree_map(lambda x: x[None], action)  # (1, nu)
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
		# Pass a list so get_data_into uses the batched path, which leaves
		# DATA_NON_VMAP fields (e.g. nacon shape (1,)) un-indexed.
		mjx.get_data_into([physics_data], physics_model, state.data)
		for _ in range(max_steps):
			rng, action_key = jax.random.split(rng)
			# Inference needs unbatched obs; step needs batched action.
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
		description="Evaluate trained ARCDrone vision-landing IL student"
	)
	parser.add_argument(
		"--model_path",
		type=str,
		default=CHECKPOINT_PATH,
		help="Path to trained model checkpoint (trained_model.pkl)",
	)
	parser.add_argument(
		"--teacher_checkpoint_path",
		type=str,
		default="checkpoints/trained_model.pkl",
		help="Teacher checkpoint for loading action_head (only needed for old-format student checkpoints)",
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
		teacher_checkpoint_path=args.teacher_checkpoint_path,
	)


if __name__ == "__main__":
	main()

# how to use:
# python src/arcdrone/vision_landing_il/evaluate.py
# python src/arcdrone/vision_landing_il/evaluate.py --model_path outputs/2026-03-10/12-00-00/trained_model.pkl