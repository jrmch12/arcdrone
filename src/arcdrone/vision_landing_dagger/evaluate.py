"""Evaluation script for trained ARCDrone vision-landing DAgger student."""

from pathlib import Path
from typing import Optional
import functools

import jax
from brax.io import model
from brax.training.acme import running_statistics
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import mujoco

OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
from mujoco import mjx
import mujoco.viewer

from arcdrone.vision_landing_il.task.arcdrone import ARCDroneRL_VisionLanding_StudentTeacher
from arcdrone.vision_landing_il.training import networks as il_networks
from arcdrone.vision_landing_dagger.training import networks as dagger_networks


def find_latest_checkpoint(outputs_dir: str = "outputs") -> str:
    """Find the latest trained_model.pkl file in the outputs directory."""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    pkl_files = list(outputs_path.rglob("trained_model.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No trained_model.pkl files found in {outputs_dir}")
    return str(max(pkl_files, key=lambda p: p.stat().st_mtime))


# ========== Configuration ==========
CFG_DIR = Path(__file__).resolve().parent / "cfg"
_project_root = Path(__file__).resolve().parent.parent.parent.parent
MUJOCO_PATH = str(_project_root / "assets" / "skydio_x2" / "scene.xml")


def _squeeze(tree):
    """Remove the leading batch-1 axis added by vmap."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def evaluate(
    model_path: Optional[str] = None,
    policy: str = "student",
    num_episodes: int = 20,
    max_steps: int = 200,
    teacher_checkpoint_path: Optional[str] = None,
):
    """Evaluate a trained DAgger student (or frozen teacher) policy."""
    initialize_config_dir(
        config_dir=str(CFG_DIR), job_name="dagger_evaluate", version_base=None
    )
    cfg = compose(config_name="config")
    cfg_env = cfg.env
    cfg_train = cfg.train

    policy = policy.lower()
    if policy not in {"student", "teacher"}:
        raise ValueError("policy must be either 'student' or 'teacher'")

    if policy == "student":
        if model_path is None:
            model_path = find_latest_checkpoint(str(_project_root / "outputs"))
        print("=" * 60)
        print("ARCDrone Vision Landing DAgger Evaluation")
        print("=" * 60)
        print(f"Model path: {model_path}")
    else:
        if not teacher_checkpoint_path:
            raise ValueError("policy 'teacher' requires --teacher_checkpoint_path")
        print("=" * 60)
        print("ARCDrone Vision Landing DAgger Evaluation (Teacher)")
        print("=" * 60)
        print(f"Teacher checkpoint: {teacher_checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    # nworld=1 for single-env evaluation
    cfg_env = OmegaConf.to_container(cfg_env, resolve=True)
    cfg_env["vision_config"]["nworld"] = 1
    cfg_env["naconmax"] = cfg_env["njmax"]

    env = ARCDroneRL_VisionLanding_StudentTeacher(cfg=cfg_env)
    print("env instantiated successfully")

    vmap_reset = jax.vmap(env.reset)
    vmap_step  = jax.vmap(env.step)
    jit_reset  = jax.jit(vmap_reset)
    jit_step   = jax.jit(vmap_step)

    print("Setting up reset function...")
    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    keys_1 = jax.random.split(reset_key, 1)
    state = jit_reset(keys_1)
    print("✓ Reset function compiled successfully")

    print("Setting up inference function...")
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)
    action_size = env._mj_model.nu
    network_factory = functools.partial(
        il_networks.make_il_networks,
        preprocess_observations_fn=(
            running_statistics.normalize
            if cfg_train.normalize_observations
            else (lambda x, y: x)
        ),
        teacher_dec_hidden_layers=cfg_train.teacher_dec_hidden_layers,
        policy_dec_hidden_layers=cfg_train.policy_dec_hidden_layers,
        policy_proprio_proj_hidden_layers=cfg_train.policy_proprio_proj_hidden_layers,
        action_hidden_layer_sizes=cfg_train.action_hidden_layers,
        value_hidden_layer_sizes=cfg_train.value_hidden_layers,
        cnn_num_filters=cfg_train.cnn_num_filters,
        cnn_kernel_sizes=cfg_train.cnn_kernel_sizes,
        cnn_strides=cfg_train.cnn_strides,
        policy_pixels_key=cfg_train.policy_pixels_key,
        policy_pixels_key_1=cfg_train.policy_pixels_key_1,
        policy_proprio_key=cfg_train.policy_proprio_key,
        teacher_obs_key=cfg_train.teacher_obs_key,
        value_obs_key=cfg_train.value_obs_key,
    )
    il_net = network_factory(obs_shape, action_size)

    if policy == "student":
        print("Loading trained student checkpoint...")
        params = model.load_params(model_path)

        # Checkpoint layout: (proprio_norm, (student_enc, student_action_head))
        proprio_norm = params[0]
        student_enc  = params[1][0]
        action_head  = params[1][1]

        # Guard: if the checkpoint normalizer is dict-shaped (from an older
        # run) or has a mismatched size (e.g. before linvel was added to
        # proprio), replace it with a fresh identity normalizer so the
        # network runs without error.
        try:
            # Flat RunningStatisticsState has mean.shape; dict-shaped does not
            _ = proprio_norm.mean.shape
        except AttributeError:
            print("  ⚠ Checkpoint normalizer is dict-shaped; reinitialising "
                  "a fresh identity normalizer for proprio.")
            from brax.training.acme import specs as _specs
            proprio_size = obs_shape["proprio_obs"][-1]
            proprio_norm = running_statistics.init_state(
                _specs.Array((proprio_size,), jax.numpy.dtype("float32"))
            )

        # DAgger make_student_inference_fn expects a 3-tuple:
        #   (proprio_norm, student_enc, student_action_head)
        make_policy = dagger_networks.make_student_inference_fn(il_net)
        inference_fn = make_policy((proprio_norm, student_enc, action_head), deterministic=True)
    else:
        print("Loading frozen teacher checkpoint...")
        teacher_ckpt = model.load_params(teacher_checkpoint_path)
        try:
            teacher_norm = il_networks._select_normalizer_by_path(
                teacher_ckpt[0], cfg_train.teacher_obs_key
            )
        except KeyError:
            policy_obs_key = getattr(cfg_train, "policy_obs_key", "policy_obs")
            teacher_norm = il_networks._select_normalizer_by_path(
                teacher_ckpt[0], policy_obs_key
            )
        inference_fn = il_networks.make_frozen_teacher_policy(
            il_net,
            teacher_norm_params=teacher_norm,
            teacher_policy_params=teacher_ckpt[1],
            deterministic=True,
        )

    jit_inference_fn = jax.jit(inference_fn)
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
        for _ in range(max_steps):
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
        description="Evaluate trained ARCDrone vision-landing DAgger policy"
    )
    parser.add_argument(
        "--policy",
        choices=["student", "teacher"],
        default="student",
        help="Policy to evaluate",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained student checkpoint (trained_model.pkl); defaults to latest.",
    )
    parser.add_argument(
        "--teacher_checkpoint_path",
        type=str,
        default=None,
        help="Path to the frozen teacher checkpoint (required for --policy teacher).",
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
        policy=args.policy,
        num_episodes=args.episodes,
        max_steps=args.steps,
        teacher_checkpoint_path=args.teacher_checkpoint_path,
    )


if __name__ == "__main__":
    main()
