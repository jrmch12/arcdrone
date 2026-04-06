"""Networks for DAgger vision-landing training.

Identical to vision_landing_il/training/networks.py EXCEPT:

``make_student_inference_fn`` — the action head is NO LONGER a frozen
constant closed over from the teacher checkpoint.  Instead it is a fully
trainable parameter that is passed in as part of ``params`` at call time:

    params = (proprio_norm, student_enc_params, student_action_head_params)

This lets the DAgger loop train the action head from scratch alongside the
student encoder, rather than inheriting and freezing the teacher's head.

Everything else (ILNetworks dataclass, make_il_networks factory,
make_frozen_teacher_policy) is unchanged from the IL module and is
re-exported from here so DAgger code only needs to import from this file.
"""

from typing import Any, Callable, Mapping, Sequence, Tuple

from brax.training import distribution
from brax.training import types
from brax.training.networks import normalizer_select
from brax.training.types import PRNGKey
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from arcdrone.common.networks import CNN, MLP, _get_obs_size

# Re-export everything from IL networks that DAgger does not override
from arcdrone.vision_landing_il.training.networks import (
    FeedForwardNetwork,
    ILNetworks,
    PolicyVisionProprioEncoder,
    PolicyProprioEncoder,
    make_il_networks,
    make_frozen_teacher_policy,
    _split_path,
    _get_by_path,
    _select_normalizer_by_path,
    _shape_last_dim,
)


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]


# ---------------------------------------------------------------------------
# DAgger-specific inference helper
# ---------------------------------------------------------------------------

def make_student_inference_fn(il_networks: ILNetworks):
    """Student (vision) policy factory for DAgger — trainable action head.

    Unlike the IL version, the action head is NOT closed over as a constant.
    It is part of the params tuple so the training loop can update it.

    Expected params at call time (from _pack_student_params):
        params[0] = proprio_norm              (RunningStatisticsState)
        params[1] = student_enc_params        (trainable)
        params[2] = student_action_head_params (trainable)

    Checkpoint layout (saved by train.py):
        (proprio_norm, (student_enc_params, student_action_head_params))
    """

    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        proprio_norm, student_enc, student_action_head = params

        def policy(observations: types.Observation, key_sample: PRNGKey):
            logits = il_networks.student_network.apply(
                proprio_norm, (student_enc, student_action_head), observations
            )
            if deterministic:
                return il_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = (
                il_networks.parametric_action_distribution
                .sample_no_postprocessing(logits, key_sample)
            )
            log_prob = il_networks.parametric_action_distribution.log_prob(
                logits, raw_actions
            )
            postprocessed = il_networks.parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "distribution_params": logits,
            }

        return policy

    return make_policy
