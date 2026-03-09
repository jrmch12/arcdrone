"""SITT-style alignment step for vision IL training.

The alignment loss drives the student (vision) encoder to produce the same
feature representations — and therefore same actions — as the frozen teacher
(privileged-state) decoder, via a proxy intermediate.

Params tuple layout (matching train.py)::

    (teacher_dec_params, student_enc_params, proxy_dec_params,
     action_head_params, norm_params)

Only ``student_enc_params`` and ``proxy_dec_params`` receive gradients.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax


@partial(
    jax.jit,
    static_argnames=("il_network", "optimizer", "align_updates_per_trigger"),
)
def align(
    params: Tuple,
    opt_state: optax.OptState,
    teacher_obs,
    student_obs,
    *,
    il_network,
    optimizer,
    align_updates_per_trigger: int,
):
    """Run ``align_updates_per_trigger`` gradient steps on the alignment loss.

    Args:
        params: 5-tuple ``(teacher_dec_params, student_enc_params,
                           proxy_dec_params, action_head_params, norm_params)``.
        opt_state: optax state for ``(student_enc_params, proxy_dec_params)``.
        teacher_obs: observations fed to the frozen teacher decoder.
        student_obs: vision observations fed to the trainable student encoder.
        il_network: ``ILNetworks`` instance (must be hashable / static).
        optimizer: optax optimiser for the alignment step.
        align_updates_per_trigger: number of gradient steps per call.

    Returns:
        ``(new_params, new_opt_state, align_loss)``
    """

    (
        teacher_dec_params,
        student_enc_params,
        proxy_dec_params,
        action_head_params,
        norm_params,
    ) = params

    def loss_fn(student_enc, proxy_dec):
        # ── Features ──────────────────────────────────────────────────────
        teacher_feat = il_network.teacher_decoder.apply(
            norm_params, teacher_dec_params, teacher_obs
        )
        student_feat = il_network.student_encoder.apply(
            norm_params, student_enc, student_obs
        )
        proxy_feat = il_network.proxy_decoder.apply(
            norm_params, proxy_dec, teacher_obs
        )

        # ── Embedding loss ─────────────────────────────────────────────────
        embed_loss = (
            jnp.mean(jnp.abs(student_feat - jax.lax.stop_gradient(teacher_feat)))
            + jnp.mean(jnp.abs(jax.lax.stop_gradient(student_feat) - proxy_feat))
        )

        # ── Action loss ────────────────────────────────────────────────────
        teacher_logits = il_network.action_head.apply(
            None, action_head_params, jax.lax.stop_gradient(teacher_feat)
        )
        student_logits = il_network.action_head.apply(
            None, action_head_params, student_feat
        )
        proxy_logits = il_network.action_head.apply(
            None, action_head_params, proxy_feat
        )

        action_loss = (
            jnp.mean(jnp.abs(student_logits - jax.lax.stop_gradient(teacher_logits)))
            + jnp.mean(jnp.abs(jax.lax.stop_gradient(student_logits) - proxy_logits))
        )

        return embed_loss + action_loss

    def step(carry, _):
        student_enc, proxy_dec, opt_st = carry
        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(
            student_enc, proxy_dec
        )
        updates, opt_st = optimizer.update(grads, opt_st, (student_enc, proxy_dec))
        student_enc, proxy_dec = jax.tree_util.tree_map(
            lambda p, u: p + u, (student_enc, proxy_dec), updates
        )
        return (student_enc, proxy_dec, opt_st), loss

    (student_enc_params, proxy_dec_params, opt_state), losses = jax.lax.scan(
        step,
        (student_enc_params, proxy_dec_params, opt_state),
        None,
        length=align_updates_per_trigger,
    )
    align_loss = jnp.mean(losses)

    new_params = (
        teacher_dec_params,
        student_enc_params,
        proxy_dec_params,
        action_head_params,
        norm_params,
    )
    return new_params, opt_state, align_loss
