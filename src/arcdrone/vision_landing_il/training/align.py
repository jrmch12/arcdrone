"""Alignment step for vision IL training.

The alignment loss drives the student (vision) encoder to produce the same
feature representations — and therefore the same actions — as the frozen
teacher (privileged-state) decoder.

Only ``student_enc_params`` is trainable.  All teacher constants
(``teacher_dec_params``, ``action_head_params``, ``teacher_norm``) are
pre-bound via ``functools.partial`` in the caller so they never appear
inside the JAX training state.

Minibatching mirrors PPO's SGD step exactly:
  - data in:        (batch_size * num_minibatches, unroll_length, ...)
  - per minibatch:  (batch_size * unroll_length, ...)   [T flattened into N]
  - outer scan:     align_updates_per_trigger passes over all minibatches
  - inner scan:     num_minibatches gradient steps per pass
  CNN sees batch_size * unroll_length images per gradient step.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax


@partial(
    jax.jit,
    static_argnames=("il_network", "optimizer", "align_updates_per_trigger", "num_minibatches"),
)
def align(
    student_enc_params,
    opt_state: optax.OptState,
    teacher_obs,
    student_obs,
    proprio_norm,
    *,
    teacher_dec_params,
    action_head_params,
    teacher_norm,
    il_network,
    optimizer,
    align_updates_per_trigger: int,
    num_minibatches: int,
):
    """Run minibatched alignment.

    Data is split into ``num_minibatches`` minibatches.  This whole pass is
    repeated ``align_updates_per_trigger`` times (≈ ``num_updates_per_batch``
    in PPO).  Total gradient steps = align_updates_per_trigger * num_minibatches.

    Args:
        student_enc_params: trainable student encoder params.
        opt_state: optax state for ``student_enc_params``.
        teacher_obs: dict of obs, shape ``(batch_size * num_minibatches, T, ...)``.
        student_obs: same shape as ``teacher_obs``.
        proprio_norm: live ``RunningStatisticsState`` for proprio normalisation.
        teacher_dec_params: frozen teacher decoder params (pre-bound via partial).
        action_head_params: frozen action head params (pre-bound via partial).
        teacher_norm: frozen teacher normaliser sub-state (pre-bound via partial).
        il_network: ``ILNetworks`` instance (static).
        optimizer: optax optimiser.
        align_updates_per_trigger: outer loop repetitions (like num_updates_per_batch).
        num_minibatches: number of gradient steps per outer repetition.

    Returns:
        ``(new_student_enc_params, new_opt_state, mean_align_loss)``
    """

    # ── Pre-split data into minibatches ────────────────────────────────────
    # Input leaf shape:  (batch_size * num_minibatches, unroll_length, ...)
    # After reshape:     (num_minibatches, batch_size * unroll_length, ...)
    # The CNN therefore sees batch_size * unroll_length images per step.
    def _to_minibatches(x):
        # x: (B*M, T, ...) → (M, B*T, ...)
        return jnp.reshape(x, (num_minibatches, -1) + x.shape[2:])

    mb_teacher_obs = jax.tree_util.tree_map(_to_minibatches, teacher_obs)
    mb_student_obs = jax.tree_util.tree_map(_to_minibatches, student_obs)

    # ── Per-minibatch gradient step ────────────────────────────────────────
    def minibatch_step(carry, data):
        student_enc, opt_st = carry
        t_obs, s_obs = data

        def loss_fn(student_enc):
            teacher_feat = il_network.teacher_decoder.apply(
                teacher_norm, teacher_dec_params, t_obs
            )
            student_feat = il_network.student_encoder.apply(
                proprio_norm, student_enc, s_obs
            )
            embed_loss = jnp.mean(
                jnp.abs(student_feat - jax.lax.stop_gradient(teacher_feat))
            )
            teacher_logits = il_network.action_head.apply(
                None, action_head_params, jax.lax.stop_gradient(teacher_feat)
            )
            student_logits = il_network.action_head.apply(
                None, action_head_params, student_feat
            )
            action_loss = jnp.mean(
                jnp.abs(student_logits - jax.lax.stop_gradient(teacher_logits))
            )
            total_loss = embed_loss + action_loss
            return total_loss, (embed_loss, action_loss)

        (loss, (embed_loss, action_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(student_enc)
        updates, opt_st = optimizer.update(grads, opt_st, student_enc)
        student_enc = optax.apply_updates(student_enc, updates)
        # Stack losses into a single array for scan compatibility
        losses = jnp.stack([loss, embed_loss, action_loss])
        return (student_enc, opt_st), losses

    # ── One pass over all minibatches ─────────────────────────────────────
    def sgd_step(carry, _):
        student_enc, opt_st = carry
        (student_enc, opt_st), losses = jax.lax.scan(
            minibatch_step,
            (student_enc, opt_st),
            (mb_teacher_obs, mb_student_obs),
            length=num_minibatches,
        )
        # losses: (num_minibatches, 3) [total, embed, action]
        return (student_enc, opt_st), jnp.mean(losses, axis=0)

    # ── Outer loop: repeat align_updates_per_trigger times ────────────────
    (student_enc_params, opt_state), epoch_losses = jax.lax.scan(
        sgd_step,
        (student_enc_params, opt_state),
        None,
        length=align_updates_per_trigger,
    )
    # epoch_losses: (align_updates_per_trigger, 3)
    mean_total, mean_embed, mean_action = jnp.mean(epoch_losses, axis=0)
    return student_enc_params, opt_state, mean_total, mean_embed, mean_action
