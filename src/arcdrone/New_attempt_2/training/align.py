"""Alignment step for DAgger training — trainable action head.

Key difference from vision_landing_il/training/align.py:

    The action head is TRAINABLE in DAgger.  Gradients flow through both
    the student encoder *and* the student action head jointly.

    ``trainable_params = (student_enc_params, student_action_head_params)``

    The frozen teacher action head (``teacher_action_head_params``) is still
    used to generate action *labels* — but it no longer participates in the
    gradient graph.

Loss structure (unchanged):
    embed_loss  = mean |student_feat - stop_grad(teacher_feat)|
    action_loss = mean |student_head(student_feat) -
                        stop_grad(teacher_head(stop_grad(teacher_feat)))|
    total_loss  = embed_loss + action_loss
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax


@partial(
    jax.jit,
    static_argnames=(
        "il_network",
        "optimizer",
        "align_updates_per_trigger",
        "num_minibatches",
        "embed_coef",
        "action_coef",
    ),
)
def align(
    trainable_params,           # (student_enc_params, student_action_head_params)
    opt_state: optax.OptState,
    teacher_obs,
    student_obs,
    proprio_norm,
    *,
    teacher_dec_params,
    teacher_action_head_params,  # frozen — action label source only
    teacher_norm,
    il_network,
    optimizer,
    align_updates_per_trigger: int,
    num_minibatches: int,
    embed_coef: float = 1.0,
    action_coef: float = 1.0,
):
    """Run minibatched alignment, training both encoder and action head.

    Args:
        trainable_params: tuple ``(student_enc_params, student_action_head_params)``.
        opt_state: optax state covering ``trainable_params``.
        teacher_obs: rollout observations, shape ``(B*M, T, ...)``.
        student_obs: same (identical in practice — same rollout).
        proprio_norm: live proprio ``RunningStatisticsState``.
        teacher_dec_params: frozen teacher MLP decoder (pre-bound).
        teacher_action_head_params: frozen teacher action head used only to
            produce action *labels* — never differentiated (pre-bound).
        teacher_norm: frozen teacher normaliser sub-state (pre-bound).
        il_network: ``ILNetworks`` instance (static).
        optimizer: optax optimiser covering the full ``trainable_params`` tree.
        align_updates_per_trigger: outer loop repetitions.
        num_minibatches: gradient steps per outer repetition.

    Returns:
        ``(new_trainable_params, new_opt_state, mean_total, mean_embed, mean_action)``
    """

    def _to_minibatches(x):
        return jnp.reshape(x, (num_minibatches, -1) + x.shape[2:])

    mb_teacher_obs = jax.tree_util.tree_map(_to_minibatches, teacher_obs)
    mb_student_obs = jax.tree_util.tree_map(_to_minibatches, student_obs)

    def minibatch_step(carry, data):
        trainable, opt_st = carry
        t_obs, s_obs = data

        def loss_fn(trainable):
            student_enc, student_head = trainable

            # ── Teacher features (frozen target) ──
            teacher_feat = il_network.teacher_decoder.apply(
                teacher_norm, teacher_dec_params, t_obs
            )

            # ── Student features (trainable) ──
            student_feat = il_network.student_encoder.apply(
                proprio_norm, student_enc, s_obs
            )

            # Feature alignment loss
            embed_loss = jnp.mean(
                jnp.abs(student_feat - jax.lax.stop_gradient(teacher_feat))
            )

            # Teacher action labels (frozen — stop_grad on both feat and head)
            teacher_logits = il_network.action_head.apply(
                None,
                teacher_action_head_params,
                jax.lax.stop_gradient(teacher_feat),
            )

            # Student actions through the trainable student head
            student_logits = il_network.action_head.apply(
                None, student_head, student_feat
            )

            action_loss = jnp.mean(
                jnp.abs(student_logits - jax.lax.stop_gradient(teacher_logits))
            )

            total_loss = embed_coef * embed_loss + action_coef * action_loss
            return total_loss, (embed_loss, action_loss)

        (loss, (embed_loss, action_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(trainable)
        updates, opt_st = optimizer.update(grads, opt_st, trainable)
        trainable = optax.apply_updates(trainable, updates)
        losses = jnp.stack([loss, embed_loss, action_loss])
        return (trainable, opt_st), losses

    def sgd_step(carry, _):
        trainable, opt_st = carry
        (trainable, opt_st), losses = jax.lax.scan(
            minibatch_step,
            (trainable, opt_st),
            (mb_teacher_obs, mb_student_obs),
            length=num_minibatches,
        )
        return (trainable, opt_st), jnp.mean(losses, axis=0)

    (trainable_params, opt_state), epoch_losses = jax.lax.scan(
        sgd_step,
        (trainable_params, opt_state),
        None,
        length=align_updates_per_trigger,
    )
    mean_total, mean_embed, mean_action = jnp.mean(epoch_losses, axis=0)
    return trainable_params, opt_state, mean_total, mean_embed, mean_action
