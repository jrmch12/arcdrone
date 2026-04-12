"""Alignment step for DAgger training — vel-in-the-loop architecture.

    ``trainable_params = (student_enc, action_head, vel_estimator)``

Architecture in the loss:
    encoder_feats = student_enc(pixels, proprio)
    pred_linvel   = vel_estimator([encoder_feats, tilt_history])
    student_logits = action_head([encoder_feats, pred_linvel])

Loss structure:
    embed_loss    = mean |student_feat - stop_grad(teacher_feat)|
    action_loss   = mean |student_logits - stop_grad(teacher_logits)|
                    (gradients flow through action_head AND vel_estimator!)
    aux_vel_loss  = mean (pred_linvel - linvel_gt)^2
    total_loss    = embed_coef * embed_loss + action_coef * action_loss
                  + aux_vel_coef * aux_vel_loss
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
        "aux_vel_coef",
    ),
)
def align(
    trainable_params,           # (student_enc, action_head, vel_estimator)
    opt_state: optax.OptState,
    teacher_obs,
    student_obs,
    proprio_norm,
    *,
    teacher_dec_params,
    teacher_action_head_params,
    teacher_norm,
    il_network,
    optimizer,
    align_updates_per_trigger: int,
    num_minibatches: int,
    embed_coef: float = 1.0,
    action_coef: float = 1.0,
    aux_vel_coef: float = 0.0,
):
    """Run minibatched alignment with vel-in-the-loop.

    The predicted linvel flows INTO the action head, so action_loss gradients
    propagate through both the action head AND the vel estimator.

    Returns:
        ``(new_trainable_params, new_opt_state, mean_total, mean_embed, mean_action, mean_aux_vel)``
    """

    def _to_minibatches(x):
        return jnp.reshape(x, (num_minibatches, -1) + x.shape[2:])

    mb_teacher_obs = jax.tree_util.tree_map(_to_minibatches, teacher_obs)
    mb_student_obs = jax.tree_util.tree_map(_to_minibatches, student_obs)

    def minibatch_step(carry, data):
        trainable, opt_st = carry
        t_obs, s_obs = data

        def loss_fn(trainable):
            student_enc, student_head, vel_est_params = trainable

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

            # Teacher action labels (frozen) — uses teacher_action_head
            # which has teacher_feat_dim input (no vel concat)
            teacher_logits = il_network.teacher_action_head.apply(
                None,
                teacher_action_head_params,
                jax.lax.stop_gradient(teacher_feat),
            )

            # ── Vel-in-the-loop: estimate velocity, then feed to action head ──
            tilt_info = s_obs["aux_tilt"]
            vel_input = jnp.concatenate([student_feat, tilt_info], axis=-1)
            pred_linvel = il_network.vel_estimator.apply(
                None, vel_est_params, vel_input
            )

            # Action head receives [encoder_feats, predicted_linvel]
            action_input = jnp.concatenate([student_feat, pred_linvel], axis=-1)
            student_logits = il_network.action_head.apply(
                None, student_head, action_input
            )

            action_loss = jnp.mean(
                jnp.abs(student_logits - jax.lax.stop_gradient(teacher_logits))
            )

            # Vel supervision: MSE against ground-truth linvel
            linvel_gt = s_obs["aux_linvel"]
            aux_vel_loss = jnp.mean(
                (pred_linvel - jax.lax.stop_gradient(linvel_gt)) ** 2
            )

            total_loss = (embed_coef * embed_loss
                         + action_coef * action_loss
                         + aux_vel_coef * aux_vel_loss)
            return total_loss, (embed_loss, action_loss, aux_vel_loss)

        (loss, (embed_loss, action_loss, aux_vel_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(trainable)
        updates, opt_st = optimizer.update(grads, opt_st, trainable)
        trainable = optax.apply_updates(trainable, updates)
        losses = jnp.stack([loss, embed_loss, action_loss, aux_vel_loss])
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
    mean_total, mean_embed, mean_action, mean_aux_vel = jnp.mean(epoch_losses, axis=0)
    return trainable_params, opt_state, mean_total, mean_embed, mean_action, mean_aux_vel
