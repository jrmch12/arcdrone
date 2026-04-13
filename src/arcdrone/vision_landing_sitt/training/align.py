"""SITT alignment step.

Trains **both** student_enc_params (vision encoder) and proxy_dec_params
(proxy decoder) to match the frozen teacher decoder features.

Loss structure (from controller/sitt):
  embed_loss = |student_feat - sg(teacher_feat)|
             + |sg(student_feat) - proxy_feat|
  action_loss = |student_logits - sg(teacher_logits)|
              + |sg(student_logits) - proxy_logits|

The proxy learns to mimic the student; the student learns to mimic the
teacher.  The proxy is then used in the PPO reward shaping (KL) because
it only needs flat obs (no vision), making it usable during RL rollouts
on the teacher env.

Minibatching mirrors PPO's SGD step:
  - data in:        (batch_size * num_minibatches, unroll_length, ...)
  - per minibatch:  (batch_size * unroll_length, ...)
  - outer scan:     align_updates_per_trigger passes
  - inner scan:     num_minibatches gradient steps per pass
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax


@partial(
    jax.jit,
    static_argnames=(
        "sitt_network", "optimizer",
        "align_updates_per_trigger", "num_minibatches",
        "embed_coef", "action_coef",
    ),
)
def align(
    student_enc_params,
    proxy_dec_params,
    opt_state: optax.OptState,
    teacher_obs,
    student_obs,
    proprio_norm,
    *,
    teacher_dec_params,
    action_head_params,
    teacher_norm,
    sitt_network,
    optimizer,
    align_updates_per_trigger: int,
    num_minibatches: int,
    embed_coef: float = 1.0,
    action_coef: float = 1.0,
):
    """Run minibatched SITT alignment.

    Args:
        student_enc_params: trainable student encoder params.
        proxy_dec_params: trainable proxy decoder params.
        opt_state: optax state for ``(student_enc_params, proxy_dec_params)``.
        teacher_obs: dict of obs from student env, shape
            ``(batch_size * num_minibatches, T, ...)``.
        student_obs: same shape / content as ``teacher_obs`` (both are the
            full student env obs dict; each network extracts its keys).
        proprio_norm: live RunningStatisticsState for proprio normalisation
            (or normalizer_select(..., "policy_obs") since proprio == policy_obs).
        teacher_dec_params: frozen teacher decoder params (pre-bound).
        action_head_params: frozen action head params (pre-bound).
        teacher_norm: frozen teacher normaliser sub-state (pre-bound).
        sitt_network: ``SITTNetworks`` instance (static).
        optimizer: optax optimiser for (student_enc, proxy_dec).
        align_updates_per_trigger: outer repetitions.
        num_minibatches: gradient steps per outer repetition.

    Returns:
        ``(student_enc_params, proxy_dec_params, opt_state,
           mean_total_loss, mean_embed_loss, mean_action_loss)``
    """

    # ── Split into minibatches ─────────────────────────────────────────
    def _to_minibatches(x):
        # (B*M, T, ...) → (M, B*T, ...)
        return jnp.reshape(x, (num_minibatches, -1) + x.shape[2:])

    mb_teacher_obs = jax.tree_util.tree_map(_to_minibatches, teacher_obs)
    mb_student_obs = jax.tree_util.tree_map(_to_minibatches, student_obs)

    # ── Per-minibatch gradient step ────────────────────────────────────
    def minibatch_step(carry, data):
        (student_enc, proxy_dec, opt_st) = carry
        t_obs, s_obs = data

        def loss_fn(student_enc, proxy_dec):
            teacher_feat = sitt_network.teacher_decoder.apply(
                teacher_norm, teacher_dec_params, t_obs
            )
            student_feat = sitt_network.student_encoder.apply(
                proprio_norm, student_enc, s_obs
            )
            proxy_feat = sitt_network.proxy_decoder.apply(
                teacher_norm, proxy_dec, t_obs
            )

            # Embedding losses
            embed_loss = (
                jnp.mean(jnp.abs(
                    student_feat - jax.lax.stop_gradient(teacher_feat)
                ))
                + jnp.mean(jnp.abs(
                    jax.lax.stop_gradient(student_feat) - proxy_feat
                ))
            )

            # Action-level losses
            teacher_logits = sitt_network.action_head.apply(
                None, action_head_params,
                jax.lax.stop_gradient(teacher_feat),
            )
            student_logits = sitt_network.action_head.apply(
                None, action_head_params, student_feat,
            )
            proxy_logits = sitt_network.action_head.apply(
                None, action_head_params, proxy_feat,
            )
            action_loss = (
                jnp.mean(jnp.abs(
                    student_logits - jax.lax.stop_gradient(teacher_logits)
                ))
                + jnp.mean(jnp.abs(
                    jax.lax.stop_gradient(student_logits) - proxy_logits
                ))
            )

            total = embed_coef * embed_loss + action_coef * action_loss
            return total, (embed_loss, action_loss)

        (loss, (embed_loss, action_loss)), grads = jax.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(student_enc, proxy_dec)

        updates, opt_st = optimizer.update(
            grads, opt_st, (student_enc, proxy_dec)
        )
        student_enc, proxy_dec = jax.tree_util.tree_map(
            lambda p, u: p + u,
            (student_enc, proxy_dec),
            updates,
        )

        losses = jnp.stack([loss, embed_loss, action_loss])
        return (student_enc, proxy_dec, opt_st), losses

    # ── One pass over all minibatches ─────────────────────────────────
    def sgd_step(carry, _):
        student_enc, proxy_dec, opt_st = carry
        (student_enc, proxy_dec, opt_st), losses = jax.lax.scan(
            minibatch_step,
            (student_enc, proxy_dec, opt_st),
            (mb_teacher_obs, mb_student_obs),
            length=num_minibatches,
        )
        # losses: (num_minibatches, 3)
        return (student_enc, proxy_dec, opt_st), jnp.mean(losses, axis=0)

    # ── Outer loop ────────────────────────────────────────────────────
    (student_enc_params, proxy_dec_params, opt_state), epoch_losses = jax.lax.scan(
        sgd_step,
        (student_enc_params, proxy_dec_params, opt_state),
        None,
        length=align_updates_per_trigger,
    )
    # epoch_losses: (align_updates_per_trigger, 3)
    mean_total, mean_embed, mean_action = jnp.mean(epoch_losses, axis=0)
    return (
        student_enc_params, proxy_dec_params, opt_state,
        mean_total, mean_embed, mean_action,
    )
