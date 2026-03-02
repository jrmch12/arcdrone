import jax
import jax.numpy as jnp

# def teacher_loss_from_proxy_decoder():

#     return


# def teacher_reward_shaping_from_proxy_actions():

#     return

@jax.jit
def align(
    params,
    opt_state,
    teacher_obs,
    student_obs,
    *,
    sitt_network,
    optimizer,
    align_updates_per_trigger,
):
    (
        policy_dec_params,
        student_dec_params,
        proxy_dec_params,
        action_head_params,
        norm_params,
    ) = params

    def loss_fn(student_dec_params, proxy_dec_params):
        teacher_feat = sitt_network.policy_decoder.apply(
            norm_params, policy_dec_params, teacher_obs
        )
        student_feat = sitt_network.student_decoder.apply(
            norm_params, student_dec_params, student_obs
        )
        proxy_feat = sitt_network.proxy_decoder.apply(
            norm_params, proxy_dec_params, teacher_obs
        )

        embed_loss = (
            jnp.mean(jnp.abs(student_feat - jax.lax.stop_gradient(teacher_feat)))
            + jnp.mean(jnp.abs(jax.lax.stop_gradient(student_feat) - proxy_feat))
        )

        teacher_logits = sitt_network.action_head.apply(
            None, action_head_params, jax.lax.stop_gradient(teacher_feat)
        )
        student_logits = sitt_network.action_head.apply(
            None, action_head_params, student_feat
        )
        proxy_logits = sitt_network.action_head.apply(
            None, action_head_params, proxy_feat
        )

        action_loss = (
            jnp.mean(jnp.abs(student_logits - jax.lax.stop_gradient(teacher_logits)))
            + jnp.mean(jnp.abs(jax.lax.stop_gradient(student_logits) - proxy_logits))
        )

        return embed_loss + action_loss

    def step(carry, _):
        student_dec, proxy_dec, opt_state = carry

        grads = jax.grad(loss_fn, argnums=(0, 1))(student_dec, proxy_dec)
        updates, opt_state = optimizer.update(
            grads, opt_state, (student_dec, proxy_dec)
        )

        student_dec, proxy_dec = jax.tree_util.tree_map(
            lambda p, u: p + u,
            (student_dec, proxy_dec),
            updates,
        )
        return (student_dec, proxy_dec, opt_state), None

    (student_dec_params, proxy_dec_params, opt_state), _ = jax.lax.scan(
        step,
        (student_dec_params, proxy_dec_params, opt_state),
        None,
        length=align_updates_per_trigger,
    )

    new_params = (
        policy_dec_params,
        student_dec_params,
        proxy_dec_params,
        action_head_params,
        norm_params,
    )
    return new_params, opt_state


