def teacher_loss_from_proxy_decoder():

    return


def teacher_reward_shaping_from_proxy_actions():

    return

@jax.jit
def align_step(params, opt_state, teacher_obs, student_obs, optimizer):
    (
        policy_dec_params,
        student_dec_params,
        proxy_dec_params,
        action_head_params,
        norm_params,
    ) = params

    def loss_fn(student_dec_params, proxy_dec_params):
        # features
        teacher_feat = policy_decoder.apply(
            norm_params, policy_dec_params, teacher_obs
        )
        student_feat = student_decoder.apply(
            norm_params, student_dec_params, student_obs
        )
        proxy_feat = proxy_decoder.apply(
            norm_params, proxy_dec_params, teacher_obs
        )

        # embedding loss
        embed_loss = (
            jnp.mean(jnp.abs(student_feat - lax.stop_gradient(teacher_feat)))
            + jnp.mean(jnp.abs(lax.stop_gradient(student_feat) - proxy_feat))
        )

        # action-head alignment
        teacher_logits = action_head.apply(
            None, action_head_params, lax.stop_gradient(teacher_feat)
        )
        student_logits = action_head.apply(
            None, action_head_params, student_feat
        )
        proxy_logits = action_head.apply(
            None, action_head_params, proxy_feat
        )

        action_loss = (
            jnp.mean(jnp.abs(student_logits - lax.stop_gradient(teacher_logits)))
            + jnp.mean(jnp.abs(lax.stop_gradient(student_logits) - proxy_logits))
        )

        return embed_loss + action_loss

    grads = jax.grad(loss_fn, argnums=(0, 1))(
        student_dec_params, proxy_dec_params
    )

    updates, opt_state = optimizer.update(
        grads, opt_state, (student_dec_params, proxy_dec_params)
    )

    student_dec_params, proxy_dec_params = jax.tree_util.tree_map(
        lambda p, u: p + u,
        (student_dec_params, proxy_dec_params),
        updates,
    )

    new_params = (
        policy_dec_params,
        student_dec_params,
        proxy_dec_params,
        action_head_params,
        norm_params,
    )

    return new_params, opt_state



# # How to use

# logging_loss = 0.0

# for _ in range(n_align_epochs):
#     params, opt_state, loss = align_step(
#         params,
#         opt_state,
#         teacher_obs,
#         student_obs,
#         align_optimizer,
#     )
#     logging_loss += loss

# logging_loss /= n_align_epochs
