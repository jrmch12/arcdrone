"""PPO loss with SITT modifications.

SITT adds two terms to standard PPO:
  1. Proxy KL reward shaping: ``reward += proxy_kl_coef * KL(teacher || proxy)``
     (applied to data *before* the loss — see training_step in train.py)
  2. Auxiliary RL alignment loss: ``rl_align_loss = |teacher_feat - sg(proxy_feat)|``
     weighted by ``sitt_align_coef`` and added to the PPO total loss.
"""

from typing import Any, Tuple

from brax.training import types
from brax.training.types import Params
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class PPONetworkParams:
    """Mutable network parameters for PPO + SITT."""
    policy: Params            # (teacher_dec_params, action_head_params)
    value: Params
    proxy_dec_params: Any = None


def compute_gae(
    truncation: jnp.ndarray,
    termination: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    lambda_: float = 1.0,
    discount: float = 0.99,
):
    """Generalised Advantage Estimation."""
    truncation_mask = 1 - truncation
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
    )
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), acc

    (_, _), vs_minus_v_xs = jax.lax.scan(
        compute_vs_minus_v_xs,
        (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate(
        [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
    )
    advantages = (
        rewards + discount * (1 - termination) * vs_t_plus_1 - values
    ) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def compute_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    clipping_epsilon_value: float | None = None,
    # SITT options
    use_sitt: bool = False,
    sitt_align_coef: float = 0.01,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """PPO loss + optional SITT auxiliary alignment term."""

    parametric_action_distribution = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply

    # Time-first layout
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    policy_logits = policy_apply(normalizer_params, params.policy, data.observation)
    baseline = value_apply(normalizer_params, params.value, data.observation)

    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, terminal_obs)

    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value loss
    v_error = vs - baseline
    v_loss = v_error * v_error
    if clipping_epsilon_value is not None:
        old_values = data.extras["policy_extras"]["value"]
        v_clipped = old_values + jnp.clip(
            baseline - old_values,
            -clipping_epsilon_value,
            clipping_epsilon_value,
        )
        v_loss = jnp.maximum(v_loss, (vs - v_clipped) ** 2)
    v_loss = jnp.mean(v_loss) * 0.5 * vf_coefficient

    # Entropy
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    rl_loss = policy_loss + v_loss + entropy_loss
    total_loss = rl_loss

    # KL divergence metric
    new_dist = parametric_action_distribution.create_dist(policy_logits)
    if hasattr(new_dist, "kl_divergence"):
        old_dist_params = data.extras["policy_extras"]["distribution_params"]
        old_dist = parametric_action_distribution.create_dist(old_dist_params)
        kl = jnp.mean(new_dist.kl_divergence(old_dist))
        policy_dist_mean_std = jnp.mean(new_dist.scale)
    else:
        kl, policy_dist_mean_std = jnp.array(0.0), jnp.array(0.0)

    # SITT: auxiliary alignment loss (teacher feat vs proxy feat)
    rl_align_loss = jnp.array(0.0)
    sitt_align_coef = 1.0 # HACK hardcoded for now
    if use_sitt:
        teacher_feat = ppo_network.policy_decoder.apply(
            normalizer_params, params.policy[0], data.observation
        )
        proxy_feat = ppo_network.ppo_proxy_decoder.apply(
            normalizer_params, params.proxy_dec_params, data.observation
        )
        proxy_feat_detached = jax.lax.stop_gradient(proxy_feat)
        rl_align_loss = jnp.mean(jnp.abs(teacher_feat - proxy_feat_detached))
        rl_align_loss = rl_align_loss * sitt_align_coef
        total_loss = total_loss + rl_align_loss
        # total_loss = total_loss

    return total_loss, {
        "total_loss": total_loss,
        "rl_loss": rl_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "rl_align_loss": rl_align_loss,
        "kl_mean": kl,
        "policy_dist_mean_std": policy_dist_mean_std,
    }
