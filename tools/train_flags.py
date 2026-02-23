import sys
import numpy as np


def parse_flags(argv):
    config = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key.startswith("train."):
                key = key[len("train."):]

            try:
                value = int(value)
            except ValueError:
                pass

            config[key] = value
    return config


def main():
    config = parse_flags(sys.argv[1:])

    # Required
    num_envs = config["num_envs"]
    unroll_length = config["unroll_length"]
    batch_size = config["batch_size"]
    num_minibatches = config["num_minibatches"]
    num_updates_per_batch = config["num_updates_per_batch"]
    num_timesteps = config["num_timesteps"]
    num_evals = config["num_evals"]

    # Assumptions (explicit!)
    action_repeat = 1
    num_resets_per_eval = 1
    num_evals_after_init = max(num_evals - 1, 1)

    # -----------------------------
    # Brax interpretation (correct)
    # -----------------------------

    if (batch_size * num_minibatches) % num_envs != 0:
        print("⚠ WARNING: (batch_size * num_minibatches) must be divisible by num_envs")

    # Total rollout sequences collected before SGD
    total_rollout_sequences = batch_size * num_minibatches

    # Each rollout sequence has length unroll_length
    transitions_per_sequence = unroll_length

    # Total transitions collected per training_step
    total_transitions_collected = (
        total_rollout_sequences * unroll_length
    )

    # How many times scan runs
    scan_length = total_rollout_sequences // num_envs

    # Env steps per training_step (true simulator steps)
    env_step_per_training_step = (
        total_transitions_collected * action_repeat
    )

    # Training steps per eval
    num_training_steps_per_eval = int(np.ceil(
        num_timesteps /
        (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ))

    # Total SGD updates
    total_sgd = (
        num_evals_after_init
        * num_training_steps_per_eval
        * num_minibatches
        * num_updates_per_batch
    )

    # Total env steps actually executed
    total_env_steps = (
        num_evals_after_init
        * num_training_steps_per_eval
        * env_step_per_training_step
    )

    env_steps_per_sgd = total_env_steps / total_sgd

    # Memory proxy
    transitions_in_memory = total_transitions_collected
    data_reuse_factor = num_updates_per_batch

    # -----------------------------
    # Print
    # -----------------------------

    print("\n===== BRAx PPO Training Info =====")

    print("\nASSUMPTIONS:")
    print(f"  action_repeat = {action_repeat}")
    print(f"  num_resets_per_eval = {num_resets_per_eval}")
    print(f"  num_evals_after_init = {num_evals_after_init}")

    print("\nDEFINITIONS:")
    print("  unroll_length = number of timesteps in each rollout sequence. This will be accumulate per env and per scan step.")
    print("  batch_size = number of unroll chunks inside each minibatch")
    print(
        "  scan step (generate_unroll_parallel_calls) = "
        "(batch_size * num_minibatches) // num_envs"
    )
    print("  (this is how many times acting.generate_unroll() is played in parallel)")
    print("  for stable pipeline shapes, batch_size * num_minibatches should be divisible by num_envs")

    print("\nBATCH STRUCTURE:")
    print(f"  total_rollout_sequences    = {total_rollout_sequences}")
    print(f"  transitions_per_sequence   = {transitions_per_sequence}")
    print(f"  total_transitions_collected = {total_transitions_collected}")
    print(f"  scan_length                = {scan_length}")

    print("\nCORE NUMBERS:")
    print(f"  env_steps_requested(train.num_timesteps) = {num_timesteps}")
    print(f"  env_steps_executed                        = {int(total_env_steps)}")
    print(f"  env_step_per_training_step = {env_step_per_training_step}")
    print(f"  training_steps_per_eval    = {num_training_steps_per_eval}")
    print(f"  total_sgd_updates          = {int(total_sgd)}")

    print("\nIMPORTANT RATIOS:")
    print(f"  env_steps_per_sgd          = {env_steps_per_sgd:.2f}")
    print(f"  data_reuse_factor          = {data_reuse_factor}")

    print("\nGPU / MEMORY LOAD:")
    print(f"  transitions_held_in_memory = {transitions_in_memory}")
    print("  (samples stored before each PPO update)")

    print("===================================\n")


if __name__ == "__main__":
    main()