"""
Quick demo script to verify the pure-JAX CartPole + MuZero.act_from_params
interface and the integration with PNStep / TrajectoryReplayBuffer.

Run from muax_simple/ as:

    python train_cartpole_purejax_demo.py
"""

import jax
from jax import numpy as jnp

from cartpole_jax_env import CartPole
from episode_tracer import PNStep
from replay_buffer import TrajectoryReplayBuffer, Trajectory
from model import MuZero, optimizer as make_optimizer
from nn import (
    _init_representation_func,
    _init_prediction_func,
    _init_dynamic_func,
    Representation,
    Prediction,
    Dynamic,
)


# ---------------------------------------------------------------------------
# Temperature schedule (same as in train_jax.py)
# ---------------------------------------------------------------------------

def _temperature_fn(max_training_steps, training_steps):
    if training_steps < 0.5 * max_training_steps:
        return 1.0
    elif training_steps < 0.75 * max_training_steps:
        return 0.5
    else:
        return 0.25


# ---------------------------------------------------------------------------
# Fake train_step using pure JAX CartPole + existing Python tracer/buffer
# ---------------------------------------------------------------------------

def make_fake_train_step(
    model: MuZero,
    cartpole_env: CartPole,
    tracer: PNStep,
    buffer: TrajectoryReplayBuffer,  # unused here, but kept for interface parity
    temperature_fn,
    max_training_steps: int,
    num_simulations: int,
):
    """
    Returns a non-jitted train_step(carry, _) function that:
      - uses CartPole directly (no Gym wrapper),
      - uses existing Python tracer,
      - calls model.act_from_params(params, key, obs).

    This is just to validate the interface & shapes before we go full JAX/scan.
    """

    def train_step(carry, _):
        # Unpack carry; opt_state is included for future use, but unused here.
        params, opt_state, env_state, key, training_step = carry

        # Split RNG for acting and env stepping
        key, act_key, env_key = jax.random.split(key, 3)

        # Get observation from pure JAX env state
        obs = cartpole_env.get_obs(env_state)  # jax.Array, shape [obs_dim]

        # Same temperature schedule as in your fit loop
        temperature = temperature_fn(
            max_training_steps=max_training_steps,
            training_steps=training_step,
        )

        # Pure-style action selection: explicit params
        a, pi, v = model.act_from_params(
            params,
            act_key,
            obs,
            num_simulations=num_simulations,
            temperature=temperature,
        )

        # Step pure JAX env directly (no wrapper)
        obs_next, env_state_next, reward, done, info = cartpole_env.step(
            env_key,
            env_state,
            a,
            cartpole_env.default_params,
        )

        # Use existing Python tracer (PNStep).
        # We convert small scalars to Python types for it:
        tracer.add(
            obs,               # jax.Array
            int(a),            # Python int
            float(reward),     # Python float
            bool(done),        # Python bool
            v=v,               # jax.Array scalar
            pi=pi,             # jax.Array [num_actions]
        )

        # Increment training_step counter
        training_step += 1

        # Pack new carry. Note: params/opt_state are unchanged here.
        new_carry = (params, opt_state, env_state_next, key, training_step)

        # Optional: return info dict for checking terminal condition
        info_out = {
            "reward": float(reward),
            "done": bool(done),
        }
        return new_carry, info_out

    return train_step


# ---------------------------------------------------------------------------
# Main: build model, run one episode, push into buffer, sample & update
# ---------------------------------------------------------------------------

def main():
    # ---------------- Hyperparameters ----------------
    max_training_steps = 400_000
    num_simulations = 50
    k_steps = 4
    num_update_per_episode = 10
    seed = 0

    # ---------------- Env ----------------
    cartpole = CartPole()
    discount = 0.99
    num_actions = cartpole.num_actions
    obs_dim = cartpole.obs_shape  # (4,)

    # ---------------- Model ----------------
    support_size = 10
    embedding_size = 8
    full_support_size = support_size * 2 + 1

    repr_fn = _init_representation_func(Representation, embedding_size)
    pred_fn = _init_prediction_func(Prediction, num_actions, full_support_size)
    dy_fn = _init_dynamic_func(Dynamic, embedding_size, num_actions, full_support_size)

    gradient_transform = make_optimizer(
        init_value=0.02,
        peak_value=0.02,
        end_value=0.002,
        warmup_steps=5000,
        transition_steps=5000,
    )

    model = MuZero(
        repr_fn,
        pred_fn,
        dy_fn,
        policy="muzero",
        optimizer=gradient_transform,
        discount=discount,
        support_size=support_size,
    )

    # Init model params with a dummy observation
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    sample_input = jnp.zeros((1, *obs_dim), dtype=jnp.float32)  # [B=1, obs_dim]
    model.init(init_key, sample_input)

    params = model.params
    opt_state = model.optimizer_state

    # ---------------- Tracer & Buffer ----------------
    tracer = PNStep(n=k_steps, gamma=discount, alpha=0.5)
    buffer = TrajectoryReplayBuffer(500)

    # ---------------- Env reset ----------------
    key, reset_key = jax.random.split(key)
    obs0, env_state0 = cartpole.reset(reset_key, cartpole.default_params)

    training_step = 0
    carry = (params, opt_state, env_state0, key, training_step)

    fake_train_step = make_fake_train_step(
        model=model,
        cartpole_env=cartpole,
        tracer=tracer,
        buffer=buffer,
        temperature_fn=_temperature_fn,
        max_training_steps=max_training_steps,
        num_simulations=num_simulations,
    )

    trajectory = Trajectory()

    # ---------------- Roll out one episode ----------------
    print("Running one episode with pure JAX CartPole + act_from_params...")
    for t in range(cartpole.default_params.max_steps_in_episode):
        carry, info = fake_train_step(carry, None)
        params, opt_state, env_state, key, training_step = carry

        # Flush tracer into trajectory, as in your current fit loop
        while tracer:
            trans = tracer.pop()
            trajectory.add(trans)

        if info["done"]:
            print(f"Episode finished at step {t}, reward={info['reward']:.3f}")
            break

    # ---------------- Push trajectory into buffer ----------------
    trajectory.finalize()
    print("Collected trajectory length:", len(trajectory))

    if len(trajectory) >= k_steps:
        mean_w = float(trajectory.batched_transitions.w.mean())
        buffer.add(trajectory, mean_w)
        print("Added trajectory to buffer with mean weight:", mean_w)
    else:
        print("Trajectory too short for k_steps; skipping buffer.add")

    # ---------------- Sample from buffer & update model ----------------
    if len(buffer) > 0:
        # sample a single mini-trajectory from the buffer
        batch = buffer.sample(
            num_trajectory=1,
            k_steps=k_steps,
            sample_per_trajectory=1,
        )
        print("Sampled batch shapes:")
        print("  obs:", batch.obs.shape)
        print("  a:", batch.a.shape)
        print("  r:", batch.r.shape)
        print("  Rn:", batch.Rn.shape)
        print("  pi:", batch.pi.shape)

        for _ in range(num_update_per_episode):
            params, opt_state, loss = model.update_from_params(params, opt_state, batch)
            loss_metric = {"loss": float(loss)}
        print("Loss metric after one update:", loss_metric)

        # Sync params/opt_state back to local vars if you want to reuse them
        params = model.params
        opt_state = model.optimizer_state
    else:
        print("Buffer is empty; not running update")

    print("Done.")

if __name__ == "__main__":
    main()
