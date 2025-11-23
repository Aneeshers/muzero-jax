import os
import jax
from jax import numpy as jnp
from tqdm import tqdm
import wandb

from cartpole_jax_env import CartPole
from model import MuZero, optimizer as make_optimizer
from nn import (
    _init_representation_func,
    _init_prediction_func,
    _init_dynamic_func,
    Representation,
    Prediction,
    Dynamic,
)
from jax_tracer import (
    jax_pnstep_init,
    jax_pnstep_push,
    jax_pnstep_can_pop,
    jax_pnstep_pop,
)
from jax_transition_replay_buffer import (
    jax_trans_replay_init,
    jax_trans_replay_add_transition,
    jax_trans_replay_sample_segments,
)

def eval_pure_cartpole(
    model: MuZero,
    params,
    cartpole: CartPole,
    key,
    num_simulations: int = 50,
    num_test_episodes: int = 10,
):
    """Evaluate the model on the pure JAX CartPole env.

    Returns:
        avg_reward: float, average total reward over num_test_episodes.
        key: updated PRNGKey.
    """
    total_rewards = jnp.zeros((num_test_episodes,), dtype=jnp.float32)

    for ep in tqdm(range(num_test_episodes), desc="Testing"):
        # Reset env
        key, reset_key = jax.random.split(key)
        obs, env_state = cartpole.reset(reset_key, cartpole.default_params)

        ep_reward = 0.0
        for t in range(cartpole.default_params.max_steps_in_episode):
            key, act_key, env_key = jax.random.split(key, 3)

            # Deterministic policy at test time: temperature = 0.0
            a, pi, v = model.act_from_params(
                params,
                act_key,
                obs,
                num_simulations=num_simulations,
                temperature=0.0,
            )

            obs, env_state, reward, done, info = cartpole.step(
                env_key,
                env_state,
                a,
                cartpole.default_params,
            )

            ep_reward += float(reward)
            if bool(done):
                break

        total_rewards = total_rewards.at[ep].set(ep_reward)

    avg_reward = float(jnp.mean(total_rewards))
    return avg_reward, key


def fit_pure_cartpole(
    model: MuZero,
    max_episodes: int = 1000,
    max_training_steps: int = 100_000,
    num_simulations: int = 50,
    k_steps: int = 10,
    buffer_capacity: int = 500,     # capacity = number of transitions
    buffer_warm_up: int = 1024,     # minimum transitions before updates
    num_trajectory: int = 32,       # used to set batch_size
    sample_per_trajectory: int = 10,
    num_update_per_episode: int = 50,
    random_seed: int = 0,
    test_interval: int = 10,
    num_test_episodes: int = 10,
):
    # ---------------- Env ----------------
    cartpole = CartPole()
    discount = model._discount
    num_actions = cartpole.num_actions
    obs_dim = cartpole.obs_shape  # (4,)
    max_steps = cartpole.default_params.max_steps_in_episode

    # ---------------- Params / opt_state ----------------
    key = jax.random.PRNGKey(random_seed)
    key, init_key, test_key = jax.random.split(key, 3)
    sample_input = jnp.zeros((1, *obs_dim), dtype=jnp.float32)

    model.init(init_key, sample_input)
    params = model.params
    opt_state = model.optimizer_state

    # ---------------- JAX Transition Replay Buffer ----------------
    # Buffer stores per-step transitions; segments of length k_steps are
    # built at sample time.
    jax_rb = jax_trans_replay_init(
        capacity=buffer_capacity,
        obs_dim=obs_dim[0],
        num_actions=num_actions,
        k_steps=k_steps,
    )

    training_step = 0

    # Batch size for updates: roughly num_trajectory * sample_per_trajectory
    batch_size = num_trajectory * sample_per_trajectory

    def temperature_fn(max_training_steps, training_steps):
        if training_steps < 0.5 * max_training_steps:
            return 1.0
        elif training_steps < 0.75 * max_training_steps:
            return 0.5
        else:
            return 0.25

    # ---------------- Buffer warmup ----------------
        # ---------------- Buffer warmup ----------------
    print("Buffer warm-up stage (pure CartPole)...")
    warmup_ep = 0
    # We warm up until we have at least `buffer_warm_up` transitions.
    while int(jax_rb.size) < buffer_warm_up:
        warmup_ep += 1
        before_size = int(jax_rb.size)

        key, reset_key = jax.random.split(key)
        obs, env_state = cartpole.reset(reset_key, cartpole.default_params)

        jax_tracer = jax_pnstep_init(
            n=k_steps,
            gamma=discount,
            alpha=0.5,
            obs_dim=obs_dim[0],
            num_actions=num_actions,
            capacity=max_steps,
        )

        temperature = temperature_fn(max_training_steps, training_step)

        ep_steps = 0
        for t in range(max_steps):
            ep_steps += 1
            key, act_key, env_key = jax.random.split(key, 3)

            # act_from_params: pure-style
            a, pi, v = model.act_from_params(
                params,
                act_key,
                obs,
                num_simulations=num_simulations,
                temperature=temperature,
            )

            obs_next, env_state, reward, done, info = cartpole.step(
                env_key,
                env_state,
                a,
                cartpole.default_params,
            )

            # Push into JAX tracer and pop transitions into transition buffer
            jax_tracer = jax_pnstep_push(
                jax_tracer,
                obs,
                int(a),
                float(reward),
                bool(done),
                v,
                pi,
            )
            while jax_pnstep_can_pop(jax_tracer):
                jax_tracer, trans = jax_pnstep_pop(jax_tracer)
                jax_rb = jax_trans_replay_add_transition(jax_rb, trans)

            obs = obs_next
            if bool(done):
                break

        after_size = int(jax_rb.size)
        added = after_size - before_size
        print(
            f"[Warmup ep {warmup_ep}] steps={ep_steps}, "
            f"transitions added this ep={added}, total transitions={after_size} "
            f"(target warmup={buffer_warm_up})"
        )

        # Safety: prevent infinite loop if buffer_warm_up > capacity
        if buffer_warm_up > buffer_capacity:
            print(
                f"Warning: buffer_warm_up ({buffer_warm_up}) > buffer_capacity ({buffer_capacity}); "
                "warmup will never terminate."
            )
            break


    print("Start training (pure CartPole)...")
    best_test_reward = float("-inf")

    for ep in tqdm(range(max_episodes), desc="Training"):
        key, reset_key = jax.random.split(key)
        obs, env_state = cartpole.reset(reset_key, cartpole.default_params)

        jax_tracer = jax_pnstep_init(
            n=k_steps,
            gamma=discount,
            alpha=0.5,
            obs_dim=obs_dim[0],
            num_actions=num_actions,
            capacity=max_steps,
        )

        temperature = temperature_fn(max_training_steps, training_step)

        # ----- Rollout one episode -----
        reward_sum = 0.0
        for t in tqdm(range(max_steps), desc="Rollout"):
            key, act_key, env_key = jax.random.split(key, 3)
            a, pi, v = model.act_from_params(
                params,
                act_key,
                obs,
                num_simulations=num_simulations,
                temperature=temperature,
            )
            obs_next, env_state, reward, done, info = cartpole.step(
                env_key,
                env_state,
                a,
                cartpole.default_params,
            )
            reward_sum += float(reward)

            jax_tracer = jax_pnstep_push(
                jax_tracer,
                obs,
                int(a),
                float(reward),
                bool(done),
                v,
                pi,
            )
            while jax_pnstep_can_pop(jax_tracer):
                jax_tracer, trans = jax_pnstep_pop(jax_tracer)
                jax_rb = jax_trans_replay_add_transition(jax_rb, trans)

            obs = obs_next
            if bool(done):
                break

        # ----- Updates -----
        train_loss = 0.0
        for _ in range(num_update_per_episode):
            # Sample a batch of segments from transition-level JAX replay buffer
            batch, key = jax_trans_replay_sample_segments(
                jax_rb,
                key,
                batch_size=batch_size,
            )
            params, opt_state, loss = model.update_from_params(
                params,
                opt_state,
                batch,
            )
            train_loss += float(loss)
            training_step += 1

        train_loss /= num_update_per_episode

        # ----- Test evaluation on pure JAX env -----
        if ep % test_interval == 0:
            avg_test_reward, test_key = eval_pure_cartpole(
                model,
                params,
                cartpole,
                test_key,
                num_simulations=num_simulations,
                num_test_episodes=num_test_episodes,
            )
            best_test_reward = max(best_test_reward, avg_test_reward)
            print(
                f"[Episode {ep}] train_loss = {train_loss:.4f}, "
                f"training_step = {training_step}, "
                f"test_avg_reward = {avg_test_reward:.2f}, "
                f"best_test_reward = {best_test_reward:.2f}"
            )
            wandb.log({
                "train_loss": train_loss,
                "training_step": training_step,
                "test_avg_reward": avg_test_reward,
                "best_test_reward": best_test_reward,
            })

        else:
            print(
                f"[Episode {ep}] train_loss = {train_loss:.4f}, "
                f"training_step = {training_step}, "
                f"reward_sum = {reward_sum:.2f}"
            )
            wandb.log({
                "train_loss": train_loss,
                "training_step": training_step,
                "reward_sum": reward_sum,
            })

        if training_step >= max_training_steps:
            break

    # sync model instance for saving/testing or later reuse
    model._params = params
    model._opt_state = opt_state
    return model
