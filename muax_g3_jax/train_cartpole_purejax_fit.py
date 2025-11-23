# train_cartpole_purejax_fit.py

import os
import jax
from jax import numpy as jnp
from tqdm import tqdm
from cartpole_jax_env import CartPole
from replay_buffer import TrajectoryReplayBuffer, Trajectory
from model import MuZero, optimizer as make_optimizer
import wandb
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
# from test import test  # Not needed for pure JAX eval; we use CartPole directly.


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
    buffer_capacity: int = 500,
    buffer_warm_up: int = 128,
    num_trajectory: int = 32,
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

    # ---------------- Tracer & Buffer ----------------
    buffer = TrajectoryReplayBuffer(buffer_capacity)

    training_step = 0

    def temperature_fn(max_training_steps, training_steps):
        if training_steps < 0.5 * max_training_steps:
            return 1.0
        elif training_steps < 0.75 * max_training_steps:
            return 0.5
        else:
            return 0.25

    # ---------------- Buffer warmup ----------------
    print("Buffer warm-up stage (pure CartPole)...")
    while len(buffer) < buffer_warm_up:
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
        
        trajectory = Trajectory()
        temperature = temperature_fn(max_training_steps, training_step)

        for t in range(cartpole.default_params.max_steps_in_episode):
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
                trajectory.add(trans)

            obs = obs_next
            if bool(done):
                break

        trajectory.finalize()
        if len(trajectory) >= k_steps:
            mean_w = float(trajectory.batched_transitions.w.mean())
            buffer.add(trajectory, mean_w)

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
        
        trajectory = Trajectory()
        temperature = temperature_fn(max_training_steps, training_step)

        # ----- Rollout one episode -----
        reward_sum = 0.0
        for t in tqdm(range(cartpole.default_params.max_steps_in_episode), desc="Rollout"):
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
                trajectory.add(trans)

            obs = obs_next
            if bool(done):
                break

        trajectory.finalize()
        if len(trajectory) >= k_steps:
            mean_w = float(trajectory.batched_transitions.w.mean())
            buffer.add(trajectory, mean_w)

        # ----- Updates -----
        train_loss = 0.0
        for _ in range(num_update_per_episode):
            batch = buffer.sample(
                num_trajectory=num_trajectory,
                sample_per_trajectory=sample_per_trajectory,
                k_steps=k_steps,
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
