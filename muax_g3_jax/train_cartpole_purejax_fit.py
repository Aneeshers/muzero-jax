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
    _pnstep_push_core,
    _pnstep_can_pop_jax,
    _pnstep_pop_core,
)
from jax_transition_replay_buffer import (
    jax_trans_replay_init,
    _add_transition_arrays,
    _sample_segments_core,
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

    Uses:
      - scan over time (steps per episode)
      - vmap over episodes
      - jit over the whole batch eval
    """
    env_params = cartpole.default_params
    max_steps = env_params.max_steps_in_episode

    def episode_rollout(params, key_ep):
        """Roll out a single episode and return total reward."""
        # Reset env for this episode
        obs0, env_state0 = cartpole.reset(key_ep, env_params)

        # carry = (env_state, key_step, total_reward, done_flag)
        carry0 = (
            env_state0,
            key_ep,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(False),
        )

        def step_fn(carry, _):
            env_state, key_step, total_reward, done = carry
            key_step, act_key, env_key = jax.random.split(key_step, 3)

            # If already done, just carry state forward
            def do_step(args):
                env_state, key_step, total_reward, done = args

                obs = cartpole.get_obs(env_state)

                a, pi, v = model.act_from_params(
                    params,
                    act_key,
                    obs,
                    num_simulations=num_simulations,
                    temperature=0.0,  # deterministic at eval
                )

                obs_next, env_state_next, reward, done_step, info = cartpole.step(
                    env_key,
                    env_state,
                    a,
                    env_params,
                )

                reward = jnp.asarray(reward, dtype=jnp.float32)

                # Only accumulate reward if we weren't done yet
                total_reward2 = total_reward + jnp.where(done, 0.0, reward)
                done2 = jnp.logical_or(done, done_step)

                return (env_state_next, key_step, total_reward2, done2), None

            def skip_step(args):
                # Episode already done: keep everything as-is
                return args, None

            carry_out, _ = jax.lax.cond(
                done,
                skip_step,
                do_step,
                (env_state, key_step, total_reward, done),
            )
            return carry_out, None

        carryT, _ = jax.lax.scan(
            step_fn,
            carry0,
            xs=None,
            length=max_steps,
        )
        _, key_final, total_reward_final, _ = carryT
        return total_reward_final

    # Batch over episodes: params is shared, keys are per-episode
    batched_rollout = jax.jit(
        jax.vmap(episode_rollout, in_axes=(None, 0))
    )

    # Make per-episode keys
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_test_episodes)  # [num_test_episodes]

    # Run all episodes in parallel
    total_rewards = batched_rollout(params, keys)  # [num_test_episodes]

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
    sample_per_trajectory: int = 10,  # unused now but kept for interface
    num_update_per_episode: int = 50,
    random_seed: int = 0,
    test_interval: int = 100,
    num_test_episodes: int = 10,
):
    # ---------------- Env ----------------
    cartpole = CartPole()
    discount = model._discount
    num_actions = cartpole.num_actions
    obs_dim = cartpole.obs_shape  # (4,)
    max_steps = cartpole.default_params.max_steps_in_episode
    env_params = cartpole.default_params  # capture once

    # ---------------- Params / opt_state ----------------
    key = jax.random.PRNGKey(random_seed)
    key, init_key, test_key = jax.random.split(key, 3)
    sample_input = jnp.zeros((1, *obs_dim), dtype=jnp.float32)

    model.init(init_key, sample_input)
    params = model.params
    opt_state = model.optimizer_state

    # ---------------- JAX Transition Replay Buffer ----------------
    jax_rb = jax_trans_replay_init(
        capacity=buffer_capacity,
        obs_dim=obs_dim[0],
        num_actions=num_actions,
        k_steps=k_steps,
    )

    training_step = 0
    batch_size = num_trajectory * sample_per_trajectory

    def temperature_fn(max_training_steps, training_steps):
        if training_steps < 0.5 * max_training_steps:
            return 1.0
        elif training_steps < 0.75 * max_training_steps:
            return 0.5
        else:
            return 0.25

    # ----------------------------------------------------------------
    # Jitted per-episode rollout (no updates)
    # ----------------------------------------------------------------
    def _collect_one_episode(params, rb_state, key, temperature):
        """Roll out one episode, add transitions to rb_state, return (rb_state, key, reward_sum)."""
        # Reset env
        key, reset_key = jax.random.split(key)
        obs0, env_state0 = cartpole.reset(reset_key, env_params)

        # Init tracer
        tracer0 = jax_pnstep_init(
            n=k_steps,
            gamma=discount,
            alpha=0.5,
            obs_dim=obs_dim[0],
            num_actions=num_actions,
            capacity=max_steps,
        )

        # carry: (env_state, tracer_state, rb_state, key, reward_sum, done_flag)
        carry0 = (
            env_state0,
            tracer0,
            rb_state,
            key,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(False),
        )

        def step_fn(carry, _):
            env_state, tracer_state, rb_state, key_step, reward_sum, done_flag = carry
            key_step, act_key, env_key = jax.random.split(key_step, 3)

            def do_step(args):
                env_state, tracer_state, rb_state, key_step, reward_sum, done_flag = args
                obs = cartpole.get_obs(env_state)  # [obs_dim]

                a, pi, v = model.act_from_params(
                    params,
                    act_key,
                    obs,
                    num_simulations=num_simulations,
                    temperature=temperature,
                )

                obs_next, env_state_next, reward, done, info = cartpole.step(
                    env_key,
                    env_state,
                    a,
                    env_params,
                )

                # Push into tracer
                tracer_state2 = _pnstep_push_core(
                    tracer_state,
                    obs,
                    a,
                    reward,
                    done,
                    v,
                    pi,
                )

                # Pop at most one transition here and add to rb_state
                def add_and_pop(tp):
                    t_state, rb = tp
                    t_state2, trans = _pnstep_pop_core(t_state)
                    rb2 = _add_transition_arrays(
                        rb,
                        trans.obs,
                        trans.a,
                        trans.r,
                        trans.Rn,
                        trans.pi,
                        trans.w,
                    )
                    return (t_state2, rb2)

                tracer_state3, rb_state2 = jax.lax.cond(
                    _pnstep_can_pop_jax(tracer_state2),
                    add_and_pop,
                    lambda tp: tp,
                    (tracer_state2, rb_state),
                )

                reward_sum2 = reward_sum + reward
                done_flag2 = jnp.logical_or(done_flag, done)

                new_carry = (
                    env_state_next,
                    tracer_state3,
                    rb_state2,
                    key_step,
                    reward_sum2,
                    done_flag2,
                )
                return new_carry, None

            def skip_step(args):
                # Episode already done; carry state forward
                return args, None

            carry_out, _ = jax.lax.cond(
                done_flag,
                skip_step,
                do_step,
                (env_state, tracer_state, rb_state, key_step, reward_sum, done_flag),
            )
            return carry_out, None

        # Scan over max_steps
        carryT, _ = jax.lax.scan(step_fn, carry0, xs=None, length=max_steps)
        env_stateT, tracerT, rb_stateT, keyT, reward_sum, done_flag_final = carryT

        # Flush remaining transitions from tracer at episode end
        def flush_cond(state_rb):
            t_state, rb = state_rb
            return _pnstep_can_pop_jax(t_state)

        def flush_body(state_rb):
            t_state, rb = state_rb
            t_state2, trans = _pnstep_pop_core(t_state)
            rb2 = _add_transition_arrays(
                rb,
                trans.obs,
                trans.a,
                trans.r,
                trans.Rn,
                trans.pi,
                trans.w,
            )
            return (t_state2, rb2)

        tracer_final, rb_final = jax.lax.while_loop(
            flush_cond,
            flush_body,
            (tracerT, rb_stateT),
        )

        return rb_final, keyT, reward_sum

    collect_one_episode_jit = jax.jit(_collect_one_episode)

    # ----------------------------------------------------------------
    # Jitted per-episode rollout + updates
    # ----------------------------------------------------------------
    def _train_one_episode(params, opt_state, rb_state, key, temperature):
        """Roll out one episode, then run num_update_per_episode updates.

        Returns:
          params, opt_state, rb_state, key, reward_sum, avg_train_loss
        """
        # First collect one episode (reuse the same core as warmup)
        rb_after, key_after, reward_sum = _collect_one_episode(
            params,
            rb_state,
            key,
            temperature,
        )

        # Then run SGD updates
        def update_body(i, carry):
            params, opt_state, rb_state, key_step, loss_accum = carry
            batch, key_step = _sample_segments_core(rb_state, key_step, batch_size)
            params2, opt_state2, loss = model.update_from_params(
                params,
                opt_state,
                batch,
            )
            loss_accum2 = loss_accum + loss
            return (params2, opt_state2, rb_state, key_step, loss_accum2)

        init_carry = (params, opt_state, rb_after, key_after, jnp.array(0.0, dtype=jnp.float32))
        paramsT, opt_stateT, rbT, keyT, loss_sum = jax.lax.fori_loop(
            0,
            num_update_per_episode,
            update_body,
            init_carry,
        )
        avg_loss = loss_sum / float(num_update_per_episode)
        return paramsT, opt_stateT, rbT, keyT, reward_sum, avg_loss

    train_one_episode_jit = jax.jit(_train_one_episode)

    # ---------------- Buffer warmup (using jitted rollout) ----------------
    print("Buffer warm-up stage (pure CartPole)...")
    warmup_ep = 0
    while int(jax_rb.size) < buffer_warm_up:
        warmup_ep += 1
        before_size = int(jax_rb.size)

        temperature = temperature_fn(max_training_steps, training_step)
        jax_rb, key, ep_reward = collect_one_episode_jit(
            params,
            jax_rb,
            key,
            temperature,
        )

        after_size = int(jax_rb.size)
        added = after_size - before_size
        print(
            f"[Warmup ep {warmup_ep}] transitions added={added}, "
            f"total transitions={after_size} (target warmup={buffer_warm_up})"
        )

        if buffer_warm_up > buffer_capacity:
            print(
                f"Warning: buffer_warm_up ({buffer_warm_up}) > buffer_capacity ({buffer_capacity}); "
                "warmup will never fully satisfy the target."
            )
            break

    # ---------------- Training ----------------
    print("Start training (pure CartPole, JIT+scan episodes)...")
    best_test_reward = float("-inf")

    for ep in tqdm(range(max_episodes), desc="Training"):
        temperature = temperature_fn(max_training_steps, training_step)

        # Jitted per-episode rollout + updates
        params, opt_state, jax_rb, key, reward_sum, train_loss = train_one_episode_jit(
            params,
            opt_state,
            jax_rb,
            key,
            temperature,
        )
        training_step += num_update_per_episode

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
                f"[Episode {ep}] train_loss = {float(train_loss):.4f}, "
                f"training_step = {training_step}, "
                f"test_avg_reward = {avg_test_reward:.2f}, "
                f"best_test_reward = {best_test_reward:.2f}"
            )
            wandb.log({
                "train_loss": float(train_loss),
                "training_step": training_step,
                "test_avg_reward": avg_test_reward,
                "best_test_reward": best_test_reward,
            })
        else:
            print(
                f"[Episode {ep}] train_loss = {float(train_loss):.4f}, "
                f"training_step = {training_step}, "
                f"reward_sum = {float(reward_sum):.2f}"
            )
            wandb.log({
                "train_loss": float(train_loss),
                "training_step": training_step,
                "reward_sum": float(reward_sum),
            })

        if training_step >= max_training_steps:
            break

    # sync model instance for saving/testing or later reuse
    model._params = params
    model._opt_state = opt_state
    return model
