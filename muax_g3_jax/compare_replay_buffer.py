"""
Compare segments from Python Trajectory vs JaxReplayBufferState.

We:
  - roll out one CartPole episode using pure JAX env + JAX PNStep,
  - build a Python Trajectory,
  - sample segments of length k_steps via Trajectory.sample,
  - add those segments into a JAX replay buffer,
  - verify that the stored segments match the Python segments.

Run:

    python compare_replay_buffer.py
"""

import jax
from jax import numpy as jnp
import numpy as np

from cartpole_jax_env import CartPole
from episode_tracer import PNStep
from replay_buffer import Trajectory
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
from jax_replay_buffer import (
    jax_replay_init,
    jax_replay_add_segment,
)


def build_model(num_actions: int, obs_dim):
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
        discount=0.99,
        optimizer=gradient_transform,
        support_size=support_size,
    )

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    sample_input = jnp.zeros((1, *obs_dim), dtype=jnp.float32)
    model.init(init_key, sample_input)
    return model, key


def to_np(x):
    if isinstance(x, jnp.ndarray):
        return np.asarray(x)
    return np.asarray(x)


def main():
    cartpole = CartPole()
    obs_dim = cartpole.obs_shape  # (4,)
    num_actions = cartpole.num_actions
    max_steps = cartpole.default_params.max_steps_in_episode

    model, key = build_model(num_actions, obs_dim)
    params = model.params

    n = 4
    k_steps = 4
    gamma = 0.99
    alpha = 0.5

    # Python-style trajectory built from JAX tracer output
    py_traj = Trajectory()

    # JAX tracer state (already validated vs Python PNStep)
    jax_tracer = jax_pnstep_init(
        n=n,
        gamma=gamma,
        alpha=alpha,
        obs_dim=obs_dim[0],
        num_actions=num_actions,
        capacity=max_steps,
    )

    # ---- Roll out one episode and build a Trajectory ----
    key, reset_key = jax.random.split(key)
    obs, env_state = cartpole.reset(reset_key, cartpole.default_params)

    temperature = 1.0

    for t in range(max_steps):
        key, act_key, env_key = jax.random.split(key, 3)
        a, pi, v = model.act_from_params(
            params,
            act_key,
            obs,
            num_simulations=5,
            temperature=temperature,
        )

        obs_next, env_state, reward, done, info = cartpole.step(
            env_key,
            env_state,
            a,
            cartpole.default_params,
        )

        # JAX tracer -> Python Trajectory (using the same logic as fit_pure_cartpole)
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
            py_traj.add(trans)

        obs = obs_next
        if bool(done):
            break

    py_traj.finalize()
    print("Python trajectory length (transitions):", len(py_traj))

    # ---- Sample segments from Python Trajectory ----
    num_segments = 5
    segments = py_traj.sample(num_samples=num_segments, k_steps=k_steps)
    print(f"Sampled {len(segments)} segments of length {k_steps} from Python Trajectory.")

    # ---- Fill JAX replay buffer with these segments ----
    capacity = 32
    jax_rb = jax_replay_init(
        capacity=capacity,
        L=k_steps,
        obs_dim=obs_dim[0],
        num_actions=num_actions,
    )

    for i, seg in enumerate(segments):
        # For now, give each segment equal weight = 1.0
        jax_rb = jax_replay_add_segment(jax_rb, seg, weight=1.0)
        print(f"Added segment {i} to JAX replay buffer.")

    print("JAX replay buffer size:", int(jax_rb.size))

    # ---- Compare first few stored segments vs Python segments ----
    for i, seg in enumerate(segments):
        if i >= int(jax_rb.size):
            break
        obs_py = seg.obs[0]  # [L, obs_dim]
        a_py = seg.a[0]      # [L]
        r_py = seg.r[0]      # [L]
        Rn_py = seg.Rn[0]    # [L]
        pi_py = seg.pi[0]    # [L, num_actions]

        obs_jax = to_np(jax_rb.obs[i])
        a_jax = to_np(jax_rb.a[i])
        r_jax = to_np(jax_rb.r[i])
        Rn_jax = to_np(jax_rb.Rn[i])
        pi_jax = to_np(jax_rb.pi[i])

        ok = True
        if not np.allclose(to_np(obs_py), obs_jax, atol=1e-5):
            print(f"[{i}] obs mismatch")
            ok = False
        if not np.allclose(to_np(a_py), a_jax, atol=1e-5):
            print(f"[{i}] a mismatch")
            ok = False
        if not np.allclose(to_np(r_py), r_jax, atol=1e-5):
            print(f"[{i}] r mismatch")
            ok = False
        if not np.allclose(to_np(Rn_py), Rn_jax, atol=1e-5):
            print(f"[{i}] Rn mismatch")
            ok = False
        if not np.allclose(to_np(pi_py), pi_jax, atol=1e-5):
            print(f"[{i}] pi mismatch")
            ok = False

        if ok:
            print(f"[{i}] Segment OK")

    print("Done comparing replay buffer segments.")


if __name__ == "__main__":
    main()
