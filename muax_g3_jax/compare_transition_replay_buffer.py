"""
Compare JaxTransitionReplayBufferState against a Python list of transitions.

We:
  - roll out one CartPole episode using pure JAX env + MuZero + JaxPNStepState,
  - on each popped Transition:
      * store it in a Python list (py_transitions),
      * add it to JaxTransitionReplayBufferState,
  - verify that the stored transitions in the buffer match py_transitions,
  - verify that sampled segments from the buffer correspond to contiguous
    slices of the underlying transition sequence.

Run:

    python compare_transition_replay_buffer.py
"""

import jax
from jax import numpy as jnp
import numpy as np
from tqdm import tqdm

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

    n = 4          # PNStep horizon
    k_steps = 4    # segment length for sampling
    gamma = 0.99
    alpha = 0.5

    # Python reference: list of transitions popped from JaxPNStepState
    py_transitions = []

    # JAX tracer state
    jax_tracer = jax_pnstep_init(
        n=n,
        gamma=gamma,
        alpha=alpha,
        obs_dim=obs_dim[0],
        num_actions=num_actions,
        capacity=max_steps,
    )

    # JAX transition-level replay buffer
    capacity = max_steps  # big enough for this one episode
    jax_rb = jax_trans_replay_init(
        capacity=capacity,
        obs_dim=obs_dim[0],
        num_actions=num_actions,
        k_steps=k_steps,
    )

    # ---- Roll out one episode and fill both py_transitions and jax_rb ----
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

        # Push + pop transitions from JaxPNStepState
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
            py_transitions.append(trans)
            jax_rb = jax_trans_replay_add_transition(jax_rb, trans)

        obs = obs_next
        if bool(done):
            break

    print("Number of transitions popped from tracer:", len(py_transitions))
    print("Transition replay buffer size:", int(jax_rb.size))

    # ---- Compare transitions stored in JAX buffer vs Python list ----
    size = int(jax_rb.size)
    ok_all = True
    for i in range(size):
        py_t = py_transitions[i]
        # JAX buffer: transitions are in ring order; since we never overfilled
        # capacity, logical order == buffer order for indices 0..size-1.
        obs_jax = jax_rb.obs[i]
        a_jax = jax_rb.a[i]
        r_jax = jax_rb.r[i]
        Rn_jax = jax_rb.Rn[i]
        pi_jax = jax_rb.pi[i]
        w_jax = jax_rb.w[i]

        ok = True
        if not np.allclose(to_np(py_t.obs), to_np(obs_jax), atol=1e-5):
            print(f"[{i}] obs mismatch")
            ok = False
        if py_t.a != int(a_jax):
            print(f"[{i}] a mismatch: py={py_t.a}, jax={int(a_jax)}")
            ok = False
        if not np.allclose(py_t.r, float(r_jax), atol=1e-5):
            print(f"[{i}] r mismatch: py={py_t.r}, jax={float(r_jax)}")
            ok = False
        if not np.allclose(py_t.Rn, float(Rn_jax), atol=1e-5):
            print(f"[{i}] Rn mismatch: py={py_t.Rn}, jax={float(Rn_jax)}")
            ok = False
        if not np.allclose(to_np(py_t.pi), to_np(pi_jax), atol=1e-5):
            print(f"[{i}] pi mismatch")
            ok = False
        if not np.allclose(py_t.w, float(w_jax), atol=1e-5):
            print(f"[{i}] w mismatch: py={py_t.w}, jax={float(w_jax)}")
            ok = False

        if ok:
            print(f"[{i}] Transition OK")
        ok_all = ok_all and ok

    if not ok_all:
        print("Transition-level buffer content mismatch detected.")
        return

    # ---- Check sampled segments vs contiguous slices of transitions ----
    print("Sampling segments from transition-level replay buffer...")
    batch_size = 3
    batch, key = jax_trans_replay_sample_segments(
        jax_rb,
        key,
        batch_size=batch_size,
    )

    # Build Python arrays from py_transitions
    obs_seq = np.stack([to_np(t.obs) for t in py_transitions], axis=0)      # [N, obs_dim]
    a_seq = np.stack([t.a for t in py_transitions], axis=0)                 # [N]
    r_seq = np.stack([t.r for t in py_transitions], axis=0)                 # [N]
    Rn_seq = np.stack([t.Rn for t in py_transitions], axis=0)               # [N]
    pi_seq = np.stack([to_np(t.pi) for t in py_transitions], axis=0)        # [N, num_actions]

    B = batch.obs.shape[0]
    L = batch.obs.shape[1]

    print(f"Sampled batch shapes: obs={batch.obs.shape}, a={batch.a.shape}, r={batch.r.shape}")

    # We can't directly see the start indices from jax_trans_replay_sample_segments,
    # but we can verify that each segment is present as some contiguous slice.
    for b in range(B):
        seg_obs = to_np(batch.obs[b])   # [L, obs_dim]
        seg_a = to_np(batch.a[b])       # [L]
        seg_r = to_np(batch.r[b])       # [L]
        seg_Rn = to_np(batch.Rn[b])     # [L]
        seg_pi = to_np(batch.pi[b])     # [L, num_actions]

        found = False
        for s in range(obs_seq.shape[0] - L + 1):
            if (np.allclose(obs_seq[s:s+L], seg_obs, atol=1e-5) and
                np.allclose(a_seq[s:s+L], seg_a, atol=1e-5) and
                np.allclose(r_seq[s:s+L], seg_r, atol=1e-5) and
                np.allclose(Rn_seq[s:s+L], seg_Rn, atol=1e-5) and
                np.allclose(pi_seq[s:s+L], seg_pi, atol=1e-5)):
                print(f"Segment {b} matches Python transitions slice [{s}:{s+L}]")
                found = True
                break

        if not found:
            print(f"Segment {b} does NOT match any contiguous Python slice!")
            ok_all = False

    if ok_all:
        print("All sampled segments are consistent with Python transition sequence.")
    else:
        print("Some sampled segments did not match; see messages above.")

    print("Done comparing transition-level replay buffer.")


if __name__ == "__main__":
    main()
