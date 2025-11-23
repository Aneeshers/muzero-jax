"""
Compare Python PNStep vs JaxPNStepState on a single pure-JAX CartPole episode.

Run:

    python compare_tracers.py
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


def main():
    cartpole = CartPole()
    obs_dim = cartpole.obs_shape  # (4,)
    num_actions = cartpole.num_actions
    max_steps = cartpole.default_params.max_steps_in_episode

    model, key = build_model(num_actions, obs_dim)
    params = model.params

    n = 4
    gamma = 0.99
    alpha = 0.5

    # Python tracer
    py_tracer = PNStep(n=n, gamma=gamma, alpha=alpha)

    # JAX tracer state
    jax_tracer = jax_pnstep_init(
        n=n,
        gamma=gamma,
        alpha=alpha,
        obs_dim=obs_dim[0],
        num_actions=num_actions,
        capacity=max_steps,
    )

    # ---- Rollout one episode and feed both tracers ----
    key, reset_key = jax.random.split(key)
    obs, env_state = cartpole.reset(reset_key, cartpole.default_params)

    py_transitions = []
    jax_transitions = []

    training_step = 0
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

        # --- Python tracer path ---
        py_tracer.add(obs, int(a), float(reward), bool(done), v=v, pi=pi)
        while py_tracer:
            trans = py_tracer.pop()
            py_transitions.append(trans)

        # --- JAX tracer path ---
        # First push:
        jax_tracer = jax_pnstep_push(
            jax_tracer,
            obs,
            int(a),
            float(reward),
            bool(done),
            v,
            pi,
        )
        # Then pop as many as Python did (one per while-iteration)
        while jax_pnstep_can_pop(jax_tracer):
            jax_tracer, trans_jax = jax_pnstep_pop(jax_tracer)
            jax_transitions.append(trans_jax)

        obs = obs_next
        if bool(done):
            break

    print(f"Python PNStep produced {len(py_transitions)} transitions.")
    print(f"JAX PNStep produced   {len(jax_transitions)} transitions.")

    if len(py_transitions) != len(jax_transitions):
        print("Length mismatch between Python and JAX tracers!")
        return

    # ---- Compare field-by-field ----
    def to_np(x):
        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
        return np.asarray(x)

    for i, (py_t, jax_t) in enumerate(zip(py_transitions, jax_transitions)):
        ok = True

        if not np.allclose(to_np(py_t.obs), to_np(jax_t.obs), atol=1e-5):
            print(f"[{i}] obs mismatch")
            ok = False
        if py_t.a != jax_t.a:
            print(f"[{i}] action mismatch: py={py_t.a}, jax={jax_t.a}")
            ok = False
        if not np.allclose(py_t.r, jax_t.r, atol=1e-5):
            print(f"[{i}] reward mismatch: py={py_t.r}, jax={jax_t.r}")
            ok = False
        if not np.allclose(py_t.Rn, jax_t.Rn, atol=1e-5):
            print(f"[{i}] Rn mismatch: py={py_t.Rn}, jax={jax_t.Rn}")
            ok = False
        if not np.allclose(to_np(py_t.pi), to_np(jax_t.pi), atol=1e-5):
            print(f"[{i}] pi mismatch")
            ok = False
        if not np.allclose(py_t.w, jax_t.w, atol=1e-5):
            print(f"[{i}] w mismatch: py={py_t.w}, jax={jax_t.w}")
            ok = False

        if ok:
            print(f"[{i}] OK")

    print("Done comparing tracers.")


if __name__ == "__main__":
    main()
