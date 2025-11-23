import dataclasses
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import struct

from episode_tracer import Transition


@struct.dataclass
class JaxPNStepState:
    """JAX version of PNStep internal state.

    This mirrors the Python PNStep semantics using a ring buffer of capacity
    `capacity` and bootstrapping horizon `n`.
    """
    # Ring buffers for steps
    obs: jnp.ndarray        # [capacity, obs_dim]
    a: jnp.ndarray          # [capacity] (int32)
    r: jnp.ndarray          # [capacity] (float32)
    v: jnp.ndarray          # [capacity] (float32)
    pi: jnp.ndarray         # [capacity, num_actions]
    done: jnp.ndarray       # [capacity] (bool)

    # Book-keeping
    size: jnp.ndarray       # () how many valid entries (<= capacity)
    start: jnp.ndarray      # () index of oldest entry (like deque left)
    done_flag: jnp.ndarray  # () bool: last appended `done`

    # Hyperparameters / precomputed constants
    n: int                  # bootstrapping horizon
    capacity: int           # ring buffer capacity (>= max episode length)
    gammas: jnp.ndarray     # [n]
    gamman: jnp.ndarray     # ()
    alpha: jnp.ndarray      # ()


def jax_pnstep_init(
    n: int,
    gamma: float,
    alpha: float,
    obs_dim: int,
    num_actions: int,
    capacity: int,
) -> JaxPNStepState:
    """Initialize a JAX PNStep state.

    Args:
        n: bootstrapping horizon (same as PNStep.n).
        gamma: discount.
        alpha: PER alpha.
        obs_dim: dimension of observation (e.g. 4 for CartPole).
        num_actions: number of actions.
        capacity: max number of steps we store (>= max episode length).

    Returns:
        JaxPNStepState with zeroed buffers.
    """
    gammas = jnp.power(gamma, jnp.arange(n, dtype=jnp.float32))
    gamman = jnp.power(gamma, n)

    return JaxPNStepState(
        obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        a=jnp.zeros((capacity,), dtype=jnp.int32),
        r=jnp.zeros((capacity,), dtype=jnp.float32),
        v=jnp.zeros((capacity,), dtype=jnp.float32),
        pi=jnp.zeros((capacity, num_actions), dtype=jnp.float32),
        done=jnp.zeros((capacity,), dtype=bool),
        size=jnp.array(0, dtype=jnp.int32),
        start=jnp.array(0, dtype=jnp.int32),
        done_flag=jnp.array(False),
        n=n,
        capacity=capacity,
        gammas=gammas,
        gamman=gamman,
        alpha=jnp.array(alpha, dtype=jnp.float32),
    )


def _logical_to_physical_index(state: JaxPNStepState, k: int) -> int:
    """Convert logical index k (0..size-1) to physical ring-buffer index."""
    return int((int(state.start) + k) % state.capacity)


def jax_pnstep_push(
    state: JaxPNStepState,
    obs: jnp.ndarray,
    a: int,
    r: float,
    done: bool,
    v: jnp.ndarray,
    pi: jnp.ndarray,
) -> JaxPNStepState:
    """Push a single step (obs, a, r, done, v, pi) into the ring buffer.

    This corresponds to PNStep.add(...), but does NOT pop anything.
    """
    # Where to write: end of current deque
    size = int(state.size)
    assert size < state.capacity, "JaxPNStepState capacity exceeded"
    idx = _logical_to_physical_index(state, size)  # index of new element

    obs = jnp.asarray(obs, dtype=jnp.float32)
    v = jnp.asarray(v, dtype=jnp.float32)
    pi = jnp.asarray(pi, dtype=jnp.float32)
    a = jnp.asarray(a, dtype=jnp.int32)
    r = jnp.asarray(r, dtype=jnp.float32)
    done = jnp.asarray(done, dtype=bool)

    state = state.replace(
        obs=state.obs.at[idx].set(obs),
        a=state.a.at[idx].set(a),
        r=state.r.at[idx].set(r),
        v=state.v.at[idx].set(v),
        pi=state.pi.at[idx].set(pi),
        done=state.done.at[idx].set(done),
        size=state.size + 1,
        done_flag=done,
    )
    return state


def jax_pnstep_can_pop(state: JaxPNStepState) -> bool:
    """Equivalent of Python PNStep.__bool__ for the JAX state.

    True iff there is at least one transition to pop:

        bool(len(self)) and (self._done or len(self) > self.n)
    """
    size = int(state.size)
    if size == 0:
        return False

    # len(self) > n OR last add had done=True
    return bool((size > state.n) or bool(state.done_flag))


def jax_pnstep_pop(state: JaxPNStepState) -> Tuple[JaxPNStepState, Transition]:
    """Pop a single transition from the JAX tracer state.

    Mirrors PNStep.pop semantics as closely as possible.
    """
    size = int(state.size)
    if size == 0:
        raise RuntimeError("jax_pnstep_pop called on empty state")

    start = int(state.start)
    n = state.n
    cap = state.capacity

    # Pop earliest obs, a, v, pi (like _deque_s.popleft())
    idx0 = start
    obs0 = state.obs[idx0]
    a0 = state.a[idx0]
    v0 = state.v[idx0]
    pi0 = state.pi[idx0]

    # rs = _deque_r[:n]
    len_rs = min(n, size)  # number of rewards to use for Rn
    rs = []
    for j in range(len_rs):
        idx = (start + j) % cap
        rs.append(state.r[idx])
    rs = jnp.asarray(rs, dtype=jnp.float32)  # [len_rs]

    # Rn = sum(gammas[:len_rs] * rs)
    gammas = state.gammas[:len_rs]
    Rn = jnp.sum(gammas * rs)

    # r = _deque_r.popleft() -> reward at current start
    r0 = state.r[idx0]

    # len(self) after popping s (but before removing rewards)
    size_after_pop_s = size - 1

    # Bootstrapping v_next
    if size_after_pop_s >= n:
        # v_next = v at old s_n (like _deque_s[self.n - 1] after popleft)
        idx_vnext = (start + n) % cap
        v_next = state.v[idx_vnext]
        gamman = state.gamman
    else:
        v_next = jnp.array(0.0, dtype=jnp.float32)
        # Equivalent to _gammas[len(rs) - 1] in Python when len_rs>0
        gamman = state.gammas[len_rs - 1] if len_rs > 0 else jnp.array(0.0)

    Rn = Rn + v_next * gamman

    # Priority weight: w = |v - Rn| ** alpha
    priority = jnp.abs(v0 - Rn) ** state.alpha
    w = float(priority)

    # Update start/size (like popleft on both deques)
    new_start = (start + 1) % cap
    new_size = size - 1
    state = state.replace(
        start=jnp.array(new_start, dtype=jnp.int32),
        size=jnp.array(new_size, dtype=jnp.int32),
    )

    trans = Transition(
        obs=obs0,
        a=int(a0),
        r=float(r0),
        done=bool(size_after_pop_s < n),  # approximate "done" like Python PNStep
        Rn=float(Rn),
        v=float(v0),
        pi=pi0,
        w=w,
    )
    return state, trans


def jax_pnstep_add_step(
    state: JaxPNStepState,
    obs: jnp.ndarray,
    a: int,
    r: float,
    done: bool,
    v: jnp.ndarray,
    pi: jnp.ndarray,
) -> Tuple[JaxPNStepState, Optional[Transition], bool]:
    """Push a step and pop at most one transition if available.

    This is convenient for "per-step" logic:

        state, maybe_trans, has_trans = jax_pnstep_add_step(...)

    If has_trans is True, maybe_trans is a Transition; otherwise None.
    """
    state = jax_pnstep_push(state, obs, a, r, done, v, pi)
    if jax_pnstep_can_pop(state):
        state, trans = jax_pnstep_pop(state)
        return state, trans, True
    else:
        return state, None, False
