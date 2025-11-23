import jax
import jax.numpy as jnp
from flax import struct

from episode_tracer import Transition


@struct.dataclass
class JaxPNStepState:
    """JAX version of PNStep internal state using a ring buffer.

    We maintain:
      - obs, a, r, v, pi, done for the last `capacity` steps
      - start: logical index of the oldest element
      - size:  number of valid elements
      - done_flag: whether the *last appended* step had done=True

    n:        bootstrapping horizon
    capacity: ring buffer capacity (>= max episode length)
    gammas:   [n] powers of gamma
    gamman:   gamma^n
    alpha:    PER alpha
    """
    obs: jnp.ndarray        # [capacity, obs_dim]
    a: jnp.ndarray          # [capacity]
    r: jnp.ndarray          # [capacity]
    v: jnp.ndarray          # [capacity]
    pi: jnp.ndarray         # [capacity, num_actions]
    done: jnp.ndarray       # [capacity] (bool)

    size: jnp.ndarray       # ()
    start: jnp.ndarray      # ()
    done_flag: jnp.ndarray  # ()

    n: int                  # bootstrapping horizon (static)
    capacity: int           # ring buffer capacity (static)
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
    """Initialize a JAX PNStep state."""
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


def _pnstep_push_core(
    state: JaxPNStepState,
    obs,
    a,
    r,
    done,
    v,
    pi,
) -> JaxPNStepState:
    """Pure JAX core: push a single step into the ring buffer."""
    # Convert inputs to JAX arrays of the right dtype/shape.
    obs = jnp.asarray(obs, dtype=jnp.float32)
    a = jnp.asarray(a, dtype=jnp.int32)
    r = jnp.asarray(r, dtype=jnp.float32)
    v = jnp.asarray(v, dtype=jnp.float32)
    pi = jnp.asarray(pi, dtype=jnp.float32)
    done_arr = jnp.asarray(done, dtype=bool)

    # Logical index where we append: size (0..capacity-1).
    idx = (state.start + state.size) % state.capacity  # scalar int32 (traced)

    new_obs = state.obs.at[idx].set(obs)
    new_a = state.a.at[idx].set(a)
    new_r = state.r.at[idx].set(r)
    new_v = state.v.at[idx].set(v)
    new_pi = state.pi.at[idx].set(pi)
    new_done = state.done.at[idx].set(done_arr)

    # Update size (clamp to capacity) and done_flag
    new_size = jnp.minimum(state.size + 1, state.capacity)

    return state.replace(
        obs=new_obs,
        a=new_a,
        r=new_r,
        v=new_v,
        pi=new_pi,
        done=new_done,
        size=new_size,
        done_flag=done_arr,
    )


def _pnstep_can_pop_jax(state: JaxPNStepState) -> jnp.ndarray:
    """Pure JAX predicate: whether there is a transition to pop.

    Mirrors Python PNStep.__bool__:
        bool(len(self)) and (self._done or len(self) > self.n)
    """
    size = state.size
    cond_len = size > 0
    cond_done_or_long = jnp.logical_or(state.done_flag, size > state.n)
    return jnp.logical_and(cond_len, cond_done_or_long)


def jax_pnstep_can_pop(state: JaxPNStepState) -> bool:
    """Python-friendly wrapper for can_pop, used in Python while loops."""
    return bool(_pnstep_can_pop_jax(state))


def _pnstep_pop_core(state: JaxPNStepState) -> tuple[JaxPNStepState, Transition]:
    """Pure JAX core: pop a single transition from the buffer.

    This mirrors PNStep.pop as closely as possible but uses vectorized
    JAX ops (no Python loops).
    """
    size = state.size
    start = state.start
    cap = state.capacity
    n = state.n

    # Values at the current `start` (earliest) index.
    obs0 = state.obs[start]
    a0 = state.a[start]
    r0 = state.r[start]
    v0 = state.v[start]
    pi0 = state.pi[start]

    # Rewards r_t, r_{t+1}, ..., r_{t+n-1} (wrapped).
    j = jnp.arange(n, dtype=jnp.int32)                        # [n]
    idxs = (start + j) % cap                                  # [n]
    rs_all = state.r[idxs]                                    # [n]

    # Only first `len_rs = min(size, n)` rewards are valid
    len_rs = jnp.minimum(size, n)
    valid_mask = (j < len_rs).astype(jnp.float32)             # [n]
    rs_masked = rs_all * valid_mask                           # [n]

    # Discounted partial return Rn = sum(gamma^k * r_{t+k})
    Rn = jnp.sum(state.gammas * rs_masked)

    # Immediate reward r_t
    # (Python version does r = _deque_r.popleft(); this is that r_t)
    # r0 already defined above.

    # Length after popping the earliest state-action pair
    size_after = size - 1

    # Bootstrapping v_next, gamman:
    # if size_after >= n -> v_next at index start + n, gamman = gamma^n
    # else -> no bootstrap, v_next = 0, gamman = gamma^(len_rs - 1)
    idx_vnext = (start + n) % cap

    cond_bootstrap = size_after >= n
    v_next = jnp.where(cond_bootstrap, state.v[idx_vnext], 0.0)

    # Safely compute index len_rs-1 (>=0)
    len_rs_idx = jnp.maximum(len_rs - 1, 0)
    gamman_alt = state.gammas[len_rs_idx]

    gamman = jnp.where(cond_bootstrap, state.gamman, gamman_alt)

    Rn = Rn + v_next * gamman

    # Priority weight w = |v - Rn| ** alpha
    priority = jnp.abs(v0 - Rn) ** state.alpha

    # Update start/size like popleft
    new_start = (start + 1) % cap
    new_size = size - 1

    new_state = state.replace(
        start=new_start,
        size=new_size,
    )

    # done field for the popped transition:
    # Python PNStep sets done=True when there is no more bootstrapping.
    done_out = size_after < n

    trans = Transition(
        obs=obs0,     # [obs_dim]
        a=a0,         # scalar int32
        r=r0,         # scalar float32
        done=done_out,
        Rn=Rn,        # scalar float32
        v=v0,         # scalar float32
        pi=pi0,       # [num_actions]
        w=priority,   # scalar float32
    )
    return new_state, trans


def jax_pnstep_push(
    state: JaxPNStepState,
    obs,
    a,
    r,
    done,
    v,
    pi,
) -> JaxPNStepState:
    """Python-visible push wrapper.

    This function is JIT/scan-safe and can also be used inside Python loops.
    """
    return _pnstep_push_core(state, obs, a, r, done, v, pi)


def jax_pnstep_pop(
    state: JaxPNStepState,
) -> tuple[JaxPNStepState, Transition]:
    """Python-visible pop wrapper.

    Under JIT, you would call `_pnstep_pop_core` directly.
    """
    return _pnstep_pop_core(state)
