import jax
import jax.numpy as jnp
from flax import struct

from episode_tracer import Transition


@struct.dataclass
class JaxReplayBufferState:
    """Simple JAX replay buffer that stores fixed-length segments.

    Each entry is a Transition segment of length L = k_steps:
      obs: [L, obs_dim]
      a:   [L]
      r:   [L]
      Rn:  [L]
      pi:  [L, num_actions]

    We store up to `capacity` segments.
    """
    obs: jnp.ndarray   # [capacity, L, obs_dim]
    a: jnp.ndarray     # [capacity, L]
    r: jnp.ndarray     # [capacity, L]
    Rn: jnp.ndarray    # [capacity, L]
    pi: jnp.ndarray    # [capacity, L, num_actions]
    w: jnp.ndarray     # [capacity] segment-level weights
    size: jnp.ndarray  # ()
    head: jnp.ndarray  # ()
    capacity: int      # scalar int
    L: int             # segment length (k_steps)


def jax_replay_init(
    capacity: int,
    L: int,
    obs_dim: int,
    num_actions: int,
) -> JaxReplayBufferState:
    """Initialize empty JAX replay buffer.

    Args:
        capacity: max number of segments to store.
        L: segment length (k_steps).
        obs_dim: observation dimension.
        num_actions: number of actions.

    Returns:
        JaxReplayBufferState with zeroed buffers.
    """
    return JaxReplayBufferState(
        obs=jnp.zeros((capacity, L, obs_dim), dtype=jnp.float32),
        a=jnp.zeros((capacity, L), dtype=jnp.int32),
        r=jnp.zeros((capacity, L), dtype=jnp.float32),
        Rn=jnp.zeros((capacity, L), dtype=jnp.float32),
        pi=jnp.zeros((capacity, L, num_actions), dtype=jnp.float32),
        w=jnp.zeros((capacity,), dtype=jnp.float32),
        size=jnp.array(0, dtype=jnp.int32),
        head=jnp.array(0, dtype=jnp.int32),
        capacity=capacity,
        L=L,
    )


def _segment_from_transition(seg: Transition):
    """Convert a batched Transition segment [B, L, ...] into per-segment arrays [L, ...].

    We assume:
      seg.obs: [B, L, obs_dim]
      seg.a:   [B, L]
      seg.r:   [B, L]
      seg.Rn:  [B, L]
      seg.pi:  [B, L, num_actions]
    with B=1 from Trajectory.sample.

    Returns:
      obs: [L, obs_dim]
      a:   [L]
      r:   [L]
      Rn:  [L]
      pi:  [L, num_actions]
    """
    # seg.obs is typically a numpy array; convert to jax
    obs = jnp.asarray(seg.obs)
    a = jnp.asarray(seg.a)
    r = jnp.asarray(seg.r)
    Rn = jnp.asarray(seg.Rn)
    pi = jnp.asarray(seg.pi)

    # Remove batch dimension if present
    if obs.ndim == 3:  # [B, L, obs_dim]
        obs = obs[0]
    if a.ndim == 2:
        a = a[0]
    if r.ndim == 2:
        r = r[0]
    if Rn.ndim == 2:
        Rn = Rn[0]
    if pi.ndim == 3:
        pi = pi[0]

    # Ensure dtypes
    obs = obs.astype(jnp.float32)
    a = a.astype(jnp.int32)
    r = r.astype(jnp.float32)
    Rn = Rn.astype(jnp.float32)
    pi = pi.astype(jnp.float32)

    return obs, a, r, Rn, pi


def jax_replay_add_segment(
    state: JaxReplayBufferState,
    seg: Transition,
    weight: float,
) -> JaxReplayBufferState:
    """Add a single segment (Transition) into the replay buffer.

    `seg` should represent a contiguous segment of length L = state.L.
    """
    obs, a, r, Rn, pi = _segment_from_transition(seg)

    # Basic sanity check: L matches
    L = state.L
    if obs.shape[0] != L:
        raise ValueError(f"Segment length {obs.shape[0]} != buffer L={L}")

    idx = int(state.head)
    cap = state.capacity

    state = state.replace(
        obs=state.obs.at[idx].set(obs),
        a=state.a.at[idx].set(a),
        r=state.r.at[idx].set(r),
        Rn=state.Rn.at[idx].set(Rn),
        pi=state.pi.at[idx].set(pi),
        w=state.w.at[idx].set(float(weight)),
    )

    new_head = (idx + 1) % cap
    new_size = jnp.minimum(state.size + 1, cap)

    state = state.replace(
        head=jnp.array(new_head, dtype=jnp.int32),
        size=new_size,
    )
    return state


def jax_replay_sample(
    state: JaxReplayBufferState,
    key: jax.Array,
    batch_size: int,
):
    """Sample a batch of segments from the replay buffer.

    Returns:
      batch: Transition with batched fields:
        obs: [B, L, obs_dim]
        a:   [B, L]
        r:   [B, L]
        Rn:  [B, L]
        pi:  [B, L, num_actions]
      key: updated PRNGKey
    """
    size = int(state.size)
    if size == 0:
        raise RuntimeError("jax_replay_sample called on empty buffer")

    # Use weights w[:size] as probabilities
    weights = state.w[:size]
    # Avoid division by zero if all weights are zero
    weights = jnp.where(weights <= 0.0, jnp.ones_like(weights), weights)
    probs = weights / jnp.sum(weights)

    key, subkey = jax.random.split(key)
    idx = jax.random.choice(subkey, size, shape=(batch_size,), p=probs)

    obs = state.obs[idx]   # [B, L, obs_dim]
    a = state.a[idx]       # [B, L]
    r = state.r[idx]       # [B, L]
    Rn = state.Rn[idx]     # [B, L]
    pi = state.pi[idx]     # [B, L, num_actions]

    batch = Transition(
        obs=obs,
        a=a,
        r=r,
        done=False,     # not used in loss
        Rn=Rn,
        v=0.0,          # not used in loss
        pi=pi,
        w=1.0,          # not used (training doesn’t use w)
    )
    return batch, key
