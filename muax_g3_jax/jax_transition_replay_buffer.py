import jax
import jax.numpy as jnp
from flax import struct

from episode_tracer import Transition


@struct.dataclass
class JaxTransitionReplayBufferState:
    """Transition-level JAX replay buffer.

    Stores per-step transitions in a ring buffer. Sampling builds
    segments of length k_steps at query time.

    Fields:
      obs: [capacity, obs_dim]
      a:   [capacity]
      r:   [capacity]
      Rn:  [capacity]
      pi:  [capacity, num_actions]
      w:   [capacity]   # per-transition priority

      head: index of next write position
      size: number of valid transitions (<= capacity)

      capacity: ring buffer capacity (max number of transitions)
      k_steps:  segment length for sampling
    """
    obs: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    Rn: jnp.ndarray
    pi: jnp.ndarray
    w: jnp.ndarray

    head: jnp.ndarray       # ()
    size: jnp.ndarray       # ()

    capacity: int
    k_steps: int


def jax_trans_replay_init(
    capacity: int,
    obs_dim: int,
    num_actions: int,
    k_steps: int,
) -> JaxTransitionReplayBufferState:
    """Initialize empty transition-level replay buffer."""
    return JaxTransitionReplayBufferState(
        obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        a=jnp.zeros((capacity,), dtype=jnp.int32),
        r=jnp.zeros((capacity,), dtype=jnp.float32),
        Rn=jnp.zeros((capacity,), dtype=jnp.float32),
        pi=jnp.zeros((capacity, num_actions), dtype=jnp.float32),
        w=jnp.zeros((capacity,), dtype=jnp.float32),
        head=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
        capacity=capacity,
        k_steps=k_steps,
    )


# ---------------------------------------------------------------------------
# JAX core: add transition as raw arrays (jit/scan-safe)
# ---------------------------------------------------------------------------

def _add_transition_arrays(
    state: JaxTransitionReplayBufferState,
    obs: jnp.ndarray,       # [obs_dim]
    a: jnp.ndarray,         # scalar
    r: jnp.ndarray,         # scalar
    Rn: jnp.ndarray,        # scalar
    pi: jnp.ndarray,        # [num_actions]
    w: jnp.ndarray,         # scalar
) -> JaxTransitionReplayBufferState:
    """JAX core: add a single transition (arrays) into the ring buffer."""
    obs = jnp.asarray(obs, dtype=jnp.float32)
    a = jnp.asarray(a, dtype=jnp.int32)
    r = jnp.asarray(r, dtype=jnp.float32)
    Rn = jnp.asarray(Rn, dtype=jnp.float32)
    pi = jnp.asarray(pi, dtype=jnp.float32)
    w = jnp.asarray(w, dtype=jnp.float32)

    idx = state.head
    cap = state.capacity

    new_obs = state.obs.at[idx].set(obs)
    new_a = state.a.at[idx].set(a)
    new_r = state.r.at[idx].set(r)
    new_Rn = state.Rn.at[idx].set(Rn)
    new_pi = state.pi.at[idx].set(pi)
    new_w = state.w.at[idx].set(w)

    new_head = (idx + 1) % cap
    new_size = jnp.minimum(state.size + 1, cap)

    return state.replace(
        obs=new_obs,
        a=new_a,
        r=new_r,
        Rn=new_Rn,
        pi=new_pi,
        w=new_w,
        head=new_head,
        size=new_size,
    )


def jax_trans_replay_add_transition(
    state: JaxTransitionReplayBufferState,
    trans: Transition,
) -> JaxTransitionReplayBufferState:
    """Python-friendly wrapper: add a Transition (from JaxPNStep) to the buffer.

    Expected shapes:
      trans.obs: [obs_dim]
      trans.a:   scalar
      trans.r:   scalar
      trans.Rn:  scalar
      trans.pi:  [num_actions]
      trans.w:   scalar
    """
    return _add_transition_arrays(
        state,
        obs=trans.obs,
        a=trans.a,
        r=trans.r,
        Rn=trans.Rn,
        pi=trans.pi,
        w=trans.w,
    )


# ---------------------------------------------------------------------------
# JAX core: sample segments (jit/scan-safe)
# ---------------------------------------------------------------------------

def _sample_segments_core(
    state: JaxTransitionReplayBufferState,
    key: jax.Array,
    batch_size: int,
):
    """JAX core: sample segments [B, L, ...] from the transition buffer.

    No Python int/if logic — safe to use inside jit/scan.

    Assumes:
      state.size >= state.k_steps

    This function:
      - builds a logical view of transitions in time order,
      - samples starting positions of valid segments,
      - constructs segments of length L = k_steps.
    """
    size = state.size            # scalar int32 (JAX)
    cap = state.capacity
    L = state.k_steps

    # Logical index of oldest transition (head is next write).
    start = (state.head - size) % cap

    # Physical indices for logical positions [0..size-1]
    idxs = (start + jnp.arange(size, dtype=jnp.int32)) % cap   # [size]

    # Logical views
    obs_log = state.obs[idxs]   # [size, obs_dim]
    a_log = state.a[idxs]       # [size]
    r_log = state.r[idxs]       # [size]
    Rn_log = state.Rn[idxs]     # [size]
    pi_log = state.pi[idxs]     # [size, num_actions]
    w_log = state.w[idxs]       # [size]

    # Number of valid starting positions for segments of length L
    # valid_starts = size - L + 1
    valid_starts = size - L + 1  # scalar int32

    # Use weights from these start positions
    w_starts = w_log[:valid_starts]  # [valid_starts]

    # Avoid zero-sum weights: fall back to uniform if needed
    # If all weights <= 0, replace them with ones before soft normalisation.
    w_starts = jnp.where(w_starts <= 0.0, jnp.ones_like(w_starts), w_starts)
    probs = w_starts / jnp.sum(w_starts)

    # Sample starting indices
    key, subkey = jax.random.split(key)
    start_idx = jax.random.choice(
        subkey,
        valid_starts,
        shape=(batch_size,),
        p=probs,
    )  # [B]

    # For each start s, build indices s..s+L-1
    rel = jnp.arange(L, dtype=jnp.int32)           # [L]
    seg_idx = start_idx[:, None] + rel[None, :]    # [B, L]

    obs_batch = obs_log[seg_idx]   # [B, L, obs_dim]
    a_batch = a_log[seg_idx]       # [B, L]
    r_batch = r_log[seg_idx]       # [B, L]
    Rn_batch = Rn_log[seg_idx]     # [B, L]
    pi_batch = pi_log[seg_idx]     # [B, L, num_actions]

    batch = Transition(
        obs=obs_batch,
        a=a_batch,
        r=r_batch,
        done=False,   # not used by loss
        Rn=Rn_batch,
        v=0.0,        # not used by loss
        pi=pi_batch,
        w=1.0,        # not used by loss
    )
    return batch, key


def jax_trans_replay_sample_segments(
    state: JaxTransitionReplayBufferState,
    key: jax.Array,
    batch_size: int,
):
    """Python-friendly wrapper around the JAX core sampler.

    This is what you call from your Python training loop. It does a
    simple error check using int(state.size) and then delegates to the
    jit/scan-safe `_sample_segments_core`.

    Inside a jitted train_step, you should call `_sample_segments_core`
    directly (and ensure state.size >= state.k_steps via warmup logic).
    """
    size_int = int(state.size)
    if size_int < state.k_steps:
        raise RuntimeError(
            f"Not enough transitions ({size_int}) to sample segments of length {state.k_steps}"
        )

    return _sample_segments_core(state, key, batch_size)
