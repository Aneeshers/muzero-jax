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

      capacity: ring buffer capacity (max number of transitions) [static]
      k_steps:  segment length for sampling [static]
    """
    obs: jnp.ndarray
    a: jnp.ndarray
    r: jnp.ndarray
    Rn: jnp.ndarray
    pi: jnp.ndarray
    w: jnp.ndarray

    head: jnp.ndarray       # ()
    size: jnp.ndarray       # ()

    # Static fields (non-pytree)
    capacity: int = struct.field(pytree_node=False)
    k_steps: int = struct.field(pytree_node=False)


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
    """
    size = state.size            # scalar int32 (dynamic)
    cap = state.capacity         # Python int (static)
    L = state.k_steps            # Python int (static)

    # Logical index of oldest transition (head is next write).
    start = (state.head - size) % cap  # scalar int32

    # Physical indices for *all* positions 0..cap-1 (static length)
    all_pos = jnp.arange(cap, dtype=jnp.int32)       # [cap]
    idxs = (start + all_pos) % cap                   # [cap]

    # Logical views of arrays, shape [cap, ...]
    obs_log = state.obs[idxs]   # [cap, obs_dim]
    a_log = state.a[idxs]       # [cap]
    r_log = state.r[idxs]       # [cap]
    Rn_log = state.Rn[idxs]     # [cap]
    pi_log = state.pi[idxs]     # [cap, num_actions]
    w_log = state.w[idxs]       # [cap]

    # Candidate starting positions for segments: s in [0, num_all_starts-1]
    num_all_starts = cap - L + 1                         # static int
    s_all = jnp.arange(num_all_starts, dtype=jnp.int32)  # [num_all_starts]

    # A start s is valid iff s + L - 1 < size  <=>  s <= size - L
    size_minus_L = size - L                              # scalar int32
    valid_mask = (s_all <= size_minus_L).astype(jnp.float32)  # [num_all_starts]

    # Raw weights at starts, but only valid positions should contribute
    w_raw = w_log[s_all]                                 # [num_all_starts]
    w_masked = w_raw * valid_mask                        # [num_all_starts]

    # Sum of masked weights
    sum_w = jnp.sum(w_masked)

    # If sum_w == 0 (degenerate), use uniform over valid starts only.
    def use_uniform(_):
        return jnp.where(valid_mask > 0.0, 1.0, 0.0)

    def use_masked(_):
        return w_masked

    w_final = jax.lax.cond(sum_w <= 0.0, use_uniform, use_masked, operand=None)

    # Normalized probabilities over all candidate starts
    probs = w_final / jnp.sum(w_final)

    # Sample starting indices
    key, subkey = jax.random.split(key)
    start_idx = jax.random.choice(
        subkey,
        num_all_starts,            # static
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
    """Python-friendly wrapper around the JAX core sampler."""
    size_int = int(state.size)
    if size_int < state.k_steps:
        raise RuntimeError(
            f"Not enough transitions ({size_int}) to sample segments of length {state.k_steps}"
        )
    return _sample_segments_core(state, key, batch_size)
