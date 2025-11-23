from typing import NamedTuple, Optional
from functools import partial
import jax
from jax import numpy as jnp 
import mctx 
from mctx import qtransform_by_parent_and_siblings, qtransform_completed_by_mix_value
import optax
import haiku as hk 

import warnings

from utils import scale_gradient, scalar_to_support, support_to_scalar
from loss import default_loss_fn


class MZNetworkParams(NamedTuple):
    representation: Optional[hk.Params] = None
    prediction: Optional[hk.Params] = None
    dynamic: Optional[hk.Params] = None


def optimizer(init_value=0,
              peak_value=2e-2, 
              end_value=1e-3,
              warmup_steps=1000,
              transition_steps=10000,
              decay_rate=0.8,
              clip_by_global_norm=1.0
              ):
  r"""
    Initializes an optax optimizer that uses adam to update weights, 
    `optax.warmup_exponential_decay_schedule` to schedule the learning rate, and clip the gradient by the global norm(`optax.clip_by_global_norm`)
    This optimizer seems to be more stable.

    Parameters
    ----------
    init_value : float, initial value for the scalar to be annealed.

    peak_value : float, peak value for scalar to be annealed at end of warmup.

    end_value : float, the value at which the exponential decay stops. When
    `decay_rate` < 1, `end_value` is treated as a lower bound, otherwise as an upper bound. Has no effect when `decay_rate` = 0.

    warmup_steps: int, positive integer, the length of the linear warmup.

    transition_steps: int, positive integer, for `exponential decay`

    decay_rate: float, must not be zero. The decay rate.

    clip_by_global_norm: float, the maximum global norm for an update. Clips updates using their global norm.
    
    Returns
    -------
    gradient_transform: GradientTransformation. A single (init_fn, update_fn) tuple.
    """
  scheduler = optax.warmup_exponential_decay_schedule(
    init_value=init_value, 
    peak_value=peak_value,
    end_value=end_value,
    warmup_steps=warmup_steps,
    transition_steps=transition_steps,
    decay_rate=decay_rate)
  gradient_transform = optax.chain(
      optax.clip_by_global_norm(clip_by_global_norm),  # Clip by the gradient by the global norm.
      optax.scale_by_adam(),  # Use the updates from adam.
      optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
      # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
      optax.scale(-1.0)
  )
  return gradient_transform



class MuZero:
  r"""Muzero algorithm
    
    Parameters
    ----------
    representation_fn: A function initialized from a class which inherents hk.Module, 
        which takes raw observation `obs` from the environment as input and outputs the hidden state `s`.
        `s` will be the input of prediction_fn and dynamic_fn. 
        The first dimension of the `obs` is the batch dimension.
    
    prediction_fn: A function initialized from a class which inherents hk.Module, 
        which takes hidden state `s` as input and outputs prior logits `logits` and value `v` of the state.
    
    dynamic_fn: A function initialized from a class which inherents hk.Module,
        which takes hidden state `s` and action `a` as input and outputs reward `r` and next hidden state `ns`.
    
    policy: str, value in `['muzero', 'gumbel']`. Determines which muzero policy in `mctx` to use. 
    
    optimizer: Optimizer to update the parameters of `representation_fn`, `prediction_fn` and `dynamic_fn`.
    
    loss_fn: Callable, computes loss for the MuZero model. The default is `default_loss_fn`.
    
    discount: Any. Used for mctx.RecurrentFnOutput.

    support_size: int, the `support_size` for `scalar_to_support`, 
        the scale is nearly square root, that is, if the scalar is ~100, `support_size`=10 might be sufficient.
  """
  def __init__(self, 
               representation_fn,
               prediction_fn,
               dynamic_fn,
               policy='muzero',
               optimizer = optimizer(),
               loss_fn = default_loss_fn,
               discount: float = 0.99,
               support_size: int = 10
               ):
     
    self.repr_func = hk.without_apply_rng(hk.transform(representation_fn))
    self.pred_func = hk.without_apply_rng(hk.transform(prediction_fn))
    self.dy_func = hk.without_apply_rng(hk.transform(dynamic_fn))
    
    self._policy = self._init_policy(policy)
    self._policy_type = policy
    self._optimizer = optimizer 
    self.loss_fn = partial(loss_fn, self) if loss_fn else self._loss_fn
    # partial(default_loss_fn, muzero_instance=self)
    self._discount = discount
    self._support_size = support_size
  
  def init(self, rng_key, sample_input):
    """Inits `representation`, `prediction` and `dynamic` functions and optimizer
    
    Parameters
    ----------
    rng_key: jax.random.PRNGKey.
    sample_input: Array. The dimension is `[B, ...]` where B is the batch dimension.
    
    Returns
    ----------
    params: MZNetworkParams. {'representation': repr_params, 'prediction': pred_params, 'dynamic': dy_params}
    """
    repr_params = self.repr_func.init(rng_key, sample_input)
    s = self.repr_func.apply(repr_params, sample_input)
    pred_params = self.pred_func.init(rng_key, s)
    dy_params = self.dy_func.init(rng_key, s, jnp.zeros(s.shape[0]))
    self._params = MZNetworkParams(repr_params, pred_params, dy_params)
    self._opt_state = self._optimizer.init(self._params)
    return self._params 

  def representation(self, obs):
    r"""
    Passes the observation `obs` through MuZero's `representation_fn`.

    Parameters
    ----------
    obs: array, the first dimension is the batch dimension.
    
    Returns
    ----------
    s: jnp.array, the hidden state
    """
    s = self._repr_apply(self.params.representation, obs)
    return s 
  
  def prediction(self, s):
    r"""
    Passes the hidden state `s` through MuZero's `prediction_fn`.

    Parameters
    ----------
    s: array, the first dimension is the batch dimension.
    
    Returns
    ----------
    v: jnp.array, the value predicted 
    logits: jnp.array, the action logits predicted
    """
    v, logits = self._pred_apply(self.params.prediction, s)
    return v, logits

  def dynamic(self, s, a):
    r"""
    Passes the hidden state `s` and action `a` through MuZero's `dynamic_fn`.

    Parameters
    ----------
    s: array, the first dimension is the batch dimension.

    a: array, the first dimension is the batch dimension.
    
    Returns
    ----------
    r: jnp.array, the reward estimated
    ns: jnp.array, the next state estimated
    """
    r, ns = self._dy_apply(self.params.dynamic, s, a)
    return r, ns
  def act_from_params(self,
                        params,
                        rng_key,
                        obs,
                        num_simulations: int = 50,
                        temperature: float = 1.0):
    
      """Pure-style act: takes explicit params, returns (a, pi, v) as JAX arrays.

      This is the one we’ll eventually use inside jitted train_step / scan.
      """
      # Ensure batch dimension for _plan: [B, ...]
      if obs.ndim == 1:
          obs = obs[None, :]  # [1, obs_dim]

      plan_output, root_value = self._plan(
          params,
          rng_key,
          obs,
          num_simulations=num_simulations,
          temperature=temperature,
          invalid_actions=None,
          max_depth=None,
          loop_fn=jax.lax.fori_loop,
          qtransform=None,
          dirichlet_fraction=0.25,
          dirichlet_alpha=0.3,
          pb_c_init=1.25,
          pb_c_base=19652,
          max_num_considered_actions=16,
          gumbel_scale=1.0,
      )

      # Unbatch B=1, keep as jax.Array scalars/vectors.
      action = plan_output.action[0]          # shape: ()
      pi = plan_output.action_weights[0]      # shape: [num_actions]
      v = root_value[0]                       # shape: ()

      return action, pi, v
  
  def act(self, rng_key, obs, with_pi: bool = False, with_value: bool = False, obs_from_batch: bool = False, num_simulations: int = 50, temperature: float = 1., invalid_actions=None, max_depth: int = None, loop_fn = jax.lax.fori_loop,
        qtransform=None,
        dirichlet_fraction: float = 0.25,
        dirichlet_alpha: float = 0.3,
        pb_c_init: float = 1.25,
        pb_c_base: float = 19652,
        max_num_considered_actions: int = 16,
        gumbel_scale: float = 1):
    
    r"""Acts given environment's observations.

    Parameters are the same as before; this version keeps everything as
    JAX arrays instead of converting to Python/NumPy scalars.
    """
    # Ensure a batch dim if we're acting on a single observation.
    if not obs_from_batch:
      obs = jnp.expand_dims(obs, axis=0)

    plan_output, root_value = self._plan(
        self.params,
        rng_key,
        obs,
        num_simulations,
        temperature,
        invalid_actions=invalid_actions,
        max_depth=max_depth,
        loop_fn=loop_fn,
        qtransform=qtransform,
        dirichlet_fraction=dirichlet_fraction,
        dirichlet_alpha=dirichlet_alpha,
        pb_c_init=pb_c_init,
        pb_c_base=pb_c_base,
        max_num_considered_actions=max_num_considered_actions,
        gumbel_scale=gumbel_scale,
    )

    if not obs_from_batch:
      # Unbatched case: return scalars / 1D arrays for a single env.
      action = plan_output.action[0]          # shape: ()
      root_value = root_value[0]             # shape: ()
      if with_pi:
        action_weights = plan_output.action_weights[0]  # shape: [num_actions]
    else:
      # Batched case: keep full batch dimension.
      action = plan_output.action             # shape: [B]
      root_value = root_value                # shape: [B]
      if with_pi:
        action_weights = plan_output.action_weights  # shape: [B, num_actions]

    # Return in the same combinations as before, but as JAX arrays.
    if with_pi and with_value:
      return action, action_weights, root_value
    elif not with_pi and with_value:
      return action, root_value
    elif with_pi and not with_value:
      return action, action_weights
    else:
      return action


  @partial(jax.jit, static_argnums=(0,))
  def update_from_params(self, params, opt_state, batch):
    """Pure update: (params, opt_state, batch) -> (new_params, new_opt_state, loss).

    This does *not* mutate self._params / self._opt_state, so it’s safe to
    use inside future jitted train_step / scan.
    """
    # loss_fn already has `self` partially applied: loss_fn(params, batch)
    loss, grads = jax.value_and_grad(self.loss_fn)(params, batch)
    params, opt_state = self._update(params, opt_state, grads)
    return params, opt_state, loss

  def update(self, batch):
    r"""Updates model parameters given a batch of trajectories (stateful API).

    This wraps `update_from_params` and mutates `self._params` and
    `self._opt_state` for the existing training loop.
    """
    self._params, self._opt_state, loss = self.update_from_params(
        self._params, self._opt_state, batch
    )
    loss_metric = {"loss": float(loss)}
    return loss_metric

  
  def save(self, file):
    """Saves model parameters and optimizer state to the file"""
    to_save = {'params': self.params, 'optimizer_state': self.optimizer_state}
    jnp.save(file, to_save)
  
  def load(self, file):
    """Loads model parameters and optimizer state from the saved file"""
    if not file.endswith('.npy'):
      file = f'{file}.npy'
    saved = jnp.load(file, allow_pickle=True).item()
    self._params, self._opt_state = saved['params'], saved['optimizer_state']

  @property
  def params(self):
    return self._params

  @property
  def optimizer_state(self):
    return self._opt_state

  def _plan(self, params, rng_key, obs,
           num_simulations: int = 5,
           temperature: float = 1.,
          invalid_actions=None,
          max_depth: int = None, 
          loop_fn = jax.lax.fori_loop,
          qtransform=None, 
          dirichlet_fraction: float = 0.25, 
          dirichlet_alpha: float = 0.3, 
          pb_c_init: float = 1.25, 
          pb_c_base: float = 19652,
          max_num_considered_actions: int = 16,
          gumbel_scale: float = 1):
    root = self._root_inference(params, rng_key, obs)
  
    if qtransform is None:
      qtransform = qtransform_by_parent_and_siblings
    plan_output = self._policy(params, rng_key, root, self._recurrent_inference,
                              num_simulations=num_simulations,
                              temperature=temperature,
                              invalid_actions=invalid_actions,
                              max_depth=max_depth, 
                              loop_fn=loop_fn,
                              qtransform=qtransform, 
                              dirichlet_fraction=dirichlet_fraction, 
                              dirichlet_alpha=dirichlet_alpha, 
                              pb_c_init=pb_c_init, 
                              pb_c_base=pb_c_base)
    # elif self._policy_type == 'gumbel':
    #   if qtransform is None:
    #     qtransform = qtransform_completed_by_mix_value
    #   plan_output = self._policy(params, rng_key, root, self._recurrent_inference,
    #                             num_simulations=num_simulations,
    #                             invalid_actions=invalid_actions, 
    #                             max_depth=max_depth, 
    #                             loop_fn=loop_fn, 
    #                             qtransform=qtransform, 
    #                             max_num_considered_actions=max_num_considered_actions, 
    #                             gumbel_scale=gumbel_scale)
    return plan_output, root.value
    
  @partial(jax.jit, static_argnums=(0,))
  def _update(self, params, optimizer_state, grads):
    updates, optimizer_state = self._optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state
  
  @partial(jax.jit, static_argnums=(0,))
  def _loss_fn(self, params, batch):
    """Fallback loss if no custom loss_fn is provided.

    We simply delegate to the shared default_loss_fn to keep behaviour
    and shapes consistent.
    """
    return default_loss_fn(self, params, batch)


  @partial(jax.jit, static_argnums=(0,))
  def _root_inference(self, params, rng_key, obs):
    r"""Given the observation, a (prior_logits, value, embedding) RootFnOutput is estimated. The
    prior_logits are from a policy network. The shapes are ([B, num_actions], [B], [B, ...]), respectively."""
    s = self._repr_apply(params.representation, obs)
    v, logits = self._pred_apply(params.prediction, s)  
    v = support_to_scalar(jax.nn.softmax(v), self._support_size).flatten()
    root = mctx.RootFnOutput(
        prior_logits=logits,
        value=v,
        embedding=s
    )
    return root 

  @partial(jax.jit, static_argnums=(0,))
  def _recurrent_inference(self, params, rng_key, action, embedding):
    r"""To be called on the leaf nodes and unvisited actions retrieved by the simulation step,
    which takes as args (params, rng_key, action, embedding) and returns a `RecurrentFnOutput` and the new state embedding.
    The rng_key argument is consumed.
    """
    r, next_embedding = self._dy_apply(params.dynamic, embedding, action)
    v, logits = self._pred_apply(params.prediction, embedding)
    r = support_to_scalar(jax.nn.softmax(r), self._support_size).flatten()
    v = support_to_scalar(jax.nn.softmax(v), self._support_size).flatten()
    discount = jnp.ones_like(r) * self._discount
    recurrent_output = mctx.RecurrentFnOutput(
        reward=r,
        discount=discount,
        prior_logits=logits,
        value=v 
    )
    return recurrent_output, next_embedding
  
  @partial(jax.jit, static_argnums=(0,))
  def _repr_apply(self, repr_params, obs):
    s = self.repr_func.apply(repr_params, obs)
    return s

  @partial(jax.jit, static_argnums=(0,))
  def _pred_apply(self, pred_params, s):
    v, logits = self.pred_func.apply(pred_params, s)
    return v, logits

  @partial(jax.jit, static_argnums=(0,))
  def _dy_apply(self, dy_params, s, a):
    r, ns = self.dy_func.apply(dy_params, s, a)
    return r, ns

  def _init_policy(self, policy):
    # For MuZero, use mctx.muzero_policy
    policy_func = mctx.muzero_policy
    return jax.jit(
        policy_func,
        static_argnames=(
            "recurrent_fn",
            "num_simulations",
            "loop_fn",
            "qtransform",
            "max_depth",
        ),
        backend="gpu",
    )

    # elif policy == 'gumbel':
    #   policy_func = mctx.gumbel_muzero_policy
    #   return jax.jit(policy_func, static_argnames=('recurrent_fn', 'num_simulations', 'max_num_considered_actions', 'gumbel_scale', 'loop_fn', 'qtransform', 'max_depth'), 
    #                  backend='cpu')
    
def make_model_act(muzero, num_simulations, temperature):
    def model_act(params, rng_key, obs):
        return muzero.act_from_params(params, rng_key, obs,
                                      num_simulations=num_simulations,
                                      temperature=temperature)
    return model_act
class StochasticMuZero:
  r"""Stochastic Muzero algorithm
    
    Parameters
    ----------
    representation_fn: A function initialized from a class which inherents hk.Module, 
        which takes raw observation `obs` from the environment as input and outputs the hidden state `s`.
        `s` will be the input of prediction_fn and dynamic_fn. 
        The first dimension of the `obs` is the batch dimension.
    
    prediction_fn: A function initialized from a class which inherents hk.Module, 
        which takes hidden state `s` as input and outputs prior logits `logits` and value `v` of the state.
    
    dynamic_fn: A function initialized from a class which inherents hk.Module,
        which takes hidden state `s` and action `a` as input and outputs reward `r` and next hidden state `ns`.
    
    policy: str, value in `['muzero', 'gumbel']`. Determines which muzero policy in `mctx` to use. 
    
    optimizer: Optimizer to update the parameters of `representation_fn`, `prediction_fn` and `dynamic_fn`.
    
    loss_fn: Callable, computes loss for the MuZero model. The default is `default_loss_fn`.
    
    discount: Any. Used for mctx.RecurrentFnOutput.

    support_size: int, the `support_size` for `scalar_to_support`, 
        the scale is nearly square root, that is, if the scalar is ~100, `support_size`=10 might be sufficient.
  """
  def __init__(self, 
               representation_fn,
               prediction_fn,
               dynamic_fn,
               policy='muzero',
               optimizer = optimizer(),
               loss_fn = default_loss_fn,
               discount: float = 0.99,
               support_size: int = 10
               ):
     
    self.repr_func = hk.without_apply_rng(hk.transform(representation_fn))
    self.pred_func = hk.without_apply_rng(hk.transform(prediction_fn))
    self.dy_func = hk.without_apply_rng(hk.transform(dynamic_fn))
    
    self._policy = self._init_policy(policy)
    self._policy_type = policy
    self._optimizer = optimizer 
    self.loss_fn = partial(loss_fn, self) if loss_fn else self._loss_fn
    # partial(default_loss_fn, muzero_instance=self)
    self._discount = discount
    self._support_size = support_size
  
  def init(self, rng_key, sample_input):
    """Inits `representation`, `prediction` and `dynamic` functions and optimizer
    
    Parameters
    ----------
    rng_key: jax.random.PRNGKey.
    sample_input: Array. The dimension is `[B, ...]` where B is the batch dimension.
    
    Returns
    ----------
    params: MZNetworkParams. {'representation': repr_params, 'prediction': pred_params, 'dynamic': dy_params}
    """
    repr_params = self.repr_func.init(rng_key, sample_input)
    s = self.repr_func.apply(repr_params, sample_input)
    pred_params = self.pred_func.init(rng_key, s)
    dy_params = self.dy_func.init(rng_key, s, jnp.zeros(s.shape[0]))
    self._params = MZNetworkParams(repr_params, pred_params, dy_params)
    self._opt_state = self._optimizer.init(self._params)
    return self._params 

  def representation(self, obs):
    r"""
    Passes the observation `obs` through MuZero's `representation_fn`.

    Parameters
    ----------
    obs: array, the first dimension is the batch dimension.
    
    Returns
    ----------
    s: jnp.array, the hidden state
    """
    s = self._repr_apply(self.params.representation, obs)
    return s 
  
  def prediction(self, s):
    r"""
    Passes the hidden state `s` through MuZero's `prediction_fn`.

    Parameters
    ----------
    s: array, the first dimension is the batch dimension.
    
    Returns
    ----------
    v: jnp.array, the value predicted 
    logits: jnp.array, the action logits predicted
    """
    v, logits = self._pred_apply(self.params.prediction, s)
    return v, logits

  def dynamic(self, s, a):
    r"""
    Passes the hidden state `s` and action `a` through MuZero's `dynamic_fn`.

    Parameters
    ----------
    s: array, the first dimension is the batch dimension.

    a: array, the first dimension is the batch dimension.
    
    Returns
    ----------
    r: jnp.array, the reward estimated
    ns: jnp.array, the next state estimated
    """
    r, ns = self._dy_apply(self.params.dynamic, s, a)
    return r, ns

  def act(self, rng_key, obs,
        with_pi: bool = False,
        with_value: bool = False,
        obs_from_batch: bool = False,
        num_simulations: int = 5,
        temperature: float = 1.,
        invalid_actions=None,
        max_depth: int = None,
        loop_fn = jax.lax.fori_loop,
        qtransform=None,
        dirichlet_fraction: float = 0.25,
        dirichlet_alpha: float = 0.3,
        pb_c_init: float = 1.25,
        pb_c_base: float = 19652,
        max_num_considered_actions: int = 16,
        gumbel_scale: float = 1):
    r"""Acts given environment's observations for Stochastic MuZero.

    Same semantics as `MuZero.act`, but using the stochastic policy.
    """

    if not obs_from_batch:
      obs = jnp.expand_dims(obs, axis=0)

    plan_output, root_value = self._plan(
        self.params,
        rng_key,
        obs,
        num_simulations,
        temperature,
        invalid_actions=invalid_actions,
        max_depth=max_depth,
        loop_fn=loop_fn,
        qtransform=qtransform,
        dirichlet_fraction=dirichlet_fraction,
        dirichlet_alpha=dirichlet_alpha,
        pb_c_init=pb_c_init,
        pb_c_base=pb_c_base,
        max_num_considered_actions=max_num_considered_actions,
        gumbel_scale=gumbel_scale,
    )

    if not obs_from_batch:
      action = plan_output.action[0]          # shape: ()
      root_value = root_value[0]             # shape: ()
      if with_pi:
        action_weights = plan_output.action_weights[0]  # shape: [num_actions]
    else:
      action = plan_output.action             # [B]
      root_value = root_value                # [B]
      if with_pi:
        action_weights = plan_output.action_weights  # [B, num_actions]

    if with_pi and with_value:
      return action, action_weights, root_value
    elif not with_pi and with_value:
      return action, root_value
    elif with_pi and not with_value:
      return action, action_weights
    else:
      return action


  @partial(jax.jit, static_argnums=(0,))
  def update_from_params(self, params, opt_state, batch):
    """Pure update: (params, opt_state, batch) -> (new_params, new_opt_state, loss).

    This does *not* mutate self._params / self._opt_state, so it’s safe to
    use inside future jitted train_step / scan.
    """
    # loss_fn already has `self` partially applied: loss_fn(params, batch)
    loss, grads = jax.value_and_grad(self.loss_fn)(params, batch)
    params, opt_state = self._update(params, opt_state, grads)
    return params, opt_state, loss

  def update(self, batch):
    r"""Updates model parameters given a batch of trajectories (stateful API).

    This wraps `update_from_params` and mutates `self._params` and
    `self._opt_state` for the existing training loop.
    """
    self._params, self._opt_state, loss = self.update_from_params(
        self._params, self._opt_state, batch
    )
    loss_metric = {"loss": float(loss)}
    return loss_metric

  
  def save(self, file):
    """Saves model parameters and optimizer state to the file"""
    to_save = {'params': self.params, 'optimizer_state': self.optimizer_state}
    jnp.save(file, to_save)
  
  def load(self, file):
    """Loads model parameters and optimizer state from the saved file"""
    if not file.endswith('.npy'):
      file = f'{file}.npy'
    saved = jnp.load(file, allow_pickle=True).item()
    self._params, self._opt_state = saved['params'], saved['optimizer_state']

  @property
  def params(self):
    return self._params

  @property
  def optimizer_state(self):
    return self._opt_state

  def _plan(self, params, rng_key, obs,
           num_simulations: int = 5,
           temperature: float = 1.,
          invalid_actions=None,
          max_depth: int = None, 
          loop_fn = jax.lax.fori_loop,
          qtransform=None, 
          dirichlet_fraction: float = 0.25, 
          dirichlet_alpha: float = 0.3, 
          pb_c_init: float = 1.25, 
          pb_c_base: float = 19652,
          max_num_considered_actions: int = 16,
          gumbel_scale: float = 1):
    root = self._root_inference(params, rng_key, obs)
  
    if qtransform is None:
      qtransform = qtransform_by_parent_and_siblings
    plan_output = self._policy(params, rng_key, root, self._recurrent_inference,
                              num_simulations=num_simulations,
                              temperature=temperature,
                              invalid_actions=invalid_actions,
                              max_depth=max_depth, 
                              loop_fn=loop_fn,
                              qtransform=qtransform, 
                              dirichlet_fraction=dirichlet_fraction, 
                              dirichlet_alpha=dirichlet_alpha, 
                              pb_c_init=pb_c_init, 
                              pb_c_base=pb_c_base)
    # elif self._policy_type == 'gumbel':
    #   if qtransform is None:
    #     qtransform = qtransform_completed_by_mix_value
    #   plan_output = self._policy(params, rng_key, root, self._recurrent_inference,
    #                             num_simulations=num_simulations,
    #                             invalid_actions=invalid_actions, 
    #                             max_depth=max_depth, 
    #                             loop_fn=loop_fn, 
    #                             qtransform=qtransform, 
    #                             max_num_considered_actions=max_num_considered_actions, 
    #                             gumbel_scale=gumbel_scale)
    return plan_output, root.value
    
  @partial(jax.jit, static_argnums=(0,))
  def _update(self, params, optimizer_state, grads):
    updates, optimizer_state = self._optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state
  
  @partial(jax.jit, static_argnums=(0,))
  def _loss_fn(self, params, batch):
    """Fallback loss if no custom loss_fn is provided.

    We simply delegate to the shared default_loss_fn to keep behaviour
    and shapes consistent.
    """
    return default_loss_fn(self, params, batch)

  @partial(jax.jit, static_argnums=(0,))
  def _root_inference(self, params, rng_key, obs):
    r"""Given the observation, a (prior_logits, value, embedding) RootFnOutput is estimated. The
    prior_logits are from a policy network. The shapes are ([B, num_actions], [B], [B, ...]), respectively."""
    s = self._repr_apply(params.representation, obs)
    v, logits = self._pred_apply(params.prediction, s)  
    v = support_to_scalar(jax.nn.softmax(v), self._support_size).flatten()
    root = mctx.RootFnOutput(
        prior_logits=logits,
        value=v,
        embedding=s
    )
    return root 

  @partial(jax.jit, static_argnums=(0,))
  def _recurrent_inference(self, params, rng_key, action, embedding):
    r"""To be called on the leaf nodes and unvisited actions retrieved by the simulation step,
    which takes as args (params, rng_key, action, embedding) and returns a `RecurrentFnOutput` and the new state embedding.
    The rng_key argument is consumed.
    """
    r, next_embedding = self._dy_apply(params.dynamic, embedding, action)
    v, logits = self._pred_apply(params.prediction, embedding)
    r = support_to_scalar(jax.nn.softmax(r), self._support_size).flatten()
    v = support_to_scalar(jax.nn.softmax(v), self._support_size).flatten()
    discount = jnp.ones_like(r) * self._discount
    recurrent_output = mctx.RecurrentFnOutput(
        reward=r,
        discount=discount,
        prior_logits=logits,
        value=v 
    )
    return recurrent_output, next_embedding
  
  @partial(jax.jit, static_argnums=(0,))
  def _repr_apply(self, repr_params, obs):
    s = self.repr_func.apply(repr_params, obs)
    return s

  @partial(jax.jit, static_argnums=(0,))
  def _pred_apply(self, pred_params, s):
    v, logits = self.pred_func.apply(pred_params, s)
    return v, logits

  @partial(jax.jit, static_argnums=(0,))
  def _dy_apply(self, dy_params, s, a):
    r, ns = self.dy_func.apply(dy_params, s, a)
    return r, ns

  def _init_policy(self, policy):
    policy_func = mctx.stochastic_muzero_policy
    return jax.jit(policy_func, 
                    static_argnames=(
                      'decision_recurrent_fn', 
                      'chance_recurrent_fn', 
                      'num_simulations', 
                      'loop_fn', 
                      'qtransform', 
                      'max_depth', 
                      'dirichlet_fraction', 
                      'dirichlet_alpha', 
                      'pb_c_init', 
                      'pb_c_base', 
                      'temperature'),
                    backend='cpu')
    
    