from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
from jax import Array
from flax import nnx, struct
import optax
import matplotlib.pyplot as plt
import os
"""Pure-JAX CartPole-v1 environment (no gym / gymnax dependency). Inspired by Gymnax implementation (https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/classic_control/cartpole.py)"""

PRNGKey = jax.Array

class Discrete:
    """Simple discrete space, similar to gym/gymnax Discrete."""

    def __init__(self, n: int):
        self.n = int(n)
        self.shape = ()
        self.dtype = jnp.int32

    def sample(self, key: PRNGKey) -> jax.Array:
        return jax.random.randint(
            key,
            shape=(),
            minval=0,
            maxval=self.n,
            dtype=self.dtype,
        )


class Box:
    """Simple continuous box space."""

    def __init__(self, low, high, shape=None, dtype=jnp.float32):
        low = jnp.array(low, dtype=dtype)
        high = jnp.array(high, dtype=dtype)
        if shape is None:
            shape = low.shape
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self, key: PRNGKey) -> jax.Array:
        return jax.random.uniform(
            key,
            shape=self.shape,
            minval=self.low,
            maxval=self.high,
            dtype=self.dtype,
        )


class Dict:
    """Dictionary of spaces."""

    def __init__(self, spaces: dict[str, Any]):
        self.spaces = spaces

@struct.dataclass
class EnvState:
    x: jax.Array
    x_dot: jax.Array
    theta: jax.Array
    theta_dot: jax.Array
    time: jax.Array


@struct.dataclass
class EnvParams:
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5  # actually half the pole length
    force_mag: float = 10.0
    tau: float = 0.02  # seconds between state updates

    theta_threshold_radians: float = 12 * 2 * jnp.pi / 360.0
    x_threshold: float = 2.4

    # v1 uses 500 steps per episode (v0 used 200)
    max_steps_in_episode: int = 500


class CartPole:
    """JAX implementation of CartPole-v1 environment.

    - Pure functional API (no mutable state on the Python side).
    - Fully jittable, vmappable, and scan-able.
    - Very close to the OpenAI Gym / gymnax CartPole dynamics.
    """

    def __init__(self):
        self.obs_shape = (4,)

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for CartPole-v1."""
        return EnvParams()

    def step_env(
        self,
        key: PRNGKey,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, Any]]:
        """Performs one environment step.

        Args:
            key: PRNGKey (not used here but kept for API compatibility).
            state: current EnvState.
            action: 0 or 1 (left / right).
            params: EnvParams.

        Returns:
            obs, next_state, reward, done, info
        """
        del key

        # Reward depends on whether previous state was terminal
        prev_terminal = self.is_terminal(state, params)

        # Convert action to JAX array (0 or 1)
        action = jnp.asarray(action, dtype=jnp.float32)

        # Derived params (computed here so they stay consistent if you
        # ever change masscart/masspole/length at runtime).
        total_mass = params.masscart + params.masspole
        polemass_length = params.masspole * params.length

        # Force is either -force_mag or +force_mag
        # action=0 -> -force_mag, action=1 -> +force_mag
        force = params.force_mag * action - params.force_mag * (1.0 - action)

        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        temp = (force + polemass_length * state.theta_dot**2 * sintheta) / total_mass
        thetaacc = (params.gravity * sintheta - costheta * temp) / (
            params.length
            * (4.0 / 3.0 - params.masspole * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Euler integration
        x = state.x + params.tau * state.x_dot
        x_dot = state.x_dot + params.tau * xacc
        theta = state.theta + params.tau * state.theta_dot
        theta_dot = state.theta_dot + params.tau * thetaacc

        # Reward is based on termination of *previous* state
        reward = 1.0 - prev_terminal.astype(jnp.float32)

        # Update state and compute termination for new state
        new_state = EnvState(
            x=x,
            x_dot=x_dot,
            theta=theta,
            theta_dot=theta_dot,
            time=state.time + jnp.array(1, dtype=jnp.int32),
        )
        done = self.is_terminal(new_state, params)

        obs = self.get_obs(new_state)

        # If you want the env to be non-differentiable, stop gradients here.
        obs = jax.lax.stop_gradient(obs)
        new_state = jax.tree_util.tree_map(jax.lax.stop_gradient, new_state)

        return (
            obs,
            new_state,
            jnp.asarray(reward, dtype=jnp.float32),
            done,
            {"discount": self.discount(new_state, params)},
        )

    def reset_env(self, key: PRNGKey, params: EnvParams) -> tuple[jax.Array, EnvState]:
        """Reset environment state.

        Returns:
            obs, state
        """
        init_state_vec = jax.random.uniform(
            key, minval=-0.05, maxval=0.05, shape=(4,)
        )
        state = EnvState(
            x=init_state_vec[0],
            x_dot=init_state_vec[1],
            theta=init_state_vec[2],
            theta_dot=init_state_vec[3],
            time=jnp.array(0, dtype=jnp.int32),
        )
        obs = self.get_obs(state)
        return obs, state

    def step(
        self,
        key: PRNGKey,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams | None = None,
    ):
        """Step with optional params (defaults to self.default_params)."""
        if params is None:
            params = self.default_params
        return self.step_env(key, state, action, params)

    def reset(
        self,
        key: PRNGKey,
        params: EnvParams | None = None,
    ) -> tuple[jax.Array, EnvState]:
        """Reset with optional params (defaults to self.default_params)."""
        if params is None:
            params = self.default_params
        return self.reset_env(key, params)

    def init(
        self,
        key: PRNGKey,
        params: EnvParams | None = None,
    ) -> EnvState:
        """Gymnax-style init: returns state only (no observation)."""
        if params is None:
            params = self.default_params
        _, state = self.reset_env(key, params)
        return state

    def get_obs(
        self,
        state: EnvState,
        params: EnvParams | None = None,
        key: PRNGKey | None = None,
    ) -> jax.Array:
        """Construct observation from state."""
        del params, key  # unused, kept for API compatibility
        return jnp.stack(
            [state.x, state.x_dot, state.theta, state.theta_dot],
            axis=-1,
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        done_x = jnp.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )
        done_theta = jnp.logical_or(
            state.theta < -params.theta_threshold_radians,
            state.theta > params.theta_threshold_radians,
        )
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(jnp.logical_or(done_x, done_theta), done_steps)
        return done

    def discount(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Standard terminal-absorbing discount."""
        done = self.is_terminal(state, params).astype(jnp.float32)
        return 1.0 - done

    @property
    def name(self) -> str:
        """Environment name."""
        return "CartPole-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams | None = None) -> Discrete:
        """Action space of the environment."""
        del params
        return Discrete(2)

    def observation_space(self, params: EnvParams | None = None) -> Box:
        """Observation space of the environment."""
        if params is None:
            params = self.default_params
        high = jnp.array(
            [
                params.x_threshold * 2.0,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2.0,
                jnp.finfo(jnp.float32).max,
            ],
            dtype=jnp.float32,
        )
        return Box(-high, high, (4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams | None = None) -> Dict:
        """State space of the environment."""
        if params is None:
            params = self.default_params
        high = jnp.array(
            [
                params.x_threshold * 2.0,
                jnp.finfo(jnp.float32).max,
                params.theta_threshold_radians * 2.0,
                jnp.finfo(jnp.float32).max,
            ],
            dtype=jnp.float32,
        )
        return Dict(
            {
                "x": Box(-high[0], high[0], (), jnp.float32),
                "x_dot": Box(-high[1], high[1], (), jnp.float32),
                "theta": Box(-high[2], high[2], (), jnp.float32),
                "theta_dot": Box(-high[3], high[3], (), jnp.float32),
                "time": Discrete(params.max_steps_in_episode + 1),
            }
        )
    def render(
        self,
        state: EnvState,
        params: EnvParams | None = None,
        screen_height: int = 400,
        screen_width: int = 600,
    ) -> jax.Array:
        """Pure-JAX RGB render of the CartPole state.
        
        Returns:
            (H, W, 3) uint8 image array.
        """
        if params is None:
            params = self.default_params

        H = int(screen_height)
        W = int(screen_width)

        # ----------------------------------------------------------------------------
        # Coordinate grid (image space, y downwards, x rightwards)
        # ----------------------------------------------------------------------------
        ys = jnp.arange(H, dtype=jnp.float32)  # rows
        xs = jnp.arange(W, dtype=jnp.float32)  # cols
        grid_y = ys[:, None]                   # (H,1)
        grid_x = xs[None, :]                   # (1,W)

        # Background: white
        img = jnp.ones((H, W, 3), dtype=jnp.float32) * 255.0

        def paint(mask: jax.Array, color_rgb, base_img: jax.Array) -> jax.Array:
            """Paint color on base_img where mask is True."""
            color = jnp.array(color_rgb, dtype=jnp.float32)
            return jnp.where(mask[..., None], color, base_img)

        # ----------------------------------------------------------------------------
        # Map world x (state.x in [-x_threshold, x_threshold]) to pixel x
        # ----------------------------------------------------------------------------
        x_threshold = jnp.asarray(params.x_threshold, dtype=jnp.float32)
        world_width = 2.0 * x_threshold
        scale = jnp.asarray(W, jnp.float32) / world_width

        cartwidth = 50.0
        cartheight = 30.0
        polewidth = 10.0
        polelen = scale * (2.0 * jnp.asarray(params.length, jnp.float32))
        axleoffset = cartheight / 4.0

        # Cart horizontal position
        cartx = state.x * scale + jnp.asarray(W, jnp.float32) / 2.0

        # Use "top-of-cart" y coordinate like Gym (but in image coordinates)
        cart_top = jnp.asarray(H, jnp.float32) * 0.75
        cart_bottom = cart_top + cartheight
        cart_center_y = (cart_top + cart_bottom) / 2.0

        # ----------------------------------------------------------------------------
        # Track (ground) line
        # ----------------------------------------------------------------------------
        ground_y = cart_bottom + 5.0
        ground_mask = jnp.abs(grid_y - ground_y) <= 0.5
        img = paint(ground_mask, (0, 0, 0), img)

        # ----------------------------------------------------------------------------
        # Cart rectangle
        # ----------------------------------------------------------------------------
        cart_left = cartx - cartwidth / 2.0
        cart_right = cartx + cartwidth / 2.0

        in_cart = (
            (grid_x >= cart_left)
            & (grid_x <= cart_right)
            & (grid_y >= cart_top)
            & (grid_y <= cart_bottom)
        )
        img = paint(in_cart, (0, 0, 0), img)

        # ----------------------------------------------------------------------------
        # Pole as a rotated rectangle around pivot at top center of cart
        # ----------------------------------------------------------------------------
        pivot_x = cartx
        pivot_y = cart_top + axleoffset

        # Direction of the pole. In the dynamics, theta=0 is vertical;
        # here we define an angle so that theta=0 means "pole straight up".
        angle = state.theta - jnp.pi / 2.0  # rotate so 0 => pointing up
        ux = jnp.cos(angle)                 # pole direction x
        uy = jnp.sin(angle)                 # pole direction y

        # Vector from pivot to each pixel
        vx = grid_x - pivot_x
        vy = grid_y - pivot_y

        # Projection of each pixel onto pole axis (0 at pivot)
        proj = vx * ux + vy * uy
        # Perpendicular distance from pole axis
        perp = vx * uy - vy * ux

        half_width = polewidth / 2.0
        in_pole = (
            (proj >= 0.0)
            & (proj <= polelen)
            & (jnp.abs(perp) <= half_width)
        )

        # Pole color (similar to Gym's brownish pole)
        img = paint(in_pole, (202, 152, 101), img)

        # ----------------------------------------------------------------------------
        # Axle (circle at pivot)
        # ----------------------------------------------------------------------------
        axle_radius = polewidth / 2.0
        dx = grid_x - pivot_x
        dy = grid_y - pivot_y
        in_axle = dx**2 + dy**2 <= axle_radius**2

        img = paint(in_axle, (129, 132, 203), img)

        # Clip and cast
        img = jnp.clip(img, 0.0, 255.0)
        return img.astype(jnp.uint8)