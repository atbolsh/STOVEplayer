"""Gymnax-style environment wrapper for the 3D ball-catching game."""
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from .room import RoomState, create_room, get_room_bounds, render_image
from .physics import BallState, create_balls, update_balls
from .agent_controller import (
    AgentState, create_agent, apply_action, update_agent, Action
)


class EnvParams(NamedTuple):
    """Environment parameters (configurable)."""
    num_balls: int = 5
    max_steps: int = 1000


class EnvState(NamedTuple):
    """Full environment state (immutable)."""
    agent: AgentState
    balls: BallState
    room: RoomState
    step_count: jnp.ndarray
    total_caught: jnp.ndarray


class BallCatchEnv:
    """Gymnax-style 3D ball-catching environment.
    
    The agent must navigate a 3D room and catch bouncing balls.
    Observations are 128x128x3 RGB images from first-person perspective.
    """
    
    def __init__(self, params: EnvParams = None):
        """Initialize environment with parameters.
        
        Args:
            params: Environment parameters (uses defaults if None)
        """
        self.params = params if params is not None else EnvParams()
    
    @property
    def num_actions(self) -> int:
        """Number of discrete actions."""
        return 5  # FORWARD, STOP, TURN_LEFT, TURN_RIGHT, STOP_TURNING
    
    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        """Shape of observation."""
        return (128, 128, 3)
    
    def reset(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, EnvState]:
        """Reset environment to initial state.
        
        Args:
            key: JAX random key
        
        Returns:
            (observation, state) tuple
        """
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Create room
        room = create_room(key1)
        bounds = get_room_bounds(room)
        min_x, max_x, min_z, max_z = bounds
        
        # Create agent at random position near center
        agent_x = jax.random.uniform(key2, (), minval=-0.5, maxval=0.5)
        agent_z = jax.random.uniform(key3, (), minval=-0.5, maxval=0.5)
        agent_yaw = jax.random.uniform(key2, (), minval=-jnp.pi, maxval=jnp.pi)
        agent = create_agent(x=agent_x, y=0.0, z=agent_z, yaw=agent_yaw)
        
        # Create balls
        key4, _ = jax.random.split(key3)
        balls = create_balls(key4, self.params.num_balls, bounds, room.height)
        
        # Initial state
        state = EnvState(
            agent=agent,
            balls=balls,
            room=room,
            step_count=jnp.array(0, dtype=jnp.int32),
            total_caught=jnp.array(0, dtype=jnp.int32),
        )
        
        # Render initial observation
        obs = render_image(agent, balls, room)
        
        return obs, state
    
    def step(self, key: jax.random.PRNGKey, state: EnvState, 
             action: int) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict]:
        """Take one environment step.
        
        Args:
            key: JAX random key (unused currently, kept for API compatibility)
            state: Current environment state
            action: Action to take (0-4)
        
        Returns:
            (observation, new_state, reward, done, info)
        """
        bounds = get_room_bounds(state.room)
        
        # Apply action to agent
        agent = apply_action(state.agent, action)
        
        # Update agent position
        agent = update_agent(agent, bounds)
        
        # Update ball physics
        balls, num_caught = update_balls(
            state.balls, agent, bounds, state.room.height
        )
        
        # Update step count and total caught
        step_count = state.step_count + 1
        total_caught = state.total_caught + num_caught
        
        # Check done condition
        all_caught = ~jnp.any(balls.active)
        max_steps_reached = step_count >= self.params.max_steps
        done = all_caught | max_steps_reached
        
        # New state
        new_state = EnvState(
            agent=agent,
            balls=balls,
            room=state.room,
            step_count=step_count,
            total_caught=total_caught,
        )
        
        # Render observation
        obs = render_image(agent, balls, state.room)
        
        # Reward is number of balls caught this step
        reward = num_caught.astype(jnp.float32)
        
        # Info dict
        info = {
            "total_caught": total_caught,
            "step_count": step_count,
            "all_caught": all_caught,
        }
        
        return obs, new_state, reward, done, info
    
    def render(self, state: EnvState) -> jnp.ndarray:
        """Render current state to image.
        
        Args:
            state: Environment state
        
        Returns:
            (128, 128, 3) RGB image
        """
        return render_image(state.agent, state.balls, state.room)


def make_env(num_balls: int = 5, max_steps: int = 1000) -> BallCatchEnv:
    """Create a BallCatchEnv with specified parameters.
    
    Args:
        num_balls: Number of balls to spawn
        max_steps: Maximum steps per episode
    
    Returns:
        BallCatchEnv instance
    """
    params = EnvParams(num_balls=num_balls, max_steps=max_steps)
    return BallCatchEnv(params)


# Functional interface for jit/vmap compatibility
def reset(key: jax.random.PRNGKey, 
          params: EnvParams = EnvParams()) -> Tuple[jnp.ndarray, EnvState]:
    """Functional reset interface.
    
    Args:
        key: JAX random key
        params: Environment parameters
    
    Returns:
        (observation, state)
    """
    env = BallCatchEnv(params)
    return env.reset(key)


def step(key: jax.random.PRNGKey, state: EnvState, action: int,
         params: EnvParams = EnvParams()) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict]:
    """Functional step interface.
    
    Args:
        key: JAX random key
        state: Current state
        action: Action to take
        params: Environment parameters
    
    Returns:
        (observation, new_state, reward, done, info)
    """
    env = BallCatchEnv(params)
    return env.step(key, state, action)


# Create jit-compiled versions
@jax.jit
def jit_reset(key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, EnvState]:
    """JIT-compiled reset with default params."""
    return reset(key)


@jax.jit
def jit_step(key: jax.random.PRNGKey, state: EnvState, 
             action: int) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict]:
    """JIT-compiled step with default params."""
    return step(key, state, action)


# Vectorized versions for batch processing
def vmap_reset(keys: jnp.ndarray, 
               params: EnvParams = EnvParams()) -> Tuple[jnp.ndarray, EnvState]:
    """Vectorized reset across multiple environments.
    
    Args:
        keys: (batch_size, 2) array of random keys
        params: Environment parameters
    
    Returns:
        (observations, states) with batch dimension
    """
    return jax.vmap(lambda k: reset(k, params))(keys)


def vmap_step(keys: jnp.ndarray, states: EnvState, actions: jnp.ndarray,
              params: EnvParams = EnvParams()) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict]:
    """Vectorized step across multiple environments.
    
    Args:
        keys: (batch_size, 2) array of random keys
        states: Batched environment states
        actions: (batch_size,) array of actions
        params: Environment parameters
    
    Returns:
        (observations, new_states, rewards, dones, infos) with batch dimensions
    """
    return jax.vmap(lambda k, s, a: step(k, s, a, params))(keys, states, actions)
