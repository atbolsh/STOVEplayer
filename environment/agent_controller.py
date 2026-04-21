"""Agent state and movement controller."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from enum import IntEnum


class Action(IntEnum):
    """Available agent actions."""
    FORWARD = 0
    STOP = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    STOP_TURNING = 4


class AgentState(NamedTuple):
    """Agent state containing position, rotation, and velocities."""
    x: jnp.ndarray  # position x
    y: jnp.ndarray  # position y (height)
    z: jnp.ndarray  # position z
    yaw: jnp.ndarray  # rotation angle (radians)
    moving: jnp.ndarray  # whether agent is moving forward
    turn_direction: jnp.ndarray  # -1 left, 0 none, 1 right


# Agent physical properties
AGENT_RADIUS = 0.3
AGENT_HEIGHT = 1.0
AGENT_EYE_HEIGHT = 0.8
MOVE_SPEED = 0.05
TURN_SPEED = 0.03


def create_agent(x: float = 0.0, y: float = 0.0, z: float = 0.0, 
                 yaw: float = 0.0) -> AgentState:
    """Create a new agent at the specified position."""
    return AgentState(
        x=jnp.array(x, dtype=jnp.float32),
        y=jnp.array(y, dtype=jnp.float32),
        z=jnp.array(z, dtype=jnp.float32),
        yaw=jnp.array(yaw, dtype=jnp.float32),
        moving=jnp.array(False),
        turn_direction=jnp.array(0, dtype=jnp.int32),
    )


def apply_action(agent: AgentState, action: int) -> AgentState:
    """Apply an action to update agent control state."""
    action = jnp.asarray(action, dtype=jnp.int32)
    
    # Update moving state
    moving = jax.lax.cond(
        action == Action.FORWARD,
        lambda: jnp.array(True),
        lambda: jax.lax.cond(
            action == Action.STOP,
            lambda: jnp.array(False),
            lambda: agent.moving,
        )
    )
    
    # Update turn direction
    turn_direction = jax.lax.cond(
        action == Action.TURN_LEFT,
        lambda: jnp.array(-1, dtype=jnp.int32),
        lambda: jax.lax.cond(
            action == Action.TURN_RIGHT,
            lambda: jnp.array(1, dtype=jnp.int32),
            lambda: jax.lax.cond(
                action == Action.STOP_TURNING,
                lambda: jnp.array(0, dtype=jnp.int32),
                lambda: agent.turn_direction,
            )
        )
    )
    
    return agent._replace(moving=moving, turn_direction=turn_direction)


def update_agent(agent: AgentState, room_bounds: tuple) -> AgentState:
    """Update agent position and rotation based on current control state.
    
    Args:
        agent: Current agent state
        room_bounds: (min_x, max_x, min_z, max_z) room boundaries
    
    Returns:
        Updated agent state
    """
    min_x, max_x, min_z, max_z = room_bounds
    
    # Update rotation
    new_yaw = agent.yaw + agent.turn_direction.astype(jnp.float32) * TURN_SPEED
    new_yaw = jnp.mod(new_yaw + jnp.pi, 2 * jnp.pi) - jnp.pi
    
    # Calculate movement direction
    dx = jnp.sin(new_yaw) * MOVE_SPEED
    dz = jnp.cos(new_yaw) * MOVE_SPEED
    
    # Apply movement if moving
    new_x = jax.lax.cond(
        agent.moving,
        lambda: agent.x + dx,
        lambda: agent.x,
    )
    new_z = jax.lax.cond(
        agent.moving,
        lambda: agent.z + dz,
        lambda: agent.z,
    )
    
    # Clamp to room bounds (with margin for agent radius)
    new_x = jnp.clip(new_x, min_x + AGENT_RADIUS, max_x - AGENT_RADIUS)
    new_z = jnp.clip(new_z, min_z + AGENT_RADIUS, max_z - AGENT_RADIUS)
    
    return agent._replace(x=new_x, z=new_z, yaw=new_yaw)


def get_camera_params(agent: AgentState) -> tuple:
    """Get camera position and direction from agent state.
    
    Returns:
        (pos, forward, right, up) vectors for camera
    """
    pos = jnp.array([agent.x, agent.y + AGENT_EYE_HEIGHT, agent.z])
    
    forward = jnp.array([
        jnp.sin(agent.yaw),
        0.0,
        jnp.cos(agent.yaw),
    ])
    
    right = jnp.array([
        jnp.cos(agent.yaw),
        0.0,
        -jnp.sin(agent.yaw),
    ])
    
    up = jnp.array([0.0, 1.0, 0.0])
    
    return pos, forward, right, up


def get_agent_cylinder(agent: AgentState) -> tuple:
    """Get cylinder parameters for collision detection.
    
    Returns:
        (center_x, center_z, radius, y_bottom, y_top)
    """
    return (
        agent.x,
        agent.z,
        AGENT_RADIUS,
        agent.y,
        agent.y + AGENT_HEIGHT,
    )
