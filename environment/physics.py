"""Ball physics with elastic collisions."""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from .agent_controller import AgentState, AGENT_RADIUS, AGENT_HEIGHT


# Ball properties
BALL_RADIUS = 0.15
BALL_SPEED = 0.02  # Slow movement for easy catching
BALL_COLOR = jnp.array([0.3, 0.0, 0.5], dtype=jnp.float32)  # Indigo


class BallState(NamedTuple):
    """State of all balls in the environment."""
    positions: jnp.ndarray  # (num_balls, 3) x, y, z
    velocities: jnp.ndarray  # (num_balls, 3) vx, vy, vz
    active: jnp.ndarray  # (num_balls,) bool, whether ball exists


def create_balls(key: jax.random.PRNGKey, num_balls: int,
                 room_bounds: tuple, room_height: float) -> BallState:
    """Create balls with random positions and velocities.
    
    Args:
        key: Random key
        num_balls: Number of balls to create
        room_bounds: (min_x, max_x, min_z, max_z)
        room_height: Height of room
    
    Returns:
        Initial ball state
    """
    min_x, max_x, min_z, max_z = room_bounds
    margin = BALL_RADIUS * 2
    
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Random positions within room
    x = jax.random.uniform(key1, (num_balls,), 
                           minval=min_x + margin, maxval=max_x - margin)
    y = jax.random.uniform(key2, (num_balls,), 
                           minval=BALL_RADIUS + 0.1, maxval=room_height - margin)
    z = jax.random.uniform(key3, (num_balls,), 
                           minval=min_z + margin, maxval=max_z - margin)
    positions = jnp.stack([x, y, z], axis=-1)
    
    # Random velocities (normalized to BALL_SPEED)
    key4, key5 = jax.random.split(key3)
    velocities = jax.random.normal(key4, (num_balls, 3))
    velocities = velocities / (jnp.linalg.norm(velocities, axis=-1, keepdims=True) + 1e-8)
    velocities = velocities * BALL_SPEED
    
    # All balls start active
    active = jnp.ones(num_balls, dtype=jnp.bool_)
    
    return BallState(positions=positions, velocities=velocities, active=active)


def _ball_ball_collision(positions: jnp.ndarray, velocities: jnp.ndarray,
                         active: jnp.ndarray) -> tuple:
    """Handle elastic collisions between balls.
    
    Returns:
        (updated_positions, updated_velocities) with overlap separation
    """
    num_balls = positions.shape[0]
    min_dist = 2 * BALL_RADIUS
    
    def collision_pair(carry, idx):
        pos, vels = carry
        i = idx // num_balls
        j = idx % num_balls
        
        should_process = (i < j) & active[i] & active[j]
        
        pos_i = pos[i]
        pos_j = pos[j]
        vel_i = vels[i]
        vel_j = vels[j]
        
        diff = pos_i - pos_j
        dist = jnp.linalg.norm(diff)
        
        colliding = (dist < min_dist) & should_process
        
        normal = diff / (dist + 1e-8)
        rel_vel = vel_i - vel_j
        rel_speed = jnp.dot(rel_vel, normal)
        
        # Reflect velocity (elastic, equal mass)
        should_collide = colliding & (rel_speed > 0)
        new_vel_i = jnp.where(should_collide, vel_i - rel_speed * normal, vel_i)
        new_vel_j = jnp.where(should_collide, vel_j + rel_speed * normal, vel_j)
        
        # Push overlapping balls apart so they no longer intersect
        overlap = min_dist - dist
        should_separate = colliding & (overlap > 0)
        half_push = jnp.where(should_separate, overlap / 2 + 1e-4, 0.0)
        new_pos_i = pos_i + normal * half_push
        new_pos_j = pos_j - normal * half_push
        
        pos = pos.at[i].set(jnp.where(colliding, new_pos_i, pos_i))
        pos = pos.at[j].set(jnp.where(colliding, new_pos_j, pos_j))
        vels = vels.at[i].set(new_vel_i)
        vels = vels.at[j].set(new_vel_j)
        
        return (pos, vels), None
    
    (positions, velocities), _ = jax.lax.scan(
        collision_pair, 
        (positions, velocities), 
        jnp.arange(num_balls * num_balls)
    )
    
    return positions, velocities


def _ball_wall_collision(positions: jnp.ndarray, velocities: jnp.ndarray,
                         room_bounds: tuple, room_height: float) -> tuple:
    """Handle ball-wall elastic collisions.
    
    Returns:
        (updated_positions, updated_velocities)
    """
    min_x, max_x, min_z, max_z = room_bounds
    
    # Floor and ceiling
    floor_collision = positions[:, 1] < BALL_RADIUS
    ceiling_collision = positions[:, 1] > room_height - BALL_RADIUS
    
    # Reflect y velocity on floor/ceiling collision
    velocities = jnp.where(
        (floor_collision | ceiling_collision)[:, None] & (jnp.array([False, True, False])),
        -velocities,
        velocities
    )
    
    # Clamp y position
    positions = positions.at[:, 1].set(
        jnp.clip(positions[:, 1], BALL_RADIUS, room_height - BALL_RADIUS)
    )
    
    # X walls
    x_min_collision = positions[:, 0] < min_x + BALL_RADIUS
    x_max_collision = positions[:, 0] > max_x - BALL_RADIUS
    
    velocities = jnp.where(
        (x_min_collision | x_max_collision)[:, None] & (jnp.array([True, False, False])),
        -velocities,
        velocities
    )
    
    positions = positions.at[:, 0].set(
        jnp.clip(positions[:, 0], min_x + BALL_RADIUS, max_x - BALL_RADIUS)
    )
    
    # Z walls
    z_min_collision = positions[:, 2] < min_z + BALL_RADIUS
    z_max_collision = positions[:, 2] > max_z - BALL_RADIUS
    
    velocities = jnp.where(
        (z_min_collision | z_max_collision)[:, None] & (jnp.array([False, False, True])),
        -velocities,
        velocities
    )
    
    positions = positions.at[:, 2].set(
        jnp.clip(positions[:, 2], min_z + BALL_RADIUS, max_z - BALL_RADIUS)
    )
    
    return positions, velocities


def _ball_agent_collision(positions: jnp.ndarray, active: jnp.ndarray,
                          agent: AgentState) -> tuple:
    """Check for ball-agent collisions.
    
    Returns:
        (new_active, num_caught) - updated active flags and count of caught balls
    """
    # Agent cylinder parameters
    agent_x = agent.x
    agent_z = agent.z
    agent_y_bottom = agent.y
    agent_y_top = agent.y + AGENT_HEIGHT
    
    # Distance from ball center to agent cylinder axis (in xz plane)
    dx = positions[:, 0] - agent_x
    dz = positions[:, 2] - agent_z
    dist_xz = jnp.sqrt(dx**2 + dz**2)
    
    # Check if ball is within cylinder height
    ball_y = positions[:, 1]
    in_height = (ball_y + BALL_RADIUS > agent_y_bottom) & (ball_y - BALL_RADIUS < agent_y_top)
    
    # Check if ball is within cylinder radius
    in_radius = dist_xz < (AGENT_RADIUS + BALL_RADIUS)
    
    # Ball is caught if colliding and active
    caught = active & in_height & in_radius
    
    # Update active status
    new_active = active & ~caught
    num_caught = jnp.sum(caught.astype(jnp.int32))
    
    return new_active, num_caught


def update_balls(balls: BallState, agent: AgentState, 
                 room_bounds: tuple, room_height: float) -> tuple:
    """Update ball physics for one timestep.
    
    Args:
        balls: Current ball state
        agent: Current agent state
        room_bounds: (min_x, max_x, min_z, max_z)
        room_height: Height of room
    
    Returns:
        (new_balls, num_caught)
    """
    positions = balls.positions
    velocities = balls.velocities
    active = balls.active
    
    # Move balls
    positions = positions + velocities
    
    # Ball-ball collisions (also separates overlapping balls)
    positions, velocities = _ball_ball_collision(positions, velocities, active)
    
    # Ball-wall collisions
    positions, velocities = _ball_wall_collision(positions, velocities, 
                                                   room_bounds, room_height)
    
    # Ball-agent collisions
    active, num_caught = _ball_agent_collision(positions, active, agent)
    
    new_balls = BallState(positions=positions, velocities=velocities, active=active)
    
    return new_balls, num_caught


def get_active_ball_positions(balls: BallState) -> tuple:
    """Get positions and mask of active balls for rendering.
    
    Returns:
        (positions, active_mask)
    """
    return balls.positions, balls.active
