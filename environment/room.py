"""Simple JAX-based 3D room renderer for 128x128 POV images."""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from .agent_controller import AgentState, get_camera_params, AGENT_RADIUS, AGENT_HEIGHT
from .physics import BallState, BALL_RADIUS, BALL_COLOR


# Image dimensions
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
FOV = 1.2  # Field of view in radians (~70 degrees)

# Colors
WALL_COLOR = jnp.array([0.7, 0.8, 1.0], dtype=jnp.float32)  # Light blue
FLOOR_COLOR = jnp.array([0.6, 0.5, 0.4], dtype=jnp.float32)  # Brownish
CEILING_COLOR = jnp.array([0.9, 0.9, 0.95], dtype=jnp.float32)  # White-ish
MIRROR_TINT = jnp.array([0.7, 1.0, 0.8], dtype=jnp.float32)  # Green tint for reflections
AGENT_COLOR = jnp.array([0.9, 0.8, 0.7], dtype=jnp.float32)  # Beige


class RoomState(NamedTuple):
    """Room geometry state."""
    width: jnp.ndarray  # Room width (x dimension)
    height: jnp.ndarray  # Room height (y dimension)
    depth: jnp.ndarray  # Room depth (z dimension)
    mirror_walls: jnp.ndarray  # (4,) bool for which walls have mirrors
    mirror_positions: jnp.ndarray  # (4, 2) min/max positions along wall


def create_room(key: jax.random.PRNGKey) -> RoomState:
    """Create a room with random dimensions and mirror placement.
    
    Args:
        key: Random key
    
    Returns:
        RoomState with room geometry
    """
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Random room dimensions (base 4-6 with variation)
    width = jax.random.uniform(key1, (), minval=4.0, maxval=6.0)
    height = jax.random.uniform(key2, (), minval=2.5, maxval=3.5)
    depth = jax.random.uniform(key3, (), minval=4.0, maxval=6.0)
    
    # Occasional mirrors on walls (rare)
    key4, key5 = jax.random.split(key3)
    mirror_probs = jax.random.uniform(key4, (4,))
    mirror_walls = mirror_probs < 0.15  # ~15% chance per wall
    
    # Mirror positions along each wall (relative 0-1 range)
    mirror_pos_raw = jax.random.uniform(key5, (4, 2))
    mirror_positions = jnp.sort(mirror_pos_raw, axis=-1)
    # Make mirrors smaller (0.2 to 0.4 of wall)
    mirror_positions = mirror_positions.at[:, 0].set(mirror_positions[:, 0] * 0.3)
    mirror_positions = mirror_positions.at[:, 1].set(0.3 + mirror_positions[:, 1] * 0.4)
    
    return RoomState(
        width=width,
        height=height,
        depth=depth,
        mirror_walls=mirror_walls,
        mirror_positions=mirror_positions,
    )


def get_room_bounds(room: RoomState) -> tuple:
    """Get room boundaries.
    
    Returns:
        (min_x, max_x, min_z, max_z)
    """
    half_w = room.width / 2
    half_d = room.depth / 2
    return (-half_w, half_w, -half_d, half_d)


def _ray_sphere_intersect(ray_origin: jnp.ndarray, ray_dir: jnp.ndarray,
                          sphere_center: jnp.ndarray, sphere_radius: float) -> tuple:
    """Ray-sphere intersection test.
    
    Returns:
        (hit, t, normal) where t is distance along ray
    """
    oc = ray_origin - sphere_center
    a = jnp.dot(ray_dir, ray_dir)
    b = 2.0 * jnp.dot(oc, ray_dir)
    c = jnp.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4 * a * c
    
    hit = discriminant > 0
    t = jax.lax.cond(
        hit,
        lambda: (-b - jnp.sqrt(discriminant)) / (2 * a),
        lambda: jnp.array(1e10, dtype=jnp.float32),
    )
    
    # Only valid if t > 0
    hit = hit & (t > 0.001)
    
    # Calculate normal at hit point
    hit_point = ray_origin + t * ray_dir
    normal = (hit_point - sphere_center) / sphere_radius
    
    return hit, t, normal


def _ray_plane_intersect(ray_origin: jnp.ndarray, ray_dir: jnp.ndarray,
                         plane_point: jnp.ndarray, plane_normal: jnp.ndarray) -> tuple:
    """Ray-plane intersection test.
    
    Returns:
        (hit, t, normal)
    """
    denom = jnp.dot(plane_normal, ray_dir)
    hit = jnp.abs(denom) > 1e-6
    
    t = jax.lax.cond(
        hit,
        lambda: jnp.dot(plane_point - ray_origin, plane_normal) / denom,
        lambda: jnp.array(1e10, dtype=jnp.float32),
    )
    
    # Only valid if t > 0 and ray is hitting front of plane
    hit = hit & (t > 0.001) & (denom < 0)
    
    return hit, t, plane_normal


def _ray_cylinder_intersect(ray_origin: jnp.ndarray, ray_dir: jnp.ndarray,
                            cyl_center: jnp.ndarray, cyl_radius: float,
                            cyl_height: float) -> tuple:
    """Ray-cylinder intersection (vertical cylinder).
    
    Args:
        cyl_center: (x, y, z) center at bottom of cylinder
    
    Returns:
        (hit, t, normal)
    """
    # Project to xz plane for infinite cylinder test
    oc = jnp.array([ray_origin[0] - cyl_center[0], ray_origin[2] - cyl_center[2]])
    d = jnp.array([ray_dir[0], ray_dir[2]])
    
    a = jnp.dot(d, d)
    b = 2.0 * jnp.dot(oc, d)
    c = jnp.dot(oc, oc) - cyl_radius * cyl_radius
    discriminant = b * b - 4 * a * c
    
    # Check infinite cylinder hit
    inf_hit = discriminant > 0
    t_cyl = jax.lax.cond(
        inf_hit,
        lambda: (-b - jnp.sqrt(discriminant)) / (2 * a + 1e-8),
        lambda: jnp.array(1e10, dtype=jnp.float32),
    )
    
    # Check if hit is within cylinder height
    hit_y = ray_origin[1] + t_cyl * ray_dir[1]
    in_height = (hit_y >= cyl_center[1]) & (hit_y <= cyl_center[1] + cyl_height)
    
    hit = inf_hit & (t_cyl > 0.001) & in_height
    
    # Calculate normal (horizontal, pointing outward)
    hit_point = ray_origin + t_cyl * ray_dir
    normal = jnp.array([
        hit_point[0] - cyl_center[0],
        0.0,
        hit_point[2] - cyl_center[2],
    ])
    normal = normal / (jnp.linalg.norm(normal) + 1e-8)
    
    return hit, t_cyl, normal


def _check_mirror_hit(hit_point: jnp.ndarray, wall_idx: int,
                      room: RoomState) -> jnp.ndarray:
    """Check if a hit point on a wall is within a mirror.
    
    Args:
        hit_point: 3D hit point
        wall_idx: Which wall (0=+x, 1=-x, 2=+z, 3=-z)
        room: Room state
    
    Returns:
        Boolean indicating if hit is on mirror
    """
    has_mirror = room.mirror_walls[wall_idx]
    mirror_min = room.mirror_positions[wall_idx, 0]
    mirror_max = room.mirror_positions[wall_idx, 1]
    
    # For x walls, check z coordinate; for z walls, check x coordinate
    is_x_wall = wall_idx < 2
    
    wall_extent = jax.lax.cond(is_x_wall, lambda: room.depth, lambda: room.width)
    coord = jax.lax.cond(is_x_wall, lambda: hit_point[2], lambda: hit_point[0])
    
    # Convert to 0-1 range along wall
    half_extent = wall_extent / 2
    normalized = (coord + half_extent) / wall_extent
    
    # Check height (mirror covers middle portion)
    height_ok = (hit_point[1] > room.height * 0.2) & (hit_point[1] < room.height * 0.8)
    
    in_mirror = has_mirror & (normalized >= mirror_min) & (normalized <= mirror_max) & height_ok
    
    return in_mirror


def _trace_ray(ray_origin: jnp.ndarray, ray_dir: jnp.ndarray,
               balls: BallState, agent: AgentState, room: RoomState,
               is_reflection: bool = False) -> tuple:
    """Trace a single ray and return color and depth.
    
    Args:
        ray_origin: Ray start point
        ray_dir: Normalized ray direction
        balls: Ball state for sphere positions
        agent: Agent state (for reflection rendering)
        room: Room geometry
        is_reflection: Whether this is a reflected ray
    
    Returns:
        (color, depth)
    """
    bounds = get_room_bounds(room)
    min_x, max_x, min_z, max_z = bounds
    
    best_t = jnp.array(1e10, dtype=jnp.float32)
    best_color = jnp.zeros(3, dtype=jnp.float32)
    hit_mirror = jnp.array(False)
    mirror_normal = jnp.zeros(3, dtype=jnp.float32)
    hit_point = jnp.zeros(3, dtype=jnp.float32)
    
    # Test floor
    floor_hit, floor_t, floor_normal = _ray_plane_intersect(
        ray_origin, ray_dir,
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
    )
    floor_hit = floor_hit & (floor_t < best_t)
    best_t = jnp.where(floor_hit, floor_t, best_t)
    best_color = jnp.where(floor_hit, FLOOR_COLOR, best_color)
    
    # Test ceiling
    ceil_hit, ceil_t, ceil_normal = _ray_plane_intersect(
        ray_origin, ray_dir,
        jnp.array([0.0, room.height, 0.0]),
        jnp.array([0.0, -1.0, 0.0]),
    )
    ceil_hit = ceil_hit & (ceil_t < best_t)
    best_t = jnp.where(ceil_hit, ceil_t, best_t)
    best_color = jnp.where(ceil_hit, CEILING_COLOR, best_color)
    
    # Test walls (4 walls)
    # Wall 0: +x wall
    wall_hit, wall_t, wall_normal = _ray_plane_intersect(
        ray_origin, ray_dir,
        jnp.array([max_x, 0.0, 0.0]),
        jnp.array([-1.0, 0.0, 0.0]),
    )
    hp = ray_origin + wall_t * ray_dir
    in_bounds = (hp[1] >= 0) & (hp[1] <= room.height) & (hp[2] >= min_z) & (hp[2] <= max_z)
    wall_hit = wall_hit & in_bounds & (wall_t < best_t)
    is_mirror = _check_mirror_hit(hp, 0, room)
    
    best_t = jnp.where(wall_hit, wall_t, best_t)
    best_color = jnp.where(wall_hit, WALL_COLOR, best_color)
    hit_mirror = jnp.where(wall_hit & is_mirror, True, hit_mirror)
    mirror_normal = jnp.where(wall_hit & is_mirror, wall_normal, mirror_normal)
    hit_point = jnp.where(wall_hit & is_mirror, hp, hit_point)
    
    # Wall 1: -x wall
    wall_hit, wall_t, wall_normal = _ray_plane_intersect(
        ray_origin, ray_dir,
        jnp.array([min_x, 0.0, 0.0]),
        jnp.array([1.0, 0.0, 0.0]),
    )
    hp = ray_origin + wall_t * ray_dir
    in_bounds = (hp[1] >= 0) & (hp[1] <= room.height) & (hp[2] >= min_z) & (hp[2] <= max_z)
    wall_hit = wall_hit & in_bounds & (wall_t < best_t)
    is_mirror = _check_mirror_hit(hp, 1, room)
    
    best_t = jnp.where(wall_hit, wall_t, best_t)
    best_color = jnp.where(wall_hit, WALL_COLOR, best_color)
    hit_mirror = jnp.where(wall_hit & is_mirror, True, hit_mirror)
    mirror_normal = jnp.where(wall_hit & is_mirror, wall_normal, mirror_normal)
    hit_point = jnp.where(wall_hit & is_mirror, hp, hit_point)
    
    # Wall 2: +z wall
    wall_hit, wall_t, wall_normal = _ray_plane_intersect(
        ray_origin, ray_dir,
        jnp.array([0.0, 0.0, max_z]),
        jnp.array([0.0, 0.0, -1.0]),
    )
    hp = ray_origin + wall_t * ray_dir
    in_bounds = (hp[1] >= 0) & (hp[1] <= room.height) & (hp[0] >= min_x) & (hp[0] <= max_x)
    wall_hit = wall_hit & in_bounds & (wall_t < best_t)
    is_mirror = _check_mirror_hit(hp, 2, room)
    
    best_t = jnp.where(wall_hit, wall_t, best_t)
    best_color = jnp.where(wall_hit, WALL_COLOR, best_color)
    hit_mirror = jnp.where(wall_hit & is_mirror, True, hit_mirror)
    mirror_normal = jnp.where(wall_hit & is_mirror, wall_normal, mirror_normal)
    hit_point = jnp.where(wall_hit & is_mirror, hp, hit_point)
    
    # Wall 3: -z wall
    wall_hit, wall_t, wall_normal = _ray_plane_intersect(
        ray_origin, ray_dir,
        jnp.array([0.0, 0.0, min_z]),
        jnp.array([0.0, 0.0, 1.0]),
    )
    hp = ray_origin + wall_t * ray_dir
    in_bounds = (hp[1] >= 0) & (hp[1] <= room.height) & (hp[0] >= min_x) & (hp[0] <= max_x)
    wall_hit = wall_hit & in_bounds & (wall_t < best_t)
    is_mirror = _check_mirror_hit(hp, 3, room)
    
    best_t = jnp.where(wall_hit, wall_t, best_t)
    best_color = jnp.where(wall_hit, WALL_COLOR, best_color)
    hit_mirror = jnp.where(wall_hit & is_mirror, True, hit_mirror)
    mirror_normal = jnp.where(wall_hit & is_mirror, wall_normal, mirror_normal)
    hit_point = jnp.where(wall_hit & is_mirror, hp, hit_point)
    
    # Test balls
    def test_ball(carry, ball_idx):
        best_t, best_color = carry
        ball_pos = balls.positions[ball_idx]
        ball_active = balls.active[ball_idx]
        
        hit, t, normal = _ray_sphere_intersect(ray_origin, ray_dir, ball_pos, BALL_RADIUS)
        hit = hit & ball_active & (t < best_t)
        
        # Simple shading
        shade = jnp.maximum(0.3, jnp.dot(normal, jnp.array([0.5, 0.7, 0.3])))
        color = BALL_COLOR * shade
        
        best_t = jnp.where(hit, t, best_t)
        best_color = jnp.where(hit, color, best_color)
        
        return (best_t, best_color), None
    
    (best_t, best_color), _ = jax.lax.scan(
        test_ball, (best_t, best_color), jnp.arange(balls.positions.shape[0])
    )
    
    # Test agent cylinder (only in reflections)
    def test_agent():
        cyl_center = jnp.array([agent.x, agent.y, agent.z])
        hit, t, normal = _ray_cylinder_intersect(
            ray_origin, ray_dir, cyl_center, AGENT_RADIUS, AGENT_HEIGHT
        )
        shade = jnp.maximum(0.3, jnp.dot(normal, jnp.array([0.5, 0.7, 0.3])))
        return hit & (t < best_t), t, AGENT_COLOR * shade
    
    agent_hit, agent_t, agent_color = jax.lax.cond(
        jnp.array(is_reflection),
        test_agent,
        lambda: (jnp.array(False), jnp.array(1e10), AGENT_COLOR),
    )
    
    best_t = jnp.where(agent_hit, agent_t, best_t)
    best_color = jnp.where(agent_hit, agent_color, best_color)
    
    return best_color, best_t, hit_mirror, mirror_normal, hit_point


def _trace_ray_with_reflection(ray_origin: jnp.ndarray, ray_dir: jnp.ndarray,
                               balls: BallState, agent: AgentState, 
                               room: RoomState) -> tuple:
    """Trace ray with single-bounce mirror reflection."""
    # First trace
    color, depth, hit_mirror, mirror_normal, hit_point = _trace_ray(
        ray_origin, ray_dir, balls, agent, room, is_reflection=False
    )
    
    # If we hit a mirror, trace reflected ray
    def do_reflection():
        # Calculate reflected direction
        reflected_dir = ray_dir - 2.0 * jnp.dot(ray_dir, mirror_normal) * mirror_normal
        reflected_dir = reflected_dir / (jnp.linalg.norm(reflected_dir) + 1e-8)
        
        # Offset origin slightly to avoid self-intersection
        reflected_origin = hit_point + mirror_normal * 0.01
        
        # Trace reflected ray (no further reflections)
        refl_color, refl_depth, _, _, _ = _trace_ray(
            reflected_origin, reflected_dir, balls, agent, room, is_reflection=True
        )
        
        # Apply mirror tint
        return refl_color * MIRROR_TINT
    
    final_color = jax.lax.cond(
        hit_mirror,
        do_reflection,
        lambda: color,
    )
    
    return final_color, depth


def render_image(agent: AgentState, balls: BallState, room: RoomState) -> jnp.ndarray:
    """Render a 128x128 POV image from agent's perspective.
    
    Args:
        agent: Agent state (for camera position/direction)
        balls: Ball state (positions of balls to render)
        room: Room geometry
    
    Returns:
        (128, 128, 3) RGB image, float32 in [0, 1]
    """
    cam_pos, cam_forward, cam_right, cam_up = get_camera_params(agent)
    
    # Generate pixel coordinates
    aspect = IMAGE_WIDTH / IMAGE_HEIGHT
    
    def render_pixel(pixel_idx):
        y_idx = pixel_idx // IMAGE_WIDTH
        x_idx = pixel_idx % IMAGE_WIDTH
        
        # Convert to normalized device coordinates (-1 to 1)
        ndc_x = (2.0 * x_idx / IMAGE_WIDTH - 1.0) * aspect
        ndc_y = 1.0 - 2.0 * y_idx / IMAGE_HEIGHT  # Flip Y
        
        # Calculate ray direction
        half_fov = jnp.tan(FOV / 2)
        ray_dir = cam_forward + ndc_x * half_fov * cam_right + ndc_y * half_fov * cam_up
        ray_dir = ray_dir / jnp.linalg.norm(ray_dir)
        
        color, _ = _trace_ray_with_reflection(cam_pos, ray_dir, balls, agent, room)
        return color
    
    # Render all pixels
    pixel_indices = jnp.arange(IMAGE_WIDTH * IMAGE_HEIGHT)
    colors = jax.vmap(render_pixel)(pixel_indices)
    
    # Reshape to image
    image = colors.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    
    # Clamp to [0, 1]
    image = jnp.clip(image, 0.0, 1.0)
    
    return image
