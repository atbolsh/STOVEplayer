"""3D POV environment with bouncing balls."""
from .env import BallCatchEnv, EnvState, EnvParams, make_env, reset, step
from .room import RoomState, create_room, render_image, get_room_bounds
from .physics import BallState, create_balls, update_balls, BALL_RADIUS, BALL_COLOR
from .agent_controller import (
    AgentState, Action, create_agent, apply_action, update_agent,
    AGENT_RADIUS, AGENT_HEIGHT, MOVE_SPEED, TURN_SPEED
)
