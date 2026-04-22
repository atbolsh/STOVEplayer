"""Launch script for interactive BallCatchEnv play.

Launch: python -m nextPlayer.gui.play

Keyboard Controls:
  W / UP      - Move forward (action=0)
  S / DOWN    - Stop moving (action=1)
  A / LEFT    - Turn left (action=2)
  D / RIGHT   - Turn right (action=3)
  SPACE       - Stop turning (action=4)
  R           - Reset environment
  P           - Pause/unpause
  ESC / Q     - Quit

Debug Controls:
  I           - Print environment state to console
  SHIFT+arrows - Slow motion mode
  N           - Step one frame (when paused)
"""

import argparse
import sys
from typing import Optional, Tuple, Any

import numpy as np
import pygame

try:
    import jax
    import jax.numpy as jnp
    from jax import random
except ImportError:
    print("Error: JAX is required. Install with: pip install jax jaxlib")
    sys.exit(1)

from nextPlayer.environment import BallCatchEnv, EnvParams
from nextPlayer.gui.viewer import Viewer


class Actions:
    """Action constants for BallCatchEnv."""
    MOVE_FORWARD = 0
    STOP_MOVING = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    STOP_TURNING = 4
    NO_OP = 1  # Default to stop moving as no-op


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive BallCatchEnv player",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--watch",
        action="store_true",
        help="Autonomous mode (no agent input, physics only)"
    )
    mode_group.add_argument(
        "--play",
        action="store_true",
        default=True,
        help="Manual control (default)"
    )
    mode_group.add_argument(
        "--random",
        action="store_true",
        help="Random agent actions each frame"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Set random seed (default: 42)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        metavar="N",
        help="Set target FPS (default: 30)"
    )
    parser.add_argument(
        "--num-balls",
        type=int,
        default=5,
        metavar="N",
        help="Number of balls (default: 5)"
    )
    
    args = parser.parse_args()
    
    if args.watch:
        args.mode = "watch"
    elif args.random:
        args.mode = "random"
    else:
        args.mode = "play"
        
    return args


def get_keyboard_action(slow_motion: bool = False) -> Tuple[Optional[int], dict]:
    """Get action from keyboard state.
    
    Args:
        slow_motion: Whether shift is held for slow motion.
        
    Returns:
        Tuple of (action or None, control_events dict).
    """
    keys = pygame.key.get_pressed()
    mods = pygame.key.get_mods()
    
    action = None
    
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        action = Actions.MOVE_FORWARD
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        action = Actions.STOP_MOVING
    elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
        action = Actions.TURN_LEFT
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        action = Actions.TURN_RIGHT
    elif keys[pygame.K_SPACE]:
        action = Actions.STOP_TURNING
        
    return action, {
        "slow_motion": bool(mods & pygame.KMOD_SHIFT)
    }


def process_events(viewer: "Viewer") -> dict:
    """Process pygame events and return control signals.
    
    Args:
        viewer: Viewer instance, used for button hit-testing on mouse clicks.
    
    Returns:
        Dictionary with control signals:
        - quit: Should quit the game
        - reset: Should reset the environment
        - pause_toggle: Should toggle pause
        - debug_info: Should print debug info
        - step_frame: Should step one frame (when paused)
        - button_action: action int from a sidebar button click, or None
    """
    events = {
        "quit": False,
        "reset": False,
        "pause_toggle": False,
        "debug_info": False,
        "step_frame": False,
        "button_action": None,
    }
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            events["quit"] = True
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                events["quit"] = True
            elif event.key == pygame.K_r:
                events["reset"] = True
            elif event.key == pygame.K_p:
                events["pause_toggle"] = True
            elif event.key == pygame.K_i:
                events["debug_info"] = True
            elif event.key == pygame.K_n:
                events["step_frame"] = True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            viewer.handle_mouse_click(event.pos)

    BUTTON_TO_ACTION = {
        "move_forward": Actions.MOVE_FORWARD,
        "stop_moving":  Actions.STOP_MOVING,
        "turn_left":    Actions.TURN_LEFT,
        "turn_right":   Actions.TURN_RIGHT,
        "stop_turning": Actions.STOP_TURNING,
    }

    for click_id in viewer.pop_button_clicks():
        if click_id in BUTTON_TO_ACTION:
            events["button_action"] = BUTTON_TO_ACTION[click_id]
        elif click_id == "quit":
            events["quit"] = True
        elif click_id == "reset":
            events["reset"] = True
        elif click_id == "pause_toggle":
            events["pause_toggle"] = True
        elif click_id == "debug_info":
            events["debug_info"] = True
        elif click_id == "step_frame":
            events["step_frame"] = True
                
    return events


def extract_state_info(state: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
    """Extract position and rotation from environment state.
    
    Args:
        state: Environment state object.
        
    Returns:
        Tuple of (position, rotation) or (None, None) if not available.
    """
    position = None
    rotation = None
    
    if hasattr(state, 'agent_position'):
        pos = state.agent_position
        if hasattr(pos, '__len__') and len(pos) >= 3:
            position = (float(pos[0]), float(pos[1]), float(pos[2]))
        elif hasattr(pos, '__len__') and len(pos) >= 2:
            position = (float(pos[0]), float(pos[1]), 0.0)
            
    if hasattr(state, 'agent_rotation'):
        rotation = float(state.agent_rotation)
    elif hasattr(state, 'agent_angle'):
        rotation = float(state.agent_angle)
        
    return position, rotation


def print_debug_info(state: Any, obs: Any, score: float, frame: int) -> None:
    """Print detailed environment state to console."""
    print("\n" + "=" * 50)
    print(f"Frame: {frame}")
    print(f"Score: {score:.2f}")
    print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Observation range: [{float(obs.min()):.3f}, {float(obs.max()):.3f}]")
    
    if hasattr(state, '__dict__'):
        print("State attributes:")
        for key, value in state.__dict__.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")
    elif hasattr(state, '_fields'):
        print("State fields (NamedTuple):")
        for field in state._fields:
            value = getattr(state, field)
            if hasattr(value, 'shape'):
                print(f"  {field}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {field}: {value}")
    else:
        print(f"State type: {type(state)}")
        
    print("=" * 50 + "\n")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    print(f"Starting BallCatchEnv in '{args.mode}' mode")
    print(f"  Seed: {args.seed}")
    print(f"  FPS: {args.fps}")
    print(f"  Balls: {args.num_balls}")
    print("\nControls:")
    print("  W/UP: Forward | S/DOWN: Stop | A/LEFT: Turn left | D/RIGHT: Turn right")
    print("  SPACE: Stop turning | R: Reset | P: Pause | I: Debug | N: Step frame")
    print("  ESC/Q: Quit | SHIFT+arrows: Slow motion")
    print()
    
    key = random.PRNGKey(args.seed)
    
    env = BallCatchEnv(EnvParams(num_balls=args.num_balls))
    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)
    
    print("Compiling JIT kernels (first run only)...")
    key, reset_key = random.split(key)
    obs, state = jit_reset(reset_key)
    
    # Warmup: run one step so step kernel is also compiled before game loop
    key, warmup_key = random.split(key)
    _ = jit_step(warmup_key, state, Actions.NO_OP)
    print("Ready!\n")
    
    viewer = Viewer(fps=args.fps, title=f"BallCatch - {args.mode.capitalize()} Mode")
    
    paused = False
    total_score = 0.0
    episode_score = 0.0
    balls_caught_this_episode = 0
    running = True
    slow_motion_factor = 4
    episode_over = False
    episode_end_reason = ""
    
    try:
        viewer.init()
        
        while running:
            events = process_events(viewer)
            
            if events["quit"]:
                running = False
                continue
            
            # While on the episode-end screen, wait for R (or button) to start new episode
            if episode_over:
                if events["reset"]:
                    key, reset_key = random.split(key)
                    obs, state = jit_reset(reset_key)
                    episode_score = 0.0
                    balls_caught_this_episode = 0
                    viewer.reset_frame_count()
                    episode_over = False
                    print("New episode started")
                
                position, rotation = extract_state_info(state)
                viewer.render(
                    obs,
                    score=total_score,
                    position=position,
                    rotation=rotation,
                    paused=True,
                    mode=args.mode,
                    overlay_message=f"{episode_end_reason}\n"
                                    f"Caught {balls_caught_this_episode}/{args.num_balls} balls "
                                    f"(episode score: {episode_score:.0f})\n\n"
                                    f"Press R or click Reset for a new room",
                )
                continue
                
            if events["pause_toggle"]:
                paused = not paused
                viewer.set_pause_state(paused)
                print(f"{'Paused' if paused else 'Resumed'}")
                
            if events["reset"]:
                key, reset_key = random.split(key)
                obs, state = jit_reset(reset_key)
                total_score = 0.0
                episode_score = 0.0
                balls_caught_this_episode = 0
                viewer.reset_frame_count()
                print("Environment reset")
                
            if events["debug_info"]:
                print_debug_info(state, obs, total_score, viewer.frame_count)
                
            should_step = not paused or events["step_frame"]
            
            if should_step:
                prev_caught = balls_caught_this_episode
                
                if args.mode == "watch":
                    action = Actions.NO_OP
                elif args.mode == "random":
                    key, action_key = random.split(key)
                    action = random.randint(action_key, (), 0, 5)
                else:
                    keyboard_action, controls = get_keyboard_action()
                    if events["button_action"] is not None:
                        action = events["button_action"]
                    elif keyboard_action is not None:
                        action = keyboard_action
                    else:
                        action = Actions.NO_OP
                    
                    if controls["slow_motion"]:
                        for _ in range(slow_motion_factor - 1):
                            viewer.render(
                                obs,
                                score=total_score,
                                position=extract_state_info(state)[0],
                                rotation=extract_state_info(state)[1],
                                paused=paused,
                                mode=args.mode
                            )
                
                key, step_key = random.split(key)
                obs, state, reward, done, info = jit_step(step_key, state, action)
                caught_now = int(float(reward))
                total_score += caught_now
                episode_score += caught_now
                balls_caught_this_episode += caught_now
                
                if caught_now > 0:
                    print(f"  Nice! Caught {caught_now} ball{'s' if caught_now > 1 else ''}! "
                          f"({balls_caught_this_episode}/{args.num_balls} total)")
                
                if done:
                    all_caught = bool(info["all_caught"])
                    if all_caught:
                        episode_end_reason = "Congratulations! All balls caught!"
                        print(f"Congratulations! All {args.num_balls} balls caught! "
                              f"Episode score: {episode_score:.0f}")
                    else:
                        episode_end_reason = "Time's up!"
                        print(f"Time's up! Caught {balls_caught_this_episode}/{args.num_balls} balls. "
                              f"Episode score: {episode_score:.0f}")
                    episode_over = True
            
            position, rotation = extract_state_info(state)
            viewer.render(
                obs,
                score=total_score,
                position=position,
                rotation=rotation,
                paused=paused,
                mode=args.mode
            )
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        viewer.close()
        
    print(f"Session ended. Total score across all episodes: {total_score:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
