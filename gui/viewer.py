"""Pygame-based visualization for BallCatchEnv."""

import numpy as np
import pygame
from typing import Optional, Tuple, Any

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class Viewer:
    """Real-time pygame viewer for 128x128 environment frames."""
    
    ENV_SIZE = 128
    WINDOW_SIZE = 512
    SCALE = WINDOW_SIZE // ENV_SIZE
    
    HUD_FONT_SIZE = 20
    HUD_COLOR = (255, 255, 255)
    HUD_SHADOW_COLOR = (0, 0, 0)
    HUD_PADDING = 10
    
    def __init__(self, fps: int = 30, title: str = "BallCatch Environment"):
        """Initialize the viewer.
        
        Args:
            fps: Target frames per second for display.
            title: Window title.
        """
        self.target_fps = fps
        self.title = title
        
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None
        self._initialized = False
        
        self._frame_count = 0
        self._actual_fps = 0.0
        
    def init(self) -> None:
        """Initialize pygame and create the window."""
        if self._initialized:
            return
            
        pygame.init()
        pygame.display.set_caption(self.title)
        
        self._screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, self.HUD_FONT_SIZE)
        self._initialized = True
        
    def close(self) -> None:
        """Clean up pygame resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
            self._screen = None
            self._clock = None
            self._font = None
            
    def _jax_to_surface(self, obs: Any) -> pygame.Surface:
        """Convert JAX array [128, 128, 3] float32 to pygame surface.
        
        Args:
            obs: JAX array with shape [128, 128, 3], values in [0, 1].
            
        Returns:
            Pygame surface scaled to window size.
        """
        if HAS_JAX and hasattr(obs, 'device'):
            arr = np.asarray(obs)
        else:
            arr = np.array(obs)
        
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        
        if arr.shape != (self.ENV_SIZE, self.ENV_SIZE, 3):
            raise ValueError(f"Expected shape ({self.ENV_SIZE}, {self.ENV_SIZE}, 3), got {arr.shape}")
        
        surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        scaled = pygame.transform.scale(surface, (self.WINDOW_SIZE, self.WINDOW_SIZE))
        
        return scaled
    
    def _render_text(self, text: str, pos: Tuple[int, int], shadow: bool = True) -> None:
        """Render text with optional drop shadow for readability."""
        if shadow:
            shadow_surface = self._font.render(text, True, self.HUD_SHADOW_COLOR)
            self._screen.blit(shadow_surface, (pos[0] + 1, pos[1] + 1))
        
        text_surface = self._font.render(text, True, self.HUD_COLOR)
        self._screen.blit(text_surface, pos)
        
    def _render_hud(
        self,
        score: float = 0.0,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[float] = None,
        paused: bool = False,
        mode: str = "play"
    ) -> None:
        """Render the heads-up display overlay.
        
        Args:
            score: Current game score.
            position: Agent position as (x, y, z) tuple.
            rotation: Agent rotation in degrees.
            paused: Whether the game is paused.
            mode: Current play mode.
        """
        y = self.HUD_PADDING
        line_height = self.HUD_FONT_SIZE + 4
        
        self._render_text(f"Score: {score:.1f}", (self.HUD_PADDING, y))
        y += line_height
        
        self._render_text(f"Frame: {self._frame_count}", (self.HUD_PADDING, y))
        y += line_height
        
        self._render_text(f"FPS: {self._actual_fps:.1f}", (self.HUD_PADDING, y))
        y += line_height
        
        if position is not None:
            pos_str = f"Pos: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})"
            self._render_text(pos_str, (self.HUD_PADDING, y))
            y += line_height
            
        if rotation is not None:
            self._render_text(f"Rot: {rotation:.1f}°", (self.HUD_PADDING, y))
            y += line_height
            
        mode_y = self.HUD_PADDING
        mode_text = f"Mode: {mode}"
        text_width = self._font.size(mode_text)[0]
        self._render_text(mode_text, (self.WINDOW_SIZE - text_width - self.HUD_PADDING, mode_y))
        
        if paused:
            pause_text = "PAUSED"
            text_width = self._font.size(pause_text)[0]
            x = (self.WINDOW_SIZE - text_width) // 2
            y = self.WINDOW_SIZE // 2
            
            pygame.draw.rect(
                self._screen,
                (0, 0, 0, 128),
                (x - 10, y - 5, text_width + 20, self.HUD_FONT_SIZE + 10)
            )
            self._render_text(pause_text, (x, y))
            
    def render(
        self,
        obs: Any,
        score: float = 0.0,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[float] = None,
        paused: bool = False,
        mode: str = "play"
    ) -> float:
        """Render a frame and return the delta time.
        
        Args:
            obs: JAX array with shape [128, 128, 3], values in [0, 1].
            score: Current game score.
            position: Agent position as (x, y, z) tuple.
            rotation: Agent rotation in degrees.
            paused: Whether the game is paused.
            mode: Current play mode.
            
        Returns:
            Delta time in seconds since last frame.
        """
        if not self._initialized:
            self.init()
            
        surface = self._jax_to_surface(obs)
        self._screen.blit(surface, (0, 0))
        
        self._render_hud(
            score=score,
            position=position,
            rotation=rotation,
            paused=paused,
            mode=mode
        )
        
        pygame.display.flip()
        
        dt = self._clock.tick(self.target_fps) / 1000.0
        self._actual_fps = self._clock.get_fps()
        self._frame_count += 1
        
        return dt
    
    def reset_frame_count(self) -> None:
        """Reset the frame counter."""
        self._frame_count = 0
        
    @property
    def frame_count(self) -> int:
        """Current frame count."""
        return self._frame_count
    
    @property
    def fps(self) -> float:
        """Actual FPS being achieved."""
        return self._actual_fps

    def __enter__(self) -> "Viewer":
        """Context manager entry."""
        self.init()
        return self
        
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()
