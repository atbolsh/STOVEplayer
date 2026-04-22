"""Pygame-based visualization for BallCatchEnv."""

import numpy as np
import pygame
from typing import Optional, Tuple, Any, List

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class Button:
    """Clickable button with hover feedback."""

    COLOR_NORMAL = (60, 63, 70)
    COLOR_HOVER = (80, 85, 95)
    COLOR_ACTIVE = (100, 140, 200)
    COLOR_TEXT = (220, 220, 225)
    COLOR_KEY_HINT = (140, 145, 155)
    CORNER_RADIUS = 6

    def __init__(self, rect: pygame.Rect, label: str, action_id: str,
                 key_hint: str = "", toggle: bool = False):
        self.rect = rect
        self.label = label
        self.action_id = action_id
        self.key_hint = key_hint
        self.toggle = toggle
        self.toggled = False
        self._hovered = False

    def update_hover(self, mouse_pos: Tuple[int, int]) -> None:
        self._hovered = self.rect.collidepoint(mouse_pos)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font,
             small_font: pygame.font.Font) -> None:
        if self.toggle and self.toggled:
            color = self.COLOR_ACTIVE
        elif self._hovered:
            color = self.COLOR_HOVER
        else:
            color = self.COLOR_NORMAL

        pygame.draw.rect(surface, color, self.rect,
                         border_radius=self.CORNER_RADIUS)

        text_surf = font.render(self.label, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(centerx=self.rect.centerx,
                                       centery=self.rect.centery - (6 if self.key_hint else 0))
        surface.blit(text_surf, text_rect)

        if self.key_hint:
            hint_surf = small_font.render(self.key_hint, True, self.COLOR_KEY_HINT)
            hint_rect = hint_surf.get_rect(centerx=self.rect.centerx,
                                           top=text_rect.bottom + 2)
            surface.blit(hint_surf, hint_rect)

    def handle_click(self, mouse_pos: Tuple[int, int]) -> bool:
        if self.rect.collidepoint(mouse_pos):
            if self.toggle:
                self.toggled = not self.toggled
            return True
        return False


class Viewer:
    """Real-time pygame viewer for 128x128 environment frames with sidebar controls."""

    ENV_SIZE = 128
    GAME_SIZE = 512
    SIDEBAR_WIDTH = 200
    SCALE = GAME_SIZE // ENV_SIZE

    HUD_FONT_SIZE = 20
    HUD_COLOR = (255, 255, 255)
    HUD_SHADOW_COLOR = (0, 0, 0)
    HUD_PADDING = 10

    SIDEBAR_BG = (35, 38, 45)
    SIDEBAR_SECTION_COLOR = (170, 175, 185)

    def __init__(self, fps: int = 30, title: str = "BallCatch Environment"):
        self.target_fps = fps
        self.title = title

        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._section_font: Optional[pygame.font.Font] = None
        self._initialized = False

        self._frame_count = 0
        self._actual_fps = 0.0

        self._buttons: List[Button] = []
        self._pending_clicks: List[str] = []

    def _create_buttons(self) -> None:
        """Build the sidebar button layout."""
        x = self.GAME_SIZE + 16
        w = self.SIDEBAR_WIDTH - 32
        h = 40
        gap = 6

        y = 36

        movement_buttons = [
            ("Forward",       "move_forward",  "W / Up"),
            ("Stop",          "stop_moving",   "S / Down"),
            ("Turn Left",     "turn_left",     "A / Left"),
            ("Turn Right",    "turn_right",    "D / Right"),
            ("Stop Turning",  "stop_turning",  "Space"),
        ]
        for label, action_id, hint in movement_buttons:
            self._buttons.append(
                Button(pygame.Rect(x, y, w, h), label, action_id, key_hint=hint)
            )
            y += h + gap

        y += 12

        control_buttons = [
            ("Pause",      "pause_toggle", "P",   True),
            ("Reset",      "reset",        "R",   False),
            ("Debug Info", "debug_info",   "I",   False),
            ("Step Frame", "step_frame",   "N",   False),
            ("Quit",       "quit",         "Esc", False),
        ]
        for label, action_id, hint, toggle in control_buttons:
            self._buttons.append(
                Button(pygame.Rect(x, y, w, h), label, action_id,
                       key_hint=hint, toggle=toggle)
            )
            y += h + gap

    def init(self) -> None:
        """Initialize pygame and create the window."""
        if self._initialized:
            return

        pygame.init()
        pygame.display.set_caption(self.title)

        total_w = self.GAME_SIZE + self.SIDEBAR_WIDTH
        self._screen = pygame.display.set_mode((total_w, self.GAME_SIZE))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, self.HUD_FONT_SIZE)
        self._small_font = pygame.font.Font(None, 16)
        self._section_font = pygame.font.Font(None, 18)
        self._initialized = True

        self._create_buttons()

    def close(self) -> None:
        """Clean up pygame resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
            self._screen = None
            self._clock = None
            self._font = None
            self._small_font = None
            self._section_font = None

    def handle_mouse_click(self, mouse_pos: Tuple[int, int]) -> None:
        """Test all buttons against a click and queue any hits."""
        for btn in self._buttons:
            if btn.handle_click(mouse_pos):
                self._pending_clicks.append(btn.action_id)

    def pop_button_clicks(self) -> List[str]:
        """Return and clear the list of button action_ids clicked since last call."""
        clicks = self._pending_clicks
        self._pending_clicks = []
        return clicks

    def set_pause_state(self, paused: bool) -> None:
        """Sync the Pause toggle button with external state."""
        for btn in self._buttons:
            if btn.action_id == "pause_toggle":
                btn.toggled = paused

    def _jax_to_surface(self, obs: Any) -> pygame.Surface:
        if HAS_JAX and hasattr(obs, 'device'):
            arr = np.asarray(obs)
        else:
            arr = np.array(obs)

        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)

        if arr.shape != (self.ENV_SIZE, self.ENV_SIZE, 3):
            raise ValueError(
                f"Expected shape ({self.ENV_SIZE}, {self.ENV_SIZE}, 3), got {arr.shape}"
            )

        surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        scaled = pygame.transform.scale(surface, (self.GAME_SIZE, self.GAME_SIZE))
        return scaled

    def _render_text(self, text: str, pos: Tuple[int, int],
                     shadow: bool = True) -> None:
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
            self._render_text(f"Rot: {rotation:.1f}\u00b0", (self.HUD_PADDING, y))
            y += line_height

        mode_y = self.HUD_PADDING
        mode_text = f"Mode: {mode}"
        text_width = self._font.size(mode_text)[0]
        self._render_text(
            mode_text,
            (self.GAME_SIZE - text_width - self.HUD_PADDING, mode_y)
        )

        if paused:
            pause_text = "PAUSED"
            text_width = self._font.size(pause_text)[0]
            x = (self.GAME_SIZE - text_width) // 2
            y = self.GAME_SIZE // 2

            pygame.draw.rect(
                self._screen,
                (0, 0, 0, 128),
                (x - 10, y - 5, text_width + 20, self.HUD_FONT_SIZE + 10)
            )
            self._render_text(pause_text, (x, y))

    def _render_sidebar(self) -> None:
        sidebar_rect = pygame.Rect(self.GAME_SIZE, 0,
                                   self.SIDEBAR_WIDTH, self.GAME_SIZE)
        pygame.draw.rect(self._screen, self.SIDEBAR_BG, sidebar_rect)

        section_x = self.GAME_SIZE + 16
        label_surf = self._section_font.render(
            "MOVEMENT", True, self.SIDEBAR_SECTION_COLOR
        )
        self._screen.blit(label_surf, (section_x, 16))

        control_y = None
        for btn in self._buttons:
            if btn.action_id == "pause_toggle" and control_y is None:
                control_y = btn.rect.top - 20
        if control_y is not None:
            label_surf = self._section_font.render(
                "CONTROLS", True, self.SIDEBAR_SECTION_COLOR
            )
            self._screen.blit(label_surf, (section_x, control_y))

        mouse_pos = pygame.mouse.get_pos()
        for btn in self._buttons:
            btn.update_hover(mouse_pos)
            btn.draw(self._screen, self._font, self._small_font)

    def _render_overlay(self, message: str) -> None:
        """Render a semi-transparent overlay with a multi-line message."""
        overlay = pygame.Surface((self.GAME_SIZE, self.GAME_SIZE), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self._screen.blit(overlay, (0, 0))

        lines = message.split("\n")
        line_height = self.HUD_FONT_SIZE + 6
        total_height = len(lines) * line_height
        start_y = (self.GAME_SIZE - total_height) // 2

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            text_surf = self._font.render(line.strip(), True, self.HUD_COLOR)
            text_rect = text_surf.get_rect(
                centerx=self.GAME_SIZE // 2,
                y=start_y + i * line_height
            )
            self._screen.blit(text_surf, text_rect)

    def render(
        self,
        obs: Any,
        score: float = 0.0,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[float] = None,
        paused: bool = False,
        mode: str = "play",
        overlay_message: Optional[str] = None,
    ) -> float:
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

        if overlay_message is not None:
            self._render_overlay(overlay_message)

        self._render_sidebar()

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
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._actual_fps

    def __enter__(self) -> "Viewer":
        self.init()
        return self

    def __exit__(self, *args) -> None:
        self.close()
