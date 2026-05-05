"""Step 5 HUD overlay helpers.

This package contains HUD drawing functions for the video renderer:
- hud_renderer: Extracted skeleton/bar-path/legend drawing, HUD orchestration
- sparkline: Velocity sparkline rendering
- power_band: Power zone band rendering
- joint_angles: Knee angle display with color coding
- error_markers: Fault marker placement and rendering
"""

from dataclasses import dataclass


@dataclass
class HUDConfig:
    """Configuration for HUD element visibility."""
    show_skeleton: bool = True
    show_sparkline: bool = True
    show_power_zones: bool = True
    show_angles: bool = True
    show_error_markers: bool = True


# Lazy imports — modules created in subsequent waves
def draw_hud_overlay(*args, **kwargs):
    from .hud_renderer import draw_hud_overlay as _impl
    return _impl(*args, **kwargs)


def draw_velocity_sparkline(*args, **kwargs):
    from .sparkline import draw_velocity_sparkline as _impl
    return _impl(*args, **kwargs)


def draw_power_zone_band(*args, **kwargs):
    from .power_band import draw_power_zone_band as _impl
    return _impl(*args, **kwargs)


def draw_knee_angles(*args, **kwargs):
    from .joint_angles import draw_knee_angles as _impl
    return _impl(*args, **kwargs)


def draw_error_markers(*args, **kwargs):
    from .error_markers import draw_error_markers as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "HUDConfig",
    "draw_hud_overlay",
    "draw_velocity_sparkline",
    "draw_power_zone_band",
    "draw_knee_angles",
    "draw_error_markers",
]
