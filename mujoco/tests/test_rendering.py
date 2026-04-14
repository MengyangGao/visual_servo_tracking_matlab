from __future__ import annotations

import numpy as np

from mujoco_servo.rendering import mask_thumbnail, overlay_thumbnail, world_map_panel


def test_mask_thumbnail_and_overlay_thumbnail() -> None:
    mask = np.zeros((16, 24), dtype=np.uint8)
    mask[4:12, 8:20] = 1

    thumb = mask_thumbnail(mask, size=(40, 28))
    assert thumb is not None
    assert thumb.shape == (28, 40, 3)

    canvas = np.zeros((64, 80, 3), dtype=np.uint8)
    out = overlay_thumbnail(canvas, thumb, corner="upper_left", margin=2)

    assert out.shape == canvas.shape
    assert np.count_nonzero(out) > 0


def test_world_map_panel_renders_trails_and_markers() -> None:
    target = [np.array([0.0, 0.0, 0.4]), np.array([0.1, 0.0, 0.4]), np.array([0.2, 0.1, 0.4])]
    ee = [np.array([0.0, 0.2, 0.4]), np.array([0.1, 0.15, 0.4]), np.array([0.15, 0.1, 0.4])]

    panel = world_map_panel(
        target,
        ee,
        current_target=target[-1],
        current_ee=ee[-1],
        center_xy=(0.1, 0.05),
        extent_xy=(0.8, 0.6),
        phase="tracking",
        position_error_norm=0.12,
        feature_error_norm=0.03,
    )

    assert panel.shape == (240, 320, 3)
    assert np.count_nonzero(panel) > 0
