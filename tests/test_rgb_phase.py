import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rgb_phase


def test_fourth_channel_joins_fractional_total():
    # Left half: JCO dominates (grey), right half: CO dominates (red).
    small, big = np.full((4, 6), 1e-8), np.full((4, 6), 1e-4)
    co, jco = small.copy(), small.copy()
    co[:, 3:], jco[:, :3] = big[:, 3:], big[:, :3]
    grey = (0.5, 0.5, 0.5)
    rgb, fr = rgb_phase.compose_rgb_image(
        [small, small, co, jco], colors=tuple(rgb_phase.DEFAULT_RGB_COLORS) + (grey,))
    assert np.allclose(sum(fr), 1.0)
    assert np.allclose(rgb[:, 0], grey, atol=1e-3)
    assert rgb[0, -1, 0] > 0.8 and rgb[0, -1, 2] < 0.2
    pos = rgb_phase.auto_label_positions(fr, np.arange(6.0), np.arange(4.0))
    assert len(pos) == 4 and pos[3][0] < pos[2][0]
