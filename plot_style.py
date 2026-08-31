"""Shared Plotly font sizes for every figure in the explorer.

Change these values to rescale titles, axes, ticks, colorbars, and legends
across the app (profiles, grids, intensities, RGB, χ², map fits, …).
"""

PLOT_FONT_FAMILY = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"

# Axis titles (x / y labels)
PLOT_AXIS_FONT_SIZE = 14

# Tick labels on axes
PLOT_TICK_FONT_SIZE = 12

# Figure titles
PLOT_TITLE_FONT_SIZE = 14

# Colorbar title and ticks
PLOT_CBAR_FONT_SIZE = 13
PLOT_CBAR_TICK_FONT_SIZE = 12

# Legends
PLOT_LEGEND_FONT_SIZE = 11


def axis_title_font(color=None):
    kw = dict(size=PLOT_AXIS_FONT_SIZE, family=PLOT_FONT_FAMILY)
    if color is not None:
        kw['color'] = color
    return kw


def tick_font(color=None):
    kw = dict(size=PLOT_TICK_FONT_SIZE, family=PLOT_FONT_FAMILY)
    if color is not None:
        kw['color'] = color
    return kw


def title_font(color=None):
    kw = dict(size=PLOT_TITLE_FONT_SIZE, family=PLOT_FONT_FAMILY)
    if color is not None:
        kw['color'] = color
    return kw


def cbar_title_font(color=None):
    kw = dict(size=PLOT_CBAR_FONT_SIZE, family=PLOT_FONT_FAMILY)
    if color is not None:
        kw['color'] = color
    return kw


def cbar_tick_font(color=None):
    kw = dict(size=PLOT_CBAR_TICK_FONT_SIZE, family=PLOT_FONT_FAMILY)
    if color is not None:
        kw['color'] = color
    return kw


def legend_font(color=None):
    kw = dict(size=PLOT_LEGEND_FONT_SIZE, family=PLOT_FONT_FAMILY)
    if color is not None:
        kw['color'] = color
    return kw


def layout_font(color=None):
    kw = dict(family=PLOT_FONT_FAMILY, size=PLOT_TICK_FONT_SIZE)
    if color is not None:
        kw['color'] = color
    return kw
