"""
RGB chemical / intensity phase diagrams (KoSens ``plot_species_phase_diagram``).

At each grid point the three channel values are converted to fractional
contributions of their sum; those fractions paint an RGB image.  Optional
iso-ratio contours mark H↔H₂, C↔CO, and CO↔JCO transitions.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None


# KoSens defaults: dark blue / green / tab:red
DEFAULT_RGB_COLORS = (
    (0.0, 0.0, 0.545098),
    (0.0, 0.501961, 0.0),
    (0.839216, 0.152941, 0.156863),
)

TransitionSpec = Tuple[np.ndarray, str, str]  # (ratio grid, linestyle, name)


def exact_base_key(base_name: str, candidates: Iterable[str]) -> Optional[str]:
    """Match a bare species name exactly (``C`` not ``C+``); skip ratio keys.

    Ratio keys look like ``CO(1-0)/C(1-0)`` (slash in the species head).
    Fine-structure labels such as ``C+(3/2-1/2)`` are kept — the ``/`` is
    inside the transition parentheses.
    """
    base = str(base_name)
    for key in candidates:
        if not key:
            continue
        head = str(key).split('(', 1)[0]
        if '/' in head:
            continue
        if head == base:
            return str(key)
    return None


def positive_grid(values) -> np.ndarray:
    g = np.asarray(values, dtype=float).copy()
    g[g <= 0] = np.nan
    return g


def scale_channel_grids(
    grids: Sequence[np.ndarray],
    conv_factors: Optional[Sequence[float]] = None,
) -> List[np.ndarray]:
    """Multiply each channel grid by its conversion factor (default 1)."""
    if conv_factors is None:
        return [np.asarray(g, dtype=float) for g in grids]
    if len(conv_factors) != len(grids):
        raise ValueError(
            f'conv_factors length {len(conv_factors)} != grids length {len(grids)}'
        )
    out = []
    for g, cf in zip(grids, conv_factors):
        try:
            factor = float(cf)
        except (TypeError, ValueError):
            factor = 1.0
        if not np.isfinite(factor) or factor <= 0:
            factor = 1.0
        out.append(np.asarray(g, dtype=float) * factor)
    return out


def compose_rgb_image(
    grids: Sequence[np.ndarray],
    colors: Sequence[Sequence[float]] = DEFAULT_RGB_COLORS,
    conv_factors: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Build an ``(ny, nx, 3)`` RGB image in [0, 1] from three channel grids.

    Optional ``conv_factors`` scale each channel before the fractional mix
    (KoSens mass-density / unit conversion weights).

    Returns ``(rgb, fractions)`` where ``fractions[i]`` is the fractional
    contribution of channel *i* to the per-pixel total.
    """
    if len(grids) != 3:
        raise ValueError(f'Exactly 3 channel grids required, got {len(grids)}')
    if len(colors) != 3:
        raise ValueError(f'Exactly 3 colours required, got {len(colors)}')

    cleaned = [positive_grid(g) for g in scale_channel_grids(grids, conv_factors)]
    stacked = np.stack(cleaned, axis=0)
    total = np.nansum(stacked, axis=0)
    total = np.where(total > 0, total, np.nan)

    fractions = []
    for g in cleaned:
        frac = g / total
        frac = np.where(np.isfinite(frac), frac, 0.0)
        fractions.append(frac)

    ny, nx = cleaned[0].shape
    rgb = np.zeros((ny, nx, 3), dtype=float)
    for i in range(3):
        col = np.asarray(colors[i], dtype=float)
        for ch in range(3):
            rgb[:, :, ch] += fractions[i] * float(col[ch])
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb, fractions


def ratio_grid(numerator, denominator) -> np.ndarray:
    """Positive-valued ratio grid ``num / den`` (NaN where undefined)."""
    num = positive_grid(numerator)
    den = positive_grid(denominator)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def hh2_ratio(h_grid, h2_grid) -> np.ndarray:
    """``H / (2 H2)``; contour level 1 marks the H↔H₂ transition."""
    return ratio_grid(h_grid, 2.0 * positive_grid(h2_grid))


def auto_label_positions(
    fractions: Sequence[np.ndarray],
    x_plot: np.ndarray,
    y_plot: np.ndarray,
) -> List[Tuple[Optional[float], Optional[float]]]:
    """Centroid of each channel's dominance region in plot coordinates."""
    stacked = np.stack([np.asarray(f, dtype=float) for f in fractions], axis=0)
    dominant = np.argmax(stacked, axis=0)
    x_plot = np.asarray(x_plot, dtype=float)
    y_plot = np.asarray(y_plot, dtype=float)
    out = []
    for i in range(3):
        mask = dominant == i
        if not np.any(mask):
            frac = stacked[i]
            if not np.any(np.isfinite(frac)):
                out.append((None, None))
                continue
            flat = np.nanargmax(frac)
            row, col = np.unravel_index(flat, frac.shape)
            out.append((float(x_plot[col]), float(y_plot[row])))
            continue
        rows, cols = np.where(mask)
        out.append((float(np.median(x_plot[cols])), float(np.median(y_plot[rows]))))
    return out


def _resample_rgb_square(rgb: np.ndarray, n_pix: int = 120) -> np.ndarray:
    """Nearest-neighbour resample to an ``n_pix × n_pix`` RGB texture."""
    rgb = np.asarray(rgb, dtype=float)
    ny, nx, nch = rgb.shape
    if ny < 1 or nx < 1:
        return rgb
    if ny == n_pix and nx == n_pix:
        return rgb
    # Map each output pixel to the nearest native cell.
    row_idx = np.clip(
        np.floor(np.linspace(0, ny, n_pix, endpoint=False)).astype(int),
        0, ny - 1,
    )
    col_idx = np.clip(
        np.floor(np.linspace(0, nx, n_pix, endpoint=False)).astype(int),
        0, nx - 1,
    )
    return rgb[np.ix_(row_idx, col_idx)]


def _rgb_png_data_uri(rgb_u8: np.ndarray) -> str:
    """Encode an RGB ``uint8`` array as a PNG data URI for layout images."""
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8)
    if PILImage is None:
        raise ImportError(
            'Pillow is required for RGB phase diagrams (pip install Pillow)'
        )
    buf = BytesIO()
    PILImage.fromarray(rgb_u8, mode='RGB').save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def fig_rgb_phase(
    rgb: np.ndarray,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    channel_labels: Sequence[str],
    label_xy: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    transitions: Optional[Sequence[TransitionSpec]] = None,
    theme_colors: Optional[dict] = None,
    width: int = 630,
    height: int = 510,
) -> go.Figure:
    """Plotly RGB phase diagram with optional white iso-ratio contours.

    Uses a layout image with ``sizing='stretch'`` so every panel fills the same
    axes box.  ``go.Image`` preserves data-coordinate aspect and makes slices
    with unequal log spans (e.g. dens–FUV) look non-square.
    """
    t = theme_colors or {}
    paper = t.get('paper_bg', 'white')
    plot_bg = t.get('plot_bg', 'white')
    font = t.get('font', '#222')
    title_c = t.get('title', font)
    grid_c = t.get('grid', '#ddd')
    axis_c = t.get('axis_line', '#444')

    x_plot = np.asarray(x_plot, dtype=float)
    y_plot = np.asarray(y_plot, dtype=float)
    ny, nx, _ = np.asarray(rgb).shape
    if nx < 1 or ny < 1:
        fig = go.Figure()
        fig.update_layout(title=title, paper_bgcolor=paper, plot_bgcolor=plot_bg)
        return fig

    dx_native = float(x_plot[-1] - x_plot[0]) / max(nx - 1, 1)
    dy_native = float(y_plot[-1] - y_plot[0]) / max(ny - 1, 1)
    x_lo = float(x_plot[0]) - 0.5 * dx_native
    x_hi = float(x_plot[-1]) + 0.5 * dx_native
    y_lo = float(y_plot[0]) - 0.5 * dy_native
    y_hi = float(y_plot[-1]) + 0.5 * dy_native

    n_pix = 120
    rgb_sq = _resample_rgb_square(rgb, n_pix=n_pix)
    rgb_u8 = np.clip(np.round(rgb_sq * 255.0), 0, 255).astype(np.uint8)
    # layout images have y upward when yref is a cartesian axis; array row 0
    # is the top of the PNG, so flip to match Contour (y increases upward).
    png_uri = _rgb_png_data_uri(rgb_u8[::-1, :, :])

    fig = go.Figure()
    # Invisible anchors keep axis autorange/export paths happy when there are
    # no transition contours yet.
    fig.add_trace(go.Scatter(
        x=[x_lo, x_hi],
        y=[y_lo, y_hi],
        mode='markers',
        marker=dict(opacity=0, size=1),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_layout_image(dict(
        source=png_uri,
        xref='x',
        yref='y',
        x=x_lo,
        y=y_hi,
        sizex=x_hi - x_lo,
        sizey=y_hi - y_lo,
        sizing='stretch',
        layer='below',
    ))

    for ratio, linestyle, _name in (transitions or []):
        if ratio is None:
            continue
        z = np.asarray(ratio, dtype=float)
        finite = z[np.isfinite(z)]
        if finite.size == 0:
            continue
        if not (float(np.nanmin(finite)) <= 1.0 <= float(np.nanmax(finite))):
            continue
        if linestyle in ('--', 'dash'):
            dash = 'dash'
        elif linestyle in ('-.', 'dashdot'):
            dash = 'dashdot'
        elif linestyle in (':', 'dot'):
            dash = 'dot'
        else:
            dash = 'solid'
        fig.add_trace(go.Contour(
            x=x_plot,
            y=y_plot,
            z=z,
            contours=dict(
                coloring='none',
                showlabels=False,
                start=1.0,
                end=1.0,
                size=1.0,
            ),
            line=dict(color='white', width=2.0, dash=dash),
            showscale=False,
            hoverinfo='skip',
            name=_name,
        ))

    annotations = []
    positions = list(label_xy or [(None, None)] * 3)
    for i, label in enumerate(channel_labels):
        if i >= len(positions):
            break
        lx, ly = positions[i]
        if lx is None or ly is None or not label:
            continue
        annotations.append(dict(
            x=float(lx),
            y=float(ly),
            text=str(label),
            showarrow=False,
            font=dict(size=14, color='white', family='Arial, sans-serif'),
            bgcolor='rgba(0,0,0,0.35)',
            borderpad=3,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color=title_c), x=0.02, xanchor='left'),
        paper_bgcolor=paper,
        plot_bgcolor=plot_bg,
        width=width,
        height=height,
        autosize=False,
        margin=dict(l=64, r=90, t=52, b=58),
        font=dict(family='Arial, sans-serif', size=12, color=font),
        annotations=annotations,
        showlegend=False,
        uirevision=f'rgb|{title}',
        xaxis=dict(
            title=dict(text=xlabel, font=dict(size=11, color=font)),
            range=[x_lo, x_hi],
            autorange=False,
            showgrid=False,
            gridcolor=grid_c,
            linecolor=axis_c,
            tickfont=dict(color=font),
            constrain='domain',
            domain=[0.0, 0.84],
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(size=11, color=font)),
            range=[y_lo, y_hi],
            autorange=False,
            showgrid=False,
            gridcolor=grid_c,
            linecolor=axis_c,
            tickfont=dict(color=font),
            constrain='domain',
            domain=[0.0, 1.0],
        ),
    )
    return fig


def build_rgb_figure(
    grids: Sequence[np.ndarray],
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    channel_labels: Sequence[str],
    colors: Sequence[Sequence[float]] = DEFAULT_RGB_COLORS,
    transitions: Optional[Sequence[TransitionSpec]] = None,
    theme_colors: Optional[dict] = None,
    conv_factors: Optional[Sequence[float]] = None,
    width: int = 630,
    height: int = 510,
) -> go.Figure:
    """Compose RGB + auto labels and return a ready Plotly figure."""
    rgb, fractions = compose_rgb_image(
        grids, colors=colors, conv_factors=conv_factors)
    label_xy = auto_label_positions(fractions, x_plot, y_plot)
    return fig_rgb_phase(
        rgb, x_plot, y_plot,
        xlabel=xlabel, ylabel=ylabel, title=title,
        channel_labels=channel_labels,
        label_xy=label_xy,
        transitions=transitions,
        theme_colors=theme_colors,
        width=width, height=height,
    )
