"""Photochemical transition fronts over a PDR model grid (Grids -> 3D fronts).

Front definitions follow Gaches, Bisbas, Brendel & Zhu (arXiv:2608.04116,
Sec. 3.4): the H2 dissociation front sits where x(H2) = 0.25 (x(H) = 2x(H2)
= 0.5) and the C+ recombination front where x(C+)/x_C = 1/e. The paper does
not define the C/CO transition; here it is where CO holds half the carbon,
x(CO)/x_C = 0.5. x_C = x(C+) + x(C) + x(CO) (the dominant carbon carriers).

Two views:
  * front depth  -- A_V of each front for every model of a 2-D parameter
    plane, drawn as surfaces (the paper's Fig. 7 analogue, with depth as the
    spatial axis);
  * parameter regimes -- the full n x chi x zeta cube, one point per model
    coloured by its dominant zone, with isosurfaces of the column-integrated
    fractions at the same thresholds.
"""
import math

import numpy as np
import plotly.graph_objects as go
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

import plot_style as ps

FRONTS = [
    dict(key='h2', label='H/H₂ dissociation', colour='#E5484D', default_level=0.25,
         rule='x(H₂)', rising=True),
    dict(key='cplus', label='C⁺/C recombination', colour='#3DD6F5',
         default_level=round(1.0 / math.e, 3), rule='x(C⁺)/x_C', rising=False),
    dict(key='co', label='C/CO', colour='#46A758', default_level=0.5,
         rule='x(CO)/x_C', rising=True),
]
FRONT_BY_KEY = {f['key']: f for f in FRONTS}
FRONT_SPECIES = ('H', 'H2', 'C+', 'C', 'CO')

# Media between the carbon fronts: index 0 / 1 / 2 in ``classify_media``.
MEDIA = [
    dict(key='C+', label='Ionized (C\u207a)', colour='#E5484D'),
    dict(key='C', label='Atomic (C)', colour='#3DD6F5'),
    dict(key='CO', label='Molecular (CO)', colour='#46A758'),
]


def _carbon_fractions(c_plus, c, co):
    c_plus, c, co = (np.asarray(a, dtype=float) for a in (c_plus, c, co))
    x_c = c_plus + c + co
    with np.errstate(divide='ignore', invalid='ignore'):
        safe = np.where(x_c > 0, x_c, np.nan)
        return c_plus / safe, co / safe


def front_fractions(rel, species_idx):
    """Per-depth fractions that define each front, from ``rel`` = n/n_H [depth, species].

    A front is omitted when one of its species is not in the network.
    """
    rel = np.asarray(rel, dtype=float)
    out = {}
    if rel.ndim != 2 or rel.shape[0] == 0:
        return out
    idx = species_idx or {}
    if 'H2' in idx:
        out['h2'] = rel[:, idx['H2']]
    if all(s in idx for s in ('C+', 'C', 'CO')):
        out['cplus'], out['co'] = _carbon_fractions(
            rel[:, idx['C+']], rel[:, idx['C']], rel[:, idx['CO']])
    return out


def front_depth(av, frac, level, rising=True):
    """A_V where the front is first reached going inward; NaN if never reached.

    A front is reached where ``frac >= level`` (``rising``: H2, CO build up
    inward) or ``frac <= level`` (C+ is used up inward). Gas that is already
    past the front at the surface gives the surface A_V.
    """
    av = np.asarray(av, dtype=float)
    frac = np.asarray(frac, dtype=float)
    n = min(av.size, frac.size)
    if n == 0 or level is None or not np.isfinite(level):
        return np.nan
    av, frac = av[:n], frac[:n]
    with np.errstate(invalid='ignore'):
        reached = frac >= level if rising else frac <= level
    hits = np.flatnonzero(reached & np.isfinite(frac))
    if hits.size == 0:
        return np.nan
    i = int(hits[0])
    if i == 0 or not np.isfinite(frac[i - 1]):
        return float(av[i])
    f0, f1 = frac[i - 1], frac[i]
    w = (level - f0) / (f1 - f0) if f1 != f0 else 1.0
    return float(av[i - 1] + w * (av[i] - av[i - 1]))


def column_fractions(cols):
    """Column-integrated front fractions from ``cols`` = {species: N}."""
    out = {}
    if 'H' in cols and 'H2' in cols:
        n_h = cols['H'] + 2.0 * cols['H2']
        # x(H2) = n(H2)/n_H, so the column analogue is N(H2)/N_H.
        out['h2'] = cols['H2'] / n_h if n_h > 0 else np.nan
    if all(s in cols for s in ('C+', 'C', 'CO')):
        cp, co = _carbon_fractions(cols['C+'], cols['C'], cols['CO'])
        out['cplus'], out['co'] = float(cp), float(co)
    return out


def _level(levels, key):
    return levels.get(key, FRONT_BY_KEY[key]['default_level'])


def classify_media(fracs, levels):
    """Medium index per point (0 ionized C+, 1 atomic C, 2 molecular CO, -1 unknown)
    and a molecular-hydrogen mask. ``fracs`` maps front key -> array (same shape)."""
    co = np.asarray(fracs.get('co', np.nan), dtype=float)
    cp = np.asarray(fracs.get('cplus', np.nan), dtype=float)
    h2 = np.asarray(fracs.get('h2', np.nan), dtype=float)
    shape = np.broadcast_shapes(co.shape, cp.shape, h2.shape)
    co, cp, h2 = (np.broadcast_to(a, shape) for a in (co, cp, h2))
    with np.errstate(invalid='ignore'):
        medium = np.where(np.isfinite(cp), np.where(cp >= _level(levels, 'cplus'), 0, 1), -1)
        medium = np.where(np.isfinite(co) & (co >= _level(levels, 'co')), 2, medium)
        molecular = np.isfinite(h2) & (h2 >= _level(levels, 'h2'))
    return medium, molecular


def zone_of(fracs, levels):
    """Medium key ('C+', 'C', 'CO' or None) and molecular-H flag for one model."""
    medium, molecular = classify_media(fracs, levels)
    m = int(medium)
    return (MEDIA[m]['key'] if m >= 0 else None), bool(molecular)


def refine(axes, values, n):
    """Linearly resample ``values`` (shape = axis lengths) onto ``n`` points per axis.

    Axes are plot coordinates (log10 parameters), so this interpolates in log
    parameter space. NaNs (fronts not reached) stay NaN around their cells.
    """
    axes = [np.asarray(a, dtype=float) for a in axes]
    fine = [np.linspace(a.min(), a.max(), int(n)) if a.size > 1 else a for a in axes]
    interp = RegularGridInterpolator(axes, np.asarray(values, dtype=float),
                                     bounds_error=False, fill_value=np.nan)
    mesh = np.meshgrid(*fine, indexing='ij')
    pts = np.stack([m.ravel() for m in mesh], axis=-1)
    return fine, interp(pts).reshape(mesh[0].shape)


def fill_nan_nearest(vol):
    """Replace NaNs with the nearest finite value (marching cubes needs a full grid)."""
    vol = np.asarray(vol, dtype=float)
    bad = ~np.isfinite(vol)
    if not bad.any() or bad.all():
        return vol
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return vol[tuple(idx)]


# --- Figures -------------------------------------------------------------------

def _title_box(text, t, dark):
    return dict(
        text=f'<b>{text}</b>', xref='paper', yref='paper',
        x=0.0, xanchor='left', y=1.0, yanchor='bottom', yshift=6,
        showarrow=False, align='left',
        font=dict(size=ps.PLOT_TITLE_FONT_SIZE + 2, color=t['title'],
                  family=ps.PLOT_FONT_FAMILY),
        bgcolor=('#1F1F23' if dark else '#EAF2FF'),
        bordercolor=('#52525B' if dark else t.get('accent', '#1560CE')),
        borderwidth=1.2, borderpad=5,
    )


def _scene_layout(t, labels, title, height, dark):
    axis_kw = dict(backgroundcolor=t['plot_bg'], gridcolor=t['grid'], showbackground=True,
                   zerolinecolor=t['axis_line'], tickfont=ps.tick_font(t['font']))
    return dict(
        paper_bgcolor=t['paper_bg'],
        font=ps.layout_font(t['font']),
        height=height,
        margin=dict(l=0, r=0, t=56, b=0),
        annotations=[_title_box(title, t, dark)],
        legend=dict(bgcolor=t['legend_bg'], bordercolor=t['legend_border'], borderwidth=1,
                    font=ps.legend_font(t['font']), x=0.01, y=0.95, xanchor='left'),
        scene=dict(
            xaxis=dict(title=dict(text=labels[0], font=ps.axis_title_font(t['font'])), **axis_kw),
            yaxis=dict(title=dict(text=labels[1], font=ps.axis_title_font(t['font'])), **axis_kw),
            zaxis=dict(title=dict(text=labels[2], font=ps.axis_title_font(t['font'])), **axis_kw),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
        ),
    )


def _flat_colorscale(colour):
    return [[0.0, colour], [1.0, colour]]


def fig_front_surfaces(x, y, depth_by_front, t, labels, *, av_center=None, dark=True,
                       title='Front depth', height=680, uirevision=None):
    """One surface A_V_front(x, y) per front; NaN (front not reached) leaves a hole.

    ``depth_by_front`` maps front key -> array [len(y), len(x)].
    """
    fig = go.Figure()
    for key, Z in depth_by_front.items():
        spec = FRONT_BY_KEY[key]
        fig.add_trace(go.Surface(
            x=x, y=y, z=Z, name=spec['label'], showlegend=True, showscale=False,
            colorscale=_flat_colorscale(spec['colour']), opacity=0.85,
            hovertemplate=(f'{spec["label"]}<br>{labels[0]} = %{{x:.2f}}<br>'
                           f'{labels[1]} = %{{y:.2f}}<br>'
                           'A<sub>V</sub> = %{z:.3g} mag<extra></extra>'),
        ))
    if av_center is not None:
        fig.add_trace(go.Surface(
            x=x, y=y, z=av_center, name='clump centre', showlegend=True, showscale=False,
            colorscale=_flat_colorscale(t['muted']), opacity=0.18,
            hovertemplate='clump centre A<sub>V</sub> = %{z:.3g} mag<extra></extra>',
        ))
    fig.update_layout(**_scene_layout(t, labels, title, height, dark),
                      uirevision=uirevision or 'g3d-front')
    return fig


def _model_points_3d(fig, GX, GY, GZ, medium, molecular, labels):
    for m, spec in enumerate(MEDIA):
        for is_mol in (True, False):
            sel = (medium == m) & (molecular == is_mol)
            if not sel.any():
                continue
            name = spec['label'] + (', H\u2082' if is_mol else ', atomic H')
            fig.add_trace(go.Scatter3d(
                x=GX[sel], y=GY[sel], z=GZ[sel], mode='markers', name=name,
                marker=dict(size=6, color=spec['colour'] if is_mol else 'rgba(0,0,0,0)',
                            line=dict(color=spec['colour'], width=2)),
                hovertemplate=(f'{name}<br>{labels[0]} = %{{x:.2f}}<br>'
                               f'{labels[1]} = %{{y:.2f}}<br>'
                               f'{labels[2]} = %{{z:.2f}}<extra></extra>'),
            ))


def fig_regime_cube(X, Y, Z, fracs_cube, levels, t, labels, *, fronts=None, dark=True,
                    style='points', fine=None, title='Parameter regimes', height=680,
                    uirevision=None):
    """The n x chi x zeta cube coloured by medium, plus one isosurface per front.

    ``fracs_cube`` maps front key -> array [len(X), len(Y), len(Z)] of
    column-integrated fractions on the native grid. ``style='points'`` draws
    one marker per model (filled = molecular H, hollow = atomic H);
    ``style='volumes'`` draws each medium as a closed translucent volume.
    ``fine`` = ((X, Y, Z), fracs) is an optional interpolated copy used for
    the volumes and isosurfaces; the points always show the real models.
    """
    fronts = list(fronts if fronts is not None else fracs_cube)
    fig = go.Figure()
    sX, sY, sZ, sfr = (*fine[0], fine[1]) if fine is not None else (X, Y, Z, fracs_cube)
    SX, SY, SZ = np.meshgrid(sX, sY, sZ, indexing='ij')

    if style == 'volumes':
        medium, _ = classify_media({k: fill_nan_nearest(v) for k, v in sfr.items()}, levels)
        for m, spec in enumerate(MEDIA):
            ind = (medium == m).astype(float)
            if not ind.any():
                continue
            fig.add_trace(go.Isosurface(
                x=SX.ravel(), y=SY.ravel(), z=SZ.ravel(), value=ind.ravel(),
                isomin=0.5, isomax=1.0, surface_count=1,
                caps=dict(x_show=True, y_show=True, z_show=True),
                colorscale=_flat_colorscale(spec['colour']), showscale=False, opacity=0.45,
                name=spec['label'], showlegend=True,
                hovertemplate=f'{spec["label"]}<extra></extra>',
            ))
    else:
        GX, GY, GZ = np.meshgrid(X, Y, Z, indexing='ij')
        medium, molecular = classify_media(fracs_cube, levels)
        _model_points_3d(fig, GX, GY, GZ, medium, molecular, labels)

    for key in fronts:
        vol = sfr.get(key)
        if vol is None or not np.isfinite(vol).any():
            continue
        level = _level(levels, key)
        filled = fill_nan_nearest(vol)
        if not (np.nanmin(filled) < level < np.nanmax(filled)):
            continue  # the whole cube sits on one side: no surface to draw
        spec = FRONT_BY_KEY[key]
        fig.add_trace(go.Isosurface(
            x=SX.ravel(), y=SY.ravel(), z=SZ.ravel(), value=filled.ravel(),
            isomin=level, isomax=level, surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale=_flat_colorscale(spec['colour']), showscale=False,
            opacity=0.2 if style == 'volumes' else 0.35,
            name=f'{spec["label"]} front', showlegend=True,
            hovertemplate=f'{spec["label"]} front<extra></extra>',
        ))
    fig.update_layout(**_scene_layout(t, labels, title, height, dark),
                      uirevision=uirevision or 'g3d-regime')
    return fig


_FRONT_DASH = {'h2': 'solid', 'cplus': 'dash', 'co': 'dot'}


def fig_media_map(x, y, fracs, levels, t, labels, *, fronts=None, dark=True,
                  style='filled', fine=None, title='Media', height=620, uirevision=None):
    """2-D map of the medium over two grid parameters at a fixed third one.

    ``fracs`` maps front key -> array [len(y), len(x)] (native grid).
    ``style='filled'`` colours the plane by medium with the fronts as contour
    lines; ``style='points'`` draws one marker per model. ``fine`` = ((y, x),
    fracs) is an optional interpolated copy for the filled view.
    """
    fronts = list(fronts if fronts is not None else fracs)
    fig = go.Figure()
    if style == 'filled':
        (fy, fx), ffr = fine if fine is not None else ((y, x), fracs)
        medium, _ = classify_media(ffr, levels)
        z = np.where(medium >= 0, medium, np.nan).astype(float)
        n = len(MEDIA)
        scale = []
        for m, spec in enumerate(MEDIA):
            scale += [[m / n, spec['colour']], [(m + 1) / n, spec['colour']]]
        names = np.array([s['label'] for s in MEDIA] + [''])[np.where(medium >= 0, medium, n)]
        fig.add_trace(go.Heatmap(
            x=fx, y=fy, z=z, zmin=-0.5, zmax=n - 0.5, colorscale=scale, opacity=0.8,
            showscale=False, customdata=names, name='medium',
            hovertemplate=(f'%{{customdata}}<br>{labels[0]} = %{{x:.2f}}<br>'
                           f'{labels[1]} = %{{y:.2f}}<extra></extra>'),
        ))
        for spec in MEDIA:  # legend swatches for the filled colours
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=spec['label'],
                                     marker=dict(symbol='square', size=12, color=spec['colour'])))
        for key in fronts:
            Z = ffr.get(key)
            if Z is None or not np.isfinite(Z).any():
                continue
            spec = FRONT_BY_KEY[key]
            level = _level(levels, key)
            fig.add_trace(go.Contour(
                x=fx, y=fy, z=Z, showscale=False, name=f'{spec["label"]} front',
                showlegend=True, contours=dict(start=level, end=level, size=1,
                                               coloring='lines'),
                line=dict(color=t['title'], width=2.5, dash=_FRONT_DASH[key]),
                colorscale=_flat_colorscale(t['title']), hoverinfo='skip',
            ))
    else:
        GX, GY = np.meshgrid(x, y)
        medium, molecular = classify_media(fracs, levels)
        for m, spec in enumerate(MEDIA):
            for is_mol in (True, False):
                sel = (medium == m) & (molecular == is_mol)
                if not sel.any():
                    continue
                name = spec['label'] + (', H\u2082' if is_mol else ', atomic H')
                fig.add_trace(go.Scatter(
                    x=GX[sel], y=GY[sel], mode='markers', name=name,
                    marker=dict(size=16, symbol='square',
                                color=spec['colour'] if is_mol else 'rgba(0,0,0,0)',
                                line=dict(color=spec['colour'], width=2.5)),
                    hovertemplate=(f'{name}<br>{labels[0]} = %{{x:.2f}}<br>'
                                   f'{labels[1]} = %{{y:.2f}}<extra></extra>'),
                ))
    axis_kw = dict(showgrid=True, gridcolor=t['grid'], linecolor=t['axis_line'], mirror=True,
                   zeroline=False, tickfont=ps.tick_font(t['font']))
    fig.update_layout(
        paper_bgcolor=t['paper_bg'], plot_bgcolor=t['plot_bg'],
        font=ps.layout_font(t['font']), height=height,
        margin=dict(l=70, r=20, t=60, b=60),
        annotations=[_title_box(title, t, dark)],
        legend=dict(bgcolor=t['legend_bg'], bordercolor=t['legend_border'], borderwidth=1,
                    font=ps.legend_font(t['font']), x=1.02, xanchor='left', y=1.0),
        xaxis=dict(title=dict(text=labels[0], font=ps.axis_title_font(t['font'])), **axis_kw),
        yaxis=dict(title=dict(text=labels[1], font=ps.axis_title_font(t['font'])), **axis_kw),
        uirevision=uirevision or 'g3d-media',
    )
    return fig


if __name__ == '__main__':
    _av = np.linspace(0, 10, 11)
    assert abs(front_depth(_av, _av / 10, 0.25) - 2.5) < 1e-9
    assert abs(front_depth(_av, 1 - _av / 10, 0.25, rising=False) - 7.5) < 1e-9
    assert np.isnan(front_depth(_av, np.zeros(11), 0.25))
    print('ok')
