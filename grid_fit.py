"""
Build 3-D SIMLINE intensity cubes and fit observational FITS maps.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import grid_interp as gi
import grid_naming as gn
import map_fit as _map_fit_mod
import smli_labels

try:
    from astropy.io import fits
except ImportError:  # pragma: no cover
    fits = None

AXIS_X = 'density'
AXIS_Y = 'fuv'
AXIS_Z = 'crir'

# Default mesh keys for the classic PDR triple (overridden per grid via ``resolve_fit_axes``).
X_MESH_KEY = gn.PARAM_MESH_KEYS['density']
Y_MESH_KEY = gn.PARAM_MESH_KEYS['fuv']
Z_MESH_KEY = gn.PARAM_MESH_KEYS['crir']

_PARAM_FIT_META = {
    'density': dict(
        set_name='n',
        units=r'cm$^{-3}$',
        map_title='Best-fit hydrogen density',
        colorbar='log<sub>10</sub> n (cm<sup>-3</sup>)',
    ),
    'mass': dict(
        set_name='M',
        units=r'M$_\odot$',
        map_title='Best-fit clump mass',
        colorbar='log<sub>10</sub> M',
    ),
    'fuv': dict(
        set_name='FUV',
        units='Draine',
        map_title='Best-fit FUV field',
        colorbar='log<sub>10</sub> χ',
    ),
    'metal': dict(
        set_name='Z',
        units=r'Z$_\odot$',
        map_title='Best-fit metallicity',
        colorbar='log<sub>10</sub> Z',
    ),
    'crir': dict(
        set_name=r'$\zeta_{\mathrm{H}}$',
        units=r'$s^{-1}$',
        map_title='Best-fit cosmic-ray rate',
        colorbar='log<sub>10</sub> ζ',
    ),
    'atten': dict(
        set_name='atten',
        units='',
        map_title='Best-fit attenuation',
        colorbar='atten',
    ),
}

DEFAULT_TARGET_SHAPE_3D = (60, 60, 60)
TARGET_UNITS_JTEMP = 'K km/s'
TARGET_UNITS_JERG = 'erg s-1 cm-2'


def _format_transition_label(raw_transition: str) -> str:
    return smli_labels.format_smli_transition_label(raw_transition)


def spectroscopic_line_key(species: str, raw_transition: str) -> str:
    """KoSens-style line name, e.g. ``CO(1-0)``."""
    label = _format_transition_label(raw_transition)
    return f'{species}({label})'


def list_simline_line_keys(simline: dict, idef: str) -> List[str]:
    """All spectroscopic line keys available in a scanned SIMLINE directory."""
    if not simline:
        return []
    keys = set()
    transitions = simline.get('transitions') or {}
    for (species, idf), rows in transitions.items():
        if idf != idef:
            continue
        for row in rows:
            raw = row.get('transition')
            if raw:
                keys.add(spectroscopic_line_key(species, raw))
    return sorted(keys)


def _axis_log10(phys: np.ndarray, logscale: bool) -> np.ndarray:
    phys = np.asarray(phys, dtype=float)
    if logscale:
        return np.log10(np.maximum(phys, np.finfo(float).tiny))
    return phys


def _sort_axis(phys: np.ndarray, logscale: bool) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(_axis_log10(phys, logscale))
    return phys[order], order


def _line_lookup(simline: dict, idef: str) -> Dict[str, Tuple[str, int]]:
    """Map spectroscopic line key -> (species, transition row index)."""
    out = {}
    transitions = simline.get('transitions') or {}
    for (species, idf), rows in transitions.items():
        if idf != idef:
            continue
        for row in rows:
            raw = row.get('transition')
            if not raw:
                continue
            key = spectroscopic_line_key(species, raw)
            out[key] = (species, int(row['idx']))
    return out


def fit_axis_config(axis_tokens: dict) -> dict:
    """Resolve 3-D fit axes and mesh keys from loaded grid ``axis_tokens``."""
    axis_x, axis_y, axis_z = gn.resolve_fit_axes(axis_tokens)
    x_mesh = gn.mesh_key_for_param(axis_x)
    y_mesh = gn.mesh_key_for_param(axis_y)
    z_mesh = gn.mesh_key_for_param(axis_z)
    meta_x = _PARAM_FIT_META[axis_x]
    meta_y = _PARAM_FIT_META[axis_y]
    meta_z = _PARAM_FIT_META[axis_z]
    return {
        'axis_x': axis_x,
        'axis_y': axis_y,
        'axis_z': axis_z,
        'x_mesh_key': x_mesh,
        'y_mesh_key': y_mesh,
        'z_mesh_key': z_mesh,
        'x_axis_name': x_mesh,
        'y_axis_name': y_mesh,
        'z_axis_name': z_mesh,
        'x_axis_units': meta_x['units'],
        'y_axis_units': meta_y['units'],
        'z_axis_units': meta_z['units'],
        'set_x_name': meta_x['set_name'],
        'set_y_name': meta_y['set_name'],
        'set_z_name': meta_z['set_name'],
        'set_mass_name': meta_z['set_name'],
        'x_map_title': meta_x['map_title'],
        'y_map_title': meta_y['map_title'],
        'z_map_title': meta_z['map_title'],
        'x_colorbar': meta_x['colorbar'],
        'y_colorbar': meta_y['colorbar'],
        'z_colorbar': meta_z['colorbar'],
    }


def build_native_grid_3d(
    axis_tokens: dict,
    decode: Callable[[str, int], float],
    logscale: Callable[[str], bool],
    get_intensity: Callable[[tuple, str, str, int], float],
    idef: str,
    species: str,
    transition_idx: int,
    *,
    axis_x: str,
    axis_y: str,
    axis_z: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Native (nz, ny, nx) intensity cube on the selected parameter triple."""
    x_tokens = list(axis_tokens[axis_x])
    y_tokens = list(axis_tokens[axis_y])
    z_tokens = list(axis_tokens[axis_z])

    nx, ny, nz = len(x_tokens), len(y_tokens), len(z_tokens)
    grid = np.full((nz, ny, nx), np.nan, dtype=float)

    for iz, zt in enumerate(z_tokens):
        for iy, yt in enumerate(y_tokens):
            for ix, xt in enumerate(x_tokens):
                tokens = _tokens_from_axes(
                    axis_tokens, axis_x=axis_x, axis_y=axis_y, axis_z=axis_z,
                    xt=xt, yt=yt, zt=zt,
                )
                grid[iz, iy, ix] = get_intensity(tuple(tokens), species, idef, transition_idx)

    x_phys = np.array([decode(axis_x, t) for t in x_tokens], dtype=float)
    y_phys = np.array([decode(axis_y, t) for t in y_tokens], dtype=float)
    z_phys = np.array([decode(axis_z, t) for t in z_tokens], dtype=float)

    x_phys, x_ord = _sort_axis(x_phys, logscale(axis_x))
    y_phys, y_ord = _sort_axis(y_phys, logscale(axis_y))
    z_phys, z_ord = _sort_axis(z_phys, logscale(axis_z))
    grid = grid[np.ix_(z_ord, y_ord, x_ord)]
    return grid, x_phys, y_phys, z_phys


def _tokens_from_axes(axis_tokens, *, axis_x, axis_y, axis_z, xt, yt, zt):
    """Build full 6-token tuple with middle values for non-fit axes."""
    fixed = {}
    for k in gn.PARAM_KEYS:
        toks = axis_tokens.get(k) or []
        fixed[k] = toks[len(toks) // 2] if toks else 0
    fixed[axis_x] = xt
    fixed[axis_y] = yt
    fixed[axis_z] = zt
    return [fixed[k] for k in gn.PARAM_KEYS]


def resample_grid_3d(
    grid: np.ndarray,
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    z_phys: np.ndarray,
    target_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE_3D,
    method: str = 'linear',
    logscale_x: bool = True,
    logscale_y: bool = True,
    logscale_z: bool = True,
    clip_to_bounds: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample native cube (KoSens ``process_grids_3d`` + ``create_final_grids_3d``)."""
    grid_out, final_x, final_y, final_z = gi.resample_grid_3d_kosens(
        grid, x_phys, y_phys, z_phys,
        target_shape=target_shape,
        x_logscale=logscale_x,
        y_logscale=logscale_y,
        z_logscale=logscale_z,
        interpolation_method=method,
        clip_to_bounds=clip_to_bounds,
        log_values=True,
    )
    return grid_out, final_x, final_y, final_z


def build_grids_3d_dict(
    axis_tokens: dict,
    decode: Callable[[str, int], float],
    logscale: Callable[[str], bool],
    get_intensity: Callable[[tuple, str, str, int], float],
    simline: dict,
    idef: str,
    line_names: Optional[List[str]] = None,
    target_shape: Tuple[int, int, int] = DEFAULT_TARGET_SHAPE_3D,
    method: str = 'linear',
) -> Dict[str, dict]:
    """
    Build ``my_grids_3d``-compatible dict for map fitting.

    The three cube axes are chosen from ``axis_tokens`` via
    :func:`grid_naming.resolve_fit_axes` (PDR triple when available, otherwise
    the first three varying parameters).
    """
    axis_cfg = fit_axis_config(axis_tokens)
    axis_x = axis_cfg['axis_x']
    axis_y = axis_cfg['axis_y']
    axis_z = axis_cfg['axis_z']
    x_mesh_key = axis_cfg['x_mesh_key']
    y_mesh_key = axis_cfg['y_mesh_key']
    z_mesh_key = axis_cfg['z_mesh_key']

    lookup = _line_lookup(simline, idef)
    if line_names is None:
        line_names = sorted(lookup.keys())
    else:
        missing = [n for n in line_names if n not in lookup]
        if missing:
            raise ValueError(
                f'No SIMLINE transition for line(s): {missing}. '
                f'Available: {sorted(lookup.keys())}'
            )

    first_species, first_tidx = lookup[line_names[0]]
    native, x_phys, y_phys, z_phys = build_native_grid_3d(
        axis_tokens, decode, logscale, get_intensity, idef, first_species, first_tidx,
        axis_x=axis_x, axis_y=axis_y, axis_z=axis_z,
    )
    resampled, fx, fy, fz = resample_grid_3d(
        native, x_phys, y_phys, z_phys, target_shape=target_shape, method=method,
        logscale_x=logscale(axis_x), logscale_y=logscale(axis_y), logscale_z=logscale(axis_z),
    )
    z_mesh, y_mesh, x_mesh = np.meshgrid(fz, fy, fx, indexing='ij')

    out = {}
    for name in line_names:
        species, tidx = lookup[name]
        if name == line_names[0]:
            cube = resampled
        else:
            nat, _, _, _ = build_native_grid_3d(
                axis_tokens, decode, logscale, get_intensity, idef, species, tidx,
                axis_x=axis_x, axis_y=axis_y, axis_z=axis_z,
            )
            cube, _, _, _ = resample_grid_3d(
                nat, x_phys, y_phys, z_phys, target_shape=target_shape, method=method,
                logscale_x=logscale(axis_x), logscale_y=logscale(axis_y),
                logscale_z=logscale(axis_z),
            )
        out[name] = {
            'grid': cube,
            x_mesh_key: x_mesh,
            y_mesh_key: y_mesh,
            z_mesh_key: z_mesh,
        }
    out['_fit_axis_config'] = axis_cfg
    return out


def parse_json_dict(text: Optional[str]) -> dict:
    if not text or not str(text).strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('Expected a JSON object (dict).')
    return data


def target_units_for_idef(idef: str) -> str:
    return TARGET_UNITS_JERG if idef == 'jerg' else TARGET_UNITS_JTEMP


def run_map_fit(
    observed_fits_files: dict,
    obs_errors: Optional[dict],
    grid_dicts_3d: dict,
    output_dir: str,
    target_units: str = TARGET_UNITS_JTEMP,
    chi2_pixel: Optional[Tuple[int, int]] = None,
    create_ratios: bool = True,
    create_model_ratios: bool = True,
    create_chi2_analysis: bool = True,
    create_uncertainty_maps: bool = False,
    plot_contours: bool = False,
    axis_config: Optional[dict] = None,
) -> dict:
    """Run ``fit_fits_maps_to_grids_3d`` and return the results dict."""
    os.makedirs(output_dir, exist_ok=True)
    axis_cfg = axis_config or grid_dicts_3d.get('_fit_axis_config') or {}
    if not axis_cfg:
        raise ValueError(
            'Missing fit axis configuration — rebuild 3-D grids with build_grids_3d_dict().'
        )
    line_grids = {
        k: v for k, v in grid_dicts_3d.items()
        if isinstance(v, dict) and 'grid' in v
    }
    result = _map_fit_mod.fit_fits_maps_to_grids_3d(
        observed_fits_files=observed_fits_files,
        obs_errors=obs_errors or {},
        grid_dicts_3d=line_grids,
        x_axis_name=axis_cfg['x_axis_name'],
        y_axis_name=axis_cfg['y_axis_name'],
        z_axis_name=axis_cfg['z_axis_name'],
        x_axis_units=axis_cfg['x_axis_units'],
        y_axis_units=axis_cfg['y_axis_units'],
        z_axis_units=axis_cfg['z_axis_units'],
        set_x_name=axis_cfg['set_x_name'],
        set_y_name=axis_cfg['set_y_name'],
        set_z_name=axis_cfg['set_z_name'],
        set_mass_name=axis_cfg['set_mass_name'],
        target_units=target_units,
        output_dir=output_dir,
        output_prefix='fitted',
        create_plots=False,
        fig_dir_PATH=False,
        or_PATH=False,
        plot_contours=plot_contours,
        create_ratios=create_ratios,
        create_model_ratios=create_model_ratios,
        create_chi2_analysis=create_chi2_analysis,
        chi2_analysis_pixel=chi2_pixel,
        chi2_plot_results=False,
        chi2_plot_projections=False,
        chi2_plot_volume=False,
        chi2_plot_species_slices=False,
        create_uncertainty_maps=create_uncertainty_maps,
        create_chi2_dominant_map=True,
    )
    result['fit_axis_config'] = axis_cfg
    return result


def read_fits_header(fits_path: str):
    """Load the primary HDU header from a FITS file."""
    if fits is None:
        return None
    path = os.path.expanduser(str(fits_path).strip())
    with fits.open(path) as hdul:
        return hdul[0].header.copy()


def _celestial_wcs(header):
    """Return a 2-D celestial WCS from a FITS header, or None."""
    if header is None:
        return None
    try:
        from astropy.wcs import WCS
        wcs = WCS(header)
        wcs_2d = wcs.celestial if wcs.naxis > 2 else wcs
        return wcs_2d if wcs_2d.naxis == 2 else None
    except Exception:
        return None


def _wcs_axis_labels(wcs) -> Tuple[str, str]:
    """Human-readable axis titles from WCS CTYPE."""
    if wcs is None:
        return 'Column', 'Row'
    ctype1 = str(wcs.wcs.ctype[0]).upper()
    ctype2 = str(wcs.wcs.ctype[1]).upper()
    if 'RA' in ctype1:
        x_lab = 'RA (deg)'
    elif 'GLON' in ctype1 or 'HLON' in ctype1:
        x_lab = 'Galactic lon (deg)'
    elif 'ELON' in ctype1:
        x_lab = 'Longitude (deg)'
    else:
        x_lab = str(wcs.wcs.ctype[0])
    if 'DEC' in ctype2:
        y_lab = 'Dec (deg)'
    elif 'GLAT' in ctype2 or 'HLAT' in ctype2:
        y_lab = 'Galactic lat (deg)'
    elif 'ELAT' in ctype2:
        y_lab = 'Latitude (deg)'
    else:
        y_lab = str(wcs.wcs.ctype[1])
    return x_lab, y_lab


def extract_spatial_coordinates(header, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    World coordinates at pixel centres for each map row/column (KoSens map_fit).

    Returns 1-D ``x_coords`` (length nx) and ``y_coords`` (length ny) suitable for
    Plotly heatmaps with ``origin='lower'`` sky orientation.
    """
    ny, nx = int(shape[0]), int(shape[1])
    wcs = _celestial_wcs(header)
    if wcs is None:
        return np.arange(nx, dtype=float), np.arange(ny, dtype=float)

    y_pix, x_pix = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    try:
        world = wcs.pixel_to_world_values(x_pix, y_pix)
        if isinstance(world, tuple) and len(world) >= 2:
            x_2d = np.asarray(world[0], dtype=float)
            y_2d = np.asarray(world[1], dtype=float)
        else:
            return np.arange(nx, dtype=float), np.arange(ny, dtype=float)
    except Exception:
        return np.arange(nx, dtype=float), np.arange(ny, dtype=float)

    x_1d = x_2d[ny // 2, :]
    y_1d = y_2d[:, nx // 2]
    return x_1d, y_1d


def _align_fits_map_orientation(data, x_coords, y_coords, wcs=None):
    """Align Plotly heatmap rows with FITS ``origin='lower'`` (matplotlib parity).

    Do **not** mirror along RA — astronomical maps keep RA decreasing left-to-right
    via ``xaxis.autorange='reversed'`` (see ``_plotly_fits_axis_kw``).
    """
    z = np.asarray(data, dtype=float)
    x = np.asarray(x_coords, dtype=float).copy()
    y = np.asarray(y_coords, dtype=float).copy()
    ny, nx = z.shape
    if x.shape[0] != nx:
        x = np.arange(nx, dtype=float)
    if y.shape[0] != ny:
        y = np.arange(ny, dtype=float)
    # Plotly draws row 0 at the bottom only when y is ascending.
    if ny > 1 and y[0] > y[-1]:
        y = y[::-1]
        z = z[::-1, :]
    return z, x, y


def _plotly_fits_axis_kw(wcs, x_coords, y_coords):
    """Plotly axis options so WCS maps match KoSens ``imshow(..., origin='lower'``."""
    x_kw = {}
    y_kw = {}
    if wcs is None:
        return x_kw, y_kw
    ctype1 = str(wcs.wcs.ctype[0]).upper()
    ctype2 = str(wcs.wcs.ctype[1]).upper()
    if 'RA' in ctype1 or 'HLON' in ctype1:
        x_kw['autorange'] = 'reversed'
    if 'DEC' in ctype2 or 'LAT' in ctype2:
        if len(y_coords) > 1 and float(y_coords[0]) > float(y_coords[-1]):
            y_kw['autorange'] = 'reversed'
    return x_kw, y_kw


_DISCRETE_MAP_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]


_FIT_THEMES = {
    'light': dict(
        paper_bg='#ffffff',
        plot_bg='#f8f9fa',
        grid='#e0e0e0',
        title='#333333',
        font='#333333',
    ),
    'dark': dict(
        paper_bg='#1a1a2e',
        plot_bg='#16213e',
        grid='#2a3a5c',
        title='#e8eaf0',
        font='#e0e0e0',
    ),
}


def _fit_theme_colors(theme='light'):
    return _FIT_THEMES.get(theme if theme in _FIT_THEMES else 'light', _FIT_THEMES['light'])


def _first_present(mapping: dict, *keys):
    """Return the first non-None value for ``keys`` (safe for numpy arrays)."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def fig_fits_map(
    data: np.ndarray,
    title: str,
    header=None,
    zscale: str = 'log',
    colorscale: str = 'Viridis',
    colorbar_title: str = '',
    theme: str = 'light',
) -> Any:
    """Plotly heatmap for a 2-D parameter or diagnostic map."""
    import plotly.graph_objects as go

    arr = np.asarray(data, dtype=float)
    ny, nx = arr.shape
    wcs = _celestial_wcs(header)
    x_coords, y_coords = extract_spatial_coordinates(header, (ny, nx))
    use_wcs = wcs is not None
    x_label, y_label = _wcs_axis_labels(wcs) if use_wcs else ('Column', 'Row')

    zplot, x_coords, y_coords = _align_fits_map_orientation(arr, x_coords, y_coords, wcs=wcs)
    x_axis_kw, y_axis_kw = _plotly_fits_axis_kw(wcs, x_coords, y_coords)
    if zscale == 'log':
        zplot = np.where(zplot > 0, np.log10(zplot), np.nan)

    if use_wcs:
        hover = 'RA=%{x:.5f}°<br>Dec=%{y:.5f}°<br>value=%{z:.4g}<extra></extra>'
    else:
        hover = 'col=%{x}<br>row=%{y}<br>value=%{z:.4g}<extra></extra>'

    dx = abs(float(x_coords[-1] - x_coords[0])) if nx > 1 else 1.0
    dy = abs(float(y_coords[-1] - y_coords[0])) if ny > 1 else 1.0
    if use_wcs and dx > 0 and dy > 0:
        dec_mean = float(np.nanmean(y_coords))
        cos_dec = max(abs(np.cos(np.deg2rad(dec_mean))), 0.05)
        sky_ratio = dy / (dx * cos_dec)
    elif dx > 0 and dy > 0:
        sky_ratio = dy / dx
    else:
        sky_ratio = ny / max(nx, 1)

    fig = go.Figure(go.Heatmap(
        x=x_coords, y=y_coords, z=zplot,
        colorscale=colorscale,
        colorbar=dict(title=dict(text=colorbar_title or title)),
        hovertemplate=hover,
    ))
    plot_height = int(max(520, min(780, 420 * sky_ratio)))
    t = _fit_theme_colors(theme)
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor='left',
                   font=dict(size=13, color=t['title'])),
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        margin=dict(l=60, r=20, t=48, b=54),
        autosize=True,
        height=plot_height,
        font=dict(color=t['font']),
        xaxis=dict(
            title=dict(text=x_label, font=dict(color=t['font'])),
            showgrid=True, gridcolor=t['grid'],
            tickfont=dict(color=t['font']),
            **x_axis_kw,
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color=t['font'])),
            showgrid=True, gridcolor=t['grid'],
            tickfont=dict(color=t['font']),
            scaleanchor='x', scaleratio=sky_ratio,
            **y_axis_kw,
        ),
    )
    return fig


def fig_chi2_dominant_map(
    index_map,
    species_labels,
    header=None,
    theme: str = 'light',
) -> Any:
    """Categorical map of the dominant chi^2 contributor per pixel."""
    import plotly.graph_objects as go

    labels = list(species_labels or [])
    if not labels:
        return None

    arr = np.asarray(index_map, dtype=float)
    ny, nx = arr.shape
    wcs = _celestial_wcs(header)
    x_coords, y_coords = extract_spatial_coordinates(header, (ny, nx))
    zplot, x_coords, y_coords = _align_fits_map_orientation(arr, x_coords, y_coords, wcs=wcs)
    x_axis_kw, y_axis_kw = _plotly_fits_axis_kw(wcs, x_coords, y_coords)
    use_wcs = wcs is not None
    x_label, y_label = _wcs_axis_labels(wcs) if use_wcs else ('Column', 'Row')

    n = len(labels)
    palette = (_DISCRETE_MAP_COLORS * ((n // len(_DISCRETE_MAP_COLORS)) + 1))[:n]
    if n == 1:
        colorscale = [[0.0, palette[0]], [1.0, palette[0]]]
    else:
        colorscale = []
        for i, color in enumerate(palette):
            t0 = i / n
            t1 = (i + 1) / n
            colorscale.append([t0, color])
            colorscale.append([t1, color])

    label_lookup = np.array(labels, dtype=object)
    custom = np.full((ny, nx), '', dtype=object)
    finite = np.isfinite(zplot)
    idx = zplot[finite].astype(int)
    valid = (idx >= 0) & (idx < n)
    custom_flat = custom[finite]
    custom_flat[valid] = label_lookup[idx[valid]]
    custom[finite] = custom_flat

    if use_wcs:
        hover = 'RA=%{x:.5f}°<br>Dec=%{y:.5f}°<br>%{customdata}<extra></extra>'
    else:
        hover = 'col=%{x}<br>row=%{y}<br>%{customdata}<extra></extra>'

    dx = abs(float(x_coords[-1] - x_coords[0])) if nx > 1 else 1.0
    dy = abs(float(y_coords[-1] - y_coords[0])) if ny > 1 else 1.0
    if use_wcs and dx > 0 and dy > 0:
        dec_mean = float(np.nanmean(y_coords))
        cos_dec = max(abs(np.cos(np.deg2rad(dec_mean))), 0.05)
        sky_ratio = dy / (dx * cos_dec)
    elif dx > 0 and dy > 0:
        sky_ratio = dy / dx
    else:
        sky_ratio = ny / max(nx, 1)

    fig = go.Figure(go.Heatmap(
        x=x_coords, y=y_coords, z=zplot,
        zmin=-0.5, zmax=n - 0.5,
        customdata=custom,
        colorscale=colorscale,
        showscale=False,
        hovertemplate=hover,
    ))
    present = sorted({int(v) for v in zplot[np.isfinite(zplot)] if 0 <= int(v) < n})
    for idx in present:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color=palette[idx]),
            name=labels[idx], showlegend=True,
        ))

    plot_height = int(max(520, min(780, 420 * sky_ratio)))
    t = _fit_theme_colors(theme)
    fig.update_layout(
        title=dict(
            text='Dominant χ² contributor at best-fit grid point',
            x=0.02, xanchor='left', font=dict(size=13, color=t['title']),
        ),
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        margin=dict(l=60, r=160, t=48, b=54),
        autosize=True,
        height=plot_height,
        font=dict(color=t['font']),
        legend=dict(title='Dominant line', font=dict(color=t['font'])),
        xaxis=dict(
            title=dict(text=x_label, font=dict(color=t['font'])),
            showgrid=True, gridcolor=t['grid'],
            tickfont=dict(color=t['font']),
            **x_axis_kw,
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color=t['font'])),
            showgrid=True, gridcolor=t['grid'],
            tickfont=dict(color=t['font']),
            scaleanchor='x', scaleratio=sky_ratio,
            **y_axis_kw,
        ),
    )
    return fig


_CHI2_DELTA_2D = getattr(_map_fit_mod, '_CHI2_DELTA_2D', (2.30, 6.17, 11.62))
_CHI2_CONF_LABELS = getattr(_map_fit_mod, '_CHI2_CONF_LABELS', ('68%', '95%', '99.7%'))


def run_chi2_at_pixel(result: dict, pixel: Tuple[int, int]) -> Optional[dict]:
    """Re-run ``chi2_analysis_3d`` at ``pixel`` using stored fit context."""
    ctx = (result or {}).get('chi2_reanalysis')
    if not ctx:
        return None
    from map_fit_extras import chi2_analysis_3d

    data_names = result.get('data_names') or []
    grid_arrays = {name: True for name in data_names}
    corner_obs, corner_err, subtitle = _map_fit_mod._pixel_observations_for_chi2_analysis(
        ctx['all_observed_data'],
        ctx['all_observed_errors'],
        ctx['ref_mask'],
        result.get('data_names') or [],
        grid_arrays,
        pixel=pixel,
    )
    if not corner_obs:
        return None
    chi2_results = chi2_analysis_3d(
        ctx['grid_dicts_3d'],
        observed_values=corner_obs,
        observed_errors=corner_err,
        set_x_name=ctx.get('set_x_name', 'n'),
        set_y_name=ctx.get('set_y_name', 'FUV'),
        set_mass_name=ctx.get('set_mass_name', 'M'),
        set_z_name=ctx.get('set_z_name'),
        plot_results=False,
        plot_projections=False,
    )
    if not chi2_results:
        return None
    out = dict(result)
    out['chi2_analysis'] = chi2_results
    out['chi2_analysis_pixel'] = pixel
    out['chi2_observed_values'] = corner_obs
    out['chi2_observed_errors'] = corner_err
    out['chi2_subtitle'] = subtitle
    return out


def fig_chi2_pixel_map(
    chi2_map: np.ndarray,
    *,
    chi2_pixel: Optional[Tuple[int, int]] = None,
    use_reduced: bool = False,
    theme: str = 'light',
    colorscale: str = 'Viridis',
) -> Any:
    """Full χ² map in pixel row/column coordinates (KoSens-style picker view)."""
    import plotly.graph_objects as go

    arr = np.asarray(chi2_map, dtype=float)
    if arr.ndim != 2 or not np.any(np.isfinite(arr)):
        return None
    ny, nx = arr.shape
    x = np.arange(nx)
    y = np.arange(ny)
    title = 'Reduced χ² map' if use_reduced else 'χ² map'
    cbar = 'χ²<sub>ν</sub>' if use_reduced else 'χ²'

    fig = go.Figure(go.Heatmap(
        x=x, y=y, z=arr,
        colorscale=colorscale,
        colorbar=dict(title=dict(text=cbar)),
        hovertemplate='row=%{y}<br>col=%{x}<br>χ²=%{z:.4g}<extra></extra>',
    ))
    if chi2_pixel is not None:
        i_pix, j_pix = chi2_pixel
        fig.add_trace(go.Scatter(
            x=[j_pix], y=[i_pix],
            mode='markers',
            marker=dict(symbol='x', size=14, color='white', line=dict(width=2, color='black')),
            name=f'Analysis pixel ({i_pix}, {j_pix})',
            hovertemplate=f'Analysis pixel<br>row={i_pix}<br>col={j_pix}<extra></extra>',
        ))
    t = _fit_theme_colors(theme)
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor='left', font=dict(size=13, color=t['title'])),
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        margin=dict(l=60, r=20, t=48, b=54),
        autosize=True,
        height=int(max(520, min(780, 420 * ny / max(nx, 1)))),
        font=dict(color=t['font']),
        xaxis=dict(title='Pixel column', showgrid=True, gridcolor=t['grid']),
        yaxis=dict(title='Pixel row', showgrid=True, gridcolor=t['grid'], scaleanchor='x'),
        showlegend=chi2_pixel is not None,
        legend=dict(font=dict(color=t['font'])),
    )
    return fig


def fig_chi2_corner(
    chi2_results: dict,
    *,
    subtitle: str = '',
    set_x_name: str = 'n',
    set_y_name: str = 'FUV',
    theme: str = 'light',
    colorscale: str = 'Viridis',
    reverse_colorscale: bool = True,
) -> Any:
    """Plotly corner plot (Δχ² surface + marginalized profiles) from 3D chi² analysis."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    chi2_grid = np.asarray(chi2_results['chi2_grid'], dtype=float)
    chi2_min = float(chi2_results['chi2_min'])
    x_values = np.asarray(chi2_results['x_values'], dtype=float)
    y_values = np.asarray(chi2_results['y_values'], dtype=float)
    if chi2_grid.ndim == 3:
        chi2_2d = np.nanmin(chi2_grid, axis=0)
    else:
        chi2_2d = chi2_grid
    if not np.any(np.isfinite(chi2_2d)) or np.any(x_values <= 0) or np.any(y_values <= 0):
        return None

    delta = chi2_2d - chi2_min
    finite = delta[np.isfinite(delta)]
    vmax = 25.0
    if finite.size:
        vmax = min(25.0, float(np.nanpercentile(finite, 95)))
    if vmax <= 0:
        vmax = 25.0

    log_x = np.log10(x_values)
    log_y = np.log10(y_values)
    log_x_mesh, log_y_mesh = np.meshgrid(log_x, log_y, indexing='xy')

    chi2_vs_x = np.nanmin(chi2_2d, axis=0) - chi2_min
    chi2_vs_y = np.nanmin(chi2_2d, axis=1) - chi2_min
    best_x = float(chi2_results['best_x'])
    best_y = float(chi2_results['best_y'])
    profile_max = min(15.0, max(float(np.nanmax(chi2_vs_x)), float(np.nanmax(chi2_vs_y))) + 2.0)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.78, 0.22],
        row_heights=[0.22, 0.78],
        horizontal_spacing=0.04,
        vertical_spacing=0.04,
        specs=[[{'type': 'xy'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}]],
    )
    fig.add_trace(go.Contour(
        x=log_x, y=log_y, z=delta,
        colorscale=colorscale,
        reversescale=reverse_colorscale,
        zmin=0, zmax=vmax,
        contours=dict(start=0, end=vmax, size=vmax / 50, coloring='fill'),
        colorbar=dict(title='Δχ²', len=0.55, y=0.25),
        hovertemplate='log₁₀ x=%{x:.2f}<br>log₁₀ y=%{y:.2f}<br>Δχ²=%{z:.3f}<extra></extra>',
        showscale=True,
    ), row=2, col=1)

    for delta_level, label, color, dash in zip(
        _CHI2_DELTA_2D, _CHI2_CONF_LABELS, ('white', 'cyan', 'yellow'), ('solid', 'dash', 'dot'),
    ):
        fig.add_trace(go.Contour(
            x=log_x, y=log_y, z=delta,
            contours=dict(start=delta_level, end=delta_level, size=1, coloring='lines'),
            line=dict(color=color, width=2, dash=dash),
            showscale=False,
            name=f'{label} (Δχ²={delta_level:.1f})',
            hoverinfo='skip',
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[np.log10(best_x)], y=[np.log10(best_y)],
        mode='markers',
        marker=dict(symbol='star', size=16, color='red', line=dict(width=1, color='white')),
        name='Best fit',
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=log_x, y=chi2_vs_x, mode='lines',
        line=dict(color='#2E86AB', width=2),
        showlegend=False,
        hovertemplate='Δχ²=%{y:.3f}<extra></extra>',
    ), row=1, col=1)
    fig.add_vline(x=np.log10(best_x), line=dict(color='red', dash='dash'), row=1, col=1)
    fig.add_hline(y=1.0, line=dict(color='gray', dash='dot'), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=chi2_vs_y, y=log_y, mode='lines',
        line=dict(color='#C73E1D', width=2),
        showlegend=False,
        hovertemplate='Δχ²=%{x:.3f}<extra></extra>',
    ), row=2, col=2)
    fig.add_hline(y=np.log10(best_y), line=dict(color='red', dash='dash'), row=2, col=2)
    fig.add_vline(x=1.0, line=dict(color='gray', dash='dot'), row=2, col=2)

    dof = chi2_results.get('dof', 1)
    chi2_red = chi2_results.get('chi2_reduced', chi2_min / max(dof, 1))
    title = (
        f'χ² corner: {set_x_name} vs {set_y_name}  '
        f'(χ²<sub>min</sub>={chi2_min:.2f}, χ²<sub>ν</sub>={chi2_red:.2f})'
    )
    if subtitle:
        title += f' — {subtitle}'

    t = _fit_theme_colors(theme)
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor='left', font=dict(size=13, color=t['title'])),
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        margin=dict(l=60, r=20, t=56, b=54),
        autosize=True,
        height=620,
        font=dict(color=t['font']),
        showlegend=True,
        legend=dict(x=0.02, y=0.02, bgcolor='rgba(255,255,255,0.7)', font=dict(size=10)),
    )
    fig.update_xaxes(title_text=f'log₁₀({set_x_name})', row=2, col=1, gridcolor=t['grid'])
    fig.update_yaxes(title_text=f'log₁₀({set_y_name})', row=2, col=1, gridcolor=t['grid'])
    fig.update_yaxes(title_text='Δχ²', row=1, col=1, range=[-0.5, profile_max], gridcolor=t['grid'])
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='Δχ²', row=2, col=2, range=[-0.5, profile_max], gridcolor=t['grid'])
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=2)
    return fig


def fig_chi2_vs_third_axis(chi2_results: dict, *, theme: str = 'light') -> Any:
    """Marginalized χ² profile along the third grid axis (CRIR / mass)."""
    import plotly.graph_objects as go

    chi2_grid = np.asarray(chi2_results['chi2_grid'], dtype=float)
    z_raw = _first_present(chi2_results, 'mass_values', 'z_values')
    if z_raw is None:
        return None
    z_values = np.asarray(z_raw, dtype=float)
    z_label = chi2_results.get('z_axis_name') or 'ζ'
    if chi2_grid.ndim != 3 or z_values.size == 0:
        return None
    chi2_by_z = np.nanmin(np.nanmin(chi2_grid, axis=2), axis=1)
    chi2_min = float(chi2_results['chi2_min'])
    delta = chi2_by_z - chi2_min
    best_z_raw = _first_present(chi2_results, 'best_mass', 'best_z')
    if best_z_raw is None:
        return None
    best_z = float(best_z_raw)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=z_values, y=delta, mode='lines+markers',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=6),
        hovertemplate=f'{z_label}=%{{x:.3e}}<br>Δχ²=%{{y:.3f}}<extra></extra>',
    ))
    fig.add_vline(x=best_z, line=dict(color='red', dash='dash'))
    fig.add_hline(y=1.0, line=dict(color='gray', dash='dot'))
    t = _fit_theme_colors(theme)
    fig.update_layout(
        title=dict(
            text=f'Δχ² vs {z_label} (marginalized over other axes)',
            x=0.02, xanchor='left', font=dict(size=13, color=t['title']),
        ),
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        margin=dict(l=60, r=20, t=48, b=54),
        autosize=True,
        height=420,
        font=dict(color=t['font']),
        xaxis=dict(title=z_label, type='log', gridcolor=t['grid']),
        yaxis=dict(title='Δχ²', gridcolor=t['grid']),
    )
    return fig


def fig_chi2_obs_model_bars(
    chi2_results: dict,
    observed_values: dict,
    observed_errors: dict,
    *,
    theme: str = 'light',
) -> Any:
    """Grouped bar chart of observed vs model intensities at the best-fit grid point."""
    import plotly.graph_objects as go

    species = chi2_results.get('valid_species') or []
    model = chi2_results.get('model_values_at_best') or {}
    if not species:
        return None
    obs_y, mod_y, err_y, labels = [], [], [], []
    for name in species:
        if name not in observed_values or name not in model:
            continue
        obs_val = float(observed_values[name])
        if obs_val <= 0 or float(model[name]) <= 0:
            continue
        labels.append(name)
        obs_y.append(obs_val)
        mod_y.append(float(model[name]))
        err_y.append(float(observed_errors.get(name, np.nan)))

    if not labels:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Observed', x=labels, y=obs_y, marker_color='#2E86AB',
                         error_y=dict(type='data', array=err_y, visible=True)))
    fig.add_trace(go.Bar(name='Model (best fit)', x=labels, y=mod_y, marker_color='#C73E1D'))
    t = _fit_theme_colors(theme)
    fig.update_layout(
        title=dict(
            text='Observed vs model at best-fit point',
            x=0.02, xanchor='left', font=dict(size=13, color=t['title']),
        ),
        barmode='group',
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        margin=dict(l=60, r=20, t=48, b=120),
        autosize=True,
        height=460,
        font=dict(color=t['font']),
        xaxis=dict(tickangle=-35, gridcolor=t['grid']),
        yaxis=dict(title='Intensity', type='log', gridcolor=t['grid']),
        legend=dict(font=dict(color=t['font'])),
    )
    return fig


def fig_fitted_params_kde(result: dict, *, theme: str = 'light') -> Any:
    """1×3 KDE panels for the fitted x/y/z parameter maps (KoSens-style)."""
    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde
    import plotly.graph_objects as go

    if not result:
        return None
    axis_cfg = result.get('fit_axis_config') or {}
    panels = (
        ('x_param_map', 'set_x_name', axis_cfg.get('x_map_title', 'x')),
        ('y_param_map', 'set_y_name', axis_cfg.get('y_map_title', 'y')),
        ('z_param_map', 'set_z_name', axis_cfg.get('z_map_title', 'z')),
    )
    t = _fit_theme_colors(theme)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[title for _, _, title in panels],
        horizontal_spacing=0.08,
    )
    n_traces = 0
    for col, (map_key, name_key, title) in enumerate(panels, start=1):
        param_map = result.get(map_key)
        if param_map is None:
            continue
        vals = np.asarray(param_map, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]
        if vals.size < 2:
            continue
        plot_vals = np.log10(vals)
        label = axis_cfg.get(name_key, title)
        inner = label.strip('$') if label.startswith('$') and label.endswith('$') else label
        x_label = f'log₁₀({inner})'
        kde = gaussian_kde(plot_vals)
        x_grid = np.linspace(float(plot_vals.min()), float(plot_vals.max()), 256)
        density = kde(x_grid)
        fig.add_trace(go.Scatter(
            x=x_grid, y=density,
            mode='lines',
            line=dict(color='#2E86AB', width=2.2),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.35)',
            name=title,
            showlegend=False,
            hovertemplate=f'{x_label}=%{{x:.3f}}<br>density=%{{y:.3g}}<extra></extra>',
        ), row=1, col=col)
        fig.add_vline(
            x=float(np.median(plot_vals)),
            line=dict(color='black', dash='dash', width=1.5),
            row=1, col=col,
        )
        fig.add_vline(
            x=float(np.mean(plot_vals)),
            line=dict(color='gray', dash='dot', width=1.5),
            row=1, col=col,
        )
        fig.update_xaxes(title_text=x_label, row=1, col=col, gridcolor=t['grid'])
        fig.update_yaxes(title_text='Density' if col == 1 else '', row=1, col=col, gridcolor=t['grid'])
        n_traces += 1
    if n_traces == 0:
        return None
    fig.update_layout(
        title=dict(
            text='KDE of fitted parameters (valid map pixels)',
            x=0.02, xanchor='left', font=dict(size=13, color=t['title']),
        ),
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        margin=dict(l=60, r=20, t=56, b=54),
        autosize=True,
        height=380,
        font=dict(color=t['font']),
    )
    return fig


def _format_axis_value(val: float) -> str:
    if not np.isfinite(val):
        return 'nan'
    if val < 0.1 or val >= 1000:
        return f'{val:.2e}'
    return f'{val:.2f}'


def _extract_contour_paths(x, y, z, level):
    """Return vertex paths for a single contour level (headless matplotlib)."""
    from matplotlib.figure import Figure

    z_arr = np.asarray(z, dtype=float)
    lev = float(level)
    zmin, zmax = np.nanmin(z_arr), np.nanmax(z_arr)
    if not np.isfinite(lev) or lev < zmin or lev > zmax:
        return []
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    fig_tmp = Figure()
    ax = fig_tmp.add_subplot(111)
    cs = ax.contour(x_arr, y_arr, z_arr, levels=[lev])
    paths = []
    if cs.levels.size and cs.allsegs:
        for seg in cs.allsegs[0]:
            seg = np.asarray(seg, dtype=float)
            if seg.shape[0] >= 2:
                paths.append(seg)
    return paths


def _add_obs_contour(fig, *, x, y, z, obs_val, color, row, col):
    """Add model=observed contour as coloured line segments (one colour per species)."""
    import plotly.graph_objects as go

    paths = _extract_contour_paths(x, y, z, obs_val)
    for path in paths:
        fig.add_trace(go.Scatter(
            x=path[:, 0], y=path[:, 1],
            mode='lines',
            line=dict(color=color, width=2.5),
            showlegend=False,
            hoverinfo='skip',
        ), row=row, col=col)
    return len(paths) > 0


def fig_species_contour_slices(
    chi2_results: dict,
    grid_dicts_3d: dict,
    observed_values: dict,
    *,
    axis_cfg: dict,
    theme: str = 'light',
    max_species: int = 8,
) -> Any:
    """
    Species contours (model = observed) at best-fit slices — KoSens spaghetti plot.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    if not chi2_results or not grid_dicts_3d or not observed_values:
        return None
    species = [
        sp for sp in (chi2_results.get('valid_species') or [])
        if sp in observed_values and sp in grid_dicts_3d
    ]
    if not species:
        return None
    if len(species) > max_species:
        species = species[:max_species]

    x_values = np.asarray(chi2_results['x_values'], dtype=float)
    y_values = np.asarray(chi2_results['y_values'], dtype=float)
    z_raw = _first_present(chi2_results, 'mass_values', 'z_values')
    if z_raw is None:
        return None
    z_values = np.asarray(z_raw, dtype=float)
    if min(x_values.size, y_values.size, z_values.size) == 0:
        return None

    bx = int(chi2_results['best_x_idx'])
    by = int(chi2_results['best_y_idx'])
    bz = int(_first_present(chi2_results, 'best_mass_idx', 'best_z_idx') or 0)
    best_x = float(chi2_results['best_x'])
    best_y = float(chi2_results['best_y'])
    best_z = float(_first_present(chi2_results, 'best_mass', 'best_z'))

    set_x = axis_cfg.get('set_x_name', 'n')
    set_y = axis_cfg.get('set_y_name', 'FUV')
    set_z = axis_cfg.get('set_z_name', 'ζ')
    z_title = _format_axis_value(best_z)

    log_x = np.log10(x_values)
    log_y = np.log10(y_values)
    log_z = np.log10(z_values)
    colors = _DISCRETE_MAP_COLORS[:len(species)]
    t = _fit_theme_colors(theme)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f'Slice at {set_z} = {z_title}',
            f'Slice at {set_x} = {best_x:.2e}',
            f'Slice at {set_y} = {best_y:.2e}',
        ],
        horizontal_spacing=0.08,
    )
    n_drawn = 0
    for i, sp in enumerate(species):
        cube = np.asarray(grid_dicts_3d[sp]['grid'], dtype=float)
        obs = observed_values[sp]
        if _add_obs_contour(
            fig, x=log_x, y=log_y, z=cube[bz, :, :], obs_val=obs,
            color=colors[i], row=1, col=1,
        ):
            n_drawn += 1
        _add_obs_contour(
            fig, x=log_z, y=log_y, z=cube[:, :, bx].T, obs_val=obs,
            color=colors[i], row=1, col=2,
        )
        _add_obs_contour(
            fig, x=log_z, y=log_x, z=cube[:, by, :].T, obs_val=obs,
            color=colors[i], row=1, col=3,
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color=colors[i], width=2.5),
            legendgroup=sp,
            name=f'{sp}  (obs={obs:.2e})',
            showlegend=True,
        ))

    if n_drawn == 0:
        return None

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='star', size=12, color='black', line=dict(width=1, color='white')),
        name='Best fit', showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[np.log10(best_x)], y=[np.log10(best_y)],
        mode='markers',
        marker=dict(symbol='star', size=14, color='black', line=dict(width=1, color='white')),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[np.log10(best_z)], y=[np.log10(best_y)],
        mode='markers',
        marker=dict(symbol='star', size=14, color='black', line=dict(width=1, color='white')),
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[np.log10(best_z)], y=[np.log10(best_x)],
        mode='markers',
        marker=dict(symbol='star', size=14, color='black', line=dict(width=1, color='white')),
        showlegend=False,
    ), row=1, col=3)

    fig.update_xaxes(title_text=f'log₁₀({set_x})', row=1, col=1, gridcolor=t['grid'])
    fig.update_yaxes(title_text=f'log₁₀({set_y})', row=1, col=1, gridcolor=t['grid'])
    fig.update_xaxes(title_text=f'log₁₀({set_z})', row=1, col=2, gridcolor=t['grid'])
    fig.update_yaxes(title_text=f'log₁₀({set_y})', row=1, col=2, gridcolor=t['grid'])
    fig.update_xaxes(title_text=f'log₁₀({set_z})', row=1, col=3, gridcolor=t['grid'])
    fig.update_yaxes(title_text=f'log₁₀({set_x})', row=1, col=3, gridcolor=t['grid'])

    fig.update_layout(
        title=dict(
            text='Species contours (model = observed) at best-fit slices',
            x=0.02, xanchor='left', font=dict(size=13, color=t['title']),
        ),
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        margin=dict(l=60, r=220, t=56, b=54),
        autosize=True,
        height=480,
        font=dict(color=t['font']),
        legend=dict(
            orientation='v',
            yanchor='top', y=1.0,
            xanchor='left', x=1.01,
            bgcolor='rgba(255,255,255,0.92)' if theme == 'light' else 'rgba(22,33,62,0.92)',
            bordercolor=t['grid'], borderwidth=1,
            font=dict(size=10, color=t['font']),
            tracegroupgap=4,
        ),
    )
    return fig


def chi2_analysis_figures(
    result: dict,
    *,
    theme: str = 'light',
    colorscale: str = 'Viridis',
    chi2_pixel: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Build Plotly figures for per-pixel χ² maps and detailed χ² analysis."""
    if not result:
        return {}
    figs: Dict[str, Any] = {}
    pixel = chi2_pixel or result.get('chi2_analysis_pixel')

    chi2_min_map = result.get('chi2_min_map')
    if chi2_min_map is not None and np.any(np.isfinite(chi2_min_map)):
        fig = fig_chi2_pixel_map(
            chi2_min_map, chi2_pixel=pixel, use_reduced=False,
            theme=theme, colorscale=colorscale,
        )
        if fig is not None:
            figs['chi2_pixel_map'] = fig

    chi2 = result.get('chi2_analysis')
    if not chi2:
        return figs

    ctx = (result.get('chi2_reanalysis') or {})
    axis_cfg = result.get('fit_axis_config') or {}
    subtitle = result.get('chi2_subtitle') or ''
    obs_vals = result.get('chi2_observed_values') or {}
    obs_errs = result.get('chi2_observed_errors') or {}

    corner = fig_chi2_corner(
        chi2,
        subtitle=subtitle,
        set_x_name=axis_cfg.get('set_x_name') or ctx.get('set_x_name', 'n'),
        set_y_name=axis_cfg.get('set_y_name') or ctx.get('set_y_name', 'FUV'),
        theme=theme,
        colorscale=colorscale,
        reverse_colorscale=True,
    )
    if corner is not None:
        figs['chi2_corner'] = corner

    zprof = fig_chi2_vs_third_axis(chi2, theme=theme)
    if zprof is not None:
        figs['chi2_z_profile'] = zprof

    bars = fig_chi2_obs_model_bars(chi2, obs_vals, obs_errs, theme=theme)
    if bars is not None:
        figs['chi2_bars'] = bars

    ctx_grids = ctx.get('grid_dicts_3d')
    if ctx_grids and obs_vals:
        spaghetti = fig_species_contour_slices(
            chi2, ctx_grids, obs_vals, axis_cfg=axis_cfg, theme=theme,
        )
        if spaghetti is not None:
            figs['species_contours'] = spaghetti
    return figs


def fit_result_figures(result: dict, fits_header=None, *,
                       theme='light', colorscale='Viridis') -> Dict[str, Any]:
    """Build Plotly figures for fitted x/y/z parameter maps."""
    if not result:
        return {}
    header = fits_header if fits_header is not None else result.get('fits_header')
    axis_cfg = result.get('fit_axis_config') or {}
    figs = {
        'x': fig_fits_map(
            result['x_param_map'],
            axis_cfg.get('x_map_title', 'Best-fit x parameter'),
            header=header, zscale='log',
            colorbar_title=axis_cfg.get('x_colorbar', ''),
            theme=theme, colorscale=colorscale,
        ),
        'y': fig_fits_map(
            result['y_param_map'],
            axis_cfg.get('y_map_title', 'Best-fit y parameter'),
            header=header, zscale='log',
            colorbar_title=axis_cfg.get('y_colorbar', ''),
            theme=theme, colorscale=colorscale,
        ),
        'z': fig_fits_map(
            result['z_param_map'],
            axis_cfg.get('z_map_title', 'Best-fit z parameter'),
            header=header, zscale='log',
            colorbar_title=axis_cfg.get('z_colorbar', ''),
            theme=theme, colorscale=colorscale,
        ),
    }
    chi2_red = result.get('chi2_reduced_map')
    if chi2_red is not None and np.any(np.isfinite(chi2_red)):
        figs['chi2'] = fig_fits_map(
            chi2_red, 'Reduced χ² per pixel',
            header=header, zscale='linear', colorscale=colorscale,
            colorbar_title='χ²<sub>ν</sub>',
            theme=theme,
        )
    dom_map = result.get('chi2_dominant_index_map')
    dom_species = result.get('chi2_dominant_species')
    if dom_map is not None and dom_species and np.any(np.isfinite(dom_map)):
        dom_fig = fig_chi2_dominant_map(
            dom_map, dom_species, header=header, theme=theme,
        )
        if dom_fig is not None:
            figs['chi2_dominant'] = dom_fig
    kde_fig = fig_fitted_params_kde(result, theme=theme)
    if kde_fig is not None:
        figs['kde'] = kde_fig
    figs.update(chi2_analysis_figures(result, theme=theme, colorscale=colorscale))
    return figs
