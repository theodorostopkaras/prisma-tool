"""
Build 3-D SIMLINE intensity cubes and fit observational FITS maps.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import grid_interp as gi
import map_fit as _map_fit_mod
import smli_labels

try:
    from astropy.io import fits
except ImportError:  # pragma: no cover
    fits = None

AXIS_X = 'density'
AXIS_Y = 'fuv'
AXIS_Z = 'crir'
X_MESH_KEY = 'densities'
Y_MESH_KEY = 'fuv_values'
Z_MESH_KEY = 'crir_values'

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


def build_native_grid_3d(
    axis_tokens: dict,
    decode: Callable[[str, int], float],
    logscale: Callable[[str], bool],
    get_intensity: Callable[[tuple, str, str, int], float],
    idef: str,
    species: str,
    transition_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Native (nz, ny, nx) intensity cube on density x FUV x CRIR."""
    x_tokens = list(axis_tokens[AXIS_X])
    y_tokens = list(axis_tokens[AXIS_Y])
    z_tokens = list(axis_tokens[AXIS_Z])

    nx, ny, nz = len(x_tokens), len(y_tokens), len(z_tokens)
    grid = np.full((nz, ny, nx), np.nan, dtype=float)

    for iz, zt in enumerate(z_tokens):
        for iy, yt in enumerate(y_tokens):
            for ix, xt in enumerate(x_tokens):
                tokens = [None] * 6
                # caller passes full token tuple builder via indices
                tokens = _tokens_from_axes(xt, yt, zt, axis_tokens)
                grid[iz, iy, ix] = get_intensity(tuple(tokens), species, idef, transition_idx)

    x_phys = np.array([decode(AXIS_X, t) for t in x_tokens], dtype=float)
    y_phys = np.array([decode(AXIS_Y, t) for t in y_tokens], dtype=float)
    z_phys = np.array([decode(AXIS_Z, t) for t in z_tokens], dtype=float)

    x_phys, x_ord = _sort_axis(x_phys, logscale(AXIS_X))
    y_phys, y_ord = _sort_axis(y_phys, logscale(AXIS_Y))
    z_phys, z_ord = _sort_axis(z_phys, logscale(AXIS_Z))
    grid = grid[np.ix_(z_ord, y_ord, x_ord)]
    return grid, x_phys, y_phys, z_phys


def _tokens_from_axes(xt, yt, zt, axis_tokens):
    """Build full 6-token tuple with middle values for fixed axes."""
    # Order matches PARAM_DEFS: density, mass, fuv, metal, crir, atten
    keys = ['density', 'mass', 'fuv', 'metal', 'crir', 'atten']
    fixed = {}
    for k in keys:
        toks = axis_tokens.get(k) or []
        fixed[k] = toks[len(toks) // 2] if toks else 0
    fixed['density'] = xt
    fixed['fuv'] = yt
    fixed['crir'] = zt
    return [fixed[k] for k in keys]


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

    Axes: x = density, y = FUV, z = CRIR (matching ``3d_grids.ipynb`` defaults).
    """
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

    # Build one native cube to get shared axes
    first_species, first_tidx = lookup[line_names[0]]
    native, x_phys, y_phys, z_phys = build_native_grid_3d(
        axis_tokens, decode, logscale, get_intensity, idef, first_species, first_tidx,
    )
    resampled, fx, fy, fz = resample_grid_3d(
        native, x_phys, y_phys, z_phys, target_shape=target_shape, method=method,
        logscale_x=logscale(AXIS_X), logscale_y=logscale(AXIS_Y), logscale_z=logscale(AXIS_Z),
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
            )
            cube, _, _, _ = resample_grid_3d(
                nat, x_phys, y_phys, z_phys, target_shape=target_shape, method=method,
                logscale_x=logscale(AXIS_X), logscale_y=logscale(AXIS_Y),
                logscale_z=logscale(AXIS_Z),
            )
        out[name] = {
            'grid': cube,
            X_MESH_KEY: x_mesh,
            Y_MESH_KEY: y_mesh,
            Z_MESH_KEY: z_mesh,
        }
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
) -> dict:
    """Run ``fit_fits_maps_to_grids_3d`` and return the results dict."""
    os.makedirs(output_dir, exist_ok=True)
    return _map_fit_mod.fit_fits_maps_to_grids_3d(
        observed_fits_files=observed_fits_files,
        obs_errors=obs_errors or {},
        grid_dicts_3d=grid_dicts_3d,
        x_axis_name=X_MESH_KEY,
        y_axis_name=Y_MESH_KEY,
        z_axis_name=Z_MESH_KEY,
        x_axis_units=r'cm$^{-3}$',
        y_axis_units='Draine',
        z_axis_units=r'$s^{-1}$',
        set_x_name='n',
        set_y_name='FUV',
        set_z_name=r'$\zeta_{\mathrm{H}}$',
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
    )


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

    zplot = arr.copy()
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
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color=t['font'])),
            showgrid=True, gridcolor=t['grid'],
            tickfont=dict(color=t['font']),
            scaleanchor='x', scaleratio=sky_ratio,
        ),
    )
    return fig


def fit_result_figures(result: dict, fits_header=None, *,
                       theme='light', colorscale='Viridis') -> Dict[str, Any]:
    """Build Plotly figures for fitted x/y/z parameter maps."""
    if not result:
        return {}
    header = fits_header if fits_header is not None else result.get('fits_header')
    figs = {
        'x': fig_fits_map(
            result['x_param_map'],
            'Best-fit hydrogen density',
            header=header, zscale='log',
            colorbar_title='log<sub>10</sub> n (cm<sup>-3</sup>)',
            theme=theme, colorscale=colorscale,
        ),
        'y': fig_fits_map(
            result['y_param_map'],
            f'Best-fit FUV field ({Y_MESH_KEY})',
            header=header, zscale='log',
            colorbar_title='log<sub>10</sub> χ',
            theme=theme, colorscale=colorscale,
        ),
        'z': fig_fits_map(
            result['z_param_map'],
            f'Best-fit cosmic-ray rate ({Z_MESH_KEY})',
            header=header, zscale='log',
            colorbar_title='log<sub>10</sub> ζ',
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
    return figs
