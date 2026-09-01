#!/usr/bin/env python3
"""
PRISMA - PDR Results Interactive Simulation Mapping & Analysis
==============================================================

Interactive browser for KOSMA-tau / KoSens3D photodissociation-region (PDR)
model grids.

Unlike the original 3D-PDR explorer (which reads a single combined HDF5 file),
KoSens3D stores **one HDF5 file per model point**.  Each filename encodes the
model parameters:

    Model<tag>_DD_MM_FF_ZZ_CC_AA.hdf5
                |  |  |  |  |  |
                |  |  |  |  |  +-- AA : attenuation tag
                |  |  |  |  +----- CC : cosmic-ray ionisation rate  -> 10^(-CC) s^-1
                |  |  |  +-------- ZZ : metallicity                 -> 10^((ZZ-10)/10) Zsun
                |  |  +----------- FF : FUV field (Draine chi)      -> 10^(FF/10)
                |  +-------------- MM : clump mass                  -> 10^(MM/10) Msun
                +----------------- DD : gas density n_H             -> 10^(DD/10) cm^-3

When a filename does not follow that convention, grid values are read from the
HDF5 itself (``protdens``, ``cmass``, ``radm_ini``, ``metal``, ``cosray`` /
``zeta``).  A single imported model still loads; a set of such files still
builds a grid from whichever parameters vary.

The tool scans a directory of such files, works out which parameters vary,
and exposes one slider per varying parameter.  The current grid varies
density x FUV x CRIR (a 3-D cube) at fixed mass/metallicity, but the code is
written generically so that mass (or any other axis) can become a free
parameter without changes -- its slider simply appears once more than one
value is present on disk.

For each selected model the depth profiles (vs Av or n_H) of gas temperature,
H/H2, C+/C/CO and a set of custom species are plotted, mirroring the layout of
the original 3D-PDR explorer.

Usage:
    python app.py [--dir /path/to/pdrgrid_hdf5] [--recursive]
                  [--port 8050] [--host 127.0.0.1] [--debug]
"""

import argparse
import os
import glob
import re

import numpy as np
import h5py

import dash
from dash import dcc, html, Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate

# Enlarge every dropdown's open menu so more species/options are visible at once
# (Dash defaults maxHeight to 200px, which only shows a few rows). Applies to all
# dcc.Dropdown instances across every tab unless a call passes its own maxHeight.
_DROPDOWN_MENU_MAX_HEIGHT = 360
_OrigDropdown = dcc.Dropdown


def _TallDropdown(*args, **kwargs):
    kwargs.setdefault('maxHeight', _DROPDOWN_MENU_MAX_HEIGHT)
    return _OrigDropdown(*args, **kwargs)


dcc.Dropdown = _TallDropdown
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import grid_fit as gf
import grid_interp as gi
import model_config as mc
import simline_spectra as ss
import obs_spectrum_fits as osf
import cr_attenuation as cra
import grid_naming as gn
import rgb_phase
import chem_network
import plot_style as ps
import map_fit_extras as mfe


# --- CLI ----------------------------------------------------------------------

OBS_ROW_OPTIONS = [
    {'label': ' Mean (all spectra / pixels)', 'value': 'mean'},
    {'label': ' Peak region (bright half)', 'value': 'peak'},
    {'label': ' Single row / pixel index', 'value': 'row'},
]
DEFAULT_OBS_V_LOW = -20.0
DEFAULT_OBS_V_HIGH = 20.0
DEFAULT_DIR = ''


parser = argparse.ArgumentParser(description="PRISMA - PDR Results Interactive Simulation Mapping & Analysis")
parser.add_argument('--dir', default=DEFAULT_DIR, type=str,
                    help='Directory of per-model .hdf5 files, or a single .hdf5 model.')
parser.add_argument('--recursive', action='store_true',
                    help='Search the directory tree recursively for .hdf5 files.')
parser.add_argument('--port',  default=8050, type=int)
parser.add_argument('--host',  default='127.0.0.1')
parser.add_argument('--debug', action='store_true')
args, _ = parser.parse_known_args()


# --- Constants ----------------------------------------------------------------

MIN_AB = 1e-30                      # floor for log abundance plots
MIN_RATE = 1e-40                    # floor for log heating/cooling rate plots
AV_FLOOR = 1e-5                     # optional A_V display floor (mag); off by default
DEFAULT_AV_RANGE = 'full'           # 'full' = all HDF5 A_V; 'floor' = clip below AV_FLOOR
METADATA_PATH   = 'Metadata/Metadata'
SPECIES_PATH    = 'Additional output/species involved'
RELDENS_PATH    = 'Local quantities/Densities/Relative densities'
DENS_PATH       = 'Local quantities/Densities/Densities'
HEATING_PATH    = 'Local quantities/Auxiliary/Thermal balance/Heating rates'
COOLING_PATH    = 'Local quantities/Auxiliary/Thermal balance/Cooling rates'

# Chemistry (reaction-rate) grid: a separate set of HDF5 files, one per model,
# named like ``chem_Model<tag>_DD_MM_FF_ZZ_CC_AA.hdf5``.  Per species they hold
# formation/destruction rate matrices (n_depth x n_reactions).
CHEM_GROUP   = 'Local quantities/Chemistry'        # parent group of per-species groups
CHEM_POS     = 'Local quantities/Positions'        # av axis for chem data (col 0)
CHEM_DEFAULT_SPECIES = 'CO'
CHEM_DEFAULT_NREAC = 3                              # top-N reactions per depth point
REACT_RANKING_OPTIONS = [
    {'label': ' Fractional contribution (% of total rate)', 'value': 'fractional_contribution'},
    {'label': ' Mass-weighted rate', 'value': 'mass_weighted_rate'},
]
DEFAULT_REACT_RANKING = 'fractional_contribution'

SIMLINE_DEFAULT_SPECIES = 'CO'
SIMLINE_DEFAULT_IDEF = 'jtemp'                      # jtemp, jerg, or tau
# KoSens-style observational detection limit on jtemp intensity maps (K km/s).
INT_OBS_BOUNDARY_JTEMP = 0.1
INT_OBS_BOUNDARY_COLOR = 'red'
INT_OBS_BOUNDARY_WIDTH = 2.5
INT_EXTRA_CONTOUR_WIDTH = 1.5
# Discrete palette for multi-level / spaghetti contours (Plotly tab20-like).
INT_CONTOUR_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]
INT_SPAGHETTI_DEFAULT_ERROR_FRAC = 0.20
INT_SPAGHETTI_LINE_WIDTH = 2.5
INT_SPAGHETTI_BAND_OPACITY = 0.35
SIMLINE_IDEF_OPTIONS = [
    {'label': ' I [K km/s]  (jtemp)', 'value': 'jtemp'},
    {'label': ' I [erg s\u207B\u00B9 cm\u207B\u00B2 Hz\u207B\u00B9]  (jerg)', 'value': 'jerg'},
    {'label': ' \u03c4  optical depth  (tau)', 'value': 'tau'},
]
SIMLINE_MAPFIT_IDEF_OPTIONS = [
    o for o in SIMLINE_IDEF_OPTIONS if o['value'] in ('jtemp', 'jerg')
]
SIMLINE_PV_QUANTITY_OPTIONS = [
    {'label': ' T<sub>mb</sub> [K]  (brightness temperature)', 'value': 'intensity'},
    {'label': ' \u03c4  optical depth  (tau)', 'value': 'tau'},
]
SIMLINE_DEFAULT_PV_QUANTITY = 'intensity'

# RGB phase-diagram defaults (KoSens ``plot_species_phase_diagram``).
RGB_DEFAULT_GRID_SPECIES = ('C+', 'C', 'CO')
RGB_CHANNEL_COLORS = (
    {'label': 'Channel 1 (blue)', 'color': '#00008B'},
    {'label': 'Channel 2 (green)', 'color': '#008000'},
    {'label': 'Channel 3 (red)', 'color': '#D62728'},
)
RGB_GRID_TRANSITION_OPTIONS = [
    {'label': ' H \u2194 H\u2082 (dashed)', 'value': 'hh2'},
    {'label': ' C \u2194 CO (solid)', 'value': 'cco'},
    {'label': ' CO \u2194 JCO (dash-dotted)', 'value': 'cojco'},
]
RGB_INT_TRANSITION_OPTIONS = [
    {'label': ' H \u2194 H\u2082 from HDF5 abundances (dashed)', 'value': 'hh2'},
    {'label': ' C \u2194 CO from line intensities (solid)', 'value': 'cco'},
]

# Attenuation x-shift matching (KoSens grid_attenuation_compare / plot_triple_grid_ratio)
X_SHIFT_MATCH_RTOL = 0.02
X_SHIFT_SCAN_DIRECTION = 'rightward_then_leftward'
X_SHIFT_DIRECTION_OPTIONS = [
    {'label': ' Rightward then leftward', 'value': 'rightward_then_leftward'},
    {'label': ' Rightward only', 'value': 'rightward'},
    {'label': ' Leftward only', 'value': 'leftward'},
]

# KoSens-style 2-D grid resampling (3d_grids.ipynb / grid_functions resample path)
DEFAULT_INTERP_NY = 60
DEFAULT_INTERP_NX = 60
DEFAULT_INTERP_METHOD = 'linear'
DEFAULT_INTERP_CLIP = False
DEFAULT_CONTOUR_PANEL_W = 520
# Square axes box (px) for every side-by-side Grid / Intensities / RGB panel.
# Domains leave a right strip for the colorbar; figure size is derived so the
# plotted box is exactly COMPACT_PLOT_BOX × COMPACT_PLOT_BOX pixels.
COMPACT_PLOT_BOX = 400
_COMPACT_FIG_MARGIN = dict(l=64, r=56, t=52, b=58)
_COMPACT_XDOMAIN = [0.0, 0.84]   # ~16% reserved for colorbar
_COMPACT_YDOMAIN = [0.0, 1.0]    # full paper height → square with x-domain span
_COMPACT_XSPAN = _COMPACT_XDOMAIN[1] - _COMPACT_XDOMAIN[0]
_COMPACT_YSPAN = _COMPACT_YDOMAIN[1] - _COMPACT_YDOMAIN[0]
_COMPACT_CBAR_X = _COMPACT_XDOMAIN[1] + 0.015  # chi²-style: just right of the plot
COMPACT_FIG_WIDTH = int(round(
    _COMPACT_FIG_MARGIN['l'] + COMPACT_PLOT_BOX / _COMPACT_XSPAN
    + _COMPACT_FIG_MARGIN['r']
))
COMPACT_FIG_HEIGHT = int(round(
    _COMPACT_FIG_MARGIN['t'] + COMPACT_PLOT_BOX / _COMPACT_YSPAN
    + _COMPACT_FIG_MARGIN['b']
))
# Page shell: wide enough for three compact panels side by side.
APP_MAX_WIDTH = 'min(1920px, 98vw)'
# Plot text sizes live in plot_style.py (PLOT_AXIS_FONT_SIZE, PLOT_TICK_FONT_SIZE, …).
# Legacy alias used by a few non-compact figure paths.
COMPACT_CONTOUR_PANEL_W = COMPACT_PLOT_BOX
INTERP_METHOD_OPTIONS = [
    {'label': ' Linear', 'value': 'linear'},
    {'label': ' Cubic', 'value': 'cubic'},
    {'label': ' Nearest', 'value': 'nearest'},
    {'label': ' Spline', 'value': 'spline'},
]
GRID_COLORMAP_OPTIONS = [
    {'label': ' Viridis', 'value': 'Viridis'},
    {'label': ' Magma', 'value': 'Magma'},
    {'label': ' Plasma', 'value': 'Plasma'},
    {'label': ' Inferno', 'value': 'Inferno'},
    {'label': ' Cividis', 'value': 'Cividis'},
    {'label': ' Turbo', 'value': 'Turbo'},
    {'label': ' Hot', 'value': 'Hot'},
    {'label': ' Blues', 'value': 'Blues'},
    {'label': ' YlOrRd', 'value': 'YlOrRd'},
]
DEFAULT_GRID_COLORMAP = 'Viridis'
PLOT_THEME_OPTIONS = [
    {
        'label': html.Span([
            html.Span(className='theme-ico theme-ico-sun'),
            html.Span('Light', className='theme-opt-text'),
        ], className='theme-opt'),
        'value': 'light',
    },
    {
        'label': html.Span([
            html.Span(className='theme-ico theme-ico-moon'),
            html.Span('Dark', className='theme-opt-text'),
        ], className='theme-opt'),
        'value': 'dark',
    },
]
DEFAULT_PLOT_THEME = 'light'
PLOT_THEMES = {
    'light': dict(
        paper_bg='#ffffff',
        plot_bg='#ffffff',
        grid='#e8edf3',
        axis_line='#cbd5e1',
        title='#0f172a',
        font='#1e293b',
        muted='#64748b',
        heading='#0f172a',
        legend_bg='rgba(255,255,255,0.92)',
        legend_border='#e2e8f0',
        placeholder='#94a3b8',
        placeholder_plot='#f1f4f8',
        contour_line='rgba(255,255,255,0.85)',
        vline='rgba(60,60,60,0.45)',
        controls_bg='#ffffff',
        page_bg='#f1f4f8',
        card_bg='#ffffff',
        card_border='#e2e8f0',
        input_bg='#ffffff',
        input_border='#cbd5e1',
        tab_border='#e2e8f0',
        tab_bg='#ffffff',
        tab_sel_bg='#eff6ff',
        accent='#2563eb',
    ),
    'dark': dict(
        paper_bg='#1e293b',
        plot_bg='#0f172a',
        grid='#334155',
        axis_line='#475569',
        title='#f1f5f9',
        font='#e2e8f0',
        muted='#94a3b8',
        heading='#f1f5f9',
        legend_bg='rgba(30,41,59,0.94)',
        legend_border='#475569',
        placeholder='#94a3b8',
        placeholder_plot='#151b24',
        contour_line='rgba(255,255,255,0.35)',
        vline='rgba(220,220,230,0.45)',
        controls_bg='#1e293b',
        page_bg='#0f1419',
        card_bg='#1e293b',
        card_border='#334155',
        input_bg='#0f172a',
        input_border='#475569',
        tab_border='#334155',
        tab_bg='#151b24',
        tab_sel_bg='#1e293b',
        accent='#3b82f6',
    ),
}
UI_FONT = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
DEFAULT_ERROR_DECIMATION = 2
DEFAULT_ERROR_REL_THRESHOLD = 0.01
ERROR_METRIC_OPTIONS = [
    {'label': html.Span('Relative (%)', className='seg-opt'), 'value': 'relative'},
    {'label': html.Span('Absolute', className='seg-opt'), 'value': 'absolute'},
]
ERROR_PANEL_CMAP = 'RdYlGn_r'

# Internal HDF5 metadata field keys (KOSMA-tau convention).
KEY_AV    = 'av'             # visual extinction profile (Positions, col 0)
KEY_NH    = 'protdens'       # proton/H nucleus density profile (Gas state, col 0)
KEY_TGAS  = 'tgas'           # gas temperature  (Gas state, col 2)
KEY_TDUST = 'tdust'          # dust temperature (Gas state, col 3)
KEY_NELECTR = 'n_electr'     # electron density n(e-) (Densities)
KEY_HEAT_CR = 'heatrate_cr'  # cosmic-ray heating component (Heating rates, col 3)
KEY_COSRAY  = 'cosray'       # CR ionisation rate profile (Gas state)
KEY_NH2_PROFILE = 'cd_prof_h2'  # H2 column-density profile (Local quantities)
KEY_RADIUS  = 'radius'       # radial profile (pc) for integrated abundances

CR_ATTEN_XAXIS_OPTIONS = [
    {'label': ' N<sub>H₂</sub> [cm⁻²]', 'value': 'nh2'},
    {'label': ' A<sub>V</sub> [mag]', 'value': 'av'},
]

PC_TO_CM = 3.08567758128e18  # parsec to cm

# Parameter definitions.  ``token_idx`` is the position of the value in the
# underscore-split filename (Model<tag>_DD_MM_FF_ZZ_CC_AA -> indices 1..6).
PARAM_DEFS = [
    dict(key='density', token_idx=1, name='Density n_H',     unit='cm\u207B\u00B3',
         color='#1f77b4', logscale=True, decode=lambda t: 10.0 ** (t / 10.0)),
    dict(key='mass',    token_idx=2, name='Clump mass',      unit='M\u2299',
         color='#9467bd', logscale=True, decode=lambda t: 10.0 ** (t / 10.0)),
    dict(key='fuv',     token_idx=3, name='FUV field \u03C7', unit='Draine',
         color='#ff7f0e', logscale=True, decode=lambda t: 10.0 ** (t / 10.0)),
    dict(key='metal',   token_idx=4, name='Metallicity Z',   unit='Z\u2299',
         color='#2ca02c', logscale=True, decode=lambda t: 10.0 ** ((t - 10.0) / 10.0)),
    dict(key='crir',    token_idx=5, name='Cosmic-ray rate \u03B6', unit='s\u207B\u00B9',
         color='#d62728', logscale=True, decode=lambda t: 10.0 ** (-t)),
    dict(key='atten',   token_idx=6, name='Attenuation',     unit='',
         color='#8c564b', logscale=False, decode=lambda t: float(t)),
]
N_PARAMS = len(PARAM_DEFS)
_PARAM_IDX = {p['key']: i for i, p in enumerate(PARAM_DEFS)}

# Three fixed UI slots; axes are assigned at load time from varying parameters.
SLICE_PLANE_SLOT_IDS = ('dens-fuv', 'dens-crir', 'fuv-crir')
_SLICE_PLANES_ACTIVE = None

SLICE_PLANES = [dict(id=pid) for pid in SLICE_PLANE_SLOT_IDS]

_PLANE_AXIS_SHORT = {
    'density': ('n<sub>H</sub>', 'nH'),
    'mass': ('M', 'M'),
    'fuv': ('FUV \u03C7', 'FUV'),
    'metal': ('Z', 'Z'),
    'crir': ('\u03B6', '\u03B6'),
    'atten': ('atten', 'atten'),
}

CONTOUR_DIAGNOSTICS = [
    ('tgas', 'T<sub>gas</sub> (cloud edge)'),
    ('tgas_col', 'T<sub>gas</sub> (column avg)'),
    ('tdust', 'T<sub>dust</sub> (cloud edge)'),
    ('nh', 'n<sub>H</sub> (cloud edge)'),
    ('xe', 'x<sub>e</sub> = n(e<sup>-</sup>)/n<sub>H</sub> (cloud edge)'),
]
CONTOUR_DEFAULT_QUANTITY = 'species:CO'

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

DEFAULT_CUSTOM = ['HCO+', 'N2H+', 'O']

_MAP_FIT_MAPS_EXAMPLE = (
    '{\n'
    '  "H13CO+(1-0)": "/path/to/dr21_h13co+10_-10_20_repro.fits",\n'
    '  "HCO+(1-0)": "/path/to/dr21_hco+10_-30_30_repro.fits",\n'
    '  "CO(3-2)": "/path/to/jcmt_dr21_12co32_-10_20_repro.fits"\n'
    '}'
)
_MAP_FIT_ERRORS_EXAMPLE = (
    '{\n'
    '  "H13CO+(1-0)": 0.059,\n'
    '  "HCO+(1-0)": 0.050,\n'
    '  "CO(3-2)": 0.348\n'
    '}'
)


# --- Server-side data store ---------------------------------------------------
# Single global dict (last loaded directory wins).  For multi-user deployments
# move this into flask_caching keyed by session.

_grid = {}            # main grid (populated on load)
_overlay = {}         # optional second grid (e.g. attenuated), overplotted
_chem = {}            # optional chemistry (reaction-rate) grid
_chem_overlay = {}    # optional overlay chemistry grid (comparison)
_simline = {}         # optional SIMLINE (.smli) intensity grid
_simline_overlay = {} # optional attenuated SIMLINE grid (intensity comparison)
_fit_results = {}     # last map-fit output (parameter maps + paths)
_model_config_summary = None  # scan of Models/**/config_files/*.json
_field_map = {}       # field_key -> (hdf5_path, column_index)
_profile_cache = {}   # filepath -> dict of profile arrays
_reaction_cache = {}  # (filepath, species, mode) -> dict of reaction data
_smli_cache = {}      # smli filepath -> list of transition rows
_scalar_cache = {}    # (filepath, quantity) -> float
_heat_components = []  # list of (label, column_index) for heating rates
_cool_components = []  # list of (label, column_index) for cooling rates
_cr_heat_idx = None    # column index of cosmic-ray heating in Heating rates


def _dec(x):
    return x.decode() if isinstance(x, bytes) else x


# --- Grid scanning ------------------------------------------------------------

def _sorted_axis_tokens(key, tokens):
    """Sort filename tokens by decoded physical value (needed for CRIR: token↑ ⇒ ζ↓)."""
    pdef = PARAM_DEFS[_PARAM_IDX[key]]
    return sorted(set(tokens), key=lambda t: (pdef['decode'](t), t))


def order_axis_tokens_physically(axis_tokens):
    """Return axis_tokens with each axis ordered by ascending physical value."""
    return {p['key']: _sorted_axis_tokens(p['key'], axis_tokens.get(p['key']) or [])
            for p in PARAM_DEFS}


def _tokens_for_path(store, path):
    """Token tuple that maps to ``path`` in a scanned chemistry/HDF5 store."""
    if not store or not path:
        return None
    for tok, p in (store.get('files') or {}).items():
        if p == path:
            return tuple(tok)
    return None


def _varying_param_keys(axis_tokens):
    return [p['key'] for p in PARAM_DEFS if len(axis_tokens.get(p['key'], [])) > 1]


def _fixed_param_keys(axis_tokens):
    return [p['key'] for p in PARAM_DEFS if len(axis_tokens.get(p['key'], [])) == 1]


def _plane_title_html(xk, yk):
    xl, _ = _PLANE_AXIS_SHORT.get(xk, (xk, xk))
    yl, _ = _PLANE_AXIS_SHORT.get(yk, (yk, yk))
    return f'{xl} vs {yl}'


def _plane_title_plain(xk, yk):
    _, xp = _PLANE_AXIS_SHORT.get(xk, (xk, xk))
    _, yp = _PLANE_AXIS_SHORT.get(yk, (yk, yk))
    return f'{xp} vs {yp}'


def _inactive_slot_planes():
    return [dict(id=pid, x=None, y=None, slice=None, title='', active=False)
            for pid in SLICE_PLANE_SLOT_IDS]


def build_slice_planes(axis_tokens):
    """Build up to three 2-D slice planes from whichever parameters vary."""
    varying = _varying_param_keys(axis_tokens)
    fixed = _fixed_param_keys(axis_tokens)
    combos = []
    n = len(varying)
    if n >= 3:
        v0, v1, v2 = varying[0], varying[1], varying[2]
        combos = [(v0, v1, v2), (v0, v2, v1), (v1, v2, v0)]
    elif n == 2:
        sk = fixed[0] if fixed else varying[0]
        combos = [(varying[0], varying[1], sk)]
    elif n == 1:
        others = [p['key'] for p in PARAM_DEFS if p['key'] != varying[0]]
        if len(others) >= 2:
            combos = [(varying[0], others[0], others[1])]

    planes = []
    for i, pid in enumerate(SLICE_PLANE_SLOT_IDS):
        if i < len(combos):
            xk, yk, sk = combos[i]
            planes.append(dict(
                id=pid, x=xk, y=yk, slice=sk,
                title=_plane_title_html(xk, yk), active=True,
            ))
        else:
            planes.append(dict(id=pid, x=None, y=None, slice=None, title='', active=False))
    return planes


def configure_slice_planes(axis_tokens):
    """Assign slice-plane axes from the parameters that vary on disk."""
    global _SLICE_PLANES_ACTIVE
    _SLICE_PLANES_ACTIVE = build_slice_planes(axis_tokens)


def active_slice_planes():
    if _SLICE_PLANES_ACTIVE is not None:
        return _SLICE_PLANES_ACTIVE
    return _inactive_slot_planes()


def _ie_plane_label(plane_id):
    for p in active_slice_planes():
        if p['id'] == plane_id and p.get('active'):
            return _plane_title_plain(p['x'], p['y'])
    return ''


def _plane_by_slot(slot_id):
    for p in active_slice_planes():
        if p['id'] == slot_id:
            return p
    return dict(id=slot_id, active=False)


def parse_filename(fname):
    """Return a tuple of 6 encoded tokens (DD, MM, FF, ZZ, CC, AA) or None.

    Works for both ``Model<tag>_DD_MM_FF_ZZ_CC_AA.hdf5`` and
    ``Model<tag>_DD_MM_FF_ZZ_CC.hdf5`` (missing ``AA`` -> atten 0), plus
    chemistry-grid ``chem_Model<tag>_…`` names.  Names that parse as numbers
    but are physical values (e.g. ``n_H=100000`` instead of token ``50``)
    return None so the HDF5 fallback can run.
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    tokens = gn.parse_model_tokens_from_stem(stem)
    if tokens is None or not gn.tokens_look_encoded(tokens):
        return None
    return tokens


def _hdf5_cell_float(cell):
    """Parse one HDF5 cell (numeric, bytes, or string) to float, or None."""
    if cell is None:
        return None
    if isinstance(cell, (bytes, np.bytes_)):
        text = cell.decode('utf-8', 'replace').strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if np.isfinite(value) else None
    if isinstance(cell, str):
        text = cell.strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if np.isfinite(value) else None
    try:
        value = float(cell)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _hdf5_first_float(hf, path, idx):
    """First (surface / initial) value of a metadata-mapped HDF5 column."""
    try:
        arr = np.asarray(hf[path][:])
    except (KeyError, TypeError, OSError, ValueError):
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        return _hdf5_cell_float(arr.item())
    if arr.ndim == 1:
        if 0 <= int(idx) < arr.size:
            value = _hdf5_cell_float(arr[int(idx)])
            if value is not None:
                return value
        return _hdf5_cell_float(arr[0])
    col = int(idx) if 0 <= int(idx) < arr.shape[1] else 0
    return _hdf5_cell_float(arr[0, col])


def _hdf5_wanted_field_map(hf, wanted):
    """Map selected metadata keys to ``(dataset_path, column_index)``."""
    field_map = {}
    if METADATA_PATH not in hf:
        return field_map
    try:
        md = hf[METADATA_PATH][:]
    except (KeyError, TypeError, OSError):
        return field_map
    for row in md:
        key = _dec(row[3])
        if key not in wanted or key in field_map:
            continue
        group = _dec(row[0])
        dset = _dec(row[1])
        try:
            idx = int(float(_dec(row[2])))
        except (ValueError, TypeError):
            continue
        path = group.rstrip('/') + '/' + dset
        field_map[key] = (path, idx)
        if len(field_map) >= len(wanted):
            break
    return field_map


_PDR_CONFIG_PATHS = ('Parameters/pdr_config.json', 'pdr_config.json')
_PDR_CONFIG_KEY_MAP = {
    'surface_density': 'density',
    'radiation_field_strength': 'fuv',
    'cosmic_ray_rate': 'crir',
    'metallicity': 'metal',
}
_PDR_CONFIG_RE = re.compile(
    r'"(surface_density|radiation_field_strength|cosmic_ray_rate|metallicity)"'
    r'\s*:\s*([-+0-9.eE]+)'
)


def _pdr_config_physicals(hf):
    """Fill missing grid params from an embedded ``pdr_config.json`` dataset."""
    out = {}
    ds = None
    for path in _PDR_CONFIG_PATHS:
        if path in hf:
            ds = hf[path]
            break
    if ds is None:
        return out
    try:
        arr = np.asarray(ds[()])
    except (TypeError, OSError, ValueError):
        return out
    chunks = []
    for cell in np.ravel(arr):
        if isinstance(cell, (bytes, np.bytes_)):
            chunks.append(cell.decode('utf-8', 'replace'))
        else:
            chunks.append(str(cell))
    for raw_key, raw_val in _PDR_CONFIG_RE.findall('\n'.join(chunks)):
        param = _PDR_CONFIG_KEY_MAP.get(raw_key)
        if param is None or param in out:
            continue
        value = _hdf5_cell_float(raw_val)
        if value is not None:
            out[param] = value
    return out


def _hdf5_column_floats(hf, path, idx):
    """Numeric values of one metadata-mapped column, or None."""
    try:
        arr = np.asarray(hf[path][:])
    except (KeyError, TypeError, OSError, ValueError):
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        value = _hdf5_cell_float(arr.item())
        return np.array([value], dtype=float) if value is not None else None
    if arr.ndim == 1:
        cells = arr
    else:
        col = int(idx) if 0 <= int(idx) < arr.shape[1] else 0
        cells = arr[:, col]
    values = [_hdf5_cell_float(cell) for cell in np.ravel(cells)]
    values = [v for v in values if v is not None]
    return np.asarray(values, dtype=float) if values else None


def read_hdf5_grid_physicals(path):
    """Physical grid (and extra) values from one KOSMA-τ HDF5 file.

    Returns a dict with any of ``density``, ``mass``, ``fuv``, ``metal``,
    ``crir``, plus extras ``av_max`` and ``radius`` when those columns exist.
    Missing keys are omitted (not defaulted).
    """
    wanted = {
        'protdens', 'cmass', 'metal', 'radm_ini', 'cosray', 'zeta',
        'av', 'radius', 'distance',
    }
    phys = {}
    try:
        with h5py.File(path, 'r') as hf:
            fmap = _hdf5_wanted_field_map(hf, wanted)
            if 'protdens' in fmap:
                phys['density'] = _hdf5_first_float(hf, *fmap['protdens'])
            if 'cmass' in fmap:
                phys['mass'] = _hdf5_first_float(hf, *fmap['cmass'])
            if 'radm_ini' in fmap:
                phys['fuv'] = _hdf5_first_float(hf, *fmap['radm_ini'])
            if 'metal' in fmap:
                phys['metal'] = _hdf5_first_float(hf, *fmap['metal'])
            crir = None
            if 'cosray' in fmap:
                crir = _hdf5_first_float(hf, *fmap['cosray'])
            if crir is None and 'zeta' in fmap:
                crir = _hdf5_first_float(hf, *fmap['zeta'])
            if crir is not None:
                phys['crir'] = abs(crir)
            for key, value in _pdr_config_physicals(hf).items():
                if phys.get(key) is None:
                    phys[key] = abs(value) if key == 'crir' else value
            if 'av' in fmap:
                av_col = _hdf5_column_floats(hf, *fmap['av'])
                if av_col is not None and av_col.size:
                    phys['av_max'] = float(np.nanmax(av_col))
            radius_key = 'radius' if 'radius' in fmap else (
                'distance' if 'distance' in fmap else None)
            if radius_key:
                rad_col = _hdf5_column_floats(hf, *fmap[radius_key])
                if rad_col is not None and rad_col.size:
                    radius = float(np.nanmax(rad_col))
                    if radius > 1e15:
                        radius = radius / PC_TO_CM
                    phys['radius'] = radius
    except OSError:
        return {}
    return {k: v for k, v in phys.items() if v is not None and np.isfinite(v)}


def parse_tokens_from_hdf5(path):
    """Encode grid tokens from KOSMA-τ HDF5 fields when the filename is nonstandard.

    Returns ``(tokens, phys)`` or ``None``.  ``phys`` holds the actual HDF5
    numbers used for display (not the rounded decoded tokens).
    """
    phys = read_hdf5_grid_physicals(path)
    tokens = gn.physical_values_to_tokens(
        density=phys.get('density'),
        mass=phys.get('mass'),
        fuv=phys.get('fuv'),
        metal=phys.get('metal'),
        crir=phys.get('crir'),
    )
    if tokens is None:
        return None
    return tokens, phys


def _clean_rate_label(label, kind):
    """Strip the 'Heating - ' / 'Cooling - ' prefix from a metadata label."""
    for prefix in ('Heating - ', 'Cooling - ', 'Heating-', 'Cooling-'):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


def _species_name_from_density_label(label, key=''):
    """Turn a Densities metadata label ``n(C+)`` into the species name ``C+``."""
    text = str(label or '').strip()
    match = re.match(r'^n\((.+)\)$', text)
    if match:
        name = match.group(1).strip()
        if name in ('e-', 'e'):
            return 'ELECTR'
        return name
    key = str(key or '').strip()
    if key.startswith('n_') and key != 'n_electr':
        return key[2:]
    if key == 'n_electr':
        return 'ELECTR'
    return text or None


def _add_isotope_caret(name):
    """KoSens ``add_isotope_caret``: insert ``^`` before isotope mass numbers.

    ``c18o`` → ``c^18o`` so the HDF5 key is ``cd_c^18o``; leading digits such as
    ``13co`` stay as-is (``cd_13co``).
    """
    text = str(name or '')
    parts = []
    last = 0
    for match in re.finditer(r'\d+', text):
        start, end = match.span()
        number = match.group()
        parts.append(text[last:start])
        if start > 0 and (
            (len(number) == 3)
            or (len(number) == 2 and int(number) >= 6)
            or (len(number) == 1 and int(number) >= 6)
        ):
            if len(number) == 3:
                parts.append(number[0] + '^' + number[1:])
            else:
                parts.append('^' + number)
        else:
            parts.append(number)
        last = end
    parts.append(text[last:])
    return ''.join(parts)


def tot_col_dens_hdf5_key(species):
    """KoSens ``dens_names_translation`` key for total column density.

    Metadata field ``cd_<name>`` in ``Integrated quantities/Column densities``
    (e.g. CO → ``cd_co``, C+ → ``cd_cp``, C18O → ``cd_c^18o``).
    """
    name = str(species or '').lower().replace('-', '_').replace('+', 'p').replace('^', '')
    if name in ('e', 'e-', 'electr'):
        name = 'electr'
    return 'cd_' + _add_isotope_caret(name)


def _species_from_density_metadata(md):
    """Species names ordered by Densities column index (older KOSMA-τ HDF5)."""
    by_idx = {}
    for row in md:
        dset = _dec(row[1])
        group = _dec(row[0])
        if dset != 'Densities' or 'Column' in group:
            continue
        try:
            idx = int(float(_dec(row[2])))
        except (ValueError, TypeError):
            continue
        if idx in by_idx:
            continue
        name = _species_name_from_density_label(_dec(row[4]), _dec(row[3]))
        if name:
            by_idx[idx] = name
    return [by_idx[i] for i in sorted(by_idx)]


def _read_species_list(hf, md):
    """Species names from ``species involved``, else Densities metadata labels."""
    if SPECIES_PATH in hf:
        try:
            return [_dec(x[0]) for x in hf[SPECIES_PATH][:]]
        except (KeyError, TypeError, OSError, ValueError):
            pass
    return _species_from_density_metadata(md)


def build_structure(sample_file):
    """Read metadata + species list from one representative file.

    Returns
    -------
    field_map : dict   field_key -> (hdf5_internal_path, column_index)
    species   : list[str]
    heat_comp : list[(label, idx)]  heating-rate components
    cool_comp : list[(label, idx)]  cooling-rate components
    cr_idx    : int | None          column index of CR heating
    """
    field_map = {}
    heat_comp, cool_comp = [], []
    cr_idx = None
    with h5py.File(sample_file, 'r') as hf:
        md = hf[METADATA_PATH][:]
        for row in md:
            group = _dec(row[0])
            dset = _dec(row[1])
            key = _dec(row[3])
            label = _dec(row[4])
            try:
                idx = int(float(_dec(row[2])))
            except (ValueError, TypeError):
                continue
            path = group.rstrip('/') + '/' + dset
            if key not in field_map:
                field_map[key] = (path, idx)
            if dset == 'Heating rates':
                heat_comp.append((_clean_rate_label(label, 'h'), idx))
                if key == KEY_HEAT_CR:
                    cr_idx = idx
            elif dset == 'Cooling rates':
                cool_comp.append((_clean_rate_label(label, 'c'), idx))
        species = _read_species_list(hf, md)
    heat_comp.sort(key=lambda t: t[1])
    cool_comp.sort(key=lambda t: t[1])
    return field_map, species, heat_comp, cool_comp, cr_idx


def _scan_status_note(n_from_hdf5=0, n_skipped=0):
    """Short parenthetical for load-status lines."""
    bits = []
    if n_from_hdf5:
        bits.append(f'{n_from_hdf5} from HDF5 metadata')
    if n_skipped:
        bits.append(f'{n_skipped} file(s) skipped')
    return f'  ({"; ".join(bits)})' if bits else ''


def _collect_hdf5_paths(source, recursive=False):
    """Return ``(source, hdf5_paths)`` for a directory or a single HDF5 file."""
    source = os.path.expanduser((source or '').strip())
    if not source:
        raise FileNotFoundError('No grid directory or HDF5 file specified.')
    source = os.path.abspath(source)
    if os.path.isfile(source):
        if not source.lower().endswith(('.hdf5', '.h5')):
            raise FileNotFoundError(f'Not an HDF5 file: {source!r}')
        return source, [source]
    if os.path.isdir(source):
        paths = []
        for ext in ('*.hdf5', '*.h5'):
            pattern = os.path.join(source, '**', ext) if recursive \
                else os.path.join(source, ext)
            paths.extend(glob.glob(pattern, recursive=recursive))
        paths = sorted(set(paths))
        if not paths:
            raise FileNotFoundError(f'No .hdf5 files found in {source!r}')
        return source, paths
    raise FileNotFoundError(f'Directory or HDF5 file not found: {source!r}')


def _config_scan_root(source):
    """Directory used to look up ``Models/**/config_files`` next to a grid."""
    if os.path.isdir(source):
        return source
    return os.path.dirname(source) or source


def _scan_files(directory, recursive):
    """Return (source, files, axis_tokens, n_skipped, n_from_hdf5, phys_by_path).

    ``directory`` may be a folder of models or a single ``.hdf5`` file.
    Filenames that follow the encoded-token convention are used as-is (fast
    path).  Anything else is opened and tagged from HDF5 metadata.
    """
    source, paths = _collect_hdf5_paths(directory, recursive)

    files = {}
    phys_by_path = {}
    skipped = 0
    n_from_hdf5 = 0
    for path in paths:
        tokens = parse_filename(path)
        if tokens is None:
            parsed = parse_tokens_from_hdf5(path)
            if parsed is None:
                skipped += 1
                continue
            tokens, phys = parsed
            n_from_hdf5 += 1
            if phys:
                phys_by_path[path] = phys
        files[tokens] = path
    if not files:
        raise ValueError(
            'Found .hdf5 files but none had encoded <tag>_DD_MM_FF_ZZ_CC[_AA] '
            'names or readable KOSMA-τ grid parameters (protdens / cmass / '
            'radm_ini / metal / cosray).'
        )

    axis_tokens = {}
    for d, p in enumerate(PARAM_DEFS):
        axis_tokens[p['key']] = _sorted_axis_tokens(p['key'], {tok[d] for tok in files})
    return source, files, axis_tokens, skipped, n_from_hdf5, phys_by_path


def scan_directory(directory, recursive=False):
    """Scan ``directory`` for per-model HDF5 files and build the main grid."""
    global _grid, _field_map, _profile_cache
    global _heat_components, _cool_components, _cr_heat_idx
    global _model_config_summary

    source, files, axis_tokens, skipped, n_from_hdf5, phys_by_path = _scan_files(
        directory, recursive)
    field_map, species, heat_comp, cool_comp, cr_idx = \
        build_structure(next(iter(files.values())))

    _profile_cache = {}
    _scalar_cache = {}
    _heatcool_crir_cache.clear()
    _col_dens_profile_cache.clear()
    _field_map = field_map
    _heat_components = heat_comp
    _cool_components = cool_comp
    _cr_heat_idx = cr_idx
    _model_config_summary = mc.scan_model_configs(_config_scan_root(source))
    _grid = dict(
        directory=source,
        files=files,
        axis_tokens=axis_tokens,
        species=species,
        species_idx={s: i for i, s in enumerate(species)},
        n_files=len(files),
        n_skipped=skipped,
        n_from_hdf5=n_from_hdf5,
        phys_by_path=phys_by_path,
        simline_only=False,
        chem_only=False,
        has_hdf5=True,
    )
    configure_slice_planes(axis_tokens)
    return _grid


def scan_overlay(directory, recursive=False):
    """Scan a second (e.g. attenuated) grid to be overplotted on the figures.

    Matching to the main grid is done on every axis *except* attenuation, so an
    attenuated counterpart (different AA tag) lines up with each main model.
    """
    global _overlay
    source, files, axis_tokens, skipped, n_from_hdf5, phys_by_path = _scan_files(
        directory, recursive)
    # Index ignoring the attenuation token (last entry in PARAM order).
    by_non_atten = {}
    for tokens, path in files.items():
        by_non_atten.setdefault(tokens[:N_PARAMS - 1], path)
    _overlay = dict(
        directory=source,
        files=files,
        by_non_atten=by_non_atten,
        axis_tokens=axis_tokens,
        n_files=len(files),
        n_skipped=skipped,
        n_from_hdf5=n_from_hdf5,
        phys_by_path=phys_by_path,
    )
    return _overlay


def clear_overlay():
    global _overlay
    _overlay = {}


def _tokens_from_values(values):
    """Return the full token tuple for the current slider indices, or None."""
    try:
        return tuple(
            _grid['axis_tokens'][PARAM_DEFS[d]['key']][int(values[d])]
            for d in range(N_PARAMS)
        )
    except (IndexError, KeyError, TypeError):
        return None


def current_file(values):
    """Map a list of slider indices (PARAM order) to a main-grid filepath."""
    if not _grid:
        return None
    tokens = _tokens_from_values(values)
    if tokens is None:
        return None
    return _grid['files'].get(tokens)


def overlay_file(values):
    """Find the overlay-grid file matching the current slider selection."""
    if not _overlay:
        return None
    tokens = _tokens_from_values(values)
    if tokens is None:
        return None
    # Prefer an exact match (same attenuation), else match all other axes.
    return _overlay['files'].get(tokens) \
        or _overlay['by_non_atten'].get(tokens[:N_PARAMS - 1])


def _reaction_label(raw):
    """Turn a metadata label 'Formation rate CO | A + B > C + D' into 'A + B \u2192 C + D'."""
    txt = raw.split('|')[-1].strip() if '|' in raw else raw.strip()
    return txt.replace(' > ', ' \u2192 ')


def build_chem_structure(sample_file):
    """Read the reaction labels (per species, formation & destruction) once.

    Returns
    -------
    species : list[str]   species that have a chemistry group
    labels  : dict        species -> {'formation': [...], 'destruction': [...]}
    """
    labels = {}
    with h5py.File(sample_file, 'r') as hf:
        species = list(hf[CHEM_GROUP].keys()) if CHEM_GROUP in hf else []
        md = hf[METADATA_PATH][:]
    # Group metadata rows by their dataset name (col 1), keeping (idx, label).
    by_dset = {}
    for row in md:
        dset = _dec(row[1])
        if not (dset.startswith('Formation rates ') or dset.startswith('Destruction rates ')):
            continue
        try:
            idx = int(float(_dec(row[2])))
        except (ValueError, TypeError):
            continue
        by_dset.setdefault(dset, []).append((idx, _reaction_label(_dec(row[4]))))
    for sp in species:
        form = sorted(by_dset.get(f'Formation rates {sp}', []))
        dest = sorted(by_dset.get(f'Destruction rates {sp}', []))
        labels[sp] = {
            'formation':   [lab for _, lab in form],
            'destruction': [lab for _, lab in dest],
        }
    return species, labels


def scan_chem(directory, recursive=False):
    """Scan a chemistry (reaction-rate) grid directory."""
    global _chem, _reaction_cache
    source, files, axis_tokens, skipped, n_from_hdf5, phys_by_path = _scan_files(
        directory, recursive)
    species, labels = build_chem_structure(next(iter(files.values())))
    by_non_atten = {}
    for tokens, path in files.items():
        by_non_atten.setdefault(tokens[:N_PARAMS - 1], path)
    _reaction_cache = {}
    clear_chem_overlay()
    _chem = dict(
        directory=source,
        files=files,
        by_non_atten=by_non_atten,
        axis_tokens=axis_tokens,
        species=species,
        labels=labels,
        n_files=len(files),
        n_skipped=skipped,
        n_from_hdf5=n_from_hdf5,
        phys_by_path=phys_by_path,
    )
    return _chem


def _chem_phys_signature(phys):
    """Rounded (n_H, M, chi, Z, zeta) tuple for cross-grid chemistry matching."""
    if not phys:
        return None
    sig = []
    for key in ('density', 'mass', 'fuv', 'metal', 'crir'):
        val = phys.get(key)
        if val is None:
            return None
        try:
            sig.append(round(float(val), 6))
        except (TypeError, ValueError):
            return None
    return tuple(sig)


def _chem_path_for_phys(store, phys):
    """Chemistry file whose embedded physical parameters match ``phys``."""
    sig = _chem_phys_signature(phys)
    if sig is None or not store:
        return None
    for path in set((store.get('files') or {}).values()):
        pphys = (store.get('phys_by_path') or {}).get(path)
        if _chem_phys_signature(pphys) == sig:
            return path
    return None


def _build_chem_overlay_map(primary, overlay):
    """Map each primary chemistry token tuple to the best overlay HDF5 path."""
    if not primary or not overlay:
        return {}
    by_phys = {}
    for opath in set((overlay.get('files') or {}).values()):
        sig = _chem_phys_signature((overlay.get('phys_by_path') or {}).get(opath))
        if sig is not None:
            by_phys.setdefault(sig, []).append(opath)
    mapping = {}
    for ptok, ppath in (primary.get('files') or {}).items():
        path = _chem_path_for_store(overlay, ptok, allow_by_non_atten=True)
        psig = _chem_phys_signature((primary.get('phys_by_path') or {}).get(ppath))
        if psig is not None and psig in by_phys:
            candidates = [p for p in by_phys[psig] if not _same_data_path(p, ppath)]
            path = candidates[0] if candidates else by_phys[psig][0]
        mapping[tuple(ptok)] = path
    return mapping


def _refresh_chem_overlay_map():
    """Rebuild primary-token -> overlay-path map after chemistry loads."""
    if not _chem or not _chem_overlay:
        return
    _chem_overlay['for_primary'] = _build_chem_overlay_map(_chem, _chem_overlay)


def clear_chem():
    global _chem, _reaction_cache
    _chem = {}
    _reaction_cache = {}
    clear_chem_overlay()


def scan_chem_overlay(directory, recursive=False):
    """Scan a second chemistry grid to be overplotted (e.g. attenuated models)."""
    global _chem_overlay, _reaction_cache
    source, files, axis_tokens, skipped, n_from_hdf5, phys_by_path = _scan_files(
        directory, recursive)
    species, labels = build_chem_structure(next(iter(files.values())))
    by_non_atten = {}
    for tokens, path in files.items():
        by_non_atten.setdefault(tokens[:N_PARAMS - 1], path)
    _reaction_cache = {}
    _chem_overlay = dict(
        directory=source,
        files=files,
        by_non_atten=by_non_atten,
        axis_tokens=axis_tokens,
        species=species,
        labels=labels,
        n_files=len(files),
        n_skipped=skipped,
        n_from_hdf5=n_from_hdf5,
        phys_by_path=phys_by_path,
        for_primary={},
    )
    _refresh_chem_overlay_map()
    return _chem_overlay


def clear_chem_overlay():
    global _chem_overlay, _reaction_cache
    _chem_overlay = {}
    _reaction_cache = {}


def parse_smli_filename(fname):
    """Parse ``{jtemp,jerg,tau}_<tag>_DD_MM_FF_ZZ_<species>.smli`` (or ``.smlc``).

    Returns (quantity_key, full_tokens, file_key, species) or None.
    ``file_key`` is the raw token run stored on disk (4–6 integers).
    """
    base = os.path.basename(fname)
    if base.endswith('.smli'):
        ext = '.smli'
    elif base.endswith('.smlc'):
        ext = '.smlc'
    else:
        return None

    idef = core = None
    for prefix, key in (('jtemp_', 'jtemp'), ('jerg_', 'jerg'), ('tau_', 'tau')):
        if base.startswith(prefix):
            idef, core = key, base[len(prefix):-len(ext)]
            break
    if idef is None:
        return None

    parts = core.split('_')
    tag_idx = gn.find_model_tag_index(parts)
    if tag_idx is None:
        return None
    n_tok = gn.param_token_count(parts, tag_idx)
    if n_tok < gn.MIN_GRID_PARAMS:
        return None
    start = tag_idx + 1
    try:
        raw = tuple(int(parts[start + i]) for i in range(n_tok))
    except (ValueError, IndexError):
        return None
    full = gn.expand_partial_tokens(raw)
    if full is None:
        return None

    sp_start = tag_idx + 1 + n_tok
    if sp_start >= len(parts):
        return None
    species = '_'.join(parts[sp_start:])
    file_key = gn.simline_file_key(full, n_tok)
    return idef, full, file_key, species


def read_smli_file(path):
    """Read one SIMLINE .smli file into a list of transition dicts."""
    if path in _smli_cache:
        return _smli_cache[path]
    rows = []
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            transition = parts[0]
            if len(parts) >= 3:
                try:
                    freq = float(parts[1])
                    intensity = float(parts[2])
                except ValueError:
                    continue
            else:
                freq = np.nan
                try:
                    intensity = float(parts[1])
                except ValueError:
                    continue
            rows.append(dict(transition=transition, frequency=freq, intensity=intensity))
    _smli_cache[path] = rows
    return rows


def _smli_transition_label(raw):
    """Turn ``1--0`` into ``1-0`` for display."""
    return str(raw).replace('--', '-')


def scan_simline(directory, recursive=False):
    """Scan a SIMLINE output directory for per-model ``.smli`` intensity files."""
    global _simline, _smli_cache
    _smli_cache = {}
    _simline = _scan_simline_directory(directory, recursive, build_by_non_atten=False)
    return _simline


def scan_simline_overlay(directory, recursive=False):
    """Scan an attenuated SIMLINE directory (matched to the main grid on non-atten axes)."""
    global _simline_overlay, _smli_cache
    _smli_cache = {}
    _simline_overlay = _scan_simline_directory(directory, recursive, build_by_non_atten=True)
    return _simline_overlay


def _prefer_smli_path(existing, candidate):
    """Prefer ``.smli`` over ``.smlc`` for the same model / species / idef key.

    ``jtemp`` / ``jerg`` / ``tau`` select intensity units via the filename prefix;
    when both extensions exist for one model, always keep the ``.smli`` table.
    """
    if not existing:
        return candidate
    cand_smli = candidate.lower().endswith('.smli')
    exist_smli = existing.lower().endswith('.smli')
    if cand_smli and not exist_smli:
        return candidate
    if exist_smli and not cand_smli:
        return existing
    return candidate


def _scan_simline_directory(directory, recursive, build_by_non_atten=False):
    directory = os.path.expanduser((directory or '').strip())
    if not directory or not os.path.isdir(directory):
        raise FileNotFoundError(f'Directory not found: {directory!r}')

    pattern_smli = os.path.join(directory, '**', '*.smli') if recursive \
        else os.path.join(directory, '*.smli')
    pattern_smlc = os.path.join(directory, '**', '*.smlc') if recursive \
        else os.path.join(directory, '*.smlc')
    paths = sorted(set(glob.glob(pattern_smli, recursive=recursive))
                 | set(glob.glob(pattern_smlc, recursive=recursive)))
    if not paths:
        raise FileNotFoundError(f'No .smli / .smlc files found in {directory!r}')

    files = {}
    by_non_atten = {}
    skipped = 0
    species_set = set()
    transitions = {}
    transition_paths = {}
    filename_token_count = None
    for path in paths:
        parsed = parse_smli_filename(path)
        if parsed is None:
            skipped += 1
            continue
        idef, _full, file_key, species = parsed
        if filename_token_count is None:
            filename_token_count = len(file_key)
        key = (file_key, species, idef)
        chosen = _prefer_smli_path(files.get(key), path)
        files[key] = chosen
        if build_by_non_atten:
            bn_key = (file_key[:N_PARAMS - 1], species, idef)
            by_non_atten[bn_key] = _prefer_smli_path(by_non_atten.get(bn_key), path)
        species_set.add(species)
        tkey = (species, idef)
        prev_tpath = transition_paths.get(tkey)
        chosen_tpath = _prefer_smli_path(prev_tpath, path)
        if chosen_tpath != prev_tpath:
            transition_paths[tkey] = chosen_tpath
            rows = read_smli_file(chosen_tpath)
            transitions[tkey] = [
                dict(
                    idx=i,
                    transition=r['transition'],
                    frequency=r['frequency'],
                    label=_smli_transition_option_label(i, r['transition'], r['frequency']),
                    value=str(i),
                )
                for i, r in enumerate(rows)
            ]

    if not files:
        raise ValueError('Found .smli / .smlc files but none matched the expected '
                         'jtemp_/jerg_/tau_<tag>_DD_MM_FF_[ZZ]_<species> naming convention.')

    out = dict(
        directory=directory,
        files=files,
        filename_token_count=filename_token_count or gn.N_GRID_PARAMS,
        species=sorted(species_set),
        transitions=transitions,
        n_files=len(files),
        n_skipped=skipped,
    )
    pv_index, pv_transitions, pv_tau_index, pv_tau_transitions = ss.scan_pv_index(
        directory, recursive=recursive)
    out['pv_index'] = pv_index
    out['pv_transitions'] = pv_transitions
    out['pv_tau_index'] = pv_tau_index
    out['pv_tau_transitions'] = pv_tau_transitions
    out['n_pv_files'] = len(pv_index) + len(pv_tau_index)
    if build_by_non_atten:
        out['by_non_atten'] = by_non_atten
    return out


def clear_simline():
    global _simline, _smli_cache
    _simline = {}
    _smli_cache = {}


def clear_simline_overlay():
    global _simline_overlay, _smli_cache
    _simline_overlay = {}
    _smli_cache = {}


def _grid_has_hdf5():
    """True when the main grid comes from an HDF5 structure scan (not chemistry/SIMLINE)."""
    return bool(_grid and _grid.get('has_hdf5'))


def _grid_is_virtual():
    """True when parameter axes come from SIMLINE or chemistry, not an HDF5 grid."""
    return bool(_grid and (_grid.get('simline_only') or _grid.get('chem_only')))


def bootstrap_grid_from_chem():
    """When no HDF5 grid is loaded, build a virtual parameter index from chemistry.

    Same pattern as ``bootstrap_grid_from_simline``: ``_grid`` holds only
    ``axis_tokens`` for the sliders; chemistry HDF5 paths live in ``_chem``.
    Overlay chemistry stays in ``_chem_overlay`` (matched like HDF5 overlays).
    """
    global _grid
    if not _chem or not _chem.get('files'):
        return False
    if _grid_has_hdf5():
        return False
    if _grid and not _grid.get('chem_only'):
        return False

    axis_tokens = order_axis_tokens_physically(dict(_chem['axis_tokens']))
    species = list(_chem.get('species', []))
    _grid = dict(
        directory=_chem['directory'],
        files={},
        axis_tokens=axis_tokens,
        species=species,
        species_idx={s: i for i, s in enumerate(species)},
        n_files=len(_chem['files']),
        n_skipped=_chem.get('n_skipped', 0),
        n_from_hdf5=_chem.get('n_from_hdf5', 0),
        phys_by_path=dict(_chem.get('phys_by_path') or {}),
        chem_only=True,
        simline_only=False,
        has_hdf5=False,
    )
    configure_slice_planes(axis_tokens)
    return True


def clear_chem_only_grid():
    """Drop the virtual parameter index bootstrapped from chemistry only."""
    global _grid, _SLICE_PLANES_ACTIVE
    if _grid and _grid.get('chem_only') and not _grid_has_hdf5():
        _grid = {}
        _SLICE_PLANES_ACTIVE = None


def bootstrap_grid_from_simline():
    """When no HDF5 grid is loaded, build parameter axes from SIMLINE filenames."""
    global _grid
    if not _simline or not _simline.get('files'):
        return False
    if _grid_has_hdf5() or _grid:
        return False

    token_tuples = sorted({
        gn.expand_partial_tokens(key[0])
        for key in _simline['files']
        if gn.expand_partial_tokens(key[0]) is not None
    })
    axis_tokens = order_axis_tokens_physically(gn.axis_tokens_from_tuples(token_tuples))
    token_cores = {}
    for key, path in _simline['files'].items():
        full = gn.expand_partial_tokens(key[0])
        if full and full not in token_cores:
            core = gn.model_core_from_simline_path(path)
            if core:
                token_cores[full] = core

    species = list(_simline.get('species', []))
    _grid = dict(
        directory=_simline['directory'],
        files={},
        axis_tokens=axis_tokens,
        species=species,
        species_idx={s: i for i, s in enumerate(species)},
        n_files=len(token_tuples),
        n_skipped=_simline.get('n_skipped', 0),
        n_from_hdf5=0,
        phys_by_path={},
        simline_only=True,
        chem_only=False,
        has_hdf5=False,
        token_cores=token_cores,
        filename_token_count=_simline.get('filename_token_count', gn.N_GRID_PARAMS),
    )
    configure_slice_planes(axis_tokens)
    return True


def clear_simline_only_grid():
    """Drop the virtual parameter index bootstrapped from SIMLINE only."""
    global _grid, _SLICE_PLANES_ACTIVE
    if _grid and _grid.get('simline_only') and not _grid_has_hdf5():
        _grid = {}
        _SLICE_PLANES_ACTIVE = None


def clear_main_grid():
    """Clear the main HDF5 / virtual grid and related caches."""
    global _grid, _field_map, _profile_cache, _scalar_cache, _SLICE_PLANES_ACTIVE
    global _heat_components, _cool_components, _cr_heat_idx, _model_config_summary
    _grid = {}
    _field_map = {}
    _profile_cache = {}
    _scalar_cache = {}
    _heat_components = []
    _cool_components = []
    _cr_heat_idx = None
    _model_config_summary = None
    _SLICE_PLANES_ACTIVE = None


def _simline_sample_path(tok_tuple):
    if not _simline or not tok_tuple:
        return None
    n = _simline.get('filename_token_count', gn.N_GRID_PARAMS)
    file_key = gn.simline_file_key(tuple(tok_tuple), n)
    for file_key_stored, path in _simline['files'].items():
        if file_key_stored[0] == file_key:
            return path
    return None


def _chem_sample_path(tok_tuple):
    """Chemistry HDF5 path for a parameter token tuple (chemistry-only mode)."""
    return _chem_path_for_store(_chem, tok_tuple)


def _chem_path_for_store(store, tokens, *, allow_by_non_atten=False):
    """Resolve a chemistry HDF5 path from a token tuple.

    Primary chemistry uses exact matches only (like ``current_file`` on the HDF5
    grid).  Overlay chemistry also falls back to the non-attenuation key
    (like ``overlay_file``).
    """
    if not store or tokens is None:
        return None
    t = tuple(tokens)
    path = (store.get('files') or {}).get(t)
    if path or not allow_by_non_atten:
        return path
    return (store.get('by_non_atten') or {}).get(t[:N_PARAMS - 1])


def smli_file(tokens, species, idef, overlay=False):
    """Return the .smli path for one model point, or None."""
    store = _simline_overlay if overlay else _simline
    if not store:
        return None
    n = store.get('filename_token_count', gn.N_GRID_PARAMS)
    file_key = gn.simline_file_key(tuple(tokens), n)
    path = store['files'].get((file_key, species, idef))
    if path:
        return path
    if overlay:
        return store.get('by_non_atten', {}).get((file_key[:N_PARAMS - 1], species, idef))
    return None


def model_core_from_token_tuple(tok_tuple):
    """Model folder stem ``Model…_DD_MM_…`` for a full parameter token tuple."""
    if not tok_tuple:
        return None
    tok_tuple = tuple(tok_tuple)
    if _grid and _grid.get('token_cores'):
        core = _grid['token_cores'].get(tok_tuple)
        if core:
            return core
    if _grid and _grid.get('files'):
        path = _grid['files'].get(tok_tuple)
        if path:
            return os.path.splitext(os.path.basename(path))[0]
    spath = _simline_sample_path(tok_tuple)
    if spath:
        return gn.model_core_from_simline_path(spath)
    return None


def model_core_from_tokens(slider_values):
    """Model folder stem for the current slider selection."""
    return model_core_from_token_tuple(_tokens_from_values(slider_values))


def pv_fits_path(tokens, species, transition, quantity='intensity'):
    """Return the PV FITS path for one model point / species / transition."""
    if not _simline or tokens is None or not species or transition is None:
        return None
    key = (tuple(tokens), species, str(transition))
    store_key = 'pv_tau_index' if quantity == 'tau' else 'pv_index'
    path = _simline.get(store_key, {}).get(key)
    if path and os.path.isfile(path):
        return path
    model_core = model_core_from_token_tuple(tokens)
    if not model_core:
        return None
    candidate = ss.expected_pv_fits_path(
        _simline['directory'], model_core, species, str(transition),
        quantity=quantity)
    return candidate if os.path.isfile(candidate) else None


def _sp_transition_options(tokens, species, quantity='intensity'):
    """Dropdown options for PV transitions at the current model point."""
    if not _simline or tokens is None or not species:
        return []
    trans_key = 'pv_tau_transitions' if quantity == 'tau' else 'pv_transitions'
    raw_list = _simline.get(trans_key, {}).get((tuple(tokens), species), [])
    if not raw_list:
        return []
    idef = 'tau' if quantity == 'tau' else SIMLINE_DEFAULT_IDEF
    smli_rows = {
        r['transition']: r
        for r in (_simline['transitions'].get((species, idef)) or [])
    }

    def _sort_key(tr):
        row = smli_rows.get(tr)
        if row and np.isfinite(row.get('frequency', np.nan)):
            return float(row['frequency'])
        return str(tr)

    opts = []
    for tr in sorted(raw_list, key=_sort_key):
        row = smli_rows.get(tr)
        label = row['label'] if row else _smli_transition_label(tr)
        opts.append({'label': label, 'value': tr})
    return opts


def get_smli_intensity(tokens, species, idef, transition_idx, overlay=False):
    """Intensity for one grid point and transition row index."""
    path = smli_file(tokens, species, idef, overlay=overlay)
    if not path:
        return np.nan
    rows = read_smli_file(path)
    try:
        return float(rows[int(transition_idx)]['intensity'])
    except (IndexError, TypeError, ValueError):
        return np.nan


def _smli_transition_option_label(idx, transition, frequency):
    label = _smli_transition_label(transition)
    if np.isfinite(frequency):
        return f'{label}  ({frequency:.4g} GHz)'
    return label


def _intensity_unit_label(idef):
    if idef == 'jerg':
        return 'erg s\u207B\u00B9 cm\u207B\u00B2 Hz\u207B\u00B9'
    if idef == 'tau':
        return '\u03c4'
    return 'K km/s'


def _quantity_axis_label(idef):
    """Y-axis / colorbar label for SIMLINE .smli quantities."""
    if idef == 'tau':
        return '\u03c4'
    unit = _intensity_unit_label(idef)
    return f'I [{unit}]'


def _simline_transition_options(species, idef):
    if not _simline or not species:
        return []
    return list(_simline['transitions'].get((species, idef), []))


def _default_simline_transition(species, idef):
    opts = _simline_transition_options(species, idef)
    return opts[0]['value'] if opts else None


def _grid_decode_param(key, token):
    return _physical_param_value(key, token)


def _grid_param_logscale(key):
    return PARAM_DEFS[_PARAM_IDX[key]]['logscale']


def _simline_line_key_options(idef):
    """Dropdown options for spectroscopic line keys in loaded SIMLINE."""
    keys = gf.list_simline_line_keys(_simline, idef or SIMLINE_DEFAULT_IDEF)
    return [{'label': k, 'value': k} for k in keys]


def run_map_fit_job(maps_json, errors_json, output_dir, idef,
                    target_nz, target_ny, target_nx, method,
                    chi2_i, chi2_j, create_ratios, create_model_ratios,
                    chi2_analysis, uncertainty_maps):
    """Build 3-D intensity cubes and run KoSens3D map fit."""
    global _fit_results
    if not _grid:
        raise RuntimeError('Load a main grid directory on the Load tab first.')
    if not _simline:
        raise RuntimeError('Load a SIMLINE directory on the Load tab first.')

    maps = gf.parse_json_dict(maps_json)
    if not maps:
        raise ValueError('Provide at least one observed FITS map (JSON object).')
    for line, path in maps.items():
        path = os.path.expanduser(str(path).strip())
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Observed map not found for {line!r}: {path}')
        maps[line] = path

    errors = gf.parse_json_dict(errors_json)
    idef = idef or SIMLINE_DEFAULT_IDEF
    line_names = [k for k in maps if '/' not in k]
    if not line_names:
        raise ValueError('Need at least one individual line (not a ratio) in maps JSON.')

    ref_path = maps[line_names[0]]
    fits_header = gf.read_fits_header(ref_path)

    shape = (
        _parse_interp_resolution(target_nz, gf.DEFAULT_TARGET_SHAPE_3D[0]),
        _parse_interp_resolution(target_ny, gf.DEFAULT_TARGET_SHAPE_3D[1]),
        _parse_interp_resolution(target_nx, gf.DEFAULT_TARGET_SHAPE_3D[2]),
    )
    grids_3d = gf.build_grids_3d_dict(
        _grid['axis_tokens'],
        _grid_decode_param,
        _grid_param_logscale,
        get_smli_intensity,
        _simline,
        idef,
        line_names=line_names,
        target_shape=shape,
        method=method or DEFAULT_INTERP_METHOD,
    )

    out_dir = os.path.expanduser((output_dir or '').strip()) or os.path.join(
        os.getcwd(), 'map_fit_output')
    chi2_pixel = None
    if chi2_i is not None and chi2_j is not None:
        try:
            chi2_pixel = (int(chi2_i), int(chi2_j))
        except (TypeError, ValueError):
            chi2_pixel = None

    axis_cfg = grids_3d.get('_fit_axis_config') or gf.fit_axis_config(_grid['axis_tokens'])

    _fit_results = gf.run_map_fit(
        observed_fits_files=maps,
        obs_errors=errors,
        grid_dicts_3d=grids_3d,
        output_dir=out_dir,
        target_units=gf.target_units_for_idef(idef),
        chi2_pixel=chi2_pixel,
        create_ratios=bool(create_ratios and 'ratios' in create_ratios),
        create_model_ratios=bool(create_model_ratios and 'model_ratios' in create_model_ratios),
        create_chi2_analysis=bool(chi2_analysis and 'chi2' in chi2_analysis),
        create_uncertainty_maps=bool(uncertainty_maps and 'unc' in uncertainty_maps),
        axis_config=axis_cfg,
    )
    _fit_results['output_dir'] = out_dir
    _fit_results['grid_shape'] = shape
    _fit_results['lines_fitted'] = line_names
    _fit_results['fits_header'] = fits_header
    _fit_results['reference_fits_path'] = ref_path
    _fit_results['fit_axes_label'] = (
        f"{axis_cfg['axis_x']} × {axis_cfg['axis_y']} × {axis_cfg['axis_z']}"
    )
    return _fit_results


def chem_file(values):
    """Find the primary chemistry HDF5 for the current slider selection."""
    if not _chem:
        return None
    tokens = _tokens_from_values(values)
    if tokens is None:
        return None
    path = _chem_path_for_store(_chem, tokens)
    if path or not _grid_has_hdf5():
        return path
    struct = current_file(values)
    if not struct:
        return None
    phys = (_grid.get('phys_by_path') or {}).get(struct)
    return _chem_path_for_phys(_chem, phys)


def chem_overlay_file(values):
    """Find the overlay chemistry file matching the current slider selection."""
    if not _chem_overlay or not _chem_overlay.get('files'):
        return None
    primary = chem_file(values)
    if not primary:
        return None
    ptok = _tokens_for_path(_chem, primary)
    if ptok is None:
        tokens = _tokens_from_values(values)
        ptok = tuple(tokens) if tokens else None
    path = None
    if ptok is not None:
        path = (_chem_overlay.get('for_primary') or {}).get(ptok)
    if not path and ptok is not None:
        path = _chem_path_for_store(_chem_overlay, ptok, allow_by_non_atten=True)
    if not path:
        struct = current_file(values) if _grid_has_hdf5() else None
        if struct:
            phys = (_grid.get('phys_by_path') or {}).get(struct)
            path = _chem_path_for_phys(_chem_overlay, phys)
        if not path:
            pphys = (_chem.get('phys_by_path') or {}).get(primary)
            path = _chem_path_for_phys(_chem_overlay, pphys)
    if path and _same_data_path(path, primary):
        return None
    return path


def _chem_labels(species, mode):
    """Reaction labels from the primary or overlay chemistry store."""
    for store in (_chem, _chem_overlay):
        labels = (store or {}).get('labels', {}).get(species, {}).get(mode, [])
        if labels:
            return labels
    return []


def _chem_labels_for_file(filepath, species, mode):
    """Reaction labels for a specific chemistry HDF5 path."""
    for store in (_chem, _chem_overlay):
        if not store:
            continue
        paths = set((store.get('files') or {}).values())
        paths.update((store.get('by_non_atten') or {}).values())
        if filepath in paths:
            return store.get('labels', {}).get(species, {}).get(mode, [])
    return _chem_labels(species, mode)


# --- Profile reading ----------------------------------------------------------

def _read_field(hf, key):
    """Read a full depth profile for a metadata field key."""
    if key not in _field_map:
        return None
    path, idx = _field_map[key]
    arr = np.asarray(hf[path][:], dtype=float)
    if arr.ndim == 1:
        return arr
    return arr[:, idx]


def _read_optional(hf, path):
    """Read a dataset that may be absent; return None on failure."""
    try:
        return np.asarray(hf[path][:], dtype=float)
    except (KeyError, TypeError):
        return None


def _relative_from_number_densities(dens, nH):
    """Convert n(species) [cm⁻³] to x = n / n_H when Relative densities are absent."""
    dens = np.asarray(dens, dtype=float)
    nH = np.asarray(nH, dtype=float)
    if dens.ndim != 2 or nH.size == 0:
        return dens
    nH_col = nH.reshape(-1, 1)
    n_rows = min(dens.shape[0], nH_col.shape[0])
    with np.errstate(divide='ignore', invalid='ignore'):
        return dens[:n_rows] / np.where(nH_col[:n_rows] > 0, nH_col[:n_rows], np.nan)


def get_model(filepath):
    """Load (and cache) the depth profiles needed for plotting one model."""
    if filepath in _profile_cache:
        return _profile_cache[filepath]
    with h5py.File(filepath, 'r') as hf:
        nH = _read_field(hf, KEY_NH)
        dens = _read_optional(hf, DENS_PATH)
        rel = _read_optional(hf, RELDENS_PATH)
        if rel is None and dens is not None and nH is not None:
            rel = _relative_from_number_densities(dens, nH)
        if rel is None:
            rel = np.zeros((0, 0))
        if dens is None:
            dens = np.zeros((0, 0))
        model = dict(
            av     = _read_field(hf, KEY_AV),
            nH     = nH,
            tgas   = _read_field(hf, KEY_TGAS),
            tdust  = _read_field(hf, KEY_TDUST),
            nelectr = _read_field(hf, KEY_NELECTR),
            cosray = _read_field(hf, KEY_COSRAY),
            nh2_profile = _read_field(hf, KEY_NH2_PROFILE),
            radius = _read_field(hf, KEY_RADIUS),
            rel    = rel,
            dens   = dens,
            heat   = _read_optional(hf, HEATING_PATH),
            cool   = _read_optional(hf, COOLING_PATH),
        )
    _profile_cache[filepath] = model
    return model


def species_abundance(model, name, yscale):
    """Relative abundance profile for a species, clipped for log scale."""
    idx = _grid['species_idx'].get(name)
    rel = model.get('rel')
    if idx is None or rel is None or rel.ndim != 2 or idx >= rel.shape[1]:
        return None
    ab = rel[:, idx].astype(float)
    if yscale == 'log':
        ab = np.where(ab > 0, ab, MIN_AB)
    return ab


def _los_profiles(radius_pc, profile):
    """Flip + pad radius/profile for KoSens line-of-sight integrals.

    Matches ``column_density_func``: reverse the depth grid, insert r=0, convert
    pc → cm, and repeat the first flipped profile sample at the centre.
    """
    radius = np.asarray(radius_pc, dtype=float)
    profile = np.asarray(profile, dtype=float)
    n = min(radius.size, profile.size)
    if n == 0:
        return None, None
    radius, profile = radius[:n], profile[:n]
    flipped_radius = radius[::-1]
    radius_cm = np.insert(flipped_radius, 0, 0.0) * PC_TO_CM
    flipped_prof = profile[::-1]
    prof_cm = np.insert(flipped_prof, 0, flipped_prof[0])
    return radius_cm, prof_cm


def _integrate_sphere(radius_pc, profile):
    """Volume integral 4 pi r^2 n(r) dr (KoSens Abundance_Calculator convention)."""
    radius_cm, prof_cm = _los_profiles(radius_pc, profile)
    if radius_cm is None:
        return np.nan
    return float(np.trapezoid(4.0 * np.pi * radius_cm ** 2 * prof_cm, radius_cm))


def _integrate_sphere_columns(radius_pc, dens):
    """Volume integrals for every species column of ``dens[n_r, n_sp]``."""
    radius = np.asarray(radius_pc, dtype=float)
    dens = np.asarray(dens, dtype=float)
    if dens.ndim == 1:
        dens = dens[:, None]
    n = min(radius.size, dens.shape[0])
    n_sp = dens.shape[1]
    if n == 0 or n_sp == 0:
        return np.full(n_sp, np.nan)
    radius = radius[:n]
    dens = dens[:n]
    radius_cm = np.empty(n + 1, dtype=float)
    radius_cm[0] = 0.0
    radius_cm[1:] = radius[::-1] * PC_TO_CM
    dens_cm = np.empty((n + 1, n_sp), dtype=float)
    flipped = dens[::-1]
    dens_cm[0] = flipped[0]
    dens_cm[1:] = flipped
    integrand = (4.0 * np.pi * radius_cm[:, None] ** 2) * dens_cm
    return np.trapezoid(integrand, radius_cm, axis=0)


def _integrated_x_for_species(model, species_names):
    """Clump-integrated X for many species from one dens array."""
    out = {sp: np.nan for sp in species_names}
    if not species_names or not _grid or model.get('radius') is None:
        return out
    dens = np.asarray(model.get('dens'), dtype=float)
    if dens.ndim != 2 or dens.size == 0:
        return out
    idx_map = _grid['species_idx']
    idx_h = idx_map.get('H')
    idx_h2 = idx_map.get('H2')
    if idx_h is None or idx_h2 is None:
        return out
    wanted = []
    for sp in species_names:
        idx = idx_map.get(sp)
        if idx is not None and 0 <= idx < dens.shape[1]:
            wanted.append(idx)
    cols = sorted(set([idx_h, idx_h2] + wanted))
    cols = [c for c in cols if 0 <= c < dens.shape[1]]
    if not cols:
        return out
    masses = _integrate_sphere_columns(model['radius'], dens[:, cols])
    mass_of = {c: masses[i] for i, c in enumerate(cols)}
    m_total = mass_of.get(idx_h, np.nan) + 2.0 * mass_of.get(idx_h2, np.nan)
    if not np.isfinite(m_total) or m_total <= 0:
        return out
    for sp in species_names:
        idx = idx_map.get(sp)
        if idx in mass_of:
            out[sp] = mass_of[idx] / m_total
    return out


def _integrate_rate_volume(radius_pc, rate, density):
    """Integral of 4 pi r^2 rate(r) n_spec(r) dr (KoSens chem contribution)."""
    radius = np.asarray(radius_pc, dtype=float)
    rate = np.asarray(rate, dtype=float)
    density = np.asarray(density, dtype=float)
    n = min(radius.size, rate.size, density.size)
    if n == 0:
        return np.nan
    radius, rate, density = radius[:n], rate[:n], density[:n]
    flipped_radius = radius[::-1]
    radius_cm = np.insert(flipped_radius, 0, 0.0) * PC_TO_CM
    rate_cm = np.insert(rate[::-1], 0, rate[::-1][0])
    dens_cm = np.insert(density[::-1], 0, density[::-1][0])
    integrand = 4.0 * np.pi * radius_cm ** 2 * rate_cm * dens_cm
    return float(np.trapezoid(integrand, radius_cm))


def _species_density_profile(model, species):
    """Number density profile n_spec(r) from the structure grid."""
    if not _grid or model.get('dens') is None:
        return None
    idx = _grid['species_idx'].get(species)
    dens = np.asarray(model['dens'], dtype=float)
    if idx is None or idx >= dens.shape[1]:
        return None
    return dens[:, idx]


def _align_depth_profiles(*profiles):
    """Trim profiles to a common depth length."""
    lengths = [p.size for p in profiles if p is not None and np.asarray(p).size]
    if not lengths:
        return profiles
    n = min(lengths)
    return tuple(
        (np.asarray(p, dtype=float)[:n] if p is not None else None)
        for p in profiles
    )


def compute_reaction_contributions(matrix, model, species):
    """KoSens ``fractional_contribution`` stats for each reaction column.

    Returns dict column_index -> {weighted_rate, fraction_pct}.
    Uses volume-weighted integration (4 pi r^2 rate n dr) when radius and species
    density are available; otherwise falls back to |rate| integrated over A_V.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return {}

    radius = model.get('radius') if model else None
    density = _species_density_profile(model, species) if model else None
    av = model.get('av') if model else None

    stats = {}
    n_reac = matrix.shape[1]

    if radius is not None and density is not None:
        radius, density = _align_depth_profiles(radius, density)[:2]
        total_rate = np.sum(matrix[:radius.size], axis=1)
        total_int = _integrate_rate_volume(radius, total_rate, density)
        mass_int = _integrate_sphere(radius, density)
        for j in range(n_reac):
            col = matrix[:radius.size, j]
            if not np.any(col != 0):
                continue
            num_int = _integrate_rate_volume(radius, col, density)
            stats[j] = dict(
                weighted_rate=(num_int / mass_int if mass_int > 0 else np.nan),
                fraction_pct=(100.0 * num_int / total_int if total_int != 0 else 0.0),
                method='volume',
            )
        return stats

    if av is not None:
        av = np.asarray(av, dtype=float)
        n = min(av.size, matrix.shape[0])
        av, matrix = av[:n], matrix[:n]
        abs_m = np.abs(matrix)
        total_int = float(np.trapezoid(np.sum(abs_m, axis=1), av))
        for j in range(n_reac):
            if not np.any(matrix[:, j] != 0):
                continue
            num_int = float(np.trapezoid(abs_m[:, j], av))
            stats[j] = dict(
                weighted_rate=num_int,
                fraction_pct=(100.0 * num_int / total_int if total_int != 0 else 0.0),
                method='av',
            )
    return stats


def integrated_rel_abundance(model, species):
    """Clump-integrated relative abundance X(species) = M_spec / M_H_total."""
    idx = _grid['species_idx'].get(species)
    if idx is None or model.get('radius') is None:
        return np.nan
    dens = np.asarray(model['dens'], dtype=float)
    idx_h = _grid['species_idx'].get('H')
    idx_h2 = _grid['species_idx'].get('H2')
    if idx_h is None or idx_h2 is None:
        return np.nan
    radius = model['radius']
    m_h = _integrate_sphere(radius, dens[:, idx_h])
    m_h2 = _integrate_sphere(radius, dens[:, idx_h2])
    m_sp = _integrate_sphere(radius, dens[:, idx])
    m_total = m_h + 2.0 * m_h2
    if not np.isfinite(m_total) or m_total <= 0:
        return np.nan
    return m_sp / m_total


def _total_column_density_from_hf(hf, species):
    """Scalar N(species) [cm⁻²] from KoSens ``cd_*`` integrated column densities."""
    key = tot_col_dens_hdf5_key(species)
    if key not in _field_map:
        return np.nan
    path, idx = _field_map[key]
    try:
        arr = np.asarray(hf[path][:], dtype=float)
    except (KeyError, TypeError, OSError, ValueError):
        return np.nan
    arr = np.squeeze(arr)
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return np.nan
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        if 0 <= idx < arr.size:
            val = float(arr[idx])
            return val if np.isfinite(val) else np.nan
        return np.nan
    if arr.ndim >= 2 and 0 <= idx < arr.shape[-1]:
        val = float(np.ravel(arr[..., idx])[0])
        return val if np.isfinite(val) else np.nan
    return np.nan


def total_column_density(filepath, species):
    """Total column density N(species) from one model HDF5 file."""
    try:
        with h5py.File(filepath, 'r') as hf:
            return _total_column_density_from_hf(hf, species)
    except OSError:
        return np.nan


def column_averaged_tgas(model):
    """n_H-weighted column-averaged gas temperature (KoSens LOS convention).

    ⟨T_gas⟩ = ∫ T_gas(r) n_H(r) ds / ∫ n_H(r) ds, with the same radius flip/pad
    as ``column_density_func`` / ``pdr_grid`` column integrals.
    """
    radius = model.get('radius')
    tgas = model.get('tgas')
    nH = model.get('nH')
    if radius is None or tgas is None or nH is None:
        return np.nan
    radius_cm, nH_cm = _los_profiles(radius, nH)
    _, tgas_cm = _los_profiles(radius, tgas)
    if radius_cm is None or tgas_cm is None:
        return np.nan
    # Profiles may differ in raw length; _los_profiles already truncates each
    # pair, so re-align to the common LOS length.
    n = min(radius_cm.size, nH_cm.size, tgas_cm.size)
    if n < 2:
        return np.nan
    radius_cm, nH_cm, tgas_cm = radius_cm[:n], nH_cm[:n], tgas_cm[:n]
    denom = float(np.trapezoid(nH_cm, radius_cm))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.trapezoid(tgas_cm * nH_cm, radius_cm) / denom)


def electron_fraction_edge(model):
    """Cloud-edge electron fraction x_e = n(e-) / n_H (protdens)."""
    ne = model.get('nelectr')
    nH = model.get('nH')
    if ne is None or nH is None:
        return np.nan
    ne0 = float(np.asarray(ne, dtype=float)[0])
    nH0 = float(np.asarray(nH, dtype=float)[0])
    if not np.isfinite(ne0) or not np.isfinite(nH0) or nH0 <= 0:
        return np.nan
    return ne0 / nH0


def _rate_total(matrix, yscale):
    """Sum of all rate components per depth point, clipped for log scale."""
    if matrix is None or matrix.size == 0:
        return None
    tot = np.nansum(matrix, axis=1).astype(float)
    if yscale == 'log':
        tot = np.where(tot > 0, tot, MIN_RATE)
    return tot


def _rate_column(matrix, idx, yscale):
    """One rate component column, clipped for log scale."""
    if matrix is None or idx is None or idx >= matrix.shape[1]:
        return None
    col = matrix[:, idx].astype(float)
    if yscale == 'log':
        col = np.where(col > 0, col, MIN_RATE)
    return col


# ======================================================================
# KoSens grid-integrated heating / cooling vs CRIR  +  abundance PDF
# (ports of grid_functions.py plot_*_vs_crir and
#  universal_plot_functions.plot_abundance_pdf, matplotlib -> Plotly)
# ======================================================================

_heatcool_crir_cache = {}   # signature -> aggregated heat/cool dict
_col_dens_profile_cache = {}  # (filepath, species) -> per-depth profile

_GAS_GRAIN_HEAT_LABEL = 'gas-grain (heating)'
_CRIR_XLABEL = 'log\u2081\u2080(\u03B6) [s\u207B\u00B9]'

# KoSens display names (grid_functions.py heating_names_dict / cooling_names_dict)
_HEAT_NAME_DICT = {
    'H2 vib. deexcitation': '\u0393(H\u2082 vib)',
    'H2 photodissociation': '\u0393(H\u2082 ph)',
    'H2 formation on grains': '\u0393(H\u2082 grain)',
    'Cosmic rays': '\u0393(CR)',
    'Photo-electric effect': '\u0393(PE)',
    'C ionization': '\u0393(C\u207A)',
    'Chemical reactions': '\u0393(chem)',
    'gas-grain': '\u0393(gas-grain)',
    'coolrate_gas_grain': '\u0393(gas-grain)',
    _GAS_GRAIN_HEAT_LABEL: '\u0393(gas-grain)',
}
_COOL_NAME_DICT = {
    'O 63mu': '\u039B(O 63)', 'O 44mu': '\u039B(O 44)', 'O 146mu': '\u039B(O 146)',
    'CO': '\u039B(CO)', 'C+': '\u039B(C\u207A)', 'C 610mu': '\u039B(C 610)',
    'C 230mu': '\u039B(C 230)', 'C 370mu': '\u039B(C 370)', 'Si+': '\u039B(Si\u207A)',
    '13CO': '\u039B(\u00B9\u00B3CO)', 'Lyman alpha': '\u039B(Ly\u03B1)', 'H2O': '\u039B(H\u2082O)',
    'gas-grain': '\u039B(gas-grain)', 'OH': '\u039B(OH)', 'O 6300 Angstrom': '\u039B(O 6300)',
    'H2 kinetic dissociation': '\u039B(H\u2082 kin)',
}
# KoSens explicit colours
_HEAT_COLORS = {
    'H2 vib. deexcitation': '#e66101',
    'H2 photodissociation': '#fdb863',
    'H2 formation on grains': '#fdd0a2',
    'Cosmic rays': '#4daf4a',
    'Photo-electric effect': '#8073ac',
    'C ionization': '#b2182b',
    'Chemical reactions': '#ef8a62',
    'coolrate_gas_grain': 'deeppink',
    _GAS_GRAIN_HEAT_LABEL: 'deeppink',
}
_COOL_COLORS = {
    'O 63mu': '#008080', 'O 44mu': '#7b68ee', 'O 146mu': '#9a6324',
    'CO': '#00ced1', 'C+': '#800000', 'C 610mu': '#228b22',
    'C 230mu': '#808000', 'C 370mu': '#cd853f', 'Si+': '#000075',
    '13CO': '#696969', 'Lyman alpha': '#ff6347', 'H2O': '#4682b4',
    'gas-grain': '#bcf60c', 'OH': '#daa520', 'O 6300 Angstrom': '#2e8b57',
    'H2 kinetic dissociation': '#dc143c',
}
_DISTINCT_PALETTE = [
    '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#469990', '#9A6324',
    '#800000', '#808000', '#000075', '#ffe119', '#ff1493',
    '#00ced1', '#dcbeff', '#fabed4', '#aaffc3', '#ff8ba7',
]


def _heat_color(label, i=0):
    return _HEAT_COLORS.get(label) or _DISTINCT_PALETTE[i % len(_DISTINCT_PALETTE)]


def _cool_color(label, i=0):
    return _COOL_COLORS.get(label) or _DISTINCT_PALETTE[i % len(_DISTINCT_PALETTE)]


def _is_gas_grain_cooling_label(label):
    """KoSens ``_is_gas_grain_cooling_label``."""
    norm = str(label).lower().replace(' ', '').replace('_', '')
    return 'gasgrain' in norm or norm == 'gas-grain'


def _gas_grain_shell_split(rate_combined, tgas, tdust):
    """Split a (flipped/prepended) gas-grain rate into cooling vs dust->gas heating.

    Port of KoSens ``_gas_grain_shell_split``: T_gas/T_dust < 1 -> dust heats gas.
    ``tgas`` / ``tdust`` are the raw per-depth profiles (flipped/prepended here).
    """
    rate = np.asarray(rate_combined, dtype=float)
    mag = np.abs(rate)
    tg = np.asarray(tgas, dtype=float) if tgas is not None else np.zeros(0)
    td = np.asarray(tdust, dtype=float) if tdust is not None else np.zeros(0)
    if tg.size and td.size:
        tg_c = np.insert(tg[::-1], 0, tg[::-1][0])
        td_c = np.insert(td[::-1], 0, td[::-1][0])
        n = min(tg_c.size, td_c.size, rate.size)
        tg_c, td_c = tg_c[:n], td_c[:n]
        mag_n, rate_n = mag[:n], rate[:n]
        td_safe = np.where(td_c > 0.0, td_c, np.nan)
        ratio = tg_c / td_safe
        dust_heats_gas = np.isfinite(ratio) & (ratio < 1.0)
        invalid = ~np.isfinite(ratio)
        cooling = np.where(dust_heats_gas, 0.0, mag_n)
        heating = np.where(dust_heats_gas, mag_n, 0.0)
        cooling = np.where(invalid, np.maximum(rate_n, 0.0), cooling)
        heating = np.where(invalid, np.maximum(-rate_n, 0.0), heating)
        return cooling, heating
    return np.maximum(rate, 0.0), np.maximum(-rate, 0.0)


def _grid_heat_cool_dataframes(values):
    """Integrate heating/cooling per model over the (density, CRIR) subgrid.

    FUV / mass / metallicity / attenuation are held at the current slider tokens;
    density and CRIR vary. Each component is volume-integrated KoSens-style
    (``4*pi*r^2 * rate * n dr`` with radius flipped, prepended, converted pc->cm),
    and the gas-grain cooling channel is split into cooling vs dust->gas heating.

    Returns ``dict(n=..., crir=..., heat={label: arr}, cool={label: arr})`` with
    one entry per model, or ``None`` when unavailable.
    """
    if not (_grid and _grid.get('has_hdf5')):
        return None
    base = _tokens_from_values(values)
    if base is None:
        return None
    idx_dens = _PARAM_IDX['density']
    idx_crir = _PARAM_IDX['crir']
    fixed = tuple((i, base[i]) for i in range(N_PARAMS)
                  if i not in (idx_dens, idx_crir))
    if fixed in _heatcool_crir_cache:
        return _heatcool_crir_cache[fixed]

    dens_def = PARAM_DEFS[idx_dens]
    crir_def = PARAM_DEFS[idx_crir]
    n_vals, crir_vals = [], []
    heat_acc = {lbl: [] for lbl, _ in _heat_components}
    heat_acc[_GAS_GRAIN_HEAT_LABEL] = []
    cool_acc = {lbl: [] for lbl, _ in _cool_components}

    for tokens, path in _grid['files'].items():
        if any(tokens[i] != tok for i, tok in fixed):
            continue
        model = get_model(path)
        if model is None or model.get('heat') is None or model.get('radius') is None:
            continue
        radius = np.asarray(model['radius'], dtype=float)
        nH = np.asarray(model['nH'], dtype=float)
        if radius.size == 0 or nH.size == 0:
            continue
        r_comb = np.insert(radius[::-1], 0, 0.0) * PC_TO_CM
        d_comb = np.insert(nH[::-1], 0, nH[::-1][0])
        m = min(r_comb.size, d_comb.size)
        r_comb, d_comb = r_comb[:m], d_comb[:m]
        shell = 4.0 * np.pi * r_comb ** 2 * d_comb

        def _integrate(col):
            c_comb = np.insert(col[::-1], 0, col[::-1][0])
            k = min(m, c_comb.size)
            return float(np.trapezoid(shell[:k] * c_comb[:k], r_comb[:k]))

        n_vals.append(float(np.log10(dens_def['decode'](tokens[idx_dens]))))
        crir_vals.append(float(np.log10(crir_def['decode'](tokens[idx_crir]))))

        heat = np.asarray(model['heat'], dtype=float)
        for lbl, idx in _heat_components:
            if idx < heat.shape[1]:
                heat_acc[lbl].append(_integrate(heat[:, idx]))
            else:
                heat_acc[lbl].append(np.nan)

        cool = model.get('cool')
        cool = np.asarray(cool, dtype=float) if cool is not None else None
        tgas, tdust = model.get('tgas'), model.get('tdust')
        gg_heat = np.nan
        for lbl, idx in _cool_components:
            if cool is None or idx >= cool.shape[1]:
                cool_acc[lbl].append(np.nan)
                continue
            col = cool[:, idx]
            if _is_gas_grain_cooling_label(lbl):
                c_comb = np.insert(col[::-1], 0, col[::-1][0])
                cool_shell, heat_shell = _gas_grain_shell_split(c_comb, tgas, tdust)
                k = min(m, cool_shell.size)
                cool_acc[lbl].append(
                    float(np.trapezoid(shell[:k] * cool_shell[:k], r_comb[:k])))
                kh = min(m, heat_shell.size)
                gg_heat = float(np.trapezoid(shell[:kh] * heat_shell[:kh], r_comb[:kh]))
            else:
                cool_acc[lbl].append(_integrate(col))
        heat_acc[_GAS_GRAIN_HEAT_LABEL].append(gg_heat)

    if not n_vals:
        return None
    result = {
        'n': np.asarray(n_vals, dtype=float),
        'crir': np.asarray(crir_vals, dtype=float),
        'heat': {lbl: np.asarray(v, dtype=float) for lbl, v in heat_acc.items()},
        'cool': {lbl: np.asarray(v, dtype=float) for lbl, v in cool_acc.items()},
    }
    gg = result['heat'].get(_GAS_GRAIN_HEAT_LABEL)
    if gg is None or not np.any(np.isfinite(gg) & (gg != 0.0)):
        result['heat'].pop(_GAS_GRAIN_HEAT_LABEL, None)
    _heatcool_crir_cache[fixed] = result
    return result


def _select_density(data, n_val):
    """Row indices for models at ``log10(n)=n_val``, ordered by ascending CRIR."""
    idx = np.where(np.isclose(data['n'], n_val))[0]
    if idx.size == 0:
        return None, None
    order = idx[np.argsort(data['crir'][idx])]
    return order, data['crir'][order]


def _apply_crir_layout(fig, theme, height, title='', legends='single'):
    """Shared theming for the multi-panel CRIR / PDF figures.

    ``legends``: ``'heat_cool'`` draws two separate legends (heating above,
    cooling below) on the right; ``'single'`` one right-side legend;
    ``'none'`` hides it. Traces opt in via ``legend='legend'`` / ``'legend2'``.
    """
    t = _theme_colors(theme)
    right = 228 if legends in ('single', 'heat_cool') else 78
    fig.update_layout(
        paper_bgcolor=t['paper_bg'], plot_bgcolor=t['plot_bg'],
        font=ps.layout_font(t['font']), height=height,
        margin=dict(l=74, r=right, t=98, b=62),
    )
    common = dict(bgcolor=t['legend_bg'], borderwidth=1,
                  bordercolor=t['legend_border'], font=ps.legend_font(t['font']),
                  orientation='v', itemsizing='constant', itemwidth=36)
    if legends == 'heat_cool':
        # Heating grows down from the top, cooling grows up from the bottom
        # so the two boxes stay separated even with many channels.
        fig.update_layout(
            showlegend=True,
            legend=dict(**common, title_text='<b>Heating</b>',
                        xref='paper', x=1.02, xanchor='left',
                        yref='paper', y=1.0, yanchor='top'),
            legend2=dict(**common, title_text='<b>Cooling</b>',
                         xref='paper', x=1.02, xanchor='left',
                         yref='paper', y=0.0, yanchor='bottom'),
        )
    elif legends == 'single':
        fig.update_layout(
            showlegend=True,
            legend=dict(**common, xref='paper', x=1.012, xanchor='left',
                        yref='paper', y=1.0, yanchor='top'),
        )
    else:
        fig.update_layout(showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor=t['grid'], zeroline=False,
                     linecolor=t['axis_line'], mirror=True,
                     tickfont=ps.tick_font(t['font']))
    fig.update_yaxes(showgrid=True, gridcolor=t['grid'], zeroline=False,
                     linecolor=t['axis_line'], mirror=True,
                     tickfont=ps.tick_font(t['font']))
    _apply_title_box(fig, title, theme, yshift=34)
    return fig


def _empty_panel_note(fig, row, col, text='No models'):
    fig.add_annotation(text=text, showarrow=False, xref='x domain', yref='y domain',
                       x=0.5, y=0.5, row=row, col=col)


def fig_heating_contributions_vs_crir(data, densities, threshold=0.01, theme='light'):
    """Normalized stacked heating (0->+1) / cooling (0->-1) contributions vs CRIR."""
    dens_list = [float(d) for d in (densities or [])]
    if not dens_list:
        return placeholder_fig('Select at least one density', theme=theme)
    t = _theme_colors(theme)
    n_cols = len(dens_list)
    titles = [f'log\u2081\u2080(n) = {d:.1f}' for d in dens_list]
    fig = make_subplots(rows=1, cols=n_cols, shared_yaxes=True,
                        subplot_titles=titles, horizontal_spacing=0.05)
    heat_cols = list(data['heat'].keys())
    cool_cols = list(data['cool'].keys())
    seen = set()
    for ci, n_val in enumerate(dens_list, start=1):
        order, x = _select_density(data, n_val)
        if order is None:
            _empty_panel_note(fig, 1, ci)
            fig.update_xaxes(title_text=_CRIR_XLABEL, row=1, col=ci)
            continue

        def _stack(cols, dct, color_fn, name_dict, sign, prefix, legend_id):
            fig.add_trace(go.Scatter(x=x, y=np.zeros_like(x, dtype=float), mode='lines',
                          line=dict(width=0), hoverinfo='skip', showlegend=False),
                          row=1, col=ci)
            total = np.nansum([dct[c][order] for c in cols], axis=0)
            total = np.where(total == 0.0, np.nan, total)
            cum = np.zeros(len(x), dtype=float)
            for i, c in enumerate(cols):
                frac = np.nan_to_num(dct[c][order] / total, nan=0.0,
                                     posinf=0.0, neginf=0.0)
                if np.nanmax(frac) < threshold:
                    continue
                cum = cum + sign * frac
                key = prefix + c
                show = key not in seen
                seen.add(key)
                fig.add_trace(go.Scatter(
                    x=x, y=cum.copy(), mode='lines',
                    line=dict(width=0.55, color='#1f1f1f'),
                    fill='tonexty', fillcolor=color_fn(c, i),
                    name=name_dict.get(c, c), legendgroup=key, showlegend=show,
                    legend=legend_id, customdata=frac,
                    hovertemplate=name_dict.get(c, c) + ': %{customdata:.2f}<extra></extra>'),
                    row=1, col=ci)

        _stack(heat_cols, data['heat'], _heat_color, _HEAT_NAME_DICT, 1.0, 'h:', 'legend')
        _stack(cool_cols, data['cool'], _cool_color, _COOL_NAME_DICT, -1.0, 'c:', 'legend2')
        fig.add_hline(y=0.0, line=dict(color=t['axis_line'], width=1.2), row=1, col=ci)
        fig.update_yaxes(range=[-1.05, 1.05], row=1, col=ci)
        fig.update_xaxes(title_text=_CRIR_XLABEL, row=1, col=ci)
    fig.update_yaxes(title_text='Normalized contribution (heat +, cool \u2212)', row=1, col=1)
    return _apply_crir_layout(fig, theme, height=640, legends='heat_cool',
                              title='Normalized heating / cooling contributions vs CRIR')


def fig_absolute_heating_cooling_vs_crir(data, densities, min_rate=1e23,
                                         min_points=2, show_totals=True, theme='light'):
    """Absolute heating (top row) and cooling (bottom row) vs CRIR, log y."""
    dens_list = [float(d) for d in (densities or [])]
    if not dens_list:
        return placeholder_fig('Select at least one density', theme=theme)
    heat_cols = list(data['heat'].keys())
    cool_cols = list(data['cool'].keys())
    heat_cmap = {c: _DISTINCT_PALETTE[i % len(_DISTINCT_PALETTE)]
                 for i, c in enumerate(heat_cols)}
    cool_cmap = {c: _DISTINCT_PALETTE[i % len(_DISTINCT_PALETTE)]
                 for i, c in enumerate(cool_cols)}
    heat_keep = [c for c in heat_cols
                 if data['heat'][c].size and np.nanmax(np.abs(data['heat'][c])) >= min_rate]
    cool_keep = [c for c in cool_cols
                 if data['cool'][c].size and np.nanmax(np.abs(data['cool'][c])) >= min_rate]
    n_cols = len(dens_list)
    titles = [f'log\u2081\u2080(n) = {d:.1f}' for d in dens_list]
    fig = make_subplots(rows=2, cols=n_cols, shared_xaxes=True, shared_yaxes='rows',
                        subplot_titles=titles, vertical_spacing=0.09,
                        horizontal_spacing=0.05)
    seen = set()
    for ci, n_val in enumerate(dens_list, start=1):
        order, x = _select_density(data, n_val)
        if order is None:
            _empty_panel_note(fig, 1, ci)
            _empty_panel_note(fig, 2, ci)
            fig.update_xaxes(title_text=_CRIR_XLABEL, row=2, col=ci)
            continue
        for c in heat_keep:
            vals = np.abs(data['heat'][c][order])
            vals = np.where(vals >= min_rate, vals, np.nan)
            if int(np.sum(np.isfinite(vals))) < min_points:
                continue
            key = 'h:' + c
            show = key not in seen
            seen.add(key)
            fig.add_trace(go.Scatter(x=x, y=vals, mode='lines+markers',
                          line=dict(color=heat_cmap[c], width=2.2),
                          marker=dict(size=5),
                          name=_HEAT_NAME_DICT.get(c, c), legendgroup=key,
                          showlegend=show, legend='legend'), row=1, col=ci)
        if show_totals:
            tot = np.abs(np.nansum([data['heat'][c][order] for c in heat_keep], axis=0))
            tot = np.where(tot >= min_rate, tot, np.nan)
            fig.add_trace(go.Scatter(x=x, y=tot, mode='lines',
                          line=dict(color='#111111', width=3),
                          name='Total heating', legendgroup='tot_h',
                          showlegend='tot_h' not in seen, legend='legend'),
                          row=1, col=ci)
            seen.add('tot_h')
        for c in cool_keep:
            vals = np.abs(data['cool'][c][order])
            vals = np.where(vals >= min_rate, vals, np.nan)
            if int(np.sum(np.isfinite(vals))) < min_points:
                continue
            key = 'c:' + c
            show = key not in seen
            seen.add(key)
            fig.add_trace(go.Scatter(x=x, y=vals, mode='lines+markers',
                          line=dict(color=cool_cmap[c], width=2.2, dash='dash'),
                          marker=dict(size=5, symbol='square'),
                          name=_COOL_NAME_DICT.get(c, c), legendgroup=key,
                          showlegend=show, legend='legend2'), row=2, col=ci)
        if show_totals:
            tot = np.abs(np.nansum([data['cool'][c][order] for c in cool_keep], axis=0))
            tot = np.where(tot >= min_rate, tot, np.nan)
            fig.add_trace(go.Scatter(x=x, y=tot, mode='lines',
                          line=dict(color='#111111', width=3),
                          name='Total cooling', legendgroup='tot_c',
                          showlegend='tot_c' not in seen, legend='legend2'),
                          row=2, col=ci)
            seen.add('tot_c')
        fig.update_xaxes(title_text=_CRIR_XLABEL, row=2, col=ci)
    for ci in range(1, n_cols + 1):
        fig.update_yaxes(type='log', row=1, col=ci)
        fig.update_yaxes(type='log', row=2, col=ci)
    fig.update_yaxes(title_text='Heating [erg s\u207B\u00B9]', row=1, col=1)
    fig.update_yaxes(title_text='Cooling [erg s\u207B\u00B9]', row=2, col=1)
    return _apply_crir_layout(fig, theme, height=980, legends='heat_cool',
                              title='Absolute heating (top) and cooling (bottom) vs CRIR')


# --- Abundance PDF ----------------------------------------------------------

def _col_dens_profile_key(species):
    """Per-depth column-density profile key ``cd_prof_<name>`` (KoSens)."""
    return 'cd_prof_' + tot_col_dens_hdf5_key(species)[len('cd_'):]


def _col_dens_profile_from_file(filepath, species):
    """Read the per-depth column-density profile for ``species`` from one file."""
    cache_key = (filepath, species)
    if cache_key in _col_dens_profile_cache:
        return _col_dens_profile_cache[cache_key]
    key = _col_dens_profile_key(species)
    prof = None
    if key in _field_map:
        path, idx = _field_map[key]
        try:
            with h5py.File(filepath, 'r') as hf:
                arr = np.asarray(hf[path][:], dtype=float)
            arr = np.squeeze(arr)
            idx = int(idx)
            if arr.ndim == 1:
                prof = arr
            elif arr.ndim == 2:
                if idx < arr.shape[1]:
                    prof = arr[:, idx]
                elif idx < arr.shape[0]:
                    prof = arr[idx, :]
        except (OSError, KeyError, TypeError, ValueError):
            prof = None
    _col_dens_profile_cache[cache_key] = prof
    return prof


def _abundance_xy(cosray, nH, y_arr):
    """(log10(cosray/nH), log10(abundance)) for valid depth points."""
    c = np.asarray(cosray, dtype=float)
    p = np.asarray(nH, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    n = min(c.size, p.size, y.size)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    c, p, y = c[:n], p[:n], y[:n]
    valid = (c > 0) & (p > 0) & np.isfinite(c) & np.isfinite(p) & (y > 0) & np.isfinite(y)
    if not np.any(valid):
        return np.zeros(0), np.zeros(0)
    return np.log10(c[valid] / p[valid]), np.log10(y[valid])


def _binned_profile(x, y, n_bins=50, statistic='mean'):
    """KoSens ``_binned_profile``: statistic of y within equal-width x bins."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return np.zeros(0), np.zeros(0)
    edges = np.linspace(x.min(), x.max(), int(n_bins) + 1)
    ids = np.digitize(x, edges)
    func = np.median if statistic == 'median' else np.mean
    bx, by = [], []
    for b in range(1, int(n_bins) + 1):
        mask = ids == b
        if np.any(mask):
            bx.append(func(x[mask]))
            by.append(func(y[mask]))
    return np.asarray(bx), np.asarray(by)


def _abundance_pdf_collect(values, species, dens_choice):
    """Collect per-model (log CRIR init, x, y) points over the grid.

    Mass / metallicity / attenuation are fixed to the current sliders; density,
    FUV and CRIR vary. Returns a list of ``(log_crir_init, x_arr, y_arr)``.
    """
    if not (_grid and _grid.get('has_hdf5')):
        return None
    base = _tokens_from_values(values)
    if base is None:
        return None
    fixed_axes = [_PARAM_IDX['mass'], _PARAM_IDX['metal'], _PARAM_IDX['atten']]
    fixed = [(i, base[i]) for i in fixed_axes]
    sp_idx = _grid.get('species_idx', {}).get(species)
    out = []
    for tokens, path in _grid['files'].items():
        if any(tokens[i] != tok for i, tok in fixed):
            continue
        model = get_model(path)
        if model is None:
            continue
        cosray, nH = model.get('cosray'), model.get('nH')
        if cosray is None or nH is None:
            continue
        if dens_choice == 'column':
            y_arr = _col_dens_profile_from_file(path, species)
            if y_arr is None:
                continue
        else:
            arr = model.get('rel' if dens_choice == 'rel' else 'dens')
            if arr is None or getattr(arr, 'ndim', 0) != 2 or sp_idx is None \
                    or sp_idx >= arr.shape[1]:
                continue
            y_arr = arr[:, sp_idx]
        x, y = _abundance_xy(cosray, nH, y_arr)
        if x.size == 0:
            continue
        cr0 = float(np.asarray(cosray, dtype=float)[0])
        log_cr = round(float(np.log10(cr0)), 1) if cr0 > 0 else None
        out.append((log_cr, x, y))
    return out


_APDF_YLABEL = {
    'rel': 'log \u03C7 (n/n_H)',
    'number': 'log n [cm\u207B\u00B3]',
    'column': 'log N [cm\u207B\u00B2]',
}
_APDF_XLABEL = 'log(\u03B6_H / n_H) [cm\u00B3 s\u207B\u00B9]'


def _add_abundance_hist2d(fig, row, col, xs, ys, n_bins, log_color, show_cbar):
    if xs.size == 0:
        _empty_panel_note(fig, row, col, 'No data')
        return
    try:
        nb = int(n_bins)
    except (TypeError, ValueError):
        nb = 100
    nb = max(5, min(nb, 400))
    hist, xe, ye = np.histogram2d(xs, ys, bins=nb)
    z = hist.T
    z = np.where(z >= 1, z, np.nan)
    if log_color:
        z = np.log10(z)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    fig.add_trace(go.Heatmap(
        x=xc, y=yc, z=z, colorscale='Viridis', showscale=show_cbar,
        colorbar=(dict(title='log\u2081\u2080 counts' if log_color else 'counts')
                  if show_cbar else None),
        hovertemplate='\u03B6/n_H=%{x:.2f}<br>abund=%{y:.2f}<br>%{z:.2f}<extra></extra>'),
        row=row, col=col)


def fig_abundance_pdf(values, species, dens_choice='rel', n_bins=100,
                      split_by_crir=True, log_color=True, overplot_average=False,
                      theme='light'):
    """2D histogram of species abundance vs ζ_H/n_H over the grid (KoSens PDF)."""
    models = _abundance_pdf_collect(values, species, dens_choice)
    if not models:
        return placeholder_fig(f'No abundance data for {species}', theme=theme)
    ylabel = _APDF_YLABEL.get(dens_choice, 'log abundance')
    if split_by_crir:
        groups = {}
        for log_cr, x, y in models:
            if log_cr is None:
                continue
            groups.setdefault(log_cr, []).append((x, y))
        crs = sorted(groups.keys())
        if not crs:
            return placeholder_fig(f'No CRIR groups for {species}', theme=theme)
        n_cols = min(3, len(crs))
        n_rows = (len(crs) + n_cols - 1) // n_cols
        titles = [f'\u03B6_H = 10^{c:.1f} s\u207B\u00B9' for c in crs]
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                            horizontal_spacing=0.09, vertical_spacing=0.14)
        for gi, cr in enumerate(crs):
            r, c = gi // n_cols + 1, gi % n_cols + 1
            xs = np.concatenate([a for a, _ in groups[cr]])
            ys = np.concatenate([b for _, b in groups[cr]])
            _add_abundance_hist2d(fig, r, c, xs, ys, n_bins, log_color,
                                  show_cbar=(gi == 0))
            if overplot_average:
                bx, by = _binned_profile(xs, ys)
                if bx.size:
                    fig.add_trace(go.Scatter(x=bx, y=by, mode='lines',
                                  line=dict(color='#ff2d2d', width=2),
                                  name='Binned average',
                                  showlegend=(gi == 0)), row=r, col=c)
            fig.update_xaxes(title_text=_APDF_XLABEL, row=r, col=c)
            fig.update_yaxes(title_text=ylabel, row=r, col=c)
        height = 380 * n_rows
    else:
        xs = np.concatenate([x for _, x, _ in models])
        ys = np.concatenate([y for _, _, y in models])
        fig = make_subplots(rows=1, cols=1)
        _add_abundance_hist2d(fig, 1, 1, xs, ys, n_bins, log_color, show_cbar=True)
        if overplot_average:
            bx, by = _binned_profile(xs, ys)
            if bx.size:
                fig.add_trace(go.Scatter(x=bx, y=by, mode='lines',
                              line=dict(color='#ff2d2d', width=2),
                              name='Binned average'), row=1, col=1)
        fig.update_xaxes(title_text=_APDF_XLABEL, row=1, col=1)
        fig.update_yaxes(title_text=ylabel, row=1, col=1)
        height = 440
    _apply_crir_layout(fig, theme, height=height, legends='none',
                       title=f'Abundance PDF \u2014 {species}')
    if overplot_average:
        t = _theme_colors(theme)
        fig.update_layout(showlegend=True, legend=dict(
            bgcolor=t['legend_bg'], borderwidth=1, bordercolor=t['legend_border'],
            font=ps.legend_font(t['font']), orientation='h',
            xref='paper', x=0.5, xanchor='center',
            yref='paper', y=1.08, yanchor='bottom'))
    return fig


def get_reaction_data(filepath, species, mode):
    """Load (and cache) the reaction-rate matrix + av for one species/mode.

    Returns dict with keys 'av' (n_depth,), 'matrix' (n_depth, n_reac),
    'labels' (list[str]); or None if unavailable.
    """
    cache_key = (filepath, species, mode)
    if cache_key in _reaction_cache:
        return _reaction_cache[cache_key]
    dset_kind = 'Formation' if mode == 'formation' else 'Destruction'
    dpath = f'{CHEM_GROUP}/{species}/{dset_kind} rates {species}'
    try:
        with h5py.File(filepath, 'r') as hf:
            if dpath not in hf:
                return None
            matrix = np.asarray(hf[dpath][:], dtype=float)
            av = np.asarray(hf[CHEM_POS][:], dtype=float)[:, 0]
    except (KeyError, OSError, IndexError):
        return None
    labels = _chem_labels_for_file(filepath, species, mode)
    # Pad/truncate labels to match matrix columns.
    if len(labels) < matrix.shape[1]:
        labels = list(labels) + [f'reaction {i}' for i in range(len(labels), matrix.shape[1])]
    data = dict(av=av, matrix=matrix, labels=labels[:matrix.shape[1]])
    _reaction_cache[cache_key] = data
    return data


def select_top_reactions(matrix, av, top_n):
    """Reproduce the top_reactions_plot selection: a reaction is kept if it is in
    the top-N (by |rate|) at *any* depth point.  Returned indices are ordered by
    integrated |rate| over the depth axis (most important first).
    """
    if matrix is None or matrix.size == 0:
        return []
    absM = np.abs(matrix)
    active = np.where(np.any(matrix != 0, axis=0))[0]
    if active.size == 0:
        return []
    top_n = max(1, int(top_n))
    selected = set()
    for i in range(matrix.shape[0]):
        row = absM[i, active]
        order = active[np.argsort(row)[::-1][:top_n]]
        selected.update(int(j) for j in order)
    importance = {j: float(np.trapezoid(absM[:, j], av)) for j in selected}
    return sorted(selected, key=lambda j: importance[j], reverse=True)


def _parse_react_ranking(value):
    allowed = {o['value'] for o in REACT_RANKING_OPTIONS}
    return value if value in allowed else DEFAULT_REACT_RANKING


def sort_reactions_by_ranking(order, stats, ranking_metric):
    """Order selected reactions by KoSens ``ranking_metric``."""

    def _key(j):
        st = stats.get(j, {})
        if ranking_metric == 'mass_weighted_rate':
            v = st.get('weighted_rate', np.nan)
        else:
            v = st.get('fraction_pct', np.nan)
        return v if np.isfinite(v) else -np.inf

    return sorted(order, key=_key, reverse=True)


def _reaction_legend_label(lab, stats_j, ranking_metric):
    """Legend / hover primary metric label."""
    wrate = stats_j.get('weighted_rate', np.nan)
    pct = stats_j.get('fraction_pct', np.nan)
    if ranking_metric == 'mass_weighted_rate' and np.isfinite(wrate):
        return f'{lab} ({wrate:.2e})'
    if np.isfinite(pct):
        return f'{lab} ({pct:.1f}%)'
    return lab


# --- Label helpers ------------------------------------------------------------

_SUP = str.maketrans('0123456789-', '\u2070\u00B9\u00B2\u00B3\u2074\u2075\u2076\u2077\u2078\u2079\u207B')


def _sup(n):
    return str(n).translate(_SUP)


def _sci_label(v):
    """Compact scientific label with unicode superscripts, e.g. 10\u00B3 or 3.0\u00D710\u00B2."""
    if v == 0 or not np.isfinite(v):
        return '0'
    exp = int(np.floor(np.log10(abs(v))))
    man = v / 10.0 ** exp
    if abs(man - 1) < 0.02:
        return f'10{_sup(exp)}'
    return f'{man:.1f}\u00D710{_sup(exp)}'


def _atten_disp(token):
    try:
        return f'{int(round(float(token))):02d}'
    except (TypeError, ValueError):
        return str(token)


def _format_phys_number(val):
    """Format an HDF5 / decoded physical value for the info bar and sliders."""
    if val is None:
        return '—'
    try:
        val = float(val)
    except (TypeError, ValueError):
        return str(val)
    if not np.isfinite(val):
        return '—'
    if val == 0:
        return '0'
    av = abs(val)
    if av >= 1e4 or av < 1e-3:
        return f'{val:.4g}'
    return f'{val:.6g}'


def _phys_lookup(filepath, key):
    """Actual HDF5 value for ``key`` if this file was tagged from metadata."""
    if not filepath:
        return None
    for store in (_grid, _overlay, _chem, _chem_overlay):
        phys = (store or {}).get('phys_by_path', {}).get(filepath)
        if phys and phys.get(key) is not None:
            return phys[key]
    return None


def _path_for_param_token(key, token, store=None):
    """A model file whose encoded token on ``key`` matches ``token``."""
    store = store or _grid
    files = (store or {}).get('files') or {}
    idx = _PARAM_IDX.get(key)
    if idx is None:
        return None
    for tokens, path in files.items():
        if len(tokens) > idx and tokens[idx] == token:
            return path
    return None


def _chem_model_path(values):
    """Chemistry HDF5 path for the current slider selection."""
    return chem_file(values) or chem_overlay_file(values)


def get_chem_model(filepath):
    """Minimal depth model from a chemistry HDF5 (A_V axis; structure when present)."""
    if not filepath:
        return None
    if filepath in _profile_cache:
        return _profile_cache[filepath]
    try:
        with h5py.File(filepath, 'r') as hf:
            av = np.asarray(hf[CHEM_POS][:], dtype=float)[:, 0]
            model = dict(av=av)
            fmap = _hdf5_wanted_field_map(hf, {'protdens', 'radius', 'distance'})
            if 'protdens' in fmap:
                nH = _hdf5_column_floats(hf, *fmap['protdens'])
                if nH is not None:
                    model['nH'] = nH
            radius_key = 'radius' if 'radius' in fmap else (
                'distance' if 'distance' in fmap else None)
            if radius_key:
                radius = _hdf5_column_floats(hf, *fmap[radius_key])
                if radius is not None and radius.size:
                    rad = np.asarray(radius, dtype=float)
                    if np.nanmax(rad) > 1e15:
                        rad = rad / PC_TO_CM
                    model['radius'] = rad
            dens = _read_optional(hf, DENS_PATH)
            if dens is not None:
                model['dens'] = dens
            rel = _read_optional(hf, RELDENS_PATH)
            if rel is not None:
                model['rel'] = rel
    except (OSError, KeyError, IndexError):
        return None
    _profile_cache[filepath] = model
    return model


def _same_data_path(path_a, path_b):
    if not path_a or not path_b:
        return False
    try:
        return os.path.samefile(path_a, path_b)
    except (OSError, ValueError):
        return os.path.normpath(path_a) == os.path.normpath(path_b)


def _physical_param_value(key, token, filepath=None, tokens=None):
    """Prefer the HDF5 number; fall back to decoding the grid token."""
    path = filepath
    if path is None and tokens is not None:
        path = (_grid.get('files') or {}).get(tuple(tokens))
        if path is None and _overlay:
            path = (_overlay.get('files') or {}).get(tuple(tokens))
    value = _phys_lookup(path, key)
    if value is not None:
        return float(value)
    if path is None:
        path = _path_for_param_token(key, token)
        if path is None and _chem:
            path = _path_for_param_token(key, token, store=_chem)
        if path is None and _chem_overlay:
            path = _path_for_param_token(key, token, store=_chem_overlay)
        value = _phys_lookup(path, key)
        if value is not None:
            return float(value)
    return PARAM_DEFS[_PARAM_IDX[key]]['decode'](token)


def _param_token_disp(sdef, token):
    """Slider / slice title text for one parameter token."""
    if sdef['key'] == 'atten':
        return _atten_disp(token)
    return _format_phys_number(_physical_param_value(sdef['key'], token))


def param_value_label(param, token):
    """Physical value formatted for a parameter (HDF5 value when available)."""
    if param['key'] == 'atten':
        return _atten_disp(token)
    val = _physical_param_value(param['key'], token)
    return _sci_label(val)


def build_marks(param, axis_tokens=None):
    """Slider marks for a parameter (decoded values), thinned to <=9 labels."""
    tokens = (axis_tokens or _grid['axis_tokens'])[param['key']]
    n = len(tokens)
    max_marks = 9
    step = max(1, (n - 1) // (max_marks - 1)) if n > 1 else 1
    indices = list(range(0, n, step))
    if (n - 1) not in indices:
        indices.append(n - 1)
    return {i: param_value_label(param, tokens[i]) for i in indices}


def format_species_html(name):
    """Render a KOSMA-tau species name with HTML sub/superscripts for plotly."""
    if not name:
        return name
    # Leading isotope number -> superscript prefix (e.g. 13CO -> <sup>13</sup>CO).
    out = []
    i = 0
    n = len(name)
    if name[0].isdigit():
        j = 0
        while j < n and name[j].isdigit():
            j += 1
        out.append(f'<sup>{name[:j]}</sup>')
        i = j
    while i < n:
        c = name[i]
        if c == '+':
            out.append('<sup>+</sup>')
        elif c == '-':
            out.append('<sup>\u2212</sup>')
        elif c.isdigit():
            out.append(f'<sub>{c}</sub>')
        else:
            out.append(c)
        i += 1
    return ''.join(out)


# --- Figure styling -----------------------------------------------------------

def _parse_plot_theme(value):
    return value if value in PLOT_THEMES else DEFAULT_PLOT_THEME


def _parse_grid_colorscale(value):
    allowed = {o['value'] for o in GRID_COLORMAP_OPTIONS}
    return value if value in allowed else DEFAULT_GRID_COLORMAP


def _theme_colors(theme='light'):
    return PLOT_THEMES.get(_parse_plot_theme(theme), PLOT_THEMES['light'])


def _base_layout(theme='light'):
    t = _theme_colors(theme)
    return dict(
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        margin=dict(l=70, r=20, t=54, b=54),
        font=ps.layout_font(t['font']),
        height=320,
        legend=dict(bgcolor=t['legend_bg'], borderwidth=1, bordercolor=t['legend_border'],
                    font=ps.legend_font(t['font'])),
    )


def _title_box(text, theme, yshift=8):
    """Bold, boxed figure title (paper annotation) shared across all figures."""
    t = _theme_colors(theme)
    dark = _parse_plot_theme(theme) == 'dark'
    return dict(
        text=f'<b>{text}</b>', xref='paper', yref='paper',
        x=0.0, xanchor='left', y=1.0, yanchor='bottom', yshift=yshift,
        showarrow=False, align='left',
        font=dict(size=ps.PLOT_TITLE_FONT_SIZE + 2, color=t['title'],
                  family=ps.PLOT_FONT_FAMILY),
        bgcolor=('#111c30' if dark else '#eef2ff'),
        bordercolor=t['accent'], borderwidth=1.2, borderpad=5,
    )


def _apply_title_box(fig, text, theme, yshift=8):
    """Add the shared bold/boxed title annotation to *fig* (no-op if blank)."""
    if text:
        fig.add_annotation(**_title_box(text, theme, yshift=yshift))
    return fig


def _axis_style(theme='light'):
    t = _theme_colors(theme)
    return dict(
        showgrid=True, gridcolor=t['grid'], gridwidth=1,
        zeroline=False, linecolor=t['axis_line'], mirror=True,
        exponentformat='e', showexponent='all',
        tickfont=ps.tick_font(t['font']),
    )


def _apply_layout(fig, title, xlabel, xtype, xrange, ylabel, ytype, theme='light',
                  uirevision_extra=''):
    t = _theme_colors(theme)
    uirev = f'{xtype}|{ytype}|{_parse_plot_theme(theme)}'
    if uirevision_extra:
        uirev = f'{uirev}|{uirevision_extra}'
    fig.update_layout(
        **_base_layout(theme),
        # Reset zoom/colorbar when log↔linear / A_V range changes (uirevision must change).
        uirevision=uirev,
        xaxis=dict(**_axis_style(theme),
                   title=dict(text=xlabel, font=ps.axis_title_font(t['font'])),
                   type=xtype, range=xrange, autorange=False),
        yaxis=dict(**_axis_style(theme),
                   title=dict(text=ylabel, font=ps.axis_title_font(t['font'])),
                   type=ytype, autorange=True),
    )
    _apply_title_box(fig, title, theme, yshift=6)


def _parse_av_range(value):
    return value if value in ('full', 'floor') else DEFAULT_AV_RANGE


def _av_display_floor(av_range):
    """Return A_V clip threshold, or None when the full HDF5 axis is requested."""
    return AV_FLOOR if _parse_av_range(av_range) == 'floor' else None


def _xvals(model, xvar, xscale, av_range=DEFAULT_AV_RANGE):
    """Return (x_array, x_label, x_type, x_range) with an explicit axis range.

    For ``type='log'`` Plotly expects ``range`` in log10 units.  For linear axes
    the range is in physical data units.

    ``av_range='floor'`` hides A_V below ``AV_FLOOR``; ``'full'`` keeps every
    HDF5 sample (log axes still omit non-positive values, which cannot be drawn).
    """
    if xvar == 'nH':
        xv = np.asarray(model['nH'], dtype=float)
        xl = 'n<sub>H</sub> (cm<sup>-3</sup>)'
    else:
        xv = np.asarray(model['av'], dtype=float)
        xl = 'A<sub>V</sub> (mag)'

    floor = _av_display_floor(av_range) if xvar == 'Av' else None
    if floor is not None:
        xv = np.where(xv >= floor, xv, np.nan)

    pos = xv[np.isfinite(xv) & (xv > 0)]
    if xscale == 'log' and pos.size:
        xv = np.where(np.isfinite(xv) & (xv > 0), xv, np.nan)
        lo_exp = float(np.floor(np.log10(pos.min())))
        if floor is not None:
            lo_exp = max(lo_exp, np.log10(floor))
        xrange = [lo_exp, float(np.ceil(np.log10(pos.max())))]
        xtype = 'log'
    else:
        finite = xv[np.isfinite(xv)]
        if finite.size:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
        else:
            lo, hi = (floor if floor is not None else 0.0), 1.0
        if floor is not None:
            lo = max(lo, floor)
        if not np.isfinite(hi) or hi <= lo:
            hi = lo * 1.02 if lo else 1.0
        xrange = [lo, hi * 1.02]
        xtype = 'linear'
    return xv, xl, xtype, xrange


def _x_array(model, xvar, xscale, av_range=DEFAULT_AV_RANGE):
    """x-array for a model, masked consistently with ``_xvals`` (for overlays)."""
    xv = np.asarray(model['nH'] if xvar == 'nH' else model['av'], dtype=float)
    floor = _av_display_floor(av_range) if xvar == 'Av' else None
    if floor is not None:
        xv = np.where(xv >= floor, xv, np.nan)
    if xscale == 'log':
        xv = np.where(np.isfinite(xv) & (xv > 0), xv, np.nan)
    return xv


def _line(fig, xv, yv, color, label, dash='solid', width=2, overlay=False):
    """Add one line trace; overlay traces are dashed, thinner and translucent."""
    if xv is None or yv is None:
        return
    fig.add_trace(go.Scatter(
        x=xv, y=yv, mode='lines',
        line=dict(color=color, width=(width - 0.4) if overlay else width,
                  dash='dash' if overlay else dash),
        opacity=0.5 if overlay else 1.0,
        name=(label + ' \u00B7 att') if overlay else label,
        legendgroup=label,
    ))


def find_h_h2_transition(model, xvar):
    """x-coordinate (Av or nH) where x(H) = x(H2) (first crossing)."""
    idx_h = _grid['species_idx'].get('H')
    idx_h2 = _grid['species_idx'].get('H2')
    rel = model.get('rel') if model else None
    if idx_h is None or idx_h2 is None or rel is None or rel.ndim != 2:
        return None
    if idx_h >= rel.shape[1] or idx_h2 >= rel.shape[1]:
        return None
    ab_h = rel[:, idx_h].astype(float)
    ab_h2 = rel[:, idx_h2].astype(float)
    xv = np.asarray(model['nH'] if xvar == 'nH' else model['av'], dtype=float)

    diff = ab_h - ab_h2
    for i in range(len(diff) - 1):
        if diff[i] >= 0 > diff[i + 1]:
            x0, x1 = float(xv[i]), float(xv[i + 1])
            d0, d1 = float(diff[i]), float(diff[i + 1])
            frac = -d0 / (d1 - d0) if abs(d1 - d0) > 1e-100 else 0.5
            if x0 > 0 and x1 > 0:
                return 10 ** (np.log10(x0) + frac * (np.log10(x1) - np.log10(x0)))
            return x0 + frac * (x1 - x0)
    return None


def add_h_h2_vline(fig, x_cross, theme='light'):
    if x_cross is None:
        return
    t = _theme_colors(theme)
    fig.add_shape(type='line', xref='x', yref='paper',
                  x0=x_cross, x1=x_cross, y0=0, y1=1,
                  line=dict(color=t['vline'], width=1, dash='dot'))


def placeholder_fig(msg='Load a grid directory or HDF5 file to begin', theme='light'):
    t = _theme_colors(theme)
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False,
                       font=dict(size=14, color=t['placeholder']),
                       xref='paper', yref='paper', x=0.5, y=0.5)
    fig.update_layout(
        paper_bgcolor=t['paper_bg'], plot_bgcolor=t['placeholder_plot'],
        margin=dict(l=20, r=20, t=20, b=20), height=320,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def _temperature(model, key, yscale):
    arr = model.get(key) if model else None
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return None
    if yscale == 'log':
        arr = np.where(arr > 0, arr, np.nan)
    return arr


def fig_tgas(model, overlay, xvar, xscale, yscale, theme='light',
             av_range=DEFAULT_AV_RANGE):
    xv, xl, xt, xr = _xvals(model, xvar, xscale, av_range=av_range)
    xvo = _x_array(overlay, xvar, xscale, av_range=av_range) if overlay else None
    fig = go.Figure()
    _line(fig, xv, _temperature(model, 'tgas', yscale), COLORS[0], 'T<sub>gas</sub>')
    _line(fig, xv, _temperature(model, 'tdust', yscale), COLORS[1], 'T<sub>dust</sub>', dash='dot')
    if overlay:
        _line(fig, xvo, _temperature(overlay, 'tgas', yscale), COLORS[0], 'T<sub>gas</sub>', overlay=True)
        _line(fig, xvo, _temperature(overlay, 'tdust', yscale), COLORS[1], 'T<sub>dust</sub>', overlay=True)
    _apply_layout(fig, 'Gas / Dust Temperature', xl, xt, xr, 'T (K)', yscale, theme=theme,
                  uirevision_extra=_parse_av_range(av_range))
    return fig


def _fig_species(model, overlay, xvar, xscale, yscale, specs, title, theme='light',
                 av_range=DEFAULT_AV_RANGE):
    xv, xl, xt, xr = _xvals(model, xvar, xscale, av_range=av_range)
    xvo = _x_array(overlay, xvar, xscale, av_range=av_range) if overlay else None
    fig = go.Figure()
    for name, color in specs:
        _line(fig, xv, species_abundance(model, name, yscale), color, format_species_html(name))
    if overlay:
        for name, color in specs:
            _line(fig, xvo, species_abundance(overlay, name, yscale), color,
                  format_species_html(name), overlay=True)
    _apply_layout(fig, title, xl, xt, xr, 'x(species)', yscale, theme=theme,
                  uirevision_extra=_parse_av_range(av_range))
    return fig


def fig_h_h2(model, overlay, xvar, xscale, yscale, theme='light',
             av_range=DEFAULT_AV_RANGE):
    return _fig_species(model, overlay, xvar, xscale, yscale,
                        [('H', COLORS[0]), ('H2', COLORS[1])], 'H / H<sub>2</sub>',
                        theme=theme, av_range=av_range)


def fig_cplus_c_co(model, overlay, xvar, xscale, yscale, theme='light',
                   av_range=DEFAULT_AV_RANGE):
    return _fig_species(model, overlay, xvar, xscale, yscale,
                        [('C+', COLORS[3]), ('C', COLORS[2]), ('CO', COLORS[0])],
                        'C<sup>+</sup> / C / CO', theme=theme, av_range=av_range)


def fig_custom(model, overlay, xvar, xscale, yscale, sel_species, theme='light',
               av_range=DEFAULT_AV_RANGE):
    specs = [(sp, COLORS[k % len(COLORS)]) for k, sp in enumerate(sel_species or [])]
    return _fig_species(model, overlay, xvar, xscale, yscale, specs, 'Custom Species',
                        theme=theme, av_range=av_range)


_RATE_YLABEL = '\u0393, \u039B (erg cm<sup>-3</sup> s<sup>-1</sup>)'

_RATE_YLABEL_heating = '\u0393 (erg cm<sup>-3</sup> s<sup>-1</sup>)'
_RATE_YLABEL_cooling = '\u039B (erg cm<sup>-3</sup> s<sup>-1</sup>)'


def fig_thermal(model, overlay, xvar, xscale, yscale, theme='light',
                av_range=DEFAULT_AV_RANGE):
    """Total heating vs total cooling, with the cosmic-ray heating highlighted."""
    xv, xl, xt, xr = _xvals(model, xvar, xscale, av_range=av_range)
    xvo = _x_array(overlay, xvar, xscale, av_range=av_range) if overlay else None
    fig = go.Figure()
    _line(fig, xv, _rate_total(model['heat'], yscale), '#d62728', '\u0393 total (heating)')
    _line(fig, xv, _rate_total(model['cool'], yscale), '#1f77b4', '\u039B total (cooling)')
    _line(fig, xv, _rate_column(model['heat'], _cr_heat_idx, yscale),
          '#ff7f0e', '\u0393 cosmic rays', dash='dash')
    if overlay:
        _line(fig, xvo, _rate_total(overlay['heat'], yscale), '#d62728', '\u0393 total (heating)', overlay=True)
        _line(fig, xvo, _rate_total(overlay['cool'], yscale), '#1f77b4', '\u039B total (cooling)', overlay=True)
        _line(fig, xvo, _rate_column(overlay['heat'], _cr_heat_idx, yscale),
              '#ff7f0e', '\u0393 cosmic rays', overlay=True)
    _apply_layout(fig, 'Heating / Cooling balance  (CR highlighted)',
                  xl, xt, xr, _RATE_YLABEL, yscale, theme=theme,
                  uirevision_extra=_parse_av_range(av_range))
    return fig


_DASH_CYCLE = ['solid', 'dot', 'dash', 'dashdot']


def _fig_rate_breakdown(model, overlay, xvar, xscale, yscale,
                        key, components, title, emphasize_idx=None, theme='light',
                        av_range=DEFAULT_AV_RANGE):
    """Plot every component of a rate matrix (heating or cooling) individually.

    With more components than colours, the dash pattern is cycled too so all
    lines stay distinguishable.
    """
    xv, xl, xt, xr = _xvals(model, xvar, xscale, av_range=av_range)
    xvo = _x_array(overlay, xvar, xscale, av_range=av_range) if overlay else None
    fig = go.Figure()
    for k, (label, idx) in enumerate(components):
        emph = (emphasize_idx is not None and idx == emphasize_idx)
        color = '#ff7f0e' if emph else COLORS[k % len(COLORS)]
        dash = 'solid' if emph else _DASH_CYCLE[(k // len(COLORS)) % len(_DASH_CYCLE)]
        _line(fig, xv, _rate_column(model[key], idx, yscale),
              color, label, dash=dash, width=3.2 if emph else 1.6)
        if overlay:
            _line(fig, xvo, _rate_column(overlay[key], idx, yscale),
                  color, label, overlay=True)
    _apply_layout(fig, title, xl, xt, xr,
                  _RATE_YLABEL_heating if key == 'heat' else _RATE_YLABEL_cooling,
                  yscale, theme=theme,
                  uirevision_extra=_parse_av_range(av_range))
    return fig


def fig_heat_breakdown(model, overlay, xvar, xscale, yscale, theme='light',
                       av_range=DEFAULT_AV_RANGE):
    """All heating-rate components; cosmic-ray heating drawn thicker."""
    return _fig_rate_breakdown(model, overlay, xvar, xscale, yscale,
                               'heat', _heat_components,
                               'Heating-rate components (all)', _cr_heat_idx,
                               theme=theme, av_range=av_range)


def fig_cool_breakdown(model, overlay, xvar, xscale, yscale, theme='light',
                       av_range=DEFAULT_AV_RANGE):
    """All cooling-rate components."""
    return _fig_rate_breakdown(model, overlay, xvar, xscale, yscale,
                               'cool', _cool_components,
                               'Cooling-rate components (all)', theme=theme,
                               av_range=av_range)


_REACT_YLABEL = 'rate (cm<sup>-3</sup> s<sup>-1</sup>)'
_REACT_TABLE_STYLE = {
    'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '12px',
    'marginTop': '8px',
}
_REACT_TABLE_CELL = {
    'padding': '4px 8px', 'borderBottom': '1px solid #e0e0e0',
    'verticalAlign': 'top',
}


def _reaction_y_range(ys_list, yscale, decade_pad=1.0):
    """Y-axis range from the currently displayed (top-N) reaction rates.

    Uses the raw min/max of those series (no filtering of small values), then
    pads by ``decade_pad`` orders of magnitude on both ends.  For log axes the
    returned range is in log10 units (Plotly convention).
    """
    chunks = []
    for y in ys_list:
        if y is None:
            continue
        vals = np.asarray(y, dtype=float)
        vals = vals[np.isfinite(vals)]
        if yscale == 'log':
            vals = vals[vals > 0]
        if vals.size:
            chunks.append(vals)
    if not chunks:
        return None
    allv = np.concatenate(chunks)
    lo, hi = float(np.min(allv)), float(np.max(allv))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if yscale == 'log':
        if lo <= 0 or hi <= 0:
            return None
        return [float(np.log10(lo)) - decade_pad,
                float(np.log10(hi)) + decade_pad]
    # Linear: one decade of headroom around the data span.
    if hi <= 0 and lo <= 0:
        return [hi * 10.0 ** decade_pad, lo / 10.0 ** decade_pad]
    if lo <= 0:
        return [lo - abs(hi) * (10.0 ** decade_pad - 1), hi * 10.0 ** decade_pad]
    return [lo / 10.0 ** decade_pad, hi * 10.0 ** decade_pad]


def _reaction_contribution_table(order, labels, stats, mode, ranking_metric):
    """HTML summary table (KoSens ``top_reactions_plot`` console output)."""
    rate_label = 'formation' if mode == 'formation' else 'destruction'
    ranking_metric = _parse_react_ranking(ranking_metric)
    if not order:
        return html.P('No reactions to rank.', style={'fontSize': '12px', 'color': '#666'})

    vol_method = next((stats[j]['method'] for j in order if j in stats), 'volume')
    if vol_method == 'volume':
        rate_hdr = 'Weighted rate (cm\u207B\u00B3 s\u207B\u00B9)'
        weight_note = ('Volume integrals \u222B 4\u03C0 r\u00B2 k n dr (KoSens chem read).')
    else:
        rate_hdr = 'Integrated |rate|'
        weight_note = ('Structure grid unavailable \u2014 |rate| integrated over A_V.')

    if ranking_metric == 'mass_weighted_rate':
        rank_note = ('Reactions ordered by mass-weighted rate '
                     '(\u222B 4\u03C0 r\u00B2 k n dr / \u222B 4\u03C0 r\u00B2 n dr).')
    else:
        rank_note = ('Reactions ordered by fractional contribution to the total '
                     f'{rate_label} rate (\u222B 4\u03C0 r\u00B2 k n dr / total).')

    hdr_wrate = {**_REACT_TABLE_CELL, **(
        {'fontWeight': '700'} if ranking_metric == 'mass_weighted_rate' else {})}
    hdr_pct = {**_REACT_TABLE_CELL, **(
        {'fontWeight': '700'} if ranking_metric == 'fractional_contribution' else {})}

    rows = []
    for i, j in enumerate(order, 1):
        lab = labels[j] if j < len(labels) else f'reaction {j}'
        st = stats.get(j, {})
        wrate = st.get('weighted_rate', np.nan)
        pct = st.get('fraction_pct', np.nan)
        rows.append(html.Tr([
            html.Td(f'{i}.', style={**_REACT_TABLE_CELL, 'width': '28px', 'color': '#666'}),
            html.Td(lab, style={**_REACT_TABLE_CELL, 'fontFamily': 'monospace'}),
            html.Td(f'{wrate:.2e}' if np.isfinite(wrate) else '\u2014',
                    style={**_REACT_TABLE_CELL, 'whiteSpace': 'nowrap'}),
            html.Td(f'{pct:6.2f}%' if np.isfinite(pct) else '\u2014',
                    style={**_REACT_TABLE_CELL, 'whiteSpace': 'nowrap'}),
        ]))

    return html.Div([
        html.P(f'{rank_note} {weight_note}',
               style={'fontSize': '11px', 'color': '#666', 'margin': '0 0 6px'}),
        html.Table([
            html.Thead(html.Tr([
                html.Th('#', style=_REACT_TABLE_CELL),
                html.Th('Reaction', style=_REACT_TABLE_CELL),
                html.Th(rate_hdr, style=hdr_wrate),
                html.Th(f'% of total {rate_label}', style=hdr_pct),
            ])),
            html.Tbody(rows),
        ], style=_REACT_TABLE_STYLE),
    ])


def fig_reactions(filepath, species, mode, xscale, yscale, top_n, model=None,
                  ranking_metric=DEFAULT_REACT_RANKING, theme='light',
                  av_range=DEFAULT_AV_RANGE, overlay_filepath=None):
    """Top-N formation or destruction reactions for a species (vs A_V)."""
    ranking_metric = _parse_react_ranking(ranking_metric)
    av_range = _parse_av_range(av_range)
    title = f'{species}: {"formation" if mode == "formation" else "destruction"} reactions'
    if filepath is None:
        return placeholder_fig('Load a chemistry grid to see reactions', theme=theme), {}, []

    data = get_reaction_data(filepath, species, mode)
    if data is None:
        return placeholder_fig(f'No {mode} data for {species}', theme=theme), {}, []

    matrix, av, labels = data['matrix'], data['av'], data['labels']
    order = select_top_reactions(matrix, av, top_n)
    if not order:
        return placeholder_fig(f'No non-zero {mode} reactions for {species}', theme=theme), {}, []

    if model is None:
        model = dict(av=av)
    elif model.get('av') is None:
        model = {**model, 'av': av}
    stats = compute_reaction_contributions(matrix, model, species)
    order = sort_reactions_by_ranking(order, stats, ranking_metric)

    # Reuse the shared A_V axis helper (full HDF5 axis vs optional AV_FLOOR).
    xv, _xl, xt, xr = _xvals(dict(av=av), 'Av', xscale, av_range=av_range)
    x_ok = np.isfinite(xv)

    fig = go.Figure()
    # Y-range follows the currently plotted top-N reactions (±1 decade pad),
    # using only depth points that are actually drawn on the x-axis.
    displayed_ys = []
    for k, j in enumerate(order):
        # Rates may be signed (esp. destruction); plots show magnitudes.
        y = np.abs(matrix[:, j].astype(float))
        if yscale == 'log':
            y = np.where(y > 0, y, np.nan)
        y_plot = np.where(x_ok, y, np.nan)
        displayed_ys.append(y_plot)
        lab = labels[j] if j < len(labels) else f'reaction {j}'
        st = stats.get(j, {})
        pct = st.get('fraction_pct', np.nan)
        wrate = st.get('weighted_rate', np.nan)
        legend = _reaction_legend_label(lab, st, ranking_metric)
        hover = (
            f'{lab}<br>A_V=%{{x:.3g}}<br>rate=%{{y:.3g}}'
            + (f'<br>weighted rate={wrate:.2e}' if np.isfinite(wrate) else '')
            + (f'<br>contribution={pct:.2f}%' if np.isfinite(pct) else '')
            + '<extra></extra>'
        )
        fig.add_trace(go.Scatter(
            x=xv, y=y_plot, mode='lines',
            line=dict(color=COLORS[k % len(COLORS)], width=1.9, dash='solid'),
            name=legend,
            legendrank=k + 1,
            hovertemplate=hover,
        ))
    # Sum of every |reaction| column — always shown, independent of top-N.
    y_total = np.sum(np.abs(matrix), axis=1).astype(float)
    if yscale == 'log':
        y_total = np.where(y_total > 0, y_total, np.nan)
    y_total = np.where(x_ok, y_total, np.nan)
    total_name = f'Total {mode}'
    fig.add_trace(go.Scatter(
        x=xv, y=y_total, mode='lines',
        line=dict(color='black', width=2.5),
        name=total_name,
        legendrank=0,
        hovertemplate=(
            f'{total_name}<br>A_V=%{{x:.3g}}<br>rate=%{{y:.3g}}<extra></extra>'
        ),
    ))

    if overlay_filepath and not _same_data_path(overlay_filepath, filepath):
        odata = get_reaction_data(overlay_filepath, species, mode)
        if odata is not None:
            oxv, _, _, _ = _xvals(dict(av=odata['av']), 'Av', xscale, av_range=av_range)
            x_ok_o = np.isfinite(oxv)
            omatrix = odata['matrix']
            olabels = odata['labels']
            for k, j in enumerate(order):
                if j >= omatrix.shape[1]:
                    continue
                y = np.abs(omatrix[:, j].astype(float))
                if yscale == 'log':
                    y = np.where(y > 0, y, np.nan)
                y_plot = np.where(x_ok_o, y, np.nan)
                displayed_ys.append(y_plot)
                lab = olabels[j] if j < len(olabels) else f'reaction {j}'
                legend = _reaction_legend_label(lab, {}, ranking_metric) + ' \u00B7 att'
                fig.add_trace(go.Scatter(
                    x=oxv, y=y_plot, mode='lines',
                    line=dict(color=COLORS[k % len(COLORS)], width=1.5, dash='dash'),
                    opacity=0.5,
                    name=legend,
                    legendrank=k + 1,
                    hovertemplate=(
                        f'{lab}<br>A_V=%{{x:.3g}}<br>rate=%{{y:.3g}}<extra></extra>'
                    ),
                ))
            y_ototal = np.sum(np.abs(omatrix), axis=1).astype(float)
            if yscale == 'log':
                y_ototal = np.where(y_ototal > 0, y_ototal, np.nan)
            y_ototal = np.where(x_ok_o, y_ototal, np.nan)
            displayed_ys.append(y_ototal)
            fig.add_trace(go.Scatter(
                x=oxv, y=y_ototal, mode='lines',
                line=dict(color='black', width=2.0, dash='dash'),
                opacity=0.5,
                name=f'{total_name} \u00B7 att',
                legendrank=0,
                hovertemplate=(
                    f'{total_name}<br>A_V=%{{x:.3g}}<br>rate=%{{y:.3g}}<extra></extra>'
                ),
            ))

    yr = _reaction_y_range(displayed_ys, yscale)
    _apply_layout(fig, title, 'A<sub>V</sub> (mag)', xt, xr, _REACT_YLABEL, yscale, theme=theme,
                  uirevision_extra=av_range)
    yr_tag = f'{yr[0]:.4g}:{yr[1]:.4g}' if yr is not None else 'auto'
    fig.update_layout(
        legend=dict(font=ps.legend_font()),
        uirevision=(
            f'react|{xt}|{yscale}|{av_range}|{top_n}|{species}|{mode}|'
            f'{ranking_metric}|{yr_tag}|{_parse_plot_theme(theme)}'
        ),
    )
    if yr is not None:
        fig.update_yaxes(type=yscale, range=yr, autorange=False)
    else:
        fig.update_yaxes(type=yscale, autorange=True)
    return fig, stats, order


def make_reaction_plots(values, species, xscale, yscale, top_n,
                        ranking_metric=DEFAULT_REACT_RANKING, theme='light',
                        av_range=DEFAULT_AV_RANGE,
                        chain_upstream=None, chain_downstream=None,
                        chain_include_isotopes=False, chain_include_ice=False,
                        network_highlight=None, network_hidden=None,
                        partner_label_size=None, species_label_size=None):
    """Return formation/destruction figures, tables, and pathway network."""
    ranking_metric = _parse_react_ranking(ranking_metric)
    av_range = _parse_av_range(av_range)
    empty = html.Div()
    theme = _parse_plot_theme(theme)
    tc = _theme_colors(theme)
    net_placeholder = placeholder_fig(
        'Load a chemistry grid and pick a species', theme=theme)
    net_status = chem_network.status_message({})
    try:
        up = int(chain_upstream) if chain_upstream is not None else chem_network.DEFAULT_CHAIN_UPSTREAM
    except (TypeError, ValueError):
        up = chem_network.DEFAULT_CHAIN_UPSTREAM
    try:
        down = int(chain_downstream) if chain_downstream is not None else chem_network.DEFAULT_CHAIN_DOWNSTREAM
    except (TypeError, ValueError):
        down = chem_network.DEFAULT_CHAIN_DOWNSTREAM
    include_isotopes = bool(chain_include_isotopes)
    include_ice = bool(chain_include_ice)
    partner_pt = partner_label_size
    species_pt = species_label_size
    if not _chem or not species:
        p = placeholder_fig('Load a chemistry grid and pick a species', theme=theme)
        return (p, empty, p, empty, net_placeholder, net_status)
    chem_path = chem_file(values)
    chem_overlay_path = chem_overlay_file(values)
    if chem_overlay_path and _same_data_path(chem_overlay_path, chem_path):
        chem_overlay_path = None
    struct_path = current_file(values) if _grid_has_hdf5() else None
    model = get_model(struct_path) if struct_path else get_chem_model(chem_path)
    if chem_path:
        fig_f, stats_f, order_f = fig_reactions(
            chem_path, species, 'formation', xscale, yscale, top_n, model=model,
            ranking_metric=ranking_metric, theme=theme, av_range=av_range,
            overlay_filepath=chem_overlay_path)
        fig_d, stats_d, order_d = fig_reactions(
            chem_path, species, 'destruction', xscale, yscale, top_n, model=model,
            ranking_metric=ranking_metric, theme=theme, av_range=av_range,
            overlay_filepath=chem_overlay_path)
        labels_f = (get_reaction_data(chem_path, species, 'formation') or {}).get('labels', [])
        labels_d = (get_reaction_data(chem_path, species, 'destruction') or {}).get('labels', [])
        tbl_f = _reaction_contribution_table(
            order_f, labels_f, stats_f, 'formation', ranking_metric)
        tbl_d = _reaction_contribution_table(
            order_d, labels_d, stats_d, 'destruction', ranking_metric)
        selected_f = [labels_f[i] for i in (order_f or []) if 0 <= int(i) < len(labels_f)]
        selected_d = [labels_d[i] for i in (order_d or []) if 0 <= int(i) < len(labels_d)]
    else:
        missing = placeholder_fig('No chemistry file for this model point', theme=theme)
        fig_f = fig_d = missing
        tbl_f = tbl_d = empty
        selected_f = selected_d = None
    sp_html = format_species_html(species)
    fig_net, net_meta = chem_network.build_network_panels(
        _chem.get('labels') or {}, species,
        upstream_depth=up, downstream_depth=down,
        include_isotopes=include_isotopes,
        include_ice=include_ice,
        highlight_species=network_highlight,
        theme_colors=tc,
        title=f'Reaction pathway network — {sp_html}',
        focal_formation=selected_f,
        focal_destruction=selected_d,
        hidden_species=network_hidden,
        partner_label_size=partner_pt,
        species_label_size=species_pt,
    )
    return (fig_f, tbl_f, fig_d, tbl_d, fig_net,
            chem_network.status_message(net_meta))


# --- 2-D parameter-slice contour grids ----------------------------------------

def _middle_token(key):
    toks = _grid['axis_tokens'][key]
    return toks[len(toks) // 2]


def _token_list_from_sliders(slider_values):
    """Full parameter token list from profile sliders, else grid midpoints."""
    if slider_values is not None:
        toks = _tokens_from_values(list(slider_values))
        if toks is not None:
            return list(toks)
    return [_middle_token(p['key']) for p in PARAM_DEFS]


def _model_point_tokens(slider_values=None, int_slice_indices=None):
    """Resolve the full model token tuple for the current UI selection."""
    if not _grid:
        return None
    if int_slice_indices and _grid.get('simline_only'):
        tokens = _token_list_from_sliders(slider_values)
        for plane, idx in zip(active_slice_planes(), int_slice_indices):
            if not plane.get('active') or not plane.get('slice'):
                continue
            sk = plane['slice']
            try:
                tokens[_PARAM_IDX[sk]] = _grid['axis_tokens'][sk][int(idx)]
            except (IndexError, TypeError, ValueError):
                pass
        return tuple(tokens)
    if slider_values is not None:
        toks = _tokens_from_values(list(slider_values))
        if toks is not None:
            return toks
    return tuple(_token_list_from_sliders(slider_values))


def _param_def(key):
    return PARAM_DEFS[_PARAM_IDX[key]]


def _plane_plot_axes(plane):
    """Return (x_key, y_key, slice_key); prefer ζ on the x-axis when it is plotted."""
    if not plane.get('active'):
        return plane.get('x'), plane.get('y'), plane.get('slice')
    xk, yk, sk = plane['x'], plane['y'], plane['slice']
    if yk == 'crir' and xk != 'crir':
        xk, yk = yk, xk
    return xk, yk, sk


def _axis_label(pdef):
    if pdef['key'] == 'density':
        return 'n<sub>H</sub> (cm<sup>-3</sup>)'
    if pdef['key'] == 'fuv':
        return '\u03C7 (Draine)'
    if pdef['key'] == 'crir':
        return '\u03B6 (s<sup>-1</sup>)'
    if pdef['key'] == 'mass':
        return 'M (M<sub>\u2299</sub>)'
    return pdef['name']


def _quantity_label(quantity):
    if quantity == 'tgas':
        return 'T<sub>gas</sub> (K)'
    if quantity == 'tgas_col':
        return '⟨T<sub>gas</sub>⟩<sub>N</sub> (K)'
    if quantity == 'tdust':
        return 'T<sub>dust</sub> (K)'
    if quantity == 'nh':
        return 'n<sub>H</sub> (cm<sup>-3</sup>)'
    if quantity == 'xe':
        return 'x<sub>e</sub> = n(e<sup>-</sup>)/n<sub>H</sub>'
    if quantity.startswith('species:'):
        sp = quantity.split(':', 1)[1]
        return f'X<sub>{format_species_html(sp)}</sub>'
    if quantity.startswith('cdens:'):
        sp = quantity.split(':', 1)[1]
        return f'N({format_species_html(sp)}) (cm<sup>-2</sup>)'
    if quantity.startswith('X(') and ')/X(' in quantity:
        try:
            left, right = quantity.split(')/X(', 1)
            num = left[2:]
            den = right.rstrip(')')
            return (f'X<sub>{format_species_html(num)}</sub>/'
                    f'X<sub>{format_species_html(den)}</sub>')
        except ValueError:
            return quantity
    return quantity


def _as_str_list(value):
    """Normalize a dropdown value to a list of non-empty strings."""
    if value is None or value == '':
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and v != '']
    return [str(value)]


def _first_str(value):
    items = _as_str_list(value)
    return items[0] if items else None


def get_grid_scalar(filepath, quantity):
    """Scalar for contour grids (species: integrated X; diagnostics: see labels)."""
    cache_key = (filepath, quantity)
    if cache_key in _scalar_cache:
        return _scalar_cache[cache_key]
    try:
        if str(quantity).startswith('cdens:'):
            val = total_column_density(filepath, quantity.split(':', 1)[1])
        else:
            model = get_model(filepath)
            if quantity == 'tgas':
                val = float(np.asarray(model['tgas'], float)[0])
            elif quantity == 'tgas_col':
                val = column_averaged_tgas(model)
            elif quantity == 'tdust':
                val = float(np.asarray(model['tdust'], float)[0])
            elif quantity == 'nh':
                val = float(np.asarray(model['nH'], float)[0])
            elif quantity == 'xe':
                val = electron_fraction_edge(model)
            elif quantity.startswith('species:'):
                sp = quantity.split(':', 1)[1]
                val = integrated_rel_abundance(model, sp)
            else:
                val = np.nan
    except (IndexError, TypeError, ValueError):
        val = np.nan
    _scalar_cache[cache_key] = val
    return val


def _axis_plot_coords(values, pdef):
    """Map physical axis values to plot coordinates (log params -> log10)."""
    arr = np.asarray(values, dtype=float)
    if pdef['logscale']:
        return np.where(arr > 0, np.log10(arr), np.nan)
    return arr


def _axis_label_contour(pdef):
    """Axis label matching KoSens grid plots (log10 on a linear axis)."""
    if pdef['key'] == 'density':
        return 'log<sub>10</sub> n<sub>H</sub> (cm<sup>-3</sup>)'
    if pdef['key'] == 'fuv':
        return 'log<sub>10</sub> \u03C7 (Draine)'
    if pdef['key'] == 'crir':
        return 'log<sub>10</sub> \u03B6 (s<sup>-1</sup>)'
    if pdef['key'] == 'mass':
        return 'log<sub>10</sub> M (M<sub>\u2299</sub>)'
    return pdef['name']


def _plot_coord_span(coords):
    """Span of axis values used for square-ish contour aspect ratio."""
    arr = np.asarray(coords, dtype=float)
    fin = arr[np.isfinite(arr)]
    if fin.size < 2:
        return 1.0
    span = float(np.max(fin) - np.min(fin))
    return span if span > 0 else 1.0


def _contour_data_aspect(x_plot, y_plot):
    """y/x span ratio in plot coordinates (log axes when applicable)."""
    return _plot_coord_span(y_plot) / _plot_coord_span(x_plot)


def _contour_colorbar(title, theme='light', zscale='log', compact=True):
    """Colorbar geometry; compact grids sit next to the plot like the χ² maps."""
    t = _theme_colors(theme)
    # Log mode plots log10(Z), so plain decimal ticks.  Linear mode often spans
    # many decades → scientific exponents on the colorbar.
    if zscale == 'log':
        tick_kw = dict(exponentformat='none', showexponent='none')
    else:
        tick_kw = dict(exponentformat='e', showexponent='all')
    if compact:
        pos_kw = dict(
            len=_COMPACT_YSPAN,
            lenmode='fraction',
            thickness=16,
            x=_COMPACT_CBAR_X,
            xanchor='left',
            xpad=2,
            y=0.5,
            yanchor='middle',
        )
    else:
        pos_kw = dict(
            len=0.82,
            thickness=14,
            x=1.02,
            xanchor='left',
            xpad=4,
            y=0.5,
            yanchor='middle',
        )
    return dict(
        title=dict(text=title, font=ps.cbar_title_font(t['font'])),
        tickfont=ps.cbar_tick_font(t['font']),
        outlinewidth=0,
        **pos_kw,
        **tick_kw,
    )


def _subplot_colorbar(title, x, theme='light', zscale='log', **extra):
    """Colorbar tucked against one panel in a 1×3 subplot row."""
    t = _theme_colors(theme)
    if zscale == 'log':
        tick_kw = dict(exponentformat='none', showexponent='none')
    else:
        tick_kw = dict(exponentformat='e', showexponent='all')
    return dict(
        title=dict(text=title, font=ps.cbar_title_font(t['font'])),
        tickfont=ps.cbar_tick_font(t['font']),
        len=0.88,
        thickness=12,
        x=x,
        xanchor='left',
        xpad=4,
        xref='paper',
        outlinewidth=0,
        **tick_kw,
        **extra,
    )


def _apply_zscale(Z, zscale):
    """Map physical Z to plot values; return (Zplot, zmin, zmax)."""
    Zplot = np.asarray(Z, dtype=float)
    if zscale == 'log':
        Zplot = np.where(Zplot > 0, np.log10(Zplot), np.nan)
    fin = Zplot[np.isfinite(Zplot)]
    if fin.size == 0:
        return Zplot, None, None
    zmin = float(np.nanmin(fin))
    zmax = float(np.nanmax(fin))
    if zmin == zmax:
        pad = 0.5 if zscale == 'log' else (abs(zmin) * 0.05 + 1e-30)
        zmin, zmax = zmin - pad, zmax + pad
    return Zplot, zmin, zmax


def _compact_slice_layout_kw(title, xlabel, ylabel, theme='light', uirevision='slice',
                             x_range=None, y_range=None, showgrid=True):
    """Identical figure box for every side-by-side slice panel.

    Uses a fixed width/height and shared axis domains.  Deliberately does
    *not* set ``scaleanchor``: equal log-decade scaling would shrink panels
    differently when the three planes have unequal axis spans (e.g. 4 vs 5
    decades), which makes the three plots look like different shapes.

    Optional ``x_range`` / ``y_range`` lock the data window (needed for RGB
    ``go.Image`` so Plotly does not re-derive a pixel-based aspect).
    """
    t = _theme_colors(theme)
    axis_common = dict(
        type='linear', showgrid=bool(showgrid), gridcolor=t['grid'],
        linecolor=t['axis_line'], tickfont=ps.tick_font(t['font']),
        constrain='domain',
        fixedrange=False,
    )
    xaxis = dict(
        title=dict(text=xlabel, font=ps.axis_title_font(t['font'])),
        domain=list(_COMPACT_XDOMAIN),
        **axis_common,
    )
    yaxis = dict(
        title=dict(text=ylabel, font=ps.axis_title_font(t['font'])),
        domain=list(_COMPACT_YDOMAIN),
        **axis_common,
    )
    if x_range is not None:
        xaxis['range'] = [float(x_range[0]), float(x_range[1])]
        xaxis['autorange'] = False
    else:
        xaxis['autorange'] = True
    if y_range is not None:
        yaxis['range'] = [float(y_range[0]), float(y_range[1])]
        yaxis['autorange'] = False
    else:
        yaxis['autorange'] = True
    return dict(
        title=dict(text=f'<b>{title}</b>', font=ps.title_font(t['title']),
                   x=0.02, xanchor='left'),
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        width=COMPACT_FIG_WIDTH,
        height=COMPACT_FIG_HEIGHT,
        autosize=False,
        margin=dict(_COMPACT_FIG_MARGIN),
        font=ps.layout_font(t['font']),
        uirevision=uirevision,
        xaxis=xaxis,
        yaxis=yaxis,
    )


def _apply_square_contour_layout(fig, x_plot, y_plot, xdef, ydef, title,
                                 panel_w=DEFAULT_CONTOUR_PANEL_W,
                                 fixed_size=False, theme='light',
                                 zscale='log'):
    """Layout for a single contour panel.

    ``fixed_size=True`` — identical figsize and axis box for every panel in a
    side-by-side row (Grid / Intensities / RGB).
    """
    t = _theme_colors(theme)
    # Include zscale so colorbar range resets when switching log↔linear.
    uirev = f'contour|{zscale}|{_parse_plot_theme(theme)}'
    if fixed_size:
        fig.update_layout(**_compact_slice_layout_kw(
            title, _axis_label_contour(xdef), _axis_label_contour(ydef),
            theme=theme, uirevision=uirev,
        ))
        return fig

    title_kw = dict(text=f'<b>{title}</b>', font=ps.title_font(t['title']),
                    x=0.02, xanchor='left')
    axis_common = dict(
        type='linear', showgrid=True, gridcolor=t['grid'],
        linecolor=t['axis_line'], tickfont=ps.tick_font(t['font']),
        autorange=True,
    )
    aspect = _contour_data_aspect(x_plot, y_plot)
    fig_w = panel_w + 90
    fig_h = int(panel_w * aspect) + 106
    fig.update_layout(
        title=title_kw,
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        width=fig_w,
        height=fig_h,
        autosize=False,
        margin=dict(l=60, r=56, t=52, b=54),
        font=ps.layout_font(t['font']),
        uirevision=uirev,
        xaxis=dict(
            title=dict(text=_axis_label_contour(xdef), font=ps.axis_title_font(t['font'])),
            constrain='domain',
            **axis_common,
        ),
        yaxis=dict(
            title=dict(text=_axis_label_contour(ydef), font=ps.axis_title_font(t['font'])),
            scaleanchor='x', scaleratio=aspect,
            constrain='domain',
            **axis_common,
        ),
    )
    return fig


def _contour_trace_kw(theme='light', show_lines=True):
    t = _theme_colors(theme)
    return dict(
        contours=dict(coloring='heatmap', showlines=show_lines,
                      labelfont=dict(color='white', size=10 if show_lines else 9)),
        line=dict(color=t['contour_line'], width=0.8 if show_lines else 0.6),
        connectgaps=False,
        hovertemplate='x=%{x:.3g}<br>y=%{y:.3g}<br>z=%{z:.3g}<extra></extra>',
    )


def _parse_int_extra_contour_levels(text):
    """Parse comma/semicolon/whitespace-separated intensity levels (physical units)."""
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    levels = []
    for part in re.split(r'[,;\s]+', raw):
        if not part:
            continue
        try:
            v = float(part)
        except ValueError:
            continue
        if np.isfinite(v):
            levels.append(v)
    return levels


def _parse_int_extra_contour_colors(text, n_levels):
    """Parse CSS colors for extra contours; cycle palette when under-specified.

    Empty / blank → one colour per level from ``INT_CONTOUR_COLORS``.
    One colour → reuse for every level (legacy single-colour behaviour).
    Several colours → assign in order, cycling if fewer than ``n_levels``.
    """
    n = max(int(n_levels), 0)
    if n == 0:
        return []
    raw = (text or '').strip()
    if not raw:
        # Legacy single-contour default is black; multi-level uses a palette.
        if n == 1:
            return ['black']
        return [INT_CONTOUR_COLORS[i % len(INT_CONTOUR_COLORS)] for i in range(n)]
    colors = [c.strip() for c in re.split(r'[,;]+', raw) if c.strip()]
    if not colors:
        if n == 1:
            return ['black']
        return [INT_CONTOUR_COLORS[i % len(INT_CONTOUR_COLORS)] for i in range(n)]
    if len(colors) == 1:
        return [colors[0]] * n
    return [colors[i % len(colors)] for i in range(n)]


def _physical_to_plot_intensity(level, zscale):
    """Map a physical intensity level to the Z array used in contour plots."""
    if level is None or not np.isfinite(level):
        return None
    if zscale == 'log':
        if level <= 0:
            return None
        return float(np.log10(level))
    return float(level)


def _build_intensity_line_contours(idef, zscale, show_obs_boundary,
                                   extra_levels_text, extra_color):
    """Build line-only contour specs for observational and user levels."""
    lines = []
    if show_obs_boundary and idef == 'jtemp':
        lvl = _physical_to_plot_intensity(INT_OBS_BOUNDARY_JTEMP, zscale)
        if lvl is not None:
            lines.append(dict(
                level=lvl,
                color=INT_OBS_BOUNDARY_COLOR,
                width=INT_OBS_BOUNDARY_WIDTH,
                label=f'obs. limit ({INT_OBS_BOUNDARY_JTEMP:g} K km/s)',
            ))
    phys_levels = _parse_int_extra_contour_levels(extra_levels_text)
    colors = _parse_int_extra_contour_colors(extra_color, len(phys_levels))
    unit = _intensity_unit_label(idef)
    for phys, color in zip(phys_levels, colors):
        lvl = _physical_to_plot_intensity(phys, zscale)
        if lvl is not None:
            lines.append(dict(
                level=lvl,
                color=color,
                width=INT_EXTRA_CONTOUR_WIDTH,
                label=f'{phys:g} {unit}',
            ))
    return lines


def _add_contour_level_lines(fig, x_plot, y_plot, z_plot, line_contours, row=None, col=None):
    """Overlay constant-intensity contour lines on an existing contour figure."""
    for spec in line_contours:
        trace = go.Contour(
            x=x_plot, y=y_plot, z=z_plot,
            contours=dict(
                coloring='none',
                showlabels=True,
                start=spec['level'], end=spec['level'], size=1,
                labelfont=dict(size=10, color=spec['color']),
            ),
            line=dict(color=spec['color'], width=spec.get('width', INT_EXTRA_CONTOUR_WIDTH)),
            showscale=False,
            hoverinfo='skip',
            name=spec.get('label', ''),
        )
        if row is not None and col is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)


def _hex_to_rgba(color, alpha=0.35):
    """Best-effort CSS colour → rgba() string for translucent contour bands."""
    raw = (color or '').strip()
    if not raw:
        return f'rgba(31,119,180,{alpha})'
    if raw.startswith('rgba(') or raw.startswith('rgb('):
        return raw
    if raw.startswith('#') and len(raw) in (4, 7):
        h = raw[1:]
        if len(h) == 3:
            h = ''.join(c * 2 for c in h)
        try:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f'rgba({r},{g},{b},{alpha})'
        except ValueError:
            pass
    named = {
        'black': (0, 0, 0), 'white': (255, 255, 255), 'red': (214, 39, 40),
        'blue': (31, 119, 180), 'green': (44, 160, 44), 'orange': (255, 127, 14),
        'purple': (148, 103, 189), 'cyan': (23, 190, 207), 'gray': (127, 127, 127),
        'grey': (127, 127, 127),
    }
    rgb = named.get(raw.lower())
    if rgb is not None:
        return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'
    return f'rgba(31,119,180,{alpha})'


def _parse_spaghetti_contour_spec(text, available_keys=None):
    """Parse named observed intensities for spaghetti / χ² intersection.

    Accepts JSON ``{"CO(1-0)": 1.2, ...}`` or line-oriented text::

        CO(1-0) = 1.2
        CO(2-1): 3.5 ± 0.7
        C+(158um): 5.0, 1.0

    Optional second number / ± term is the intensity error; otherwise
    ``INT_SPAGHETTI_DEFAULT_ERROR_FRAC`` of the level is used.

    Returns
    -------
    list of dict
        ``{name, level, error}`` for each successfully parsed entry.
    """
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []

    available = set(available_keys or [])
    entries = []

    def _append(name, level, error=None):
        name = str(name).strip()
        if not name:
            return
        try:
            level = float(level)
        except (TypeError, ValueError):
            return
        if not np.isfinite(level):
            return
        if error is None:
            error = abs(level) * INT_SPAGHETTI_DEFAULT_ERROR_FRAC
        else:
            try:
                error = float(error)
            except (TypeError, ValueError):
                error = abs(level) * INT_SPAGHETTI_DEFAULT_ERROR_FRAC
        if not np.isfinite(error) or error <= 0:
            error = abs(level) * INT_SPAGHETTI_DEFAULT_ERROR_FRAC or 1e-30
        if available and name not in available:
            # Soft match: case-insensitive / unique prefix among available keys.
            lower = name.lower()
            matches = [k for k in available if k.lower() == lower]
            if not matches:
                matches = [k for k in available if k.lower().startswith(lower)]
            if len(matches) == 1:
                name = matches[0]
            else:
                return
        entries.append(dict(name=name, level=level, error=error))

    if raw.startswith('{'):
        try:
            data = gf.parse_json_dict(raw)
        except Exception:
            data = None
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (list, tuple)) and len(v) >= 1:
                    _append(k, v[0], v[1] if len(v) > 1 else None)
                elif isinstance(v, dict):
                    _append(k, v.get('level', v.get('value')),
                            v.get('error', v.get('err')))
                else:
                    _append(k, v)
            return entries

    line_re = re.compile(
        r'^\s*(.+?)\s*[=:]\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
        r'(?:\s*(?:±|\+/-|\+\/\-|,|;)\s*'
        r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))?\s*$'
    )
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = line_re.match(line)
        if not m:
            continue
        _append(m.group(1), m.group(2), m.group(3))
    return entries


def compute_chi2_surface(x_plot, y_plot, grids, levels, errors):
    """Compute χ² = Σ ((Z − level)/error)² surface over the grid.

    Returns
    -------
    chi2_grid : 2-D array shaped (len(y_plot), len(x_plot))
        The χ² value at each grid point.
    """
    x_arr = np.asarray(x_plot, dtype=float)
    y_arr = np.asarray(y_plot, dtype=float)
    chi2_grid = np.zeros((y_arr.size, x_arr.size), dtype=float)
    
    for grid, level, err in zip(grids, levels, errors):
        z = np.asarray(grid, dtype=float)
        if z.shape == (y_arr.size, x_arr.size) and err > 0:
            residual = (z - level) / err
            chi2_grid += np.where(np.isfinite(residual), residual ** 2, 0)
    
    return chi2_grid


def find_nearest_contour_point(x_plot, y_plot, grids, levels, errors):
    """Minimise χ² = Σ ((Z − level)/error)² over plot-coordinate space.

    Uses multiple starting points to avoid local minima:
    - Grid center
    - Grid corners
    - Points near the minimum of the chi2 surface

    Parameters
    ----------
    x_plot, y_plot : 1-D arrays
        Plot-axis coordinates (log10 for logscale parameters).
    grids : sequence of 2-D arrays
        Intensity grids shaped ``(len(y_plot), len(x_plot))``.
    levels, errors : sequence of float
        Observed contour levels and uncertainties.

    Returns
    -------
    dict or None
        ``x``, ``y`` (plot coords), ``chi2``, ``relative_chi2``, ``chi2_surface`` on success.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.optimize import minimize

    if len(grids) < 1:
        return None
    x_arr = np.asarray(x_plot, dtype=float)
    y_arr = np.asarray(y_plot, dtype=float)
    if x_arr.size < 2 or y_arr.size < 2:
        return None
    x_min, x_max = float(np.nanmin(x_arr)), float(np.nanmax(x_arr))
    y_min, y_max = float(np.nanmin(y_arr)), float(np.nanmax(y_arr))
    if not np.isfinite([x_min, x_max, y_min, y_max]).all():
        return None

    # Compute chi2 surface for plotting and finding starting points
    chi2_surface = compute_chi2_surface(x_plot, y_plot, grids, levels, errors)
    
    # Find approximate minimum from grid (for starting point)
    finite_chi2 = np.where(np.isfinite(chi2_surface), chi2_surface, np.inf)
    min_idx = np.unravel_index(np.argmin(finite_chi2), finite_chi2.shape)
    grid_min_x = x_arr[min_idx[1]]
    grid_min_y = y_arr[min_idx[0]]

    interpolators = []
    for grid in grids:
        z = np.asarray(grid, dtype=float)
        if z.shape != (y_arr.size, x_arr.size):
            return None
        interpolators.append(RegularGridInterpolator(
            (y_arr, x_arr), z, method='linear',
            bounds_error=False, fill_value=np.nan,
        ))

    def chi2_func(pt):
        chi2 = 0.0
        for interp, level, err in zip(interpolators, levels, errors):
            val = float(interp([[pt[1], pt[0]]])[0])
            if np.isfinite(val) and err > 0:
                chi2 += ((val - level) / err) ** 2
        return chi2

    # Multiple starting points: center, corners, grid minimum, and intermediate points
    starting_points = [
        ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),  # center
        (grid_min_x, grid_min_y),                         # grid-based minimum
        (x_min, y_min), (x_max, y_min),                   # corners
        (x_min, y_max), (x_max, y_max),
        ((x_min + x_max) / 2.0, y_min),                   # edge midpoints
        ((x_min + x_max) / 2.0, y_max),
        (x_min, (y_min + y_max) / 2.0),
        (x_max, (y_min + y_max) / 2.0),
    ]
    
    best_result = None
    best_chi2 = np.inf
    
    for x0, y0 in starting_points:
        try:
            result = minimize(
                chi2_func,
                np.array([x0, y0]),
                method='L-BFGS-B',
                bounds=[(x_min, x_max), (y_min, y_max)],
            )
            if result.success and result.fun < best_chi2:
                best_chi2 = result.fun
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fall back to grid minimum
        best_x, best_y = grid_min_x, grid_min_y
        best_chi2 = float(finite_chi2[min_idx])
    else:
        best_x = float(best_result.x[0])
        best_y = float(best_result.x[1])
        best_chi2 = float(best_result.fun)
    
    n = len(levels)
    return dict(
        x=best_x,
        y=best_y,
        chi2=best_chi2,
        relative_chi2=best_chi2 / n if n else best_chi2,
        chi2_surface=chi2_surface,
    )


def _add_spaghetti_contour(fig, x_plot, y_plot, z, level, error, color, name,
                           show_band=True):
    """Draw one observed-level contour (+ optional ±error band) on ``fig``.
    
    Error bands are shown as filled regions between level-error and level+error,
    with thin dashed boundary lines.
    """
    z_arr = np.asarray(z, dtype=float)
    if not np.any(np.isfinite(z_arr)):
        return False
    drawn = False
    fill_color = _hex_to_rgba(color, INT_SPAGHETTI_BAND_OPACITY)

    # Draw error band as filled region between lower and upper contours
    if show_band and error is not None and error > 0:
        lo, hi = float(level - error), float(level + error)
        
        # Extract paths for lower and upper bounds
        paths_lo = gf._extract_contour_paths(x_plot, y_plot, z_arr, lo)
        paths_hi = gf._extract_contour_paths(x_plot, y_plot, z_arr, hi)
        
        # Create filled polygons between lower and upper contour paths
        # Try to pair paths and create filled regions
        if paths_lo and paths_hi:
            # For each pair of lo/hi paths, create a filled polygon
            for i, path_lo in enumerate(paths_lo):
                path_hi = paths_hi[i] if i < len(paths_hi) else paths_hi[-1]
                # Create closed polygon: go along hi path, then reverse along lo path
                poly_x = list(path_hi[:, 0]) + list(path_lo[::-1, 0]) + [path_hi[0, 0]]
                poly_y = list(path_hi[:, 1]) + list(path_lo[::-1, 1]) + [path_hi[0, 1]]
                fig.add_trace(go.Scatter(
                    x=poly_x, y=poly_y,
                    mode='lines',
                    line=dict(width=0),
                    fill='toself',
                    fillcolor=fill_color,
                    legendgroup=name,
                    showlegend=False,
                    hoverinfo='skip',
                ))
                drawn = True
        elif paths_lo or paths_hi:
            # If only one bound exists, draw it as a thin line
            for path in (paths_lo or paths_hi):
                fig.add_trace(go.Scatter(
                    x=path[:, 0], y=path[:, 1],
                    mode='lines',
                    line=dict(color=color, width=1.0, dash='dot'),
                    legendgroup=name,
                    showlegend=False,
                    hoverinfo='skip',
                    opacity=0.5,
                ))
                drawn = True
        
        # Draw boundary lines (thin dashed) for the error bounds
        for path in paths_lo:
            fig.add_trace(go.Scatter(
                x=path[:, 0], y=path[:, 1],
                mode='lines',
                line=dict(color=color, width=0.8, dash='dot'),
                legendgroup=name,
                showlegend=False,
                hoverinfo='skip',
                opacity=0.7,
            ))
        for path in paths_hi:
            fig.add_trace(go.Scatter(
                x=path[:, 0], y=path[:, 1],
                mode='lines',
                line=dict(color=color, width=0.8, dash='dot'),
                legendgroup=name,
                showlegend=False,
                hoverinfo='skip',
                opacity=0.7,
            ))

    # Draw the main contour line at the central level (on top)
    paths = gf._extract_contour_paths(x_plot, y_plot, z_arr, level)
    for i, path in enumerate(paths):
        fig.add_trace(go.Scatter(
            x=path[:, 0], y=path[:, 1],
            mode='lines',
            line=dict(color=color, width=INT_SPAGHETTI_LINE_WIDTH),
            legendgroup=name,
            name=name if i == 0 else None,
            showlegend=(i == 0),
            hoverinfo='skip',
        ))
        drawn = True
    
    if not paths:
        # Fallback: Plotly Contour line when matplotlib finds no path.
        fig.add_trace(go.Contour(
            x=x_plot, y=y_plot, z=z_arr,
            contours=dict(
                coloring='none', showlabels=False,
                start=level, end=level, size=1,
            ),
            line=dict(color=color, width=INT_SPAGHETTI_LINE_WIDTH),
            showscale=False,
            hoverinfo='skip',
            legendgroup=name,
            name=name,
            showlegend=True,
        ))
        fin = z_arr[np.isfinite(z_arr)]
        drawn = fin.size > 0 and float(np.nanmin(fin)) <= level <= float(np.nanmax(fin))

    return drawn


def fig_spaghetti_plane(plane, slice_idx, contour_entries, idef,
                        interp_config=None, theme='light', slider_values=None,
                        show_error_bands=True, mark_closest=True):
    """KoSens-style spaghetti plot: multi-line observed contours on one slice."""
    if not plane.get('active'):
        return placeholder_fig(theme=theme), None, None
    if not _grid or not _simline:
        return placeholder_fig('Load grids and a SIMLINE directory', theme=theme), None, None
    if not contour_entries:
        return placeholder_fig(
            'Provide observed intensities (line = value) above', theme=theme), None, None

    sk = plane['slice']
    slice_tokens = _grid['axis_tokens'][sk]
    if not slice_tokens:
        return placeholder_fig('No slice axis', theme=theme), None, None
    try:
        slice_token = slice_tokens[int(slice_idx)]
    except (IndexError, TypeError, ValueError):
        slice_token = slice_tokens[0]

    lookup = gf._line_lookup(_simline, idef or SIMLINE_DEFAULT_IDEF)
    names, grids, levels, errors, colors = [], [], [], [], []
    x_plot = y_plot = None
    xdef = ydef = None
    for i, entry in enumerate(contour_entries):
        resolved = lookup.get(entry['name'])
        if resolved is None:
            continue
        species, tidx = resolved
        x_phys, y_phys, Z, xd, yd = build_intensity_slice_grid(
            plane, slice_token, species, idef, tidx,
            interp_config=interp_config, slider_values=slider_values,
        )
        if not np.any(np.isfinite(Z)):
            continue
        if x_plot is None:
            x_plot = _axis_plot_coords(x_phys, xd)
            y_plot = _axis_plot_coords(y_phys, yd)
            xdef, ydef = xd, yd
        names.append(entry['name'])
        grids.append(Z)
        levels.append(float(entry['level']))
        errors.append(float(entry['error']))
        colors.append(INT_CONTOUR_COLORS[i % len(INT_CONTOUR_COLORS)])

    if not names or x_plot is None:
        return placeholder_fig('No matching SIMLINE lines for the given names',
                               theme=theme), None, None

    t = _theme_colors(theme)
    sdef = _param_def(sk)
    slice_disp = _param_token_disp(sdef, slice_token)
    slice_unit = f' {sdef["unit"]}' if sdef['unit'] else ''
    unit = _intensity_unit_label(idef)
    title = (f'{plane["title"]}<br>'
             f'<sup style="font-size:11px">{sdef["name"]} = {slice_disp}{slice_unit}'
             f'  \u2014  spaghetti contours [{unit}]</sup>')

    fig = go.Figure()
    n_drawn = 0
    for name, Z, level, err, color in zip(names, grids, levels, errors, colors):
        label = f'{name}  ({level:g} \u00b1 {err:g})'
        if _add_spaghetti_contour(
            fig, x_plot, y_plot, Z, level, err, color, label,
            show_band=show_error_bands,
        ):
            n_drawn += 1

    chi2_info = None
    if mark_closest and len(grids) >= 2:
        chi2_info = find_nearest_contour_point(
            x_plot, y_plot, grids, levels, errors,
        )
        if chi2_info is not None:
            fig.add_trace(go.Scatter(
                x=[chi2_info['x']], y=[chi2_info['y']],
                mode='markers',
                marker=dict(
                    symbol='star', size=14, color='red',
                    line=dict(width=1.5, color='white'),
                ),
                name=f'\u03c7\u00b2 min ({chi2_info["chi2"]:.3g})',
                showlegend=True,
            ))

    if n_drawn == 0:
        return placeholder_fig('Contours outside grid range for this slice',
                               theme=theme), None, chi2_info

    fig = _apply_square_contour_layout(
        fig, x_plot, y_plot, xdef, ydef, title,
        fixed_size=True, theme=theme, zscale='linear',
    )
    # Legend sits in the reserved right strip (axis domain ends at 0.84),
    # outside the heatmap and still inside this panel.
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top', y=1.0,
            xanchor='left', x=_COMPACT_CBAR_X,
            bgcolor=t['legend_bg'],
            bordercolor=t['legend_border'],
            borderwidth=1,
            font=ps.legend_font(t['font']),
            tracegroupgap=1,
            itemsizing='constant',
            itemwidth=30,
        ),
    )

    # Create chi2 surface figure
    chi2_fig = None
    if chi2_info is not None and 'chi2_surface' in chi2_info:
        chi2_surface = chi2_info['chi2_surface']
        chi2_title = (f'{plane["title"]} \u2014 \u03c7\u00b2 surface<br>'
                      f'<sup style="font-size:13px">{sdef["name"]} = {slice_disp}{slice_unit}'
                      f'  |  min \u03c7\u00b2 = {chi2_info["chi2"]:.3g}</sup>')

        # Use log scale for better visualization of chi2 variations
        chi2_log = np.log10(np.clip(chi2_surface, 1e-10, None))

        chi2_fig = go.Figure()
        chi2_fig.add_trace(go.Heatmap(
            x=x_plot, y=y_plot, z=chi2_log,
            colorscale='Viridis_r',
            colorbar=_contour_colorbar(
                'log\u2081\u2080(\u03c7\u00b2)', theme=theme, zscale='log',
            ),
            hovertemplate='x: %{x:.3g}<br>y: %{y:.3g}<br>log₁₀(χ²): %{z:.3g}<extra></extra>',
        ))
        # Mark the minimum
        chi2_fig.add_trace(go.Scatter(
            x=[chi2_info['x']], y=[chi2_info['y']],
            mode='markers',
            marker=dict(symbol='star', size=14, color='red',
                        line=dict(width=1.5, color='white')),
            name='\u03c7\u00b2 min',
            showlegend=False,
            hovertemplate=(
                f'χ² min = {chi2_info["chi2"]:.4g}<br>'
                f'x: {chi2_info["x"]:.4g}<br>y: {chi2_info["y"]:.4g}<extra></extra>'
            ),
        ))
        chi2_fig = _apply_square_contour_layout(
            chi2_fig, x_plot, y_plot, xdef, ydef, chi2_title,
            fixed_size=True, theme=theme, zscale='linear',
        )
        chi2_fig.update_layout(showlegend=False)
    
    return fig, chi2_fig, chi2_info


def make_spaghetti_plots(contour_text, idef, slice_indices,
                         interp_config=None, theme='light', slider_values=None,
                         show_error_bands=True, mark_closest=True):
    """Build spaghetti figures for every active slice plane.
    
    Returns
    -------
    spaghetti_figs : list of go.Figure
        Spaghetti contour figures (one per active slice plane).
    chi2_figs : list of go.Figure
        Chi2 surface figures (one per active slice plane).
    chi2_info : dict or None
        Info about the best chi2 minimum across all planes.
    """
    available = gf.list_simline_line_keys(_simline, idef or SIMLINE_DEFAULT_IDEF) if _simline else []
    entries = _parse_spaghetti_contour_spec(contour_text, available_keys=available)
    spaghetti_figs = []
    chi2_figs = []
    chi2_first = None
    for i, plane in enumerate(active_slice_planes()):
        if plane.get('active'):
            fig, chi2_fig, chi2_info = fig_spaghetti_plane(
                plane, slice_indices[i], entries, idef,
                interp_config=interp_config, theme=theme,
                slider_values=slider_values,
                show_error_bands=show_error_bands,
                mark_closest=mark_closest,
            )
        else:
            fig = placeholder_fig(theme=theme)
            chi2_fig = placeholder_fig(theme=theme)
            chi2_info = None
        spaghetti_figs.append(fig)
        chi2_figs.append(chi2_fig if chi2_fig else placeholder_fig('Need ≥1 line for χ² surface', theme=theme))
        if chi2_first is None and chi2_info is not None:
            chi2_first = dict(chi2_info)
            # Remove the surface from the summary dict (too large to store)
            chi2_first.pop('chi2_surface', None)
            chi2_first['plane'] = plane.get('title') or plane.get('id')
            chi2_first['n_lines'] = len(entries)
            chi2_first['names'] = [e['name'] for e in entries]
    return tuple(spaghetti_figs), tuple(chi2_figs), chi2_first


def _spaghetti_status_children(chi2_info, theme='light'):
    """Compact status line under the spaghetti controls."""
    t = _theme_colors(theme)
    if not chi2_info:
        return html.Span(
            'Enter at least two matched lines to mark the χ² closest point.',
            style={'fontSize': '12px', 'color': t['muted']},
        )
    names = ', '.join(chi2_info.get('names') or [])
    return html.Span([
        html.Span('χ² intersection  ', style={'fontWeight': '700', 'color': t['heading']}),
        html.Span(
            f'plane={chi2_info.get("plane", "?")}  '
            f'χ²={chi2_info["chi2"]:.4g}  '
            f'χ²/N={chi2_info["relative_chi2"]:.4g}  '
            f'(x,y)_plot=({chi2_info["x"]:.4g}, {chi2_info["y"]:.4g})',
            style={'fontSize': '12px', 'color': t['font']},
        ),
        html.Span(f'   [{names}]',
                  style={'fontSize': '11px', 'color': t['muted'], 'marginLeft': '6px'}),
    ])


def _multi_panel_layout_kw(theme='light', height=420):
    t = _theme_colors(theme)
    return dict(
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        autosize=True,
        width=None,
        height=height,
        margin=dict(l=58, r=56, t=64, b=50),
        font=ps.layout_font(t['font']),
    )


def _subplot_axis_kw(theme='light'):
    t = _theme_colors(theme)
    return dict(showgrid=True, gridcolor=t['grid'], zeroline=False,
                linecolor=t['axis_line'], tickfont=ps.tick_font(t['font']))


def _overlay_path_for_tokens(tokens):
    """HDF5 overlay file for a full or attenuation-matched token tuple."""
    if not _overlay:
        return None
    t = tuple(tokens)
    return _overlay['files'].get(t) or _overlay['by_non_atten'].get(t[:N_PARAMS - 1])


def _parse_optional_float(value):
    if value is None or value == '':
        return None
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _parse_interp_resolution(value, default):
    try:
        n = int(value)
        return max(n, 2)
    except (TypeError, ValueError):
        return default


def _parse_interp_method(value):
    allowed = {o['value'] for o in INTERP_METHOD_OPTIONS}
    return value if value in allowed else DEFAULT_INTERP_METHOD


def _interp_config(ny, nx, x_lim, y_lim, method, clip):
    return dict(
        target_shape=(_parse_interp_resolution(ny, DEFAULT_INTERP_NY),
                      _parse_interp_resolution(nx, DEFAULT_INTERP_NX)),
        x_lim=_parse_optional_float(x_lim),
        y_lim=_parse_optional_float(y_lim),
        interpolation_method=_parse_interp_method(method),
        clip_to_bounds=bool(clip and 'clip' in clip),
        log_values=True,
    )


def _parse_error_decimation(value):
    try:
        n = int(value)
        return max(n, 2)
    except (TypeError, ValueError):
        return DEFAULT_ERROR_DECIMATION


def _parse_error_metric(value):
    allowed = {o['value'] for o in ERROR_METRIC_OPTIONS}
    return value if value in allowed else ERROR_METRIC_OPTIONS[0]['value']


def _parse_error_rel_threshold(value):
    try:
        v = float(value)
        return v if np.isfinite(v) and v >= 0 else DEFAULT_ERROR_REL_THRESHOLD
    except (TypeError, ValueError):
        return DEFAULT_ERROR_REL_THRESHOLD


def _ie_analysis_config(ny, nx, x_lim, y_lim, method, clip,
                        decimation, error_metric, rel_threshold):
    cfg = _interp_config(ny, nx, x_lim, y_lim, method, clip)
    cfg['decimation_factor'] = _parse_error_decimation(decimation)
    cfg['error_metric'] = _parse_error_metric(error_metric)
    cfg['relative_threshold'] = _parse_error_rel_threshold(rel_threshold)
    return cfg


def resample_slice_grid(x_phys, y_phys, Z, xdef, ydef,
                        target_shape=(DEFAULT_INTERP_NY, DEFAULT_INTERP_NX),
                        x_lim=None, y_lim=None,
                        method=DEFAULT_INTERP_METHOD,
                        clip_to_bounds=DEFAULT_INTERP_CLIP,
                        log_values=True,
                        interpolation_method=None,
                        smoothing_order=3):
    """Resample a native 2-D slice grid (KoSens ``resampled_grid_data`` + log axes)."""
    imethod = interpolation_method if interpolation_method is not None else method
    final_x, final_y, Z_new = gi.resample_grid_2d_kosens(
        x_phys, y_phys, Z,
        x_logscale=xdef['logscale'],
        y_logscale=ydef['logscale'],
        target_shape=target_shape,
        x_lim=x_lim,
        y_lim=y_lim,
        interpolation_method=imethod,
        smoothing_order=smoothing_order,
        clip_to_bounds=clip_to_bounds,
        log_values=log_values,
    )
    return final_x, final_y, Z_new, xdef, ydef


def _native_abundance_grid(plane, slice_token, quantity):
    """Native (unresampled) 2-D abundance / diagnostic grid for one slice plane."""
    xk, yk, sk = _plane_plot_axes(plane)
    x_tokens = _grid['axis_tokens'][xk]
    y_tokens = _grid['axis_tokens'][yk]
    xdef, ydef = _param_def(xk), _param_def(yk)

    nx, ny = len(x_tokens), len(y_tokens)
    Z = np.full((ny, nx), np.nan)
    for ix, xt in enumerate(x_tokens):
        for iy, yt in enumerate(y_tokens):
            tokens = [_middle_token(p['key']) for p in PARAM_DEFS]
            tokens[_PARAM_IDX[xk]] = xt
            tokens[_PARAM_IDX[yk]] = yt
            tokens[_PARAM_IDX[sk]] = slice_token
            path = _grid['files'].get(tuple(tokens))
            if path:
                Z[iy, ix] = get_grid_scalar(path, quantity)

    x_phys = np.array([_physical_param_value(xdef['key'], t) for t in x_tokens], dtype=float)
    y_phys = np.array([_physical_param_value(ydef['key'], t) for t in y_tokens], dtype=float)
    x_phys, y_phys, Z = gi.align_grid_axes_ascending(
        x_phys, y_phys, Z, x_logscale=xdef['logscale'], y_logscale=ydef['logscale'],
    )
    return x_phys, y_phys, Z, xdef, ydef


def _native_intensity_grid(plane, slice_token, species, idef, transition_idx,
                           slider_values=None):
    """Native (unresampled) 2-D SIMLINE intensity grid for one slice plane."""
    xk, yk, sk = _plane_plot_axes(plane)
    x_tokens = _grid['axis_tokens'][xk]
    y_tokens = _grid['axis_tokens'][yk]
    xdef, ydef = _param_def(xk), _param_def(yk)

    nx, ny = len(x_tokens), len(y_tokens)
    Z = np.full((ny, nx), np.nan)
    for ix, xt in enumerate(x_tokens):
        for iy, yt in enumerate(y_tokens):
            tokens = _token_list_from_sliders(slider_values)
            tokens[_PARAM_IDX[xk]] = xt
            tokens[_PARAM_IDX[yk]] = yt
            tokens[_PARAM_IDX[sk]] = slice_token
            Z[iy, ix] = get_smli_intensity(
                tokens, species, idef, transition_idx)

    x_phys = np.array([_physical_param_value(xdef['key'], t) for t in x_tokens], dtype=float)
    y_phys = np.array([_physical_param_value(ydef['key'], t) for t in y_tokens], dtype=float)
    x_phys, y_phys, Z = gi.align_grid_axes_ascending(
        x_phys, y_phys, Z, x_logscale=xdef['logscale'], y_logscale=ydef['logscale'],
    )
    return x_phys, y_phys, Z, xdef, ydef


def fig_interpolation_comparison(result, xdef, ydef, *, title, zscale, color_map,
                                 error_metric, plot_contours, unit_label,
                                 theme='light'):
    """Three-panel original / interpolated / error figure (KoSens ``plot_interpolation_comparison``)."""
    if not result or not np.any(np.isfinite(result['original_linear'])):
        return placeholder_fig('No data for interpolation check', theme=theme)

    x_plot_n = _axis_plot_coords(result['x_native'], xdef)
    y_plot_n = _axis_plot_coords(result['y_native'], ydef)
    x_plot_f = _axis_plot_coords(result['x_fine'], xdef)
    y_plot_f = _axis_plot_coords(result['y_fine'], ydef)

    orig = result['original_linear']
    interp = result['resampled_linear']
    err = result['error_grid']
    stats = result['error_stats']

    if zscale == 'log':
        z0, z0a, z0b = _apply_zscale(orig, 'log')
        z1, z1a, z1b = _apply_zscale(interp, 'log')
        flux_cbar = f'log<sub>10</sub>({unit_label})'
    else:
        z0, z0a, z0b = _apply_zscale(orig, 'linear')
        z1, z1a, z1b = _apply_zscale(interp, 'linear')
        flux_cbar = unit_label

    if error_metric == 'relative':
        err_title = 'Relative error (%)'
        err_cbar = 'Error (%)'
    else:
        err_title = f'Absolute error ({unit_label})'
        err_cbar = f'Error ({unit_label})'

    show_lines = bool(plot_contours and 'contours' in (plot_contours or []))
    t = _theme_colors(theme)
    cmap = _parse_grid_colorscale(color_map)
    contour_kw = _contour_trace_kw(theme, show_lines=show_lines)
    if not show_lines:
        contour_kw['contours'] = dict(coloring='heatmap', showlines=False,
                                      labelfont=dict(color='white', size=9))
        contour_kw['line']['width'] = 0
    flux_z = {}
    if z0a is not None and z1a is not None:
        flux_z = dict(zmin=min(z0a, z1a), zmax=max(z0b, z1b), zauto=False)

    mean_e = stats.get('mean_error', np.nan)
    max_e = stats.get('max_error', np.nan)
    med_e = stats.get('median_error', np.nan)
    stats_note = f'mean={mean_e:.3g}  max={max_e:.3g}  median={med_e:.3g}'

    xlab, ylab = _axis_label_contour(xdef), _axis_label_contour(ydef)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Original (native grid)', 'Interpolated (resampled)', err_title],
        horizontal_spacing=0.05,
        column_widths=[1, 1, 1],
    )
    fig.add_trace(go.Contour(
        x=x_plot_n, y=y_plot_n, z=z0, colorscale=cmap,
        colorbar=_subplot_colorbar(flux_cbar, 0.308, theme, zscale),
        **flux_z, **contour_kw), row=1, col=1)
    fig.add_trace(go.Contour(
        x=x_plot_f, y=y_plot_f, z=z1, colorscale=cmap,
        colorbar=_subplot_colorbar(flux_cbar, 0.658, theme, zscale),
        **flux_z, **contour_kw), row=1, col=2)
    fig.add_trace(go.Contour(
        x=x_plot_n, y=y_plot_n, z=err, colorscale=ERROR_PANEL_CMAP,
        colorbar=_subplot_colorbar(err_cbar, 1.006, theme, zscale='linear'),
        connectgaps=False,
        hovertemplate='x=%{x:.3g}<br>y=%{y:.3g}<br>z=%{z:.3g}<extra></extra>',
    ), row=1, col=3)

    axis_kw = _subplot_axis_kw(theme)
    for col in (1, 2, 3):
        xanchor = f'x{col}' if col > 1 else 'x'
        fig.update_xaxes(title_text=xlab, row=1, col=col, **axis_kw)
        fig.update_yaxes(
            title_text=ylab if col == 1 else '',
            scaleanchor=xanchor, scaleratio=1,
            row=1, col=col, **axis_kw,
        )

    fig.update_layout(
        title=dict(text=f'{title}<br><sup style="font-size:11px">{stats_note}</sup>',
                   font=ps.title_font(t['title']), x=0.01, xanchor='left'),
        uirevision=f'interp|{zscale}|{_parse_plot_theme(theme)}',
        **_multi_panel_layout_kw(theme, height=420),
    )
    return fig


def _interpolate_segment_match(x0, x1, z0, z1, i_target, tol):
    """Match intensity along log10(x) between two samples (KoSens convention)."""
    if not np.all(np.isfinite([x0, x1, z0, z1, i_target])) or x0 <= 0 or x1 <= 0:
        return np.nan
    u0, u1 = np.log10(float(x0)), np.log10(float(x1))
    du = u1 - u0
    if abs(du) < 1e-15 * (abs(u0) + 1.0):
        if abs(z0 - i_target) <= tol:
            return float(min(x0, x1))
        return np.nan
    dz = z1 - z0
    if dz == 0:
        if abs(z0 - i_target) <= tol:
            return float(min(x0, x1))
        return np.nan
    s1 = (i_target - tol - z0) / dz
    s2 = (i_target + tol - z0) / dz
    s_lo = max(0.0, min(s1, s2))
    s_hi = min(1.0, max(s1, s2))
    if s_lo > s_hi + 1e-15:
        return np.nan
    xb = float(10.0 ** (u0 + s_lo * du))
    return xb if xb > 0 and np.isfinite(xb) else np.nan


def _x_match_scan_sorted(xs_scan, zs_scan, i_target, tol):
    if xs_scan.size == 0:
        return np.nan
    for xi, zi in zip(xs_scan, zs_scan):
        if np.isfinite(zi) and abs(zi - i_target) <= tol:
            return float(xi)
    for k in range(len(xs_scan) - 1):
        xb = _interpolate_segment_match(
            xs_scan[k], xs_scan[k + 1], zs_scan[k], zs_scan[k + 1], i_target, tol)
        if np.isfinite(xb):
            return float(xb)
    return np.nan


def _x_match_intensity_to_x(x_row, z_slice, i_target, x_prefer, *,
                            atol=0.0, rtol=0.0, scan_direction='rightward_then_leftward',
                            skip_if_nominal_at_row_max=False, skip_if_nominal_at_row_min=False,
                            x_edge_rtol=1e-12):
    if not np.isfinite(i_target) or not np.isfinite(x_prefer) or x_prefer <= 0:
        return np.nan

    order = np.argsort(x_row.astype(float))
    xs = x_row[order].astype(float)
    zs = z_slice[order].astype(float)
    fin = np.isfinite(xs) & np.isfinite(zs) & (xs > 0)
    if not np.any(fin):
        return np.nan
    xs, zs = xs[fin], zs[fin]
    x_min_row, x_max_row = float(np.min(xs)), float(np.max(xs))
    span = x_max_row - x_min_row
    edge_atol = x_edge_rtol * (span + np.finfo(float).tiny)

    if skip_if_nominal_at_row_max and x_prefer >= x_max_row - edge_atol:
        return np.nan
    if skip_if_nominal_at_row_min and x_prefer <= x_min_row + edge_atol:
        return np.nan

    tol = float(atol) + float(rtol) * abs(float(i_target))
    x_lo_bound = x_prefer * (1.0 - max(x_edge_rtol, 1e-12))
    x_hi_bound = x_prefer * (1.0 + max(x_edge_rtol, 1e-12))

    def _scan_half_row(*, right_half, descending=False):
        mask = xs >= x_lo_bound if right_half else xs <= x_hi_bound
        xs_scan, zs_scan = xs[mask], zs[mask]
        if xs_scan.size == 0:
            return np.nan
        idx = np.argsort(xs_scan)[::-1] if descending else np.argsort(xs_scan)
        return _x_match_scan_sorted(xs_scan[idx], zs_scan[idx], i_target, tol)

    if scan_direction == 'rightward':
        x_hit = _scan_half_row(right_half=True, descending=False)
        if np.isfinite(x_hit) and x_hit + edge_atol < x_lo_bound:
            return np.nan
        return x_hit
    if scan_direction == 'leftward':
        x_hit = _scan_half_row(right_half=False, descending=False)
        if np.isfinite(x_hit) and x_hit - edge_atol > x_hi_bound:
            return np.nan
        return x_hit

    x_hit = _scan_half_row(right_half=True, descending=False)
    if np.isfinite(x_hit):
        return x_hit
    z_good = zs[np.isfinite(zs)]
    if z_good.size == 0:
        return np.nan
    if (i_target < float(np.min(z_good)) - tol - 1e-15
            or i_target > float(np.max(z_good)) + tol + 1e-15):
        return np.nan
    if x_prefer <= x_min_row + edge_atol:
        return np.nan
    return _scan_half_row(right_half=False, descending=True)


def horizontal_intensity_matched_x_shift_dex(g_ref, g_atten, x_mesh_phys, *,
                                             match_atol=0.0, match_rtol=X_SHIFT_MATCH_RTOL,
                                             scan_direction=X_SHIFT_SCAN_DIRECTION,
                                             skip_x_nom_at_row_max=False,
                                             skip_x_nom_at_row_min=False,
                                             x_edge_rtol=1e-12):
    """Per-pixel log10(x_match/x_nom) dex map (KoSens grid_functions)."""
    g_ref = np.asarray(g_ref, dtype=float)
    g_atten = np.asarray(g_atten, dtype=float)
    x_mesh_phys = np.asarray(x_mesh_phys, dtype=float)
    ny, nx = g_ref.shape
    out = np.full((ny, nx), np.nan, dtype=float)
    for iy in range(ny):
        x_row = x_mesh_phys[iy, :]
        z_ref = g_ref[iy, :]
        z_atten = g_atten[iy, :]
        for ix in range(nx):
            x_nom = x_row[ix]
            i_targ = z_ref[ix]
            if not (np.isfinite(x_nom) and x_nom > 0 and np.isfinite(i_targ)):
                continue
            x_match = _x_match_intensity_to_x(
                x_row, z_atten, i_targ, x_nom,
                atol=match_atol, rtol=match_rtol,
                scan_direction=scan_direction,
                skip_if_nominal_at_row_max=skip_x_nom_at_row_max,
                skip_if_nominal_at_row_min=skip_x_nom_at_row_min,
                x_edge_rtol=x_edge_rtol,
            )
            if np.isfinite(x_match) and x_match > 0:
                out[iy, ix] = np.log10(x_match / x_nom)
    return out


def _shift_panel_colorbar_title(xdef):
    if xdef['key'] == 'crir':
        return 'log<sub>10</sub>(\u03B6<sub>match</sub>/\u03B6<sub>nom</sub>) [dex]'
    if xdef['key'] == 'density':
        return 'log<sub>10</sub>(n<sub>match</sub>/n<sub>nom</sub>) [dex]'
    if xdef['key'] == 'fuv':
        return 'log<sub>10</sub>(\u03C7<sub>match</sub>/\u03C7<sub>nom</sub>) [dex]'
    return 'log<sub>10</sub>(x<sub>match</sub>/x<sub>nom</sub>) [dex]'


def _parse_shift_rtol(value):
    try:
        v = float(value)
        return v if np.isfinite(v) and v >= 0 else X_SHIFT_MATCH_RTOL
    except (TypeError, ValueError):
        return X_SHIFT_MATCH_RTOL


def _parse_shift_scan_direction(value):
    allowed = {o['value'] for o in X_SHIFT_DIRECTION_OPTIONS}
    return value if value in allowed else X_SHIFT_SCAN_DIRECTION


def fig_triple_atten_grid(plane, slice_title, Z_ref, Z_atten, x_phys, y_phys,
                          xdef, ydef, zscale, quantity_cbar_title,
                          shift_rtol=X_SHIFT_MATCH_RTOL,
                          shift_scan_direction=X_SHIFT_SCAN_DIRECTION,
                          colorscale=DEFAULT_GRID_COLORMAP, theme='light',
                          line_contours=None):
    """Three side-by-side square panels: reference, overlay, horizontal x-shift (dex)."""
    ny, nx = Z_ref.shape
    x_mesh = np.broadcast_to(np.asarray(x_phys, dtype=float), (ny, nx)).copy()
    scan_dir = _parse_shift_scan_direction(shift_scan_direction)
    rtol = _parse_shift_rtol(shift_rtol)
    shift = horizontal_intensity_matched_x_shift_dex(
        Z_ref, Z_atten, x_mesh, match_rtol=rtol, scan_direction=scan_dir)

    x_plot = _axis_plot_coords(x_phys, xdef)
    y_plot = _axis_plot_coords(y_phys, ydef)

    with np.errstate(divide='ignore'):
        p0, z0min, z0max = _apply_zscale(Z_ref, zscale)
        p1, z1min, z1max = _apply_zscale(Z_atten, zscale)
    cbar_ref = quantity_cbar_title

    fin3 = shift[np.isfinite(shift)]
    rmax = float(np.nanpercentile(np.abs(fin3), 99.0)) if fin3.size else 0.5
    rmax = max(rmax, 0.02)
    if scan_dir == 'rightward':
        shift_cmap = 'YlOrRd'
        shift_vmin, shift_vmax = 0.0, rmax
    else:
        shift_cmap = 'RdBu_r'
        shift_vmin, shift_vmax = -rmax, rmax

    xlab, ylab = _axis_label_contour(xdef), _axis_label_contour(ydef)
    t = _theme_colors(theme)
    cmap = _parse_grid_colorscale(colorscale)
    contour_kw = _contour_trace_kw(theme, show_lines=True)
    contour_kw['contours']['labelfont'] = dict(color='white', size=9)
    contour_kw['line']['width'] = 0.6

    panel = 380
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            'Main grid (reference)',
            'Overlay grid (attenuated)',
            'Horizontal x-shift (dex)',
        ],
        horizontal_spacing=0.05,
        column_widths=[1, 1, 1],
    )
    shift_trace_kw = dict(
        zmin=shift_vmin, zmax=shift_vmax,
        **({} if scan_dir == 'rightward' else {'zmid': 0}),
    )
    flux_z = {}
    if z0min is not None and z1min is not None:
        flux_z = dict(zmin=min(z0min, z1min), zmax=max(z0max, z1max), zauto=False)
    fig.add_trace(go.Contour(
        x=x_plot, y=y_plot, z=p0, colorscale=cmap,
        colorbar=_subplot_colorbar(cbar_ref, 0.308, theme, zscale),
        **flux_z, **contour_kw), row=1, col=1)
    fig.add_trace(go.Contour(
        x=x_plot, y=y_plot, z=p1, colorscale=cmap,
        colorbar=_subplot_colorbar(cbar_ref, 0.658, theme, zscale),
        **flux_z, **contour_kw), row=1, col=2)
    fig.add_trace(go.Contour(
        x=x_plot, y=y_plot, z=shift, colorscale=shift_cmap,
        colorbar=_subplot_colorbar(_shift_panel_colorbar_title(xdef), 1.006, theme,
                                   zscale='linear'),
        **shift_trace_kw, **contour_kw), row=1, col=3)

    axis_kw = _subplot_axis_kw(theme)
    for col in (1, 2, 3):
        xanchor = f'x{col}' if col > 1 else 'x'
        fig.update_xaxes(title_text=xlab, row=1, col=col, **axis_kw)
        fig.update_yaxes(
            title_text=ylab if col == 1 else '',
            scaleanchor=xanchor, scaleratio=1,
            row=1, col=col, **axis_kw,
        )

    layout_kw = _multi_panel_layout_kw(theme, height=panel + 95)
    layout_kw.update(
        margin=dict(l=62, r=80, t=70, b=48),
        font=ps.layout_font(t['font']),
        title=dict(text=slice_title, font=ps.title_font(t['title']),
                   x=0.01, xanchor='left'),
        uirevision=f'triple|{zscale}|{_parse_plot_theme(theme)}',
    )
    fig.update_layout(**layout_kw)
    if line_contours:
        _add_contour_level_lines(fig, x_plot, y_plot, p0, line_contours, row=1, col=1)
        _add_contour_level_lines(fig, x_plot, y_plot, p1, line_contours, row=1, col=2)
    return fig


def build_slice_grid(plane, slice_token, quantity, interp_config=None):
    """Build a 2-D array Z[iy, ix] for one parameter plane at fixed ``slice_token``."""
    xk, yk, sk = _plane_plot_axes(plane)
    x_tokens = _grid['axis_tokens'][xk]
    y_tokens = _grid['axis_tokens'][yk]
    xdef, ydef = _param_def(xk), _param_def(yk)

    nx, ny = len(x_tokens), len(y_tokens)
    Z = np.full((ny, nx), np.nan)
    for ix, xt in enumerate(x_tokens):
        for iy, yt in enumerate(y_tokens):
            tokens = [_middle_token(p['key']) for p in PARAM_DEFS]
            tokens[_PARAM_IDX[xk]] = xt
            tokens[_PARAM_IDX[yk]] = yt
            tokens[_PARAM_IDX[sk]] = slice_token
            path = _grid['files'].get(tuple(tokens))
            if path:
                Z[iy, ix] = get_grid_scalar(path, quantity)

    x_phys = np.array([_physical_param_value(xdef['key'], t) for t in x_tokens], dtype=float)
    y_phys = np.array([_physical_param_value(ydef['key'], t) for t in y_tokens], dtype=float)

    x_phys, y_phys, Z = gi.align_grid_axes_ascending(
        x_phys, y_phys, Z,
        x_logscale=xdef['logscale'],
        y_logscale=ydef['logscale'],
    )

    if interp_config:
        return resample_slice_grid(x_phys, y_phys, Z, xdef, ydef, **interp_config)
    return x_phys, y_phys, Z, xdef, ydef


def build_overlay_slice_grid(plane, slice_token, quantity, interp_config=None):
    """Like ``build_slice_grid`` but reads scalars from the overlay HDF5 grid."""
    xk, yk, sk = _plane_plot_axes(plane)
    x_tokens = _grid['axis_tokens'][xk]
    y_tokens = _grid['axis_tokens'][yk]
    xdef, ydef = _param_def(xk), _param_def(yk)

    nx, ny = len(x_tokens), len(y_tokens)
    Z = np.full((ny, nx), np.nan)
    for ix, xt in enumerate(x_tokens):
        for iy, yt in enumerate(y_tokens):
            tokens = [_middle_token(p['key']) for p in PARAM_DEFS]
            tokens[_PARAM_IDX[xk]] = xt
            tokens[_PARAM_IDX[yk]] = yt
            tokens[_PARAM_IDX[sk]] = slice_token
            path = _overlay_path_for_tokens(tokens)
            if path:
                Z[iy, ix] = get_grid_scalar(path, quantity)

    x_phys = np.array([_physical_param_value(xdef['key'], t) for t in x_tokens], dtype=float)
    y_phys = np.array([_physical_param_value(ydef['key'], t) for t in y_tokens], dtype=float)

    x_phys, y_phys, Z = gi.align_grid_axes_ascending(
        x_phys, y_phys, Z,
        x_logscale=xdef['logscale'],
        y_logscale=ydef['logscale'],
    )

    if interp_config:
        return resample_slice_grid(x_phys, y_phys, Z, xdef, ydef, **interp_config)
    return x_phys, y_phys, Z, xdef, ydef


def build_slice_grids_for_quantities(plane, slice_token, quantities, interp_config=None,
                                     overlay=False):
    """Build several 2-D slice maps from one walk over the model files.

    Species abundances are integrated together from each model's dens array so
    ``Add all species`` does not repeat the HDF5 / volume-integral work.
    """
    quantities = [q for q in dict.fromkeys(quantities) if q]
    if not quantities or not plane.get('active') or not _grid:
        return {q: None for q in quantities}

    xk, yk, sk = _plane_plot_axes(plane)
    x_tokens = _grid['axis_tokens'][xk]
    y_tokens = _grid['axis_tokens'][yk]
    xdef, ydef = _param_def(xk), _param_def(yk)
    nx, ny = len(x_tokens), len(y_tokens)
    Zs = {q: np.full((ny, nx), np.nan) for q in quantities}

    species_qtys = [q for q in quantities if str(q).startswith('species:')]
    cdens_qtys = [q for q in quantities if str(q).startswith('cdens:')]
    other_qtys = [q for q in quantities if q not in species_qtys and q not in cdens_qtys]
    species_names = [q.split(':', 1)[1] for q in species_qtys]
    cdens_names = [q.split(':', 1)[1] for q in cdens_qtys]

    ixk, iyk, isk = _PARAM_IDX[xk], _PARAM_IDX[yk], _PARAM_IDX[sk]
    base_tokens = [_middle_token(p['key']) for p in PARAM_DEFS]

    for ix, xt in enumerate(x_tokens):
        for iy, yt in enumerate(y_tokens):
            tokens = list(base_tokens)
            tokens[ixk] = xt
            tokens[iyk] = yt
            tokens[isk] = slice_token
            path = (_overlay_path_for_tokens(tokens) if overlay
                    else _grid['files'].get(tuple(tokens)))
            if not path:
                continue
            if species_qtys:
                model = get_model(path)
                xs = _integrated_x_for_species(model, species_names)
                for q, sp in zip(species_qtys, species_names):
                    val = xs[sp]
                    Zs[q][iy, ix] = val
                    _scalar_cache[(path, q)] = val
            if cdens_qtys:
                try:
                    with h5py.File(path, 'r') as hf:
                        for q, sp in zip(cdens_qtys, cdens_names):
                            val = _total_column_density_from_hf(hf, sp)
                            Zs[q][iy, ix] = val
                            _scalar_cache[(path, q)] = val
                except OSError:
                    pass
            for q in other_qtys:
                Zs[q][iy, ix] = get_grid_scalar(path, q)

    x_phys = np.array([_physical_param_value(xdef['key'], t) for t in x_tokens], dtype=float)
    y_phys = np.array([_physical_param_value(ydef['key'], t) for t in y_tokens], dtype=float)
    results = {}
    for q, Z in Zs.items():
        xp, yp, Zq = gi.align_grid_axes_ascending(
            x_phys, y_phys, Z,
            x_logscale=xdef['logscale'],
            y_logscale=ydef['logscale'],
        )
        if interp_config:
            results[q] = resample_slice_grid(xp, yp, Zq, xdef, ydef, **interp_config)
        else:
            results[q] = (xp, yp, Zq, xdef, ydef)
    return results


def fig_contour_plane(plane, slice_idx, quantity, zscale,
                      shift_rtol=X_SHIFT_MATCH_RTOL,
                      shift_scan_direction=X_SHIFT_SCAN_DIRECTION,
                      interp_config=None,
                      colorscale=DEFAULT_GRID_COLORMAP, theme='light',
                      slice_grid=None, overlay_grid=None, show_lines=True):
    """Contour plot for one (x, y) slice plane with the third axis on a slider.

    ``slice_grid`` / ``overlay_grid`` are optional precomputed
    ``(x_phys, y_phys, Z, xdef, ydef)`` tuples from
    ``build_slice_grids_for_quantities``.
    """
    if not plane.get('active'):
        return placeholder_fig(theme=theme)
    if not _grid:
        return placeholder_fig('Load a grid directory', theme=theme)

    sk = plane['slice']
    slice_tokens = _grid['axis_tokens'][sk]
    if not slice_tokens:
        return placeholder_fig('No slice axis', theme=theme)

    try:
        slice_token = slice_tokens[int(slice_idx)]
    except (IndexError, TypeError, ValueError):
        slice_token = slice_tokens[0]

    sdef = _param_def(sk)
    if slice_grid is not None:
        x_phys, y_phys, Z, xdef, ydef = slice_grid
    else:
        x_phys, y_phys, Z, xdef, ydef = build_slice_grid(
            plane, slice_token, quantity, interp_config=interp_config)

    if not np.any(np.isfinite(Z)):
        return placeholder_fig('No model points for this slice', theme=theme)

    slice_disp = _param_token_disp(sdef, slice_token)
    unit = f' {sdef["unit"]}' if sdef['unit'] else ''
    qlabel = _quantity_label(quantity)
    slice_title = (f'{plane["title"]}<br>'
                   f'<sup style="font-size:11px">{sdef["name"]} = {slice_disp}{unit}'
                   f'  \u2014  {qlabel}</sup>')

    if zscale == 'log':
        cbar_title = f'log<sub>10</sub>({qlabel})'
    else:
        cbar_title = qlabel

    cmap = _parse_grid_colorscale(colorscale)
    Z_ov = None
    if overlay_grid is not None:
        _, _, Z_ov, _, _ = overlay_grid
    elif _overlay and slice_grid is None:
        _, _, Z_ov, _, _ = build_overlay_slice_grid(
            plane, slice_token, quantity, interp_config=interp_config)
    if Z_ov is not None and np.any(np.isfinite(Z_ov)):
        return fig_triple_atten_grid(
            plane, slice_title, Z, Z_ov, x_phys, y_phys, xdef, ydef, zscale, cbar_title,
            shift_rtol=shift_rtol, shift_scan_direction=shift_scan_direction,
            colorscale=cmap, theme=theme)

    Zplot, zmin, zmax = _apply_zscale(Z, zscale)

    x_plot = _axis_plot_coords(x_phys, xdef)
    y_plot = _axis_plot_coords(y_phys, ydef)

    trace_kw = _contour_trace_kw(theme, show_lines=show_lines)
    if zmin is not None:
        trace_kw = {**trace_kw, 'zmin': zmin, 'zmax': zmax, 'zauto': False}
    fig = go.Figure(go.Contour(
        x=x_plot, y=y_plot, z=Zplot,
        colorscale=cmap,
        colorbar=_contour_colorbar(cbar_title, theme=theme, zscale=zscale),
        **trace_kw,
    ))
    return _apply_square_contour_layout(
        fig, x_plot, y_plot, xdef, ydef, slice_title,
        fixed_size=True, theme=theme, zscale=zscale,
    )


def make_contour_plots(quantity, zscale, slice_indices,
                       shift_rtol=X_SHIFT_MATCH_RTOL,
                       shift_scan_direction=X_SHIFT_SCAN_DIRECTION,
                       interp_config=None,
                       colorscale=DEFAULT_GRID_COLORMAP, theme='light'):
    """Return one contour figure per active slice-plane slot.

    ``quantity`` may be a single value or a list; the first entry is plotted
    on the primary three panels.
    """
    qty = _first_str(quantity)
    if not qty:
        return tuple(placeholder_fig('Load a grid and pick a quantity', theme=theme)
                     for _ in SLICE_PLANES)
    return tuple(
        fig_contour_plane(plane, slice_indices[i], qty, zscale,
                          shift_rtol=shift_rtol, shift_scan_direction=shift_scan_direction,
                          interp_config=interp_config,
                          colorscale=colorscale, theme=theme)
        if plane.get('active') else placeholder_fig(theme=theme)
        for i, plane in enumerate(active_slice_planes())
    )


def _intensity_map_combos(species_value, transition_value, idef):
    """``(species, transition_idx)`` pairs that exist in the loaded SIMLINE grid.

    Transition dropdown indices come from the first selected species. Other
    species are matched by spectroscopic transition name when possible, then
    by row index.
    """
    combos = []
    seen = set()
    idef = idef or SIMLINE_DEFAULT_IDEF
    species_list = _as_str_list(species_value)
    trans_vals = _as_str_list(transition_value)
    if not species_list or not trans_vals:
        return combos

    primary_opts = _simline_transition_options(species_list[0], idef)
    wanted = []
    for raw in trans_vals:
        try:
            tidx = int(raw)
        except (TypeError, ValueError):
            continue
        tname = None
        if 0 <= tidx < len(primary_opts):
            tname = primary_opts[tidx].get('transition')
        wanted.append((tidx, tname))

    for sp in species_list:
        opts = _simline_transition_options(sp, idef)
        n = len(opts)
        name_to_idx = {}
        for o in opts:
            name = o.get('transition')
            if name is None:
                continue
            try:
                name_to_idx[str(name)] = int(o.get('idx', o.get('value')))
            except (TypeError, ValueError):
                continue
        for tidx, tname in wanted:
            if tname is not None and str(tname) in name_to_idx:
                mapped = name_to_idx[str(tname)]
            elif 0 <= tidx < n:
                mapped = tidx
            else:
                continue
            key = (sp, mapped)
            if key not in seen:
                seen.add(key)
                combos.append(key)
    return combos


def build_intensity_slice_grid(plane, slice_token, species, idef, transition_idx,
                               overlay=False, interp_config=None,
                               slider_values=None):
    """Build a 2-D SIMLINE intensity array for one parameter plane."""
    xk, yk, sk = _plane_plot_axes(plane)
    x_tokens = _grid['axis_tokens'][xk]
    y_tokens = _grid['axis_tokens'][yk]
    xdef, ydef = _param_def(xk), _param_def(yk)

    nx, ny = len(x_tokens), len(y_tokens)
    Z = np.full((ny, nx), np.nan)
    for ix, xt in enumerate(x_tokens):
        for iy, yt in enumerate(y_tokens):
            tokens = _token_list_from_sliders(slider_values)
            tokens[_PARAM_IDX[xk]] = xt
            tokens[_PARAM_IDX[yk]] = yt
            tokens[_PARAM_IDX[sk]] = slice_token
            Z[iy, ix] = get_smli_intensity(
                tokens, species, idef, transition_idx, overlay=overlay)

    x_phys = np.array([_physical_param_value(xdef['key'], t) for t in x_tokens], dtype=float)
    y_phys = np.array([_physical_param_value(ydef['key'], t) for t in y_tokens], dtype=float)

    x_phys, y_phys, Z = gi.align_grid_axes_ascending(
        x_phys, y_phys, Z,
        x_logscale=xdef['logscale'],
        y_logscale=ydef['logscale'],
    )

    if interp_config:
        return resample_slice_grid(x_phys, y_phys, Z, xdef, ydef, **interp_config)
    return x_phys, y_phys, Z, xdef, ydef


def fig_intensity_contour_plane(plane, slice_idx, species, idef, transition_idx, zscale,
                                shift_rtol=X_SHIFT_MATCH_RTOL,
                                shift_scan_direction=X_SHIFT_SCAN_DIRECTION,
                                interp_config=None,
                                colorscale=DEFAULT_GRID_COLORMAP, theme='light',
                                slider_values=None,
                                line_contours=None):
    """Contour plot of SIMLINE intensity on one parameter slice plane."""
    if not plane.get('active'):
        return placeholder_fig(theme=theme)
    if not _grid:
        return placeholder_fig('Load a grid or SIMLINE directory', theme=theme)
    if not _simline:
        return placeholder_fig('Load a SIMLINE directory on the Load tab', theme=theme)

    sk = plane['slice']
    slice_tokens = _grid['axis_tokens'][sk]
    if not slice_tokens:
        return placeholder_fig('No slice axis', theme=theme)

    try:
        slice_token = slice_tokens[int(slice_idx)]
    except (IndexError, TypeError, ValueError):
        slice_token = slice_tokens[0]

    sdef = _param_def(sk)
    x_phys, y_phys, Z, xdef, ydef = build_intensity_slice_grid(
        plane, slice_token, species, idef, transition_idx, interp_config=interp_config,
        slider_values=slider_values)

    if not np.any(np.isfinite(Z)):
        return placeholder_fig('No SIMLINE data for this slice', theme=theme)

    unit = _intensity_unit_label(idef)
    sp_html = format_species_html(species)
    trans_rows = _simline_transition_options(species, idef)
    try:
        tidx = int(transition_idx)
        tlabel = trans_rows[tidx]['label'] if 0 <= tidx < len(trans_rows) else str(transition_idx)
    except (TypeError, ValueError, IndexError):
        tlabel = str(transition_idx)

    if idef == 'tau':
        cbar_title = f'log<sub>10</sub>(\u03c4)' if zscale == 'log' else '\u03c4'
    elif zscale == 'log':
        cbar_title = f'log<sub>10</sub>(I) [{unit}]'
    else:
        cbar_title = f'I [{unit}]'

    slice_disp = _param_token_disp(sdef, slice_token)
    slice_unit = f' {sdef["unit"]}' if sdef['unit'] else ''
    slice_title = (f'{plane["title"]}<br>'
                   f'<sup style="font-size:11px">{sdef["name"]} = {slice_disp}{slice_unit}'
                   f'  \u2014  {sp_html} {tlabel}</sup>')

    cmap = _parse_grid_colorscale(colorscale)
    if _simline_overlay:
        _, _, Z_ov, _, _ = build_intensity_slice_grid(
            plane, slice_token, species, idef, transition_idx, overlay=True,
            interp_config=interp_config, slider_values=slider_values)
        if np.any(np.isfinite(Z_ov)):
            return fig_triple_atten_grid(
                plane, slice_title, Z, Z_ov, x_phys, y_phys, xdef, ydef, zscale, cbar_title,
                shift_rtol=shift_rtol, shift_scan_direction=shift_scan_direction,
                colorscale=cmap, theme=theme, line_contours=line_contours)

    Zplot, zmin, zmax = _apply_zscale(Z, zscale)

    x_plot = _axis_plot_coords(x_phys, xdef)
    y_plot = _axis_plot_coords(y_phys, ydef)

    trace_kw = _contour_trace_kw(theme, show_lines=True)
    if zmin is not None:
        trace_kw = {**trace_kw, 'zmin': zmin, 'zmax': zmax, 'zauto': False}
    fig = go.Figure(go.Contour(
        x=x_plot, y=y_plot, z=Zplot,
        colorscale=cmap,
        colorbar=_contour_colorbar(cbar_title, theme=theme, zscale=zscale),
        **trace_kw,
    ))
    if line_contours:
        _add_contour_level_lines(fig, x_plot, y_plot, Zplot, line_contours)
    return _apply_square_contour_layout(
        fig, x_plot, y_plot, xdef, ydef, slice_title,
        fixed_size=True, theme=theme, zscale=zscale,
    )


def make_intensity_contour_plots(species, idef, transition_idx, zscale, slice_indices,
                                 shift_rtol=X_SHIFT_MATCH_RTOL,
                                 shift_scan_direction=X_SHIFT_SCAN_DIRECTION,
                                 interp_config=None,
                                 colorscale=DEFAULT_GRID_COLORMAP, theme='light',
                                 slider_values=None,
                                 show_obs_boundary=True,
                                 extra_contour_levels=None,
                                 extra_contour_color='black'):
    """Return one SIMLINE intensity contour per active slice plane.

    ``species`` / ``transition_idx`` may be lists; the first valid
    (species, transition) pair is drawn on the primary three panels.
    """
    combos = _intensity_map_combos(species, transition_idx, idef)
    if not combos:
        return tuple(placeholder_fig('Load grids and pick species / transition',
                                     theme=theme)
                     for _ in SLICE_PLANES)
    species, tidx = combos[0]
    idef = idef or SIMLINE_DEFAULT_IDEF
    line_contours = _build_intensity_line_contours(
        idef, zscale or 'log', show_obs_boundary,
        extra_contour_levels, extra_contour_color,
    )
    return tuple(
        fig_intensity_contour_plane(
            plane, slice_indices[i], species, idef, tidx, zscale,
            shift_rtol=shift_rtol, shift_scan_direction=shift_scan_direction,
            interp_config=interp_config,
            colorscale=colorscale, theme=theme,
            slider_values=slider_values,
            line_contours=line_contours)
        if plane.get('active') else placeholder_fig(theme=theme)
        for i, plane in enumerate(active_slice_planes())
    )


def _slice_token_from_index(plane, slice_idx):
    """Resolve the fixed-axis token for a slice plane slider index."""
    sk = plane['slice']
    slice_tokens = _grid['axis_tokens'][sk]
    if not slice_tokens:
        return None
    try:
        return slice_tokens[int(slice_idx)]
    except (IndexError, TypeError, ValueError):
        return slice_tokens[0]


def _rgb_slice_title(plane, slice_token, subtitle):
    sdef = _param_def(plane['slice'])
    slice_disp = _param_token_disp(sdef, slice_token)
    unit = f' {sdef["unit"]}' if sdef['unit'] else ''
    return (f'{plane["title"]}<br>'
            f'<sup style="font-size:11px">{sdef["name"]} = {slice_disp}{unit}'
            f'  \u2014  {subtitle}</sup>')


def _default_rgb_grid_species(species_list):
    species_list = list(species_list or [])
    chosen = [s for s in RGB_DEFAULT_GRID_SPECIES if s in species_list]
    for s in species_list:
        if s not in chosen:
            chosen.append(s)
        if len(chosen) >= 3:
            break
    while len(chosen) < 3:
        chosen.append(None)
    return chosen[:3]


def _default_rgb_line_keys(idef):
    keys = gf.list_simline_line_keys(_simline, idef or SIMLINE_DEFAULT_IDEF)
    chosen = []
    for base in RGB_DEFAULT_GRID_SPECIES:
        match = rgb_phase.exact_base_key(base, keys)
        if match:
            chosen.append(match)
    for k in keys:
        if k not in chosen:
            chosen.append(k)
        if len(chosen) >= 3:
            break
    while len(chosen) < 3:
        chosen.append(None)
    return chosen[:3]


def _resolve_simline_line(line_key, idef):
    """Return ``(species, transition_idx)`` for a spectroscopic line key."""
    if not line_key or not _simline:
        return None
    lookup = gf._line_lookup(_simline, idef or SIMLINE_DEFAULT_IDEF)
    return lookup.get(line_key)


def _find_simline_base_line(base_name, idef, preferred_keys=None):
    """Find a SIMLINE line key whose bare species matches ``base_name``."""
    keys = list(preferred_keys or [])
    keys.extend(gf.list_simline_line_keys(_simline, idef or SIMLINE_DEFAULT_IDEF))
    return rgb_phase.exact_base_key(base_name, keys)


def _parse_rgb_conv_factors(*values):
    """Three positive channel conversion factors (default 1)."""
    out = []
    for v in list(values)[:3]:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = 1.0
        if not np.isfinite(f) or f <= 0:
            f = 1.0
        out.append(f)
    while len(out) < 3:
        out.append(1.0)
    return out[:3]


def _channel_conv_factor(key, channel_keys, conv_factors):
    """Conversion factor for ``key`` if it is one of the RGB channels."""
    if not key or not channel_keys or not conv_factors:
        return 1.0
    try:
        return float(conv_factors[list(channel_keys).index(key)])
    except (ValueError, IndexError, TypeError):
        return 1.0


def _grid_rgb_transition_layers(plane, slice_token, flags, interp_config,
                                species_trio=None, conv_factors=None):
    """Optional H–H₂ / C–CO / CO–JCO ratio grids from HDF5 abundances."""
    flags = set(flags or [])
    species = list((_grid or {}).get('species') or [])
    trio = list(species_trio or [])
    cfs = _parse_rgb_conv_factors(*(conv_factors or []))
    layers = []
    if 'hh2' in flags and 'H' in species and 'H2' in species:
        _, _, zh, _, _ = build_slice_grid(
            plane, slice_token, 'species:H', interp_config=interp_config)
        _, _, zh2, _, _ = build_slice_grid(
            plane, slice_token, 'species:H2', interp_config=interp_config)
        layers.append((rgb_phase.hh2_ratio(zh, zh2), '--', 'H-H2'))
    if 'cco' in flags:
        # Prefer RGB-channel C/CO (same data + conversion factors) like KoSens.
        c_sp = rgb_phase.exact_base_key('C', trio) or rgb_phase.exact_base_key('C', species)
        co_sp = rgb_phase.exact_base_key('CO', trio) or rgb_phase.exact_base_key('CO', species)
        if c_sp and co_sp:
            _, _, zc, _, _ = build_slice_grid(
                plane, slice_token, f'species:{c_sp}', interp_config=interp_config)
            _, _, zco, _, _ = build_slice_grid(
                plane, slice_token, f'species:{co_sp}', interp_config=interp_config)
            zc = zc * _channel_conv_factor(c_sp, trio, cfs)
            zco = zco * _channel_conv_factor(co_sp, trio, cfs)
            layers.append((rgb_phase.ratio_grid(zc, zco), '-', 'C-CO'))
    if 'cojco' in flags:
        co_sp = 'CO' if 'CO' in species else rgb_phase.exact_base_key('CO', species)
        jco_sp = 'JCO' if 'JCO' in species else rgb_phase.exact_base_key('JCO', species)
        if co_sp and jco_sp:
            _, _, zco, _, _ = build_slice_grid(
                plane, slice_token, f'species:{co_sp}', interp_config=interp_config)
            _, _, zjco, _, _ = build_slice_grid(
                plane, slice_token, f'species:{jco_sp}', interp_config=interp_config)
            layers.append((rgb_phase.ratio_grid(zco, zjco), '-.', 'CO-JCO'))
    return layers


def _intensity_rgb_transition_layers(plane, slice_token, flags, idef,
                                     channel_keys, interp_config, slider_values,
                                     conv_factors=None):
    """H–H₂ from HDF5; C–CO from SIMLINE intensities. No CO–JCO for lines."""
    flags = set(flags or [])
    channel_keys = list(channel_keys or [])
    cfs = _parse_rgb_conv_factors(*(conv_factors or []))
    layers = []
    if 'hh2' in flags and _grid_has_hdf5():
        species = list((_grid or {}).get('species') or [])
        if 'H' in species and 'H2' in species:
            _, _, zh, _, _ = build_slice_grid(
                plane, slice_token, 'species:H', interp_config=interp_config)
            _, _, zh2, _, _ = build_slice_grid(
                plane, slice_token, 'species:H2', interp_config=interp_config)
            layers.append((rgb_phase.hh2_ratio(zh, zh2), '--', 'H-H2'))
    if 'cco' in flags and _simline:
        c_key = _find_simline_base_line('C', idef, channel_keys)
        co_key = _find_simline_base_line('CO', idef, channel_keys)
        c_res = _resolve_simline_line(c_key, idef) if c_key else None
        co_res = _resolve_simline_line(co_key, idef) if co_key else None
        if c_res and co_res:
            _, _, zc, _, _ = build_intensity_slice_grid(
                plane, slice_token, c_res[0], idef, c_res[1],
                interp_config=interp_config, slider_values=slider_values)
            _, _, zco, _, _ = build_intensity_slice_grid(
                plane, slice_token, co_res[0], idef, co_res[1],
                interp_config=interp_config, slider_values=slider_values)
            zc = zc * _channel_conv_factor(c_key, channel_keys, cfs)
            zco = zco * _channel_conv_factor(co_key, channel_keys, cfs)
            layers.append((rgb_phase.ratio_grid(zc, zco), '-', 'C-CO'))
    return layers


def fig_grid_rgb_plane(plane, slice_idx, species_trio, transition_flags,
                       interp_config=None, theme='light', conv_factors=None):
    """RGB abundance dominance map for one grid slice plane."""
    if not plane.get('active'):
        return placeholder_fig(theme=theme)
    if not _grid or not _grid_has_hdf5():
        return placeholder_fig('Load an HDF5 grid for abundance RGB maps', theme=theme)
    species_trio = list(species_trio or [])
    if len(species_trio) < 3 or any(not s for s in species_trio[:3]):
        return placeholder_fig('Select three species for the RGB channels', theme=theme)
    cfs = _parse_rgb_conv_factors(*(conv_factors or []))

    slice_token = _slice_token_from_index(plane, slice_idx)
    if slice_token is None:
        return placeholder_fig('No slice axis', theme=theme)

    grids = []
    x_phys = y_phys = xdef = ydef = None
    for sp in species_trio[:3]:
        x_phys, y_phys, Z, xdef, ydef = build_slice_grid(
            plane, slice_token, f'species:{sp}', interp_config=interp_config)
        grids.append(Z)
    if not any(np.any(np.isfinite(g)) for g in grids):
        return placeholder_fig('No abundance data for this RGB slice', theme=theme)

    labels = [format_species_html(sp) for sp in species_trio[:3]]
    subtitle = ' / '.join(labels)
    transitions = _grid_rgb_transition_layers(
        plane, slice_token, transition_flags, interp_config,
        species_trio=species_trio[:3], conv_factors=cfs)
    x_plot = _axis_plot_coords(x_phys, xdef)
    y_plot = _axis_plot_coords(y_phys, ydef)
    title = _rgb_slice_title(plane, slice_token, subtitle)
    fig = rgb_phase.build_rgb_figure(
        grids, x_plot, y_plot,
        xlabel=_axis_label_contour(xdef),
        ylabel=_axis_label_contour(ydef),
        title=title,
        channel_labels=labels,
        transitions=transitions,
        theme_colors=_theme_colors(theme),
        conv_factors=cfs,
        width=COMPACT_FIG_WIDTH,
        height=COMPACT_FIG_HEIGHT,
    )
    xr = list(fig.layout.xaxis.range) if fig.layout.xaxis.range else None
    yr = list(fig.layout.yaxis.range) if fig.layout.yaxis.range else None
    fig.update_layout(**_compact_slice_layout_kw(
        title, _axis_label_contour(xdef), _axis_label_contour(ydef),
        theme=theme, uirevision=f'rgb|{_parse_plot_theme(theme)}',
        x_range=xr, y_range=yr, showgrid=False,
    ))
    return fig


def fig_intensity_rgb_plane(plane, slice_idx, line_keys, idef, transition_flags,
                            interp_config=None, theme='light', slider_values=None,
                            conv_factors=None):
    """RGB line-intensity dominance map for one SIMLINE slice plane."""
    if not plane.get('active'):
        return placeholder_fig(theme=theme)
    if not _grid or not _simline:
        return placeholder_fig('Load a grid and SIMLINE directory', theme=theme)
    line_keys = list(line_keys or [])
    if len(line_keys) < 3 or any(not k for k in line_keys[:3]):
        return placeholder_fig('Select three lines for the RGB channels', theme=theme)
    idef = idef or SIMLINE_DEFAULT_IDEF
    cfs = _parse_rgb_conv_factors(*(conv_factors or []))

    slice_token = _slice_token_from_index(plane, slice_idx)
    if slice_token is None:
        return placeholder_fig('No slice axis', theme=theme)

    grids = []
    x_phys = y_phys = xdef = ydef = None
    labels = []
    for key in line_keys[:3]:
        resolved = _resolve_simline_line(key, idef)
        if not resolved:
            return placeholder_fig(f'Unknown SIMLINE line: {key}', theme=theme)
        sp, tidx = resolved
        x_phys, y_phys, Z, xdef, ydef = build_intensity_slice_grid(
            plane, slice_token, sp, idef, tidx,
            interp_config=interp_config, slider_values=slider_values)
        grids.append(Z)
        labels.append(key)
    if not any(np.any(np.isfinite(g)) for g in grids):
        return placeholder_fig('No SIMLINE data for this RGB slice', theme=theme)

    transitions = _intensity_rgb_transition_layers(
        plane, slice_token, transition_flags, idef, line_keys[:3],
        interp_config, slider_values, conv_factors=cfs)
    x_plot = _axis_plot_coords(x_phys, xdef)
    y_plot = _axis_plot_coords(y_phys, ydef)
    title = _rgb_slice_title(plane, slice_token, ' / '.join(labels))
    fig = rgb_phase.build_rgb_figure(
        grids, x_plot, y_plot,
        xlabel=_axis_label_contour(xdef),
        ylabel=_axis_label_contour(ydef),
        title=title,
        channel_labels=labels,
        transitions=transitions,
        theme_colors=_theme_colors(theme),
        conv_factors=cfs,
        width=COMPACT_FIG_WIDTH,
        height=COMPACT_FIG_HEIGHT,
    )
    xr = list(fig.layout.xaxis.range) if fig.layout.xaxis.range else None
    yr = list(fig.layout.yaxis.range) if fig.layout.yaxis.range else None
    fig.update_layout(**_compact_slice_layout_kw(
        title, _axis_label_contour(xdef), _axis_label_contour(ydef),
        theme=theme, uirevision=f'rgb-int|{_parse_plot_theme(theme)}',
        x_range=xr, y_range=yr, showgrid=False,
    ))
    return fig


def make_grid_rgb_plots(species_trio, transition_flags, slice_indices,
                        interp_config=None, theme='light', conv_factors=None):
    return tuple(
        fig_grid_rgb_plane(
            plane, slice_indices[i], species_trio, transition_flags,
            interp_config=interp_config, theme=theme, conv_factors=conv_factors)
        if plane.get('active') else placeholder_fig(theme=theme)
        for i, plane in enumerate(active_slice_planes())
    )


def make_intensity_rgb_plots(line_keys, idef, transition_flags, slice_indices,
                             interp_config=None, theme='light', slider_values=None,
                             conv_factors=None):
    return tuple(
        fig_intensity_rgb_plane(
            plane, slice_indices[i], line_keys, idef, transition_flags,
            interp_config=interp_config, theme=theme, slider_values=slider_values,
            conv_factors=conv_factors)
        if plane.get('active') else placeholder_fig(theme=theme)
        for i, plane in enumerate(active_slice_planes())
    )


def fig_intensity_spectrum(values, species, idef, theme='light', int_slice_indices=None):
    """All SIMLINE line intensities for the selected model point.

    ``species`` may be a list; each selected species is overplotted.
    """
    if not _grid:
        return placeholder_fig('Load a grid or SIMLINE directory', theme=theme)
    if not _simline:
        return placeholder_fig('Load a SIMLINE directory on the Load tab', theme=theme)
    species_list = _as_str_list(species)
    if not species_list:
        return placeholder_fig('Select a species', theme=theme)

    tokens = _model_point_tokens(values, int_slice_indices)
    if tokens is None:
        return placeholder_fig('No model file for this parameter combination', theme=theme)

    idef = idef or SIMLINE_DEFAULT_IDEF
    unit = _intensity_unit_label(idef)
    ylab = _quantity_axis_label(idef)
    t = _theme_colors(theme)
    qty_word = 'optical depths' if idef == 'tau' else 'intensities'
    y_hover = '\u03c4 = %{y:.4g}' if idef == 'tau' else 'I = %{y:.4g}'

    fig = go.Figure()
    any_rows = False
    all_positive = True
    for i, sp in enumerate(species_list):
        path = smli_file(tokens, sp, idef)
        if not path:
            continue
        rows = read_smli_file(path)
        if not rows:
            continue
        any_rows = True
        freqs = np.array([r['frequency'] for r in rows], dtype=float)
        ints = np.array([r['intensity'] for r in rows], dtype=float)
        labels = [_smli_transition_label(r['transition']) for r in rows]
        color = COLORS[i % len(COLORS)]
        if not np.any(ints[np.isfinite(ints)] > 0):
            all_positive = False
        fig.add_trace(go.Scatter(
            x=freqs, y=ints, mode='lines+markers',
            marker=dict(size=5, color=color),
            line=dict(color=color, width=1.5),
            name=sp,
            text=labels,
            hovertemplate=f'{sp} %{{text}}<br>\u03bd = %{{x:.4g}} GHz<br>{y_hover}<extra></extra>',
        ))
    if not any_rows:
        return placeholder_fig('No .smli file for this model point / species', theme=theme)

    names = ', '.join(format_species_html(sp) for sp in species_list)
    fig.update_layout(
        **_base_layout(theme),
        title=dict(
            text=f'{names} line {qty_word}  [{unit}]',
            font=ps.title_font(t['title']), x=0.02, xanchor='left'),
        xaxis=dict(**_axis_style(theme), title=dict(text='Frequency (GHz)', font=ps.axis_title_font()),
                   type='linear'),
        yaxis=dict(**_axis_style(theme), title=dict(text=ylab, font=ps.axis_title_font()),
                   type='log' if idef != 'tau' and all_positive else 'linear'),
        showlegend=len(species_list) > 1,
    )
    return fig


def _obs_fit_val_err(val, err, decimals=3):
    if not np.isfinite(val):
        return '—'
    if not np.isfinite(err) or err == 0:
        return f'{val:.{decimals}g}'
    return f'{val:.{decimals}g} ± {err:.{decimals}g}'


def _obs_fit_summary_layout(result):
    """Dash-native layout for observational Gaussian fit results."""
    if result is None:
        return html.Div()

    sel = result.spectrum_selection
    sel_detail = ''
    if result.n_spectra_selected is not None and sel in ('mean', 'peak'):
        if sel == 'peak' and result.peak_n_selected is not None:
            sel_detail = (f' ({result.peak_n_selected} / {result.n_spectra} pixels')
            if result.peak_threshold is not None:
                sel_detail += f', threshold = {result.peak_threshold:.4g}'
            sel_detail += ')'
        else:
            sel_detail = f' ({result.n_spectra_selected} / {result.n_spectra} spectra)'

    cell = {'padding': '4px 10px'}
    center = {**cell, 'textAlign': 'center'}
    header_row = html.Tr([
        html.Th('Component', style={**cell, 'textAlign': 'left'}),
        html.Th('A (K)', style=center),
        html.Th('v₀ (km/s)', style=center),
        html.Th('σ (km/s)', style=center),
        html.Th('∫I dv (K km/s)', style=center),
    ], style={'backgroundColor': '#eef2f7'})

    rows = [header_row]
    errs = result.param_errors
    for j in range(result.n_components):
        k = 3 * j
        rows.append(html.Tr([
            html.Td(f'Gaussian {j + 1}', style=cell),
            html.Td(_obs_fit_val_err(result.amplitudes[j], errs[k]), style=center),
            html.Td(_obs_fit_val_err(result.centers[j], errs[k + 1]), style=center),
            html.Td(_obs_fit_val_err(result.sigmas[j], errs[k + 2]), style=center),
            html.Td(f'{float(result.integrated_intensity_per_component[j]):.4g}',
                    style=center),
        ]))

    off_err = errs[-1] if errs.size else float('nan')
    rows.append(html.Tr([
        html.Td('Offset (post-cont.)', style=cell),
        html.Td(_obs_fit_val_err(result.offset, off_err) + ' K',
                colSpan=4, style=center),
    ]))
    tot = _obs_fit_val_err(
        result.integrated_intensity_total, result.integrated_intensity_total_error)
    rows.append(html.Tr([
        html.Td('Total ∫I dv', style={**cell, 'fontWeight': '600'}),
        html.Td(f'{tot} K km/s', colSpan=4,
                style={**center, 'fontWeight': '600'}),
    ], style={'backgroundColor': '#f0f8f0'}))

    meta = [
        html.P([
            html.B('Observational fit'), ' — ',
            html.Code(os.path.basename(result.file_path)),
        ], style={'margin': '0 0 4px'}),
        html.P(f'Selection: {sel}{sel_detail}',
               style={'margin': '0 0 4px'}),
        html.P(
            f'Noise (line-free): RMS = {result.noise_sigma_rms:.4g} K, '
            f'MAD = {result.noise_sigma_mad:.4g} K',
            style={'margin': '0 0 8px'}),
    ]
    if result.mean_noise_rms is not None:
        meta.append(html.P(
            f'Mean per-spectrum noise: RMS = {result.mean_noise_rms:.4g} K, '
            f'MAD = {result.mean_noise_mad:.4g} K',
            style={'margin': '0 0 8px'}))

    return html.Div(meta + [
        html.Table(rows, style={'borderCollapse': 'collapse', 'marginTop': '4px'}),
    ], style={'fontSize': '13px', 'lineHeight': '1.55'})


def _run_obs_spectrum_fit(obs_path, obs_hdu, obs_row_mode, obs_row_index,
                          obs_v_low, obs_v_high, obs_n_gauss, do_fit):
    """Load observational spectrum; optionally run Gaussian fit."""
    path = (obs_path or '').strip()
    if not path or not os.path.isfile(os.path.expanduser(path)):
        return None, None, None

    try:
        row = osf.parse_row_selection(obs_row_mode, obs_row_index)
        v_lo = float(obs_v_low if obs_v_low is not None else DEFAULT_OBS_V_LOW)
        v_hi = float(obs_v_high if obs_v_high is not None else DEFAULT_OBS_V_HIGH)
        hdu = int(obs_hdu if obs_hdu is not None else 1)
        n_g = max(1, int(obs_n_gauss or 1))
    except (TypeError, ValueError) as exc:
        return None, None, str(exc)

    try:
        if do_fit:
            result = osf.spectrum_fitting(
                path, row, v_lo, v_hi, hdu_index=hdu, n_gaussians=n_g)
            return result.v, result.y, result
        v, y, _, _ = osf.extract_spectrum_only(path, row, v_lo, v_hi, hdu_index=hdu)
        return v, y, None
    except Exception as exc:
        return None, None, str(exc)


def _pv_spectrum_ylabel(quantity, bunit='K'):
    if quantity == 'tau':
        return '\u03c4'
    return f'T<sub>mb</sub> ({bunit})'


def _pv_spectrum_hover_y(quantity):
    if quantity == 'tau':
        return '\u03c4 = %{y:.4g}'
    return 'T<sub>mb</sub> = %{y:.4g} K'


def fig_spectra_plot(values, species, transitions, positions_text,
                     obs_path=None, obs_hdu=None, obs_row_mode='mean',
                     obs_row_index=None, obs_v_low=None, obs_v_high=None,
                     obs_n_gauss=1, obs_overlay=False, obs_fit=False,
                     pv_quantity='intensity', theme='light'):
    """SimLine PV spectra with optional observational overlay and Gaussian fit."""
    t = _theme_colors(theme)
    fig = go.Figure()
    bunit = 'K'
    pv_quantity = pv_quantity or SIMLINE_DEFAULT_PV_QUANTITY
    trace_idx = 0
    n_simline = 0
    has_simline_request = bool(species and transitions)
    simline_subtitle = ''

    if has_simline_request and _grid and _simline:
        if isinstance(transitions, (list, tuple)):
            trans_list = [tr for tr in transitions if tr]
        else:
            trans_list = [transitions] if transitions else []

        tokens = _tokens_from_values(values) if _grid else None
        positions = ss.parse_position_list(positions_text)

        if tokens is not None and trans_list:
            y_hover = _pv_spectrum_hover_y(pv_quantity)
            for transition in trans_list:
                fits_path = pv_fits_path(tokens, species, transition, quantity=pv_quantity)
                if not fits_path:
                    continue
                try:
                    result = ss.load_and_average_spectrum(fits_path, positions=positions)
                    bunit, hdr_transition = ss.pv_header_info(fits_path)
                    if pv_quantity == 'tau':
                        bunit = '\u03c4'
                except Exception:
                    continue
                tr_label = hdr_transition or _smli_transition_label(transition)
                n_simline += 1
                if positions is None:
                    velocities, spectrum, _ = result
                    fig.add_trace(go.Scatter(
                        x=velocities, y=spectrum, mode='lines',
                        line=dict(color=COLORS[trace_idx % len(COLORS)], width=1.5, shape='hv'),
                        name=f'SimLine {tr_label}',
                        hovertemplate=(
                            f'{tr_label}<br>v = %{{x:.3g}} km/s'
                            f'<br>{y_hover}<extra></extra>'
                        ),
                    ))
                    trace_idx += 1
                else:
                    velocities, spectra, selected = result
                    spectra = np.atleast_2d(spectra)
                    selected = np.atleast_1d(selected)
                    for pos, spec in zip(selected, spectra):
                        pos_name = f'{pos:.2f}"'
                        trace_name = (f'SimLine {tr_label} @ {pos_name}'
                                      if len(trans_list) > 1 else f'SimLine {pos_name}')
                        fig.add_trace(go.Scatter(
                            x=velocities, y=spec, mode='lines',
                            line=dict(color=COLORS[trace_idx % len(COLORS)],
                                      width=1.5, shape='hv'),
                            name=trace_name,
                            hovertemplate=(
                                f'{tr_label}<br>offset = {pos:.3f}"<br>v = %{{x:.3g}} km/s'
                                f'<br>{y_hover}<extra></extra>'
                            ),
                        ))
                        trace_idx += 1

            if positions is None:
                simline_subtitle = 'SimLine spatial mean'
            elif n_simline == 1:
                simline_subtitle = f'SimLine, {len(selected)} position(s)'
            else:
                simline_subtitle = f'SimLine {n_simline} transitions'

    fit_result = None
    fit_error = None
    if obs_overlay or obs_fit:
        v_obs, y_obs, fit_or_err = _run_obs_spectrum_fit(
            obs_path, obs_hdu, obs_row_mode, obs_row_index,
            obs_v_low, obs_v_high, obs_n_gauss, do_fit=obs_fit)
        if isinstance(fit_or_err, str):
            fit_error = fit_or_err
        elif fit_or_err is not None:
            fit_result = fit_or_err
            v_obs, y_obs = fit_result.v, fit_result.y

        if v_obs is not None and y_obs is not None and obs_overlay:
            sel = fit_result.spectrum_selection if fit_result else 'obs'
            fig.add_trace(go.Scatter(
                x=v_obs, y=y_obs, mode='lines',
                line=dict(color='#d62728', width=2.0),
                name=f'Observed ({sel})',
                hovertemplate='v = %{x:.3g} km/s<br>T<sub>mb</sub> = %{y:.4g} K<extra></extra>',
            ))
            trace_idx += 1

        if fit_result is not None:
            v_fit = fit_result.v
            fig.add_trace(go.Scatter(
                x=v_fit, y=fit_result.cont, mode='lines',
                line=dict(color='#888', width=1.2, dash='dot'),
                name='Obs continuum',
                hovertemplate='continuum = %{y:.4g}<extra></extra>',
            ))
            for j in range(fit_result.n_components):
                comp = fit_result.cont + fit_result.offset + fit_result.gaussian_components[j]
                fig.add_trace(go.Scatter(
                    x=v_fit, y=comp, mode='lines',
                    line=dict(color='#ff7f0e', width=1.2, dash='dash'),
                    name=f'Obs Gauss {j + 1}',
                    hovertemplate=f'Gauss {j + 1}<extra></extra>',
                ))
            fig.add_trace(go.Scatter(
                x=v_fit, y=fit_result.fit_total, mode='lines',
                line=dict(color='#2ca02c', width=2.0),
                name='Obs total fit',
                hovertemplate='fit = %{y:.4g}<extra></extra>',
            ))
            v_lo = float(obs_v_low if obs_v_low is not None else DEFAULT_OBS_V_LOW)
            v_hi = float(obs_v_high if obs_v_high is not None else DEFAULT_OBS_V_HIGH)
            for vlim, lbl in ((v_lo, 'line core low'), (v_hi, 'line core high')):
                fig.add_vline(x=vlim, line=dict(color='rgba(60,120,200,0.45)',
                                                width=1, dash='dot'))

    if not fig.data:
        if fit_error:
            return placeholder_fig(f'Observational spectrum: {fit_error}', theme=theme), None, fit_error
        if has_simline_request:
            if not _grid:
                return placeholder_fig('Load a grid or SIMLINE directory', theme=theme), None, None
            if not _simline:
                return placeholder_fig('Load a SIMLINE directory on the Load tab',
                                       theme=theme), None, None
            return placeholder_fig('Select species and transition(s)', theme=theme), None, None
        if obs_path and str(obs_path).strip():
            return placeholder_fig('Enter a valid observational FITS path', theme=theme), None, None
        return placeholder_fig('Load SimLine data and/or an observational FITS cube',
                              theme=theme), None, None

    sp_html = format_species_html(species) if species else ''
    if n_simline == 1 and species:
        title = f'{sp_html} spectrum  [{simline_subtitle}]'
    elif n_simline > 1:
        title = f'{sp_html} — {n_simline} SimLine transitions'
    elif fit_result is not None:
        title = f'Observed spectrum — {os.path.basename(fit_result.file_path)}'
    elif obs_overlay:
        title = 'Observed + SimLine spectra'
    else:
        title = 'Spectra'

    if obs_overlay and n_simline > 0:
        title += ' + observation'
    elif fit_result is not None and not obs_overlay:
        title += ' (fit only)'

    layout_kw = {
        **_base_layout(theme),
        'height': 380,
        'title': dict(text=title, font=ps.title_font(t['title']),
                      x=0.02, xanchor='left'),
        'xaxis': dict(**_axis_style(theme),
                      title=dict(text='Velocity (km/s)', font=ps.axis_title_font()),
                      type='linear'),
        'yaxis': dict(**_axis_style(theme),
                      title=dict(text=_pv_spectrum_ylabel(pv_quantity, bunit),
                                 font=ps.axis_title_font()),
                      type='linear'),
        'showlegend': len(fig.data) > 1,
    }
    fig.update_layout(**layout_kw)
    summary = _obs_fit_summary_layout(fit_result) if fit_result else html.Div()
    if fit_error and not fit_result:
        summary = html.Span(fit_error, style={'color': '#d62728'})
    return fig, summary, fit_error


def fig_simline_spectrum(values, species, transitions, positions_text, theme='light'):
    """Backward-compatible wrapper (SimLine only, no observation)."""
    fig, _, _ = fig_spectra_plot(
        values, species, transitions, positions_text, theme=theme)
    return fig


def fig_simline_pv(values, species, transition, pos_min, pos_max, zscale,
                   colorscale, pv_quantity='intensity', theme='light'):
    """Position-velocity diagram from a SimLine PV FITS cube."""
    if not _grid:
        return placeholder_fig('Load a grid or SIMLINE directory', theme=theme)
    if not _simline:
        return placeholder_fig('Load a SIMLINE directory on the Load tab', theme=theme)
    if isinstance(transition, (list, tuple)):
        transition = transition[0] if transition else None
    if not species or not transition:
        return placeholder_fig('Select species and transition', theme=theme)

    tokens = _tokens_from_values(values)
    if tokens is None:
        return placeholder_fig('No model file for this parameter combination', theme=theme)

    pv_quantity = pv_quantity or SIMLINE_DEFAULT_PV_QUANTITY
    fits_path = pv_fits_path(tokens, species, transition, quantity=pv_quantity)
    if not fits_path:
        return placeholder_fig('No PV FITS file for this transition', theme=theme)

    pos_range = None
    if pos_min is not None and pos_max is not None:
        try:
            p0, p1 = float(pos_min), float(pos_max)
            if np.isfinite(p0) and np.isfinite(p1):
                pos_range = (min(p0, p1), max(p0, p1))
        except (TypeError, ValueError):
            pos_range = None

    try:
        positions, velocities, data, header = ss.load_pv_diagram(
            fits_path, position_range=pos_range)
    except Exception as exc:
        return placeholder_fig(f'Could not read PV FITS: {exc}', theme=theme)

    zplot = np.asarray(data, dtype=float)
    bunit = ss.brightness_unit_from_header(header)
    if pv_quantity == 'tau':
        zplot, zmin, zmax = _apply_zscale(zplot, 'linear')
        z_label = '\u03c4'
        z_hover = '\u03c4 = %{z:.4g}'
        cbar_zscale = 'linear'
    elif zscale == 'log':
        zplot, zmin, zmax = _apply_zscale(zplot, 'log')
        z_label = f'log<sub>10</sub>(T<sub>mb</sub>) ({bunit})'
        z_hover = 'log<sub>10</sub> T<sub>mb</sub> = %{z:.4g}'
        cbar_zscale = 'log'
    else:
        zplot, zmin, zmax = _apply_zscale(zplot, 'linear')
        z_label = f'T<sub>mb</sub> ({bunit})'
        z_hover = 'T<sub>mb</sub> = %{z:.4g}'
        cbar_zscale = 'linear'
    tr_label = ss.transition_from_header(header) or _smli_transition_label(transition)
    sp_html = format_species_html(species)
    t = _theme_colors(theme)

    heat_kw = dict(
        x=positions, y=velocities, z=zplot,
        colorscale=colorscale or 'Inferno',
        colorbar=_contour_colorbar(z_label, theme=theme, zscale=cbar_zscale,
                                   compact=False),
        hovertemplate=(
            'offset = %{x:.3f}"<br>v = %{y:.3g} km/s'
            f'<br>{z_hover}<extra></extra>'
        ),
    )
    if zmin is not None:
        heat_kw.update(zmin=zmin, zmax=zmax, zauto=False)
    fig = go.Figure(go.Heatmap(**heat_kw))
    layout_kw = {
        **_base_layout(theme),
        'height': 520,
        'uirevision': f'pv|{cbar_zscale}|{_parse_plot_theme(theme)}',
        'title': dict(
            text=f'PV diagram — {sp_html} {tr_label}',
            font=ps.title_font(t['title']), x=0.02, xanchor='left'),
        'xaxis': dict(**_axis_style(theme),
                      title=dict(text='Position offset (arcsec)', font=ps.axis_title_font(t['font'])),
                      type='linear'),
        'yaxis': dict(**_axis_style(theme),
                      title=dict(text='Velocity (km/s)', font=ps.axis_title_font(t['font'])),
                      type='linear'),
    }
    fig.update_layout(**layout_kw)
    return fig


def _tokens_key(tokens):
    return tuple(int(t) for t in tokens)


def _filepath_for_tokens(tokens):
    """Resolve an HDF5 path from a token tuple (main grid, then overlay)."""
    if not tokens:
        return None
    key = _tokens_key(tokens)
    if _grid:
        path = _grid.get('files', {}).get(key)
        if path:
            return path
    if _overlay:
        path = _overlay.get('files', {}).get(key)
        if path:
            return path
    return None


def _model_profile_label(tokens, include_atten=True):
    """Legend label from model token tuple: n_H, FUV χ, and ζ (optional atten tag)."""
    if not tokens or len(tokens) < N_PARAMS:
        return 'model'
    dens = _physical_param_value('density', tokens[_PARAM_IDX['density']], tokens=tokens)
    fuv = _physical_param_value('fuv', tokens[_PARAM_IDX['fuv']], tokens=tokens)
    zeta = _physical_param_value('crir', tokens[_PARAM_IDX['crir']], tokens=tokens)
    parts = [
        f'n<sub>H</sub> = {_sci_label(dens)} cm\u207b\u00b3',
        f'\u03c7 = {_sci_label(fuv)}',
        f'\u03b6 = {_sci_label(zeta)} s\u207b\u00b9',
    ]
    if include_atten:
        parts.append(f'atten {_atten_disp(tokens[_PARAM_IDX["atten"]])}')
    return ', '.join(parts)


def _cr_atten_profile_label(tokens):
    """Legend label for one CR attenuation model profile."""
    return _model_profile_label(tokens, include_atten=True)


def _overlay_variants_for_sliders(values):
    """All overlay models matching the current density / mass / FUV / metallicity."""
    if not _overlay or not _grid:
        return []
    tokens = _tokens_from_values(values)
    if tokens is None:
        return []
    prefix = tuple(tokens[:4])
    out = []
    for tok in _overlay['files']:
        if tuple(tok[:4]) == prefix:
            out.append(list(tok))
    return sorted(out)


def _plotly_log_range(vmin, vmax):
    """Plotly log-axis ``range`` is in log10(data) units, not linear data values."""
    return [float(np.log10(vmin)), float(np.log10(vmax))]


def _cr_atten_profile_curve(model, x_axis='nh2', extrapolate=True):
    """Return (x, zeta_h2) arrays for one loaded model."""
    if model is None:
        return None, None
    cosray = model.get('cosray')
    if cosray is None:
        return None, None
    zeta = cra.zeta_h2_from_cosray(cosray)
    if x_axis == 'av':
        x = model.get('av')
        x_label_kind = 'av'
    else:
        x = model.get('nh2_profile')
        x_label_kind = 'nh2'
    if x is None:
        return None, None
    x, zeta = _align_depth_profiles(x, zeta)
    x = np.asarray(x, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    mask = np.isfinite(x) & np.isfinite(zeta) & (x > 0) & (zeta > 0)
    x, zeta = x[mask], zeta[mask]
    if x.size == 0:
        return None, None
    if extrapolate and x_label_kind == 'nh2':
        x, zeta = cra.extrapolate_profile_loglog(x, zeta)
    return x, zeta


def _add_padovani_band(fig, n_h2, zeta, color, name, show_legend=True):
    """Add a Padovani reference line and ± factor-of-2 band."""
    z_lo = zeta / 2.0
    z_hi = zeta * 2.0
    fig.add_trace(go.Scatter(
        x=n_h2, y=zeta, mode='lines',
        line=dict(color=color, width=2),
        name=name,
        legendgroup=name,
        showlegend=show_legend,
        hovertemplate='N<sub>H₂</sub> = %{x:.3g} cm⁻²<br>ζ = %{y:.3g} s⁻¹<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([n_h2, n_h2[::-1]]),
        y=np.concatenate([z_hi, z_lo[::-1]]),
        fill='toself',
        fillcolor=_hex_to_rgba(color, 0.25),
        line=dict(width=0),
        showlegend=False,
        legendgroup=name,
        hoverinfo='skip',
    ))


def _hex_to_rgba(color, alpha):
    named = {
        'blue': '#1f77b4', 'green': '#2ca02c', 'orange': '#ff7f0e',
        'red': '#d62728',
    }
    hex_color = named.get(color, color)
    if isinstance(hex_color, str) and hex_color.startswith('#') and len(hex_color) == 7:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f'rgba({r},{g},{b},{alpha})'
    return f'rgba(128,128,128,{alpha})'


def fig_cr_attenuation(profiles, active_keys, x_axis='nh2', show_padovani=True,
                       extrapolate=True, stopping_rate=None, show_threshold=True,
                       theme='light'):
    """ζ_H2 attenuation profiles with optional Padovani reference bands."""
    t = _theme_colors(theme)
    fig = go.Figure()
    stopping_rate = float(stopping_rate or cra.DEFAULT_STOPPING_RATE)
    x_axis = x_axis or 'nh2'
    active = set(active_keys or [])

    if show_padovani and x_axis == 'nh2':
        n_ref, (z_l, z_h, z_u) = cra.padovani_reference_grid()
        for _tag, _coef, color, name in cra.PADOVANI_MODELS:
            zeta = {'L': z_l, 'H': z_h, 'U': z_u}[_tag]
            _add_padovani_band(fig, n_ref, zeta, color, name)

    threshold_x = None
    if show_threshold and x_axis == 'nh2' and stopping_rate > 0:
        # Match CR_atten_plot.ipynb: axvline(stopping_rate), not profile-derived N_H2.
        threshold_x = float(stopping_rate)

    n_plotted = 0
    for entry in profiles or []:
        key = entry.get('key')
        if key not in active:
            continue
        path = _filepath_for_tokens(entry.get('tokens'))
        if not path:
            continue
        try:
            model = get_model(path)
        except Exception:
            continue
        x, zeta = _cr_atten_profile_curve(model, x_axis=x_axis, extrapolate=extrapolate)
        if x is None:
            continue
        color = COLORS[n_plotted % len(COLORS)]
        label = _cr_atten_profile_label(entry.get('tokens'))
        fig.add_trace(go.Scatter(
            x=x, y=zeta, mode='lines',
            line=dict(color=color, width=3),
            name=label,
            hovertemplate=(
                'x = %{x:.3g}<br>ζ<sub>H₂</sub> = %{y:.3g} s⁻¹<extra></extra>'
            ),
        ))
        n_plotted += 1

    if not fig.data:
        msg = 'Add model profiles with the buttons above'
        if not _grid:
            msg = 'Load a grid directory on the Load tab'
        return placeholder_fig(msg, theme=theme)

    if show_threshold and threshold_x is not None and np.isfinite(threshold_x):
        fig.add_vline(
            x=threshold_x,
            line=dict(color='black', dash='dashdot', width=1.5),
        )

    if x_axis == 'av':
        x_title = 'A<sub>V</sub> (mag)'
        x_type = 'log'
        x_range = _plotly_log_range(1e-2, 1e3)
    else:
        x_title = 'N<sub>H₂</sub> (cm⁻²)'
        x_type = 'log'
        x_range = _plotly_log_range(cra.DEFAULT_NH2_MIN, cra.DEFAULT_NH2_MAX)

    y_range = _plotly_log_range(cra.DEFAULT_ZETA_MIN, cra.DEFAULT_ZETA_MAX)

    layout_kw = {
        **_base_layout(theme),
        'height': 560,
        'title': dict(
            text='Cosmic-ray attenuation profiles',
            font=ps.title_font(t['title']), x=0.02, xanchor='left'),
        'xaxis': dict(**_axis_style(theme),
                      title=dict(text=x_title, font=ps.axis_title_font()),
                      type=x_type, range=x_range),
        'yaxis': dict(**_axis_style(theme),
                      title=dict(text='ζ<sub>H₂</sub> (s⁻¹)', font=ps.axis_title_font()),
                      type='log',
                      range=y_range),
        'showlegend': True,
        'legend': dict(font=ps.legend_font()),
    }
    fig.update_layout(**layout_kw)
    return fig


def _load_model_pair(values):
    """Return (model, overlay) for the selected grid point, or (None, None)."""
    if not _grid:
        return None, None
    filepath = current_file(values)
    if filepath is None:
        return None, None
    model = get_model(filepath)
    ov_path = overlay_file(values)
    overlay = get_model(ov_path) if ov_path else None
    return model, overlay


def make_profile_plots(values, xvar, xscale, yscale, custom_species, theme='light',
                       av_range=DEFAULT_AV_RANGE):
    av_range = _parse_av_range(av_range)
    model, overlay = _load_model_pair(values)
    if model is None:
        if _grid and _grid.get('simline_only'):
            bad = placeholder_fig('Depth profiles require an HDF5 grid directory', theme=theme)
        elif _grid and _grid.get('chem_only'):
            bad = placeholder_fig('Depth profiles require an HDF5 model grid', theme=theme)
        elif not _grid:
            bad = placeholder_fig(theme=theme)
        else:
            bad = placeholder_fig('No model file for this parameter combination', theme=theme)
        return (bad,) * 4
    x_cross = find_h_h2_transition(model, xvar)
    figs = [
        fig_tgas(model, overlay, xvar, xscale, yscale, theme=theme, av_range=av_range),
        fig_h_h2(model, overlay, xvar, xscale, yscale, theme=theme, av_range=av_range),
        fig_cplus_c_co(model, overlay, xvar, xscale, yscale, theme=theme, av_range=av_range),
        fig_custom(model, overlay, xvar, xscale, yscale, custom_species, theme=theme,
                   av_range=av_range),
    ]
    for f in figs:
        add_h_h2_vline(f, x_cross, theme=theme)
    return tuple(figs)


def make_thermal_plots(values, xvar, xscale, yscale, theme='light',
                       av_range=DEFAULT_AV_RANGE):
    av_range = _parse_av_range(av_range)
    model, overlay = _load_model_pair(values)
    if model is None:
        if _grid and _grid.get('simline_only'):
            bad = placeholder_fig('Heating/cooling profiles require an HDF5 grid directory', theme=theme)
        elif _grid and _grid.get('chem_only'):
            bad = placeholder_fig('Heating/cooling profiles require an HDF5 model grid', theme=theme)
        elif not _grid:
            bad = placeholder_fig(theme=theme)
        else:
            bad = placeholder_fig('No model file for this parameter combination', theme=theme)
        return (bad,) * 3
    return (
        fig_thermal(model, overlay, xvar, xscale, yscale, theme=theme, av_range=av_range),
        fig_heat_breakdown(model, overlay, xvar, xscale, yscale, theme=theme, av_range=av_range),
        fig_cool_breakdown(model, overlay, xvar, xscale, yscale, theme=theme, av_range=av_range),
    )


# --- App layout ---------------------------------------------------------------

app = dash.Dash(
    __name__,
    title='PRISMA - PDR Results Interactive Simulation Mapping & Analysis',
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
    suppress_callback_exceptions=True,
)
server = app.server

_RADIO = dict(labelStyle={'display': 'block', 'marginBottom': '3px', 'fontSize': '13px'})
_SEG = dict(
    className='seg-control',
    inputClassName='seg-input',
    labelClassName='seg-btn',
    labelStyle={'display': 'inline-flex', 'margin': '0'},
)
_SEG_STACK = dict(
    className='seg-control seg-stack',
    inputClassName='seg-input',
    labelClassName='seg-btn',
    labelStyle={'display': 'inline-flex', 'margin': '0'},
)

_SCALE_OPTIONS = [
    {'label': html.Span('log', className='seg-opt'), 'value': 'log'},
    {'label': html.Span('linear', className='seg-opt'), 'value': 'linear'},
]

_AV_RANGE_OPTIONS = [
    {'label': html.Span('full HDF5', className='seg-opt'), 'value': 'full'},
    {'label': html.Span(f'floor ≥ {AV_FLOOR:g}', className='seg-opt'), 'value': 'floor'},
]

_XVAR_OPTIONS = [
    {'label': html.Span('Aᵥ (mag)', className='seg-opt'), 'value': 'Av'},
    {'label': html.Span('n_H (cm⁻³)', className='seg-opt'), 'value': 'nH'},
]

_CTRL_LABEL = {'fontWeight': '600', 'fontSize': '13px', 'display': 'block', 'marginBottom': '5px'}
_ALL_BTN_STYLE = {'marginTop': '22px'}
_CTRL_BOX = {'flex': '1', 'minWidth': '110px', 'marginRight': '18px'}
_GRAPH_CFG = {
    'toImageButtonOptions': {'format': 'png', 'scale': 2},
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'lasso2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian',
        'hoverCompareCartesian', 'toggleSpikelines',
    ],
}
_SLICE_GRAPH_CFG = {**_GRAPH_CFG, 'responsive': False}
_TAB_STYLE = {}
_TAB_SEL = {}
_NESTED_TAB_STYLE = {}
_NESTED_TAB_SEL = {}
_PAGE_INTRO = {'margin': '0 0 14px', 'fontSize': '13px', 'lineHeight': '1.5',
               'color': 'inherit', 'opacity': 0.88}
_PANEL_ROW = {
    'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
    'padding': '12px 18px', 'marginBottom': '12px',
}
_INPUT_STYLE = {
    'padding': '8px 10px', 'fontSize': '13px',
}


def _advanced_details(summary, *children):
    """Collapsible card for advanced / secondary controls."""
    return html.Details(
        [html.Summary(summary),
         html.Div(list(children), className='kosma-details-body')],
        className='kosma-details',
        open=False,
    )


def _load_card(title, subtitle, body, *, accent='blue'):
    """Neutral load-page card with a coloured left accent."""
    accent_cls = {
        'blue': '',
        'green': ' accent-green',
        'purple': ' accent-purple',
        'overlay': ' accent-overlay',
    }.get(accent, '')
    return html.Div([
        html.Div([
            html.Span(title, className='load-card-title'),
            html.Span(f' — {subtitle}', className='load-card-sub') if subtitle else None,
        ], className='load-card-head'),
        body,
    ], className=f'load-card{accent_cls}')



def _slider_block(d):
    """One labelled slider for parameter index d (hidden until grid loads)."""
    p = PARAM_DEFS[d]
    unit = f'  ({p["unit"]})' if p['unit'] else ''
    return html.Div(id={'role': 'slider-wrap', 'idx': d},
                    style={'display': 'none', 'flex': '1 1 240px',
                           'minWidth': '210px', 'marginRight': '24px',
                           'marginBottom': '8px'},
                    children=[
        html.Label(f'{p["name"]}{unit}', style=_CTRL_LABEL),
        dcc.Slider(id=f'slider-{d}', min=0, max=1, step=1, value=0, marks={},
                   tooltip={'placement': 'top', 'always_visible': False}),
        html.Div(id=f'slabel-{d}',
                 style={'textAlign': 'center', 'color': p['color'],
                        'fontSize': '12px', 'marginTop': '2px', 'fontWeight': '600'}),
    ])


def _shift_control_row(prefix=''):
    """Attenuation x-shift controls; ``prefix`` distinguishes Grid vs Intensities tab IDs."""
    scan_id = f'{prefix}shift-scan-direction'
    rtol_id = f'{prefix}shift-match-rtol'
    return html.Div([
        html.Div([
            html.Label('X-shift scan direction', style=_CTRL_LABEL),
            dcc.Dropdown(
                id=scan_id, options=X_SHIFT_DIRECTION_OPTIONS,
                value=X_SHIFT_SCAN_DIRECTION, clearable=False,
                className='kosma-dropdown', style={'fontSize': '13px'}),
        ], style={'flex': '2', 'minWidth': '240px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Intensity match tolerance (rtol)', style=_CTRL_LABEL),
            dcc.Input(id=rtol_id, type='number', value=X_SHIFT_MATCH_RTOL,
                      min=0, max=1, step=0.005,
                      className='kosma-input',
                      style={**_INPUT_STYLE, 'width': '100px'}),
            html.Span('  relative band around target intensity',
                      className='kosma-muted',
                      style={'fontSize': '11px', 'marginLeft': '8px'}),
        ], style={'flex': '1.2', 'minWidth': '200px'}),
    ], className='kosma-panel', style=_PANEL_ROW)


def _int_contour_overlay_row():
    """Observational boundary and optional user contour levels for intensity grids."""
    return html.Div([
        html.Div([
            html.Label('Observational boundary', style=_CTRL_LABEL),
            dcc.Checklist(
                id='int-obs-boundary',
                options=[{'label': f'  {INT_OBS_BOUNDARY_JTEMP:g} K km/s detection limit (red)',
                          'value': 'show'}],
                value=['show'],
                style={'fontSize': '13px'},
            ),
            html.Span('  jtemp only — KoSens-style observational limit',
                      className='kosma-muted',
                      style={'fontSize': '11px', 'marginLeft': '8px'}),
        ], style={'flex': '1.6', 'minWidth': '280px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Extra contour levels', style=_CTRL_LABEL),
            dcc.Input(
                id='int-extra-contours', type='text', value='',
                placeholder='e.g. 0.5, 1, 5  (physical units of quantity)',
                className='kosma-input',
                style={**_INPUT_STYLE, 'width': '100%'}),
            html.Span('  comma-separated; drawn on all slice panels',
                      className='kosma-muted',
                      style={'fontSize': '11px', 'marginTop': '4px',
                             'display': 'block'}),
        ], style={'flex': '2', 'minWidth': '260px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Extra contour color(s)', style=_CTRL_LABEL),
            dcc.Input(
                id='int-extra-contour-color', type='text', value='',
                placeholder='empty = auto palette; or black, #00ff00, cyan',
                className='kosma-input',
                style={**_INPUT_STYLE, 'width': '100%'}),
            html.Span('  one colour for all levels, or comma-separated per level',
                      className='kosma-muted',
                      style={'fontSize': '11px', 'marginTop': '4px',
                             'display': 'block'}),
        ], style={'flex': '1.2', 'minWidth': '180px'}),
    ], className='kosma-panel', style=_PANEL_ROW)


def _rgb_channel_dropdowns(prefix, placeholder):
    """Three channel selectors + conversion factors for an RGB phase diagram."""
    boxes = []
    for i, meta in enumerate(RGB_CHANNEL_COLORS):
        boxes.append(html.Div([
            html.Label(meta['label'], style={**_CTRL_LABEL, 'color': meta['color']}),
            dcc.Dropdown(
                id=f'{prefix}rgb-sp{i}',
                options=[], value=None, clearable=False,
                placeholder=placeholder,
                style={'fontSize': '13px'}),
            html.Label('Conversion factor',
                       style={**_CTRL_LABEL, 'fontSize': '11px', 'marginTop': '6px'}),
            dcc.Input(
                id=f'{prefix}rgb-cf{i}',
                type='number', value=1, min=0, step='any',
                debounce=True,
                style={**_INPUT_STYLE, 'width': '100%'}),
        ], style={'flex': '1', 'minWidth': '150px',
                  'marginRight': '12px' if i < 2 else '0'}))
    return boxes


def _rgb_slice_panel(slot_id, *, prefix, plot_id):
    """One RGB panel with its own third-axis slider (same pattern as contour slices)."""
    return html.Div([
        html.Div([
            html.Label(id=f'{prefix}slice-fixed-label-{slot_id}',
                       style={**_CTRL_LABEL, 'fontSize': '12px'}),
            dcc.Slider(id=f'{prefix}slice-slider-{slot_id}', min=0, max=1, step=1,
                       value=0, marks={},
                       tooltip={'placement': 'top', 'always_visible': False}),
            html.Div(id=f'{prefix}slice-label-{slot_id}',
                     style={'textAlign': 'center', 'fontSize': '11px',
                            'marginTop': '2px', 'fontWeight': '600'}),
        ], style={'padding': '0 4px 8px'}),
        dcc.Graph(id=plot_id, figure=placeholder_fig(),
                  config=_SLICE_GRAPH_CFG, style=_SLICE_GRAPH_STYLE),
    ], id=f'{prefix}panel-{slot_id}', style=_SLICE_PANEL_ROW)


def _rgb_grid_controls():
    """Abundance RGB phase-diagram controls (Grid slices tab)."""
    return html.Div([
        html.H4('RGB phase diagram',
                style={'fontSize': '14px', 'margin': '18px 0 6px', 'fontWeight': '600'}),
        html.P('Fractional abundance dominance of three species (KoSens-style RGB map). '
               'Per-channel conversion factors scale each species before the mix '
               '(e.g. mass-density weights). Optional white contours mark chemistry transitions.',
               style={**_PAGE_INTRO, 'marginBottom': '8px'}),
        html.Div(
            _rgb_channel_dropdowns('grid-', 'Load a grid\u2026')
            + [html.Div([
                html.Label('Transition contours', style=_CTRL_LABEL),
                dcc.Checklist(
                    id='grid-rgb-transitions',
                    options=RGB_GRID_TRANSITION_OPTIONS,
                    value=[],
                    style={'fontSize': '13px'}),
            ], style={'flex': '1.4', 'minWidth': '220px'})],
            className='kosma-panel',
            style={'display': 'flex', 'alignItems': 'flex-start',
                   'padding': '12px 18px', 'marginBottom': '12px', 'flexWrap': 'wrap'}),
        html.Div(
            [_rgb_slice_panel(p['id'], prefix='rgb-', plot_id=f'plot-rgb-{p["id"]}')
             for p in SLICE_PLANES],
            id='rgb-panels-wrap',
            style=_SLICE_PANELS_ROW_STYLE,
        ),
    ])


def _rgb_intensity_controls():
    """Line-intensity RGB phase-diagram controls (Intensities tab)."""
    return html.Div([
        html.H4('RGB phase diagram',
                style={'fontSize': '14px', 'margin': '18px 0 6px', 'fontWeight': '600'}),
        html.P('Fractional intensity dominance of three SIMLINE lines. '
               'Per-channel conversion factors scale each line before the mix '
               '(and the C\u2194CO intensity contour). '
               'H\u2194H\u2082 uses HDF5 abundances when a PDR grid is loaded; '
               'CO\u2194JCO is unavailable for intensities (no ice line).',
               style={**_PAGE_INTRO, 'marginBottom': '8px'}),
        html.Div(
            _rgb_channel_dropdowns('int-', 'Load SIMLINE\u2026')
            + [html.Div([
                html.Label('Transition contours', style=_CTRL_LABEL),
                dcc.Checklist(
                    id='int-rgb-transitions',
                    options=RGB_INT_TRANSITION_OPTIONS,
                    value=[],
                    style={'fontSize': '13px'}),
            ], style={'flex': '1.6', 'minWidth': '260px'})],
            className='kosma-panel',
            style={'display': 'flex', 'alignItems': 'flex-start',
                   'padding': '12px 18px', 'marginBottom': '12px', 'flexWrap': 'wrap'}),
        html.Div(
            [_rgb_slice_panel(p['id'], prefix='int-rgb-',
                              plot_id=f'plot-int-rgb-{p["id"]}')
             for p in SLICE_PLANES],
            id='int-rgb-panels-wrap',
            style=_SLICE_PANELS_ROW_STYLE,
        ),
    ])


def _spaghetti_controls():
    """KoSens-style multi-line spaghetti / χ² intersection (Intensities tab)."""
    example = (
        'CO(1-0) = 1.2\n'
        'CO(2-1) = 3.5 \u00b1 0.7\n'
        'C+(158um) = 5.0, 1.0'
    )
    return html.Div([
        html.H4('Spaghetti contours (observed intensities)',
                style={'fontSize': '14px', 'margin': '18px 0 6px', 'fontWeight': '600'}),
        html.P(
            'Overlay model = observed intensity contours for several SIMLINE lines '
            'on each parameter slice (KoSens spaghetti_plot_grid). Optional error '
            'bands default to 20% of the level; with two or more lines the χ² '
            'closest point in parameter space is marked (find_nearest_contour_points).',
            style={**_PAGE_INTRO, 'marginBottom': '8px'},
        ),
        html.Div([
            html.Div([
                html.Label('Lines (optional multi-select helper)', style=_CTRL_LABEL),
                dcc.Dropdown(
                    id='int-spaghetti-lines',
                    options=[], value=[], multi=True,
                    placeholder='Load SIMLINE\u2026 then pick lines to seed the text box',
                    style={'fontSize': '13px'}),
            ], style={'flex': '1.4', 'minWidth': '240px', 'marginRight': '18px'}),
            html.Div([
                html.Label('Observed intensities', style=_CTRL_LABEL),
                dcc.Textarea(
                    id='int-spaghetti-contours',
                    value='',
                    placeholder=example,
                    className='kosma-input',
                    style={**_INPUT_STYLE, 'width': '100%', 'height': '96px',
                           'fontFamily': 'monospace', 'resize': 'vertical'},
                ),
                html.Span(
                    '  JSON or lines: Name = value [± err]. Names must match SIMLINE keys.',
                    className='kosma-muted',
                    style={'fontSize': '11px', 'marginTop': '4px', 'display': 'block'},
                ),
            ], style={'flex': '2', 'minWidth': '280px', 'marginRight': '18px'}),
            html.Div([
                html.Label('Options', style=_CTRL_LABEL),
                dcc.Checklist(
                    id='int-spaghetti-options',
                    options=[
                        {'label': '  error bands (±σ)', 'value': 'bands'},
                        {'label': '  mark χ² closest point', 'value': 'chi2'},
                    ],
                    value=['bands', 'chi2'],
                    style={'fontSize': '13px'},
                ),
            ], style={'flex': '1', 'minWidth': '180px'}),
        ], className='kosma-panel',
           style={'display': 'flex', 'alignItems': 'flex-start', 'flexWrap': 'wrap',
                  'padding': '12px 18px', 'marginBottom': '8px'}),
        html.Div(id='int-spaghetti-status', style={'padding': '0 18px 8px'}),
        # Spaghetti contour plots
        html.Div(
            [_rgb_slice_panel(p['id'], prefix='int-spag-',
                              plot_id=f'plot-int-spag-{p["id"]}')
             for p in SLICE_PLANES],
            id='int-spag-panels-wrap',
            style=_SLICE_PANELS_ROW_STYLE,
        ),
        # Chi2 surface plots (below spaghetti)
        html.Details([
            html.Summary('χ² surface plots'),
            html.Div([
                html.P('The χ² = Σ((Z − obs)/σ)² surface for each slice plane. '
                       'Lower values (darker) indicate better fit. The star marks the minimum.',
                       style={**_PAGE_INTRO, 'marginTop': '6px', 'marginBottom': '10px',
                              'padding': '0 16px'}),
                html.Div(
                    [html.Div([
                        dcc.Graph(id=f'plot-int-spag-chi2-{p["id"]}', figure=placeholder_fig(),
                                  config=_SLICE_GRAPH_CFG, style=_SLICE_GRAPH_STYLE),
                    ], id=f'int-spag-chi2-panel-{p["id"]}', style=_SLICE_PANEL_ROW)
                     for p in SLICE_PLANES],
                    id='int-spag-chi2-panels-wrap',
                    style={**_SLICE_PANELS_ROW_STYLE, 'flexWrap': 'nowrap',
                           'overflowX': 'auto', 'padding': '0 8px 12px'},
                ),
            ], className='kosma-details-body'),
        ], className='kosma-details', open=False),
    ])


def _interp_control_row(prefix=''):
    """KoSens-style grid interpolation controls (``3d_grids.ipynb``)."""
    ny_id = f'{prefix}interp-ny'
    nx_id = f'{prefix}interp-nx'
    xlim_id = f'{prefix}interp-x-lim'
    ylim_id = f'{prefix}interp-y-lim'
    method_id = f'{prefix}interp-method'
    clip_id = f'{prefix}interp-clip'
    input_style = {**_INPUT_STYLE, 'width': '72px'}
    return html.Div([
        html.Div([
            html.Label('Interpolated grid size (ny \u00d7 nx)', style=_CTRL_LABEL),
            html.Div([
                dcc.Input(id=ny_id, type='number', value=DEFAULT_INTERP_NY,
                          min=2, max=500, step=1, className='kosma-input', style=input_style),
                html.Span(' \u00d7 ', className='kosma-muted', style={'margin': '0 6px'}),
                dcc.Input(id=nx_id, type='number', value=DEFAULT_INTERP_NX,
                          min=2, max=500, step=1, className='kosma-input', style=input_style),
            ]),
        ], style={'flex': '1.1', 'minWidth': '170px', 'marginRight': '18px'}),
        html.Div([
            html.Label('X-axis log\u2081\u2080 limit (upper)', style=_CTRL_LABEL),
            dcc.Input(id=xlim_id, type='number', value=None, placeholder='no limit',
                      className='kosma-input', style={**input_style, 'width': '100px'}),
            html.Span('  keep points with log\u2081\u2080(x) < limit',
                      className='kosma-muted',
                      style={'fontSize': '11px', 'marginLeft': '6px'}),
        ], style={'flex': '1.4', 'minWidth': '220px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Y-axis log\u2081\u2080 limit (upper)', style=_CTRL_LABEL),
            dcc.Input(id=ylim_id, type='number', value=None, placeholder='no limit',
                      className='kosma-input', style={**input_style, 'width': '100px'}),
            html.Span('  keep points with log\u2081\u2080(y) < limit',
                      className='kosma-muted',
                      style={'fontSize': '11px', 'marginLeft': '6px'}),
        ], style={'flex': '1.4', 'minWidth': '220px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Interpolation method', style=_CTRL_LABEL),
            dcc.Dropdown(id=method_id, options=INTERP_METHOD_OPTIONS,
                         value=DEFAULT_INTERP_METHOD, clearable=False,
                         className='kosma-dropdown', style={'fontSize': '13px'}),
        ], style={'flex': '1', 'minWidth': '130px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Clip to bounds', style=_CTRL_LABEL),
            dcc.Checklist(id=clip_id, options=[{'label': ' clip overshoot to [0, max]',
                                                'value': 'clip'}],
                          value=[], style={'fontSize': '13px'}),
        ], style={'flex': '1', 'minWidth': '160px'}),
    ], className='kosma-panel', style=_PANEL_ROW)


def _interp_error_control_row():
    """Decimation / error-metric controls for the interpolation-error tab."""
    input_style = {**_INPUT_STYLE, 'width': '72px'}
    return html.Div([
        html.Div([
            html.Label('Error decimation factor', style=_CTRL_LABEL),
            dcc.Input(id='ie-decimation', type='number', value=DEFAULT_ERROR_DECIMATION,
                      min=2, max=8, step=1, className='kosma-input', style=input_style),
            html.Span('  checkerboard decimation (KoSens default: 2)',
                      className='kosma-muted',
                      style={'fontSize': '11px', 'marginLeft': '8px'}),
        ], style={'flex': '1.3', 'minWidth': '220px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Error metric', style=_CTRL_LABEL),
            dcc.RadioItems(id='ie-error-metric', options=ERROR_METRIC_OPTIONS,
                           value=ERROR_METRIC_OPTIONS[0]['value'], **_SEG),
        ], style={'flex': '1.2', 'minWidth': '200px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Relative threshold', style=_CTRL_LABEL),
            dcc.Input(id='ie-rel-threshold', type='number',
                      value=DEFAULT_ERROR_REL_THRESHOLD, min=0, max=1, step=0.01,
                      className='kosma-input', style={**input_style, 'width': '90px'}),
            html.Span('  mask relative error below this fraction of max',
                      className='kosma-muted',
                      style={'fontSize': '11px', 'marginLeft': '8px'}),
        ], style={'flex': '1.5', 'minWidth': '260px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Flux / abundance scale', style=_CTRL_LABEL),
            dcc.RadioItems(id='ie-flux-scale', options=_SCALE_OPTIONS,
                           value='log', **_SEG),
        ], style={**_CTRL_BOX, 'marginRight': '18px'}),
        html.Div([
            html.Label('Contour lines', style=_CTRL_LABEL),
            dcc.Checklist(id='ie-plot-contours',
                          options=[{'label': ' white contours on original & interpolated',
                                    'value': 'contours'}],
                          value=['contours'], style={'fontSize': '13px'}),
        ], style={'flex': '1.4', 'minWidth': '220px'}),
    ], className='kosma-panel', style=_PANEL_ROW)


_SLICE_GRAPH_STYLE_COMPACT = {
    'margin': '0 auto',
    'width': f'{COMPACT_FIG_WIDTH}px',
    'height': f'{COMPACT_FIG_HEIGHT}px',
}
_SLICE_GRAPH_STYLE_FULL = {
    'margin': '0 auto',
    'width': '100%',
    'height': 'auto',
}
_SLICE_GRAPH_STYLE = _SLICE_GRAPH_STYLE_COMPACT
_SLICE_PANEL_ROW = {
    'flex': f'0 0 {COMPACT_FIG_WIDTH}px',
    'width': f'{COMPACT_FIG_WIDTH}px',
    'maxWidth': f'{COMPACT_FIG_WIDTH}px',
    'display': 'flex',
    'flexDirection': 'column',
    'marginBottom': '8px',
}
_SLICE_PANEL_ROW_FULL = {
    'width': '100%',
    'maxWidth': '100%',
    'display': 'flex',
    'flexDirection': 'column',
    'marginBottom': '8px',
}
_SLICE_PANELS_ROW_STYLE = {
    'display': 'flex',
    'flexDirection': 'row',
    'flexWrap': 'wrap',
    'gap': '12px',
    'alignItems': 'flex-start',
    'justifyContent': 'center',
}
_SLICE_PANELS_COL_STYLE = {
    'display': 'flex',
    'flexDirection': 'column',
    'gap': '16px',
}


def _slice_panel(slot_id):
    """One 2-D contour panel with its own third-axis slider (axes set at load time)."""
    return html.Div([
        html.Div([
            html.Label(id=f'slice-fixed-label-{slot_id}',
                       style={**_CTRL_LABEL, 'fontSize': '12px'}),
            dcc.Slider(id=f'slice-slider-{slot_id}', min=0, max=1, step=1, value=0, marks={},
                       tooltip={'placement': 'top', 'always_visible': False}),
            html.Div(id=f'slice-label-{slot_id}',
                     style={'textAlign': 'center', 'fontSize': '11px',
                            'marginTop': '2px', 'fontWeight': '600'}),
        ], style={'padding': '0 4px 8px'}),
        dcc.Graph(id=f'plot-contour-{slot_id}', figure=placeholder_fig(),
                  config=_SLICE_GRAPH_CFG,
                  style=_SLICE_GRAPH_STYLE),
    ], id=f'slice-panel-{slot_id}', style=_SLICE_PANEL_ROW)


def _int_slice_panel(slot_id):
    """One SIMLINE intensity contour panel with its own third-axis slider."""
    return html.Div([
        html.Div([
            html.Label(id=f'int-slice-fixed-label-{slot_id}',
                       style={**_CTRL_LABEL, 'fontSize': '12px'}),
            dcc.Slider(id=f'int-slice-slider-{slot_id}', min=0, max=1, step=1, value=0, marks={},
                       tooltip={'placement': 'top', 'always_visible': False}),
            html.Div(id=f'int-slice-label-{slot_id}',
                     style={'textAlign': 'center', 'fontSize': '11px',
                            'marginTop': '2px', 'fontWeight': '600'}),
        ], style={'padding': '0 4px 8px'}),
        dcc.Graph(id=f'plot-int-contour-{slot_id}', figure=placeholder_fig(),
                  config=_SLICE_GRAPH_CFG,
                  style=_SLICE_GRAPH_STYLE),
    ], id=f'int-slice-panel-{slot_id}', style=_SLICE_PANEL_ROW)


def _ie_plane_section(slot_id):
    """One slice plane on the interpolation-error tab (abundance + intensity rows)."""
    slider_block = html.Div([
        html.Label(id=f'ie-slice-fixed-label-{slot_id}',
                   style={**_CTRL_LABEL, 'fontSize': '12px'}),
        dcc.Slider(id=f'ie-slice-slider-{slot_id}', min=0, max=1, step=1, value=0, marks={},
                   tooltip={'placement': 'top', 'always_visible': False}),
        html.Div(id=f'ie-slice-label-{slot_id}',
                 style={'textAlign': 'center', 'fontSize': '11px',
                        'marginTop': '2px', 'fontWeight': '600'}),
    ], style={'padding': '0 4px 8px', 'maxWidth': f'{COMPACT_FIG_WIDTH}px'})
    return html.Div([
        html.H4(id=f'ie-plane-title-{slot_id}', style={'fontSize': '14px', 'margin': '8px 0 4px',
                                                        'color': '#333', 'fontWeight': '600'}),
        slider_block,
        html.Div(id=f'ie-abund-wrap-{slot_id}', children=[
            dcc.Graph(id=f'plot-ie-abund-{slot_id}', figure=placeholder_fig(),
                      config=_SLICE_GRAPH_CFG,
                      style={'width': '100%', 'height': 'auto', 'marginBottom': '8px'}),
        ]),
        html.Div(id=f'ie-int-wrap-{slot_id}', children=[
            dcc.Graph(id=f'plot-ie-int-{slot_id}', figure=placeholder_fig(),
                      config=_SLICE_GRAPH_CFG,
                      style={'width': '100%', 'height': 'auto'}),
        ]),
    ], style={'marginBottom': '20px', 'paddingBottom': '12px',
              'borderBottom': '1px solid #eee'})


app.layout = html.Div(
    id='app-root',
    className='theme-light',
    style={'fontFamily': UI_FONT, 'maxWidth': APP_MAX_WIDTH,
           'margin': '0 auto', 'padding': '12px 24px 28px', 'backgroundColor': '#f1f4f8',
           'color': '#1e293b', 'minHeight': '100vh'},
    children=[

    # Header + compact theme toggle
    html.Div(id='app-header', className='app-header', children=[
        html.Img(src='/assets/prisma_logo.png', id='app-logo', className='app-logo',
                 alt='PRISMA'),
        html.Div(id='app-brand', className='app-brand', children=[
            html.H1([
                html.Span('PRISMA', className='app-brand-name'),
            ], id='app-title', className='app-brand-title'),
            html.P(
                'PDR Results Interactive Simulation Mapping & Analysis — '
                'KOSMA-τ / KoSens3D model grids.',
                id='app-subtitle',
                className='app-brand-lead',
            ),
            html.Details(className='app-about', children=[
                html.Summary('About'),
                html.P(
                    'Browse photodissociation-region (PDR) model grids: depth profiles, '
                    'heating & cooling, chemistry, and SIMLINE line intensities across '
                    'density, FUV, cosmic-ray rate, and related parameters. Load an HDF5 '
                    'model directory and/or a SIMLINE output folder, then use the tabs '
                    'to explore profiles, 2-D grid slices, spectra, and map fits.',
                    id='app-blurb',
                ),
            ]),
        ]),
        html.Div(id='theme-toggle-wrap', className='theme-toggle', children=[
            dcc.RadioItems(
                id='plot-theme',
                options=PLOT_THEME_OPTIONS,
                value=DEFAULT_PLOT_THEME,
                className='theme-seg',
                inputClassName='theme-seg-input',
                labelClassName='theme-seg-btn',
                labelStyle={'display': 'inline-flex', 'margin': '0'},
            ),
        ]),
    ]),

    dcc.Store(id='grid-loaded', data=False),

    # Shared profile controls (hidden until a grid is loaded; not shown on Load tab)
    html.Div(id='controls-wrap', style={'display': 'none'}, children=[
        html.Div([_slider_block(d) for d in range(N_PARAMS)],
                 id='controls-sliders-wrap',
                 className='kosma-panel',
                 style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-start',
                        'padding': '12px 16px', 'marginTop': '4px'}),

        html.Div([
            html.Div([
                html.Label('Grid colormap', style=_CTRL_LABEL),
                dcc.Dropdown(id='grid-colorscale', options=GRID_COLORMAP_OPTIONS,
                             value=DEFAULT_GRID_COLORMAP, clearable=False,
                             className='kosma-dropdown', style={'fontSize': '13px'}),
            ], style={'flex': '1.2', 'minWidth': '140px', 'marginRight': '18px'}),
            html.Div([
                html.Label('X-axis', style=_CTRL_LABEL),
                dcc.RadioItems(id='xvar-choice', options=_XVAR_OPTIONS, value='Av', **_SEG),
            ], style=_CTRL_BOX),
            html.Div([
                html.Label('X scale', style=_CTRL_LABEL),
                dcc.RadioItems(id='xscale', options=_SCALE_OPTIONS, value='log', **_SEG),
            ], style=_CTRL_BOX),
            html.Div([
                html.Label('Y scale', style=_CTRL_LABEL),
                dcc.RadioItems(id='yscale', options=_SCALE_OPTIONS, value='log', **_SEG),
            ], style=_CTRL_BOX),
            html.Div([
                html.Label('Aᵥ range', style=_CTRL_LABEL),
                dcc.RadioItems(id='av-range', options=_AV_RANGE_OPTIONS,
                               value=DEFAULT_AV_RANGE, **_SEG),
            ], style={**_CTRL_BOX, 'marginRight': '0'}),
        ], id='controls-axis-wrap',
           className='kosma-panel',
           style={'display': 'flex', 'alignItems': 'flex-start',
                  'padding': '10px 16px', 'marginTop': '8px'}),

        html.Div(id='model-info', className='kosma-panel', style={
            'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'alignItems': 'center',
            'padding': '7px 14px',
            'marginTop': '8px', 'marginBottom': '4px', 'fontSize': '13px'}),
    ]),

    html.Div(className='nav-groups', children=[
        html.Span('Data'),
        html.Span('Local profiles'),
        html.Span('Grids & chemistry'),
        html.Span('Lines & fits'),
    ]),

    dcc.Tabs(
        id='main-tabs',
        value='load',
        className='prisma-tabs',
        style={'marginTop': '0'},
        colors={'border': '#e2e8f0', 'primary': '#2563eb', 'background': '#ffffff'},
        children=[

        # --- Page 1: grid loaders -------------------------------------------
        dcc.Tab(label='Load', value='load', style=_TAB_STYLE, selected_style=_TAB_SEL,
                children=[
            html.Div(style={'paddingTop': '14px'}, children=[
                html.P('Load model grids and SIMLINE directories to explore PDR models. '
                       'Primary grids drive the parameter sliders; overlay grids appear as '
                       'dashed lines for comparison (e.g. attenuated vs. unattenuated models).',
                       style=_PAGE_INTRO),

                html.Div(className='load-grid', children=[
                    # Primary
                    html.Div([
                        html.H3([
                            html.Span(className='accent-bar'),
                            'Primary grids',
                        ], className='load-section-title'),
                        html.P(
                            'Main HDF5 grid and SIMLINE output that define the parameter space. '
                            'These populate the sliders and drive visualizations.',
                            className='load-section-desc',
                        ),
                        _load_card(
                            'HDF5 model grid',
                            'depth profiles, abundances, heating/cooling',
                            html.Div([
                                html.Div([
                                    dcc.Input(
                                        id='dir-input', type='text', value=args.dir,
                                        placeholder='/path/to/pdrgrid_hdf5  or  /path/to/model.hdf5',
                                        className='kosma-input',
                                    ),
                                    dcc.Checklist(
                                        id='recursive-check',
                                        options=[{'label': ' recursive', 'value': 'rec'}],
                                        value=['rec'] if args.recursive else [],
                                        style={'fontSize': '13px', 'whiteSpace': 'nowrap',
                                               'alignSelf': 'center'},
                                    ),
                                    html.Button('Load grid', id='btn-load', n_clicks=0,
                                                className='btn btn-primary'),
                                ], className='load-card-row'),
                                html.Div(id='load-status', className='load-status'),
                            ]),
                            accent='blue',
                        ),
                        _load_card(
                            'Chemistry grid',
                            'reaction rates (can load standalone or with HDF5)',
                            html.Div([
                                html.Div([
                                    dcc.Input(
                                        id='chem-dir-input', type='text', value='',
                                        placeholder='/path/to/chemistrygrid  or  /path/to/chem_model.hdf5',
                                        className='kosma-input',
                                    ),
                                    dcc.Checklist(
                                        id='chem-recursive-check',
                                        options=[{'label': ' recursive', 'value': 'rec'}],
                                        value=[],
                                        style={'fontSize': '13px', 'whiteSpace': 'nowrap',
                                               'alignSelf': 'center'},
                                    ),
                                    html.Button('Load chemistry', id='btn-load-chem', n_clicks=0,
                                                className='btn btn-success'),
                                    html.Button('Clear', id='btn-clear-chem', n_clicks=0,
                                                className='btn btn-ghost'),
                                ], className='load-card-row'),
                                html.Div(id='chem-status', className='load-status'),
                            ]),
                            accent='green',
                        ),
                        dcc.Store(id='chem-state', data=0),
                        _load_card(
                            'SIMLINE directory',
                            'line intensities & spectra (can load standalone)',
                            html.Div([
                                html.Div([
                                    dcc.Input(
                                        id='simline-dir-input', type='text', value='',
                                        placeholder='/path/to/simlineoutput',
                                        className='kosma-input',
                                    ),
                                    dcc.Checklist(
                                        id='simline-recursive-check',
                                        options=[{'label': ' recursive', 'value': 'rec'}],
                                        value=[],
                                        style={'fontSize': '13px', 'whiteSpace': 'nowrap',
                                               'alignSelf': 'center'},
                                    ),
                                    html.Button('Load SIMLINE', id='btn-load-simline', n_clicks=0,
                                                className='btn btn-purple'),
                                    html.Button('Clear', id='btn-clear-simline', n_clicks=0,
                                                className='btn btn-ghost'),
                                ], className='load-card-row'),
                                html.Div(id='simline-status', className='load-status'),
                            ]),
                            accent='purple',
                        ),
                        dcc.Store(id='simline-state', data=0),
                    ], className='kosma-panel load-section'),

                    # Overlay
                    html.Div([
                        html.H3([
                            html.Span(className='accent-bar overlay'),
                            'Overlay grids',
                            html.Span(' (comparison)', className='kosma-muted',
                                      style={'fontWeight': '400', 'fontSize': '13px',
                                             'marginLeft': '6px', 'letterSpacing': '0',
                                             'textTransform': 'none'}),
                        ], className='load-section-title'),
                        html.P(
                            'Secondary grids for comparison (e.g. attenuated CRIR models). '
                            'Overlays appear as dashed lines on profiles, reaction rates on the '
                            'Chemistry tab, and enable triple-panel intensity views.',
                            className='load-section-desc',
                        ),
                        _load_card(
                            'Overlay HDF5 grid',
                            'dashed lines on profiles',
                            html.Div([
                                html.Div([
                                    dcc.Input(
                                        id='overlay-dir-input', type='text', value='',
                                        placeholder='/path/to/attenuated_grid_hdf5  or  /path/to/model.hdf5',
                                        className='kosma-input',
                                    ),
                                    dcc.Checklist(
                                        id='overlay-recursive-check',
                                        options=[{'label': ' recursive', 'value': 'rec'}],
                                        value=[],
                                        style={'fontSize': '13px', 'whiteSpace': 'nowrap',
                                               'alignSelf': 'center'},
                                    ),
                                    html.Button('Load overlay', id='btn-load-overlay', n_clicks=0,
                                                className='btn btn-overlay'),
                                    html.Button('Clear', id='btn-clear-overlay', n_clicks=0,
                                                className='btn btn-ghost'),
                                ], className='load-card-row'),
                                html.Div(id='overlay-status', className='load-status'),
                            ]),
                            accent='overlay',
                        ),
                        dcc.Store(id='overlay-state', data=0),
                        _load_card(
                            'Overlay chemistry grid',
                            'dashed reaction rates on Chemistry tab',
                            html.Div([
                                html.Div([
                                    dcc.Input(
                                        id='chem-overlay-dir-input', type='text', value='',
                                        placeholder='/path/to/attenuated_chemistrygrid  or  /path/to/chem_model.hdf5',
                                        className='kosma-input',
                                    ),
                                    dcc.Checklist(
                                        id='chem-overlay-recursive-check',
                                        options=[{'label': ' recursive', 'value': 'rec'}],
                                        value=[],
                                        style={'fontSize': '13px', 'whiteSpace': 'nowrap',
                                               'alignSelf': 'center'},
                                    ),
                                    html.Button('Load overlay chemistry',
                                                id='btn-load-chem-overlay', n_clicks=0,
                                                className='btn btn-overlay'),
                                    html.Button('Clear', id='btn-clear-chem-overlay',
                                                n_clicks=0, className='btn btn-ghost'),
                                ], className='load-card-row'),
                                html.Div(id='chem-overlay-status', className='load-status'),
                            ]),
                            accent='overlay',
                        ),
                        dcc.Store(id='chem-overlay-state', data=0),
                        _load_card(
                            'Overlay SIMLINE',
                            'triple-panel shift on Intensities tab',
                            html.Div([
                                html.Div([
                                    dcc.Input(
                                        id='simline-overlay-dir-input', type='text', value='',
                                        placeholder='/path/to/attenuated/simlineoutput',
                                        className='kosma-input',
                                    ),
                                    dcc.Checklist(
                                        id='simline-overlay-recursive-check',
                                        options=[{'label': ' recursive', 'value': 'rec'}],
                                        value=[],
                                        style={'fontSize': '13px', 'whiteSpace': 'nowrap',
                                               'alignSelf': 'center'},
                                    ),
                                    html.Button('Load overlay SIMLINE',
                                                id='btn-load-simline-overlay', n_clicks=0,
                                                className='btn btn-overlay'),
                                    html.Button('Clear', id='btn-clear-simline-overlay',
                                                n_clicks=0, className='btn btn-ghost'),
                                ], className='load-card-row'),
                                html.Div(id='simline-overlay-status', className='load-status'),
                            ]),
                            accent='overlay',
                        ),
                        dcc.Store(id='simline-overlay-state', data=0),
                    ], className='kosma-panel load-section'),
                ]),

                dcc.Store(id='cr-atten-profiles', data=[]),
            ]),
        ]),

        # --- Page 2: model setup (JSON configs) -----------------------------
        dcc.Tab(label='Model setup', value='modelsetup', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Simulation parameters from PDR config JSON files under '
                       '``Models/Model…/config_files/``. Shared settings are grouped '
                       'once; grid parameters that differ between models are listed '
                       'explicitly. Updates when you load a grid or move the sliders.',
                       style=_PAGE_INTRO),
                html.Div(id='model-setup-panel'),
            ]),
        ]),

        # --- Page 3: abundance profiles -------------------------------------
        dcc.Tab(label='Profiles', value='profiles', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Gas and dust temperatures plus H/H\u2082, C\u207A/C/CO and custom '
                       'species abundance profiles vs depth.',
                       style=_PAGE_INTRO),
                dcc.Loading(id='loading-profiles', type='circle', children=[
                    html.Div([
                        html.Div([
                            dcc.Graph(id='plot-tgas', figure=placeholder_fig(), config=_GRAPH_CFG,
                                      style={'flex': '1', 'minWidth': '0'}),
                            dcc.Graph(id='plot-h-h2', figure=placeholder_fig(), config=_GRAPH_CFG,
                                      style={'flex': '1', 'minWidth': '0'}),
                        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '12px'}),
                        html.Div([
                            dcc.Graph(id='plot-cco', figure=placeholder_fig(), config=_GRAPH_CFG,
                                      style={'flex': '1', 'minWidth': '0'}),
                            html.Div([
                                html.Div([
                                    html.Label('Species:', style={'fontWeight': '600', 'fontSize': '12px',
                                                                  'whiteSpace': 'nowrap',
                                                                  'marginRight': '8px'}),
                                    dcc.Dropdown(id='species-selector', options=[], value=[], multi=True,
                                                 placeholder='Select species to plot\u2026',
                                                 style={'fontSize': '12px', 'flex': '1'}),
                                ], style={'display': 'flex', 'alignItems': 'center',
                                          'padding': '4px 0', 'marginBottom': '4px'}),
                                dcc.Graph(id='plot-custom', figure=placeholder_fig(), config=_GRAPH_CFG,
                                          style={'flex': '1'}),
                            ], style={'flex': '1', 'minWidth': '0', 'display': 'flex',
                                      'flexDirection': 'column'}),
                        ], style={'display': 'flex', 'gap': '12px'}),
                    ]),
                ]),
            ]),
        ]),

        # --- Page 3: heating & cooling --------------------------------------
        dcc.Tab(label='Thermal', value='thermal', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Thermal balance and per-component heating and cooling rates vs depth.',
                       style=_PAGE_INTRO),
                dcc.Tabs(id='thermal-subtabs', value='thermal-depth',
                         style={'marginTop': '4px'},
                         content_style={'paddingTop': '16px'}, children=[
                    dcc.Tab(label='Depth profiles', value='thermal-depth',
                            style=_NESTED_TAB_STYLE, selected_style=_NESTED_TAB_SEL,
                            children=[
                        dcc.Loading(id='loading-thermal', type='circle', children=[
                            html.Div([
                                dcc.Graph(id='plot-thermal', figure=placeholder_fig(), config=_GRAPH_CFG,
                                          style={'flex': '1', 'minWidth': '0'}),
                                dcc.Graph(id='plot-heat-breakdown', figure=placeholder_fig(), config=_GRAPH_CFG,
                                          style={'flex': '1', 'minWidth': '0'}),
                            ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '12px'}),
                            html.Div([
                                dcc.Graph(id='plot-cool-breakdown', figure=placeholder_fig(), config=_GRAPH_CFG,
                                          style={'flex': '1', 'minWidth': '0'}),
                                html.Div(style={'flex': '1', 'minWidth': '0'}),
                            ], style={'display': 'flex', 'gap': '12px'}),
                        ]),
                    ]),
                    dcc.Tab(label='Grid-integrated vs CRIR', value='thermal-crir',
                            style=_NESTED_TAB_STYLE, selected_style=_NESTED_TAB_SEL,
                            children=[
                        html.P('KoSens grid-integrated diagnostics: each heating / cooling '
                               'mechanism is volume-integrated (\u222B 4\u03C0r\u00B2 rate n dr) per '
                               'model and plotted against the cosmic-ray ionization rate \u03B6 for '
                               'the selected densities. FUV, clump mass, metallicity and '
                               'attenuation are held at the current slider values.',
                               style=_PAGE_INTRO),
                        html.Div([
                            html.Div([
                                html.Label('Densities  log\u2081\u2080(n)', style=_CTRL_LABEL),
                                dcc.Dropdown(id='therm-crir-densities', options=[], value=[],
                                             multi=True,
                                             placeholder='Load a grid\u2026 then pick densities',
                                             style={'fontSize': '13px'}),
                            ], style={'flex': '2', 'minWidth': '240px', 'marginRight': '18px'}),
                            html.Div([
                                html.Label('Stack threshold', style=_CTRL_LABEL),
                                dcc.Slider(id='therm-crir-threshold', min=0.0, max=0.3,
                                           step=0.01, value=0.01,
                                           marks={0: '0', 0.1: '0.1', 0.2: '0.2', 0.3: '0.3'},
                                           tooltip={'placement': 'bottom'}),
                            ], style={'flex': '1', 'minWidth': '180px', 'marginRight': '18px'}),
                        ], className='kosma-panel',
                           style={'display': 'flex', 'alignItems': 'flex-start',
                                  'flexWrap': 'wrap', 'padding': '12px 18px',
                                  'marginBottom': '12px'}),
                        html.Div([
                            html.Div([
                                html.Label('Min rate floor [erg s\u207B\u00B9]', style=_CTRL_LABEL),
                                dcc.Input(id='therm-minrate', type='number', value=1e23,
                                          debounce=True, style={'width': '140px'}),
                            ], style={**_CTRL_BOX, 'minWidth': '160px'}),
                            html.Div([
                                html.Label('Min points / channel', style=_CTRL_LABEL),
                                dcc.Input(id='therm-minpoints', type='number', value=2,
                                          min=1, step=1, debounce=True,
                                          style={'width': '90px'}),
                            ], style={**_CTRL_BOX, 'minWidth': '120px'}),
                            html.Div([
                                html.Label('Totals', style=_CTRL_LABEL),
                                dcc.Checklist(id='therm-show-totals',
                                              options=[{'label': ' show totals', 'value': 'on'}],
                                              value=['on'], **_RADIO),
                            ], style={**_CTRL_BOX, 'minWidth': '120px'}),
                        ], className='kosma-panel',
                           style={'display': 'flex', 'alignItems': 'flex-start',
                                  'flexWrap': 'wrap', 'padding': '12px 18px',
                                  'marginBottom': '12px'}),
                        dcc.Loading(id='loading-thermal-crir', type='circle', children=[
                            dcc.Graph(id='plot-heat-contrib-crir', figure=placeholder_fig(),
                                      config=_GRAPH_CFG, style={'marginBottom': '12px'}),
                            dcc.Graph(id='plot-abs-heatcool-crir', figure=placeholder_fig(),
                                      config=_GRAPH_CFG),
                        ]),
                    ]),
                ]),
            ]),
        ]),

        # --- Page 4: 2-D grid slices --------------------------------------
        dcc.Tab(label='Grids', value='grids', style=_TAB_STYLE, selected_style=_TAB_SEL,
                children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Contour maps over the full model grid. Use the abundance box, the '
                       'column-density box, or both — the first selection fills the three '
                       'main panels, and every extra quantity gets its own row of slice maps. '
                       'Add all species / Add all column densities fill each box. Without an '
                       'overlay grid the three slice planes appear side by side as square '
                       'panels. Grids are resampled in log parameter space (KoSens-style) '
                       'before plotting. With an overlay loaded, each plane expands to three '
                       'sub-plots (reference, overlay, x-shift) stacked vertically.',
                       style=_PAGE_INTRO),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Label('Abundance / diagnostics', style=_CTRL_LABEL),
                            dcc.Dropdown(id='contour-quantity', options=[], value=[],
                                         multi=True,
                                         placeholder='Load a grid\u2026 then pick X(species) or a diagnostic',
                                         style={'fontSize': '13px'}),
                        ], style={'flex': '2', 'minWidth': '260px', 'marginRight': '12px'}),
                        html.Button('Add all species', id='btn-contour-all-species', n_clicks=0,
                                    className='btn btn-primary btn-sm', style=_ALL_BTN_STYLE),
                        html.Div([
                            html.Label('Contour Z scale', style=_CTRL_LABEL),
                            dcc.RadioItems(id='contour-zscale', options=_SCALE_OPTIONS,
                                           value='log', **_SEG),
                        ], style={**_CTRL_BOX, 'marginLeft': '18px', 'marginRight': '0'}),
                    ], style={'display': 'flex', 'alignItems': 'flex-start',
                              'flexWrap': 'wrap', 'width': '100%',
                              'marginBottom': '10px'}),
                    html.Div([
                        html.Div([
                            html.Label('Column density N (cm\u207B\u00B2)', style=_CTRL_LABEL),
                            dcc.Dropdown(id='contour-cdens', options=[], value=[],
                                         multi=True,
                                         placeholder='Optional \u2014 pick N(species) to show alongside or instead of abundances',
                                         style={'fontSize': '13px'}),
                        ], style={'flex': '2', 'minWidth': '260px', 'marginRight': '12px'}),
                        html.Button('Add all column densities', id='btn-contour-all-cdens', n_clicks=0,
                                    title='HDF5 Integrated quantities/Column densities (KoSens cd_*)',
                                    className='btn btn-success btn-sm', style=_ALL_BTN_STYLE),
                    ], style={'display': 'flex', 'alignItems': 'flex-start',
                              'flexWrap': 'wrap', 'width': '100%'}),
                ], className='kosma-panel',
                style={'padding': '12px 18px', 'marginBottom': '12px'}),
                _advanced_details('Interpolation & resampling', _interp_control_row(prefix='')),
                _advanced_details('Attenuation x-shift matching', _shift_control_row(prefix='')),
                dcc.Tabs(id='grid-subtabs', value='grid-maps',
                         style={'marginTop': '4px'},
                         content_style={'paddingTop': '16px'}, children=[
                    dcc.Tab(label='Maps', value='grid-maps',
                            style=_NESTED_TAB_STYLE, selected_style=_NESTED_TAB_SEL,
                            children=[
                        html.Div([_slice_panel(p['id']) for p in SLICE_PLANES],
                                 id='slice-panels-wrap',
                                 style=_SLICE_PANELS_ROW_STYLE),
                        dcc.Loading(id='loading-grid-contour-extra', type='circle', children=[
                            html.Div(id='grid-contour-extra-rows'),
                        ]),
                        _rgb_grid_controls(),
                    ]),
                    dcc.Tab(label='Ratios', value='grid-ratios',
                            style=_NESTED_TAB_STYLE, selected_style=_NESTED_TAB_SEL,
                            children=[
                        html.P('KoSens abundance ratios from the species selected above '
                               '(needs two or more). Intra-list pairs use the KoSens '
                               'molecule priority so the higher-priority species is in '
                               'the numerator (C⁺ > C > HCO⁺ > CO > ¹³CO > …). '
                               'Equal-priority pairs keep your selection order.',
                               style=_PAGE_INTRO),
                        html.Div([
                            html.Div([
                                html.Label('Abundance ratios', style=_CTRL_LABEL),
                                dcc.Dropdown(id='grid-ratio-selector', options=[], value=[],
                                             multi=True,
                                             placeholder='Select two or more species above\u2026',
                                             style={'fontSize': '13px'}),
                            ], style={'flex': '2', 'minWidth': '260px', 'marginRight': '12px'}),
                            html.Button('Add all ratios', id='btn-grid-all-ratios', n_clicks=0,
                                        className='btn btn-primary btn-sm', style=_ALL_BTN_STYLE),
                        ], className='kosma-panel',
                           style={'display': 'flex', 'alignItems': 'flex-start',
                                  'flexWrap': 'wrap',
                                  'padding': '12px 18px', 'marginBottom': '12px'}),
                        html.Div(id='grid-ratio-hint', className='kosma-muted',
                                 style={'marginBottom': '8px'}),
                        dcc.Loading(id='loading-grid-ratios', type='circle', children=[
                            html.Div(id='grid-ratio-rows'),
                        ]),
                    ]),
                    dcc.Tab(label='Abundance PDF', value='grid-apdf',
                            style=_NESTED_TAB_STYLE, selected_style=_NESTED_TAB_SEL,
                            children=[
                        html.P('KoSens abundance PDF: a 2-D histogram of species abundance '
                               'versus \u03B6_H / n_H, aggregating every depth point across the '
                               'grid (mass, metallicity and attenuation fixed to the sliders; '
                               'density, FUV and CRIR vary). With "split by CRIR" one panel is '
                               'made per initial cosmic-ray rate; one figure is produced per '
                               'selected species.',
                               style=_PAGE_INTRO),
                        html.Div([
                            html.Div([
                                html.Label('Species', style=_CTRL_LABEL),
                                dcc.Dropdown(id='apdf-species', options=[], value=[],
                                             multi=True,
                                             placeholder='Load a grid\u2026 then pick species',
                                             style={'fontSize': '13px'}),
                            ], style={'flex': '2', 'minWidth': '240px', 'marginRight': '18px'}),
                            html.Div([
                                html.Label('Abundance', style=_CTRL_LABEL),
                                dcc.RadioItems(id='apdf-dens-choice',
                                               options=[{'label': 'relative n/n_H', 'value': 'rel'},
                                                        {'label': 'number n', 'value': 'number'},
                                                        {'label': 'column N', 'value': 'column'}],
                                               value='rel', **_RADIO),
                            ], style={**_CTRL_BOX, 'minWidth': '150px'}),
                            html.Div([
                                html.Label('Histogram bins', style=_CTRL_LABEL),
                                dcc.Input(id='apdf-nbins', type='number', value=100,
                                          min=5, max=400, step=5, debounce=True,
                                          style={'width': '90px'}),
                            ], style={**_CTRL_BOX, 'minWidth': '120px'}),
                            html.Div([
                                html.Label('Options', style=_CTRL_LABEL),
                                dcc.Checklist(id='apdf-split-crir',
                                              options=[{'label': ' split by CRIR', 'value': 'on'}],
                                              value=['on'], **_RADIO),
                                dcc.Checklist(id='apdf-log-color',
                                              options=[{'label': ' log color', 'value': 'on'}],
                                              value=['on'], **_RADIO),
                                dcc.Checklist(id='apdf-overplot-avg',
                                              options=[{'label': ' binned average', 'value': 'on'}],
                                              value=[], **_RADIO),
                            ], style={**_CTRL_BOX, 'minWidth': '150px'}),
                        ], className='kosma-panel',
                           style={'display': 'flex', 'alignItems': 'flex-start',
                                  'flexWrap': 'wrap', 'padding': '12px 18px',
                                  'marginBottom': '12px'}),
                        dcc.Loading(id='loading-apdf', type='circle', children=[
                            html.Div(id='apdf-rows'),
                        ]),
                    ]),
                ]),
            ]),
        ]),

        # --- Page 5: chemistry ----------------------------------------------
        dcc.Tab(label='Chemistry', value='chemistry', style=_TAB_STYLE, selected_style=_TAB_SEL,
                children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Top formation and destruction reactions for a selected species '
                       '(load a chemistry grid on the Load tab — standalone or with an HDF5 '
                       'structure grid; an optional overlay chemistry grid adds dashed '
                       'comparison curves). In chemistry-only mode the sliders index the '
                       'chemistry filenames; load an HDF5 grid for depth profiles and '
                       'abundance colormaps. Choose how reactions '
                       'are ranked: fractional contribution to the total rate, or '
                       'mass-weighted rate (KoSens ``top_reactions_plot`` metrics). '
                       'Below, separate formation and destruction networks show all '
                       'partners in those same top-N reactions (reactants left, products right).',
                       style=_PAGE_INTRO),
                html.Div([
                    html.Div([
                        html.Label('Reaction species', style=_CTRL_LABEL),
                        dcc.Dropdown(id='react-species', options=[], value=None,
                                     placeholder='Load a chemistry grid\u2026',
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '2', 'minWidth': '200px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Top reactions / depth point', style=_CTRL_LABEL),
                        dcc.Input(id='react-n', type='number', value=CHEM_DEFAULT_NREAC,
                                  min=1, max=50, step=1,
                                  style={'width': '90px', 'padding': '7px 9px', 'fontSize': '13px',
                                         'border': '1px solid #bbc', 'borderRadius': '6px'}),
                    ], style={'flex': '1', 'minWidth': '150px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Ranking metric', style=_CTRL_LABEL),
                        dcc.RadioItems(id='react-ranking', options=REACT_RANKING_OPTIONS,
                                       value=DEFAULT_REACT_RANKING, **_RADIO),
                    ], style={**_CTRL_BOX, 'marginRight': '0'}),
                ], className='kosma-panel',
                style={'display': 'flex', 'alignItems': 'flex-start',
                       'padding': '12px 18px'}),
                dcc.Loading(id='loading-chemistry', type='circle', children=[
                    html.Div([
                        html.Div([
                            dcc.Graph(id='plot-react-formation', figure=placeholder_fig(), config=_GRAPH_CFG,
                                      style={'width': '100%'}),
                            html.Div(id='react-contrib-formation'),
                        ], style={'flex': '1', 'minWidth': '0'}),
                        html.Div([
                            dcc.Graph(id='plot-react-destruction', figure=placeholder_fig(), config=_GRAPH_CFG,
                                      style={'width': '100%'}),
                            html.Div(id='react-contrib-destruction'),
                        ], style={'flex': '1', 'minWidth': '0'}),
                    ], style={'display': 'flex', 'gap': '12px', 'marginTop': '12px',
                              'alignItems': 'flex-start'}),
                ]),
                html.H4('Reaction pathway network',
                        style={'fontSize': '14px', 'margin': '22px 0 6px', 'fontWeight': '600'}),
                html.Div(id='react-network-status', style={**_PAGE_INTRO, 'marginBottom': '8px'}),
                dcc.Store(id='react-net-highlight', data=None),
                dcc.Store(id='react-net-hidden', data=[]),
                html.Div([
                    html.Div([
                        html.Label('Upstream depth', style=_CTRL_LABEL),
                        dcc.Input(id='react-chain-upstream', type='number',
                                  min=0, max=6, step=1,
                                  value=chem_network.DEFAULT_CHAIN_UPSTREAM,
                                  style={'width': '90px', 'padding': '7px 9px', 'fontSize': '13px',
                                         'border': '1px solid #bbc', 'borderRadius': '6px'}),
                    ], style={'marginRight': '18px'}),
                    html.Div([
                        html.Label('Downstream depth', style=_CTRL_LABEL),
                        dcc.Input(id='react-chain-downstream', type='number',
                                  min=0, max=6, step=1,
                                  value=chem_network.DEFAULT_CHAIN_DOWNSTREAM,
                                  style={'width': '90px', 'padding': '7px 9px', 'fontSize': '13px',
                                         'border': '1px solid #bbc', 'borderRadius': '6px'}),
                    ], style={'marginRight': '18px'}),
                    html.Div([
                        html.Label('Species text', style=_CTRL_LABEL,
                                   title='Font size of species names in the boxes (pt)'),
                        dcc.Input(id='react-species-label-size', type='number',
                                  min=8, max=40, step=1,
                                  value=chem_network.DEFAULT_SPECIES_LABEL_SIZE,
                                  style={'width': '90px', 'padding': '7px 9px', 'fontSize': '13px',
                                         'border': '1px solid #bbc', 'borderRadius': '6px'}),
                    ], style={'marginRight': '18px'}),
                    html.Div([
                        html.Label('Partner text', style=_CTRL_LABEL,
                                   title='Font size of reaction partners on the arrows (pt)'),
                        dcc.Input(id='react-partner-label-size', type='number',
                                  min=8, max=40, step=1,
                                  value=chem_network.DEFAULT_PARTNER_LABEL_SIZE,
                                  style={'width': '90px', 'padding': '7px 9px', 'fontSize': '13px',
                                         'border': '1px solid #bbc', 'borderRadius': '6px'}),
                    ], style={'marginRight': '22px'}),
                    html.Div([
                        html.Label('Include', style=_CTRL_LABEL),
                        dcc.Checklist(
                            id='react-chain-isotopes',
                            options=[{'label': ' isotopes', 'value': 'iso'}],
                            value=[],
                            style={'fontSize': '13px', 'whiteSpace': 'nowrap'}),
                    ], style={'marginRight': '16px', 'alignSelf': 'flex-end'}),
                    html.Div([
                        html.Label('\u00a0', style=_CTRL_LABEL),
                        dcc.Checklist(
                            id='react-chain-ice',
                            options=[{'label': ' ice / grain', 'value': 'ice'}],
                            value=[],
                            style={'fontSize': '13px', 'whiteSpace': 'nowrap'}),
                    ], style={'marginRight': '18px', 'alignSelf': 'flex-end'}),
                    html.Div([
                        html.Label('\u00a0', style=_CTRL_LABEL),
                        html.Button('Hide selected', id='btn-react-net-hide', n_clicks=0,
                                    title='Click a species box, then hide it from the figure',
                                    className='btn btn-ghost btn-sm',
                                    style={'marginRight': '8px'}),
                        html.Button('Show all', id='btn-react-net-show-all', n_clicks=0,
                                    title='Restore every hidden species',
                                    className='btn btn-ghost btn-sm',
                                    style={'marginRight': '8px'}),
                        html.Button('Clear highlight', id='btn-react-net-clear', n_clicks=0,
                                    className='btn btn-ghost btn-sm'),
                    ], style={'alignSelf': 'flex-end'}),
                ], style={'display': 'flex', 'alignItems': 'flex-end',
                          'flexWrap': 'wrap', 'marginBottom': '8px'}),
                html.Div(id='react-net-hidden-bar', style={'marginBottom': '10px'}),
                dcc.Graph(
                    id='plot-react-network',
                    figure=placeholder_fig(),
                    config={**_GRAPH_CFG, 'responsive': True},
                    style={'width': '100%', 'maxWidth': '100%',
                           'height': '1020px', 'maxHeight': '1020px',
                           'minHeight': '720px', 'marginBottom': '16px',
                           'overflow': 'hidden'},
                ),
            ]),
        ]),

        # --- Page 6: SIMLINE intensities ------------------------------------
        dcc.Tab(label='Intensities', value='intensities', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('SIMLINE line intensities from per-model .smli files. Select several '
                       'species and/or transitions (or Add all species / Add all transitions) '
                       'to overlay spectra and add extra contour rows under the main maps. '
                       'Without overlay SIMLINE the three slice planes appear side by side as '
                       'square panels; grids are resampled in log parameter space before '
                       'plotting. With overlay SIMLINE each plane expands to three sub-plots '
                       'side by side (reference, overlay, x-shift) stacked vertically.',
                       style=_PAGE_INTRO),
                html.Div([
                    html.Div([
                        html.Label('Quantity', style=_CTRL_LABEL),
                        dcc.RadioItems(id='int-idef', options=SIMLINE_IDEF_OPTIONS,
                                       value=SIMLINE_DEFAULT_IDEF, **_RADIO),
                    ], style={'flex': '1.4', 'minWidth': '220px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Species', style=_CTRL_LABEL),
                        dcc.Dropdown(id='int-species', options=[], value=[],
                                     multi=True,
                                     placeholder='Load a SIMLINE directory\u2026',
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '1.2', 'minWidth': '160px', 'marginRight': '8px'}),
                    html.Button('Add all species', id='btn-int-all-species', n_clicks=0,
                                className='btn btn-primary btn-sm', style=_ALL_BTN_STYLE),
                    html.Div([
                        html.Label('Transition (for grid maps)', style=_CTRL_LABEL),
                        dcc.Dropdown(id='int-transition', options=[], value=[],
                                     multi=True,
                                     placeholder='Select species\u2026',
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '1.4', 'minWidth': '180px', 'marginLeft': '12px',
                              'marginRight': '8px'}),
                    html.Button('Add all transitions', id='btn-int-all-transitions', n_clicks=0,
                                className='btn btn-primary btn-sm', style=_ALL_BTN_STYLE),
                    html.Div([
                        html.Label('Contour Z scale', style=_CTRL_LABEL),
                        dcc.RadioItems(id='int-zscale', options=_SCALE_OPTIONS,
                                       value='log', **_SEG),
                    ], style={**_CTRL_BOX, 'marginRight': '0'}),
                ], className='kosma-panel',
                style={'display': 'flex', 'alignItems': 'flex-start',
                       'flexWrap': 'wrap',
                       'padding': '12px 18px', 'marginBottom': '12px'}),
                _advanced_details('Interpolation & resampling', _interp_control_row(prefix='int-')),
                _advanced_details('Attenuation x-shift matching', _shift_control_row(prefix='int-')),
                _advanced_details('Contour overlays', _int_contour_overlay_row()),
                dcc.Tabs(id='int-subtabs', value='int-maps',
                         style={'marginTop': '4px'}, children=[
                    dcc.Tab(label='Maps', value='int-maps',
                            style=_NESTED_TAB_STYLE, selected_style=_NESTED_TAB_SEL,
                            children=[
                        dcc.Loading(id='loading-int-spectrum', type='circle', children=[
                            dcc.Graph(id='plot-int-spectrum', figure=placeholder_fig(),
                                      config=_GRAPH_CFG,
                                      style={'marginBottom': '12px'}),
                        ]),
                        html.Div([_int_slice_panel(p['id']) for p in SLICE_PLANES],
                                 id='int-slice-panels-wrap',
                                 style=_SLICE_PANELS_ROW_STYLE),
                        dcc.Loading(id='loading-int-contour-extra', type='circle', children=[
                            html.Div(id='int-contour-extra-rows'),
                        ]),
                        _spaghetti_controls(),
                        _rgb_intensity_controls(),
                    ]),
                    dcc.Tab(label='Ratios', value='int-ratios',
                            style=_NESTED_TAB_STYLE, selected_style=_NESTED_TAB_SEL,
                            children=[
                        html.P('KoSens line-intensity ratios from the species and transitions '
                               'selected above. Intra-molecule (one species, two or more '
                               'transitions): higher-energy / lower-energy '
                               '(e.g. CO(2-1)/CO(1-0)). Inter-molecule (two or more species): '
                               'higher KoSens priority / lower '
                               '(C⁺ > C > HCO⁺ > CO > …).',
                               style=_PAGE_INTRO),
                        html.Div([
                            html.Div([
                                html.Label('Line ratios', style=_CTRL_LABEL),
                                dcc.Dropdown(id='int-ratio-selector', options=[], value=[],
                                             multi=True,
                                             placeholder='Select two transitions or two species\u2026',
                                             style={'fontSize': '13px'}),
                            ], style={'flex': '2', 'minWidth': '260px', 'marginRight': '12px'}),
                            html.Button('Add all ratios', id='btn-int-all-ratios', n_clicks=0,
                                        className='btn btn-primary btn-sm',
                                        style=_ALL_BTN_STYLE),
                        ], className='kosma-panel',
                           style={'display': 'flex', 'alignItems': 'flex-start',
                                  'flexWrap': 'wrap',
                                  'padding': '12px 18px', 'marginBottom': '12px'}),
                        html.Div(id='int-ratio-hint', className='kosma-muted',
                                 style={'marginBottom': '8px'}),
                        dcc.Loading(id='loading-int-ratios', type='circle', children=[
                            html.Div(id='int-ratio-rows'),
                        ]),
                    ]),
                ]),
            ]),
        ]),

        # --- Page 7: SIMLINE spectra (PV FITS) --------------------------------
        dcc.Tab(label='Spectra', value='spectra', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Velocity-resolved spectra and position–velocity diagrams from SimLine '
                       'PV FITS cubes (brightness temperature or optical depth), with optional '
                       'observational CLASS MATRIX / cube FITS overlay and multi-Gaussian '
                       'line fitting.',
                       style=_PAGE_INTRO),
                html.Div([
                    html.Div([
                        html.Label('Quantity', style=_CTRL_LABEL),
                        dcc.RadioItems(id='sp-quantity', options=SIMLINE_PV_QUANTITY_OPTIONS,
                                       value=SIMLINE_DEFAULT_PV_QUANTITY, **_RADIO),
                    ], style={'flex': '1.4', 'minWidth': '220px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Species', style=_CTRL_LABEL),
                        dcc.Dropdown(id='sp-species', options=[], value=None,
                                     placeholder='Load a SIMLINE directory\u2026',
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '1', 'minWidth': '140px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Transition(s)', style=_CTRL_LABEL),
                        dcc.Dropdown(id='sp-transition', options=[], value=[],
                                     multi=True,
                                     placeholder='Select species\u2026',
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '1.8', 'minWidth': '240px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Position offsets (arcsec)', style=_CTRL_LABEL),
                        dcc.Input(id='sp-positions', type='text', value='',
                                  placeholder='empty = spatial mean; e.g. 0, 21, 42',
                                  style={'width': '100%', 'fontSize': '13px'}),
                    ], style={'flex': '2', 'minWidth': '220px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('PV Z scale', style=_CTRL_LABEL),
                        dcc.RadioItems(id='sp-zscale', options=_SCALE_OPTIONS,
                                       value='linear', **_RADIO),
                    ], style={**_CTRL_BOX, 'marginRight': '18px'}),
                    html.Div([
                        html.Label('PV colormap', style=_CTRL_LABEL),
                        dcc.Dropdown(id='sp-colorscale', options=GRID_COLORMAP_OPTIONS,
                                     value=DEFAULT_GRID_COLORMAP,
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '1', 'minWidth': '120px'}),
                ], className='kosma-panel',
                style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
                       'padding': '12px 18px'}),
                html.Div([
                    html.Div([
                        html.Label('PV position min (arcsec)', style=_CTRL_LABEL),
                        dcc.Input(id='sp-pos-min', type='number', value=None,
                                  placeholder='auto', style={'width': '100%'}),
                    ], style={'flex': '1', 'minWidth': '140px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('PV position max (arcsec)', style=_CTRL_LABEL),
                        dcc.Input(id='sp-pos-max', type='number', value=None,
                                  placeholder='auto', style={'width': '100%'}),
                    ], style={'flex': '1', 'minWidth': '140px'}),
                ], style={'display': 'flex', 'alignItems': 'flex-end',
                          'padding': '0 18px 12px', 'marginBottom': '4px'}),
                html.Div([
                    html.H4('Observational spectrum (FITS cube / CLASS MATRIX)',
                            style={'margin': '0 0 8px', 'fontSize': '14px', 'color': '#444'}),
                    html.Div([
                        html.Div([
                            html.Label('FITS file path', style=_CTRL_LABEL),
                            dcc.Input(id='obs-fits-path', type='text', value='',
                                      placeholder='/path/to/cube.fits or matrix table',
                                      style={'width': '100%', 'fontSize': '13px'}),
                        ], style={'flex': '3', 'minWidth': '280px', 'marginRight': '18px'}),
                        html.Div([
                            html.Label('HDU index', style=_CTRL_LABEL),
                            dcc.Input(id='obs-hdu', type='number', value=1, min=0, step=1,
                                      style={'width': '100%'}),
                        ], style={'flex': '0.6', 'minWidth': '70px', 'marginRight': '18px'}),
                        html.Div([
                            html.Label('Spectrum selection', style=_CTRL_LABEL),
                            dcc.Dropdown(id='obs-row-mode', options=OBS_ROW_OPTIONS,
                                         value='peak', style={'fontSize': '13px'}),
                        ], style={'flex': '1.4', 'minWidth': '180px', 'marginRight': '18px'}),
                        html.Div([
                            html.Label('Row / pixel index', style=_CTRL_LABEL),
                            dcc.Input(id='obs-row-index', type='number', value=0, min=0,
                                      step=1, style={'width': '100%'}),
                        ], style={'flex': '0.8', 'minWidth': '90px', 'marginRight': '18px'}),
                        html.Div([
                            html.Label('N Gaussians', style=_CTRL_LABEL),
                            dcc.Input(id='obs-n-gauss', type='number', value=1, min=1,
                                      max=6, step=1, style={'width': '100%'}),
                        ], style={'flex': '0.7', 'minWidth': '80px'}),
                    ], style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
                              'marginBottom': '10px'}),
                    html.Div([
                        html.Div([
                            html.Label('Line core low (km/s)', style=_CTRL_LABEL),
                            dcc.Input(id='obs-v-low', type='number', value=DEFAULT_OBS_V_LOW,
                                      step=0.5, style={'width': '100%'}),
                        ], style={'flex': '1', 'minWidth': '120px', 'marginRight': '18px'}),
                        html.Div([
                            html.Label('Line core high (km/s)', style=_CTRL_LABEL),
                            dcc.Input(id='obs-v-high', type='number', value=DEFAULT_OBS_V_HIGH,
                                      step=0.5, style={'width': '100%'}),
                        ], style={'flex': '1', 'minWidth': '120px', 'marginRight': '18px'}),
                        html.Div([
                            dcc.Checklist(id='obs-overlay', options=[
                                {'label': ' Overlay on SimLine spectrum', 'value': 'on'},
                            ], value=['on'], style={'fontSize': '13px'}),
                        ], style={'flex': '1.4', 'minWidth': '200px', 'marginRight': '18px'}),
                        html.Div([
                            dcc.Checklist(id='obs-fit', options=[
                                {'label': ' Fit Gaussians (+ continuum)', 'value': 'on'},
                            ], value=['on'], style={'fontSize': '13px'}),
                        ], style={'flex': '1.4', 'minWidth': '200px'}),
                    ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap'}),
                ], className='kosma-panel',
                          style={'padding': '12px 18px', 'marginBottom': '12px'}),
                dcc.Loading(id='loading-spectra', type='circle', children=[
                    dcc.Graph(id='plot-sp-spectrum', figure=placeholder_fig(), config=_GRAPH_CFG,
                              style={'marginBottom': '8px'}),
                    html.Div(id='sp-fit-summary', style={'padding': '0 18px 12px'}),
                    dcc.Graph(id='plot-sp-pv', figure=placeholder_fig(), config=_GRAPH_CFG),
                ]),
            ]),
        ]),

        # --- Page 8: interpolation error check --------------------------------
        dcc.Tab(label='Interpolation', value='interperror', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Compare native model grids with KoSens-style resampling and '
                       'estimate interpolation error (decimate \u2192 re-interpolate on the '
                       'native mesh, matching ``resampled_grid_data`` with '
                       '``calculate_error=True``). Shows abundance / diagnostic grids when '
                       'the main HDF5 grid is loaded, and SIMLINE intensity grids when a '
                       'SIMLINE directory is loaded.',
                       style=_PAGE_INTRO),
                html.Div([
                    html.Div([
                        html.Label('Abundance / diagnostic quantity', style=_CTRL_LABEL),
                        dcc.Dropdown(id='ie-quantity', options=[], value=None,
                                     placeholder='Load a main grid\u2026',
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '2', 'minWidth': '220px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('SIMLINE species', style=_CTRL_LABEL),
                        dcc.Dropdown(id='ie-int-species', options=[], value=None,
                                     placeholder='Load SIMLINE\u2026',
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '1', 'minWidth': '120px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Transition', style=_CTRL_LABEL),
                        dcc.Dropdown(id='ie-int-transition', options=[], value=None,
                                     placeholder='Select species\u2026',
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '1.2', 'minWidth': '160px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Quantity', style=_CTRL_LABEL),
                        dcc.RadioItems(id='ie-int-idef', options=SIMLINE_IDEF_OPTIONS,
                                       value=SIMLINE_DEFAULT_IDEF, **_RADIO),
                    ], style={**_CTRL_BOX, 'marginRight': '0'}),
                ], className='kosma-panel',
                style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
                       'padding': '12px 18px'}),
                _interp_control_row(prefix='ie-'),
                _advanced_details('Error metric & display', _interp_error_control_row()),
                html.Div([_ie_plane_section(p['id']) for p in SLICE_PLANES],
                         id='ie-panels-wrap', style=_SLICE_PANELS_COL_STYLE),
            ]),
        ]),

        # --- Page 9: CR attenuation profiles ----------------------------------
        dcc.Tab(label='CR attenuation', value='cratten', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Cosmic-ray ionisation rate ζ<sub>H₂</sub> vs column density, '
                       'following the KoSens CR_atten_plot notebook. Padovani et al. '
                       '(2018/2024) reference bands 𝓛, 𝓗, 𝓤 are always available; '
                       'add one or more KOSMA model profiles from the sliders or overlay grid.',
                       style=_PAGE_INTRO),
                html.Div([
                    html.Div([
                        html.Label('X axis', style=_CTRL_LABEL),
                        dcc.RadioItems(id='cr-atten-xaxis', options=CR_ATTEN_XAXIS_OPTIONS,
                                       value='nh2', **_RADIO),
                    ], style={'flex': '1', 'minWidth': '160px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('N<sub>H₂</sub> threshold line (cm⁻²)', style=_CTRL_LABEL),
                        dcc.Input(id='cr-atten-stop-rate', type='number',
                                  value=cra.DEFAULT_STOPPING_RATE,
                                  placeholder='1e20',
                                  style={'width': '100%', 'fontSize': '13px'}),
                    ], style={'flex': '1', 'minWidth': '160px', 'marginRight': '18px'}),
                    html.Div([
                        dcc.Checklist(
                            id='cr-atten-options',
                            options=[
                                {'label': ' Padovani 𝓛 / 𝓗 / 𝓤 bands', 'value': 'padovani'},
                                {'label': ' Extrapolate profiles to 10²⁵ cm⁻²', 'value': 'extrap'},
                                {'label': ' Attenuation threshold line', 'value': 'threshold'},
                            ],
                            value=['padovani', 'extrap', 'threshold'],
                            style={'fontSize': '13px'},
                        ),
                    ], style={'flex': '1.6', 'minWidth': '240px'}),
                ], style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
                          'padding': '12px 18px',
                          'borderRadius': '8px', 'marginBottom': '10px'}),
                html.Div([
                    html.Button('Add current model', id='cr-atten-add', n_clicks=0,
                                className='btn btn-primary',
                                style={'marginRight': '8px'}),
                    html.Button('Add all overlay matches', id='cr-atten-add-overlay', n_clicks=0,
                                className='btn btn-overlay',
                                style={'marginRight': '8px'}),
                    html.Button('Clear profiles', id='cr-atten-clear', n_clicks=0,
                                className='btn btn-ghost'),
                ], style={'padding': '0 18px 10px'}),
                html.Div([
                    html.Label('Profiles to plot', style=_CTRL_LABEL),
                    dcc.Checklist(id='cr-atten-active', options=[], value=[],
                                  style={'fontSize': '13px'}),
                ], style={'padding': '0 18px 12px'}),
                dcc.Loading(id='loading-cr-atten', type='circle', children=[
                    dcc.Graph(id='plot-cr-atten', figure=placeholder_fig(), config=_GRAPH_CFG),
                ]),
            ]),
        ]),

        # --- Page 10: observational map fit ---------------------------------
        dcc.Tab(label='Map fit', value='mapfit', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Fit observational FITS intensity maps to the 3-D SIMLINE model grid '
                       '(KoSens3D ``fit_fits_maps_to_grids_3d``). Requires a main grid and '
                       'SIMLINE directory on the Load tab. The three fit axes are chosen '
                       'automatically from the varying parameters in the loaded grid '
                       '(density × FUV × ζ when all three vary; otherwise the first three '
                       'varying axes, e.g. density × mass × FUV for 4-token grids).',
                       style=_PAGE_INTRO),
                html.Div([
                    html.Div([
                        html.Label('Intensity units (SIMLINE)', style=_CTRL_LABEL),
                        dcc.RadioItems(id='fit-idef', options=SIMLINE_MAPFIT_IDEF_OPTIONS,
                                       value=SIMLINE_DEFAULT_IDEF, **_RADIO),
                    ], style={'flex': '1.2', 'minWidth': '200px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('3-D grid size (nz \u00d7 ny \u00d7 nx)', style=_CTRL_LABEL),
                        html.Div([
                            dcc.Input(id='fit-nz', type='number',
                                      value=gf.DEFAULT_TARGET_SHAPE_3D[0],
                                      min=2, max=200, step=1,
                                      style={'width': '58px', 'padding': '7px 6px',
                                             'fontSize': '13px', 'border': '1px solid #bbc',
                                             'borderRadius': '6px'}),
                            html.Span(' \u00d7 ', style={'margin': '0 4px', 'color': '#666'}),
                            dcc.Input(id='fit-ny', type='number',
                                      value=gf.DEFAULT_TARGET_SHAPE_3D[1],
                                      min=2, max=200, step=1,
                                      style={'width': '58px', 'padding': '7px 6px',
                                             'fontSize': '13px', 'border': '1px solid #bbc',
                                             'borderRadius': '6px'}),
                            html.Span(' \u00d7 ', style={'margin': '0 4px', 'color': '#666'}),
                            dcc.Input(id='fit-nx', type='number',
                                      value=gf.DEFAULT_TARGET_SHAPE_3D[2],
                                      min=2, max=200, step=1,
                                      style={'width': '58px', 'padding': '7px 6px',
                                             'fontSize': '13px', 'border': '1px solid #bbc',
                                             'borderRadius': '6px'}),
                        ]),
                    ], style={'flex': '1.1', 'minWidth': '200px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Interpolation method', style=_CTRL_LABEL),
                        dcc.Dropdown(id='fit-method', options=INTERP_METHOD_OPTIONS,
                                     value=DEFAULT_INTERP_METHOD, clearable=False,
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '1', 'minWidth': '120px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Output directory', style=_CTRL_LABEL),
                        dcc.Input(id='fit-output-dir', type='text', value='',
                                  placeholder='./map_fit_output (default)',
                                  style={'width': '100%', 'padding': '7px 9px',
                                         'fontSize': '13px', 'border': '1px solid #bbc',
                                         'borderRadius': '6px'}),
                    ], style={'flex': '2', 'minWidth': '220px'}),
                ], className='kosma-panel',
                style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
                       'padding': '12px 18px'}),
                html.Div([
                    html.Div([
                        html.Label('Observed FITS maps (JSON: line \u2192 path)', style=_CTRL_LABEL),
                        dcc.Textarea(
                            id='fit-maps-json', value='', placeholder=_MAP_FIT_MAPS_EXAMPLE,
                            style={'width': '100%', 'height': '120px', 'fontFamily': 'monospace',
                                   'fontSize': '12px', 'padding': '8px',
                                   'border': '1px solid #bbc', 'borderRadius': '6px'}),
                    ], style={'flex': '1', 'minWidth': '280px', 'marginRight': '14px'}),
                    html.Div([
                        html.Label('Map errors (JSON: line \u2192 scalar or FITS path)', style=_CTRL_LABEL),
                        dcc.Textarea(
                            id='fit-errors-json', value='', placeholder=_MAP_FIT_ERRORS_EXAMPLE,
                            style={'width': '100%', 'height': '120px', 'fontFamily': 'monospace',
                                   'fontSize': '12px', 'padding': '8px',
                                   'border': '1px solid #bbc', 'borderRadius': '6px'}),
                    ], style={'flex': '1', 'minWidth': '240px'}),
                ], style={'display': 'flex', 'padding': '0 18px 12px'}),
                html.Div([
                    html.Div([
                        html.Label('χ² analysis pixel (row, col)', style=_CTRL_LABEL),
                        html.Div([
                            dcc.Input(id='fit-chi2-i', type='number', value=98, min=0, step=1,
                                      style={'width': '64px', 'padding': '7px 6px',
                                             'fontSize': '13px', 'border': '1px solid #bbc',
                                             'borderRadius': '6px'}),
                            html.Span(', ', style={'margin': '0 4px'}),
                            dcc.Input(id='fit-chi2-j', type='number', value=48, min=0, step=1,
                                      style={'width': '64px', 'padding': '7px 6px',
                                             'fontSize': '13px', 'border': '1px solid #bbc',
                                             'borderRadius': '6px'}),
                        ]),
                    ], style={'flex': '1', 'minWidth': '160px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Fit options', style=_CTRL_LABEL),
                        dcc.Checklist(
                            id='fit-options',
                            options=[
                                {'label': ' create observed ratios', 'value': 'ratios'},
                                {'label': ' create model ratios', 'value': 'model_ratios'},
                                {'label': ' χ² analysis at pixel', 'value': 'chi2'},
                                {'label': ' uncertainty maps (slow)', 'value': 'unc'},
                            ],
                            value=['ratios', 'model_ratios', 'chi2'],
                            style={'fontSize': '13px'},
                        ),
                    ], style={'flex': '2', 'minWidth': '260px', 'marginRight': '18px'}),
                    html.Div([
                        html.Button('Run map fit', id='btn-run-map-fit', n_clicks=0,
                                    className='btn btn-success'),
                    ], style={'flex': '0', 'alignSelf': 'flex-end'}),
                ], className='kosma-panel',
                style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
                       'padding': '12px 18px'}),
                html.Div(id='fit-available-lines',
                         style={'padding': '0 18px 8px', 'fontSize': '12px', 'color': '#666'}),
                html.Div(id='fit-status', style={'padding': '0 18px 12px'}),
                dcc.Store(id='fit-state', data=0),
                dcc.Loading(type='circle', children=[
                    html.Div([
                        dcc.Graph(id='plot-fit-x', figure=placeholder_fig('Run a fit'),
                                  config=_GRAPH_CFG, style={'width': '100%'}),
                    ], style={'padding': '0 18px', 'marginBottom': '12px'}),
                    html.Div([
                        dcc.Graph(id='plot-fit-y', figure=placeholder_fig(),
                                  config=_GRAPH_CFG, style={'width': '100%'}),
                    ], style={'padding': '0 18px', 'marginBottom': '12px'}),
                    html.Div([
                        dcc.Graph(id='plot-fit-z', figure=placeholder_fig(),
                                  config=_GRAPH_CFG, style={'width': '100%'}),
                    ], style={'padding': '0 18px', 'marginBottom': '12px'}),
                    html.Div([
                        dcc.Graph(id='plot-fit-chi2', figure=placeholder_fig(),
                                  config=_GRAPH_CFG, style={'width': '100%'}),
                    ], style={'padding': '0 18px', 'marginBottom': '12px'}),
                    html.Div([
                        dcc.Graph(id='plot-fit-dominant', figure=placeholder_fig(),
                                  config=_GRAPH_CFG, style={'width': '100%'}),
                    ], style={'padding': '0 18px', 'marginBottom': '12px'}),
                    html.H4('Fitted parameter distributions', style={'padding': '0 18px',
                                                                    'margin': '8px 0 4px',
                                                                    'fontSize': '15px', 'color': '#444'}),
                    html.Div([
                        dcc.Graph(id='plot-fit-kde', figure=placeholder_fig(),
                                  config=_GRAPH_CFG, style={'width': '100%'}),
                    ], style={'padding': '0 18px', 'marginBottom': '12px'}),
                    html.H4('χ² analysis at pixel', style={'padding': '0 18px', 'margin': '8px 0 4px',
                                                           'fontSize': '15px', 'color': '#444'}),
                    html.P('Set row/column above and re-run the fit, or change the pixel after a fit '
                             'to refresh the detailed χ² plots.',
                           style={'padding': '0 18px', 'margin': '0 0 8px', 'fontSize': '12px',
                                  'color': '#666'}),
                    html.Div([
                        dcc.Graph(id='plot-fit-chi2-pixel', figure=placeholder_fig(),
                                  config=_GRAPH_CFG, style={'width': '100%'}),
                    ], style={'padding': '0 18px', 'marginBottom': '12px'}),
                    html.Div([
                        dcc.Graph(id='plot-fit-chi2-corner', figure=placeholder_fig(),
                                  config=_GRAPH_CFG, style={'width': '100%'}),
                    ], style={'padding': '0 18px', 'marginBottom': '12px'}),
                    html.Div([
                        dcc.Graph(id='plot-fit-chi2-z', figure=placeholder_fig(),
                                  config=_GRAPH_CFG, style={'width': '100%'}),
                    ], style={'padding': '0 18px', 'marginBottom': '12px'}),
                    html.Div([
                        dcc.Graph(id='plot-fit-chi2-bars', figure=placeholder_fig(),
                                  config=_GRAPH_CFG, style={'width': '100%'}),
                    ], style={'padding': '0 18px', 'marginBottom': '12px'}),
                    html.Div([
                        dcc.Graph(id='plot-fit-species-contours', figure=placeholder_fig(),
                                  config=_GRAPH_CFG, style={'width': '100%'}),
                    ], style={'padding': '0 18px'}),
                ]),
            ]),
        ]),
    ]),

    # Footer with export options
    html.Div([
        html.Hr(style={'margin': '16px 0 12px', 'borderColor': '#e8e8e8'}),
        html.Div([
            html.Div([
                html.Span('\u2B07 Export format for camera icon:', 
                          style={'fontWeight': '600', 'fontSize': '12px',
                                 'marginRight': '10px', 'color': '#555'}),
                dcc.RadioItems(
                    id='export-format-selector',
                    options=[
                        {'label': ' PNG (raster)', 'value': 'png'},
                        {'label': ' SVG (vector)', 'value': 'svg'},
                    ],
                    value='png',
                    inline=True,
                    style={'fontSize': '12px'},
                    labelStyle={'marginRight': '16px', 'cursor': 'pointer'},
                    inputStyle={'marginRight': '4px'},
                ),
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                      'flexWrap': 'wrap', 'gap': '4px'}),
            html.P([
                'Select format above, then click the ',
                html.Span('\U0001F4F7', style={'fontSize': '14px'}),
                ' camera icon on any plot. ',
                html.Span('SVG', style={'fontWeight': '600'}),
                ' is vector format \u2014 ideal for papers & easily converted to PDF.',
            ], style={'color': '#888', 'fontSize': '12px', 'textAlign': 'center', 'margin': '8px 0 6px'}),
        ], className='kosma-panel', style={'padding': '12px 18px', 'marginBottom': '10px'}),
        html.P('PRISMA  \u2014  PDR Results Interactive Simulation Mapping & Analysis',
               style={'color': '#ccc', 'fontSize': '11px', 'textAlign': 'center', 'margin': '0 0 6px'}),
    ]),
])


# --- Callbacks ----------------------------------------------------------------

# Outputs: status, controls style, then per-slider (max, marks, value, wrap-style),
# then species options + value.
_slider_outputs = []
for d in range(N_PARAMS):
    _slider_outputs += [
        Output(f'slider-{d}', 'max'),
        Output(f'slider-{d}', 'marks'),
        Output(f'slider-{d}', 'value'),
        Output({'role': 'slider-wrap', 'idx': d}, 'style'),
    ]

_slice_slider_outputs = []
for plane in SLICE_PLANES:
    pid = plane['id']
    _slice_slider_outputs += [
        Output(f'slice-slider-{pid}', 'max'),
        Output(f'slice-slider-{pid}', 'marks'),
        Output(f'slice-slider-{pid}', 'value'),
    ]

_slice_slider_inputs = [Input(f'slice-slider-{p["id"]}', 'value') for p in SLICE_PLANES]
_slice_label_outputs = [Output(f'slice-label-{p["id"]}', 'children') for p in SLICE_PLANES]
_contour_outputs = [Output(f'plot-contour-{p["id"]}', 'figure') for p in SLICE_PLANES]

_rgb_slice_slider_outputs = []
for plane in SLICE_PLANES:
    pid = plane['id']
    _rgb_slice_slider_outputs += [
        Output(f'rgb-slice-slider-{pid}', 'max'),
        Output(f'rgb-slice-slider-{pid}', 'marks'),
        Output(f'rgb-slice-slider-{pid}', 'value'),
    ]
_rgb_slice_slider_inputs = [Input(f'rgb-slice-slider-{p["id"]}', 'value') for p in SLICE_PLANES]
_rgb_slice_label_outputs = [Output(f'rgb-slice-label-{p["id"]}', 'children') for p in SLICE_PLANES]
_rgb_outputs = [Output(f'plot-rgb-{p["id"]}', 'figure') for p in SLICE_PLANES]
_rgb_panel_style_outputs = [Output(f'rgb-panel-{p["id"]}', 'style') for p in SLICE_PLANES]
_rgb_graph_style_outputs = [Output(f'plot-rgb-{p["id"]}', 'style') for p in SLICE_PLANES]

_int_slice_slider_outputs = []
for plane in SLICE_PLANES:
    pid = plane['id']
    _int_slice_slider_outputs += [
        Output(f'int-slice-slider-{pid}', 'max'),
        Output(f'int-slice-slider-{pid}', 'marks'),
        Output(f'int-slice-slider-{pid}', 'value'),
    ]

_int_slice_slider_inputs = [Input(f'int-slice-slider-{p["id"]}', 'value') for p in SLICE_PLANES]
_int_slice_label_outputs = [Output(f'int-slice-label-{p["id"]}', 'children') for p in SLICE_PLANES]
_int_contour_outputs = [Output(f'plot-int-contour-{p["id"]}', 'figure') for p in SLICE_PLANES]

_int_rgb_slice_slider_outputs = []
for plane in SLICE_PLANES:
    pid = plane['id']
    _int_rgb_slice_slider_outputs += [
        Output(f'int-rgb-slice-slider-{pid}', 'max'),
        Output(f'int-rgb-slice-slider-{pid}', 'marks'),
        Output(f'int-rgb-slice-slider-{pid}', 'value'),
    ]
_int_rgb_slice_slider_inputs = [
    Input(f'int-rgb-slice-slider-{p["id"]}', 'value') for p in SLICE_PLANES
]
_int_rgb_slice_label_outputs = [
    Output(f'int-rgb-slice-label-{p["id"]}', 'children') for p in SLICE_PLANES
]
_int_rgb_outputs = [Output(f'plot-int-rgb-{p["id"]}', 'figure') for p in SLICE_PLANES]
_int_rgb_panel_style_outputs = [Output(f'int-rgb-panel-{p["id"]}', 'style') for p in SLICE_PLANES]
_int_rgb_graph_style_outputs = [Output(f'plot-int-rgb-{p["id"]}', 'style') for p in SLICE_PLANES]

_int_spag_slice_slider_outputs = []
for plane in SLICE_PLANES:
    pid = plane['id']
    _int_spag_slice_slider_outputs += [
        Output(f'int-spag-slice-slider-{pid}', 'max'),
        Output(f'int-spag-slice-slider-{pid}', 'marks'),
        Output(f'int-spag-slice-slider-{pid}', 'value'),
    ]
_int_spag_slice_slider_inputs = [
    Input(f'int-spag-slice-slider-{p["id"]}', 'value') for p in SLICE_PLANES
]
_int_spag_slice_label_outputs = [
    Output(f'int-spag-slice-label-{p["id"]}', 'children') for p in SLICE_PLANES
]
_int_spag_outputs = [Output(f'plot-int-spag-{p["id"]}', 'figure') for p in SLICE_PLANES]
_int_spag_chi2_outputs = [Output(f'plot-int-spag-chi2-{p["id"]}', 'figure') for p in SLICE_PLANES]
_int_spag_panel_style_outputs = [Output(f'int-spag-panel-{p["id"]}', 'style') for p in SLICE_PLANES]
_int_spag_graph_style_outputs = [Output(f'plot-int-spag-{p["id"]}', 'style') for p in SLICE_PLANES]

_ie_slice_slider_outputs = []
for plane in SLICE_PLANES:
    pid = plane['id']
    _ie_slice_slider_outputs += [
        Output(f'ie-slice-slider-{pid}', 'max'),
        Output(f'ie-slice-slider-{pid}', 'marks'),
        Output(f'ie-slice-slider-{pid}', 'value'),
    ]

_ie_slice_slider_inputs = [Input(f'ie-slice-slider-{p["id"]}', 'value') for p in SLICE_PLANES]
_ie_slice_label_outputs = [Output(f'ie-slice-label-{p["id"]}', 'children') for p in SLICE_PLANES]
_ie_abund_outputs = [Output(f'plot-ie-abund-{p["id"]}', 'figure') for p in SLICE_PLANES]
_ie_int_outputs = [Output(f'plot-ie-int-{p["id"]}', 'figure') for p in SLICE_PLANES]
_ie_abund_wrap_outputs = [Output(f'ie-abund-wrap-{p["id"]}', 'style') for p in SLICE_PLANES]
_ie_int_wrap_outputs = [Output(f'ie-int-wrap-{p["id"]}', 'style') for p in SLICE_PLANES]


def _contour_quantity_options(species):
    opts = [{'label': lb, 'value': v} for v, lb in CONTOUR_DIAGNOSTICS]
    opts += [{'label': f'X({s})', 'value': f'species:{s}'} for s in species]
    return opts


def _contour_cdens_options(species):
    return [{'label': f'N({s})', 'value': f'cdens:{s}'}
            for s in species if tot_col_dens_hdf5_key(s) in _field_map]


def _combined_contour_quantities(abund, cdens):
    """Abundance/diagnostic selection, then column densities (both may be empty)."""
    return _as_str_list(abund) + _as_str_list(cdens)


def _default_contour_quantity(species_idx):
    if 'CO' in species_idx:
        return CONTOUR_DEFAULT_QUANTITY
    if species_idx:
        return f'species:{next(iter(species_idx))}'
    return 'tgas'


def _species_dropdown_cfg(species, species_idx):
    sp_opts = [{'label': s, 'value': s} for s in species]
    defaults = [s for s in DEFAULT_CUSTOM if s in species_idx][:3]
    cq_opts = _contour_quantity_options(species)
    cq_val = _default_contour_quantity(species_idx)
    cd_opts = _contour_cdens_options(species)
    return (sp_opts, defaults, cq_opts, [cq_val] if cq_val else [],
            cd_opts, [], list(cq_opts), cq_val)


def _active_slice_plane_ids():
    planes = [p for p in active_slice_planes() if p.get('active')]
    return [p['id'] for p in planes] or [p['id'] for p in SLICE_PLANES]


def _extra_slice_caption(text):
    return html.Div(text, style={
        'fontSize': '12px', 'fontWeight': '600', 'opacity': 0.88,
        'margin': '12px 0 4px', 'textAlign': 'center',
    })


def _match_output_id():
    """Pattern-matching component id of the first Output of this callback."""
    out = dash.callback_context.outputs_list[0]
    if isinstance(out, list):
        out = out[0]
    return out['id']


def _ratio_pair_value(num, den):
    return f'{num}||{den}'


def _ratio_pair_parts(value):
    text = str(value or '')
    if '||' in text:
        num, den = text.split('||', 1)
        return num, den
    if '/' in text:
        return text.split('/', 1)
    return text, ''


def _species_from_contour_quantities(quantities):
    return [q.split(':', 1)[1] for q in _as_str_list(quantities)
            if str(q).startswith('species:')]


def _grid_ratio_dropdown(quantities, current=None):
    species = _species_from_contour_quantities(quantities)
    hint = ''
    if len(species) < 2:
        hint = ('Select at least two species in the quantity list to build '
                'KoSens abundance ratios.')
        return [], [], hint
    pairs = mfe.abundance_ratio_pairs(species)
    opts = [{'label': f'X({n})/X({d})', 'value': _ratio_pair_value(n, d)}
            for n, d in pairs]
    valid = {o['value'] for o in opts}
    kept = [v for v in _as_str_list(current) if v in valid]
    default = kept or ([opts[0]['value']] if opts else [])
    return opts, default, hint


def _intensity_line_key(species, tidx, idef):
    opts = _simline_transition_options(species, idef)
    try:
        raw = opts[int(tidx)].get('transition')
    except (TypeError, ValueError, IndexError, AttributeError):
        return None
    if not raw:
        return None
    return gf.spectroscopic_line_key(species, raw)


def _selected_line_lookup(species, transition, idef):
    lookup = {}
    for sp, tidx in _intensity_map_combos(species, transition, idef):
        key = _intensity_line_key(sp, tidx, idef)
        if key:
            lookup[key] = (sp, tidx)
    return lookup


def _int_ratio_dropdown(species, transition, idef, current=None):
    hint = ''
    species_list = _as_str_list(species)
    trans_list = _as_str_list(transition)
    if not species_list:
        hint = ('Select a species and at least two transitions, or two or more '
                'species, to build KoSens line-intensity ratios.')
        return [], [], hint
    if not trans_list:
        hint = ('Select at least one transition to build KoSens line-intensity ratios.')
        return [], [], hint
    if len(species_list) < 2 and len(trans_list) < 2:
        hint = ('Select at least two transitions of this species, or a second '
                'species, to build KoSens line-intensity ratios.')
        return [], [], hint
    lookup = _selected_line_lookup(species, transition, idef)
    pairs = mfe.transition_ratio_pairs(list(lookup))
    opts = [{'label': f'{n}/{d}', 'value': _ratio_pair_value(n, d)}
            for n, d in pairs]
    valid = {o['value'] for o in opts}
    kept = [v for v in _as_str_list(current) if v in valid]
    default = kept or ([opts[0]['value']] if opts else [])
    if lookup and not pairs:
        hint = ('No KoSens ratios for these lines (need parseable spectroscopic '
                'keys with distinct energy levels; inter-molecule pairs must '
                'differ in priority).')
    return opts, default, hint


def _divided_slice_grid(num_grid, den_grid):
    if num_grid is None or den_grid is None:
        return None
    x_phys, y_phys, zn, xdef, ydef = num_grid
    _, _, zd, _, _ = den_grid
    if zn.shape != zd.shape:
        return None
    z_ratio = mfe.divide_ratio_grids(zn, zd)
    return x_phys, y_phys, z_ratio, xdef, ydef


def _pattern_role(ident, role):
    out = dict(ident)
    out['role'] = role
    return out


def _extra_slice_key(ident):
    return (ident.get('qty'), ident.get('sp'), ident.get('tidx'),
            ident.get('pair'), ident.get('plane'))


def _saved_extra_slices(ids, values):
    saved = {}
    for ident, val in zip(ids or [], values or []):
        if ident:
            saved[_extra_slice_key(ident)] = val
    return saved


def _slice_seed_by_plane(slice_indices):
    return {
        p['id']: (slice_indices[i] if i < len(slice_indices) else 0)
        for i, p in enumerate(SLICE_PLANES)
    }


def _extra_panel_slice_idx(ident, seed_by_plane, saved):
    key = _extra_slice_key(ident)
    if key in saved:
        return saved[key]
    return seed_by_plane.get(ident.get('plane'), 0)


def _extra_slice_slider_props(plane_id, value=None):
    plane = _plane_by_slot(plane_id)
    if not _grid or not plane.get('active') or not plane.get('slice'):
        return dict(min=0, max=1, step=1, value=0, marks={},
                    tooltip={'placement': 'top', 'always_visible': False})
    sdef = _param_def(plane['slice'])
    tokens = _grid['axis_tokens'][plane['slice']]
    n = len(tokens)
    vmax = max(n - 1, 0)
    try:
        ival = int(value) if value is not None else n // 2
    except (TypeError, ValueError):
        ival = n // 2
    ival = min(max(ival, 0), vmax)
    return dict(
        min=0, max=vmax, step=1, value=ival,
        marks=build_marks(sdef, _grid['axis_tokens']),
        tooltip={'placement': 'top', 'always_visible': False},
    )


def _extra_slice_graph_row(idents, full_width=False, figures=None, slice_indices=None,
                           slider_role='grid-extra-slider',
                           label_role='grid-extra-label',
                           fixed_role='grid-extra-fixed'):
    """One row of extra slice panels, each with its own third-axis slider."""
    wrap = _SLICE_PANELS_COL_STYLE if full_width else {
        **_SLICE_PANELS_ROW_STYLE, 'marginTop': '4px',
    }
    panel = _SLICE_PANEL_ROW_FULL if full_width else _SLICE_PANEL_ROW
    gstyle = _SLICE_GRAPH_STYLE_FULL if full_width else _SLICE_GRAPH_STYLE
    if not figures or len(figures) != len(idents):
        figures = [placeholder_fig() for _ in idents]
    if not slice_indices or len(slice_indices) != len(idents):
        slice_indices = [0] * len(idents)
    children = []
    for ident, fig, sl_idx in zip(idents, figures, slice_indices):
        props = _extra_slice_slider_props(ident['plane'], sl_idx)
        fixed, val, style = _slice_axis_ui(_plane_by_slot(ident['plane']), props['value'])
        children.append(html.Div([
            html.Div([
                html.Label(fixed, id=_pattern_role(ident, fixed_role),
                           style={**_CTRL_LABEL, 'fontSize': '12px'}),
                dcc.Slider(id=_pattern_role(ident, slider_role), **props),
                html.Div(val, id=_pattern_role(ident, label_role), style=style),
            ], style={'padding': '0 4px 8px'}),
            dcc.Graph(
                id=ident,
                figure=fig,
                config=_SLICE_GRAPH_CFG,
                style=gstyle,
            ),
        ], style=panel))
    return html.Div(children, style=wrap)


def _plain_quantity_caption(quantity):
    if str(quantity).startswith('species:'):
        return f'X({quantity.split(":", 1)[1]})'
    if str(quantity).startswith('cdens:'):
        return f'N({quantity.split(":", 1)[1]})'
    for key, label in CONTOUR_DIAGNOSTICS:
        if key == quantity:
            return (label.replace('<sub>', '').replace('</sub>', '')
                    .replace('<sup>', '').replace('</sup>', ''))
    return str(quantity)


def _intensity_combo_caption(species, tidx, idef):
    opts = _simline_transition_options(species, idef)
    try:
        tlabel = opts[int(tidx)]['label']
    except (IndexError, TypeError, ValueError, KeyError):
        tlabel = str(tidx)
    return f'{species}  —  {tlabel}'


def _grid_ratio_caption(pair_value):
    num, den = _ratio_pair_parts(pair_value)
    return f'X({num}) / X({den})'


def _int_ratio_caption(pair_value):
    num, den = _ratio_pair_parts(pair_value)
    return f'{num} / {den}'


def _slice_index_for_plane(plane_id, slice_indices):
    for i, p in enumerate(SLICE_PLANES):
        if p['id'] == plane_id and i < len(slice_indices):
            return slice_indices[i]
    return 0


def _grid_contour_extra_figures(idents, zscale, shift_dir, shift_rtol,
                                interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                                interp_method, interp_clip, plot_theme, grid_colorscale,
                                slice_indices):
    """Figures for dynamically created extra grid-slice graphs.

    Species that share a plane and slice index are filled in one grid walk.
    """
    theme = _parse_plot_theme(plot_theme)
    if not idents:
        return []
    if not _grid:
        return [placeholder_fig('Load a grid directory', theme=theme)] * len(idents)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    icfg = _interp_config(interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip)
    per_ident = len(slice_indices) == len(idents)
    groups = {}
    for i, ident in enumerate(idents):
        sl_idx = slice_indices[i] if per_ident else _slice_index_for_plane(
            ident['plane'], slice_indices)
        groups.setdefault((ident['plane'], sl_idx), []).append((i, ident))

    figs = [None] * len(idents)
    for (plane_id, sl_idx), items in groups.items():
        plane = _plane_by_slot(plane_id)
        if not plane.get('active'):
            pfig = placeholder_fig(theme=theme)
            for i, _ident in items:
                figs[i] = pfig
            continue
        slice_token = _slice_token_from_index(plane, sl_idx)
        qtys = [ident['qty'] for _, ident in items]
        grids = build_slice_grids_for_quantities(
            plane, slice_token, qtys, interp_config=icfg)
        ov_grids = {}
        if _overlay:
            ov_grids = build_slice_grids_for_quantities(
                plane, slice_token, qtys, interp_config=icfg, overlay=True)
        for i, ident in items:
            grid = grids.get(ident['qty'])
            figs[i] = fig_contour_plane(
                plane, sl_idx, ident['qty'], zscale or 'log',
                shift_rtol=_parse_shift_rtol(shift_rtol),
                shift_scan_direction=_parse_shift_scan_direction(shift_dir),
                interp_config=icfg, colorscale=colorscale, theme=theme,
                slice_grid=grid, overlay_grid=ov_grids.get(ident['qty']),
                show_lines=False,
            )
    return figs


def _int_contour_extra_figures(idents, idef, zscale, shift_dir, shift_rtol,
                               interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                               interp_method, interp_clip, obs_boundary, extra_contours,
                               extra_contour_color, plot_theme, grid_colorscale,
                               slider_values, slice_indices):
    """Figures for dynamically created extra intensity-slice graphs."""
    theme = _parse_plot_theme(plot_theme)
    if not idents:
        return []
    if not _grid or not _simline:
        p = placeholder_fig('Load grids and a SIMLINE directory', theme=theme)
        return [p] * len(idents)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    icfg = _interp_config(interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip)
    line_contours = _build_intensity_line_contours(
        idef or SIMLINE_DEFAULT_IDEF, zscale or 'log',
        bool(obs_boundary and 'show' in obs_boundary),
        extra_contours, extra_contour_color,
    )
    figs = []
    per_ident = len(slice_indices) == len(idents)
    for i, ident in enumerate(idents):
        try:
            tidx = int(ident['tidx'])
        except (TypeError, ValueError):
            tidx = 0
        sl_idx = slice_indices[i] if per_ident else _slice_index_for_plane(
            ident['plane'], slice_indices)
        plane = _plane_by_slot(ident['plane'])
        figs.append(fig_intensity_contour_plane(
            plane, sl_idx, ident['sp'], idef or SIMLINE_DEFAULT_IDEF, tidx, zscale or 'log',
            shift_rtol=_parse_shift_rtol(shift_rtol),
            shift_scan_direction=_parse_shift_scan_direction(shift_dir),
            interp_config=icfg, colorscale=colorscale, theme=theme,
            slider_values=slider_values, line_contours=line_contours,
        ))
    return figs


def _grid_ratio_figures(idents, zscale, shift_dir, shift_rtol,
                        interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                        interp_method, interp_clip, plot_theme, grid_colorscale,
                        slice_indices):
    """KoSens abundance-ratio contour figures for extra ratio rows."""
    theme = _parse_plot_theme(plot_theme)
    if not idents:
        return []
    if not _grid:
        return [placeholder_fig('Load a grid directory', theme=theme)] * len(idents)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    icfg = _interp_config(interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip)
    per_ident = len(slice_indices) == len(idents)
    groups = {}
    for i, ident in enumerate(idents):
        sl_idx = slice_indices[i] if per_ident else _slice_index_for_plane(
            ident['plane'], slice_indices)
        groups.setdefault((ident['plane'], sl_idx), []).append((i, ident))

    figs = [None] * len(idents)
    for (plane_id, sl_idx), items in groups.items():
        plane = _plane_by_slot(plane_id)
        if not plane.get('active'):
            pfig = placeholder_fig(theme=theme)
            for i, _ident in items:
                figs[i] = pfig
            continue
        species = []
        for _i, ident in items:
            num, den = _ratio_pair_parts(ident['pair'])
            species.extend([num, den])
        qtys = [f'species:{s}' for s in dict.fromkeys(species) if s]
        slice_token = _slice_token_from_index(plane, sl_idx)
        grids = build_slice_grids_for_quantities(
            plane, slice_token, qtys, interp_config=icfg)
        ov_grids = {}
        if _overlay:
            ov_grids = build_slice_grids_for_quantities(
                plane, slice_token, qtys, interp_config=icfg, overlay=True)
        for i, ident in items:
            num, den = _ratio_pair_parts(ident['pair'])
            label = f'X({num})/X({den})'
            grid = _divided_slice_grid(grids.get(f'species:{num}'),
                                       grids.get(f'species:{den}'))
            ov = _divided_slice_grid(ov_grids.get(f'species:{num}'),
                                     ov_grids.get(f'species:{den}')) if ov_grids else None
            figs[i] = fig_contour_plane(
                plane, sl_idx, label, zscale or 'log',
                shift_rtol=_parse_shift_rtol(shift_rtol),
                shift_scan_direction=_parse_shift_scan_direction(shift_dir),
                interp_config=icfg, colorscale=colorscale, theme=theme,
                slice_grid=grid, overlay_grid=ov, show_lines=False,
            )
    return figs


def _int_ratio_figures(idents, idef, zscale, shift_dir, shift_rtol,
                       interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                       interp_method, interp_clip, plot_theme, grid_colorscale,
                       slider_values, slice_indices, species_value, transition_value):
    """KoSens line-intensity ratio contour figures."""
    theme = _parse_plot_theme(plot_theme)
    if not idents:
        return []
    if not _grid or not _simline:
        p = placeholder_fig('Load grids and a SIMLINE directory', theme=theme)
        return [p] * len(idents)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    icfg = _interp_config(interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip)
    lookup = _selected_line_lookup(species_value, transition_value, idef)
    per_ident = len(slice_indices) == len(idents)
    groups = {}
    for i, ident in enumerate(idents):
        sl_idx = slice_indices[i] if per_ident else _slice_index_for_plane(
            ident['plane'], slice_indices)
        groups.setdefault((ident['plane'], sl_idx), []).append((i, ident))

    figs = [None] * len(idents)
    for (plane_id, sl_idx), items in groups.items():
        plane = _plane_by_slot(plane_id)
        if not plane.get('active'):
            pfig = placeholder_fig(theme=theme)
            for i, _ident in items:
                figs[i] = pfig
            continue
        needed = {}
        for _i, ident in items:
            num, den = _ratio_pair_parts(ident['pair'])
            for key in (num, den):
                combo = lookup.get(key)
                if combo:
                    needed[key] = combo
        line_grids = {}
        ov_line_grids = {}
        for key, (sp, tidx) in needed.items():
            line_grids[key] = build_intensity_slice_grid(
                plane, _slice_token_from_index(plane, sl_idx),
                sp, idef or SIMLINE_DEFAULT_IDEF, tidx,
                interp_config=icfg, slider_values=slider_values)
            if _simline_overlay:
                ov_line_grids[key] = build_intensity_slice_grid(
                    plane, _slice_token_from_index(plane, sl_idx),
                    sp, idef or SIMLINE_DEFAULT_IDEF, tidx, overlay=True,
                    interp_config=icfg, slider_values=slider_values)
        for i, ident in items:
            num, den = _ratio_pair_parts(ident['pair'])
            label = f'{num}/{den}'
            grid = _divided_slice_grid(line_grids.get(num), line_grids.get(den))
            ov = _divided_slice_grid(ov_line_grids.get(num), ov_line_grids.get(den)
                                     ) if ov_line_grids else None
            figs[i] = fig_contour_plane(
                plane, sl_idx, label, zscale or 'log',
                shift_rtol=_parse_shift_rtol(shift_rtol),
                shift_scan_direction=_parse_shift_scan_direction(shift_dir),
                interp_config=icfg, colorscale=colorscale, theme=theme,
                slice_grid=grid, overlay_grid=ov, show_lines=False,
            )
    return figs


def _slider_wrap_styles():
    hidden = {'display': 'none', 'flex': '1 1 240px', 'minWidth': '210px',
              'marginRight': '24px', 'marginBottom': '8px'}
    shown = {**hidden, 'display': 'block'}
    return hidden, shown


def _ui_slider_config(axis_tokens):
    """Build Dash slider outputs from an ``axis_tokens`` dict."""
    hidden, shown = _slider_wrap_styles()
    slider_cfg = []
    for d, p in enumerate(PARAM_DEFS):
        tokens = axis_tokens[p['key']]
        n = len(tokens)
        varies = n > 1
        marks = build_marks(p, axis_tokens)
        value = n // 2
        slider_cfg += [max(n - 1, 0), marks, value, (shown if varies else hidden)]

    slice_cfg = []
    for plane in active_slice_planes():
        if plane.get('active') and plane.get('slice'):
            sdef = _param_def(plane['slice'])
            tokens = axis_tokens[plane['slice']]
            n = len(tokens)
            slice_cfg += [max(n - 1, 0), build_marks(sdef, axis_tokens), n // 2]
        else:
            slice_cfg += [1, {}, 0]
    return slider_cfg, slice_cfg, hidden


def _empty_param_slider_ui(hidden):
    empty = []
    for _ in range(N_PARAMS):
        empty += [1, {}, 0, hidden]
    return empty


def _empty_slice_slider_ui():
    empty = []
    for _ in SLICE_PLANES:
        empty += [1, {}, 0]
    return empty


def _interleave_slice_tab_cfgs(slice_cfg, n_tabs=6):
    """Match ``_grid_sync_outputs``: per plane, one (max, marks, value) block per tab.

    Tabs: contour slice, intensity slice, IE slice, grid RGB, intensity RGB,
    intensity spaghetti.
    ``handle_load`` uses grouped outputs (all of tab A, then B, …), so it can
    pass ``slice_cfg`` six times.  SIMLINE bootstrap uses interleaved outputs.
    """
    out = []
    for i in range(len(SLICE_PLANES)):
        plane = slice_cfg[i * 3:(i + 1) * 3]
        out += plane * n_tabs
    return out


_grid_sync_outputs = [Output('grid-loaded', 'data', allow_duplicate=True)]
for d in range(N_PARAMS):
    _grid_sync_outputs += [
        Output(f'slider-{d}', 'max', allow_duplicate=True),
        Output(f'slider-{d}', 'marks', allow_duplicate=True),
        Output(f'slider-{d}', 'value', allow_duplicate=True),
        Output({'role': 'slider-wrap', 'idx': d}, 'style', allow_duplicate=True),
    ]
for plane in SLICE_PLANES:
    pid = plane['id']
    _grid_sync_outputs += [
        Output(f'slice-slider-{pid}', 'max', allow_duplicate=True),
        Output(f'slice-slider-{pid}', 'marks', allow_duplicate=True),
        Output(f'slice-slider-{pid}', 'value', allow_duplicate=True),
        Output(f'int-slice-slider-{pid}', 'max', allow_duplicate=True),
        Output(f'int-slice-slider-{pid}', 'marks', allow_duplicate=True),
        Output(f'int-slice-slider-{pid}', 'value', allow_duplicate=True),
        Output(f'ie-slice-slider-{pid}', 'max', allow_duplicate=True),
        Output(f'ie-slice-slider-{pid}', 'marks', allow_duplicate=True),
        Output(f'ie-slice-slider-{pid}', 'value', allow_duplicate=True),
        Output(f'rgb-slice-slider-{pid}', 'max', allow_duplicate=True),
        Output(f'rgb-slice-slider-{pid}', 'marks', allow_duplicate=True),
        Output(f'rgb-slice-slider-{pid}', 'value', allow_duplicate=True),
        Output(f'int-rgb-slice-slider-{pid}', 'max', allow_duplicate=True),
        Output(f'int-rgb-slice-slider-{pid}', 'marks', allow_duplicate=True),
        Output(f'int-rgb-slice-slider-{pid}', 'value', allow_duplicate=True),
        Output(f'int-spag-slice-slider-{pid}', 'max', allow_duplicate=True),
        Output(f'int-spag-slice-slider-{pid}', 'marks', allow_duplicate=True),
        Output(f'int-spag-slice-slider-{pid}', 'value', allow_duplicate=True),
    ]
_grid_sync_outputs += [
    Output('species-selector', 'options', allow_duplicate=True),
    Output('species-selector', 'value', allow_duplicate=True),
    Output('contour-quantity', 'options', allow_duplicate=True),
    Output('contour-quantity', 'value', allow_duplicate=True),
    Output('contour-cdens', 'options', allow_duplicate=True),
    Output('contour-cdens', 'value', allow_duplicate=True),
    Output('ie-quantity', 'options', allow_duplicate=True),
    Output('ie-quantity', 'value', allow_duplicate=True),
]
_N_GRID_SYNC = len(_grid_sync_outputs)


def _empty_grid_sync(loaded=False):
    hidden, _ = _slider_wrap_styles()
    empty = _empty_param_slider_ui(hidden)
    empty_slice = _empty_slice_slider_ui()
    empty_dropdowns = [[], [], [], [], [], [], [], None]
    return ([loaded] + empty + _interleave_slice_tab_cfgs(empty_slice)
            + empty_dropdowns)


def _grid_sync_from_axis_tokens(axis_tokens):
    slider_cfg, slice_cfg, _ = _ui_slider_config(axis_tokens)
    species = _grid.get('species', [])
    species_idx = _grid.get('species_idx', {})
    sp_opts, defaults, cq_opts, cq_val, cd_opts, cd_val, ie_cq_opts, ie_cq_val = _species_dropdown_cfg(
        species, species_idx)
    return ([True] + slider_cfg + _interleave_slice_tab_cfgs(slice_cfg)
            + [sp_opts, defaults, cq_opts, cq_val, cd_opts, cd_val, ie_cq_opts, ie_cq_val])


@app.callback(
    [Output('load-status', 'children'),
     Output('grid-loaded', 'data')]
    + _slider_outputs
    + _slice_slider_outputs
    + _int_slice_slider_outputs
    + _ie_slice_slider_outputs
    + _rgb_slice_slider_outputs
    + _int_rgb_slice_slider_outputs
    + _int_spag_slice_slider_outputs
    + [Output('species-selector', 'options'),
       Output('species-selector', 'value'),
       Output('contour-quantity', 'options'),
       Output('contour-quantity', 'value'),
       Output('contour-cdens', 'options'),
       Output('contour-cdens', 'value'),
       Output('ie-quantity', 'options'),
       Output('ie-quantity', 'value')],
    Input('btn-load', 'n_clicks'),
    State('dir-input', 'value'),
    State('recursive-check', 'value'),
    prevent_initial_call=True,
)
def handle_load(n_clicks, directory, recursive):
    hidden, shown = _slider_wrap_styles()
    empty_slice = _empty_slice_slider_ui()

    try:
        grid = scan_directory(directory or '', recursive=bool(recursive))
    except Exception as exc:
        err = html.Span(f'\u2717  {exc}', style={'color': '#d62728', 'fontWeight': '600'})
        empty = _empty_param_slider_ui(hidden)
        return ([err, False] + empty + empty_slice * 6
                + [[], [], [], [], [], [], [], None])

    slider_cfg, slice_cfg, _ = _ui_slider_config(grid['axis_tokens'])
    sp_opts, defaults, cq_opts, cq_val, cd_opts, cd_val, ie_cq_opts, ie_cq_val = _species_dropdown_cfg(
        grid['species'], grid['species_idx'])

    cube = ' \u00D7 '.join(
        f'{len(grid["axis_tokens"][p["key"]])} {p["name"].split()[0]}'
        for p in PARAM_DEFS if len(grid['axis_tokens'][p['key']]) > 1
    ) or 'single model'

    note = _scan_status_note(grid.get('n_from_hdf5', 0), grid.get('n_skipped', 0))
    cfg_note = ''
    if _model_config_summary and not _model_config_summary.get('error'):
        cfg_note = f'   \u2014   {_model_config_summary["n_configs"]} JSON configs'
    status = html.Span([
        html.Span('Loaded', className='status-chip ok'),
        html.Code(grid['directory'], style={'marginLeft': '8px'}),
        html.Span(f'  {grid["n_files"]} models  —  grid: {cube}{note}{cfg_note}',
                  className='kosma-muted',
                  style={'marginLeft': '10px'}),
    ])

    return ([status, True] + slider_cfg + slice_cfg * 6
            + [sp_opts, defaults, cq_opts, cq_val, cd_opts, cd_val, ie_cq_opts, ie_cq_val])


app.clientside_callback(
    """
    function(tab, loaded) {
        if (!loaded || tab === 'load') {
            return {'display': 'none'};
        }
        return {'display': 'block'};
    }
    """,
    Output('controls-wrap', 'style'),
    Input('main-tabs', 'value'),
    Input('grid-loaded', 'data'),
)


@app.callback(
    Output('app-root', 'style'),
    Output('app-root', 'className'),
    Output('app-header', 'style'),
    Output('controls-sliders-wrap', 'style'),
    Output('controls-axis-wrap', 'style'),
    Output('model-info', 'style'),
    Output('main-tabs', 'colors'),
    Input('plot-theme', 'value'),
    Input('grid-loaded', 'data'),
    Input('simline-state', 'data'),
)
def apply_plot_theme(theme, loaded, _simline_state):
    t = _theme_colors(theme)
    name = _parse_plot_theme(theme)
    root = {
        'fontFamily': UI_FONT, 'maxWidth': APP_MAX_WIDTH,
        'margin': '0 auto', 'padding': '12px 24px 28px',
        'backgroundColor': t['page_bg'], 'color': t['font'],
        'minHeight': '100vh',
    }
    header = {
        'display': 'flex', 'alignItems': 'center',
        'justifyContent': 'space-between', 'gap': '16px',
        'borderBottom': f'1px solid {t["card_border"]}',
        'paddingBottom': '14px', 'marginBottom': '12px',
    }
    sliders = {
        'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-start',
        'padding': '12px 16px', 'marginTop': '4px',
        'color': t['font'],
    }
    if loaded and _grid and _grid.get('simline_only'):
        axis = {'display': 'none'}
    else:
        axis = {
            'display': 'flex', 'alignItems': 'flex-start',
            'padding': '10px 16px', 'marginTop': '8px',
            'color': t['font'],
        }
    info = {
        'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'alignItems': 'center',
        'padding': '7px 14px',
        'marginTop': '8px', 'marginBottom': '4px', 'fontSize': '13px', 'color': t['font'],
    }
    tab_colors = {
        'border': t['tab_border'],
        'primary': t['accent'],
        'background': t['tab_bg'],
    }
    return root, f'theme-{name}', header, sliders, axis, info, tab_colors


# Per-slider value labels + model info bar.
_label_outputs = [Output(f'slabel-{d}', 'children') for d in range(N_PARAMS)]
_slider_value_inputs = [Input(f'slider-{d}', 'value') for d in range(N_PARAMS)]


@app.callback(
    Output('model-setup-panel', 'children'),
    Input('grid-loaded', 'data'),
    _slider_value_inputs,
)
def update_model_setup_panel(loaded, *slider_values):
    if not loaded or not _grid:
        return mc.build_model_setup_panel(None)
    tokens = _tokens_from_values(list(slider_values))
    return mc.build_model_setup_panel(_model_config_summary, tokens)


@app.callback(
    _label_outputs + [Output('model-info', 'children')],
    _slider_value_inputs,
)
def update_labels(*values):
    if not _grid:
        return [''] * N_PARAMS + ['']

    labels = []
    info = []
    filepath = current_file(values) if _grid_has_hdf5() else None
    chem_path = chem_file(values) if _chem and not filepath else None
    phys = {}
    if filepath:
        phys = (_grid.get('phys_by_path') or {}).get(filepath) or {}
    elif chem_path:
        phys = ((_chem.get('phys_by_path') or {}).get(chem_path)
                or (_grid.get('phys_by_path') or {}).get(chem_path) or {})
    for d, p in enumerate(PARAM_DEFS):
        tokens = _grid['axis_tokens'][p['key']]
        try:
            tok = tokens[int(values[d])]
        except (IndexError, TypeError, ValueError):
            tok = tokens[0]
        varies = len(tokens) > 1
        if p['key'] == 'atten':
            disp = _atten_disp(tok)
        else:
            val = _physical_param_value(p['key'], tok, filepath=filepath or chem_path)
            disp = _format_phys_number(val)
        unit = f' {p["unit"]}' if p['unit'] else ''
        labels.append(f'{disp}{unit}')
        if phys and p['key'] not in phys and p['key'] in ('mass', 'atten'):
            continue
        info.append(html.Span(
            f'{p["name"]} = {disp}{unit}',
            style={'marginRight': '20px', 'color': p['color'],
                   'fontWeight': '600' if varies else '400',
                   'opacity': 1.0 if varies else 0.55}))
    if phys.get('av_max') is not None:
        info.append(html.Span(
            f'Max A_V = {_format_phys_number(phys["av_max"])} mag',
            style={'marginRight': '20px', 'color': '#555'}))
    if phys.get('radius') is not None:
        info.append(html.Span(
            f'Radius = {_format_phys_number(phys["radius"])} pc',
            style={'marginRight': '20px', 'color': '#555'}))

    if filepath:
        fname = os.path.basename(filepath)
    elif _grid.get('simline_only'):
        tok = _tokens_from_values(values)
        spath = _simline_sample_path(tok)
        if spath:
            fname = f'{os.path.basename(spath)}  (SIMLINE only — no HDF5)'
        else:
            fname = 'SIMLINE model point (no HDF5)'
    elif _grid.get('chem_only'):
        tok = _tokens_from_values(values)
        cpath = _chem_sample_path(tok)
        if cpath:
            fname = f'{os.path.basename(cpath)}  (chemistry only — no HDF5)'
        else:
            fname = 'Chemistry model point (no HDF5)'
    else:
        fname = '(no matching file)'
    info.append(html.Span(fname, style={'color': '#999', 'fontSize': '12px'}))
    return labels + [info]


def _slice_axis_ui(plane, idx):
    """Return fixed-axis label, value label, and value style for one slice slot."""
    if not _grid or not plane.get('active') or not plane.get('slice'):
        return '', '', {'display': 'none'}
    sdef = _param_def(plane['slice'])
    tokens = _grid['axis_tokens'][plane['slice']]
    varies = len(tokens) > 1
    try:
        tok = tokens[int(idx)]
    except (IndexError, TypeError, ValueError):
        tok = tokens[0]
    if sdef['key'] == 'atten':
        disp = _atten_disp(tok)
    else:
        disp = _format_phys_number(_physical_param_value(sdef['key'], tok))
    unit = f' {sdef["unit"]}' if sdef['unit'] else ''
    role = 'Slice' if varies else 'Fixed'
    fixed_lbl = f'{role}: {sdef["name"]}{unit}'
    value_style = {
        'textAlign': 'center', 'color': sdef['color'],
        'fontSize': '11px', 'marginTop': '2px', 'fontWeight': '600',
    }
    return fixed_lbl, f'{disp}{unit}', value_style


_slice_fixed_label_outputs = [Output(f'slice-fixed-label-{p["id"]}', 'children')
                              for p in SLICE_PLANES]
_slice_value_style_outputs = [Output(f'slice-label-{p["id"]}', 'style')
                              for p in SLICE_PLANES]
_int_slice_fixed_label_outputs = [Output(f'int-slice-fixed-label-{p["id"]}', 'children')
                                  for p in SLICE_PLANES]
_int_slice_value_style_outputs = [Output(f'int-slice-label-{p["id"]}', 'style')
                                    for p in SLICE_PLANES]
_ie_slice_fixed_label_outputs = [Output(f'ie-slice-fixed-label-{p["id"]}', 'children')
                                 for p in SLICE_PLANES]
_ie_plane_title_outputs = [Output(f'ie-plane-title-{p["id"]}', 'children')
                           for p in SLICE_PLANES]
_rgb_slice_fixed_label_outputs = [
    Output(f'rgb-slice-fixed-label-{p["id"]}', 'children') for p in SLICE_PLANES
]
_rgb_slice_value_style_outputs = [
    Output(f'rgb-slice-label-{p["id"]}', 'style') for p in SLICE_PLANES
]
_int_rgb_slice_fixed_label_outputs = [
    Output(f'int-rgb-slice-fixed-label-{p["id"]}', 'children') for p in SLICE_PLANES
]
_int_rgb_slice_value_style_outputs = [
    Output(f'int-rgb-slice-label-{p["id"]}', 'style') for p in SLICE_PLANES
]
_int_spag_slice_fixed_label_outputs = [
    Output(f'int-spag-slice-fixed-label-{p["id"]}', 'children') for p in SLICE_PLANES
]
_int_spag_slice_value_style_outputs = [
    Output(f'int-spag-slice-label-{p["id"]}', 'style') for p in SLICE_PLANES
]


def _update_slice_label_bundle(slice_indices):
    """Shared label/fixed-label/style bundle for one set of slice sliders."""
    n = len(SLICE_PLANES)
    if not _grid:
        empty = [''] * n
        hide = {'display': 'none'}
        return empty + empty + [hide] * n
    labels, fixed_labels, styles = [], [], []
    for plane, idx in zip(active_slice_planes(), slice_indices):
        fixed, val, style = _slice_axis_ui(plane, idx)
        fixed_labels.append(fixed)
        labels.append(val)
        styles.append(style if val else {'display': 'none'})
    return labels + fixed_labels + styles


@app.callback(
    _slice_label_outputs + _slice_fixed_label_outputs + _slice_value_style_outputs,
    _slice_slider_inputs
    + [Input('grid-loaded', 'data'), Input('simline-state', 'data')],
)
def update_slice_labels(*args_in):
    return _update_slice_label_bundle(list(args_in[:len(SLICE_PLANES)]))


@app.callback(
    _rgb_slice_label_outputs + _rgb_slice_fixed_label_outputs
    + _rgb_slice_value_style_outputs,
    _rgb_slice_slider_inputs
    + [Input('grid-loaded', 'data'), Input('simline-state', 'data')],
)
def update_rgb_slice_labels(*args_in):
    return _update_slice_label_bundle(list(args_in[:len(SLICE_PLANES)]))


_slice_panel_style_outputs = [Output(f'slice-panel-{p["id"]}', 'style') for p in SLICE_PLANES]
_contour_graph_style_outputs = [Output(f'plot-contour-{p["id"]}', 'style') for p in SLICE_PLANES]
_int_slice_panel_style_outputs = [Output(f'int-slice-panel-{p["id"]}', 'style') for p in SLICE_PLANES]
_int_contour_graph_style_outputs = [Output(f'plot-int-contour-{p["id"]}', 'style') for p in SLICE_PLANES]


def _slice_panel_row_style(active, full_width=False):
    if not active:
        return {'display': 'none'}
    return _SLICE_PANEL_ROW_FULL if full_width else _SLICE_PANEL_ROW


@app.callback(
    [Output('slice-panels-wrap', 'style'),
     *_slice_panel_style_outputs,
     *_contour_graph_style_outputs],
    Input('overlay-state', 'data'),
    Input('grid-loaded', 'data'),
    Input('simline-state', 'data'),
)
def update_slice_panels_layout(_overlay_state, _loaded, _simline_state):
    """Side-by-side equal-size panels without overlay; hide unused slice slots."""
    if _overlay:
        wrap = _SLICE_PANELS_COL_STYLE
        full = True
    else:
        wrap = _SLICE_PANELS_ROW_STYLE
        full = False
    panel_styles = [
        _slice_panel_row_style(p.get('active'), full_width=full)
        for p in active_slice_planes()
    ]
    graph_styles = [
        _SLICE_GRAPH_STYLE_FULL if full and p.get('active') else
        (_SLICE_GRAPH_STYLE_COMPACT if p.get('active') else {'display': 'none'})
        for p in active_slice_planes()
    ]
    return (wrap, *panel_styles, *graph_styles)


@app.callback(
    [Output('int-slice-panels-wrap', 'style'),
     *_int_slice_panel_style_outputs,
     *_int_contour_graph_style_outputs],
    Input('simline-overlay-state', 'data'),
    Input('grid-loaded', 'data'),
    Input('simline-state', 'data'),
)
def update_int_slice_panels_layout(_simline_overlay_state, _loaded, _simline_state):
    if _simline_overlay:
        wrap = _SLICE_PANELS_COL_STYLE
        full = True
    else:
        wrap = _SLICE_PANELS_ROW_STYLE
        full = False
    panel_styles = [
        _slice_panel_row_style(p.get('active'), full_width=full)
        for p in active_slice_planes()
    ]
    graph_styles = [
        _SLICE_GRAPH_STYLE_FULL if full and p.get('active') else
        (_SLICE_GRAPH_STYLE_COMPACT if p.get('active') else {'display': 'none'})
        for p in active_slice_planes()
    ]
    return (wrap, *panel_styles, *graph_styles)


@app.callback(
    [Output('rgb-panels-wrap', 'style'),
     *_rgb_panel_style_outputs,
     *_rgb_graph_style_outputs],
    Input('grid-loaded', 'data'),
    Input('simline-state', 'data'),
)
def update_rgb_panels_layout(_loaded, _simline_state):
    panel_styles = [
        _slice_panel_row_style(p.get('active'), full_width=False)
        for p in active_slice_planes()
    ]
    graph_styles = [
        _SLICE_GRAPH_STYLE_COMPACT if p.get('active') else {'display': 'none'}
        for p in active_slice_planes()
    ]
    return (_SLICE_PANELS_ROW_STYLE, *panel_styles, *graph_styles)


@app.callback(
    [Output('int-rgb-panels-wrap', 'style'),
     *_int_rgb_panel_style_outputs,
     *_int_rgb_graph_style_outputs],
    Input('grid-loaded', 'data'),
    Input('simline-state', 'data'),
)
def update_int_rgb_panels_layout(_loaded, _simline_state):
    panel_styles = [
        _slice_panel_row_style(p.get('active'), full_width=False)
        for p in active_slice_planes()
    ]
    graph_styles = [
        _SLICE_GRAPH_STYLE_COMPACT if p.get('active') else {'display': 'none'}
        for p in active_slice_planes()
    ]
    return (_SLICE_PANELS_ROW_STYLE, *panel_styles, *graph_styles)


@app.callback(
    [Output('int-spag-panels-wrap', 'style'),
     *_int_spag_panel_style_outputs,
     *_int_spag_graph_style_outputs],
    Input('grid-loaded', 'data'),
    Input('simline-state', 'data'),
)
def update_int_spag_panels_layout(_loaded, _simline_state):
    panel_styles = [
        _slice_panel_row_style(p.get('active'), full_width=False)
        for p in active_slice_planes()
    ]
    graph_styles = [
        _SLICE_GRAPH_STYLE_COMPACT if p.get('active') else {'display': 'none'}
        for p in active_slice_planes()
    ]
    return (_SLICE_PANELS_ROW_STYLE, *panel_styles, *graph_styles)


@app.callback(
    [Output(f'grid-rgb-sp{i}', 'options') for i in range(3)]
    + [Output(f'grid-rgb-sp{i}', 'value') for i in range(3)],
    Input('grid-loaded', 'data'),
    Input('simline-state', 'data'),
)
def update_grid_rgb_species(_loaded, _simline_state):
    if not _grid or not _grid_has_hdf5():
        return ([],) * 3 + (None,) * 3
    species = list(_grid.get('species') or [])
    opts = [{'label': s, 'value': s} for s in species]
    defaults = _default_rgb_grid_species(species)
    return (*[opts] * 3, *defaults)


@app.callback(
    [Output(f'int-rgb-sp{i}', 'options') for i in range(3)]
    + [Output(f'int-rgb-sp{i}', 'value') for i in range(3)],
    Input('simline-state', 'data'),
    Input('int-idef', 'value'),
)
def update_int_rgb_lines(_simline_state, idef):
    if not _simline:
        return ([],) * 3 + (None,) * 3
    opts = _simline_line_key_options(idef or SIMLINE_DEFAULT_IDEF)
    defaults = _default_rgb_line_keys(idef or SIMLINE_DEFAULT_IDEF)
    return (*[opts] * 3, *defaults)


@app.callback(
    Output('int-spaghetti-lines', 'options'),
    Input('simline-state', 'data'),
    Input('int-idef', 'value'),
)
def update_spaghetti_line_options(_simline_state, idef):
    if not _simline:
        return []
    return _simline_line_key_options(idef or SIMLINE_DEFAULT_IDEF)


@app.callback(
    Output('int-spaghetti-contours', 'value'),
    Input('int-spaghetti-lines', 'value'),
    State('int-spaghetti-contours', 'value'),
    prevent_initial_call=True,
)
def seed_spaghetti_contours_from_lines(selected, current_text):
    """Append newly selected line keys into the intensity text box."""
    selected = list(selected or [])
    if not selected:
        raise PreventUpdate
    current = current_text or ''
    available = gf.list_simline_line_keys(
        _simline, SIMLINE_DEFAULT_IDEF) if _simline else []
    # Re-parse with whatever keys we can; idef-specific keys come from dropdown.
    existing = {
        e['name'] for e in _parse_spaghetti_contour_spec(current, available_keys=None)
    }
    # Also treat bare "Name =" lines without a value as present.
    for line in current.splitlines():
        m = re.match(r'^\s*(.+?)\s*[=:]\s*', line)
        if m:
            existing.add(m.group(1).strip())
    additions = []
    for name in selected:
        if name not in existing:
            additions.append(f'{name} = ')
            existing.add(name)
    if not additions:
        raise PreventUpdate
    prefix = current.rstrip()
    sep = '\n' if prefix else ''
    return prefix + sep + '\n'.join(additions)


@app.callback(
    _contour_outputs,
    [Input('contour-quantity', 'value'),
     Input('contour-cdens', 'value'),
     Input('contour-zscale', 'value'),
     Input('overlay-state', 'data'),
     Input('shift-scan-direction', 'value'),
     Input('shift-match-rtol', 'value'),
     Input('interp-ny', 'value'),
     Input('interp-nx', 'value'),
     Input('interp-x-lim', 'value'),
     Input('interp-y-lim', 'value'),
     Input('interp-method', 'value'),
     Input('interp-clip', 'value'),
     Input('plot-theme', 'value'),
     Input('grid-colorscale', 'value')]
    + _slice_slider_inputs,
    prevent_initial_call=True,
)
def update_contour_plots(abund, cdens, zscale, _overlay_state, shift_dir, shift_rtol,
                         interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                         interp_method, interp_clip, plot_theme, grid_colorscale,
                         *slice_indices):
    theme = _parse_plot_theme(plot_theme)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    quantity = _combined_contour_quantities(abund, cdens)
    if not _grid or not quantity:
        p = placeholder_fig('Load a grid and pick a quantity', theme=theme)
        return (p,) * len(SLICE_PLANES)
    icfg = _interp_config(interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip)
    return make_contour_plots(
        quantity, zscale or 'log', list(slice_indices),
        shift_rtol=_parse_shift_rtol(shift_rtol),
        shift_scan_direction=_parse_shift_scan_direction(shift_dir),
        interp_config=icfg,
        colorscale=colorscale, theme=theme,
    )


@app.callback(
    Output('contour-quantity', 'value', allow_duplicate=True),
    Input('btn-contour-all-species', 'n_clicks'),
    State('contour-quantity', 'options'),
    State('contour-quantity', 'value'),
    prevent_initial_call=True,
)
def add_all_contour_species(_n, options, current):
    species_vals = [o['value'] for o in (options or [])
                    if str(o.get('value', '')).startswith('species:')]
    if not species_vals:
        raise PreventUpdate
    kept = [v for v in _as_str_list(current) if not str(v).startswith('species:')]
    return kept + species_vals


@app.callback(
    Output('contour-cdens', 'value', allow_duplicate=True),
    Input('btn-contour-all-cdens', 'n_clicks'),
    State('contour-cdens', 'options'),
    prevent_initial_call=True,
)
def add_all_contour_cdens(_n, options):
    vals = [o['value'] for o in (options or [])]
    if not vals:
        raise PreventUpdate
    return vals


@app.callback(
    Output('grid-contour-extra-rows', 'children'),
    Input('contour-quantity', 'value'),
    Input('contour-cdens', 'value'),
    Input('overlay-state', 'data'),
    Input('grid-loaded', 'data'),
    Input('simline-state', 'data'),
    State('contour-zscale', 'value'),
    State('shift-scan-direction', 'value'),
    State('shift-match-rtol', 'value'),
    State('interp-ny', 'value'),
    State('interp-nx', 'value'),
    State('interp-x-lim', 'value'),
    State('interp-y-lim', 'value'),
    State('interp-method', 'value'),
    State('interp-clip', 'value'),
    State('plot-theme', 'value'),
    State('grid-colorscale', 'value'),
    *[State(f'slice-slider-{p["id"]}', 'value') for p in SLICE_PLANES],
    State({'role': 'grid-extra-slider', 'qty': ALL, 'plane': ALL}, 'id'),
    State({'role': 'grid-extra-slider', 'qty': ALL, 'plane': ALL}, 'value'),
)
def build_grid_contour_extra_rows(abund, cdens, _overlay_state, _loaded, _simline_state,
                                  zscale, shift_dir, shift_rtol,
                                  interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                                  interp_method, interp_clip, plot_theme, grid_colorscale,
                                  *rest):
    n_planes = len(SLICE_PLANES)
    slice_indices = list(rest[:n_planes])
    saved = _saved_extra_slices(rest[n_planes], rest[n_planes + 1] if len(rest) > n_planes + 1 else [])
    extras = _combined_contour_quantities(abund, cdens)[1:]
    if not extras:
        return []
    full_width = bool(_overlay)
    plane_ids = _active_slice_plane_ids()
    seed = _slice_seed_by_plane(slice_indices)
    rows = []
    for qty in extras:
        idents = [
            {'role': 'grid-contour-extra', 'qty': qty, 'plane': pid}
            for pid in plane_ids
        ]
        sl_idxs = [_extra_panel_slice_idx(ident, seed, saved) for ident in idents]
        figs = _grid_contour_extra_figures(
            idents, zscale, shift_dir, shift_rtol,
            interp_ny, interp_nx, interp_x_lim, interp_y_lim,
            interp_method, interp_clip, plot_theme, grid_colorscale,
            sl_idxs,
        )
        rows.append(html.Div([
            _extra_slice_caption(_plain_quantity_caption(qty)),
            _extra_slice_graph_row(
                idents, full_width=full_width, figures=figs, slice_indices=sl_idxs,
            ),
        ]))
    return rows


@app.callback(
    Output({'role': 'grid-contour-extra', 'qty': MATCH, 'plane': MATCH}, 'figure'),
    Output({'role': 'grid-extra-label', 'qty': MATCH, 'plane': MATCH}, 'children'),
    Output({'role': 'grid-extra-label', 'qty': MATCH, 'plane': MATCH}, 'style'),
    Output({'role': 'grid-extra-fixed', 'qty': MATCH, 'plane': MATCH}, 'children'),
    Input({'role': 'grid-extra-slider', 'qty': MATCH, 'plane': MATCH}, 'value'),
    Input('contour-zscale', 'value'),
    Input('overlay-state', 'data'),
    Input('shift-scan-direction', 'value'),
    Input('shift-match-rtol', 'value'),
    Input('interp-ny', 'value'),
    Input('interp-nx', 'value'),
    Input('interp-x-lim', 'value'),
    Input('interp-y-lim', 'value'),
    Input('interp-method', 'value'),
    Input('interp-clip', 'value'),
    Input('plot-theme', 'value'),
    Input('grid-colorscale', 'value'),
    prevent_initial_call=True,
)
def update_one_grid_contour_extra(slice_idx, zscale, _overlay_state, shift_dir,
                                  shift_rtol, interp_ny, interp_nx, interp_x_lim,
                                  interp_y_lim, interp_method, interp_clip, plot_theme,
                                  grid_colorscale):
    ident = _match_output_id()
    figs = _grid_contour_extra_figures(
        [ident], zscale, shift_dir, shift_rtol,
        interp_ny, interp_nx, interp_x_lim, interp_y_lim,
        interp_method, interp_clip, plot_theme, grid_colorscale,
        [slice_idx],
    )
    plane = _plane_by_slot(ident['plane'])
    fixed, val, style = _slice_axis_ui(plane, slice_idx)
    return figs[0], val, style, fixed


@app.callback(
    Output('grid-ratio-selector', 'options'),
    Output('grid-ratio-selector', 'value'),
    Output('grid-ratio-hint', 'children'),
    Input('contour-quantity', 'value'),
    State('grid-ratio-selector', 'value'),
)
def update_grid_ratio_selector(quantities, current):
    return _grid_ratio_dropdown(quantities, current)


@app.callback(
    Output('grid-ratio-selector', 'value', allow_duplicate=True),
    Input('btn-grid-all-ratios', 'n_clicks'),
    State('grid-ratio-selector', 'options'),
    prevent_initial_call=True,
)
def add_all_grid_ratios(_n, options):
    vals = [o['value'] for o in (options or [])]
    if not vals:
        raise PreventUpdate
    return vals


@app.callback(
    Output('grid-ratio-rows', 'children'),
    Input('grid-ratio-selector', 'value'),
    Input('overlay-state', 'data'),
    Input('grid-loaded', 'data'),
    Input('simline-state', 'data'),
    State('contour-zscale', 'value'),
    State('shift-scan-direction', 'value'),
    State('shift-match-rtol', 'value'),
    State('interp-ny', 'value'),
    State('interp-nx', 'value'),
    State('interp-x-lim', 'value'),
    State('interp-y-lim', 'value'),
    State('interp-method', 'value'),
    State('interp-clip', 'value'),
    State('plot-theme', 'value'),
    State('grid-colorscale', 'value'),
    *[State(f'slice-slider-{p["id"]}', 'value') for p in SLICE_PLANES],
    State({'role': 'grid-ratio-slider', 'pair': ALL, 'plane': ALL}, 'id'),
    State({'role': 'grid-ratio-slider', 'pair': ALL, 'plane': ALL}, 'value'),
)
def build_grid_ratio_rows(pairs, _overlay_state, _loaded, _simline_state,
                          zscale, shift_dir, shift_rtol,
                          interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip, plot_theme, grid_colorscale,
                          *rest):
    selected = _as_str_list(pairs)
    if not selected:
        return []
    n_planes = len(SLICE_PLANES)
    slice_indices = list(rest[:n_planes])
    saved = _saved_extra_slices(rest[n_planes], rest[n_planes + 1] if len(rest) > n_planes + 1 else [])
    full_width = bool(_overlay)
    plane_ids = _active_slice_plane_ids()
    seed = _slice_seed_by_plane(slice_indices)
    rows = []
    for pair in selected:
        idents = [
            {'role': 'grid-ratio', 'pair': pair, 'plane': pid}
            for pid in plane_ids
        ]
        sl_idxs = [_extra_panel_slice_idx(ident, seed, saved) for ident in idents]
        figs = _grid_ratio_figures(
            idents, zscale, shift_dir, shift_rtol,
            interp_ny, interp_nx, interp_x_lim, interp_y_lim,
            interp_method, interp_clip, plot_theme, grid_colorscale,
            sl_idxs,
        )
        rows.append(html.Div([
            _extra_slice_caption(_grid_ratio_caption(pair)),
            _extra_slice_graph_row(
                idents, full_width=full_width, figures=figs, slice_indices=sl_idxs,
                slider_role='grid-ratio-slider',
                label_role='grid-ratio-label',
                fixed_role='grid-ratio-fixed',
            ),
        ]))
    return rows


@app.callback(
    Output({'role': 'grid-ratio', 'pair': MATCH, 'plane': MATCH}, 'figure'),
    Output({'role': 'grid-ratio-label', 'pair': MATCH, 'plane': MATCH}, 'children'),
    Output({'role': 'grid-ratio-label', 'pair': MATCH, 'plane': MATCH}, 'style'),
    Output({'role': 'grid-ratio-fixed', 'pair': MATCH, 'plane': MATCH}, 'children'),
    Input({'role': 'grid-ratio-slider', 'pair': MATCH, 'plane': MATCH}, 'value'),
    Input('contour-zscale', 'value'),
    Input('overlay-state', 'data'),
    Input('shift-scan-direction', 'value'),
    Input('shift-match-rtol', 'value'),
    Input('interp-ny', 'value'),
    Input('interp-nx', 'value'),
    Input('interp-x-lim', 'value'),
    Input('interp-y-lim', 'value'),
    Input('interp-method', 'value'),
    Input('interp-clip', 'value'),
    Input('plot-theme', 'value'),
    Input('grid-colorscale', 'value'),
    prevent_initial_call=True,
)
def update_one_grid_ratio(slice_idx, zscale, _overlay_state, shift_dir,
                          shift_rtol, interp_ny, interp_nx, interp_x_lim,
                          interp_y_lim, interp_method, interp_clip, plot_theme,
                          grid_colorscale):
    ident = _match_output_id()
    figs = _grid_ratio_figures(
        [ident], zscale, shift_dir, shift_rtol,
        interp_ny, interp_nx, interp_x_lim, interp_y_lim,
        interp_method, interp_clip, plot_theme, grid_colorscale,
        [slice_idx],
    )
    plane = _plane_by_slot(ident['plane'])
    fixed, val, style = _slice_axis_ui(plane, slice_idx)
    return figs[0], val, style, fixed


@app.callback(
    _rgb_outputs,
    [Input('grid-rgb-sp0', 'value'),
     Input('grid-rgb-sp1', 'value'),
     Input('grid-rgb-sp2', 'value'),
     Input('grid-rgb-cf0', 'value'),
     Input('grid-rgb-cf1', 'value'),
     Input('grid-rgb-cf2', 'value'),
     Input('grid-rgb-transitions', 'value'),
     Input('interp-ny', 'value'),
     Input('interp-nx', 'value'),
     Input('interp-x-lim', 'value'),
     Input('interp-y-lim', 'value'),
     Input('interp-method', 'value'),
     Input('interp-clip', 'value'),
     Input('plot-theme', 'value'),
     Input('grid-loaded', 'data')]
    + _rgb_slice_slider_inputs,
    prevent_initial_call=True,
)
def update_grid_rgb_plots(sp0, sp1, sp2, cf0, cf1, cf2, transitions,
                          interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip, plot_theme, _loaded,
                          *slice_indices):
    theme = _parse_plot_theme(plot_theme)
    if not _grid or not _grid_has_hdf5():
        p = placeholder_fig('Load an HDF5 grid for abundance RGB maps', theme=theme)
        return (p,) * len(SLICE_PLANES)
    icfg = _interp_config(interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip)
    return make_grid_rgb_plots(
        [sp0, sp1, sp2], transitions or [], list(slice_indices),
        interp_config=icfg, theme=theme,
        conv_factors=_parse_rgb_conv_factors(cf0, cf1, cf2),
    )


@app.callback(
    Output('overlay-status', 'children'),
    Output('overlay-state', 'data'),
    Input('btn-load-overlay', 'n_clicks'),
    Input('btn-clear-overlay', 'n_clicks'),
    State('overlay-dir-input', 'value'),
    State('overlay-recursive-check', 'value'),
    State('overlay-state', 'data'),
    prevent_initial_call=True,
)
def handle_overlay(n_load, n_clear, directory, recursive, state):
    trigger = dash.callback_context.triggered[0]['prop_id'] if dash.callback_context.triggered else ''
    state = (state or 0)

    if trigger.startswith('btn-clear-overlay'):
        clear_overlay()
        return (html.Span('Overlay cleared.', style={'color': '#888'}), state + 1)

    if not _grid:
        return (html.Span('\u2717  Load a main grid first.',
                          style={'color': '#d62728', 'fontWeight': '600'}), state)
    try:
        ov = scan_overlay(directory or '', recursive=bool(recursive))
    except Exception as exc:
        return (html.Span(f'\u2717  {exc}',
                          style={'color': '#d62728', 'fontWeight': '600'}), state + 1)

    note = _scan_status_note(ov.get('n_from_hdf5', 0), ov.get('n_skipped', 0))
    status = html.Span([
        html.Span('\u2713  Overlay ', style={'color': '#8c564b', 'fontWeight': '700'}),
        html.Code(ov['directory']),
        html.Span(f'   {ov["n_files"]} models (dashed){note}',
                  style={'color': '#555', 'marginLeft': '10px'}),
    ])
    return (status, state + 1)


@app.callback(
    Output('simline-overlay-status', 'children'),
    Output('simline-overlay-state', 'data'),
    Input('btn-load-simline-overlay', 'n_clicks'),
    Input('btn-clear-simline-overlay', 'n_clicks'),
    State('simline-overlay-dir-input', 'value'),
    State('simline-overlay-recursive-check', 'value'),
    State('simline-overlay-state', 'data'),
    prevent_initial_call=True,
)
def handle_simline_overlay(n_load, n_clear, directory, recursive, state):
    trigger = dash.callback_context.triggered[0]['prop_id'] if dash.callback_context.triggered else ''
    state = (state or 0)

    if trigger.startswith('btn-clear-simline-overlay'):
        clear_simline_overlay()
        return (html.Span('Overlay SIMLINE cleared.', style={'color': '#888'}), state + 1)

    if not _simline:
        return (html.Span('\u2717  Load a main SIMLINE directory first.',
                          style={'color': '#d62728', 'fontWeight': '600'}), state)

    try:
        sl = scan_simline_overlay(directory or '', recursive=bool(recursive))
    except Exception as exc:
        return (html.Span(f'\u2717  {exc}',
                          style={'color': '#d62728', 'fontWeight': '600'}), state + 1)

    note = f'  ({sl["n_skipped"]} skipped)' if sl['n_skipped'] else ''
    status = html.Span([
        html.Span('\u2713  Overlay SIMLINE ', style={'color': '#8c564b', 'fontWeight': '700'}),
        html.Code(sl['directory']),
        html.Span(f'   {sl["n_files"]} files{note}',
                  style={'color': '#555', 'marginLeft': '10px'}),
    ])
    return (status, state + 1)


@app.callback(
    [Output('chem-overlay-status', 'children'),
     Output('chem-overlay-state', 'data')],
    Input('btn-load-chem-overlay', 'n_clicks'),
    Input('btn-clear-chem-overlay', 'n_clicks'),
    State('chem-overlay-dir-input', 'value'),
    State('chem-overlay-recursive-check', 'value'),
    State('chem-overlay-state', 'data'),
    prevent_initial_call=True,
)
def handle_chem_overlay(n_load, n_clear, directory, recursive, state):
    trigger = dash.callback_context.triggered[0]['prop_id'] if dash.callback_context.triggered else ''
    state = (state or 0)

    if trigger.startswith('btn-clear-chem-overlay'):
        clear_chem_overlay()
        return (html.Span('Overlay chemistry cleared.', style={'color': '#888'}),
                state + 1)

    if not _chem:
        return (html.Span('\u2717  Load a main chemistry grid first.',
                          style={'color': '#d62728', 'fontWeight': '600'}),
                state)

    try:
        ch = scan_chem_overlay(directory or '', recursive=bool(recursive))
    except Exception as exc:
        return (html.Span(f'\u2717  {exc}',
                          style={'color': '#d62728', 'fontWeight': '600'}),
                state + 1)

    note = _scan_status_note(ch.get('n_from_hdf5', 0), ch.get('n_skipped', 0))
    status = html.Span([
        html.Span('\u2713  Overlay chemistry ', style={'color': '#8c564b', 'fontWeight': '700'}),
        html.Code(ch['directory']),
        html.Span(f'   {ch["n_files"]} models (dashed){note}',
                  style={'color': '#555', 'marginLeft': '10px'}),
    ])
    return (status, state + 1)


@app.callback(
    [Output('chem-status', 'children'),
     Output('chem-state', 'data'),
     Output('react-species', 'options'),
     Output('react-species', 'value')]
    + _grid_sync_outputs,
    Input('btn-load-chem', 'n_clicks'),
    Input('btn-clear-chem', 'n_clicks'),
    State('chem-dir-input', 'value'),
    State('chem-recursive-check', 'value'),
    State('chem-state', 'data'),
    State('react-species', 'value'),
    prevent_initial_call=True,
)
def handle_chem(n_load, n_clear, directory, recursive, state, cur_species):
    trigger = dash.callback_context.triggered[0]['prop_id'] if dash.callback_context.triggered else ''
    state = (state or 0)
    no_grid_sync = [dash.no_update] * _N_GRID_SYNC

    if trigger.startswith('btn-clear-chem'):
        was_chem_only = bool(_grid and _grid.get('chem_only'))
        clear_chem()
        clear_chem_only_grid()
        grid_sync = _empty_grid_sync(loaded=False) if was_chem_only else no_grid_sync
        return (html.Span('Chemistry grid cleared.', style={'color': '#888'}),
                state + 1, [], None) + tuple(grid_sync)

    try:
        ch = scan_chem(directory or '', recursive=bool(recursive))
        bootstrapped = bootstrap_grid_from_chem()
    except Exception as exc:
        return (html.Span(f'\u2717  {exc}',
                          style={'color': '#d62728', 'fontWeight': '600'}),
                state + 1, [], None) + tuple(no_grid_sync)

    species = ch['species']
    # Order: common diagnostics first, then the rest alphabetically.
    preferred = [s for s in ('CO', 'C+', 'C', 'HCO+', 'H2O', 'OH', 'CH', 'CN', 'HCN', 'CS')
                 if s in species]
    rest = sorted(s for s in species if s not in preferred)
    opts = [{'label': s, 'value': s} for s in preferred + rest]
    value = cur_species if cur_species in species else \
        (CHEM_DEFAULT_SPECIES if CHEM_DEFAULT_SPECIES in species else (species[0] if species else None))

    note = _scan_status_note(ch.get('n_from_hdf5', 0), ch.get('n_skipped', 0))
    mode_note = ''
    if bootstrapped:
        cube = ' \u00D7 '.join(
            f'{len(_grid["axis_tokens"][p["key"]])} {p["name"].split()[0]}'
            for p in PARAM_DEFS if len(_grid['axis_tokens'][p['key']]) > 1
        ) or 'single model'
        mode_note = f'   \u2014   grid from filenames: {cube}  (chemistry-only mode)'
    elif _grid_has_hdf5():
        mode_note = '   \u2014   using loaded HDF5 grid axes'
    elif _grid and _grid.get('simline_only'):
        mode_note = '   \u2014   using SIMLINE grid axes'
    elif _chem and not bootstrapped and not _grid_has_hdf5():
        mode_note = ('   \u2014   chemistry loaded (clear SIMLINE or load HDF5 '
                     'for chemistry-only navigation)')
    grid_sync = (_grid_sync_from_axis_tokens(_grid['axis_tokens']) if bootstrapped
                 else no_grid_sync)
    status = html.Span([
        html.Span('\u2713  Chemistry ', style={'color': '#2ca02c', 'fontWeight': '700'}),
        html.Code(ch['directory']),
        html.Span(f'   {ch["n_files"]} models, {len(species)} species{note}{mode_note}',
                  style={'color': '#555', 'marginLeft': '10px'}),
    ])
    return (status, state + 1, opts, value) + tuple(grid_sync)


@app.callback(
    [Output('simline-status', 'children'),
     Output('simline-state', 'data'),
     Output('int-species', 'options'),
     Output('int-species', 'value'),
     Output('ie-int-species', 'options'),
     Output('ie-int-species', 'value'),
     Output('sp-species', 'options'),
     Output('sp-species', 'value')]
    + _grid_sync_outputs,
    Input('btn-load-simline', 'n_clicks'),
    Input('btn-clear-simline', 'n_clicks'),
    State('simline-dir-input', 'value'),
    State('simline-recursive-check', 'value'),
    State('simline-state', 'data'),
    State('int-species', 'value'),
    State('ie-int-species', 'value'),
    State('sp-species', 'value'),
    prevent_initial_call=True,
)
def handle_simline(n_load, n_clear, directory, recursive, state,
                   cur_species, cur_ie_species, cur_sp_species):
    trigger = dash.callback_context.triggered[0]['prop_id'] if dash.callback_context.triggered else ''
    state = (state or 0)
    no_grid_sync = [dash.no_update] * _N_GRID_SYNC

    if trigger.startswith('btn-clear-simline'):
        was_simline_only = bool(_grid and _grid.get('simline_only'))
        clear_simline()
        clear_simline_only_grid()
        grid_sync = _empty_grid_sync(loaded=False) if was_simline_only else no_grid_sync
        return ([html.Span('SIMLINE directory cleared.', style={'color': '#888'}),
                 state + 1, [], [], [], None, [], None]
                + grid_sync)

    try:
        sl = scan_simline(directory or '', recursive=bool(recursive))
    except Exception as exc:
        return ([html.Span(f'\u2717  {exc}',
                           style={'color': '#d62728', 'fontWeight': '600'}),
                state + 1, [], [], [], None, [], None]
                + no_grid_sync)

    bootstrapped = bootstrap_grid_from_simline()

    preferred = [s for s in ('CO', '13CO', 'C18O', 'HCO+', 'N2H+', 'CS', 'HCN', 'C+', 'CI')
                 if s in sl['species']]
    rest = sorted(s for s in sl['species'] if s not in preferred)
    opts = [{'label': s, 'value': s} for s in preferred + rest]
    default_sp = (SIMLINE_DEFAULT_SPECIES if SIMLINE_DEFAULT_SPECIES in sl['species']
                  else (sl['species'][0] if sl['species'] else None))
    kept_species = [s for s in _as_str_list(cur_species) if s in sl['species']]
    value = kept_species or ([default_sp] if default_sp else [])
    ie_default = value[0] if value else None
    ie_value = cur_ie_species if cur_ie_species in sl['species'] else ie_default
    sp_value = cur_sp_species if cur_sp_species in sl['species'] else ie_default

    note = f'  ({sl["n_skipped"]} skipped)' if sl['n_skipped'] else ''
    pv_note = ''
    if sl.get('n_pv_files'):
        pv_note = f', {sl["n_pv_files"]} PV FITS'
    mode_note = ''
    if bootstrapped:
        cube = ' \u00D7 '.join(
            f'{len(_grid["axis_tokens"][p["key"]])} {p["name"].split()[0]}'
            for p in PARAM_DEFS if len(_grid['axis_tokens'][p['key']]) > 1
        ) or 'single model'
        mode_note = f'   \u2014   grid from filenames: {cube}  (SIMLINE-only mode)'
    elif _grid_has_hdf5():
        mode_note = '   \u2014   using loaded HDF5 grid axes'
    grid_sync = (_grid_sync_from_axis_tokens(_grid['axis_tokens']) if bootstrapped
                 else no_grid_sync)
    status = html.Span([
        html.Span('\u2713  SIMLINE ', style={'color': '#9467bd', 'fontWeight': '700'}),
        html.Code(sl['directory']),
        html.Span(f'   {sl["n_files"]} files, {len(sl["species"])} species{pv_note}{note}{mode_note}',
                  style={'color': '#555', 'marginLeft': '10px'}),
    ])
    return ([status, state + 1, opts, value, opts, ie_value, opts, sp_value]
            + grid_sync)


@app.callback(
    Output('int-transition', 'options'),
    Output('int-transition', 'value'),
    Output('ie-int-transition', 'options'),
    Output('ie-int-transition', 'value'),
    Input('int-species', 'value'),
    Input('ie-int-species', 'value'),
    Input('int-idef', 'value'),
    Input('ie-int-idef', 'value'),
    Input('simline-state', 'data'),
    State('int-transition', 'value'),
    State('ie-int-transition', 'value'),
)
def update_int_transition(species, ie_species, idef, ie_idef, _state, cur_trans, cur_ie_trans):
    if not _simline:
        return [], [], [], None
    primary = _first_str(species)
    opts = _simline_transition_options(primary, idef or SIMLINE_DEFAULT_IDEF) if primary else []
    dd_opts = [{'label': o['label'], 'value': o['value']} for o in opts]
    valid = {str(o['value']) for o in opts}
    kept = [t for t in _as_str_list(cur_trans) if t in valid]
    value = kept or ([_default_simline_transition(primary, idef)] if opts else [])

    ie_opts = _simline_transition_options(
        ie_species, ie_idef or SIMLINE_DEFAULT_IDEF) if ie_species else []
    ie_dd = [{'label': o['label'], 'value': o['value']} for o in ie_opts]
    ie_valid = {o['value'] for o in ie_opts}
    ie_value = cur_ie_trans if cur_ie_trans in ie_valid else _default_simline_transition(
        ie_species, ie_idef)
    return dd_opts, value, ie_dd, ie_value


@app.callback(
    Output('sp-transition', 'options'),
    Output('sp-transition', 'value'),
    _slider_value_inputs
    + [Input('sp-species', 'value'),
       Input('sp-quantity', 'value'),
       Input('simline-state', 'data')],
    State('sp-transition', 'value'),
)
def update_sp_transition(*args_in):
    values = list(args_in[:N_PARAMS])
    species, quantity, _state = args_in[N_PARAMS:N_PARAMS + 3]
    cur_trans = args_in[-1]
    if not _simline or not species:
        return [], []
    tokens = _tokens_from_values(values)
    opts = _sp_transition_options(tokens, species, quantity=quantity or SIMLINE_DEFAULT_PV_QUANTITY)
    dd_opts = [{'label': o['label'], 'value': o['value']} for o in opts]
    valid = {o['value'] for o in opts}
    if isinstance(cur_trans, list):
        kept = [t for t in cur_trans if t in valid]
    elif cur_trans in valid:
        kept = [cur_trans]
    else:
        kept = []
    if not kept and opts:
        kept = [opts[0]['value']]
    return dd_opts, kept


@app.callback(
    Output('plot-sp-spectrum', 'figure'),
    Output('sp-fit-summary', 'children'),
    _slider_value_inputs
    + [Input('sp-species', 'value'),
       Input('sp-transition', 'value'),
       Input('sp-positions', 'value'),
       Input('sp-quantity', 'value'),
       Input('obs-fits-path', 'value'),
       Input('obs-hdu', 'value'),
       Input('obs-row-mode', 'value'),
       Input('obs-row-index', 'value'),
       Input('obs-v-low', 'value'),
       Input('obs-v-high', 'value'),
       Input('obs-n-gauss', 'value'),
       Input('obs-overlay', 'value'),
       Input('obs-fit', 'value'),
       Input('simline-state', 'data'),
       Input('plot-theme', 'value')],
    prevent_initial_call=True,
)
def update_sp_spectrum(*args_in):
    values = list(args_in[:N_PARAMS])
    (species, transition, positions_text, pv_quantity, obs_path, obs_hdu, obs_row_mode,
     obs_row_index, obs_v_low, obs_v_high, obs_n_gauss,
     obs_overlay, obs_fit, _state, plot_theme) = args_in[N_PARAMS:]
    fig, summary, _err = fig_spectra_plot(
        values, species, transition, positions_text,
        obs_path=obs_path, obs_hdu=obs_hdu, obs_row_mode=obs_row_mode or 'mean',
        obs_row_index=obs_row_index, obs_v_low=obs_v_low, obs_v_high=obs_v_high,
        obs_n_gauss=obs_n_gauss,
        obs_overlay='on' in (obs_overlay or []),
        obs_fit='on' in (obs_fit or []),
        pv_quantity=pv_quantity or SIMLINE_DEFAULT_PV_QUANTITY,
        theme=_parse_plot_theme(plot_theme),
    )
    return fig, summary if summary is not None else html.Div()


@app.callback(
    Output('plot-sp-pv', 'figure'),
    _slider_value_inputs
    + [Input('sp-species', 'value'),
       Input('sp-transition', 'value'),
       Input('sp-pos-min', 'value'),
       Input('sp-pos-max', 'value'),
       Input('sp-zscale', 'value'),
       Input('sp-colorscale', 'value'),
       Input('sp-quantity', 'value'),
       Input('simline-state', 'data'),
       Input('plot-theme', 'value')],
    prevent_initial_call=True,
)
def update_sp_pv(*args_in):
    values = list(args_in[:N_PARAMS])
    (species, transition, pos_min, pos_max, zscale,
     colorscale, pv_quantity, _state, plot_theme) = args_in[N_PARAMS:]
    if isinstance(transition, list):
        transition = transition[0] if transition else None
    return fig_simline_pv(
        values, species, transition, pos_min, pos_max,
        zscale or 'linear', _parse_grid_colorscale(colorscale),
        pv_quantity=pv_quantity or SIMLINE_DEFAULT_PV_QUANTITY,
        theme=_parse_plot_theme(plot_theme),
    )


@app.callback(
    _int_slice_label_outputs + _int_slice_fixed_label_outputs + _int_slice_value_style_outputs,
    _int_slice_slider_inputs
    + [Input('grid-loaded', 'data'), Input('simline-state', 'data')],
)
def update_int_slice_labels(*args_in):
    return _update_slice_label_bundle(list(args_in[:len(SLICE_PLANES)]))


@app.callback(
    _int_rgb_slice_label_outputs + _int_rgb_slice_fixed_label_outputs
    + _int_rgb_slice_value_style_outputs,
    _int_rgb_slice_slider_inputs
    + [Input('grid-loaded', 'data'), Input('simline-state', 'data')],
)
def update_int_rgb_slice_labels(*args_in):
    return _update_slice_label_bundle(list(args_in[:len(SLICE_PLANES)]))


@app.callback(
    _int_spag_slice_label_outputs + _int_spag_slice_fixed_label_outputs
    + _int_spag_slice_value_style_outputs,
    _int_spag_slice_slider_inputs
    + [Input('grid-loaded', 'data'), Input('simline-state', 'data')],
)
def update_int_spag_slice_labels(*args_in):
    return _update_slice_label_bundle(list(args_in[:len(SLICE_PLANES)]))


def _ie_slice_token(plane, idx):
    tokens = _grid['axis_tokens'][plane['slice']]
    try:
        return tokens[int(idx)]
    except (IndexError, TypeError, ValueError):
        return tokens[0]


def _ie_quantity_unit(quantity):
    if quantity in ('tgas', 'tgas_col'):
        return 'K'
    if quantity == 'tdust':
        return 'K'
    if quantity == 'nh':
        return 'cm^-3'
    if quantity == 'xe':
        return 'n(e-)/nH'
    if quantity.startswith('species:'):
        return 'rel. abund.'
    if quantity.startswith('cdens:'):
        return 'cm^-2'
    return ''


@app.callback(
    _ie_slice_label_outputs + _ie_slice_fixed_label_outputs + _ie_plane_title_outputs,
    _ie_slice_slider_inputs
    + [Input('grid-loaded', 'data'), Input('simline-state', 'data')],
)
def update_ie_slice_labels(*args_in):
    n = len(SLICE_PLANES)
    slice_indices = list(args_in[:n])
    if not _grid:
        empty = [''] * n
        return empty + empty + empty
    labels, fixed_labels, titles = [], [], []
    for plane, idx in zip(active_slice_planes(), slice_indices):
        fixed, val, _ = _slice_axis_ui(plane, idx)
        fixed_labels.append(fixed)
        labels.append(val)
        titles.append(_plane_title_plain(plane['x'], plane['y']) if plane.get('active') else '')
    return labels + fixed_labels + titles


@app.callback(
    _ie_abund_outputs + _ie_int_outputs + _ie_abund_wrap_outputs + _ie_int_wrap_outputs,
    [Input('ie-quantity', 'value'),
     Input('ie-int-species', 'value'),
     Input('ie-int-transition', 'value'),
     Input('ie-int-idef', 'value'),
     Input('ie-flux-scale', 'value'),
     Input('grid-colorscale', 'value'),
     Input('plot-theme', 'value'),
     Input('ie-error-metric', 'value'),
     Input('ie-plot-contours', 'value'),
     Input('ie-decimation', 'value'),
     Input('ie-rel-threshold', 'value'),
     Input('grid-loaded', 'data'),
     Input('simline-state', 'data'),
     Input('ie-interp-ny', 'value'),
     Input('ie-interp-nx', 'value'),
     Input('ie-interp-x-lim', 'value'),
     Input('ie-interp-y-lim', 'value'),
     Input('ie-interp-method', 'value'),
     Input('ie-interp-clip', 'value')]
    + _slider_value_inputs
    + _ie_slice_slider_inputs,
    prevent_initial_call=True,
)
def update_ie_panels(*args_in):
    n_sl = N_PARAMS
    n_ie = len(_ie_slice_slider_inputs)
    n_fixed = len(args_in) - n_sl - n_ie
    slider_values = list(args_in[n_fixed:n_fixed + n_sl])
    slice_indices = list(args_in[n_fixed + n_sl:])
    (quantity, ie_species, ie_transition, ie_idef, flux_scale, grid_colorscale,
     plot_theme, error_metric, plot_contours, decimation, rel_threshold,
     _grid_loaded, _simline_state,
     interp_ny, interp_nx, interp_x_lim, interp_y_lim,
     interp_method, interp_clip) = args_in[:n_fixed]
    n = len(SLICE_PLANES)
    theme = _parse_plot_theme(plot_theme)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    ie_cfg = _ie_analysis_config(
        interp_ny, interp_nx, interp_x_lim, interp_y_lim, interp_method, interp_clip,
        decimation, error_metric, rel_threshold,
    )
    zscale = flux_scale or 'log'
    err_metric = _parse_error_metric(error_metric)
    show_abund = bool(_grid and quantity and _grid_has_hdf5())
    show_int = bool(_grid and _simline and ie_species and ie_transition is not None)

    abund_figs, int_figs = [], []
    abund_wrap, int_wrap = [], []
    wrap_show = {'display': 'block', 'marginBottom': '8px'}
    wrap_hide = {'display': 'none'}

    for plane, sidx in zip(active_slice_planes(), slice_indices):
        xdef = ydef = None
        if show_abund:
            slice_token = _ie_slice_token(plane, sidx)
            x_phys, y_phys, Z, xdef, ydef = _native_abundance_grid(
                plane, slice_token, quantity)
            if np.any(np.isfinite(Z)):
                result = gi.analyze_slice_interpolation(
                    x_phys, y_phys, Z,
                    x_logscale=xdef['logscale'],
                    y_logscale=ydef['logscale'],
                    **ie_cfg,
                )
                sk = plane['slice']
                sdef = _param_def(sk)
                slice_disp = _param_token_disp(sdef, slice_token)
                unit_l = _ie_quantity_unit(quantity)
                title = (f'Abundance / diagnostic &mdash; {_ie_plane_label(plane["id"])}'
                         f'<br><sup>{_quantity_label(quantity)}'
                         f'  (fixed {sdef["name"]} = {slice_disp})</sup>')
                abund_figs.append(fig_interpolation_comparison(
                    result, xdef, ydef, title=title, zscale=zscale,
                    color_map=colorscale,
                    error_metric=err_metric, plot_contours=plot_contours,
                    unit_label=unit_l or _quantity_label(quantity),
                    theme=theme,
                ))
                abund_wrap.append(wrap_show)
            else:
                abund_figs.append(placeholder_fig('No grid points for this slice', theme=theme))
                abund_wrap.append(wrap_show)
        else:
            abund_figs.append(placeholder_fig('Load a main grid and pick a quantity', theme=theme))
            abund_wrap.append(wrap_hide)

        if show_int:
            slice_token = _ie_slice_token(plane, sidx)
            try:
                tidx = int(ie_transition)
            except (TypeError, ValueError):
                tidx = 0
            x_phys, y_phys, Z, xdef, ydef = _native_intensity_grid(
                plane, slice_token, ie_species, ie_idef or SIMLINE_DEFAULT_IDEF, tidx,
                slider_values=slider_values)
            if np.any(np.isfinite(Z)):
                result = gi.analyze_slice_interpolation(
                    x_phys, y_phys, Z,
                    x_logscale=xdef['logscale'],
                    y_logscale=ydef['logscale'],
                    **ie_cfg,
                )
                sk = plane['slice']
                sdef = _param_def(sk)
                slice_disp = _param_token_disp(sdef, slice_token)
                trans_rows = _simline_transition_options(ie_species, ie_idef)
                tlabel = trans_rows[tidx]['label'] if 0 <= tidx < len(trans_rows) else str(tidx)
                unit_l = _intensity_unit_label(ie_idef or SIMLINE_DEFAULT_IDEF)
                sp_html = format_species_html(ie_species)
                title = (f'Intensity &mdash; {_ie_plane_label(plane["id"])}'
                         f'<br><sup>{sp_html} {tlabel}'
                         f'  (fixed {sdef["name"]} = {slice_disp})</sup>')
                int_figs.append(fig_interpolation_comparison(
                    result, xdef, ydef, title=title, zscale=zscale,
                    color_map=colorscale,
                    error_metric=err_metric, plot_contours=plot_contours,
                    unit_label=unit_l,
                    theme=theme,
                ))
                int_wrap.append(wrap_show)
            else:
                int_figs.append(placeholder_fig('No SIMLINE data for this slice', theme=theme))
                int_wrap.append(wrap_show)
        else:
            int_figs.append(placeholder_fig('Load SIMLINE and pick species / transition',
                                            theme=theme))
            int_wrap.append(wrap_hide)

    return abund_figs + int_figs + abund_wrap + int_wrap


@app.callback(
    _int_contour_outputs,
    [Input('int-species', 'value'),
     Input('int-idef', 'value'),
     Input('int-transition', 'value'),
     Input('int-zscale', 'value'),
     Input('simline-state', 'data'),
     Input('simline-overlay-state', 'data'),
     Input('int-shift-scan-direction', 'value'),
     Input('int-shift-match-rtol', 'value'),
     Input('int-interp-ny', 'value'),
     Input('int-interp-nx', 'value'),
     Input('int-interp-x-lim', 'value'),
     Input('int-interp-y-lim', 'value'),
     Input('int-interp-method', 'value'),
     Input('int-interp-clip', 'value'),
     Input('int-obs-boundary', 'value'),
     Input('int-extra-contours', 'value'),
     Input('int-extra-contour-color', 'value'),
     Input('plot-theme', 'value'),
     Input('grid-colorscale', 'value')]
    + _slider_value_inputs
    + _int_slice_slider_inputs,
    prevent_initial_call=True,
)
def update_intensity_contours(*args_in):
    n_sl = N_PARAMS
    n_int = len(_int_slice_slider_inputs)
    n_fixed = len(args_in) - n_sl - n_int
    slider_values = list(args_in[n_fixed:n_fixed + n_sl])
    slice_indices = list(args_in[n_fixed + n_sl:])
    (species, idef, transition, zscale, _state, _ov_state,
     shift_dir, shift_rtol,
     interp_ny, interp_nx, interp_x_lim, interp_y_lim,
     interp_method, interp_clip, obs_boundary, extra_contours, extra_contour_color,
     plot_theme, grid_colorscale) = args_in[:n_fixed]
    theme = _parse_plot_theme(plot_theme)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    if not _grid or not _simline or not _as_str_list(species) or not _as_str_list(transition):
        p = placeholder_fig('Load grids and a SIMLINE directory', theme=theme)
        return (p,) * len(SLICE_PLANES)
    icfg = _interp_config(interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip)
    return make_intensity_contour_plots(
        species, idef, transition, zscale or 'log', slice_indices,
        shift_rtol=_parse_shift_rtol(shift_rtol),
        shift_scan_direction=_parse_shift_scan_direction(shift_dir),
        interp_config=icfg,
        colorscale=colorscale, theme=theme,
        slider_values=slider_values,
        show_obs_boundary=bool(obs_boundary and 'show' in obs_boundary),
        extra_contour_levels=extra_contours,
        extra_contour_color=extra_contour_color,
    )


@app.callback(
    Output('int-species', 'value', allow_duplicate=True),
    Input('btn-int-all-species', 'n_clicks'),
    State('int-species', 'options'),
    prevent_initial_call=True,
)
def add_all_int_species(_n, options):
    vals = [o['value'] for o in (options or [])]
    if not vals:
        raise PreventUpdate
    return vals


@app.callback(
    Output('int-transition', 'value', allow_duplicate=True),
    Input('btn-int-all-transitions', 'n_clicks'),
    State('int-transition', 'options'),
    prevent_initial_call=True,
)
def add_all_int_transitions(_n, options):
    vals = [o['value'] for o in (options or [])]
    if not vals:
        raise PreventUpdate
    return vals


@app.callback(
    Output('int-contour-extra-rows', 'children'),
    Input('int-species', 'value'),
    Input('int-transition', 'value'),
    Input('int-idef', 'value'),
    Input('simline-state', 'data'),
    Input('simline-overlay-state', 'data'),
    Input('grid-loaded', 'data'),
    State('int-zscale', 'value'),
    State('int-shift-scan-direction', 'value'),
    State('int-shift-match-rtol', 'value'),
    State('int-interp-ny', 'value'),
    State('int-interp-nx', 'value'),
    State('int-interp-x-lim', 'value'),
    State('int-interp-y-lim', 'value'),
    State('int-interp-method', 'value'),
    State('int-interp-clip', 'value'),
    State('int-obs-boundary', 'value'),
    State('int-extra-contours', 'value'),
    State('int-extra-contour-color', 'value'),
    State('plot-theme', 'value'),
    State('grid-colorscale', 'value'),
    *[State(f'slider-{d}', 'value') for d in range(N_PARAMS)],
    *[State(f'int-slice-slider-{p["id"]}', 'value') for p in SLICE_PLANES],
    State({'role': 'int-extra-slider', 'sp': ALL, 'tidx': ALL, 'plane': ALL}, 'id'),
    State({'role': 'int-extra-slider', 'sp': ALL, 'tidx': ALL, 'plane': ALL}, 'value'),
)
def build_int_contour_extra_rows(species, transition, idef, _state, _ov_state, _loaded,
                                 zscale, shift_dir, shift_rtol,
                                 interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                                 interp_method, interp_clip, obs_boundary, extra_contours,
                                 extra_contour_color, plot_theme, grid_colorscale,
                                 *rest):
    combos = _intensity_map_combos(species, transition, idef)[1:]
    if not combos:
        return []
    n_sl = N_PARAMS
    n_planes = len(SLICE_PLANES)
    slider_values = list(rest[:n_sl])
    slice_indices = list(rest[n_sl:n_sl + n_planes])
    saved = _saved_extra_slices(rest[n_sl + n_planes], rest[n_sl + n_planes + 1])
    full_width = bool(_simline_overlay)
    plane_ids = _active_slice_plane_ids()
    seed = _slice_seed_by_plane(slice_indices)
    rows = []
    for sp, tidx in combos:
        idents = [
            {'role': 'int-contour-extra', 'sp': sp, 'tidx': str(tidx),
             'plane': pid}
            for pid in plane_ids
        ]
        sl_idxs = [_extra_panel_slice_idx(ident, seed, saved) for ident in idents]
        figs = _int_contour_extra_figures(
            idents, idef, zscale, shift_dir, shift_rtol,
            interp_ny, interp_nx, interp_x_lim, interp_y_lim,
            interp_method, interp_clip, obs_boundary, extra_contours,
            extra_contour_color, plot_theme, grid_colorscale,
            slider_values, sl_idxs,
        )
        rows.append(html.Div([
            _extra_slice_caption(_intensity_combo_caption(sp, tidx, idef)),
            _extra_slice_graph_row(
                idents, full_width=full_width, figures=figs, slice_indices=sl_idxs,
                slider_role='int-extra-slider',
                label_role='int-extra-label',
                fixed_role='int-extra-fixed',
            ),
        ]))
    return rows


@app.callback(
    Output({'role': 'int-contour-extra', 'sp': MATCH, 'tidx': MATCH, 'plane': MATCH}, 'figure'),
    Output({'role': 'int-extra-label', 'sp': MATCH, 'tidx': MATCH, 'plane': MATCH}, 'children'),
    Output({'role': 'int-extra-label', 'sp': MATCH, 'tidx': MATCH, 'plane': MATCH}, 'style'),
    Output({'role': 'int-extra-fixed', 'sp': MATCH, 'tidx': MATCH, 'plane': MATCH}, 'children'),
    Input({'role': 'int-extra-slider', 'sp': MATCH, 'tidx': MATCH, 'plane': MATCH}, 'value'),
    Input('int-idef', 'value'),
    Input('int-zscale', 'value'),
    Input('simline-state', 'data'),
    Input('simline-overlay-state', 'data'),
    Input('int-shift-scan-direction', 'value'),
    Input('int-shift-match-rtol', 'value'),
    Input('int-interp-ny', 'value'),
    Input('int-interp-nx', 'value'),
    Input('int-interp-x-lim', 'value'),
    Input('int-interp-y-lim', 'value'),
    Input('int-interp-method', 'value'),
    Input('int-interp-clip', 'value'),
    Input('int-obs-boundary', 'value'),
    Input('int-extra-contours', 'value'),
    Input('int-extra-contour-color', 'value'),
    Input('plot-theme', 'value'),
    Input('grid-colorscale', 'value'),
    *_slider_value_inputs,
    prevent_initial_call=True,
)
def update_one_int_contour_extra(slice_idx, idef, zscale, _state, _ov_state,
                                 shift_dir, shift_rtol, interp_ny, interp_nx,
                                 interp_x_lim, interp_y_lim, interp_method, interp_clip,
                                 obs_boundary, extra_contours, extra_contour_color,
                                 plot_theme, grid_colorscale, *slider_values):
    ident = _match_output_id()
    figs = _int_contour_extra_figures(
        [ident], idef, zscale, shift_dir, shift_rtol,
        interp_ny, interp_nx, interp_x_lim, interp_y_lim,
        interp_method, interp_clip, obs_boundary, extra_contours,
        extra_contour_color, plot_theme, grid_colorscale,
        list(slider_values), [slice_idx],
    )
    plane = _plane_by_slot(ident['plane'])
    fixed, val, style = _slice_axis_ui(plane, slice_idx)
    return figs[0], val, style, fixed


@app.callback(
    Output('int-ratio-selector', 'options'),
    Output('int-ratio-selector', 'value'),
    Output('int-ratio-hint', 'children'),
    Input('int-species', 'value'),
    Input('int-transition', 'value'),
    Input('int-idef', 'value'),
    State('int-ratio-selector', 'value'),
)
def update_int_ratio_selector(species, transition, idef, current):
    return _int_ratio_dropdown(species, transition, idef, current)


@app.callback(
    Output('int-ratio-selector', 'value', allow_duplicate=True),
    Input('btn-int-all-ratios', 'n_clicks'),
    State('int-ratio-selector', 'options'),
    prevent_initial_call=True,
)
def add_all_int_ratios(_n, options):
    vals = [o['value'] for o in (options or [])]
    if not vals:
        raise PreventUpdate
    return vals


@app.callback(
    Output('int-ratio-rows', 'children'),
    Input('int-ratio-selector', 'value'),
    Input('simline-state', 'data'),
    Input('simline-overlay-state', 'data'),
    Input('grid-loaded', 'data'),
    State('int-idef', 'value'),
    State('int-species', 'value'),
    State('int-transition', 'value'),
    State('int-zscale', 'value'),
    State('int-shift-scan-direction', 'value'),
    State('int-shift-match-rtol', 'value'),
    State('int-interp-ny', 'value'),
    State('int-interp-nx', 'value'),
    State('int-interp-x-lim', 'value'),
    State('int-interp-y-lim', 'value'),
    State('int-interp-method', 'value'),
    State('int-interp-clip', 'value'),
    State('plot-theme', 'value'),
    State('grid-colorscale', 'value'),
    *[State(f'slider-{d}', 'value') for d in range(N_PARAMS)],
    *[State(f'int-slice-slider-{p["id"]}', 'value') for p in SLICE_PLANES],
    State({'role': 'int-ratio-slider', 'pair': ALL, 'plane': ALL}, 'id'),
    State({'role': 'int-ratio-slider', 'pair': ALL, 'plane': ALL}, 'value'),
)
def build_int_ratio_rows(pairs, _state, _ov_state, _loaded,
                         idef, species, transition, zscale, shift_dir, shift_rtol,
                         interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                         interp_method, interp_clip, plot_theme, grid_colorscale,
                         *rest):
    selected = _as_str_list(pairs)
    if not selected:
        return []
    n_sl = N_PARAMS
    n_planes = len(SLICE_PLANES)
    slider_values = list(rest[:n_sl])
    slice_indices = list(rest[n_sl:n_sl + n_planes])
    saved = _saved_extra_slices(rest[n_sl + n_planes], rest[n_sl + n_planes + 1])
    full_width = bool(_simline_overlay)
    plane_ids = _active_slice_plane_ids()
    seed = _slice_seed_by_plane(slice_indices)
    rows = []
    for pair in selected:
        idents = [
            {'role': 'int-ratio', 'pair': pair, 'plane': pid}
            for pid in plane_ids
        ]
        sl_idxs = [_extra_panel_slice_idx(ident, seed, saved) for ident in idents]
        figs = _int_ratio_figures(
            idents, idef, zscale, shift_dir, shift_rtol,
            interp_ny, interp_nx, interp_x_lim, interp_y_lim,
            interp_method, interp_clip, plot_theme, grid_colorscale,
            slider_values, sl_idxs, species, transition,
        )
        rows.append(html.Div([
            _extra_slice_caption(_int_ratio_caption(pair)),
            _extra_slice_graph_row(
                idents, full_width=full_width, figures=figs, slice_indices=sl_idxs,
                slider_role='int-ratio-slider',
                label_role='int-ratio-label',
                fixed_role='int-ratio-fixed',
            ),
        ]))
    return rows


@app.callback(
    Output({'role': 'int-ratio', 'pair': MATCH, 'plane': MATCH}, 'figure'),
    Output({'role': 'int-ratio-label', 'pair': MATCH, 'plane': MATCH}, 'children'),
    Output({'role': 'int-ratio-label', 'pair': MATCH, 'plane': MATCH}, 'style'),
    Output({'role': 'int-ratio-fixed', 'pair': MATCH, 'plane': MATCH}, 'children'),
    Input({'role': 'int-ratio-slider', 'pair': MATCH, 'plane': MATCH}, 'value'),
    Input('int-idef', 'value'),
    Input('int-zscale', 'value'),
    Input('simline-state', 'data'),
    Input('simline-overlay-state', 'data'),
    Input('int-shift-scan-direction', 'value'),
    Input('int-shift-match-rtol', 'value'),
    Input('int-interp-ny', 'value'),
    Input('int-interp-nx', 'value'),
    Input('int-interp-x-lim', 'value'),
    Input('int-interp-y-lim', 'value'),
    Input('int-interp-method', 'value'),
    Input('int-interp-clip', 'value'),
    Input('plot-theme', 'value'),
    Input('grid-colorscale', 'value'),
    *_slider_value_inputs,
    State('int-species', 'value'),
    State('int-transition', 'value'),
    prevent_initial_call=True,
)
def update_one_int_ratio(slice_idx, idef, zscale, _state, _ov_state,
                         shift_dir, shift_rtol, interp_ny, interp_nx,
                         interp_x_lim, interp_y_lim, interp_method, interp_clip,
                         plot_theme, grid_colorscale, *rest):
    slider_values = list(rest[:N_PARAMS])
    species = rest[N_PARAMS] if len(rest) > N_PARAMS else None
    transition = rest[N_PARAMS + 1] if len(rest) > N_PARAMS + 1 else None
    ident = _match_output_id()
    figs = _int_ratio_figures(
        [ident], idef, zscale, shift_dir, shift_rtol,
        interp_ny, interp_nx, interp_x_lim, interp_y_lim,
        interp_method, interp_clip, plot_theme, grid_colorscale,
        slider_values, [slice_idx], species, transition,
    )
    plane = _plane_by_slot(ident['plane'])
    fixed, val, style = _slice_axis_ui(plane, slice_idx)
    return figs[0], val, style, fixed


@app.callback(
    list(_int_spag_outputs) + list(_int_spag_chi2_outputs) + [Output('int-spaghetti-status', 'children')],
    [Input('int-spaghetti-contours', 'value'),
     Input('int-spaghetti-options', 'value'),
     Input('int-idef', 'value'),
     Input('simline-state', 'data'),
     Input('grid-loaded', 'data'),
     Input('int-interp-ny', 'value'),
     Input('int-interp-nx', 'value'),
     Input('int-interp-x-lim', 'value'),
     Input('int-interp-y-lim', 'value'),
     Input('int-interp-method', 'value'),
     Input('int-interp-clip', 'value'),
     Input('plot-theme', 'value')]
    + _slider_value_inputs
    + _int_spag_slice_slider_inputs,
    prevent_initial_call=True,
)
def update_spaghetti_plots(*args_in):
    n_sl = N_PARAMS
    n_sp = len(_int_spag_slice_slider_inputs)
    n_fixed = len(args_in) - n_sl - n_sp
    slider_values = list(args_in[n_fixed:n_fixed + n_sl])
    slice_indices = list(args_in[n_fixed + n_sl:])
    (contour_text, options, idef, _state, _loaded,
     interp_ny, interp_nx, interp_x_lim, interp_y_lim,
     interp_method, interp_clip, plot_theme) = args_in[:n_fixed]
    theme = _parse_plot_theme(plot_theme)
    options = options or []
    n_planes = len(SLICE_PLANES)
    if not _grid or not _simline:
        p = placeholder_fig('Load grids and a SIMLINE directory', theme=theme)
        return (p,) * n_planes + (p,) * n_planes + (
            _spaghetti_status_children(None, theme=theme),
        )
    icfg = _interp_config(interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip)
    spaghetti_figs, chi2_figs, chi2_info = make_spaghetti_plots(
        contour_text, idef or SIMLINE_DEFAULT_IDEF, slice_indices,
        interp_config=icfg, theme=theme, slider_values=slider_values,
        show_error_bands=('bands' in options),
        mark_closest=('chi2' in options),
    )
    return tuple(spaghetti_figs) + tuple(chi2_figs) + (_spaghetti_status_children(chi2_info, theme=theme),)


@app.callback(
    _int_rgb_outputs,
    [Input('int-rgb-sp0', 'value'),
     Input('int-rgb-sp1', 'value'),
     Input('int-rgb-sp2', 'value'),
     Input('int-rgb-cf0', 'value'),
     Input('int-rgb-cf1', 'value'),
     Input('int-rgb-cf2', 'value'),
     Input('int-idef', 'value'),
     Input('int-rgb-transitions', 'value'),
     Input('simline-state', 'data'),
     Input('grid-loaded', 'data'),
     Input('int-interp-ny', 'value'),
     Input('int-interp-nx', 'value'),
     Input('int-interp-x-lim', 'value'),
     Input('int-interp-y-lim', 'value'),
     Input('int-interp-method', 'value'),
     Input('int-interp-clip', 'value'),
     Input('plot-theme', 'value')]
    + _slider_value_inputs
    + _int_rgb_slice_slider_inputs,
    prevent_initial_call=True,
)
def update_intensity_rgb_plots(*args_in):
    n_sl = N_PARAMS
    n_int = len(_int_rgb_slice_slider_inputs)
    n_fixed = len(args_in) - n_sl - n_int
    slider_values = list(args_in[n_fixed:n_fixed + n_sl])
    slice_indices = list(args_in[n_fixed + n_sl:])
    (line0, line1, line2, cf0, cf1, cf2, idef, transitions, _state, _loaded,
     interp_ny, interp_nx, interp_x_lim, interp_y_lim,
     interp_method, interp_clip, plot_theme) = args_in[:n_fixed]
    theme = _parse_plot_theme(plot_theme)
    if not _grid or not _simline:
        p = placeholder_fig('Load grids and a SIMLINE directory', theme=theme)
        return (p,) * len(SLICE_PLANES)
    icfg = _interp_config(interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip)
    return make_intensity_rgb_plots(
        [line0, line1, line2], idef, transitions or [], slice_indices,
        interp_config=icfg, theme=theme, slider_values=slider_values,
        conv_factors=_parse_rgb_conv_factors(cf0, cf1, cf2),
    )


@app.callback(
    Output('plot-int-spectrum', 'figure'),
    _slider_value_inputs
    + _int_slice_slider_inputs
    + [Input('int-species', 'value'),
       Input('int-idef', 'value'),
       Input('simline-state', 'data'),
       Input('grid-loaded', 'data'),
       Input('plot-theme', 'value')],
    prevent_initial_call=True,
)
def update_intensity_spectrum(*args_in):
    n_int = len(_int_slice_slider_inputs)
    values = list(args_in[:N_PARAMS])
    int_slices = list(args_in[N_PARAMS:N_PARAMS + n_int])
    species, idef, _state, _loaded, plot_theme = args_in[N_PARAMS + n_int:]
    return fig_intensity_spectrum(
        values, species, idef,
        theme=_parse_plot_theme(plot_theme),
        int_slice_indices=int_slices,
    )


@app.callback(
    Output('plot-tgas', 'figure'),
    Output('plot-h-h2', 'figure'),
    Output('plot-cco', 'figure'),
    Output('plot-custom', 'figure'),
    _slider_value_inputs
    + [Input('xvar-choice', 'value'),
       Input('xscale', 'value'),
       Input('yscale', 'value'),
       Input('av-range', 'value'),
       Input('species-selector', 'value'),
       Input('overlay-state', 'data'),
       Input('plot-theme', 'value')],
    prevent_initial_call=True,
)
def update_profile_plots(*args_in):
    values = list(args_in[:N_PARAMS])
    xvar, xscale, yscale, av_range, custom_species, _overlay_state, plot_theme = (
        args_in[N_PARAMS:])
    return make_profile_plots(values, xvar, xscale, yscale, custom_species or [],
                              theme=_parse_plot_theme(plot_theme),
                              av_range=av_range)


@app.callback(
    Output('plot-thermal', 'figure'),
    Output('plot-heat-breakdown', 'figure'),
    Output('plot-cool-breakdown', 'figure'),
    _slider_value_inputs
    + [Input('xvar-choice', 'value'),
       Input('xscale', 'value'),
       Input('yscale', 'value'),
       Input('av-range', 'value'),
       Input('overlay-state', 'data'),
       Input('plot-theme', 'value')],
    prevent_initial_call=True,
)
def update_thermal_plots(*args_in):
    values = list(args_in[:N_PARAMS])
    xvar, xscale, yscale, av_range, _overlay_state, plot_theme = args_in[N_PARAMS:]
    return make_thermal_plots(values, xvar, xscale, yscale,
                              theme=_parse_plot_theme(plot_theme),
                              av_range=av_range)


@app.callback(
    Output('therm-crir-densities', 'options'),
    Output('therm-crir-densities', 'value'),
    Input('grid-loaded', 'data'),
    Input('thermal-subtabs', 'value'),
    State('therm-crir-densities', 'value'),
    prevent_initial_call=True,
)
def populate_therm_crir_densities(_loaded, _subtab, current):
    if not (_grid and _grid.get('has_hdf5')):
        return [], []
    dens_def = PARAM_DEFS[_PARAM_IDX['density']]
    toks = _grid['axis_tokens'].get('density', [])
    dvals = [round(float(np.log10(dens_def['decode'](t))), 1) for t in toks]
    opts = [{'label': f'log\u2081\u2080(n) = {v:.1f}', 'value': v} for v in dvals]
    kept = [v for v in (current or []) if v in dvals]
    if kept:
        value = kept
    else:
        value = dvals[:3] if len(dvals) > 3 else dvals
    return opts, value


@app.callback(
    Output('apdf-species', 'options'),
    Output('apdf-species', 'value'),
    Input('grid-loaded', 'data'),
    Input('grid-subtabs', 'value'),
    State('apdf-species', 'value'),
    prevent_initial_call=True,
)
def populate_apdf_species(_loaded, _subtab, current):
    if not (_grid and _grid.get('has_hdf5')):
        return [], []
    species = _grid.get('species', [])
    species_idx = _grid.get('species_idx', {})
    opts = [{'label': s, 'value': s} for s in species]
    kept = [s for s in _as_str_list(current) if s in species_idx]
    if kept:
        value = kept
    else:
        value = [s for s in DEFAULT_CUSTOM if s in species_idx][:3] or species[:1]
    return opts, value


@app.callback(
    Output('plot-heat-contrib-crir', 'figure'),
    Output('plot-abs-heatcool-crir', 'figure'),
    _slider_value_inputs
    + [Input('therm-crir-densities', 'value'),
       Input('therm-crir-threshold', 'value'),
       Input('therm-minrate', 'value'),
       Input('therm-minpoints', 'value'),
       Input('therm-show-totals', 'value'),
       Input('thermal-subtabs', 'value'),
       Input('overlay-state', 'data'),
       Input('plot-theme', 'value')],
    prevent_initial_call=True,
)
def update_heatcool_crir_plots(*args_in):
    values = list(args_in[:N_PARAMS])
    (densities, threshold, min_rate, min_points, show_totals,
     _subtab, _overlay_state, plot_theme) = args_in[N_PARAMS:]
    theme = _parse_plot_theme(plot_theme)
    if not (_grid and _grid.get('has_hdf5')):
        ph = placeholder_fig('Grid-integrated heating/cooling requires an HDF5 model grid',
                             theme=theme)
        return ph, ph
    data = _grid_heat_cool_dataframes(values)
    if not data:
        ph = placeholder_fig('No models for the current FUV / mass / metallicity selection',
                             theme=theme)
        return ph, ph
    dens = [float(d) for d in (densities or [])]
    if not dens:
        ph = placeholder_fig('Select one or more densities above', theme=theme)
        return ph, ph
    try:
        thr = float(threshold)
    except (TypeError, ValueError):
        thr = 0.01
    try:
        mr = float(min_rate)
    except (TypeError, ValueError):
        mr = 1e23
    try:
        mp = int(min_points)
    except (TypeError, ValueError):
        mp = 2
    show = bool(show_totals)
    f1 = fig_heating_contributions_vs_crir(data, dens, threshold=thr, theme=theme)
    f3 = fig_absolute_heating_cooling_vs_crir(
        data, dens, min_rate=mr, min_points=max(1, mp), show_totals=show, theme=theme)
    return f1, f3


@app.callback(
    Output('apdf-rows', 'children'),
    _slider_value_inputs
    + [Input('apdf-species', 'value'),
       Input('apdf-dens-choice', 'value'),
       Input('apdf-nbins', 'value'),
       Input('apdf-split-crir', 'value'),
       Input('apdf-log-color', 'value'),
       Input('apdf-overplot-avg', 'value'),
       Input('grid-subtabs', 'value'),
       Input('grid-loaded', 'data'),
       Input('plot-theme', 'value')],
    prevent_initial_call=True,
)
def update_abundance_pdf(*args_in):
    values = list(args_in[:N_PARAMS])
    (species, dens_choice, nbins, split, log_color, overplot_avg,
     _subtab, _loaded, plot_theme) = args_in[N_PARAMS:]
    theme = _parse_plot_theme(plot_theme)
    if not (_grid and _grid.get('has_hdf5')):
        return [dcc.Graph(figure=placeholder_fig(
            'Abundance PDF requires an HDF5 model grid', theme=theme), config=_GRAPH_CFG)]
    selected = _as_str_list(species)
    if not selected:
        return [html.Div('Select one or more species to build abundance PDFs.',
                         className='kosma-muted', style={'padding': '8px'})]
    split_on = bool(split)
    log_on = bool(log_color)
    over_on = bool(overplot_avg)
    try:
        nb = int(nbins)
    except (TypeError, ValueError):
        nb = 100
    rows = []
    for sp in selected:
        fig = fig_abundance_pdf(values, sp, dens_choice=(dens_choice or 'rel'),
                                n_bins=nb, split_by_crir=split_on,
                                log_color=log_on, overplot_average=over_on,
                                theme=theme)
        rows.append(dcc.Graph(figure=fig, config=_GRAPH_CFG,
                              style={'marginBottom': '14px'}))
    return rows


@app.callback(
    Output('plot-react-formation', 'figure'),
    Output('react-contrib-formation', 'children'),
    Output('plot-react-destruction', 'figure'),
    Output('react-contrib-destruction', 'children'),
    Output('plot-react-network', 'figure'),
    Output('react-network-status', 'children'),
    _slider_value_inputs
    + [Input('react-species', 'value'),
       Input('react-n', 'value'),
       Input('react-ranking', 'value'),
       Input('xscale', 'value'),
       Input('yscale', 'value'),
       Input('av-range', 'value'),
       Input('chem-state', 'data'),
       Input('chem-overlay-state', 'data'),
       Input('plot-theme', 'value'),
       Input('react-chain-upstream', 'value'),
       Input('react-chain-downstream', 'value'),
       Input('react-chain-isotopes', 'value'),
       Input('react-chain-ice', 'value'),
       Input('react-species-label-size', 'value'),
       Input('react-partner-label-size', 'value'),
       Input('react-net-highlight', 'data'),
       Input('react-net-hidden', 'data')],
    prevent_initial_call=True,
)
def update_reaction_plots(*args_in):
    values = list(args_in[:N_PARAMS])
    (species, top_n, ranking, xscale, yscale, av_range, _chem_state, _chem_overlay_state,
     plot_theme, chain_up, chain_down, chain_iso, chain_ice,
     species_pt, partner_pt, net_highlight, net_hidden) = args_in[N_PARAMS:]
    top_n = top_n or CHEM_DEFAULT_NREAC
    return make_reaction_plots(values, species, xscale, yscale, top_n,
                               ranking_metric=ranking,
                               theme=_parse_plot_theme(plot_theme),
                               av_range=av_range,
                               chain_upstream=chain_up,
                               chain_downstream=chain_down,
                               chain_include_isotopes=bool(chain_iso and 'iso' in chain_iso),
                               chain_include_ice=bool(chain_ice and 'ice' in chain_ice),
                               network_highlight=net_highlight,
                               network_hidden=net_hidden,
                               partner_label_size=partner_pt,
                               species_label_size=species_pt)


@app.callback(
    Output('react-net-highlight', 'data'),
    Input('plot-react-network', 'clickData'),
    Input('btn-react-net-clear', 'n_clicks'),
    Input('react-species', 'value'),
    State('react-net-highlight', 'data'),
    prevent_initial_call=True,
)
def update_react_net_highlight(click_data, clear_clicks, species, current):
    """Click a species box to isolate its links; Clear / species change resets."""
    from dash import ctx
    import re
    trig = getattr(ctx, 'triggered_id', None)
    if trig in ('btn-react-net-clear', 'react-species'):
        return None
    if not click_data:
        return current
    pts = click_data.get('points') or []
    if not pts:
        return current
    pt = pts[0]
    # Node traces carry plain species names in customdata.
    name = None
    cd = pt.get('customdata')
    if isinstance(cd, (list, tuple)):
        cd = cd[0] if cd else None
    if cd is not None and str(cd).strip():
        name = str(cd).strip()
    if not name:
        # Fallback: marker text may be HTML-formatted (CO⁺, H₂O, …).
        raw = pt.get('text')
        if raw:
            plain = re.sub(r'<[^>]+>', '', str(raw)).strip()
            # Reject empty / multi-line hover garbage; species labels are short.
            if plain and '\n' not in plain and len(plain) <= 24:
                name = plain
    if not name:
        # Clicked an edge / empty area — keep current highlight.
        return current
    # Toggle off if the same box is clicked again.
    return None if current == name else name


def _hidden_species_bar(hidden):
    """Chip row of hidden species; click a chip to restore that box."""
    names = [str(n).strip() for n in (hidden or []) if str(n).strip()]
    if not names:
        return html.Div()
    chips = []
    for n in names:
        chips.append(html.Button(
            n,
            id={'type': 'react-net-unhide', 'name': n},
            n_clicks=0,
            title=f'Show {n} again',
            style={'padding': '3px 9px', 'fontSize': '12px', 'marginRight': '6px',
                   'marginBottom': '4px', 'border': '1px solid #c5b48a',
                   'borderRadius': '12px', 'backgroundColor': '#fff6e0',
                   'cursor': 'pointer', 'color': '#5a4a20'},
        ))
    return html.Div([
        html.Span('Hidden: ', style={'fontSize': '12px', 'color': '#666',
                                     'marginRight': '6px'}),
        html.Div(chips, style={'display': 'inline'}),
        html.Span(' (click a name to restore)',
                  style={'fontSize': '12px', 'color': '#888', 'marginLeft': '4px'}),
    ])


@app.callback(
    Output('react-net-hidden', 'data'),
    Input('btn-react-net-hide', 'n_clicks'),
    Input('btn-react-net-show-all', 'n_clicks'),
    Input({'type': 'react-net-unhide', 'name': ALL}, 'n_clicks'),
    Input('react-species', 'value'),
    State('react-net-highlight', 'data'),
    State('react-net-hidden', 'data'),
    prevent_initial_call=True,
)
def update_react_net_hidden(hide_clicks, show_all_clicks, _unhide_clicks,
                            species, highlight, current):
    """Hide the highlighted species, restore one chip, or restore all."""
    from dash import ctx
    trig = getattr(ctx, 'triggered_id', None)
    hidden = [str(n).strip() for n in (current or []) if str(n).strip()]
    if trig in ('react-species', 'btn-react-net-show-all'):
        return []
    if isinstance(trig, dict) and trig.get('type') == 'react-net-unhide':
        name = str(trig.get('name') or '').strip()
        return [n for n in hidden if n != name]
    if trig != 'btn-react-net-hide':
        return hidden
    name = str(highlight or '').strip()
    if not name:
        return hidden
    # The chemistry-tab species is the tree root and cannot be removed.
    if species and chem_network._species_match_key(name) == chem_network._species_match_key(species):
        return hidden
    if any(chem_network._species_match_key(n) == chem_network._species_match_key(name)
           for n in hidden):
        return hidden
    hidden.append(name)
    return hidden


@app.callback(
    Output('react-net-hidden-bar', 'children'),
    Input('react-net-hidden', 'data'),
)
def update_react_net_hidden_bar(hidden):
    return _hidden_species_bar(hidden)


@app.callback(
    Output('fit-available-lines', 'children'),
    Input('simline-state', 'data'),
    Input('fit-idef', 'value'),
)
def update_fit_available_lines(_simline_state, idef):
    if not _simline:
        return 'Load a SIMLINE directory on the Load tab to see available line keys.'
    keys = gf.list_simline_line_keys(_simline, idef or SIMLINE_DEFAULT_IDEF)
    if not keys:
        return 'No transitions found for the selected intensity units.'
    preview = ', '.join(keys[:12])
    suffix = f' … (+{len(keys) - 12} more)' if len(keys) > 12 else ''
    return html.Span([
        html.Strong(f'{len(keys)} line keys available'),
        html.Span(f' (e.g. {preview}{suffix}). Use exact names in the maps JSON.'),
    ])


def _chi2_detail_figure(figs, key, default_msg, *, theme='light'):
    """Return a chi² analysis figure or a themed placeholder."""
    fig = figs.get(key)
    if fig is not None:
        return fig
    return placeholder_fig(default_msg, theme=theme)


@app.callback(
    Output('fit-status', 'children'),
    Output('fit-state', 'data'),
    Output('plot-fit-x', 'figure'),
    Output('plot-fit-y', 'figure'),
    Output('plot-fit-z', 'figure'),
    Output('plot-fit-chi2', 'figure'),
    Output('plot-fit-dominant', 'figure'),
    Output('plot-fit-kde', 'figure'),
    Output('plot-fit-chi2-pixel', 'figure'),
    Output('plot-fit-chi2-corner', 'figure'),
    Output('plot-fit-chi2-z', 'figure'),
    Output('plot-fit-chi2-bars', 'figure'),
    Output('plot-fit-species-contours', 'figure'),
    Input('btn-run-map-fit', 'n_clicks'),
    State('fit-maps-json', 'value'),
    State('fit-errors-json', 'value'),
    State('fit-output-dir', 'value'),
    State('fit-idef', 'value'),
    State('fit-nz', 'value'),
    State('fit-ny', 'value'),
    State('fit-nx', 'value'),
    State('fit-method', 'value'),
    State('fit-chi2-i', 'value'),
    State('fit-chi2-j', 'value'),
    State('fit-options', 'value'),
    State('plot-theme', 'value'),
    State('grid-colorscale', 'value'),
    State('fit-state', 'data'),
    prevent_initial_call=True,
)
def handle_map_fit(n_clicks, maps_json, errors_json, output_dir, idef,
                   nz, ny, nx, method, chi2_i, chi2_j, fit_options,
                   plot_theme, grid_colorscale, state):
    theme = _parse_plot_theme(plot_theme)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    empty = placeholder_fig('Run a map fit', theme=theme)
    chi2_empty = placeholder_fig('Enable “χ² analysis at pixel” and run a fit', theme=theme)
    kde_empty = placeholder_fig('Run a map fit to see parameter KDEs', theme=theme)
    if not n_clicks:
        return (dash.no_update, dash.no_update, empty, empty, empty, empty, empty, kde_empty,
                chi2_empty, chi2_empty, chi2_empty, chi2_empty, chi2_empty)

    opts = fit_options or []
    try:
        result = run_map_fit_job(
            maps_json, errors_json, output_dir, idef, nz, ny, nx, method,
            chi2_i, chi2_j, opts, opts, opts, opts,
        )
    except Exception as exc:
        err = html.Span(f'\u2717  {exc}',
                        style={'color': '#d62728', 'fontWeight': '600'})
        return (err, (state or 0) + 1, empty, empty, empty, empty, empty, kde_empty,
                chi2_empty, chi2_empty, chi2_empty, chi2_empty, chi2_empty)

    # WCS from the reference observed FITS (stored on the result)
    figs = gf.fit_result_figures(result, theme=theme, colorscale=colorscale)
    ref = result.get('reference_line', '')
    nlines = len(result.get('lines_fitted') or [])
    axes_label = result.get('fit_axes_label') or result.get('fit_axis_config', {}).get('axis_x', '')
    status = html.Span([
        html.Span('\u2713  Map fit complete. ', style={'color': '#2ca02c', 'fontWeight': '700'}),
        html.Span(f'{nlines} line(s), fit axes: {axes_label}, reference footprint: {ref}. '),
        html.Span('FITS saved to '),
        html.Code(result.get('output_dir', '')),
        html.Span(f"  \u2014  x: {os.path.basename(result.get('x_fits_path', ''))}, "
                  f"y: {os.path.basename(result.get('y_fits_path', ''))}, "
                  f"z: {os.path.basename(result.get('z_fits_path', ''))}"),
    ])
    chi2_fig = figs.get('chi2', placeholder_fig('No reduced χ² map', theme=theme))
    dom_fig = figs.get(
        'chi2_dominant',
        placeholder_fig('No dominant χ² contributor map', theme=theme),
    )
    return (
        status, (state or 0) + 1,
        figs.get('x', empty), figs.get('y', empty),
        figs.get('z', empty), chi2_fig, dom_fig,
        _chi2_detail_figure(figs, 'kde', 'No KDE (need ≥2 valid pixels per axis)', theme=theme),
        _chi2_detail_figure(figs, 'chi2_pixel_map', 'No χ² map (run fit with χ² analysis)', theme=theme),
        _chi2_detail_figure(figs, 'chi2_corner', 'No χ² corner plot (enable χ² analysis)', theme=theme),
        _chi2_detail_figure(figs, 'chi2_z_profile', 'No third-axis χ² profile', theme=theme),
        _chi2_detail_figure(figs, 'chi2_bars', 'No observed vs model bars', theme=theme),
        _chi2_detail_figure(figs, 'species_contours', 'No species contours (enable χ² analysis)', theme=theme),
    )


@app.callback(
    Output('plot-fit-x', 'figure', allow_duplicate=True),
    Output('plot-fit-y', 'figure', allow_duplicate=True),
    Output('plot-fit-z', 'figure', allow_duplicate=True),
    Output('plot-fit-chi2', 'figure', allow_duplicate=True),
    Output('plot-fit-dominant', 'figure', allow_duplicate=True),
    Output('plot-fit-kde', 'figure', allow_duplicate=True),
    Output('plot-fit-chi2-pixel', 'figure', allow_duplicate=True),
    Output('plot-fit-chi2-corner', 'figure', allow_duplicate=True),
    Output('plot-fit-chi2-z', 'figure', allow_duplicate=True),
    Output('plot-fit-chi2-bars', 'figure', allow_duplicate=True),
    Output('plot-fit-species-contours', 'figure', allow_duplicate=True),
    Input('plot-theme', 'value'),
    Input('grid-colorscale', 'value'),
    Input('fit-state', 'data'),
    prevent_initial_call=True,
)
def refresh_fit_figures_on_theme(plot_theme, grid_colorscale, _fit_state):
    if not _fit_results:
        return (dash.no_update,) * 11
    theme = _parse_plot_theme(plot_theme)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    figs = gf.fit_result_figures(_fit_results, theme=theme, colorscale=colorscale)
    empty = placeholder_fig('Run a map fit', theme=theme)
    return (
        figs.get('x', empty), figs.get('y', empty),
        figs.get('z', empty),
        figs.get('chi2', placeholder_fig('No reduced χ² map', theme=theme)),
        figs.get('chi2_dominant', placeholder_fig('No dominant χ² contributor map', theme=theme)),
        _chi2_detail_figure(figs, 'kde', 'No KDE (need ≥2 valid pixels per axis)', theme=theme),
        _chi2_detail_figure(figs, 'chi2_pixel_map', 'No χ² map (run fit with χ² analysis)', theme=theme),
        _chi2_detail_figure(figs, 'chi2_corner', 'No χ² corner plot (enable χ² analysis)', theme=theme),
        _chi2_detail_figure(figs, 'chi2_z_profile', 'No third-axis χ² profile', theme=theme),
        _chi2_detail_figure(figs, 'chi2_bars', 'No observed vs model bars', theme=theme),
        _chi2_detail_figure(figs, 'species_contours', 'No species contours (enable χ² analysis)', theme=theme),
    )


@app.callback(
    Output('plot-fit-chi2-pixel', 'figure', allow_duplicate=True),
    Output('plot-fit-chi2-corner', 'figure', allow_duplicate=True),
    Output('plot-fit-chi2-z', 'figure', allow_duplicate=True),
    Output('plot-fit-chi2-bars', 'figure', allow_duplicate=True),
    Output('plot-fit-species-contours', 'figure', allow_duplicate=True),
    Input('fit-chi2-i', 'value'),
    Input('fit-chi2-j', 'value'),
    Input('fit-state', 'data'),
    State('plot-theme', 'value'),
    State('grid-colorscale', 'value'),
    prevent_initial_call=True,
)
def refresh_chi2_at_pixel(chi2_i, chi2_j, _fit_state, plot_theme, grid_colorscale):
    """Re-run χ² analysis when the pixel row/col changes after a fit."""
    global _fit_results
    theme = _parse_plot_theme(plot_theme)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    chi2_empty = placeholder_fig('Run a map fit with χ² analysis enabled', theme=theme)
    if not _fit_results or not _fit_results.get('chi2_reanalysis'):
        return chi2_empty, chi2_empty, chi2_empty, chi2_empty, chi2_empty
    if chi2_i is None or chi2_j is None:
        figs = gf.chi2_analysis_figures(_fit_results, theme=theme, colorscale=colorscale)
    else:
        try:
            pixel = (int(chi2_i), int(chi2_j))
        except (TypeError, ValueError):
            figs = gf.chi2_analysis_figures(_fit_results, theme=theme, colorscale=colorscale)
        else:
            updated = gf.run_chi2_at_pixel(_fit_results, pixel)
            if updated:
                _fit_results.update({
                    k: updated[k]
                    for k in (
                        'chi2_analysis', 'chi2_analysis_pixel', 'chi2_observed_values',
                        'chi2_observed_errors', 'chi2_subtitle',
                    )
                    if k in updated
                })
            figs = gf.chi2_analysis_figures(
                _fit_results, theme=theme, colorscale=colorscale, chi2_pixel=pixel,
            )
    return (
        _chi2_detail_figure(figs, 'chi2_pixel_map', 'No χ² map', theme=theme),
        _chi2_detail_figure(figs, 'chi2_corner', 'No χ² corner plot at this pixel', theme=theme),
        _chi2_detail_figure(figs, 'chi2_z_profile', 'No third-axis χ² profile', theme=theme),
        _chi2_detail_figure(figs, 'chi2_bars', 'No observed vs model bars', theme=theme),
        _chi2_detail_figure(figs, 'species_contours', 'No species contours at this pixel', theme=theme),
    )


@app.callback(
    Output('cr-atten-profiles', 'data'),
    Output('cr-atten-active', 'options'),
    Output('cr-atten-active', 'value'),
    Input('cr-atten-add', 'n_clicks'),
    Input('cr-atten-add-overlay', 'n_clicks'),
    Input('cr-atten-clear', 'n_clicks'),
    _slider_value_inputs,
    State('cr-atten-profiles', 'data'),
    State('cr-atten-active', 'value'),
    prevent_initial_call=True,
)
def manage_cr_atten_profiles(n_add, n_add_ov, n_clear, *args_in):
    slider_values = list(args_in[:N_PARAMS])
    profiles = list(args_in[N_PARAMS] or [])
    cur_active = list(args_in[N_PARAMS + 1] or [])

    trigger = (dash.callback_context.triggered[0]['prop_id'].split('.')[0]
               if dash.callback_context.triggered else '')

    def _entry(tokens):
        tokens = [int(t) for t in tokens]
        key = ','.join(str(t) for t in tokens)
        return {'key': key, 'tokens': tokens, 'label': _cr_atten_profile_label(tokens)}

    def _sync_options_active(data, active):
        opts = [
            {'label': _cr_atten_profile_label(p['tokens']), 'value': p['key']}
            for p in data
        ]
        valid = {o['value'] for o in opts}
        active = [k for k in active if k in valid]
        if not active and opts:
            active = [o['value'] for o in opts]
        return opts, active

    if trigger == 'cr-atten-clear':
        return [], [], []

    existing = {p['key']: p for p in profiles}

    if trigger == 'cr-atten-add':
        tokens = _tokens_from_values(slider_values)
        if tokens is not None and _filepath_for_tokens(tokens):
            existing[_entry(tokens)['key']] = _entry(tokens)

    elif trigger == 'cr-atten-add-overlay':
        for tokens in _overlay_variants_for_sliders(slider_values):
            if _filepath_for_tokens(tokens):
                ent = _entry(tokens)
                existing[ent['key']] = ent

    profiles = list(existing.values())
    opts, active = _sync_options_active(profiles, cur_active)
    return profiles, opts, active


@app.callback(
    Output('plot-cr-atten', 'figure'),
    Input('cr-atten-profiles', 'data'),
    Input('cr-atten-active', 'value'),
    Input('cr-atten-xaxis', 'value'),
    Input('cr-atten-options', 'value'),
    Input('cr-atten-stop-rate', 'value'),
    Input('plot-theme', 'value'),
    prevent_initial_call=True,
)
def update_cr_atten_plot(profiles, active, x_axis, options, stop_rate, plot_theme):
    opts = options or []
    return fig_cr_attenuation(
        profiles,
        active,
        x_axis=x_axis or 'nh2',
        show_padovani='padovani' in opts,
        extrapolate='extrap' in opts,
        stopping_rate=stop_rate,
        show_threshold='threshold' in opts,
        theme=_parse_plot_theme(plot_theme),
    )


# --- Entry point --------------------------------------------------------------

if __name__ == '__main__':
    if args.dir:
        try:
            g = scan_directory(args.dir, recursive=args.recursive)
            print(f'Pre-loaded {g["n_files"]} models from {g["directory"]}')
        except Exception as exc:
            print(f'Could not pre-load --dir: {exc}')
    run = getattr(app, 'run', None) or app.run_server
    run(debug=args.debug, host=args.host, port=args.port)
