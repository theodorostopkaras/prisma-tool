#!/usr/bin/env python3
"""
KOSMA-tau Grid Explorer
=======================

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

import numpy as np
import h5py

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import grid_fit as gf
import grid_interp as gi
import model_config as mc
import simline_spectra as ss
import obs_spectrum_fits as osf
import cr_attenuation as cra
import grid_naming as gn


# --- CLI ----------------------------------------------------------------------

OBS_ROW_OPTIONS = [
    {'label': ' Mean (all spectra / pixels)', 'value': 'mean'},
    {'label': ' Peak region (bright half)', 'value': 'peak'},
    {'label': ' Single row / pixel index', 'value': 'row'},
]
DEFAULT_OBS_V_LOW = -20.0
DEFAULT_OBS_V_HIGH = 20.0
DEFAULT_DIR = '/home/teotopkaras/Desktop/new_teo_grid/pdrgrid_hdf5'


parser = argparse.ArgumentParser(description="KOSMA-tau Grid Explorer")
parser.add_argument('--dir', default=DEFAULT_DIR, type=str,
                    help='Directory containing the per-model .hdf5 files.')
parser.add_argument('--recursive', action='store_true',
                    help='Search the directory tree recursively for .hdf5 files.')
parser.add_argument('--port',  default=8050, type=int)
parser.add_argument('--host',  default='127.0.0.1')
parser.add_argument('--debug', action='store_true')
args, _ = parser.parse_known_args()


# --- Constants ----------------------------------------------------------------

MIN_AB = 1e-30                      # floor for log abundance plots
MIN_RATE = 1e-40                    # floor for log heating/cooling rate plots
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
COMPACT_CONTOUR_PANEL_W = 380
DEFAULT_CONTOUR_PANEL_W = 520
COMPACT_FIG_WIDTH = COMPACT_CONTOUR_PANEL_W + 90
COMPACT_FIG_HEIGHT = COMPACT_CONTOUR_PANEL_W + 106
# Identical axes box + colorbar slot for every side-by-side slice figure.
_COMPACT_FIG_MARGIN = dict(l=58, r=72, t=44, b=50)
_COMPACT_XDOMAIN = [0.0, 0.80]
_COMPACT_YDOMAIN = [0.0, 0.94]
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
    {'label': ' Light', 'value': 'light'},
    {'label': ' Dark', 'value': 'dark'},
]
DEFAULT_PLOT_THEME = 'light'
PLOT_THEMES = {
    'light': dict(
        paper_bg='#ffffff',
        plot_bg='#f8f9fa',
        grid='#e0e0e0',
        axis_line='#bbbbbb',
        title='#333333',
        font='#333333',
        legend_bg='rgba(255,255,255,0.85)',
        legend_border='#cccccc',
        placeholder='#aaaaaa',
        placeholder_plot='#f4f4f4',
        contour_line='rgba(255,255,255,0.85)',
        vline='rgba(60,60,60,0.45)',
        controls_bg='#f0f4ff',
        page_bg='#ffffff',
    ),
    'dark': dict(
        paper_bg='#1a1a2e',
        plot_bg='#16213e',
        grid='#2a3a5c',
        axis_line='#4a5a7a',
        title='#e8eaf0',
        font='#e0e0e0',
        legend_bg='rgba(26,26,46,0.92)',
        legend_border='#4a5a7a',
        placeholder='#888888',
        placeholder_plot='#121528',
        contour_line='rgba(255,255,255,0.35)',
        vline='rgba(220,220,230,0.45)',
        controls_bg='#1e293b',
        page_bg='#0f1419',
    ),
}
DEFAULT_ERROR_DECIMATION = 2
DEFAULT_ERROR_REL_THRESHOLD = 0.01
ERROR_METRIC_OPTIONS = [
    {'label': ' Relative (%)', 'value': 'relative'},
    {'label': ' Absolute', 'value': 'absolute'},
]
ERROR_PANEL_CMAP = 'RdYlGn_r'

# Internal HDF5 metadata field keys (KOSMA-tau convention).
KEY_AV    = 'av'             # visual extinction profile (Positions, col 0)
KEY_NH    = 'protdens'       # proton/H nucleus density profile (Gas state, col 0)
KEY_TGAS  = 'tgas'           # gas temperature  (Gas state, col 2)
KEY_TDUST = 'tdust'          # dust temperature (Gas state, col 3)
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

# 2-D contour slice planes: (x, y) axes on the plot; ``slice`` is the third axis
# controlled by that panel's own slider (independent of the profile sliders).
SLICE_PLANES = [
    dict(id='dens-fuv',  x='density', y='fuv',  slice='crir',
         title='n<sub>H</sub> vs FUV'),
    dict(id='dens-crir', x='crir', y='density', slice='fuv',
         title='\u03B6 vs n<sub>H</sub>'),
    dict(id='fuv-crir',  x='crir', y='fuv',     slice='density',
         title='\u03B6 vs FUV'),
]
_IE_PLANE_LABELS = {
    'dens-fuv': 'nH vs FUV',
    'dens-crir': '\u03B6 vs nH',
    'fuv-crir': '\u03B6 vs FUV',
}

CONTOUR_DIAGNOSTICS = [
    ('tgas', 'T<sub>gas</sub> (cloud edge)'),
    ('tdust', 'T<sub>dust</sub> (cloud edge)'),
    ('nh', 'n<sub>H</sub> (cloud edge)'),
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

def parse_filename(fname):
    """Return a tuple of 6 integer tokens (DD, MM, FF, ZZ, CC, AA) or None.

    Works for both ``Model<tag>_DD_MM_FF_ZZ_CC_AA.hdf5`` and
    ``Model<tag>_DD_MM_FF_ZZ_CC.hdf5`` (missing ``AA`` -> atten 0), plus
    chemistry-grid ``chem_Model<tag>_…`` names.
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    return gn.parse_model_tokens_from_stem(stem)


def _clean_rate_label(label, kind):
    """Strip the 'Heating - ' / 'Cooling - ' prefix from a metadata label."""
    for prefix in ('Heating - ', 'Cooling - ', 'Heating-', 'Cooling-'):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


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
        species = [_dec(x[0]) for x in hf[SPECIES_PATH][:]]
    heat_comp.sort(key=lambda t: t[1])
    cool_comp.sort(key=lambda t: t[1])
    return field_map, species, heat_comp, cool_comp, cr_idx


def _scan_files(directory, recursive):
    """Return (directory, files, axis_tokens, n_skipped) for a grid directory."""
    directory = os.path.expanduser((directory or '').strip())
    if not directory or not os.path.isdir(directory):
        raise FileNotFoundError(f'Directory not found: {directory!r}')

    pattern = os.path.join(directory, '**', '*.hdf5') if recursive \
        else os.path.join(directory, '*.hdf5')
    paths = sorted(glob.glob(pattern, recursive=recursive))
    if not paths:
        raise FileNotFoundError(f'No .hdf5 files found in {directory!r}')

    files = {}
    skipped = 0
    for path in paths:
        tokens = parse_filename(path)
        if tokens is None:
            skipped += 1
            continue
        files[tokens] = path
    if not files:
        raise ValueError('Found .hdf5 files but none matched the expected '
                         'Model_DD_MM_FF_ZZ_CC[_AA] naming convention.')

    axis_tokens = {}
    for d, p in enumerate(PARAM_DEFS):
        axis_tokens[p['key']] = sorted({tok[d] for tok in files})
    return directory, files, axis_tokens, skipped


def scan_directory(directory, recursive=False):
    """Scan ``directory`` for per-model HDF5 files and build the main grid."""
    global _grid, _field_map, _profile_cache
    global _heat_components, _cool_components, _cr_heat_idx
    global _model_config_summary

    directory, files, axis_tokens, skipped = _scan_files(directory, recursive)
    field_map, species, heat_comp, cool_comp, cr_idx = \
        build_structure(next(iter(files.values())))

    _profile_cache = {}
    _scalar_cache = {}
    _field_map = field_map
    _heat_components = heat_comp
    _cool_components = cool_comp
    _cr_heat_idx = cr_idx
    _model_config_summary = mc.scan_model_configs(directory)
    _grid = dict(
        directory=directory,
        files=files,
        axis_tokens=axis_tokens,
        species=species,
        species_idx={s: i for i, s in enumerate(species)},
        n_files=len(files),
        n_skipped=skipped,
    )
    return _grid


def scan_overlay(directory, recursive=False):
    """Scan a second (e.g. attenuated) grid to be overplotted on the figures.

    Matching to the main grid is done on every axis *except* attenuation, so an
    attenuated counterpart (different AA tag) lines up with each main model.
    """
    global _overlay
    directory, files, axis_tokens, skipped = _scan_files(directory, recursive)
    # Index ignoring the attenuation token (last entry in PARAM order).
    by_non_atten = {}
    for tokens, path in files.items():
        by_non_atten.setdefault(tokens[:N_PARAMS - 1], path)
    _overlay = dict(
        directory=directory,
        files=files,
        by_non_atten=by_non_atten,
        axis_tokens=axis_tokens,
        n_files=len(files),
        n_skipped=skipped,
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
    directory, files, axis_tokens, skipped = _scan_files(directory, recursive)
    species, labels = build_chem_structure(next(iter(files.values())))
    by_non_atten = {}
    for tokens, path in files.items():
        by_non_atten.setdefault(tokens[:N_PARAMS - 1], path)
    _reaction_cache = {}
    _chem = dict(
        directory=directory,
        files=files,
        by_non_atten=by_non_atten,
        species=species,
        labels=labels,
        n_files=len(files),
        n_skipped=skipped,
    )
    return _chem


def clear_chem():
    global _chem, _reaction_cache
    _chem = {}
    _reaction_cache = {}


def parse_smli_filename(fname):
    """Parse ``{jtemp,jerg,tau}_Model<tag>_DD_MM_FF_ZZ_CC_AA_<species>.smli`` (or ``.smlc``).

    Returns (quantity_key, tokens, species) or None.
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
    mi = next((i for i, p in enumerate(parts) if p.startswith('Model')), None)
    if mi is None:
        return None
    tokens = gn.parse_model_tokens_from_parts(parts, mi)
    if tokens is None:
        return None

    sp_start = mi + 1 + gn.param_token_count(parts, mi)
    if sp_start >= len(parts):
        return None
    species = '_'.join(parts[sp_start:])
    return idef, tokens, species


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
    for path in paths:
        parsed = parse_smli_filename(path)
        if parsed is None:
            skipped += 1
            continue
        idef, tokens, species = parsed
        key = (tokens, species, idef)
        files[key] = path
        if build_by_non_atten:
            by_non_atten.setdefault((tokens[:N_PARAMS - 1], species, idef), path)
        species_set.add(species)
        tkey = (species, idef)
        if tkey not in transitions:
            rows = read_smli_file(path)
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
                         'jtemp_/jerg_/tau_Model_DD_MM_FF_ZZ_CC_AA_<species> naming convention.')

    out = dict(
        directory=directory,
        files=files,
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


def smli_file(tokens, species, idef, overlay=False):
    """Return the .smli path for one model point, or None."""
    store = _simline_overlay if overlay else _simline
    if not store:
        return None
    t = tuple(tokens)
    path = store['files'].get((t, species, idef))
    if path:
        return path
    if overlay:
        return store.get('by_non_atten', {}).get((t[:N_PARAMS - 1], species, idef))
    return None


def model_core_from_tokens(tokens):
    """HDF5 / SimLine model folder stem for the current slider selection."""
    path = current_file(tokens)
    if not path:
        return None
    return os.path.splitext(os.path.basename(path))[0]


def pv_fits_path(tokens, species, transition, quantity='intensity'):
    """Return the PV FITS path for one model point / species / transition."""
    if not _simline or tokens is None or not species or transition is None:
        return None
    key = (tuple(tokens), species, str(transition))
    store_key = 'pv_tau_index' if quantity == 'tau' else 'pv_index'
    path = _simline.get(store_key, {}).get(key)
    if path and os.path.isfile(path):
        return path
    model_core = model_core_from_tokens(tokens)
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
    return PARAM_DEFS[_PARAM_IDX[key]]['decode'](token)


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
    )
    _fit_results['output_dir'] = out_dir
    _fit_results['grid_shape'] = shape
    _fit_results['lines_fitted'] = line_names
    _fit_results['fits_header'] = fits_header
    _fit_results['reference_fits_path'] = ref_path
    return _fit_results


def chem_file(values):
    """Find the chem-grid file matching the current slider selection."""
    if not _chem:
        return None
    tokens = _tokens_from_values(values)
    if tokens is None:
        return None
    return _chem['files'].get(tokens) \
        or _chem['by_non_atten'].get(tokens[:N_PARAMS - 1])


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


def get_model(filepath):
    """Load (and cache) the depth profiles needed for plotting one model."""
    if filepath in _profile_cache:
        return _profile_cache[filepath]
    with h5py.File(filepath, 'r') as hf:
        model = dict(
            av     = _read_field(hf, KEY_AV),
            nH     = _read_field(hf, KEY_NH),
            tgas   = _read_field(hf, KEY_TGAS),
            tdust  = _read_field(hf, KEY_TDUST),
            cosray = _read_field(hf, KEY_COSRAY),
            nh2_profile = _read_field(hf, KEY_NH2_PROFILE),
            radius = _read_field(hf, KEY_RADIUS),
            rel    = np.asarray(hf[RELDENS_PATH][:], dtype=float),   # (n_depth, n_species)
            dens   = np.asarray(hf[DENS_PATH][:], dtype=float),
            heat   = _read_optional(hf, HEATING_PATH),               # (n_depth, n_heat)
            cool   = _read_optional(hf, COOLING_PATH),               # (n_depth, n_cool)
        )
    _profile_cache[filepath] = model
    return model


def species_abundance(model, name, yscale):
    """Relative abundance profile for a species, clipped for log scale."""
    idx = _grid['species_idx'].get(name)
    if idx is None or idx >= model['rel'].shape[1]:
        return None
    ab = model['rel'][:, idx].astype(float)
    if yscale == 'log':
        ab = np.where(ab > 0, ab, MIN_AB)
    return ab


def _integrate_sphere(radius_pc, profile):
    """Volume integral 4 pi r^2 n(r) dr (KoSens Abundance_Calculator convention)."""
    radius = np.asarray(radius_pc, dtype=float)
    profile = np.asarray(profile, dtype=float)
    if radius.size == 0 or profile.size != radius.size:
        return np.nan
    flipped_radius = radius[::-1]
    radius_cm = np.insert(flipped_radius, 0, 0.0) * PC_TO_CM
    flipped_prof = profile[::-1]
    prof_cm = np.insert(flipped_prof, 0, flipped_prof[0])
    return float(np.trapezoid(4.0 * np.pi * radius_cm ** 2 * prof_cm, radius_cm))


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
    labels = _chem.get('labels', {}).get(species, {}).get(mode, [])
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


def param_value_label(param, token):
    """Decoded physical value formatted for a parameter."""
    val = param['decode'](token)
    if param['key'] == 'atten':
        return f'{token:02d}'
    return _sci_label(val)


def build_marks(param):
    """Slider marks for a parameter (decoded values), thinned to <=9 labels."""
    tokens = _grid['axis_tokens'][param['key']]
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
        margin=dict(l=70, r=20, t=44, b=54),
        font=dict(family='Arial, sans-serif', size=12, color=t['font']),
        height=320,
        legend=dict(bgcolor=t['legend_bg'], borderwidth=1, bordercolor=t['legend_border']),
    )


def _axis_style(theme='light'):
    t = _theme_colors(theme)
    return dict(
        showgrid=True, gridcolor=t['grid'], gridwidth=1,
        zeroline=False, linecolor=t['axis_line'], mirror=True,
        exponentformat='e', showexponent='all',
        tickfont=dict(color=t['font']),
    )


def _apply_layout(fig, title, xlabel, xtype, xrange, ylabel, ytype, theme='light'):
    t = _theme_colors(theme)
    fig.update_layout(
        **_base_layout(theme),
        title=dict(text=title, font=dict(size=13, color=t['title']), x=0.02, xanchor='left'),
        xaxis=dict(**_axis_style(theme), title=dict(text=xlabel, font=dict(size=12)),
                   type=xtype, range=xrange),
        yaxis=dict(**_axis_style(theme), title=dict(text=ylabel, font=dict(size=12)), type=ytype),
    )


def _xvals(model, xvar, xscale):
    """Return (x_array, x_label, x_type, x_range) with an explicit axis range."""
    if xvar == 'nH':
        xv = np.asarray(model['nH'], dtype=float)
        xl = 'n<sub>H</sub> (cm<sup>-3</sup>)'
    else:
        xv = np.asarray(model['av'], dtype=float)
        xl = 'A<sub>V</sub> (mag)'

    # For Av, never show values below this floor (mag).
    av_floor = 1e-5

    pos = xv[xv > 0]
    if xscale == 'log' and pos.size:
        lo_exp = np.floor(np.log10(pos.min()))
        if xvar == 'Av':
            lo_exp = max(lo_exp, np.log10(av_floor))
            xv = np.where(xv >= av_floor, xv, np.nan)
        else:
            xv = np.where(xv > 0, xv, np.nan)
        xrange = [lo_exp, np.ceil(np.log10(pos.max()))]
    else:
        if xvar == 'Av':
            lo = av_floor
            xv = np.where(xv >= av_floor, xv, np.nan)
        else:
            lo = float(np.nanmin(xv))
        xrange = [lo, float(np.nanmax(xv)) * 1.02]
        xscale = 'linear'
    return xv, xl, xscale, xrange


def _x_array(model, xvar, xscale):
    """x-array for a model, masked consistently with ``_xvals`` (for overlays)."""
    av_floor = 1e-5
    xv = np.asarray(model['nH'] if xvar == 'nH' else model['av'], dtype=float)
    if xvar == 'Av':
        xv = np.where(xv >= av_floor, xv, np.nan)
    elif xscale == 'log':
        xv = np.where(xv > 0, xv, np.nan)
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
    if idx_h is None or idx_h2 is None:
        return None
    ab_h = model['rel'][:, idx_h].astype(float)
    ab_h2 = model['rel'][:, idx_h2].astype(float)
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


def placeholder_fig(msg='Load a grid directory to begin', theme='light'):
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
    arr = np.asarray(model[key], dtype=float)
    if yscale == 'log':
        arr = np.where(arr > 0, arr, np.nan)
    return arr


def fig_tgas(model, overlay, xvar, xscale, yscale, theme='light'):
    xv, xl, xt, xr = _xvals(model, xvar, xscale)
    xvo = _x_array(overlay, xvar, xscale) if overlay else None
    fig = go.Figure()
    _line(fig, xv, _temperature(model, 'tgas', yscale), COLORS[0], 'T<sub>gas</sub>')
    _line(fig, xv, _temperature(model, 'tdust', yscale), COLORS[1], 'T<sub>dust</sub>', dash='dot')
    if overlay:
        _line(fig, xvo, _temperature(overlay, 'tgas', yscale), COLORS[0], 'T<sub>gas</sub>', overlay=True)
        _line(fig, xvo, _temperature(overlay, 'tdust', yscale), COLORS[1], 'T<sub>dust</sub>', overlay=True)
    _apply_layout(fig, 'Gas / Dust Temperature', xl, xt, xr, 'T (K)', yscale, theme=theme)
    return fig


def _fig_species(model, overlay, xvar, xscale, yscale, specs, title, theme='light'):
    xv, xl, xt, xr = _xvals(model, xvar, xscale)
    xvo = _x_array(overlay, xvar, xscale) if overlay else None
    fig = go.Figure()
    for name, color in specs:
        _line(fig, xv, species_abundance(model, name, yscale), color, format_species_html(name))
    if overlay:
        for name, color in specs:
            _line(fig, xvo, species_abundance(overlay, name, yscale), color,
                  format_species_html(name), overlay=True)
    _apply_layout(fig, title, xl, xt, xr, 'x(species)', yscale, theme=theme)
    return fig


def fig_h_h2(model, overlay, xvar, xscale, yscale, theme='light'):
    return _fig_species(model, overlay, xvar, xscale, yscale,
                        [('H', COLORS[0]), ('H2', COLORS[1])], 'H / H<sub>2</sub>',
                        theme=theme)


def fig_cplus_c_co(model, overlay, xvar, xscale, yscale, theme='light'):
    return _fig_species(model, overlay, xvar, xscale, yscale,
                        [('C+', COLORS[3]), ('C', COLORS[2]), ('CO', COLORS[0])],
                        'C<sup>+</sup> / C / CO', theme=theme)


def fig_custom(model, overlay, xvar, xscale, yscale, sel_species, theme='light'):
    specs = [(sp, COLORS[k % len(COLORS)]) for k, sp in enumerate(sel_species or [])]
    return _fig_species(model, overlay, xvar, xscale, yscale, specs, 'Custom Species',
                        theme=theme)


_RATE_YLABEL = '\u0393, \u039B (erg cm<sup>-3</sup> s<sup>-1</sup>)'

_RATE_YLABEL_heating = '\u0393 (erg cm<sup>-3</sup> s<sup>-1</sup>)'
_RATE_YLABEL_cooling = '\u039B (erg cm<sup>-3</sup> s<sup>-1</sup>)'


def fig_thermal(model, overlay, xvar, xscale, yscale, theme='light'):
    """Total heating vs total cooling, with the cosmic-ray heating highlighted."""
    xv, xl, xt, xr = _xvals(model, xvar, xscale)
    xvo = _x_array(overlay, xvar, xscale) if overlay else None
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
                  xl, xt, xr, _RATE_YLABEL, yscale, theme=theme)
    return fig


_DASH_CYCLE = ['solid', 'dot', 'dash', 'dashdot']


def _fig_rate_breakdown(model, overlay, xvar, xscale, yscale,
                        key, components, title, emphasize_idx=None, theme='light'):
    """Plot every component of a rate matrix (heating or cooling) individually.

    With more components than colours, the dash pattern is cycled too so all
    lines stay distinguishable.
    """
    xv, xl, xt, xr = _xvals(model, xvar, xscale)
    xvo = _x_array(overlay, xvar, xscale) if overlay else None
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
                  yscale, theme=theme)
    return fig


def fig_heat_breakdown(model, overlay, xvar, xscale, yscale, theme='light'):
    """All heating-rate components; cosmic-ray heating drawn thicker."""
    return _fig_rate_breakdown(model, overlay, xvar, xscale, yscale,
                               'heat', _heat_components,
                               'Heating-rate components (all)', _cr_heat_idx,
                               theme=theme)


def fig_cool_breakdown(model, overlay, xvar, xscale, yscale, theme='light'):
    """All cooling-rate components."""
    return _fig_rate_breakdown(model, overlay, xvar, xscale, yscale,
                               'cool', _cool_components,
                               'Cooling-rate components (all)', theme=theme)


_REACT_YLABEL = 'rate (cm<sup>-3</sup> s<sup>-1</sup>)'
_REACT_TABLE_STYLE = {
    'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '12px',
    'marginTop': '8px',
}
_REACT_TABLE_CELL = {
    'padding': '4px 8px', 'borderBottom': '1px solid #e0e0e0',
    'verticalAlign': 'top',
}


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
                  ranking_metric=DEFAULT_REACT_RANKING, theme='light'):
    """Top-N formation or destruction reactions for a species (vs A_V)."""
    ranking_metric = _parse_react_ranking(ranking_metric)
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

    # x handling (Av only; chem grid carries the A_V axis).
    av_floor = 1e-5
    xv = np.where(av >= av_floor, av, np.nan) if xscale == 'log' else av.copy()
    pos = av[av > 0]
    if xscale == 'log' and pos.size:
        xr = [max(np.floor(np.log10(pos.min())), np.log10(av_floor)),
              np.ceil(np.log10(pos.max()))]
        xt = 'log'
    else:
        xr = [av_floor if xscale == 'log' else float(np.nanmin(av)),
              float(np.nanmax(av)) * 1.02]
        xt = 'linear'

    fig = go.Figure()
    for k, j in enumerate(order):
        y = matrix[:, j].astype(float)
        if yscale == 'log':
            y = np.where(y > 0, y, np.nan)   # rates can be negative-signed; show positive part on log
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
            x=xv, y=y, mode='lines',
            line=dict(color=COLORS[k % len(COLORS)], width=1.9,
                      dash=_DASH_CYCLE[(k // len(COLORS)) % len(_DASH_CYCLE)]),
            name=legend,
            hovertemplate=hover,
        ))
    _apply_layout(fig, title, 'A<sub>V</sub> (mag)', xt, xr, _REACT_YLABEL, yscale, theme=theme)
    fig.update_layout(legend=dict(font=dict(size=10)))
    return fig, stats, order


def make_reaction_plots(values, species, xscale, yscale, top_n,
                        ranking_metric=DEFAULT_REACT_RANKING, theme='light'):
    """Return formation/destruction figures and contribution summary tables."""
    ranking_metric = _parse_react_ranking(ranking_metric)
    empty = html.Div()
    if not _chem or not species:
        p = placeholder_fig('Load a chemistry grid and pick a species', theme=theme)
        return p, empty, p, empty
    chem_path = chem_file(values)
    struct_path = current_file(values)
    model = get_model(struct_path) if struct_path else None
    fig_f, stats_f, order_f = fig_reactions(
        chem_path, species, 'formation', xscale, yscale, top_n, model=model,
        ranking_metric=ranking_metric, theme=theme)
    fig_d, stats_d, order_d = fig_reactions(
        chem_path, species, 'destruction', xscale, yscale, top_n, model=model,
        ranking_metric=ranking_metric, theme=theme)
    labels_f = (get_reaction_data(chem_path, species, 'formation') or {}).get('labels', [])
    labels_d = (get_reaction_data(chem_path, species, 'destruction') or {}).get('labels', [])
    tbl_f = _reaction_contribution_table(
        order_f, labels_f, stats_f, 'formation', ranking_metric)
    tbl_d = _reaction_contribution_table(
        order_d, labels_d, stats_d, 'destruction', ranking_metric)
    return fig_f, tbl_f, fig_d, tbl_d


# --- 2-D parameter-slice contour grids ----------------------------------------

def _middle_token(key):
    toks = _grid['axis_tokens'][key]
    return toks[len(toks) // 2]


def _param_def(key):
    return PARAM_DEFS[_PARAM_IDX[key]]


def _plane_plot_axes(plane):
    """Return (x_key, y_key, slice_key) with cosmic-ray rate always on the x-axis."""
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
    if quantity == 'tdust':
        return 'T<sub>dust</sub> (K)'
    if quantity == 'nh':
        return 'n<sub>H</sub> (cm<sup>-3</sup>)'
    if quantity.startswith('species:'):
        sp = quantity.split(':', 1)[1]
        return f'X<sub>{format_species_html(sp)}</sub>'
    return quantity


def get_grid_scalar(filepath, quantity):
    """Scalar for contour grids (species: integrated X; diagnostics: cloud edge)."""
    cache_key = (filepath, quantity)
    if cache_key in _scalar_cache:
        return _scalar_cache[cache_key]
    try:
        model = get_model(filepath)
        if quantity == 'tgas':
            val = float(np.asarray(model['tgas'], float)[0])
        elif quantity == 'tdust':
            val = float(np.asarray(model['tdust'], float)[0])
        elif quantity == 'nh':
            val = float(np.asarray(model['nH'], float)[0])
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


def _contour_colorbar(title):
    """Fixed colorbar geometry so every slice figure shares the same figsize."""
    return dict(
        title=dict(text=title, font=dict(size=11)),
        len=0.82,
        thickness=14,
        x=1.02,
        xpad=2,
        y=0.5,
        yanchor='middle',
        outlinewidth=0,
    )


def _apply_square_contour_layout(fig, x_plot, y_plot, xdef, ydef, title,
                                 panel_w=DEFAULT_CONTOUR_PANEL_W,
                                 fixed_size=False, theme='light'):
    """Layout for a single contour panel.

    ``fixed_size=True`` — identical figsize for every panel in a side-by-side row
    (same width/height, axes domain, and 1:1 log-decade scaling like KoSens triple plots).
    """
    t = _theme_colors(theme)
    title_kw = dict(text=title, font=dict(size=12 if fixed_size else 13, color=t['title']),
                    x=0.02, xanchor='left')
    axis_common = dict(
        type='linear', showgrid=True, gridcolor=t['grid'],
        linecolor=t['axis_line'], tickfont=dict(color=t['font']),
    )
    if fixed_size:
        fig.update_layout(
            title=title_kw,
            paper_bgcolor=t['paper_bg'],
            plot_bgcolor=t['plot_bg'],
            width=COMPACT_FIG_WIDTH,
            height=COMPACT_FIG_HEIGHT,
            autosize=False,
            margin=_COMPACT_FIG_MARGIN,
            font=dict(family='Arial, sans-serif', size=12, color=t['font']),
            xaxis=dict(
                title=dict(text=_axis_label_contour(xdef), font=dict(size=11, color=t['font'])),
                domain=_COMPACT_XDOMAIN,
                constrain='domain',
                **axis_common,
            ),
            yaxis=dict(
                title=dict(text=_axis_label_contour(ydef), font=dict(size=11, color=t['font'])),
                domain=_COMPACT_YDOMAIN,
                scaleanchor='x', scaleratio=1,
                constrain='domain',
                **axis_common,
            ),
        )
        return fig

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
        margin=dict(l=60, r=80, t=52, b=54),
        font=dict(family='Arial, sans-serif', size=12, color=t['font']),
        xaxis=dict(
            title=dict(text=_axis_label_contour(xdef), font=dict(size=12, color=t['font'])),
            constrain='domain',
            **axis_common,
        ),
        yaxis=dict(
            title=dict(text=_axis_label_contour(ydef), font=dict(size=12, color=t['font'])),
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


def _multi_panel_layout_kw(theme='light', height=420):
    t = _theme_colors(theme)
    return dict(
        paper_bgcolor=t['paper_bg'],
        plot_bgcolor=t['plot_bg'],
        autosize=True,
        width=None,
        height=height,
        margin=dict(l=58, r=88, t=64, b=50),
        font=dict(family='Arial, sans-serif', size=12, color=t['font']),
    )


def _subplot_axis_kw(theme='light'):
    t = _theme_colors(theme)
    return dict(showgrid=True, gridcolor=t['grid'], zeroline=False,
                linecolor=t['axis_line'], tickfont=dict(color=t['font']))


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

    x_phys = np.array([xdef['decode'](t) for t in x_tokens], dtype=float)
    y_phys = np.array([ydef['decode'](t) for t in y_tokens], dtype=float)
    x_phys, y_phys, Z = gi.align_grid_axes_ascending(
        x_phys, y_phys, Z, x_logscale=xdef['logscale'], y_logscale=ydef['logscale'],
    )
    return x_phys, y_phys, Z, xdef, ydef


def _native_intensity_grid(plane, slice_token, species, idef, transition_idx):
    """Native (unresampled) 2-D SIMLINE intensity grid for one slice plane."""
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
            Z[iy, ix] = get_smli_intensity(
                tokens, species, idef, transition_idx)

    x_phys = np.array([xdef['decode'](t) for t in x_tokens], dtype=float)
    y_phys = np.array([ydef['decode'](t) for t in y_tokens], dtype=float)
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
        z0 = np.where(orig > 0, np.log10(orig), np.nan)
        z1 = np.where(interp > 0, np.log10(interp), np.nan)
        flux_cbar = f'log<sub>10</sub>({unit_label})'
    else:
        z0, z1 = orig.astype(float), interp.astype(float)
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

    mean_e = stats.get('mean_error', np.nan)
    max_e = stats.get('max_error', np.nan)
    med_e = stats.get('median_error', np.nan)
    stats_note = f'mean={mean_e:.3g}  max={max_e:.3g}  median={med_e:.3g}'

    xlab, ylab = _axis_label_contour(xdef), _axis_label_contour(ydef)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Original (native grid)', 'Interpolated (resampled)', err_title],
        horizontal_spacing=0.07,
        column_widths=[1, 1, 1],
    )
    fig.add_trace(go.Contour(
        x=x_plot_n, y=y_plot_n, z=z0, colorscale=cmap,
        colorbar=dict(title=dict(text=flux_cbar, font=dict(size=10, color=t['font'])),
                      len=0.88, thickness=12, x=0.28, xref='paper'),
        **contour_kw), row=1, col=1)
    fig.add_trace(go.Contour(
        x=x_plot_f, y=y_plot_f, z=z1, colorscale=cmap,
        colorbar=dict(title=dict(text=flux_cbar, font=dict(size=10, color=t['font'])),
                      len=0.88, thickness=12, x=0.635, xref='paper'),
        **contour_kw), row=1, col=2)
    fig.add_trace(go.Contour(
        x=x_plot_n, y=y_plot_n, z=err, colorscale=ERROR_PANEL_CMAP,
        colorbar=dict(title=dict(text=err_cbar, font=dict(size=10, color=t['font'])),
                      len=0.88, thickness=12, x=1.01, xref='paper'),
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
                   font=dict(size=13, color=t['title']), x=0.01, xanchor='left'),
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
                          colorscale=DEFAULT_GRID_COLORMAP, theme='light'):
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
        if zscale == 'log':
            p0 = np.where(Z_ref > 0, np.log10(Z_ref), np.nan)
            p1 = np.where(Z_atten > 0, np.log10(Z_atten), np.nan)
        else:
            p0, p1 = Z_ref.astype(float), Z_atten.astype(float)
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
        horizontal_spacing=0.07,
        column_widths=[1, 1, 1],
    )
    shift_trace_kw = dict(
        zmin=shift_vmin, zmax=shift_vmax,
        **({} if scan_dir == 'rightward' else {'zmid': 0}),
    )
    fig.add_trace(go.Contour(
        x=x_plot, y=y_plot, z=p0, colorscale=cmap,
        colorbar=dict(title=dict(text=cbar_ref, font=dict(size=10, color=t['font'])),
                      len=0.88, thickness=12, x=0.30, xref='paper'),
        **contour_kw), row=1, col=1)
    fig.add_trace(go.Contour(
        x=x_plot, y=y_plot, z=p1, colorscale=cmap,
        colorbar=dict(title=dict(text=cbar_ref, font=dict(size=10, color=t['font'])),
                      len=0.88, thickness=12, x=0.635, xref='paper'),
        **contour_kw), row=1, col=2)
    fig.add_trace(go.Contour(
        x=x_plot, y=y_plot, z=shift, colorscale=shift_cmap,
        colorbar=dict(title=dict(text=_shift_panel_colorbar_title(xdef),
                                 font=dict(size=10, color=t['font'])),
                      len=0.88, thickness=12, x=1.01, xref='paper'),
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
        font=dict(family='Arial, sans-serif', size=11, color=t['font']),
        title=dict(text=slice_title, font=dict(size=13, color=t['title']),
                   x=0.01, xanchor='left'),
    )
    fig.update_layout(**layout_kw)
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

    x_phys = np.array([xdef['decode'](t) for t in x_tokens], dtype=float)
    y_phys = np.array([ydef['decode'](t) for t in y_tokens], dtype=float)

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

    x_phys = np.array([xdef['decode'](t) for t in x_tokens], dtype=float)
    y_phys = np.array([ydef['decode'](t) for t in y_tokens], dtype=float)

    x_phys, y_phys, Z = gi.align_grid_axes_ascending(
        x_phys, y_phys, Z,
        x_logscale=xdef['logscale'],
        y_logscale=ydef['logscale'],
    )

    if interp_config:
        return resample_slice_grid(x_phys, y_phys, Z, xdef, ydef, **interp_config)
    return x_phys, y_phys, Z, xdef, ydef


def fig_contour_plane(plane, slice_idx, quantity, zscale,
                      shift_rtol=X_SHIFT_MATCH_RTOL,
                      shift_scan_direction=X_SHIFT_SCAN_DIRECTION,
                      interp_config=None,
                      colorscale=DEFAULT_GRID_COLORMAP, theme='light'):
    """Contour plot for one (x, y) slice plane with the third axis on a slider."""
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
    x_phys, y_phys, Z, xdef, ydef = build_slice_grid(
        plane, slice_token, quantity, interp_config=interp_config)

    if not np.any(np.isfinite(Z)):
        return placeholder_fig('No model points for this slice', theme=theme)

    if sdef['key'] == 'atten':
        slice_disp = f'{slice_token:02d}'
    else:
        slice_disp = f'{sdef["decode"](slice_token):.4g}'
    unit = f' {sdef["unit"]}' if sdef['unit'] else ''
    slice_title = (f'{plane["title"]}<br>'
                   f'<sup style="font-size:11px">{sdef["name"]} = {slice_disp}{unit}</sup>')

    qlabel = _quantity_label(quantity)
    if zscale == 'log':
        cbar_title = f'log<sub>10</sub>({qlabel})'
    else:
        cbar_title = qlabel

    cmap = _parse_grid_colorscale(colorscale)
    if _overlay:
        _, _, Z_ov, _, _ = build_overlay_slice_grid(
            plane, slice_token, quantity, interp_config=interp_config)
        if np.any(np.isfinite(Z_ov)):
            return fig_triple_atten_grid(
                plane, slice_title, Z, Z_ov, x_phys, y_phys, xdef, ydef, zscale, cbar_title,
                shift_rtol=shift_rtol, shift_scan_direction=shift_scan_direction,
                colorscale=cmap, theme=theme)

    Zplot = Z.astype(float)
    if zscale == 'log':
        Zplot = np.where(Zplot > 0, np.log10(Zplot), np.nan)

    x_plot = _axis_plot_coords(x_phys, xdef)
    y_plot = _axis_plot_coords(y_phys, ydef)

    trace_kw = _contour_trace_kw(theme, show_lines=True)
    fig = go.Figure(go.Contour(
        x=x_plot, y=y_plot, z=Zplot,
        colorscale=cmap,
        colorbar=_contour_colorbar(cbar_title),
        **trace_kw,
    ))
    return _apply_square_contour_layout(
        fig, x_plot, y_plot, xdef, ydef, slice_title,
        fixed_size=True, theme=theme,
    )


def make_contour_plots(quantity, zscale, slice_indices,
                       shift_rtol=X_SHIFT_MATCH_RTOL,
                       shift_scan_direction=X_SHIFT_SCAN_DIRECTION,
                       interp_config=None,
                       colorscale=DEFAULT_GRID_COLORMAP, theme='light'):
    """Return one contour figure per entry in SLICE_PLANES."""
    return tuple(
        fig_contour_plane(plane, slice_indices[i], quantity, zscale,
                          shift_rtol=shift_rtol, shift_scan_direction=shift_scan_direction,
                          interp_config=interp_config,
                          colorscale=colorscale, theme=theme)
        for i, plane in enumerate(SLICE_PLANES)
    )


def build_intensity_slice_grid(plane, slice_token, species, idef, transition_idx,
                               overlay=False, interp_config=None):
    """Build a 2-D SIMLINE intensity array for one parameter plane."""
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
            Z[iy, ix] = get_smli_intensity(
                tokens, species, idef, transition_idx, overlay=overlay)

    x_phys = np.array([xdef['decode'](t) for t in x_tokens], dtype=float)
    y_phys = np.array([ydef['decode'](t) for t in y_tokens], dtype=float)

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
                                colorscale=DEFAULT_GRID_COLORMAP, theme='light'):
    """Contour plot of SIMLINE intensity on one parameter slice plane."""
    if not _grid:
        return placeholder_fig('Load a main grid directory', theme=theme)
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
        plane, slice_token, species, idef, transition_idx, interp_config=interp_config)

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

    if sdef['key'] == 'atten':
        slice_disp = f'{slice_token:02d}'
    else:
        slice_disp = f'{sdef["decode"](slice_token):.4g}'
    slice_unit = f' {sdef["unit"]}' if sdef['unit'] else ''
    slice_title = (f'{plane["title"]}<br>'
                   f'<sup style="font-size:11px">{sdef["name"]} = {slice_disp}{slice_unit}'
                   f'  \u2014  {sp_html} {tlabel}</sup>')

    cmap = _parse_grid_colorscale(colorscale)
    if _simline_overlay:
        _, _, Z_ov, _, _ = build_intensity_slice_grid(
            plane, slice_token, species, idef, transition_idx, overlay=True,
            interp_config=interp_config)
        if np.any(np.isfinite(Z_ov)):
            return fig_triple_atten_grid(
                plane, slice_title, Z, Z_ov, x_phys, y_phys, xdef, ydef, zscale, cbar_title,
                shift_rtol=shift_rtol, shift_scan_direction=shift_scan_direction,
                colorscale=cmap, theme=theme)

    Zplot = Z.astype(float)
    if zscale == 'log':
        Zplot = np.where(Zplot > 0, np.log10(Zplot), np.nan)

    x_plot = _axis_plot_coords(x_phys, xdef)
    y_plot = _axis_plot_coords(y_phys, ydef)

    trace_kw = _contour_trace_kw(theme, show_lines=True)
    fig = go.Figure(go.Contour(
        x=x_plot, y=y_plot, z=Zplot,
        colorscale=cmap,
        colorbar=_contour_colorbar(cbar_title),
        **trace_kw,
    ))
    return _apply_square_contour_layout(
        fig, x_plot, y_plot, xdef, ydef, slice_title,
        fixed_size=True, theme=theme,
    )


def make_intensity_contour_plots(species, idef, transition_idx, zscale, slice_indices,
                                 shift_rtol=X_SHIFT_MATCH_RTOL,
                                 shift_scan_direction=X_SHIFT_SCAN_DIRECTION,
                                 interp_config=None,
                                 colorscale=DEFAULT_GRID_COLORMAP, theme='light'):
    """Return one SIMLINE intensity contour per SLICE_PLANES entry."""
    try:
        tidx = int(transition_idx)
    except (TypeError, ValueError):
        tidx = 0
    return tuple(
        fig_intensity_contour_plane(
            plane, slice_indices[i], species, idef or SIMLINE_DEFAULT_IDEF, tidx, zscale,
            shift_rtol=shift_rtol, shift_scan_direction=shift_scan_direction,
            interp_config=interp_config,
            colorscale=colorscale, theme=theme)
        for i, plane in enumerate(SLICE_PLANES)
    )


def fig_intensity_spectrum(values, species, idef, theme='light'):
    """All SIMLINE line intensities for the selected model point."""
    if not _grid:
        return placeholder_fig('Load a main grid directory', theme=theme)
    if not _simline:
        return placeholder_fig('Load a SIMLINE directory on the Load tab', theme=theme)
    if not species:
        return placeholder_fig('Select a species', theme=theme)

    tokens = _tokens_from_values(values)
    if tokens is None:
        return placeholder_fig('No model file for this parameter combination', theme=theme)

    path = smli_file(tokens, species, idef or SIMLINE_DEFAULT_IDEF)
    if not path:
        return placeholder_fig('No .smli file for this model point / species', theme=theme)

    rows = read_smli_file(path)
    if not rows:
        return placeholder_fig('Empty .smli file', theme=theme)

    freqs = np.array([r['frequency'] for r in rows], dtype=float)
    ints = np.array([r['intensity'] for r in rows], dtype=float)
    labels = [_smli_transition_label(r['transition']) for r in rows]
    idef = idef or SIMLINE_DEFAULT_IDEF
    unit = _intensity_unit_label(idef)
    ylab = _quantity_axis_label(idef)
    sp_html = format_species_html(species)
    t = _theme_colors(theme)
    qty_word = 'optical depths' if idef == 'tau' else 'intensities'
    y_hover = '\u03c4 = %{y:.4g}' if idef == 'tau' else 'I = %{y:.4g}'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=ints, mode='lines+markers',
        marker=dict(size=5, color='#1f77b4'),
        line=dict(color='#1f77b4', width=1.5),
        text=labels,
        hovertemplate=f'%{{text}}<br>\u03bd = %{{x:.4g}} GHz<br>{y_hover}<extra></extra>',
    ))
    fig.update_layout(
        **_base_layout(theme),
        title=dict(
            text=f'{sp_html} line {qty_word}  [{unit}]',
            font=dict(size=13, color=t['title']), x=0.02, xanchor='left'),
        xaxis=dict(**_axis_style(theme), title=dict(text='Frequency (GHz)', font=dict(size=12)),
                   type='linear'),
        yaxis=dict(**_axis_style(theme), title=dict(text=ylab, font=dict(size=12)),
                   type='log' if idef != 'tau' and np.all(ints[ints > 0] > 0) else 'linear'),
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
                return placeholder_fig('Load a main grid directory', theme=theme), None, None
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
        'title': dict(text=title, font=dict(size=13, color=t['title']),
                      x=0.02, xanchor='left'),
        'xaxis': dict(**_axis_style(theme),
                      title=dict(text='Velocity (km/s)', font=dict(size=12)),
                      type='linear'),
        'yaxis': dict(**_axis_style(theme),
                      title=dict(text=_pv_spectrum_ylabel(pv_quantity, bunit),
                                 font=dict(size=12)),
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
        return placeholder_fig('Load a main grid directory', theme=theme)
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
    if zscale == 'log' and pv_quantity != 'tau':
        zplot = np.where(zplot > 0, zplot, np.nan)

    if pv_quantity == 'tau':
        z_label = '\u03c4'
        z_hover = '\u03c4 = %{z:.4g}'
    else:
        bunit = ss.brightness_unit_from_header(header)
        z_label = f'T<sub>mb</sub> ({bunit})'
        z_hover = 'T<sub>mb</sub> = %{z:.4g}'
    tr_label = ss.transition_from_header(header) or _smli_transition_label(transition)
    sp_html = format_species_html(species)
    t = _theme_colors(theme)

    fig = go.Figure(go.Heatmap(
        x=positions, y=velocities, z=zplot,
        colorscale=colorscale or 'Inferno',
        colorbar=dict(title=dict(text=z_label)),
        hovertemplate=(
            'offset = %{x:.3f}"<br>v = %{y:.3g} km/s'
            f'<br>{z_hover}<extra></extra>'
        ),
    ))
    layout_kw = {
        **_base_layout(theme),
        'height': 520,
        'title': dict(
            text=f'PV diagram — {sp_html} {tr_label}',
            font=dict(size=13, color=t['title']), x=0.02, xanchor='left'),
        'xaxis': dict(**_axis_style(theme),
                      title=dict(text='Position offset (arcsec)', font=dict(size=12)),
                      type='linear'),
        'yaxis': dict(**_axis_style(theme),
                      title=dict(text='Velocity (km/s)', font=dict(size=12)),
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
    dens = PARAM_DEFS[_PARAM_IDX['density']]['decode'](tokens[_PARAM_IDX['density']])
    fuv = PARAM_DEFS[_PARAM_IDX['fuv']]['decode'](tokens[_PARAM_IDX['fuv']])
    zeta = PARAM_DEFS[_PARAM_IDX['crir']]['decode'](tokens[_PARAM_IDX['crir']])
    parts = [
        f'n<sub>H</sub> = {_sci_label(dens)} cm\u207b\u00b3',
        f'\u03c7 = {_sci_label(fuv)}',
        f'\u03b6 = {_sci_label(zeta)} s\u207b\u00b9',
    ]
    if include_atten:
        parts.append(f'atten {tokens[_PARAM_IDX["atten"]]:02d}')
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
            font=dict(size=14, color=t['title']), x=0.02, xanchor='left'),
        'xaxis': dict(**_axis_style(theme),
                      title=dict(text=x_title, font=dict(size=13)),
                      type=x_type, range=x_range),
        'yaxis': dict(**_axis_style(theme),
                      title=dict(text='ζ<sub>H₂</sub> (s⁻¹)', font=dict(size=13)),
                      type='log',
                      range=y_range),
        'showlegend': True,
        'legend': dict(font=dict(size=11)),
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


def make_profile_plots(values, xvar, xscale, yscale, custom_species, theme='light'):
    model, overlay = _load_model_pair(values)
    if model is None:
        bad = placeholder_fig(theme=theme) if not _grid else placeholder_fig(
            'No model file for this parameter combination', theme=theme)
        return (bad,) * 4
    x_cross = find_h_h2_transition(model, xvar)
    figs = [
        fig_tgas(model, overlay, xvar, xscale, yscale, theme=theme),
        fig_h_h2(model, overlay, xvar, xscale, yscale, theme=theme),
        fig_cplus_c_co(model, overlay, xvar, xscale, yscale, theme=theme),
        fig_custom(model, overlay, xvar, xscale, yscale, custom_species, theme=theme),
    ]
    for f in figs:
        add_h_h2_vline(f, x_cross, theme=theme)
    return tuple(figs)


def make_thermal_plots(values, xvar, xscale, yscale, theme='light'):
    model, overlay = _load_model_pair(values)
    if model is None:
        bad = placeholder_fig(theme=theme) if not _grid else placeholder_fig(
            'No model file for this parameter combination', theme=theme)
        return (bad,) * 3
    return (
        fig_thermal(model, overlay, xvar, xscale, yscale, theme=theme),
        fig_heat_breakdown(model, overlay, xvar, xscale, yscale, theme=theme),
        fig_cool_breakdown(model, overlay, xvar, xscale, yscale, theme=theme),
    )


# --- App layout ---------------------------------------------------------------

app = dash.Dash(
    __name__,
    title='KOSMA-\u03C4 Grid Explorer',
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
)
server = app.server

_RADIO = dict(labelStyle={'display': 'block', 'marginBottom': '3px', 'fontSize': '13px'})

_SCALE_OPTIONS = [{'label': ' log', 'value': 'log'},
                  {'label': ' linear', 'value': 'linear'}]

_XVAR_OPTIONS = [{'label': ' A\u1D65 (mag)', 'value': 'Av'},
                 {'label': ' n_H (cm\u207B\u00B3)', 'value': 'nH'}]

_CTRL_LABEL = {'fontWeight': '600', 'fontSize': '13px', 'display': 'block', 'marginBottom': '5px'}
_CTRL_BOX = {'flex': '1', 'minWidth': '110px', 'marginRight': '18px'}
_GRAPH_CFG = {'toImageButtonOptions': {'format': 'png', 'scale': 2}}
_SLICE_GRAPH_CFG = {**_GRAPH_CFG, 'responsive': False}
_TAB_STYLE = {'padding': '10px 18px', 'fontWeight': '600', 'fontSize': '13px'}
_TAB_SEL = {'borderTop': '3px solid #1f77b4', 'padding': '10px 18px',
            'fontWeight': '700', 'fontSize': '13px', 'backgroundColor': '#f5f7ff'}
_PAGE_INTRO = {'margin': '0 0 12px', 'color': '#666', 'fontSize': '13px'}


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
                style={'fontSize': '13px'}),
        ], style={'flex': '2', 'minWidth': '240px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Intensity match tolerance (rtol)', style=_CTRL_LABEL),
            dcc.Input(id=rtol_id, type='number', value=X_SHIFT_MATCH_RTOL,
                      min=0, max=1, step=0.005,
                      style={'width': '100px', 'padding': '7px 9px', 'fontSize': '13px',
                             'border': '1px solid #bbc', 'borderRadius': '6px'}),
            html.Span('  relative band around target intensity',
                      style={'fontSize': '11px', 'color': '#888', 'marginLeft': '8px'}),
        ], style={'flex': '1.2', 'minWidth': '200px'}),
    ], style={'display': 'flex', 'alignItems': 'flex-end',
              'padding': '12px 18px', 'backgroundColor': '#faf6f4',
              'borderRadius': '8px', 'marginBottom': '12px',
              'border': '1px dashed #c9a99b'})


def _interp_control_row(prefix=''):
    """KoSens-style grid interpolation controls (``3d_grids.ipynb``)."""
    ny_id = f'{prefix}interp-ny'
    nx_id = f'{prefix}interp-nx'
    xlim_id = f'{prefix}interp-x-lim'
    ylim_id = f'{prefix}interp-y-lim'
    method_id = f'{prefix}interp-method'
    clip_id = f'{prefix}interp-clip'
    input_style = {'width': '72px', 'padding': '7px 9px', 'fontSize': '13px',
                   'border': '1px solid #bbc', 'borderRadius': '6px'}
    return html.Div([
        html.Div([
            html.Label('Interpolated grid size (ny \u00d7 nx)', style=_CTRL_LABEL),
            html.Div([
                dcc.Input(id=ny_id, type='number', value=DEFAULT_INTERP_NY,
                          min=2, max=500, step=1, style=input_style),
                html.Span(' \u00d7 ', style={'margin': '0 6px', 'color': '#666'}),
                dcc.Input(id=nx_id, type='number', value=DEFAULT_INTERP_NX,
                          min=2, max=500, step=1, style=input_style),
            ]),
        ], style={'flex': '1.1', 'minWidth': '170px', 'marginRight': '18px'}),
        html.Div([
            html.Label('X-axis log\u2081\u2080 limit (upper)', style=_CTRL_LABEL),
            dcc.Input(id=xlim_id, type='number', value=None, placeholder='no limit',
                      style={**input_style, 'width': '100px'}),
            html.Span('  keep points with log\u2081\u2080(x) < limit',
                      style={'fontSize': '11px', 'color': '#888', 'marginLeft': '6px'}),
        ], style={'flex': '1.4', 'minWidth': '220px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Y-axis log\u2081\u2080 limit (upper)', style=_CTRL_LABEL),
            dcc.Input(id=ylim_id, type='number', value=None, placeholder='no limit',
                      style={**input_style, 'width': '100px'}),
            html.Span('  keep points with log\u2081\u2080(y) < limit',
                      style={'fontSize': '11px', 'color': '#888', 'marginLeft': '6px'}),
        ], style={'flex': '1.4', 'minWidth': '220px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Interpolation method', style=_CTRL_LABEL),
            dcc.Dropdown(id=method_id, options=INTERP_METHOD_OPTIONS,
                         value=DEFAULT_INTERP_METHOD, clearable=False,
                         style={'fontSize': '13px'}),
        ], style={'flex': '1', 'minWidth': '130px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Clip to bounds', style=_CTRL_LABEL),
            dcc.Checklist(id=clip_id, options=[{'label': ' clip overshoot to [0, max]',
                                                'value': 'clip'}],
                          value=[], style={'fontSize': '13px'}),
        ], style={'flex': '1', 'minWidth': '160px'}),
    ], style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
              'padding': '12px 18px', 'backgroundColor': '#f4f8f4',
              'borderRadius': '8px', 'marginBottom': '12px',
              'border': '1px dashed #9cb89c'})


def _interp_error_control_row():
    """Decimation / error-metric controls for the interpolation-error tab."""
    input_style = {'width': '72px', 'padding': '7px 9px', 'fontSize': '13px',
                   'border': '1px solid #bbc', 'borderRadius': '6px'}
    return html.Div([
        html.Div([
            html.Label('Error decimation factor', style=_CTRL_LABEL),
            dcc.Input(id='ie-decimation', type='number', value=DEFAULT_ERROR_DECIMATION,
                      min=2, max=8, step=1, style=input_style),
            html.Span('  checkerboard decimation (KoSens default: 2)',
                      style={'fontSize': '11px', 'color': '#888', 'marginLeft': '8px'}),
        ], style={'flex': '1.3', 'minWidth': '220px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Error metric', style=_CTRL_LABEL),
            dcc.RadioItems(id='ie-error-metric', options=ERROR_METRIC_OPTIONS,
                           value=ERROR_METRIC_OPTIONS[0]['value'], **_RADIO),
        ], style={'flex': '1.2', 'minWidth': '200px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Relative threshold', style=_CTRL_LABEL),
            dcc.Input(id='ie-rel-threshold', type='number',
                      value=DEFAULT_ERROR_REL_THRESHOLD, min=0, max=1, step=0.01,
                      style={**input_style, 'width': '90px'}),
            html.Span('  mask relative error below this fraction of max',
                      style={'fontSize': '11px', 'color': '#888', 'marginLeft': '8px'}),
        ], style={'flex': '1.5', 'minWidth': '260px', 'marginRight': '18px'}),
        html.Div([
            html.Label('Flux / abundance scale', style=_CTRL_LABEL),
            dcc.RadioItems(id='ie-flux-scale', options=_SCALE_OPTIONS,
                           value='log', **_RADIO),
        ], style={**_CTRL_BOX, 'marginRight': '18px'}),
        html.Div([
            html.Label('Contour lines', style=_CTRL_LABEL),
            dcc.Checklist(id='ie-plot-contours',
                          options=[{'label': ' white contours on original & interpolated',
                                    'value': 'contours'}],
                          value=['contours'], style={'fontSize': '13px'}),
        ], style={'flex': '1.4', 'minWidth': '220px'}),
    ], style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
              'padding': '12px 18px', 'backgroundColor': '#faf4f8',
              'borderRadius': '8px', 'marginBottom': '12px',
              'border': '1px dashed #c9a0c0'})


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


def _slice_panel(plane):
    """One 2-D contour panel with its own third-axis slider."""
    pid = plane['id']
    sdef = _param_def(plane['slice'])
    unit = f'  ({sdef["unit"]})' if sdef['unit'] else ''
    return html.Div([
        html.Div([
            html.Label(f'Fixed: {sdef["name"]}{unit}', style={**_CTRL_LABEL, 'fontSize': '12px'}),
            dcc.Slider(id=f'slice-slider-{pid}', min=0, max=1, step=1, value=0, marks={},
                       tooltip={'placement': 'top', 'always_visible': False}),
            html.Div(id=f'slice-label-{pid}',
                     style={'textAlign': 'center', 'color': sdef['color'],
                            'fontSize': '11px', 'marginTop': '2px', 'fontWeight': '600'}),
        ], style={'padding': '0 4px 8px'}),
        dcc.Graph(id=f'plot-contour-{pid}', figure=placeholder_fig(),
                  config=_SLICE_GRAPH_CFG,
                  style=_SLICE_GRAPH_STYLE),
    ], id=f'slice-panel-{pid}', style=_SLICE_PANEL_ROW)


def _int_slice_panel(plane):
    """One SIMLINE intensity contour panel with its own third-axis slider."""
    pid = plane['id']
    sdef = _param_def(plane['slice'])
    unit = f'  ({sdef["unit"]})' if sdef['unit'] else ''
    return html.Div([
        html.Div([
            html.Label(f'Fixed: {sdef["name"]}{unit}', style={**_CTRL_LABEL, 'fontSize': '12px'}),
            dcc.Slider(id=f'int-slice-slider-{pid}', min=0, max=1, step=1, value=0, marks={},
                       tooltip={'placement': 'top', 'always_visible': False}),
            html.Div(id=f'int-slice-label-{pid}',
                     style={'textAlign': 'center', 'color': sdef['color'],
                            'fontSize': '11px', 'marginTop': '2px', 'fontWeight': '600'}),
        ], style={'padding': '0 4px 8px'}),
        dcc.Graph(id=f'plot-int-contour-{pid}', figure=placeholder_fig(),
                  config=_SLICE_GRAPH_CFG,
                  style=_SLICE_GRAPH_STYLE),
    ], id=f'int-slice-panel-{pid}', style=_SLICE_PANEL_ROW)


def _ie_plane_section(plane):
    """One slice plane on the interpolation-error tab (abundance + intensity rows)."""
    pid = plane['id']
    sdef = _param_def(plane['slice'])
    unit = f'  ({sdef["unit"]})' if sdef['unit'] else ''
    slider_block = html.Div([
        html.Label(f'Fixed: {sdef["name"]}{unit}', style={**_CTRL_LABEL, 'fontSize': '12px'}),
        dcc.Slider(id=f'ie-slice-slider-{pid}', min=0, max=1, step=1, value=0, marks={},
                   tooltip={'placement': 'top', 'always_visible': False}),
        html.Div(id=f'ie-slice-label-{pid}',
                 style={'textAlign': 'center', 'color': sdef['color'],
                        'fontSize': '11px', 'marginTop': '2px', 'fontWeight': '600'}),
    ], style={'padding': '0 4px 8px', 'maxWidth': f'{COMPACT_FIG_WIDTH}px'})
    return html.Div([
        html.H4(_IE_PLANE_LABELS.get(pid, pid), style={'fontSize': '14px', 'margin': '8px 0 4px',
                                                        'color': '#333', 'fontWeight': '600'}),
        slider_block,
        html.Div(id=f'ie-abund-wrap-{pid}', children=[
            dcc.Graph(id=f'plot-ie-abund-{pid}', figure=placeholder_fig(),
                      config=_SLICE_GRAPH_CFG,
                      style={'width': '100%', 'height': 'auto', 'marginBottom': '8px'}),
        ]),
        html.Div(id=f'ie-int-wrap-{pid}', children=[
            dcc.Graph(id=f'plot-ie-int-{pid}', figure=placeholder_fig(),
                      config=_SLICE_GRAPH_CFG,
                      style={'width': '100%', 'height': 'auto'}),
        ]),
    ], style={'marginBottom': '20px', 'paddingBottom': '12px',
              'borderBottom': '1px solid #eee'})


app.layout = html.Div(
    id='app-root',
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1460px',
           'margin': '0 auto', 'padding': '14px 22px', 'backgroundColor': '#fff'},
    children=[

    # Header
    html.Div([
        html.H1('KOSMA-\u03C4 Grid Explorer',
                style={'margin': '0 0 4px', 'color': '#1a1a2e',
                       'fontSize': '26px', 'fontWeight': '700'}),
        html.P('Interactive browser for KoSens3D photodissociation-region model grids '
               '(one HDF5 file per model point).',
               style={'margin': '0', 'color': '#666', 'fontSize': '13px'}),
    ], style={'borderBottom': '2px solid #1f77b4',
              'paddingBottom': '10px', 'marginBottom': '14px'}),

    dcc.Store(id='grid-loaded', data=False),

    # Shared profile controls (hidden until a grid is loaded; not shown on Load tab)
    html.Div(id='controls-wrap', style={'display': 'none'}, children=[
        html.Div([_slider_block(d) for d in range(N_PARAMS)],
                 id='controls-sliders-wrap',
                 style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-start',
                        'padding': '14px 18px', 'marginTop': '4px',
                        'backgroundColor': '#f0f4ff', 'borderRadius': '8px'}),

        html.Div([
            html.Div([
                html.Label('Plot theme', style=_CTRL_LABEL),
                dcc.RadioItems(id='plot-theme', options=PLOT_THEME_OPTIONS,
                               value=DEFAULT_PLOT_THEME, **_RADIO),
            ], style={**_CTRL_BOX, 'minWidth': '100px'}),
            html.Div([
                html.Label('Grid colormap', style=_CTRL_LABEL),
                dcc.Dropdown(id='grid-colorscale', options=GRID_COLORMAP_OPTIONS,
                             value=DEFAULT_GRID_COLORMAP, clearable=False,
                             style={'fontSize': '13px'}),
            ], style={'flex': '1.2', 'minWidth': '140px', 'marginRight': '18px'}),
            html.Div([
                html.Label('X-axis', style=_CTRL_LABEL),
                dcc.RadioItems(id='xvar-choice', options=_XVAR_OPTIONS, value='Av', **_RADIO),
            ], style=_CTRL_BOX),
            html.Div([
                html.Label('X scale', style=_CTRL_LABEL),
                dcc.RadioItems(id='xscale', options=_SCALE_OPTIONS, value='log', **_RADIO),
            ], style=_CTRL_BOX),
            html.Div([
                html.Label('Y scale', style=_CTRL_LABEL),
                dcc.RadioItems(id='yscale', options=_SCALE_OPTIONS, value='log', **_RADIO),
            ], style={**_CTRL_BOX, 'marginRight': '0'}),
        ], id='controls-axis-wrap',
           style={'display': 'flex', 'alignItems': 'flex-start',
                  'padding': '10px 18px', 'marginTop': '8px',
                  'backgroundColor': '#f0f4ff', 'borderRadius': '8px'}),

        html.Div(id='model-info', style={
            'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'alignItems': 'center',
            'backgroundColor': '#e8f4f8', 'padding': '7px 16px', 'borderRadius': '6px',
            'marginTop': '8px', 'marginBottom': '4px', 'fontSize': '13px', 'color': '#333'}),
    ]),

    dcc.Tabs(
        id='main-tabs',
        value='load',
        style={'marginTop': '10px'},
        colors={'border': '#dde', 'primary': '#1f77b4', 'background': '#fafbff'},
        children=[

        # --- Page 1: grid loaders -------------------------------------------
        dcc.Tab(label='Load grids', value='load', style=_TAB_STYLE, selected_style=_TAB_SEL,
                children=[
            html.Div(style={'paddingTop': '14px'}, children=[
                html.P('Load the main model grid and, optionally, an overlay grid '
                       '(e.g. attenuated models), a chemistry reaction-rate grid, '
                       'and a SIMLINE intensity directory.',
                       style=_PAGE_INTRO),

                html.Div([
                    html.Label('Grid directory', style=_CTRL_LABEL),
                    html.Div([
                        dcc.Input(id='dir-input', type='text', value=args.dir,
                                  placeholder='/path/to/pdrgrid_hdf5',
                                  style={'flex': '1', 'padding': '8px 10px', 'fontSize': '13px',
                                         'border': '1px solid #bbc', 'borderRadius': '6px',
                                         'marginRight': '10px'}),
                        dcc.Checklist(id='recursive-check',
                                      options=[{'label': ' recursive', 'value': 'rec'}],
                                      value=['rec'] if args.recursive else [],
                                      style={'fontSize': '13px', 'marginRight': '10px',
                                             'whiteSpace': 'nowrap', 'alignSelf': 'center'}),
                        html.Button('Load grid', id='btn-load', n_clicks=0,
                                    style={'padding': '8px 22px', 'backgroundColor': '#1f77b4',
                                           'color': 'white', 'border': 'none', 'borderRadius': '6px',
                                           'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': '600'}),
                    ], style={'display': 'flex', 'alignItems': 'stretch'}),
                ], style={'padding': '14px 18px', 'backgroundColor': '#f5f7ff',
                          'borderRadius': '8px', 'border': '1px dashed #9ab'}),

                html.Div(id='load-status',
                         style={'marginTop': '7px', 'fontSize': '13px', 'minHeight': '22px'}),

                html.Div([
                    html.Label('Overlay grid  (optional \u2014 e.g. attenuated; dashed lines)',
                               style=_CTRL_LABEL),
                    html.Div([
                        dcc.Input(id='overlay-dir-input', type='text', value='',
                                  placeholder='/path/to/attenuated_grid_hdf5',
                                  style={'flex': '1', 'padding': '8px 10px', 'fontSize': '13px',
                                         'border': '1px solid #cbb', 'borderRadius': '6px',
                                         'marginRight': '10px'}),
                        dcc.Checklist(id='overlay-recursive-check',
                                      options=[{'label': ' recursive', 'value': 'rec'}], value=[],
                                      style={'fontSize': '13px', 'marginRight': '10px',
                                             'whiteSpace': 'nowrap', 'alignSelf': 'center'}),
                        html.Button('Load overlay', id='btn-load-overlay', n_clicks=0,
                                    style={'padding': '8px 18px', 'backgroundColor': '#8c564b',
                                           'color': 'white', 'border': 'none', 'borderRadius': '6px',
                                           'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': '600',
                                           'marginRight': '8px'}),
                        html.Button('Clear', id='btn-clear-overlay', n_clicks=0,
                                    style={'padding': '8px 16px', 'backgroundColor': '#eee',
                                           'color': '#555', 'border': '1px solid #ccc',
                                           'borderRadius': '6px', 'cursor': 'pointer', 'fontSize': '13px'}),
                    ], style={'display': 'flex', 'alignItems': 'stretch'}),
                ], style={'padding': '12px 18px', 'marginTop': '10px',
                          'backgroundColor': '#faf6f4', 'borderRadius': '8px',
                          'border': '1px dashed #c9a99b'}),

                html.Div(id='overlay-status',
                         style={'marginTop': '6px', 'fontSize': '13px', 'minHeight': '20px'}),
                html.Div([
                    html.Label('Overlay SIMLINE directory  (optional \u2014 attenuated .smli; '
                               'triple-panel shift on Intensities tab)', style=_CTRL_LABEL),
                    html.Div([
                        dcc.Input(id='simline-overlay-dir-input', type='text', value='',
                                  placeholder='/path/to/attenuated/simlineoutput',
                                  style={'flex': '1', 'padding': '8px 10px', 'fontSize': '13px',
                                         'border': '1px solid #cbb', 'borderRadius': '6px',
                                         'marginRight': '10px'}),
                        dcc.Checklist(id='simline-overlay-recursive-check',
                                      options=[{'label': ' recursive', 'value': 'rec'}], value=[],
                                      style={'fontSize': '13px', 'marginRight': '10px',
                                             'whiteSpace': 'nowrap', 'alignSelf': 'center'}),
                        html.Button('Load overlay SIMLINE', id='btn-load-simline-overlay', n_clicks=0,
                                    style={'padding': '8px 18px', 'backgroundColor': '#8c564b',
                                           'color': 'white', 'border': 'none', 'borderRadius': '6px',
                                           'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': '600',
                                           'marginRight': '8px'}),
                        html.Button('Clear', id='btn-clear-simline-overlay', n_clicks=0,
                                    style={'padding': '8px 16px', 'backgroundColor': '#eee',
                                           'color': '#555', 'border': '1px solid #ccc',
                                           'borderRadius': '6px', 'cursor': 'pointer', 'fontSize': '13px'}),
                    ], style={'display': 'flex', 'alignItems': 'stretch'}),
                ], style={'padding': '10px 18px', 'marginTop': '8px',
                          'backgroundColor': '#faf6f4', 'borderRadius': '8px',
                          'border': '1px dashed #c9a99b'}),
                html.Div(id='simline-overlay-status',
                         style={'marginTop': '6px', 'fontSize': '13px', 'minHeight': '20px'}),
                dcc.Store(id='overlay-state', data=0),
                dcc.Store(id='simline-overlay-state', data=0),

                html.Div([
                    html.Label('Chemistry grid  (optional \u2014 reaction rates; enables the '
                               'Chemistry tab)', style=_CTRL_LABEL),
                    html.Div([
                        dcc.Input(id='chem-dir-input', type='text', value='',
                                  placeholder='/path/to/chemistrygrid',
                                  style={'flex': '1', 'padding': '8px 10px', 'fontSize': '13px',
                                         'border': '1px solid #b9c5b0', 'borderRadius': '6px',
                                         'marginRight': '10px'}),
                        dcc.Checklist(id='chem-recursive-check',
                                      options=[{'label': ' recursive', 'value': 'rec'}], value=[],
                                      style={'fontSize': '13px', 'marginRight': '10px',
                                             'whiteSpace': 'nowrap', 'alignSelf': 'center'}),
                        html.Button('Load chemistry', id='btn-load-chem', n_clicks=0,
                                    style={'padding': '8px 18px', 'backgroundColor': '#2ca02c',
                                           'color': 'white', 'border': 'none', 'borderRadius': '6px',
                                           'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': '600',
                                           'marginRight': '8px'}),
                        html.Button('Clear', id='btn-clear-chem', n_clicks=0,
                                    style={'padding': '8px 16px', 'backgroundColor': '#eee',
                                           'color': '#555', 'border': '1px solid #ccc',
                                           'borderRadius': '6px', 'cursor': 'pointer', 'fontSize': '13px'}),
                    ], style={'display': 'flex', 'alignItems': 'stretch'}),
                ], style={'padding': '12px 18px', 'marginTop': '10px',
                          'backgroundColor': '#f3f9f1', 'borderRadius': '8px',
                          'border': '1px dashed #a9c79b'}),

                html.Div(id='chem-status',
                         style={'marginTop': '6px', 'fontSize': '13px', 'minHeight': '20px'}),
                dcc.Store(id='chem-state', data=0),

                html.Div([
                    html.Label('SIMLINE directory  (optional \u2014 line intensities; '
                               'enables the Intensities tab)', style=_CTRL_LABEL),
                    html.Div([
                        dcc.Input(id='simline-dir-input', type='text', value='',
                                  placeholder='/path/to/simlineoutput',
                                  style={'flex': '1', 'padding': '8px 10px', 'fontSize': '13px',
                                         'border': '1px solid #b8c4d8', 'borderRadius': '6px',
                                         'marginRight': '10px'}),
                        dcc.Checklist(id='simline-recursive-check',
                                      options=[{'label': ' recursive', 'value': 'rec'}], value=[],
                                      style={'fontSize': '13px', 'marginRight': '10px',
                                             'whiteSpace': 'nowrap', 'alignSelf': 'center'}),
                        html.Button('Load SIMLINE', id='btn-load-simline', n_clicks=0,
                                    style={'padding': '8px 18px', 'backgroundColor': '#9467bd',
                                           'color': 'white', 'border': 'none', 'borderRadius': '6px',
                                           'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': '600',
                                           'marginRight': '8px'}),
                        html.Button('Clear', id='btn-clear-simline', n_clicks=0,
                                    style={'padding': '8px 16px', 'backgroundColor': '#eee',
                                           'color': '#555', 'border': '1px solid #ccc',
                                           'borderRadius': '6px', 'cursor': 'pointer', 'fontSize': '13px'}),
                    ], style={'display': 'flex', 'alignItems': 'stretch'}),
                ], style={'padding': '12px 18px', 'marginTop': '10px',
                          'backgroundColor': '#f7f3fa', 'borderRadius': '8px',
                          'border': '1px dashed #b9a3c9'}),

                html.Div(id='simline-status',
                         style={'marginTop': '6px', 'fontSize': '13px', 'minHeight': '20px'}),
                dcc.Store(id='simline-state', data=0),
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
        dcc.Tab(label='Abundance profiles', value='profiles', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Gas and dust temperatures plus H/H\u2082, C\u207A/C/CO and custom '
                       'species abundance profiles vs depth.',
                       style=_PAGE_INTRO),
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

        # --- Page 3: heating & cooling --------------------------------------
        dcc.Tab(label='Heating & cooling', value='thermal', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Thermal balance and per-component heating and cooling rates vs depth.',
                       style=_PAGE_INTRO),
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

        # --- Page 4: 2-D grid slices --------------------------------------
        dcc.Tab(label='Grid slices', value='grids', style=_TAB_STYLE, selected_style=_TAB_SEL,
                children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Contour maps over the full model grid. Without an overlay grid the '
                       'three slice planes appear side by side as square panels. Grids are '
                       'resampled in log parameter space (KoSens-style) before plotting. '
                       'With an overlay loaded, each plane expands to three sub-plots '
                       '(reference, overlay, x-shift) stacked vertically.',
                       style=_PAGE_INTRO),
                html.Div([
                    html.Div([
                        html.Label('Contoured quantity (cloud edge)', style=_CTRL_LABEL),
                        dcc.Dropdown(id='contour-quantity', options=[], value=None,
                                     placeholder='Load a grid\u2026',
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '2', 'minWidth': '220px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Contour Z scale', style=_CTRL_LABEL),
                        dcc.RadioItems(id='contour-zscale', options=_SCALE_OPTIONS,
                                       value='log', **_RADIO),
                    ], style={**_CTRL_BOX, 'marginRight': '0'}),
                ], style={'display': 'flex', 'alignItems': 'flex-start',
                          'padding': '12px 18px', 'backgroundColor': '#f0f4ff',
                          'borderRadius': '8px', 'marginBottom': '12px'}),
                _interp_control_row(prefix=''),
                _shift_control_row(prefix=''),
                html.Div([_slice_panel(plane) for plane in SLICE_PLANES],
                         id='slice-panels-wrap',
                         style=_SLICE_PANELS_ROW_STYLE),
            ]),
        ]),

        # --- Page 5: chemistry ----------------------------------------------
        dcc.Tab(label='Chemistry', value='chemistry', style=_TAB_STYLE, selected_style=_TAB_SEL,
                children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Top formation and destruction reactions for a selected species '
                       '(requires a chemistry grid on the Load tab). Choose how reactions '
                       'are ranked: fractional contribution to the total rate, or '
                       'mass-weighted rate (KoSens ``top_reactions_plot`` metrics).',
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
                ], style={'display': 'flex', 'alignItems': 'flex-start',
                          'padding': '12px 18px', 'backgroundColor': '#f3f9f1',
                          'borderRadius': '8px'}),
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
        ]),

        # --- Page 6: SIMLINE intensities ------------------------------------
        dcc.Tab(label='Intensities', value='intensities', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('SIMLINE line intensities from per-model .smli files. Without overlay '
                       'SIMLINE the three slice planes appear side by side as square panels; '
                       'grids are resampled in log parameter space before plotting. With '
                       'overlay SIMLINE each plane expands to three sub-plots side by side '
                       '(reference, overlay, x-shift) stacked vertically.',
                       style=_PAGE_INTRO),
                html.Div([
                    html.Div([
                        html.Label('Quantity', style=_CTRL_LABEL),
                        dcc.RadioItems(id='int-idef', options=SIMLINE_IDEF_OPTIONS,
                                       value=SIMLINE_DEFAULT_IDEF, **_RADIO),
                    ], style={'flex': '1.4', 'minWidth': '220px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Species', style=_CTRL_LABEL),
                        dcc.Dropdown(id='int-species', options=[], value=None,
                                     placeholder='Load a SIMLINE directory\u2026',
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '1', 'minWidth': '140px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Transition (for grid maps)', style=_CTRL_LABEL),
                        dcc.Dropdown(id='int-transition', options=[], value=None,
                                     placeholder='Select species\u2026',
                                     style={'fontSize': '13px'}),
                    ], style={'flex': '1.2', 'minWidth': '180px', 'marginRight': '18px'}),
                    html.Div([
                        html.Label('Contour Z scale', style=_CTRL_LABEL),
                        dcc.RadioItems(id='int-zscale', options=_SCALE_OPTIONS,
                                       value='log', **_RADIO),
                    ], style={**_CTRL_BOX, 'marginRight': '0'}),
                ], style={'display': 'flex', 'alignItems': 'flex-start',
                          'padding': '12px 18px', 'backgroundColor': '#f7f3fa',
                          'borderRadius': '8px', 'marginBottom': '12px'}),
                _interp_control_row(prefix='int-'),
                _shift_control_row(prefix='int-'),
                dcc.Graph(id='plot-int-spectrum', figure=placeholder_fig(), config=_GRAPH_CFG,
                          style={'marginBottom': '12px'}),
                html.Div([_int_slice_panel(plane) for plane in SLICE_PLANES],
                         id='int-slice-panels-wrap',
                         style=_SLICE_PANELS_ROW_STYLE),
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
                ], style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
                          'padding': '12px 18px', 'backgroundColor': '#f3f6fa',
                          'borderRadius': '8px', 'marginBottom': '12px'}),
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
                ], style={'padding': '12px 18px', 'backgroundColor': '#faf6f0',
                          'borderRadius': '8px', 'marginBottom': '12px'}),
                dcc.Graph(id='plot-sp-spectrum', figure=placeholder_fig(), config=_GRAPH_CFG,
                          style={'marginBottom': '8px'}),
                html.Div(id='sp-fit-summary', style={'padding': '0 18px 12px'}),
                dcc.Graph(id='plot-sp-pv', figure=placeholder_fig(), config=_GRAPH_CFG),
            ]),
        ]),

        # --- Page 8: interpolation error check --------------------------------
        dcc.Tab(label='Interpolation error', value='interperror', style=_TAB_STYLE,
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
                ], style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
                          'padding': '12px 18px', 'backgroundColor': '#f7f5fa',
                          'borderRadius': '8px', 'marginBottom': '12px'}),
                _interp_control_row(prefix='ie-'),
                _interp_error_control_row(),
                html.Div([_ie_plane_section(plane) for plane in SLICE_PLANES],
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
                          'padding': '12px 18px', 'backgroundColor': '#f3f6fa',
                          'borderRadius': '8px', 'marginBottom': '10px'}),
                html.Div([
                    html.Button('Add current model', id='cr-atten-add', n_clicks=0,
                                style={'padding': '8px 16px', 'marginRight': '8px',
                                       'backgroundColor': '#1f77b4', 'color': 'white',
                                       'border': 'none', 'borderRadius': '6px',
                                       'cursor': 'pointer', 'fontSize': '13px',
                                       'fontWeight': '600'}),
                    html.Button('Add all overlay matches', id='cr-atten-add-overlay', n_clicks=0,
                                style={'padding': '8px 16px', 'marginRight': '8px',
                                       'backgroundColor': '#8c564b', 'color': 'white',
                                       'border': 'none', 'borderRadius': '6px',
                                       'cursor': 'pointer', 'fontSize': '13px',
                                       'fontWeight': '600'}),
                    html.Button('Clear profiles', id='cr-atten-clear', n_clicks=0,
                                style={'padding': '8px 16px', 'backgroundColor': '#eee',
                                       'color': '#555', 'border': '1px solid #ccc',
                                       'borderRadius': '6px', 'cursor': 'pointer',
                                       'fontSize': '13px'}),
                ], style={'padding': '0 18px 10px'}),
                html.Div([
                    html.Label('Profiles to plot', style=_CTRL_LABEL),
                    dcc.Checklist(id='cr-atten-active', options=[], value=[],
                                  style={'fontSize': '13px'}),
                ], style={'padding': '0 18px 12px'}),
                dcc.Graph(id='plot-cr-atten', figure=placeholder_fig(), config=_GRAPH_CFG),
            ]),
        ]),

        # --- Page 10: observational map fit ---------------------------------
        dcc.Tab(label='Map fit', value='mapfit', style=_TAB_STYLE,
                selected_style=_TAB_SEL, children=[
            html.Div(style={'paddingTop': '10px'}, children=[
                html.P('Fit observational FITS intensity maps to the 3-D SIMLINE model grid '
                       '(KoSens3D ``fit_fits_maps_to_grids_3d``). Requires a main grid and '
                       'SIMLINE directory on the Load tab. Axes: density (x), FUV (y), '
                       'cosmic-ray rate (z).',
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
                ], style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
                          'padding': '12px 18px', 'backgroundColor': '#f4f6fa',
                          'borderRadius': '8px', 'marginBottom': '12px'}),
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
                                    style={'padding': '10px 22px', 'fontSize': '14px',
                                           'fontWeight': '600', 'backgroundColor': '#2ca02c',
                                           'color': 'white', 'border': 'none',
                                           'borderRadius': '6px', 'cursor': 'pointer'}),
                    ], style={'flex': '0', 'alignSelf': 'flex-end'}),
                ], style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap',
                          'padding': '12px 18px', 'backgroundColor': '#faf8f4',
                          'borderRadius': '8px', 'marginBottom': '12px'}),
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
                    ], style={'padding': '0 18px'}),
                ]),
            ]),
        ]),
    ]),

    # Footer
    html.Div([
        html.Hr(style={'margin': '16px 0 8px', 'borderColor': '#e8e8e8'}),
        html.P('Use the camera icon on each plot to save it as PNG.',
               style={'color': '#888', 'fontSize': '12px', 'textAlign': 'center', 'margin': '0 0 6px'}),
        html.P('KOSMA-\u03C4 Grid Explorer  \u2014  KoSens3D photodissociation-region model navigator.',
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


def _default_contour_quantity(species_idx):
    if 'CO' in species_idx:
        return CONTOUR_DEFAULT_QUANTITY
    if species_idx:
        return f'species:{next(iter(species_idx))}'
    return 'tgas'


@app.callback(
    [Output('load-status', 'children'),
     Output('grid-loaded', 'data')]
    + _slider_outputs
    + _slice_slider_outputs
    + _int_slice_slider_outputs
    + _ie_slice_slider_outputs
    + [Output('species-selector', 'options'),
       Output('species-selector', 'value'),
       Output('contour-quantity', 'options'),
       Output('contour-quantity', 'value'),
       Output('ie-quantity', 'options'),
       Output('ie-quantity', 'value')],
    Input('btn-load', 'n_clicks'),
    State('dir-input', 'value'),
    State('recursive-check', 'value'),
    prevent_initial_call=True,
)
def handle_load(n_clicks, directory, recursive):
    hidden = {'display': 'none', 'flex': '1 1 240px', 'minWidth': '210px',
              'marginRight': '24px', 'marginBottom': '8px'}
    shown = {**hidden, 'display': 'block'}

    empty_slice = []
    for _ in SLICE_PLANES:
        empty_slice += [1, {}, 0]
    empty_int_slice = list(empty_slice)
    empty_ie_slice = list(empty_slice)

    try:
        grid = scan_directory(directory or '', recursive=bool(recursive))
    except Exception as exc:
        err = html.Span(f'\u2717  {exc}', style={'color': '#d62728', 'fontWeight': '600'})
        empty = []
        for _ in range(N_PARAMS):
            empty += [1, {}, 0, hidden]
        return ([err, False] + empty + empty_slice + empty_int_slice + empty_ie_slice
                + [[], [], [], None, [], None])

    # Build per-slider configuration.
    slider_cfg = []
    for d, p in enumerate(PARAM_DEFS):
        tokens = grid['axis_tokens'][p['key']]
        n = len(tokens)
        varies = n > 1
        marks = build_marks(p)
        value = n // 2
        slider_cfg += [max(n - 1, 0), marks, value, (shown if varies else hidden)]

    slice_cfg = []
    for plane in SLICE_PLANES:
        sdef = _param_def(plane['slice'])
        tokens = grid['axis_tokens'][plane['slice']]
        n = len(tokens)
        slice_cfg += [max(n - 1, 0), build_marks(sdef), n // 2]

    species = grid['species']
    sp_opts = [{'label': s, 'value': s} for s in species]
    defaults = [s for s in DEFAULT_CUSTOM if s in grid['species_idx']][:3]
    cq_opts = _contour_quantity_options(species)
    cq_val = _default_contour_quantity(grid['species_idx'])
    ie_cq_opts = list(cq_opts)
    ie_cq_val = cq_val

    cube = ' \u00D7 '.join(
        f'{len(grid["axis_tokens"][p["key"]])} {p["name"].split()[0]}'
        for p in PARAM_DEFS if len(grid['axis_tokens'][p['key']]) > 1
    ) or 'single model'

    note = ''
    if grid['n_skipped']:
        note = f'  ({grid["n_skipped"]} file(s) skipped: bad name)'
    cfg_note = ''
    if _model_config_summary and not _model_config_summary.get('error'):
        cfg_note = f'   \u2014   {_model_config_summary["n_configs"]} JSON configs'
    status = html.Span([
        html.Span('\u2713  Loaded ', style={'color': '#2ca02c', 'fontWeight': '700'}),
        html.Code(grid['directory']),
        html.Span(f'   {grid["n_files"]} models   \u2014   grid: {cube}{note}{cfg_note}',
                  style={'color': '#555', 'marginLeft': '10px'}),
    ])

    return ([status, True] + slider_cfg + slice_cfg + slice_cfg + slice_cfg
            + [sp_opts, defaults, cq_opts, cq_val, ie_cq_opts, ie_cq_val])


@app.callback(
    Output('controls-wrap', 'style'),
    Input('main-tabs', 'value'),
    Input('grid-loaded', 'data'),
)
def show_controls(tab, loaded):
    if not loaded or tab == 'load':
        return {'display': 'none'}
    return {'display': 'block'}


@app.callback(
    Output('app-root', 'style'),
    Output('controls-sliders-wrap', 'style'),
    Output('controls-axis-wrap', 'style'),
    Output('model-info', 'style'),
    Input('plot-theme', 'value'),
)
def apply_plot_theme(theme):
    t = _theme_colors(theme)
    root = {'fontFamily': 'Arial, sans-serif', 'maxWidth': '1460px',
            'margin': '0 auto', 'padding': '14px 22px', 'backgroundColor': t['page_bg']}
    sliders = {'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-start',
                 'padding': '14px 18px', 'marginTop': '4px',
                 'backgroundColor': t['controls_bg'], 'borderRadius': '8px'}
    axis = {'display': 'flex', 'alignItems': 'flex-start',
            'padding': '10px 18px', 'marginTop': '8px',
            'backgroundColor': t['controls_bg'], 'borderRadius': '8px'}
    info = {'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'alignItems': 'center',
            'backgroundColor': t['controls_bg'], 'padding': '7px 16px', 'borderRadius': '6px',
            'marginTop': '8px', 'marginBottom': '4px', 'fontSize': '13px', 'color': t['font']}
    return root, sliders, axis, info


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
    for d, p in enumerate(PARAM_DEFS):
        tokens = _grid['axis_tokens'][p['key']]
        try:
            tok = tokens[int(values[d])]
        except (IndexError, TypeError, ValueError):
            tok = tokens[0]
        val = p['decode'](tok)
        unit = f' {p["unit"]}' if p['unit'] else ''
        varies = len(tokens) > 1
        if p['key'] == 'atten':
            disp = f'{tok:02d}'
        else:
            disp = f'{val:.4g}'
        labels.append(f'{disp}{unit}')
        info.append(html.Span(
            f'{p["name"]} = {disp}{unit}',
            style={'marginRight': '20px', 'color': p['color'],
                   'fontWeight': '600' if varies else '400',
                   'opacity': 1.0 if varies else 0.55}))

    filepath = current_file(values)
    fname = os.path.basename(filepath) if filepath else '(no matching file)'
    info.append(html.Span(fname, style={'color': '#999', 'fontSize': '12px'}))
    return labels + [info]


@app.callback(
    _slice_label_outputs,
    _slice_slider_inputs,
)
def update_slice_labels(*slice_indices):
    if not _grid:
        return [''] * len(SLICE_PLANES)
    labels = []
    for plane, idx in zip(SLICE_PLANES, slice_indices):
        sdef = _param_def(plane['slice'])
        tokens = _grid['axis_tokens'][plane['slice']]
        try:
            tok = tokens[int(idx)]
        except (IndexError, TypeError, ValueError):
            tok = tokens[0]
        if sdef['key'] == 'atten':
            disp = f'{tok:02d}'
        else:
            disp = f'{sdef["decode"](tok):.4g}'
        unit = f' {sdef["unit"]}' if sdef['unit'] else ''
        labels.append(f'{disp}{unit}')
    return labels


_slice_panel_style_outputs = [Output(f'slice-panel-{p["id"]}', 'style') for p in SLICE_PLANES]
_contour_graph_style_outputs = [Output(f'plot-contour-{p["id"]}', 'style') for p in SLICE_PLANES]
_int_slice_panel_style_outputs = [Output(f'int-slice-panel-{p["id"]}', 'style') for p in SLICE_PLANES]
_int_contour_graph_style_outputs = [Output(f'plot-int-contour-{p["id"]}', 'style') for p in SLICE_PLANES]


@app.callback(
    [Output('slice-panels-wrap', 'style'),
     *_slice_panel_style_outputs,
     *_contour_graph_style_outputs],
    Input('overlay-state', 'data'),
)
def update_slice_panels_layout(_overlay_state):
    """Side-by-side equal-size panels without overlay; full-width rows with overlay."""
    if _overlay:
        return (
            _SLICE_PANELS_COL_STYLE,
            *([_SLICE_PANEL_ROW_FULL] * len(SLICE_PLANES)),
            *([_SLICE_GRAPH_STYLE_FULL] * len(SLICE_PLANES)),
        )
    return (
        _SLICE_PANELS_ROW_STYLE,
        *([_SLICE_PANEL_ROW] * len(SLICE_PLANES)),
        *([_SLICE_GRAPH_STYLE_COMPACT] * len(SLICE_PLANES)),
    )


@app.callback(
    [Output('int-slice-panels-wrap', 'style'),
     *_int_slice_panel_style_outputs,
     *_int_contour_graph_style_outputs],
    Input('simline-overlay-state', 'data'),
)
def update_int_slice_panels_layout(_simline_overlay_state):
    if _simline_overlay:
        return (
            _SLICE_PANELS_COL_STYLE,
            *([_SLICE_PANEL_ROW_FULL] * len(SLICE_PLANES)),
            *([_SLICE_GRAPH_STYLE_FULL] * len(SLICE_PLANES)),
        )
    return (
        _SLICE_PANELS_ROW_STYLE,
        *([_SLICE_PANEL_ROW] * len(SLICE_PLANES)),
        *([_SLICE_GRAPH_STYLE_COMPACT] * len(SLICE_PLANES)),
    )


@app.callback(
    _contour_outputs,
    [Input('contour-quantity', 'value'),
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
)
def update_contour_plots(quantity, zscale, _overlay_state, shift_dir, shift_rtol,
                         interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                         interp_method, interp_clip, plot_theme, grid_colorscale,
                         *slice_indices):
    theme = _parse_plot_theme(plot_theme)
    colorscale = _parse_grid_colorscale(grid_colorscale)
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

    note = f'  ({ov["n_skipped"]} skipped)' if ov['n_skipped'] else ''
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
    Output('chem-status', 'children'),
    Output('chem-state', 'data'),
    Output('react-species', 'options'),
    Output('react-species', 'value'),
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

    if trigger.startswith('btn-clear-chem'):
        clear_chem()
        return (html.Span('Chemistry grid cleared.', style={'color': '#888'}),
                state + 1, [], None)

    try:
        ch = scan_chem(directory or '', recursive=bool(recursive))
    except Exception as exc:
        return (html.Span(f'\u2717  {exc}',
                          style={'color': '#d62728', 'fontWeight': '600'}),
                state + 1, [], None)

    species = ch['species']
    # Order: common diagnostics first, then the rest alphabetically.
    preferred = [s for s in ('CO', 'C+', 'C', 'HCO+', 'H2O', 'OH', 'CH', 'CN', 'HCN', 'CS')
                 if s in species]
    rest = sorted(s for s in species if s not in preferred)
    opts = [{'label': s, 'value': s} for s in preferred + rest]
    value = cur_species if cur_species in species else \
        (CHEM_DEFAULT_SPECIES if CHEM_DEFAULT_SPECIES in species else (species[0] if species else None))

    note = f'  ({ch["n_skipped"]} skipped)' if ch['n_skipped'] else ''
    status = html.Span([
        html.Span('\u2713  Chemistry ', style={'color': '#2ca02c', 'fontWeight': '700'}),
        html.Code(ch['directory']),
        html.Span(f'   {ch["n_files"]} models, {len(species)} species{note}',
                  style={'color': '#555', 'marginLeft': '10px'}),
    ])
    return (status, state + 1, opts, value)


@app.callback(
    Output('simline-status', 'children'),
    Output('simline-state', 'data'),
    Output('int-species', 'options'),
    Output('int-species', 'value'),
    Output('ie-int-species', 'options'),
    Output('ie-int-species', 'value'),
    Output('sp-species', 'options'),
    Output('sp-species', 'value'),
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

    if trigger.startswith('btn-clear-simline'):
        clear_simline()
        return (html.Span('SIMLINE directory cleared.', style={'color': '#888'}),
                state + 1, [], None, [], None, [], None)

    try:
        sl = scan_simline(directory or '', recursive=bool(recursive))
    except Exception as exc:
        return (html.Span(f'\u2717  {exc}',
                          style={'color': '#d62728', 'fontWeight': '600'}),
                state + 1, [], None, [], None, [], None)

    preferred = [s for s in ('CO', '13CO', 'C18O', 'HCO+', 'N2H+', 'CS', 'HCN', 'C+', 'CI')
                 if s in sl['species']]
    rest = sorted(s for s in sl['species'] if s not in preferred)
    opts = [{'label': s, 'value': s} for s in preferred + rest]
    value = cur_species if cur_species in sl['species'] else \
        (SIMLINE_DEFAULT_SPECIES if SIMLINE_DEFAULT_SPECIES in sl['species']
         else (sl['species'][0] if sl['species'] else None))
    ie_value = cur_ie_species if cur_ie_species in sl['species'] else value
    sp_value = cur_sp_species if cur_sp_species in sl['species'] else value

    note = f'  ({sl["n_skipped"]} skipped)' if sl['n_skipped'] else ''
    pv_note = ''
    if sl.get('n_pv_files'):
        pv_note = f', {sl["n_pv_files"]} PV FITS'
    status = html.Span([
        html.Span('\u2713  SIMLINE ', style={'color': '#9467bd', 'fontWeight': '700'}),
        html.Code(sl['directory']),
        html.Span(f'   {sl["n_files"]} files, {len(sl["species"])} species{pv_note}{note}',
                  style={'color': '#555', 'marginLeft': '10px'}),
    ])
    return (status, state + 1, opts, value, opts, ie_value, opts, sp_value)


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
        return [], None, [], None
    opts = _simline_transition_options(species, idef or SIMLINE_DEFAULT_IDEF) if species else []
    dd_opts = [{'label': o['label'], 'value': o['value']} for o in opts]
    valid = {o['value'] for o in opts}
    value = cur_trans if cur_trans in valid else _default_simline_transition(species, idef)

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
    _int_slice_label_outputs,
    _int_slice_slider_inputs,
)
def update_int_slice_labels(*slice_indices):
    if not _grid:
        return [''] * len(SLICE_PLANES)
    labels = []
    for plane, idx in zip(SLICE_PLANES, slice_indices):
        sdef = _param_def(plane['slice'])
        tokens = _grid['axis_tokens'][plane['slice']]
        try:
            tok = tokens[int(idx)]
        except (IndexError, TypeError, ValueError):
            tok = tokens[0]
        if sdef['key'] == 'atten':
            disp = f'{tok:02d}'
        else:
            disp = f'{sdef["decode"](tok):.4g}'
        unit = f' {sdef["unit"]}' if sdef['unit'] else ''
        labels.append(f'{disp}{unit}')
    return labels


def _ie_slice_token(plane, idx):
    tokens = _grid['axis_tokens'][plane['slice']]
    try:
        return tokens[int(idx)]
    except (IndexError, TypeError, ValueError):
        return tokens[0]


def _ie_quantity_unit(quantity):
    if quantity == 'tgas':
        return 'K'
    if quantity == 'tdust':
        return 'K'
    if quantity == 'nh':
        return 'cm^-3'
    if quantity.startswith('species:'):
        return 'rel. abund.'
    return ''


@app.callback(
    _ie_slice_label_outputs,
    _ie_slice_slider_inputs,
)
def update_ie_slice_labels(*slice_indices):
    if not _grid:
        return [''] * len(SLICE_PLANES)
    labels = []
    for plane, idx in zip(SLICE_PLANES, slice_indices):
        sdef = _param_def(plane['slice'])
        tokens = _grid['axis_tokens'][plane['slice']]
        try:
            tok = tokens[int(idx)]
        except (IndexError, TypeError, ValueError):
            tok = tokens[0]
        if sdef['key'] == 'atten':
            disp = f'{tok:02d}'
        else:
            disp = f'{sdef["decode"](tok):.4g}'
        unit = f' {sdef["unit"]}' if sdef['unit'] else ''
        labels.append(f'{disp}{unit}')
    return labels


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
    + _ie_slice_slider_inputs,
)
def update_ie_panels(quantity, ie_species, ie_transition, ie_idef, flux_scale, grid_colorscale,
                     plot_theme, error_metric, plot_contours, decimation, rel_threshold,
                     _grid_loaded, _simline_state,
                     interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                     interp_method, interp_clip, *slice_indices):
    n = len(SLICE_PLANES)
    theme = _parse_plot_theme(plot_theme)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    ie_cfg = _ie_analysis_config(
        interp_ny, interp_nx, interp_x_lim, interp_y_lim, interp_method, interp_clip,
        decimation, error_metric, rel_threshold,
    )
    zscale = flux_scale or 'log'
    err_metric = _parse_error_metric(error_metric)
    show_abund = bool(_grid and quantity)
    show_int = bool(_grid and _simline and ie_species and ie_transition is not None)

    abund_figs, int_figs = [], []
    abund_wrap, int_wrap = [], []
    wrap_show = {'display': 'block', 'marginBottom': '8px'}
    wrap_hide = {'display': 'none'}

    for plane, sidx in zip(SLICE_PLANES, slice_indices):
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
                if sdef['key'] == 'atten':
                    slice_disp = f'{slice_token:02d}'
                else:
                    slice_disp = f'{sdef["decode"](slice_token):.4g}'
                unit_l = _ie_quantity_unit(quantity)
                title = (f'Abundance / diagnostic &mdash; {_IE_PLANE_LABELS.get(plane["id"], "")}'
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
                plane, slice_token, ie_species, ie_idef or SIMLINE_DEFAULT_IDEF, tidx)
            if np.any(np.isfinite(Z)):
                result = gi.analyze_slice_interpolation(
                    x_phys, y_phys, Z,
                    x_logscale=xdef['logscale'],
                    y_logscale=ydef['logscale'],
                    **ie_cfg,
                )
                sk = plane['slice']
                sdef = _param_def(sk)
                if sdef['key'] == 'atten':
                    slice_disp = f'{slice_token:02d}'
                else:
                    slice_disp = f'{sdef["decode"](slice_token):.4g}'
                trans_rows = _simline_transition_options(ie_species, ie_idef)
                tlabel = trans_rows[tidx]['label'] if 0 <= tidx < len(trans_rows) else str(tidx)
                unit_l = _intensity_unit_label(ie_idef or SIMLINE_DEFAULT_IDEF)
                sp_html = format_species_html(ie_species)
                title = (f'Intensity &mdash; {_IE_PLANE_LABELS.get(plane["id"], "")}'
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
     Input('plot-theme', 'value'),
     Input('grid-colorscale', 'value')]
    + _int_slice_slider_inputs,
)
def update_intensity_contours(species, idef, transition, zscale, _state, _ov_state,
                              shift_dir, shift_rtol,
                              interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                              interp_method, interp_clip, plot_theme, grid_colorscale,
                              *slice_indices):
    theme = _parse_plot_theme(plot_theme)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    if not _grid or not _simline or not species or transition is None:
        p = placeholder_fig('Load grids and a SIMLINE directory', theme=theme)
        return (p,) * len(SLICE_PLANES)
    icfg = _interp_config(interp_ny, interp_nx, interp_x_lim, interp_y_lim,
                          interp_method, interp_clip)
    return make_intensity_contour_plots(
        species, idef, transition, zscale or 'log', list(slice_indices),
        shift_rtol=_parse_shift_rtol(shift_rtol),
        shift_scan_direction=_parse_shift_scan_direction(shift_dir),
        interp_config=icfg,
        colorscale=colorscale, theme=theme,
    )


@app.callback(
    Output('plot-int-spectrum', 'figure'),
    _slider_value_inputs
    + [Input('int-species', 'value'),
       Input('int-idef', 'value'),
       Input('simline-state', 'data'),
       Input('plot-theme', 'value')],
)
def update_intensity_spectrum(*args_in):
    values = list(args_in[:N_PARAMS])
    species, idef, _state, plot_theme = args_in[N_PARAMS:]
    return fig_intensity_spectrum(values, species, idef, theme=_parse_plot_theme(plot_theme))


@app.callback(
    Output('plot-tgas', 'figure'),
    Output('plot-h-h2', 'figure'),
    Output('plot-cco', 'figure'),
    Output('plot-custom', 'figure'),
    _slider_value_inputs
    + [Input('xvar-choice', 'value'),
       Input('xscale', 'value'),
       Input('yscale', 'value'),
       Input('species-selector', 'value'),
       Input('overlay-state', 'data'),
       Input('plot-theme', 'value')],
)
def update_profile_plots(*args_in):
    values = list(args_in[:N_PARAMS])
    xvar, xscale, yscale, custom_species, _overlay_state, plot_theme = args_in[N_PARAMS:]
    return make_profile_plots(values, xvar, xscale, yscale, custom_species or [],
                              theme=_parse_plot_theme(plot_theme))


@app.callback(
    Output('plot-thermal', 'figure'),
    Output('plot-heat-breakdown', 'figure'),
    Output('plot-cool-breakdown', 'figure'),
    _slider_value_inputs
    + [Input('xvar-choice', 'value'),
       Input('xscale', 'value'),
       Input('yscale', 'value'),
       Input('overlay-state', 'data'),
       Input('plot-theme', 'value')],
)
def update_thermal_plots(*args_in):
    values = list(args_in[:N_PARAMS])
    xvar, xscale, yscale, _overlay_state, plot_theme = args_in[N_PARAMS:]
    return make_thermal_plots(values, xvar, xscale, yscale,
                              theme=_parse_plot_theme(plot_theme))


@app.callback(
    Output('plot-react-formation', 'figure'),
    Output('react-contrib-formation', 'children'),
    Output('plot-react-destruction', 'figure'),
    Output('react-contrib-destruction', 'children'),
    _slider_value_inputs
    + [Input('react-species', 'value'),
       Input('react-n', 'value'),
       Input('react-ranking', 'value'),
       Input('xscale', 'value'),
       Input('yscale', 'value'),
       Input('chem-state', 'data'),
       Input('plot-theme', 'value')],
)
def update_reaction_plots(*args_in):
    values = list(args_in[:N_PARAMS])
    species, top_n, ranking, xscale, yscale, _chem_state, plot_theme = args_in[N_PARAMS:]
    top_n = top_n or CHEM_DEFAULT_NREAC
    return make_reaction_plots(values, species, xscale, yscale, top_n,
                               ranking_metric=ranking,
                               theme=_parse_plot_theme(plot_theme))


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


@app.callback(
    Output('fit-status', 'children'),
    Output('fit-state', 'data'),
    Output('plot-fit-x', 'figure'),
    Output('plot-fit-y', 'figure'),
    Output('plot-fit-z', 'figure'),
    Output('plot-fit-chi2', 'figure'),
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
    if not n_clicks:
        return dash.no_update, dash.no_update, empty, empty, empty, empty

    opts = fit_options or []
    try:
        result = run_map_fit_job(
            maps_json, errors_json, output_dir, idef, nz, ny, nx, method,
            chi2_i, chi2_j, opts, opts, opts, opts,
        )
    except Exception as exc:
        err = html.Span(f'\u2717  {exc}',
                        style={'color': '#d62728', 'fontWeight': '600'})
        return err, (state or 0) + 1, empty, empty, empty, empty

    # WCS from the reference observed FITS (stored on the result)
    figs = gf.fit_result_figures(result, theme=theme, colorscale=colorscale)
    ref = result.get('reference_line', '')
    nlines = len(result.get('lines_fitted') or [])
    status = html.Span([
        html.Span('\u2713  Map fit complete. ', style={'color': '#2ca02c', 'fontWeight': '700'}),
        html.Span(f'{nlines} line(s), reference footprint: {ref}. '),
        html.Span('FITS saved to '),
        html.Code(result.get('output_dir', '')),
        html.Span(f"  \u2014  x: {os.path.basename(result.get('x_fits_path', ''))}, "
                  f"y: {os.path.basename(result.get('y_fits_path', ''))}, "
                  f"z: {os.path.basename(result.get('z_fits_path', ''))}"),
    ])
    chi2_fig = figs.get('chi2', placeholder_fig('No reduced χ² map', theme=theme))
    return (
        status, (state or 0) + 1,
        figs.get('x', empty), figs.get('y', empty),
        figs.get('z', empty), chi2_fig,
    )


@app.callback(
    Output('plot-fit-x', 'figure', allow_duplicate=True),
    Output('plot-fit-y', 'figure', allow_duplicate=True),
    Output('plot-fit-z', 'figure', allow_duplicate=True),
    Output('plot-fit-chi2', 'figure', allow_duplicate=True),
    Input('plot-theme', 'value'),
    Input('grid-colorscale', 'value'),
    Input('fit-state', 'data'),
    prevent_initial_call=True,
)
def refresh_fit_figures_on_theme(plot_theme, grid_colorscale, _fit_state):
    if not _fit_results:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    theme = _parse_plot_theme(plot_theme)
    colorscale = _parse_grid_colorscale(grid_colorscale)
    figs = gf.fit_result_figures(_fit_results, theme=theme, colorscale=colorscale)
    empty = placeholder_fig('Run a map fit', theme=theme)
    return (
        figs.get('x', empty), figs.get('y', empty),
        figs.get('z', empty),
        figs.get('chi2', placeholder_fig('No reduced χ² map', theme=theme)),
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
