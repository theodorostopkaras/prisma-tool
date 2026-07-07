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
import io

import numpy as np
import h5py

import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go


# --- CLI ----------------------------------------------------------------------

DEFAULT_DIR = '/home/teotopkaras/Desktop/Teo_AR_included_grid/pdrgrid_hdf5'

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
METADATA_PATH   = 'Metadata/Metadata'
SPECIES_PATH    = 'Additional output/species involved'
RELDENS_PATH    = 'Local quantities/Densities/Relative densities'
DENS_PATH       = 'Local quantities/Densities/Densities'

# Internal HDF5 metadata field keys (KOSMA-tau convention).
KEY_AV    = 'av'         # visual extinction profile (Positions, col 0)
KEY_NH    = 'protdens'   # proton/H nucleus density profile (Gas state, col 0)
KEY_TGAS  = 'tgas'       # gas temperature  (Gas state, col 2)
KEY_TDUST = 'tdust'      # dust temperature (Gas state, col 3)

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

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

DEFAULT_CUSTOM = ['OH', 'H2O', 'HCN', 'CN', 'CS', 'HCO+']


# --- Server-side data store ---------------------------------------------------
# Single global dict (last loaded directory wins).  For multi-user deployments
# move this into flask_caching keyed by session.

_grid = {}            # populated on load
_field_map = {}       # field_key -> (hdf5_path, column_index)
_profile_cache = {}   # filepath -> dict of profile arrays


def _dec(x):
    return x.decode() if isinstance(x, bytes) else x


# --- Grid scanning ------------------------------------------------------------

def parse_filename(fname):
    """Return a tuple of 6 integer tokens (DD, MM, FF, ZZ, CC, AA) or None."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split('_')
    if len(parts) < 7:
        return None
    try:
        return tuple(int(parts[p['token_idx']]) for p in PARAM_DEFS)
    except (ValueError, IndexError):
        return None


def build_structure(sample_file):
    """Read metadata + species list from one representative file.

    Returns
    -------
    field_map : dict  field_key -> (hdf5_internal_path, column_index)
    species   : list[str]
    """
    field_map = {}
    with h5py.File(sample_file, 'r') as hf:
        md = hf[METADATA_PATH][:]
        for row in md:
            group = _dec(row[0])
            dset = _dec(row[1])
            key = _dec(row[3])
            if key in field_map:
                continue
            try:
                idx = int(float(_dec(row[2])))
            except (ValueError, TypeError):
                continue
            path = group.rstrip('/') + '/' + dset
            field_map[key] = (path, idx)
        species = [_dec(x[0]) for x in hf[SPECIES_PATH][:]]
    return field_map, species


def scan_directory(directory, recursive=False):
    """Scan ``directory`` for per-model HDF5 files and build the global grid."""
    global _grid, _field_map, _profile_cache

    directory = os.path.expanduser(directory.strip())
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
                         'Model_DD_MM_FF_ZZ_CC_AA naming convention.')

    # Per-parameter sorted unique tokens.
    axis_tokens = {}
    for d, p in enumerate(PARAM_DEFS):
        axis_tokens[p['key']] = sorted({tok[d] for tok in files})

    field_map, species = build_structure(next(iter(files.values())))

    _profile_cache = {}
    _field_map = field_map
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


def current_file(values):
    """Map a list of slider indices (PARAM order) to a filepath, or None."""
    if not _grid:
        return None
    try:
        tokens = tuple(
            _grid['axis_tokens'][PARAM_DEFS[d]['key']][int(values[d])]
            for d in range(N_PARAMS)
        )
    except (IndexError, KeyError, TypeError):
        return None
    return _grid['files'].get(tokens)


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


def get_model(filepath):
    """Load (and cache) the depth profiles needed for plotting one model."""
    if filepath in _profile_cache:
        return _profile_cache[filepath]
    with h5py.File(filepath, 'r') as hf:
        model = dict(
            av    = _read_field(hf, KEY_AV),
            nH    = _read_field(hf, KEY_NH),
            tgas  = _read_field(hf, KEY_TGAS),
            tdust = _read_field(hf, KEY_TDUST),
            rel   = np.asarray(hf[RELDENS_PATH][:], dtype=float),   # (n_depth, n_species)
            dens  = np.asarray(hf[DENS_PATH][:], dtype=float),
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

_BASE = dict(
    paper_bgcolor='white',
    plot_bgcolor='#f8f9fa',
    margin=dict(l=70, r=20, t=44, b=54),
    font=dict(family='Arial, sans-serif', size=12),
    height=320,
    legend=dict(bgcolor='rgba(255,255,255,0.85)', borderwidth=1, bordercolor='#ccc'),
)

_AXIS_STYLE = dict(
    showgrid=True, gridcolor='#e0e0e0', gridwidth=1,
    zeroline=False, linecolor='#bbb', mirror=True,
    exponentformat='e', showexponent='all',
)


def _apply_layout(fig, title, xlabel, xtype, xrange, ylabel, ytype):
    fig.update_layout(
        **_BASE,
        title=dict(text=title, font=dict(size=13, color='#333'), x=0.02, xanchor='left'),
        xaxis=dict(**_AXIS_STYLE, title=dict(text=xlabel, font=dict(size=12)),
                   type=xtype, range=xrange),
        yaxis=dict(**_AXIS_STYLE, title=dict(text=ylabel, font=dict(size=12)), type=ytype),
    )


def _xvals(model, xvar, xscale):
    """Return (x_array, x_label, x_type, x_range) with an explicit axis range."""
    if xvar == 'nH':
        xv = np.asarray(model['nH'], dtype=float)
        xl = 'n<sub>H</sub> (cm<sup>-3</sup>)'
    else:
        xv = np.asarray(model['av'], dtype=float)
        xl = 'A<sub>V</sub> (mag)'

    pos = xv[xv > 0]
    if xscale == 'log' and pos.size:
        xrange = [np.floor(np.log10(pos.min())), np.ceil(np.log10(pos.max()))]
        xv = np.where(xv > 0, xv, np.nan)
    else:
        lo = 0.0 if xvar == 'Av' else float(np.nanmin(xv))
        xrange = [lo, float(np.nanmax(xv)) * 1.02]
        xscale = 'linear'
    return xv, xl, xscale, xrange


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


def add_h_h2_vline(fig, x_cross):
    if x_cross is None:
        return
    fig.add_shape(type='line', xref='x', yref='paper',
                  x0=x_cross, x1=x_cross, y0=0, y1=1,
                  line=dict(color='rgba(60,60,60,0.45)', width=1, dash='dot'))


def placeholder_fig(msg='Load a grid directory to begin'):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False,
                       font=dict(size=14, color='#aaa'),
                       xref='paper', yref='paper', x=0.5, y=0.5)
    fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='#f4f4f4',
        margin=dict(l=20, r=20, t=20, b=20), height=320,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def fig_tgas(model, xvar, xscale, yscale):
    xv, xl, xt, xr = _xvals(model, xvar, xscale)
    tgas = np.asarray(model['tgas'], dtype=float)
    tdust = np.asarray(model['tdust'], dtype=float)
    if yscale == 'log':
        tgas = np.where(tgas > 0, tgas, np.nan)
        tdust = np.where(tdust > 0, tdust, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xv, y=tgas, mode='lines',
                             line=dict(color=COLORS[0], width=2), name='T<sub>gas</sub>'))
    fig.add_trace(go.Scatter(x=xv, y=tdust, mode='lines',
                             line=dict(color=COLORS[1], width=2, dash='dot'),
                             name='T<sub>dust</sub>'))
    _apply_layout(fig, 'Gas / Dust Temperature', xl, xt, xr, 'T (K)', yscale)
    return fig


def fig_h_h2(model, xvar, xscale, yscale):
    xv, xl, xt, xr = _xvals(model, xvar, xscale)
    fig = go.Figure()
    for name, color in (('H', COLORS[0]), ('H2', COLORS[1])):
        ab = species_abundance(model, name, yscale)
        if ab is not None:
            fig.add_trace(go.Scatter(x=xv, y=ab, mode='lines',
                                     line=dict(color=color, width=2),
                                     name=format_species_html(name)))
    _apply_layout(fig, 'H / H<sub>2</sub>', xl, xt, xr,
                  'x(species)', yscale)
    return fig


def fig_cplus_c_co(model, xvar, xscale, yscale):
    xv, xl, xt, xr = _xvals(model, xvar, xscale)
    fig = go.Figure()
    for name, color in (('C+', COLORS[3]), ('C', COLORS[2]), ('CO', COLORS[0])):
        ab = species_abundance(model, name, yscale)
        if ab is not None:
            fig.add_trace(go.Scatter(x=xv, y=ab, mode='lines',
                                     line=dict(color=color, width=2),
                                     name=format_species_html(name)))
    _apply_layout(fig, 'C<sup>+</sup> / C / CO', xl, xt, xr,
                  'x(species)', yscale)
    return fig


def fig_custom(model, xvar, xscale, yscale, sel_species):
    xv, xl, xt, xr = _xvals(model, xvar, xscale)
    fig = go.Figure()
    for k, sp in enumerate(sel_species or []):
        ab = species_abundance(model, sp, yscale)
        if ab is not None:
            fig.add_trace(go.Scatter(x=xv, y=ab, mode='lines',
                                     line=dict(color=COLORS[k % len(COLORS)], width=2),
                                     name=format_species_html(sp)))
    _apply_layout(fig, 'Custom Species', xl, xt, xr,
                  'x(species)', yscale)
    return fig


def make_all_plots(values, xvar, xscale, yscale, custom_species):
    if not _grid:
        p = placeholder_fig()
        return p, p, p, p

    filepath = current_file(values)
    if filepath is None:
        p = placeholder_fig('No model file for this parameter combination')
        return p, p, p, p

    model = get_model(filepath)
    x_cross = find_h_h2_transition(model, xvar)
    figs = [
        fig_tgas(model, xvar, xscale, yscale),
        fig_h_h2(model, xvar, xscale, yscale),
        fig_cplus_c_co(model, xvar, xscale, yscale),
        fig_custom(model, xvar, xscale, yscale, custom_species),
    ]
    for f in figs:
        add_h_h2_vline(f, x_cross)
    return tuple(figs)


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


app.layout = html.Div(
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

    # Directory loader
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

    # Controls (hidden until a grid is loaded)
    html.Div(id='controls-wrap', style={'display': 'none'}, children=[

        html.Div([_slider_block(d) for d in range(N_PARAMS)],
                 style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-start',
                        'padding': '14px 18px', 'marginTop': '14px',
                        'backgroundColor': '#f0f4ff', 'borderRadius': '8px'}),

        html.Div([
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
        ], style={'display': 'flex', 'alignItems': 'flex-start',
                  'padding': '10px 18px', 'marginTop': '8px',
                  'backgroundColor': '#f0f4ff', 'borderRadius': '8px'}),

        html.Div(id='model-info', style={
            'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'alignItems': 'center',
            'backgroundColor': '#e8f4f8', 'padding': '7px 16px', 'borderRadius': '6px',
            'marginTop': '8px', 'marginBottom': '14px', 'fontSize': '13px', 'color': '#333'}),
    ]),

    # 2x2 plot grid
    html.Div([
        html.Div([
            dcc.Graph(id='plot-tgas', figure=placeholder_fig(),
                      config={'toImageButtonOptions': {'format': 'png', 'scale': 2}},
                      style={'flex': '1', 'minWidth': '0'}),
            dcc.Graph(id='plot-h-h2', figure=placeholder_fig(),
                      config={'toImageButtonOptions': {'format': 'png', 'scale': 2}},
                      style={'flex': '1', 'minWidth': '0'}),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '12px'}),

        html.Div([
            dcc.Graph(id='plot-cco', figure=placeholder_fig(),
                      config={'toImageButtonOptions': {'format': 'png', 'scale': 2}},
                      style={'flex': '1', 'minWidth': '0'}),
            html.Div([
                html.Div([
                    html.Label('Species:', style={'fontWeight': '600', 'fontSize': '12px',
                                                  'whiteSpace': 'nowrap', 'marginRight': '8px'}),
                    dcc.Dropdown(id='species-selector', options=[], value=[], multi=True,
                                 placeholder='Select species to plot\u2026',
                                 style={'fontSize': '12px', 'flex': '1'}),
                ], style={'display': 'flex', 'alignItems': 'center',
                          'padding': '4px 0', 'marginBottom': '4px'}),
                dcc.Graph(id='plot-custom', figure=placeholder_fig(),
                          config={'toImageButtonOptions': {'format': 'png', 'scale': 2}},
                          style={'flex': '1'}),
            ], style={'flex': '1', 'minWidth': '0', 'display': 'flex', 'flexDirection': 'column'}),
        ], style={'display': 'flex', 'gap': '12px'}),
    ]),

    # Download toolbar
    html.Div([
        html.Button('\u2B07  Download model (ASCII)', id='btn-download', n_clicks=0,
                    style={'padding': '8px 18px', 'backgroundColor': '#1f77b4',
                           'color': 'white', 'border': 'none', 'borderRadius': '6px',
                           'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': '600',
                           'marginRight': '14px'}),
        html.Span('Use the camera icon on each plot to save it as PNG.',
                  style={'color': '#888', 'fontSize': '12px'}),
        dcc.Download(id='download-ascii'),
    ], style={'marginTop': '16px', 'display': 'flex', 'alignItems': 'center',
              'borderTop': '1px solid #ddd', 'paddingTop': '14px'}),

    # Footer
    html.Div([
        html.Hr(style={'margin': '16px 0 8px', 'borderColor': '#e8e8e8'}),
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


@app.callback(
    [Output('load-status', 'children'),
     Output('controls-wrap', 'style')]
    + _slider_outputs
    + [Output('species-selector', 'options'),
       Output('species-selector', 'value')],
    Input('btn-load', 'n_clicks'),
    State('dir-input', 'value'),
    State('recursive-check', 'value'),
    prevent_initial_call=True,
)
def handle_load(n_clicks, directory, recursive):
    hidden = {'display': 'none', 'flex': '1 1 240px', 'minWidth': '210px',
              'marginRight': '24px', 'marginBottom': '8px'}
    shown = {**hidden, 'display': 'block'}

    try:
        grid = scan_directory(directory or '', recursive=bool(recursive))
    except Exception as exc:
        err = html.Span(f'\u2717  {exc}', style={'color': '#d62728', 'fontWeight': '600'})
        empty = []
        for _ in range(N_PARAMS):
            empty += [1, {}, 0, hidden]
        return [err, {'display': 'none'}] + empty + [[], []]

    # Build per-slider configuration.
    n_varying = []
    slider_cfg = []
    for d, p in enumerate(PARAM_DEFS):
        tokens = grid['axis_tokens'][p['key']]
        n = len(tokens)
        varies = n > 1
        if varies:
            n_varying.append(p['name'])
        marks = build_marks(p)
        value = n // 2
        slider_cfg += [max(n - 1, 0), marks, value, (shown if varies else hidden)]

    species = grid['species']
    sp_opts = [{'label': s, 'value': s} for s in species]
    defaults = [s for s in DEFAULT_CUSTOM if s in grid['species_idx']][:3]

    cube = ' \u00D7 '.join(
        f'{len(grid["axis_tokens"][p["key"]])} {p["name"].split()[0]}'
        for p in PARAM_DEFS if len(grid['axis_tokens'][p['key']]) > 1
    ) or 'single model'

    note = ''
    if grid['n_skipped']:
        note = f'  ({grid["n_skipped"]} file(s) skipped: bad name)'
    status = html.Span([
        html.Span('\u2713  Loaded ', style={'color': '#2ca02c', 'fontWeight': '700'}),
        html.Code(grid['directory']),
        html.Span(f'   {grid["n_files"]} models   \u2014   grid: {cube}{note}',
                  style={'color': '#555', 'marginLeft': '10px'}),
    ])

    return [status, {'display': 'block'}] + slider_cfg + [sp_opts, defaults]


# Per-slider value labels + model info bar.
_label_outputs = [Output(f'slabel-{d}', 'children') for d in range(N_PARAMS)]
_slider_value_inputs = [Input(f'slider-{d}', 'value') for d in range(N_PARAMS)]


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
    Output('plot-tgas', 'figure'),
    Output('plot-h-h2', 'figure'),
    Output('plot-cco', 'figure'),
    Output('plot-custom', 'figure'),
    _slider_value_inputs
    + [Input('xvar-choice', 'value'),
       Input('xscale', 'value'),
       Input('yscale', 'value'),
       Input('species-selector', 'value')],
)
def update_plots(*args_in):
    values = list(args_in[:N_PARAMS])
    xvar, xscale, yscale, custom_species = args_in[N_PARAMS:]
    return make_all_plots(values, xvar, xscale, yscale, custom_species or [])


@app.callback(
    Output('download-ascii', 'data'),
    Input('btn-download', 'n_clicks'),
    _slider_value_inputs,
    prevent_initial_call=True,
)
def download_ascii(n_clicks, *values):
    if not _grid:
        return no_update
    filepath = current_file(list(values))
    if filepath is None:
        return no_update

    model = get_model(filepath)
    species = _grid['species']

    # Header with the decoded model parameters.
    pars = []
    for d, p in enumerate(PARAM_DEFS):
        tok = _grid['axis_tokens'][p['key']][int(values[d])]
        val = p['decode'](tok)
        pars.append(f'{p["name"]}={val:.4g}{p["unit"]}')
    header_lines = [
        '# KOSMA-tau model  ' + '   '.join(pars),
        '# source file: ' + os.path.basename(filepath),
        '# Columns: Av[mag]  nH[cm-3]  Tgas[K]  Tdust[K]  then relative abundance x(species)',
        '# Species order: ' + ' '.join(species),
    ]

    av, nH = model['av'], model['nH']
    tgas, tdust = model['tgas'], model['tdust']
    rel = model['rel']
    n_depth = len(av)

    buf = io.StringIO()
    buf.write('\n'.join(header_lines) + '\n')
    base_cols = ['Av', 'nH', 'Tgas', 'Tdust'] + list(species)
    buf.write('# ' + '  '.join(f'{c:>14s}' for c in base_cols) + '\n')
    for i in range(n_depth):
        row = [av[i], nH[i], tgas[i], tdust[i]] + list(rel[i, :len(species)])
        buf.write('  '.join(f'{v:14.6E}' for v in row) + '\n')

    stem = os.path.splitext(os.path.basename(filepath))[0]
    return dict(content=buf.getvalue(), filename=f'{stem}.dat', type='text/plain')


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
