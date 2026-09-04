"""
Regime-aware CRIR probe ranking on 3-D PDR grids.

For each tracer (species, column density, line, or ratio) the quantity is
read along the cosmic-ray axis at every fixed environment (typically n_H ×
FUV).  The CRIR axis is split into regimes; slope, plateau detection and a
probe score are computed independently in each regime so a high score is
always tied to a specific cosmic-ray range.

Score
-----
``S = F × V`` in [0, 1]:

* **Variation** ``V`` maps the log10 dynamic range of the curve onto [0, 1],
  saturating around a 100× swing by default.
* **Functional** ``F`` is the fraction of the log-CRIR span that is *active*
  (local |d log y / d log ζ| at or above the plateau slope).  More than one
  trend reversal halves ``F``.

A local slope below the plateau threshold (default: factor 3 per CRIR decade)
is treated as flat / uninformative.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

import plot_style as ps


DEFAULT_PLATEAU_SLOPE = float(np.log10(3.0))
DEFAULT_VARIATION_R_CAP = float(np.log10(100.0))
DEFAULT_MIN_ABUNDANCE = 1e-15
KOSENS_DEFAULT_N_ENV = (1e1, 1e3, 1e5)
KOSENS_DEFAULT_FUV_ENV = (1e1, 1e2, 1e5)
DEFAULT_GOOD_SCORE_THRESHOLD = 0.3
DEFAULT_CRIR_REGIME_EDGES = (1e-16,)
DEFAULT_MIN_POINTS = 4
DEFAULT_MIN_LOG_CRIR_SPAN = 0.5
LEVEL_MATCH_RTOL = 1e-6

QUARTILE_COLORS = {
    'green': '#2ca02c',
    'yellow': '#bcbd22',
    'orange': '#ff7f0e',
    'red': '#d62728',
    'gray': '#7f7f7f',
}

QUARTILE_LABELS = {
    'green': 'Q1 (best 25%)',
    'yellow': 'Q2',
    'orange': 'Q3',
    'red': 'Q4 (worst 25%)',
    'gray': 'Plateau / not observable',
}

ABSOLUTE_BAND_COLORS = {
    'strong': '#2ca02c',
    'good': '#bcbd22',
    'moderate': '#ff7f0e',
    'weak': '#d62728',
    'inactive': '#7f7f7f',
}

ABSOLUTE_BAND_LABELS = {
    'strong': 'Strong  (S ≥ 0.59)',
    'good': 'Good  (S ≥ 0.39)',
    'moderate': 'Moderate  (S ≥ 0.20)',
    'weak': 'Weak  (S > 0)',
    'inactive': 'Inactive / plateau',
}

Y_LABELS = {
    'rel_abund': 'Clump-integrated X',
    'intensity': 'Line intensity',
    'column_density': r'$N~(\mathrm{cm}^{-2})$',
}

FUV_LINE_COLORS = ['#1a1a1a', '#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

_NAN_SCORE = {
    'probe_score': np.nan,
    'functional_factor': np.nan,
    'variation_factor': np.nan,
    'trend': 'unknown',
    'n_reversals': 0,
    'log_dynamic_range': np.nan,
    'spearman_rho': np.nan,
    'net_log_slope': np.nan,
    'observable': False,
    'status': 'unknown',
}


_SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')


def sci_html(value, *, power_tol=0.05):
    """HTML scientific label for Plotly hover (MathJax is not applied there)."""
    if value is None or not np.isfinite(value):
        return '∞' if value is None else '—'
    if value == 0:
        return '0'
    if value < 0:
        return f'−{sci_html(-value, power_tol=power_tol)}'
    exp = int(np.round(np.log10(value)))
    if abs(value - 10.0 ** exp) / max(value, np.finfo(float).tiny) < power_tol:
        return f'10<sup>{exp}</sup>'
    return f'{value:.2g}'


def sci_tex(value, *, power_tol=0.05):
    """LaTeX math without delimiters, e.g. ``10^{-16}``."""
    if value is None or not np.isfinite(value):
        return r'\infty' if value is None else r'\mathrm{—}'
    if value == 0:
        return '0'
    if value < 0:
        return rf'-{sci_tex(-value, power_tol=power_tol)}'
    exp = int(np.round(np.log10(value)))
    if abs(value - 10.0 ** exp) / max(value, np.finfo(float).tiny) < power_tol:
        return rf'10^{{{exp}}}'
    man = value / (10.0 ** exp)
    return rf'{man:.2g}\times 10^{{{exp}}}'


def sci_latex(value, *, power_tol=0.05):
    """MathJax-ready Plotly label, e.g. ``$10^{-16}$``."""
    inner = sci_tex(value, power_tol=power_tol)
    if inner == '0':
        return '0'
    return f'${inner}$'


def sci_plain(value, *, power_tol=0.05):
    """Unicode label for dropdown chips and tables (no MathJax there)."""
    if value is None or not np.isfinite(value):
        return '∞' if value is None else '—'
    if value == 0:
        return '0'
    if value < 0:
        return f'−{sci_plain(-value, power_tol=power_tol)}'
    exp = int(np.round(np.log10(value)))
    if abs(value - 10.0 ** exp) / max(value, np.finfo(float).tiny) < power_tol:
        return f'10{str(exp).translate(_SUP)}'
    man = value / (10.0 ** exp)
    return f'{man:.2g}×10{str(exp).translate(_SUP)}'


def format_crir_value(value):
    """Compact CRIR bound, preferring ``1e{k}`` for powers of ten."""
    if value is None or not np.isfinite(value):
        return '∞'
    if value <= 0:
        return f'{value:g}'
    exp = np.log10(value)
    exp_round = int(round(exp))
    if abs(exp - exp_round) < 1e-6:
        return f'1e{exp_round}'
    return f'{value:.1e}'


def crir_regime_label(lo, hi, *, wrap=True):
    """CRIR-regime span. ``wrap=True`` → MathJax for plots; else Unicode for widgets."""
    inner = rf'{sci_tex(lo)}\text{{–}}{sci_tex(hi)}\,\mathrm{{s}}^{{-1}}'
    if wrap:
        return f'${inner}$'
    return f'{sci_plain(lo)}–{sci_plain(hi)} s⁻¹'


def parse_regime_edges(text):
    """Parse a comma-separated CRIR-edge string into interior floats.

    Empty / blank → default split at 10⁻¹⁶.  Use ``none`` or ``full`` for a
    single regime covering the whole axis.
    """
    raw = (text or '').strip()
    if not raw:
        return list(DEFAULT_CRIR_REGIME_EDGES)
    if raw.lower() in ('none', 'full', 'off'):
        return []
    edges = []
    for part in raw.replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        edges.append(float(part))
    return edges


def build_crir_regimes(regimes_spec, crir_min, crir_max):
    """Normalise a CRIR regime spec into ``[{name, lo, hi, label}, ...]``."""
    if regimes_spec is None:
        regimes_spec = DEFAULT_CRIR_REGIME_EDGES
    regimes_spec = list(regimes_spec)

    explicit = (
        len(regimes_spec) > 0
        and isinstance(regimes_spec[0], (tuple, list))
        and len(regimes_spec[0]) == 3
    )
    if explicit:
        regimes = []
        for name, lo, hi in regimes_spec:
            lo_f = float(lo) if lo is not None else float(crir_min)
            hi_f = float(hi) if hi is not None else float(crir_max)
            regimes.append({
                'name': str(name),
                'lo': lo_f,
                'hi': hi_f,
                'label': crir_regime_label(lo_f, hi_f),
            })
        return regimes

    edges = sorted(float(e) for e in regimes_spec)
    bounds = [float(crir_min)]
    bounds += [e for e in edges if crir_min < e < crir_max]
    bounds += [float(crir_max)]
    bounds = sorted(set(bounds))
    regimes = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        regimes.append({
            'name': f'[{format_crir_value(lo)}, {format_crir_value(hi)}]',
            'lo': float(lo),
            'hi': float(hi),
            'label': crir_regime_label(lo, hi),
        })
    return regimes


def _classify_trend(slope, plateau_slope):
    if slope > plateau_slope:
        return 'increasing'
    if slope < -plateau_slope:
        return 'decreasing'
    return 'flat'


def _snap_native_level(value):
    """Map a physical coordinate to the nearest integer-log10 native node."""
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    return float(10.0 ** np.round(np.log10(value)))


def _theil_slope(y, x):
    """Median pairwise slope (Theil–Sen). Faster than ``scipy.stats.theilslopes``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        return 0.0
    if n == 2:
        dx = x[1] - x[0]
        return float((y[1] - y[0]) / dx) if dx != 0 else 0.0
    slopes = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        ok = dx != 0
        if np.any(ok):
            slopes.append(dy[ok] / dx[ok])
    if not slopes:
        return 0.0
    return float(np.median(np.concatenate(slopes)))


def _coord_axis_index(mesh):
    """Index of the ndarray axis along which a coordinate mesh varies."""
    mesh = np.asarray(mesh, dtype=float)
    if mesh.ndim != 3:
        raise ValueError(f'Expected 3D coordinate mesh, got shape {mesh.shape}')

    for ax in range(3):
        if mesh.shape[ax] < 2:
            continue
        ref = np.take(mesh, 0, axis=ax)
        if np.allclose(ref, np.take(mesh, 1, axis=ax), rtol=1e-6, atol=0.0):
            continue
        constant = True
        for other in range(3):
            if other == ax:
                continue
            if mesh.shape[other] < 2:
                continue
            if not np.allclose(
                np.take(mesh, 0, axis=other),
                np.take(mesh, 1, axis=other),
                rtol=1e-6,
                atol=0.0,
            ):
                constant = False
                break
        if constant:
            return ax

    spans = []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        vals = np.unique(np.take(mesh, 0, axis=other))
        if vals.size > 1 and np.all(vals > 0):
            spans.append((np.log10(vals.max()) - np.log10(vals.min()), ax))
    if spans:
        return max(spans)[1]
    return 0


def _iter_crir_slices(shape, crir_axis):
    """Yield (advanced_index, env_meta) for each fixed-environment CRIR curve."""
    other_axes = [a for a in range(3) if a != crir_axis]
    for idxs in itertools.product(*(range(shape[a]) for a in other_axes)):
        index: List[Union[int, slice]] = [0, 0, 0]
        index[crir_axis] = slice(None)
        env = {}
        for ax, idx in zip(other_axes, idxs):
            index[ax] = idx
            env[ax] = idx
        yield tuple(index), env


def _is_crir_axis_key(key):
    z = str(key).lower()
    return any(token in z for token in ('crir', 'cosray', 'zeta'))


def _is_density_axis_key(key):
    z = str(key).lower()
    if z in ('n', 'densities', 'density', 'protdens'):
        return True
    return z.startswith('n_') or z.endswith('_n')


def _is_fuv_axis_key(key):
    z = str(key).lower()
    return any(token in z for token in (
        'fuv', 'radm', 'habing', 'draine', 'g_0', 'g0', 'chi_0'))


def coord_keys_for_grid(grid_data):
    """Return (density_key, crir_key, fuv_key) present in a species grid dict."""
    keys = set(grid_data.keys()) - {'grid'}
    density_key = next((k for k in keys if _is_density_axis_key(k)), None)
    crir_key = next((k for k in keys if _is_crir_axis_key(k)), None)
    fuv_key = next((k for k in keys if _is_fuv_axis_key(k)), None)
    return density_key, crir_key, fuv_key


def axis_display_label(axis_key, *, log=False, html=True):
    """Axis label with units.

    ``html=True``: Plotly hover (HTML).  ``html=False``: MathJax ``$...$`` for
    figure titles, ticks, and Dash Markdown labels.
    """
    key = str(axis_key or '')
    if html:
        if _is_density_axis_key(key):
            inner, unit = 'n<sub>H</sub>', 'cm<sup>−3</sup>'
        elif _is_fuv_axis_key(key):
            inner, unit = 'G<sub>0</sub>', 'Draine'
        elif _is_crir_axis_key(key):
            inner, unit = 'ζ', 's<sup>−1</sup>'
        elif 'mass' in key.lower():
            inner, unit = 'M', 'M<sub>☉</sub>'
        else:
            inner, unit = key, ''
        core = f'log<sub>10</sub> {inner}' if log else inner
        if unit:
            return f'{core} ({unit})'
        return core

    if _is_density_axis_key(key):
        inner, unit = r'n_{\mathrm{H}}', r'\mathrm{cm}^{-3}'
    elif _is_fuv_axis_key(key):
        inner, unit = r'G_0', r'\mathrm{Draine}'
    elif _is_crir_axis_key(key):
        inner, unit = r'\zeta', r'\mathrm{s}^{-1}'
    elif 'mass' in key.lower():
        inner, unit = r'M', r'\mathrm{M}_{\odot}'
    else:
        inner, unit = key.replace('_', r'\_'), ''
    core = rf'\log_{{10}} {inner}' if log else inner
    if unit:
        return rf'${core}~({unit})$'
    return rf'${core}$'


def probe_score(
    crir,
    abundance,
    *,
    min_abundance=DEFAULT_MIN_ABUNDANCE,
    min_points=DEFAULT_MIN_POINTS,
    min_log_crir_span=DEFAULT_MIN_LOG_CRIR_SPAN,
    plateau_slope=DEFAULT_PLATEAU_SLOPE,
    variation_r_cap=DEFAULT_VARIATION_R_CAP,
    crir_range=None,
    compute_spearman=False,
):
    """Probe score for one quantity-vs-CRIR curve (optionally one regime)."""
    crir = np.asarray(crir, dtype=float)
    abundance = np.asarray(abundance, dtype=float)
    valid = np.isfinite(crir) & np.isfinite(abundance) & (crir > 0) & (abundance > 0)
    if crir_range is not None:
        lo, hi = crir_range
        valid &= (crir >= lo) & (crir <= hi)
    crir = crir[valid]
    abundance = abundance[valid]

    nan_result = {**_NAN_SCORE, 'n_points': int(crir.size)}

    if crir.size < min_points:
        return {**nan_result, 'status': 'too_few_points'}

    if np.nanmax(abundance) < min_abundance:
        return {
            **nan_result,
            'probe_score': 0.0,
            'functional_factor': 0.0,
            'variation_factor': 0.0,
            'trend': 'not_observable',
            'observable': False,
            'status': 'not_observable',
        }

    order = np.argsort(crir)
    log_x = np.log10(crir[order])
    log_y = np.log10(abundance[order])

    log_span = float(log_x.max() - log_x.min())
    if log_span < min_log_crir_span:
        return {**nan_result, 'status': 'too_narrow_span'}

    log_dynamic_range = float(log_y.max() - log_y.min())
    x_var = log_dynamic_range / variation_r_cap if variation_r_cap > 0 else 0.0
    variation_factor = float(x_var / (1.0 + x_var) if x_var > 1.0 else min(x_var, 1.0))

    rho = np.nan
    if compute_spearman:
        with np.errstate(invalid='ignore'):
            if not np.allclose(log_y, log_y[0]):
                rho, _ = spearmanr(log_x, log_y)
    net_log_slope = float((log_y[-1] - log_y[0]) / log_span) if log_span > 0 else 0.0

    dx = np.diff(log_x)
    dy = np.diff(log_y)
    with np.errstate(divide='ignore', invalid='ignore'):
        local_slope = np.divide(dy, dx, out=np.zeros_like(dy), where=dx > 0)

    active_steps = np.abs(local_slope) >= plateau_slope
    point_active = np.zeros(log_x.size, dtype=bool)
    point_active[:-1] |= active_steps
    point_active[1:] |= active_steps

    if not np.any(point_active):
        return {
            'probe_score': 0.0,
            'functional_factor': 0.0,
            'variation_factor': variation_factor,
            'trend': 'plateau',
            'n_reversals': 0,
            'log_dynamic_range': log_dynamic_range,
            'spearman_rho': float(rho) if np.isfinite(rho) else np.nan,
            'net_log_slope': net_log_slope,
            'observable': True,
            'n_points': int(crir.size),
            'status': 'plateau',
        }

    segments: List[Tuple[str, float]] = []
    i = 0
    while i < log_x.size:
        if not point_active[i]:
            i += 1
            continue
        j = i
        while j < log_x.size and point_active[j]:
            j += 1
        lx, ly = log_x[i:j], log_y[i:j]
        if lx.size >= 2:
            ts_slope = _theil_slope(ly, lx)
            segments.append((_classify_trend(float(ts_slope), plateau_slope),
                             float(lx.max() - lx.min())))
        i = j

    trends = [t for t, _ in segments if t != 'flat']
    n_reversals = sum(1 for k in range(1, len(trends)) if trends[k] != trends[k - 1])

    usable_span = sum(span for t, span in segments if t != 'flat')
    functional_factor = float(usable_span / log_span) if log_span > 0 else 0.0
    if n_reversals > 1:
        functional_factor *= 0.5

    score = functional_factor * variation_factor
    if trends:
        trend = trends[0] if len(set(trends)) == 1 else 'mixed'
    else:
        trend = 'plateau'

    return {
        'probe_score': float(score),
        'functional_factor': functional_factor,
        'variation_factor': variation_factor,
        'trend': trend,
        'n_reversals': int(n_reversals),
        'log_dynamic_range': log_dynamic_range,
        'spearman_rho': float(rho) if np.isfinite(rho) else np.nan,
        'net_log_slope': net_log_slope,
        'observable': True,
        'n_points': int(crir.size),
        'status': 'ok' if float(score) > 0 else 'plateau',
    }


def plateau_spans_crir(crir, abundance, plateau_slope=DEFAULT_PLATEAU_SLOPE):
    """Return (xmin, xmax) spans where the curve is locally plateau."""
    crir = np.asarray(crir, dtype=float)
    abundance = np.asarray(abundance, dtype=float)
    valid = np.isfinite(crir) & np.isfinite(abundance) & (crir > 0) & (abundance > 0)
    crir, abundance = crir[valid], abundance[valid]
    if crir.size < 2:
        return []
    order = np.argsort(crir)
    log_x = np.log10(crir[order])
    log_y = np.log10(abundance[order])
    spans = []
    for i in range(len(log_x) - 1):
        dx = log_x[i + 1] - log_x[i]
        if dx <= 0:
            continue
        slope = (log_y[i + 1] - log_y[i]) / dx
        if abs(slope) < plateau_slope:
            spans.append((10.0 ** log_x[i], 10.0 ** log_x[i + 1]))
    return spans


def absolute_quartile_label(score):
    """Absolute score band labels (KoSens ``Absolute_Quartile``)."""
    if not np.isfinite(score) or score <= 0:
        return 'gray (plateau / zero)'
    if score >= 0.59:
        return 'green (Q1)'
    if score >= 0.39:
        return 'yellow (Q2)'
    if score >= 0.20:
        return 'orange (Q3)'
    return 'red (Q4)'


def absolute_score_band(score):
    """Absolute score band (independent of other tracers)."""
    if not np.isfinite(score) or score <= 0:
        return 'inactive'
    if score >= 0.59:
        return 'strong'
    if score >= 0.39:
        return 'good'
    if score >= 0.20:
        return 'moderate'
    return 'weak'


def assign_quartiles(scores):
    """Map scores to relative quartile labels among observable (score > 0) values."""
    labels = ['gray'] * len(scores)
    finite = [(i, float(s)) for i, s in enumerate(scores)
              if np.isfinite(s) and s > 0]
    if not finite:
        return labels
    vals = np.array([s for _, s in finite], dtype=float)
    q1, q2, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
    for i, val in finite:
        if val >= q3:
            labels[i] = 'green'
        elif val >= q2:
            labels[i] = 'yellow'
        elif val >= q1:
            labels[i] = 'orange'
        else:
            labels[i] = 'red'
    return labels


def _median(values):
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def _mean(values):
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def _quantile(values, q):
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else np.nan


def _safe_nanmax(values):
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else np.nan


def _best_score_row(rows, key='probe_score'):
    best, best_s = rows[0], -np.inf
    found = False
    for row in rows:
        s = row.get(key, np.nan)
        if np.isfinite(s) and s > best_s:
            best, best_s, found = row, float(s), True
    return best if found else rows[0]


def _merge_score_status(statuses):
    """Keep the most informative status when several curves collapse to one cell."""
    rank = {
        'ok': 0,
        'plateau': 1,
        'not_observable': 2,
        'too_narrow_span': 3,
        'too_few_points': 4,
        'unknown': 5,
    }
    best, best_rank = 'unknown', 99
    for raw in statuses:
        key = str(raw or 'unknown')
        r = rank.get(key, 5)
        if r < best_rank:
            best, best_rank = key, r
    return best


def _resolve_species_list(grids_3d, species_list, include_ratios):
    skip = {'_fit_axis_config', '_meta'}
    names = [k for k, v in grids_3d.items()
             if k not in skip and isinstance(v, dict) and 'grid' in v]
    if species_list is not None:
        return [s for s in species_list if s in grids_3d]
    if include_ratios:
        return names
    return [s for s in names if '/' not in s]


def _resolve_environment_levels(requested, available):
    """Map requested n or FUV values to nearest native log10 grid nodes."""
    available = np.sort(np.unique([
        _snap_native_level(v) for v in available
        if _snap_native_level(v) is not None
    ]))
    if available.size == 0:
        return []
    resolved = []
    seen = set()
    for val in requested:
        native = _snap_native_level(val)
        if native is None:
            continue
        if native not in available:
            idx = int(np.argmin(np.abs(np.log10(available) - np.log10(native))))
            native = float(available[idx])
        if native not in seen:
            seen.add(native)
            resolved.append(native)
    return resolved


def _nearest_levels(requested, available):
    available = np.sort(np.unique(np.asarray(available, dtype=float)))
    available = available[np.isfinite(available) & (available > 0)]
    if available.size == 0:
        return []
    resolved = []
    seen = set()
    for val in requested:
        val = float(val)
        if not np.isfinite(val) or val <= 0:
            continue
        idx = int(np.argmin(np.abs(np.log10(available) - np.log10(val))))
        native = float(available[idx])
        if native not in seen:
            seen.add(native)
            resolved.append(native)
    return resolved


def _pick_span_levels(available, n_pick=3):
    """Pick up to ``n_pick`` values spanning a positive axis (log-spaced)."""
    available = np.sort(np.unique(np.asarray(available, dtype=float)))
    available = available[np.isfinite(available) & (available > 0)]
    if available.size == 0:
        return []
    if available.size <= n_pick:
        return [float(v) for v in available]
    idx = np.linspace(0, available.size - 1, n_pick, dtype=int)
    return [float(available[i]) for i in idx]


def _has_level_selection(values):
    """True when the caller asked for specific axis values (not 'use defaults')."""
    if values is None:
        return False
    if isinstance(values, (str, bytes)):
        return bool(str(values).strip())
    try:
        return len(values) > 0
    except TypeError:
        return True


def _levels_close(a, b):
    a_level = _snap_native_level(a)
    b_level = _snap_native_level(b)
    if a_level is not None and b_level is not None:
        return a_level == b_level
    return bool(np.isclose(a, b, rtol=LEVEL_MATCH_RTOL, atol=0.0))


def _attach_native_levels(row):
    """Ensure ``n_level`` / ``fuv_level`` are present on a slice or env row."""
    out = dict(row)
    if 'n_level' not in out or not np.isfinite(out.get('n_level', np.nan)):
        out['n_level'] = _snap_native_level(out.get('n'))
    if 'fuv_level' not in out or not np.isfinite(out.get('fuv_level', np.nan)):
        out['fuv_level'] = _snap_native_level(out.get('fuv'))
    return out


def _heatmap_pivot(panel, value_col='probe_score'):
    """Median probe score on native (n, G₀) nodes — KoSens ``pivot_table``."""
    if not panel:
        return np.array([]), np.array([]), np.full((0, 0), np.nan)
    rows = [_attach_native_levels(r) for r in panel]
    by_cell: Dict[Tuple[float, float], List[float]] = {}
    for row in rows:
        n_level = row.get('n_level')
        fuv_level = row.get('fuv_level')
        val = row.get(value_col, np.nan)
        if (n_level is None or fuv_level is None
                or not np.isfinite(val)):
            continue
        key = (float(n_level), float(fuv_level))
        by_cell.setdefault(key, []).append(float(val))
    if not by_cell:
        return np.array([]), np.array([]), np.full((0, 0), np.nan)
    n_vals = np.sort(np.unique([k[0] for k in by_cell]))
    fuv_vals = np.sort(np.unique([k[1] for k in by_cell]))
    z = np.full((n_vals.size, fuv_vals.size), np.nan)
    for iy, n in enumerate(n_vals):
        for ix, fuv in enumerate(fuv_vals):
            vals = by_cell.get((float(n), float(fuv)))
            if vals:
                z[iy, ix] = float(np.median(vals))
    return n_vals, fuv_vals, z


def compute_probe_ranking(
    grids_3d,
    *,
    species_list=None,
    include_ratios=True,
    crir_axis_key=None,
    crir_regimes=None,
    min_abundance=DEFAULT_MIN_ABUNDANCE,
    min_points=DEFAULT_MIN_POINTS,
    min_log_crir_span=DEFAULT_MIN_LOG_CRIR_SPAN,
    plateau_slope=DEFAULT_PLATEAU_SLOPE,
    variation_r_cap=DEFAULT_VARIATION_R_CAP,
    good_score_threshold=DEFAULT_GOOD_SCORE_THRESHOLD,
    progress=None,
):
    """Regime-aware CRIR probe ranking over all environments in a 3-D grid.

    Returns a dict of list-of-dict tables plus axis / scoring metadata.
    ``progress`` is an optional ``(frac, message)`` callback, with ``frac``
    in [0, 1] over the scoring loop.
    """
    if not grids_3d:
        raise ValueError('grids_3d is empty')

    species = _resolve_species_list(grids_3d, species_list, include_ratios)
    if not species:
        raise ValueError('No valid tracers found in the 3-D grid')

    ref = grids_3d[species[0]]
    density_key, crir_key_auto, fuv_key = coord_keys_for_grid(ref)
    crir_key = crir_axis_key or crir_key_auto
    if crir_key is None or crir_key not in ref:
        raise ValueError(
            'Could not resolve the CRIR axis in the grid. '
            f'Available keys: {list(ref.keys())}'
        )

    crir_axis = _coord_axis_index(ref[crir_key])
    grid_shape = ref['grid'].shape
    crir_all = np.asarray(ref[crir_key], dtype=float)
    crir_finite = crir_all[np.isfinite(crir_all) & (crir_all > 0)]
    if crir_finite.size == 0:
        raise ValueError('CRIR axis contains no positive finite values')
    crir_min = float(crir_finite.min())
    crir_max = float(crir_finite.max())
    regimes = build_crir_regimes(crir_regimes, crir_min, crir_max)

    if density_key is None:
        density_key = next(
            (k for k in ref if k not in ('grid', crir_key, fuv_key)),
            None,
        )

    slice_rows: List[dict] = []
    n_sp = len(species)
    for i_sp, sp in enumerate(species):
        if callable(progress):
            progress(i_sp / max(n_sp, 1), f'Scoring {sp} ({i_sp + 1}/{n_sp})')
        gdata = grids_3d[sp]
        grid = np.asarray(gdata['grid'], dtype=float)
        crir_mesh = np.asarray(gdata[crir_key], dtype=float)
        for index, env in _iter_crir_slices(grid.shape, crir_axis):
            curve = np.asarray(grid[index], dtype=float).ravel()
            crir_vals = np.asarray(crir_mesh[index], dtype=float).ravel()
            n_val = float(gdata[density_key][index].ravel()[0]) if density_key else np.nan
            fuv_val = float(gdata[fuv_key][index].ravel()[0]) if fuv_key else np.nan
            for regime in regimes:
                score = probe_score(
                    crir_vals, curve,
                    min_abundance=min_abundance,
                    min_points=min_points,
                    min_log_crir_span=min_log_crir_span,
                    plateau_slope=plateau_slope,
                    variation_r_cap=variation_r_cap,
                    crir_range=(regime['lo'], regime['hi']),
                )
                if not np.isfinite(score['probe_score']):
                    continue
                n_level = _snap_native_level(n_val)
                fuv_level = _snap_native_level(fuv_val)
                slice_rows.append({
                    'Species': sp,
                    'Regime': regime['name'],
                    'Regime_Label': regime['label'],
                    'probe_score': score['probe_score'],
                    'functional_factor': score['functional_factor'],
                    'variation_factor': score['variation_factor'],
                    'trend': score['trend'],
                    'n_reversals': score['n_reversals'],
                    'log_dynamic_range': score['log_dynamic_range'],
                    'spearman_rho': score['spearman_rho'],
                    'net_log_slope': score['net_log_slope'],
                    'observable': score['observable'],
                    'n': float(n_level if n_level is not None else n_val),
                    'fuv': float(fuv_level if fuv_level is not None else fuv_val),
                    'n_level': n_level,
                    'fuv_level': fuv_level,
                    'n_points': score['n_points'],
                    'score_status': score.get('status', 'unknown'),
                })

    if callable(progress):
        progress(1.0, 'Building ranking tables')
    if not slice_rows:
        raise ValueError('No valid CRIR curves found for probe ranking')

    # One score per (species, regime, native n, native FUV).  Interpolated
    # cubes can map many points to the same log10 bin — take the median.
    grouped: Dict[Tuple, List[dict]] = {}
    for row in slice_rows:
        row = _attach_native_levels(row)
        key = (
            row['Species'],
            row['Regime'],
            row.get('n_level'),
            row.get('fuv_level'),
        )
        grouped.setdefault(key, []).append(row)

    slice_df: List[dict] = []
    for rows in grouped.values():
        base = dict(rows[0])
        base['probe_score'] = _median(r['probe_score'] for r in rows)
        base['functional_factor'] = _median(r['functional_factor'] for r in rows)
        base['variation_factor'] = _median(r['variation_factor'] for r in rows)
        base['log_dynamic_range'] = _median(r['log_dynamic_range'] for r in rows)
        base['spearman_rho'] = _median(r['spearman_rho'] for r in rows)
        base['net_log_slope'] = _median(r['net_log_slope'] for r in rows)
        base['n_reversals'] = int(max(r['n_reversals'] for r in rows))
        base['observable'] = any(r['observable'] for r in rows)
        base['score_status'] = _merge_score_status(r.get('score_status') for r in rows)
        base['n'] = float(base['n_level']) if base.get('n_level') is not None else base.get('n')
        base['fuv'] = float(base['fuv_level']) if base.get('fuv_level') is not None else base.get('fuv')
        slice_df.append(base)

    summary_df = _summary_from_slice(slice_df)
    environment_rankings_df = environment_rankings_from_slice(
        slice_df, all_environments=True,
    )
    robustness_df = build_robustness_summary(
        slice_df, good_score_threshold=good_score_threshold,
    )
    regime_top_df = top_tracers_per_regime(environment_rankings_df, top_n=5)

    n_levels = sorted({
        r['n_level'] for r in slice_df
        if r.get('n_level') is not None and np.isfinite(r['n_level']) and r['n_level'] > 0
    })
    fuv_levels = sorted({
        r['fuv_level'] for r in slice_df
        if r.get('fuv_level') is not None and np.isfinite(r['fuv_level']) and r['fuv_level'] > 0
    })

    return {
        'environment_rankings': environment_rankings_df,
        'regime_top': regime_top_df,
        'robustness': robustness_df,
        'summary': summary_df,
        'slice_scores': slice_df,
        'species_list': species,
        'crir_regimes': regimes,
        'crir_axis_key': crir_key,
        'density_key': density_key,
        'fuv_key': fuv_key,
        'crir_axis_index': crir_axis,
        'plateau_slope': float(plateau_slope),
        'good_score_threshold': float(good_score_threshold),
        'min_abundance': float(min_abundance),
        'min_points': int(min_points),
        'min_log_crir_span': float(min_log_crir_span),
        'n_levels': n_levels,
        'fuv_levels': fuv_levels,
        'crir_min': crir_min,
        'crir_max': crir_max,
    }


def _summary_from_slice(slice_df):
    by_regime: Dict[str, List[dict]] = {}
    for row in slice_df:
        by_regime.setdefault(row['Regime'], []).append(row)

    summary = []
    for regime_name, regime_rows in by_regime.items():
        by_sp: Dict[str, List[dict]] = {}
        for row in regime_rows:
            by_sp.setdefault(row['Species'], []).append(row)
        part = []
        for sp, grp in by_sp.items():
            ps = [r['probe_score'] for r in grp]
            part.append({
                'Species': sp,
                'Regime': regime_name,
                'Regime_Label': grp[0]['Regime_Label'],
                'Median_Probe_Score': _median(ps),
                'Mean_Probe_Score': _mean(ps),
                'P25_Probe_Score': _quantile(ps, 0.25),
                'P75_Probe_Score': _quantile(ps, 0.75),
                'Max_Probe_Score': _safe_nanmax(ps),
                'Frac_Strong_Environments': float(np.mean([s >= 0.59 for s in ps])),
                'Frac_Good_Environments': float(np.mean([s >= 0.39 for s in ps])),
                'Frac_Observable': float(np.mean([bool(r['observable']) for r in grp])),
                'Frac_Monotonic': float(np.mean([
                    r['trend'] in ('increasing', 'decreasing') for r in grp])),
                'Median_Functional': _median(r['functional_factor'] for r in grp),
                'Median_Variation': _median(r['variation_factor'] for r in grp),
                'Median_Spearman_Rho': _median(r['spearman_rho'] for r in grp),
                'Median_Net_Log_Slope': _median(r['net_log_slope'] for r in grp),
                'Median_Log_Dynamic_Range': _median(r['log_dynamic_range'] for r in grp),
                'N_Environments': len(grp),
                'Is_Ratio': '/' in sp,
            })
        scores = [r['Median_Probe_Score'] for r in part]
        quartiles = assign_quartiles(scores)
        for row, q in zip(part, quartiles):
            row['Quartile'] = q
        part.sort(key=lambda r: (-(r['Median_Probe_Score']
                                   if np.isfinite(r['Median_Probe_Score']) else -1),
                                 r['Species']))
        for i, row in enumerate(part, 1):
            row['Rank'] = i
        summary.extend(part)
    return summary


def environment_rankings_from_slice(
    slice_df,
    *,
    n_levels=None,
    fuv_levels=None,
    all_environments=False,
    species_only=False,
    ratios_only=False,
):
    """Per-(regime, n, FUV) tracer rankings from deduplicated slice scores."""
    if not slice_df:
        return []

    work = [_attach_native_levels(r) for r in slice_df]
    if all_environments or (n_levels is None and fuv_levels is None):
        n_use = sorted({
            r['n_level'] for r in work
            if r.get('n_level') is not None and np.isfinite(r['n_level'])
        })
        fuv_use = sorted({
            r['fuv_level'] for r in work
            if r.get('fuv_level') is not None and np.isfinite(r['fuv_level'])
        })
    else:
        avail_n = [r['n_level'] for r in work if r.get('n_level') is not None]
        avail_fuv = [r['fuv_level'] for r in work if r.get('fuv_level') is not None]
        n_use = _resolve_environment_levels(n_levels or [1e1, 1e3, 1e5], avail_n)
        fuv_use = _resolve_environment_levels(fuv_levels or [1e1, 1e2, 1e5], avail_fuv)

    by_regime: Dict[str, List[dict]] = {}
    for row in work:
        by_regime.setdefault(row['Regime'], []).append(row)

    out = []
    for regime_name, regime_rows in by_regime.items():
        regime_label = regime_rows[0]['Regime_Label']
        for n in n_use:
            for fuv in fuv_use:
                env = [
                    r for r in regime_rows
                    if r.get('n_level') == n and r.get('fuv_level') == fuv
                ]
                if species_only:
                    env = [r for r in env if '/' not in r['Species']]
                elif ratios_only:
                    env = [r for r in env if '/' in r['Species']]
                if not env:
                    continue
                ranked = sorted(
                    env,
                    key=lambda r: (-r['probe_score'] if np.isfinite(r['probe_score']) else -1,
                                   r['Species']),
                )
                seen = set()
                unique = []
                for row in ranked:
                    if row['Species'] in seen:
                        continue
                    seen.add(row['Species'])
                    unique.append(row)
                quartiles = assign_quartiles([r['probe_score'] for r in unique])
                for rank, (row, q) in enumerate(zip(unique, quartiles), 1):
                    score = float(row['probe_score'])
                    out.append({
                        'Species': row['Species'],
                        'Regime': regime_name,
                        'Regime_Label': regime_label,
                        'n': float(n),
                        'fuv': float(fuv),
                        'n_level': float(n),
                        'fuv_level': float(fuv),
                        'probe_score': score,
                        'functional_factor': float(row.get('functional_factor', np.nan)),
                        'variation_factor': float(row.get('variation_factor', np.nan)),
                        'trend': str(row.get('trend', '')),
                        'n_reversals': int(row.get('n_reversals', 0)),
                        'log_dynamic_range': float(row.get('log_dynamic_range', np.nan)),
                        'spearman_rho': float(row.get('spearman_rho', np.nan)),
                        'Quartile': q,
                        'Absolute_Quartile': absolute_quartile_label(score),
                        'Score_Band': absolute_score_band(score),
                        'Rank': rank,
                        'Is_Ratio': '/' in row['Species'],
                        'score_status': row.get('score_status', 'unknown'),
                    })
    return out


def ranking_at_environments(
    ranking_results,
    n_values=None,
    fuv_values=None,
    *,
    all_environments=False,
    species_only=False,
    ratios_only=False,
):
    """Filter or rebuild per-environment rankings at selected (n, FUV) nodes."""
    env_df = ranking_results.get('environment_rankings') or []
    if env_df and not all_environments:
        n_sel = _has_level_selection(n_values)
        fuv_sel = _has_level_selection(fuv_values)
        if not n_sel and not fuv_sel:
            sub = list(env_df)
        else:
            n_levels = (_nearest_levels(n_values, [r['n'] for r in env_df])
                        if n_sel else None)
            fuv_levels = (_nearest_levels(fuv_values, [r['fuv'] for r in env_df])
                          if fuv_sel else None)
            sub = [
                r for r in env_df
                if (n_levels is None or any(_levels_close(r['n'], n) for n in n_levels))
                and (fuv_levels is None or any(_levels_close(r['fuv'], fuv)
                                               for fuv in fuv_levels))
            ]
        if species_only:
            sub = [r for r in sub if not r['Is_Ratio']]
        elif ratios_only:
            sub = [r for r in sub if r['Is_Ratio']]
        return sub

    slice_df = ranking_results.get('slice_scores') or []
    return environment_rankings_from_slice(
        slice_df,
        n_levels=n_values,
        fuv_levels=fuv_values,
        all_environments=all_environments,
        species_only=species_only,
        ratios_only=ratios_only,
    )


def build_robustness_summary(slice_df, *, good_score_threshold=DEFAULT_GOOD_SCORE_THRESHOLD):
    """Species-level reliability across environments, per CRIR regime."""
    if not slice_df:
        return []
    work = [_attach_native_levels(r) for r in slice_df]
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for row in work:
        groups.setdefault((row['Regime'], row['Species']), []).append(row)

    out = []
    for (regime_name, sp), grp in groups.items():
        ps = np.array([r['probe_score'] for r in grp], dtype=float)
        best = _best_score_row(grp)
        n_good = int(np.sum(np.isfinite(ps) & (ps >= good_score_threshold)))
        best_n = best.get('n_level', best.get('n'))
        best_fuv = best.get('fuv_level', best.get('fuv'))
        out.append({
            'Species': sp,
            'Regime': regime_name,
            'Regime_Label': grp[0]['Regime_Label'],
            'Peak_Probe_Score': _safe_nanmax(ps),
            'Best_n': float(best_n) if best_n is not None else np.nan,
            'Best_fuv': float(best_fuv) if best_fuv is not None else np.nan,
            'Median_Probe_Score': _median(ps),
            'Frac_Good_Environments': float(n_good / len(grp)),
            'N_Good_Environments': n_good,
            'N_Environments': int(len(grp)),
            'Is_Ratio': '/' in sp,
        })
    out.sort(key=lambda r: (r['Regime'],
                            -(r['Peak_Probe_Score'] if np.isfinite(r['Peak_Probe_Score']) else -1),
                            -r['Frac_Good_Environments'],
                            r['Species']))
    return out


def top_tracers_per_regime(environment_rankings, *, top_n=5,
                           species_only=False, ratios_only=False):
    sub = list(environment_rankings or [])
    if species_only:
        sub = [r for r in sub if not r['Is_Ratio']]
    elif ratios_only:
        sub = [r for r in sub if r['Is_Ratio']]
    return [r for r in sub if r['Rank'] <= top_n]


def extract_curve_at_environment(
    grids_3d, species, n_level, fuv_level, *, crir_axis_key=None,
):
    """Deduplicated quantity-vs-CRIR curve at one native (n, FUV)."""
    if species not in grids_3d:
        return np.array([]), np.array([])
    gdata = grids_3d[species]
    density_key, crir_key_auto, fuv_key = coord_keys_for_grid(gdata)
    crir_key = crir_axis_key or crir_key_auto
    if crir_key is None or density_key is None or fuv_key is None:
        return np.array([]), np.array([])

    crir_axis = _coord_axis_index(gdata[crir_key])
    n_tgt = _snap_native_level(n_level)
    fuv_tgt = _snap_native_level(fuv_level)
    if n_tgt is None:
        n_tgt = float(n_level)
    if fuv_tgt is None:
        fuv_tgt = float(fuv_level)

    crir_pts, y_pts = [], []
    for index, _ in _iter_crir_slices(gdata['grid'].shape, crir_axis):
        n_v = _snap_native_level(float(np.asarray(gdata[density_key][index]).ravel()[0]))
        f_v = _snap_native_level(float(np.asarray(gdata[fuv_key][index]).ravel()[0]))
        if n_v is None or f_v is None:
            continue
        if not _levels_close(n_v, n_tgt) or not _levels_close(f_v, fuv_tgt):
            continue
        c = np.asarray(gdata[crir_key][index], dtype=float).ravel()
        y = np.asarray(gdata['grid'][index], dtype=float).ravel()
        valid = np.isfinite(c) & np.isfinite(y) & (c > 0) & (y > 0)
        crir_pts.extend(c[valid].tolist())
        y_pts.extend(y[valid].tolist())

    if not crir_pts:
        return np.array([]), np.array([])

    # Median-combine duplicate CRIR nodes.
    order = np.argsort(crir_pts)
    c = np.asarray(crir_pts, dtype=float)[order]
    y = np.asarray(y_pts, dtype=float)[order]
    uniq, inv = np.unique(np.round(np.log10(c), decimals=8), return_inverse=True)
    y_med = np.array([np.median(y[inv == i]) for i in range(uniq.size)])
    c_med = 10.0 ** uniq
    return c_med, y_med


def lookup_env_score(env_table, species, n_level, fuv_level, regime=None):
    empty = {
        'probe_score': np.nan,
        'functional_factor': np.nan,
        'variation_factor': np.nan,
        'trend': '',
        'Quartile': 'gray',
        'Score_Band': 'inactive',
    }
    if not env_table:
        return empty
    for row in env_table:
        if row['Species'] != species:
            continue
        if not _levels_close(row.get('n'), n_level):
            continue
        if not _levels_close(row.get('fuv'), fuv_level):
            continue
        if regime is not None and row.get('Regime') != regime:
            continue
        return {
            'probe_score': float(row['probe_score']),
            'functional_factor': float(row.get('functional_factor', np.nan)),
            'variation_factor': float(row.get('variation_factor', np.nan)),
            'trend': str(row.get('trend', '')),
            'Quartile': str(row.get('Quartile', 'gray')),
            'Score_Band': str(row.get('Score_Band', absolute_score_band(row['probe_score']))),
        }
    return empty


def add_ratio_grids(grids_3d, pairs):
    """In-place: add ``num/den`` cubes for each (num, den) pair."""
    if not pairs:
        return grids_3d
    ref_name = next(
        (k for k, v in grids_3d.items()
         if isinstance(v, dict) and 'grid' in v),
        None,
    )
    if ref_name is None:
        return grids_3d
    mesh_keys = [k for k in grids_3d[ref_name] if k != 'grid']
    for num, den in pairs:
        name = f'{num}/{den}'
        if name in grids_3d or num not in grids_3d or den not in grids_3d:
            continue
        a = np.asarray(grids_3d[num]['grid'], dtype=float)
        b = np.asarray(grids_3d[den]['grid'], dtype=float)
        out = np.full(a.shape, np.nan, dtype=float)
        ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
        np.divide(a, b, out=out, where=ok)
        entry = {'grid': out}
        for k in mesh_keys:
            if k in grids_3d[num]:
                entry[k] = grids_3d[num][k]
        grids_3d[name] = entry
    return grids_3d


def _theme_fallback(theme):
    if isinstance(theme, dict) and 'paper_bg' in theme:
        return theme
    return dict(
        paper_bg='#ffffff', plot_bg='#ffffff', grid='#e8edf3',
        axis_line='#cbd5e1', title='#0f172a', font='#1e293b',
        muted='#64748b', legend_bg='rgba(255,255,255,0.92)',
        legend_border='#e2e8f0', accent='#2563eb',
        placeholder='#94a3b8', placeholder_plot='#f1f4f8',
    )


def _axis_style(t):
    return dict(
        showgrid=True, gridcolor=t['grid'], gridwidth=1,
        zeroline=False, linecolor=t['axis_line'], mirror=True,
        exponentformat='power', showexponent='all',
        tickfont=ps.tick_font(t['font']),
    )


def _apply_log_ticks(axis_dict, levels, max_ticks=6):
    """Decade ticks on a log axis via Plotly's native 10^n (no MathJax ticktext).

    ``$10^{n}$`` in ``ticktext`` is drawn *on top of* Plotly's own log labels.
    """
    levels = np.sort(np.unique(np.asarray(levels, dtype=float)))
    levels = levels[np.isfinite(levels) & (levels > 0)]
    if levels.size == 0:
        return axis_dict
    if levels.size > max_ticks:
        idx = np.linspace(0, levels.size - 1, max_ticks, dtype=int)
        ticks = levels[idx]
    else:
        ticks = levels
    axis_dict['tickmode'] = 'array'
    axis_dict['tickvals'] = [float(v) for v in ticks]
    axis_dict.pop('ticktext', None)
    axis_dict['exponentformat'] = 'power'
    axis_dict['showexponent'] = 'all'
    return axis_dict


def _index_decade_axis(levels, title, t):
    """Equal-width heatmap axis with plain decade labels (no MathJax).

    ``go.Heatmap`` + ``type='log'`` double-draws exponents. MathJax
    ``$10^{n}$`` ticktext is drawn on top of Plotly's own tick numbers.
    """
    levels = np.asarray(levels, dtype=float)
    n = int(levels.size)
    style = {k: v for k, v in _axis_style(t).items()
             if k not in ('exponentformat', 'showexponent')}
    return dict(
        **style,
        type='linear',
        title=dict(text=title, font=ps.axis_title_font(t['font'])),
        tickmode='array',
        tickvals=list(range(n)),
        ticktext=[sci_plain(v) for v in levels],
        range=[-0.5, n - 0.5] if n else None,
        constrain='domain',
        automargin=True,
        exponentformat='none',
        showexponent='none',
    )


def _short_label(name, max_len=22):
    name = str(name)
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + '…'


def _color_for_row(row, color_mode):
    if color_mode == 'absolute':
        return ABSOLUTE_BAND_COLORS.get(row.get('Score_Band', 'inactive'), '#888')
    return QUARTILE_COLORS.get(row.get('Quartile', 'gray'), '#888')


def _as_name_list(value):
    if value is None or value == '':
        return []
    if isinstance(value, (str, bytes)):
        text = str(value)
        return [] if text == 'all' else [text]
    out = []
    for item in value:
        if item in (None, '', 'all'):
            continue
        out.append(str(item))
    return out


def _ordered_regimes(ranking_results, regime=None):
    """Regime dicts to plot, in increasing-ζ order.

    ``regime`` may be a name, a list of names, ``None``, or ``'all'``.
    Always taken from scoring metadata so a CRIR range with no finite
    scores still appears as its own panel.
    """
    regimes = list(ranking_results.get('crir_regimes') or [])
    regimes.sort(key=lambda r: (r.get('lo', 0.0), r.get('name', '')))
    wanted = _as_name_list(regime)
    if wanted:
        by_name = {r['name']: r for r in regimes}
        picked = []
        seen = set()
        for name in wanted:
            if name in seen:
                continue
            seen.add(name)
            if name in by_name:
                picked.append(by_name[name])
            else:
                picked.append({'name': name, 'label': name, 'lo': np.nan, 'hi': np.nan})
        return picked
    if regimes:
        return regimes
    present = list(dict.fromkeys(
        r.get('Regime') for r in (ranking_results.get('slice_scores') or [])
        if r.get('Regime')
    ))
    return [{'name': n, 'label': n, 'lo': np.nan, 'hi': np.nan} for n in present]


def explain_regime_plot(ranking_results, regime_name, *, species=None):
    """Human-readable reason a regime figure may look empty, or ``None``."""
    slice_df = ranking_results.get('slice_scores') or []
    rows = [r for r in slice_df if r.get('Regime') == regime_name]
    wanted = _as_name_list(species)
    if wanted:
        rows = [r for r in rows if r.get('Species') in wanted]
    min_pts = ranking_results.get('min_points', DEFAULT_MIN_POINTS)
    min_span = ranking_results.get('min_log_crir_span', DEFAULT_MIN_LOG_CRIR_SPAN)
    meta = next((r for r in (ranking_results.get('crir_regimes') or [])
                 if r.get('name') == regime_name), None)
    label = (meta or {}).get('label') or regime_name

    if not rows:
        return (
            f'No scores in {label}. This CRIR range usually has fewer than '
            f'{min_pts} ζ points or a log-span below {min_span} dex, so curves '
            f'cannot be scored.'
        )

    statuses = [r.get('score_status', '') for r in rows]
    finite = [r['probe_score'] for r in rows
              if np.isfinite(r.get('probe_score', np.nan))]
    positive = [s for s in finite if s > 0]
    if positive:
        return None

    bits = []
    n_few = sum(1 for s in statuses if s == 'too_few_points')
    n_nar = sum(1 for s in statuses if s == 'too_narrow_span')
    n_obs = sum(1 for r in rows if r.get('trend') == 'not_observable'
                or r.get('score_status') == 'not_observable')
    n_plat = sum(1 for r in rows if r.get('trend') == 'plateau'
                 or r.get('score_status') == 'plateau')
    if n_few:
        bits.append(f'too few ζ points (< {min_pts})')
    if n_nar:
        bits.append(f'ζ span too narrow (< {min_span} dex)')
    if n_obs:
        bits.append('below the observable cut')
    if n_plat or (finite and not positive):
        bits.append('flat / plateau (S = 0)')
    reason = ', '.join(bits) if bits else 'no tracer responds (S = 0)'
    return (
        f'Nothing informative to rank in {label}: {reason}. '
        f'Green / Q1 colours only appear when at least one tracer has S > 0 '
        f'in that CRIR range — relative ranking cannot pick a winner among all zeros.'
    )


def _empty_panel_message(rows, ranking_results=None):
    """Short in-panel note when a single (n, G₀) cell has no bars worth seeing."""
    min_pts = (ranking_results or {}).get('min_points', DEFAULT_MIN_POINTS)
    min_span = (ranking_results or {}).get('min_log_crir_span', DEFAULT_MIN_LOG_CRIR_SPAN)
    if not rows:
        return 'No scores in this environment for this CRIR regime.'
    statuses = [r.get('score_status', '') for r in rows]
    finite = [r['probe_score'] for r in rows
              if np.isfinite(r.get('probe_score', np.nan))]
    if not finite:
        if all(s == 'too_few_points' for s in statuses if s):
            return f'Not scored: fewer than {min_pts} ζ points in this CRIR range.'
        if all(s == 'too_narrow_span' for s in statuses if s):
            return f'Not scored: ζ span below {min_span} dex in this CRIR range.'
        return 'Not scored in this CRIR range (too few ζ points or too narrow a span).'
    if max(finite) <= 0:
        return 'All S = 0 (plateau or not observable). No green ranking in this panel.'
    return None


def environment_ranking_figures(
    ranking_results,
    *,
    regime=None,
    n_values=None,
    fuv_values=None,
    species_only=False,
    ratios_only=False,
    top_n=8,
    color_mode='quartile',
    theme=None,
):
    """One figure per selected CRIR regime (independent n_H × G₀ ranking grids)."""
    regimes = _ordered_regimes(ranking_results, regime)
    figs = []
    for spec in regimes:
        fig = fig_environment_ranking(
            ranking_results,
            regime=spec['name'],
            n_values=n_values,
            fuv_values=fuv_values,
            species_only=species_only,
            ratios_only=ratios_only,
            top_n=top_n,
            color_mode=color_mode,
            theme=theme,
        )
        if fig is not None:
            figs.append(fig)
    return figs


def fig_environment_ranking(
    ranking_results,
    *,
    regime=None,
    n_values=None,
    fuv_values=None,
    species_only=False,
    ratios_only=False,
    top_n=8,
    color_mode='quartile',
    theme=None,
):
    """One figure for one CRIR regime: rows = density, columns = FUV."""
    t = _theme_fallback(theme)
    env_table = ranking_at_environments(
        ranking_results,
        n_values=n_values,
        fuv_values=fuv_values,
        species_only=species_only,
        ratios_only=ratios_only,
    ) or []

    avail_n = ranking_results.get('n_levels') or [r['n'] for r in env_table]
    avail_fuv = ranking_results.get('fuv_levels') or [r['fuv'] for r in env_table]
    if _has_level_selection(n_values):
        n_levels = _resolve_environment_levels(n_values, avail_n)
    else:
        n_levels = default_kosens_env_values(avail_n, KOSENS_DEFAULT_N_ENV)
    if _has_level_selection(fuv_values):
        fuv_levels = _resolve_environment_levels(fuv_values, avail_fuv)
    else:
        fuv_levels = default_kosens_env_values(avail_fuv, KOSENS_DEFAULT_FUV_ENV)
    if not n_levels or not fuv_levels:
        return None

    picked = _ordered_regimes(ranking_results, regime)
    if not picked:
        return None
    spec = picked[0]
    rname, rlabel = spec['name'], spec.get('label') or spec['name']
    reg_table = [r for r in env_table if r.get('Regime') == rname]

    n_rows, n_cols = len(n_levels), len(fuv_levels)
    dens_key = ranking_results.get('density_key', 'n')
    fuv_key = ranking_results.get('fuv_key', 'fuv')
    fuv_math = axis_display_label(fuv_key, log=False, html=False).strip('$')
    dens_math = axis_display_label(dens_key, log=False, html=False).strip('$')
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=max(0.08, min(0.14, 0.22 / max(n_rows, 1))),
        column_titles=[rf'${fuv_math} = {sci_tex(f)}$' for f in fuv_levels],
    )
    for i, n in enumerate(n_levels):
        for j, fuv in enumerate(fuv_levels):
            raw = [
                r for r in reg_table
                if (r.get('n_level') == n or _levels_close(r.get('n'), n))
                and (r.get('fuv_level') == fuv or _levels_close(r.get('fuv'), fuv))
            ]
            panel = sorted(
                raw,
                key=lambda r: (-r['probe_score'] if np.isfinite(r['probe_score']) else -1),
            )[: int(top_n)]
            scored = [r for r in panel if np.isfinite(r.get('probe_score', np.nan))]
            note = _empty_panel_message(raw, ranking_results)
            if scored:
                names = [_short_label(r['Species']) for r in scored]
                scores = [max(0.0, float(r['probe_score'])) for r in scored]
                colors = [_color_for_row(r, color_mode) for r in scored]
                hover = [
                    (f"{r['Species']}<br>S = {r['probe_score']:.3f}"
                     f"<br>F = {r['functional_factor']:.3f}"
                     f"<br>V = {r['variation_factor']:.3f}"
                     f"<br>trend = {r['trend']}"
                     f"<br>reversals = {r['n_reversals']}")
                    for r in scored
                ]
                fig.add_trace(
                    go.Bar(
                        x=scores, y=names, orientation='h',
                        marker=dict(color=colors, line=dict(color=t['axis_line'], width=0.4)),
                        hovertext=hover, hoverinfo='text',
                        showlegend=False,
                    ),
                    row=i + 1, col=j + 1,
                )
            if j == 0:
                y_title = sci_latex(n)
                if i == n_rows // 2:
                    y_title = rf'${dens_math} = {sci_tex(n)}$'
            else:
                y_title = ''
            fig.update_yaxes(
                autorange='reversed',
                automargin=True,
                tickfont=ps.tick_font(t['font']),
                title=dict(
                    text=y_title,
                    font=ps.axis_title_font(t['font']),
                    standoff=12,
                ),
                row=i + 1, col=j + 1,
            )
            fig.update_xaxes(
                range=[0, 1.0],
                title=dict(text='Probe score' if i == n_rows - 1 else '',
                           font=ps.axis_title_font(t['font'])),
                **_axis_style(t),
                row=i + 1, col=j + 1,
            )
            if note and not scored:
                fig.add_annotation(
                    text=note,
                    xref='x domain', yref='y domain',
                    x=0.5, y=0.5, showarrow=False,
                    xanchor='center', yanchor='middle',
                    align='center',
                    font=dict(size=11, color=t['muted']),
                    row=i + 1, col=j + 1,
                )

    legend_src = ABSOLUTE_BAND_LABELS if color_mode == 'absolute' else QUARTILE_LABELS
    legend_col = ABSOLUTE_BAND_COLORS if color_mode == 'absolute' else QUARTILE_COLORS
    for key, lab in legend_src.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=legend_col[key]),
            name=lab, showlegend=True,
        ))

    row_h = 300
    fig.update_layout(
        paper_bgcolor=t['paper_bg'], plot_bgcolor=t['plot_bg'],
        font=ps.layout_font(t['font']),
        height=max(420, 90 + n_rows * row_h),
        margin=dict(l=88, r=28, t=88, b=78),
        legend=dict(
            orientation='h', yanchor='top', y=-0.10 if n_rows > 1 else -0.16,
            x=0.0, xanchor='left',
            bgcolor=t['legend_bg'], bordercolor=t['legend_border'], borderwidth=1,
            font=ps.legend_font(t['font']),
        ),
        bargap=0.25,
        title=dict(
            text=(f'CRIR probe ranking by environment'
                  f'<br>CRIR ∈ {rlabel}'),
            font=ps.title_font(t['title']),
            x=0.0, xanchor='left',
        ),
    )
    return fig


def _as_species_list(species):
    if species is None or species == '':
        return []
    if isinstance(species, (str, bytes)):
        return [str(species)]
    return [str(s) for s in species if s not in (None, '')]


def filter_ranking_by_quartile(
    ranking_results,
    quartile='green',
    *,
    regime=None,
    species_only=False,
    ratios_only=False,
):
    """Summary rows in a quartile, matching the KoSens3D selection."""
    summary = ranking_results.get('summary') or []
    out = []
    for row in summary:
        if row.get('Quartile') != quartile:
            continue
        if regime not in (None, 'all') and row.get('Regime') != regime:
            continue
        if species_only and row.get('Is_Ratio'):
            continue
        if ratios_only and not row.get('Is_Ratio'):
            continue
        out.append(row)
    return out


def species_in_quartile(ranking_results, quartile='green', *, regime=None):
    """Unique tracer names that reach ``quartile`` in any (or one) regime."""
    seen = []
    have = set()
    for row in filter_ranking_by_quartile(
            ranking_results, quartile, regime=regime):
        sp = row.get('Species')
        if sp and sp not in have:
            have.add(sp)
            seen.append(sp)
    return seen


def _heatmap_figure_height(n_reg):
    # Landscape: two cards per row, two regime panels share a wide x-axis.
    return 300 if n_reg <= 2 else 330


def score_heatmap_figures(
    ranking_results,
    species,
    *,
    regime=None,
    value_col='probe_score',
    theme=None,
    colorscale='Viridis',
    use_log_colorbar=False,
):
    """One heatmap figure per tracer (regimes side by side), as in KoSens3D."""
    figs = []
    for sp in _as_species_list(species):
        fig = fig_score_heatmap(
            ranking_results, sp,
            regime=regime,
            value_col=value_col,
            theme=theme,
            colorscale=colorscale,
            use_log_colorbar=use_log_colorbar,
        )
        if fig is not None:
            figs.append((sp, fig))
    return figs


def fig_score_heatmap(
    ranking_results,
    species,
    *,
    regime=None,
    value_col='probe_score',
    theme=None,
    colorscale='Viridis',
    use_log_colorbar=False,
):
    """One tracer: heatmap vs (n, FUV), one panel per CRIR regime (KoSens3D)."""
    t = _theme_fallback(theme)
    names = list(dict.fromkeys(_as_species_list(species)))
    slice_df = ranking_results.get('slice_scores') or []
    by_sp: Dict[str, List[dict]] = {}
    for row in slice_df:
        by_sp.setdefault(row['Species'], []).append(row)
    names = [s for s in names if s in by_sp]
    if not names:
        return None
    sp = names[0]

    picked = _ordered_regimes(ranking_results, regime)
    if not picked:
        return None
    ordered = [r['name'] for r in picked]

    n_reg = len(ordered)
    n_rows, n_cols = 1, n_reg
    cells = []
    subplot_titles = []
    for j, spec in enumerate(picked):
        cells.append((0, j, sp, spec['name']))
        subplot_titles.append(crir_regime_label(spec['lo'], spec['hi'], wrap=False))

    vspace = max(0.04, min(0.12, 0.28 / max(n_rows, 1)))
    hspace = max(0.03, min(0.05, 0.12 / max(n_cols, 1)))
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=hspace,
        vertical_spacing=vspace,
        subplot_titles=subplot_titles,
    )

    dens_key = ranking_results.get('density_key', 'n')
    fuv_key = ranking_results.get('fuv_key', 'fuv')
    fallback_n = np.asarray(ranking_results.get('n_levels') or [], dtype=float)
    fallback_fuv = np.asarray(ranking_results.get('fuv_levels') or [], dtype=float)
    last_cell = cells[-1] if cells else None
    for row_i, col_j, sp, rname in cells:
        panel = [r for r in by_sp.get(sp, []) if r['Regime'] == rname]
        n_vals, fuv_vals, z = _heatmap_pivot(panel, value_col=value_col)
        x_title = axis_display_label(fuv_key, log=False, html=True) if row_i == n_rows - 1 else ''
        y_title = axis_display_label(dens_key, log=False, html=True) if col_j == 0 else ''
        if n_vals.size == 0 or fuv_vals.size == 0:
            fig.add_annotation(
                text=_empty_panel_message(panel, ranking_results),
                xref='x domain', yref='y domain',
                x=0.5, y=0.5, showarrow=False,
                xanchor='center', yanchor='middle',
                font=dict(size=11, color=t['muted']),
                row=row_i + 1, col=col_j + 1,
            )
            n_axis = n_vals if n_vals.size else fallback_n
            fuv_axis = fuv_vals if fuv_vals.size else fallback_fuv
            if fuv_axis.size:
                fig.update_xaxes(
                    **_index_decade_axis(fuv_axis, x_title, t),
                    row=row_i + 1, col=col_j + 1,
                )
            if n_axis.size:
                fig.update_yaxes(
                    **_index_decade_axis(n_axis, y_title, t),
                    row=row_i + 1, col=col_j + 1,
                )
            continue
        if use_log_colorbar:
            plot_z = np.log10(np.maximum(np.where(np.isfinite(z) & (z > 0), z, np.nan), 1e-30))
        else:
            plot_z = z
        show_cbar = (row_i, col_j, sp, rname) == last_cell
        hovertext = [
            [
                (
                    f'{sp}<br>'
                    f'G<sub>0</sub>={sci_html(fuv)}<br>'
                    f'n<sub>H</sub>={sci_html(n)}<br>'
                    f'{value_col}='
                    + ('—' if not np.isfinite(plot_z[iy, ix]) else f'{float(z[iy, ix]):.3g}')
                )
                for ix, fuv in enumerate(fuv_vals)
            ]
            for iy, n in enumerate(n_vals)
        ]
        fig.add_trace(
            go.Heatmap(
                x=list(range(int(fuv_vals.size))),
                y=list(range(int(n_vals.size))),
                z=plot_z,
                colorscale=colorscale,
                zmin=0 if (not use_log_colorbar and value_col == 'probe_score') else None,
                zmax=1 if (not use_log_colorbar and value_col == 'probe_score') else None,
                colorbar=dict(
                    title=dict(
                        text=(
                            f'median log₁₀ {value_col.replace("_", " ")}'
                            if use_log_colorbar
                            else f'median {value_col.replace("_", " ")}'
                        ),
                        font=ps.cbar_title_font(t['font']),
                    ),
                    tickfont=ps.cbar_tick_font(t['font']),
                    x=1.02,
                    len=0.92,
                    y=0.5,
                    yanchor='middle',
                    thickness=12,
                    exponentformat='none',
                    showexponent='none',
                ),
                hovertext=hovertext,
                hoverinfo='text',
                showscale=show_cbar,
            ),
            row=row_i + 1, col=col_j + 1,
        )
        finite_z = z[np.isfinite(z)]
        if finite_z.size == 0 or np.nanmax(finite_z) <= 0:
            fig.add_annotation(
                text=_empty_panel_message(panel, ranking_results),
                xref='x domain', yref='y domain',
                x=0.5, y=0.5, showarrow=False,
                xanchor='center', yanchor='middle',
                align='center',
                font=dict(size=11, color=t['muted']),
                row=row_i + 1, col=col_j + 1,
            )
        fig.update_xaxes(
            **_index_decade_axis(fuv_vals, x_title, t),
            row=row_i + 1, col=col_j + 1,
        )
        fig.update_yaxes(
            **_index_decade_axis(n_vals, y_title, t),
            row=row_i + 1, col=col_j + 1,
        )

    fig.update_layout(
        paper_bgcolor=t['paper_bg'], plot_bgcolor=t['plot_bg'],
        font=ps.layout_font(t['font']),
        height=_heatmap_figure_height(n_reg),
        margin=dict(l=58, r=56, t=56, b=52),
        title=None,
        autosize=True,
    )
    title_set = set(subplot_titles)
    for ann in fig.layout.annotations or []:
        if getattr(ann, 'text', None) in title_set:
            ann.font = dict(size=11, color=t['font'])
    return fig


def fig_response_curves(
    grids_3d,
    ranking_results,
    *,
    species_list=None,
    n_level=None,
    fuv_values=None,
    grid_type='rel_abund',
    y_scale='log',
    shade_plateaus=True,
    ncol=3,
    theme=None,
):
    """Quantity vs CRIR at one density; coloured lines are FUV values."""
    t = _theme_fallback(theme)
    if not grids_3d:
        return None

    skip = {'_fit_axis_config', '_meta'}
    available = [k for k, v in grids_3d.items()
                 if k not in skip and isinstance(v, dict) and 'grid' in v]
    if species_list:
        species = [s for s in species_list if s in grids_3d]
    else:
        summary = ranking_results.get('summary') or []
        ranked = sorted(
            {r['Species']: r.get('Median_Probe_Score', 0) for r in summary
             if '/' not in r['Species']}.items(),
            key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else -1),
        )
        species = [s for s, _ in ranked[:6]] or available[:6]
    if not species:
        return None

    n_levels = ranking_results.get('n_levels') or []
    fuv_levels_all = ranking_results.get('fuv_levels') or []
    if n_level is None:
        n_use = _pick_span_levels(n_levels, 3)
        n_level = n_use[len(n_use) // 2] if n_use else None
    else:
        n_level = _nearest_levels([n_level], n_levels)[0] if n_levels else float(n_level)
    if n_level is None:
        return None

    if fuv_values:
        fuv_levels = _nearest_levels(fuv_values, fuv_levels_all or fuv_values)
    else:
        fuv_levels = _pick_span_levels(fuv_levels_all, 3)
    if not fuv_levels:
        return None

    env_table = ranking_at_environments(
        ranking_results, n_values=[n_level], fuv_values=fuv_levels,
    )
    regimes = ranking_results.get('crir_regimes') or []
    interior_edges = [r['lo'] for r in sorted(regimes, key=lambda x: x['lo'])[1:]]
    plateau_slope = ranking_results.get('plateau_slope', DEFAULT_PLATEAU_SLOPE)
    crir_key = ranking_results.get('crir_axis_key')
    y_label = Y_LABELS.get(grid_type, 'Quantity')

    n_species = len(species)
    ncol = max(1, int(ncol))
    nrows = int(np.ceil(n_species / ncol))
    fig = make_subplots(
        rows=nrows, cols=ncol,
        subplot_titles=species + [''] * (nrows * ncol - n_species),
        horizontal_spacing=0.07,
        vertical_spacing=0.12,
    )

    for idx, sp in enumerate(species):
        row, col = divmod(idx, ncol)
        r, c = row + 1, col + 1
        y_vals = []
        last_crir = None
        for j, fuv in enumerate(fuv_levels):
            crir, y = extract_curve_at_environment(
                grids_3d, sp, n_level, fuv, crir_axis_key=crir_key,
            )
            if crir.size < 2:
                continue
            last_crir = crir
            y_vals.extend(y[np.isfinite(y) & (y > 0)].tolist())
            color = FUV_LINE_COLORS[j % len(FUV_LINE_COLORS)]
            info_bits = []
            is_green = False
            for reg in regimes or [None]:
                rname = reg['name'] if isinstance(reg, dict) else None
                info = lookup_env_score(env_table, sp, n_level, fuv, regime=rname)
                sc = info['probe_score']
                if not np.isfinite(sc):
                    continue
                is_green = is_green or (info['Quartile'] == 'green')
                tag = (reg['label'] + ': ') if isinstance(reg, dict) else ''
                info_bits.append(f'{tag}S={sc:.2f}')
            hover = (f'{sp}<br>G<sub>0</sub>={sci_html(fuv)}'
                     + (('<br>' + '<br>'.join(info_bits)) if info_bits else '')
                     + '<br>ζ=%{x:.3g}<br>y=%{y:.3g}<extra></extra>')
            if shade_plateaus:
                for x0, x1 in plateau_spans_crir(crir, y, plateau_slope=plateau_slope):
                    fig.add_vrect(
                        x0=x0, x1=x1, fillcolor='#9aa3ad', opacity=0.12,
                        line_width=0, layer='below', row=r, col=c,
                    )
            fig.add_trace(
                go.Scatter(
                    x=crir, y=y, mode='lines+markers',
                    line=dict(color=color, width=2.4 if is_green else 1.7),
                    marker=dict(size=7, color=color),
                    name=rf'$G_0 = {sci_tex(fuv)}$',
                    legendgroup=f'fuv{j}',
                    showlegend=(idx == 0),
                    hovertemplate=hover,
                ),
                row=r, col=c,
            )

        for edge in interior_edges:
            fig.add_vline(
                x=edge, line=dict(color='#666666', width=1, dash='dash'),
                row=r, col=c,
            )

        xax = dict(
            type='log',
            title=dict(text=axis_display_label(crir_key or 'crir', log=False, html=False)
                       if r == nrows else '',
                       font=ps.axis_title_font(t['font'])),
            **_axis_style(t),
        )
        if last_crir is not None:
            _apply_log_ticks(xax, last_crir)
        fig.update_xaxes(**xax, row=r, col=c)
        fig.update_yaxes(
            type='log' if y_scale == 'log' else 'linear',
            title=dict(text=y_label if c == 1 else '',
                       font=ps.axis_title_font(t['font'])),
            **_axis_style(t),
            row=r, col=c,
        )

    for idx in range(n_species, nrows * ncol):
        r, c = divmod(idx, ncol)
        fig.update_xaxes(visible=False, row=r + 1, col=c + 1)
        fig.update_yaxes(visible=False, row=r + 1, col=c + 1)

    fig.update_layout(
        paper_bgcolor=t['paper_bg'], plot_bgcolor=t['plot_bg'],
        font=ps.layout_font(t['font']),
        height=max(420, 80 + nrows * 340),
        margin=dict(l=72, r=28, t=78, b=56),
        legend=dict(
            orientation='h', y=1.04, yanchor='bottom', x=1.0, xanchor='right',
            bgcolor=t['legend_bg'], bordercolor=t['legend_border'], borderwidth=1,
            font=ps.legend_font(t['font']),
        ),
        title=dict(
            text=rf'Response vs CRIR  —  $n_{{\mathrm{{H}}}} = {sci_tex(n_level)}\,\mathrm{{cm}}^{{-3}}$',
            font=ps.title_font(t['title']),
            x=0.0, xanchor='left',
        ),
    )
    return fig


def level_options(values):
    """Dropdown options from a list of native physical levels."""
    vals = np.sort(np.unique(np.asarray(values, dtype=float)))
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return [{'label': sci_plain(float(v)), 'value': float(v)} for v in vals]


def _plain_sci(value):
    return sci_plain(value)


def default_span_values(values, n_pick=3):
    return _pick_span_levels(values, n_pick)


def default_kosens_env_values(available, requested=KOSENS_DEFAULT_N_ENV):
    """Nearest native nodes to the KoSens notebook environment set."""
    resolved = _resolve_environment_levels(requested, available)
    return resolved or _pick_span_levels(available, len(requested))


def format_score(val, digits=3):
    if val is None or not np.isfinite(val):
        return '—'
    return f'{val:.{digits}f}'


def format_phys(val):
    if val is None or not np.isfinite(val):
        return '—'
    return sci_plain(float(val))
