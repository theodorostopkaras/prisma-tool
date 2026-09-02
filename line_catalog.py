"""Molecular line identification in a plotted frequency window.

A local 3 mm catalog (IRAM 30 m / CLASS coverage ~80–115 GHz, plus a few
neighbouring mm lines) works offline.  Splatalogue (CDMS/JPL) is used when
``astroquery`` is installed and the user selects that catalog.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

C_LIGHT_KMS = 299792.458


@dataclass(frozen=True)
class CatalogLine:
    species: str
    transition: str
    rest_ghz: float
    eu_k: Optional[float] = None


def radio_obs_freq_ghz(rest_ghz: float, v_lsr_kms: float) -> float:
    """Observed frequency (radio definition): ν = ν₀ (1 − v/c)."""
    return float(rest_ghz) * (1.0 - float(v_lsr_kms) / C_LIGHT_KMS)


def radio_velocity_kms(obs_ghz: float, rest_ghz: float) -> float:
    """Radio LSR velocity of an observed frequency relative to rest frequency."""
    nu0 = float(rest_ghz)
    if not np.isfinite(nu0) or nu0 == 0.0:
        return float('nan')
    return C_LIGHT_KMS * (1.0 - float(obs_ghz) / nu0)


def remap_velocity_to_axis(velocity_kms: np.ndarray, axis: np.ndarray,
                           v_query) -> np.ndarray:
    """Interpolate ``axis`` (same length as ``velocity_kms``) at velocity samples."""
    v = np.asarray(velocity_kms, dtype=float)
    x = np.asarray(axis, dtype=float)
    q = np.asarray(v_query, dtype=float)
    order = np.argsort(v)
    return np.interp(q, v[order], x[order], left=np.nan, right=np.nan)


# Rest frequencies from CDMS / LAMDA (GHz).  3 mm band plus nearby mm lines.
_LOCAL_LINES: Tuple[CatalogLine, ...] = (
    # CO and isotopologues
    CatalogLine('CO', '1-0', 115.2712018, 5.53),
    CatalogLine('13CO', '1-0', 110.2013543, 5.29),
    CatalogLine('C18O', '1-0', 109.7821734, 5.27),
    CatalogLine('C17O', '1-0', 112.3592778, 5.39),
    CatalogLine('CO', '2-1', 230.5380000, 16.60),
    CatalogLine('13CO', '2-1', 220.3986841, 15.87),
    # CS
    CatalogLine('CS', '2-1', 97.9809533, 7.05),
    CatalogLine('C34S', '2-1', 96.4129495, 6.94),
    CatalogLine('13CS', '2-1', 92.4943080, 6.66),
    CatalogLine('CS', '3-2', 146.9690287, 14.11),
    # SO / SO2
    CatalogLine('SO', '2_2-1_1', 86.0939830, 19.31),
    CatalogLine('SO', '3_2-2_1', 99.2998700, 9.23),
    CatalogLine('SO', '4_5-4_4', 100.0295650, 38.58),
    CatalogLine('SO2', '3_1_3-2_0_2', 104.2392950, 7.74),
    # SiO / SiS
    CatalogLine('SiO', '2-1', 86.8469850, 6.25),
    CatalogLine('29SiO', '2-1', 85.7591990, 6.18),
    CatalogLine('SiO', '3-2', 130.2686100, 12.50),
    CatalogLine('SiS', '5-4', 90.7715640, 13.09),
    # HCO+ family
    CatalogLine('HCO+', '1-0', 89.1885247, 4.28),
    CatalogLine('H13CO+', '1-0', 86.7542884, 4.16),
    CatalogLine('HC18O+', '1-0', 85.1622230, 4.09),
    CatalogLine('DCO+', '1-0', 72.0393310, 3.46),
    CatalogLine('HOC+', '1-0', 89.4874140, 4.30),
    # HCN / HNC
    CatalogLine('HCN', '1-0 F=1-1', 88.6304157, 4.25),
    CatalogLine('HCN', '1-0 F=2-1', 88.6318473, 4.25),
    CatalogLine('HCN', '1-0 F=0-1', 88.6339360, 4.25),
    CatalogLine('H13CN', '1-0', 86.3399214, 4.14),
    CatalogLine('HC15N', '1-0', 86.0549664, 4.13),
    CatalogLine('DCN', '1-0', 72.4149228, 3.48),
    CatalogLine('HNC', '1-0', 90.6635680, 4.35),
    CatalogLine('HN13C', '1-0', 87.0908590, 4.18),
    CatalogLine('H15NC', '1-0', 88.8657150, 4.26),
    # N2H+ hyperfine (J=1-0)
    CatalogLine('N2H+', '1-0 F1=1 F=0-1', 93.1716210, 4.47),
    CatalogLine('N2H+', '1-0 F1=1 F=2-2', 93.1719170, 4.47),
    CatalogLine('N2H+', '1-0 F1=1 F=1-0', 93.1720500, 4.47),
    CatalogLine('N2H+', '1-0 F1=2 F=2-1', 93.1734770, 4.47),
    CatalogLine('N2H+', '1-0 F1=2 F=3-2', 93.1737760, 4.47),
    CatalogLine('N2H+', '1-0 F1=2 F=1-1', 93.1739670, 4.47),
    CatalogLine('N2H+', '1-0 F1=0 F=1-2', 93.1762650, 4.47),
    CatalogLine('N2D+', '1-0', 77.1096140, 3.70),
    # CF+ / H2CS (DR21 IRAM set)
    CatalogLine('CF+', '1-0', 102.5875330, 4.93),
    CatalogLine('H2CS', '3_1_2-2_1_1', 101.4778850, 23.19),
    CatalogLine('H2CS', '3_0_3-2_0_2', 103.0405480, 14.85),
    CatalogLine('H2CS', '3_1_3-2_1_2', 104.6169880, 22.91),
    # CCH / CN
    CatalogLine('CCH', 'N=1-0 J=3/2-1/2 F=2-1', 87.3169250, 4.19),
    CatalogLine('CCH', 'N=1-0 J=3/2-1/2 F=1-0', 87.3286240, 4.19),
    CatalogLine('CCH', 'N=1-0 J=1/2-1/2 F=1-1', 87.4019890, 4.21),
    CatalogLine('CCH', 'N=1-0 J=1/2-1/2 F=0-1', 87.4071650, 4.21),
    CatalogLine('CN', 'N=1-0 J=3/2-1/2', 113.4909820, 5.45),
    CatalogLine('CN', 'N=1-0 J=1/2-1/2', 113.1913170, 5.43),
    # H2CO / HNCO / CH3OH / OCS
    CatalogLine('p-H2CO', '2_0_2-1_0_1', 145.6029490, 10.48),
    CatalogLine('HNCO', '4_0_4-3_0_3', 87.9252370, 10.55),
    CatalogLine('CH3OH', '5_1_4-4_1_3 A+', 84.5212060, 34.25),
    CatalogLine('CH3OH', '2_1_1-1_1_0 E', 96.7393620, 12.54),
    CatalogLine('CH3OH', '2_0_2-1_0_1 A+', 96.7413750, 6.97),
    CatalogLine('OCS', '7-6', 85.1391030, 16.34),
    CatalogLine('OCS', '8-7', 97.3012085, 21.01),
    CatalogLine('OCS', '9-8', 109.4630630, 26.34),
    CatalogLine('HC3N', '10-9', 90.9790230, 24.01),
    CatalogLine('HC3N', '11-10', 100.0763920, 28.82),
    CatalogLine('HC3N', '12-11', 109.1736340, 34.06),
    CatalogLine('c-C3H2', '2_1_2-1_0_1', 85.3388940, 6.45),
    CatalogLine('N2H+', '1-0 (blend)', 93.1737000, 4.47),
)


def local_catalog() -> Tuple[CatalogLine, ...]:
    return _LOCAL_LINES


def lines_in_window(freq_min_ghz: float, freq_max_ghz: float,
                    v_source_kms: float = 0.0,
                    catalog: Optional[Sequence[CatalogLine]] = None,
                    eu_max_k: Optional[float] = None) -> List[dict]:
    """Catalog lines whose *observed* frequency falls in ``[freq_min, freq_max]``."""
    lo, hi = sorted((float(freq_min_ghz), float(freq_max_ghz)))
    src = float(v_source_kms or 0.0)
    rows = []
    for line in (catalog if catalog is not None else _LOCAL_LINES):
        if eu_max_k is not None and line.eu_k is not None and line.eu_k > float(eu_max_k):
            continue
        nu_obs = radio_obs_freq_ghz(line.rest_ghz, src)
        if lo <= nu_obs <= hi:
            rows.append({
                'species': line.species,
                'transition': line.transition,
                'rest_ghz': float(line.rest_ghz),
                'obs_ghz': float(nu_obs),
                'eu_k': line.eu_k,
                'source': 'local',
            })
    rows.sort(key=lambda r: r['obs_ghz'])
    return rows


_HTML_TAG_RE = re.compile(r'<[^>]+>')
_LEADING_FLOAT_RE = re.compile(r'([0-9]+(?:\.[0-9]+)?)')


def splatalogue_available() -> bool:
    try:
        from astroquery.splatalogue import Splatalogue  # noqa: F401
        return True
    except Exception:
        return False


def _strip_html(text) -> str:
    if text is None:
        return ''
    s = html.unescape(str(text))
    s = _HTML_TAG_RE.sub('', s)
    return ' '.join(s.split())


def _cell_float(val) -> float:
    """Numeric table cell, or the first number in a Splatalogue HTML string."""
    if val is None:
        return float('nan')
    try:
        if hasattr(val, 'mask') and bool(val.mask):
            return float('nan')
    except (TypeError, ValueError):
        pass
    try:
        x = float(val)
        if np.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    m = _LEADING_FLOAT_RE.search(_strip_html(val))
    return float(m.group(1)) if m else float('nan')


def rest_ghz_from_splatalogue(value, *, from_mhz: bool = False) -> float:
    """Splatalogue JSON ``orderedfreq`` is MHz; older tables used GHz."""
    x = _cell_float(value)
    if not np.isfinite(x) or x <= 0:
        return float('nan')
    if from_mhz or x >= 3000.0:
        return x / 1000.0
    return x


def _table_col(table, *names):
    for name in names:
        if name in table.colnames:
            return table[name]
    return None


def _rest_window_for_observed(freq_min_ghz: float, freq_max_ghz: float,
                              v_source_kms: float) -> Tuple[float, float]:
    """Rest-frequency interval that Doppler-shifts into the observed cube window."""
    lo, hi = sorted((float(freq_min_ghz), float(freq_max_ghz)))
    src = float(v_source_kms or 0.0)
    denom = 1.0 - src / C_LIGHT_KMS
    if not np.isfinite(denom) or denom == 0.0:
        return lo, hi
    return tuple(sorted((lo / denom, hi / denom)))


def rows_from_splatalogue_table(table, freq_min_ghz: float, freq_max_ghz: float,
                                v_source_kms: float = 0.0,
                                limit: int = 80) -> List[dict]:
    """Keep Splatalogue rows whose rest frequency (the line ID) falls in the window."""
    lo, hi = sorted((float(freq_min_ghz), float(freq_max_ghz)))
    src = float(v_source_kms or 0.0)

    species = _table_col(table, 'name', 'Species', 'chemical_name', 'Chemical Name')
    trans = _table_col(table, 'resolved_QNs', 'Resolved QNs', 'QNs', 'Quantum Number')
    # New JSON API: orderedfreq is MHz. Legacy HTML tables used GHz column names.
    freq_mhz = _table_col(table, 'orderedfreq')
    freq_ghz = _table_col(
        table,
        'Freq-GHz(rest frame,redshifted)', 'Freq-GHz',
        'Rest Frequency (GHz)', 'nu',
    )
    freq_fmt = _table_col(table, 'orderedFreq', 'measFreq')
    eu = _table_col(table, 'upper_state_energy_K', 'E_U (K)', 'Eu_K',
                    'Upper State Energy (K)')
    linelist = _table_col(table, 'linelist', 'Line List')

    rows = []
    n = len(table)
    for i in range(n):
        if freq_mhz is not None:
            nu0 = rest_ghz_from_splatalogue(freq_mhz[i], from_mhz=True)
        elif freq_ghz is not None:
            nu0 = rest_ghz_from_splatalogue(freq_ghz[i], from_mhz=False)
        elif freq_fmt is not None:
            nu0 = rest_ghz_from_splatalogue(freq_fmt[i], from_mhz=False)
        else:
            continue
        if not np.isfinite(nu0):
            continue
        nu_obs = radio_obs_freq_ghz(nu0, src)
        if not (lo <= nu_obs <= hi):
            continue
        sp = _strip_html(species[i]) if species is not None else ''
        qn = _strip_html(trans[i]) if trans is not None else ''
        eu_val = None
        if eu is not None:
            e = _cell_float(eu[i])
            if np.isfinite(e):
                eu_val = e
        catalog = 'splatalogue'
        if linelist is not None:
            tag = _strip_html(linelist[i])
            if tag:
                catalog = f'splatalogue/{tag}'
        rows.append({
            'species': sp or '—',
            'transition': qn or '—',
            'rest_ghz': float(nu0),
            'obs_ghz': float(nu_obs),
            'eu_k': eu_val,
            'source': catalog,
        })

    rows.sort(key=lambda r: (r['eu_k'] is None,
                             r['eu_k'] if r['eu_k'] is not None else 0,
                             r['obs_ghz']))
    seen = set()
    unique = []
    for rec in rows:
        key = (rec['species'], rec['transition'], round(rec['rest_ghz'], 5))
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique[: int(limit)]


def query_splatalogue(freq_min_ghz: float, freq_max_ghz: float,
                      v_source_kms: float = 0.0, eu_max_k: float = 150.0,
                      limit: int = 80) -> Tuple[List[dict], Optional[str]]:
    """Query Splatalogue CDMS/JPL for rest frequencies in the observed window."""
    try:
        from astroquery.splatalogue import Splatalogue
        import astropy.units as u
    except Exception:
        return [], 'astroquery is not installed (pip install astroquery)'

    lo, hi = sorted((float(freq_min_ghz), float(freq_max_ghz)))
    rest_lo, rest_hi = _rest_window_for_observed(lo, hi, v_source_kms)
    pad = 0.001  # 1 MHz catalog / rounding pad
    try:
        kw = dict(
            energy_type='eu_k',
            line_lists=['CDMS', 'JPL'],
            exclude=['atmospheric'],
            energy_levels=['Four'],
        )
        if eu_max_k:
            kw['energy_max'] = float(eu_max_k)
        table = Splatalogue.query_lines(
            (rest_lo - pad) * u.GHz, (rest_hi + pad) * u.GHz, **kw)
    except Exception as exc:
        return [], str(exc)

    if table is None or len(table) == 0:
        return [], None
    return rows_from_splatalogue_table(
        table, lo, hi, v_source_kms=v_source_kms, limit=limit), None


def identify_lines(freq_min_ghz: float, freq_max_ghz: float,
                   catalog: str = 'local', v_source_kms: float = 0.0,
                   eu_max_k: Optional[float] = 150.0) -> Tuple[List[dict], Optional[str]]:
    """Return (rows, error) for the chosen catalog in an observed frequency window."""
    cat = (catalog or 'local').strip().lower()
    if cat in ('splatalogue', 'splat', 'cdms'):
        return query_splatalogue(
            freq_min_ghz, freq_max_ghz, v_source_kms=v_source_kms,
            eu_max_k=float(eu_max_k) if eu_max_k else 150.0)
    return lines_in_window(
        freq_min_ghz, freq_max_ghz, v_source_kms=v_source_kms,
        eu_max_k=eu_max_k), None
