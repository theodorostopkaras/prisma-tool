"""Molecular line identification in a plotted frequency window.

A local 3 mm catalog (IRAM 30 m / CLASS coverage ~80–115 GHz, plus a few
neighbouring mm lines) works offline.  The CDMS and JPL catalogues are
queried directly through ``astroquery`` (network needed) when selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


_MHZ_PER_CM1 = 29979.2458   # 1 cm⁻¹ in MHz
_K_PER_CM1 = 1.4387769      # hc/k: 1 cm⁻¹ in K
_QN_MISSING = -999999       # CDMS fill value for unused quantum numbers
_UPPER_QN = ('Ju', 'Ku', 'vu', 'F1u', 'F2u', 'F3u')   # CDMS columns
_LOWER_QN = ('Jl', 'Kl', 'vl', 'F1l', 'F2l', 'F3l')
CATALOG_SERVICES = ('CDMS', 'JPL')
# Per-species documentation pages (the catalogues have no per-transition page).
_SPECIES_URL = {
    'CDMS': 'https://cdms.astro.uni-koeln.de/cgi-bin/cdmsinfo?file=e{tag:06d}.cat',
    'JPL': 'https://spec.jpl.nasa.gov/ftp/pub/catalog/doc/d{tag:06d}.pdf',
}


def species_url(source: str, tag: int) -> Optional[str]:
    """Documentation page of catalogue species ``tag`` in ``source`` (CDMS / JPL)."""
    fmt = _SPECIES_URL.get(source)
    return fmt.format(tag=abs(int(tag))) if fmt else None


def _jpl_class():
    try:
        from astroquery.linelists.jplspec import JPLSpec
    except ImportError:  # astroquery < 0.4.8
        from astroquery.jplspec import JPLSpec
    return JPLSpec


def catalog_query_available() -> bool:
    try:
        from astroquery.linelists.cdms import CDMS  # noqa: F401
        _jpl_class()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _jpl_species_names() -> dict:
    """JPL result tables carry only the species TAG; map it to a name."""
    try:
        st = _jpl_class().get_species_table()
        return {int(t): str(n).strip() for t, n in zip(st['TAG'], st['NAME'])}
    except Exception:
        return {}


def _qn_part(value) -> Optional[str]:
    if np.ma.is_masked(value):
        return None
    if isinstance(value, (str, bytes, np.str_)):
        return ','.join(str(value).split()) or None
    if int(value) == _QN_MISSING:
        return None
    return str(int(value))


def _transition(row, upper, lower) -> str:
    up = [p for p in (_qn_part(row[c]) for c in upper) if p]
    lo = [p for p in (_qn_part(row[c]) for c in lower) if p]
    return f"{','.join(up)} - {','.join(lo)}" if (up or lo) else '—'


def _rest_window_for_observed(freq_min_ghz: float, freq_max_ghz: float,
                              v_source_kms: float) -> Tuple[float, float]:
    """Rest-frequency interval that Doppler-shifts into the observed cube window."""
    lo, hi = sorted((float(freq_min_ghz), float(freq_max_ghz)))
    src = float(v_source_kms or 0.0)
    denom = 1.0 - src / C_LIGHT_KMS
    if not np.isfinite(denom) or denom == 0.0:
        return lo, hi
    return tuple(sorted((lo / denom, hi / denom)))


def rows_from_catalog_table(table, freq_min_ghz: float, freq_max_ghz: float,
                            v_source_kms: float = 0.0,
                            eu_max_k: Optional[float] = None,
                            source: str = 'CDMS',
                            species_names: Optional[dict] = None,
                            limit: int = 80) -> List[dict]:
    """CDMS / JPL catalogue rows (FREQ MHz, ELO cm⁻¹) whose observed frequency is in the window.

    E_u = E_lo + hν, converted to K.  Rows are sorted by E_u so the ``limit``
    keeps the low-excitation lines that are most likely in a PDR cube.
    """
    lo, hi = sorted((float(freq_min_ghz), float(freq_max_ghz)))
    src = float(v_source_kms or 0.0)
    cols = table.colnames
    upper = [c for c in cols if c in _UPPER_QN or c.startswith("QN'")]
    lower = [c for c in cols if c in _LOWER_QN or c.startswith('QN"')]
    names = species_names or {}

    rows = []
    for row in table:
        freq_mhz = float(row['FREQ'])
        nu0 = freq_mhz / 1000.0
        nu_obs = radio_obs_freq_ghz(nu0, src)
        if not (lo <= nu_obs <= hi):
            continue
        eu = (float(row['ELO']) + freq_mhz / _MHZ_PER_CM1) * _K_PER_CM1
        if eu_max_k is not None and eu > float(eu_max_k):
            continue
        tag = abs(int(row['TAG']))  # negative TAG = laboratory frequency
        species = str(row['name']).strip() if 'name' in cols else names.get(tag, f'tag {tag:06d}')
        rows.append({
            'species': species,
            'transition': _transition(row, upper, lower),
            'rest_ghz': nu0,
            'obs_ghz': float(nu_obs),
            'eu_k': float(eu),
            'source': source,
            'tag': tag,
            'url': species_url(source, tag),
        })
    return _dedup_by_energy(rows, limit)


def _dedup_by_energy(rows: List[dict], limit: int) -> List[dict]:
    rows = sorted(rows, key=lambda r: (r['eu_k'] is None, r['eu_k'] or 0.0, r['obs_ghz']))
    seen, unique = set(), []
    for rec in rows:
        key = (rec['species'], rec['transition'], round(rec['rest_ghz'], 5))
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique[: int(limit)]


def query_catalog(service: str, freq_min_ghz: float, freq_max_ghz: float,
                  v_source_kms: float = 0.0, eu_max_k: Optional[float] = 150.0,
                  limit: int = 80) -> Tuple[List[dict], Optional[str]]:
    """Query the CDMS or JPL catalogue (astroquery) for lines in the observed window."""
    try:
        import astropy.units as u
        from astroquery.linelists.cdms import CDMS
        JPLSpec = _jpl_class()
    except Exception:
        return [], 'astroquery is not installed (pip install astroquery)'

    rest_lo, rest_hi = _rest_window_for_observed(freq_min_ghz, freq_max_ghz, v_source_kms)
    pad = 0.001  # 1 MHz catalog / rounding pad
    fmin, fmax = (rest_lo - pad) * u.GHz, (rest_hi + pad) * u.GHz
    names = None
    try:
        if service == 'JPL':
            # The all-species search uses JPL's catform service; no per-file fallback.
            table = JPLSpec.query_lines(fmin, fmax, molecule='All',
                                        use_getmolecule=False,
                                        fallback_to_getmolecule=False)
            names = _jpl_species_names()
        else:
            table = CDMS.query_lines(fmin, fmax, molecule='All')
    except np.ma.MaskError:
        return [], None  # astroquery's parser chokes on an empty result
    except OSError as exc:  # requests.ConnectionError / Timeout are OSErrors
        return [], f'{service} search service is unreachable ({type(exc).__name__}).'
    except Exception as exc:
        return [], f'{service} query failed ({type(exc).__name__}): {str(exc)[:200]}'

    if table is None or len(table) == 0:
        return [], None
    return rows_from_catalog_table(
        table, freq_min_ghz, freq_max_ghz, v_source_kms=v_source_kms,
        eu_max_k=eu_max_k, source=service, species_names=names, limit=limit), None


def identify_lines(freq_min_ghz: float, freq_max_ghz: float,
                   catalog: str = 'local', v_source_kms: float = 0.0,
                   eu_max_k: Optional[float] = 150.0,
                   limit: int = 80) -> Tuple[List[dict], Optional[str]]:
    """Return (rows, error) for ``catalog`` = 'local', 'cdms', 'jpl' or 'cdms+jpl'.

    With several services, rows from the ones that answered are returned
    together with the error text of the ones that failed.
    """
    cat = (catalog or 'local').strip().lower()
    services = [s for s in CATALOG_SERVICES if s.lower() in cat.split('+')]
    if not services:
        return lines_in_window(
            freq_min_ghz, freq_max_ghz, v_source_kms=v_source_kms,
            eu_max_k=eu_max_k), None
    rows, errors = [], []
    for svc in services:
        got, err = query_catalog(svc, freq_min_ghz, freq_max_ghz,
                                 v_source_kms=v_source_kms, eu_max_k=eu_max_k,
                                 limit=limit)
        rows += got
        if err:
            errors.append(err)
    if len(services) > 1:
        rows = sorted(rows, key=lambda r: (r['eu_k'], r['obs_ghz']))[: int(limit)]
    return rows, ('\n'.join(errors) or None)
