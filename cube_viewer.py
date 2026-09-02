"""
CARTA-like spectral-cube viewer for CLASS/GILDAS MATRIX tables and image cubes.

IRAM 30 m OTF maps are often exported as CLASS binary tables (one spectrum per
row, spatial offsets in CDELT2/CDELT3), not NAXIS=3 FITS images.  This module
grids those rows onto the regular offset lattice, collapses a moment / peak
map, and extracts the spectrum of a clicked pixel or a selected region.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import obs_spectrum_fits as osf

try:
    from astropy.io import fits
except ImportError:  # pragma: no cover
    fits = None

FWHM_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548
_DEFAULT_GRID_ARCSEC = 15.0
_CACHE_MAX = 4
_cube_cache: Dict[Tuple[str, int], Tuple[float, 'SpectralCube']] = {}

_FILE_SPEC_RE = re.compile(
    r'(?:^|_)(?P<sp>h13co\+|h13cn|n2h\+|hco\+|h2cs|hcn|hnc|sio|cf\+|[a-z][a-z0-9+]*)'
    r'(?P<u>\d)(?P<l>\d)(?:_|$|\.)',
    re.IGNORECASE,
)
_SPECIES_PRETTY = {
    'cf+': 'CF+',
    'h13cn': 'H13CN',
    'h13co+': 'H13CO+',
    'h2cs': 'H2CS',
    'hcn': 'HCN',
    'hco+': 'HCO+',
    'hnc': 'HNC',
    'n2h+': 'N2H+',
    'sio': 'SiO',
}


@dataclass
class SpectralCube:
    """Gridded spectral cube (CLASS table or image HDU)."""

    path: str
    hdu_index: int
    spectra: np.ndarray
    velocity_kms: np.ndarray
    ra_off_deg: np.ndarray
    dec_off_deg: np.ndarray
    iy: np.ndarray
    ix: np.ndarray
    ra_axis_deg: np.ndarray
    dec_axis_deg: np.ndarray
    row_index: np.ndarray
    deltav_kms: float
    source_kind: str
    freq_ghz: Optional[np.ndarray] = None
    restfreq_ghz: Optional[float] = None
    crval2_deg: Optional[float] = None
    crval3_deg: Optional[float] = None
    object_name: str = ''
    species_label: str = ''
    intensity_unit: str = 'K'
    is_main_beam: bool = False
    grid_step_arcsec: float = _DEFAULT_GRID_ARCSEC
    header_extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def ny(self) -> int:
        return int(self.dec_axis_deg.size)

    @property
    def nx(self) -> int:
        return int(self.ra_axis_deg.size)

    @property
    def n_spectra(self) -> int:
        return int(self.spectra.shape[0])

    @property
    def nchan(self) -> int:
        return int(self.spectra.shape[1])

    @property
    def ra_axis_arcmin(self) -> np.ndarray:
        return self.ra_axis_deg * 60.0

    @property
    def dec_axis_arcmin(self) -> np.ndarray:
        return self.dec_axis_deg * 60.0

    @property
    def frequency_ghz(self) -> Optional[np.ndarray]:
        """Observed frequency (GHz): CLASS header axis, else radio from rest frequency."""
        if self.freq_ghz is not None:
            return np.asarray(self.freq_ghz, dtype=float)
        if self.restfreq_ghz is not None and np.isfinite(self.restfreq_ghz):
            return (float(self.restfreq_ghz)
                    * (1.0 - np.asarray(self.velocity_kms, dtype=float) / osf._C_LIGHT_KMS))
        return None

    def freq_window_ghz(self) -> Optional[Tuple[float, float]]:
        f = self.frequency_ghz
        if f is None or f.size == 0 or not np.any(np.isfinite(f)):
            return None
        return float(np.nanmin(f)), float(np.nanmax(f))

    @property
    def n_blank(self) -> int:
        return int(np.count_nonzero(self.row_index < 0))


def species_label_from_path(path: str, restfreq_ghz: Optional[float] = None) -> str:
    """Species / transition from filename; LINE in CLASS headers is often wrong."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r'^dr21[_-]?', '', stem, flags=re.IGNORECASE)
    match = _FILE_SPEC_RE.search(stem)
    if match:
        raw = match.group('sp').lower()
        pretty = _SPECIES_PRETTY.get(raw, raw.upper())
        label = f'{pretty} ({int(match.group("u"))}-{int(match.group("l"))})'
    else:
        label = stem
    if restfreq_ghz is not None and np.isfinite(restfreq_ghz) and restfreq_ghz > 0:
        label = f'{label}  {restfreq_ghz:.6f} GHz'
    return label


def list_cube_fits(path: str) -> List[Dict[str, str]]:
    """List FITS files for a path that is either a file or a directory."""
    raw = os.path.abspath(os.path.expanduser(str(path or '').strip()))
    if not raw:
        return []
    files: List[str] = []
    if os.path.isfile(raw) and raw.lower().endswith(('.fits', '.fit', '.fts')):
        files = [raw]
    elif os.path.isdir(raw):
        files = sorted(
            glob.glob(os.path.join(raw, '*.fits'))
            + glob.glob(os.path.join(raw, '*.FITS'))
            + glob.glob(os.path.join(raw, '*.fit'))
        )
    options = []
    for fp in files:
        options.append({
            'label': f'{os.path.basename(fp)}  —  {species_label_from_path(fp)}',
            'value': fp,
        })
    return options


def default_hdu_index(path: str) -> int:
    """CLASS MATRIX tables live in HDU 1; image cubes usually in HDU 0."""
    if fits is None:
        return 1
    with fits.open(os.path.expanduser(path), memmap=False) as hdul:
        if len(hdul) > 1 and getattr(hdul[0], 'data', None) is None:
            return 1
        return 0


def _header_float(header, *keys):
    for key in keys:
        if key not in header:
            continue
        try:
            val = float(header[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            return val
    return None


def _frequency_ghz(header, nchan: int) -> Optional[np.ndarray]:
    rest = osf.rest_frequency_hz_from_header(header)
    if rest is None:
        return None
    crval = float(header.get('CRVAL1', 0.0) or 0.0)
    cdelt = float(header.get('CDELT1', 0.0) or 0.0)
    crpix = float(header.get('CRPIX1', 1.0) or 1.0)
    pix = np.arange(nchan) + 1.0
    return (rest + crval + (pix - crpix) * cdelt) / 1e9


def _unique_sorted(values: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return a
    a = np.sort(a)
    if tol is None:
        diffs = np.diff(a)
        diffs = diffs[diffs > 0]
        tol = 0.2 * float(np.median(diffs)) if diffs.size else 1e-12
    out = [float(a[0])]
    for x in a[1:]:
        if abs(float(x) - out[-1]) > tol:
            out.append(float(x))
    return np.asarray(out, dtype=float)


def _regular_axis(offsets_deg: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fill a regular offset axis from min to max using the median step."""
    vals = _unique_sorted(offsets_deg)
    if vals.size == 0:
        return vals, _DEFAULT_GRID_ARCSEC / 3600.0
    if vals.size == 1:
        return vals.copy(), _DEFAULT_GRID_ARCSEC / 3600.0
    diffs = np.diff(vals)
    diffs = diffs[diffs > 1e-15]
    step = float(np.median(diffs)) if diffs.size else (_DEFAULT_GRID_ARCSEC / 3600.0)
    if step <= 0:
        step = _DEFAULT_GRID_ARCSEC / 3600.0
    n = int(round((vals[-1] - vals[0]) / step)) + 1
    n = max(n, 1)
    axis = vals[0] + step * np.arange(n, dtype=float)
    return axis, step


def _assign_grid(ra_off: np.ndarray, dec_off: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    ra_axis, step_ra = _regular_axis(ra_off)
    dec_axis, step_dec = _regular_axis(dec_off)
    step = 0.5 * (step_ra + step_dec)
    nx, ny = ra_axis.size, dec_axis.size
    row_index = np.full((ny, nx), -1, dtype=int)
    ix = np.zeros(ra_off.size, dtype=int)
    iy = np.zeros(dec_off.size, dtype=int)
    if nx == 0 or ny == 0:
        return iy, ix, ra_axis, dec_axis, row_index, step * 3600.0
    for i, (ra, dec) in enumerate(zip(ra_off, dec_off)):
        jx = int(np.argmin(np.abs(ra_axis - ra)))
        jy = int(np.argmin(np.abs(dec_axis - dec)))
        ix[i] = jx
        iy[i] = jy
        row_index[jy, jx] = i
    return iy, ix, ra_axis, dec_axis, row_index, step * 3600.0


def _spatial_axes_from_image_header(header, spectral_axis: int, ny: int, nx: int
                                    ) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    """Pixel-centre offsets (deg) for the two non-spectral axes of an image cube."""
    axes = [ax for ax in range(1, int(header.get('NAXIS', 2)) + 1)
            if ax != int(spectral_axis)]
    if len(axes) < 2:
        ra_axis = (np.arange(nx) - 0.5 * (nx - 1)) * (_DEFAULT_GRID_ARCSEC / 3600.0)
        dec_axis = (np.arange(ny) - 0.5 * (ny - 1)) * (_DEFAULT_GRID_ARCSEC / 3600.0)
        return ra_axis, dec_axis, None, None

    def _axis(ax, n):
        crpix = float(header.get(f'CRPIX{ax}', 1.0) or 1.0)
        crval = float(header.get(f'CRVAL{ax}', 0.0) or 0.0)
        cdelt = float(header.get(f'CDELT{ax}', 0.0) or 0.0)
        cunit = str(header.get(f'CUNIT{ax}', 'deg')).strip().lower()
        pix = np.arange(n) + 1.0
        world = crval + (pix - crpix) * cdelt
        if cunit in ('arcsec', '"'):
            world = world / 3600.0
        elif cunit in ('arcmin', "'"):
            world = world / 60.0
        return world, crval

    # FITS image cubes are typically (x=NAXIS1=RA, y=NAXIS2=Dec) with spectral last
    # or spectral first.  numpy shape after moving spectral to last is (ny, nx, nchan)
    # corresponding to FITS axes (axis_y, axis_x) = (axes[1], axes[0]) when spectral
    # is axis 3, or similar.  obs_spectrum_fits moves the spectral axis to last, so
    # remaining dims are in original C-order (NAXIS_n ... NAXIS_1 excluding spectral).
    x_world, crval_x = _axis(axes[0], nx)
    y_world, crval_y = _axis(axes[1], ny)
    ctype_x = str(header.get(f'CTYPE{axes[0]}', '')).upper()
    if 'DEC' in ctype_x or ctype_x.startswith('LAT'):
        x_world, y_world = y_world, x_world
        crval_x, crval_y = crval_y, crval_x
    if np.nanmax(np.abs(x_world)) > 1.0:
        ra_off = x_world - crval_x
        crval2 = crval_x
    else:
        ra_off = x_world
        crval2 = crval_x
    if np.nanmax(np.abs(y_world)) > 1.0:
        dec_off = y_world - crval_y
        crval3 = crval_y
    else:
        dec_off = y_world
        crval3 = crval_y
    return ra_off, dec_off, crval2, crval3


def _load_class_matrix(hdu, path: str, hdu_index: int) -> SpectralCube:
    header = hdu.header
    data = hdu.data
    names = list(getattr(data, 'names', []) or [])
    if 'SPECTRUM' not in names:
        raise KeyError(f'SPECTRUM column missing; available: {names}')
    spectra = np.asarray(data['SPECTRUM'], dtype=np.float64)
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)
    nchan = spectra.shape[1]
    v = osf.velocity_axis_lsr_kms(header, nchan=nchan)
    if 'CDELT2' not in names or 'CDELT3' not in names:
        raise KeyError('CLASS MATRIX needs CDELT2 (RA offset) and CDELT3 (Dec offset)')
    ra_off = np.asarray(data['CDELT2'], dtype=np.float64)
    dec_off = np.asarray(data['CDELT3'], dtype=np.float64)
    iy, ix, ra_axis, dec_axis, row_index, step_as = _assign_grid(ra_off, dec_off)
    rest_hz = osf.rest_frequency_hz_from_header(header)
    rest_ghz = None if rest_hz is None else rest_hz / 1e9
    deltav = float(header.get('DELTAV', 0.0) or 0.0) / 1e3
    if not np.isfinite(deltav) or deltav == 0.0:
        dv = np.diff(v)
        deltav = float(np.median(dv)) if dv.size else float('nan')
    stem = os.path.basename(path).lower()
    return SpectralCube(
        path=os.path.abspath(path),
        hdu_index=hdu_index,
        spectra=spectra,
        velocity_kms=np.asarray(v, dtype=float),
        freq_ghz=_frequency_ghz(header, nchan),
        ra_off_deg=ra_off,
        dec_off_deg=dec_off,
        iy=iy,
        ix=ix,
        ra_axis_deg=ra_axis,
        dec_axis_deg=dec_axis,
        row_index=row_index,
        restfreq_ghz=rest_ghz,
        deltav_kms=deltav,
        crval2_deg=_header_float(header, 'CRVAL2'),
        crval3_deg=_header_float(header, 'CRVAL3'),
        object_name=str(header.get('OBJECT', '') or ''),
        species_label=species_label_from_path(path, rest_ghz),
        intensity_unit='K',
        is_main_beam=('_mb' in stem),
        source_kind='class_matrix',
        grid_step_arcsec=float(step_as),
        header_extras={
            'BEAMEFF': _header_float(header, 'BEAMEFF'),
            'FORWEFF': _header_float(header, 'FORWEFF'),
            'ORIGIN': str(header.get('ORIGIN', '') or ''),
        },
    )


def _load_image_cube(hdu, path: str, hdu_index: int) -> SpectralCube:
    header = hdu.header
    spectral_axis, v = osf._spectral_axis_from_header_kms(header)
    cube_sl = osf._cube_spectral_last(np.asarray(hdu.data, dtype=float), spectral_axis)
    if cube_sl.ndim == 1:
        cube_sl = cube_sl.reshape(1, 1, -1)
    elif cube_sl.ndim == 2:
        cube_sl = cube_sl.reshape(1, cube_sl.shape[0], cube_sl.shape[1])
    ny, nx, nchan = cube_sl.shape
    spectra = cube_sl.reshape(ny * nx, nchan)
    ra_axis, dec_axis, crval2, crval3 = _spatial_axes_from_image_header(
        header, spectral_axis, ny, nx)
    ra_grid, dec_grid = np.meshgrid(ra_axis, dec_axis)
    ra_off = ra_grid.ravel()
    dec_off = dec_grid.ravel()
    iy = np.repeat(np.arange(ny), nx)
    ix = np.tile(np.arange(nx), ny)
    row_index = np.arange(ny * nx, dtype=int).reshape(ny, nx)
    finite_spec = np.any(np.isfinite(spectra), axis=1)
    blank = np.where(~finite_spec)[0]
    if blank.size:
        for r in blank:
            row_index[iy[r], ix[r]] = -1
    rest_hz = osf.rest_frequency_hz_from_header(header)
    rest_ghz = None if rest_hz is None else rest_hz / 1e9
    dv = np.diff(v)
    deltav = float(np.median(dv)) if dv.size else float('nan')
    step_as = float(np.median(np.abs(np.diff(ra_axis)))) * 3600.0 if ra_axis.size > 1 else _DEFAULT_GRID_ARCSEC
    bunit = str(header.get('BUNIT', '') or 'K').strip() or 'K'
    return SpectralCube(
        path=os.path.abspath(path),
        hdu_index=hdu_index,
        spectra=spectra,
        velocity_kms=np.asarray(v, dtype=float),
        freq_ghz=(None if rest_ghz is None else
                  rest_ghz * (1.0 - np.asarray(v, dtype=float) / osf._C_LIGHT_KMS)),
        ra_off_deg=ra_off,
        dec_off_deg=dec_off,
        iy=iy,
        ix=ix,
        ra_axis_deg=np.asarray(ra_axis, dtype=float),
        dec_axis_deg=np.asarray(dec_axis, dtype=float),
        row_index=row_index,
        restfreq_ghz=rest_ghz,
        deltav_kms=deltav,
        crval2_deg=crval2,
        crval3_deg=crval3,
        object_name=str(header.get('OBJECT', '') or ''),
        species_label=species_label_from_path(path, rest_ghz),
        intensity_unit=bunit,
        is_main_beam=('_mb' in os.path.basename(path).lower()),
        source_kind='image_cube',
        grid_step_arcsec=step_as,
    )


def load_spectral_cube(path: str, hdu_index: Optional[int] = None,
                       use_cache: bool = True) -> SpectralCube:
    """Load a CLASS MATRIX table or a spectral image cube, with a small memory cache."""
    if fits is None:
        raise ImportError('astropy is required to read spectral cubes')
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if hdu_index is None:
        hdu_index = default_hdu_index(path)
    hdu_index = int(hdu_index)
    mtime = os.path.getmtime(path)
    key = (path, hdu_index)
    if use_cache and key in _cube_cache and _cube_cache[key][0] == mtime:
        return _cube_cache[key][1]

    with fits.open(path, memmap=False) as hdul:
        if hdu_index >= len(hdul):
            raise IndexError(f'HDU {hdu_index} not in {path} ({len(hdul)} HDUs)')
        hdu = hdul[hdu_index]
        data = hdu.data
        if data is None:
            raise ValueError(f'HDU {hdu_index} has no data')
        if hasattr(data, 'names'):
            cube = _load_class_matrix(hdu, path, hdu_index)
        else:
            cube = _load_image_cube(hdu, path, hdu_index)

    if use_cache:
        _cube_cache[key] = (mtime, cube)
        if len(_cube_cache) > _CACHE_MAX:
            oldest = next(iter(_cube_cache))
            if oldest != key:
                _cube_cache.pop(oldest, None)
    return cube


def clear_cube_cache():
    _cube_cache.clear()


def collapse_map(cube: SpectralCube, v_low: float, v_high: float,
                 metric: str = 'moment0') -> np.ndarray:
    """2-D map (ny, nx); blank cells stay NaN (no interpolation)."""
    v = cube.velocity_kms
    chan = (v >= float(v_low)) & (v <= float(v_high))
    if not np.any(chan):
        return np.full((cube.ny, cube.nx), np.nan)
    line = cube.spectra[:, chan]
    metric_l = str(metric or 'moment0').strip().lower()
    if metric_l in ('moment0', 'mom0', 'integrated'):
        values = np.nansum(line, axis=1) * abs(float(cube.deltav_kms))
    elif metric_l in ('peak', 'mom8'):
        values = np.nanmax(line, axis=1)
    else:
        raise ValueError("metric must be 'moment0' or 'peak'")
    out = np.full((cube.ny, cube.nx), np.nan, dtype=float)
    filled = cube.row_index >= 0
    if not np.any(filled):
        return out
    rows = cube.row_index[filled]
    out[filled] = values[rows]
    return out


def peak_cell(cube: SpectralCube, map2d: np.ndarray
              ) -> Optional[Tuple[int, int, int]]:
    """(iy, ix, row) of the brightest finite map pixel."""
    finite = np.isfinite(map2d)
    if not np.any(finite):
        return None
    masked = np.where(finite, map2d, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(masked)), map2d.shape)
    row = int(cube.row_index[iy, ix])
    if row < 0:
        return None
    return int(iy), int(ix), row


def nearest_filled_cell(cube: SpectralCube, ra_arcmin: float, dec_arcmin: float
                        ) -> Optional[Tuple[int, int, int]]:
    """Nearest row that actually has a spectrum (skip blank OTF cells)."""
    if cube.n_spectra == 0:
        return None
    ra = cube.ra_off_deg * 60.0
    dec = cube.dec_off_deg * 60.0
    d2 = (ra - float(ra_arcmin)) ** 2 + (dec - float(dec_arcmin)) ** 2
    i = int(np.argmin(d2))
    return int(cube.iy[i]), int(cube.ix[i]), i


def rows_in_box(cube: SpectralCube, ra_min: float, ra_max: float,
                dec_min: float, dec_max: float) -> np.ndarray:
    """Table rows whose offsets (arcmin) fall inside a box (any axis order)."""
    ra0, ra1 = sorted((float(ra_min), float(ra_max)))
    dec0, dec1 = sorted((float(dec_min), float(dec_max)))
    ra = cube.ra_off_deg * 60.0
    dec = cube.dec_off_deg * 60.0
    mask = (ra >= ra0) & (ra <= ra1) & (dec >= dec0) & (dec <= dec1)
    return np.where(mask)[0]


def extract_spectrum(cube: SpectralCube, rows: Sequence[int]
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Mean spectrum over ``rows`` (a single pixel or a region)."""
    idx = np.asarray(list(rows), dtype=int)
    idx = idx[(idx >= 0) & (idx < cube.n_spectra)]
    if idx.size == 0:
        raise ValueError('no valid spectra in selection')
    spec = cube.spectra[idx]
    if spec.shape[0] == 1:
        y = spec[0]
    else:
        y = np.nanmean(spec, axis=0)
    return cube.velocity_kms.copy(), np.asarray(y, dtype=float)


def fwhm_halfmax_kms(velocity_kms: np.ndarray, flux: np.ndarray,
                     continuum: float = 0.0) -> float:
    """FWHM (km/s) from linear interpolation around the half-maximum."""
    v = np.asarray(velocity_kms, dtype=float)
    y = np.asarray(flux, dtype=float) - float(continuum)
    finite = np.isfinite(v) & np.isfinite(y)
    v, y = v[finite], y[finite]
    if v.size < 3:
        return float('nan')
    ipeak = int(np.argmax(y))
    peak = float(y[ipeak])
    if peak <= 0.0:
        return float('nan')
    half = 0.5 * peak

    def _cross(side: str) -> float:
        if side == 'left':
            sl = slice(None, ipeak + 1)
            yy, vv = y[sl], v[sl]
            below = np.where(yy <= half)[0]
            if below.size == 0:
                return float(vv[0])
            i = int(below[-1])
            if i >= yy.size - 1:
                return float(vv[-1])
            y0, y1 = float(yy[i]), float(yy[i + 1])
            if y1 == y0:
                return float(vv[i])
            frac = (half - y0) / (y1 - y0)
            return float(vv[i] + frac * (vv[i + 1] - vv[i]))
        sl = slice(ipeak, None)
        yy, vv = y[sl], v[sl]
        below = np.where(yy <= half)[0]
        if below.size == 0:
            return float(vv[-1])
        i = int(below[0])
        if i == 0:
            return float(vv[0])
        y0, y1 = float(yy[i - 1]), float(yy[i])
        if y1 == y0:
            return float(vv[i])
        frac = (half - y0) / (y1 - y0)
        return float(vv[i - 1] + frac * (vv[i] - vv[i - 1]))

    left = _cross('left')
    right = _cross('right')
    width = abs(right - left)
    return float(width) if np.isfinite(width) and width > 0 else float('nan')


def measure_spectrum(velocity_kms: np.ndarray, flux: np.ndarray,
                     v_low: float, v_high: float) -> Dict[str, float]:
    """Line-window peak, integrated intensity, centroid, RMS, measured FWHM."""
    v = np.asarray(velocity_kms, dtype=float)
    y = np.asarray(flux, dtype=float)
    line = (v >= float(v_low)) & (v <= float(v_high)) & np.isfinite(v) & np.isfinite(y)
    continuum = ~line & np.isfinite(v) & np.isfinite(y)
    rms, mad = osf.noise_estimates_line_free(y, continuum)
    dv = abs(float(np.median(np.diff(v)))) if v.size > 1 else float('nan')

    empty = dict(
        peak_k=float('nan'), v_peak_kms=float('nan'),
        integrated_k_kms=float('nan'), centroid_kms=float('nan'),
        fwhm_kms=float('nan'), rms_k=float(rms), mad_k=float(mad),
        snr=float('nan'), mean_k=float('nan'), dv_kms=dv,
    )
    if not np.any(line):
        return empty

    y_line, v_line = y[line], v[line]
    ipeak = int(np.argmax(y_line))
    peak = float(y_line[ipeak])
    v_peak = float(v_line[ipeak])
    integ = float(np.nansum(y_line) * dv) if np.isfinite(dv) else float('nan')
    w = np.clip(y_line, 0.0, None)
    wsum = float(np.nansum(w))
    centroid = float(np.nansum(v_line * w) / wsum) if wsum > 0 else float('nan')
    fwhm = fwhm_halfmax_kms(v_line, y_line, continuum=0.0)
    snr = peak / rms if np.isfinite(rms) and rms > 0 else float('nan')
    empty.update(
        peak_k=peak, v_peak_kms=v_peak, integrated_k_kms=integ,
        centroid_kms=centroid, fwhm_kms=fwhm, snr=snr,
        mean_k=float(np.nanmean(y_line)),
    )
    return empty


def selection_sky(cube: SpectralCube, rows: Sequence[int]) -> Dict[str, Any]:
    """Mean RA/Dec offsets (and absolute coords if the map centre is known)."""
    idx = np.asarray(list(rows), dtype=int)
    idx = idx[(idx >= 0) & (idx < cube.n_spectra)]
    if idx.size == 0:
        return {}
    ra_off = float(np.mean(cube.ra_off_deg[idx]))
    dec_off = float(np.mean(cube.dec_off_deg[idx]))
    info: Dict[str, Any] = {
        'n_pixels': int(idx.size),
        'ra_off_arcmin': ra_off * 60.0,
        'dec_off_arcmin': dec_off * 60.0,
        'ra_off_arcsec': ra_off * 3600.0,
        'dec_off_arcsec': dec_off * 3600.0,
        'iy': int(np.round(np.mean(cube.iy[idx]))),
        'ix': int(np.round(np.mean(cube.ix[idx]))),
        'row_index': int(idx[0]) if idx.size == 1 else None,
        'rows': [int(r) for r in idx],
    }
    if cube.crval2_deg is not None:
        info['ra_deg'] = cube.crval2_deg + ra_off
    if cube.crval3_deg is not None:
        info['dec_deg'] = cube.crval3_deg + dec_off
    return info


def customdata_stack(cube: SpectralCube) -> np.ndarray:
    """(ny, nx, 3) array of [row, iy, ix] for Plotly heatmap hover / click."""
    ny, nx = cube.ny, cube.nx
    iy = np.repeat(np.arange(ny)[:, None], nx, axis=1)
    ix = np.repeat(np.arange(nx)[None, :], ny, axis=0)
    return np.stack([cube.row_index, iy, ix], axis=-1)


def parse_click_point(point: Dict[str, Any], cube: SpectralCube
                      ) -> Optional[Tuple[int, int, int]]:
    """(iy, ix, row) from a Plotly heatmap click/select point."""
    cd = point.get('customdata')
    row = iy = ix = None
    if isinstance(cd, (list, tuple, np.ndarray)) and len(cd) >= 1:
        try:
            row = int(cd[0])
            if len(cd) >= 3:
                iy, ix = int(cd[1]), int(cd[2])
        except (TypeError, ValueError):
            row = None
    elif cd is not None and not isinstance(cd, (list, tuple, np.ndarray)):
        try:
            row = int(cd)
        except (TypeError, ValueError):
            row = None
    if row is not None and row >= 0 and row < cube.n_spectra:
        if iy is None:
            iy, ix = int(cube.iy[row]), int(cube.ix[row])
        return iy, ix, row
    x = point.get('x')
    y = point.get('y')
    if x is None or y is None:
        return None
    return nearest_filled_cell(cube, float(x), float(y))


def rows_from_selected_points(points: Sequence[Dict[str, Any]], cube: SpectralCube
                              ) -> List[int]:
    rows = []
    seen = set()
    for pt in points:
        parsed = parse_click_point(pt, cube)
        if parsed is None:
            continue
        _iy, _ix, row = parsed
        if row not in seen:
            seen.add(row)
            rows.append(row)
    return rows
