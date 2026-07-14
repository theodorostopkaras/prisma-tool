"""
SimLine position-velocity FITS spectra (vendored from KoSens ``simline_functions``).

Reads PV cubes written by SimLine (``-fits``) and extracts 1-D spectra or 2-D
PV maps.  FITS naming convention (same directory as ``.smli`` files)::

    Model<tag>_DD_MM_FF_ZZ_CC_AA_<species>.<transition>.fits
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

import grid_naming as gn

try:
    from astropy.io import fits
except ImportError:  # pragma: no cover
    fits = None

N_GRID_TOKENS = 6
_TAU_SUFFIX = '-tau'


def tokens_from_model_core(model_core: str) -> Optional[Tuple[int, ...]]:
    """Parse ``Model…_DD_MM_FF_ZZ_CC[_AA]`` into six integer grid tokens."""
    return gn.parse_model_tokens_from_stem(model_core)


def parse_pv_fits_filename(path: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse ``Model…_<species>.<transition>.fits`` or ``…<transition>-tau.fits``.

    Returns ``(model_core, species, transition, quantity)`` where ``quantity``
    is ``'intensity'`` (brightness temperature) or ``'tau'`` (optical depth).
    """
    base = os.path.basename(path)
    if not base.lower().endswith('.fits'):
        return None
    stem = base[:-5]
    if '.' not in stem:
        return None
    namebase, transition = stem.rsplit('.', 1)
    quantity = 'intensity'
    if transition.endswith(_TAU_SUFFIX):
        transition = transition[:-len(_TAU_SUFFIX)]
        quantity = 'tau'
    if not transition:
        return None
    if '_' not in namebase:
        return None
    species = namebase.rsplit('_', 1)[-1]
    model_core = namebase[: -(len(species) + 1)]
    if not model_core.startswith('Model'):
        return None
    return model_core, species, transition, quantity


def expected_pv_fits_path(directory: str, model_core: str, species: str,
                          transition: str, quantity: str = 'intensity') -> str:
    """Build the canonical PV FITS path for one model / species / transition."""
    suffix = f'{transition}-tau' if quantity == 'tau' else transition
    return os.path.join(
        directory,
        f'{model_core}_{species}.{suffix}.fits',
    )


def scan_pv_index(directory: str, recursive: bool = False
                  ) -> Tuple[Dict[Tuple[Tuple[int, ...], str, str], str],
                             Dict[Tuple[Tuple[int, ...], str], List[str]],
                             Dict[Tuple[Tuple[int, ...], str, str], str],
                             Dict[Tuple[Tuple[int, ...], str], List[str]]]:
    """
    Index SimLine PV FITS files under ``directory``.

    Returns
    -------
    index, by_model_species
        Brightness-temperature cubes ``{(tokens, species, transition): path}``
    tau_index, tau_by_model_species
        Optical-depth cubes (``*-tau.fits``) with the same key layout
    """
    directory = os.path.abspath(os.path.expanduser(directory))
    pattern = (
        os.path.join(directory, '**', '*.fits') if recursive
        else os.path.join(directory, '*.fits')
    )
    index: Dict[Tuple[Tuple[int, ...], str, str], str] = {}
    by_model_species: Dict[Tuple[Tuple[int, ...], str], List[str]] = {}
    tau_index: Dict[Tuple[Tuple[int, ...], str, str], str] = {}
    tau_by_model_species: Dict[Tuple[Tuple[int, ...], str], List[str]] = {}

    for path in sorted(glob.glob(pattern, recursive=recursive)):
        parsed = parse_pv_fits_filename(path)
        if parsed is None:
            continue
        model_core, species, transition, quantity = parsed
        tokens = tokens_from_model_core(model_core)
        if tokens is None:
            continue
        key = (tokens, species, transition)
        if quantity == 'tau':
            store, by_store = tau_index, tau_by_model_species
        else:
            store, by_store = index, by_model_species
        store[key] = path
        msk = (tokens, species)
        by_store.setdefault(msk, [])
        if transition not in by_store[msk]:
            by_store[msk].append(transition)

    for by_store in (by_model_species, tau_by_model_species):
        for msk in by_store:
            by_store[msk] = sorted(by_store[msk])
    return index, by_model_species, tau_index, tau_by_model_species


def _velocity_axis_from_header(header) -> np.ndarray:
    """Velocity axis (km/s) from FITS WCS keywords on axis 2."""
    n_vel = int(header.get('NAXIS2', 0))
    crpix2 = float(header.get('CRPIX2', 1.0))
    crval2 = float(header.get('CRVAL2', 0.0))
    cdelt2 = float(header.get('CDELT2', 1.0))
    cunit2 = str(header.get('CUNIT2', 'km/s'))

    velocities = (np.arange(n_vel) + 1 - crpix2) * cdelt2 + crval2
    if cunit2.strip().lower() in ('m/s', 'ms-1'):
        velocities = velocities / 1e3
    return velocities


def _position_axis_from_header(header) -> np.ndarray:
    """Position offset axis (arcsec) from FITS WCS keywords on axis 1."""
    n_pos = int(header.get('NAXIS1', 0))
    crpix1 = float(header.get('CRPIX1', 1.0))
    crval1 = float(header.get('CRVAL1', 0.0))
    cdelt1 = float(header.get('CDELT1', 1.0))

    positions = (np.arange(n_pos) + 1 - crpix1) * cdelt1 + crval1
    return positions * 3600.0


def _axis_edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges for an evenly spaced coordinate axis."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    step = centers[1] - centers[0]
    return np.concatenate([centers - step / 2.0, [centers[-1] + step / 2.0]])


def transition_from_header(header) -> str:
    """Extract transition label from SimLine COMMENT cards."""
    comments = header.get('COMMENT', [])
    if not isinstance(comments, (list, np.ndarray)):
        comments = [comments]
    for card in comments:
        text = str(card)
        if 'Transition:' in text:
            part = text.split('Transition:', 1)[1]
            if 'Rest' in part:
                part = part.split('Rest', 1)[0]
            return part.strip()
    return ''


def brightness_unit_from_header(header) -> str:
    bunit = str(header.get('BUNIT', 'K')).strip() or 'K'
    return bunit


def pv_header_info(fits_path: str) -> Tuple[str, str]:
    """Return ``(brightness_unit, transition_label)`` from a PV FITS file."""
    if fits is None:
        raise ImportError('astropy is required to read SimLine FITS spectra')
    with fits.open(os.path.expanduser(fits_path)) as hdul:
        header = hdul[0].header
    return brightness_unit_from_header(header), transition_from_header(header)


def load_and_average_spectrum(fits_path: str, positions=None):
    """
    Load a SimLine PV FITS cube and return velocity + spectrum.

    Data shape is ``(n_vel, n_pos)``.  By default the mean over all positions
    is returned (notebook default).  Pass ``positions`` (arcsec) to extract
    the nearest column(s) instead.
    """
    if fits is None:
        raise ImportError('astropy is required to read SimLine FITS spectra')

    with fits.open(os.path.expanduser(fits_path)) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header

    velocities = _velocity_axis_from_header(header)

    if positions is None:
        spectrum = np.nanmean(data, axis=1)
        return velocities, spectrum, None

    position_axis = _position_axis_from_header(header)
    is_scalar = np.ndim(positions) == 0
    requested = np.atleast_1d(np.asarray(positions, dtype=float))
    indices = np.array(
        [int(np.argmin(np.abs(position_axis - p))) for p in requested]
    )
    spectra = data[:, indices].T
    selected_positions = position_axis[indices]

    if is_scalar:
        return velocities, spectra[0], float(selected_positions[0])
    return velocities, spectra, selected_positions


def load_pv_diagram(fits_path: str, position_range=None):
    """
    Load a SimLine PV map without collapsing either axis.

    Returns ``positions`` (arcsec), ``velocities`` (km/s), ``data`` (K),
    and the FITS ``header``.
    """
    if fits is None:
        raise ImportError('astropy is required to read SimLine FITS spectra')

    with fits.open(os.path.expanduser(fits_path)) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header

    positions = _position_axis_from_header(header)
    velocities = _velocity_axis_from_header(header)

    if position_range is not None:
        p_min, p_max = position_range
        mask = (positions >= p_min) & (positions <= p_max)
        positions = positions[mask]
        data = data[:, mask]

    return positions, velocities, data, header


def parse_position_list(text: Optional[str]) -> Optional[List[float]]:
    """Parse comma / whitespace separated position offsets (arcsec)."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    parts = re.split(r'[,;\s]+', text)
    out = []
    for part in parts:
        if not part:
            continue
        out.append(float(part))
    return out or None
