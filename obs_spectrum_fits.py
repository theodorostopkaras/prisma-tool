"""
Observational 1-D spectral line fitting (vendored from KoSens ``spectrum_fits``).

Reads CLASS/GILDAS MATRIX tables or spectral image cubes, extracts spectra
(mean / peak-region / single row), fits a polynomial continuum and sum of
Gaussians, and reports integrated intensities with propagated uncertainties.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from astropy.io import fits
    from astropy.modeling import fitting, models
except ImportError:  # pragma: no cover
    fits = None
    fitting = None
    models = None

_C_LIGHT_KMS = 299792.458


# ---------------------------------------------------------------------------
# Velocity axes
# ---------------------------------------------------------------------------

def velocity_axis_lsr_kms(header, nchan=None):
    """LSR velocity (km/s) from CLASS MATRIX header (VELO-LSR, CRPIX1, DELTAV m/s)."""
    if nchan is None:
        nchan = int(header['MAXIS1'])
    j = np.arange(1, nchan + 1, dtype=float)
    v_ref = float(header['VELO-LSR'])
    crpix = float(header['CRPIX1'])
    deltav_m_s = float(header['DELTAV'])
    return v_ref + (j - crpix) * (deltav_m_s * 1.0e-3)


def _header_channel_axis_values(header, axis_index):
    naxis = int(header[f'NAXIS{axis_index}'])
    j = np.arange(1, naxis + 1, dtype=float)
    crpix = float(header.get(f'CRPIX{axis_index}', 1.0))
    crval = float(header[f'CRVAL{axis_index}'])
    cdelt = float(header[f'CDELT{axis_index}'])
    return crval + (j - crpix) * cdelt


def radio_velocity_kms_from_frequency_ghz(observed_freq_ghz, rest_freq_ghz):
    nu = np.asarray(observed_freq_ghz, dtype=float)
    nu0 = float(rest_freq_ghz)
    return _C_LIGHT_KMS * (nu - nu0) / nu0


def rest_frequency_hz_from_header(header):
    for key in ('RESTFREQ', 'RESTFRQ', 'LINEFREQ', 'FREQ0'):
        if key not in header:
            continue
        try:
            val = float(header[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(val) and val > 0.0:
            return val
    return None


def _spectral_axis_from_header_kms(header):
    naxis = int(header.get('NAXIS', 0))
    if naxis < 1:
        raise ValueError('header has no FITS axes (NAXIS < 1)')

    candidates = []
    for ax in range(1, naxis + 1):
        ctype = str(header.get(f'CTYPE{ax}', '')).strip().upper()
        if any(tag in ctype for tag in ('VRAD', 'VELO', 'VOPT', 'FELO', 'FREQ')):
            candidates.append((ax, ctype))
    if not candidates:
        raise KeyError('could not identify a spectral axis from CTYPE* keywords')

    spectral_axis, ctype = candidates[0]
    axis_vals = _header_channel_axis_values(header, spectral_axis)
    cunit = str(header.get(f'CUNIT{spectral_axis}', '')).strip().lower()

    if 'freq' in ctype:
        rest_hz = rest_frequency_hz_from_header(header)
        if rest_hz is None:
            raise ValueError(
                'frequency spectral axis requires RESTFREQ / RESTFRQ / LINEFREQ / FREQ0')
        if 'ghz' in cunit:
            obs_ghz = axis_vals
        elif 'mhz' in cunit:
            obs_ghz = axis_vals / 1.0e3
        elif 'khz' in cunit:
            obs_ghz = axis_vals / 1.0e6
        else:
            obs_ghz = axis_vals / 1.0e9
        velocity_kms = radio_velocity_kms_from_frequency_ghz(
            obs_ghz, rest_hz / 1.0e9)
    else:
        if 'km/s' in cunit or 'km s-1' in cunit:
            velocity_kms = axis_vals
        else:
            velocity_kms = axis_vals * 1.0e-3
    return spectral_axis, np.asarray(velocity_kms, dtype=float)


# ---------------------------------------------------------------------------
# Cube / table loading
# ---------------------------------------------------------------------------

def _reshape_cube_to_spectra(cube_data, spectral_axis_1based):
    data = np.asarray(cube_data, dtype=float)
    spectral_numpy_axis = data.ndim - int(spectral_axis_1based)
    if spectral_numpy_axis < 0 or spectral_numpy_axis >= data.ndim:
        raise ValueError(
            f'spectral axis {spectral_axis_1based} invalid for ndim {data.ndim}')
    moved = np.moveaxis(data, spectral_numpy_axis, -1)
    return moved.reshape(-1, moved.shape[-1])


def _cube_spectral_last(cube_data, spectral_axis_1based):
    data = np.asarray(cube_data, dtype=float)
    spectral_numpy_axis = data.ndim - int(spectral_axis_1based)
    return np.moveaxis(data, spectral_numpy_axis, -1)


def _line_core_channel_mask(velocity_kms, low_line_limit, high_line_limit):
    v = np.asarray(velocity_kms, dtype=float)
    return (v > float(low_line_limit)) & (v < float(high_line_limit))


def _spatial_intensity_map(cube_spectral_last, line_channel_mask, metric='integrated'):
    line_data = cube_spectral_last[..., line_channel_mask]
    if line_data.shape[-1] < 1:
        raise ValueError('no channels inside line velocity window')
    metric_l = str(metric).strip().lower()
    if metric_l == 'integrated':
        return np.nansum(line_data, axis=-1)
    if metric_l == 'peak':
        return np.nanmax(line_data, axis=-1)
    raise ValueError("peak_intensity_metric must be 'integrated' or 'peak'")


def _peak_region_row_mask(intensity_map, threshold_fraction=0.5,
                          absolute_threshold=None):
    frac = float(threshold_fraction)
    flat = np.asarray(intensity_map, dtype=float).ravel()
    finite = np.isfinite(flat)
    if not np.any(finite):
        raise ValueError('no finite intensity values in the spatial map')
    peak_val = float(np.nanmax(flat[finite]))
    thresh = float(absolute_threshold) if absolute_threshold is not None else frac * peak_val
    row_mask = finite & (flat >= thresh)
    if not np.any(row_mask):
        raise ValueError('peak-region mask is empty')
    return row_mask, thresh, peak_val


def _peak_region_mask_from_cube(cube_data, spectral_axis_1based, velocity_kms,
                                low_line_limit, high_line_limit,
                                peak_threshold_fraction=0.5,
                                peak_absolute_threshold=None,
                                peak_intensity_metric='integrated'):
    cube_sl = _cube_spectral_last(cube_data, spectral_axis_1based)
    line_mask = _line_core_channel_mask(velocity_kms, low_line_limit, high_line_limit)
    intensity_map = _spatial_intensity_map(cube_sl, line_mask, metric=peak_intensity_metric)
    row_mask, thresh, peak_val = _peak_region_row_mask(
        intensity_map, threshold_fraction=peak_threshold_fraction,
        absolute_threshold=peak_absolute_threshold)
    return {
        'intensity_map': intensity_map,
        'row_mask': row_mask,
        'intensity_threshold': thresh,
        'peak_intensity': peak_val,
        'n_selected': int(np.count_nonzero(row_mask)),
    }


def _peak_region_mask_from_spectra_table(spectra, velocity_kms, low_line_limit,
                                         high_line_limit, peak_threshold_fraction=0.5,
                                         peak_absolute_threshold=None,
                                         peak_intensity_metric='integrated'):
    spec = np.asarray(spectra, dtype=float)
    line_mask = _line_core_channel_mask(velocity_kms, low_line_limit, high_line_limit)
    line_data = spec[:, line_mask]
    if line_data.shape[-1] < 1:
        raise ValueError('no channels inside line velocity window')
    metric_l = str(peak_intensity_metric).strip().lower()
    if metric_l == 'integrated':
        intensity_map = np.nansum(line_data, axis=1)
    elif metric_l == 'peak':
        intensity_map = np.nanmax(line_data, axis=1)
    else:
        raise ValueError("peak_intensity_metric must be 'integrated' or 'peak'")
    row_mask, thresh, peak_val = _peak_region_row_mask(
        intensity_map, threshold_fraction=peak_threshold_fraction,
        absolute_threshold=peak_absolute_threshold)
    return {
        'intensity_map': intensity_map,
        'row_mask': row_mask,
        'intensity_threshold': thresh,
        'peak_intensity': peak_val,
        'n_selected': int(np.count_nonzero(row_mask)),
    }


def _aggregate_spectra_and_noise(v, spectra, row, low_line_limit, high_line_limit, *,
                                 cube_data=None, spectral_axis_1based=None,
                                 peak_threshold_fraction=0.5,
                                 peak_absolute_threshold=None,
                                 peak_intensity_metric='integrated',
                                 velocity_kms_native=None):
    n_spectra = int(spectra.shape[0])
    continuum_for_agg = (v <= low_line_limit) | (v >= high_line_limit)
    peak_info = None
    v_peak = v if velocity_kms_native is None else velocity_kms_native

    if isinstance(row, str):
        row_key = row.strip().lower()
    else:
        row_key = None

    if row_key == 'mean':
        selected = spectra
        spectrum_selection = 'mean'
    elif row_key == 'peak':
        if cube_data is not None and spectral_axis_1based is not None:
            peak_info = _peak_region_mask_from_cube(
                cube_data, spectral_axis_1based, v_peak,
                low_line_limit, high_line_limit,
                peak_threshold_fraction=peak_threshold_fraction,
                peak_absolute_threshold=peak_absolute_threshold,
                peak_intensity_metric=peak_intensity_metric)
        elif spectra.ndim == 2 and n_spectra > 1:
            peak_info = _peak_region_mask_from_spectra_table(
                spectra, v_peak, low_line_limit, high_line_limit,
                peak_threshold_fraction=peak_threshold_fraction,
                peak_absolute_threshold=peak_absolute_threshold,
                peak_intensity_metric=peak_intensity_metric)
        else:
            raise ValueError('row="peak" requires a cube or multi-row table')
        selected = spectra[peak_info['row_mask']]
        spectrum_selection = 'peak'
    else:
        ri = int(row)
        if ri < 0 or ri >= n_spectra:
            raise IndexError(f'row index {ri} out of range for {n_spectra} spectrum(s)')
        y = np.asarray(spectra[ri], dtype=float)
        return {
            'y': y, 'spectrum_selection': f'row_{ri}', 'row_index': ri,
            'n_spectra': n_spectra, 'n_spectra_selected': 1,
            'mean_noise_rms': None, 'mean_noise_mad': None,
            'noise_rms_per_row': None, 'noise_mad_per_row': None,
            'peak_info': None,
        }

    y = np.nanmean(selected, axis=0)
    noise_rms_per_row = np.empty(n_spectra, dtype=float)
    noise_mad_per_row = np.empty(n_spectra, dtype=float)
    for i in range(n_spectra):
        rms_i, mad_i = noise_estimates_line_free(spectra[i], continuum_for_agg)
        noise_rms_per_row[i] = rms_i
        noise_mad_per_row[i] = mad_i
    return {
        'y': y,
        'spectrum_selection': spectrum_selection,
        'row_index': None,
        'n_spectra': n_spectra,
        'n_spectra_selected': int(selected.shape[0]),
        'mean_noise_rms': float(np.nanmean(noise_rms_per_row)),
        'mean_noise_mad': float(np.nanmean(noise_mad_per_row)),
        'noise_rms_per_row': noise_rms_per_row,
        'noise_mad_per_row': noise_mad_per_row,
        'peak_info': peak_info,
    }


def load_spectra_from_fits(file, hdu_index=1, spectrum_column='SPECTRUM',
                           resample_vel_res_kms=None):
    """Load velocity axis and spectra array from a FITS cube or MATRIX table."""
    if fits is None:
        raise ImportError('astropy is required to read observational FITS spectra')

    path = os.path.abspath(os.path.expanduser(str(file)))
    with fits.open(path) as hdul:
        hdu = hdul[hdu_index]
        h = hdu.header
        data = hdu.data
        if data is None:
            raise ValueError(f'HDU {hdu_index} has no data')

        cube_data = None
        spectral_axis = None
        if hasattr(data, 'names'):
            v = velocity_axis_lsr_kms(h)
            names = getattr(data, 'names', None)
            if names is not None and spectrum_column not in names:
                raise KeyError(
                    f'column {spectrum_column!r} not in table; available: {list(names)}')
            spectra = np.asarray(data[spectrum_column], dtype=float)
            if spectra.ndim == 1:
                spectra = spectra.reshape(1, -1)
        else:
            spectral_axis, v = _spectral_axis_from_header_kms(h)
            cube_data = np.asarray(data, dtype=float)
            spectra = _reshape_cube_to_spectra(cube_data, spectral_axis)

    if spectra.shape[-1] != v.size:
        raise ValueError(
            f'spectral length {spectra.shape[-1]} != velocity length {v.size}')

    v_native = np.asarray(v, dtype=float).copy()
    if resample_vel_res_kms is not None:
        step_out = float(resample_vel_res_kms)
        if step_out <= 0.0:
            raise ValueError('resample_vel_res_kms must be > 0')
        v_sorted = np.sort(v)
        v_out = np.arange(float(v_sorted[0]), float(v_sorted[-1]) + 0.5 * step_out,
                          step_out, dtype=float)
        spectra_out = np.empty((spectra.shape[0], v_out.size), dtype=float)
        for i in range(spectra.shape[0]):
            spectra_out[i] = np.interp(v_out, v, spectra[i])
        v = v_out
        spectra = spectra_out

    return v, spectra, cube_data, spectral_axis, v_native, path


# ---------------------------------------------------------------------------
# Gaussian models and fitting
# ---------------------------------------------------------------------------

def gaussian(x, amplitude, center, sigma, offset=0.0):
    x = np.asarray(x, dtype=float)
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def multi_gaussian_sum(x, amplitudes, centers, sigmas, offset=0.0):
    x = np.asarray(x, dtype=float)
    total = np.full_like(x, float(offset), dtype=float)
    for ai, ci, si in zip(amplitudes, centers, sigmas):
        total = total + ai * np.exp(-0.5 * ((x - ci) / si) ** 2)
    return total


def noise_estimates_line_free(flux, continuum_mask):
    y = np.asarray(flux, dtype=float)
    m = np.asarray(continuum_mask, dtype=bool)
    y_free = y[m]
    if y_free.size == 0:
        return float('nan'), float('nan')
    median = np.median(y_free)
    mad = np.median(np.abs(y_free - median))
    rms = float(np.sqrt(np.mean((y_free - np.mean(y_free)) ** 2)))
    return rms, 1.4826 * mad


def fit_polynomial_continuum(velocity_kms, flux, continuum_mask, degree=1):
    if models is None or fitting is None:
        raise ImportError('astropy.modeling is required for continuum fitting')
    v = np.asarray(velocity_kms, dtype=float)
    y = np.asarray(flux, dtype=float)
    m = np.asarray(continuum_mask, dtype=bool) & np.isfinite(v) & np.isfinite(y)
    if np.count_nonzero(m) < degree + 1:
        raise ValueError(
            f'need at least {degree + 1} continuum points; got {np.count_nonzero(m)}')
    poly_init = models.Polynomial1D(degree=degree)
    fitter = fitting.LinearLSQFitter()
    poly_fit = fitter(poly_init, v[m], y[m])
    return poly_fit(v)


FWHM_OVER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548


def fwhm_to_sigma(fwhm_kms):
    return abs(float(fwhm_kms)) / FWHM_OVER_SIGMA


def sigma_to_fwhm(sigma_kms):
    return abs(float(sigma_kms)) * FWHM_OVER_SIGMA


def coerce_optional_floats(values, n):
    """Pad/trim a list of optional numbers to length *n* (None = unset)."""
    n = int(n)
    out = [None] * max(n, 0)
    for i, val in enumerate(list(values or [])[:n]):
        if val is None or val == '':
            continue
        try:
            x = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            out[i] = x
    return out


def auto_gaussian_p0(velocity_kms, flux, n_components=1):
    """Default LevMar starting vector: [A, v0, σ] × N + offset."""
    v = np.asarray(velocity_kms, dtype=float)
    y = np.asarray(flux, dtype=float)
    finite = np.isfinite(v) & np.isfinite(y)
    v, y = v[finite], y[finite]
    n_components = int(n_components)
    if y.size == 0:
        p0 = []
        for _ in range(n_components):
            p0.extend([0.0, 0.0, 1.0])
        p0.append(0.0)
        return tuple(p0)
    offset_guess = float(np.median(y))
    peak = float(np.max(y) - offset_guess)
    if abs(peak) < abs(float(np.min(y) - offset_guess)):
        peak = float(np.min(y) - offset_guess)
    center_guess = float(v[np.argmax(np.abs(y - offset_guess))])
    width = float(v.max() - v.min()) or 1.0
    p0_list = []
    for i in range(n_components):
        if i == 0:
            sigma_guess = max(width / 20.0, 1.0e-6)
            amp_guess = peak * 0.85 if peak != 0.0 else 0.0
        elif i == 1:
            sigma_guess = max(width / 5.0, 1.0e-6)
            amp_guess = peak * 0.45 if peak != 0.0 else 0.0
        else:
            sigma_guess = max(width / (3.0 + 2.0 * i), 1.0e-6)
            amp_guess = (peak * 0.25 / max(i, 1)) if peak != 0.0 else 0.0
        p0_list.extend([amp_guess, center_guess, sigma_guess])
    p0_list.append(offset_guess)
    return tuple(p0_list)


def merge_user_gaussian_p0(n_gaussians, auto_p0, amplitudes=None, centers=None,
                           fwhms=None, offset=None):
    """Overwrite auto-guess slots with user Peak / v0 / FWHM (unset stays auto).

    Returns ``(p0, user_set)`` where *user_set* is a bool list of the same
    length marking which parameters the user actually provided.
    """
    n = int(n_gaussians)
    n_params = 3 * n + 1
    p0 = list(auto_p0)
    if len(p0) != n_params:
        raise ValueError(f'auto_p0 must have length {n_params}; got {len(p0)}')
    user_set = [False] * n_params
    amps = coerce_optional_floats(amplitudes, n)
    ctrs = coerce_optional_floats(centers, n)
    widths = coerce_optional_floats(fwhms, n)
    for i in range(n):
        i0 = 3 * i
        if amps[i] is not None:
            p0[i0] = float(amps[i])
            user_set[i0] = True
        if ctrs[i] is not None:
            p0[i0 + 1] = float(ctrs[i])
            user_set[i0 + 1] = True
        if widths[i] is not None:
            p0[i0 + 2] = max(fwhm_to_sigma(widths[i]), 1.0e-6)
            user_set[i0 + 2] = True
    if offset is not None and offset != '':
        try:
            off = float(offset)
        except (TypeError, ValueError):
            off = None
        if off is not None and np.isfinite(off):
            p0[-1] = off
            user_set[-1] = True
    return tuple(p0), user_set


def fit_multi_gaussian(velocity_kms, flux, n_components=1, p0=None, bounds=None,
                       fixed=None):
    if models is None or fitting is None:
        raise ImportError('astropy.modeling is required for line fitting')
    v = np.asarray(velocity_kms, dtype=float)
    y = np.asarray(flux, dtype=float)
    finite = np.isfinite(v) & np.isfinite(y)
    v, y = v[finite], y[finite]
    n_components = int(n_components)
    n_params = 3 * n_components + 1
    if len(y) < n_params:
        raise ValueError(f'need at least {n_params} points for fit; got {len(y)}')

    if p0 is None:
        p0 = auto_gaussian_p0(v, y, n_components)
    else:
        p0 = tuple(p0)
        if len(p0) != n_params:
            raise ValueError(f'p0 must have length {n_params}')

    if fixed is None:
        fixed = [False] * n_params
    else:
        fixed = [bool(x) for x in fixed]
        if len(fixed) != n_params:
            raise ValueError(f'fixed must have length {n_params}')

    if bounds is None:
        lowers, uppers = [], []
        for _ in range(n_components):
            lowers.extend([-np.inf, -np.inf, 1.0e-6])
            uppers.extend([np.inf, np.inf, np.inf])
        lowers.append(-np.inf)
        uppers.append(np.inf)
        bounds = (tuple(lowers), tuple(uppers))
    lower, upper = bounds

    gaussians = []
    for i in range(n_components):
        i0 = 3 * i
        g = models.Gaussian1D(amplitude=p0[i0], mean=p0[i0 + 1], stddev=p0[i0 + 2], name=f'g{i}')
        g.bounds['amplitude'] = (lower[i0], upper[i0])
        g.bounds['mean'] = (lower[i0 + 1], upper[i0 + 1])
        g.bounds['stddev'] = (max(lower[i0 + 2], 1.0e-12), upper[i0 + 2])
        g.fixed['amplitude'] = fixed[i0]
        g.fixed['mean'] = fixed[i0 + 1]
        g.fixed['stddev'] = fixed[i0 + 2]
        gaussians.append(g)
    const = models.Const1D(amplitude=p0[-1], name='offset')
    const.bounds['amplitude'] = (lower[-1], upper[-1])
    const.fixed['amplitude'] = fixed[-1]

    if all(fixed):
        return np.asarray(p0, dtype=float), np.full((n_params, n_params), np.nan)

    model_init = gaussians[0]
    for g in gaussians[1:]:
        model_init = model_init + g
    model_init = model_init + const

    fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
    model_fit = fitter(model_init, v, y, maxiter=50000)
    popt = np.asarray(model_fit.parameters, dtype=float)
    pcov = fitter.fit_info.get('param_cov')
    if pcov is None:
        pcov = np.full((n_params, n_params), np.nan, dtype=float)
    return popt, pcov


def integrated_intensity_per_component_kms(amplitudes, sigmas):
    a = np.asarray(amplitudes, dtype=float).ravel()
    s = np.asarray(sigmas, dtype=float).ravel()
    return np.asarray(a * s * np.sqrt(2.0 * np.pi), dtype=float)


def integrated_intensity_total_error_kms(amplitudes, sigmas, pcov, n_components):
    a = np.asarray(amplitudes, dtype=float).ravel()
    s = np.asarray(sigmas, dtype=float).ravel()
    n_g = int(n_components)
    n_par = 3 * n_g + 1
    pcov = np.asarray(pcov, dtype=float)
    if pcov.shape != (n_par, n_par) or not np.any(np.isfinite(pcov)):
        return float('nan')
    c = np.sqrt(2.0 * np.pi)
    grad = np.zeros(n_par, dtype=float)
    for i in range(n_g):
        grad[3 * i] = c * s[i]
        grad[3 * i + 2] = c * a[i]
    var = float(grad @ pcov @ grad)
    return float(np.sqrt(var)) if var >= 0.0 and np.isfinite(var) else float('nan')


def _fit_gaussian_line_core(v, y, low_line_limit, high_line_limit, degree=1,
                            n_gaussians=1, line_p0=None, line_bounds=None,
                            amplitudes=None, centers=None, fwhms=None,
                            offset=None, lock_user=False):
    continuum_mask = (v <= low_line_limit) | (v >= high_line_limit)
    noise_sigma_rms, noise_sigma_mad = noise_estimates_line_free(y, continuum_mask)
    cont = fit_polynomial_continuum(v, y, continuum_mask, degree=degree)
    residual = y - cont
    n_g = int(n_gaussians)
    if line_p0 is None:
        auto = auto_gaussian_p0(v, residual, n_g)
        line_p0, user_set = merge_user_gaussian_p0(
            n_g, auto, amplitudes=amplitudes, centers=centers,
            fwhms=fwhms, offset=offset)
        fixed = user_set if lock_user else None
    else:
        fixed = None
    popt, pcov = fit_multi_gaussian(
        v, residual, n_components=n_g, p0=line_p0, bounds=line_bounds,
        fixed=fixed)
    amps = popt[0:3 * n_g:3]
    ctrs = popt[1:3 * n_g:3]
    sigs = popt[2:3 * n_g:3]
    off = float(popt[-1])
    gaussian_components = np.empty((n_g, v.size), dtype=float)
    for i in range(n_g):
        gaussian_components[i] = gaussian(v, amps[i], ctrs[i], sigs[i], 0.0)
    model_line = np.sum(gaussian_components, axis=0)
    return {
        'continuum_mask': continuum_mask,
        'cont': cont,
        'residual': residual,
        'model_line': model_line,
        'gaussian_components': gaussian_components,
        'fit_total': cont + model_line + off,
        'n_components': n_g,
        'amplitudes': amps,
        'centers': ctrs,
        'sigmas': sigs,
        'offset': off,
        'popt': popt,
        'pcov': pcov,
        'param_errors': np.sqrt(np.diag(pcov)),
        'noise_sigma_rms': noise_sigma_rms,
        'noise_sigma_mad': noise_sigma_mad,
    }


@dataclass
class SpectrumFitResult:
    v: np.ndarray
    y: np.ndarray
    continuum_mask: np.ndarray
    cont: np.ndarray
    residual: np.ndarray
    model_line: np.ndarray
    fit_total: np.ndarray
    gaussian_components: np.ndarray
    n_components: int
    amplitudes: np.ndarray
    centers: np.ndarray
    sigmas: np.ndarray
    offset: float
    popt: np.ndarray
    pcov: np.ndarray
    param_errors: np.ndarray
    integrated_intensity_per_component: np.ndarray
    integrated_intensity_total: float
    integrated_intensity_total_error: float
    noise_sigma_rms: float
    noise_sigma_mad: float
    spectrum_selection: str
    row_index: Optional[int]
    n_spectra: int
    n_spectra_selected: Optional[int]
    mean_noise_rms: Optional[float] = None
    mean_noise_mad: Optional[float] = None
    file_path: str = ''
    peak_n_selected: Optional[int] = None
    peak_threshold: Optional[float] = None


def parse_row_selection(row_mode: str, row_index: Optional[int] = None):
    """Parse UI row mode into spectrum_fitting ``row`` argument."""
    mode = (row_mode or 'mean').strip().lower()
    if mode in ('mean', 'peak'):
        return mode
    if mode == 'row':
        if row_index is None:
            raise ValueError('row index required when spectrum selection is "row"')
        return int(row_index)
    raise ValueError(f'unknown row mode: {row_mode!r}')


def spectrum_fitting(
    file,
    row,
    low_line_limit,
    high_line_limit,
    hdu_index=1,
    spectrum_column='SPECTRUM',
    n_gaussians=1,
    resample_vel_res_kms=None,
    peak_threshold_fraction=0.5,
    peak_intensity_metric='integrated',
    amplitudes=None,
    centers=None,
    fwhms=None,
    lock_user=False,
) -> SpectrumFitResult:
    """Fit continuum + multi-Gaussian line model to an observational spectrum."""
    v, spectra, cube_data, spectral_axis, v_native, path = load_spectra_from_fits(
        file, hdu_index=hdu_index, spectrum_column=spectrum_column,
        resample_vel_res_kms=resample_vel_res_kms)

    agg = _aggregate_spectra_and_noise(
        v, spectra, row, low_line_limit, high_line_limit,
        cube_data=cube_data, spectral_axis_1based=spectral_axis,
        peak_threshold_fraction=peak_threshold_fraction,
        peak_intensity_metric=peak_intensity_metric,
        velocity_kms_native=v_native,
    )
    y = agg['y']
    peak_info = agg['peak_info']

    out = _fit_gaussian_line_core(
        v, y, low_line_limit, high_line_limit,
        n_gaussians=int(n_gaussians),
        amplitudes=amplitudes, centers=centers, fwhms=fwhms,
        lock_user=bool(lock_user))

    return _spectrum_fit_result_from_core(
        v, y, out,
        file_path=path,
        spectrum_selection=agg['spectrum_selection'],
        row_index=agg['row_index'],
        n_spectra=agg['n_spectra'],
        n_spectra_selected=agg['n_spectra_selected'],
        mean_noise_rms=agg['mean_noise_rms'],
        mean_noise_mad=agg['mean_noise_mad'],
        peak_n_selected=peak_info['n_selected'] if peak_info else None,
        peak_threshold=peak_info['intensity_threshold'] if peak_info else None,
    )


def _spectrum_fit_result_from_core(
    v, y, out, *,
    file_path='',
    spectrum_selection='',
    row_index=None,
    n_spectra=1,
    n_spectra_selected=1,
    mean_noise_rms=None,
    mean_noise_mad=None,
    peak_n_selected=None,
    peak_threshold=None,
) -> SpectrumFitResult:
    amps_fit = np.asarray(out['amplitudes'], dtype=float).ravel()
    sigs_fit = np.asarray(out['sigmas'], dtype=float).ravel()
    int_per = integrated_intensity_per_component_kms(amps_fit, sigs_fit)
    int_total = float(np.sum(int_per))
    int_total_err = integrated_intensity_total_error_kms(
        amps_fit, sigs_fit, out['pcov'], out['n_components'])
    return SpectrumFitResult(
        v=v, y=y,
        continuum_mask=out['continuum_mask'],
        cont=out['cont'],
        residual=out['residual'],
        model_line=out['model_line'],
        fit_total=out['fit_total'],
        gaussian_components=out['gaussian_components'],
        n_components=out['n_components'],
        amplitudes=out['amplitudes'],
        centers=out['centers'],
        sigmas=out['sigmas'],
        offset=float(out['offset']),
        popt=out['popt'],
        pcov=out['pcov'],
        param_errors=out['param_errors'],
        integrated_intensity_per_component=int_per,
        integrated_intensity_total=int_total,
        integrated_intensity_total_error=int_total_err,
        noise_sigma_rms=float(out['noise_sigma_rms']),
        noise_sigma_mad=float(out['noise_sigma_mad']),
        spectrum_selection=spectrum_selection,
        row_index=row_index,
        n_spectra=n_spectra,
        n_spectra_selected=n_spectra_selected,
        mean_noise_rms=mean_noise_rms,
        mean_noise_mad=mean_noise_mad,
        file_path=file_path,
        peak_n_selected=peak_n_selected,
        peak_threshold=peak_threshold,
    )


def fit_spectrum_arrays(velocity_kms, flux, low_line_limit, high_line_limit,
                        n_gaussians=1, file_path='', spectrum_selection='selection',
                        row_index=None, n_spectra=1, n_spectra_selected=1,
                        amplitudes=None, centers=None, fwhms=None,
                        lock_user=False,
                        ) -> SpectrumFitResult:
    """Fit continuum + Gaussians to an already extracted (v, y) spectrum."""
    v = np.asarray(velocity_kms, dtype=float)
    y = np.asarray(flux, dtype=float)
    out = _fit_gaussian_line_core(
        v, y, low_line_limit, high_line_limit, n_gaussians=int(n_gaussians),
        amplitudes=amplitudes, centers=centers, fwhms=fwhms,
        lock_user=bool(lock_user))
    return _spectrum_fit_result_from_core(
        v, y, out,
        file_path=file_path,
        spectrum_selection=spectrum_selection,
        row_index=row_index,
        n_spectra=n_spectra,
        n_spectra_selected=n_spectra_selected,
    )


def extract_spectrum_only(file, row, low_line_limit, high_line_limit,
                          hdu_index=1, spectrum_column='SPECTRUM',
                          peak_threshold_fraction=0.5,
                          peak_intensity_metric='integrated'):
    """Load and aggregate spectrum without fitting (for overlay-only)."""
    v, spectra, cube_data, spectral_axis, v_native, path = load_spectra_from_fits(
        file, hdu_index=hdu_index, spectrum_column=spectrum_column)
    agg = _aggregate_spectra_and_noise(
        v, spectra, row, low_line_limit, high_line_limit,
        cube_data=cube_data, spectral_axis_1based=spectral_axis,
        peak_threshold_fraction=peak_threshold_fraction,
        peak_intensity_metric=peak_intensity_metric,
        velocity_kms_native=v_native,
    )
    return v, agg['y'], agg['spectrum_selection'], path


# ---------------------------------------------------------------------------
# Formatting for the Dash UI
# ---------------------------------------------------------------------------

def _fmt_val_err(val, err, decimals=3):
    if not np.isfinite(val):
        return '—'
    if not np.isfinite(err) or err == 0:
        return f'{val:.{decimals}g}'
    return f'{val:.{decimals}g} \u00b1 {err:.{decimals}g}'


def format_fit_summary_html(result: SpectrumFitResult) -> str:
    """HTML summary table for a :class:`SpectrumFitResult`."""
    lines = [
        '<div style="font-size:13px;line-height:1.55">',
        f'<b>Observational fit</b> — <code>{os.path.basename(result.file_path)}</code>',
        f'<br>Selection: <b>{result.spectrum_selection}</b>',
    ]
    if result.n_spectra_selected is not None and result.spectrum_selection in ('mean', 'peak'):
        extra = f' ({result.n_spectra_selected} / {result.n_spectra} spectra)'
        if result.spectrum_selection == 'peak' and result.peak_n_selected is not None:
            extra = f' ({result.peak_n_selected} / {result.n_spectra} pixels'
            if result.peak_threshold is not None:
                extra += f', threshold = {result.peak_threshold:.4g}'
            extra += ')'
        lines.append(extra)
    lines.append(
        f'<br>Noise (line-free): RMS = {result.noise_sigma_rms:.4g} K, '
        f'MAD = {result.noise_sigma_mad:.4g} K')
    if result.mean_noise_rms is not None:
        lines.append(
            f'<br>Mean per-spectrum noise: RMS = {result.mean_noise_rms:.4g} K, '
            f'MAD = {result.mean_noise_mad:.4g} K')

    lines.append('<table style="margin-top:8px;border-collapse:collapse">')
    lines.append(
        '<tr style="background:#eef2f7">'
        '<th style="padding:4px 10px;text-align:left">Component</th>'
        '<th style="padding:4px 10px">A (K)</th>'
        '<th style="padding:4px 10px">v\u2080 (km/s)</th>'
        '<th style="padding:4px 10px">\u03c3 (km/s)</th>'
        '<th style="padding:4px 10px">\u222bI dv (K km/s)</th></tr>')

    errs = result.param_errors
    for j in range(result.n_components):
        k = 3 * j
        ipc = float(result.integrated_intensity_per_component[j])
        lines.append(
            f'<tr><td style="padding:4px 10px">Gaussian {j + 1}</td>'
            f'<td style="padding:4px 10px;text-align:center">'
            f'{_fmt_val_err(result.amplitudes[j], errs[k])}</td>'
            f'<td style="padding:4px 10px;text-align:center">'
            f'{_fmt_val_err(result.centers[j], errs[k + 1])}</td>'
            f'<td style="padding:4px 10px;text-align:center">'
            f'{_fmt_val_err(result.sigmas[j], errs[k + 2])}</td>'
            f'<td style="padding:4px 10px;text-align:center">{ipc:.4g}</td></tr>')

    off_err = errs[-1] if errs.size else float('nan')
    lines.append(
        f'<tr><td style="padding:4px 10px">Offset (post-cont.)</td>'
        f'<td colspan="4" style="padding:4px 10px;text-align:center">'
        f'{_fmt_val_err(result.offset, off_err)} K</td></tr>')
    tot_str = _fmt_val_err(
        result.integrated_intensity_total, result.integrated_intensity_total_error)
    lines.append(
        f'<tr style="background:#f0f8f0;font-weight:600">'
        f'<td style="padding:4px 10px">Total \u222bI dv</td>'
        f'<td colspan="4" style="padding:4px 10px;text-align:center">'
        f'{tot_str} K km/s</td></tr>')
    lines.append('</table></div>')
    return ''.join(lines)
