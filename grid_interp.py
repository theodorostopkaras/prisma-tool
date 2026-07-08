"""
KoSens3D-compatible grid interpolation (``resampled_grid_data`` + axis log spacing).

Matches ``kosens3d.main_functions.resampled_grid_data`` for 2-D value resampling
(index-normalised coordinates, log10 values) and ``grid_functions`` /
``create_final_grids_3d`` for log-spaced physical axis coordinates.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, griddata


def interpolate_axis_logspace(axis_log10: np.ndarray, n_points: int) -> np.ndarray:
    """Interpolate axis in log space (``axis_log10`` are log10 physical values)."""
    axis_log10 = np.asarray(axis_log10, dtype=float)
    n_points = max(int(n_points), 1)
    if axis_log10.size < 1:
        return np.array([], dtype=float)
    if axis_log10.size < 2:
        return np.full(n_points, 10.0 ** axis_log10[0], dtype=float)
    indices_new = np.linspace(0, axis_log10.size - 1, n_points)
    log_interp = np.interp(indices_new, np.arange(axis_log10.size), axis_log10)
    return 10.0 ** log_interp


def _norm_log_coords(log_axis: np.ndarray) -> np.ndarray:
    """Map log10 axis values to [0, 1] for interpolation coordinates."""
    log_axis = np.asarray(log_axis, dtype=float)
    if log_axis.size == 0:
        return log_axis
    lo, hi = float(np.min(log_axis)), float(np.max(log_axis))
    span = hi - lo
    if span <= 0:
        return np.zeros_like(log_axis)
    return (log_axis - lo) / span


def impute_along_axis_kosens(
    grid: np.ndarray,
    axis: int = 1,
    *,
    log_values: bool = True,
) -> np.ndarray:
    """
    Linearly impute NaN / zero values along one axis (KoSens ``resampled_grid_data``).

    For a 2-D slice with shape ``(n_y, n_x)``, ``axis=1`` imputes along x within
    each y row (grouped by the y-axis parameter); ``axis=0`` imputes along y within
    each x column.  Interpolation is performed in log10(value) space when
    ``log_values`` is True.
    """
    grid = np.asarray(grid, dtype=float).copy()
    if grid.size == 0:
        return grid

    if axis == 1:
        for i in range(grid.shape[0]):
            row = grid[i, :]
            bad = ~np.isfinite(row) | ((row <= 0) if log_values else False)
            good = np.isfinite(row) & ((row > 0) if log_values else True)
            if not np.any(good) or not np.any(bad):
                continue
            x_good = np.flatnonzero(good).astype(np.float64)
            x_all = np.arange(row.size, dtype=np.float64)
            if log_values:
                filled = np.interp(x_all, x_good, np.log10(row[good]))
                row[bad] = 10.0 ** filled[bad]
            else:
                row[bad] = np.interp(x_all, x_good, row[good])[bad]
            grid[i, :] = row
    elif axis == 0:
        for j in range(grid.shape[1]):
            col = grid[:, j]
            bad = ~np.isfinite(col) | ((col <= 0) if log_values else False)
            good = np.isfinite(col) & ((col > 0) if log_values else True)
            if not np.any(good) or not np.any(bad):
                continue
            y_good = np.flatnonzero(good).astype(np.float64)
            y_all = np.arange(col.size, dtype=np.float64)
            if log_values:
                filled = np.interp(y_all, y_good, np.log10(col[good]))
                col[bad] = 10.0 ** filled[bad]
            else:
                col[bad] = np.interp(y_all, y_good, col[good])[bad]
            grid[:, j] = col
    return grid


def impute_grid_kosens(grid: np.ndarray, *, log_values: bool = True) -> np.ndarray:
    """Impute missing/zero cells along x then y (matches KoSens row-wise flux fill)."""
    grid = impute_along_axis_kosens(grid, axis=1, log_values=log_values)
    grid = impute_along_axis_kosens(grid, axis=0, log_values=log_values)
    return grid


def align_grid_axes_ascending(
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    grid: np.ndarray,
    *,
    x_logscale: bool = True,
    y_logscale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure x and y run in ascending log10(physical) order (matches KoSens plots)."""
    x_phys = np.asarray(x_phys, dtype=float).copy()
    y_phys = np.asarray(y_phys, dtype=float).copy()
    grid = np.asarray(grid, dtype=float).copy()
    if x_phys.size > 1:
        x_key = _axis_log10(x_phys, x_logscale)
        if x_key[0] > x_key[-1]:
            x_phys = x_phys[::-1]
            grid = grid[:, ::-1]
    if y_phys.size > 1:
        y_key = _axis_log10(y_phys, y_logscale)
        if y_key[0] > y_key[-1]:
            y_phys = y_phys[::-1]
            grid = grid[::-1, :]
    return x_phys, y_phys, grid


def _griddata_resample_2d(
    values_2d: np.ndarray,
    x_orig_norm: np.ndarray,
    y_orig_norm: np.ndarray,
    ty: int,
    tx: int,
    interpolation_method: str,
    smoothing_order: int,
) -> np.ndarray:
    """Interpolate ``values_2d`` on normalised (x, y) coords; ignore invalid samples."""
    ny, nx = values_2d.shape
    yy, xx = np.meshgrid(y_orig_norm, x_orig_norm, indexing='ij')
    points = np.column_stack((xx.ravel(), yy.ravel()))
    flat = values_2d.ravel()
    valid = np.isfinite(flat)
    if not np.any(valid):
        return np.full((ty, tx), np.nan, dtype=np.float64)

    y_t, x_t = np.mgrid[0:ty, 0:tx]
    x_target_norm = x_t / max(tx - 1, 1)
    y_target_norm = y_t / max(ty - 1, 1)
    xi = np.column_stack((x_target_norm.ravel(), y_target_norm.ravel()))

    method = interpolation_method or 'linear'
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 4 and method in ('cubic', 'spline'):
        method = 'linear'
    if n_valid < 3 and method == 'linear':
        method = 'nearest'

    if method == 'spline' and ny >= 2 and nx >= 2 and n_valid == flat.size:
        try:
            spline = RectBivariateSpline(
                y_orig_norm, x_orig_norm, values_2d,
                kx=min(smoothing_order, nx - 1),
                ky=min(smoothing_order, ny - 1),
            )
            y_coords = np.linspace(float(y_orig_norm[0]), float(y_orig_norm[-1]), ty)
            x_coords = np.linspace(float(x_orig_norm[0]), float(x_orig_norm[-1]), tx)
            return spline(y_coords, x_coords).astype(np.float64)
        except Exception:
            method = 'linear'

    resampled = griddata(
        points[valid], flat[valid], xi, method=method, fill_value=np.nan,
    ).reshape(ty, tx).astype(np.float64)
    if np.any(np.isnan(resampled)):
        resampled = griddata(
            points[valid], flat[valid], xi, method='nearest',
        ).reshape(ty, tx).astype(np.float64)
    return resampled


def resample_grid_values_kosens(
    grid: np.ndarray,
    target_shape: Tuple[int, int],
    interpolation_method: str = 'linear',
    smoothing_order: int = 3,
    clip_to_bounds: bool = False,
    log_values: bool = True,
    x_coords_norm: Optional[np.ndarray] = None,
    y_coords_norm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Resample a 2-D grid to ``target_shape`` using KoSens ``resampled_grid_data``.

    Values are interpolated in log10 space (when ``log_values``).  Coordinates are
    normalised to [0, 1]; pass ``x_coords_norm`` / ``y_coords_norm`` from log10 axis
    values for log-parameter-space interpolation (recommended for PDR grids).
    """
    grid = np.asarray(grid, dtype=float)
    ty, tx = int(target_shape[0]), int(target_shape[1])
    ny, nx = grid.shape

    if ty == ny and tx == nx:
        return grid.copy()

    if log_values:
        min_positive = (
            float(np.nanmin(grid[grid > 0]))
            if np.any(grid > 0)
            else 1e-30
        )
        values_2d = np.log10(np.maximum(grid, min_positive))
        original_linear = grid
    else:
        values_2d = grid.copy()
        original_linear = grid

    if x_coords_norm is not None and y_coords_norm is not None:
        x_orig_norm = np.asarray(x_coords_norm, dtype=float)
        y_orig_norm = np.asarray(y_coords_norm, dtype=float)
    else:
        x_orig_norm = np.linspace(0.0, 1.0, nx) if nx > 1 else np.array([0.0])
        y_orig_norm = np.linspace(0.0, 1.0, ny) if ny > 1 else np.array([0.0])

    if ny < 2 and nx < 2:
        fill = float(values_2d[0, 0])
        resampled = np.full((ty, tx), fill, dtype=np.float64)
    elif ny < 2:
        x_new = np.linspace(0.0, 1.0, tx)
        row_interp = np.interp(
            x_new, x_orig_norm, values_2d[0, :],
        ).astype(np.float64)
        resampled = np.tile(row_interp, (ty, 1))
    elif nx < 2:
        y_new = np.linspace(0.0, 1.0, ty)
        col_interp = np.interp(
            y_new, y_orig_norm, values_2d[:, 0],
        ).astype(np.float64)
        resampled = np.tile(col_interp.reshape(-1, 1), (1, tx))
    else:
        resampled = _griddata_resample_2d(
            values_2d, x_orig_norm, y_orig_norm, ty, tx,
            interpolation_method, smoothing_order,
        )

    if log_values:
        resampled = 10.0 ** resampled
        if clip_to_bounds and np.any(np.isfinite(original_linear)):
            resampled = np.clip(resampled, 0.0, float(np.nanmax(original_linear)))
    elif clip_to_bounds and np.any(np.isfinite(original_linear)):
        zmin, zmax = float(np.nanmin(original_linear)), float(np.nanmax(original_linear))
        resampled = np.clip(resampled, zmin, zmax)

    return resampled


def _axis_log10(phys: np.ndarray, logscale: bool) -> np.ndarray:
    phys = np.asarray(phys, dtype=float)
    if logscale:
        return np.log10(np.maximum(phys, np.finfo(float).tiny))
    return phys


def _filter_axis_by_limit(phys, logscale, limit):
    phys = np.asarray(phys, dtype=float)
    idx = np.arange(phys.size)
    if limit is None or not np.isfinite(limit):
        return phys, idx
    if logscale:
        mask = _axis_log10(phys, True) < float(limit)
    else:
        mask = phys < float(limit)
    keep = idx[mask]
    if keep.size == 0:
        return phys, idx
    return phys[keep], keep


def resample_grid_2d_kosens(
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    grid: np.ndarray,
    *,
    x_logscale: bool = True,
    y_logscale: bool = True,
    target_shape: Tuple[int, int] = (60, 60),
    x_lim: Optional[float] = None,
    y_lim: Optional[float] = None,
    interpolation_method: str = 'linear',
    smoothing_order: int = 3,
    clip_to_bounds: bool = False,
    log_values: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full KoSens-style 2-D resample: axis limits, log-spaced physical mesh, log10 values.

    Returns ``(final_x, final_y, resampled_grid)``.
    """
    grid = np.asarray(grid, dtype=float)
    x_phys = np.asarray(x_phys, dtype=float)
    y_phys = np.asarray(y_phys, dtype=float)

    x_phys, x_idx = _filter_axis_by_limit(x_phys, x_logscale, x_lim)
    y_phys, y_idx = _filter_axis_by_limit(y_phys, y_logscale, y_lim)
    grid = grid[np.ix_(y_idx, x_idx)]

    if x_phys.size == 0 or y_phys.size == 0:
        return x_phys, y_phys, grid

    x_order = np.argsort(_axis_log10(x_phys, x_logscale))
    y_order = np.argsort(_axis_log10(y_phys, y_logscale))
    x_phys = x_phys[x_order]
    y_phys = y_phys[y_order]
    grid = grid[np.ix_(y_order, x_order)]

    grid = impute_grid_kosens(grid, log_values=log_values)

    ty, tx = int(target_shape[0]), int(target_shape[1])
    x_log = _axis_log10(x_phys, x_logscale)
    y_log = _axis_log10(y_phys, y_logscale)
    final_x = interpolate_axis_logspace(x_log, tx)
    final_y = interpolate_axis_logspace(y_log, ty)

    x_norm = _norm_log_coords(x_log)
    y_norm = _norm_log_coords(y_log)

    grid_out = resample_grid_values_kosens(
        grid, (ty, tx),
        interpolation_method=interpolation_method,
        smoothing_order=smoothing_order,
        clip_to_bounds=clip_to_bounds,
        log_values=log_values,
        x_coords_norm=x_norm,
        y_coords_norm=y_norm,
    )
    return final_x, final_y, grid_out


def resample_grid_3d_kosens(
    grid: np.ndarray,
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    z_phys: np.ndarray,
    *,
    target_shape: Tuple[int, int, int] = (60, 60, 60),
    x_logscale: bool = True,
    y_logscale: bool = True,
    z_logscale: bool = True,
    interpolation_method: str = 'linear',
    smoothing_order: int = 3,
    clip_to_bounds: bool = False,
    log_values: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    KoSens3D ``process_grids_3d`` + ``create_final_grids_3d`` style 3-D resampling.

    1. Each z slice is resampled in 2-D with ``resample_grid_values_kosens``.
    2. The stacked axis is interpolated in log10(physical z) with normalised y/x
       indices (``RegularGridInterpolator``), matching ``grid_3d.py``.
    """
    grid = np.asarray(grid, dtype=float)
    nz, ny, nx = grid.shape
    tz, ty, tx = (int(target_shape[0]), int(target_shape[1]), int(target_shape[2]))

    if ty != ny or tx != nx:
        x_log = _axis_log10(x_phys, x_logscale)
        y_log = _axis_log10(y_phys, y_logscale)
        x_norm = _norm_log_coords(x_log)
        y_norm = _norm_log_coords(y_log)
        slices = []
        for iz in range(nz):
            sl = impute_grid_kosens(grid[iz], log_values=log_values)
            slices.append(resample_grid_values_kosens(
                sl, (ty, tx),
                interpolation_method=interpolation_method,
                smoothing_order=smoothing_order,
                clip_to_bounds=clip_to_bounds,
                log_values=log_values,
                x_coords_norm=x_norm,
                y_coords_norm=y_norm,
            ))
        grid = np.stack(slices, axis=0)
        ny, nx = ty, tx

    x_log = _axis_log10(x_phys, x_logscale)
    y_log = _axis_log10(y_phys, y_logscale)
    z_log = _axis_log10(z_phys, z_logscale)
    final_x = interpolate_axis_logspace(x_log, tx)
    final_y = interpolate_axis_logspace(y_log, ty)
    final_z = interpolate_axis_logspace(z_log, tz)

    if tz == grid.shape[0]:
        return grid, final_x, final_y, final_z

    if grid.shape[0] < 2:
        fill = float(np.nanmean(grid)) if np.any(np.isfinite(grid)) else np.nan
        grid_out = np.full((tz, ny, nx), fill, dtype=float)
        return grid_out, final_x, final_y, final_z

    log_z_orig = np.log10(np.maximum(np.asarray(z_phys, float), np.finfo(float).tiny))
    log_z_new = np.log10(final_z)
    y_coords = np.arange(ny) / max(ny - 1, 1)
    x_coords = np.arange(nx) / max(nx - 1, 1)
    y_target = np.linspace(0.0, 1.0, ny) if ny > 1 else np.array([0.0])
    x_target = np.linspace(0.0, 1.0, nx) if nx > 1 else np.array([0.0])

    rgi = RegularGridInterpolator(
        (log_z_orig, y_coords, x_coords),
        grid,
        method='linear',
        bounds_error=False,
        fill_value=np.nan,
    )
    Z_new, Y_new, X_new = np.meshgrid(log_z_new, y_target, x_target, indexing='ij')
    points = np.stack([Z_new.ravel(), Y_new.ravel(), X_new.ravel()], axis=-1)
    grid_out = rgi(points).reshape(tz, ny, nx)
    return grid_out, final_x, final_y, final_z


def calculate_interpolation_error_kosens(
    grid_log10: np.ndarray,
    *,
    interpolation_method: str = 'linear',
    smoothing_order: int = 3,
    clip_to_bounds: bool = True,
    decimation_factor: int = 2,
    error_metric: str = 'relative',
    relative_threshold: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Decimate a native 2-D grid, re-interpolate, and estimate error (KoSens
    ``_calculate_interpolation_error_internal``).

    ``grid_log10`` holds log10(flux) values.  Returns error grid and statistics
    in linear flux units; ``error_grid`` matches the input shape.
    """
    original_grid = np.asarray(grid_log10, dtype=float)
    original_shape = original_grid.shape
    dec = max(int(decimation_factor), 2)

    boundary_mask = np.zeros(original_shape, dtype=bool)
    dec_y_indices = list(range(0, original_shape[0], dec))
    if dec_y_indices[-1] != original_shape[0] - 1:
        dec_y_indices.append(original_shape[0] - 1)
    dec_x_indices = list(range(0, original_shape[1], dec))
    if dec_x_indices[-1] != original_shape[1] - 1:
        dec_x_indices.append(original_shape[1] - 1)

    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            is_kept_row = i in dec_y_indices
            is_kept_col = j in dec_x_indices
            is_boundary = (
                i == 0 or i == original_shape[0] - 1
                or j == 0 or j == original_shape[1] - 1
            )
            if (is_kept_row and is_kept_col) or is_boundary:
                boundary_mask[i, j] = True

    dec_y_all, dec_x_all = np.where(boundary_mask)
    decimated_values = original_grid[dec_y_all, dec_x_all].astype(np.float64)
    finite_mask = np.isfinite(decimated_values)
    if not np.all(finite_mask):
        finite_vals = decimated_values[finite_mask]
        fill_val = float(np.nanmin(finite_vals) - 10) if finite_vals.size else -30.0
        decimated_for_interp = np.where(finite_mask, decimated_values, fill_val)
    else:
        decimated_for_interp = decimated_values

    points = np.column_stack((dec_x_all, dec_y_all))
    y_tgt, x_tgt = np.meshgrid(
        np.arange(original_shape[0]), np.arange(original_shape[1]), indexing='ij',
    )
    xi = np.column_stack((x_tgt.ravel(), y_tgt.ravel()))
    interp_method = 'cubic' if interpolation_method == 'spline' else interpolation_method
    interpolated_grid = griddata(
        points, decimated_for_interp, xi, method=interp_method, fill_value=np.nan,
    ).reshape(original_shape).astype(np.float64)
    if np.any(np.isnan(interpolated_grid)):
        nearest = griddata(points, decimated_for_interp, xi, method='nearest')
        interpolated_grid = np.where(
            np.isnan(interpolated_grid), nearest.reshape(original_shape), interpolated_grid,
        )

    if clip_to_bounds:
        finite_orig = original_grid[np.isfinite(original_grid)]
        min_val = float(np.nanmin(finite_orig)) if finite_orig.size else -30.0
        max_val = float(np.nanmax(finite_orig)) if finite_orig.size else 0.0
        interpolated_grid = np.clip(interpolated_grid, min_val, max_val)

    interpolated_grid[boundary_mask] = original_grid[boundary_mask]

    with np.errstate(divide='ignore', invalid='ignore'):
        orig_linear = np.where(
            np.isfinite(original_grid),
            10.0 ** np.clip(original_grid, -50.0, 100.0),
            np.nan,
        )
        interp_linear = np.where(
            np.isfinite(interpolated_grid),
            10.0 ** np.clip(interpolated_grid, -50.0, 100.0),
            np.nan,
        )

    with np.errstate(divide='ignore', invalid='ignore'):
        if error_metric == 'relative':
            max_linear = np.nanmax(orig_linear[np.isfinite(orig_linear)])
            thresh_linear = (
                max_linear * max(relative_threshold, 1e-30)
                if np.isfinite(max_linear) else 0.0
            )
            denom_linear = np.where(
                np.abs(orig_linear) > 1e-100, np.abs(orig_linear), np.nan,
            )
            error_grid = np.abs(interp_linear - orig_linear) / denom_linear * 100.0
            error_grid = np.where(orig_linear < thresh_linear, 0.0, error_grid)
            error_grid = np.where(~np.isfinite(denom_linear), 0.0, error_grid)
        elif error_metric == 'absolute':
            error_grid = np.abs(interp_linear - orig_linear)
        else:
            raise ValueError(
                f"Unknown error_metric: {error_metric!r}. Use 'relative' or 'absolute'.",
            )
        error_grid = np.where(~np.isfinite(error_grid), 0.0, error_grid)

    valid_errors = error_grid[np.isfinite(error_grid)]
    n = valid_errors.size
    error_statistics = {
        'mean_error': float(np.mean(valid_errors)) if n else np.nan,
        'max_error': float(np.max(valid_errors)) if n else np.nan,
        'min_error': float(np.min(valid_errors)) if n else np.nan,
        'std_error': float(np.std(valid_errors)) if n else np.nan,
        'median_error': float(np.median(valid_errors)) if n else np.nan,
        'decimation_factor': dec,
        'interpolation_method': interpolation_method,
        'error_metric': error_metric,
    }
    return error_grid, error_statistics, interp_linear, orig_linear


def analyze_slice_interpolation(
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    grid: np.ndarray,
    *,
    x_logscale: bool = True,
    y_logscale: bool = True,
    target_shape: Tuple[int, int] = (60, 60),
    x_lim: Optional[float] = None,
    y_lim: Optional[float] = None,
    interpolation_method: str = 'linear',
    smoothing_order: int = 3,
    clip_to_bounds: bool = False,
    log_values: bool = True,
    decimation_factor: int = 2,
    error_metric: str = 'relative',
    relative_threshold: float = 0.01,
) -> Dict[str, object]:
    """
    Native-grid error analysis plus KoSens-style fine resampling for one slice.

    Returns dict with native/fine coordinates, original & resampled linear grids,
    error grid on the native mesh, and error statistics.
    """
    grid = np.asarray(grid, dtype=float)
    x_phys = np.asarray(x_phys, dtype=float)
    y_phys = np.asarray(y_phys, dtype=float)

    x_phys, x_idx = _filter_axis_by_limit(x_phys, x_logscale, x_lim)
    y_phys, y_idx = _filter_axis_by_limit(y_phys, y_logscale, y_lim)
    grid = grid[np.ix_(y_idx, x_idx)]

    x_phys, y_phys, grid = align_grid_axes_ascending(
        x_phys, y_phys, grid, x_logscale=x_logscale, y_logscale=y_logscale,
    )
    grid = impute_grid_kosens(grid, log_values=log_values)

    if log_values:
        min_positive = (
            float(np.nanmin(grid[grid > 0])) if np.any(grid > 0) else 1e-30
        )
        grid_log10 = np.log10(np.maximum(grid, min_positive))
    else:
        with np.errstate(divide='ignore'):
            grid_log10 = np.log10(np.maximum(grid, np.finfo(float).tiny))

    error_grid, error_stats, _, orig_linear = calculate_interpolation_error_kosens(
        grid_log10,
        interpolation_method=interpolation_method,
        smoothing_order=smoothing_order,
        clip_to_bounds=clip_to_bounds,
        decimation_factor=decimation_factor,
        error_metric=error_metric,
        relative_threshold=relative_threshold,
    )

    final_x, final_y, resampled = resample_grid_2d_kosens(
        x_phys, y_phys, grid,
        x_logscale=x_logscale,
        y_logscale=y_logscale,
        target_shape=target_shape,
        interpolation_method=interpolation_method,
        smoothing_order=smoothing_order,
        clip_to_bounds=clip_to_bounds,
        log_values=log_values,
    )

    return dict(
        x_native=x_phys,
        y_native=y_phys,
        original_linear=orig_linear,
        x_fine=final_x,
        y_fine=final_y,
        resampled_linear=resampled,
        error_grid=error_grid,
        error_stats=error_stats,
    )
