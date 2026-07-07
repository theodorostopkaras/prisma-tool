"""
KoSens3D-compatible grid interpolation (``resampled_grid_data`` + axis log spacing).

Matches ``kosens3d.main_functions.resampled_grid_data`` for 2-D value resampling
(index-normalised coordinates, log10 values) and ``grid_functions`` /
``create_final_grids_3d`` for log-spaced physical axis coordinates.
"""

from __future__ import annotations

from typing import Optional, Tuple

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
