"""
Reference vs cosmic-ray-attenuated grid comparison.

Port of the KoSens attenuation workflow (``kosens.grid.grid_functions``):
``collect_attenuation_x_shift_table`` → ``plot_attenuation_shift_summary`` →
``plot_attenuation_fuv_response`` → ``reconcile_attenuation_species``.
The numerics are kept identical to KoSens (``tests/test_atten_compare.py``
checks parity); only the figures differ (Plotly instead of matplotlib).

Two questions, deliberately kept apart:

* *Which lines mislead me most if I ignore attenuation?*  → ``problematic``
* *Which lines are the cleanest cosmic-ray probes?*       → ``tracers``

Three per-cell quantities carry everything:

* Δ = log10(ζ_match / ζ_nom)  — horizontal shift = dex error in the inferred ζ
* S = dlog10(I_ref)/dlog10(ζ) — local CR sensitivity of the reference grid
* R = log10(I_atten / I_ref)  — intensity lost at fixed ζ;   |Δ| ≈ |R| / |S|

3-D extension: every ζ-row sits at a fixed (n_H, χ).  Pixels carry the density,
so the same reducers run on one density slice (exactly KoSens), on the pooled
3-D grid, per density, or per (n_H, χ, ζ) environment.  How to read every
number: ``docs/attenuation_workflow.md``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import linregress

import grid_naming as gn

X_SHIFT_MATCH_RTOL = 0.02
OBS_INTENSITY_LIMIT = 0.1
# Rare-isotope markers as they appear in KOSMA-τ species names (13CO, C18O, H13CN...).
ISOTOPE_MARKERS = ("13C", "18O", "17O", "15N", "34S", "33S")

VERDICT_ORDER = [
    "CRIR probe", "CR-led, small bias", "CR-led, narrow", "FUV tracer",
    "unmeasurable", "no response", "unobservable", "too few models", "no data",
]
CR_FAMILY = {"CRIR probe", "CR-led, small bias", "CR-led, narrow"}


# ---------------------------------------------------------------------------
# Horizontal intensity match (KoSens, row preparation hoisted)
# ---------------------------------------------------------------------------

def _interpolate_segment_match(x0, x1, z0, z1, i_target, tol):
    """Smallest x in [x0, x1] where intensity matches ``i_target`` within ``tol``.

    Intensity is taken as linear in log10(x) between samples.
    """
    if not np.all(np.isfinite([x0, x1, z0, z1, i_target])) or x0 <= 0 or x1 <= 0:
        return np.nan
    u0, u1 = np.log10(float(x0)), np.log10(float(x1))
    du = u1 - u0
    if abs(du) < 1e-15 * (abs(u0) + 1.0):
        return float(min(x0, x1)) if abs(z0 - i_target) <= tol else np.nan
    dz = z1 - z0
    if dz == 0:
        return float(min(x0, x1)) if abs(z0 - i_target) <= tol else np.nan
    s1 = (i_target - tol - z0) / dz
    s2 = (i_target + tol - z0) / dz
    s_lo = max(0.0, min(s1, s2))
    s_hi = min(1.0, max(s1, s2))
    if s_lo > s_hi + 1e-15:
        return np.nan
    xb = float(10.0 ** (u0 + s_lo * du))
    return xb if xb > 0 and np.isfinite(xb) else np.nan


def _x_match_scan_sorted(xs_scan, zs_scan, i_target, tol):
    """First match on grid nodes, then on segments, in ``xs_scan`` order."""
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


def _normalize_x_shift_scan_direction(scan_direction):
    """Canonical label: 'rightward', 'leftward' or 'rightward_then_leftward'."""
    if scan_direction is None:
        return "rightward_then_leftward"
    key = str(scan_direction).strip().lower()
    key = {"rtl": "rightward_then_leftward", "right": "rightward",
           "left": "leftward"}.get(key, key)
    allowed = ("rightward", "leftward", "rightward_then_leftward")
    if key not in allowed:
        raise ValueError(f"scan_direction must be one of {allowed}, got {scan_direction!r}.")
    return key


def _prepare_matched_row(x_row, z_slice):
    """Sort/filter one row once: ``(xs, zs, x_min_row, x_max_row)``."""
    order = np.argsort(x_row.astype(float))
    xs = x_row[order].astype(float)
    zs = z_slice[order].astype(float)
    fin = np.isfinite(xs) & np.isfinite(zs) & (xs > 0)
    xs, zs = xs[fin], zs[fin]
    if xs.size == 0:
        return xs, zs, np.nan, np.nan
    return xs, zs, float(np.min(xs)), float(np.max(xs))


def _x_match_intensity_to_x(xs, zs, x_min_row, x_max_row, i_target, x_prefer, *,
                            atol=0.0, rtol=0.0, scan_direction="rightward_then_leftward",
                            skip_if_nominal_at_row_max=False,
                            skip_if_nominal_at_row_min=False, x_edge_rtol=1e-12):
    """``x_match > 0`` where the prepared row matches ``i_target``, or NaN."""
    if not np.isfinite(i_target) or not np.isfinite(x_prefer) or x_prefer <= 0:
        return np.nan
    if xs.size == 0:
        return np.nan
    scan_direction = _normalize_x_shift_scan_direction(scan_direction)
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

    if scan_direction == "rightward":
        x_hit = _scan_half_row(right_half=True)
        if np.isfinite(x_hit) and x_hit + edge_atol < x_lo_bound:
            return np.nan
        return x_hit
    if scan_direction == "leftward":
        x_hit = _scan_half_row(right_half=False)
        if np.isfinite(x_hit) and x_hit - edge_atol > x_hi_bound:
            return np.nan
        return x_hit

    x_hit = _scan_half_row(right_half=True)
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


def horizontal_intensity_matched_x_shift_dex(g_const, g_atten, x_mesh_phys, *,
                                             match_atol=0.0, match_rtol=X_SHIFT_MATCH_RTOL,
                                             scan_direction="rightward_then_leftward",
                                             skip_x_nom_at_row_max=False,
                                             skip_x_nom_at_row_min=False,
                                             x_edge_rtol=1e-12):
    """Per-cell ``log10(x_match / x_nom)`` [dex]; NaN where no match.

    At each ``(x_nom, y)`` the constant-grid value is the target; the attenuated
    row at the same ``y`` is scanned for the ``x`` that reproduces it within
    ``match_atol + match_rtol * |target|``.
    """
    scan_direction = _normalize_x_shift_scan_direction(scan_direction)
    g_const = np.asarray(g_const, dtype=float)
    g_atten = np.asarray(g_atten, dtype=float)
    x_mesh_phys = np.asarray(x_mesh_phys, dtype=float)
    if g_const.shape != g_atten.shape or g_const.shape != x_mesh_phys.shape:
        raise ValueError(
            f"Shape mismatch g_const {g_const.shape}, g_atten {g_atten.shape}, "
            f"x_mesh_phys {x_mesh_phys.shape}.")
    ny, nx = g_const.shape
    out = np.full((ny, nx), np.nan, dtype=float)
    for iy in range(ny):
        x_row = x_mesh_phys[iy, :]
        z_const = g_const[iy, :]
        xs_row, zs_row, x_min_row, x_max_row = _prepare_matched_row(x_row, g_atten[iy, :])
        if xs_row.size == 0:
            continue
        for ix in range(nx):
            x_nom, i_targ = x_row[ix], z_const[ix]
            if not (np.isfinite(x_nom) and x_nom > 0 and np.isfinite(i_targ)):
                continue
            x_match = _x_match_intensity_to_x(
                xs_row, zs_row, x_min_row, x_max_row, i_targ, x_nom,
                atol=match_atol, rtol=match_rtol, scan_direction=scan_direction,
                skip_if_nominal_at_row_max=skip_x_nom_at_row_max,
                skip_if_nominal_at_row_min=skip_x_nom_at_row_min,
                x_edge_rtol=x_edge_rtol)
            if not (np.isfinite(x_match) and x_match > 0):
                continue
            dex = np.log10(x_match / x_nom)
            if scan_direction == "rightward" and dex < -max(x_edge_rtol, 1e-12):
                continue
            if scan_direction == "leftward" and dex > max(x_edge_rtol, 1e-12):
                continue
            out[iy, ix] = dex
    return out


# ---------------------------------------------------------------------------
# Small helpers (KoSens verbatim)
# ---------------------------------------------------------------------------

def _log_intensity_slope(log_intensity, log_x):
    """``dlog10(I)/dlog10(x)`` along each row, central differences, one-sided edges."""
    li = np.asarray(log_intensity, dtype=float)
    lx = np.asarray(log_x, dtype=float)
    if li.ndim != 2 or li.shape != lx.shape or li.shape[1] < 2:
        return np.full(li.shape, np.nan)
    num = np.full(li.shape, np.nan)
    den = np.full(li.shape, np.nan)
    num[:, 1:-1] = li[:, 2:] - li[:, :-2]
    den[:, 1:-1] = lx[:, 2:] - lx[:, :-2]
    num[:, 0] = li[:, 1] - li[:, 0]
    den[:, 0] = lx[:, 1] - lx[:, 0]
    num[:, -1] = li[:, -1] - li[:, -2]
    den[:, -1] = lx[:, -1] - lx[:, -2]
    good = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 0.0)
    return np.divide(num, den, out=np.full(li.shape, np.nan), where=good)


def _grid_entry_xy_keys(entry):
    """``(y_key, x_key)``: the two non-``grid`` keys, read positionally (KoSens)."""
    keys = [k for k in entry.keys() if k != "grid"]
    if len(keys) < 2:
        raise ValueError(f"Grid entry needs 'grid' plus x and y mesh keys, got {list(entry)}.")
    return keys[0], keys[1]


def _observable_mask(species, grids_ref, grids_atten, shape, obs_limit):
    """Cells where every line behind ``species`` clears ``obs_limit`` in both grids.

    A ratio ``Num/Den`` is judged on its two constituent line grids, never on
    the ratio value.  An unknown constituent cannot be judged and stays observable.
    """
    mask = np.ones(shape, dtype=bool)
    if obs_limit is None:
        return mask
    lim = float(obs_limit)
    sides = [s.strip() for s in str(species).split("/")]
    lines = sides if len(sides) == 2 else [species]
    for line in lines:
        if line not in grids_ref or line not in grids_atten:
            continue
        for grids in (grids_ref, grids_atten):
            g = np.asarray(grids[line]["grid"], dtype=float)
            if g.shape != shape:
                break
            mask &= g >= lim          # NaN compares False: undefined is not observable
    return mask


def _is_ice_species_key(species_key):
    """True if any side of a key is a grain-surface (``J``-prefixed) species."""
    for part in str(species_key).split("/"):
        base = part.split("(")[0].strip()
        if len(base) > 1 and base.startswith("J"):
            return True
    return False


def _is_isotopologue_key(species_key):
    """True if any side of a key is an isotopologue (tested on base names only)."""
    for part in str(species_key).split("/"):
        base = part.split("(")[0].strip()
        if any(marker in base for marker in ISOTOPE_MARKERS):
            return True
    return False


def _drop_keys(pixels, test, what, notes):
    keys = [s for s in pixels["species"].unique() if test(s)]
    if not keys:
        return pixels
    shown = ", ".join(str(k) for k in keys[:8])
    more = f" (+{len(keys) - 8} more)" if len(keys) > 8 else ""
    notes.append(f"Skipped {len(keys)} {what} species/ratios: {shown}{more}.")
    return pixels[~pixels["species"].isin(keys)]


def _cap_per_constituent(ranked, n, max_per=2):
    """Top ``n`` rows of a pre-sorted table, at most ``max_per`` sharing a species.

    Ratios on a common numerator with inert denominators restate one finding;
    uncapped they fill every slot.  Returns ``(kept_frame, held_keys)``.
    """
    limit = max(int(max_per), 1)
    used, keep, held = {}, [], []
    for idx, key in ranked["species"].items():
        parts = {p.split("(")[0].strip() for p in str(key).split("/")}
        if any(used.get(part, 0) >= limit for part in parts):
            held.append(key)
            continue
        for part in parts:
            used[part] = used.get(part, 0) + 1
        keep.append(idx)
        if len(keep) >= int(n):
            break
    return ranked.loc[keep], held


def _capped_note(held, max_per):
    if not held:
        return None
    shown = ", ".join(str(k) for k in held[:8])
    more = f" (+{len(held) - 8} more)" if len(held) > 8 else ""
    return (f"{len(held)} higher-ranked entries suppressed by max per constituent = "
            f"{max_per:g}: {shown}{more} — the same species restated against other "
            "denominators, one finding not many.")


def _abs_shift_bin_edges_and_labels(shift_abs_edges):
    """|Δ| bin edges (0 … inf) and plain-text labels."""
    edges = [0.0] + [float(v) for v in shift_abs_edges]
    if np.isfinite(edges[-1]):
        edges.append(np.inf)
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if not np.isfinite(hi):
            labels.append(f"|Δ| ≥ {lo:g}")
        elif lo == 0.0:
            labels.append(f"|Δ| < {hi:g}")
        else:
            labels.append(f"{lo:g}–{hi:g}")
    return np.asarray(edges, dtype=float), labels


def _positive_logspace_edges(values, n_bins):
    fin = np.asarray(values, dtype=float)
    fin = fin[np.isfinite(fin) & (fin > 0.0)]
    if fin.size == 0:
        return None
    lo, hi = float(np.min(fin)), float(np.max(fin))
    n_bins = max(int(n_bins), 1)
    if hi <= lo or np.isclose(np.log10(hi), np.log10(lo)):
        return np.array([lo * 0.5, hi * 2.0 if hi > lo else lo * 2.0], dtype=float)
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    edges[0] = lo * (1.0 - 1e-12)
    edges[-1] = hi * (1.0 + 1e-12)
    return edges


def _log_bin_labels(edges):
    labels = [f"{lo:.1e}–{hi:.1e}" for lo, hi in zip(edges[:-1], edges[1:])]
    seen, unique = {}, []
    for lab in labels:
        n = seen.get(lab, 0)
        seen[lab] = n + 1
        unique.append(lab if n == 0 else f"{lab} ({n + 1})")
    return unique


# ---------------------------------------------------------------------------
# The pixel table: one row per (species, cell)
# ---------------------------------------------------------------------------

def collect_attenuation_x_shift_table(grids_ref, grids_atten, *, species_list=None,
                                      x_shift_match_atol=0.0,
                                      x_shift_match_rtol=X_SHIFT_MATCH_RTOL,
                                      x_shift_scan_direction="rightward",
                                      x_shift_edge_rtol=1e-12, obs_limit=None):
    """KoSens pixel table for one 2-D plane (``my_grids``-style dicts).

    Columns: species, iy, ix, x, y, x_name, y_name, shift_dex, slope_dex,
    value_ref, value_atten, observable, abs_shift_dex.
    """
    shared = sorted(set(grids_ref) & set(grids_atten))
    species_list = shared if species_list is None else [s for s in species_list if s in shared]
    scan_direction = _normalize_x_shift_scan_direction(x_shift_scan_direction)
    frames = []
    for species in species_list:
        d0, d1 = grids_ref[species], grids_atten[species]
        y_key, x_key = _grid_entry_xy_keys(d0)
        g0 = np.asarray(d0["grid"], dtype=float)
        g1 = np.asarray(d1["grid"], dtype=float)
        xm = np.asarray(d0[x_key], dtype=float)
        ym = np.asarray(d0[y_key], dtype=float)
        if g0.shape != g1.shape or g0.shape != xm.shape or g0.shape != ym.shape:
            continue
        shift = horizontal_intensity_matched_x_shift_dex(
            g0, g1, xm, match_atol=x_shift_match_atol, match_rtol=x_shift_match_rtol,
            scan_direction=scan_direction, x_edge_rtol=x_shift_edge_rtol)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ref = np.where(g0 > 0.0, np.log10(g0), np.nan)
            log_x = np.where(xm > 0.0, np.log10(xm), np.nan)
        slope = _log_intensity_slope(log_ref, log_x)
        observable = _observable_mask(species, grids_ref, grids_atten, g0.shape, obs_limit)
        iy, ix = np.indices(shift.shape)
        frames.append(pd.DataFrame({
            "species": species, "iy": iy.ravel(), "ix": ix.ravel(),
            "x": xm.ravel(), "y": ym.ravel(), "x_name": x_key, "y_name": y_key,
            "shift_dex": shift.ravel(), "slope_dex": slope.ravel(),
            "value_ref": g0.ravel(), "value_atten": g1.ravel(),
            "observable": observable.ravel(),
        }))
    if not frames:
        return pd.DataFrame()
    pixel_df = pd.concat(frames, ignore_index=True)
    pixel_df["abs_shift_dex"] = np.abs(pixel_df["shift_dex"])
    return pixel_df


def cube_plane(entry, density_index):
    """(χ-row × ζ-col) plane of one prisma-tool cube at one density index.

    Cubes are ``(n_ζ, n_χ, n_n)`` with meshgrids under ``gn.PARAM_MESH_KEYS``.
    Key order is ``grid, y, x`` because :func:`_grid_entry_xy_keys` is positional.
    """
    k = int(density_index)
    grid = np.asarray(entry["grid"], dtype=float)
    fuv = np.asarray(entry[gn.PARAM_MESH_KEYS["fuv"]], dtype=float)
    crir = np.asarray(entry[gn.PARAM_MESH_KEYS["crir"]], dtype=float)
    return {"grid": grid[:, :, k].T, "fuv": fuv[:, :, k].T, "crir": crir[:, :, k].T}


def cube_densities(entry):
    dens = np.asarray(entry[gn.PARAM_MESH_KEYS["density"]], dtype=float)
    return dens[0, 0, :]


def collect_pixels(ref_cubes, att_cubes, *, species_list=None, density_indices=None,
                   match_rtol=X_SHIFT_MATCH_RTOL, scan_direction="rightward",
                   obs_limit=None, progress=None):
    """Pixel table over one or more density slices of two cube dicts.

    Adds ``n`` (density), ``i_n``, ``row_id`` (one fixed-(n, χ) ζ-row),
    ``response_dex`` = R and ``bound_dex`` = log10(ζ_row_max / ζ): both are
    fixed on the **full** row here, so later subsetting by environment can
    never move a row end.
    """
    shared = [s for s in ref_cubes if s in att_cubes]
    species = shared if species_list is None else [s for s in species_list if s in shared]
    if not species:
        return pd.DataFrame()
    densities = cube_densities(ref_cubes[species[0]])
    idx = list(range(densities.size)) if density_indices is None else list(density_indices)
    n_work = max(1, len(idx) * len(species))
    frames, done = [], 0
    for k in idx:
        ref2 = {s: cube_plane(ref_cubes[s], k) for s in shared}
        att2 = {s: cube_plane(att_cubes[s], k) for s in shared}
        for s in species:
            df = collect_attenuation_x_shift_table(
                ref2, att2, species_list=[s], x_shift_match_rtol=match_rtol,
                x_shift_scan_direction=scan_direction, obs_limit=obs_limit)
            if len(df):
                df["n"] = float(densities[k])
                df["i_n"] = int(k)
                frames.append(df)
            done += 1
            if callable(progress):
                progress(done / n_work, f"Shift search: {s} at n_H = {densities[k]:.3g} "
                                        f"({done}/{n_work})")
    if not frames:
        return pd.DataFrame()
    return _ensure_cells(pd.concat(frames, ignore_index=True))


def _ensure_cells(pixels):
    """Fill the per-cell columns the reducers need (idempotent).

    Tables built by hand (tests, KoSens pixel tables) have no density columns;
    they are one slice.
    """
    if "observable" not in pixels.columns:
        pixels = pixels.assign(observable=True)
    if "i_n" not in pixels.columns:
        pixels = pixels.assign(i_n=0, n=np.nan)
    if "row_id" not in pixels.columns:
        pixels = pixels.assign(
            row_id=pixels["i_n"].astype(np.int64) * 1_000_000 + pixels["iy"].astype(np.int64))
    if "response_dex" not in pixels.columns:
        ref = pixels["value_ref"].to_numpy(dtype=float)
        att = pixels["value_atten"].to_numpy(dtype=float)
        pos = np.isfinite(ref) & np.isfinite(att) & (ref > 0.0) & (att > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            resp = np.where(pos, np.log10(np.where(pos, att / np.where(pos, ref, 1.0), 1.0)),
                            np.nan)
        pixels = pixels.assign(response_dex=resp)
    if "bound_dex" not in pixels.columns:
        x = pixels["x"].to_numpy(dtype=float)
        x_ok = np.isfinite(x) & (x > 0.0)
        x_row_max = (pd.Series(np.where(x_ok, x, 0.0), index=pixels.index)
                     .groupby([pixels["species"], pixels["row_id"]]).transform("max")
                     .to_numpy(dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            bound = np.where(x_ok & (x_row_max > 0.0),
                             np.log10(x_row_max / np.where(x_ok, x, 1.0)), np.nan)
        pixels = pixels.assign(bound_dex=bound)
    return pixels


# ---------------------------------------------------------------------------
# Step 1 — breadth: which species move, and where
# ---------------------------------------------------------------------------

def _empty_summary(pixels, notes):
    return {"pixels": pixels if pixels is not None else pd.DataFrame(),
            "species_summary": pd.DataFrame(), "top_models": pd.DataFrame(),
            "top_overall": pd.DataFrame(), "regime_median": pd.DataFrame(),
            "regime_winners": pd.DataFrame(), "heat_x": None, "heat_y": None,
            "heat_n": None, "x_labels": None, "y_labels": None,
            "bin_counts": pd.DataFrame(), "bin_labels": [],
            "top_species": [], "n_models_all": np.nan, "notes": notes}


def shift_summary(pixels, *, shift_abs_edges=(0.1, 0.3, 0.5, 1.0), min_abs_shift_dex=0.1,
                  min_slope_dex=0.1, max_frac_unmatched=0.5, exclude_ice=True,
                  exclude_isotopologues=False, n_x_bins=4, n_y_bins=4, n_top_models=8,
                  n_top_summary_species=12, max_per_constituent=2,
                  min_frac_observable=0.5):
    """Numerics of KoSens ``plot_attenuation_shift_summary``.

    Species are sorted by ``n_affected = n_shifted + n_unmatched`` (breadth:
    an unmatched cell counts as movement).  Counts ignore the slope gate;
    medians, regime tables and top-model lists use only trusted cells
    (``|S| >= min_slope_dex``).  Species above ``max_frac_unmatched`` or below
    ``min_frac_observable`` stay in ``species_summary`` with ``ranked=False``.
    With more than one density, regimes and winners are also split by ``n``.
    """
    notes = []
    if pixels is None or len(pixels) == 0:
        return _empty_summary(pixels, notes)
    pixels = _ensure_cells(pixels)
    if exclude_ice:
        pixels = _drop_keys(pixels, _is_ice_species_key, "ice", notes)
    if exclude_isotopologues:
        pixels = _drop_keys(pixels, _is_isotopologue_key, "isotopologue", notes)
    if len(pixels) == 0:
        return _empty_summary(pixels, notes)

    pixels_out = pixels
    frac_obs = pixels.groupby("species")["observable"].mean()
    never_obs = frac_obs.index[frac_obs <= 0.0].tolist()
    if never_obs:
        notes.append(f"{len(never_obs)} species are below the detection limit in every "
                     f"cell and are left out: {', '.join(map(str, never_obs[:12]))}.")
    pixel_df = pixels[pixels["observable"].astype(bool)]

    min_abs = float(min_abs_shift_dex)
    min_slope = abs(float(min_slope_dex))
    valid = pixel_df.dropna(subset=["shift_dex"]).copy()
    if len(valid) == 0:
        notes.append("No finite x-shifts were found for any species.")
        return _empty_summary(pixels_out, notes)

    bin_edges, bin_labels = _abs_shift_bin_edges_and_labels(shift_abs_edges)
    valid["shift_bin"] = pd.cut(valid["abs_shift_dex"], bins=bin_edges, labels=bin_labels,
                                include_lowest=True, right=False)
    valid["slope_ok"] = (valid["slope_dex"].abs() >= min_slope if min_slope > 0.0
                         else np.ones(len(valid), dtype=bool))
    trusted = valid[valid["slope_ok"]]

    n_cells = pixel_df.groupby("species").size().rename("n_cells")
    n_valid = valid.groupby("species").size().rename("n_valid")
    n_valid = n_valid.reindex(n_cells.index).fillna(0).astype(int)
    n_unmatched = (n_cells - n_valid).rename("n_unmatched")
    n_trusted = trusted.groupby("species").size().rename("n_trusted")
    n_trusted = n_trusted.reindex(n_cells.index).fillna(0).astype(int)
    shifted = valid[valid["abs_shift_dex"] >= min_abs]
    n_shift = shifted.groupby("species").size().rename("n_shifted")
    stats = trusted.groupby("species")["abs_shift_dex"].agg(
        mean_abs_shift_dex="mean", median_abs_shift_dex="median",
        std_abs_shift_dex="std", max_abs_shift_dex="max")
    unmatched_label = "no match"
    bin_counts = (valid.groupby(["species", "shift_bin"], observed=False).size()
                  .unstack(fill_value=0))
    for lab in bin_labels:
        if lab not in bin_counts.columns:
            bin_counts[lab] = 0
    bin_counts = bin_counts.reindex(n_cells.index).fillna(0)[bin_labels]
    bin_counts[unmatched_label] = n_unmatched.to_numpy()

    summary = (n_cells.to_frame().join(n_valid, how="left").join(n_unmatched, how="left")
               .join(n_trusted, how="left").join(n_shift, how="left")
               .join(stats, how="left").join(bin_counts, how="left").reset_index())
    for col in ("n_shifted", "n_valid", "n_cells", "n_unmatched", "n_trusted"):
        summary[col] = summary[col].fillna(0).astype(int)
    summary["n_affected"] = summary["n_shifted"] + summary["n_unmatched"]
    zeros = np.zeros(len(summary), dtype=float)
    summary["frac_shifted"] = np.divide(summary["n_shifted"], summary["n_valid"],
                                        out=zeros.copy(), where=summary["n_valid"] > 0)
    summary["frac_unmatched"] = np.divide(summary["n_unmatched"], summary["n_cells"],
                                          out=zeros.copy(), where=summary["n_cells"] > 0)
    summary["frac_trusted"] = np.divide(summary["n_trusted"], summary["n_valid"],
                                        out=zeros.copy(), where=summary["n_valid"] > 0)
    summary["label"] = summary["species"].astype(str)
    summary["frac_observable"] = summary["species"].map(frac_obs).astype(float)
    over_unmatched = summary["frac_unmatched"] > float(max_frac_unmatched)
    low_obs = summary["frac_observable"] < float(min_frac_observable)
    summary["ranked"] = ~over_unmatched & ~low_obs
    summary = summary.sort_values(["n_affected", "n_unmatched", "median_abs_shift_dex"],
                                  ascending=False, na_position="last").reset_index(drop=True)

    low_obs = summary["frac_observable"] < float(min_frac_observable)
    over_unmatched = summary["frac_unmatched"] > float(max_frac_unmatched)
    for held_mask, reason in (
            (low_obs, f"less than {float(min_frac_observable):g} of their grid above the "
                      "detection limit"),
            (over_unmatched & ~low_obs, f"more than {float(max_frac_unmatched):g} of their "
                                        "shift grid unmatched")):
        held = summary.loc[held_mask, "species"].tolist()
        if held:
            notes.append(f"Not ranked — {len(held)} species with {reason} (kept in the "
                         f"table with ranked = no): {', '.join(map(str, held[:12]))}"
                         f"{'' if len(held) <= 12 else ', …'}.")

    censored_species = summary.loc[~summary["ranked"], "species"].tolist()
    if censored_species:
        valid = valid[~valid["species"].isin(censored_species)]
        trusted = valid[valid["slope_ok"]]

    top_n = max(int(n_top_models), 1)
    top_cols = ["species", "n", "x", "y", "shift_dex", "abs_shift_dex", "slope_dex",
                "value_ref", "value_atten"]
    top_parts = [g.nlargest(top_n, "abs_shift_dex") for _, g in trusted.groupby("species")]
    top_models = (pd.concat(top_parts, ignore_index=True)[top_cols] if top_parts
                  else pd.DataFrame(columns=top_cols))
    top_overall = (trusted.nlargest(min(20, len(trusted)), "abs_shift_dex")[top_cols]
                   .reset_index(drop=True))

    multi_n = pixel_df["i_n"].nunique() > 1
    regime_keys = (["n"] if multi_n else []) + ["x_bin", "y_bin"]
    x_edges = _positive_logspace_edges(valid["x"], n_x_bins)
    y_edges = _positive_logspace_edges(valid["y"], n_y_bins)
    regime_median, winners = pd.DataFrame(), pd.DataFrame()
    x_labels = y_labels = None
    heat_x = heat_y = heat_n = None
    plot_summary = summary[summary["ranked"]]
    species_order = plot_summary["species"].tolist()
    if x_edges is not None and y_edges is not None:
        x_labels, y_labels = _log_bin_labels(x_edges), _log_bin_labels(y_edges)
        valid["x_bin"] = pd.cut(valid["x"], bins=x_edges, labels=x_labels, include_lowest=True)
        valid["y_bin"] = pd.cut(valid["y"], bins=y_edges, labels=y_labels, include_lowest=True)
        trusted = valid[valid["slope_ok"]]
        regime_median = (trusted.groupby(["species", *regime_keys], observed=True)
                         .agg(median_abs_shift_dex=("abs_shift_dex", "median"),
                              n_models=("abs_shift_dex", "count"))
                         .reset_index())
        if len(regime_median) > 0:
            peak = regime_median.groupby(regime_keys, observed=True)[
                "median_abs_shift_dex"].idxmax()
            winners = regime_median.loc[peak].reset_index(drop=True)
            winners["label"] = winners["species"].astype(str)

        def _heat(key):
            return (trusted.groupby(["species", key], observed=True)["abs_shift_dex"]
                    .median().unstack().reindex(species_order))
        heat_x, heat_y = _heat("x_bin"), _heat("y_bin")
        if multi_n:
            heat_n = _heat("n")

    top_n_sum = min(max(int(n_top_summary_species), 1), max(len(species_order), 1))
    top_by_median = plot_summary.sort_values("median_abs_shift_dex", ascending=False,
                                             na_position="last")
    top_tab, held_top = _cap_per_constituent(top_by_median, top_n_sum, max_per_constituent)
    note = _capped_note(held_top, max_per_constituent)
    if note:
        notes.append(note)

    cell_totals = summary["n_cells"].to_numpy(dtype=float)
    n_models_all = (float(cell_totals[0]) if len(cell_totals)
                    and np.all(cell_totals == cell_totals[0]) else np.nan)

    leading = ["species", "label", "ranked", "frac_observable", "n_cells", "n_valid",
               "n_unmatched", "n_shifted", "n_affected", "frac_shifted", "frac_unmatched",
               "median_abs_shift_dex", "mean_abs_shift_dex", "std_abs_shift_dex",
               "max_abs_shift_dex"]
    out_summary = summary[leading + [c for c in summary.columns if c not in leading]]
    return {
        "pixels": pixels_out, "species_summary": out_summary, "top_models": top_models,
        "top_overall": top_overall, "regime_median": regime_median,
        "regime_winners": winners, "heat_x": heat_x, "heat_y": heat_y, "heat_n": heat_n,
        "x_labels": x_labels, "y_labels": y_labels,
        "bin_counts": bin_counts, "bin_labels": bin_labels,
        "top_species": top_tab["species"].tolist(), "n_models_all": n_models_all,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Step 2 — magnitude (ζ error) and probe quality (CR or FUV?)
# ---------------------------------------------------------------------------

RESPONSE_COLUMNS = [
    "species", "label", "error_dex", "error_p10", "error_p90", "frac_censored",
    "n_error_cells", "response_dex", "abs_response_dex", "swing_dex", "fuv_swing_dex",
    "n_swing_dex", "fuv_trend", "frac_trusted", "n_fuv_rows", "frac_observable",
]


def _swing(values):
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, 90) - np.percentile(values, 10)) if values.size >= 2 \
        else np.nan


def response_table(pixels, *, max_species=8, min_abs_response_dex=0.0, min_slope_dex=0.1,
                   max_frac_censored=0.5, max_per_constituent=2, exclude_ice=True,
                   exclude_isotopologues=False, min_frac_observable=0.5):
    """Numerics of KoSens ``plot_attenuation_fuv_response``.

    ``error_dex`` is the median |Δ| over trusted cells, censored (no-match)
    cells entering at their lower bound log10(ζ_row_max/ζ); ``probe_margin =
    |R| − swing_dex``, where the swing is p90 − p10 of the per-row median R.
    Each row is one CR experiment at fixed (n_H, χ); with a single density
    ``swing_dex`` is exactly the KoSens ``fuv_swing_dex``.  In 3-D two
    attribution columns tell which axis drives the swing: ``fuv_swing_dex``
    (median over densities of the per-density χ-swing) and ``n_swing_dex``
    (p90 − p10 across densities of the per-density median R).

    Returns ``{'table', 'notes'}``; the table is sorted by ``error_dex`` and
    carries ``ranked`` (left panel) and ``probe_ranked`` (right panel).
    """
    notes = []
    empty = pd.DataFrame(columns=[*RESPONSE_COLUMNS, "probe_margin", "den_contrib",
                                  "ranked", "probe_ranked"])
    if pixels is None or len(pixels) == 0:
        return {"table": empty, "notes": notes}
    pixels = _ensure_cells(pixels)
    if exclude_isotopologues:
        pixels = _drop_keys(pixels, _is_isotopologue_key, "isotopologue", notes)
    if exclude_ice:
        pixels = _drop_keys(pixels, _is_ice_species_key, "ice", notes)
    if len(pixels) == 0:
        return {"table": empty, "notes": notes}

    records = []
    for species, sub in pixels.groupby("species"):
        obs = sub["observable"].to_numpy(dtype=bool)
        response = np.where(obs, sub["response_dex"].to_numpy(dtype=float), np.nan)
        slope = sub["slope_dex"].to_numpy(dtype=float)
        shift = sub["abs_shift_dex"].to_numpy(dtype=float)
        bound = sub["bound_dex"].to_numpy(dtype=float)
        trusted = np.isfinite(slope) & (np.abs(slope) >= float(min_slope_dex)) & obs
        domain = np.isfinite(response)
        frac_trusted = (float(np.sum(trusted & domain) / np.sum(domain))
                        if np.any(domain) else np.nan)
        measured = trusted & np.isfinite(shift)
        # bound <= 0: ζ_nom sits at the row end, no rightward room, no information.
        censored = trusted & ~np.isfinite(shift) & np.isfinite(bound) & (bound > 0.0)
        err = np.where(censored, bound, np.where(measured, shift, np.nan))
        err_vals = err[np.isfinite(err)]
        n_err = int(err_vals.size)

        rows = pd.DataFrame({"row": sub["row_id"].to_numpy(), "i_n": sub["i_n"].to_numpy(),
                             "r": response, "y": sub["y"].to_numpy(dtype=float)})
        per_row = rows.groupby("row").agg(r=("r", "median"), y=("y", "median"),
                                          i_n=("i_n", "first"))
        keep = np.isfinite(per_row["r"]) & np.isfinite(per_row["y"]) & (per_row["y"] > 0.0)
        per_row = per_row[keep]
        n_rows = len(per_row)
        if n_rows == 0:
            continue
        r_keep = per_row["r"].to_numpy(dtype=float)
        y_keep = per_row["y"].to_numpy(dtype=float)
        swing = _swing(r_keep)
        trend = (float(linregress(np.log10(y_keep), r_keep).slope)
                 if n_rows >= 3 and np.ptp(y_keep) > 0.0 else np.nan)
        by_n = per_row.groupby("i_n")["r"]
        if by_n.ngroups > 1:
            fuv_sw = np.array([_swing(g) for _, g in by_n], dtype=float)
            fuv_swing = float(np.nanmedian(fuv_sw)) if np.isfinite(fuv_sw).any() else np.nan
            n_swing = _swing(by_n.median().to_numpy(dtype=float))
        else:
            fuv_swing, n_swing = swing, np.nan
        records.append({
            "species": species, "label": str(species),
            "error_dex": float(np.median(err_vals)) if n_err else np.nan,
            "error_p10": float(np.percentile(err_vals, 10)) if n_err else np.nan,
            "error_p90": float(np.percentile(err_vals, 90)) if n_err else np.nan,
            "frac_censored": float(np.sum(censored) / n_err) if n_err else np.nan,
            "n_error_cells": n_err,
            "response_dex": float(np.median(r_keep)),
            "abs_response_dex": float(np.abs(np.median(r_keep))),
            "swing_dex": swing, "fuv_swing_dex": fuv_swing, "n_swing_dex": n_swing,
            "fuv_trend": trend, "frac_trusted": frac_trusted, "n_fuv_rows": n_rows,
            "frac_observable": float(obs.mean()),
        })
    if not records:
        notes.append("No species had a usable intensity response.")
        return {"table": empty, "notes": notes}

    table = pd.DataFrame.from_records(records)[RESPONSE_COLUMNS]
    table["probe_margin"] = table["abs_response_dex"] - table["swing_dex"]
    r_by_key = dict(zip(table["species"], table["response_dex"]))

    def _denominator_share(key):
        sides = str(key).split("/")
        if len(sides) != 2:
            return np.nan
        r_num = r_by_key.get(sides[0].strip(), np.nan)
        r_den = r_by_key.get(sides[1].strip(), np.nan)
        total = abs(r_num) + abs(r_den)
        if not np.isfinite(total) or total <= 0.0:
            return np.nan
        return float(abs(r_den) / total)

    table["den_contrib"] = table["species"].map(_denominator_share)
    table = table.sort_values("error_dex", ascending=False,
                              na_position="last").reset_index(drop=True)

    over_censored = (np.isfinite(table["error_dex"])
                     & (table["frac_censored"].fillna(0.0) > float(max_frac_censored)))
    low_obs = table["frac_observable"] < float(min_frac_observable)
    gated = table[(table["abs_response_dex"] >= float(min_abs_response_dex))
                  & np.isfinite(table["error_dex"]) & ~over_censored & ~low_obs]
    n_ungated = int((~np.isfinite(table["error_dex"])).sum())
    if n_ungated:
        notes.append(f"{n_ungated} species have no cell with |S| ≥ {min_slope_dex:g}; "
                     "their ζ error is unmeasurable (listed only).")
    if over_censored.any():
        held = ", ".join(f"{r['label']} ({r['frac_censored']:.0%})"
                         for _, r in table[over_censored].iterrows())
        notes.append(f"Not ranked — {int(over_censored.sum())} species with more than "
                     f"{float(max_frac_censored):.0%} censored cells: {held}. Their error "
                     "would be an imputed bound, and a no-match cell need not be a CR effect.")
    if low_obs.any():
        held = ", ".join(f"{r['label']} ({r['frac_observable']:.0%})"
                         for _, r in table[low_obs].iterrows())
        notes.append(f"Not ranked — {int(low_obs.sum())} species observable in less than "
                     f"{float(min_frac_observable):.0%} of the grid: {held}.")

    n_draw = max(int(max_species), 1)
    plot_tab, held_error = _cap_per_constituent(gated, n_draw, max_per_constituent)
    probe_tab, held_probe = _cap_per_constituent(
        gated.dropna(subset=["probe_margin"]).sort_values("probe_margin", ascending=False),
        n_draw, max_per_constituent)
    for held in (held_error, held_probe):
        note = _capped_note(held, max_per_constituent)
        if note:
            notes.append(note)
    table["ranked"] = table["species"].isin(set(plot_tab["species"]))
    table["probe_ranked"] = table["species"].isin(set(probe_tab["species"]))
    return {"table": table, "notes": notes}


# ---------------------------------------------------------------------------
# Step 3 — one verdict per species
# ---------------------------------------------------------------------------

RECONCILE_COLUMNS = [
    "species", "label", "observable", "broad", "honest", "cr_led", "matters", "verdict",
    "frac_observable", "frac_shifted", "frac_unmatched", "median_abs_shift_dex", "error_dex",
    "error_p90", "bound_gap", "frac_censored", "abs_response_dex", "swing_dex",
    "fuv_swing_dex", "n_swing_dex", "probe_margin", "n_cells", "n_fuv_rows",
]


def reconcile(summary, response, *, min_frac_shifted=0.5, max_frac_unmatched=0.5,
              max_frac_censored=0.5, response_floor_dex=0.1, min_error_dex=0.3,
              min_frac_observable=0.5):
    """KoSens ``reconcile_attenuation_species``: four questions, first failure names the verdict.

    broad → frac_shifted ≥ min_frac_shifted; honest → frac_unmatched ≤
    max_frac_unmatched and frac_censored ≤ max_frac_censored; cr_led →
    probe_margin > 0; matters → error_dex ≥ min_error_dex.  Returns
    ``table`` (cascade order), ``problematic`` (honest, by error_dex:
    correction priority) and ``tracers`` (CR-led, by probe_margin: line choice).
    """
    if isinstance(summary, dict):
        summary = summary["species_summary"]
    if isinstance(response, dict):
        response = response["table"]
    if summary is None or response is None or not len(summary) or not len(response):
        empty = pd.DataFrame(columns=RECONCILE_COLUMNS)
        return {"table": empty, "problematic": empty.copy(), "tracers": empty.copy()}

    merged = summary.merge(response, on="species", how="outer", suffixes=("_shift", "_fuv"))
    merged["label"] = merged.get(
        "label_shift", pd.Series(index=merged.index, dtype=object)).fillna(merged.get("label_fuv"))
    merged["bound_gap"] = merged["error_dex"] - merged["median_abs_shift_dex"]

    frac_shifted = merged["frac_shifted"].astype(float)
    frac_unmatched = merged["frac_unmatched"].astype(float)
    frac_censored = merged["frac_censored"].astype(float).fillna(0.0)
    resp = merged["abs_response_dex"].astype(float)
    margin = merged["probe_margin"].astype(float)
    error = merged["error_dex"].astype(float)

    frac_obs = pd.Series(np.nan, index=merged.index, dtype=float)
    for col in ("frac_observable_shift", "frac_observable_fuv", "frac_observable"):
        if col in merged.columns:
            frac_obs = frac_obs.fillna(merged[col].astype(float))
    merged["frac_observable"] = frac_obs.fillna(1.0)
    merged["observable"] = merged["frac_observable"] >= float(min_frac_observable)
    merged["broad"] = frac_shifted >= float(min_frac_shifted)
    merged["honest"] = ((frac_unmatched <= float(max_frac_unmatched))
                        & (frac_censored <= float(max_frac_censored)))
    merged["cr_led"] = margin > 0.0
    merged["matters"] = error >= float(min_error_dex)

    has_data = np.isfinite(resp) & np.isfinite(margin)
    merged["verdict"] = np.select(
        [~has_data, ~merged["observable"], resp < float(response_floor_dex),
         ~merged["honest"], ~merged["cr_led"], ~merged["broad"], merged["matters"]],
        ["no data", "unobservable", "no response", "unmeasurable", "FUV tracer",
         "CR-led, narrow", "CRIR probe"],
        default="CR-led, small bias")
    merged["_rank"] = merged["verdict"].map({v: i for i, v in enumerate(VERDICT_ORDER)})
    merged = merged.sort_values(["_rank", "error_dex"], ascending=[True, False],
                                na_position="last").reset_index(drop=True)
    for col in RECONCILE_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan
    table = merged[RECONCILE_COLUMNS]

    problematic = table[table["honest"].astype(bool) & table["observable"].astype(bool)
                        & np.isfinite(table["error_dex"].astype(float))].sort_values(
        "error_dex", ascending=False).reset_index(drop=True)
    tracers = table[table["verdict"].isin(CR_FAMILY)].sort_values(
        "probe_margin", ascending=False).reset_index(drop=True)
    return {"table": table, "problematic": problematic, "tracers": tracers}


# ---------------------------------------------------------------------------
# 3-D: verdicts per density and per environment
# ---------------------------------------------------------------------------

def _run_trio(pixels, summary_kw, response_kw, reconcile_kw):
    s = shift_summary(pixels, **summary_kw)
    r = response_table(pixels, **response_kw)
    return s, r, reconcile(s, r, **reconcile_kw)


def verdicts_by_density(pixels, summary_kw, response_kw, reconcile_kw):
    """Long table of the reconciled verdict per (species, density slice)."""
    pixels = _ensure_cells(pixels)
    frames = []
    for i_n, sub in pixels.groupby("i_n"):
        _s, _r, v = _run_trio(sub, summary_kw, response_kw, reconcile_kw)
        t = v["table"].copy()
        t["n"] = float(sub["n"].iloc[0])
        t["i_n"] = int(i_n)
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_axis_edges(text, values):
    """Interior bin edges for one axis.

    ``auto`` (or blank) splits once at the log midpoint of ``values``;
    ``full``/``none`` keeps one bin; otherwise comma-separated numbers.
    """
    raw = str(text or "auto").strip().lower()
    if raw in ("full", "none", "off"):
        return []
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0.0)]
    if raw == "auto":
        if v.size == 0:
            return []
        lo, hi = np.log10(v.min()), np.log10(v.max())
        return [float(10.0 ** (0.5 * (lo + hi)))] if hi > lo else []
    return sorted(float(p) for p in raw.replace(";", ",").split(",") if p.strip())


def _bin_names(n_bins):
    return {1: [""], 2: ["low", "high"], 3: ["low", "mid", "high"]}.get(
        n_bins, [f"bin {i + 1}" for i in range(n_bins)])


ENV_AXES = (("n", "n_H"), ("y", "χ"), ("x", "ζ"))


def assign_environments(pixels, edges):
    """Label every cell with its (n_H, χ, ζ) environment.

    ``edges`` maps ``'n'``, ``'y'`` (χ), ``'x'`` (ζ) to interior edges.  A cell
    on an edge goes to the upper bin.  Returns ``(labels, env_info)``;
    ``env_info`` lists every environment in axis order with its ranges.
    """
    parts, split = [], []
    for col, sym in ENV_AXES:
        e = sorted(float(v) for v in (edges.get(col) or []))
        if not e:
            continue
        names = _bin_names(len(e) + 1)
        idx = np.searchsorted(np.asarray(e), pixels[col].to_numpy(dtype=float), side="right")
        parts.append((sym, np.asarray(names, dtype=object)[idx]))
        split.append((sym, names, [-np.inf, *e, np.inf]))
    if parts:
        labels = [" · ".join(f"{names[i]} {sym}" for sym, names in parts)
                  for i in range(len(pixels))]
    else:
        labels = ["whole grid"] * len(pixels)
    env_info = []
    for combo in itertools.product(*[range(len(a[1])) for a in split]):
        label = " · ".join(f"{a[1][i]} {a[0]}" for a, i in zip(split, combo)) or "whole grid"
        ranges = {a[0]: (a[2][i], a[2][i + 1]) for a, i in zip(split, combo)}
        env_info.append({"label": label, "ranges": ranges})
    return np.asarray(labels, dtype=object), env_info


def environment_verdicts(pixels, edges, summary_kw, response_kw, reconcile_kw, *,
                         min_env_cells=10, progress=None):
    """Verdict per (species, environment), plus per-environment views.

    The swing inside an environment spans only the (n_H, χ) rows in it, so a
    verdict is local: *"at low n_H and high ζ this line is a clean probe"*.
    Censoring bounds and slopes come from the full rows (``collect_pixels``).
    An environment with fewer than ``min_env_cells`` cells or fewer than two
    rows for a species grades it ``too few models`` instead of a noisy verdict.
    """
    pixels = _ensure_cells(pixels)
    labels, env_info = assign_environments(pixels, edges)
    pixels = pixels.assign(environment=labels)
    frames, views = [], {}
    for k, env in enumerate(env_info, 1):
        sub = pixels[pixels["environment"] == env["label"]]
        if callable(progress):
            progress(k / max(len(env_info), 1), f"Environment {env['label']}")
        if not len(sub):
            continue
        _s, _r, v = _run_trio(sub, summary_kw, response_kw, reconcile_kw)
        t = v["table"].copy()
        thin = ((t["n_cells"].astype(float) < float(min_env_cells))
                | (t["n_fuv_rows"].astype(float) < 2))
        t.loc[thin & (t["verdict"] != "no data"), "verdict"] = "too few models"
        t["environment"] = env["label"]
        frames.append(t)
        ok = t["verdict"] != "too few models"
        views[env["label"]] = {
            "tracers": t[ok & t["verdict"].isin(CR_FAMILY)].sort_values(
                "probe_margin", ascending=False).reset_index(drop=True),
            "problematic": t[ok & t["honest"].astype(bool) & t["observable"].astype(bool)
                             & np.isfinite(t["error_dex"].astype(float))].sort_values(
                "error_dex", ascending=False).reset_index(drop=True),
        }
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {"table": table, "environments": env_info, "views": views}


def run_workflow(pixels, *, summary_kw=None, response_kw=None, reconcile_kw=None,
                 env_edges=None, min_env_cells=10, progress=None):
    """Whole procedure on one pixel table.

    Thresholds that must agree across the steps (``min_slope_dex``,
    ``exclude_ice``, ``exclude_isotopologues``, ``max_frac_unmatched``,
    ``max_frac_censored``, ``min_frac_observable``) are copied into every
    step here, so they cannot drift apart.  With more than one density the
    per-density and per-environment verdicts are added.
    """
    summary_kw = dict(summary_kw or {})
    response_kw = dict(response_kw or {})
    reconcile_kw = dict(reconcile_kw or {})
    for key in ("min_slope_dex", "exclude_ice", "exclude_isotopologues", "min_frac_observable",
                "max_per_constituent"):
        if key in summary_kw:
            response_kw.setdefault(key, summary_kw[key])
    for key in ("max_frac_unmatched", "min_frac_observable"):
        if key in summary_kw:
            reconcile_kw.setdefault(key, summary_kw[key])
    if "max_frac_censored" in response_kw:
        reconcile_kw.setdefault("max_frac_censored", response_kw["max_frac_censored"])

    pixels = _ensure_cells(pixels)
    s, r, v = _run_trio(pixels, summary_kw, response_kw, reconcile_kw)
    out = {"summary": s, "response": r, "verdicts": v, "by_density": None,
           "environments": None, "multi_density": pixels["i_n"].nunique() > 1}
    if out["multi_density"]:
        if callable(progress):
            progress(0.3, "Verdicts per density…")
        out["by_density"] = verdicts_by_density(pixels, summary_kw, response_kw, reconcile_kw)
    if env_edges is not None:
        out["environments"] = environment_verdicts(
            pixels, env_edges, summary_kw, response_kw, reconcile_kw,
            min_env_cells=min_env_cells,
            progress=(lambda f, m: progress(0.4 + 0.6 * f, m)) if callable(progress) else None)
    return out


def density_strip(workflow, max_species=40):
    """Long table for the verdict-by-density matrix: every slice plus ``pooled``.

    Returns ``(long_table, col_order, col_labels, species)``; species follow the
    pooled cascade order, capped at ``max_species``.
    """
    by_n = workflow.get("by_density")
    pooled = workflow["verdicts"]["table"]
    if by_n is None or not len(by_n) or not len(pooled):
        return None, [], [], []
    long = pd.concat([by_n, pooled.assign(i_n=-1)], ignore_index=True)
    order = sorted(int(i) for i in by_n["i_n"].unique())
    dens = by_n.drop_duplicates("i_n").set_index("i_n")["n"]
    labels = [f"n_H = {dens[i]:.2g}" for i in order] + ["pooled"]
    return long, order + [-1], labels, pooled["species"].tolist()[:max_species]


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

import plotly.graph_objects as go                      # noqa: E402
from plotly.colors import sample_colorscale           # noqa: E402
from plotly.subplots import make_subplots             # noqa: E402

import plot_style as ps                               # noqa: E402
import probe_ranking as pr                            # noqa: E402

VERDICT_COLORS = {
    "CRIR probe": "#16a34a", "CR-led, small bias": "#86efac", "CR-led, narrow": "#facc15",
    "FUV tracer": "#a855f7", "unmeasurable": "#f97316", "no response": "#94a3b8",
    "unobservable": "#cbd5e1", "too few models": "#e2e8f0", "no data": "#f8fafc",
}
VERDICT_SHORT = {
    "CRIR probe": "probe", "CR-led, small bias": "small bias", "CR-led, narrow": "narrow",
    "FUV tracer": "FUV", "unmeasurable": "unmeas.", "no response": "no resp.",
    "unobservable": "unobs.", "too few models": "few", "no data": "—",
}


def pretty_key(key, fmt=None):
    """HTML label of a grid key: species part through ``fmt``, transition kept verbatim."""
    if fmt is None:
        return str(key)
    out = []
    for part in str(key).split("/"):
        base = part.split("(")[0].strip()
        out.append(part.replace(base, fmt(base), 1) if base else part)
    return "/".join(out)


def _style(fig, t, title, height):
    fig.update_layout(
        paper_bgcolor=t["paper_bg"], plot_bgcolor=t["plot_bg"],
        font=ps.layout_font(t["font"]), height=height,
        title=dict(text=title, font=ps.title_font(t["title"]), x=0.01, xanchor="left"),
        margin=dict(l=80, r=30, t=80, b=70),
        legend=dict(bgcolor=t["legend_bg"], bordercolor=t["legend_border"], borderwidth=1,
                    font=ps.legend_font(t["font"])),
    )
    fig.update_xaxes(**pr._axis_style(t))
    fig.update_yaxes(**pr._axis_style(t))
    fig.update_annotations(font=ps.axis_title_font(t["font"]))
    return fig


def empty_fig(message, theme=None):
    t = pr._theme_fallback(theme)
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=13, color=t["muted"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, t, "", 260)


def fig_shift_bars(summary_out, species=None, *, theme=None, fmt=None, title=None):
    """Step 1: models per |Δ| bin (stacked, lowest bin omitted) and median |Δ|."""
    t = pr._theme_fallback(theme)
    species = list(species if species is not None else summary_out.get("top_species") or [])
    counts = summary_out.get("bin_counts")
    if not species or counts is None or not len(counts):
        return empty_fig("No ranked species to draw.", theme)
    labels = list(summary_out["bin_labels"][1:]) + ["no match"]
    stack = counts.reindex(species)[labels].fillna(0.0)
    names = [pretty_key(s, fmt) for s in species]
    shades = sample_colorscale("YlOrRd", [0.30 + 0.65 * i / max(len(labels) - 2, 1)
                                          for i in range(len(labels) - 1)])
    colors = shades + ["#737373"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        subplot_titles=("Models that moved (|Δ| bins)", "Typical shift"))
    for lab, col in zip(labels, colors):
        fig.add_trace(go.Bar(x=names, y=stack[lab], name=lab if lab == "no match" else f"{lab} dex",
                             marker=dict(color=col, line=dict(color="black", width=0.6)),
                             hovertemplate="%{x}<br>" + lab + ": %{y}<extra></extra>"),
                      row=1, col=1)
    n_all = summary_out.get("n_models_all")
    if n_all is not None and np.isfinite(n_all):
        fig.add_hline(y=n_all, line=dict(color=t["muted"], dash="dot", width=1.4),
                      annotation_text=f"all models ({int(n_all)})",
                      annotation_position="top left", row=1, col=1)
    med = (summary_out["species_summary"].set_index("species")["median_abs_shift_dex"]
           .reindex(species))
    fig.add_trace(go.Bar(x=names, y=med, name="median |Δ|", showlegend=False,
                         marker=dict(color=sample_colorscale("YlOrRd", [0.65])[0],
                                     line=dict(color="black", width=0.8)),
                         hovertemplate="%{x}<br>median |Δ| = %{y:.3f} dex<extra></extra>"),
                  row=1, col=2)
    fig.update_layout(barmode="stack", legend_title_text="|Δ| bin")
    fig.update_yaxes(title_text="number of models", row=1, col=1)
    fig.update_yaxes(title_text="median |Δ| [dex]", row=1, col=2)
    fig.update_xaxes(tickangle=-40)
    return _style(fig, t, title or "Step 1 · How many models move, and by how much",
                  440 + 4 * max(len(n) for n in names))


def fig_regime_heatmaps(summary_out, species=None, *, theme=None, fmt=None):
    """Step 1: median |Δ| per ζ bin, χ bin (and n_H in 3-D) — *where* a species shifts."""
    t = pr._theme_fallback(theme)
    species = list(species if species is not None else summary_out.get("top_species") or [])
    heats = [(summary_out.get("heat_x"), "ζ bin [s⁻¹]"), (summary_out.get("heat_y"), "χ bin"),
             (summary_out.get("heat_n"), "n_H [cm⁻³]")]
    heats = [(h.reindex(species), lab) for h, lab in heats if h is not None]
    if not species or not heats:
        return empty_fig("No regime heatmaps (no trusted cells).", theme)
    vmax = max([0.05] + [float(np.nanmax(h.to_numpy(dtype=float)))
                         for h, _ in heats if np.isfinite(h.to_numpy(dtype=float)).any()])
    fig = make_subplots(rows=1, cols=len(heats), horizontal_spacing=0.08,
                        subplot_titles=[f"median |Δ| vs {lab}" for _, lab in heats])
    names = [pretty_key(s, fmt) for s in species]
    for j, (h, lab) in enumerate(heats, 1):
        cols = [f"{c:.2g}" if isinstance(c, float) else str(c) for c in h.columns]
        z = h.to_numpy(dtype=float)
        fig.add_trace(go.Heatmap(
            z=z, x=cols, y=names, zmin=0.0, zmax=vmax, colorscale="YlOrRd",
            text=np.where(np.isfinite(z), np.round(z, 2).astype(str), ""),
            texttemplate="%{text}", showscale=(j == len(heats)),
            colorbar=dict(title=dict(text="median |Δ| [dex]", font=ps.cbar_title_font(t["font"])),
                          tickfont=ps.cbar_tick_font(t["font"])),
            hovertemplate="%{y}<br>" + lab + " %{x}<br>median |Δ| = %{z:.3f} dex<extra></extra>"),
            row=1, col=j)
        fig.update_xaxes(title_text=lab, tickangle=-30, row=1, col=j)
        fig.update_yaxes(autorange="reversed", showticklabels=(j == 1), row=1, col=j)
    return _style(fig, t, "Step 1 · Where the shift happens", max(320, 60 + 34 * len(species)))


def fig_winner_map(summary_out, *, theme=None, fmt=None):
    """Step 1: in each (ζ, χ) regime the species with the largest median |Δ|.

    One panel per density in 3-D.  Read as *"here, follow up this species"* —
    not as a tracer ranking (an FUV tracer can win).
    """
    t = pr._theme_fallback(theme)
    win = summary_out.get("regime_winners")
    x_labels, y_labels = summary_out.get("x_labels"), summary_out.get("y_labels")
    if win is None or not len(win) or not x_labels:
        return empty_fig("No winner map (no trusted cells).", theme)
    panels = ([(None, win)] if "n" not in win.columns
              else [(n, g) for n, g in win.groupby("n")])
    n_cols = min(len(panels), 4)
    n_rows = int(np.ceil(len(panels) / n_cols))
    titles = ["" if n is None else f"n_H = {pr.format_phys(n)} cm⁻³" for n, _ in panels]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.16)
    vmax = max(float(win["median_abs_shift_dex"].max()), 0.05)
    for k, (_n, g) in enumerate(panels):
        lab = g.pivot(index="y_bin", columns="x_bin", values="label").reindex(
            index=y_labels, columns=x_labels)
        val = g.pivot(index="y_bin", columns="x_bin", values="median_abs_shift_dex").reindex(
            index=y_labels, columns=x_labels)
        cnt = g.pivot(index="y_bin", columns="x_bin", values="n_models").reindex(
            index=y_labels, columns=x_labels)
        text = lab.fillna("").map(lambda s: pretty_key(s, fmt).replace("/", "/<br>", 1))
        fig.add_trace(go.Heatmap(
            z=val.to_numpy(dtype=float), x=x_labels, y=y_labels, zmin=0.0, zmax=vmax,
            colorscale="YlOrRd", text=text.to_numpy(), texttemplate="%{text}", textfont=dict(size=14),
            customdata=np.dstack([lab.fillna("").to_numpy(), cnt.to_numpy(dtype=float)]),
            showscale=(k == 0),
            colorbar=dict(title=dict(text="winner median |Δ| [dex]",
                                     font=ps.cbar_title_font(t["font"])),
                          tickfont=ps.cbar_tick_font(t["font"])),
            hovertemplate=("ζ %{x}<br>χ %{y}<br>%{customdata[0]}<br>median |Δ| = %{z:.3f} dex"
                           "<br>trusted cells = %{customdata[1]:.0f}<extra></extra>")),
            row=k // n_cols + 1, col=k % n_cols + 1)
    fig.update_xaxes(title_text="ζ bin [s⁻¹]", tickangle=-30)
    fig.update_yaxes(title_text="χ bin")
    return _style(fig, t, "Step 1 · Winner map — which species to follow up where",
                  max(420, 360 * n_rows))


def fig_error_bars(table, *, theme=None, fmt=None):
    """Step 2 (left): ζ error from ignoring attenuation, largest at the top.

    Bar = median |Δ|; band = p10–p90 over grid cells (a spread, not an
    uncertainty); colour = fraction of censored cells; ▶ at p90 = lower limit.
    """
    t = pr._theme_fallback(theme)
    tab = table[table["ranked"].astype(bool)] if len(table) else table
    if not len(tab):
        return empty_fig("No species passed the gates for the error ranking.", theme)
    names = [pretty_key(s, fmt) for s in tab["species"]]
    cens = np.nan_to_num(tab["frac_censored"].to_numpy(dtype=float), nan=0.0)
    p10 = tab["error_p10"].to_numpy(dtype=float)
    p90 = tab["error_p90"].to_numpy(dtype=float)
    err = tab["error_dex"].to_numpy(dtype=float)
    mark = np.where(cens > 0.0, ">", "")
    custom = np.column_stack([mark, p10, p90, cens])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=err, orientation="h", name="median |Δ|", customdata=custom,
        marker=dict(color=cens, cmin=0.0, cmax=1.0, colorscale="YlOrRd",
                    line=dict(color="black", width=1.0),
                    colorbar=dict(title=dict(text="fraction of cells<br>that are lower limits",
                                             font=ps.cbar_title_font(t["font"])),
                                  tickfont=ps.cbar_tick_font(t["font"]))),
        hovertemplate=("%{y}<br>error = %{customdata[0]}%{x:.3f} dex"
                       "<br>p10–p90 = %{customdata[1]:.2f}–%{customdata[2]:.2f}"
                       "<br>censored = %{customdata[3]:.0%}<extra></extra>")))
    fig.add_trace(go.Bar(
        y=names, x=np.clip(p90 - p10, 0.0, None), base=p10, orientation="h", width=0.3,
        name="p10–p90 over cells", marker=dict(color="rgba(0,0,0,0.25)"), hoverinfo="skip"))
    lim = cens > 0.0
    if lim.any():
        fig.add_trace(go.Scatter(
            y=np.asarray(names, dtype=object)[lim], x=p90[lim], mode="markers",
            name="lower limit", hoverinfo="skip",
            marker=dict(symbol="triangle-right", size=11, color=t["font"])))
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=-0.18))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="|Δ| = |log₁₀(ζ_match / ζ_nom)| [dex]", rangemode="tozero")
    return _style(fig, t, "Step 2 · Error in ζ if attenuation is ignored",
                  max(640, 130 + 42 * len(names)))


def fig_probe_plane(table, *, response_floor_dex=0.1, probe_axis_max=1.5, theme=None,
                    fmt=None, swing_label="G₀ swing of R"):
    """Step 2 (right): |R| against its own environment swing.

    Below the diagonal the response beats its swing → the line reports cosmic
    rays; above it → FUV (or density) tracer; left of the floor → no response.
    """
    t = pr._theme_fallback(theme)
    tab = table[table["probe_ranked"].astype(bool)] if len(table) else table
    if not len(tab):
        return empty_fig("No species passed the gates for the probe panel.", theme)
    tab = tab.sort_values("probe_margin", ascending=False)
    x = tab["abs_response_dex"].to_numpy(dtype=float)
    y = tab["swing_dex"].to_numpy(dtype=float)
    err = tab["error_dex"].to_numpy(dtype=float)
    data_max = float(np.nanmax(np.concatenate([x, y]))) if len(tab) else 0.0
    lim = max(float(probe_axis_max or 0.0), 1.08 * np.nan_to_num(data_max), 0.2)
    floor = float(response_floor_dex)
    fig = go.Figure()
    for xs, ys, col in (([floor, lim, lim, floor], [0, 0, lim, floor], "rgba(22,163,74,0.08)"),
                        ([0, lim, 0], [0, lim, lim], "rgba(147,51,234,0.08)"),
                        ([0, floor, floor, 0], [0, 0, lim, lim], "rgba(100,116,139,0.12)")):
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=col, mode="none",
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", hoverinfo="skip",
                             line=dict(color=t["muted"], dash="dash"), showlegend=False))
    fig.add_vline(x=floor, line=dict(color=t["muted"], dash="dot"))
    size = 10.0 + 26.0 * np.nan_to_num(err, nan=0.0) / max(np.nanmax(err) if
                                                         np.isfinite(err).any() else 1.0, 1e-9)
    palette = sample_colorscale("Turbo", list(np.linspace(0.05, 0.95, max(len(tab), 2))))
    for i, (_, row) in enumerate(tab.iterrows()):
        mark = ">" if row["frac_censored"] > 0.0 else ""
        name = pretty_key(row["species"], fmt)
        fig.add_trace(go.Scatter(
            x=[row["abs_response_dex"]], y=[row["swing_dex"]], mode="markers",
            name=f"{name}  ({mark}{row['error_dex']:.2f})",
            marker=dict(size=size[i], color=palette[i], line=dict(color="black", width=1.2)),
            hovertemplate=(f"{name}<br>|R| = %{{x:.3f}} dex<br>swing = %{{y:.3f}} dex"
                           f"<br>margin = {row['probe_margin']:+.3f} dex"
                           f"<br>|Δ| = {mark}{row['error_dex']:.3f} dex<extra></extra>")))
    for txt, ax, ay, col in (("CRIR probe<br>response beats its own spread", 0.62, 0.40,
                              "#15803d"),
                             ("FUV tracer<br>spread beats the response", 0.38, 0.62, "#7e22ce")):
        fig.add_annotation(x=ax * lim, y=ay * lim, text=f"<i>{txt}</i>", showarrow=False,
                           font=dict(size=11, color=col), bgcolor="rgba(255,255,255,0.55)")
    fig.add_annotation(x=0.5 * floor, y=0.5 * lim, text="<i>no response</i>", textangle=-90,
                       showarrow=False, font=dict(size=10, color=t["muted"]))
    fig.update_xaxes(range=[0, lim], constrain="domain",
                     title_text="|R| [dex]  (does attenuation change it?)")
    fig.update_yaxes(range=[0, lim], scaleanchor="x", scaleratio=1, constrain="domain",
                     title_text=f"{swing_label} [dex]  (does the environment matter?)")
    # Legend on the right, one species per row (plotly widens the right margin
    # to fit it).  Grow the figure with the species count so the legend never
    # scrolls or clips.
    n_items = sum(tr.showlegend is not False for tr in fig.data)
    fig = _style(fig, t, "Step 2 · Probe quality — marker size = |Δ|",
                 max(640, 190 + 26 * n_items))
    fig.update_layout(legend=dict(title=dict(text="species  (|Δ| [dex])", side="top"),
                                  orientation="v", x=1.02, xanchor="left", y=1.0,
                                  yanchor="top", maxheight=4000,
                                  font=ps.legend_font(t["font"]) | dict(size=15),
                                  title_font=dict(size=15)))
    return fig


def fig_verdict_matrix(long_table, *, col_key, col_order, col_labels=None, species=None,
                       theme=None, fmt=None, title="Verdicts"):
    """Step 3: species × (density | environment) grid coloured by verdict."""
    t = pr._theme_fallback(theme)
    if long_table is None or not len(long_table):
        return empty_fig("No verdicts to show.", theme)
    species = list(species if species is not None else dict.fromkeys(long_table["species"]))
    if not species:
        return empty_fig("No verdicts to show.", theme)
    col_labels = list(col_labels or [str(c) for c in col_order])
    piv = long_table.drop_duplicates(["species", col_key]).set_index(["species", col_key])
    k = len(VERDICT_ORDER)
    z = np.full((len(species), len(col_order)), np.nan)
    text = np.full(z.shape, "", dtype=object)
    hover = np.full(z.shape, "", dtype=object)
    for i, sp in enumerate(species):
        for j, c in enumerate(col_order):
            if (sp, c) not in piv.index:
                continue
            row = piv.loc[(sp, c)]
            v = str(row["verdict"])
            z[i, j] = VERDICT_ORDER.index(v) if v in VERDICT_ORDER else k - 1
            text[i, j] = VERDICT_SHORT.get(v, v)
            hover[i, j] = (f"{pretty_key(sp, fmt)} · {col_labels[j]}<br><b>{v}</b>"
                           f"<br>|Δ| = {pr.format_score(row.get('error_dex'))} dex"
                           f"<br>margin = {pr.format_score(row.get('probe_margin'))} dex"
                           f"<br>|R| = {pr.format_score(row.get('abs_response_dex'))} dex"
                           f"<br>cells = {pr.format_score(row.get('n_cells'), 0)}")
    scale = []
    for idx, v in enumerate(VERDICT_ORDER):
        scale += [[idx / k, VERDICT_COLORS[v]], [(idx + 1) / k, VERDICT_COLORS[v]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=col_labels, y=[pretty_key(s, fmt) for s in species], zmin=-0.5, zmax=k - 0.5,
        colorscale=scale, text=text, texttemplate="%{text}", customdata=hover,
        hovertemplate="%{customdata}<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(tickvals=list(range(k)), ticktext=VERDICT_ORDER,
                      tickfont=ps.cbar_tick_font(t["font"]))))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", tickangle=-25, showgrid=False)
    return _style(fig, t, title, max(320, 120 + 30 * len(species)))
