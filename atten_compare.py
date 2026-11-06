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

Every cell with a model counts for every species: unmatched cells and cells
below the detection limit enter at Δ = 0 (and R = 0); no species is excluded and
no ``|S|`` gate is applied.  Thresholds are labels on the numbers, not cuts.

3-D (``kosens.grid.grid_3d``): every ζ-row sits at a fixed (n_H, χ).  Pixels
carry the density, so the same reducers run on one density slice, on the pooled
cube, or per (n_H, χ, ζ) box; the swing is split into its χ part
(tested by the verdict) and its n_H part (reported).  How to read every number:
``docs/attenuation_workflow.md``.
"""

from __future__ import annotations

import functools
import itertools
import re

import numpy as np
import pandas as pd
from scipy.stats import linregress

import grid_naming as gn

X_SHIFT_MATCH_RTOL = 0.02
OBS_INTENSITY_LIMIT = 0.1
# Rare-isotope markers as they appear in KOSMA-τ species names (13CO, C18O, H13CN...).
ISOTOPE_MARKERS = ("13C", "18O", "17O", "15N", "34S", "33S")

VERDICT_ORDER = ["robust attenuation probe", "robust, small bias", "G0-modulated", "no response", "no data"]
CR_FAMILY = {"robust attenuation probe", "robust, small bias"}


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
    """First match in ``xs_scan`` order, with grid nodes and segments as one candidate set.

    Each node is tested, then the segment that follows it, so the match nearest the
    start of the scan wins.  Testing all nodes before any segment let a node on the far
    branch of a non-monotonic row (one that happens to sit inside the tolerance band)
    beat a true crossing between two nodes close by.
    """
    if xs_scan.size == 0:
        return np.nan
    for k, (xi, zi) in enumerate(zip(xs_scan, zs_scan)):
        if np.isfinite(zi) and abs(zi - i_target) <= tol:
            return float(xi)
        if k + 1 < len(xs_scan):
            xb = _interpolate_segment_match(xi, xs_scan[k + 1], zi, zs_scan[k + 1], i_target, tol)
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

SLOPE_BASELINE_DEX = 1.0  # model node spacing (KoSens GRf.SLOPE_BASELINE_DEX)


def _log_intensity_slope(log_intensity, log_x, baseline_dex=SLOPE_BASELINE_DEX):
    """``dlog10(I)/dlog10(x)`` along each row over a fixed ``baseline_dex`` window.

    ``S(x) = [log I(x + h) - log I(x - h)] / 2h`` with ``h = baseline_dex / 2``, ``log I``
    linear between the row's finite points and the window clipped to the row's range
    (one-sided at the edges). A neighbour-cell difference on an interpolated grid is
    constant inside each model cell and jumps at every node (a staircase in S and D);
    on raw nodes 1 dex apart this is the centred difference.
    """
    li = np.asarray(log_intensity, dtype=float)
    lx = np.asarray(log_x, dtype=float)
    out = np.full(li.shape, np.nan)
    if li.ndim != 2 or li.shape != lx.shape or li.shape[1] < 2:
        return out
    h = 0.5 * baseline_dex
    for i in range(li.shape[0]):
        ok = np.isfinite(li[i]) & np.isfinite(lx[i])
        if ok.sum() < 2:
            continue
        order = np.argsort(lx[i, ok])
        xs, ys = lx[i, ok][order], li[i, ok][order]
        x = lx[i, ok]
        lo, hi = np.clip(x - h, xs[0], None), np.clip(x + h, None, xs[-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            sl = (np.interp(hi, xs, ys) - np.interp(lo, xs, ys)) / (hi - lo)
        out[i, ok] = np.where(hi > lo, sl, np.nan)
    return out


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


# KOSMA-τ names that CDMS/JPL spell differently (compared upper-case).
KOSMA_CATALOG_ALIASES = {"C2H": "CCH", "O2H": "HO2"}
# Observable without a CDMS/JPL entry (e.g. H3+ has no rotational spectrum: IR absorption).
KOSMA_CATALOG_EXTRA = {"H3+", "HE", "HE+", "AR+", "ARH+"}


@functools.lru_cache(maxsize=1)
def catalog_species():
    """Species with laboratory spectroscopy: ``{UPPERCASE name: 'CDMS' | 'JPL' | 'CDMS+JPL'}``.

    Read from the CDMS and JPL species tables bundled with astroquery (no
    network).  CDMS names carry state info after a comma and sometimes a
    structural prefix (``c-``, ``l-``); only the bare formula is kept.
    """
    from astroquery.linelists.cdms import CDMS
    try:
        from astroquery.linelists.jplspec import JPLSpec
    except ImportError:
        from astroquery.jplspec import JPLSpec

    cdms = (re.sub(r"^[a-z]{1,3}-", "", str(n).split(",")[0].strip())
            for n in CDMS.get_species_table()["Name"])
    jpl = (str(n).strip() for n in JPLSpec.get_species_table()["NAME"])
    found = {}
    for cat, names in (("CDMS", cdms), ("JPL", jpl)):
        for name in names:
            key = name.upper()
            found[key] = cat if found.get(key, cat) == cat else "CDMS+JPL"
    return found


def drop_uncatalogued(pixels, known=None):
    """KoSens ``require_catalog``: drop species/ratios with no CDMS or JPL entry.

    A ratio is kept only if both sides are catalogued; each side is reduced to
    its base name.  Ice keys are left for ``exclude_ice``.  Returns
    ``(pixels, no_catalog)``; ``no_catalog`` lists each dropped key and the
    constituent(s) missing from both catalogues.
    """
    known = catalog_species() if known is None else known
    rows = []
    for key in pixels["species"].unique():
        if _is_ice_species_key(key):
            continue
        bases = [p.split("(")[0].strip() for p in str(key).split("/")]
        missing = [b for b in bases
                   if KOSMA_CATALOG_ALIASES.get(b.upper(), b.upper()) not in known
                   and b.upper() not in KOSMA_CATALOG_EXTRA]
        if missing:
            rows.append({"species": key, "missing": ", ".join(missing)})
    no_catalog = pd.DataFrame(rows, columns=["species", "missing"])
    return pixels[~pixels["species"].isin(no_catalog["species"])], no_catalog


def error_cells(shift, g_const, g_atten, x_mesh, obs=None):
    """|Δ| map on the cell set of the rankings (KoSens ``_error_cells``).

    A cell with a model in both grids counts as 0 when it has no match or lies
    below the detection limit (``obs`` False; None = no limit).  Only cells
    with no model are NaN.
    """
    defined = (np.isfinite(g_const) & np.isfinite(g_atten)
               & np.isfinite(x_mesh) & (x_mesh > 0.0))
    d = np.where(defined, np.abs(np.nan_to_num(shift, nan=0.0)), np.nan)
    if obs is not None:
        d = np.where(obs, d, np.where(defined, 0.0, np.nan))
    return d


def row_median_delta(shift, g_const, g_atten, x_mesh, obs=None):
    """Per-row median |Δ| over :func:`error_cells` (KoSens ``overplot_delta_median``).

    Returns ``(d_row, ok)``: one value per row with a model, and the row mask.
    """
    d = error_cells(shift, g_const, g_atten, x_mesh, obs)
    ok = np.isfinite(d).any(axis=1)
    return (np.nanmedian(d[ok], axis=1) if ok.any() else np.array([])), ok


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


def _full_edges(values, interior):
    """Interior edges closed by the data's own range, for ``pandas.cut``."""
    fin = np.asarray(values, dtype=float)
    fin = fin[np.isfinite(fin) & (fin > 0.0)]
    if fin.size == 0:
        return None
    lo, hi = float(np.min(fin)), float(np.max(fin))
    inner = [float(e) for e in sorted(interior) if lo < float(e) < hi]
    return np.array([lo * (1.0 - 1e-12), *inner, hi * (1.0 + 1e-12)], dtype=float)


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
    value_ref, value_atten, observable, abs_shift_dex.  ``shift_dex`` is 0 where
    both grids have a model but no match was found (read as "no shift") and NaN
    only where there is no model.  ``slope_dex`` is a diagnostic, never a filter.
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
        # KoSens convention: an unmatched cell with a model in both grids is "no shift".
        defined = np.isfinite(g0) & np.isfinite(g1) & np.isfinite(xm) & (xm > 0.0)
        shift = np.where(defined & ~np.isfinite(shift), 0.0, shift)
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
                   obs_limit=None, f_att_cube=None, nh_cube=None, progress=None):
    """Pixel table over one or more density slices of two cube dicts.

    Adds ``n`` (density), ``i_n``, ``row_id`` (one fixed-(n, χ) ζ-row),
    and ``response_dex`` = R.  Shifts are matched on the **full** row here, so
    later subsetting by environment is not biased by where a bin stops.
    ``f_att_cube`` (a cube entry of the attenuated
    models' CR-attenuated mass fraction) adds an ``f_att`` column; ``nh_cube``
    (clump column N_H, cm⁻²) adds ``log_N_H``.
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
        f_plane = None if f_att_cube is None else cube_plane(f_att_cube, k)["grid"]
        nh_plane = None if nh_cube is None else cube_plane(nh_cube, k)["grid"]
        for s in species:
            df = collect_attenuation_x_shift_table(
                ref2, att2, species_list=[s], x_shift_match_rtol=match_rtol,
                x_shift_scan_direction=scan_direction, obs_limit=obs_limit)
            if len(df):
                df["n"] = float(densities[k])
                df["i_n"] = int(k)
                if f_plane is not None:
                    df["f_att"] = f_plane[df["iy"].to_numpy(), df["ix"].to_numpy()]
                if nh_plane is not None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        df["log_N_H"] = np.log10(
                            nh_plane[df["iy"].to_numpy(), df["ix"].to_numpy()])
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


def _zero_unobservable_shifts(pixels):
    """``shift_dex`` / ``abs_shift_dex`` = 0 in cells below the detection limit (KoSens).

    An undetectable line shows no shift, so such a cell counts as "no shift" -- the
    same rule as an unmatched cell.  Every species is then summarised over the same
    cells.  Cells with no model (NaN shift) stay NaN.
    """
    hide = ~pixels["observable"].astype(bool) & np.isfinite(pixels["shift_dex"])
    if not hide.any():
        return pixels
    pixels = pixels.copy()
    pixels.loc[hide, ["shift_dex", "abs_shift_dex"]] = 0.0
    return pixels


def shift_summary(pixels, *, shift_abs_edges=(0.1, 0.3, 0.5, 1.0), exclude_ice=True,
                  exclude_isotopologues=False, n_x_bins=4, n_y_bins=4, n_top_models=8,
                  n_top_summary_species=12, max_per_constituent=2, regime_edges=None):
    """Numerics of KoSens ``plot_attenuation_shift_summary``.

    Every cell with a model enters every count, median, regime table and
    top-model list; unmatched and unobservable cells enter at Δ = 0.  No ``|S|``
    gate and no species cut: species are sorted by ``median_abs_shift_dex``, one
    number computed identically for all of them.  ``frac_observable`` is reported,
    not a cut.  With more than one density, regimes and winners are also split by ``n``.
    ``regime_edges`` (``'n'``, ``'y'``, ``'x'`` to interior edges, as ``parse_axis_edges``)
    sets the regimes: an axis named there replaces ``n_x_bins`` / ``n_y_bins``, and
    ``'n'`` groups the densities into ``n_bin`` instead of one regime per density.
    ``n_top_summary_species=None`` puts every species in ``top_species``.
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
                     f"cell and rank at |Δ| = 0: {', '.join(map(str, never_obs[:12]))}"
                     f"{'' if len(never_obs) <= 12 else ', …'}.")
    pixel_df = _zero_unobservable_shifts(pixels)

    valid = pixel_df.dropna(subset=["shift_dex"]).copy()
    if len(valid) == 0:
        notes.append("No finite x-shifts were found for any species.")
        return _empty_summary(pixels_out, notes)

    bin_edges, bin_labels = _abs_shift_bin_edges_and_labels(shift_abs_edges)
    valid["shift_bin"] = pd.cut(valid["abs_shift_dex"], bins=bin_edges, labels=bin_labels,
                                include_lowest=True, right=False)

    n_cells = pixel_df.groupby("species").size().rename("n_cells")
    n_valid = valid.groupby("species").size().rename("n_valid")
    n_valid = n_valid.reindex(n_cells.index).fillna(0).astype(int)
    stats = valid.groupby("species")["abs_shift_dex"].agg(
        mean_abs_shift_dex="mean", median_abs_shift_dex="median",
        std_abs_shift_dex="std", max_abs_shift_dex="max")
    bin_counts = (valid.groupby(["species", "shift_bin"], observed=False).size()
                  .unstack(fill_value=0))
    for lab in bin_labels:
        if lab not in bin_counts.columns:
            bin_counts[lab] = 0
    bin_counts = bin_counts.reindex(n_cells.index).fillna(0)[bin_labels]

    summary = (n_cells.to_frame().join(n_valid, how="left").join(stats, how="left")
               .join(bin_counts, how="left").reset_index())
    for col in ("n_valid", "n_cells"):
        summary[col] = summary[col].fillna(0).astype(int)
    summary["label"] = summary["species"].astype(str)
    summary["frac_observable"] = summary["species"].map(frac_obs).astype(float)
    summary = summary.sort_values("median_abs_shift_dex", ascending=False,
                                  na_position="last").reset_index(drop=True)

    top_n = max(int(n_top_models), 1)
    top_cols = ["species", "n", "x", "y", "shift_dex", "abs_shift_dex", "slope_dex",
                "value_ref", "value_atten"]
    top_parts = [g.nlargest(top_n, "abs_shift_dex") for _, g in valid.groupby("species")]
    top_models = (pd.concat(top_parts, ignore_index=True)[top_cols] if top_parts
                  else pd.DataFrame(columns=top_cols))
    top_overall = (valid.nlargest(min(20, len(valid)), "abs_shift_dex")[top_cols]
                   .reset_index(drop=True))

    multi_n = pixel_df["i_n"].nunique() > 1
    regime_edges = regime_edges or {}
    n_key = "n_bin" if "n" in regime_edges else "n"
    regime_keys = ([n_key] if multi_n else []) + ["x_bin", "y_bin"]

    def _edges(col, n_bins):
        if col in regime_edges:
            return _full_edges(valid[col], regime_edges[col])
        return _positive_logspace_edges(valid[col], n_bins)
    x_edges, y_edges = _edges("x", n_x_bins), _edges("y", n_y_bins)
    regime_median, winners = pd.DataFrame(), pd.DataFrame()
    x_labels = y_labels = None
    heat_x = heat_y = heat_n = None
    species_order = summary["species"].tolist()
    if x_edges is not None and y_edges is not None:
        x_labels, y_labels = _log_bin_labels(x_edges), _log_bin_labels(y_edges)
        valid["x_bin"] = pd.cut(valid["x"], bins=x_edges, labels=x_labels, include_lowest=True)
        valid["y_bin"] = pd.cut(valid["y"], bins=y_edges, labels=y_labels, include_lowest=True)
        if multi_n and n_key == "n_bin":
            n_edges = _full_edges(valid["n"], regime_edges["n"])
            valid["n_bin"] = pd.cut(valid["n"], bins=n_edges, labels=_log_bin_labels(n_edges),
                                    include_lowest=True)
        regime_median = (valid.groupby(["species", *regime_keys], observed=True)
                         .agg(median_abs_shift_dex=("abs_shift_dex", "median"),
                              n_models=("abs_shift_dex", "count"))
                         .reset_index())
        if len(regime_median) > 0:
            peak = regime_median.groupby(regime_keys, observed=True)[
                "median_abs_shift_dex"].idxmax()
            winners = regime_median.loc[peak].reset_index(drop=True)
            winners["label"] = winners["species"].astype(str)

        def _heat(key):
            return (valid.groupby(["species", key], observed=True)["abs_shift_dex"]
                    .median().unstack().reindex(species_order))
        heat_x, heat_y = _heat("x_bin"), _heat("y_bin")
        if multi_n:
            heat_n = _heat(n_key)

    if n_top_summary_species is None:  # every species is drawn: no count and no constituent cap
        top_tab = summary
    else:
        top_n_sum = min(max(int(n_top_summary_species), 1), max(len(species_order), 1))
        top_tab, held_top = _cap_per_constituent(summary, top_n_sum, max_per_constituent)
        note = _capped_note(held_top, max_per_constituent)
        if note:
            notes.append(note)

    cell_totals = summary["n_cells"].to_numpy(dtype=float)
    n_models_all = (float(cell_totals[0]) if len(cell_totals)
                    and np.all(cell_totals == cell_totals[0]) else np.nan)

    leading = ["species", "label", "frac_observable", "n_cells", "n_valid",
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
    "species", "label", "error_dex", "error_p10", "error_p90", "n_error_cells",
    "response_dex", "abs_response_dex", "fuv_swing_dex", "g0_swing_dex", "n_swing_dex",
    "fuv_trend", "n_fuv_rows", "frac_observable",
]


def _swing(values):
    """Spread (max − min) of the finite ``values``.

    The grid is a complete enumeration of environments, not a noisy sample, so
    nothing is trimmed: "the same response everywhere" must hold for every row.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if v.size else np.nan


def _split_swing(r_row, g0_row, n_row=None):
    """Swing of the per-row R split into its χ and density parts (KoSens).

    ``g0_swing`` is the spread across χ at fixed density, ``n_swing`` the spread
    across density at fixed χ; each is the largest such spread over the grid.
    Without a density id (2-D tables) ``g0_swing`` is the plain swing and
    ``n_swing`` is NaN.
    """
    r_row = np.asarray(r_row, dtype=float)
    if n_row is None:
        return _swing(r_row), np.nan
    g_key = np.round(np.log10(np.asarray(g0_row, dtype=float)), 6)

    def _max_group_range(keys):
        out = [_swing(r_row[keys == k]) for k in np.unique(keys) if np.sum(keys == k) >= 2]
        out = [v for v in out if np.isfinite(v)]
        return max(out) if out else np.nan

    return _max_group_range(np.asarray(n_row)), _max_group_range(g_key)


def response_table(pixels, *, max_species=8, min_abs_response_dex=0.0,
                   max_per_constituent=2, exclude_ice=True, exclude_isotopologues=False):
    """Numerics of KoSens ``plot_attenuation_fuv_response``.

    ``error_dex`` is the median |Δ| over every cell with a model (no ``|S|``
    gate; unmatched and unobservable cells at 0), so it equals the summary's
    ``median_abs_shift_dex``.  Unobservable cells also enter R at 0.  Each row
    is one CR experiment at fixed (n_H, χ) and collapses to its median R.
    ``fuv_swing_dex`` is max − min of those row medians; in 3-D it is split into
    ``g0_swing_dex`` (across χ at fixed n_H) and ``n_swing_dex`` (across n_H at
    fixed χ).  ``cr_dominated = g0_swing_dex < abs_response_dex``: only the χ
    part is tested, because the density part is largely the attenuation itself
    (more column, more shielded mass) and is reported, not penalised.

    Returns ``{'table', 'notes', 'profiles'}``; ``profiles[species]`` is
    ``(χ, median R, median Δ)`` per χ (rows pooled over densities), the curve
    the swing summarises.  The table is sorted by ``error_dex`` and carries
    ``ranked`` (left panel) and ``probe_ranked`` (right panel): display only.
    ``max_species=None`` flags every species that can be drawn.
    """
    notes = []
    empty = pd.DataFrame(columns=[*RESPONSE_COLUMNS, "cr_dominated", "den_contrib",
                                  "ranked", "probe_ranked"])
    if pixels is None or len(pixels) == 0:
        return {"table": empty, "notes": notes, "profiles": {}}
    pixels = _ensure_cells(pixels)
    if exclude_isotopologues:
        pixels = _drop_keys(pixels, _is_isotopologue_key, "isotopologue", notes)
    if exclude_ice:
        pixels = _drop_keys(pixels, _is_ice_species_key, "ice", notes)
    if len(pixels) == 0:
        return {"table": empty, "notes": notes, "profiles": {}}
    pixels = _zero_unobservable_shifts(pixels)
    # Hand-made / single-plane tables carry no density: the swing is not split.
    has_density = bool(np.isfinite(pixels["n"].to_numpy(dtype=float)).any())

    records, profiles = [], {}
    for species, sub in pixels.groupby("species"):
        obs = sub["observable"].to_numpy(dtype=bool)
        raw = sub["response_dex"].to_numpy(dtype=float)
        # An undetectable line shows no change: R = 0 where there is a model.
        response = np.where(obs, raw, np.where(np.isfinite(raw), 0.0, np.nan))
        err = sub["abs_shift_dex"].to_numpy(dtype=float)
        err_vals = err[np.isfinite(err)]
        n_err = int(err_vals.size)

        rows = pd.DataFrame({"row": sub["row_id"].to_numpy(), "i_n": sub["i_n"].to_numpy(),
                             "r": response, "d": err,
                             "y": sub["y"].to_numpy(dtype=float)})
        per_row = rows.groupby("row").agg(r=("r", "median"), y=("y", "median"),
                                          d=("d", "median"), i_n=("i_n", "first"))
        keep = np.isfinite(per_row["r"]) & np.isfinite(per_row["y"]) & (per_row["y"] > 0.0)
        per_row = per_row[keep]
        n_rows = len(per_row)
        if n_rows == 0:
            continue
        r_keep = per_row["r"].to_numpy(dtype=float)
        y_keep = per_row["y"].to_numpy(dtype=float)
        swing = _swing(r_keep) if n_rows >= 2 else np.nan
        g0_swing, n_swing = _split_swing(
            r_keep, y_keep, per_row["i_n"].to_numpy() if has_density else None)
        prof = per_row.groupby("y")[["r", "d"]].median().sort_index()
        profiles[species] = (prof.index.to_numpy(float), prof["r"].to_numpy(float),
                             prof["d"].to_numpy(float))
        trend = (float(linregress(np.log10(y_keep), r_keep).slope)
                 if n_rows >= 3 and np.ptp(y_keep) > 0.0 else np.nan)
        records.append({
            "species": species, "label": str(species),
            "error_dex": float(np.median(err_vals)) if n_err else np.nan,
            "error_p10": float(np.percentile(err_vals, 10)) if n_err else np.nan,
            "error_p90": float(np.percentile(err_vals, 90)) if n_err else np.nan,
            "n_error_cells": n_err,
            "response_dex": float(np.median(r_keep)),
            "abs_response_dex": float(np.abs(np.median(r_keep))),
            "fuv_swing_dex": swing, "g0_swing_dex": g0_swing, "n_swing_dex": n_swing,
            "fuv_trend": trend, "n_fuv_rows": n_rows,
            "frac_observable": float(obs.mean()),
        })
    if not records:
        notes.append("No species had a usable intensity response.")
        return {"table": empty, "notes": notes, "profiles": {}}

    table = pd.DataFrame.from_records(records)[RESPONSE_COLUMNS]
    # NaN swing compares False: never called CR-dominated.
    table["cr_dominated"] = table["g0_swing_dex"] < table["abs_response_dex"]
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

    # A species with no model cell has no error: listed, never drawn. Nothing else is held.
    gated = table[(table["abs_response_dex"] >= float(min_abs_response_dex))
                  & np.isfinite(table["error_dex"])]
    n_ungated = int((~np.isfinite(table["error_dex"])).sum())
    if n_ungated:
        notes.append(f"{n_ungated} species have no cell with a model; their error is "
                     "undefined (listed only).")

    probe_gated = gated.dropna(subset=["g0_swing_dex"]).sort_values("abs_response_dex",
                                                                    ascending=False)
    if max_species is None:  # every species is drawn: no count and no constituent cap
        plot_tab, probe_tab = gated, probe_gated
    else:
        n_draw = max(int(max_species), 1)
        plot_tab, held_error = _cap_per_constituent(gated, n_draw, max_per_constituent)
        probe_tab, held_probe = _cap_per_constituent(probe_gated, n_draw, max_per_constituent)
        for held in (held_error, held_probe):
            note = _capped_note(held, max_per_constituent)
            if note:
                notes.append(note)
    table["ranked"] = table["species"].isin(set(plot_tab["species"]))
    table["probe_ranked"] = table["species"].isin(set(probe_tab["species"]))
    return {"table": table, "notes": notes, "profiles": profiles}


# ---------------------------------------------------------------------------
# Step 3 — one verdict per species
# ---------------------------------------------------------------------------

# The paper's question -- robust attenuation probe, or degenerate with the FUV field? -- as one label
# per cascade verdict (KoSens ``_PROBE_ANSWER``); the verdict stays next to it for traceability.
PROBE_ANSWER = {
    "robust attenuation probe": "robust attenuation probe", "robust, small bias": "robust, weak",
    "G0-modulated": "G0-modulated", "no response": "unusable", "no data": "unusable",
}
ANSWER_ORDER = ["robust attenuation probe", "robust, weak", "G0-modulated", "unusable"]

RECONCILE_COLUMNS = [
    "species", "label", "cr_led", "matters", "verdict", "answer", "frac_observable",
    "median_abs_shift_dex", "error_dex", "error_p90", "response_dex", "abs_response_dex",
    "fuv_swing_dex", "g0_swing_dex", "n_swing_dex", "fuv_trend", "cr_dominated", "n_cells",
    "n_fuv_rows",
]


def reconcile(summary, response, *, response_floor_dex=0.1, min_error_dex=0.3):
    """KoSens ``reconcile_attenuation_species``: three questions, first failure names the verdict.

    responds → |R| ≥ response_floor_dex; cr_led → g0_swing_dex < |R| (CR- rather
    tha G0-modulated response); matters → error_dex ≥ min_error_dex.  These are
    labels on numbers computed identically for every species, not cuts: no
    species is excluded.  Returns ``table`` (cascade order), ``problematic``
    (every species with an error, by error_dex: correction priority) and
    ``tracers`` (CR-led, by |R|: line choice).
    """
    if isinstance(summary, dict):
        summary = summary["species_summary"]
    if isinstance(response, dict):
        response = response["table"]
    if summary is None or response is None or not len(summary) or not len(response):
        empty = pd.DataFrame(columns=RECONCILE_COLUMNS)
        return {"table": empty, "problematic": empty.copy(), "tracers": empty.copy()}

    merged = summary[["species", "median_abs_shift_dex", "n_cells"]].merge(
        response, on="species", how="outer")
    merged["label"] = merged["label"].fillna(merged["species"].astype(str))
    resp = merged["abs_response_dex"].astype(float)
    swing = merged["g0_swing_dex"].astype(float)      # the χ part only; = fuv_swing_dex in 2-D
    error = merged["error_dex"].astype(float)
    merged["cr_led"] = swing < resp
    merged["matters"] = error >= float(min_error_dex)

    has_data = (np.isfinite(resp) & np.isfinite(swing)
                & np.isfinite(merged["median_abs_shift_dex"].astype(float)))
    merged["verdict"] = np.select(
        [~has_data, resp < float(response_floor_dex), ~merged["cr_led"], merged["matters"]],
        ["no data", "no response", "G0-modulated", "robust attenuation probe"],
        default="robust, small bias")
    merged["_rank"] = merged["verdict"].map({v: i for i, v in enumerate(VERDICT_ORDER)})
    merged = merged.sort_values(["_rank", "error_dex"], ascending=[True, False],
                                na_position="last").reset_index(drop=True)
    merged["answer"] = merged["verdict"].map(PROBE_ANSWER)
    table = merged[RECONCILE_COLUMNS]

    problematic = table[np.isfinite(table["error_dex"].astype(float))].sort_values(
        "error_dex", ascending=False).reset_index(drop=True)
    tracers = table[table["verdict"].isin(CR_FAMILY)].sort_values(
        "abs_response_dex", ascending=False).reset_index(drop=True)
    return {"table": table, "problematic": problematic, "tracers": tracers}


# ---------------------------------------------------------------------------
# 3-D: verdicts per environment
# ---------------------------------------------------------------------------

def _run_trio(pixels, summary_kw, response_kw, reconcile_kw):
    s = shift_summary(pixels, **summary_kw)
    r = response_table(pixels, **response_kw)
    return s, r, reconcile(s, r, **reconcile_kw)


def parse_axis_edges(text, values):
    """Interior bin edges for one axis.

    An integer (default ``3``, as KoSens) cuts the axis into that many
    equal-log-width bins between its own min and max; ``full``/``none``/``1``
    keeps one bin; otherwise comma-separated edge values.
    """
    raw = str(text if text not in (None, "") else 3).strip().lower()
    if raw in ("full", "none", "off"):
        return []
    if raw == "auto":
        raw = "3"
    if raw.isdigit():
        edges = _positive_logspace_edges(np.unique(np.asarray(values, dtype=float)), int(raw))
        return [] if edges is None else [float(e) for e in edges[1:-1]]
    return sorted(float(p) for p in raw.replace(";", ",").split(",") if p.strip())


def _bin_names(n_bins):
    return {1: [""], 2: ["low", "high"], 3: ["low", "mid", "high"]}.get(
        n_bins, [f"bin {i + 1}" for i in range(n_bins)])


# (pixel column, display symbol, key of the ``<key>_level`` column)
ENV_AXES = (("n", "n_H", "n"), ("y", "χ", "chi"), ("x", "ζ", "zeta"))
LEVEL_COLS = [f"{key}_level" for _c, _s, key in ENV_AXES]
_LEVEL_RANK = {"low": 0, "mid": 1, "high": 2}


def assign_environments(pixels, edges):
    """Label every cell with its (n_H, χ, ζ) environment.

    ``edges`` maps ``'n'``, ``'y'`` (χ), ``'x'`` (ζ) to interior edges.  A cell
    on an edge goes to the lower bin (bins are right-closed, as ``pandas.cut``
    in KoSens).  Returns ``(labels, env_info)``; ``env_info`` lists every
    environment in axis order with its ranges and its level on each axis.
    """
    parts, split = [], []
    for col, sym, key in ENV_AXES:
        e = sorted(float(v) for v in (edges.get(col) or []))
        if not e:
            continue
        names = _bin_names(len(e) + 1)
        idx = np.searchsorted(np.asarray(e), pixels[col].to_numpy(dtype=float), side="left")
        parts.append((sym, np.asarray(names, dtype=object)[idx]))
        split.append((sym, names, [-np.inf, *e, np.inf], key))
    if parts:
        labels = [" · ".join(f"{names[i]} {sym}" for sym, names in parts)
                  for i in range(len(pixels))]
    else:
        labels = ["whole grid"] * len(pixels)
    env_info = []
    for combo in itertools.product(*[range(len(a[1])) for a in split]):
        label = " · ".join(f"{a[1][i]} {a[0]}" for a, i in zip(split, combo)) or "whole grid"
        ranges = {a[0]: (a[2][i], a[2][i + 1]) for a, i in zip(split, combo)}
        levels = {f"{a[3]}_level": a[1][i] for a, i in zip(split, combo)}
        env_info.append({"label": label, "ranges": ranges, "levels": levels})
    return np.asarray(labels, dtype=object), env_info


def env_col_labels(env_info, labels):
    """Matrix column labels: the environment name over the ranges it actually covers.

    ``low`` / ``mid`` / ``high`` are cuts of this grid between its own min and max,
    not physical thresholds, so the numbers have to travel with the name or the
    column reads as a claim about the ISM rather than about the models loaded.
    """
    def one(sym, lo, hi):
        # KoSens prints log10 edges with two decimals.
        if not np.isfinite(lo):
            return f"log {sym} ≤ {np.log10(hi):.2f}"
        if not np.isfinite(hi):
            return f"log {sym} > {np.log10(lo):.2f}"
        return f"log {sym} {np.log10(lo):.2f}–{np.log10(hi):.2f}"

    by = {e["label"]: (e.get("ranges") or {}) for e in env_info}
    return [lab + "".join(f"<br>{one(sym, *rng)}" for sym, rng in by.get(lab, {}).items())
            for lab in labels]


def environment_verdicts(pixels, edges, summary_kw, response_kw, reconcile_kw, *,
                         progress=None):
    """Verdict per (species, environment box), plus per-environment views.

    KoSens ``attenuation_regime_verdicts_3d``: the unchanged chain runs on the
    cells of each box, so a verdict means what it means on the whole grid but
    is local: *"at low n_H and high ζ this line is a clean probe"*.  Shifts were
    matched on the full rows (``collect_pixels``).  Every species is graded in
    every box with the same rule; a box where a species has no usable row is
    ``no data``.  The table carries one ``*_level`` column per split axis.
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
        t = v["table"].assign(environment=env["label"], **env["levels"])
        if "f_att" in sub.columns:
            # One value per box (all species share it): median over its models.
            env["f_att"] = float(sub.drop_duplicates(["i_n", "iy", "ix"])["f_att"].median())
            t["f_att"] = env["f_att"]
        if "log_N_H" in sub.columns:
            env["log_N_H"] = float(
                sub.drop_duplicates(["i_n", "iy", "ix"])["log_N_H"].median())
            t["log_N_H"] = env["log_N_H"]
        frames.append(t)
        views[env["label"]] = {
            "tracers": t[t["verdict"].isin(CR_FAMILY)].sort_values(
                "abs_response_dex", ascending=False).reset_index(drop=True),
            "problematic": t[np.isfinite(t["error_dex"].astype(float))].sort_values(
                "error_dex", ascending=False).reset_index(drop=True),
        }
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {"table": table, "environments": env_info, "views": views}


def regime_tally(env_table):
    """KoSens ``attenuation_verdict_frame_3d``: one compact row per species.

    The environment matrix answers "where is this a probe?" by eye; this answers
    "in how many boxes?".  ``boxes_probe`` / ``boxes_cr_led`` / ``boxes_fuv``
    count the ``robust attenuation probe`` / any-CR-led / ``G0-modulated`` boxes out of
    ``boxes_graded``, and ``best_box`` is the species' best environment — best
    verdict, then largest |R| — with that box's |R|, χ swing (the one the
    verdict tests), n_H swing (reported) and ζ error.  Nothing is recomputed.

    A species with ``boxes_probe`` well below ``boxes_graded`` is a probe only in
    the conditions named by ``best_box``, not one you can quote for the grid.
    """
    if env_table is None or not len(env_table):
        return pd.DataFrame()
    reg = env_table.assign(
        _rank=env_table["verdict"].map({v: i for i, v in enumerate(VERDICT_ORDER)}),
        boxes_probe=env_table["verdict"] == "robust attenuation probe",
        boxes_cr_led=env_table["verdict"].isin(CR_FAMILY),
        boxes_fuv=env_table["verdict"] == "G0-modulated",
    )
    counts = reg.groupby("species")[["boxes_probe", "boxes_cr_led", "boxes_fuv"]].sum()
    counts.insert(0, "boxes_graded", reg.groupby("species").size())
    best = (reg.sort_values(["_rank", "abs_response_dex"], ascending=[True, False])
            .groupby("species").head(1).set_index("species")[
                ["answer", "verdict", "environment", "abs_response_dex", "g0_swing_dex",
                 "n_swing_dex", "error_dex"]]
            .rename(columns={"answer": "best_box_answer", "environment": "best_box"}))
    out = best.join(counts)
    out["_tier"] = out["best_box_answer"].map({a: i for i, a in enumerate(ANSWER_ORDER)})
    return (out.sort_values(["_tier", "boxes_probe", "abs_response_dex"],
                            ascending=[True, False, False])
            .drop(columns="_tier").round(2).reset_index())


def best_regimes(env_table, cube_table=None):
    """KoSens ``attenuation_regime_verdicts_3d`` → ``best_regimes``.

    One row per species CR-led in at least one box: ``cube_verdict``,
    ``n_crir_probe`` / ``n_cr_led`` box counts and the boxes themselves
    (largest |R| first), sorted by ``n_crir_probe`` then ``n_cr_led``.  This is
    the row order of the verdict matrix.
    """
    cols = ["species", "cube_verdict", "n_crir_probe", "n_cr_led",
            "crir_probe_regimes", "cr_led_regimes"]
    if env_table is None or not len(env_table):
        return pd.DataFrame(columns=cols)
    cr = env_table[env_table["verdict"].isin(CR_FAMILY)].sort_values(
        "abs_response_dex", ascending=False)
    if not len(cr):
        return pd.DataFrame(columns=cols)
    best = cr.groupby("species", sort=False).agg(
        n_cr_led=("environment", "size"),
        cr_led_regimes=("environment", lambda v: "; ".join(v)))
    probes = cr[cr["verdict"] == "robust attenuation probe"].groupby("species")["environment"]
    best["n_crir_probe"] = probes.size().reindex(best.index, fill_value=0)
    best["crir_probe_regimes"] = probes.agg(lambda v: "; ".join(v)).reindex(best.index).fillna("")
    if cube_table is not None and len(cube_table):
        best["cube_verdict"] = cube_table.set_index("species")["verdict"].reindex(best.index)
    else:
        best["cube_verdict"] = ""
    return (best.reset_index()[cols]
            .sort_values(["n_crir_probe", "n_cr_led"], ascending=False).reset_index(drop=True))


SCOPE_ORDER = ["universal probe", "conditional probe", "not a probe", "no response"]


def graded_boxes(env_table, min_f_att=0.0):
    """Boxes that carry a grade: ``no data`` and never-attenuated (``f_att <= min_f_att``) dropped."""
    reg = env_table[env_table["verdict"] != "no data"]
    if "f_att" in reg.columns:
        reg = reg[~(reg["f_att"].astype(float) <= float(min_f_att))]
    return reg


def probe_scope(env_table, *, response_floor_dex=0.1, min_f_att=0.0):
    """KoSens ``plot_probe_scope_3d``: a probe everywhere, or only once n_H and χ are known?

    A box verdict asks whether a line is clean *inside* one environment; whether
    its response is the *same* in every environment is a separate question.
    Only responding boxes (``|R| >= response_floor_dex``) enter the numbers:

    * ``within_ratio`` = median box χ swing / median box |R| — sets the class;
    * ``within_n_ratio`` = the same against density (reported: part of that
      swing is the attenuation signal itself, see ``r_fatt_corr``);
    * ``between_ratio`` = (max − min over boxes of the signed R) / median |R|;
    * ``between_<axis>_dex`` = max − min over that axis' levels of the median R;
      ``conditional_on`` names the largest, for conditional probes;
    * ``r_fatt_corr`` = Pearson correlation of the per-box R with ``f_att`` over
      all graded boxes; ``r_per_fatt_dex`` = median R / f_att over responding boxes.

    ``scope``: ``no response`` (no responding box), ``not a probe``
    (within_ratio ≥ 1), ``universal probe`` (both ratios < 1), ``conditional
    probe`` (clean inside each box, different between boxes).
    """
    if env_table is None or not len(env_table):
        return pd.DataFrame()
    reg = graded_boxes(env_table, min_f_att)
    if not len(reg):
        return pd.DataFrame()
    g = reg.groupby("species")
    resp = reg[reg["abs_response_dex"] >= float(response_floor_dex)]
    gr = resp.groupby("species")
    out = pd.DataFrame({
        "n_boxes": g.size(),
        "n_boxes_responding": gr.size(),
        "boxes_probe": g["verdict"].apply(lambda v: int((v == "robust attenuation probe").sum())),
        "median_abs_response_dex": gr["abs_response_dex"].median(),
        "within_swing_dex": gr["g0_swing_dex"].median(),
        "within_n_swing_dex": gr["n_swing_dex"].median(),
        "between_spread_dex": gr["response_dex"].apply(_swing),
    })
    level_cols = [c for c in LEVEL_COLS if c in reg.columns]
    axes = [c[:-6] for c in level_cols]
    for ax_name, col in zip(axes, level_cols):
        per_level = resp.groupby(["species", col], observed=True)["response_dex"].median()
        out[f"between_{ax_name}_dex"] = per_level.groupby(level=0).apply(_swing)
    if "f_att" in reg.columns:
        out["r_fatt_corr"] = pd.Series({
            sp: (d["response_dex"].corr(d["f_att"].astype(float))
                 if d["f_att"].nunique() > 2 and d["response_dex"].nunique() > 1 else np.nan)
            for sp, d in g})
        out["r_per_fatt_dex"] = (resp["response_dex"] / resp["f_att"].astype(float)
                                 ).groupby(resp["species"]).median()
    else:
        out["r_fatt_corr"] = out["r_per_fatt_dex"] = np.nan
    out["n_boxes_responding"] = out["n_boxes_responding"].fillna(0).astype(int)
    out["frac_responding"] = out["n_boxes_responding"] / out["n_boxes"]
    r_med = out["median_abs_response_dex"].where(out["median_abs_response_dex"] > 0.0)
    out["within_ratio"] = out["within_swing_dex"] / r_med
    out["within_n_ratio"] = out["within_n_swing_dex"] / r_med
    out["between_ratio"] = out["between_spread_dex"] / r_med
    out["scope"] = np.select(
        [out["n_boxes_responding"] == 0, ~(out["within_ratio"] < 1.0),
         out["between_ratio"] < 1.0],
        ["no response", "not a probe", "universal probe"], default="conditional probe")
    axis_cols = [f"between_{a}_dex" for a in axes]
    if axis_cols:
        dominant = (out[axis_cols].fillna(-np.inf).idxmax(axis=1)
                    .str[len("between_"):-len("_dex")])
        out["conditional_on"] = dominant.where(out["scope"] == "conditional probe", "")
    else:
        out["conditional_on"] = ""
    out["_t"] = out["scope"].map({sc: i for i, sc in enumerate(SCOPE_ORDER)})
    out = (out.sort_values(["_t", "median_abs_response_dex"], ascending=[True, False])
           .drop(columns="_t").rename_axis("species").reset_index())
    return out[["species", "scope", "conditional_on", "n_boxes", "n_boxes_responding",
                "frac_responding", "boxes_probe", "median_abs_response_dex",
                "within_swing_dex", "within_n_swing_dex", "between_spread_dex",
                "within_ratio", "within_n_ratio", "between_ratio", *axis_cols,
                "r_fatt_corr", "r_per_fatt_dex"]]


def zeta_regime_verdicts(pixels, n_bins, summary_kw, response_kw, reconcile_kw, whole=None):
    """In which CRIR (ζ) regime does each species become CR-led?

    Whole-grid verdicts take the median R over the whole ζ axis, so a line inert
    at low ζ and strongly responding at high ζ grades ``no response`` although
    it is a good probe at high ζ.  Here the ζ axis is cut into ``n_bins``
    equal-log-width regimes and the chain is re-run on each; the χ (and n_H)
    swing still spans its full range inside a regime.  Shifts were matched on
    the full row, so a regime is not biased by where its bin stops; the top
    regime has less room for a rightward match, holds more Δ = 0 cells and its
    errors are more conservative (a grid limit).

    Returns ``{'regimes', 'best', 'edges', 'levels'}``: ``regimes`` one graded
    row per (species, regime); ``best`` one row per species CR-led in at least
    one regime (robust attenuation probe first, then largest |R|) with its regime, the
    ζ / χ extent of its observable cells, ``whole_grid_verdict`` and every
    CR-led regime.
    """
    pixels = _ensure_cells(pixels)
    edges = _positive_logspace_edges(pixels["x"].unique(), max(int(n_bins), 1))
    if edges is None or len(edges) < 2:
        raise ValueError("The CRIR axis has no positive values.")
    levels = _bin_names(len(edges) - 1)
    labels = [f"{lv} ζ".strip() for lv in levels]
    env = environment_verdicts(pixels, {"x": list(edges[1:-1])}, summary_kw, response_kw,
                               reconcile_kw)
    regimes = env["table"].copy()
    if not len(regimes):
        return {"regimes": regimes, "best": pd.DataFrame(), "edges": edges, "levels": labels}
    if len(labels) == 1:
        regimes["environment"] = labels[0]
    k = {lab: i for i, lab in enumerate(labels)}
    regimes["regime"] = regimes["environment"].map(lambda e: str(e))
    regimes["zeta_lo"] = regimes["regime"].map(lambda r: edges[k[r]] if r in k else np.nan)
    regimes["zeta_hi"] = regimes["regime"].map(lambda r: edges[k[r] + 1] if r in k else np.nan)
    cr = regimes[regimes["verdict"].isin(CR_FAMILY)]
    if not len(cr):
        return {"regimes": regimes, "best": pd.DataFrame(), "edges": edges, "levels": labels}
    order = {v: i for i, v in enumerate(["robust attenuation probe", "robust, small bias"])}
    cr = cr.assign(_rank=cr["verdict"].map(order))
    top = (cr.sort_values(["_rank", "abs_response_dex"], ascending=[True, False])
           .groupby("species", sort=False).head(1).set_index("species"))
    led = cr.sort_values("zeta_lo").groupby("species")["regime"].agg("; ".join)
    if whole is not None and len(whole):
        whole_v = whole.drop_duplicates("species").set_index("species")["verdict"]
    else:
        whole_v = pd.Series(dtype=object)
    best = pd.DataFrame({
        "label": top["label"], "best_regime": top["regime"], "zeta_lo": top["zeta_lo"],
        "zeta_hi": top["zeta_hi"], "verdict": top["verdict"],
        "abs_response_dex": top["abs_response_dex"], "fuv_swing_dex": top["fuv_swing_dex"],
        "error_dex": top["error_dex"], "frac_observable": top["frac_observable"],
        "whole_grid_verdict": whole_v.reindex(top.index), "cr_led_regimes": led.reindex(top.index),
        "_rank": top["_rank"],
    })
    seen = pixels[pixels["observable"].astype(bool)]
    ext = seen.assign(regime=pd.cut(seen["x"], bins=edges, labels=labels,
                                    include_lowest=True)).groupby(
        ["species", "regime"], observed=True)[["x", "y"]].agg(["min", "max"])
    keys = list(zip(best.index, top["regime"]))
    for col, name in (("x", "zeta"), ("y", "chi")):
        best[f"{name}_obs_lo"] = [ext[(col, "min")].get(kk, np.nan) for kk in keys]
        best[f"{name}_obs_hi"] = [ext[(col, "max")].get(kk, np.nan) for kk in keys]
    best["zeta_range"] = (best["zeta_lo"].map("{:.1e}".format) + " – "
                          + best["zeta_hi"].map("{:.1e}".format))
    best["zeta_observable"] = (best["zeta_obs_lo"].map("{:.1e}".format) + " – "
                               + best["zeta_obs_hi"].map("{:.1e}".format))
    best["chi_observable"] = (best["chi_obs_lo"].map("{:.1e}".format) + " – "
                              + best["chi_obs_hi"].map("{:.1e}".format))
    best = (best.sort_values(["_rank", "abs_response_dex"], ascending=[True, False])
            .drop(columns="_rank").reset_index())
    return {"regimes": regimes, "best": best, "edges": edges, "levels": labels}


def run_workflow(pixels, *, summary_kw=None, response_kw=None, reconcile_kw=None,
                 env_edges=None, require_catalog=False, progress=None):
    """Whole procedure on one pixel table.

    Settings that must agree across the steps (``exclude_ice``,
    ``exclude_isotopologues``, ``max_per_constituent``) are copied into every
    step here, so they cannot drift apart.  ``env_edges`` adds the verdicts per
    (n_H, χ, ζ) box.  ``require_catalog``
    (KoSens 3-D default) keeps only species with CDMS/JPL spectroscopy, before
    anything is graded; the dropped keys come back as ``no_catalog``.
    """
    summary_kw = dict(summary_kw or {})
    response_kw = dict(response_kw or {})
    reconcile_kw = dict(reconcile_kw or {})
    for key in ("exclude_ice", "exclude_isotopologues", "max_per_constituent"):
        if key in summary_kw:
            response_kw.setdefault(key, summary_kw[key])

    pixels = _ensure_cells(pixels)
    no_catalog, notes = pd.DataFrame(columns=["species", "missing"]), []
    if require_catalog:
        try:
            pixels, no_catalog = drop_uncatalogued(pixels)
        except Exception as exc:          # astroquery missing or its tables unreadable
            notes.append(f"CDMS/JPL filter not applied ({exc}).")
        if len(no_catalog):
            notes.append(f"Skipped {len(no_catalog)} species/ratios with no CDMS/JPL "
                         "spectroscopy: " + ", ".join(
                             f"{r.species} ({r.missing})" if "/" in str(r.species) else str(r.species)
                             for r in no_catalog.itertuples()) + ".")
    # Drop the excluded keys once, here: every later step (ζ regimes, densities, boxes)
    # then works on the species that are actually graded.
    if summary_kw.get("exclude_ice", True):
        pixels = _drop_keys(pixels, _is_ice_species_key, "ice", notes)
    if summary_kw.get("exclude_isotopologues", False):
        pixels = _drop_keys(pixels, _is_isotopologue_key, "isotopologue", notes)
    s, r, v = _run_trio(pixels, summary_kw, response_kw, reconcile_kw)
    out = {"summary": s, "response": r, "verdicts": v,
           "environments": None, "multi_density": pixels["i_n"].nunique() > 1,
           "no_catalog": no_catalog, "notes": notes, "pixels": pixels,
           "response_kw": response_kw}
    if env_edges is not None:
        out["environments"] = environment_verdicts(
            pixels, env_edges, summary_kw, response_kw, reconcile_kw,
            progress=progress)
    return out


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

import plotly.graph_objects as go                      # noqa: E402
from plotly.colors import sample_colorscale           # noqa: E402
from plotly.subplots import make_subplots             # noqa: E402

import plot_style as ps                               # noqa: E402
import probe_ranking as pr                            # noqa: E402

# KoSens ``_VERDICT_COLORS`` (cascade order, best first).
VERDICT_COLORS = {
    "robust attenuation probe": "#1b7837", "robust, small bias": "#5aae61", "G0-modulated": "#e08214",
    "no response": "#e0e0e0", "no data": "#ffffff",
}
VERDICT_SHORT = {
    "robust attenuation probe": "probe", "robust, small bias": "small bias", "G0-modulated": "G0",
    "no response": "no resp.", "no data": "—",
}
SCOPE_COLORS = {
    "universal probe": "#1b7837", "conditional probe": "#5aae61", "not a probe": "#e08214",
    "no response": "#bdbdbd",
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


def fig_shift_bars(summary_out, species=None, *, theme=None, fmt=None, title=None,
                   scale_species=None, page=""):
    """Step 1: models per |Δ| bin (stacked, lowest bin omitted) and median |Δ|.

    The gap between a bar and the "all models" line is the models that did not
    move — unmatched and unobservable cells included, at |Δ| = 0.  ``scale_species``
    (all the pages of a split list) fixes both y ranges, so the figures can be compared.
    """
    t = pr._theme_fallback(theme)
    species = list(species if species is not None else summary_out.get("top_species") or [])
    counts = summary_out.get("bin_counts")
    if not species or counts is None or not len(counts):
        return empty_fig("No species to draw.", theme)
    labels = list(summary_out["bin_labels"][1:])
    stack = counts.reindex(species)[labels].fillna(0.0)
    names = [pretty_key(s, fmt) for s in species]
    colors = sample_colorscale("YlOrRd", [0.30 + 0.65 * i / max(len(labels) - 1, 1)
                                          for i in range(len(labels))])
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        subplot_titles=("Models that moved (|Δ| bins)", "Typical shift"))
    for lab, col in zip(labels, colors):
        fig.add_trace(go.Bar(x=names, y=stack[lab], name=f"{lab} dex",
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
    if scale_species is not None:
        tall = counts.reindex(scale_species)[labels].fillna(0.0).sum(axis=1).max()
        tall = max(tall, n_all) if n_all is not None and np.isfinite(n_all) else tall
        fig.update_yaxes(range=[0.0, 1.05 * max(float(tall), 1.0)], row=1, col=1)
        top = (summary_out["species_summary"].set_index("species")["median_abs_shift_dex"]
               .reindex(scale_species).max())
        if np.isfinite(top) and top > 0.0:
            fig.update_yaxes(range=[0.0, 1.05 * float(top)], row=1, col=2)
    fig.update_xaxes(tickangle=-40)
    return _style(fig, t, (title or "Step 1 · How many models move, and by how much") + page,
                  440 + 4 * max(len(n) for n in names))


def fig_regime_heatmaps(summary_out, species=None, *, theme=None, fmt=None, scale_species=None,
                        page=""):
    """Step 1: median |Δ| per ζ bin, χ bin (and n_H in 3-D) — *where* a species shifts.

    ``scale_species`` (all the pages of a split list) fixes the colour range.
    """
    t = pr._theme_fallback(theme)
    species = list(species if species is not None else summary_out.get("top_species") or [])
    heats = [(summary_out.get("heat_x"), "ζ bin [s⁻¹]"), (summary_out.get("heat_y"), "χ bin"),
             (summary_out.get("heat_n"), "n_H [cm⁻³]")]
    heats = [(h.reindex(species), lab) for h, lab in heats if h is not None]
    if not species or not heats:
        return empty_fig("No regime heatmaps (no cells with a model).", theme)
    scale = (summary_out.get(k) for k in ("heat_x", "heat_y", "heat_n"))
    arrays = [h.reindex(scale_species if scale_species is not None else species)
              .to_numpy(dtype=float) for h in scale if h is not None]
    vmax = max([0.05] + [float(np.nanmax(a)) for a in arrays if np.isfinite(a).any()])
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
    return _style(fig, t, f"Step 1 · Where the shift happens{page}",
                  max(320, 60 + 34 * len(species)))


def fig_winner_map(summary_out, *, theme=None, fmt=None):
    """Step 1: in each (ζ, χ) regime the species with the largest median |Δ|.

    One panel per density bin in 3-D, each with its own colorbar on one shared scale.  Read as *"here, follow up this species"* —
    not as a tracer ranking (a G0-modulated species can win).
    """
    t = pr._theme_fallback(theme)
    win = summary_out.get("regime_winners")
    x_labels, y_labels = summary_out.get("x_labels"), summary_out.get("y_labels")
    if win is None or not len(win) or not x_labels:
        return empty_fig("No winner map (no cells with a model).", theme)
    key = next((k for k in ("n_bin", "n") if k in win.columns), None)
    panels = [(None, win)] if key is None else list(win.groupby(key, observed=True))
    n_cols = min(len(panels), 3)
    n_rows = int(np.ceil(len(panels) / n_cols))
    titles = ["" if n is None else
              (f"n_H {n} cm⁻³" if key == "n_bin" else f"n_H = {pr.format_phys(n)} cm⁻³")
              for n, _ in panels]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                        horizontal_spacing=0.10, vertical_spacing=0.18 if n_rows > 1 else 0.1)
    vmax = max(float(win["median_abs_shift_dex"].max()), 0.05)
    for k, (_n, g) in enumerate(panels):
        row, col = k // n_cols + 1, k % n_cols + 1
        lab = g.pivot(index="y_bin", columns="x_bin", values="label").reindex(
            index=y_labels, columns=x_labels)
        val = g.pivot(index="y_bin", columns="x_bin", values="median_abs_shift_dex").reindex(
            index=y_labels, columns=x_labels)
        cnt = g.pivot(index="y_bin", columns="x_bin", values="n_models").reindex(
            index=y_labels, columns=x_labels)
        # Split on the key's own "/": the formatted label holds </sub> tags.
        text = lab.fillna("").map(
            lambda s: "/<br>".join(pretty_key(part, fmt) for part in s.split("/")))
        sub = fig.get_subplot(row, col)
        fig.add_trace(go.Heatmap(
            z=val.to_numpy(dtype=float), x=x_labels, y=y_labels, zmin=0.0, zmax=vmax,
            colorscale="YlOrRd", text=text.to_numpy(), texttemplate="%{text}", textfont=dict(size=14),
            customdata=np.dstack([lab.fillna("").to_numpy(), cnt.to_numpy(dtype=float)]),
            # One bar per panel beside its own map, all on the same scale.
            colorbar=dict(title=dict(text="winner median |Δ| [dex]", side="right",
                                     font=ps.cbar_title_font(t["font"])),
                          tickfont=ps.cbar_tick_font(t["font"]), thickness=12,
                          x=sub.xaxis.domain[1] + 0.005, xanchor="left",
                          y=float(np.mean(sub.yaxis.domain)), yanchor="middle",
                          len=sub.yaxis.domain[1] - sub.yaxis.domain[0]),
            hovertemplate=("ζ %{x}<br>χ %{y}<br>%{customdata[0]}<br>median |Δ| = %{z:.3f} dex"
                           "<br>cells = %{customdata[1]:.0f}<extra></extra>")),
            row=row, col=col)
    fig.update_xaxes(title_text="ζ bin [s⁻¹]", tickangle=-30)
    # χ labels on the first column only: elsewhere they would run into the colorbar beside them.
    fig.update_yaxes(showticklabels=False)
    fig.update_yaxes(title_text="χ bin", showticklabels=True, col=1)
    return _style(fig, t, "Step 1 · Winner map — which species to follow up where",
                  max(420, 480 * n_rows))


def species_pages(table, flag, per_page, *, by=None):
    """Rows of ``table`` (those with ``flag`` set, if given) in groups of ``per_page``.

    Always at least one group, so an empty table still draws its empty figure.
    """
    tab = table[table[flag].astype(bool)] if flag and len(table) else table
    if by is not None:
        tab = tab.sort_values(by, ascending=False)
    n = max(int(per_page), 1)
    return [tab.iloc[i:i + n] for i in range(0, len(tab), n)] or [tab]


def fig_error_bars(table, *, theme=None, fmt=None, scale_table=None, page=""):
    """Step 2 (left): ζ error from ignoring attenuation, largest at the top.

    Bar = median |Δ| over every cell (unmatched and unobservable at 0); band =
    p10–p90 over grid cells (a spread, not an uncertainty on the median);
    colour = fraction of the cells above the detection limit.  ``scale_table`` (all
    the pages of a split list) fixes the x range, so the figures can be compared.
    """
    t = pr._theme_fallback(theme)
    tab = table[table["ranked"].astype(bool)] if len(table) else table
    if not len(tab):
        return empty_fig("No species with a ζ error to rank.", theme)
    names = [pretty_key(s, fmt) for s in tab["species"]]
    obs = np.nan_to_num(tab["frac_observable"].to_numpy(dtype=float), nan=1.0)
    p10 = tab["error_p10"].to_numpy(dtype=float)
    p90 = tab["error_p90"].to_numpy(dtype=float)
    err = tab["error_dex"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=err, orientation="h", name="median |Δ|",
        customdata=np.column_stack([p10, p90, obs]),
        marker=dict(color=obs, cmin=0.0, cmax=1.0, colorscale="YlOrRd",
                    line=dict(color="black", width=1.0),
                    colorbar=dict(title=dict(text="fraction of cells above<br>the detection limit",
                                             font=ps.cbar_title_font(t["font"])),
                                  tickfont=ps.cbar_tick_font(t["font"]))),
        hovertemplate=("%{y}<br>error = %{x:.3f} dex"
                       "<br>p10–p90 = %{customdata[0]:.2f}–%{customdata[1]:.2f}"
                       "<br>observable = %{customdata[2]:.0%}<extra></extra>")))
    fig.add_trace(go.Bar(
        y=names, x=np.clip(p90 - p10, 0.0, None), base=p10, orientation="h", width=0.3,
        name="p10–p90 over cells", marker=dict(color="rgba(0,0,0,0.25)"), hoverinfo="skip"))
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=-0.18))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="|Δ| = |log₁₀(ζ_match / ζ_nom)| [dex]", rangemode="tozero")
    if scale_table is not None:
        hi = pd.concat([scale_table["error_dex"], scale_table["error_p90"]]).astype(float).max()
        if np.isfinite(hi) and hi > 0.0:
            fig.update_xaxes(range=[0.0, 1.05 * hi])
    return _style(fig, t, f"Step 2 · Error in ζ if attenuation is ignored{page}",
                  max(640, 130 + 42 * len(names)))


def _probe_limits(vals, scale, axis_min, axis_max):
    """Shared (lo, hi) of both probe-plane axes; blank user limits mean auto.

    Log: zoomed to the data with at least one decade, so y = x stays at 45°.
    Linear: 0 to just past the largest value.  Log limits must be positive.
    """
    vals = vals[np.isfinite(vals)]
    if scale == "log":
        pos = vals[vals > 0.0]
        pad = 10.0 ** 0.15
        lo, hi = (pos.min() / pad, pos.max() * pad) if pos.size else (0.01, 1.0)
        if hi / lo < 10.0:
            mid = np.sqrt(lo * hi)
            lo, hi = mid / np.sqrt(10.0), mid * np.sqrt(10.0)
    else:
        lo, hi = 0.0, max(1.08 * (vals.max() if vals.size else 0.0), 0.2)
    if axis_min is not None:
        lo = float(axis_min)
    if axis_max is not None:
        hi = float(axis_max)
    if scale == "log" and lo <= 0.0:
        lo = 1e-3
    if hi <= lo:
        hi = lo * 10.0 if scale == "log" else lo + 1.0
    return lo, hi


def _species_colors(n):
    return sample_colorscale("Turbo", list(np.linspace(0.05, 0.95, max(n, 2))))[:n]


def fig_probe_plane(table, *, response_floor_dex=0.1, axis_scale="log", axis_min=None,
                    axis_max=None, theme=None, fmt=None, swing_label="χ swing of R",
                    scale_table=None, page=""):
    """Step 2 (right): |R| against its χ swing (``g0_swing_dex``), kept as two numbers.

    Below the diagonal the response beats its swing (CR-dominated response);
    above it the swing wins (G0-modulated response, which does not make the
    species an FUV tracer); left of the floor there is hardly any
    response.  ``axis_scale`` is ``'log'`` or ``'linear'``; both axes share one
    range so the diagonal stays at 45°, and ``axis_min`` / ``axis_max`` override
    it.  A zero swing is pinned to the lower edge on a log axis.  ``scale_table``
    (all the pages of a split list) fixes the axes and the marker size scale.
    """
    t = pr._theme_fallback(theme)
    tab = table[table["probe_ranked"].astype(bool)] if len(table) else table
    if not len(tab):
        return empty_fig("No species to draw in the probe panel.", theme)
    tab = tab.sort_values("abs_response_dex", ascending=False)
    log = axis_scale != "linear"
    err = tab["error_dex"].to_numpy(dtype=float)
    ref = tab if scale_table is None else scale_table
    lo, hi = _probe_limits(np.concatenate([ref["abs_response_dex"].to_numpy(dtype=float),
                                           ref["g0_swing_dex"].to_numpy(dtype=float)]),
                           "log" if log else "linear", axis_min, axis_max)
    edge = lo * 10.0 ** 0.04 if log else lo
    floor = float(response_floor_dex)
    x_cr = max(floor, lo)
    fig = go.Figure()
    if x_cr < hi:
        fig.add_trace(go.Scatter(x=[x_cr, hi, hi, x_cr], y=[lo, lo, hi, x_cr], fill="toself",
                                 fillcolor="rgba(22,163,74,0.08)", mode="none",
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[lo, hi, lo], y=[lo, hi, hi], fill="toself",
                             fillcolor="rgba(147,51,234,0.08)", mode="none",
                             hoverinfo="skip", showlegend=False))
    if lo < floor < hi:
        fig.add_trace(go.Scatter(x=[lo, floor, floor, lo], y=[lo, lo, hi, hi], fill="toself",
                                 fillcolor="rgba(100,116,139,0.12)", mode="none",
                                 hoverinfo="skip", showlegend=False))
        fig.add_vline(x=floor, line=dict(color=t["muted"], dash="dot"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", hoverinfo="skip",
                             line=dict(color=t["muted"], dash="dash"), showlegend=False))
    ref_err = ref["error_dex"].to_numpy(dtype=float)
    size = 10.0 + 26.0 * np.nan_to_num(err, nan=0.0) / max(
        np.nanmax(ref_err) if np.isfinite(ref_err).any() else 1.0, 1e-9)
    palette = _species_colors(len(tab))
    for i, (_, row) in enumerate(tab.iterrows()):
        name = pretty_key(row["species"], fmt)
        xi, yi = row["abs_response_dex"], np.nan_to_num(row["g0_swing_dex"])
        fig.add_trace(go.Scatter(
            x=[max(xi, edge) if log else xi], y=[max(yi, edge) if log else yi], mode="markers",
            name=f"{name}  ({row['error_dex']:.2f})",
            marker=dict(size=size[i], color=palette[i], line=dict(color="black", width=1.2)),
            hovertemplate=(f"{name}<br>|R| = {xi:.3f} dex<br>χ swing = {row['g0_swing_dex']:.3f} dex"
                           f"<br>{'CR' if row['cr_dominated'] else 'FUV'}-dominated response"
                           f"<br>|Δ| = {row['error_dex']:.3f} dex<extra></extra>")))
    # Region tags hug the diagonal (sparsest strip); in log space "0.5 of the
    # way" is the geometric mean, so place them in axis fractions of the range.
    def at(f):
        return lo * (hi / lo) ** f if log else lo + f * (hi - lo)
    for txt, ax, ay, col in (("CR-dominated<br>response beats its own spread", 0.62, 0.40,
                              "#15803d"),
                             ("G0-modulated<br>spread beats the response", 0.38, 0.62,
                              "#7e22ce")):
        fig.add_annotation(x=np.log10(at(ax)) if log else at(ax),
                           y=np.log10(at(ay)) if log else at(ay), text=f"<i>{txt}</i>",
                           showarrow=False, font=dict(size=11, color=col),
                           bgcolor="rgba(255,255,255,0.55)")
    kind = "log" if log else "linear"
    fig.update_xaxes(type=kind, range=[np.log10(lo), np.log10(hi)] if log else [lo, hi],
                     constrain="domain", title_text="|R| [dex]  (does attenuation change it?)")
    fig.update_yaxes(type=kind, range=[np.log10(lo), np.log10(hi)] if log else [lo, hi],
                     scaleanchor="x", scaleratio=1, constrain="domain",
                     title_text=f"{swing_label} [dex]  (does the radiation field matter?)")
    if log:
        for ax in (fig.update_xaxes, fig.update_yaxes):
            ax(dtick="D2", minor=dict(showgrid=True))
    n_items = sum(tr.showlegend is not False for tr in fig.data)
    fig = _style(fig, t, "Step 2 · Attenuation response vs its FUV dependence — "
                 f"marker size = |Δ|{page}", max(640, 190 + 26 * n_items))
    fig.update_layout(legend=dict(title=dict(text="species  (|Δ| [dex])", side="top"),
                                  orientation="v", x=1.02, xanchor="left", y=1.0,
                                  yanchor="top", maxheight=4000,
                                  font=ps.legend_font(t["font"]) | dict(size=15),
                                  title_font=dict(size=15)))
    return fig


def fig_profiles(table, profiles, *, x_scale="log", x_label="χ",
                 theme=None, fmt=None, page=""):
    """Step 2 (profiles): median R and median Δ along the χ axis, per species.

    The curve the swing summarises (swing = its extent, max − min; ``fuv_trend`` its
    slope): whether R drifts, flips sign or is flat.  Same species and colours
    as the probe plane.  Δ breaks where a row has no model.
    """
    t = pr._theme_fallback(theme)
    tab = table[table["probe_ranked"].astype(bool)] if len(table) else table
    tab = tab[tab["species"].isin(profiles)] if len(tab) else tab
    if not len(tab):
        return empty_fig("No species to draw in the profile panel.", theme)
    tab = tab.sort_values("abs_response_dex", ascending=False)
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, horizontal_spacing=0.09,
                        subplot_titles=("R = log₁₀(I_atten / I_const): change at the same "
                                        "cell<br>R > 0 brighter, R < 0 fainter",
                                        f"Δ = log₁₀(ζ_match / ζ_nom): shift along ζ to match"
                                        "<br>unmatched and unobservable cells at 0"))
    palette = _species_colors(len(tab))
    for i, (_, row) in enumerate(tab.iterrows()):
        yv, rv, dv = profiles[row["species"]]
        name = pretty_key(row["species"], fmt)
        for col, vals, show in ((1, rv, False), (2, dv, True)):
            fig.add_trace(go.Scatter(
                x=yv, y=vals, mode="lines+markers", name=name, legendgroup=name,
                showlegend=show, line=dict(color=palette[i], width=2.5), marker=dict(size=6),
                connectgaps=False), row=1, col=col)
    fig.add_hline(y=0.0, line=dict(color=t["muted"], dash="dash"), row=1, col=1)
    fig.update_xaxes(type="log" if x_scale != "linear" else "linear", title_text=x_label)
    fig.update_yaxes(title_text="median R [dex]", row=1, col=1)
    fig.update_yaxes(title_text="median Δ [dex]", rangemode="tozero", row=1, col=2)
    fig = _style(fig, t, f"Step 2 · Profiles along χ{page}", 520)
    fig.update_annotations(font=dict(size=13))
    return fig


_STRIP_SCALES = {"ζ": "Purples", "n_H": "Greens", "χ": "Blues"}


def _levels(values):
    return sorted({str(v) for v in values}, key=lambda v: (_LEVEL_RANK.get(v, 99), v))


def _add_level_strips(fig, envs, x, row, order=("chi", "n", "zeta"), f_att=False):
    """Colour strips under a species × environment matrix: one row per split axis.

    Each strip marks the column's level on that axis, light → dark for low → high
    (n_H green, χ blue, ζ purple), first key in ``order`` next to the matrix.
    They replace the long environment names on the column axis.  ``f_att`` adds
    a row with the box's CR-attenuated mass fraction.  Returns the number of rows.
    """
    by_key = {key: sym for _c, sym, key in ENV_AXES}
    # Numeric y with tick text: plotly misplaces the bands of one-row heatmaps that
    # share a categorical y axis, leaving each colour under another strip's labels.
    ticks = []
    for key in order:
        col = f"{key}_level"
        if not any(col in e["levels"] for e in envs):
            continue
        lv = [e["levels"].get(col, "") for e in envs]
        names = _levels(lv)
        code = [0.25 + 0.6 * names.index(v) / max(len(names) - 1, 1) for v in lv]
        sym = by_key[key]
        fig.add_trace(go.Heatmap(
            z=[code], x=x, y=[len(ticks)], zmin=0.0, zmax=1.0, showscale=False, xgap=2, ygap=2,
            colorscale=_STRIP_SCALES.get(sym, "Greys"), text=[lv], texttemplate="%{text}",
            hovertemplate=sym + " %{text}<extra></extra>"), row=row, col=1)
        ticks.append(sym)
    fa = np.array([e.get("f_att", np.nan) for e in envs], dtype=float)
    if f_att and np.isfinite(fa).any():
        fig.add_trace(go.Heatmap(
            z=[fa], x=x, y=[len(ticks)], zmin=0.0, zmax=1.0, showscale=False, xgap=2, ygap=2,
            colorscale="Greys", text=[[f"{v:.2f}" if np.isfinite(v) else "" for v in fa]],
            texttemplate="%{text}", hovertemplate="f_att = %{z:.2f}<extra></extra>"),
            row=row, col=1)
        ticks.append("f_att")
    fig.update_yaxes(showgrid=False, autorange="reversed", tickvals=list(range(len(ticks))),
                     ticktext=ticks, row=row, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=1)
    return len(ticks)


def fig_verdict_matrix(long_table, *, col_key, col_order, col_labels=None, species=None,
                       theme=None, fmt=None, title="Verdicts", annot_col=None, group=None,
                       env_info=None):
    """Step 3: species × (ζ regime | environment) grid coloured by verdict.

    ``group`` draws a divider every that many columns (KoSens: one block per n_H level).
    With ``env_info`` the columns are environments: their names are not printed on
    the axis (they are in the hover); colour strips under the matrix give each
    column's n_H, χ and ζ level instead.
    """
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
            if annot_col:
                text[i, j] += f"<br>{pr.format_score(row.get(annot_col))}"
            hover[i, j] = (f"{pretty_key(sp, fmt)} · {col_labels[j]}<br><b>{v}</b>"
                           f"<br>|Δ| = {pr.format_score(row.get('error_dex'))} dex"
                           f"<br>χ swing = {pr.format_score(row.get('g0_swing_dex'))} dex"
                           f"<br>|R| = {pr.format_score(row.get('abs_response_dex'))} dex"
                           f"<br>cells = {pr.format_score(row.get('n_cells'), 0)}")
    scale = []
    for idx, v in enumerate(VERDICT_ORDER):
        scale += [[idx / k, VERDICT_COLORS[v]], [(idx + 1) / k, VERDICT_COLORS[v]]]
    envs = None
    if env_info:
        by_label = {e["label"]: e for e in env_info}
        envs = [by_label[c] for c in col_order if c in by_label]
    strips = envs is not None and len(envs) == len(col_order)
    x = [str(c) for c in col_order] if strips else col_labels
    fig = make_subplots(rows=2 if strips else 1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        row_heights=[len(species), 2.7] if strips else None)
    fig.add_trace(go.Heatmap(
        z=z, x=x, y=[pretty_key(s, fmt) for s in species], zmin=-0.5, zmax=k - 0.5,
        colorscale=scale, text=text, texttemplate="%{text}", customdata=hover,
        hovertemplate="%{customdata}<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(tickvals=list(range(k)), ticktext=VERDICT_ORDER,
                      tickfont=ps.cbar_tick_font(t["font"]))), row=1, col=1)
    fig.update_yaxes(autorange="reversed", showgrid=False, row=1, col=1)
    fig.update_xaxes(side="top", tickangle=-25, showgrid=False, showticklabels=not strips,
                     row=1, col=1)
    if strips:
        # ζ varies fastest along the columns, n_H slowest: innermost strip first.
        _add_level_strips(fig, envs, x, row=2, order=("zeta", "chi", "n"))
    if group:
        for j in range(int(group), len(col_order), int(group)):
            fig.add_vline(x=j - 0.5, line=dict(color=t["font"], width=2))
    return _style(fig, t, title, max(320, 120 + 30 * len(species)))


def _zeta_outer_envs(env_info, present):
    """Environments in display order, ζ outermost (``assign_environments`` has ζ fastest)."""
    n_zeta = len({e["ranges"].get("ζ") for e in env_info}) or 1
    order = sorted(range(len(env_info)), key=lambda i: (i % n_zeta, i // n_zeta))
    return [env_info[i] for i in order if env_info[i]["label"] in present]


def impact_ranking(env_table, env_info, *, metric="error_dex", min_f_att=0.0,
                   exclude_species=None, top_n=15, species_order=None):
    """Species × environment table of ``metric`` and the ranking behind the impact matrix.

    Returns ``(matrix, ranking, envs, no_atten)``.  ``no data`` boxes are blank;
    every other box carries its value (unmatched and unobservable cells at 0
    inside it), so every species is ranked over the same boxes.  ``ranking`` has
    one row per species: ``median_box_<metric>`` (median of its per-box values
    over the attenuated boxes — environments with ``f_att <= min_f_att`` never
    switched attenuation on and are left out), ``n_boxes``, ``rank`` and
    ``shown`` (one of the ``top_n`` rows drawn).  ``exclude_species`` are dropped
    before the ranking, so the rows are filled from the remaining species.

    ``env_table`` may also be the per-box :func:`observer_error_table` (no
    ``verdict`` column; ``metric="D_med"`` or ``"Delta_med"``).  ``species_order``
    (e.g. another matrix's rows) fixes the shown rows instead of the ``top_n``
    ranking, so two matrices put each species in the same row.
    """
    reg = (env_table[env_table["verdict"] != "no data"] if "verdict" in env_table
           else env_table)
    envs = _zeta_outer_envs(env_info, set(reg["environment"]))
    cols = [e["label"] for e in envs]
    f_att = np.array([e.get("f_att", np.nan) for e in envs], dtype=float)
    if "f_att" in reg:  # a native-grid observer table carries its own box values
        f_att = (reg.drop_duplicates("environment").set_index("environment")["f_att"]
                 .reindex(cols).to_numpy(dtype=float))
    no_atten = f_att <= float(min_f_att)  # NaN (no f_att grid) compares False: kept
    matrix = reg.pivot_table(index="species", columns="environment", values=metric,
                             aggfunc="first").reindex(columns=cols)
    matrix = matrix.drop(index=list(exclude_species or []), errors="ignore")
    ranked_on = matrix.loc[:, ~no_atten] if not no_atten.all() else matrix
    ranking = (pd.DataFrame({f"median_box_{metric}": ranked_on.median(axis=1),
                             f"mean_box_{metric}": ranked_on.mean(axis=1),
                             "n_boxes": ranked_on.notna().sum(axis=1)})
               .sort_values(f"median_box_{metric}", ascending=False)
               .rename_axis("species").reset_index())
    ranking.insert(1, "rank", np.arange(1, len(ranking) + 1))
    ranking["shown"] = (ranking["species"].isin(list(species_order))
                        if species_order is not None else ranking["rank"] <= int(top_n))
    return matrix, ranking, envs, no_atten


def fig_impact_matrix(env_table, env_info, *, metric="error_dex", top_n=15,
                      min_f_att=0.0, exclude_species=None, species_order=None,
                      theme=None, fmt=None):
    """KoSens ``plot_attenuation_impact_matrix_3d``: which species does attenuation
    affect most, in each (n_H, χ, ζ) environment?

    Species × environment heatmap of ``metric`` (``error_dex`` = |Δ|, the ζ error;
    or ``abs_response_dex`` = |R|) for the ``top_n`` species of
    :func:`impact_ranking`; nothing is recomputed.  Columns are grouped by the ζ
    bin (dividers), then n_H, then χ; the strips under the matrix mark each
    column's level, light → dark for low → high, and its ``f_att``.  The
    environment names are in the hover, not on the axis.  Environments with
    ``f_att <= min_f_att`` are blank.  Cell marks: ▲ R > 0 (brighter when
    attenuated), ▼ R < 0 (fainter).

    With the observer table and ``metric="D_med"`` the matrix shows the CR
    dominance D on the fixed 0–1 ``PuOr_r`` scale of the CR/FUV dominance map;
    ``species_order`` pairs it row-for-row with another matrix (KoSens notebook).
    """
    t = pr._theme_fallback(theme)
    if (env_table is None or not len(env_table)
            or ("verdict" in env_table and (env_table["verdict"] == "no data").all())):
        return empty_fig("No graded environment to plot.", theme)
    matrix, ranking, envs, no_atten = impact_ranking(
        env_table, env_info, metric=metric, min_f_att=min_f_att,
        exclude_species=exclude_species, top_n=top_n, species_order=species_order)
    cols = list(matrix.columns)
    rows = (list(species_order) if species_order is not None
            else list(ranking.loc[ranking["shown"], "species"]))
    piv = matrix.reindex(rows)
    if not len(piv):
        return empty_fig("No species left to rank.", theme)
    piv.loc[:, no_atten] = np.nan
    sign = (env_table.pivot_table(index="species", columns="environment", values="response_dex",
                                  aggfunc="first").reindex(index=piv.index, columns=cols)
            .to_numpy(float) if "response_dex" in env_table else np.zeros(piv.shape))
    z = piv.to_numpy(float)
    text = np.full(z.shape, "", dtype=object)
    for i, j in zip(*np.nonzero(np.isfinite(z))):
        mark = "▲" if sign[i, j] > 0 else ("▼" if sign[i, j] < 0 else "")
        text[i, j] = f"{z[i, j]:.2f}{mark}"
    f_att = [e.get("f_att", np.nan) for e in envs]
    log_nh = [e.get("log_N_H", np.nan) for e in envs]
    labels = [lab + (f"<br>f_att {fa:.2f}" if np.isfinite(fa) else "")
              + (f"<br>log N_H {nh:.2f}" if np.isfinite(nh) else "")
              + ("<br><i>no attenuation</i>" if na else "")
              for lab, fa, nh, na in zip(env_col_labels(env_info, cols), f_att, log_nh,
                                         no_atten)]
    cbar = {"error_dex": "median |Δ| [dex]",
            "abs_response_dex": "median |R| [dex]",
            "Delta_med": "median Δ [dex]",
            "D_med": "median D"}.get(metric, metric)
    share = metric.startswith("D_")  # bounded CR share: fixed 0-1 scale
    n_sp = len(piv)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[n_sp, 3.6])
    fig.add_trace(go.Heatmap(
        z=z, x=cols, y=[pretty_key(sp, fmt) for sp in piv.index],
        colorscale="PuOr_r" if share else "YlOrRd", zmin=0.0, zmax=1.0 if share else None,
        text=text, texttemplate="%{text}", xgap=2, ygap=2,
        customdata=np.tile(np.asarray(labels, dtype=object), (n_sp, 1)),
        hovertemplate="%{y} · %{customdata}<br>" + cbar + " = %{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text=cbar), tickfont=ps.cbar_tick_font(t["font"]))),
        row=1, col=1)
    # ζ is the outermost grouping: its strip sits at the bottom.
    _add_level_strips(fig, envs, cols, row=2, order=("chi", "n", "zeta"), f_att=True)
    fig.update_yaxes(autorange="reversed", showgrid=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
    for j in range(1, len(envs)):
        if envs[j]["ranges"].get("ζ") != envs[j - 1]["ranges"].get("ζ"):
            fig.add_vline(x=j - 0.5, line=dict(color=t["font"], width=2))
    title = ("CR dominance D per environment  ·  1 = CR only, 0 = FUV only" if share
             else "Attenuation impact per environment  ·  ▲ R > 0  ▼ R < 0")
    return _style(fig, t, title, max(420, 220 + 30 * n_sp))


def fig_env_picker(env_info, labels, selected, *, theme=None):
    """The strips of the impact matrix alone, as a clickable environment legend.

    Same colours, rows and column order as that figure; the ``selected`` column is
    framed.  ``labels`` are the environments that can be picked.
    """
    t = pr._theme_fallback(theme)
    envs = _zeta_outer_envs(env_info, set(labels))
    cols = [e["label"] for e in envs]
    fig = make_subplots(rows=1, cols=1)
    n_rows = _add_level_strips(fig, envs, cols, row=1, order=("chi", "n", "zeta"), f_att=True)
    for j in range(1, len(envs)):
        if envs[j]["ranges"].get("ζ") != envs[j - 1]["ranges"].get("ζ"):
            fig.add_vline(x=j - 0.5, line=dict(color=t["font"], width=2))
    if selected in cols:
        j = cols.index(selected)
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=0.0, y1=1.0, yref="paper",
                      line=dict(color="#d62728", width=3))
    fig = _style(fig, t, "", 40 + 30 * n_rows)
    fig.update_layout(margin=dict(l=80, r=30, t=10, b=10))
    return fig


def fig_probe_scope(scope, *, response_floor_dex=0.1, theme=None, fmt=None):
    """KoSens ``plot_probe_scope_3d``: clean inside one box (x) vs the same in every box (y).

    Two panels, same points, colours and y axis: the χ swing ratio (which sets
    the class) and the density swing ratio (which does not).  Log axes, both
    ratios = 1 as dividers; ``no response`` species are not drawn.  Hover gives
    responding / all boxes and the axis a conditional probe depends on.
    """
    t = pr._theme_fallback(theme)
    pts = scope[scope["scope"] != "no response"] if scope is not None and len(scope) else None
    if pts is None or not len(pts):
        return empty_fig("No responding box: no probe scope to draw.", theme)
    panels = (("within_ratio", "χ"), ("within_n_ratio", "n_H"))
    vals = pts[["within_ratio", "within_n_ratio", "between_ratio"]].to_numpy(float)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    lo = min(vals.min() / 1.5, 0.5) if vals.size else 0.1
    hi = max(vals.max() * 1.5, 2.0) if vals.size else 10.0
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=[f"within-box {name} swing" for _c, name in panels])
    for col, (xcol, xname) in enumerate(panels, 1):
        for sc in SCOPE_ORDER[:3]:
            sub = pts[pts["scope"] == sc]
            if not len(sub):
                continue
            names = [pretty_key(sp, fmt) for sp in sub["species"]]
            hover = [f"{nm}<br>{sc}" + (f" (needs {dep})" if dep else "")
                     + f"<br>responding boxes {nr}/{nb}"
                     for nm, dep, nr, nb in zip(names, sub["conditional_on"],
                                                sub["n_boxes_responding"], sub["n_boxes"])]
            is_probe = sc.endswith("probe") and sc != "not a probe"
            fig.add_trace(go.Scatter(
                # A zero ratio has no log: pinned just inside the lower edge.
                x=sub[xcol].clip(lower=lo * 1.3), y=sub["between_ratio"].clip(lower=lo * 1.3),
                mode="markers+text" if is_probe else "markers",
                text=[f"{nm} ({nr}/{nb}{', ' + dep if dep else ''})"
                      for nm, dep, nr, nb in zip(names, sub["conditional_on"],
                                                 sub["n_boxes_responding"], sub["n_boxes"])],
                textposition="top right", textfont=dict(size=11),
                name=f"{sc} ({len(sub)})", legendgroup=sc, showlegend=(col == 1),
                marker=dict(size=11, color=SCOPE_COLORS[sc], line=dict(color="black", width=0.8)),
                customdata=hover, hovertemplate="%{customdata}<br>x = %{x:.2f}, y = %{y:.2f}"
                                                "<extra></extra>"), row=1, col=col)
        fig.add_hline(y=1.0, line=dict(color=t["muted"], dash="dash"), row=1, col=col)
        fig.add_vline(x=1.0, line=dict(color=t["muted"], dash="dash"), row=1, col=col)
        fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)], row=1, col=col,
                         title_text=f"within-box {xname} swing / |R|  (clean inside one "
                                    "environment?)")
        fig.update_yaxes(type="log", range=[np.log10(lo), np.log10(hi)], row=1, col=col)
    fig.update_yaxes(title_text="between-box spread of R / |R|  (same response everywhere?)",
                     row=1, col=1)
    return _style(fig, t, f"Scope of each CR probe  ·  boxes with |R| ≥ {response_floor_dex:g} "
                          "dex  ·  below y = 1: universal, above: conditional, right of "
                          "x = 1 (χ panel): not a probe", 620)


_BOX_SYMBOLS = ["circle", "square", "triangle-up", "diamond", "triangle-down"]


def fig_probe_boxes(env_table, cube_table, species, *, response_floor_dex=0.1, min_f_att=0.0,
                    theme=None, fmt=None):
    """KoSens ``plot_probe_boxes_3d``: every box of one species in the probe plane.

    |R| (x) against the χ swing (y) the box verdict tests, per graded box:
    colour = n_H level, symbol = χ level, size = ζ level.  The black star is the
    whole-cube value.  Boxes below the diagonal are CR-led; the star sits above
    it when the per-box responses differ between environments, which is how a
    species can be a probe in most boxes and ``G0-modulated`` over the cube.
    Boxes below ``response_floor_dex`` are hollow (not counted in the scope).
    """
    t = pr._theme_fallback(theme)
    if env_table is None or not len(env_table) or not species:
        return empty_fig("Pick a species (3-D mode).", theme)
    reg = graded_boxes(env_table, min_f_att)
    sub = reg[reg["species"] == species]
    if not len(sub):
        return empty_fig(f"{species}: no graded box.", theme)
    lv = {c: (_levels(reg[c]) if c in reg.columns else [""]) for c in LEVEL_COLS}
    star = None
    if cube_table is not None and len(cube_table):
        hit = cube_table[cube_table["species"] == species]
        star = hit.iloc[0] if len(hit) else None
    vals = np.concatenate([sub["abs_response_dex"].to_numpy(float),
                           sub["g0_swing_dex"].to_numpy(float)]
                          + ([[star["abs_response_dex"], star["g0_swing_dex"]]]
                             if star is not None else []))
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    lo, hi = (vals.min() / 1.6, vals.max() * 1.6) if vals.size else (0.01, 1.0)
    if hi / lo < 10.0:
        mid = np.sqrt(lo * hi)
        lo, hi = mid / np.sqrt(10.0), mid * np.sqrt(10.0)
    shades = sample_colorscale("Greens", list(np.linspace(0.35, 0.9, max(len(lv["n_level"]), 2))))
    sizes = np.linspace(9.0, 20.0, max(len(lv["zeta_level"]), 1))
    floor = float(response_floor_dex)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi, hi], y=[lo, lo, hi], fill="toself", mode="none",
                             fillcolor="rgba(22,163,74,0.08)", hoverinfo="skip",
                             showlegend=False))
    if lo < floor < hi:
        fig.add_vrect(x0=lo, x1=floor, fillcolor="rgba(100,116,139,0.12)", line_width=0)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", hoverinfo="skip",
                             line=dict(color=t["muted"], dash="dash"), showlegend=False))

    def _lvl(row, col):
        return str(row[col]) if col in row.index else ""
    for _, row in sub.iterrows():
        shade = shades[lv["n_level"].index(_lvl(row, "n_level"))]
        responds = row["abs_response_dex"] >= floor
        fig.add_trace(go.Scatter(
            x=[max(row["abs_response_dex"], lo)], y=[max(row["g0_swing_dex"], lo)],
            mode="markers", showlegend=False,
            marker=dict(size=sizes[lv["zeta_level"].index(_lvl(row, "zeta_level"))],
                        symbol=_BOX_SYMBOLS[lv["chi_level"].index(_lvl(row, "chi_level"))
                                            % len(_BOX_SYMBOLS)] + ("" if responds else "-open"),
                        color=shade, line=dict(color="black" if responds else shade,
                                               width=0.8 if responds else 1.8)),
            hovertemplate=(f"{row['environment']}<br><b>{row['verdict']}</b>"
                           f"<br>|R| = {row['abs_response_dex']:.3f} dex"
                           f"<br>χ swing = {row['g0_swing_dex']:.3f} dex<extra></extra>")))
    if star is not None:
        fig.add_trace(go.Scatter(
            x=[max(star["abs_response_dex"], lo)], y=[max(star["g0_swing_dex"], lo)],
            mode="markers", name=f"whole cube ({star['verdict']})",
            marker=dict(symbol="star", size=24, color="black")))
    # Legend keys: colour = n_H, symbol = χ, size = ζ.
    for i, name in enumerate(lv["n_level"]):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"n_H {name}",
                                 marker=dict(size=11, color=shades[i],
                                             line=dict(color="black", width=0.8))))
    for i, name in enumerate(lv["chi_level"]):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"χ {name}",
                                 marker=dict(size=11, color="white",
                                             symbol=_BOX_SYMBOLS[i % len(_BOX_SYMBOLS)],
                                             line=dict(color="black", width=0.8))))
    for i, name in enumerate(lv["zeta_level"]):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"ζ {name}",
                                 marker=dict(size=sizes[i], color="#b3b3b3",
                                             line=dict(color="black", width=0.8))))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             name=f"|R| < {floor:g} dex (not counted)",
                             marker=dict(size=11, symbol="circle-open", color="#666666",
                                         line=dict(width=1.8))))
    rng = [np.log10(lo), np.log10(hi)]
    fig.update_xaxes(type="log", range=rng, constrain="domain", title_text="|R| [dex]")
    fig.update_yaxes(type="log", range=rng, scaleanchor="x", scaleratio=1, constrain="domain",
                     title_text="swing of R across χ at fixed n_H [dex]")
    n_probe = int((sub["verdict"] == "robust attenuation probe").sum())
    n_resp = int((sub["abs_response_dex"] >= floor).sum())
    return _style(fig, t, f"{pretty_key(species, fmt)}: {n_resp}/{len(sub)} boxes respond, "
                          f"{n_probe} robust attenuation probe", 620)


def fig_response_slices(env_table, species, *, metric="response_dex", min_f_att=0.0,
                        theme=None, fmt=None):
    """KoSens ``plot_response_slices_3d``: where in (n_H, χ, ζ) the response lives.

    One panel per ζ level: an (n_H × χ) heatmap of the per-box ``metric``, value
    written in each box.  ``response_dex`` (signed R) uses a diverging scale
    centred on 0, so sign flips between environments stand out; any other metric
    uses YlOrRd.  ``robust attenuation probe`` boxes are outlined in green; boxes with
    ``f_att <= min_f_att`` are crossed out and blank; ``no data`` boxes are blank.
    """
    t = pr._theme_fallback(theme)
    if env_table is None or not len(env_table) or not species:
        return empty_fig("Pick a species (3-D mode).", theme)
    sub = env_table[env_table["species"] == species]
    if not len(sub):
        return empty_fig(f"{species}: no graded box.", theme)
    lv = {c: (_levels(env_table[c]) if c in env_table.columns else [""]) for c in LEVEL_COLS}
    n_lv, g_lv, z_lv = lv["n_level"], lv["chi_level"], lv["zeta_level"]
    signed = metric == "response_dex"
    vals = sub.loc[sub["verdict"] != "no data", metric].astype(float)
    vmax = (float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0) or 1.0
    fig = make_subplots(rows=1, cols=len(z_lv), shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[f"ζ {z}".strip() for z in z_lv])

    def _lvl(row, col):
        return str(row[col]) if col in row.index else ""
    for k, z in enumerate(z_lv, 1):
        grid = np.full((len(g_lv), len(n_lv)), np.nan)
        marks = []
        for _, row in sub.iterrows():
            if _lvl(row, "zeta_level") != z:
                continue
            i, j = g_lv.index(_lvl(row, "chi_level")), n_lv.index(_lvl(row, "n_level"))
            if "f_att" in row.index and float(row["f_att"]) <= float(min_f_att):
                marks.append((i, j, "off"))
            elif row["verdict"] != "no data":
                grid[i, j] = float(row[metric])
                if row["verdict"] == "robust attenuation probe":
                    marks.append((i, j, "probe"))
        text = np.where(np.isfinite(grid),
                        np.vectorize(lambda v: f"{v:+.2f}" if signed else f"{v:.2f}")(
                            np.nan_to_num(grid)), "")
        text[[i for i, _j, kind in marks if kind == "off"],
             [j for _i, j, kind in marks if kind == "off"]] = "✕"
        fig.add_trace(go.Heatmap(
            z=grid, x=[n or "all" for n in n_lv], y=[g or "all" for g in g_lv], text=text,
            texttemplate="%{text}", colorscale="RdBu_r" if signed else "YlOrRd",
            zmin=-vmax if signed else 0.0, zmax=vmax, xgap=2, ygap=2, showscale=(k == len(z_lv)),
            colorbar=dict(title=dict(text="R [dex]" if signed else f"{metric} [dex]"),
                          tickfont=ps.cbar_tick_font(t["font"])),
            hovertemplate="n_H %{x} · χ %{y}<br>%{z:.3f} dex<extra></extra>"), row=1, col=k)
        for i, j, kind in marks:
            if kind == "probe":
                fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                              line=dict(color=VERDICT_COLORS["robust attenuation probe"], width=4),
                              row=1, col=k)
        fig.update_xaxes(title_text="n_H", showgrid=False, row=1, col=k)
        fig.update_yaxes(showgrid=False, row=1, col=k)
    fig.update_yaxes(title_text="χ", row=1, col=1)
    return _style(fig, t, f"{pretty_key(species, fmt)}  ·  green outline: robust attenuation probe box  ·  "
                          "✕: no attenuation", 420)


# ---------------------------------------------------------------------------
# CO SLED classes (Rosenberg et al. 2015, ApJ 801, 72, Eq. 1)
# ---------------------------------------------------------------------------

# Rest frequencies [GHz] of the CO J -> J-1 lines in alpha.
CO_FREQ_GHZ = {5: 576.268, 6: 691.473, 7: 806.652, 11: 1267.014, 12: 1381.995, 13: 1496.923}


def co_alpha(intensity, unit="jtemp"):
    """alpha = (L11-10 + L12-11 + L13-12) / (L5-4 + L6-5 + L7-6) per model.

    ``intensity`` maps J_upper -> array of line intensities for the six lines. One shared
    distance cancels 4πD², so alpha is a flux ratio. K km/s → flux: F ∝ ν³·(K km/s)
    (I_ν ∝ ν²T and ∫I_ν dν = ν/c ∫I_ν dv); ``jerg`` is already an integrated flux.
    """
    def flux(j):
        v = np.asarray(intensity[j], dtype=float)
        return v * CO_FREQ_GHZ[j] ** 3 if unit == "jtemp" else v
    with np.errstate(divide="ignore", invalid="ignore"):
        return sum(flux(j) for j in (11, 12, 13)) / sum(flux(j) for j in (5, 6, 7))


def co_alpha_class(alpha, bins=(0.33, 0.66)):
    """Class label per model: I (α < 0.33), II (0.33 ≤ α < 0.66), III (α ≥ 0.66); "" if α is not finite."""
    a = np.asarray(alpha, dtype=float)
    lab = np.where(a < bins[0], "I", np.where(a < bins[1], "II", "III"))
    return np.where(np.isfinite(a), lab, "")


def co_alpha_classes(alpha, bins=(0.33, 0.66)):
    """Model counts per class."""
    lab = co_alpha_class(alpha, bins)
    return {c: int(np.sum(lab == c)) for c in ("I", "II", "III")}


def fig_co_alpha_classes(counts, *, bins=(0.33, 0.66), theme=None, height=560):
    """Bar plot of models per Rosenberg class."""
    t = pr._theme_fallback(theme)
    fig = go.Figure(go.Bar(
        x=[f"I<br>α < {bins[0]}", f"II<br>{bins[0]} ≤ α < {bins[1]}", f"III<br>α ≥ {bins[1]}"],
        y=[counts["I"], counts["II"], counts["III"]], text=list(counts.values()),
        textposition="outside", marker_color=["#1f77b4", "#ff7f0e", "#d62728"]))
    fig.update_yaxes(title_text="Number of models")
    _style(fig, t, "CO SLED classes", height)
    return fig.update_layout(margin=dict(l=55, r=10, t=60, b=50))


def observer_error_table(pixels, ref_cubes, edges, *, min_abundance=1e-15,
                         ambiguous_dex=2.0, min_d=0.75, max_frac_zero=0.5):
    """Per (species, environment box) ingredients of an observer's ζ error.

    KoSens ``observer_error_table_3d``: fitting with constant-ζ models at an FUV
    field off by δ = log10(χ_adopt/χ_true) gives
    log10(ζ_fit/ζ_true) = −ε_att − (S_χ/S_ζ)·δ.  Per cell: Δ (``shift_dex``,
    unmatched = 0), Δ_lin = −P/S_ζ with P = log10(X_att/X_const) (= R), lin_ratio
    = Δ/Δ_lin (only on matched 0 < Δ ≤ ``ambiguous_dex`` cells), S_ζ
    (``slope_dex``), S_χ (central differences of the reference cube along χ),
    r = S_χ/S_ζ and D = |S_ζ|/(|S_ζ|+|S_χ|).  Slope statistics skip cells with
    X_const < ``min_abundance``.  Each statistic is median / p16 / p84 over the
    box; ``reliable`` = D_med ≥ ``min_d`` and frac_zero ≤ ``max_frac_zero``.
    """
    pix = _ensure_cells(pixels)
    pix = pix[pix["species"].isin(list(ref_cubes))].reset_index(drop=True)
    if not len(pix):
        return pd.DataFrame()
    s_chi = np.full(len(pix), np.nan)
    for (sp, k), idx in pix.groupby(["species", "i_n"]).indices.items():
        plane = cube_plane(ref_cubes[sp], k)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_x = np.where(plane["grid"] > 0, np.log10(plane["grid"]), np.nan)
            log_f = np.log10(plane["fuv"])
        slope = _log_intensity_slope(log_x.T, log_f.T).T  # rows of the plane run along χ
        rows = pix.iloc[idx]
        s_chi[idx] = slope[rows["iy"].to_numpy(int), rows["ix"].to_numpy(int)]
    pix["S_chi"] = s_chi
    pix["S_zeta"] = pix["slope_dex"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pix["P"] = np.log10(pix["value_atten"] / pix["value_ref"])
        pix["Delta_lin"] = -pix["P"] / pix["S_zeta"]
        pix["r"] = pix["S_chi"] / pix["S_zeta"]
        pix["D"] = pix["S_zeta"].abs() / (pix["S_zeta"].abs() + pix["S_chi"].abs())
    bright = pix["value_ref"] >= min_abundance
    matched = (pix["shift_dex"] > 0) & (pix["shift_dex"] <= ambiguous_dex)
    with np.errstate(divide="ignore", invalid="ignore"):
        pix["lin_ratio"] = np.where(bright & matched & (pix["Delta_lin"] > 0),
                                    pix["shift_dex"] / pix["Delta_lin"], np.nan)
    for col in ("S_chi", "S_zeta", "r", "D"):
        pix.loc[~bright, col] = np.nan
    pix["is_zero"] = pix["shift_dex"] == 0
    pix["is_ambiguous"] = pix["shift_dex"].abs() > ambiguous_dex
    pix["is_faint"] = ~bright
    labels, env_info = assign_environments(pix, edges)
    pix["environment"] = labels

    def _q(p):
        return lambda s: float(np.nanpercentile(s, p)) if np.isfinite(s).any() else np.nan

    agg = {}
    for name, col in (("Delta", "shift_dex"), ("Delta_lin", "Delta_lin"),
                      ("lin_ratio", "lin_ratio"), ("S_zeta", "S_zeta"), ("S_chi", "S_chi"),
                      ("r", "r"), ("D", "D")):
        for p in (50, 16, 84):
            agg[f"{name}_{'med' if p == 50 else f'p{p}'}"] = (col, _q(p))
    agg.update(frac_zero=("is_zero", "mean"), frac_ambiguous=("is_ambiguous", "mean"),
               frac_faint=("is_faint", "mean"), n_cells=("shift_dex", "size"))
    table = pix.groupby(["species", "environment"]).agg(**agg).reset_index()
    order = {e["label"]: i for i, e in enumerate(env_info)}
    cells = pix.drop_duplicates(["i_n", "iy", "ix"])
    for col in ("f_att", "log_N_H"):
        if col in cells.columns:
            table[col] = table["environment"].map(cells.groupby("environment")[col].median())
    table["reliable"] = (table["D_med"] >= min_d) & (table["frac_zero"] <= max_frac_zero)
    return (table.assign(_o=table["environment"].map(order))
            .sort_values(["species", "_o"]).drop(columns="_o").reset_index(drop=True))


def observer_compact(table):
    """Every box, only the columns an observer needs (KoSens ``_compact_observer_table``).

    Δ and r as median with ``+`` (p84 − median) and ``−`` (median − p16) offsets;
    ``S_zeta_med`` propagates an abundance error into ζ (σ_logX / |S_ζ|).
    """
    out = table[["species", "environment"]].copy()
    for name in ("Delta", "r"):
        out[f"{name}_med"] = table[f"{name}_med"]
        out[f"{name}_plus"] = table[f"{name}_p84"] - table[f"{name}_med"]
        out[f"{name}_minus"] = table[f"{name}_med"] - table[f"{name}_p16"]
    for col in ("D_med", "S_zeta_med", "lin_ratio_med", "frac_zero", "f_att", "log_N_H",
                "reliable"):
        if col in table.columns:
            out[col] = table[col]
    return out.round(3)


def observer_table_tex(table):
    """LaTeX rows ``species & box & Δ & r & D & frac_zero`` of the reliable boxes.

    Δ and r as median^{+(p84 − median)}_{−(median − p16)} (KoSens ``_write_condensed_tex``).
    """
    def pm(row, name):
        med, lo, hi = row[f"{name}_med"], row[f"{name}_p16"], row[f"{name}_p84"]
        if not np.isfinite(med):
            return "--"
        return f"${med:.2f}^{{+{hi - med:.2f}}}_{{-{med - lo:.2f}}}$"

    lines = [r"\begin{tabular}{llcccc}", r"\hline",
             r"Species & box & $\epsilon_\mathrm{att}$ [dex] & $S_{\chi_0}/S_{\zeta_H}$ & $D$ "
             r"& $f_{\Delta=0}$ \\", r"\hline"]
    for _, row in table[table["reliable"]].iterrows():
        lines.append(f"{row['species']} & {row['environment']} & {pm(row, 'Delta')} & "
                     f"{pm(row, 'r')} & {row['D_med']:.2f} & {row['frac_zero']:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
