#!/usr/bin/env python3
"""Direct prisma-tool vs KoSens3D probe comparison on a native PDR grid.

Compares clump-integrated X and CRIR probe scores for H3O+, SO2, and SO
on the same HDF5 models.  Isolates three layers:

1. Abundance: prisma dens-column X vs KoSens named-field X (n_h, n_h2, n_sp)
2. Scoring: prisma ``probe_score`` vs KoSens ``albertsson_probe_score`` on
   the same cubes
3. Ranking / heatmap: native-node score matrices (no 60³ interpolation)

KoSens notebooks often interpolate to (60, 60, 60) and use min_abundance=1e-15.
This script scores the *native* cube with both the UI default (1e-9) and the
notebook cutoff so those effects are visible separately.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KOSENS_SRC = os.path.abspath(os.path.join(ROOT, '..', 'KoSens3D', 'src'))
DEFAULT_GRID = os.path.expanduser(
    '/home/teotopkaras/Desktop/Projects/Models/new_teo_grid/pdrgrid_hdf5'
)
SPECIES = ('H3O+', 'SO2', 'SO')
KOSENS_DENS = {'H3O+': 'n_h3op', 'SO2': 'n_so2', 'SO': 'n_so'}
PC_TO_CM = 3.08567758128e18
SCORE_KW = dict(
    include_ratios=False,
    min_points=4,
    min_log_crir_span=0.5,
    plateau_slope=float(np.log10(3.0)),
    variation_r_cap=float(np.log10(100.0)),
    good_score_threshold=0.3,
    crir_regimes=None,
)


def _header(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


def _rel_diff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    if not np.any(both):
        return np.nan
    denom = np.maximum(np.maximum(np.abs(a[both]), np.abs(b[both])), 1e-300)
    return float(np.max(np.abs(a[both] - b[both]) / denom))


def kosens_integrate_x(radius_pc, n_h, n_h2, n_sp):
    """KoSens ``rel_abund``: ∫ 4π r² n_sp dr / ∫ 4π r² (n_H + 2 n_H2) dr."""
    radius = np.asarray(radius_pc, dtype=float)
    n_h = np.asarray(n_h, dtype=float)
    n_h2 = np.asarray(n_h2, dtype=float)
    n_sp = np.asarray(n_sp, dtype=float)
    n = min(radius.size, n_h.size, n_h2.size, n_sp.size)
    if n == 0:
        return np.nan
    radius = radius[:n]
    flipped_r = radius[::-1]
    radius_cm = np.insert(flipped_r, 0, 0.0) * PC_TO_CM

    def _pad(prof):
        flipped = prof[:n][::-1]
        return np.insert(flipped, 0, flipped[0])

    n_h_c = _pad(n_h)
    n_h2_c = _pad(n_h2)
    n_sp_c = _pad(n_sp)
    total_h = n_h_c + 2.0 * n_h2_c
    shell = 4.0 * np.pi * radius_cm ** 2
    denom = float(np.trapezoid(shell * total_h, radius_cm))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.trapezoid(shell * n_sp_c, radius_cm) / denom)


def heatmap_z(slice_rows, species, regime, n_levels, fuv_levels):
    """Native (n, FUV) score matrix used by prisma heatmaps."""
    z = np.full((len(n_levels), len(fuv_levels)), np.nan)
    for row in slice_rows:
        if row.get('Species') != species or row.get('Regime') != regime:
            continue
        n, fuv = row.get('n'), row.get('fuv')
        if not (np.isfinite(n) and np.isfinite(fuv)):
            continue
        iy = int(np.argmin(np.abs(n_levels - n)))
        ix = int(np.argmin(np.abs(fuv_levels - fuv)))
        z[iy, ix] = row.get('probe_score', np.nan)
    return z


def fmt_matrix(z, n_levels, fuv_levels, title):
    print(f'\n{title}')
    header = 'n\\G0'.rjust(10) + ''.join(f'{v:10.3g}' for v in fuv_levels)
    print(header)
    for i, n in enumerate(n_levels):
        cells = []
        for j in range(len(fuv_levels)):
            val = z[i, j]
            cells.append(f'{val:10.3f}' if np.isfinite(val) else f'{"nan":>10}')
        print(f'{n:10.3g}' + ''.join(cells))


def load_prisma(grid_dir):
    sys.path.insert(0, ROOT)
    import app as prisma  # noqa: WPS433  (side-effect: Dash app)

    print(f'Loading prisma-tool grid from {grid_dir}')
    t0 = time.time()
    grid = prisma.scan_directory(grid_dir)
    print(f'  {grid["n_files"]} HDF5 files in {time.time() - t0:.1f}s')
    axis = grid.get('axis_tokens') or {}
    for key in ('density', 'mass', 'fuv', 'metal', 'crir', 'atten'):
        toks = axis.get(key) or []
        phys = [prisma._physical_param_value(key, t) for t in toks]
        print(f'  {key:8s}  {len(toks)} values  {phys}')
    idx = grid['species_idx']
    for sp in SPECIES + ('H', 'H2'):
        print(f'  species_idx[{sp!r}] = {idx.get(sp)}')
    fmap = prisma._field_map
    for key in ('radius', 'n_h', 'n_h2', 'n_h3op', 'n_so2', 'n_so', 'protdens'):
        print(f'  field_map[{key!r}] = {fmap.get(key)}')
    return prisma, grid


def prisma_cubes(prisma, species):
    sliders = [0] * prisma.N_PARAMS
    t0 = time.time()
    cubes = prisma._build_probe_grids_3d('rel_abund', list(species), sliders, None)
    print(f'Prisma native cubes built in {time.time() - t0:.1f}s')
    ref = next(iter(cubes.values()))
    print(f'  cube shape {ref["grid"].shape}  keys {sorted(cubes[species[0]])}')
    return cubes


def kosens_named_cubes(prisma, species):
    """Rebuild the native cube with KoSens named fields (n_h, n_h2, n_sp)."""
    sliders = [0] * prisma.N_PARAMS
    tok_to_ijk, shape, phys, orders = prisma._build_probe_index(sliders)
    nz, ny, nx = shape
    cubes = {sp: np.full((nz, ny, nx), np.nan, dtype=float) for sp in species}
    missing = {sp: 0 for sp in species}
    n_ok = 0
    t0 = time.time()
    for tokens, (iz, iy, ix) in tok_to_ijk.items():
        path = prisma._grid['files'].get(tokens)
        if not path:
            continue
        with __import__('h5py').File(path, 'r') as hf:
            radius = prisma._read_field(hf, 'radius')
            n_h = prisma._read_field(hf, 'n_h')
            n_h2 = prisma._read_field(hf, 'n_h2')
            if radius is None or n_h is None or n_h2 is None:
                continue
            n_ok += 1
            for sp in species:
                n_sp = prisma._read_field(hf, KOSENS_DENS[sp])
                if n_sp is None:
                    missing[sp] += 1
                    continue
                cubes[sp][iz, iy, ix] = kosens_integrate_x(radius, n_h, n_h2, n_sp)
    print(f'KoSens named-field cubes built in {time.time() - t0:.1f}s '
          f'({n_ok} models, missing fields {missing})')
    return prisma._probe_pack_cubes(cubes, shape, phys, orders)


def compare_cubes(label_a, cubes_a, label_b, cubes_b, species):
    _header(f'Abundance cubes: {label_a} vs {label_b}')
    rows = []
    for sp in species:
        a = np.asarray(cubes_a[sp]['grid'], dtype=float)
        b = np.asarray(cubes_b[sp]['grid'], dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        only_a = np.isfinite(a) & ~np.isfinite(b)
        only_b = np.isfinite(b) & ~np.isfinite(a)
        rel = _rel_diff(a, b)
        abs_max = float(np.nanmax(np.abs(a - b))) if np.any(both) else np.nan
        print(f'{sp:6s}  finite {label_a}={np.isfinite(a).sum()}  '
              f'{label_b}={np.isfinite(b).sum()}  both={both.sum()}  '
              f'only_{label_a}={only_a.sum()}  only_{label_b}={only_b.sum()}')
        print(f'       max |ΔX|={abs_max:.3e}  max rel Δ={rel:.3e}  '
              f'X[{label_a}] median={np.nanmedian(a):.3e}  '
              f'X[{label_b}] median={np.nanmedian(b):.3e}')
        if np.any(both):
            # worst relative disagreements
            denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
            rel_map = np.where(both, np.abs(a - b) / denom, 0.0)
            worst = np.unravel_index(int(np.nanargmax(rel_map)), a.shape)
            iz, iy, ix = worst
            crir = float(cubes_a[sp]['crir_values'][iz, iy, ix])
            n = float(cubes_a[sp]['densities'][iz, iy, ix])
            fuv = float(cubes_a[sp]['fuv_values'][iz, iy, ix])
            print(f'       worst rel @ n={n:.3g}  G0={fuv:.3g}  ζ={crir:.3g}  '
                  f'{label_a}={a[worst]:.6e}  {label_b}={b[worst]:.6e}')
        rows.append((sp, rel, abs_max))
    return rows


def score_prisma(cubes, min_abundance, species):
    import probe_ranking as pr

    return pr.compute_probe_ranking(
        cubes,
        species_list=list(species),
        min_abundance=min_abundance,
        **SCORE_KW,
    )


def score_kosens(cubes, min_abundance, species):
    sys.path.insert(0, KOSENS_SRC)
    from kosens3d.grid.probe_ranking import compute_crir_probe_ranking_3d

    return compute_crir_probe_ranking_3d(
        cubes,
        species_list=list(species),
        min_abundance=min_abundance,
        return_slice_scores=True,
        verbose=False,
        **SCORE_KW,
    )


def _kosens_slices(ranking):
    df = ranking.get('slice_scores_df')
    if df is None:
        return []
    return df.to_dict('records')


def compare_scores(label_a, slices_a, label_b, slices_b, species, n_levels, fuv_levels):
    _header(f'Slice scores: {label_a} vs {label_b}')
    key = lambda r: (r.get('Species'), r.get('Regime'),
                     float(r['n']) if np.isfinite(r.get('n', np.nan)) else None,
                     float(r['fuv']) if np.isfinite(r.get('fuv', np.nan)) else None)
    map_a = {key(r): r for r in slices_a}
    map_b = {key(r): r for r in slices_b}
    keys = sorted(set(map_a) | set(map_b), key=lambda k: (str(k[0]), str(k[1]), k[2] or 0, k[3] or 0))
    disagree = []
    n_both = n_nan_a = n_nan_b = 0
    for k in keys:
        ra, rb = map_a.get(k), map_b.get(k)
        sa = ra['probe_score'] if ra else np.nan
        sb = rb['probe_score'] if rb else np.nan
        fa, fb = np.isfinite(sa), np.isfinite(sb)
        if fa and fb:
            n_both += 1
            if abs(sa - sb) > 1e-8:
                disagree.append((k, sa, sb, abs(sa - sb)))
        elif fa and not fb:
            n_nan_b += 1
            disagree.append((k, sa, sb, np.nan))
        elif fb and not fa:
            n_nan_a += 1
            disagree.append((k, sa, sb, np.nan))
    print(f'  paired finite scores: {n_both}')
    print(f'  finite only in {label_a}: {n_nan_b}   only in {label_b}: {n_nan_a}')
    print(f'  |ΔS| > 1e-8 or NaN mismatch: {len(disagree)}')
    disagree.sort(key=lambda t: -(t[3] if np.isfinite(t[3]) else 1.0))
    for k, sa, sb, d in disagree[:12]:
        print(f'    {k[0]:6s}  {k[1]:16s}  n={k[2]:.3g}  G0={k[3]:.3g}  '
              f'{label_a}={sa:.4f}  {label_b}={sb:.4f}  Δ={d}')
    regimes = sorted({r.get('Regime') for r in slices_a} | {r.get('Regime') for r in slices_b})
    for sp in species:
        for regime in regimes:
            za = heatmap_z(slices_a, sp, regime, n_levels, fuv_levels)
            zb = heatmap_z(slices_b, sp, regime, n_levels, fuv_levels)
            fmt_matrix(za, n_levels, fuv_levels, f'Heatmap {sp} / {regime}  [{label_a}]')
            fmt_matrix(zb, n_levels, fuv_levels, f'Heatmap {sp} / {regime}  [{label_b}]')
            both = np.isfinite(za) & np.isfinite(zb)
            if np.any(both):
                print(f'  max |ΔS| on heatmap = {np.nanmax(np.abs(za - zb)):.4f}  '
                      f'max rel = {_rel_diff(za, zb):.3e}')
    return disagree


def sample_curve_scores(cubes, species, min_abundance):
    """Score a few native CRIR curves with both functions."""
    import probe_ranking as pr
    sys.path.insert(0, KOSENS_SRC)
    from kosens3d.grid.probe_ranking import albertsson_probe_score

    _header(f'Per-curve scorer check  (min_abundance={min_abundance:g})')
    ref = cubes[species[0]]
    crir_mesh = np.asarray(ref['crir_values'])
    n_mesh = np.asarray(ref['densities'])
    fuv_mesh = np.asarray(ref['fuv_values'])
    crir_axis = pr._coord_axis_index(crir_mesh)
    n_diff = 0
    n_tot = 0
    for sp in species:
        grid = np.asarray(cubes[sp]['grid'])
        for index, _env in pr._iter_crir_slices(grid.shape, crir_axis):
            curve = np.asarray(grid[index], dtype=float).ravel()
            crir = np.asarray(crir_mesh[index], dtype=float).ravel()
            n = float(n_mesh[index].ravel()[0])
            fuv = float(fuv_mesh[index].ravel()[0])
            for lo, hi in ((None, None), (None, 1e-16), (1e-16, None)):
                cr = None if lo is None and hi is None else (
                    (np.nanmin(crir) if lo is None else lo,
                     np.nanmax(crir) if hi is None else hi)
                )
                a = pr.probe_score(crir, curve, min_abundance=min_abundance, crir_range=cr)
                b = albertsson_probe_score(crir, curve, min_abundance=min_abundance, crir_range=cr)
                n_tot += 1
                sa, sb = a['probe_score'], b['probe_score']
                if np.isfinite(sa) and np.isfinite(sb):
                    if abs(sa - sb) > 1e-10:
                        n_diff += 1
                        print(f'  SCORE DIFF {sp} n={n:.3g} G0={fuv:.3g} '
                              f'range={cr}  prisma={sa:.6f} kosens={sb:.6f}')
                elif np.isfinite(sa) != np.isfinite(sb):
                    n_diff += 1
                    print(f'  NaN mismatch {sp} n={n:.3g} G0={fuv:.3g} '
                          f'range={cr}  prisma={sa} ({a.get("status")})  '
                          f'kosens={sb}  n_pts={a["n_points"]}/{b["n_points"]}')
    print(f'  checked {n_tot} curves; disagreements: {n_diff}')


def check_one_model_fields(prisma):
    """Confirm dens-column H/H2/species match KoSens named fields."""
    _header('Single-model field check (dens columns vs named n_*)')
    path = next(iter(prisma._grid['files'].values()))
    model = prisma.get_model(path)
    dens = np.asarray(model['dens'], dtype=float)
    idx = prisma._grid['species_idx']
    with __import__('h5py').File(path, 'r') as hf:
        named = {
            'n_h': prisma._read_field(hf, 'n_h'),
            'n_h2': prisma._read_field(hf, 'n_h2'),
            'n_h3op': prisma._read_field(hf, 'n_h3op'),
            'n_so2': prisma._read_field(hf, 'n_so2'),
            'n_so': prisma._read_field(hf, 'n_so'),
            'radius': prisma._read_field(hf, 'radius'),
        }
    print(f'  file {os.path.basename(path)}  dens shape {dens.shape}')
    pairs = [
        ('H', 'n_h'), ('H2', 'n_h2'),
        ('H3O+', 'n_h3op'), ('SO2', 'n_so2'), ('SO', 'n_so'),
    ]
    for sp, key in pairs:
        col = dens[:, idx[sp]] if sp in idx else None
        arr = named[key]
        if col is None or arr is None:
            print(f'  {sp:6s} vs {key:8s}: missing col={col is None} named={arr is None}')
            continue
        n = min(col.size, arr.size)
        rel = _rel_diff(col[:n], arr[:n])
        print(f'  {sp:6s} vs {key:8s}: max rel Δ={rel:.3e}  '
              f'median col={np.nanmedian(col):.3e} named={np.nanmedian(arr):.3e}')
    xs_prisma = prisma._integrated_x_for_species(model, list(SPECIES))
    print('  prisma X:', {k: f'{v:.6e}' for k, v in xs_prisma.items()})
    xs_k = {}
    for sp in SPECIES:
        xs_k[sp] = kosens_integrate_x(
            named['radius'], named['n_h'], named['n_h2'], named[KOSENS_DENS[sp]],
        )
    print('  kosens X:', {k: f'{v:.6e}' for k, v in xs_k.items()})


def summarize_ranking(label, ranking, species):
    summary = ranking.get('summary') or ranking.get('summary_df')
    if summary is None:
        return
    _header(f'{label} summary')
    rows = summary.to_dict('records') if hasattr(summary, 'to_dict') else list(summary)
    print(f'{"Species":8s} {"Regime":18s} {"median":>8} {"mean":>8} {"n_env":>6}')
    for row in rows:
        if row.get('Species') not in species:
            continue
        med = row.get('Median_Probe_Score', np.nan)
        mean = row.get('Mean_Probe_Score', np.nan)
        n_env = row.get('N_Environments', '')
        print(f'{row.get("Species"):8s} {str(row.get("Regime")):18s} '
              f'{float(med):8.3f} {float(mean):8.3f} {n_env!s:>6}')


def interpolate_cubes(cubes, target_shape=(60, 60, 60)):
    """KoSens-style 3-D resampling to match the notebook sizing."""
    import grid_interp as gi

    ref = next(iter(cubes.values()))
    n_phys = np.asarray(ref['densities'], dtype=float)[0, 0, :]
    fuv_phys = np.asarray(ref['fuv_values'], dtype=float)[0, :, 0]
    crir_phys = np.asarray(ref['crir_values'], dtype=float)[:, 0, 0]

    n_ax = None
    fuv_ax = None
    out = {}
    for name, gdata in cubes.items():
        grid = np.asarray(gdata['grid'], dtype=float)
        grid_out, final_x, final_y, final_z = gi.resample_grid_3d_kosens(
            grid,
            x_phys=n_phys,
            y_phys=fuv_phys,
            z_phys=crir_phys,
            target_shape=target_shape,
            interpolation_method='linear',
            smoothing_order=2,
            clip_to_bounds=False,
            log_values=True,
        )
        if n_ax is None:
            n_ax = final_x
            fuv_ax = final_y

        z_mesh, y_mesh, x_mesh = np.meshgrid(
            final_z, final_y, final_x, indexing='ij',
        )
        out[name] = {
            'grid': grid_out,
            'densities': x_mesh,
            'fuv_values': y_mesh,
            'crir_values': z_mesh,
        }
    print(f'KoSens-style resampled cubes to {target_shape}')
    return out, np.asarray(n_ax, dtype=float), np.asarray(fuv_ax, dtype=float)


def run(grid_dir=DEFAULT_GRID):
    if not os.path.isdir(grid_dir):
        print(f'Grid directory not found: {grid_dir}')
        return 2

    prisma, _grid = load_prisma(grid_dir)
    check_one_model_fields(prisma)
    p_cubes = prisma_cubes(prisma, SPECIES)
    k_cubes = kosens_named_cubes(prisma, SPECIES)
    compare_cubes('prisma', p_cubes, 'kosens_X', k_cubes, SPECIES)

    ref = p_cubes[SPECIES[0]]
    n_levels = np.unique(np.asarray(ref['densities'], dtype=float))
    fuv_levels = np.unique(np.asarray(ref['fuv_values'], dtype=float))
    n_levels = np.sort(n_levels[np.isfinite(n_levels) & (n_levels > 0)])
    fuv_levels = np.sort(fuv_levels[np.isfinite(fuv_levels) & (fuv_levels > 0)])
    print(f'\nNative n_H levels: {n_levels}')
    print(f'Native G0 levels:  {fuv_levels}')

    for min_ab in (1e-9, 1e-15):
        sample_curve_scores(p_cubes, SPECIES, min_ab)
        _header(f'Full ranking on prisma cubes  min_abundance={min_ab:g}')
        p_rank = score_prisma(p_cubes, min_ab, SPECIES)
        k_rank = score_kosens(p_cubes, min_ab, SPECIES)
        summarize_ranking(f'prisma (min_ab={min_ab:g})', p_rank, SPECIES)
        summarize_ranking(f'KoSens scorer on prisma cubes (min_ab={min_ab:g})', k_rank, SPECIES)
        compare_scores(
            'prisma', p_rank['slice_scores'],
            'kosens', _kosens_slices(k_rank),
            SPECIES, n_levels, fuv_levels,
        )

        _header(f'Ranking on KoSens named-field cubes  min_abundance={min_ab:g}')
        p_on_k = score_prisma(k_cubes, min_ab, SPECIES)
        compare_scores(
            'prisma_on_prismaX', p_rank['slice_scores'],
            'prisma_on_kosensX', p_on_k['slice_scores'],
            SPECIES, n_levels, fuv_levels,
        )

    _header('KoSens-style resample to 60³, then score (prisma vs KoSens on same cubes)')
    fine, n_nat, fuv_nat = interpolate_cubes(p_cubes, (60, 60, 60))
    for min_ab in (1e-9, 1e-15):
        p_fine = score_prisma(fine, min_ab, SPECIES)
        k_fine = score_kosens(fine, min_ab, SPECIES)
        summarize_ranking(f'prisma scorer on 60³ cubes (min_ab={min_ab:g})', p_fine, SPECIES)
        summarize_ranking(f'KoSens scorer on 60³ cubes (min_ab={min_ab:g})', k_fine, SPECIES)
        compare_scores(
            f'prisma 60³ (min_ab={min_ab:g})', p_fine['slice_scores'],
            f'KoSens 60³ (min_ab={min_ab:g})', _kosens_slices(k_fine),
            SPECIES, n_nat, fuv_nat,
        )

    print('\nDone.')
    return 0


if __name__ == '__main__':
    grid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRID
    raise SystemExit(run(grid))
