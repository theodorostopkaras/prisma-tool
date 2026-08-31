#!/usr/bin/env python3
"""Diagnose chemistry-only + overlay pairing against real directories.

Usage:
    python diag_chem.py /path/to/primary_chem_dir /path/to/overlay_chem_dir [--recursive]

Prints exactly what the app computes when NO HDF5 grid is loaded:
  * whether the chemistry-only grid bootstraps,
  * whether the axis / colormap controls would be shown,
  * whether primary and overlay resolve to DIFFERENT files at every grid point.
"""
import os
import sys

import app as ap


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    recursive = '--recursive' in sys.argv
    if len(args) < 1:
        print(__doc__)
        return 1
    primary_dir = args[0]
    overlay_dir = args[1] if len(args) > 1 else None

    print('=' * 70)
    print('Running app module from :', os.path.abspath(ap.__file__))
    print('=' * 70)

    ap.clear_main_grid()
    ap.clear_chem()

    print(f'\n[1] scan_chem({primary_dir!r}, recursive={recursive})')
    ch = ap.scan_chem(primary_dir, recursive=recursive)
    print('    files scanned      :', len(ch['files']))
    print('    species            :', ch['species'][:10], '...' if len(ch['species']) > 10 else '')
    print('    from HDF5 metadata :', ch.get('n_from_hdf5'), ' skipped:', ch.get('n_skipped'))

    print('\n[2] bootstrap_grid_from_chem()')
    boot = ap.bootstrap_grid_from_chem()
    print('    bootstrapped       :', boot)
    print('    _grid.chem_only    :', ap._grid.get('chem_only'))
    print('    _grid.has_hdf5     :', ap._grid.get('has_hdf5'))
    if not boot:
        print('    !! Bootstrap FAILED -> no sliders/controls will show.')
        print('       (Is a SIMLINE or HDF5 grid still loaded? Clear it first.)')
        return 2
    varying = {k: v for k, v in ap._grid['axis_tokens'].items() if len(v) > 1}
    print('    varying axes       :', varying or '(single model point)')

    print('\n[3] apply_plot_theme -> axis/colormap controls visibility')
    out = ap.apply_plot_theme('light', True, 0)
    axis_style = out[4]
    visible = axis_style.get('display') != 'none'
    print('    controls visible   :', visible, f'(display={axis_style.get("display")!r})')

    if overlay_dir:
        print(f'\n[4] scan_chem_overlay({overlay_dir!r}, recursive={recursive})')
        ov = ap.scan_chem_overlay(overlay_dir, recursive=recursive)
        print('    overlay files      :', len(ov['files']))
        print('    for_primary pairs  :', len(ap._chem_overlay.get('for_primary') or {}))

        print('\n[5] primary vs overlay resolution at every grid point')
        axes = ap._grid['axis_tokens']
        n_bad = n_same = n_ok = 0
        for tok in ap._chem['files']:
            vals = []
            for d, p in enumerate(ap.PARAM_DEFS):
                try:
                    vals.append(axes[p['key']].index(tok[d]))
                except ValueError:
                    vals.append(0)
            cf = ap.chem_file(vals)
            cof = ap.chem_overlay_file(vals)
            if cof is None:
                n_bad += 1
                print(f'    NO OVERLAY  tok={tok}  primary={os.path.basename(cf or "?")}')
            elif ap._same_data_path(cf, cof):
                n_same += 1
                print(f'    SAME FILE   tok={tok}  -> {os.path.basename(cf)}')
            else:
                n_ok += 1
        print(f'\n    OK (different files): {n_ok}   '
              f'no-overlay: {n_bad}   same-file: {n_same}')
        if n_ok and not n_same and not n_bad:
            print('    => Overlay pairing is CORRECT for all points.')
    else:
        print('\n[4] (no overlay directory given; skipping overlay checks)')

    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
