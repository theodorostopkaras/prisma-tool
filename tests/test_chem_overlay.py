"""Chemistry-only overlay pairing (no HDF5 structure grid required)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as ap


PHYS = dict(density=1e4, mass=1.0, fuv=1.0, metal=1.0, crir=2e-17)


def _boot_chem_only(primary_files, primary_phys, overlay_files, overlay_phys):
    ap._grid = {}
    ap._chem = {
        'directory': '/primary',
        'files': primary_files,
        'by_non_atten': {k[: ap.N_PARAMS - 1]: v for k, v in primary_files.items()},
        'axis_tokens': ap.order_axis_tokens_physically(
            ap.gn.axis_tokens_from_tuples(primary_files.keys())),
        'species': ['CO'],
        'phys_by_path': {v: dict(PHYS, **primary_phys.get(v, {})) for v in primary_files.values()},
    }
    ap._chem_overlay = {
        'directory': '/overlay',
        'files': overlay_files,
        'by_non_atten': {k[: ap.N_PARAMS - 1]: v for k, v in overlay_files.items()},
        'axis_tokens': ap.order_axis_tokens_physically(
            ap.gn.axis_tokens_from_tuples(overlay_files.keys())),
        'species': ['CO'],
        'phys_by_path': {v: dict(PHYS, **overlay_phys.get(v, {})) for v in overlay_files.values()},
        'for_primary': {},
    }
    assert ap.bootstrap_grid_from_chem()
    ap._refresh_chem_overlay_map()
    vals = [0] * ap.N_PARAMS
    return ap.chem_file(vals), ap.chem_overlay_file(vals)


def test_overlay_by_non_atten_token():
    """Different attenuation tags, same non-attenuation key."""
    primary = {(50, 0, 0, 10, 17, 0): '/primary/m0.hdf5'}
    overlay = {(50, 0, 0, 10, 17, 1): '/overlay/m1.hdf5'}
    p, o = _boot_chem_only(primary, {}, overlay, {})
    assert p == '/primary/m0.hdf5'
    assert o == '/overlay/m1.hdf5'
    assert p != o


def test_overlay_phys_match_when_tokens_differ():
    """Overlay tokens differ from primary (HDF5 vs filename rounding) but phys matches."""
    primary = {(50, 0, 0, 10, 17, 0): '/primary/m0.hdf5'}
    overlay = {(51, 0, 0, 10, 17, 1): '/overlay/m1.hdf5'}
    p, o = _boot_chem_only(primary, {}, overlay, {})
    assert p == '/primary/m0.hdf5'
    assert o == '/overlay/m1.hdf5'
    assert p != o


def test_overlay_map_not_same_as_primary():
    primary = {(50, 0, 0, 10, 17, 0): '/primary/m0.hdf5'}
    overlay = {(50, 0, 0, 10, 17, 0): '/overlay/same_tokens.hdf5'}
    p, o = _boot_chem_only(primary, {}, overlay, {})
    assert p == '/primary/m0.hdf5'
    # Same token tuple but different path — overlay must still resolve.
    assert o == '/overlay/same_tokens.hdf5'
    assert p != o


def test_chem_only_shows_axis_controls():
    ap._grid = {'chem_only': True, 'simline_only': False}
    out = ap.apply_plot_theme('light', True, 0)
    axis_style = out[4]
    assert axis_style.get('display') != 'none'


if __name__ == '__main__':
    test_overlay_by_non_atten_token()
    test_overlay_phys_match_when_tokens_differ()
    test_overlay_map_not_same_as_primary()
    test_chem_only_shows_axis_controls()
    print('all tests passed')
