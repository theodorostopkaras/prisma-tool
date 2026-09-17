"""Profile x-axis quantities and per-species intensity transition selection."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as ap


def _model():
    # KOSMA-τ order: surface first, radius shrinking inwards.
    return dict(av=np.linspace(0.0, 5.0, 11), nH=np.full(11, 1e3),
                radius=np.linspace(1.0, 0.0, 11), nh2_profile=np.logspace(18, 22, 11))


def test_depth_and_column_density():
    m = _model()
    depth = ap._x_raw(m, 'depth')
    assert depth[0] == 0.0 and np.all(np.diff(depth) > 0)
    nh = ap._x_raw(m, 'NH')
    assert nh[0] == 0.0
    assert np.isclose(nh[-1], 1e3 * ap.PC_TO_CM)
    # Reversed storage order gives the same column per sample.
    rev = {k: v[::-1] for k, v in m.items()}
    assert np.allclose(ap._x_raw(rev, 'NH')[::-1], nh)


def test_x_min_applies_to_any_quantity_and_missing_falls_back():
    m = _model()
    xv, _, _, _ = ap._xvals(m, 'NH2', 'log', x_min=1e20)
    assert np.all(np.isnan(xv[m['nh2_profile'] < 1e20]))
    assert np.all(np.isfinite(xv[m['nh2_profile'] >= 1e20]))
    xv, xl, _, _ = ap._xvals(dict(av=m['av']), 'depth', 'linear', x_min=0.5)
    assert 'not in this model' in xl and np.isfinite(xv).all()


def test_transitions_never_cross_species():
    def opts(n):
        return [{'value': i, 'label': str(i)} for i in range(n)]
    saved = ap._simline
    ap._simline = {'transitions': {('CO', 'jtemp'): opts(5), ('HCN', 'jtemp'): opts(3)}}
    try:
        combos = ap._intensity_map_combos(['CO', 'HCN'],
                                          ['HCN||2', 'CO||0', 'XX||1', 'HCN||7'], 'jtemp')
        assert combos == [('CO', 0), ('HCN', 2)]
    finally:
        ap._simline = saved
