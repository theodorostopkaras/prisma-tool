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


def test_read_field_falls_back_to_zeta_local(tmp_path):
    # KoSens fd03c61: newer KOSMA-τ stores ζ(r) as ``zeta_local`` (no ``cosray`` column).
    import h5py
    path = tmp_path / 'm.hdf5'
    with h5py.File(path, 'w') as hf:
        hf['Aux/CR'] = np.array([[1e-16], [5e-17]])
    fmap = {'zeta_local': ('Aux/CR', 0)}
    with h5py.File(path, 'r') as hf:
        np.testing.assert_allclose(ap._read_field(hf, ap.KEY_COSRAY, fmap), [1e-16, 5e-17])
        assert ap._read_field(hf, ap.KEY_COSRAY, {'cosray': ('Aux/CR', 3)}) is None


def test_rate_regime_flip_colours():
    # Γ_CR < Γ_PE everywhere with constant ζ; the attenuated grid flips one cell.
    n, z = np.repeat([3.0, 4.0], 2), np.tile([-17.0, -16.0], 2)
    ax = dict(y=n, x=z, x_key='crir', y_key='density')
    const = dict(ax, heat={'Cosmic rays': np.ones(4), 'Photo-electric effect': np.full(4, 2.0)})
    att = dict(ax, heat={'Cosmic rays': np.array([3.0, 1, 1, 1]),
                         'Photo-electric effect': np.full(4, 2.0)})
    fig = ap.fig_rate_regime_boundary(const, 'Cosmic rays', 'Photo-electric effect',
                                      data_atten=att, x_lim=(-17, -17))
    z_reg = np.asarray(fig.data[0].z, float)
    assert z_reg[0, 0] == 2 and (z_reg.ravel()[1:] == 0).all()   # orange: PE → CR
    np.testing.assert_allclose(fig.layout.xaxis.range, [-17.5, -16.5])


def test_heat_cool_axes_follow_grid_without_zeta():
    # A grid varying in n_H, M and χ but not ζ must not default to ζ.
    assert ap._hc_default_axes(['density', 'fuv', 'crir']) == ('crir', 'density')
    x, y = ap._hc_default_axes(['density', 'mass', 'fuv'])
    assert 'crir' not in (x, y) and x != y
    data = dict(x=np.tile([0.0, 1.0], 2), y=np.repeat([3.0, 4.0], 2), x_key='mass',
                y_key='density', heat={'Cosmic rays': np.ones(4)}, cool={'CO': np.ones(4)})
    fig = ap.fig_absolute_heating_cooling_vs_crir(data, [3.0, 4.0])
    titles = [fig.layout[a].title.text or '' for a in fig.layout if a.startswith('xaxis')]
    assert any('M' in t for t in titles) and not any('\u03B6' in t for t in titles)
    assert not any('\u03B6' in (a.text or '') for a in fig.layout.annotations)
    order, x_sel = ap._select_panel(data, 4.0)
    np.testing.assert_allclose(x_sel, [0.0, 1.0])


def test_abundance_pdf_x_and_split_without_zeta():
    # A_V stays linear (surface A_V = 0 is kept); ζ_H/n_H is log10 and drops x <= 0.
    x, y = ap._abundance_xy([0.0, 1.0, 2.0], [1e-4, 1e-5, 1e-6], log_x=False)
    np.testing.assert_allclose(x, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(y, [-4.0, -5.0, -6.0])
    x, _ = ap._abundance_xy([0.0, 1e-20], [1e-4, 1e-5])
    np.testing.assert_allclose(x, [-20.0])
    m = dict(av=np.linspace(0.0, 2.0, 3), cosray=np.full(3, 2e-17), nH=np.full(3, 1e3))
    np.testing.assert_allclose(ap._apdf_x_raw(m, 'zeta_n'), [2e-20] * 3)
    toks = [0] * ap.N_PARAMS
    toks[ap._PARAM_IDX['mass']] = 10                     # M = 10^(10/10) = 10 Msun
    assert ap._apdf_split_value(m, toks, 'mass') == 1.0
    assert ap._apdf_split_value(m, toks, 'crir') == -16.7  # surface ζ_H, as KoSens
    assert ap._apdf_split_value(m, toks, None) is None
    assert 'ζ' not in ap._apdf_split_title('mass', 1.0)
