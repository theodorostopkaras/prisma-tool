"""Intensities embedded in KOSMA-τ HDF5 match the SIMLINE .smli tables."""
import os
import sys

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import app as ap

SIMS = os.path.expanduser('~/Desktop/teo_extra_sims')
MODEL = 'Model100_30_00_20_10_14_00'


@pytest.mark.skipif(not os.path.isdir(os.path.join(SIMS, 'pdrgrid_hdf5')),
                    reason='teo_extra_sims not available')
def test_hdf5_intensities_match_smli():
    ap.scan_directory(os.path.join(SIMS, 'pdrgrid_hdf5'))
    assert ap._simline.get('from_hdf5')
    assert 'CO' in ap._simline['species']
    tokens = next(t for t, p in ap._grid['files'].items() if MODEL in p)
    for idef in ('jtemp', 'jerg'):
        smli = ap.read_smli_file(os.path.join(
            SIMS, MODEL, 'simlineoutput', f'{idef}_{MODEL}_CO.smli'))
        opts = ap._simline_transition_options('CO', idef)
        assert [o['transition'] for o in opts[:3]] == [r['transition'] for r in smli[:3]]
        got = [ap.get_smli_intensity(tokens, 'CO', idef, i) for i in range(3)]
        assert np.allclose(got, [r['intensity'] for r in smli[:3]], rtol=1e-4)


def test_meudon_transition_labels():
    assert ap._meudon_transition('v=0,J=2', 'v=0,J=1') == '2--1'
    assert ap._meudon_transition('v=1,J=1', 'v=0,J=0') == '1_1--0_0'
    assert ap._meudon_transition('El=3P,J=1', 'El=3P,J=0') == '1--0'
    assert ap._meudon_transition('J=1,ka=1,kc=0', 'J=1,ka=0,kc=1') == '1_1_0--1_0_1'


MEUDON = os.path.expanduser('~/Downloads/P154G3_n_r1e0n1e4A4e1_s_20.hdf5')


@pytest.mark.skipif(not os.path.isfile(MEUDON), reason='Meudon sample not available')
def test_meudon_intensities_per_angle():
    import h5py
    _, lines = ap._meudon_line_columns(MEUDON)
    col = {tr: c for tr, c, _nu in lines['CO']}['2--1']
    with h5py.File(MEUDON, 'r') as hf:
        want = {a: hf[f'Integrated quantities/Intensities/Line emission A{a}'][0, col]
                for a in ('00', '60')}
    ap._simline = {}
    ap._hdf5_line_angle = None
    ap.scan_directory(MEUDON)
    assert ap._simline['idefs'] == ('jtemp', 'jerg') and ap._simline['angle'] == '00'
    tokens = next(iter(ap._grid['files']))
    idx = [o['transition'] for o in ap._simline_transition_options('CO', 'jerg')].index('2--1')
    assert np.isclose(ap.get_smli_intensity(tokens, 'CO', 'jerg', idx), want['00'])
    ap._hdf5_line_angle = '60'
    ap._load_hdf5_simline()
    assert np.isclose(ap.get_smli_intensity(tokens, 'CO', 'jerg', idx), want['60'])
    # K km/s from the erg value and the LAMDA frequency (CO 2-1, 230.538 GHz).
    expect = want['60'] * 2.99792458e10 ** 3 / (2 * 1.380649e-16 * 230.538e9 ** 3) / 1e5
    assert np.isclose(ap.get_smli_intensity(tokens, 'CO', 'jtemp', idx), expect, rtol=1e-4)


def test_interpolation_analysis_rejects_single_row_slice():
    import grid_interp as gi
    with pytest.raises(gi.SliceTooSmallError):
        gi.analyze_slice_interpolation(np.array([1e3, 1e4]), np.array([1.0]),
                                       np.array([[1e-6, 2e-6]]))


@pytest.mark.skipif(not os.path.isfile(MEUDON), reason='Meudon sample not available')
def test_single_meudon_model_sled():
    ap._simline = {}
    ap._hdf5_line_angle = None
    ap.scan_directory(MEUDON)
    fig = ap.fig_intensity_spectrum([0] * ap.N_PARAMS, ['CO'], 'jerg')
    assert fig.layout.xaxis.type == 'linear'  # INP frequencies
    assert np.allclose(list(fig.data[0].x[:2]), [115.271, 230.538], rtol=1e-4)
    assert np.all(np.isfinite(fig.data[0].y[:2]))
    fig = ap.fig_intensity_spectrum([0] * ap.N_PARAMS, ['H2O'], 'jerg')
    assert fig.layout.xaxis.type == 'category'  # no INP file: plot vs transition
    assert np.all(np.isfinite(fig.data[0].y[:2]))


@pytest.mark.skipif(not os.path.isfile(MEUDON), reason='Meudon sample not available')
def test_meudon_frequencies_from_lamda_files():
    _, lines = ap._meudon_line_columns(MEUDON)
    freq = {sp: {tr: nu for tr, _c, nu in rows} for sp, rows in lines.items()}
    # Reference rest frequencies (GHz), CDMS / NIST.
    for sp, tr, ref in [('CO', '1--0', 115.2712), ('CO', '2--1', 230.5380),
                        ('13CO', '1--0', 110.2014), ('C18O', '1--0', 109.7822),
                        ('CS', '2--1', 97.9810), ('HCO+', '1--0', 89.1885),
                        ('C', '1--0', 492.1607), ('C', '2--1', 809.3420),
                        ('C+', '3/2--1/2', 1900.5369), ('O', '1--2', 4744.7775),
                        ('S', '1--2', 11873.43), ('Si', '1--0', 2311.85),
                        ('CH+', '1--0', 835.137)]:  # energies, not the broken ch+.lamda FREQ column
        assert np.isclose(freq[sp][tr], ref, rtol=3e-5), (sp, tr, freq[sp][tr])
    assert np.isnan(freq['C']['1D_2--3P_2'])  # excited term: not in the data files
    assert np.isnan(freq['H2O']['1_1_1--0_0_0'])  # no INP file: stays unknown


@pytest.mark.skipif(not os.path.isdir(os.path.join(SIMS, 'pdrgrid_hdf5')),
                    reason='teo_extra_sims not available')
def test_erg_to_k_kms_matches_simline():
    """SIMLINE's own T_integrated / I_integrated pairs follow the same conversion."""
    import h5py
    path = os.path.join(SIMS, 'pdrgrid_hdf5', MODEL + '.hdf5')
    with h5py.File(path, 'r') as hf:
        for grp in ('co', 'cp', 'c', 'o'):
            ii = hf[f'SIMLINE output/{grp}/Integrated intensities']
            got = [ap._erg_to_k_kms(i, nu) for i, nu in zip(ii['I_integrated'][()], ii['frequencies'][()])]
            want = ii['T_integrated'][()]
            ok = want > 1e-6
            assert np.allclose(np.array(got)[ok], want[ok], rtol=1e-3), grp
    assert np.isnan(ap._erg_to_k_kms(1e-6, np.nan))


@pytest.mark.skipif(not os.path.isfile(MEUDON), reason='Meudon sample not available')
def test_meudon_spectrum_frequency_axis_conserves_flux():
    ap._simline = {}
    ap.scan_directory(MEUDON)
    values = [0] * ap.N_PARAMS
    by_wl = ap.fig_ir_spectrum(values, 'wavelength', 'log').data[0]
    by_nu = ap.fig_ir_spectrum(values, 'frequency', 'log').data[0]
    # ∫ I_λ dλ [Å] equals ∫ I_ν dν [Hz]
    flux_wl = np.trapezoid(np.asarray(by_wl.y), np.asarray(by_wl.x) * 1e4)
    flux_nu = -np.trapezoid(np.asarray(by_nu.y), np.asarray(by_nu.x) * 1e9)
    assert np.isclose(flux_wl, flux_nu, rtol=1e-2)


@pytest.mark.skipif(not os.path.isdir(os.path.join(SIMS, 'pdrgrid_hdf5')),
                    reason='teo_extra_sims not available')
def test_kosma_spectrum_columns_from_metadata():
    spectra = ap.read_hdf5_spectrum(os.path.join(SIMS, 'pdrgrid_hdf5', MODEL + '.hdf5'))
    spec = spectra[0]
    assert spec['name'] == 'IR Continuum/Spectrum Continuum large'
    assert np.allclose(spec['frequency_ghz'] * spec['wavelength_um'], 2.99792458e5, rtol=1e-6)
    labels = [label for label, _unit, _y in spec['series']]
    assert 'Dust emission' in labels and 'Flux central star' not in labels  # all 1e-70


@pytest.mark.skipif(not os.path.isfile(MEUDON), reason='Meudon sample not available')
def test_spectral_regimes_follow_x_axis():
    ap._simline = {}
    ap.scan_directory(MEUDON)
    values = [0] * ap.N_PARAMS
    assert not ap.fig_ir_spectrum(values, 'wavelength').layout.shapes
    for axis in ('wavelength', 'frequency'):
        fig = ap.fig_ir_spectrum(values, axis, show_regimes=True)
        labels = {a.text: a for a in fig.layout.annotations}
        mir = next(sh for sh, a in zip(fig.layout.shapes, fig.layout.annotations) if a.text == 'MIR')
        edges = sorted(10 ** np.array([mir.x0, mir.x1]))
        want = [5.0, 25.0] if axis == 'wavelength' else [2.99792458e5 / 25, 2.99792458e5 / 5]
        assert np.allclose(edges, want), (axis, edges)
        assert {'FUV', 'Vis', 'MIR', 'FIR', 'cm'} <= set(labels)
