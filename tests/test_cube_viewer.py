"""CLASS MATRIX / spectral-cube viewer (CARTA-like map + spectrum)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astropy.io import fits

import cube_viewer as cv
import obs_spectrum_fits as osf
import line_catalog as lc


def _write_class_fits(path, spectra, ra_off_deg, dec_off_deg, extra_hdr=None):
    nchan = spectra.shape[1]
    cols = [
        fits.Column(name='SCAN', format='1J', array=np.arange(len(spectra), dtype=np.int32)),
        fits.Column(name='SUBSCAN', format='1J', array=np.ones(len(spectra), dtype=np.int32)),
        fits.Column(name='CDELT2', format='1E', array=np.asarray(ra_off_deg, dtype=np.float32)),
        fits.Column(name='CDELT3', format='1E', array=np.asarray(dec_off_deg, dtype=np.float32)),
        fits.Column(name='SPECTRUM', format=f'{nchan}E',
                    array=np.asarray(spectra, dtype=np.float32)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name='MATRIX')
    hdu.header['MAXIS1'] = nchan
    hdu.header['RESTFREQ'] = 89.188523e9
    hdu.header['CRVAL1'] = 0.0
    hdu.header['CDELT1'] = 48825.0
    hdu.header['CRPIX1'] = (nchan + 1) / 2.0
    hdu.header['CTYPE1'] = 'FREQ'
    hdu.header['VELO-LSR'] = 0.0
    hdu.header['DELTAV'] = -164.13
    hdu.header['CRVAL2'] = 309.75
    hdu.header['CRVAL3'] = 42.37888889
    hdu.header['OBJECT'] = 'DR21OTF'
    hdu.header['ORIGIN'] = 'CLASS-Gildas'
    if extra_hdr:
        for k, v in extra_hdr.items():
            hdu.header[k] = v
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def _gaussian_cube(nchan=41, peak=10.0, v0=0.0, sigma=2.0, noise=0.0, seed=0):
    hdr_dummy = fits.Header()
    hdr_dummy['MAXIS1'] = nchan
    hdr_dummy['VELO-LSR'] = 0.0
    hdr_dummy['CRPIX1'] = (nchan + 1) / 2.0
    hdr_dummy['DELTAV'] = -164.13  # m/s
    v = osf.velocity_axis_lsr_kms(hdr_dummy, nchan=nchan)
    rng = np.random.default_rng(seed)
    y = peak * np.exp(-0.5 * ((v - v0) / sigma) ** 2)
    if noise:
        y = y + rng.normal(0.0, noise, size=nchan)
    return v, y


def _sparse_otf_cube(path):
    """3×3 of 15″ cells with the NE corner missing (8 spectra)."""
    step = 15.0 / 3600.0
    offs = [-step, 0.0, step]
    ra, dec, specs = [], [], []
    v, g = _gaussian_cube(peak=5.0)
    for iy, d in enumerate(offs):
        for ix, r in enumerate(offs):
            if iy == 2 and ix == 2:
                continue
            amp = 2.0 + 8.0 * (ix == 1 and iy == 1)  # centre brightest
            ra.append(r)
            dec.append(d)
            specs.append(g * (amp / 5.0))
    _write_class_fits(path, np.vstack(specs), ra, dec)
    return v


def test_species_from_filename():
    assert cv.species_label_from_path('dr21_hco+10_grid15.fits').startswith('HCO+ (1-0)')
    assert 'H2CS (3-2)' in cv.species_label_from_path('dr21_h2cs32_grid15.fits')
    assert 'N2H+ (1-0)' in cv.species_label_from_path('dr21_n2h+10_grid15_mb.fits')
    assert 'SiO (2-1)' in cv.species_label_from_path('/tmp/dr21_sio21_grid15.fits')


def test_grid_has_blank_missing_cell():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'dr21_hco+10_grid15.fits')
        _sparse_otf_cube(path)
        cube = cv.load_spectral_cube(path, hdu_index=1, use_cache=False)
        assert cube.source_kind == 'class_matrix'
        assert cube.nx == 3 and cube.ny == 3
        assert cube.n_spectra == 8
        assert cube.n_blank == 1
        assert cube.row_index[2, 2] == -1
        np.testing.assert_allclose(cube.grid_step_arcsec, 15.0, rtol=1e-4)
        mom0 = cv.collapse_map(cube, -20.0, 20.0, metric='moment0')
        assert np.isnan(mom0[2, 2])
        assert np.isfinite(mom0).sum() == 8
        peak = cv.peak_cell(cube, mom0)
        assert peak is not None
        iy, ix, row = peak
        assert (iy, ix) == (1, 1)
        v, y = cv.extract_spectrum(cube, [row])
        assert y.max() > cv.extract_spectrum(cube, [0])[1].max()


def test_velocity_and_frequency_axis():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'line.fits')
        v_exp, g = _gaussian_cube()
        _write_class_fits(path, g[None, :], [0.0], [0.0])
        cube = cv.load_spectral_cube(path, hdu_index=1, use_cache=False)
        np.testing.assert_allclose(cube.velocity_kms, v_exp, rtol=1e-6)
        assert cube.freq_ghz is not None
        np.testing.assert_allclose(cube.restfreq_ghz, 89.188523)
        # CLASS: freq increases as velocity decreases (DELTAV < 0)
        assert cube.deltav_kms < 0
        assert cube.freq_ghz[0] > cube.freq_ghz[-1] or cube.velocity_kms[0] > cube.velocity_kms[-1]


def test_click_nearest_and_region_mean():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'dr21_hco+10_grid15.fits')
        _sparse_otf_cube(path)
        cube = cv.load_spectral_cube(path, hdu_index=1, use_cache=False)
        # Click slightly off the centre cell; still snap to the filled centre.
        parsed = cv.nearest_filled_cell(cube, 0.02, -0.01)
        assert parsed is not None
        iy, ix, row = parsed
        assert (iy, ix) == (1, 1)
        point = {'x': 0.0, 'y': 0.0, 'customdata': [row, iy, ix]}
        assert cv.parse_click_point(point, cube) == (iy, ix, row)
        # Region covering the bottom row (Dec = -15")
        rows = cv.rows_in_box(cube, -0.3, 0.3, -0.3, -0.15)
        assert len(rows) == 3
        v, y_mean = cv.extract_spectrum(cube, rows)
        y0 = cv.extract_spectrum(cube, [rows[0]])[1]
        y1 = cv.extract_spectrum(cube, [rows[1]])[1]
        y2 = cv.extract_spectrum(cube, [rows[2]])[1]
        np.testing.assert_allclose(y_mean, np.mean(np.vstack([y0, y1, y2]), axis=0))


def test_map_mean_is_nanmean_of_all_filled_cells():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'dr21_hco+10_grid15.fits')
        _sparse_otf_cube(path)
        cube = cv.load_spectral_cube(path, hdu_index=1, use_cache=False)
        assert cube.n_spectra == 8
        rows = list(range(cube.n_spectra))
        _v, y_mean = cv.extract_spectrum(cube, rows)
        np.testing.assert_allclose(y_mean, np.nanmean(cube.spectra, axis=0))
        # Blank NE corner is not in cube.spectra, so the mean is over filled cells only.
        assert cube.n_blank == 1
        y_centre = cv.extract_spectrum(cube, [cv.peak_cell(
            cube, cv.collapse_map(cube, -8.0, 8.0, metric='peak'))[2]])[1]
        assert y_mean.max() < y_centre.max()


def test_fwhm_of_known_gaussian():
    v = np.linspace(-20, 20, 401)
    sigma = 2.0
    y = 8.0 * np.exp(-0.5 * (v / sigma) ** 2)
    fwhm = cv.fwhm_halfmax_kms(v, y)
    np.testing.assert_allclose(fwhm, cv.FWHM_SIGMA * sigma, rtol=1e-3)
    meas = cv.measure_spectrum(v, y, -10.0, 10.0)
    np.testing.assert_allclose(meas['peak_k'], 8.0, rtol=1e-6)
    np.testing.assert_allclose(meas['v_peak_kms'], 0.0, atol=0.05)
    np.testing.assert_allclose(meas['fwhm_kms'], cv.FWHM_SIGMA * sigma, rtol=1e-3)
    # Integrated Gaussian ≈ A σ √(2π)
    expected = 8.0 * sigma * np.sqrt(2.0 * np.pi)
    np.testing.assert_allclose(meas['integrated_k_kms'], expected, rtol=0.02)


def test_fit_spectrum_arrays_recovers_gaussian():
    v = np.linspace(-25, 25, 251)
    y = 0.2 + 6.0 * np.exp(-0.5 * ((v - 3.0) / 1.8) ** 2)
    result = osf.fit_spectrum_arrays(v, y, -10.0, 15.0, n_gaussians=1)
    np.testing.assert_allclose(result.amplitudes[0], 6.0, rtol=0.05)
    np.testing.assert_allclose(result.centers[0], 3.0, atol=0.1)
    np.testing.assert_allclose(result.sigmas[0], 1.8, rtol=0.08)


def test_user_centers_recover_two_gaussians():
    v = np.linspace(-30, 30, 601)
    y = (5.0 * np.exp(-0.5 * ((v - 6.0) / 1.3) ** 2)
         + 3.5 * np.exp(-0.5 * ((v + 8.0) / 2.0) ** 2))
    result = osf.fit_spectrum_arrays(
        v, y, -25.0, 25.0, n_gaussians=2,
        amplitudes=[5.0, 3.5],
        centers=[6.0, -8.0],
        fwhms=[osf.sigma_to_fwhm(1.3), osf.sigma_to_fwhm(2.0)],
    )
    ctrs = np.sort(np.asarray(result.centers, dtype=float))
    np.testing.assert_allclose(ctrs, [-8.0, 6.0], atol=0.25)


def test_lock_holds_user_center():
    v = np.linspace(-20, 20, 401)
    y = 7.0 * np.exp(-0.5 * (v / 1.5) ** 2)
    result = osf.fit_spectrum_arrays(
        v, y, -15.0, 15.0, n_gaussians=1,
        centers=[4.0], lock_user=True,
    )
    np.testing.assert_allclose(result.centers[0], 4.0, atol=1e-8)


def test_merge_user_p0_overwrites_only_filled_slots():
    auto = (10.0, 0.0, 1.0, 0.0)
    p0, user_set = osf.merge_user_gaussian_p0(
        1, auto, amplitudes=[None], centers=[2.5], fwhms=[None])
    np.testing.assert_allclose(p0[0], 10.0)
    np.testing.assert_allclose(p0[1], 2.5)
    np.testing.assert_allclose(p0[2], 1.0)
    assert user_set == [False, True, False, False]


def test_list_cube_fits_directory():
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, 'dr21_hcn10_grid15.fits')
        b = os.path.join(td, 'dr21_sio21_grid15.fits')
        _, g = _gaussian_cube()
        _write_class_fits(a, g[None, :], [0.0], [0.0])
        _write_class_fits(b, g[None, :], [0.0], [0.0])
        opts = cv.list_cube_fits(td)
        assert len(opts) == 2
        labels = ' '.join(o['label'] for o in opts)
        assert 'HCN (1-0)' in labels
        assert 'SiO (2-1)' in labels
        single = cv.list_cube_fits(a)
        assert len(single) == 1
        assert single[0]['value'] == os.path.abspath(a)


def test_sky_coordinates_use_map_centre():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'pos.fits')
        _, g = _gaussian_cube()
        ra_off = 15.0 / 3600.0
        dec_off = -30.0 / 3600.0
        _write_class_fits(path, g[None, :], [ra_off], [dec_off])
        cube = cv.load_spectral_cube(path, hdu_index=1, use_cache=False)
        sky = cv.selection_sky(cube, [0])
        np.testing.assert_allclose(sky['ra_deg'], 309.75 + ra_off)
        np.testing.assert_allclose(sky['dec_deg'], 42.37888889 + dec_off)


def test_image_cube_roundtrip():
    nchan, ny, nx = 21, 4, 5
    v = np.linspace(-10, 10, nchan)
    cube = np.zeros((nchan, ny, nx), dtype=np.float32)
    cube[:, 1, 2] = 4.0 * np.exp(-0.5 * (v / 1.5) ** 2)
    hdu = fits.PrimaryHDU(cube)
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CTYPE3'] = 'VRAD'
    hdu.header['CUNIT1'] = 'deg'
    hdu.header['CUNIT2'] = 'deg'
    hdu.header['CUNIT3'] = 'km/s'
    hdu.header['CRPIX1'] = 3.0
    hdu.header['CRPIX2'] = 2.0
    hdu.header['CRPIX3'] = 11.0
    hdu.header['CRVAL1'] = 309.75
    hdu.header['CRVAL2'] = 42.38
    hdu.header['CRVAL3'] = 0.0
    hdu.header['CDELT1'] = -15.0 / 3600.0
    hdu.header['CDELT2'] = 15.0 / 3600.0
    hdu.header['CDELT3'] = float(v[1] - v[0])
    hdu.header['RESTFREQ'] = 89.188523e9
    hdu.header['BUNIT'] = 'K'
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'cube.fits')
        hdu.writeto(path)
        loaded = cv.load_spectral_cube(path, hdu_index=0, use_cache=False)
        assert loaded.source_kind == 'image_cube'
        assert loaded.ny == ny and loaded.nx == nx
        mom = cv.collapse_map(loaded, -8.0, 8.0, metric='peak')
        iy, ix, row = cv.peak_cell(loaded, mom)
        assert (iy, ix) == (1, 2)
        _v, y = cv.extract_spectrum(loaded, [row])
        np.testing.assert_allclose(y.max(), 4.0, rtol=1e-5)
        assert loaded.frequency_ghz is not None
        np.testing.assert_allclose(
            loaded.frequency_ghz.size, loaded.velocity_kms.size)


def test_channel_map_matches_spectrum_and_keeps_blanks():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'dr21_hco+10_grid15.fits')
        _sparse_otf_cube(path)
        cube = cv.load_spectral_cube(path, hdu_index=1, use_cache=False)
        i0 = cv.default_channel_index(cube)
        assert 0 <= i0 < cube.nchan
        assert abs(cube.velocity_kms[i0]) == np.nanmin(np.abs(cube.velocity_kms))
        plane = cv.channel_map(cube, i0)
        assert plane.shape == (cube.ny, cube.nx)
        assert np.isnan(plane[2, 2])
        assert np.isfinite(plane).sum() == 8
        iy, ix, row = cv.peak_cell(cube, plane)
        _, y = cv.extract_spectrum(cube, [row])
        np.testing.assert_allclose(plane[iy, ix], y[i0])
        assert cv.clamp_channel(cube, -4) == 0
        assert cv.clamp_channel(cube, 10_000) == cube.nchan - 1
        label = cv.channel_label(cube, i0)
        assert f'{i0 + 1}/{cube.nchan}' in label
        assert 'km/s' in label


def test_ppv_volume_is_downsampled_and_finite():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'dr21_hco+10_grid15.fits')
        _sparse_otf_cube(path)
        cube = cv.load_spectral_cube(path, hdu_index=1, use_cache=False)
        arr = cv.ppv_volume_arrays(cube, max_side=2)
        vol = arr['intensity']
        assert vol.ndim == 3
        assert vol.shape[0] <= 2 and vol.shape[1] <= 2 and vol.shape[2] <= 2
        assert vol.shape == (arr['dec_arcmin'].size, arr['ra_arcmin'].size,
                             arr['velocity_kms'].size)
        assert np.isfinite(vol).any()
        assert arr['isomax'] >= arr['isomin']
        # Missing OTF corner stays NaN in the native grid; downsampled cells
        # that still land on filled rows remain finite.
        full = cv.ppv_volume_arrays(cube, max_side=max(cube.ny, cube.nx, cube.nchan))
        assert full['intensity'].shape == (cube.ny, cube.nx, cube.nchan)
        assert np.isnan(full['intensity'][2, 2]).all()


def _write_casa_image_cube(path, *, with_beams=True, restfrq=True, use_cd=False):
    """Small CASA-like 4-D cube: STOKES × FREQ × Dec × RA, optional BEAMS table."""
    nchan, ny, nx = 9, 4, 5
    rest_hz = 39.68e9
    nu0 = 38.986e9
    dnu = 6.0e6
    v = osf.radio_velocity_kms_from_frequency_ghz(
        (nu0 + dnu * np.arange(nchan)) / 1e9, rest_hz / 1e9)
    cube = np.zeros((1, nchan, ny, nx), dtype=np.float32)
    cube[0, :, 1, 2] = 3.0 * np.exp(-0.5 * ((v - v[nchan // 2]) / 2.0) ** 2)
    hdu = fits.PrimaryHDU(cube)
    h = hdu.header
    h['OBJECT'] = 'HELMS1'
    h['BUNIT'] = 'Jy/beam'
    h['ORIGIN'] = 'CASA 6.2.1'
    h['CTYPE1'] = 'RA---SIN'
    h['CTYPE2'] = 'DEC--SIN'
    h['CTYPE3'] = 'FREQ'
    h['CTYPE4'] = 'STOKES'
    h['CUNIT1'] = 'deg'
    h['CUNIT2'] = 'deg'
    h['CUNIT3'] = 'Hz'
    h['CUNIT4'] = ''
    h['CRPIX1'] = 3.0
    h['CRPIX2'] = 2.0
    h['CRPIX3'] = 1.0
    h['CRPIX4'] = 1.0
    h['CRVAL1'] = 353.67
    h['CRVAL2'] = -6.87
    h['CRVAL3'] = nu0
    h['CRVAL4'] = 1.0
    h['PC1_1'] = 1.0
    h['PC2_2'] = 1.0
    h['PC3_3'] = 1.0
    if use_cd:
        h['CD1_1'] = -8.333e-5
        h['CD2_2'] = 8.333e-5
        h['CD3_3'] = dnu
        h['CD4_4'] = 1.0
    else:
        h['CDELT1'] = -8.333e-5
        h['CDELT2'] = 8.333e-5
        h['CDELT3'] = dnu
        h['CDELT4'] = 1.0
    h['SPECSYS'] = 'LSRK'
    if restfrq:
        h['RESTFRQ'] = rest_hz
    hdus = [hdu]
    if with_beams:
        cols = [
            fits.Column(name='BMAJ', format='1E', array=np.full(nchan, 0.01, dtype=np.float32)),
            fits.Column(name='BMIN', format='1E', array=np.full(nchan, 0.008, dtype=np.float32)),
            fits.Column(name='BPA', format='1E', array=np.zeros(nchan, dtype=np.float32)),
            fits.Column(name='CHAN', format='1J', array=np.arange(nchan, dtype=np.int32)),
            fits.Column(name='POL', format='1J', array=np.zeros(nchan, dtype=np.int32)),
        ]
        hdus.append(fits.BinTableHDU.from_columns(cols, name='BEAMS'))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return v


def test_casa_4d_cube_skips_beams_table():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'helms01_usb_contsub_dirtycube.image.pbcor.fits')
        v_exp = _write_casa_image_cube(path, with_beams=True)
        assert cv.default_hdu_index(path) == 0
        loaded = cv.load_spectral_cube(path, hdu_index=1, use_cache=False)
        assert loaded.source_kind == 'image_cube'
        assert loaded.hdu_index == 0
        assert loaded.ny == 4 and loaded.nx == 5
        assert loaded.nchan == 9
        assert loaded.intensity_unit.lower().startswith('jy')
        np.testing.assert_allclose(loaded.velocity_kms, v_exp, rtol=1e-5)
        mom = cv.collapse_map(loaded, float(np.min(v_exp)), float(np.max(v_exp)),
                              metric='peak')
        iy, ix, row = cv.peak_cell(loaded, mom)
        assert (iy, ix) == (1, 2)
        _v, y = cv.extract_spectrum(loaded, [row])
        np.testing.assert_allclose(y.max(), 3.0, rtol=1e-5)
        assert loaded.freq_ghz is not None
        assert loaded.restfreq_ghz == 39.68


def test_casa_cube_cd_matrix_without_cdelt_or_restfrq():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'casa_cd.fits')
        _write_casa_image_cube(path, with_beams=False, restfrq=False, use_cd=True)
        loaded = cv.load_spectral_cube(path, hdu_index=0, use_cache=False)
        assert loaded.source_kind == 'image_cube'
        assert loaded.nchan == 9
        assert np.all(np.isfinite(loaded.velocity_kms))
        np.testing.assert_allclose(loaded.velocity_kms[0], 0.0, atol=1e-3)


def test_hco_window_finds_hco_plus():
    rows = lc.lines_in_window(89.165, 89.212, v_source_kms=0.0)
    names = {r['species'] for r in rows}
    assert 'HCO+' in names
    assert 'HCN' not in names


def test_n2hp_window_finds_hyperfine():
    rows = lc.lines_in_window(93.170, 93.178, v_source_kms=0.0)
    n2h = [r for r in rows if r['species'] == 'N2H+']
    assert len(n2h) >= 5


def test_source_velocity_shifts_observed_frequency():
    rows0 = lc.lines_in_window(89.1880, 89.1890, v_source_kms=0.0)
    assert any(r['species'] == 'HCO+' for r in rows0)
    # +20 km/s moves HCO+ ~6 MHz down; it should leave this 1 MHz window
    rows = lc.lines_in_window(89.1880, 89.1890, v_source_kms=20.0)
    assert not any(r['species'] == 'HCO+' for r in rows)


def test_remap_velocity_to_frequency_is_monotonic_inverse():
    v = np.linspace(-10, 10, 21)
    rest = 89.1885247
    freq = rest * (1.0 - v / lc.C_LIGHT_KMS)
    v_back = lc.remap_velocity_to_axis(freq, v, freq)
    np.testing.assert_allclose(v_back, v, atol=1e-6)


def test_splatalogue_parses_orderedfreq_mhz_as_ghz():
    from astropy.table import Table
    table = Table({
        'name': ['HCO<sup>+</sup> <font color="red">v=0</font>'],
        'resolved_QNs': ['1-0'],
        'orderedfreq': [89188.5247],
        'upper_state_energy_K': [4.28],
        'linelist': ['CDMS'],
    })
    rows = lc.rows_from_splatalogue_table(table, 89.165, 89.212)
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0]['rest_ghz'], 89.1885247, atol=1e-7)
    assert rows[0]['species'] == 'HCO+ v=0'
    assert rows[0]['source'] == 'splatalogue/CDMS'


def test_splatalogue_legacy_ghz_column_still_works():
    from astropy.table import Table
    table = Table({
        'Species': ['N2H+'],
        'Resolved QNs': ['1-0 F=0-1'],
        'Freq-GHz': [93.171621],
        'E_U (K)': [4.47],
    })
    rows = lc.rows_from_splatalogue_table(table, 93.170, 93.178)
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0]['rest_ghz'], 93.171621, atol=1e-7)
    assert rows[0]['species'] == 'N2H+'


def test_splatalogue_drops_lines_outside_frequency_window():
    from astropy.table import Table
    table = Table({
        'name': ['HCO+', 'HCN'],
        'resolved_QNs': ['1-0', '1-0'],
        'orderedfreq': [89188.5247, 88631.8473],
        'upper_state_energy_K': [4.28, 4.25],
    })
    rows = lc.rows_from_splatalogue_table(table, 89.165, 89.212)
    names = {r['species'] for r in rows}
    assert 'HCO+' in names
    assert 'HCN' not in names


def test_splatalogue_live_hco_window_if_network():
    """Live CDMS/JPL search; skip if Splatalogue is unreachable."""
    if not lc.splatalogue_available():
        return
    rows, err = lc.query_splatalogue(89.165, 89.212, v_source_kms=0.0, eu_max_k=150)
    if err or not rows:
        return
    blob = ' '.join(r['species'] for r in rows)
    assert 'HCO' in blob


if __name__ == '__main__':
    test_species_from_filename()
    test_grid_has_blank_missing_cell()
    test_velocity_and_frequency_axis()
    test_click_nearest_and_region_mean()
    test_map_mean_is_nanmean_of_all_filled_cells()
    test_fwhm_of_known_gaussian()
    test_fit_spectrum_arrays_recovers_gaussian()
    test_user_centers_recover_two_gaussians()
    test_lock_holds_user_center()
    test_merge_user_p0_overwrites_only_filled_slots()
    test_list_cube_fits_directory()
    test_sky_coordinates_use_map_centre()
    test_image_cube_roundtrip()
    test_channel_map_matches_spectrum_and_keeps_blanks()
    test_ppv_volume_is_downsampled_and_finite()
    test_casa_4d_cube_skips_beams_table()
    test_casa_cube_cd_matrix_without_cdelt_or_restfrq()
    test_hco_window_finds_hco_plus()
    test_n2hp_window_finds_hyperfine()
    test_source_velocity_shifts_observed_frequency()
    test_remap_velocity_to_frequency_is_monotonic_inverse()
    test_splatalogue_parses_orderedfreq_mhz_as_ghz()
    test_splatalogue_legacy_ghz_column_still_works()
    test_splatalogue_drops_lines_outside_frequency_window()
    test_splatalogue_live_hco_window_if_network()
    print('ok')
