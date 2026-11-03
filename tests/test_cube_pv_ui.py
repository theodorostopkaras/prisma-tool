"""PV-cut figure follows the line-core window; dragged/drawn cut syncs the inputs."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app
from test_cube_viewer import _sparse_otf_cube


def test_pv_cut_follows_velocity_window():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'c.fits')
        _sparse_otf_cube(p)
        args = (p, 1, None, None, None, None, 0)
        wide = app.fig_cube_pv_cut(*args, -3, 3, 'Viridis', 'linear')
        narrow = app.fig_cube_pv_cut(*args, -1, 1, 'Viridis', 'linear')
        vw, vn = np.asarray(wide.data[-1].y), np.asarray(narrow.data[-1].y)
        assert vn.size < vw.size and vn.min() >= -1 and vn.max() <= 1
        assert wide.layout.uirevision != narrow.layout.uirevision
        assert wide.layout.shapes[0].editable


def test_pv_cut_from_relayout():
    cur = [1.0, 0.0, -1.0, 0.0]
    f = app.pv_cut_from_relayout
    assert f(None, cur) is None
    assert f({'xaxis.range[0]': 1}, cur) is None                     # zoom: ignore
    assert f({'shapes[0].x1': -0.5, 'shapes[0].y1': 0.2}, cur) == (1.0, 0.0, -0.5, 0.2)
    drawn = {'shapes': [{'type': 'line', 'xref': 'x', 'yref': 'y',
                         'x0': 1, 'y0': 0, 'x1': -1, 'y1': 0},
                        {'type': 'line', 'xref': 'x', 'yref': 'y',
                         'x0': 0.25, 'y0': -0.5, 'x1': 0.1234, 'y1': 0.5}]}
    assert f(drawn, cur) == (0.25, -0.5, 0.123, 0.5)
    on_pv = {'shapes': [{'type': 'line', 'xref': 'x2', 'yref': 'y2',
                         'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1}]}
    assert f(on_pv, cur) is None                                    # drawn on PV panel


def test_ppv_overlays_colour_and_align():
    from astropy.io import fits
    import cube_viewer as cv
    with tempfile.TemporaryDirectory() as td:
        main = os.path.join(td, 'dr21_hco+10.fits')
        other = os.path.join(td, 'dr21_hcn10.fits')
        _sparse_otf_cube(main)
        _sparse_otf_cube(other)
        with fits.open(other, mode='update') as hdul:      # centre 0.5' north
            hdul[1].header['CRVAL3'] += 0.5 / 60.0
        m = cv.load_spectral_cube(main, hdu_index=1, use_cache=False)
        o = cv.load_spectral_cube(other, hdu_index=1, use_cache=False)
        dra, ddec = cv.sky_shift_arcmin(m, o)
        assert abs(dra) < 1e-9 and abs(ddec - 0.5) < 1e-6

        solo = app.fig_cube_ppv(main, 1, 'Viridis')
        assert len(solo.data) == 1 and not solo.layout.showlegend
        fig = app.fig_cube_ppv(main, 1, 'Viridis', overlays=[other, main, None])
        assert len(fig.data) == 2 and fig.layout.showlegend
        c0, c1 = (tr.colorscale[0][1] for tr in fig.data)
        assert c0 != c1
        assert abs(np.max(fig.data[1].y) - np.max(fig.data[0].y) - 0.5) < 1e-6

        opts = app.update_cv_ppv_overlay_options(None, td, main, ['/elsewhere/x.fits'])
        vals = [o['value'] for o in opts]
        assert other in vals and main not in vals and '/elsewhere/x.fits' in vals


def test_shown_fit_numbers_stay_auto():
    import dash
    # last fit wrote these into the boxes
    shown = {'amp': [4.0, 1.5], 'v0': [0.1, 2.0], 'fwhm': [1.2, 0.8]}
    # user changed G2 peak to 3.0; everything else still shows the old fit
    cols = [[4.0, 3.0], [0.1, 2.0], [1.2, 0.8]]
    # regression: shown numbers must NOT become starting values (stale-guess bad fits)
    assert app.user_guesses(cols, shown) == [[None, 3.0], [None, None], [None, None]]
    new_fit = {'amp': [5.0, 2.9], 'v0': [0.3, 2.2], 'fwhm': [1.1, 0.9]}
    out, rec = app.guess_boxes_after_fit(new_fit, cols, shown)
    assert out[0] == [5.0, dash.no_update]          # user's 3.0 kept
    assert out[1] == [0.3, 2.2] and out[2] == [1.1, 0.9]
    assert rec['amp'] == [5.0, None]
    # next round: the refreshed numbers are auto again
    cols2 = [[5.0, 3.0], [0.3, 2.2], [1.1, 0.9]]
    assert app.user_guesses(cols2, rec) == [[None, 3.0], [None, None], [None, None]]
    # failed fit: auto boxes go empty, the user's number stays
    out, _ = app.guess_boxes_after_fit(None, cols2, rec)
    assert out[0] == [None, dash.no_update] and out[1] == [None, None]
    # Reset: every box shows the new fit
    out, rec = app.guess_boxes_after_fit(new_fit, cols2, rec, reset=True)
    assert out[0] == [5.0, 2.9] and rec['amp'] == [5.0, 2.9]
    # first load: empty boxes, no record
    assert app.user_guesses([[None], [''], [None]], None) == [[None], [None], [None]]


def test_cube_fit_returns_values_for_boxes():
    import cube_viewer as cv
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'c.fits')
        _sparse_otf_cube(p)
        cube = cv.load_spectral_cube(p, hdu_index=1, use_cache=False)
        mom = cv.collapse_map(cube, -3, 3, metric='peak')
        _iy, _ix, row = cv.peak_cell(cube, mom)
        out = app.fig_cube_spectrum(p, 1, {'rows': [int(row)]}, -3, 3, 1, True,
                                    spec_mode='pixel')
        guess = out[3]
        assert guess is not None and len(guess['amp']) == 1
        assert all(type(guess[k][0]) is float and np.isfinite(guess[k][0])
                   for k in ('amp', 'v0', 'fwhm'))
        assert guess['fwhm'][0] > 0
        nofit = app.fig_cube_spectrum(p, 1, {'rows': [int(row)]}, -3, 3, 1, False,
                                      spec_mode='pixel')
        assert nofit[3] is None
