import math

import numpy as np

import transition_fronts as tf

T = dict(paper_bg='#18181B', plot_bg='#141416', grid='#27272A', axis_line='#3F3F46',
         title='#FAFAFA', font='#E4E4E7', muted='#A1A1AA', legend_bg='rgba(24,24,27,0.94)',
         legend_border='#3F3F46', accent='#E4E4E7')
LABELS = ('x', 'y', 'z')


def test_front_depth_interpolates_first_crossing():
    av = np.linspace(0.0, 10.0, 11)
    assert math.isclose(tf.front_depth(av, av / 10.0, 0.25), 2.5)
    # C+ falls inward: reached where the fraction drops to the level.
    assert math.isclose(tf.front_depth(av, 1.0 - av / 10.0, 0.25, rising=False), 7.5)


def test_front_depth_surface_and_unreached():
    av = np.linspace(0.0, 5.0, 6)
    assert tf.front_depth(av, np.full(6, 0.9), 0.25) == 0.0      # molecular at the surface
    assert np.isnan(tf.front_depth(av, np.zeros(6), 0.25))        # clump too thin
    assert np.isnan(tf.front_depth([], [], 0.25))


def test_front_fractions_normalise_carbon_and_drop_missing():
    rel = np.array([[0.1, 2e-4, 1e-5, 1e-6],
                    [0.4, 5e-5, 5e-5, 1e-4]])
    idx = {'H2': 0, 'C+': 1, 'C': 2, 'CO': 3}
    fr = tf.front_fractions(rel, idx)
    x_c = rel[:, 1] + rel[:, 2] + rel[:, 3]
    assert np.allclose(fr['cplus'], rel[:, 1] / x_c)
    assert np.allclose(fr['co'], rel[:, 3] / x_c)
    assert np.allclose(fr['h2'], rel[:, 0])
    assert set(tf.front_fractions(rel, {'H2': 0})) == {'h2'}


def test_column_fractions_and_zones():
    levels = {f['key']: f['default_level'] for f in tf.FRONTS}
    fr = tf.column_fractions({'H': 1.0, 'H2': 1.0, 'C+': 1.0, 'C': 1.0, 'CO': 2.0})
    assert math.isclose(fr['h2'], 1.0 / 3.0)
    assert math.isclose(fr['co'], 0.5)
    assert tf.zone_of(fr, levels) == ('CO', True)
    assert tf.zone_of({'cplus': 0.9, 'co': 0.0, 'h2': 0.01}, levels) == ('C+', False)
    assert tf.zone_of({'cplus': 0.1, 'co': 0.2, 'h2': 0.4}, levels) == ('C', True)


def test_fill_nan_nearest():
    vol = np.array([[1.0, np.nan], [3.0, 4.0]])
    out = tf.fill_nan_nearest(vol)
    assert np.isfinite(out).all()
    assert out[0, 1] in (1.0, 4.0)


def test_figures_build_with_holes():
    x, y = np.arange(4.0), np.arange(3.0)
    depth = {'h2': np.where(np.arange(12).reshape(3, 4) % 5 == 0, np.nan, 1.0),
             'co': np.full((3, 4), 2.0)}
    fig = tf.fig_front_surfaces(x, y, depth, T, LABELS, av_center=np.full((3, 4), 5.0))
    assert [tr.type for tr in fig.data] == ['surface'] * 3

    X, Y, Z = np.arange(4.0), np.arange(3.0), np.arange(2.0)
    GX = np.meshgrid(X, Y, Z, indexing='ij')[0]
    cube = {'co': GX / 3.0, 'cplus': 1.0 - GX / 3.0, 'h2': GX / 6.0}
    cube['co'][1, 1, 1] = np.nan
    levels = {f['key']: f['default_level'] for f in tf.FRONTS}
    fig = tf.fig_regime_cube(X, Y, Z, cube, levels, T, LABELS)
    kinds = [tr.type for tr in fig.data]
    assert kinds.count('isosurface') == 3
    assert 'scatter3d' in kinds


def test_classify_media_vectorised():
    levels = {f['key']: f['default_level'] for f in tf.FRONTS}
    fr = {'cplus': np.array([0.9, 0.1, 0.1, np.nan]),
          'co': np.array([0.0, 0.2, 0.8, np.nan]),
          'h2': np.array([0.01, 0.4, 0.5, np.nan])}
    medium, molecular = tf.classify_media(fr, levels)
    assert medium.tolist() == [0, 1, 2, -1]
    assert molecular.tolist() == [False, True, True, False]


def test_refine_is_linear_in_plot_coords():
    x, y = np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0])
    vals = y[:, None] + x[None, :]          # shape [len(y), len(x)]
    (fy, fx), out = tf.refine([y, x], vals, 5)
    assert out.shape == (5, 5)
    assert np.allclose(out, fy[:, None] + fx[None, :])


def test_media_map_and_volume_styles():
    levels = {f['key']: f['default_level'] for f in tf.FRONTS}
    x, y = np.arange(4.0), np.arange(3.0)
    gx = np.meshgrid(x, y)[0]
    fr = {'cplus': 1.0 - gx / 3.0, 'co': gx / 3.0, 'h2': gx / 6.0}
    filled = tf.fig_media_map(x, y, fr, levels, T, LABELS[:2])
    kinds = [tr.type for tr in filled.data]
    assert kinds.count('heatmap') == 1 and kinds.count('contour') == 3
    points = tf.fig_media_map(x, y, fr, levels, T, LABELS[:2], style='points')
    assert {tr.type for tr in points.data} == {'scatter'}

    X, Y, Z = np.arange(4.0), np.arange(3.0), np.arange(2.0)
    GX = np.meshgrid(X, Y, Z, indexing='ij')[0]
    cube = {'co': GX / 3.0, 'cplus': 1.0 - GX / 3.0, 'h2': GX / 6.0}
    fig = tf.fig_regime_cube(X, Y, Z, cube, levels, T, LABELS, style='volumes')
    assert {tr.type for tr in fig.data} == {'isosurface'}
