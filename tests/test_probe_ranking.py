"""CRIR probe ranking: score, regimes, and 3-D ranking tables."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import probe_ranking as pr


def _curve(crir, slope=1.0, scale=1e-6):
    return scale * (np.asarray(crir, dtype=float) ** slope)


def test_probe_score_monotonic_high():
    crir = np.logspace(-18, -14, 9)
    y = 1e-6 * (crir / crir.min()) ** 0.8
    out = pr.probe_score(crir, y, min_abundance=1e-12)
    assert out['observable']
    assert out['trend'] == 'increasing'
    assert out['probe_score'] > 0.3
    assert out['functional_factor'] > 0.5
    assert out['variation_factor'] > 0.2
    assert out['n_reversals'] == 0


def test_probe_score_plateau_is_zero():
    crir = np.logspace(-18, -14, 9)
    y = np.full(crir.shape, 1e-4)
    out = pr.probe_score(crir, y, min_abundance=1e-12)
    assert out['observable']
    assert out['trend'] == 'plateau'
    assert out['probe_score'] == 0.0
    assert out['functional_factor'] == 0.0


def test_probe_score_not_observable():
    crir = np.logspace(-18, -14, 9)
    y = _curve(crir, slope=1.0, scale=1e-20)
    out = pr.probe_score(crir, y, min_abundance=1e-9)
    assert not out['observable']
    assert out['trend'] == 'not_observable'
    assert out['probe_score'] == 0.0


def test_probe_score_regime_restriction():
    crir = np.logspace(-18, -14, 17)
    y = np.where(crir <= 1e-16, 1e-4, 1e-6 * (crir / 1e-16) ** 1.0)
    full = pr.probe_score(crir, y, min_abundance=1e-12)
    soft = pr.probe_score(crir, y, min_abundance=1e-12, crir_range=(crir.min(), 1e-16))
    hard = pr.probe_score(crir, y, min_abundance=1e-12, crir_range=(1e-16, crir.max()))
    assert soft['trend'] == 'plateau'
    assert soft['probe_score'] == 0.0
    assert hard['probe_score'] > soft['probe_score']
    assert np.isfinite(full['probe_score'])


def test_build_crir_regimes_default_split():
    regimes = pr.build_crir_regimes(None, 1e-18, 1e-14)
    assert len(regimes) == 2
    assert regimes[0]['hi'] == 1e-16
    assert regimes[1]['lo'] == 1e-16


def test_parse_regime_edges():
    assert pr.parse_regime_edges('') == [1e-16]
    assert pr.parse_regime_edges('full') == []
    assert pr.parse_regime_edges('1e-17, 1e-15') == [1e-17, 1e-15]


def test_absolute_score_band():
    assert pr.absolute_score_band(0.7) == 'strong'
    assert pr.absolute_score_band(0.4) == 'good'
    assert pr.absolute_score_band(0.25) == 'moderate'
    assert pr.absolute_score_band(0.05) == 'weak'
    assert pr.absolute_score_band(0.0) == 'inactive'


def test_assign_quartiles():
    labels = pr.assign_quartiles([0.9, 0.6, 0.4, 0.2, 0.0, np.nan])
    assert labels[0] == 'green'
    assert labels[-2] == 'gray'
    assert labels[-1] == 'gray'


def _synthetic_grids():
    n = np.array([1e1, 1e3, 1e5])
    fuv = np.array([1e1, 1e2, 1e5])
    crir = np.logspace(-17, -14, 7)
    z, y, x = np.meshgrid(crir, fuv, n, indexing='ij')
    good = 1e-6 * (z / z.min()) ** 0.9
    flat = np.full(x.shape, 3e-5)
    weak = 1e-6 * (1.0 + 0.02 * np.log10(np.maximum(z / z.min(), 1.0)))
    mesh = {'densities': x, 'fuv_values': y, 'crir_values': z}
    return {
        'CO': {'grid': good, **mesh},
        'HCN': {'grid': weak, **mesh},
        'N2': {'grid': flat, **mesh},
    }


def test_compute_probe_ranking_orders_tracers():
    grids = _synthetic_grids()
    ranking = pr.compute_probe_ranking(
        grids,
        min_abundance=1e-12,
        crir_regimes=[1e-16],
        good_score_threshold=0.3,
    )
    assert ranking['species_list'] == ['CO', 'HCN', 'N2']
    assert len(ranking['crir_regimes']) == 2
    summary = ranking['summary']
    by_sp = {r['Species']: r for r in summary if r['Regime'] == ranking['crir_regimes'][-1]['name']}
    assert by_sp['CO']['Median_Probe_Score'] > by_sp['N2']['Median_Probe_Score']
    env = ranking['environment_rankings']
    leaders = [r for r in env if r['Rank'] == 1]
    assert leaders
    assert {r['Species'] for r in leaders} <= {'CO', 'HCN', 'N2'}
    rob = ranking['robustness']
    assert rob[0]['Species'] in ('CO', 'HCN', 'N2')
    crir, y = pr.extract_curve_at_environment(grids, 'CO', 1e3, 1e2)
    assert crir.size == 7
    assert np.all(np.diff(crir) > 0)


def test_add_ratio_grids():
    grids = _synthetic_grids()
    pr.add_ratio_grids(grids, [('CO', 'HCN')])
    assert 'CO/HCN' in grids
    np.testing.assert_allclose(
        grids['CO/HCN']['grid'],
        grids['CO']['grid'] / grids['HCN']['grid'],
        equal_nan=True,
    )


def test_fig_builders_return_figures():
    grids = _synthetic_grids()
    ranking = pr.compute_probe_ranking(grids, min_abundance=1e-12)
    env_fig = pr.fig_environment_ranking(
        ranking, n_values=[1e1, 1e5], fuv_values=[1e1, 1e5], top_n=3,
    )
    assert env_fig is not None
    assert len(env_fig.data) >= 1
    hm = pr.fig_score_heatmap(ranking, 'CO')
    assert hm is not None
    assert 260 <= hm.layout.height <= 360
    assert not (getattr(hm.layout.title, 'text', None) or '')
    n_one = sum(getattr(tr, 'type', None) == 'heatmap' for tr in hm.data)
    assert n_one >= 1
    layout = hm.layout.to_plotly_json()
    xaxes = [v for k, v in layout.items() if str(k).startswith('xaxis') and isinstance(v, dict)]
    assert any(v.get('showexponent') == 'none' for v in xaxes)
    ticks = [t for v in xaxes for t in (v.get('ticktext') or [])]
    assert ticks
    assert all('$' not in str(t) for t in ticks)
    assert any('10' in str(t) for t in ticks)
    yaxes = [v for k, v in layout.items() if str(k).startswith('yaxis') and isinstance(v, dict)]
    titles = [(v.get('title') or {}).get('text') or '' for v in xaxes + yaxes]
    assert any('G' in t or 'n' in t for t in titles)
    assert all('$' not in t for t in titles)
    many = pr.score_heatmap_figures(ranking, ['CO', 'HCN', 'N2'])
    assert len(many) == 3
    assert all(fig is not None for _sp, fig in many)
    trends = pr.fig_response_curves(
        grids, ranking, species_list=['CO', 'HCN'], n_level=1e3,
        fuv_values=[1e1, 1e5],
    )
    assert trends is not None


def test_ranking_at_environments_empty_selection_keeps_rows():
    ranking = pr.compute_probe_ranking(_synthetic_grids(), min_abundance=1e-12)
    all_rows = ranking['environment_rankings']
    empty = pr.ranking_at_environments(ranking, n_values=[], fuv_values=[])
    assert len(empty) == len(all_rows)
    fig = pr.fig_environment_ranking(ranking, n_values=[], fuv_values=[])
    assert fig is not None
    assert len(fig.data) >= 1


def test_environment_ranking_one_figure_per_regime():
    ranking = pr.compute_probe_ranking(_synthetic_grids(), min_abundance=1e-12)
    names = [r['name'] for r in ranking['crir_regimes']]
    assert len(names) == 2
    figs = pr.environment_ranking_figures(
        ranking, n_values=[1e1, 1e5], fuv_values=[1e1, 1e5], top_n=3,
    )
    assert len(figs) == 2
    assert all(fig is not None for fig in figs)
    titles = [str(fig.layout.title.text) for fig in figs]
    assert titles[0] != titles[1]


def test_heatmap_keeps_both_regimes():
    ranking = pr.compute_probe_ranking(_synthetic_grids(), min_abundance=1e-12)
    fig = pr.fig_score_heatmap(ranking, 'CO', regime=None)
    n_heat = sum(getattr(tr, 'type', None) == 'heatmap' for tr in fig.data)
    assert n_heat >= 1
    assert len(fig.layout.annotations or []) + n_heat >= 2
    low = ranking['crir_regimes'][0]['name']
    msg = pr.explain_regime_plot(ranking, low, species=['N2'])
    assert msg is not None
    assert 'S = 0' in msg or 'ζ' in msg or 'informative' in msg
    green = pr.species_in_quartile(ranking, 'green')
    assert green
    assert 'CO' in green


def test_heatmap_uses_median_on_native_levels():
    ranking = pr.compute_probe_ranking(_synthetic_grids(), min_abundance=1e-12)
    rows = ranking['slice_scores']
    dup = dict(rows[0])
    dup['probe_score'] = rows[0]['probe_score'] * 0.5
    panel = rows[:1] + [dup]
    n_vals, fuv_vals, z = pr._heatmap_pivot(panel, value_col='probe_score')
    assert z.size == 1
    assert abs(z[0, 0] - 0.75 * rows[0]['probe_score']) < 1e-12


def test_kosens_default_environments():
    n = [10.0, 100.0, 1e3, 1e4, 1e5]
    fuv = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
    assert pr.default_kosens_env_values(n, pr.KOSENS_DEFAULT_N_ENV) == [10.0, 1e3, 1e5]
    assert pr.default_kosens_env_values(fuv, pr.KOSENS_DEFAULT_FUV_ENV) == [10.0, 100.0, 1e5]


def test_extract_curve_snaps_off_node_fuv():
    """Interpolated n/G₀ nodes are not exact decades; snap so all G₀ curves appear."""
    n = np.logspace(1, 5, 23)
    fuv = np.logspace(1, 5, 23)
    crir = np.logspace(-17, -14, 9)
    assert not np.any(np.isclose(fuv, 1e2, rtol=1e-9, atol=0))
    z, y, x = np.meshgrid(crir, fuv, n, indexing='ij')
    grids = {
        'CO': {
            'grid': 1e-6 * (z / z.min()) ** 0.9,
            'densities': x,
            'fuv_values': y,
            'crir_values': z,
        },
    }
    for g0 in (1e1, 1e2, 1e5):
        c, yv = pr.extract_curve_at_environment(grids, 'CO', 1e3, g0)
        assert c.size >= 2
        assert yv.size == c.size
    ranking = {
        'n_levels': [1e1, 1e3, 1e5],
        'fuv_levels': [1e1, 1e2, 1e5],
        'summary': [],
        'species_list': ['CO'],
        'crir_regimes': [],
        'environment_rankings': [],
        'quantity': 'rel_abund',
    }
    fig = pr.fig_response_curves(
        grids, ranking, species_list=['CO'], n_level=1e3,
        fuv_values=[1e1, 1e2, 1e5], shade_plateaus=False,
    )
    n_lines = sum(
        1 for tr in fig.data
        if 'lines' in str(getattr(tr, 'mode', '') or '')
        and tr.x is not None and len(tr.x) >= 2
    )
    assert n_lines == 3


def test_environment_density_label_stays_in_margin():
    ranking = pr.compute_probe_ranking(_synthetic_grids(), min_abundance=1e-12)
    fig = pr.fig_environment_ranking(
        ranking, n_values=[1e1, 1e5], fuv_values=[1e1, 1e5], top_n=3,
    )
    layout = fig.layout.to_plotly_json()
    ytitles = [
        (v.get('title') or {}).get('text') or ''
        for k, v in layout.items()
        if str(k).startswith('yaxis') and isinstance(v, dict)
    ]
    assert any(r'n_{\mathrm{H}}' in t or '$10^{' in t for t in ytitles)
    assert any('$10^{' in t for t in ytitles)
    assert not any('<sub>' in t or '<sup>' in t for t in ytitles)


def test_absolute_quartile_labels():
    assert 'green' in pr.absolute_quartile_label(0.7)
    assert 'gray' in pr.absolute_quartile_label(0.0)
    assert pr._snap_native_level(1.05e5) == 1e5
    assert pr._snap_native_level(9.2) == 10.0
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    assert abs(pr._theil_slope(y, x) - 2.0) < 1e-12


def test_level_options_use_unicode():
    opts = pr.level_options([1e1, 1e3, 1e5])
    labels = [o['label'] for o in opts]
    assert labels == ['10¹', '10³', '10⁵']
    assert pr.sci_plain(1e-16) == '10⁻¹⁶'
    assert pr.sci_latex(1e-16) == '$10^{-16}$'
    assert pr.sci_tex(1e3) == '10^{3}'


def test_axis_and_regime_labels_use_latex():
    fuv = pr.axis_display_label('fuv_values', html=False)
    assert fuv.startswith('$') and fuv.endswith('$')
    assert r'G_0' in fuv
    dens = pr.axis_display_label('densities', html=False)
    assert r'n_{\mathrm{H}}' in dens
    assert '<sub>' not in dens and '<sup>' not in dens
    lab = pr.crir_regime_label(1e-16, 1e-13)
    assert lab.startswith('$') and r'10^{-16}' in lab and r'\mathrm{s}^{-1}' in lab
    plain = pr.crir_regime_label(1e-16, 1e-13, wrap=False)
    assert plain == '10⁻¹⁶–10⁻¹³ s⁻¹'
    assert '^{' not in plain
    assert '<sub>' not in lab and '<sup>' not in lab
