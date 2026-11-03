"""Attenuation impact workflow: KoSens behaviour, KoSens parity, and the 3-D extension."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import atten_compare as ac
import grid_naming as gn


def _entry(values, x_mesh, y_mesh):
    # Key order matters: the two non-grid keys are read positionally as (y, x).
    return {"grid": values, "y_mesh": y_mesh, "x_mesh": x_mesh}


def _toy_grids():
    """``steep`` (1 dex/dex, attenuated ×1/3), ``flat`` (ill-conditioned), ``JCO`` (ice)."""
    x = np.logspace(-17.0, -14.0, 13)
    y = np.array([1.0e2, 1.0e3])
    x_mesh, y_mesh = np.meshgrid(x, y)
    steep_ref = x_mesh / 1.0e-17
    flat_ref = 1.0 + 0.002 * np.log10(x_mesh / 1.0e-17)
    ref = {"steep": _entry(steep_ref, x_mesh, y_mesh), "flat": _entry(flat_ref, x_mesh, y_mesh),
           "JCO": _entry(steep_ref, x_mesh, y_mesh)}
    att = {"steep": _entry(steep_ref / 3.0, x_mesh, y_mesh),
           "flat": _entry(flat_ref * 0.999, x_mesh, y_mesh),
           "JCO": _entry(steep_ref / 3.0, x_mesh, y_mesh)}
    return ref, att


def _pixels_from_spec(spec, n_x=8, n_y=6):
    """Hand-made pixel table: (name, R at G0=100, dR per dex G0, slope, shift, keep_to)."""
    x = np.logspace(-17.0, -14.0, n_x)
    y = np.logspace(1.0, 3.0, n_y)
    rows = []
    for name, base, fuv_dep, slope, shift, keep_to in spec:
        for iy, yv in enumerate(y):
            for ix, xv in enumerate(x):
                ref = 10.0 ** (slope * np.log10(xv))
                resp = base - fuv_dep * (np.log10(yv) - 2.0)
                sh = shift if ix <= keep_to else np.nan
                rows.append({"species": name, "ix": ix, "iy": iy, "x": xv, "y": yv,
                             "value_ref": ref, "value_atten": ref * 10.0 ** resp,
                             "slope_dex": slope, "shift_dex": sh, "abs_shift_dex": sh})
    return pd.DataFrame(rows)


RECONCILE_SPEC = [
    ("probe", -1.20, 0.00, 0.8, 0.9, 99),
    ("fuvtracer", -0.20, 0.70, 0.5, 1.5, 99),
    ("unmeasurable", -1.00, 0.00, 0.8, 0.9, 1),
]


# --- KoSens behaviour, ported ---------------------------------------------

def test_no_slope_gate_and_unmatched_as_zero():
    ref, att = _toy_grids()
    pixels = ac.collect_attenuation_x_shift_table(ref, att)
    # Every cell with a model in both grids has a finite shift (no match -> 0).
    assert np.isfinite(pixels["shift_dex"]).all()
    out = ac.shift_summary(pixels)
    g = out["species_summary"].set_index("species")
    assert (g["n_valid"] == g["n_cells"]).all()
    # No |S| gate: the flat species' shifts enter every median.
    assert np.isfinite(g.loc["flat", "median_abs_shift_dex"])
    assert "flat" in set(out["regime_median"]["species"])
    assert "JCO" not in g.index


def test_inference_error_equals_summary_median():
    ref, att = _toy_grids()
    pixels = ac.collect_attenuation_x_shift_table(ref, att)
    table = ac.response_table(pixels)["table"].set_index("species")
    assert abs(table.loc["steep", "error_dex"] - np.log10(3.0)) < 0.05
    assert np.isfinite(table.loc["flat", "error_dex"])
    assert "JCO" not in table.index
    # Same cells, same rule: the two steps report the same number.
    med = ac.shift_summary(pixels)["species_summary"].set_index("species")["median_abs_shift_dex"]
    np.testing.assert_allclose(table["error_dex"], med.reindex(table.index))


def test_swing_is_max_minus_min():
    assert ac._swing([0.0, 0.1, 0.2, 5.0, np.nan]) == 5.0
    # 3-D split: χ swing at fixed density, density swing at fixed χ (largest group wins).
    r = np.array([0.0, 1.0, 10.0, 10.5])
    g0, n = ac._split_swing(r, [1e1, 1e2, 1e1, 1e2], np.array([0, 0, 1, 1]))
    assert g0 == 1.0 and n == 10.0


def test_probe_panel_ranks_independently():
    pixels = _pixels_from_spec([("bigdelta", -0.30, 0.0, 0.10, 3.00, 99),
                                ("goodprobe", -1.20, 0.0, 2.00, 0.60, 99)])
    table = ac.response_table(pixels, max_species=1)["table"].set_index("species")
    assert table.loc["bigdelta", "ranked"] and not table.loc["goodprobe", "ranked"]
    assert table.loc["goodprobe", "probe_ranked"] and not table.loc["bigdelta", "probe_ranked"]


def test_constituent_cap():
    ranked = pd.DataFrame({"species": ["A/X", "A/Y", "A/Z", "B/X"]})
    kept, held = ac._cap_per_constituent(ranked, 3, max_per=2)
    assert list(kept["species"]) == ["A/X", "A/Y", "B/X"] and held == ["A/Z"]


def test_reconciliation_verdicts_and_views():
    pixels = _pixels_from_spec(RECONCILE_SPEC)
    out = ac.run_workflow(pixels)
    rec = out["verdicts"]["table"].set_index("species")
    assert rec.loc["probe", "verdict"] == "robust attenuation probe"
    assert rec.loc["fuvtracer", "verdict"] == "G0-modulated"
    # No honesty gate: a species matched in part of the grid is graded on all its cells.
    assert rec.loc["unmeasurable", "verdict"] in ac.VERDICT_ORDER
    assert set(out["verdicts"]["problematic"]["species"]) == set(rec.index)
    assert out["verdicts"]["problematic"].iloc[0]["species"] == "fuvtracer"
    assert out["verdicts"]["tracers"].iloc[0]["species"] == "probe"


def test_observability_judged_on_constituent_lines():
    x = np.logspace(-17.0, -14.0, 13)
    x_mesh, y_mesh = np.meshgrid(x, np.array([1.0e2, 1.0e3]))
    s = x_mesh / 1.0e-17
    ref = {"steep": _entry(s, x_mesh, y_mesh), "faint": _entry(1e-3 * s, x_mesh, y_mesh),
           "steep/faint": _entry(s, x_mesh, y_mesh)}
    att = {k: _entry(v["grid"] / 3.0, x_mesh, y_mesh) for k, v in ref.items()}
    pixels = ac.collect_attenuation_x_shift_table(ref, att, obs_limit=ac.OBS_INTENSITY_LIMIT)
    out = ac.run_workflow(pixels)
    rec = out["verdicts"]["table"].set_index("species")
    # Unobservable cells enter at Δ = 0 and R = 0: graded low, not excluded.
    assert rec.loc["steep/faint", "frac_observable"] < 0.5 < rec.loc["steep", "frac_observable"]
    assert rec.loc["steep/faint", "error_dex"] == 0.0
    assert rec.loc["steep/faint", "verdict"] == "no response"
    assert rec.loc["steep", "verdict"] == "robust attenuation probe"


def test_isotopologue_keys():
    assert ac._is_isotopologue_key("HCN(1-0)/H13CN(1-0)")
    assert not ac._is_isotopologue_key("SO(13.044GHz)")


# --- parity with KoSens ----------------------------------------------------

def _kosens():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from kosens.grid import grid_functions as gf
    except Exception:
        pytest.skip("kosens not importable")
    return gf, plt


SUMMARY_COLS = ["n_cells", "n_valid", "median_abs_shift_dex", "mean_abs_shift_dex",
                "std_abs_shift_dex", "max_abs_shift_dex", "frac_observable"]
RESPONSE_COLS = ["error_dex", "error_p10", "error_p90", "n_error_cells",
                 "response_dex", "abs_response_dex", "fuv_swing_dex", "g0_swing_dex",
                 "fuv_trend", "n_fuv_rows", "frac_observable", "cr_dominated", "den_contrib",
                 "ranked", "probe_ranked"]
RECONCILE_COLS = ["verdict", "cr_led", "matters", "error_dex", "abs_response_dex",
                  "g0_swing_dex"]


def _same(a, b, cols):
    a, b = a.set_index("species").sort_index(), b.set_index("species").sort_index()
    assert list(a.index) == list(b.index)
    for col in cols:
        left, right = a[col].to_numpy(), b[col].to_numpy()
        if left.dtype.kind in "fc" or right.dtype.kind in "fc":
            np.testing.assert_allclose(left.astype(float), right.astype(float),
                                       rtol=1e-12, equal_nan=True, err_msg=col)
        else:
            assert list(left) == list(right), col


@pytest.mark.parametrize("case", ["grids", "limit", "reconcile"])
def test_parity_with_kosens(case, monkeypatch):
    gf, plt = _kosens()
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    try:
        if case in ("grids", "limit"):
            ref, att = _toy_grids()
            lim = 30.0 if case == "limit" else None   # part of each row below the limit
            px_k = gf.collect_attenuation_x_shift_table(ref, att, obs_limit=lim)
            px_a = ac.collect_attenuation_x_shift_table(ref, att, obs_limit=lim)
            for col in ("shift_dex", "slope_dex"):
                np.testing.assert_allclose(px_a[col], px_k[col], equal_nan=True)
        else:
            px_k = px_a = _pixels_from_spec(RECONCILE_SPEC)
        ov_k = gf.plot_attenuation_shift_summary(None, None, pixel_df=px_k, summary_only=True)
        ov_a = ac.shift_summary(px_a)
        _same(ov_a["species_summary"], ov_k["species_summary"], SUMMARY_COLS)
        fu_k = gf.plot_attenuation_fuv_response(ov_k["pixels"], make_plots=False)
        fu_a = ac.response_table(ov_a["pixels"])["table"]
        _same(fu_a, fu_k, RESPONSE_COLS)
        rec_k = gf.reconcile_attenuation_species(ov_k, fu_k, verbose=False)["table"]
        rec_a = ac.reconcile(ov_a, fu_a)["table"]
        _same(rec_a, rec_k, RECONCILE_COLS)
    finally:
        plt.close("all")


# --- 3-D --------------------------------------------------------------------

def _cubes():
    """(ζ, χ, n) cubes: ``steady`` attenuated ×1/10 everywhere; ``lowonly`` only at low n_H."""
    z = np.logspace(-17.0, -14.0, 13)
    y = np.logspace(1.0, 3.0, 4)
    n = np.array([1e2, 1e3, 1e4, 1e5])
    zm, ym, nm = np.meshgrid(z, y, n, indexing="ij")
    ref = zm / 1e-17
    low = nm < 10 ** 3.5
    meshes = {gn.PARAM_MESH_KEYS["density"]: nm, gn.PARAM_MESH_KEYS["fuv"]: ym,
              gn.PARAM_MESH_KEYS["crir"]: zm}
    ref_c = {"steady": {"grid": ref, **meshes}, "lowonly": {"grid": ref, **meshes}}
    att_c = {"steady": {"grid": ref / 10.0, **meshes},
             "lowonly": {"grid": np.where(low, ref / 10.0, ref), **meshes}}
    return ref_c, att_c


def test_single_density_equals_filtered_3d():
    ref_c, att_c = _cubes()
    full = ac.collect_pixels(ref_c, att_c)
    one = ac.collect_pixels(ref_c, att_c, density_indices=[1])
    sub = full[full["i_n"] == 1]
    a = ac.response_table(one)["table"]
    b = ac.response_table(sub)["table"]
    _same(a, b, ["error_dex", "abs_response_dex", "fuv_swing_dex", "g0_swing_dex"])


def test_environment_verdict_flips_with_density():
    ref_c, att_c = _cubes()
    pixels = ac.collect_pixels(ref_c, att_c)
    edges = {"n": ac.parse_axis_edges(2, pixels["n"]), "y": [], "x": []}
    out = ac.run_workflow(pixels, env_edges=edges)
    env = out["environments"]["table"].set_index(["species", "environment"])
    assert env.loc[("lowonly", "low n_H"), "verdict"] == "robust attenuation probe"
    assert env.loc[("lowonly", "high n_H"), "verdict"] == "no response"
    assert env.loc[("steady", "high n_H"), "verdict"] == "robust attenuation probe"
    # Pooled, lowonly's response depends on density: the n_H swing names the cause.
    rt = out["response"]["table"].set_index("species")
    assert rt.loc["lowonly", "n_swing_dex"] > 0.5 > rt.loc["steady", "n_swing_dex"]


def test_figures_build():
    pixels = _pixels_from_spec(RECONCILE_SPEC)
    out = ac.run_workflow(pixels)
    assert ac.fig_shift_bars(out["summary"]).data
    assert ac.fig_regime_heatmaps(out["summary"]).data
    assert ac.fig_winner_map(out["summary"]).data
    assert ac.fig_error_bars(out["response"]["table"]).data
    assert ac.fig_probe_plane(out["response"]["table"]).data
    t = out["verdicts"]["table"].assign(col="pooled")
    assert ac.fig_verdict_matrix(t, col_key="col", col_order=["pooled"]).data


def test_environment_matrix_uses_level_strips():
    ref_c, att_c = _cubes()
    pixels = ac.collect_pixels(ref_c, att_c)
    edges = {c: ac.parse_axis_edges(2, pixels[c]) for c in ("n", "y", "x")}
    env = ac.run_workflow(pixels, env_edges=edges)["environments"]
    cols = [e["label"] for e in env["environments"]]
    fig = ac.fig_verdict_matrix(env["table"], col_key="environment", col_order=cols,
                                env_info=env["environments"])
    assert len(fig.data) == 1 + 3 and fig.layout.xaxis.showticklabels is False
    assert list(fig.data[1].text[0]) == ["low", "high"] * 4          # ζ strip, innermost



def test_probe_plane_profiles_and_zeta_regimes():
    px = _pixels_from_spec(RECONCILE_SPEC)
    wf = ac.run_workflow(px)
    tab, prof = wf["response"]["table"], wf["response"]["profiles"]
    assert set(prof) == set(tab["species"])
    for scale in ("log", "linear"):
        ac.fig_probe_plane(tab, axis_scale=scale).to_json()
    lim = ac.fig_probe_plane(tab, axis_min=0.05, axis_max=5.0).layout.xaxis.range
    assert list(lim) == pytest.approx([np.log10(0.05), np.log10(5.0)])
    assert len(ac.fig_profiles(tab, prof, x_scale="linear").data) >= 2
    z = ac.zeta_regime_verdicts(px, 3, {}, {}, {}, whole=wf["verdicts"]["table"])
    assert set(z["regimes"]["regime"]) == set(z["levels"])
    assert z["best"].set_index("species").loc["probe", "verdict"] == "robust attenuation probe"


def test_co_alpha_classes_and_answer():
    f = ac.CO_FREQ_GHZ
    # jerg: fluxes used as-is. Model 0: high-J = 0.1x low-J (class I); model 1: 1x (class III).
    lines = {j: np.array([1.0, 1.0]) for j in (5, 6, 7)}
    lines.update({j: np.array([0.1, 1.0]) for j in (11, 12, 13)})
    assert ac.co_alpha_classes(ac.co_alpha(lines, "jerg")) == {"I": 1, "II": 0, "III": 1}
    # jtemp: equal K km/s gives alpha = sum(nu^3 high)/sum(nu^3 low) > 1
    eq = {j: np.array([1.0]) for j in f}
    expect = sum(f[j] ** 3 for j in (11, 12, 13)) / sum(f[j] ** 3 for j in (5, 6, 7))
    assert np.isclose(ac.co_alpha(eq, "jtemp")[0], expect)
    assert ac.PROBE_ANSWER["G0-modulated"] == "G0-modulated"


def test_scan_prefers_near_crossing_over_far_node():
    """Non-monotonic row: the crossing between the first two nodes beats the far node.

    ``zs`` rises 0 -> 2 then falls back to 1, so the node at x = 100 sits exactly on
    the target while the segment x in [1, 10] crosses it first. Scanning every node
    before any segment returned the far branch.
    """
    xs, zs = np.array([1.0, 10.0, 100.0]), np.array([0.0, 2.0, 1.0])
    hit = ac._x_match_scan_sorted(xs, zs, 1.0, 0.05)
    assert hit < 10.0 and np.isclose(hit, 10.0 ** 0.475)


def test_regime_tally_counts_graded_boxes():
    ref_c, att_c = _cubes()
    pixels = ac.collect_pixels(ref_c, att_c)
    edges = {"n": ac.parse_axis_edges(2, pixels["n"]), "y": [], "x": []}
    env = ac.run_workflow(pixels, env_edges=edges)["environments"]
    tally = ac.regime_tally(env["table"]).set_index("species")
    # lowonly is a probe in one of its two n_H boxes; steady in both.
    assert tally.loc["lowonly", "boxes_probe"] == 1
    assert tally.loc["steady", "boxes_probe"] == tally.loc["steady", "boxes_graded"] == 2
    assert tally.loc["lowonly", "best_box"] == "low n_H"
    assert tally.loc["lowonly", "best_box_answer"] == "robust attenuation probe"
    # The column header has to carry the range, not just the word "low".
    labels = ac.env_col_labels(env["environments"], ["low n_H", "high n_H"])
    assert labels[0].startswith("low n_H<br>log n_H ≤ ")


def test_attenuated_mass_fraction_and_impact_matrix():
    import cr_attenuation as cra
    # Uniform n_H on r in [0, 1]: mass inside r <= 0.5 is 1/8 of the total.
    r = np.linspace(0.0, 1.0, 2001)
    n_eff = np.where(r <= 0.5, 2.0 * cra.DEFAULT_STOPPING_RATE, 0.0)
    f = cra.attenuated_mass_fraction(r, 4.0 * np.pi * r ** 2, n_eff)
    assert abs(f - 0.125) < 1e-3
    assert cra.attenuated_mass_fraction(r, 4.0 * np.pi * r ** 2, np.zeros_like(r)) == 0.0

    ref_c, att_c = _cubes()
    # f_att = 0 at the lowest density, 0.5 elsewhere.
    f_cube = dict(att_c["steady"], grid=np.where(att_c["steady"]["grid"] > 0, 0.5, 0.0))
    f_cube["grid"][:, :, 0] = 0.0
    pixels = ac.collect_pixels(ref_c, att_c, f_att_cube=f_cube)
    assert set(pixels.loc[pixels["i_n"] == 0, "f_att"]) == {0.0}
    edges = {"n": [1e2 * 10 ** 0.5], "y": [], "x": []}
    env = ac.run_workflow(pixels, env_edges=edges)["environments"]
    by = {e["label"]: e["f_att"] for e in env["environments"]}
    assert by["low n_H"] == 0.0 and by["high n_H"] == 0.5
    fig = ac.fig_impact_matrix(env["table"], env["environments"])
    z = np.asarray(fig.data[0].z, dtype=float)
    assert np.isnan(z[:, 0]).all() and np.isfinite(z[:, 1]).any()


def test_boxes_match_kosens_3d():
    """Per-box verdicts and numbers equal KoSens ``attenuation_regime_verdicts_3d``."""
    gf, plt = _kosens()
    from kosens.grid import grid_3d
    ref_c, att_c = _cubes()
    pixels = ac.collect_pixels(ref_c, att_c)
    edges = {c: ac.parse_axis_edges(3, pixels[c]) for c in ("n", "y", "x")}
    env = ac.run_workflow(pixels, env_edges=edges)["environments"]["table"]
    pix_k = pixels.assign(iz=pixels["i_n"], z=pixels["n"], z_name="n")
    try:
        views = grid_3d.attenuation_regime_verdicts_3d(
            None, None, pixel_df=pix_k, n_bins=3, require_catalog=False, plot_matrix=False)
    finally:
        plt.close("all")
    key = ["species", "n_level", "chi_level", "zeta_level"]
    k = views["regimes"].rename(columns={"G0_level": "chi_level"}).astype(
        {c: str for c in key[1:]}).set_index(key).sort_index()
    a = env.set_index(key).sort_index()
    assert list(a.index) == list(k.index) and len(a) == 2 * 27
    assert list(a["verdict"]) == list(k["verdict"])
    for col in ("error_dex", "abs_response_dex", "g0_swing_dex", "n_swing_dex"):
        np.testing.assert_allclose(a[col].astype(float), k[col].astype(float),
                                   rtol=1e-12, equal_nan=True, err_msg=col)


def test_probe_scope_and_3d_figures():
    ref_c, att_c = _cubes()
    pixels = ac.collect_pixels(ref_c, att_c)
    edges = {c: ac.parse_axis_edges(2, pixels[c]) for c in ("n", "y", "x")}
    wf = ac.run_workflow(pixels, env_edges=edges)
    env = wf["environments"]
    scope = ac.probe_scope(env["table"]).set_index("species")
    # steady: the same response in every box; lowonly: only in the low-n_H boxes.
    assert scope.loc["steady", "scope"] == "universal probe"
    assert scope.loc["steady", "n_boxes_responding"] == scope.loc["steady", "n_boxes"] == 8
    assert scope.loc["lowonly", "n_boxes_responding"] == 4
    assert ac.fig_probe_scope(scope.reset_index()).data
    assert ac.fig_probe_boxes(env["table"], wf["verdicts"]["table"], "lowonly").data
    fig = ac.fig_response_slices(env["table"], "lowonly")
    assert len(fig.data) == 2          # one panel per ζ level
    _m, ranking, _e, _na = ac.impact_ranking(env["table"], env["environments"])
    assert list(ranking["species"])[0] == "steady"


def test_species_selection_and_best_regimes():
    px = _pixels_from_spec([("CO", -1.0, 0.0, 0.8, 0.9, 99), ("XYZ", -1.0, 0.0, 0.8, 0.9, 99),
                            ("CO/XYZ", -1.0, 0.0, 0.8, 0.9, 99), ("H3+", -1.0, 0.0, 0.8, 0.9, 99),
                            ("C2H", -1.0, 0.0, 0.8, 0.9, 99), ("JCO", -1.0, 0.0, 0.8, 0.9, 99)])
    kept, dropped = ac.drop_uncatalogued(px, known={"CO": "CDMS", "CCH": "CDMS"})
    # Alias (C2H = CCH) and IR-only species (H3+) stay; a ratio needs both sides; ice is
    # left to exclude_ice.
    assert set(dropped["species"]) == {"XYZ", "CO/XYZ"}
    assert set(kept["species"]) == {"CO", "H3+", "C2H", "JCO"}

    ref_c, att_c = _cubes()
    pixels = ac.collect_pixels(ref_c, att_c)
    edges = {c: ac.parse_axis_edges(2, pixels[c]) for c in ("n", "y", "x")}
    wf = ac.run_workflow(pixels, env_edges=edges)
    env = wf["environments"]
    best = ac.best_regimes(env["table"], wf["verdicts"]["table"]).set_index("species")
    assert list(best.index) == ["steady", "lowonly"]          # most CRIR-probe boxes first
    assert best.loc["steady", "n_cr_led"] == 8 and best.loc["lowonly", "n_cr_led"] == 4
    _m, ranking, _e, _na = ac.impact_ranking(env["table"], env["environments"],
                                             exclude_species=["steady"], top_n=1)
    assert list(ranking["species"]) == ["lowonly"] and ranking["shown"].all()
    fig = ac.fig_impact_matrix(env["table"], env["environments"], metric="abs_response_dex")
    assert len(fig.data) == 1 + 3                             # matrix + one strip per axis


def test_row_median_delta_matches_kosens_cells():
    gf, plt = _kosens()
    plt.close("all")
    rng = np.random.default_rng(0)
    x = np.tile(np.logspace(-17, -14, 9), (5, 1))
    g0 = 10 ** rng.normal(size=x.shape)
    g1 = g0 * 10 ** rng.normal(scale=0.3, size=x.shape)
    g1[0, 0] = np.nan
    kw = dict(match_rtol=0.02, scan_direction="rightward")
    sh_a = ac.horizontal_intensity_matched_x_shift_dex(g0, g1, x, **kw)
    sh_k = gf.horizontal_intensity_matched_x_shift_dex(g0, g1, x, **kw)
    np.testing.assert_allclose(sh_a, sh_k, equal_nan=True)   # non-monotonic rows too
    obs = (g0 >= 0.1) & (g1 >= 0.1)
    np.testing.assert_allclose(ac.error_cells(sh_a, g0, g1, x, obs),
                               gf._error_cells(sh_k, g0, g1, x, obs), equal_nan=True)
    d_row, ok = ac.row_median_delta(sh_a, g0, g1, x, obs)
    assert ok.all() and d_row.shape == (5,)


def test_observer_error_table():
    # X ∝ ζ, χ-independent; attenuated ×1/10 → Δ = Δ_lin = 1 dex, S_χ = 0, D = 1.
    ref_c, att_c = _cubes()
    pixels = ac.collect_pixels(ref_c, att_c)
    edges = {"n": ac.parse_axis_edges(2, pixels["n"]), "y": [], "x": []}
    tab = ac.observer_error_table(pixels, ref_c, edges).set_index(["species", "environment"])
    row = tab.loc[("steady", "low n_H")]
    np.testing.assert_allclose([row["S_zeta_med"], row["S_chi_med"], row["D_med"],
                                row["r_med"], row["Delta_lin_med"]], [1, 0, 1, 0, 1],
                               atol=1e-9)
    assert abs(row["lin_ratio_med"] - 1) < 0.05 and row["reliable"]
    # lowonly is unattenuated at high n_H: every Δ = 0, so the box is not reliable.
    assert tab.loc[("lowonly", "high n_H"), "frac_zero"] == 1.0
    assert not tab.loc[("lowonly", "high n_H"), "reliable"]
    assert "steady & low n_H" in ac.observer_table_tex(tab.reset_index())


def test_windowed_slope_has_no_staircase_and_matches_nodes():
    # KoSens tests/test_windowed_slope.py: curved log X on 1-dex nodes, interpolated to 0.1 dex.
    nodes = np.arange(-18.0, -12.0)
    lx = np.linspace(-18.0, -13.0, 51)
    li = np.interp(lx, nodes, 0.1 * (nodes + 16) ** 2)
    s = ac._log_intensity_slope(li[None, :], lx[None, :])[0]
    inner = (lx >= nodes[0] + 0.5) & (lx <= nodes[-1] - 0.5)
    assert np.all(np.diff(s[inner]) > 0)
    raw = ac._log_intensity_slope((0.1 * (nodes + 16) ** 2)[None, :], nodes[None, :])[0]
    assert np.allclose(raw, np.gradient(0.1 * (nodes + 16) ** 2, nodes))
    assert np.allclose(s[::10][1:-1], raw[1:-1])


def test_impact_matrix_d_mode_and_species_order():
    ref_c, att_c = _cubes()
    pixels = ac.collect_pixels(ref_c, att_c)
    edges = {c: ac.parse_axis_edges(2, pixels[c]) for c in ("n", "y", "x")}
    env = ac.run_workflow(pixels, env_edges=edges)["environments"]
    obs = ac.observer_error_table(pixels, ref_c, edges)
    _m, d_rank, _e, _na = ac.impact_ranking(obs, env["environments"], metric="D_med")
    order = d_rank.loc[d_rank["shown"], "species"].tolist() + ["absent"]
    fig = ac.fig_impact_matrix(obs, env["environments"], metric="D_med")
    assert fig.data[0].zmax == 1.0
    paired = ac.fig_impact_matrix(env["table"], env["environments"], species_order=order)
    assert len(paired.data[0].y) == len(order)              # missing species: blank row
    assert not np.isfinite(np.asarray(paired.data[0].z, float)[-1]).any()
    compact = ac.observer_compact(obs)
    assert {"Delta_plus", "r_minus", "D_med", "reliable"} <= set(compact.columns)


def _binned_pixels():
    rows = [dict(species=sp, i_n=k, n=10.0 ** (2 + k), iy=j, ix=i, x=10.0 ** (-17 + i),
                 y=10.0 ** j, x_name="zeta", y_name="fuv", shift_dex=0.1 * (i + j + k + s),
                 abs_shift_dex=0.1 * (i + j + k + s), slope_dex=1.0, value_ref=1.0,
                 value_atten=1.0, observable=True, response_dex=0.1 * (i + 1))
            for s, sp in enumerate(["CO", "HCN", "CS"])
            for k in range(6) for j in range(4) for i in range(4)]
    return pd.DataFrame(rows)


def test_level_strips_sit_on_their_own_rows():
    env_info = ac.assign_environments(
        pd.DataFrame({"n": [1.0, 100.0], "y": [1.0, 100.0], "x": [1.0, 100.0]}),
        {"n": [10.0], "y": [10.0], "x": [10.0]})[1]
    for k, env in enumerate(env_info):
        env["f_att"] = k / 8
    fig = ac.fig_env_picker(env_info, [e["label"] for e in env_info], env_info[1]["label"])
    assert [tr.y[0] for tr in fig.data] == [0, 1, 2, 3]
    assert list(fig.layout.yaxis.ticktext) == ["χ", "n_H", "ζ", "f_att"]
    frame = fig.layout.shapes[-1]
    cols = list(fig.data[0].x)
    assert frame.x0 == cols.index(env_info[1]["label"]) - 0.5


def test_step1_uses_the_requested_density_bins():
    pixels = _binned_pixels()
    edges = {key: ac.parse_axis_edges("3", pixels[col])
             for key, col in (("n", "n"), ("y", "y"), ("x", "x"))}
    out = ac.shift_summary(pixels, regime_edges=edges)
    assert out["heat_n"].shape[1] == 3 and len(out["x_labels"]) == 3 == len(out["y_labels"])
    fig = ac.fig_winner_map(out)
    assert len(fig.data) == 3
    assert len({tr.colorbar.x for tr in fig.data}) == 3
    assert len({tr.zmax for tr in fig.data}) == 1
    tagged = ac.fig_winner_map(out, fmt=lambda name: f"{name}<sub>2</sub>")
    assert all("</<br>" not in txt for tr in tagged.data for row in tr.text for txt in row)
    # "full" keeps one bin; without regime_edges every density stays its own regime.
    assert ac.shift_summary(pixels, regime_edges={**edges, "n": []})["heat_n"].shape[1] == 1
    assert ac.shift_summary(pixels)["heat_n"].shape[1] == 6


def test_every_species_is_drawn_ten_per_figure():
    pixels = pd.concat([_binned_pixels().assign(species=f"X{i}") for i in range(23)])
    table = ac.response_table(pixels, max_species=None)["table"]
    assert table["ranked"].all()
    pages = ac.species_pages(table, "ranked", 10)
    assert [len(p) for p in pages] == [10, 10, 3]
    assert sorted(pd.concat(pages)["species"]) == sorted(table["species"])
    assert ac.fig_error_bars(pages[2], scale_table=table).layout.xaxis.range is not None


def test_breadth_draws_every_species_on_one_scale():
    pixels = pd.concat([_binned_pixels().assign(species=f"X{i}", shift_dex=0.05 * i,
                                                abs_shift_dex=0.05 * i) for i in range(23)])
    out = ac.shift_summary(pixels, n_top_summary_species=None)
    assert len(out["top_species"]) == 23 and not out["notes"]
    pages = ac.species_pages(out["species_summary"], None, 10)
    assert [len(p) for p in pages] == [10, 10, 3]
    everything = out["species_summary"]["species"].tolist()
    last = pages[2]["species"].tolist()
    bars = ac.fig_shift_bars(out, last, scale_species=everything)
    heat = ac.fig_regime_heatmaps(out, last, scale_species=everything)
    full = ac.fig_regime_heatmaps(out, everything)
    assert bars.layout.yaxis2.range[1] > 1.0  # the largest median, not this page's
    assert heat.data[0].zmax == full.data[0].zmax


def test_impact_ranking_masks_on_the_tables_own_f_att():
    info = [{"label": "a", "ranges": {}, "levels": {}, "f_att": 0.5},
            {"label": "b", "ranges": {}, "levels": {}, "f_att": 0.5}]
    table = pd.DataFrame({"species": ["CO", "CO"], "environment": ["a", "b"],
                          "D_med": [0.2, 0.8], "f_att": [0.05, 0.5]})
    _matrix, ranking, _envs, no_atten = ac.impact_ranking(table, info, metric="D_med", min_f_att=0.1)
    assert list(no_atten) == [True, False]
    assert ranking["median_box_D_med"].iloc[0] == 0.8
