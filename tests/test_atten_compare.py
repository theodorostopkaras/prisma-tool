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

def test_slope_gate_keeps_counts_and_drops_medians():
    ref, att = _toy_grids()
    pixels = ac.collect_attenuation_x_shift_table(ref, att)
    g = ac.shift_summary(pixels, min_slope_dex=0.1)["species_summary"].set_index("species")
    u = ac.shift_summary(pixels, min_slope_dex=0.0)["species_summary"].set_index("species")
    assert (g["n_valid"] == u["n_valid"]).all()
    assert g.loc["flat", "n_trusted"] == 0 and np.isnan(g.loc["flat", "median_abs_shift_dex"])
    assert u.loc["flat", "n_trusted"] == u.loc["flat", "n_valid"]
    assert "JCO" not in g.index


def test_inference_error_and_censoring():
    ref, att = _toy_grids()
    table = ac.response_table(ac.collect_attenuation_x_shift_table(ref, att))["table"]
    table = table.set_index("species")
    assert abs(table.loc["steep", "error_dex"] - np.log10(3.0)) < 0.05
    assert table.loc["steep", "frac_censored"] > 0.0
    assert np.isnan(table.loc["flat", "error_dex"])
    assert "JCO" not in table.index


def test_censored_majority_is_not_ranked():
    ref, att = _toy_grids()
    pixels = ac.collect_attenuation_x_shift_table(ref, att)
    steep = pixels["species"] == "steep"
    pixels.loc[steep, ["shift_dex", "abs_shift_dex"]] = np.nan
    table = ac.response_table(pixels)["table"].set_index("species")
    assert table.loc["steep", "frac_censored"] == 1.0 and not table.loc["steep", "ranked"]
    relaxed = ac.response_table(pixels, max_frac_censored=1.0)["table"].set_index("species")
    assert relaxed.loc["steep", "ranked"]


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
    assert rec.loc["probe", "verdict"] == "CRIR probe"
    assert rec.loc["fuvtracer", "verdict"] == "FUV tracer"
    assert rec.loc["unmeasurable", "verdict"] == "unmeasurable"
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
    assert rec.loc["steep/faint", "verdict"] == "unobservable"
    assert rec.loc["steep", "verdict"] != "unobservable"


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


SUMMARY_COLS = ["n_cells", "n_valid", "n_unmatched", "n_trusted", "n_shifted", "n_affected",
                "frac_shifted", "frac_unmatched", "frac_trusted", "median_abs_shift_dex",
                "mean_abs_shift_dex", "max_abs_shift_dex", "frac_observable", "ranked"]
RESPONSE_COLS = ["error_dex", "error_p10", "error_p90", "frac_censored", "n_error_cells",
                 "response_dex", "abs_response_dex", "fuv_swing_dex", "fuv_trend",
                 "frac_trusted", "n_fuv_rows", "probe_margin", "den_contrib", "ranked",
                 "probe_ranked"]
RECONCILE_COLS = ["verdict", "broad", "honest", "cr_led", "matters", "bound_gap"]


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


@pytest.mark.parametrize("case", ["grids", "reconcile"])
def test_parity_with_kosens(case, monkeypatch):
    gf, plt = _kosens()
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    try:
        if case == "grids":
            ref, att = _toy_grids()
            px_k = gf.collect_attenuation_x_shift_table(ref, att)
            px_a = ac.collect_attenuation_x_shift_table(ref, att)
            for col in ("shift_dex", "slope_dex"):
                np.testing.assert_allclose(px_a[col], px_k[col], equal_nan=True)
        else:
            px_k = px_a = _pixels_from_spec(RECONCILE_SPEC)
        ov_k = gf.plot_attenuation_shift_summary(None, None, pixel_df=px_k, summary_only=True)
        ov_a = ac.shift_summary(px_a)
        _same(ov_a["species_summary"], ov_k["species_summary"], SUMMARY_COLS)
        fu_k = gf.plot_attenuation_fuv_response(ov_k["pixels"])
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
    _same(a, b, ["error_dex", "frac_censored", "abs_response_dex", "swing_dex"])


def test_environment_verdict_flips_with_density():
    ref_c, att_c = _cubes()
    pixels = ac.collect_pixels(ref_c, att_c)
    edges = {"n": ac.parse_axis_edges("auto", pixels["n"]), "y": [], "x": []}
    out = ac.run_workflow(pixels, env_edges=edges)
    env = out["environments"]["table"].set_index(["species", "environment"])
    assert env.loc[("lowonly", "low n_H"), "verdict"] == "CRIR probe"
    assert env.loc[("lowonly", "high n_H"), "verdict"] == "no response"
    assert env.loc[("steady", "high n_H"), "verdict"] == "CRIR probe"
    # Pooled, lowonly's response depends on density: the n_H swing names the cause.
    rt = out["response"]["table"].set_index("species")
    assert rt.loc["lowonly", "n_swing_dex"] > 0.5 > rt.loc["steady", "n_swing_dex"]
    by_n = out["by_density"].set_index(["species", "i_n"])
    assert by_n.loc[("lowonly", 3), "verdict"] == "no response"


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

