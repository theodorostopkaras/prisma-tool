"""Dominant-mode reaction picker + cosmic-ray path filter (chem_network)."""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import chem_network as cn

R_CR = 'H2 + CRP → H2+ + e-'
R_H3 = 'H2+ + H2 → H3+ + H'
R_HCO = 'H3+ + CO → HCO+ + H2'
R_ALT = 'C+ + H2O → HCO+ + H'
R_DEST = 'HCO+ + e- → CO + H'

LABELS = {
    'H2': {'formation': [], 'destruction': [R_CR, R_H3]},
    'H2+': {'formation': [R_CR], 'destruction': [R_H3]},
    'H3+': {'formation': [R_H3], 'destruction': [R_HCO]},
    'HCO+': {'formation': [R_HCO, R_ALT], 'destruction': [R_DEST]},
    'CO': {'formation': [R_DEST], 'destruction': [R_HCO]},
    'C+': {'formation': [], 'destruction': [R_ALT]},
    'H2O': {'formation': [], 'destruction': [R_ALT]},
    'H': {'formation': [R_H3, R_ALT, R_DEST], 'destruction': []},
}


def _all(sp, mode):
    return LABELS.get(sp, {}).get(mode, [])


def _raws(edges):
    return {e['raw'] for e in edges}


def test_cr_partner_class():
    # CR ionisation also yields e-; it must still be tagged 'cr'.
    assert cn.partner_class(*cn.parse_reaction(R_CR)) == 'cr'
    assert cn.partner_class(*cn.parse_reaction('CO + CRPHOT → C + O')) == 'cr'
    assert cn.partner_class(*cn.parse_reaction('CO + CRPHOTON → C + O')) == 'cr'
    assert cn.partner_class(*cn.parse_reaction(R_DEST)) == 'electron'


def test_picker_keeps_only_picked_reactions():
    def top1(sp, mode):
        return _all(sp, mode)[:1]
    _nodes, edges, _rank = cn.build_multihop_pathway(
        LABELS, 'HCO+', upstream_depth=3, downstream_depth=1, reaction_picker=top1)
    picked = {r for sp in LABELS for m in ('formation', 'destruction') for r in top1(sp, m)}
    assert _raws(edges) <= picked
    assert R_ALT not in _raws(edges)


def test_cr_path_edges_upstream_route():
    _nodes, edges, _rank = cn.build_multihop_pathway(
        LABELS, 'HCO+', upstream_depth=3, downstream_depth=1, reaction_picker=_all)
    assert {R_ALT, R_DEST} <= _raws(edges)
    assert _raws(cn.cr_path_edges(edges, 'HCO+')) == {R_CR, R_H3, R_HCO}


def test_cr_path_edges_downstream_and_none():
    down = [dict(source='HCO+', target='CO+', partner='cr', raw='x'),
            dict(source='HCO+', target='CO', partner='electron', raw='y')]
    assert _raws(cn.cr_path_edges(down, 'HCO+')) == {'x'}
    no_cr = [dict(source='C+', target='HCO+', partner='cp', raw='z')]
    assert cn.cr_path_edges(no_cr, 'HCO+') == []


def test_cr_ion_reactant_counts_as_cr_link():
    # H3+ as co-reactant (node is CO) still ties HCO+ to cosmic rays.
    co = dict(source='CO', target='HCO+', partner='other', raw='CO + H3+ → HCO+ + H2')
    he = dict(source='HE+', target='C+', partner='other', raw='HE+ + CO → C+ + O + HE')
    cp = dict(source='C+', target='HCO+', partner='cp', raw='C+ + H2O → HCO+ + H')
    assert _raws(cn.cr_path_edges([co], 'HCO+')) == {co['raw']}
    assert _raws(cn.cr_path_edges([he, cp], 'HCO+')) == {he['raw'], cp['raw']}


def test_figure_status_cr_only():
    _fig, st = cn.build_network_figure(
        LABELS, 'HCO+', upstream_depth=3, downstream_depth=1,
        reaction_picker=_all, cr_only=True)
    assert st['ok'] and st['cr_only'] and st['n_cr_edges'] == 1
    assert st['cr_ions'] == ['H2+', 'H3+']
    cr_rx = {R_CR, R_H3, R_HCO}  # direct CR step + H2+/H3+-driven steps
    no_cr = {k: {m: [r for r in v[m] if r not in cr_rx] for m in v} for k, v in LABELS.items()}
    _fig, st = cn.build_network_figure(
        no_cr, 'HCO+', upstream_depth=3, downstream_depth=1,
        reaction_picker=lambda sp, m: no_cr.get(sp, {}).get(m, []), cr_only=True)
    assert not st['ok']
    assert 'No cosmic-ray-driven route' in cn.status_message(st)


def test_seg_hits_box_exact():
    assert cn._seg_hits_box(-1, 0, 1, 0, 0, 0, (0.2, 0.2), pad=0.0)
    assert not cn._seg_hits_box(-1, 0.5, 1, 0.5, 0, 0, (0.2, 0.2), pad=0.0)
    # Enters the box only in the last 0.5 % of the segment: ignored end zone.
    assert not cn._seg_hits_box(-9.9, 0, 0.1, 0, 0.2, 0, (0.15, 0.15), pad=0.0)


def test_grouped_traces_width_and_dotted_ion_steps():
    weights = {R_HCO: 80.0, R_ALT: 5.0}

    def weighted(sp, mode):
        return [(r, weights.get(r, 50.0)) for r in _all(sp, mode)]

    fig, _st = cn.build_network_figure(
        LABELS, 'HCO+', upstream_depth=3, downstream_depth=1, reaction_picker=weighted)
    lines = [t for t in fig.data if t.mode == 'lines']

    def trace_of(raw):
        return next(t for t in lines if any(h and h.startswith(raw) for h in t.hovertext))

    assert trace_of(R_HCO).line.width > trace_of(R_ALT).line.width
    assert trace_of(R_HCO).line.dash == 'dot'     # H3+ + CO: CR-ion driven
    assert trace_of(R_CR).line.dash == 'solid'    # direct CR stays solid orange
    assert len(lines) == len({(t.line.color, t.line.dash, t.line.width) for t in lines})


def test_graphviz_layout_draws_arrowheads_and_labels():
    import pytest
    if not cn.shutil.which('dot'):
        pytest.skip('Graphviz dot not installed')
    fig, st = cn.build_network_figure(
        LABELS, 'HCO+', upstream_depth=3, downstream_depth=1, reaction_picker=_all)
    assert st['layout'] == 'dot'
    heads = [t for t in fig.data if t.fill == 'toself']
    assert heads and all(t.mode == 'none' for t in heads)
    texts = {a.text for a in fig.layout.annotations}
    assert 'CR' in texts                                 # partner label on the arrow
    assert len(fig.layout.shapes) == st['n_nodes']       # one rounded box per species
    # Highlighting keeps every box where it was.
    fig_hl, _ = cn.build_network_figure(
        LABELS, 'HCO+', upstream_depth=3, downstream_depth=1, reaction_picker=_all,
        highlight_species='H3+')
    assert [s.path for s in fig_hl.layout.shapes] == [s.path for s in fig.layout.shapes]


def test_cr_path_edges_ignore_cycles_through_focal():
    # CR → F, and F ⇄ B: B feeds F, F is destroyed back into B (HCO+ ⇄ CO).
    cr = dict(source='A', target='F', partner='cr', raw='cr')
    fb = dict(source='F', target='B', partner='electron', raw='back')
    bf = dict(source='B', target='F', partner='other', raw='plain')
    rank = {'A': -1, 'B': -1, 'F': 0}
    assert 'back' in _raws(cn.cr_path_edges([cr, fb, bf], 'F'))       # cycle leaks in
    assert _raws(cn.cr_path_edges([cr, fb, bf], 'F', rank)) == {'cr'}


@pytest.mark.skipif(shutil.which('dot') is None, reason='graphviz not installed')
def test_dot_layout_stretches_to_page_aspect():
    chain = [f'S{i}' for i in range(8)]
    edges = [dict(source=a, target=b) for a, b in zip(chain, chain[1:])]
    rank = {n: i for i, n in enumerate(chain)}
    box = {n: (60.0, 22.0) for n in chain}
    x0, y0, x1, y1 = cn._dot_layout(chain, edges, rank, box, 10, aspect=0.6)['bb']
    assert (y1 - y0) / (x1 - x0) == pytest.approx(0.6, rel=0.05)


def test_network_depth_and_node_cap():
    # Deeper walks only add species; the result is deterministic; the cap trims.
    shallow, e_shallow, _ = cn.build_multihop_pathway(
        LABELS, 'HCO+', upstream_depth=1, downstream_depth=1)
    deep, e_deep, _ = cn.build_multihop_pathway(
        LABELS, 'HCO+', upstream_depth=3, downstream_depth=1)
    assert {R_HCO, R_ALT, R_DEST} <= _raws(e_shallow)
    assert shallow <= deep  # H3+ is a seed ion: the heuristic walk stops there
    again, _, _ = cn.build_multihop_pathway(
        LABELS, 'HCO+', upstream_depth=3, downstream_depth=1)
    assert again == deep
    capped, _, _ = cn.build_multihop_pathway(
        LABELS, 'HCO+', upstream_depth=3, downstream_depth=1, max_nodes=4)
    assert len(capped) <= len(deep)


def test_cr_path_keeps_same_column_crphot_entry():
    # X + CRPHOT → X+ + e- puts X and X+ in one column; it must still start the route.
    crphot = dict(source='H2O', target='H2O+', raw='H2O + CRPHOT → H2O+ + e-', partner='cr')
    step = dict(source='H2O+', target='HCO+', raw='H2O+ + CO → HCO+ + OH', partner='other')
    loop = dict(source='H2O+', target='H2O', raw='H2O+ + e- → H2O', partner='electron')
    rank = {'H2O': -1, 'H2O+': -1, 'HCO+': 0}
    kept = cn.cr_path_edges([crphot, step, loop], 'HCO+', rank)
    assert crphot in kept and step in kept
    assert loop not in kept  # non-CR same-column steps still close cycles: dropped
