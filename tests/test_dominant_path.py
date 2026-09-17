"""Dominant-mode reaction picker + cosmic-ray path filter (chem_network)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
