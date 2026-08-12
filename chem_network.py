"""
Chemistry reaction-network diagrams from KOSMA-τ chemistry HDF5 labels.

Top-N formation/destruction panels use a pathway style: the first-listed
species is a node, other partners sit as labels on the connecting arrow, and
the focal species is centred (formation left→focus, destruction focus→right).

A separate depth-limited chemical-chain flowchart is also provided.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import plotly.graph_objects as go


# Partner class → (legend label, line color, dash style)
PARTNER_STYLES = {
    'electron': ('+ e\u207b', '#d62728', 'solid'),
    'photon': ('+ h\u03bd / CR', '#222222', 'solid'),
    'h2': ('+ H\u2082', '#1f77b4', 'solid'),
    'c': ('+ C', '#e377c2', 'solid'),
    'cp': ('+ C\u207a', '#2ca02c', 'solid'),
    'other': ('other', '#7f7f7f', 'dash'),
}

# Typical PDR “entry” species for chemical-chain rooting (adapted to KOSMA names).
CHAIN_SEED_SPECIES = frozenset({
    'H', 'H2', 'H2*', 'H3+', 'H+', 'H2+',
    'C', 'C+', 'O', 'O+', 'N', 'N+', 'HE', 'HE+',
    'S', 'S+', 'SI', 'SI+',
})

DEFAULT_CHAIN_UPSTREAM = 3
DEFAULT_CHAIN_DOWNSTREAM = 2
CHAIN_MAX_NODES = 36
# Extra node budget when isotopes / ice toggles add partners on top of the base chain.
# Page-safe Plotly canvas: never grow past this or the Dash tab scrolls off-screen.
PAGE_FIG_HEIGHT = 1020
PAGE_FIG_HEIGHT_MIN = 720
# Approx. plot area used when converting marker px → data-unit box half.
PAGE_FIG_WIDTH_EST = 1600
CHAIN_EXTRA_NODES = 16
# Max formation / destruction channels kept per species while walking the chain.
CHAIN_BRANCH_LIMIT = 5
# Degree thresholds: change node shape when many links touch a species.
HUB_DEGREE_CIRCLE = 6
HUB_DEGREE_HEX = 10

# Tokens that set edge colour / process but are not chemistry-network nodes.
_PROCESS_TOKENS = {
    'PHOTON', 'CRPHOT', 'CRP', 'CR', 'ELECTRON', 'E', 'E-', 'HV', 'HNU',
    'H\u03bd', 'HΝ',
}

_ARROW_SEPS = (' \u2192 ', ' → ', ' > ', '->', '\u2192')


def normalize_token(raw: str) -> str:
    """Canonical species / process token from a reaction fragment."""
    tok = str(raw or '').strip()
    if not tok:
        return ''
    upper = tok.upper().replace(' ', '')
    if upper in ('E-', 'E', 'ELECTRON'):
        return 'e-'
    if upper in ('PHOTON', 'HNU', 'HV', 'H\u03bd'):
        return 'PHOTON'
    if upper in ('CRPHOT', 'CRPHOTON'):
        return 'CRPHOT'
    if upper == 'CRP':
        return 'CRP'
    if upper == 'CR':
        return 'CR'
    if upper in ('H2*', 'H2STAR'):
        return 'H2*'
    return tok


def _species_match_key(name: str) -> str:
    """Loose key so click labels (HCO⁺ / SiH+) match graph ids (HCO+ / SIH+)."""
    s = str(name or '').strip()
    if not s:
        return ''
    table = str.maketrans({
        '\u207a': '+', '\u207b': '-',  # ⁺ ⁻
        '\u208a': '+', '\u208b': '-',  # ₊ ₋
        '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3',
        '\u2084': '4', '\u2085': '5', '\u2086': '6', '\u2087': '7',
        '\u2088': '8', '\u2089': '9',
        '\u00b0': '0', '\u00b9': '1', '\u00b2': '2', '\u00b3': '3',
    })
    return s.translate(table).upper()


def is_process_token(token: str) -> bool:
    t = normalize_token(token)
    if not t:
        return True
    if t in ('e-', 'PHOTON', 'CRPHOT', 'CRP', 'CR'):
        return True
    return t.upper() in _PROCESS_TOKENS


def is_species_node(token: str) -> bool:
    """True for chemical species that should appear as graph nodes."""
    t = normalize_token(token)
    return bool(t) and not is_process_token(t)


def _split_species_list(side: str) -> List[str]:
    """Split a reaction side on `` + `` without breaking charge suffixes."""
    text = str(side or '').strip()
    if not text:
        return []
    parts = text.split(' + ') if ' + ' in text else [text]
    return [normalize_token(p) for p in parts if normalize_token(p)]


def parse_reaction(label: str) -> Tuple[List[str], List[str]]:
    """Split ``A + B → C + D`` into reactant / product token lists."""
    text = str(label or '').strip()
    if not text:
        return [], []
    lhs = rhs = None
    for sep in _ARROW_SEPS:
        if sep in text:
            lhs, rhs = text.split(sep, 1)
            break
    if lhs is None:
        return [], []
    return _split_species_list(lhs), _split_species_list(rhs)


def partner_class(reactants: Sequence[str], products: Sequence[str] = ()) -> str:
    """Classify the reaction partner for edge colouring."""
    toks = [normalize_token(t) for t in list(reactants) + list(products)]
    if any(t == 'e-' for t in toks):
        return 'electron'
    if any(t in ('PHOTON', 'CRPHOT', 'CRP', 'CR') for t in toks):
        return 'photon'
    rset = {normalize_token(t) for t in reactants}
    if 'H2' in rset or 'H2*' in rset:
        return 'h2'
    if 'C+' in rset:
        return 'cp'
    if 'C' in rset:
        return 'c'
    return 'other'


def other_partner_label(reactants: Sequence[str], focal: Optional[str] = None) -> str:
    """Short edge label for 'other' partners (e.g. CH, CR)."""
    skip = {normalize_token(focal)} if focal else set()
    for t in reactants:
        nt = normalize_token(t)
        if not nt or nt in skip or is_process_token(nt):
            continue
        if nt in ('H2', 'H2*', 'C', 'C+', 'e-'):
            continue
        return nt
    for t in reactants:
        nt = normalize_token(t)
        if nt in ('PHOTON', 'CRPHOT', 'CRP', 'CR'):
            return 'CR' if nt.startswith('CR') else 'h\u03bd'
    return ''


def reaction_edges(label: str, owner_species: Optional[str] = None):
    """Yield ``(src, dst, partner_class, edge_label, raw_label)`` for one reaction."""
    reactants, products = parse_reaction(label)
    if not reactants or not products:
        return
    pclass = partner_class(reactants, products)
    elabel = other_partner_label(reactants, owner_species) if pclass == 'other' else ''
    srcs = [t for t in reactants if is_species_node(t)]
    dsts = [t for t in products if is_species_node(t)]
    for src in srcs:
        for dst in dsts:
            if src == dst:
                continue
            yield src, dst, pclass, elabel, str(label)


def _format_node_label(name: str) -> str:
    """Plotly annotation label with HTML sub/superscripts."""
    out = []
    i = 0
    n = len(name)
    if name and name[0].isdigit():
        j = 0
        while j < n and name[j].isdigit():
            j += 1
        out.append(f'<sup>{name[:j]}</sup>')
        i = j
    while i < n:
        c = name[i]
        if c == '+':
            out.append('<sup>+</sup>')
        elif c == '-':
            out.append('<sup>\u2212</sup>')
        elif c.isdigit():
            out.append(f'<sub>{c}</sub>')
        else:
            out.append(c)
        i += 1
    return ''.join(out)


def _label_char_count(name: str) -> int:
    """Visible character count used to size species boxes."""
    return max(1, len(str(name or '').strip()))


def _label_font_size(base: int, n_chars: int) -> int:
    """Slightly smaller type for long formulas so they stay inside the box."""
    if n_chars >= 8:
        return max(9, base - 4)
    if n_chars >= 6:
        return max(10, base - 3)
    if n_chars >= 5:
        return max(10, base - 2)
    return base


def _format_mid_token(token: str) -> str:
    """Readable mid-arrow partner label (e⁻, photon, species, …)."""
    nt = normalize_token(token)
    if not nt:
        return ''
    if nt == 'e-':
        return 'e\u207b'
    if nt == 'PHOTON':
        return 'photon'
    if nt in ('CRPHOT', 'CRP', 'CR'):
        return 'CR' if nt != 'CRPHOT' else 'CRPHOT'
    return _format_node_label(nt)


def _mid_label_from_tokens(tokens: Sequence[str]) -> str:
    parts = [_format_mid_token(t) for t in tokens]
    parts = [p for p in parts if p]
    return ' + '.join(parts)


def _primary_reactant(reactants: Sequence[str]) -> Tuple[Optional[str], List[str]]:
    """First-listed species reactant as the path node; remaining reactants → mid-label."""
    if not reactants:
        return None, []
    # Prefer the first chemical species; if the line starts with a process
    # token, keep that species as the node and put the process on the arrow.
    primary = None
    primary_idx = None
    for i, tok in enumerate(reactants):
        if is_species_node(tok):
            primary = normalize_token(tok)
            primary_idx = i
            break
    if primary is None:
        return None, list(reactants)
    mid = [reactants[i] for i in range(len(reactants)) if i != primary_idx]
    return primary, mid


def _primary_product(products: Sequence[str], focal: str) -> Optional[str]:
    """First-listed product species (skipping the focal species when present)."""
    for tok in products:
        nt = normalize_token(tok)
        if not is_species_node(nt):
            continue
        if nt == focal:
            continue
        return nt
    return None


def pathway_edge_from_reaction(
    label: str,
    focal_species: str,
    mode: str,
) -> Optional[dict]:
    """One pathway edge: primary species node + mid-arrow partner label.

    Formation: first reactant species → focal; other reactants on the arrow.
    Destruction: focal → first product species; co-reactants on the arrow.
    """
    focal = str(focal_species or '')
    reactants, products = parse_reaction(label)
    if not reactants or not products or not focal:
        return None
    mode = str(mode or '')
    pclass = partner_class(reactants, products)

    if mode == 'formation':
        primary, mid_toks = _primary_reactant(reactants)
        if not primary or primary == focal:
            return None
        if focal not in products and normalize_token(focal) not in {
                normalize_token(p) for p in products}:
            # Still allow if labels are noisy; require focal among products when possible.
            pass
        return dict(
            source=primary, target=focal, partner=pclass,
            mid_label=_mid_label_from_tokens(mid_toks),
            raw=str(label), mode=mode,
        )

    if mode == 'destruction':
        primary = _primary_product(products, focal)
        if not primary:
            return None
        mid_toks = [t for t in reactants if normalize_token(t) != focal]
        return dict(
            source=focal, target=primary, partner=pclass,
            mid_label=_mid_label_from_tokens(mid_toks),
            raw=str(label), mode=mode,
        )

    return None


def build_pathway_graph(
    focal_species: str,
    selected_reactions: Sequence[Tuple[str, str]],
) -> Tuple[Set[str], List[dict]]:
    """Build pathway edges from top-N formation/destruction reaction labels."""
    focal = str(focal_species or '')
    if not focal:
        return set(), []

    nodes: Set[str] = {focal}
    edges: List[dict] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    for item in selected_reactions or []:
        if not item:
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            raw, mode = item[0], item[1]
        else:
            raw, mode = item, 'reaction'
        raw = str(raw or '').strip()
        if not raw:
            continue
        edge = pathway_edge_from_reaction(raw, focal, str(mode))
        if not edge:
            continue
        key = (edge['source'], edge['target'], edge.get('mid_label', ''), edge['raw'])
        if key in seen:
            continue
        seen.add(key)
        edges.append(edge)
        nodes.add(edge['source'])
        nodes.add(edge['target'])

    return nodes, edges


# Backwards-compatible alias used by older call sites / chemical-chain helpers.
def build_selected_reaction_graph(
    focal_species: str,
    selected_reactions: Sequence[Tuple[str, str]],
) -> Tuple[Set[str], List[dict]]:
    return build_pathway_graph(focal_species, selected_reactions)


def layout_pathway_positions(
    focal_species: str,
    edges: Sequence[dict],
    *,
    mode_filter: Optional[str] = None,
) -> Dict[str, Tuple[float, float]]:
    """Left (formation) / centre (focus) / right (destruction) pathway layout."""
    focal = str(focal_species or '')
    if not focal:
        return {}

    left: List[str] = []
    right: List[str] = []
    seen_l: Set[str] = set()
    seen_r: Set[str] = set()
    for e in edges:
        src, dst = e['source'], e['target']
        if src != focal and src not in seen_l:
            # Formation-style edge into the focus.
            if dst == focal:
                left.append(src)
                seen_l.add(src)
        if dst != focal and dst not in seen_r:
            if src == focal:
                right.append(dst)
                seen_r.add(dst)

    if mode_filter == 'formation':
        right = []
    elif mode_filter == 'destruction':
        left = []

    def _spread_y(names: Sequence[str], x: float) -> Dict[str, Tuple[float, float]]:
        names = list(names)
        if not names:
            return {}
        if len(names) == 1:
            return {names[0]: (float(x), 0.0)}
        span = max(1.6, 0.95 * (len(names) - 1))
        ys = np.linspace(span, -span, len(names))
        return {n: (float(x), float(y)) for n, y in zip(names, ys)}

    pos: Dict[str, Tuple[float, float]] = {focal: (0.0, 0.0)}
    pos.update(_spread_y(left, -2.6))
    pos.update(_spread_y(right, 2.6))
    return pos


def _pad_range(vals: Sequence[float], pad: float = 0.18,
               abs_pad: float = 0.0) -> List[float]:
    """Axis range with fractional and optional absolute padding."""
    if not vals:
        return [-1.0, 1.0]
    lo, hi = float(min(vals)), float(max(vals))
    if lo == hi:
        return [lo - 1.0 - abs_pad, hi + 1.0 + abs_pad]
    span = hi - lo
    return [lo - pad * span - abs_pad, hi + pad * span + abs_pad]


def _orthogonal_polyline(
    x0: float, y0: float, x1: float, y1: float, *, bend: float = 0.0,
) -> List[Tuple[float, float]]:
    """Manhattan route from ``(x0,y0)`` to ``(x1,y1)`` (horizontal then vertical)."""
    if abs(x0 - x1) < 1e-9 or abs(y0 - y1) < 1e-9:
        return [(x0, y0), (x1, y1)]
    xm = 0.5 * (x0 + x1) + bend
    return [(x0, y0), (xm, y0), (xm, y1), (x1, y1)]


def _port_offsets(n: int, span: float = 0.28) -> List[float]:
    """Spread attachment points along a box edge so links do not stack."""
    if n <= 1:
        return [0.0]
    return [float(v) for v in np.linspace(-span, span, n)]


def _face_port_span(box_half: float, count: int) -> float:
    """How far along a face to spread ports for ``count`` attachments.

    Uses most of the face so many exits/entries sit clearly side-by-side
    (right next to each other) instead of stacking on the centre.
    """
    if count <= 1:
        return 0.0
    return max(0.0, float(box_half) * 0.84)


def _exit_face(x0: float, y0: float, x1: float, y1: float) -> str:
    """Which side of the source box an edge leaves on."""
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) >= abs(dy):
        return 'right' if dx >= 0 else 'left'
    return 'top' if dy >= 0 else 'bottom'


def _enter_face(x0: float, y0: float, x1: float, y1: float) -> str:
    """Which side of the target box an edge arrives on."""
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) >= abs(dy):
        return 'left' if dx >= 0 else 'right'
    return 'bottom' if dy >= 0 else 'top'


def _assign_face_ports(
    draw_edges: Sequence[dict],
    pos: Dict[str, Tuple[float, float]],
    node_half: Dict[str, float],
) -> Dict[Tuple[str, str, str, str], float]:
    """Map ``(source, target, partner, 'out'|'in')`` → port offset on that face.

    Ports are grouped per node face so several links leaving the same side of a
    box are spaced evenly along that side (easy to follow individually).
    """
    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for e in draw_edges:
        s, t = e.get('source', ''), e.get('target', '')
        if s not in pos or t not in pos:
            continue
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        groups[(s, _exit_face(x0, y0, x1, y1), 'out')].append(e)
        groups[(t, _enter_face(x0, y0, x1, y1), 'in')].append(e)

    port_of: Dict[Tuple[str, str, str, str], float] = {}
    for (node, face, which), elist in groups.items():
        def _sort_key(e: dict, _face: str = face, _which: str = which) -> tuple:
            other = e['target'] if _which == 'out' else e['source']
            ox, oy = pos[other]
            # Order along the face: vertical faces by y, horizontal faces by x.
            if _face in ('left', 'right'):
                return (oy, str(e.get('partner', '')), other)
            return (ox, str(e.get('partner', '')), other)

        ordered = sorted(elist, key=_sort_key)
        half = float(node_half.get(node, 0.18))
        offs = _port_offsets(len(ordered), _face_port_span(half, len(ordered)))
        for e, off in zip(ordered, offs):
            key = (e['source'], e['target'], str(e.get('partner', '')), which)
            port_of[key] = off
    return port_of


def _lane_offset(index: int, n_lanes: int, col_gap: float) -> float:
    """Mid-corridor offset for one of ``n_lanes`` parallel orthogonal routes.

    Targets a readable gap between neighbours (~0.13–0.17 data units) and
    widens the bundle when many edges share the same column pair.
    """
    if n_lanes <= 1:
        return 0.0
    # Preferred neighbour gap; allow a bit more room in wider columns.
    gap = min(0.17, max(0.125, 0.095 * float(col_gap)))
    spread = 0.5 * gap * (n_lanes - 1)
    # Keep the bundle inside the inter-column corridor when possible.
    max_spread = max(0.42, 0.48 * float(col_gap))
    if spread > max_spread:
        spread = max_spread
    return float(np.linspace(-spread, spread, n_lanes)[min(index, n_lanes - 1)])


def _curve_bend(index: int, n: int, col_gap: float, *, strong: bool = False) -> float:
    """Perpendicular arc offset (paper-style) for one of ``n`` parallel curves."""
    if n <= 1:
        return 0.28 if strong else 0.18
    gap = 0.32 if strong else 0.24
    spread = 0.5 * gap * (n - 1)
    max_spread = max(0.70, 0.62 * float(col_gap))
    if spread > max_spread:
        spread = max_spread
    return float(np.linspace(-spread, spread, n)[min(index, n - 1)])


def _should_use_curve(
    n_pair: int, n_lanes: int, n_edges: int, *, n_undirected: int = 1,
) -> bool:
    """Prefer orthogonal routes; switch to arcs when bundles would stack.

    Same-direction multi-edges (several partners A→B) always get paper-style
    arcs. Bidirectional and busy-corridor curves only kick in once the graph
    is dense enough that Manhattan lanes would pile up.
    """
    if n_pair >= 2:
        return True
    if n_edges < 40:
        return False
    if n_undirected >= 2:
        return True
    if n_lanes >= 6:
        return True
    if n_edges >= 80 and n_lanes >= 4:
        return True
    return False


def _facing_attach(
    x0: float, y0: float, x1: float, y1: float,
    *,
    box_half_src: float,
    box_half_dst: float,
    port_src: float,
    port_dst: float,
) -> Tuple[float, float, float, float]:
    """Exit/enter points on facing sides of source and target boxes."""
    dx = x1 - x0
    dy = y1 - y0
    hs = max(0.08, float(box_half_src))
    hd = max(0.08, float(box_half_dst))
    if abs(dx) >= abs(dy):
        if dx >= 0:
            return x0 + hs, y0 + port_src, x1 - hd, y1 + port_dst
        return x0 - hs, y0 + port_src, x1 + hd, y1 + port_dst
    if dy >= 0:
        return x0 + port_src, y0 + hs, x1 + port_dst, y1 - hd
    return x0 + port_src, y0 - hs, x1 + port_dst, y1 + hd


def _sample_quadratic(
    sx: float, sy: float, cx: float, cy: float, tx: float, ty: float, n: int = 28,
) -> List[Tuple[float, float]]:
    """Sample a quadratic Bezier from ``(sx,sy)`` via control ``(cx,cy)`` to ``(tx,ty)``."""
    ts = np.linspace(0.0, 1.0, max(8, int(n)))
    pts: List[Tuple[float, float]] = []
    for t in ts:
        u = 1.0 - t
        x = u * u * sx + 2.0 * u * t * cx + t * t * tx
        y = u * u * sy + 2.0 * u * t * cy + t * t * ty
        pts.append((float(x), float(y)))
    return pts


def _curve_route(
    x0: float, y0: float, x1: float, y1: float,
    *,
    box_half_src: float,
    box_half_dst: float,
    bend: float,
    port_src: float,
    port_dst: float,
    n_samples: int = 28,
) -> List[Tuple[float, float]]:
    """Smooth arc between facing box sides (A&A-style multi-edge curves).

    Control point sits at the chord midpoint, offset perpendicular by ``bend``
    so parallel reactions between the same pair fan apart instead of stacking.
    """
    sx, sy, tx, ty = _facing_attach(
        x0, y0, x1, y1,
        box_half_src=box_half_src, box_half_dst=box_half_dst,
        port_src=port_src, port_dst=port_dst,
    )
    dx, dy = tx - sx, ty - sy
    length = float(np.hypot(dx, dy))
    if length < 1e-9:
        return [(sx, sy), (tx, ty)]
    # Unit perpendicular (left of travel direction).
    px, py = -dy / length, dx / length
    mx, my = 0.5 * (sx + tx), 0.5 * (sy + ty)
    cx, cy = mx + px * float(bend), my + py * float(bend)
    return _sample_quadratic(sx, sy, cx, cy, tx, ty, n=n_samples)


def _dedupe_pathway_edges(edges: Sequence[dict]) -> List[dict]:
    """Keep one edge per (source, target, partner) to cut parallel clutter."""
    best: Dict[Tuple[str, str, str], dict] = {}
    for e in edges or []:
        key = (e.get('source', ''), e.get('target', ''), str(e.get('partner', '')))
        if key in best:
            # Prefer the shorter / cleaner mid-label.
            old = best[key].get('mid_label') or ''
            new = e.get('mid_label') or ''
            if len(new) < len(old):
                best[key] = e
            continue
        best[key] = e
    return list(best.values())


def _edge_route(
    x0: float, y0: float, x1: float, y1: float,
    *,
    box_half_src: float,
    box_half_dst: float,
    lane: float,
    port_src: float,
    port_dst: float,
    vertical_first: bool = False,
) -> List[Tuple[float, float]]:
    """Orthogonal route that starts/ends on the facing sides of both boxes.

    ``box_half_*`` are half-sizes of the source/target markers in data units so
    connectors meet each border.  ``port_*`` offsets slide the attachment
    along that border.  ``vertical_first`` flips the Manhattan bend order.
    """
    sx, sy, tx, ty = _facing_attach(
        x0, y0, x1, y1,
        box_half_src=box_half_src, box_half_dst=box_half_dst,
        port_src=port_src, port_dst=port_dst,
    )
    dx = x1 - x0
    dy = y1 - y0
    prefer_h = abs(dx) >= abs(dy)
    if vertical_first:
        prefer_h = not prefer_h

    if prefer_h:
        if abs(sy - ty) < 1e-9 and abs(lane) < 1e-9:
            return [(sx, sy), (tx, ty)]
        xm = 0.5 * (sx + tx) + lane
        lo, hi = (sx, tx) if sx <= tx else (tx, sx)
        corridor = hi - lo
        if corridor > 0.25:
            margin = min(0.06, 0.12 * corridor)
            xm = min(max(xm, lo + margin), hi - margin)
        else:
            xm = 0.5 * (sx + tx) + 0.55 * lane
        return [(sx, sy), (xm, sy), (xm, ty), (tx, ty)]

    if abs(sx - tx) < 1e-9 and abs(lane) < 1e-9:
        return [(sx, sy), (tx, ty)]
    ym = 0.5 * (sy + ty) + lane
    lo, hi = (sy, ty) if sy <= ty else (ty, sy)
    corridor = hi - lo
    if corridor > 0.25:
        margin = min(0.06, 0.12 * corridor)
        ym = min(max(ym, lo + margin), hi - margin)
    else:
        ym = 0.5 * (sy + ty) + 0.55 * lane
    return [(sx, sy), (sx, ym), (tx, ym), (tx, ty)]


def _seg_hits_box(
    x0: float, y0: float, x1: float, y1: float,
    cx: float, cy: float, half: float, *, pad: float = 0.08,
) -> bool:
    """True if the segment passes through an expanded node box."""
    r = max(0.10, float(half) + float(pad))
    xmin, xmax = cx - r, cx + r
    ymin, ymax = cy - r, cy + r
    # Quick reject on bounding boxes.
    if max(x0, x1) < xmin or min(x0, x1) > xmax:
        return False
    if max(y0, y1) < ymin or min(y0, y1) > ymax:
        return False
    length = float(np.hypot(x1 - x0, y1 - y0))
    n = max(3, int(length / 0.06) + 1)
    for t in np.linspace(0.0, 1.0, n):
        # Skip the very ends — attachments sit on src/dst faces.
        if t < 0.02 or t > 0.98:
            continue
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
    return False


def _polyline_hits_obstacles(
    pts: Sequence[Tuple[float, float]],
    obstacles: Dict[str, Tuple[float, float, float]],
    skip: Set[str],
    *,
    pad: float = 0.08,
) -> Optional[str]:
    """Return the name of the first obstacle a polyline crosses, else None."""
    if len(pts) < 2 or not obstacles:
        return None
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        for name, (cx, cy, half) in obstacles.items():
            if name in skip:
                continue
            if _seg_hits_box(x0, y0, x1, y1, cx, cy, half, pad=pad):
                return name
    return None


def _detour_route(
    x0: float, y0: float, x1: float, y1: float,
    *,
    box_half_src: float,
    box_half_dst: float,
    port_src: float,
    port_dst: float,
    obstacles: Dict[str, Tuple[float, float, float]],
    skip: Set[str],
    side: str = 'top',
) -> List[Tuple[float, float]]:
    """Manhattan detour above or below all obstacles between the endpoints."""
    sx, sy, tx, ty = _facing_attach(
        x0, y0, x1, y1,
        box_half_src=box_half_src, box_half_dst=box_half_dst,
        port_src=port_src, port_dst=port_dst,
    )
    x_lo, x_hi = (sx, tx) if sx <= tx else (tx, sx)
    blocked: List[Tuple[float, float, float]] = []
    for name, (cx, cy, half) in obstacles.items():
        if name in skip:
            continue
        if x_lo - half <= cx <= x_hi + half:
            blocked.append((cx, cy, half))
    clear = 0.16
    if blocked:
        if side == 'top':
            y_clear = max(cy + half for _, cy, half in blocked) + clear
            y_clear = max(y_clear, sy + clear, ty + clear)
        else:
            y_clear = min(cy - half for _, cy, half in blocked) - clear
            y_clear = min(y_clear, sy - clear, ty - clear)
    else:
        y_clear = 0.5 * (sy + ty) + (0.35 if side == 'top' else -0.35)
    return [(sx, sy), (sx, y_clear), (tx, y_clear), (tx, ty)]


def _route_avoiding_nodes(
    x0: float, y0: float, x1: float, y1: float,
    *,
    box_half_src: float,
    box_half_dst: float,
    port_src: float,
    port_dst: float,
    lane: float,
    bend: float,
    prefer_curve: bool,
    obstacles: Dict[str, Tuple[float, float, float]],
    skip: Set[str],
    col_gap: float,
    n_samples: int = 24,
) -> Tuple[List[Tuple[float, float]], str]:
    """Pick a route that does not cross unrelated species boxes when possible.

    Tries the preferred style first, then alternate orthogonal/curve options,
    then an explicit above/below detour. Returns ``(points, style)``.
    """
    kw = dict(
        box_half_src=box_half_src, box_half_dst=box_half_dst,
        port_src=port_src, port_dst=port_dst,
    )
    candidates: List[Tuple[str, List[Tuple[float, float]]]] = []

    if prefer_curve:
        candidates.append(('curve', _curve_route(
            x0, y0, x1, y1, bend=bend, n_samples=n_samples, **kw)))
        for b in (bend, -bend, bend + 0.35, bend - 0.35, 0.55, -0.55, 0.85, -0.85):
            if abs(b - bend) < 1e-9 and candidates:
                continue
            candidates.append(('curve', _curve_route(
                x0, y0, x1, y1, bend=b, n_samples=n_samples, **kw)))
        candidates.append(('ortho', _edge_route(
            x0, y0, x1, y1, lane=lane, **kw)))
        candidates.append(('ortho', _edge_route(
            x0, y0, x1, y1, lane=lane, vertical_first=True, **kw)))
    else:
        candidates.append(('ortho', _edge_route(
            x0, y0, x1, y1, lane=lane, **kw)))
        candidates.append(('ortho', _edge_route(
            x0, y0, x1, y1, lane=lane, vertical_first=True, **kw)))
        for dlane in (0.25, -0.25, 0.45, -0.45, 0.65, -0.65):
            candidates.append(('ortho', _edge_route(
                x0, y0, x1, y1, lane=lane + dlane * max(0.5, col_gap * 0.2), **kw)))
            candidates.append(('ortho', _edge_route(
                x0, y0, x1, y1, lane=lane + dlane * max(0.5, col_gap * 0.2),
                vertical_first=True, **kw)))
        for b in (0.4, -0.4, 0.7, -0.7, 1.0, -1.0):
            candidates.append(('curve', _curve_route(
                x0, y0, x1, y1, bend=b, n_samples=n_samples, **kw)))

    candidates.append(('detour', _detour_route(
        x0, y0, x1, y1, side='top', obstacles=obstacles, skip=skip, **kw)))
    candidates.append(('detour', _detour_route(
        x0, y0, x1, y1, side='bottom', obstacles=obstacles, skip=skip, **kw)))

    best_hit: Optional[Tuple[str, List[Tuple[float, float]]]] = None
    for style, pts in candidates:
        hit = _polyline_hits_obstacles(pts, obstacles, skip)
        if hit is None:
            return pts, style
        if best_hit is None:
            best_hit = (style, pts)
    # Last resort: first candidate (still better than crashing).
    if best_hit is not None:
        return best_hit[1], best_hit[0]
    return candidates[0][1], candidates[0][0]


def _node_visual_style(
    degree: int,
    *,
    is_focal: bool = False,
    base_size: float = 40.0,
    base_half: float = 0.20,
    label_chars: int = 1,
) -> Tuple[str, float, float]:
    """Return ``(plotly_symbol, marker_size_px, box_half_data)``.

    Size grows with link degree (hubs) and with formula length so long names
    such as ``H213CO`` / ``SiOH+`` stay inside their box.
    """
    if is_focal:
        sym, sz, half = 'square', base_size + 10.0, base_half * 1.15
    elif degree >= HUB_DEGREE_HEX:
        sym, sz, half = 'hexagon', base_size + 16.0, base_half * 1.45
    elif degree >= HUB_DEGREE_CIRCLE:
        sym, sz, half = 'circle', base_size + 10.0, base_half * 1.25
    else:
        sym, sz, half = 'square', base_size, base_half

    n = max(1, int(label_chars))
    if n >= 4:
        # Pixel width must cover the formula; ~7–8 px per character.
        sz = max(sz, base_size * 0.55 + 7.5 * n)
        half *= min(1.0 + 0.14 * (n - 3), 2.0)
        # Long labels read better on a square name plate than a tight circle.
        if n >= 6 and degree < HUB_DEGREE_HEX:
            sym = 'square'
    if is_focal and n >= 4:
        sz = max(sz, base_size * 0.6 + 7.0 * n)
    return sym, float(sz), float(half)


def layout_rank_grid(
    nodes: Iterable[str],
    rank: Dict[str, int],
    *,
    focal_species: Optional[str] = None,
) -> Tuple[Dict[str, Tuple[float, float]], dict]:
    """Column layout by pathway rank (upstream left → focus → destruction right).

    Spacing grows with the tallest column and the number of rank columns so
    deep upstream/downstream networks do not stack boxes on top of each other.
    Returns ``(positions, metrics)`` with ``col_gap``, ``row_gap``, etc.
    """
    focal = str(focal_species or '')
    by_rank: Dict[int, List[str]] = defaultdict(list)
    for n in nodes:
        by_rank[int(rank.get(n, 0))].append(n)
    for r, names in list(by_rank.items()):
        if focal in names:
            by_rank[r] = [focal] + sorted(x for x in names if x != focal)
        else:
            by_rank[r] = sorted(names)
    ranks = sorted(by_rank)
    empty_metrics = dict(col_gap=2.8, row_gap=1.6, n_cols=0, max_col=0)
    if not ranks:
        return ({focal: (0.0, 0.0)} if focal else {}), empty_metrics

    n_cols = len(ranks)
    max_col = max(len(by_rank[r]) for r in ranks)
    # Gaps grow with density; soft-caps keep the layout compressible onto one page.
    col_gap = max(2.6, 2.25 + 0.22 * max_col + 0.10 * max(0, n_cols - 3))
    row_gap = max(1.95, 1.55 + 0.22 * max_col)
    max_span_x = 18.5
    if n_cols > 1:
        col_gap = min(col_gap, max_span_x / (n_cols - 1))
        col_gap = max(2.05, col_gap)

    if 0 in by_rank:
        x_of = {r: float(ranks.index(r) - ranks.index(0)) * col_gap for r in ranks}
    else:
        x_of = {r: float(i) * col_gap for i, r in enumerate(ranks)}

    pos: Dict[str, Tuple[float, float]] = {}
    for r, names in by_rank.items():
        x = x_of[r]
        if len(names) == 1:
            pos[names[0]] = (x, 0.0)
            continue
        span = 0.5 * row_gap * (len(names) - 1)
        ys = np.linspace(span, -span, len(names))
        if focal in names and len(names) > 1:
            pos[focal] = (x, 0.0)
            others = [n for n in names if n != focal]
            span_o = 0.5 * row_gap * max(1, len(others) - 1) if len(others) > 1 else row_gap
            if len(others) == 1:
                ys = [row_gap * 0.85]
            else:
                ys = list(np.linspace(span_o, -span_o, len(others)))
            # Keep siblings clear of the focal y=0 band.
            min_clear = 0.70 * row_gap
            ys = [y if abs(y) >= min_clear else (min_clear if y >= 0 else -min_clear)
                  for y in ys]
            for n, y in zip(others, ys):
                pos[n] = (x, float(y))
        else:
            for n, y in zip(names, ys):
                pos[n] = (x, float(y))

    metrics = dict(
        col_gap=float(col_gap),
        row_gap=float(row_gap),
        n_cols=n_cols,
        max_col=max_col,
    )
    return pos, metrics


def build_multihop_pathway(
    labels_by_species: Dict[str, dict],
    focal_species: str,
    *,
    upstream_depth: int = DEFAULT_CHAIN_UPSTREAM,
    downstream_depth: int = DEFAULT_CHAIN_DOWNSTREAM,
    max_nodes: int = CHAIN_MAX_NODES,
    include_isotopes: bool = False,
    include_ice: bool = False,
) -> Tuple[Set[str], List[dict], Dict[str, int]]:
    """Multi-hop pathway graph around ``focal_species`` (form + destroy combined).

    Walks formation channels upstream and destruction channels downstream. Each
    reaction contributes one pathway edge: first-listed species as the node,
    other partners as ``mid_label`` / partner colour — so A + X → B → … → focus
    → … reads as a progressive route rather than a one-hop star.
    """
    focal = str(focal_species or '')
    if not focal or not labels_by_species:
        return set(), [], {}

    base_iso = _is_isotope_like(focal)
    base_ice = _is_ice_like(focal)
    keep_iso = bool(include_isotopes) or base_iso
    keep_ice = bool(include_ice) or base_ice
    up_depth = max(0, int(upstream_depth))
    down_depth = max(0, int(downstream_depth))

    nodes: Set[str] = {focal}
    upstream: Set[str] = set()
    downstream: Set[str] = set()
    up_hop: Dict[str, int] = {}
    dn_hop: Dict[str, int] = {}
    edges: List[dict] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    def _keep(name: str) -> bool:
        if not is_species_node(name):
            return False
        if name == focal:
            return True
        if not keep_iso and _is_isotope_like(name):
            return False
        if not keep_ice and _is_ice_like(name):
            return False
        return True

    def _add_edge(edge: Optional[dict]) -> None:
        if not edge:
            return
        if not _keep(edge['source']) or not _keep(edge['target']):
            return
        key = (edge['source'], edge['target'], edge.get('mid_label', ''), edge['raw'])
        if key in seen:
            return
        seen.add(key)
        edges.append(edge)
        nodes.add(edge['source'])
        nodes.add(edge['target'])

    # Upstream walk: formation of each species → precursor pathway edges
    q = deque([(focal, 0)])
    visited_up: Set[str] = {focal}
    while q:
        sp, depth = q.popleft()
        if depth >= up_depth:
            continue
        for raw in _select_chain_reactions(
            _labels_for(labels_by_species, sp, 'formation'), sp,
            mode='formation',
            include_isotopes=keep_iso,
            include_ice=keep_ice,
        ):
            edge = pathway_edge_from_reaction(raw, sp, 'formation')
            if not edge or not _keep(edge['source']):
                continue
            _add_edge(edge)
            pre = edge['source']
            hop = depth + 1
            if pre not in up_hop or hop < up_hop[pre]:
                up_hop[pre] = hop
            upstream.add(pre)
            if pre in CHAIN_SEED_SPECIES:
                visited_up.add(pre)
                continue
            if pre not in visited_up:
                visited_up.add(pre)
                q.append((pre, hop))

    # Downstream walk: destruction of each species → product pathway edges
    q = deque([(focal, 0)])
    visited_dn: Set[str] = {focal}
    while q:
        sp, depth = q.popleft()
        if depth >= down_depth:
            continue
        for raw in _select_chain_reactions(
            _labels_for(labels_by_species, sp, 'destruction'), sp,
            mode='destruction',
            include_isotopes=keep_iso,
            include_ice=keep_ice,
        ):
            edge = pathway_edge_from_reaction(raw, sp, 'destruction')
            if not edge or not _keep(edge['target']):
                continue
            _add_edge(edge)
            pr = edge['target']
            hop = depth + 1
            if pr not in dn_hop or hop < dn_hop[pr]:
                dn_hop[pr] = hop
            if pr not in upstream:
                downstream.add(pr)
            if pr not in visited_dn:
                visited_dn.add(pr)
                q.append((pr, hop))

    # Also attach direct top-path edges among already-kept nodes via formation
    # of intermediates (fills lateral links like CH+ → CH2+ → CH3+).
    for sp in list(nodes):
        if sp == focal:
            continue
        for raw in _select_chain_reactions(
            _labels_for(labels_by_species, sp, 'formation'), sp,
            mode='formation', limit=3,
            include_isotopes=keep_iso, include_ice=keep_ice,
        ):
            edge = pathway_edge_from_reaction(raw, sp, 'formation')
            if not edge:
                continue
            if edge['source'] in nodes and edge['target'] in nodes:
                _add_edge(edge)

    if len(nodes) > max_nodes:
        scored = []
        for n in nodes:
            if n == focal:
                continue
            r = up_hop.get(n, dn_hop.get(n, 99))
            seed_pen = 0 if n in CHAIN_SEED_SPECIES else 1
            scored.append((r, seed_pen, n))
        scored.sort()
        keep = {focal}
        for _, _, n in scored:
            if len(keep) >= max_nodes:
                break
            keep.add(n)
        nodes = keep
        edges = [e for e in edges if e['source'] in nodes and e['target'] in nodes]
        upstream &= keep
        downstream &= keep

    rank: Dict[str, int] = {focal: 0}
    for n in nodes:
        if n == focal:
            continue
        if n in upstream:
            rank[n] = -int(up_hop.get(n, 1))
        elif n in downstream:
            rank[n] = int(dn_hop.get(n, 1))
        else:
            rank[n] = 0
    return nodes, edges, rank


def build_network_figure(
    labels_by_species: Dict[str, dict],
    focal_species: str,
    *,
    upstream_depth: int = DEFAULT_CHAIN_UPSTREAM,
    downstream_depth: int = DEFAULT_CHAIN_DOWNSTREAM,
    include_isotopes: bool = False,
    include_ice: bool = False,
    highlight_species: Optional[str] = None,
    theme_colors: Optional[dict] = None,
    title: Optional[str] = None,
) -> Tuple[go.Figure, dict]:
    """Combined multi-hop pathway network (formation + destruction) for one species.

    Orthogonal arrows with per-link ports/lanes; partner colours; click/highlight
    a species to isolate its incident links for easier tracing.
    """
    t = theme_colors or {}
    paper = t.get('paper_bg', 'white')
    plot_bg = t.get('plot_bg', 'white')
    font = t.get('font', '#222')
    title_c = t.get('title', font)
    hl = str(highlight_species or '').strip() or None

    try:
        up = max(0, min(6, int(upstream_depth)))
    except (TypeError, ValueError):
        up = DEFAULT_CHAIN_UPSTREAM
    try:
        down = max(0, min(6, int(downstream_depth)))
    except (TypeError, ValueError):
        down = DEFAULT_CHAIN_DOWNSTREAM

    # Allow denser / deeper networks more nodes so depth controls stay useful.
    max_nodes = min(
        90,
        CHAIN_MAX_NODES
        + 6 * (up + down)
        + (10 if include_isotopes else 0)
        + (10 if include_ice else 0),
    )
    nodes, edges, rank = build_multihop_pathway(
        labels_by_species, focal_species,
        upstream_depth=up, downstream_depth=down,
        max_nodes=max_nodes,
        include_isotopes=include_isotopes,
        include_ice=include_ice,
    )
    edges = _dedupe_pathway_edges(edges)
    # Map highlight id onto a real node name (HCO⁺ → HCO+, etc.).
    if hl:
        by_key = {_species_match_key(n): n for n in nodes}
        hl = by_key.get(_species_match_key(hl), hl)
        if hl not in nodes:
            hl = None
    if hl and hl not in nodes:
        hl = None
    status = dict(
        species=str(focal_species or ''),
        n_nodes=len(nodes),
        n_edges=len(edges),
        upstream_depth=up,
        downstream_depth=down,
        include_isotopes=bool(include_isotopes),
        include_ice=bool(include_ice),
        highlight=hl or '',
        ok=bool(edges) and bool(focal_species),
    )
    fig = go.Figure()
    if not focal_species or not edges:
        fig.update_layout(
            title=title or 'Reaction pathway network',
            paper_bgcolor=paper, plot_bgcolor=plot_bg,
            autosize=True,
            height=PAGE_FIG_HEIGHT,
            annotations=[dict(
                text='Select a species to build its reaction pathway network',
                xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=t.get('placeholder', '#888')),
            )],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=24, r=24, t=48, b=40),
        )
        return fig, status

    pos, layout_meta = layout_rank_grid(nodes, rank, focal_species=focal_species)
    n_edges = len(edges)
    n_nodes = len(nodes)
    edge_width = 2.2 if n_edges < 40 else (1.7 if n_edges < 80 else 1.35)
    label_size = 15 if n_nodes < 28 else (13 if n_nodes < 50 else 11)
    mid_size = 12 if n_edges < 40 else 10
    base_size = 40.0 if n_nodes < 28 else (32.0 if n_nodes < 50 else 26.0)
    base_half = 0.17 if n_nodes < 28 else (0.14 if n_nodes < 50 else 0.12)
    head_len = 0.10
    col_gap = float(layout_meta.get('col_gap', 2.8))
    row_gap = float(layout_meta.get('row_gap', 1.6))

    # Compress onto a single page canvas. Dense isotope graphs need a tighter
    # target span so height stays ≤ PAGE_FIG_HEIGHT instead of growing off-screen.
    xs_fit = [p[0] for p in pos.values()]
    ys_fit = [p[1] for p in pos.values()]
    data_w = (max(xs_fit) - min(xs_fit)) if xs_fit else 4.0
    data_h = (max(ys_fit) - min(ys_fit)) if ys_fit else 4.0
    max_col = int(layout_meta.get('max_col', 1) or 1)
    if n_nodes >= 30 or max_col >= 7:
        target_span_x, target_span_y = 16.5, 10.2
    elif n_nodes >= 22 or max_col >= 5:
        target_span_x, target_span_y = 17.5, 11.0
    else:
        target_span_x, target_span_y = 18.5, 12.0
    sx = sy = 1.0
    if data_w > target_span_x and data_w > 1e-6:
        sx = target_span_x / data_w
        pos = {n: (x * sx, y) for n, (x, y) in pos.items()}
        col_gap *= sx
        data_w = target_span_x
    if data_h > target_span_y and data_h > 1e-6:
        sy = target_span_y / data_h
        pos = {n: (x, y * sy) for n, (x, y) in pos.items()}
        row_gap *= sy
        data_h = target_span_y

    # Degree → shape / size (square → circle → hexagon for busy hubs).
    out_n: Dict[str, int] = defaultdict(int)
    in_n: Dict[str, int] = defaultdict(int)
    for e in edges:
        out_n[e['source']] += 1
        in_n[e['target']] += 1
    degree = {n: out_n.get(n, 0) + in_n.get(n, 0) for n in nodes}
    node_symbol: Dict[str, str] = {}
    node_size: Dict[str, float] = {}
    node_half: Dict[str, float] = {}
    node_font: Dict[str, float] = {}
    # Marker size is in pixels; routes use data units. Prefer a *slightly small*
    # half so tips land on the visible border (oversized half → tips in empty space).
    est_span_y = max(4.0, float(data_h)) + 2.8
    est_span_x = max(4.0, float(data_w)) + 2.8
    usable_h = float(PAGE_FIG_HEIGHT - 140)
    usable_w = float(PAGE_FIG_WIDTH_EST - 80)
    px_per = min(usable_h / est_span_y, usable_w / est_span_x)
    px_per = max(px_per, 40.0)
    half_cap = min(0.42 * max(col_gap, 1.0), 0.40 * max(row_gap, 1.0))
    half_scale = max(0.72, min(1.0, min(sx, sy) * 1.02))
    for n in nodes:
        n_chars = _label_char_count(n)
        sym, sz, half_style = _node_visual_style(
            degree.get(n, 0),
            is_focal=(n == focal_species),
            base_size=base_size,
            base_half=base_half,
            label_chars=n_chars,
        )
        node_symbol[n] = sym
        node_size[n] = sz
        # ~0.40·size ≈ half-extent in px for square/circle markers (not full 0.5).
        half_from_px = 0.40 * float(sz) / px_per
        half_from_style = float(half_style) * half_scale
        # Take the tighter of the two so arrows do not float past the drawn box.
        node_half[n] = max(0.10, min(half_from_px, half_from_style, half_cap))
        node_font[n] = _label_font_size(label_size, n_chars)

    lane_i: Dict[Tuple[int, int], int] = defaultdict(int)
    lane_totals: Dict[Tuple[int, int], int] = defaultdict(int)
    pair_i: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_totals: Dict[Tuple[str, str], int] = defaultdict(int)
    for ee in edges:
        a = int(rank.get(ee['source'], 0))
        b = int(rank.get(ee['target'], 0))
        lane_totals[(min(a, b), max(a, b))] += 1
        pair_totals[(ee['source'], ee['target'])] += 1

    annotations = []
    legend_shown: Set[str] = set()
    route_xs: List[float] = []
    route_ys: List[float] = []
    n_curve = 0
    n_ortho = 0
    n_detour = 0

    # Node boxes used as routing obstacles (avoid crossing intermediate species).
    obstacles: Dict[str, Tuple[float, float, float]] = {
        n: (pos[n][0], pos[n][1], node_half[n]) for n in nodes if n in pos
    }

    # Clicked species + direct neighbours (fully solid when highlighting).
    related: Set[str] = set()
    if hl:
        related.add(hl)
        for e in edges:
            if e['source'] == hl:
                related.add(e['target'])
            if e['target'] == hl:
                related.add(e['source'])

    others = sorted(n for n in nodes if n != focal_species and n in pos)
    shape_note = {
        'square': '',
        'circle': '<br><i>hub (≥%d links)</i>' % HUB_DEGREE_CIRCLE,
        'hexagon': '<br><i>busy hub (≥%d links)</i>' % HUB_DEGREE_HEX,
    }

    def _species_hover(n: str) -> str:
        labs = sorted({e['raw'] for e in edges
                       if e['source'] == n or e['target'] == n})
        bits = [n, shape_note.get(node_symbol.get(n, 'square'), '')]
        if hl and n in related:
            bits.append(f'<br><b>On path with {hl}</b>')
        bits.append('<br><i>Click to highlight links</i>')
        if labs:
            bits.append('<br>' + '<br>'.join(labs[:6]))
        return ''.join(bits)

    # Ghost unrelated species first (behind edges); related drawn fully opaque later.
    if hl and others:
        dim_nodes = [n for n in others if n not in related]
        if dim_nodes:
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in dim_nodes],
                y=[pos[n][1] for n in dim_nodes],
                mode='markers+text',
                text=[_format_node_label(n) for n in dim_nodes],
                textposition='middle center',
                textfont=dict(
                    size=[max(9, node_font.get(n, label_size) - 1) for n in dim_nodes],
                    color='#c0c0c0',
                ),
                customdata=dim_nodes,
                marker=dict(
                    symbol=[node_symbol[n] for n in dim_nodes],
                    size=[node_size[n] for n in dim_nodes],
                    color='#f3f3f3',
                    line=dict(color='#d0d0d0', width=1.0),
                ),
                opacity=0.22,
                hovertext=[_species_hover(n) for n in dim_nodes],
                hoverinfo='text',
                showlegend=False, name='species-dim',
                cliponaxis=True,
            ))

    # When highlighting, omit unrelated links entirely (avoids orphan stubs).
    draw_edges = [
        e for e in edges
        if hl is None or e['source'] == hl or e['target'] == hl
    ]
    # Per-face ports: many links on the same box side sit next to each other.
    face_ports = _assign_face_ports(draw_edges, pos, node_half)
    ordered = sorted(
        draw_edges,
        key=lambda e: (e.get('partner', ''), e['source'], e['target']),
    )

    for e in ordered:
        if e['source'] not in pos or e['target'] not in pos:
            continue
        pclass = e.get('partner') or 'other'
        leg, color, dash = PARTNER_STYLES.get(pclass, PARTNER_STYLES['other'])
        x0, y0 = pos[e['source']]
        x1, y1 = pos[e['target']]
        partner = str(e.get('partner', ''))
        port_src = face_ports.get((e['source'], e['target'], partner, 'out'), 0.0)
        port_dst = face_ports.get((e['source'], e['target'], partner, 'in'), 0.0)
        r0 = int(rank.get(e['source'], 0))
        r1 = int(rank.get(e['target'], 0))
        col_key = (min(r0, r1), max(r0, r1))
        li = lane_i[col_key]
        lane_i[col_key] = li + 1
        n_lanes = max(1, lane_totals.get(col_key, 1))
        pair_key = (e['source'], e['target'])
        pi = pair_i[pair_key]
        pair_i[pair_key] = pi + 1
        n_pair = max(1, pair_totals.get(pair_key, 1))
        # Opposite-direction reactions between the same species pair.
        rev_n = pair_totals.get((e['target'], e['source']), 0)
        n_undirected = n_pair + rev_n
        use_curve = _should_use_curve(
            n_pair, n_lanes, n_edges, n_undirected=n_undirected,
        )
        if n_pair >= 2:
            bend = _curve_bend(pi, n_pair, col_gap, strong=True)
        elif rev_n > 0:
            bend = _curve_bend(0, max(2, n_undirected), col_gap, strong=True)
            bend = abs(bend) * (
                1.0 if (e['source'], e['target']) < (e['target'], e['source']) else -1.0
            )
        else:
            bend = _curve_bend(li, max(n_lanes, 2), col_gap, strong=False)
        lane = _lane_offset(li, n_lanes, col_gap)
        pts, style = _route_avoiding_nodes(
            x0, y0, x1, y1,
            box_half_src=node_half[e['source']],
            box_half_dst=node_half[e['target']],
            port_src=port_src, port_dst=port_dst,
            lane=lane, bend=bend,
            prefer_curve=use_curve,
            obstacles=obstacles,
            skip={e['source'], e['target']},
            col_gap=col_gap,
            n_samples=26 if n_edges < 90 else 20,
        )
        if style == 'curve':
            n_curve += 1
        elif style == 'detour':
            n_detour += 1
            n_ortho += 1
        else:
            n_ortho += 1
        for px, py in pts:
            route_xs.append(px)
            route_ys.append(py)
        width = edge_width + (1.4 if hl else 0.0)
        if len(pts) >= 2:
            ax0, ay0 = pts[-2]
            tip_x, tip_y = pts[-1]  # exactly on the target box face
            dx, dy = tip_x - ax0, tip_y - ay0
            length = float(np.hypot(dx, dy))
            if length > 1e-9:
                ux, uy = dx / length, dy / length
                # Keep the arrow tip on the box face; shorten only the stroked
                # line so the head sits on the border (not floating in empty space).
                line_gap = min(0.025, 0.18 * length)
                pts = list(pts[:-1]) + [
                    (tip_x - ux * line_gap, tip_y - uy * line_gap)
                ]
                annotations.append(dict(
                    x=tip_x, y=tip_y,
                    ax=tip_x - ux * head_len, ay=tip_y - uy * head_len,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=3, arrowsize=1.05,
                    arrowwidth=max(1.15, width * 0.8),
                    arrowcolor=color, text='',
                    opacity=1.0,
                ))
        # Also pin the route start exactly on the source box face (already from
        # _facing_attach); record start for axis padding.
        if pts:
            route_xs.append(pts[0][0])
            route_ys.append(pts[0][1])
        mid = e.get('mid_label') or ''
        # Hide mid-edge partner text on dense graphs — it collides with nodes/lanes.
        if mid and pclass == 'other' and n_edges < 55:
            mid_i = max(0, len(pts) // 2)
            mx, my = pts[mid_i]
            annotations.append(dict(
                x=mx, y=my + 0.12, xref='x', yref='y',
                text=mid, showarrow=False,
                font=dict(size=mid_size, color=color),
                bgcolor=paper, borderpad=1,
                xanchor='center', yanchor='bottom',
            ))
        hover = str(e.get('raw') or f"{e['source']} → {e['target']}")
        show_leg = pclass not in legend_shown
        if show_leg:
            legend_shown.add(pclass)
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode='lines',
            line=dict(color=color, width=width, dash=dash),
            opacity=1.0,
            hoverinfo='text', hovertext=hover,
            name=leg, legendgroup=pclass,
            showlegend=show_leg,
        ))

    # Keep routing mix available for tests / status (not shown in UI).
    status['n_curve'] = n_curve
    status['n_ortho'] = n_ortho
    status['n_detour'] = n_detour

    # Related / normal species on top — always fully opaque (no reduced opacity).
    if others:
        if hl:
            rel_nodes = [n for n in others if n in related]
            if rel_nodes:
                fig.add_trace(go.Scatter(
                    x=[pos[n][0] for n in rel_nodes],
                    y=[pos[n][1] for n in rel_nodes],
                    mode='markers+text',
                    text=[_format_node_label(n) for n in rel_nodes],
                    textposition='middle center',
                    textfont=dict(
                        size=[max(label_size, node_font.get(n, label_size))
                              for n in rel_nodes],
                        color='#111111',
                    ),
                    customdata=rel_nodes,
                    marker=dict(
                        symbol=[node_symbol[n] for n in rel_nodes],
                        size=[node_size[n] * (1.15 if n == hl else 1.06)
                              for n in rel_nodes],
                        # Solid fills — never translucent.
                        color=['#ffd24d' if n == hl else '#ffffff' for n in rel_nodes],
                        line=dict(
                            color=['#8a6d00' if n == hl else '#111111'
                                   for n in rel_nodes],
                            width=3.0,
                        ),
                    ),
                    opacity=1.0,
                    hovertext=[_species_hover(n) for n in rel_nodes],
                    hoverinfo='text',
                    showlegend=False, name='species-related',
                    cliponaxis=True,
                ))
        else:
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in others],
                y=[pos[n][1] for n in others],
                mode='markers+text',
                text=[_format_node_label(n) for n in others],
                textposition='middle center',
                textfont=dict(
                    size=[node_font.get(n, label_size) for n in others],
                    color=font,
                ),
                customdata=others,
                marker=dict(
                    symbol=[node_symbol[n] for n in others],
                    size=[node_size[n] for n in others],
                    color=plot_bg if plot_bg != 'rgba(0,0,0,0)' else 'white',
                    line=dict(color=font, width=1.5),
                ),
                opacity=1.0,
                hovertext=[_species_hover(n) for n in others],
                hoverinfo='text',
                showlegend=False, name='species',
                cliponaxis=True,
            ))

    if focal_species in pos:
        fx, fy = pos[focal_species]
        focal_related = (hl is None) or (focal_species in related)
        if focal_related:
            # Same solid treatment as other related boxes (gold if clicked).
            if hl and focal_species == hl:
                fill, border = '#ffd24d', '#8a6d00'
            elif hl:
                fill, border = '#ffffff', '#111111'
            else:
                fill, border = '#7ec0ee', '#111111'
            fig.add_trace(go.Scatter(
                x=[fx], y=[fy], mode='markers+text',
                text=[_format_node_label(focal_species)],
                textposition='middle center',
                textfont=dict(
                    size=max(label_size + 1, node_font.get(focal_species, label_size + 1)),
                    color='#111111',
                ),
                customdata=[focal_species],
                marker=dict(
                    symbol=node_symbol.get(focal_species, 'square'),
                    size=node_size.get(focal_species, base_size + 10) * (1.12 if hl else 1.0),
                    color=fill,
                    line=dict(color=border, width=3.0),
                ),
                opacity=1.0,
                hovertext=(
                    f'{focal_species}'
                    + (f'<br><b>On path with {hl}</b>' if hl else '')
                    + '<br><i>Click to highlight links</i>'
                    + (f'<br><b>Highlighting: {hl}</b>' if hl else '')
                ),
                hoverinfo='text',
                showlegend=False, name='focus',
                cliponaxis=True,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[fx], y=[fy], mode='markers+text',
                text=[_format_node_label(focal_species)],
                textposition='middle center',
                textfont=dict(
                    size=node_font.get(focal_species, label_size + 1),
                    color='#c0c0c0',
                ),
                customdata=[focal_species],
                marker=dict(
                    symbol=node_symbol.get(focal_species, 'square'),
                    size=node_size.get(focal_species, base_size + 10),
                    color='#f3f3f3',
                    line=dict(color='#d0d0d0', width=1.0),
                ),
                opacity=0.22,
                hovertext=(
                    f'{focal_species}<br><i>Click to highlight links</i>'
                    + (f'<br><b>Highlighting: {hl}</b>' if hl else '')
                ),
                hoverinfo='text',
                showlegend=False, name='focus',
                cliponaxis=True,
            ))

    # Keep all content (nodes, curved bows, legend) inside the page-safe canvas.
    # Marker sizes are in pixels — pad axes in data units so boxes are not clipped
    # at the plot frame; keep the Partner legend inside the figure (not below it).
    xs_all = [p[0] for p in pos.values()] + route_xs
    ys_all = [p[1] for p in pos.values()] + route_ys
    half_pad = max(node_half.values()) if node_half else base_half
    # Extra room for marker radius + curve bows near the frame.
    marker_pad = max(0.45, half_pad * 2.2)
    if xs_all:
        xs_all.extend([min(xs_all) - marker_pad, max(xs_all) + marker_pad])
    if ys_all:
        ys_all.extend([min(ys_all) - marker_pad, max(ys_all) + marker_pad])
    fig_h = int(PAGE_FIG_HEIGHT)

    subtitle = ''
    if hl:
        n_rel = max(0, len(related) - 1)
        subtitle = (
            f'  ·  highlighting {hl} and {n_rel} linked species '
            f'(click empty / Clear to reset)'
        )
    fig.update_layout(
        title=dict(
            text=(title or f'Reaction pathway network — {focal_species}') + subtitle,
            font=dict(size=15, color=title_c), x=0.02, xanchor='left',
            y=0.995, yanchor='top', pad=dict(t=2, b=4)),
        paper_bgcolor=paper,
        plot_bgcolor=plot_bg,
        autosize=True,
        height=fig_h,
        # Legend sits in the top band (not y<0), so it cannot push past the page.
        margin=dict(l=28, r=28, t=108, b=24),
        font=dict(family='Arial, sans-serif', size=14, color=font),
        showlegend=True,
        legend=dict(
            title=dict(text='Partner', font=dict(size=12)),
            bgcolor=t.get('legend_bg', 'rgba(255,255,255,0.92)'),
            bordercolor=t.get('legend_border', '#ccc'),
            borderwidth=1,
            orientation='h',
            x=0.0, xanchor='left',
            y=1.0, yanchor='bottom',
            font=dict(size=12),
            tracegroupgap=4,
            itemsizing='constant',
            valign='middle',
        ),
        xaxis=dict(visible=False, showgrid=False, zeroline=False,
                   range=_pad_range(xs_all, pad=0.04, abs_pad=0.12),
                   automargin=False, fixedrange=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False,
                   range=_pad_range(ys_all, pad=0.05, abs_pad=0.15),
                   automargin=False, fixedrange=False),
        annotations=annotations,
        uirevision=(f'pathnet|{focal_species}|{up}|{down}|'
                    f'iso={int(bool(include_isotopes))}|ice={int(bool(include_ice))}|'
                    f'hl={hl or ""}'),
        hovermode='closest',
        clickmode='event+select',
    )
    return fig, status


def build_network_panels(
    labels_by_species: Dict[str, dict],
    focal_species: str,
    *,
    upstream_depth: int = DEFAULT_CHAIN_UPSTREAM,
    downstream_depth: int = DEFAULT_CHAIN_DOWNSTREAM,
    include_isotopes: bool = False,
    include_ice: bool = False,
    highlight_species: Optional[str] = None,
    theme_colors: Optional[dict] = None,
    title: Optional[str] = None,
) -> Tuple[go.Figure, dict]:
    """Build the combined pathway network figure + status."""
    return build_network_figure(
        labels_by_species, focal_species,
        upstream_depth=upstream_depth,
        downstream_depth=downstream_depth,
        include_isotopes=include_isotopes,
        include_ice=include_ice,
        highlight_species=highlight_species,
        theme_colors=theme_colors,
        title=title,
    )


def status_message(status: dict) -> str:
    """Human-readable status line for the Chemistry tab."""
    if not status or not status.get('ok'):
        return ('Load a chemistry grid and select a species — the pathway network '
                'walks formation routes into the species and destruction routes out. '
                'Spacing grows with depth; busy hubs become circles/hexagons. '
                'Hover a link for its reaction; click a species to highlight its links.')
    extras = []
    if status.get('include_isotopes'):
        extras.append('isotopes on')
    if status.get('include_ice'):
        extras.append('ice on')
    if status.get('highlight'):
        extras.append(f'highlight {status["highlight"]}')
    extra = f'; {", ".join(extras)}' if extras else ''
    return (
        f'Pathway network for {status.get("species")}: '
        f'{status.get("n_nodes", 0)} species, {status.get("n_edges", 0)} links '
        f'(upstream {status.get("upstream_depth")}, '
        f'downstream {status.get("downstream_depth")}{extra}). '
        f'Click a box to isolate its links and highlight linked species.'
    )



# ---------------------------------------------------------------------------
# Chemical-chain flowchart (depth-limited pathways around the focal species)
# ---------------------------------------------------------------------------

def _is_isotope_like(name: str) -> bool:
    """Heuristic for rare-isotope species (¹³C, ¹⁸O, ^-notation, …)."""
    s = str(name or '')
    if not s:
        return False
    if '13' in s or '18' in s or '15' in s or '17' in s:
        return True
    if '^' in s:
        return True
    return False


def _is_ice_like(name: str) -> bool:
    """Heuristic for grain-surface / desorption / ice-related tokens."""
    s = str(name or '')
    if not s:
        return False
    up = s.upper()
    if up.startswith('J') and len(s) > 1:  # grain-surface (J-species)
        return True
    if 'DES' in up:
        return True
    if up.endswith('*') and up not in ('H2*',):
        return True
    if up in ('CR-DES', 'H2-DES', 'FREEZE', 'DUMMY'):
        return True
    return False


def _chain_partner_score(
    reactants: Sequence[str],
    products: Sequence[str],
    *,
    penalize_isotopes: bool = True,
    penalize_ice: bool = True,
) -> int:
    """Lower is better — prefer seed / H₂ / C⁺ / photon channels over clutter."""
    pclass = partner_class(reactants, products)
    class_rank = {
        'h2': 0, 'cp': 1, 'c': 2, 'electron': 3, 'photon': 4, 'other': 5,
    }.get(pclass, 6)
    seeds = sum(1 for t in reactants if normalize_token(t) in CHAIN_SEED_SPECIES)
    toks = [t for t in list(reactants) + list(products) if is_species_node(t)]
    isotopes = sum(1 for t in toks if _is_isotope_like(t)) if penalize_isotopes else 0
    ice = sum(1 for t in toks if _is_ice_like(t)) if penalize_ice else 0
    return class_rank * 10 - 3 * seeds + 8 * isotopes + 6 * ice + len(reactants)


def _labels_for(labels_by_species: Dict[str, dict], species: str, mode: str) -> List[str]:
    entry = (labels_by_species or {}).get(species) or {}
    return list(entry.get(mode) or [])


def _select_chain_reactions(
    labels: Sequence[str],
    owner: str,
    *,
    mode: str,
    limit: int = CHAIN_BRANCH_LIMIT,
    include_isotopes: bool = False,
    include_ice: bool = False,
) -> List[str]:
    """Pick a readable subset of formation/destruction channels for ``owner``."""
    scored = []
    for raw in labels or []:
        reactants, products = parse_reaction(raw)
        if not reactants or not products:
            continue
        scored.append((_chain_partner_score(
            reactants, products,
            penalize_isotopes=not include_isotopes,
            penalize_ice=not include_ice,
        ), str(raw)))
    scored.sort(key=lambda x: (x[0], x[1]))
    out, seen = [], set()
    for _, raw in scored:
        if raw in seen:
            continue
        seen.add(raw)
        out.append(raw)
        if len(out) >= limit:
            break
    return out


def _append_reaction_edges(
    edges: List[dict],
    seen: Set[Tuple[str, str, str, str]],
    raw: str,
    owner: str,
    mode: str,
    *,
    only_to: Optional[Set[str]] = None,
    only_from: Optional[Set[str]] = None,
) -> None:
    for src, dst, pclass, elabel, lab in reaction_edges(raw, owner_species=owner):
        if only_to is not None and dst not in only_to:
            continue
        if only_from is not None and src not in only_from:
            continue
        key = (src, dst, pclass, lab)
        if key in seen:
            continue
        seen.add(key)
        edges.append(dict(
            source=src, target=dst, partner=pclass,
            label=elabel, raw=lab, owner=str(owner), mode=mode,
        ))


def _augment_chain_partners(
    labels_by_species: Dict[str, dict],
    *,
    focal: str,
    nodes: Set[str],
    edges: List[dict],
    seen_e: Set[Tuple[str, str, str, str]],
    upstream: Set[str],
    downstream: Set[str],
    up_hop: Dict[str, int],
    dn_hop: Dict[str, int],
    want_isotopes: bool,
    want_ice: bool,
    extra_budget: int,
) -> None:
    """Attach isotope / ice partners of already-kept nodes (additive, not replacing)."""
    if extra_budget <= 0 or (not want_isotopes and not want_ice):
        return

    def _wanted(name: str) -> bool:
        if name == focal or name in nodes or not is_species_node(name):
            return False
        is_iso = _is_isotope_like(name)
        is_ice = _is_ice_like(name)
        if is_iso and want_isotopes:
            return True
        if is_ice and want_ice:
            return True
        return False

    added = 0
    # Prefer partners of the focus, then of other base nodes.
    owners = [focal] + sorted(n for n in nodes if n != focal)
    for sp in owners:
        if added >= extra_budget:
            break
        base_up = int(up_hop.get(sp, 0 if sp == focal else 1))
        base_dn = int(dn_hop.get(sp, 0 if sp == focal else 1))
        # Formation of ``sp``: isotope/ice precursors → into ``sp``.
        for raw in _labels_for(labels_by_species, sp, 'formation'):
            if added >= extra_budget:
                break
            reactants, _products = parse_reaction(raw)
            extras = [t for t in reactants if t != sp and _wanted(t)]
            if not extras:
                continue
            before = added
            for pre in extras:
                if added >= extra_budget:
                    break
                hop = base_up + 1 if sp != focal else 1
                if pre not in up_hop or hop < up_hop[pre]:
                    up_hop[pre] = hop
                upstream.add(pre)
                nodes.add(pre)
                added += 1
            if added > before:
                _append_reaction_edges(
                    edges, seen_e, raw, sp, 'formation', only_to={sp},
                )
        # Destruction of ``sp``: isotope/ice products ← from ``sp``.
        for raw in _labels_for(labels_by_species, sp, 'destruction'):
            if added >= extra_budget:
                break
            _reactants, products = parse_reaction(raw)
            extras = [t for t in products if t != sp and _wanted(t)]
            if not extras:
                continue
            before = added
            for pr in extras:
                if added >= extra_budget:
                    break
                hop = base_dn + 1 if sp != focal else 1
                if pr not in dn_hop or hop < dn_hop[pr]:
                    dn_hop[pr] = hop
                # Keep precursor rank if already upstream.
                if pr not in upstream:
                    downstream.add(pr)
                nodes.add(pr)
                added += 1
            if added > before:
                _append_reaction_edges(
                    edges, seen_e, raw, sp, 'destruction', only_from={sp},
                )


def build_chemical_chain(
    labels_by_species: Dict[str, dict],
    focal_species: str,
    *,
    upstream_depth: int = DEFAULT_CHAIN_UPSTREAM,
    downstream_depth: int = DEFAULT_CHAIN_DOWNSTREAM,
    max_nodes: int = CHAIN_MAX_NODES,
    include_isotopes: bool = False,
    include_ice: bool = False,
) -> Tuple[Set[str], List[dict], Dict[str, int]]:
    """Walk formation (up) and destruction (down) labels around ``focal_species``.

    The base chain is always built from main (non-isotope, non-ice) species.
    Optional isotope / ice partners are then attached as extras so toggles add
    nodes instead of replacing the main chemistry.

    Returns ``(nodes, edges, rank)`` with ``rank[n]`` smaller (= higher on the
    plot) for more precursor-like species.
    """
    focal = str(focal_species or '')
    if not focal or not labels_by_species:
        return set(), [], {}

    # Base walk: only include iso/ice when the focus itself is that class.
    base_iso = _is_isotope_like(focal)
    base_ice = _is_ice_like(focal)
    up_depth = max(0, int(upstream_depth))
    down_depth = max(0, int(downstream_depth))

    nodes: Set[str] = {focal}
    upstream: Set[str] = set()
    downstream: Set[str] = set()
    up_hop: Dict[str, int] = {}
    dn_hop: Dict[str, int] = {}
    edges: List[dict] = []
    seen_e: Set[Tuple[str, str, str, str]] = set()

    def _keep_base(name: str) -> bool:
        if not is_species_node(name):
            return False
        if name == focal:
            return True
        if not base_iso and _is_isotope_like(name):
            return False
        if not base_ice and _is_ice_like(name):
            return False
        return True

    # Upstream: recursively expand formation channels (main chemistry only)
    q = deque([(focal, 0)])
    visited_up: Set[str] = {focal}
    while q:
        sp, depth = q.popleft()
        if depth >= up_depth:
            continue
        for raw in _select_chain_reactions(
            _labels_for(labels_by_species, sp, 'formation'), sp,
            mode='formation',
            include_isotopes=base_iso,
            include_ice=base_ice,
        ):
            reactants, _products = parse_reaction(raw)
            precursors = [t for t in reactants if t != sp and _keep_base(t)]
            if not precursors:
                continue
            _append_reaction_edges(
                edges, seen_e, raw, sp, 'formation', only_to={sp},
            )
            for pre in precursors:
                hop = depth + 1
                if pre not in up_hop or hop < up_hop[pre]:
                    up_hop[pre] = hop
                upstream.add(pre)
                nodes.add(pre)
                if pre in CHAIN_SEED_SPECIES:
                    visited_up.add(pre)
                    continue
                if pre not in visited_up:
                    visited_up.add(pre)
                    q.append((pre, hop))

    # Downstream: expand destruction channels (main chemistry only)
    q = deque([(focal, 0)])
    visited_dn: Set[str] = {focal}
    while q:
        sp, depth = q.popleft()
        if depth >= down_depth:
            continue
        for raw in _select_chain_reactions(
            _labels_for(labels_by_species, sp, 'destruction'), sp,
            mode='destruction',
            include_isotopes=base_iso,
            include_ice=base_ice,
        ):
            _reactants, products = parse_reaction(raw)
            prods = [t for t in products if t != sp and _keep_base(t)]
            if not prods:
                continue
            _append_reaction_edges(
                edges, seen_e, raw, sp, 'destruction', only_from={sp},
            )
            for pr in prods:
                hop = depth + 1
                if pr not in dn_hop or hop < dn_hop[pr]:
                    dn_hop[pr] = hop
                downstream.add(pr)
                nodes.add(pr)
                if pr not in visited_dn:
                    visited_dn.add(pr)
                    q.append((pr, hop))

    edges = [e for e in edges
             if e['source'] in nodes and e['target'] in nodes
             and _keep_base(e['source']) and _keep_base(e['target'])]

    if len(nodes) > max_nodes:
        scored = []
        for n in nodes:
            if n == focal:
                continue
            r = up_hop.get(n, dn_hop.get(n, 99))
            seed_pen = 0 if n in CHAIN_SEED_SPECIES else 1
            # Prefer dropping any iso/ice that snuck into the base set.
            clutter = int(_is_isotope_like(n) or _is_ice_like(n))
            scored.append((clutter, r, seed_pen, n))
        scored.sort()
        keep = {focal}
        for _, _, _, n in scored:
            if len(keep) >= max_nodes:
                break
            keep.add(n)
        nodes = keep
        upstream &= keep
        downstream &= keep
        edges = [e for e in edges if e['source'] in nodes and e['target'] in nodes]

    base_nodes = set(nodes)

    # Additive pass: attach isotope / ice partners of the kept base chain.
    _augment_chain_partners(
        labels_by_species,
        focal=focal,
        nodes=nodes,
        edges=edges,
        seen_e=seen_e,
        upstream=upstream,
        downstream=downstream,
        up_hop=up_hop,
        dn_hop=dn_hop,
        want_isotopes=bool(include_isotopes) and not base_iso,
        want_ice=bool(include_ice) and not base_ice,
        extra_budget=int(CHAIN_EXTRA_NODES),
    )

    # Drop edges to anything outside the final node set; drop edges that only
    # connected through filtered partners of the augment pass.
    edges = [e for e in edges
             if e['source'] in nodes and e['target'] in nodes
             and (e['source'] in base_nodes or e['target'] in base_nodes
                  or e['source'] == focal or e['target'] == focal)]

    rank: Dict[str, int] = {focal: 0}
    for n in nodes:
        if n == focal:
            continue
        if n in upstream:
            rank[n] = -int(up_hop.get(n, 1))
        elif n in downstream:
            rank[n] = int(dn_hop.get(n, 1))
        else:
            rank[n] = 0
    return nodes, edges, rank



def layout_chain_positions(
    nodes: Iterable[str],
    edges: Sequence[dict],
    rank: Dict[str, int],
    *,
    focal_species: Optional[str] = None,
) -> Dict[str, Tuple[float, float]]:
    """Top→bottom layered layout from precursor ranks (flowchart style)."""
    node_list = list(nodes)
    if not node_list:
        return {}
    focal = str(focal_species or '')
    by_rank: Dict[int, List[str]] = defaultdict(list)
    for n in node_list:
        by_rank[int(rank.get(n, 0))].append(n)
    for r in by_rank:
        names = by_rank[r]
        if focal in names:
            by_rank[r] = [focal] + sorted(x for x in names if x != focal)
        else:
            by_rank[r] = sorted(names)

    ranks = sorted(by_rank)
    if not ranks:
        return {focal: (0.0, 0.0)} if focal else {}
    y_of = {r: float(i) for i, r in enumerate(reversed(ranks))}  # top = precursors
    # Stretch vertical spacing.
    if len(y_of) > 1:
        ymax = max(y_of.values())
        y_of = {r: 2.2 * (y / ymax if ymax else 0.0) for r, y in y_of.items()}

    pos: Dict[str, Tuple[float, float]] = {}
    for r, names in by_rank.items():
        y = y_of[r]
        if len(names) == 1:
            pos[names[0]] = (0.0, y)
            continue
        span = max(1.6, 0.9 * (len(names) - 1))
        xs = np.linspace(-span, span, len(names))
        # Keep focal centred when alone in its preference.
        if focal in names and len(names) > 1:
            others = [n for n in names if n != focal]
            pos[focal] = (0.0, y)
            if others:
                span = max(1.6, 0.9 * len(others))
                xs = np.linspace(-span, span, len(others))
                # Avoid x≈0 collision with focal.
                xs = [x if abs(x) > 0.35 else (0.45 if x >= 0 else -0.45) for x in xs]
                for n, x in zip(others, xs):
                    pos[n] = (float(x), y)
        else:
            for n, x in zip(names, xs):
                pos[n] = (float(x), y)
    return pos


def build_chemical_chain_figure(
    labels_by_species: Dict[str, dict],
    focal_species: str,
    *,
    upstream_depth: int = DEFAULT_CHAIN_UPSTREAM,
    downstream_depth: int = DEFAULT_CHAIN_DOWNSTREAM,
    include_isotopes: bool = False,
    include_ice: bool = False,
    theme_colors: Optional[dict] = None,
    title: Optional[str] = None,
) -> Tuple[go.Figure, dict]:
    """KoSens-style chemical-chain flowchart for one focal species."""
    t = theme_colors or {}
    paper = t.get('paper_bg', 'white')
    plot_bg = t.get('plot_bg', 'white')
    font = t.get('font', '#222')
    title_c = t.get('title', font)

    try:
        up = max(0, min(6, int(upstream_depth)))
    except (TypeError, ValueError):
        up = DEFAULT_CHAIN_UPSTREAM
    try:
        down = max(0, min(6, int(downstream_depth)))
    except (TypeError, ValueError):
        down = DEFAULT_CHAIN_DOWNSTREAM

    nodes, edges, rank = build_chemical_chain(
        labels_by_species, focal_species,
        upstream_depth=up, downstream_depth=down,
        include_isotopes=include_isotopes,
        include_ice=include_ice,
    )
    status = dict(
        species=str(focal_species or ''),
        n_nodes=len(nodes),
        n_edges=len(edges),
        upstream_depth=up,
        downstream_depth=down,
        include_isotopes=bool(include_isotopes),
        include_ice=bool(include_ice),
        ok=bool(nodes) and bool(focal_species),
    )
    fig = go.Figure()
    if not focal_species or not nodes:
        fig.update_layout(
            title=title or 'Chemical chain',
            paper_bgcolor=paper, plot_bgcolor=plot_bg,
            width=1100, height=820,
            annotations=[dict(
                text='Select a species to build its chemical chain',
                xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=t.get('placeholder', '#888')),
            )],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig, status

    pos = layout_chain_positions(nodes, edges, rank, focal_species=focal_species)
    n_nodes = len(nodes)
    edge_width = 2.2 if n_nodes < 30 else 1.5
    node_size = 26 if n_nodes < 30 else 16
    label_size = 13 if n_nodes < 30 else 11
    tip_standoff = 0.12
    head_len = 0.18

    arrow_annotations = []
    for pclass, (leg, color, dash) in PARTNER_STYLES.items():
        class_edges = [e for e in edges if e['partner'] == pclass]
        if not class_edges:
            continue
        xs, ys = [], []
        for e in class_edges:
            if e['source'] not in pos or e['target'] not in pos:
                continue
            x0, y0 = pos[e['source']]
            x1, y1 = pos[e['target']]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            dx, dy = x1 - x0, y1 - y0
            length = float(np.hypot(dx, dy))
            if length < 1e-9:
                continue
            ux, uy = dx / length, dy / length
            tip_x = x1 - ux * tip_standoff
            tip_y = y1 - uy * tip_standoff
            arrow_annotations.append(dict(
                x=tip_x, y=tip_y,
                ax=tip_x - ux * head_len, ay=tip_y - uy * head_len,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=3, arrowsize=1.1,
                arrowwidth=max(1.2, edge_width * 0.7),
                arrowcolor=color, text='',
            ))
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines',
                line=dict(color=color, width=edge_width, dash=dash),
                hoverinfo='skip', name=leg, legendgroup=pclass,
            ))

    others = sorted(n for n in nodes if n != focal_species)
    for group, size, color, line_c, line_w in (
        (others, node_size, '#f7f7f7', '#444444', 1.2),
        ([focal_species] if focal_species in nodes else [],
         node_size * 1.6, '#9ecae1', '#2171b5', 2.6),
    ):
        if not group:
            continue
        xs = [pos[n][0] for n in group]
        ys = [pos[n][1] for n in group]
        texts = [_format_node_label(n) for n in group]
        hover = []
        for n in group:
            labs = sorted({e['raw'] for e in edges
                           if e['source'] == n or e['target'] == n})
            hover.append(n + ('<br>' + '<br>'.join(labs[:8]) if labs else ''))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers+text', text=texts,
            textposition='top center',
            textfont=dict(size=label_size, color=font),
            marker=dict(size=size, color=color,
                        line=dict(color=line_c, width=line_w)),
            hovertext=hover, hoverinfo='text',
            showlegend=False, name='species',
        ))

    fig.update_layout(
        title=dict(
            text=title or f'Chemical chain — {focal_species}',
            font=dict(size=15, color=title_c), x=0.02, xanchor='left'),
        paper_bgcolor=paper,
        plot_bgcolor=plot_bg,
        width=1200,
        height=900,
        autosize=True,
        margin=dict(l=40, r=40, t=60, b=60),
        font=dict(family='Arial, sans-serif', size=13, color=font),
        showlegend=True,
        legend=dict(
            title=dict(text='Partner', font=dict(size=14)),
            bgcolor=t.get('legend_bg', 'rgba(255,255,255,0.85)'),
            bordercolor=t.get('legend_border', '#ccc'),
            borderwidth=1,
            orientation='h', y=-0.08, x=0, font=dict(size=14),
        ),
        xaxis=dict(visible=False, showgrid=False, zeroline=False,
                   range=_pad_range([p[0] for p in pos.values()])),
        yaxis=dict(visible=False, showgrid=False, zeroline=False,
                   range=_pad_range([p[1] for p in pos.values()])),
        annotations=arrow_annotations,
        uirevision=(f'chemchain|{focal_species}|{up}|{down}|'
                    f'iso={int(bool(include_isotopes))}|ice={int(bool(include_ice))}'),
        hovermode='closest',
    )
    return fig, status


def chain_status_message(status: dict) -> str:
    if not status or not status.get('ok'):
        return ('Chemical chain: load a chemistry grid and pick a species. '
                'Depth controls how many reaction hops above/below the focus '
                'are included from the full chem-file network. Isotopes and ice '
                'species are off by default.')
    extras = []
    if status.get('include_isotopes'):
        extras.append('isotopes on')
    if status.get('include_ice'):
        extras.append('ice on')
    extra = f'; {", ".join(extras)}' if extras else '; isotopes/ice off'
    return (
        f'Chemical chain for {status.get("species")}: '
        f'{status.get("n_nodes", 0)} species, {status.get("n_edges", 0)} edges '
        f'(upstream depth {status.get("upstream_depth")}, '
        f'downstream depth {status.get("downstream_depth")}{extra}).'
    )