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
    'grain': ('+ grain', '#8c564b', 'solid'),
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
DEFAULT_PARTNER_LABEL_SIZE = 11
DEFAULT_SPECIES_LABEL_SIZE = 15
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
    'GRAIN', 'TH-DES', 'CR-DES', 'H2-DES', 'PH-DES', 'FREEZE', 'DUMMY',
}

# Grain / desorption partners shown on the arrow instead of as species boxes.
_SURFACE_PROCESS = {
    'GRAIN': 'grain',
    'TH-DES': 'T-des',
    'CR-DES': 'CR-des',
    'H2-DES': 'H\u2082-des',
    'PH-DES': 'UV',
    'FREEZE': 'freeze',
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
    if upper.startswith('GRAIN'):
        return 'GRAIN'
    if upper in _SURFACE_PROCESS:
        return upper
    # PH-DES lines sometimes emit lowercase j-species / products.
    if (upper.startswith('J') and len(upper) > 1
            and not tok.startswith('J') and tok[:1].isalpha()):
        return tok[0].upper() + tok[1:]
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
    # HDF5 groups use J_13CO; reaction strings use J^13CO.
    return s.translate(table).upper().replace('^', '_')


def _match_known_species(name: str, known: Optional[Iterable[str]] = None) -> str:
    """Map a reaction token onto an HDF5 species key when one exists."""
    raw = normalize_token(name)
    if not raw:
        return raw
    if not known:
        return raw.replace('^', '_') if '^' in raw else raw
    known_list = list(known)
    if raw in known_list:
        return raw
    key = _species_match_key(raw)
    by_key = {_species_match_key(n): n for n in known_list}
    if key in by_key:
        return by_key[key]
    swapped = raw.replace('^', '_') if '^' in raw else raw.replace('_', '^')
    if swapped in known_list:
        return swapped
    return by_key.get(key, raw)


def is_process_token(token: str) -> bool:
    t = normalize_token(token)
    if not t:
        return True
    if t in ('e-', 'PHOTON', 'CRPHOT', 'CRP', 'CR', 'GRAIN'):
        return True
    up = t.upper()
    if up in _PROCESS_TOKENS or up in _SURFACE_PROCESS:
        return True
    if up.startswith('GRAIN') or up.endswith('-DES'):
        return True
    return False


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
    if any(t in _SURFACE_PROCESS or t == 'GRAIN' or str(t).endswith('-DES')
           for t in toks):
        return 'grain'
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
    base = int(base)
    floor = max(8, int(round(0.55 * base)))
    if n_chars >= 8:
        return max(floor, base - 4)
    if n_chars >= 6:
        return max(floor, base - 3)
    if n_chars >= 5:
        return max(floor, base - 2)
    return max(8, base)


def _species_box_pixels(
    name: str,
    font_pt: float,
    *,
    is_focal: bool = False,
) -> Tuple[float, float]:
    """Pixel half-width / half-height that hugs a species label."""
    n = max(1, _label_char_count(name))
    fs = max(8.0, float(font_pt))
    raw = str(name or '')
    # Arial-ish width; sub/superscripts are narrower than a full em.
    text_w = max(fs * 1.05, 0.55 * fs * n)
    text_h = fs * (1.28 if any(c in raw for c in '+-') else 1.18)
    pad_x = 8.0 if is_focal else 6.0
    pad_y = 6.5 if is_focal else 5.0
    return 0.5 * (text_w + 2.0 * pad_x), 0.5 * (text_h + 2.0 * pad_y)


def _font_to_fit_box(
    name: str,
    hw_px: float,
    hh_px: float,
    *,
    is_focal: bool = False,
    max_pt: float,
) -> int:
    """Largest integer pt that still fits in a pixel-sized name plate."""
    n = max(1, _label_char_count(name))
    raw = str(name or '')
    pad_x = 8.0 if is_focal else 6.0
    pad_y = 6.5 if is_focal else 5.0
    inner_w = max(4.0, 2.0 * hw_px - 2.0 * pad_x)
    inner_h = max(4.0, 2.0 * hh_px - 2.0 * pad_y)
    h_em = 1.28 if any(c in raw for c in '+-') else 1.18
    fs_w = inner_w / max(1.05, 0.55 * n)
    fs_h = inner_h / h_em
    return int(max(8, min(float(max_pt), fs_w, fs_h)))


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
    if nt in _SURFACE_PROCESS:
        return _SURFACE_PROCESS[nt]
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


def _partner_text(edge: dict) -> str:
    """Label drawn on the arrow: co-reactants, else the partner class."""
    mid = str(edge.get('mid_label') or '').strip()
    if mid:
        return mid
    pclass = str(edge.get('partner') or '')
    fallback = {
        'electron': 'e\u207b',
        'photon': 'h\u03bd / CR',
        'h2': 'H\u2082',
        'c': 'C',
        'cp': 'C\u207a',
        'grain': 'grain',
    }
    return fallback.get(pclass, '')


def pathway_edge_from_reaction(
    label: str,
    focal_species: str,
    mode: str,
    *,
    known_species: Optional[Iterable[str]] = None,
) -> Optional[dict]:
    """One pathway edge: primary species node + mid-arrow partner label.

    Formation: first reactant species → focal; other reactants on the arrow.
    Destruction: focal → first product species; co-reactants on the arrow.
    """
    focal = str(focal_species or '')
    reactants, products = parse_reaction(label)
    if not reactants or not products or not focal:
        return None
    known = list(known_species) if known_species is not None else None
    if known:
        focal = _match_known_species(focal, known)
        reactants = [_match_known_species(t, known) if is_species_node(t) else t
                     for t in reactants]
        products = [_match_known_species(t, known) if is_species_node(t) else t
                    for t in products]
    mode = str(mode or '')
    pclass = partner_class(reactants, products)

    if mode == 'formation':
        primary, mid_toks = _primary_reactant(reactants)
        if not primary or primary == focal:
            return None
        return dict(
            source=primary, target=focal, partner=pclass,
            mid_label=_mid_label_from_tokens(mid_toks),
            raw=str(label), mode=mode,
        )

    if mode == 'destruction':
        primary = _primary_product(products, focal)
        if not primary:
            return None
        mid_toks = [t for t in reactants if normalize_token(t) != focal
                    and _species_match_key(t) != _species_match_key(focal)]
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


def _as_wh(value, default: float = 0.16) -> Tuple[float, float]:
    """Unpack a box half as ``(half_w, half_h)`` (square if a single number)."""
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return max(0.08, float(value[0])), max(0.08, float(value[1]))
    try:
        h = float(value)
    except (TypeError, ValueError):
        h = float(default)
    h = max(0.08, h)
    return h, h


def _port_offsets(n: int, span: float = 0.28) -> List[float]:
    """Spread attachment points evenly along a box edge."""
    if n <= 1:
        return [0.0]
    return [float(v) for v in np.linspace(-span, span, n)]


def _face_port_span(box_half: float, count: int) -> float:
    """How far along a face to spread ports for ``count`` attachments.

    Multiple links may share the geometrically preferred side; they are spaced
    across nearly the full edge so each path can be followed on its own.
    """
    if count <= 1:
        return 0.0
    return max(0.0, float(box_half) * 0.92)


def _point_on_face(
    x: float, y: float, hw: float, hh: float, face: str, port: float,
) -> Tuple[float, float]:
    """A point on the named side of a rectangle, port-offset along that side."""
    if face == 'right':
        p = max(-0.95 * hh, min(0.95 * hh, float(port)))
        return x + hw, y + p
    if face == 'left':
        p = max(-0.95 * hh, min(0.95 * hh, float(port)))
        return x - hw, y + p
    if face == 'top':
        p = max(-0.95 * hw, min(0.95 * hw, float(port)))
        return x + p, y + hh
    p = max(-0.95 * hw, min(0.95 * hw, float(port)))
    return x + p, y - hh


def _geom_exit_face(x0: float, y0: float, x1: float, y1: float) -> str:
    """Exit side from source→target orientation."""
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) >= abs(dy):
        return 'right' if dx >= 0 else 'left'
    return 'top' if dy >= 0 else 'bottom'


def _geom_enter_face(x0: float, y0: float, x1: float, y1: float) -> str:
    """Enter side from source→target orientation."""
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) >= abs(dy):
        return 'left' if dx >= 0 else 'right'
    return 'bottom' if dy >= 0 else 'top'


def _edge_key(e: dict) -> Tuple[str, str, str]:
    return (str(e.get('source', '')), str(e.get('target', '')),
            str(e.get('partner', '')))


def _assign_face_ports(
    draw_edges: Sequence[dict],
    pos: Dict[str, Tuple[float, float]],
    node_half: Dict[str, float],
) -> Tuple[Dict[Tuple[str, str, str, str], float],
           Dict[Tuple[str, str, str], str],
           Dict[Tuple[str, str, str], str]]:
    """Keep each link on its preferred side; space attachments along that edge.

    Returns ``(port_of, exit_face, enter_face)``. Ports are keyed
    ``(source, target, partner, 'out'|'in')``.
    """
    exit_face: Dict[Tuple[str, str, str], str] = {}
    enter_face: Dict[Tuple[str, str, str], str] = {}
    usable = []
    for e in draw_edges:
        s, t = e.get('source', ''), e.get('target', '')
        if s not in pos or t not in pos:
            continue
        key = _edge_key(e)
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        exit_face[key] = _geom_exit_face(x0, y0, x1, y1)
        enter_face[key] = _geom_enter_face(x0, y0, x1, y1)
        usable.append(e)

    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for e in usable:
        key = _edge_key(e)
        groups[(e['source'], exit_face[key], 'out')].append(e)
        groups[(e['target'], enter_face[key], 'in')].append(e)

    port_of: Dict[Tuple[str, str, str, str], float] = {}
    for (node, face, which), elist in groups.items():
        def _sort_key(e: dict, _face: str = face, _which: str = which) -> tuple:
            other = e['target'] if _which == 'out' else e['source']
            ox, oy = pos[other]
            # Order along the face matches where the other species sits, so a
            # line heading up leaves from the upper part of a vertical edge.
            if _face in ('left', 'right'):
                return (oy, ox, other)
            return (ox, oy, other)

        ordered = sorted(elist, key=_sort_key)
        hw, hh = _as_wh(node_half.get(node, 0.18))
        face_len = hh if face in ('left', 'right') else hw
        span = _face_port_span(face_len, len(ordered))
        offs = _port_offsets(len(ordered), span)
        for e, off in zip(ordered, offs):
            port_of[(*_edge_key(e), which)] = off
    return port_of, exit_face, enter_face


def _lane_offset(index: int, n_lanes: int, col_gap: float) -> float:
    """Mid-corridor offset for one of ``n_lanes`` parallel orthogonal routes.

    Targets a readable gap between neighbours (~0.13–0.17 data units) and
    widens the bundle when many edges share the same column pair.
    """
    if n_lanes <= 1:
        return 0.0
    # Preferred neighbour gap; allow a bit more room in wider columns.
    gap = min(0.20, max(0.14, 0.11 * float(col_gap)))
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
    aligned: bool = False, skip_layers: bool = False,
) -> bool:
    """Paper-style arcs except for a single aligned neighbour-to-neighbour link."""
    if n_pair >= 2 or n_undirected >= 2 or skip_layers:
        return True
    if aligned and n_lanes <= 2:
        return False
    return True


def _facing_attach(
    x0: float, y0: float, x1: float, y1: float,
    *,
    box_half_src: float,
    box_half_dst: float,
    port_src: float,
    port_dst: float,
    src_face: Optional[str] = None,
    dst_face: Optional[str] = None,
) -> Tuple[float, float, float, float]:
    """Exit/enter points on the chosen sides of source and target boxes."""
    hw_s, hh_s = _as_wh(box_half_src)
    hw_d, hh_d = _as_wh(box_half_dst)
    if not src_face:
        src_face = _geom_exit_face(x0, y0, x1, y1)
    if not dst_face:
        dst_face = _geom_enter_face(x0, y0, x1, y1)
    sx, sy = _point_on_face(x0, y0, hw_s, hh_s, src_face, port_src)
    tx, ty = _point_on_face(x1, y1, hw_d, hh_d, dst_face, port_dst)
    return sx, sy, tx, ty


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
    src_face: Optional[str] = None,
    dst_face: Optional[str] = None,
) -> List[Tuple[float, float]]:
    """Smooth arc between chosen box sides (A&A-style multi-edge curves)."""
    sx, sy, tx, ty = _facing_attach(
        x0, y0, x1, y1,
        box_half_src=box_half_src, box_half_dst=box_half_dst,
        port_src=port_src, port_dst=port_dst,
        src_face=src_face, dst_face=dst_face,
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
    """One arrow per (source, target); comma-join distinct reaction partners.

    Parallel channels such as H₂O + He⁺ / H₃⁺ / H⁺ → H₂O⁺ become a single
    curve labelled ``He⁺, H₃⁺, H⁺``, matching paper-style network diagrams.
    """
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for e in edges or []:
        src, dst = str(e.get('source') or ''), str(e.get('target') or '')
        if not src or not dst:
            continue
        groups[(src, dst)].append(e)

    out: List[dict] = []
    for (src, dst), bundle in groups.items():
        if len(bundle) == 1:
            out.append(dict(bundle[0]))
            continue
        labels: List[str] = []
        seen_lab: Set[str] = set()
        raws: List[str] = []
        class_counts: Dict[str, int] = defaultdict(int)
        for e in bundle:
            lab = _partner_text(e)
            lab_key = lab.casefold()
            if lab and lab_key not in seen_lab:
                seen_lab.add(lab_key)
                labels.append(lab)
            raw = str(e.get('raw') or '').strip()
            if raw and raw not in raws:
                raws.append(raw)
            class_counts[str(e.get('partner') or 'other')] += 1
        # Keep the most common partner class so colour still hints at the mix.
        pclass = max(
            class_counts,
            key=lambda k: (class_counts[k], k != 'other'),
        ) if class_counts else 'other'
        merged = dict(bundle[0])
        merged['source'] = src
        merged['target'] = dst
        merged['partner'] = pclass
        merged['mid_label'] = ', '.join(labels)
        merged['raw'] = '<br>'.join(raws) if raws else str(merged.get('raw') or '')
        out.append(merged)
    return out


def _edge_route(
    x0: float, y0: float, x1: float, y1: float,
    *,
    box_half_src: float,
    box_half_dst: float,
    lane: float,
    port_src: float,
    port_dst: float,
    vertical_first: bool = False,
    src_face: Optional[str] = None,
    dst_face: Optional[str] = None,
) -> List[Tuple[float, float]]:
    """Orthogonal route that starts/ends on the chosen sides of both boxes."""
    sx, sy, tx, ty = _facing_attach(
        x0, y0, x1, y1,
        box_half_src=box_half_src, box_half_dst=box_half_dst,
        port_src=port_src, port_dst=port_dst,
        src_face=src_face, dst_face=dst_face,
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
    cx: float, cy: float, half, *, pad: float = 0.08,
) -> bool:
    """True if the segment passes through an expanded node box."""
    hw, hh = _as_wh(half)
    xmin, xmax = cx - hw - pad, cx + hw + pad
    ymin, ymax = cy - hh - pad, cy + hh + pad
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
    src_face: Optional[str] = None,
    dst_face: Optional[str] = None,
) -> List[Tuple[float, float]]:
    """Manhattan detour around obstacles in the local corridor only."""
    sx, sy, tx, ty = _facing_attach(
        x0, y0, x1, y1,
        box_half_src=box_half_src, box_half_dst=box_half_dst,
        port_src=port_src, port_dst=port_dst,
        src_face=src_face, dst_face=dst_face,
    )
    clear = 0.18
    if side in ('left', 'right'):
        y_lo, y_hi = (sy, ty) if sy <= ty else (ty, sy)
        x_mid = 0.5 * (sx + tx)
        x_band = max(0.85, 0.55 * abs(sx - tx) + 0.55)
        blocked: List[Tuple[float, float, float, float]] = []
        for name, (cx, cy, half) in obstacles.items():
            if name in skip:
                continue
            hw, hh = _as_wh(half)
            if abs(cx - x_mid) > x_band + hw:
                continue
            if y_lo - hh - 0.04 <= cy <= y_hi + hh + 0.04:
                blocked.append((cx, cy, hw, hh))
        if blocked:
            if side == 'right':
                x_clear = max(cx + hw for cx, _, hw, _ in blocked) + clear
            else:
                x_clear = min(cx - hw for cx, _, hw, _ in blocked) - clear
        else:
            x_clear = x_mid + (0.40 if side == 'right' else -0.40)
        x_lim = x_mid + (1.15 if side == 'right' else -1.15)
        if side == 'right':
            x_clear = min(max(x_clear, x_mid + 0.22), x_lim)
        else:
            x_clear = max(min(x_clear, x_mid - 0.22), x_lim)
        return [(sx, sy), (x_clear, sy), (x_clear, ty), (tx, ty)]

    x_lo, x_hi = (sx, tx) if sx <= tx else (tx, sx)
    y_mid = 0.5 * (sy + ty)
    y_band = max(0.85, 0.55 * abs(sy - ty) + 0.55)
    blocked = []
    for name, (cx, cy, half) in obstacles.items():
        if name in skip:
            continue
        hw, hh = _as_wh(half)
        if abs(cy - y_mid) > y_band + hh:
            continue
        if x_lo - hw <= cx <= x_hi + hw:
            blocked.append((cx, cy, hw, hh))
    if blocked:
        if side == 'top':
            y_clear = max(cy + hh for _, cy, _, hh in blocked) + clear
        else:
            y_clear = min(cy - hh for _, cy, _, hh in blocked) - clear
    else:
        y_clear = y_mid + (0.35 if side == 'top' else -0.35)
    y_lim = y_mid + (1.15 if side == 'top' else -1.15)
    if side == 'top':
        y_clear = min(max(y_clear, y_mid + 0.22), y_lim)
    else:
        y_clear = max(min(y_clear, y_mid - 0.22), y_lim)
    return [(sx, sy), (sx, y_clear), (tx, y_clear), (tx, ty)]


def _polyline_length(pts: Sequence[Tuple[float, float]]) -> float:
    """Euclidean length of a polyline."""
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pts) - 1):
        total += float(np.hypot(pts[i + 1][0] - pts[i][0],
                                pts[i + 1][1] - pts[i][1]))
    return total


def _column_bypass_route(
    x0: float, y0: float, x1: float, y1: float,
    *,
    box_half_src: float,
    box_half_dst: float,
    port_src: float,
    port_dst: float,
    obstacles: Dict[str, Tuple[float, float, float]],
    skip: Set[str],
    side: str = 'right',
) -> List[Tuple[float, float]]:
    """Short hop around a box sitting between two vertically aligned species."""
    if y1 >= y0:
        src_face, dst_face = 'top', 'bottom'
    else:
        src_face, dst_face = 'bottom', 'top'
    sx, sy, tx, ty = _facing_attach(
        x0, y0, x1, y1,
        box_half_src=box_half_src, box_half_dst=box_half_dst,
        port_src=port_src, port_dst=port_dst,
        src_face=src_face, dst_face=dst_face,
    )
    x_mid = 0.5 * (x0 + x1)
    y_lo, y_hi = (sy, ty) if sy <= ty else (ty, sy)
    blocked: List[Tuple[float, float, float, float]] = []
    for name, (cx, cy, half) in obstacles.items():
        if name in skip:
            continue
        hw, hh = _as_wh(half)
        if abs(cx - x_mid) <= hw + 0.40 and y_lo < cy < y_hi:
            blocked.append((cx, cy, hw, hh))
    pad = 0.20
    if blocked:
        if side == 'right':
            x_clear = max(cx + hw for cx, _, hw, _ in blocked) + pad
        else:
            x_clear = min(cx - hw for cx, _, hw, _ in blocked) - pad
    else:
        x_clear = x_mid + (0.38 if side == 'right' else -0.38)
    x_clear = max(x_mid - 1.05, min(x_mid + 1.05, x_clear))
    if side == 'right':
        x_clear = max(x_clear, x_mid + 0.22)
    else:
        x_clear = min(x_clear, x_mid - 0.22)
    return [(sx, sy), (x_clear, sy), (x_clear, ty), (tx, ty)]


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
    src_face: Optional[str] = None,
    dst_face: Optional[str] = None,
) -> Tuple[List[Tuple[float, float]], str]:
    """Pick a route that does not cross unrelated species boxes when possible."""
    kw = dict(
        box_half_src=box_half_src, box_half_dst=box_half_dst,
        port_src=port_src, port_dst=port_dst,
        src_face=src_face, dst_face=dst_face,
    )
    candidates: List[Tuple[str, List[Tuple[float, float]]]] = []
    same_column = abs(x1 - x0) < 0.55 * max(abs(y1 - y0), 0.4)
    mid_blocked = False
    if same_column:
        y_lo, y_hi = (y0, y1) if y0 <= y1 else (y1, y0)
        x_mid = 0.5 * (x0 + x1)
        for name, (cx, cy, half) in obstacles.items():
            if name in skip:
                continue
            hw, hh = _as_wh(half)
            if abs(cx - x_mid) <= hw + 0.25 and y_lo < cy < y_hi:
                mid_blocked = True
                break

    if same_column and mid_blocked:
        # Stay in this column: a short C around the box in between.
        candidates.append(('detour', _column_bypass_route(
            x0, y0, x1, y1, side='right', obstacles=obstacles, skip=skip,
            box_half_src=box_half_src, box_half_dst=box_half_dst,
            port_src=port_src, port_dst=port_dst)))
        candidates.append(('detour', _column_bypass_route(
            x0, y0, x1, y1, side='left', obstacles=obstacles, skip=skip,
            box_half_src=box_half_src, box_half_dst=box_half_dst,
            port_src=port_src, port_dst=port_dst)))

    if prefer_curve:
        candidates.append(('curve', _curve_route(
            x0, y0, x1, y1, bend=bend, n_samples=n_samples, **kw)))
        for b in (bend, -bend, bend + 0.35, bend - 0.35, 0.55, -0.55, 0.85, -0.85,
                  1.15, -1.15):
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

    if not same_column:
        candidates.append(('detour', _detour_route(
            x0, y0, x1, y1, side='right', obstacles=obstacles, skip=skip, **kw)))
        candidates.append(('detour', _detour_route(
            x0, y0, x1, y1, side='left', obstacles=obstacles, skip=skip, **kw)))
        candidates.append(('detour', _detour_route(
            x0, y0, x1, y1, side='top', obstacles=obstacles, skip=skip, **kw)))
        candidates.append(('detour', _detour_route(
            x0, y0, x1, y1, side='bottom', obstacles=obstacles, skip=skip, **kw)))

    # Prefer the shortest path that misses species boxes (never a figure-wide loop).
    straight = float(np.hypot(x1 - x0, y1 - y0)) + 0.12
    max_len = max(2.8 * straight, straight + 2.4)
    clear: List[Tuple[float, str, List[Tuple[float, float]]]] = []
    fallback: Optional[Tuple[str, List[Tuple[float, float]]]] = None
    for style, pts in candidates:
        hit = _polyline_hits_obstacles(pts, obstacles, skip, pad=0.10)
        if hit is not None:
            if fallback is None:
                fallback = (style, pts)
            continue
        clear.append((_polyline_length(pts), style, pts))
    if clear:
        clear.sort(key=lambda it: it[0])
        short = [it for it in clear if it[0] <= max_len]
        _, style, pts = (short or clear)[0]
        return pts, style
    if fallback is not None:
        return fallback[1], fallback[0]
    return candidates[0][1], candidates[0][0]


def _polyline_midpoint(pts: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    if not pts:
        return 0.0, 0.0
    i = max(0, len(pts) // 2)
    return float(pts[i][0]), float(pts[i][1])


def _polyline_clearance(
    pts: Sequence[Tuple[float, float]],
    others: Sequence[Sequence[Tuple[float, float]]],
) -> float:
    """Minimum sampled distance from the *body* of ``pts`` to other polylines.

    Endpoints are ignored: many arrows legitimately meet near a box face, and
    counting those tips would hide real mid-path overlaps.
    """
    if not pts or not others:
        return 1e9
    n = len(pts)
    lo = max(1, int(0.16 * (n - 1)))
    hi = min(n - 1, int(0.84 * (n - 1)) + 1)
    body = list(pts[lo:hi]) or list(pts)
    step = max(1, len(body) // 10)
    samples = body[::step]
    if body[-1] not in samples:
        samples.append(body[-1])
    best = 1e9
    for other in others:
        if not other:
            continue
        on = len(other)
        olo = max(1, int(0.16 * (on - 1)))
        ohi = min(on - 1, int(0.84 * (on - 1)) + 1)
        obody = list(other[olo:ohi]) or list(other)
        ostep = max(1, len(obody) // 10)
        osamples = obody[::ostep]
        for x, y in samples:
            for ox, oy in osamples:
                d = abs(x - ox) + abs(y - oy)
                if d < best:
                    best = d
    return float(best)


def _polyline_point(
    pts: Sequence[Tuple[float, float]],
    frac: float = 0.5,
) -> Tuple[float, float]:
    """Point a fraction of the way along a polyline (clamped off the ends)."""
    if not pts:
        return 0.0, 0.0
    n = len(pts)
    if n == 1:
        return float(pts[0][0]), float(pts[0][1])
    t = max(0.18, min(0.82, float(frac)))
    i = max(0, min(n - 2, int(t * (n - 1))))
    local = t * (n - 1) - i
    x0, y0 = pts[i]
    x1, y1 = pts[i + 1]
    return float(x0 + local * (x1 - x0)), float(y0 + local * (y1 - y0))


def _label_anchor(
    pts: Sequence[Tuple[float, float]],
    *,
    offset: float = 0.0,
    frac: float = 0.5,
) -> Tuple[float, float]:
    """Point on a route (mid-line by default); ``offset`` nudges off the path."""
    x, y = _polyline_point(pts, frac)
    if offset == 0.0 or len(pts) < 3:
        return x, y if offset == 0.0 else (x, y + offset)
    i = max(1, min(len(pts) - 2, int(round(frac * (len(pts) - 1)))))
    x0, y0 = pts[max(0, i - 1)]
    x1, y1 = pts[min(len(pts) - 1, i + 1)]
    dx, dy = x1 - x0, y1 - y0
    length = float(np.hypot(dx, dy))
    if length < 1e-9:
        return x, y + offset
    px, py = -dy / length, dx / length
    return x + px * offset, y + py * offset


def _clamp_label_pt(value, default: int, lo: int = 8, hi: int = 24) -> int:
    try:
        pt = int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        pt = int(default)
    return max(lo, min(hi, pt))


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
    edges: Optional[Sequence[dict]] = None,
) -> Tuple[Dict[str, Tuple[float, float]], dict]:
    """Column layout by pathway rank (upstream left → focus → destruction right).

    Node order inside a column uses the Sugiyama barycentre heuristic so
    connected species sit across from each other and long-range links cross
    less often. Spacing grows with the tallest column.
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
    if edges:
        by_rank = _barycenter_order(by_rank, edges, focal=focal)
    ranks = sorted(by_rank)
    empty_metrics = dict(col_gap=2.8, row_gap=1.6, n_cols=0, max_col=0)
    if not ranks:
        return ({focal: (0.0, 0.0)} if focal else {}), empty_metrics

    n_cols = len(ranks)
    max_col = max(len(by_rank[r]) for r in ranks)
    # Gaps grow with density; soft-caps keep the layout compressible onto one page.
    col_gap = max(2.8, 2.35 + 0.24 * max_col + 0.10 * max(0, n_cols - 3))
    row_gap = max(2.15, 1.70 + 0.24 * max_col)
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


def _barycenter_order(
    by_rank: Dict[int, List[str]],
    edges: Sequence[dict],
    *,
    focal: str = '',
    n_passes: int = 4,
) -> Dict[int, List[str]]:
    """Reorder nodes in each rank column to cut edge crossings (Sugiyama)."""
    ranks = sorted(by_rank)
    order = {r: list(by_rank[r]) for r in ranks}
    neigh: Dict[str, Set[str]] = defaultdict(set)
    for e in edges or []:
        s, t = e.get('source', ''), e.get('target', '')
        if s and t:
            neigh[s].add(t)
            neigh[t].add(s)

    def _median(node: str, index_of: Dict[str, int], fallback: float) -> float:
        vals = [index_of[m] for m in neigh.get(node, ()) if m in index_of]
        if not vals:
            return fallback
        vals.sort()
        mid = len(vals) // 2
        if len(vals) % 2:
            return float(vals[mid])
        return 0.5 * (vals[mid - 1] + vals[mid])

    def _sort_layer(layer: List[str], index_of: Dict[str, int]) -> List[str]:
        keyed = []
        for i, n in enumerate(layer):
            keyed.append((_median(n, index_of, float(i)), i, n))
        keyed.sort()
        out = [n for _, _, n in keyed]
        if focal in out and len(out) > 1:
            out = [focal] + [n for n in out if n != focal]
        return out

    for _ in range(max(1, n_passes)):
        for i, r in enumerate(ranks):
            if i == 0:
                continue
            prev = order[ranks[i - 1]]
            idx = {n: j for j, n in enumerate(prev)}
            order[r] = _sort_layer(order[r], idx)
        for i in range(len(ranks) - 2, -1, -1):
            nxt = order[ranks[i + 1]]
            idx = {n: j for j, n in enumerate(nxt)}
            order[ranks[i]] = _sort_layer(order[ranks[i]], idx)
    return order


def _ice_to_gas_name(name: str) -> str:
    """JCO / J_13CO → CO / _13CO (strip the grain-surface prefix)."""
    s = str(name or '').strip()
    if len(s) > 1 and s[0] in 'Jj':
        return s[1:]
    return s


def _wanted_optional_partner(
    name: str,
    *,
    focal: str,
    nodes: Set[str],
    want_isotopes: bool,
    want_ice: bool,
) -> bool:
    """True if ``name`` should be attached as an extra isotope / ice node."""
    if not is_species_node(name):
        return False
    if name == focal or name in nodes:
        return False
    is_iso = _is_isotope_like(name)
    is_ice = _is_ice_like(name)
    if is_iso and not want_isotopes:
        return False
    if is_ice and not want_ice:
        return False
    if is_ice and want_ice:
        gas = _ice_to_gas_name(name)
        gas_key = _species_match_key(gas)
        kept = {_species_match_key(n) for n in nodes}
        kept.add(_species_match_key(focal))
        # Only the grain-surface twin of a species already on the diagram.
        return bool(gas_key) and gas_key in kept
    if is_iso and want_isotopes:
        return True
    return False


def _attach_optional_pathway_partners(
    labels_by_species: Dict[str, dict],
    *,
    focal: str,
    nodes: Set[str],
    edges: List[dict],
    seen: Set[Tuple[str, str, str, str]],
    upstream: Set[str],
    downstream: Set[str],
    up_hop: Dict[str, int],
    dn_hop: Dict[str, int],
    known_keys: Sequence[str],
    want_isotopes: bool,
    want_ice: bool,
    extra_budget: int,
) -> None:
    """Add ice / isotope counterparts of the kept gas-phase chain.

    The main walk ranks H₂ / C⁺ channels first, so freeze-out (X + grain → JX)
    and desorption (JX + des → X) never make the top-N cut. This pass scans
    every label of already-kept nodes and attaches those partners.
    """
    if extra_budget <= 0 or (not want_isotopes and not want_ice):
        return

    def _wanted(name: str) -> bool:
        return _wanted_optional_partner(
            name, focal=focal, nodes=nodes,
            want_isotopes=want_isotopes, want_ice=want_ice,
        )

    def _commit(edge: Optional[dict]) -> bool:
        if not edge:
            return False
        edge['source'] = _match_known_species(edge['source'], known_keys)
        edge['target'] = _match_known_species(edge['target'], known_keys)
        if not is_species_node(edge['source']) or not is_species_node(edge['target']):
            return False
        key = (edge['source'], edge['target'], edge.get('mid_label', ''), edge['raw'])
        if key in seen:
            return False
        seen.add(key)
        edges.append(edge)
        nodes.add(edge['source'])
        nodes.add(edge['target'])
        return True

    added = 0
    owners = [focal] + sorted(n for n in list(nodes) if n != focal)
    for sp in owners:
        if added >= extra_budget:
            break
        base_up = int(up_hop.get(sp, 0 if sp == focal else 1))
        base_dn = int(dn_hop.get(sp, 0 if sp == focal else 1))
        for raw in _labels_for(labels_by_species, sp, 'formation'):
            if added >= extra_budget:
                break
            edge = pathway_edge_from_reaction(
                raw, sp, 'formation', known_species=known_keys)
            if not edge:
                continue
            pre = edge['source']
            is_new = pre not in nodes
            if is_new and not _wanted(pre):
                continue
            if not is_new and not (_is_ice_like(pre) or _is_isotope_like(pre)):
                continue
            if is_new:
                hop = base_up + 1 if sp != focal else 1
                if pre not in up_hop or hop < up_hop[pre]:
                    up_hop[pre] = hop
                upstream.add(pre)
            if _commit(edge) and is_new:
                added += 1
        for raw in _labels_for(labels_by_species, sp, 'destruction'):
            if added >= extra_budget:
                break
            edge = pathway_edge_from_reaction(
                raw, sp, 'destruction', known_species=known_keys)
            if not edge:
                continue
            pr = edge['target']
            is_new = pr not in nodes
            if is_new and not _wanted(pr):
                continue
            if not is_new and not (_is_ice_like(pr) or _is_isotope_like(pr)):
                continue
            if is_new:
                hop = base_dn + 1 if sp != focal else 1
                if pr not in dn_hop or hop < dn_hop[pr]:
                    dn_hop[pr] = hop
                if pr not in upstream:
                    downstream.add(pr)
            if _commit(edge) and is_new:
                added += 1


def build_multihop_pathway(
    labels_by_species: Dict[str, dict],
    focal_species: str,
    *,
    upstream_depth: int = DEFAULT_CHAIN_UPSTREAM,
    downstream_depth: int = DEFAULT_CHAIN_DOWNSTREAM,
    max_nodes: int = CHAIN_MAX_NODES,
    include_isotopes: bool = False,
    include_ice: bool = False,
    focal_formation: Optional[Sequence[str]] = None,
    focal_destruction: Optional[Sequence[str]] = None,
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

    known_keys = list(labels_by_species.keys())
    focal = _match_known_species(focal, known_keys) or focal

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
        if name == focal or _species_match_key(name) == _species_match_key(focal):
            return True
        if not keep_iso and _is_isotope_like(name):
            return False
        if not keep_ice and _is_ice_like(name):
            return False
        return True

    def _add_edge(edge: Optional[dict]) -> None:
        if not edge:
            return
        edge['source'] = _match_known_species(edge['source'], known_keys)
        edge['target'] = _match_known_species(edge['target'], known_keys)
        if not _keep(edge['source']) or not _keep(edge['target']):
            return
        key = (edge['source'], edge['target'], edge.get('mid_label', ''), edge['raw'])
        if key in seen:
            return
        seen.add(key)
        edges.append(edge)
        nodes.add(edge['source'])
        nodes.add(edge['target'])

    def _pick_reactions(sp: str, mode: str, *, limit: int = CHAIN_BRANCH_LIMIT) -> List[str]:
        """Focal species uses the tabulated channels; neighbours stay capped."""
        if sp == focal:
            forced = focal_formation if mode == 'formation' else focal_destruction
            if forced:
                out, seen_raw = [], set()
                for raw in forced:
                    lab = str(raw or '').strip()
                    if lab and lab not in seen_raw:
                        seen_raw.add(lab)
                        out.append(lab)
                return out
            # No table list: keep every formation/destruction label of the focus
            # so rate-dominant partners (N₂, CO, Si, …) are not dropped by the
            # H₂ / C⁺ / e⁻ heuristic.
            return [str(x) for x in _labels_for(labels_by_species, sp, mode) if str(x).strip()]
        return _select_chain_reactions(
            _labels_for(labels_by_species, sp, mode), sp,
            mode=mode, limit=limit,
            include_isotopes=keep_iso, include_ice=keep_ice,
        )

    # Always attach the selected species' own table channels, even if
    # upstream/downstream depth is 0 (depth only controls further hops).
    for raw in _pick_reactions(focal, 'formation'):
        edge = pathway_edge_from_reaction(
            raw, focal, 'formation', known_species=known_keys)
        if not edge or not _keep(edge['source']):
            continue
        _add_edge(edge)
        pre = edge['source']
        if pre not in up_hop:
            up_hop[pre] = 1
        upstream.add(pre)
    for raw in _pick_reactions(focal, 'destruction'):
        edge = pathway_edge_from_reaction(
            raw, focal, 'destruction', known_species=known_keys)
        if not edge or not _keep(edge['target']):
            continue
        _add_edge(edge)
        pr = edge['target']
        if pr not in dn_hop:
            dn_hop[pr] = 1
        if pr not in upstream:
            downstream.add(pr)

    # Upstream walk: formation of each species → precursor pathway edges
    q = deque([(focal, 0)])
    visited_up: Set[str] = {focal}
    while q:
        sp, depth = q.popleft()
        if depth >= up_depth:
            continue
        for raw in _pick_reactions(sp, 'formation'):
            edge = pathway_edge_from_reaction(
                raw, sp, 'formation', known_species=known_keys)
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
            # Ice counterparts stay as leaves unless the focus itself is ice.
            if keep_ice and not base_ice and _is_ice_like(pre):
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
        for raw in _pick_reactions(sp, 'destruction'):
            edge = pathway_edge_from_reaction(
                raw, sp, 'destruction', known_species=known_keys)
            if not edge or not _keep(edge['target']):
                continue
            _add_edge(edge)
            pr = edge['target']
            hop = depth + 1
            if pr not in dn_hop or hop < dn_hop[pr]:
                dn_hop[pr] = hop
            if pr not in upstream:
                downstream.add(pr)
            if keep_ice and not base_ice and _is_ice_like(pr):
                visited_dn.add(pr)
                continue
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
            edge = pathway_edge_from_reaction(
                raw, sp, 'formation', known_species=known_keys)
            if not edge:
                continue
            if edge['source'] in nodes and edge['target'] in nodes:
                _add_edge(edge)

    if len(nodes) > max_nodes:
        scored = []
        # Direct table partners of the focus must stay (e.g. N2H+ from H3+ + N2).
        focal_touch = {
            e['source'] if e['target'] == focal else e['target']
            for e in edges
            if e['source'] == focal or e['target'] == focal
        }
        for n in nodes:
            if n == focal or n in focal_touch:
                continue
            r = up_hop.get(n, dn_hop.get(n, 99))
            seed_pen = 0 if n in CHAIN_SEED_SPECIES else 1
            scored.append((r, seed_pen, n))
        scored.sort()
        keep = {focal} | (focal_touch & nodes)
        for _, _, n in scored:
            if len(keep) >= max_nodes:
                break
            keep.add(n)
        nodes = keep
        edges = [e for e in edges if e['source'] in nodes and e['target'] in nodes]
        upstream &= keep
        downstream &= keep

    # Gas-phase ranking never picks freeze-out / desorption. Attach those
    # partners of the kept chain when the ice / isotope toggles are on.
    extra_budget = 0
    if keep_iso and not base_iso:
        extra_budget += 10
    if keep_ice and not base_ice:
        extra_budget += 8
    if extra_budget:
        _attach_optional_pathway_partners(
            labels_by_species,
            focal=focal,
            nodes=nodes,
            edges=edges,
            seen=seen,
            upstream=upstream,
            downstream=downstream,
            up_hop=up_hop,
            dn_hop=dn_hop,
            known_keys=known_keys,
            want_isotopes=keep_iso and not base_iso,
            want_ice=keep_ice and not base_ice,
            extra_budget=extra_budget,
        )

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


def _resolve_hidden_species(
    hidden: Optional[Sequence[str]],
    nodes: Set[str],
    focal: str,
) -> Set[str]:
    """Map UI hide-list names onto current graph ids; never hide the focus."""
    if not hidden or not nodes:
        return set()
    by_key = {_species_match_key(n): n for n in nodes}
    focal_key = _species_match_key(focal)
    out: Set[str] = set()
    for raw in hidden:
        name = str(raw or '').strip()
        if not name:
            continue
        resolved = by_key.get(_species_match_key(name), name)
        if _species_match_key(resolved) == focal_key:
            continue
        if resolved in nodes:
            out.add(resolved)
    return out


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
    focal_formation: Optional[Sequence[str]] = None,
    focal_destruction: Optional[Sequence[str]] = None,
    hidden_species: Optional[Sequence[str]] = None,
    partner_label_size: Optional[float] = None,
    species_label_size: Optional[float] = None,
) -> Tuple[go.Figure, dict]:
    """Combined multi-hop pathway network (formation + destruction) for one species.

    Curved arrows meet species rectangles at the border; co-reactants are labelled
    on the link. Click a species to isolate its incident links.
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
    partner_pt = _clamp_label_pt(partner_label_size, DEFAULT_PARTNER_LABEL_SIZE)
    species_pt = _clamp_label_pt(species_label_size, DEFAULT_SPECIES_LABEL_SIZE)

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
        focal_formation=focal_formation,
        focal_destruction=focal_destruction,
    )
    edges = _dedupe_pathway_edges(edges)
    hidden_set = _resolve_hidden_species(
        hidden_species, nodes, str(focal_species or ''))
    if hidden_set:
        nodes = {n for n in nodes if n not in hidden_set}
        edges = [e for e in edges
                 if e['source'] not in hidden_set and e['target'] not in hidden_set]
        rank = {n: r for n, r in rank.items() if n in nodes}
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
        hidden=sorted(hidden_set),
        ok=bool(nodes) and bool(focal_species),
    )
    fig = go.Figure()
    if not focal_species or not nodes:
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

    pos, layout_meta = layout_rank_grid(
        nodes, rank, focal_species=focal_species, edges=edges)
    n_edges = len(edges)
    n_nodes = len(nodes)
    edge_width = 2.2 if n_edges < 40 else (1.7 if n_edges < 80 else 1.35)
    label_size = species_pt
    mid_size = partner_pt
    fig_h = int(PAGE_FIG_HEIGHT)
    head_len = 0.11
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

    node_half: Dict[str, Tuple[float, float]] = {}
    node_font: Dict[str, float] = {}
    node_size: Dict[str, float] = {}
    xs_pos = [p[0] for p in pos.values()] or [0.0]
    ys_pos = [p[1] for p in pos.values()] or [0.0]
    # Estimate the Plotly px / data-unit mapping from the node bounding box
    # (same idea as the final axis range). Boxes are then sized in *pixels*
    # from the font, so sparse graphs stay compact and dense ones still fit.
    span_x = max(1.0, (max(xs_pos) - min(xs_pos)) * 1.10 + 1.2)
    span_y = max(1.0, (max(ys_pos) - min(ys_pos)) * 1.12 + 1.2)
    usable_h = float(fig_h - 100)
    usable_w = float(PAGE_FIG_WIDTH_EST - 80)
    px_per = min(usable_w / span_x, usable_h / span_y)
    px_per = max(12.0, float(px_per))
    hw_cap = 0.44 * max(col_gap, 0.6)
    hh_cap = 0.40 * max(row_gap, 0.6)
    for n in nodes:
        n_chars = _label_char_count(n)
        is_focal_n = n == focal_species
        fs = float(_label_font_size(label_size, n_chars))
        hw_px, hh_px = _species_box_pixels(n, fs, is_focal=is_focal_n)
        hw, hh = hw_px / px_per, hh_px / px_per
        if hw > hw_cap or hh > hh_cap:
            hw, hh = min(hw, hw_cap), min(hh, hh_cap)
            fs = float(_font_to_fit_box(
                n, hw * px_per, hh * px_per,
                is_focal=is_focal_n, max_pt=fs,
            ))
            hw_px, hh_px = _species_box_pixels(n, fs, is_focal=is_focal_n)
            hw = min(hw_px / px_per, hw_cap)
            hh = min(hh_px / px_per, hh_cap)
        node_half[n] = (float(hw), float(hh))
        node_font[n] = int(round(fs))
        node_size[n] = max(18.0, 2.0 * max(hw, hh) * px_per)

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
    shapes = []
    route_xs: List[float] = []
    route_ys: List[float] = []
    n_curve = 0
    n_ortho = 0
    n_detour = 0
    used_routes: List[List[Tuple[float, float]]] = []

    # Node boxes used as routing obstacles (avoid crossing intermediate species).
    obstacles: Dict[str, Tuple[float, float, Tuple[float, float]]] = {
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

    def _species_hover(n: str) -> str:
        labs = sorted({e['raw'] for e in edges
                       if e['source'] == n or e['target'] == n})
        bits = [n]
        if _is_ice_like(n):
            bits.append('<br><i>ice / grain surface</i>')
        if hl and n in related:
            bits.append(f'<br><b>On path with {hl}</b>')
        bits.append('<br><i>Click to highlight links</i>')
        if labs:
            bits.append('<br>' + '<br>'.join(labs[:6]))
        return ''.join(bits)

    def _box_style(n: str) -> Tuple[str, str, float, float]:
        """fill, border, borderwidth, opacity for a species rectangle."""
        if hl and n == hl:
            return '#ffd24d', '#8a6d00', 3.0, 1.0
        if hl and n not in related:
            return '#f3f3f3', '#d0d0d0', 1.0, 0.35
        if n == focal_species:
            return '#7ec0ee', '#111111', 3.0, 1.0
        if _is_ice_like(n):
            return '#f6efe8', '#8c564b', 1.6, 1.0
        fill = plot_bg if plot_bg not in ('rgba(0,0,0,0)', None) else 'white'
        return fill, font, 1.5, 1.0

    # When highlighting, omit unrelated links entirely (avoids orphan stubs).
    draw_edges = [
        e for e in edges
        if hl is None or e['source'] == hl or e['target'] == hl
    ]
    # Per-face ports: crowded sides overflow onto top/bottom so tips do not stack.
    face_ports, exit_faces, enter_faces = _assign_face_ports(
        draw_edges, pos, node_half)
    ordered = sorted(
        draw_edges,
        key=lambda e: (e.get('partner', ''), e['source'], e['target']),
    )
    label_slots: Set[Tuple[int, int]] = set()
    min_clear = 0.16

    for e in ordered:
        if e['source'] not in pos or e['target'] not in pos:
            continue
        pclass = e.get('partner') or 'other'
        _leg, color, dash = PARTNER_STYLES.get(pclass, PARTNER_STYLES['other'])
        x0, y0 = pos[e['source']]
        x1, y1 = pos[e['target']]
        ekey = _edge_key(e)
        port_src = face_ports.get((*ekey, 'out'), 0.0)
        port_dst = face_ports.get((*ekey, 'in'), 0.0)
        src_face = exit_faces.get(ekey)
        dst_face = enter_faces.get(ekey)
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
        rev_n = pair_totals.get((e['target'], e['source']), 0)
        n_undirected = n_pair + rev_n
        aligned = (abs(y0 - y1) < 0.12 * max(row_gap, 1.0)
                   or abs(x0 - x1) < 0.12 * max(col_gap, 1.0))
        skip_layers = abs(r0 - r1) > 1
        use_curve = _should_use_curve(
            n_pair, n_lanes, n_edges, n_undirected=n_undirected,
            aligned=aligned, skip_layers=skip_layers,
        )
        if n_pair >= 2:
            bend = _curve_bend(pi, n_pair, col_gap, strong=True)
        elif rev_n > 0:
            bend = _curve_bend(0, max(2, n_undirected), col_gap, strong=True)
            bend = abs(bend) * (
                1.0 if (e['source'], e['target']) < (e['target'], e['source']) else -1.0
            )
        elif skip_layers:
            bend = _curve_bend(li, max(n_lanes, 2), col_gap, strong=True)
            bend = (abs(bend) + 0.35 * abs(r0 - r1)) * (1.0 if y0 >= y1 else -1.0)
        else:
            bend = _curve_bend(li, max(n_lanes, 2), col_gap, strong=False)
        lane = _lane_offset(li, n_lanes, col_gap)
        route_kw = dict(
            box_half_src=node_half[e['source']],
            box_half_dst=node_half[e['target']],
            port_src=port_src, port_dst=port_dst,
            obstacles=obstacles,
            skip={e['source'], e['target']},
            col_gap=col_gap,
            n_samples=28 if n_edges < 90 else 20,
            src_face=src_face, dst_face=dst_face,
        )
        pts, style = _route_avoiding_nodes(
            x0, y0, x1, y1,
            lane=lane, bend=bend, prefer_curve=use_curve, **route_kw,
        )
        if used_routes:
            best_c = _polyline_clearance(pts, used_routes)
            base_len = _polyline_length(pts)
            if best_c < min_clear:
                for extra in (0.28, -0.28, 0.50, -0.50, 0.78, -0.78, 1.05, -1.05):
                    alt, alt_style = _route_avoiding_nodes(
                        x0, y0, x1, y1,
                        lane=lane + 0.45 * extra, bend=bend + extra,
                        prefer_curve=True, **route_kw,
                    )
                    if _polyline_length(alt) > max(base_len * 1.85, base_len + 1.3):
                        continue
                    c = _polyline_clearance(alt, used_routes)
                    if c > best_c:
                        pts, style, best_c = alt, alt_style, c
                    if best_c >= min_clear:
                        break
        used_routes.append(pts)
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
            tip_x, tip_y = pts[-1]
            dx, dy = tip_x - ax0, tip_y - ay0
            length = float(np.hypot(dx, dy))
            if length > 1e-9:
                ux, uy = dx / length, dy / length
                line_gap = min(0.028, 0.20 * length)
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
        if pts:
            route_xs.append(pts[0][0])
            route_ys.append(pts[0][1])
        label = _partner_text(e)
        if label and pts:
            mx = my = None
            chosen_slot = None
            for frac in (0.50, 0.42, 0.58, 0.36, 0.64, 0.30, 0.70):
                tx, ty = _label_anchor(pts, offset=0.0, frac=frac)
                blocked = False
                for _name, (cx, cy, half) in obstacles.items():
                    hw, hh = _as_wh(half)
                    if abs(tx - cx) <= hw + 0.04 and abs(ty - cy) <= hh + 0.04:
                        blocked = True
                        break
                slot = (int(round(tx * 12)), int(round(ty * 12)))
                if not blocked and slot not in label_slots:
                    mx, my = tx, ty
                    chosen_slot = slot
                    break
            if mx is None:
                mx, my = _label_anchor(pts, offset=0.0, frac=0.50)
                chosen_slot = (int(round(mx * 12)), int(round(my * 12)))
            label_slots.add(chosen_slot)
            annotations.append(dict(
                x=mx, y=my, xref='x', yref='y',
                text=label, showarrow=False,
                font=dict(size=mid_size, color=color),
                bgcolor=paper if paper else 'white',
                borderpad=2,
                xanchor='center', yanchor='middle',
            ))
        hover = str(e.get('raw') or f"{e['source']} → {e['target']}")
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode='lines',
            line=dict(color=color, width=width, dash=dash),
            opacity=1.0,
            hoverinfo='text', hovertext=hover,
            name=_leg, showlegend=False,
        ))

    # Keep routing mix available for tests / status (not shown in UI).
    status['n_curve'] = n_curve
    status['n_ortho'] = n_ortho
    status['n_detour'] = n_detour

    # Rectangles in data coordinates so arrows meet the box border exactly.
    # Invisible square markers sit underneath for click / hover.
    click_names = [n for n in nodes if n in pos]
    for n in click_names:
        x, y = pos[n]
        hw, hh = _as_wh(node_half[n])
        fill, border, bwidth, op = _box_style(n)
        shapes.append(dict(
            type='rect', xref='x', yref='y',
            x0=x - hw, x1=x + hw, y0=y - hh, y1=y + hh,
            line=dict(color=border, width=bwidth),
            fillcolor=fill,
            opacity=op,
            layer='above',
        ))
        txt_color = '#c0c0c0' if (hl and n not in related) else '#111111'
        annotations.append(dict(
            x=x, y=y, xref='x', yref='y',
            text=_format_node_label(n),
            showarrow=False,
            font=dict(size=node_font.get(n, label_size), color=txt_color),
            xanchor='center', yanchor='middle',
        ))

    if click_names:
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in click_names],
            y=[pos[n][1] for n in click_names],
            mode='markers',
            text=[_format_node_label(n) for n in click_names],
            customdata=click_names,
            marker=dict(
                symbol='square',
                size=[node_size[n] for n in click_names],
                color='rgba(255,255,255,0.01)',
                line=dict(width=0),
            ),
            hovertext=[_species_hover(n) for n in click_names],
            hoverinfo='text',
            showlegend=False, name='species',
            cliponaxis=True,
        ))

    xs_all = [p[0] for p in pos.values()] + route_xs
    ys_all = [p[1] for p in pos.values()] + route_ys
    if node_half:
        half_pad = max(max(_as_wh(h)) for h in node_half.values())
    else:
        half_pad = 0.20
    marker_pad = max(0.50, half_pad * 2.4)
    if xs_all:
        xs_all.extend([min(xs_all) - marker_pad, max(xs_all) + marker_pad])
    if ys_all:
        ys_all.extend([min(ys_all) - marker_pad, max(ys_all) + marker_pad])

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
        margin=dict(l=28, r=28, t=56, b=24),
        font=dict(family='Arial, sans-serif', size=14, color=font),
        showlegend=False,
        shapes=shapes,
        xaxis=dict(visible=False, showgrid=False, zeroline=False,
                   range=_pad_range(xs_all, pad=0.04, abs_pad=0.12),
                   automargin=False, fixedrange=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False,
                   range=_pad_range(ys_all, pad=0.05, abs_pad=0.15),
                   automargin=False, fixedrange=False),
        annotations=annotations,
        uirevision=(f'pathnet|{focal_species}|{up}|{down}|'
                    f'iso={int(bool(include_isotopes))}|ice={int(bool(include_ice))}|'
                    f'pt={partner_pt}|sp={species_pt}|hl={hl or ""}'),
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
    focal_formation: Optional[Sequence[str]] = None,
    focal_destruction: Optional[Sequence[str]] = None,
    hidden_species: Optional[Sequence[str]] = None,
    partner_label_size: Optional[float] = None,
    species_label_size: Optional[float] = None,
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
        focal_formation=focal_formation,
        focal_destruction=focal_destruction,
        hidden_species=hidden_species,
        partner_label_size=partner_label_size,
        species_label_size=species_label_size,
    )


def status_message(status: dict) -> str:
    """Human-readable status line for the Chemistry tab."""
    if not status or not status.get('ok'):
        return ('Load a chemistry grid and select a species — the pathway network '
                'walks formation routes into the species and destruction routes out. '
                'Reaction partners are labelled on the arrows. '
                'Hover a link for its full reaction; click a species to highlight its links.')
    extras = []
    if status.get('include_isotopes'):
        extras.append('isotopes on')
    if status.get('include_ice'):
        extras.append('ice on')
    if status.get('highlight'):
        extras.append(f'highlight {status["highlight"]}')
    hidden = status.get('hidden') or []
    if hidden:
        extras.append(f'{len(hidden)} hidden')
    extra = f'; {", ".join(extras)}' if extras else ''
    return (
        f'Pathway network for {status.get("species")}: '
        f'{status.get("n_nodes", 0)} species, {status.get("n_edges", 0)} links '
        f'(upstream {status.get("upstream_depth")}, '
        f'downstream {status.get("downstream_depth")}{extra}). '
        f'Click a box to isolate its links; Hide selected removes it from the '
        f'figure (the selected species cannot be hidden). '
        f'Partners are written on the arrows. Formation/destruction of the '
        f'selected species match the contribution tables; ice / grain species '
        f'appear when enabled.'
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
    if not labels_by_species:
        return []
    key = species if species in labels_by_species else _match_known_species(
        species, labels_by_species.keys())
    entry = labels_by_species.get(key) or labels_by_species.get(species) or {}
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