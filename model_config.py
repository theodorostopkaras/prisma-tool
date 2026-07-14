"""
Scan PDR model JSON configs under ``<grid_dir>/Models/**/config_files/*.json``
and summarize shared vs grid-varying simulation parameters.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import grid_naming as gn
from dash import html

# Keys that typically define the grid (shown first when they vary).
_GRID_PARAM_PATHS = (
    'physical_params.surface_density',
    'physical_params.radiation_field_strength',
    'physical_params.cosmic_ray_rate',
    'physical_params.cloud_radius',
    'physical_params.core_radius',
    'physical_params.metallicity',
    'physical_params.cr_att_coeff',
    'physical_params.stopping_rate_coeff',
    'physical_params.atten_mode_choice',
)

_SECTION_TITLES = {
    'physical_params': 'Physical parameters',
    'preshielding': 'Pre-shielding',
    'radiative_transfer': 'Radiative transfer',
    'fuv_field': 'FUV field',
    'temperature': 'Temperature',
    'heating': 'Heating',
    'dust': 'Dust',
    'h2_formation': 'H2 formation',
    'h2': 'H2',
    'surface_chemistry': 'Surface chemistry',
    'alfven_heating': 'Alfven heating',
    'numerical_params': 'Numerical parameters',
    'element_abundances': 'Element abundances',
    'species': 'Species network',
}

_SPECIES_KEY = 'species'
_SPECIES_TABLE_COLS = 4


def find_models_root(grid_directory: str) -> Optional[str]:
    """Locate ``Models/`` relative to the loaded HDF5 grid directory."""
    grid_directory = os.path.abspath(os.path.expanduser((grid_directory or '').strip()))
    if not grid_directory:
        return None
    base = os.path.basename(grid_directory.rstrip(os.sep))
    parent = os.path.dirname(grid_directory)
    candidates = [
        grid_directory,
        os.path.join(grid_directory, 'Models'),
        os.path.join(parent, 'Models'),
        os.path.join(parent, 'Models', base),
    ]
    for path in candidates:
        if not os.path.isdir(path):
            continue
        if glob.glob(os.path.join(path, '**', 'config_files', '*.json'), recursive=True):
            return path
    return None


def parse_model_folder_tokens(folder_name: str, n_params: int = 6) -> Optional[Tuple[int, ...]]:
    """Parse ``Model100_DD_MM_FF_ZZ_CC[_AA]`` folder names into slider tokens."""
    tokens = gn.parse_model_tokens_from_stem(folder_name)
    if tokens is None or len(tokens) != n_params:
        return None
    return tokens


def _list_signature(value: list) -> tuple:
    if len(value) > 25:
        payload = json.dumps(value, sort_keys=True)
        return ('__species_list__', len(value), hash(payload))
    return ('__list__', tuple(value))


def flatten_config(data: dict, prefix: str = '') -> Dict[str, Any]:
    """Flatten nested JSON; skip ``*_comment`` keys."""
    out: Dict[str, Any] = {}
    if not isinstance(data, dict):
        return out
    for key, val in data.items():
        if key.endswith('_comment') or key in ('config_comment',):
            continue
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(val, dict):
            out.update(flatten_config(val, path))
        elif isinstance(val, list):
            out[path] = _list_signature(val)
            out[f'{path}.__raw__'] = val
        else:
            out[path] = val
    return out


def comment_label(text: str) -> str:
    """Short label from comment text (before the last ``:``, INP variable suffix)."""
    text = (text or '').strip()
    if not text:
        return ''
    if ':' in text:
        return text.rsplit(':', 1)[0].strip()
    return text


def extract_comments(data: dict, prefix: str = '') -> Dict[str, str]:
    """Map flattened paths to full ``*_comment`` text."""
    comments: Dict[str, str] = {}
    if not isinstance(data, dict):
        return comments
    for key, val in data.items():
        path = f'{prefix}.{key}' if prefix else key
        if key.endswith('_comment') and isinstance(val, str):
            base = path[:-len('_comment')]
            text = val.strip()
            if text:
                comments[base] = text
        elif isinstance(val, dict):
            comments.update(extract_comments(val, path))
    return comments


def extract_comment_labels(data: dict, prefix: str = '') -> Dict[str, str]:
    """Map flattened paths to short labels (text before the last ``:`` in comments)."""
    labels: Dict[str, str] = {}
    for path, text in extract_comments(data, prefix).items():
        labels[path] = comment_label(text) or path
    return labels


def field_label(path: str, labels: Dict[str, str], comments: Dict[str, str]) -> str:
    """Human-readable parameter name."""
    if path in labels and labels[path] and labels[path] != path:
        return labels[path]
    if path in comments:
        lab = comment_label(comments[path])
        if lab:
            return lab
    return path.replace('_', ' ').replace('.', ' / ')


_ELEMENT_REF_PATH = 'element_abundances.reference'
_FILE_FIELD_COMMENTS = {
    'chemical_network_file': 'Path to chemical reaction network rate file.',
    'output_file': 'Main model output file name.',
}
_EXTRA_FIELD_COMMENTS = {
    'numerical_params.iteration_convergence_tolerance':
        'Iteration convergence tolerance for the chemical / thermal solver.',
    'numerical_params.ode_absolute_tolerance':
        'Absolute tolerance for the ODE time-stepping solver.',
    'numerical_params.ode_relative_tolerance':
        'Relative tolerance for the ODE time-stepping solver.',
    'numerical_params.step_size_co_dissociated':
        'Spatial step size (cm) when CO is dissociated.',
    'numerical_params.step_size_co_not_dissociated':
        'Spatial step size (cm) when CO is not dissociated.',
}


def resolve_field_comment(
    path: str,
    comments: Dict[str, str],
    element_ref: str = '',
) -> str:
    """Full comment text for a config field, with sensible fallbacks."""
    if path in comments and comments[path]:
        return comments[path]
    if path == _ELEMENT_REF_PATH:
        return 'Bibliographic reference for elemental abundances.'
    if path.startswith('element_abundances.'):
        elem = path.rsplit('.', 1)[-1].upper()
        if element_ref:
            return f'Abundance of {elem} relative to H (log). Reference scale: {element_ref}'
        return f'Abundance of {elem} relative to H (log).'
    if path in _FILE_FIELD_COMMENTS:
        return _FILE_FIELD_COMMENTS[path]
    if path in _EXTRA_FIELD_COMMENTS:
        return _EXTRA_FIELD_COMMENTS[path]
    return ''


def format_config_value(value: Any, raw_lists: Optional[dict] = None, path: str = '') -> str:
    """Pretty-print a config value for display."""
    if isinstance(value, tuple) and value and value[0] == '__species_list__':
        n = value[1]
        raw = (raw_lists or {}).get(f'{path}.__raw__') or (raw_lists or {}).get(path)
        if raw and isinstance(raw, list):
            preview = ', '.join(str(x) for x in raw[:8])
            if len(raw) > 8:
                preview += ', ...'
            return f'{n} species ({preview})'
        return f'{n} species'
    if isinstance(value, tuple) and value and value[0] == '__list__':
        items = value[1]
        if len(items) > 8:
            head = ', '.join(format_config_value(x) for x in items[:6])
            return f'[{head}, ...] ({len(items)} items)'
        return '[' + ', '.join(format_config_value(x) for x in items) + ']'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value == 0.0:
            return '0'
        av = abs(value)
        if av >= 1e4 or av < 1e-3:
            return f'{value:.4g}'
        return f'{value:g}'
    return str(value)


def collect_species_info(configs: Dict[str, dict]) -> dict:
    """Summarise the ``species`` list across all model configs."""
    lists: List[List[str]] = []
    for data in configs.values():
        sp = data.get(_SPECIES_KEY)
        if isinstance(sp, list) and sp:
            lists.append([str(s) for s in sp])
    if not lists:
        return dict(species=[], consistent=True, n_species=0, variants=[])

    sigs: Dict[tuple, int] = {}
    for sp in lists:
        key = tuple(sp)
        sigs[key] = sigs.get(key, 0) + 1
    consistent = len(sigs) == 1
    canonical = list(max(sigs, key=lambda k: sigs[k]))
    variants = []
    if not consistent:
        variants = [
            dict(species=list(k), count=c, n_species=len(k))
            for k, c in sorted(sigs.items(), key=lambda x: -x[1])
        ]
    return dict(
        species=canonical,
        consistent=consistent,
        n_species=len(canonical),
        variants=variants,
    )


def scan_model_configs(grid_directory: str) -> dict:
    """
    Scan all ``pdr_config_*.json`` under ``Models/**/config_files/``.

    Returns a summary dict with shared / varying parameters and per-model lookups.
    """
    models_root = find_models_root(grid_directory)
    empty = dict(
        models_root=models_root,
        n_configs=0,
        n_skipped=0,
        config_comment='',
        labels={},
        comments={},
        varying=[],
        shared_sections=[],
        by_tokens={},
        configs={},
        species_info=dict(species=[], consistent=True, n_species=0, variants=[]),
        error=None,
    )
    if models_root is None:
        empty['error'] = (
            'No Models/ directory found next to the grid path '
            '(expected ``<grid_dir>/Models/<Model...>/config_files/*.json``).'
        )
        return empty

    pattern = os.path.join(models_root, '**', 'config_files', '*.json')
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        empty['error'] = f'No JSON config files under {models_root!r}.'
        return empty

    configs: Dict[str, dict] = {}
    flats: Dict[str, Dict[str, Any]] = {}
    raw_lists: Dict[str, Dict[str, Any]] = {}
    by_tokens: Dict[Tuple[int, ...], str] = {}
    labels: Dict[str, str] = {}
    comments: Dict[str, str] = {}
    skipped = 0

    for path in paths:
        try:
            with open(path, encoding='utf-8') as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        toks = parse_model_folder_tokens(folder)
        if toks is not None:
            by_tokens[toks] = path
        configs[path] = data
        flat = flatten_config(data)
        flats[path] = flat
        for k, v in flat.items():
            if k.endswith('.__raw__'):
                raw_lists.setdefault(path, {})[k] = v
        if not labels:
            comments = extract_comments(data)
            labels = extract_comment_labels(data)
            empty['config_comment'] = str(data.get('config_comment', '')).strip()

    if not flats:
        empty['n_skipped'] = skipped
        empty['error'] = 'Found JSON files but none could be parsed.'
        return empty

    all_paths = sorted(set(k for f in flats.values() for k in f if not k.endswith('.__raw__')))
    varying: List[dict] = []
    shared_by_section: Dict[str, List[dict]] = {}

    ref_path = paths[0]
    ref_flat = flats[ref_path]
    ref_data = configs[ref_path]
    element_ref = ''
    ea = ref_data.get('element_abundances')
    if isinstance(ea, dict):
        element_ref = str(ea.get('reference', '')).strip()

    for path_key in all_paths:
        if path_key == _SPECIES_KEY:
            continue
        values = []
        for fpath, flat in flats.items():
            if path_key in flat:
                values.append((fpath, flat[path_key]))
        if not values:
            continue
        unique = {}
        for fpath, val in values:
            unique.setdefault(val, []).append(fpath)
        if len(unique) > 1:
            lab = field_label(path_key, labels, comments)
            formatted = sorted(
                format_config_value(v, raw_lists.get(paths[0], {}), path_key)
                for v in unique
            )
            varying.append(dict(
                path=path_key,
                label=lab,
                comment=resolve_field_comment(path_key, comments, element_ref),
                values=formatted,
                n_values=len(unique),
                n_models=len(values),
                is_grid=path_key in _GRID_PARAM_PATHS or path_key.startswith('physical_params.'),
            ))
        else:
            val = next(iter(unique))
            section = path_key.split('.')[0] if '.' in path_key else 'general'
            shared_by_section.setdefault(section, []).append(dict(
                path=path_key,
                label=field_label(path_key, labels, comments),
                comment=resolve_field_comment(path_key, comments, element_ref),
                value=format_config_value(val, raw_lists.get(ref_path, {}), path_key),
            ))

    def _varying_sort(item):
        try:
            pri = _GRID_PARAM_PATHS.index(item['path'])
        except ValueError:
            pri = 100 + len(item['path'])
        return (0 if item['is_grid'] else 1, pri, item['path'])

    varying.sort(key=_varying_sort)

    shared_sections = []
    for section in sorted(shared_by_section, key=lambda s: (
        list(_SECTION_TITLES).index(s) if s in _SECTION_TITLES else 999, s)):
        items = sorted(shared_by_section[section], key=lambda x: x['path'])
        shared_sections.append(dict(
            section=section,
            title=_SECTION_TITLES.get(section, section.replace('_', ' ').title()),
            items=items,
        ))

    species_info = collect_species_info(configs)
    species_info['chemical_network_file'] = str(
        ref_data.get('chemical_network_file', '')).strip()
    species_info['network_comment'] = comments.get('species', '') or (
        'Species included in the chemical network (from config ``species`` list).'
    )

    return dict(
        models_root=models_root,
        n_configs=len(configs),
        n_skipped=skipped,
        config_comment=empty['config_comment'],
        labels=labels,
        comments=comments,
        varying=varying,
        shared_sections=shared_sections,
        by_tokens=by_tokens,
        configs=configs,
        flats=flats,
        raw_lists=raw_lists,
        species_info=species_info,
        error=None,
    )


def config_for_tokens(summary: dict, tokens: Optional[Tuple[int, ...]]) -> Optional[dict]:
    """Return flattened config for the model matching ``tokens``, if any."""
    if not summary or tokens is None:
        return None
    path = summary.get('by_tokens', {}).get(tuple(tokens))
    if not path:
        return None
    flat = summary.get('flats', {}).get(path)
    raw = summary.get('raw_lists', {}).get(path, {})
    labels = summary.get('labels', {})
    comments = summary.get('comments', {})
    if not flat:
        return None
    return dict(path=path, flat=flat, raw=raw, labels=labels, comments=comments)


_BULLET = {'margin': '4px 0', 'paddingLeft': '4px', 'lineHeight': '1.45'}
_SECTION_HDR = {'fontSize': '15px', 'fontWeight': '700', 'color': '#1a1a2e',
                'margin': '18px 0 8px', 'borderBottom': '1px solid #dde', 'paddingBottom': '4px'}
_NOTE = {'fontSize': '12px', 'color': '#666', 'margin': '0 0 10px'}
_CARD = {'padding': '12px 16px', 'backgroundColor': '#f8faf8', 'borderRadius': '8px',
         'border': '1px solid #d8e8d8', 'marginBottom': '12px'}
_TABLE = {'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '12px'}
_TH = {'padding': '6px 10px', 'borderBottom': '2px solid #ccd', 'textAlign': 'left',
       'backgroundColor': '#eef2ee', 'fontWeight': '600', 'fontSize': '11px'}
_TD_IDX = {'padding': '4px 8px', 'borderBottom': '1px solid #e8ece8', 'color': '#666',
           'textAlign': 'right', 'width': '36px', 'fontSize': '11px'}
_TD_SP = {'padding': '4px 10px', 'borderBottom': '1px solid #e8ece8',
          'fontFamily': 'Consolas, monospace', 'fontSize': '12px', 'whiteSpace': 'nowrap'}
_TD_VAL = {'padding': '4px 10px', 'borderBottom': '1px solid #e8ece8', 'fontSize': '12px',
           'verticalAlign': 'top'}
_TD_LAB = {'padding': '4px 10px', 'borderBottom': '1px solid #e8ece8', 'fontSize': '12px',
           'fontWeight': '600', 'verticalAlign': 'top', 'whiteSpace': 'normal'}
_TD_COMMENT = {'padding': '4px 10px', 'borderBottom': '1px solid #e8ece8', 'fontSize': '11px',
               'color': '#555', 'lineHeight': '1.45', 'verticalAlign': 'top', 'whiteSpace': 'normal'}


def build_fields_table(rows: List[dict], value_header: str = 'Value'):
    """Three-column table: parameter name, value(s), JSON comment."""
    if not rows:
        return html.P('No parameters.', style=_NOTE)
    body = []
    for row in rows:
        body.append(html.Tr([
            html.Td(row.get('label', ''), style=_TD_LAB),
            html.Td(row.get('value', ''), style=_TD_VAL),
            html.Td(row.get('comment') or '', style=_TD_COMMENT),
        ]))
    return html.Table([
        html.Thead(html.Tr([
            html.Th('Parameter', style=_TH),
            html.Th(value_header, style=_TH),
            html.Th('Comment', style=_TH),
        ])),
        html.Tbody(body),
    ], style=_TABLE)


def build_species_table(species: List[str], ncol: int = _SPECIES_TABLE_COLS):
    """Multi-column species table (#, name) x ``ncol``."""
    if not species:
        return html.P('No species list in config files.', style=_NOTE)
    n = len(species)
    nrows = (n + ncol - 1) // ncol
    header_cells = []
    for _ in range(ncol):
        header_cells.extend([
            html.Th('#', style={**_TH, 'textAlign': 'right', 'width': '36px'}),
            html.Th('Species', style=_TH),
        ])
    body = []
    for r in range(nrows):
        cells = []
        for c in range(ncol):
            idx = r + c * nrows
            if idx < n:
                cells.extend([
                    html.Td(str(idx + 1), style=_TD_IDX),
                    html.Td(species[idx], style=_TD_SP),
                ])
            else:
                cells.extend([
                    html.Td('', style=_TD_IDX),
                    html.Td('', style=_TD_SP),
                ])
        body.append(html.Tr(cells))
    return html.Table([
        html.Thead(html.Tr(header_cells)),
        html.Tbody(body),
    ], style=_TABLE)


def build_species_section(species_info: dict):
    """Species network block for the Model setup tab."""
    species = species_info.get('species') or []
    if not species:
        return None
    notes = []
    n = species_info.get('n_species', len(species))
    if species_info.get('consistent', True):
        notes.append(f'{n} species - identical in all model configs.')
    else:
        notes.append(
            html.Span([
                f'{n} species shown (most common network). ',
                html.Strong('Warning: '),
                f'{len(species_info.get("variants", []))} distinct species lists across configs.',
            ])
        )
    net_file = species_info.get('chemical_network_file') or ''
    if net_file:
        notes.append(html.Span([
            html.Strong('Network file: '),
            html.Code(net_file, style={'fontSize': '11px'}),
        ]))
    net_comment = species_info.get('network_comment') or ''
    if net_comment:
        notes.append(net_comment)
    children = [
        html.H3('Chemical species network', style=_SECTION_HDR),
        html.P(notes, style=_NOTE),
        html.Div(build_species_table(species), style={
            'maxHeight': '420px', 'overflowY': 'auto', 'overflowX': 'auto',
            'border': '1px solid #e0e0e0', 'borderRadius': '6px',
        }),
    ]
    variants = species_info.get('variants') or []
    if len(variants) > 1:
        variant_rows = []
        for v in variants[:8]:
            variant_rows.append(html.Tr([
                html.Td(str(v['count']), style=_TD_IDX),
                html.Td(str(v['n_species']), style=_TD_SP),
                html.Td(', '.join(v['species'][:12]) + (
                    ', ...' if len(v['species']) > 12 else ''),
                    style={**_TD_SP, 'whiteSpace': 'normal'}),
            ]))
        children.append(html.Div([
            html.P('Distinct species lists found:', style={**_NOTE, 'marginTop': '10px'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th('Models', style=_TH),
                    html.Th('N species', style=_TH),
                    html.Th('Preview', style=_TH),
                ])),
                html.Tbody(variant_rows),
            ], style=_TABLE),
        ]))
    return html.Div(children, style={**_CARD, 'backgroundColor': '#faf8fc',
                                       'borderColor': '#d8d0e8'})


def build_model_setup_panel(summary: Optional[dict], tokens: Optional[Tuple[int, ...]] = None):
    """Dash layout for the Model setup tab."""
    if not summary:
        return html.P('Load a main grid directory on the Load tab to scan model configs.',
                      style=_NOTE)

    if summary.get('error'):
        return html.Div([
            html.P(summary['error'], style={'color': '#a33', 'fontSize': '13px'}),
            html.P('Configs are expected at '
                   '``<grid_dir>/Models/Model.../config_files/pdr_config_....json``.',
                   style=_NOTE),
        ])

    header_bits = [
        html.P([
            html.Strong(f'{summary["n_configs"]} model configs'),
            f' scanned under ',
            html.Code(summary['models_root'], style={'fontSize': '11px'}),
        ], style={'fontSize': '13px', 'margin': '0 0 6px'}),
    ]
    if summary.get('config_comment'):
        header_bits.append(html.P(summary['config_comment'], style=_NOTE))
    if summary.get('n_skipped'):
        header_bits.append(html.P(f'{summary["n_skipped"]} file(s) skipped (parse errors).',
                                  style={'color': '#888', 'fontSize': '11px'}))

    children = [html.Div(header_bits, style=_CARD)]

    cur = config_for_tokens(summary, tokens)
    comments = summary.get('comments') or {}
    element_ref = ''
    if cur:
        element_ref = str(cur['flat'].get(_ELEMENT_REF_PATH) or '').strip()
    if cur:
        folder = os.path.basename(os.path.dirname(os.path.dirname(cur['path'])))
        grid_rows = []
        for path in _GRID_PARAM_PATHS:
            if path in cur['flat']:
                grid_rows.append(dict(
                    label=field_label(path, cur['labels'], cur['comments']),
                    value=format_config_value(cur['flat'][path], cur['raw'], path),
                    comment=resolve_field_comment(
                        path, {**comments, **(cur['comments'] or {})}, element_ref),
                ))
        children.append(html.Div([
            html.H3(f'Current model: {folder}', style={**_SECTION_HDR, 'marginTop': '0'}),
            build_fields_table(grid_rows) if grid_rows else html.P(
                'No grid parameters in config.', style=_NOTE),
        ], style={**_CARD, 'backgroundColor': '#f0f4ff', 'borderColor': '#c5d0ef'}))

    sp_block = build_species_section(summary.get('species_info') or {})
    if sp_block is not None:
        children.append(sp_block)

    grid_varying = [v for v in summary.get('varying', []) if v.get('is_grid')]
    other_varying = [v for v in summary.get('varying', []) if not v.get('is_grid')]

    if grid_varying:
        rows = [
            dict(
                label=row['label'],
                value=', '.join(row['values']) + (
                    f'  ({row["n_values"]} distinct / {row["n_models"]} models)'),
                comment=row.get('comment') or '',
            )
            for row in grid_varying
        ]
        children.append(html.Div([
            html.H3('Grid parameters (vary across models)', style=_SECTION_HDR),
            html.P('These initial / grid parameters differ between model folders.',
                   style=_NOTE),
            build_fields_table(rows, value_header='Values across grid'),
        ], style=_CARD))

    if other_varying:
        rows = [
            dict(
                label=row['label'],
                value=', '.join(row['values'][:8]) + (
                    ', ...' if len(row['values']) > 8 else ''),
                comment=row.get('comment') or '',
            )
            for row in other_varying[:60]
        ]
        children.append(html.Div([
            html.H3('Other varying parameters', style=_SECTION_HDR),
            build_fields_table(rows, value_header='Values across grid'),
            html.P(f'{len(other_varying)} parameter(s) differ across models.',
                     style=_NOTE) if len(other_varying) > 60 else None,
        ], style=_CARD))

    shared_blocks = []
    for sec in summary.get('shared_sections', []):
        if sec.get('section') == _SPECIES_KEY:
            continue
        rows = [
            dict(label=it['label'], value=it['value'], comment=it.get('comment') or '')
            for it in sec['items']
        ]
        shared_blocks.append(html.Details([
            html.Summary(f'{sec["title"]} ({len(rows)} parameters)',
                         style={'fontWeight': '600', 'cursor': 'pointer',
                                'padding': '6px 0', 'fontSize': '13px'}),
            html.Div(build_fields_table(rows), style={'marginTop': '8px'}),
        ], style={'marginBottom': '8px', 'borderBottom': '1px solid #eee', 'paddingBottom': '6px'}))

    if shared_blocks:
        children.append(html.Div([
            html.H3('Shared simulation setup (same in all models)', style=_SECTION_HDR),
            html.P('Expand a section for parameters identical in every config file. '
                   'Comments are taken from the JSON ``*_comment`` fields (PDRNEW.INP names).',
                   style=_NOTE),
            html.Div(shared_blocks),
        ], style=_CARD))

    return html.Div(children)
