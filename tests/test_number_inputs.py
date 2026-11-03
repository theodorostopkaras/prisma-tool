"""Dash 4.2's Input stepper blanks the box (NaN) when ``min`` is set without ``max``.

Every ``dcc.Input(type='number')`` with a lower bound must therefore also carry
an upper bound (``NUM_INPUT_MAX`` in app.py).
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _number_inputs_missing_max(path):
    tree = ast.parse(path.read_text())
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'Input'
                and getattr(node.func.value, 'id', None) == 'dcc'):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        typ = kw.get('type')
        if not (isinstance(typ, ast.Constant) and typ.value == 'number'):
            continue
        has_min = 'min' in kw and not (isinstance(kw['min'], ast.Constant) and kw['min'].value is None)
        if has_min and 'max' not in kw:
            bad.append(f'{path.name}:{node.lineno}')
    return bad


def test_bounded_number_inputs_have_max():
    bad = []
    for path in ROOT.glob('*.py'):
        bad += _number_inputs_missing_max(path)
    assert not bad, f'number inputs with min but no max (Dash stepper bug): {bad}'
