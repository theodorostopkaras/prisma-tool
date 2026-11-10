# Lessons learned (agent + human)

Append short, actionable patterns after corrections.
Format: date — mistake — rule that prevents it.

## Entries

- 2026-11 — Tab CSS targeted `#main-tabs > div:first-child`, which in Dash 4 is the first
  tab, not the tab bar, so the tab groups fell back to Dash defaults. Check the rendered
  DOM before writing Dash CSS selectors.
- 2026-11 — Dash 4.2 `dcc.Input(type='number')` with `min` but no `max` blanks the box.
  Always give both (`NUM_INPUT_MAX`); `tests/test_number_inputs.py` guards it.
