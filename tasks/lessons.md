# Lessons learned (agent + human)

Append short, actionable patterns after corrections.
Format: date — mistake — rule that prevents it.

## Entries

- 2026-11 — Tab CSS targeted `#main-tabs > div:first-child`, which in Dash 4 is the first
  tab, not the tab bar, so the tab groups fell back to Dash defaults. Check the rendered
  DOM before writing Dash CSS selectors.
- 2026-11 — Dash 4.2 `dcc.Input(type='number')` with `min` but no `max` blanks the box.
  Always give both (`NUM_INPUT_MAX`); `tests/test_number_inputs.py` guards it.

- 2026-11-11 — Added the H2 + CR → H2+ edge to the H3+ network but only checked the edge list, not node placement; H2 stayed in the downstream column so the arrow ran backwards and the user still could not see the path. — For network-graph fixes, verify the rendered layout (node rank / x position), not just that the edge exists.
- 2026-11-13 — Explained the 3D-fronts C⁺/C 1/e threshold (Gaches et al., arXiv:2608.04116) by guessing at the paper's reasoning without reading it. — Before basing an idea, definition or threshold on a published work, or explaining one already cited in the code, read the relevant sections of the paper and point to section/equation; if it can't be accessed, say so instead of guessing.
