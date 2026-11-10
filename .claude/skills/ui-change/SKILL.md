---
name: ui-change
description: Change or review the look or interaction of the PRISMA Dash app (tab layout, navigation, controls, theme/dark mode, Plotly figure styling, colour scales, legends, export). Use whenever the request is about how something looks or is used rather than what is computed.
argument-hint: "<what to change in the UI>"
---

Implement this UI change following the CLAUDE.md "UI change" workflow.

1. **Locate**: grep `app.py` for the tab / component id and the matching rules in
   `assets/theme.css`; note the callbacks that read or write those ids.
2. **Design + implement** with the `ui-designer` agent: theme tokens from `theme.css`,
   fonts from `plot_style.py`, existing classes/components, both light and dark themes.
3. **If the change touches what a figure shows** (axis ranges, colour-scale limits, units,
   which values are drawn), check with `ism-physicist` or `radio-astronomer` that nothing
   scientifically relevant is hidden or mislabelled.
4. **Verify**: `python -m pytest tests/` (at least `test_number_inputs.py` and any test for
   the touched tab), then run `python app.py` and check the tab in both themes
   (screenshot if a browser tool is available).
5. Update `README.md` if the tab layout or controls changed.

Report: what changed (`file:line`), screenshot or what to check visually, test result.

The change: $ARGUMENTS
