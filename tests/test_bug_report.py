from urllib.parse import urlparse, parse_qs

import app


def test_bug_issue_url_prefills_github_issue():
    u = app._bug_issue_url('Crash in map fit', 'It & broke #1', '1. load')
    assert u.startswith(app.BUG_REPORT_REPO + '/issues/new?')
    q = parse_qs(urlparse(u).query)
    assert q['title'] == ['Crash in map fit']
    assert q['labels'] == ['bug']
    assert 'It & broke #1' in q['body'][0]
    assert 'Dash' in q['body'][0]
