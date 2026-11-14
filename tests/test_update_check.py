"""Version comparison for the in-app update notice (no network)."""
from update_check import newer_release, parse_version


def test_parse_version():
    assert parse_version('v1.10.2') == (1, 10, 2)
    assert parse_version('1.0.0') == (1, 0, 0)
    assert parse_version('v1.0') is None
    assert parse_version(None) is None


def test_newer_release_compares_numerically():
    rel = {'tag_name': 'v1.10.0'}
    assert newer_release(rel, current='1.9.3') is rel  # 10 > 9, not a string compare
    assert newer_release(rel, current='1.10.0') is None  # same version
    assert newer_release(rel, current='2.0.0') is None  # running a newer build
    assert newer_release({'tag_name': 'nightly'}, current='1.0.0') is None
    assert newer_release(None, current='1.0.0') is None
