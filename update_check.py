"""PRISMA version and the startup check for a newer GitHub release.

Bump ``__version__`` when publishing a release; the git tag must match it
with a leading ``v`` (``__version__ = '1.1.0'`` -> tag ``v1.1.0``).
Only published releases count: drafts and pre-releases are ignored by
GitHub's ``releases/latest`` endpoint, so plain commits never notify users.
"""
import functools
import json
import re
import urllib.request

__version__ = '1.1.0'

REPO = 'theodorostopkaras/prisma-tool'
_LATEST_URL = f'https://api.github.com/repos/{REPO}/releases/latest'


def parse_version(text):
    """``'v1.10.2'`` -> ``(1, 10, 2)``; ``None`` if ``text`` is not X.Y.Z."""
    m = re.fullmatch(r'v?(\d+)\.(\d+)\.(\d+)', (text or '').strip())
    return tuple(int(x) for x in m.groups()) if m else None


def newer_release(release, current=__version__):
    """Return ``release`` (GitHub API dict) if it is newer than ``current``, else ``None``."""
    latest = parse_version((release or {}).get('tag_name'))
    now = parse_version(current)
    if latest is None or now is None or latest <= now:
        return None
    return release


@functools.lru_cache(maxsize=1)
def check_for_update(timeout=2.0):
    """Ask GitHub once per run for the latest release; return it if newer.

    Returns a dict with ``tag_name``, ``html_url`` and ``body`` (release notes),
    or ``None`` when up to date, offline, rate-limited or GitHub is slow --
    the update check must never get in the way of using PRISMA.
    """
    req = urllib.request.Request(
        _LATEST_URL, headers={'Accept': 'application/vnd.github+json',
                              'User-Agent': 'PRISMA-update-check'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            release = json.load(resp)
    except (OSError, ValueError):  # offline, HTTP 404 (no release yet), bad JSON
        return None
    hit = newer_release(release)
    if hit is None:
        return None
    return {k: hit.get(k) or '' for k in ('tag_name', 'html_url', 'body')}
