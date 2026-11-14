"""Native PRISMA app window: ``python desktop_window.py URL``.

Started by ``app.py`` in a child process. Exit code 0 means the user closed
the window; anything else (pywebview missing, GUI backend crash) makes
``app.py`` fall back to the browser.
"""
import ctypes
import os
import socket
import sys
import time
from urllib.parse import urlparse

# Conda Pythons ship libxkbcommon, but Qt would take libxkbcommon-x11 from the
# system; the mismatched pair segfaults Qt. Load conda's matching -x11 first.
_XKB_X11 = os.path.join(sys.prefix, 'lib', 'libxkbcommon-x11.so.0')
if sys.platform == 'linux' and os.path.exists(_XKB_X11):
    ctypes.CDLL(_XKB_X11, mode=ctypes.RTLD_GLOBAL)

import webview  # noqa: E402  (must come after the preload above)


def main(url):
    """Wait for the Dash server at ``url`` (up to 15 s), then show it in a window."""
    u = urlparse(url)
    for _ in range(150):
        try:
            socket.create_connection((u.hostname, u.port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.1)
    webview.create_window('PRISMA', url, width=1600, height=1000)
    # Linux: use Qt directly (pywebview would try GTK first and print a traceback).
    webview.start(gui='qt' if sys.platform == 'linux' else None)


if __name__ == '__main__':
    main(sys.argv[1])
    # The window is closed. Skip Python's teardown: it deletes Qt objects in
    # arbitrary order and WebEngine warns "WebEnginePage still not deleted".
    sys.stdout.flush()
    os._exit(0)
