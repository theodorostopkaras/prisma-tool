"""Native PRISMA app window: ``python desktop_window.py URL``.

Started by ``app.py`` in a child process. Exit code 0 means the user closed
the window; anything else (pywebview missing, GUI backend crash) makes
``app.py`` fall back to the browser.

Also holds ``install_desktop_entry`` for ``python app.py --install``; importing
this module does not load Qt (that happens only in ``main``).
"""
import ctypes
import os
import socket
import sys
import time
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ICON = os.path.join(HERE, 'assets', 'prisma_logo.png')
# X11 window class of the PRISMA window (Qt takes it from this script's name);
# the menu entry names it so the dock groups the open window under its icon.
WM_CLASS = os.path.basename(__file__)


def _load_webview():
    """Import pywebview, first fixing a conda library clash that crashes Qt.

    Conda Pythons ship libxkbcommon, but Qt would take libxkbcommon-x11 from the
    system; the mismatched pair segfaults Qt. Load conda's matching -x11 first.
    """
    xkb_x11 = os.path.join(sys.prefix, 'lib', 'libxkbcommon-x11.so.0')
    if sys.platform == 'linux' and os.path.exists(xkb_x11):
        ctypes.CDLL(xkb_x11, mode=ctypes.RTLD_GLOBAL)
    import webview
    return webview


def main(url):
    """Wait for the Dash server at ``url`` (up to 15 s), then show it in a window."""
    webview = _load_webview()
    u = urlparse(url)
    for _ in range(150):
        try:
            socket.create_connection((u.hostname, u.port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.1)
    webview.create_window('PRISMA', url, width=1600, height=1000)
    # Linux: use Qt directly (pywebview would try GTK first and print a traceback).
    webview.start(gui='qt' if sys.platform == 'linux' else None, icon=ICON)


def install_desktop_entry(python=sys.executable, app_py=os.path.join(HERE, 'app.py')):
    """Add PRISMA to the Linux applications menu (with its logo); return the file path.

    Writes ``~/.local/share/applications/prisma.desktop`` (freedesktop.org
    Desktop Entry spec). It launches this checkout with this Python, so re-run
    it after moving the PRISMA folder or switching Python environments.
    """
    apps = os.path.join(os.path.expanduser('~'), '.local', 'share', 'applications')
    os.makedirs(apps, exist_ok=True)
    path = os.path.join(apps, 'prisma.desktop')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(
            '[Desktop Entry]\n'
            'Type=Application\n'
            'Name=PRISMA\n'
            'Comment=PDR Results Interactive Simulation Mapping & Analysis\n'
            f'Exec="{python}" "{app_py}"\n'
            f'Path={os.path.dirname(app_py)}\n'
            f'Icon={ICON}\n'
            'Terminal=false\n'
            'Categories=Science;Astronomy;\n'
            f'StartupWMClass={WM_CLASS}\n'
        )
    return path


if __name__ == '__main__':
    main(sys.argv[1])
    # The window is closed. Skip Python's teardown: it deletes Qt objects in
    # arbitrary order and WebEngine warns "WebEnginePage still not deleted".
    sys.stdout.flush()
    os._exit(0)
