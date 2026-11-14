// Tells the PRISMA server this page is still open (see /_prisma_alive in app.py).
// Only used to stop a menu-launched PRISMA after its browser tab is closed.
setInterval(function () { navigator.sendBeacon('/_prisma_alive'); }, 20000);
