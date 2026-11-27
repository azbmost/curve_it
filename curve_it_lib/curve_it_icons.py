"""Curve It-only additions to the shared AZBMOST icon set.

azbmost_ui.py and azbmost_icons.py are identical copies shared by every
AZBMOST module, so a tool that exists only in Curve It registers its glyph and
one-line purpose here instead of editing them, as bnp_na_icons.py does for
BNP_NA.

register() is handed the azbmost_ui module its caller actually uses.  A tool
run as a script imports azbmost_ui flat rather than from the package, and
registering into the other copy would change nothing on screen.
"""
from __future__ import annotations

import math

READ_ROPELENGTHDATA = "read_ropelengthdata"
PURPOSES = {
    READ_ROPELENGTHDATA: "Read tight knots and links from Cantarella's ropelength data set.",
}


def _tight_trefoil(icons):
    """A trefoil drawn as a thick rope with its centreline inside: the tight
    knot of the data set, against KnotPlot to XYZ's thin one."""
    def glyph(p, k) -> None:
        n = 180
        pts = [icons.trefoil(2 * math.pi * i / n, 512, 540, 92) for i in range(n + 1)]
        icons._knot_glyph(p, k, pts, 112, 12, k.a)
        icons._knot_glyph(p, k, pts, 30, 53, k.c)
    return glyph


def register(ui) -> None:
    """Add this module's glyphs and purposes to `ui`, never replacing one."""
    icons = getattr(ui, "_icons", None)
    tools = getattr(icons, "TOOLS", None)
    if isinstance(tools, dict):
        tools.setdefault(READ_ROPELENGTHDATA, (_tight_trefoil(icons), "red"))
    info = getattr(ui, "TOOL_INFO", None)
    if isinstance(info, dict):
        for key, purpose in PURPOSES.items():
            info.setdefault(key, ("curve_it", purpose))
