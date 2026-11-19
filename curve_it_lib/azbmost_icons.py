#!/usr/bin/env python3
"""AZBMOST icons: the app icons of the launcher and the three modules, and one
glyph icon per tool, drawn as vector artwork.

Every icon is drawn in a 1024-unit square through a small "painter", so the
same drawing is crisp on a Tk canvas (inside the windows, at any size) and
exact in SVG (rendered to the PNG files used for window and Dock icons).

Identical copies of this file live next to azbmost_ui.py in

    azbmost/azbmost_icons.py
    bnp_na/bnp_na_lib/azbmost_icons.py
    re_helix/re_helix_lib/azbmost_icons.py
    curve_it/curve_it_lib/azbmost_icons.py

Colours are the DiLiuLab palette (gr_colors V3.3) and blends between its
entries. App icons: a charcoal squircle (neutral T80 to T100) with
multi-colour artwork. Tool icons: a squircle in one palette hue (T80 to T100)
with a white or charcoal line glyph.

The artwork carries its own depth, so it reads the same on a Tk canvas (no
transparency, no blur): strands are lit tubes (a darker rim, a core set toward
the light, a highlight line) that darken as they turn away; a helix is drawn
back runs first, then base pairs, then front runs; a strand passing over
another cuts a gap in the colour of the surface under it (gradient and glow);
a knot is drawn whole and then its over-crossings again. Only the finishing
touches are SVG-only: the tile's soft shadow, the glow (baked into the
scanlines on a canvas), the bloom of bright strands, the sheen and the glyphs'
drop shadow.

Painters share one small API: background, glow, sheen, line (with polygon
arrowheads, identical in both), poly, circle, begin/end (an SVG effect group;
a no-op on a canvas) and `detail` (device pixels per art unit, so small canvas
icons use fewer, longer strokes).

    python3 azbmost_icons.py --svg OUT_DIR     write every icon as SVG
"""
from __future__ import annotations

import math
import re
import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ICONS_VERSION = "2.0.0"

LAB: Dict[str, Dict[str, str]] = {
    "red": {"T100": "#e64c4c", "T80": "#eb7070", "T60": "#f09494", "T40": "#f5b7b7", "T20": "#fadbdb"},
    "blue": {"T100": "#4c79e6", "T80": "#7094eb", "T60": "#94aff0", "T40": "#b7c9f5", "T20": "#dbe4fa"},
    "magenta": {"T100": "#e64cd2", "T80": "#eb70db", "T60": "#f094e4", "T40": "#f5b7ed", "T20": "#fadbf6"},
    "cyan": {"T100": "#4ce6cc", "T80": "#70ebd6", "T60": "#94f0e0", "T40": "#b7f5eb", "T20": "#dbfaf5"},
    "orange": {"T100": "#e69f4c", "T80": "#ebb270", "T60": "#f0c594", "T40": "#f5d9b7", "T20": "#faecdb"},
    "purple": {"T100": "#724ce6", "T80": "#8e70eb", "T60": "#aa94f0", "T40": "#c7b7f5", "T20": "#e3dbfa"},
    "green": {"T100": "#52e64c", "T80": "#75eb70", "T60": "#97f094", "T40": "#baf5b7", "T20": "#dcfadb"},
    "yellow": {"T100": "#d9e64c", "T80": "#e1eb70", "T60": "#e8f094", "T40": "#f0f5b7", "T20": "#f7fadb"},
    "mint": {"T100": "#4ce698", "T80": "#70ebad", "T60": "#94f0c1", "T40": "#b7f5d6", "T20": "#dbfaea"},
    "neutral": {"T100": "#000000", "T80": "#333333", "T60": "#666666", "T40": "#999999", "T20": "#cccccc"},
}


def lab(hue: str, tint: str = "T100") -> str:
    return LAB[hue][tint]


def _rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def mix(color_a: str, color_b: str, t: float) -> str:
    """Blend color_a toward color_b by t (0 keeps a, 1 gives b)."""
    ra, ga, ba = _rgb(color_a)
    rb, gb, bb = _rgb(color_b)
    t = max(0.0, min(1.0, float(t)))
    return "#%02x%02x%02x" % (round(ra + (rb - ra) * t), round(ga + (gb - ga) * t), round(ba + (bb - ba) * t))


def luminance(color: str) -> float:
    def lin(v: int) -> float:
        c = v / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = _rgb(color)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


APP_TOP = LAB["neutral"]["T80"]
APP_BOTTOM = LAB["neutral"]["T100"]
CHARCOAL = LAB["neutral"]["T80"]
WHITE = "#ffffff"
BLACK = "#000000"
# Hues light enough to need a charcoal glyph rather than a white one.
LIGHT_HUES = {"cyan", "green", "yellow", "mint", "orange"}

ART = 1024.0
BODY0, BODY1 = 100.0, 924.0
# Toward the light (upper left, mostly from above): highlights sit on this side.
LIGHT = (-0.40, -0.917)


# ------------------------------------------------------------------ shape
def squircle_points(cx: float, cy: float, half: float, n: float = 4.6, steps: int = 180) -> List[Tuple[float, float]]:
    pts = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        c, s = math.cos(t), math.sin(t)
        x = cx + half * math.copysign(abs(c) ** (2 / n), c)
        y = cy + half * math.copysign(abs(s) ** (2 / n), s)
        pts.append((x, y))
    return pts


def squircle_halfwidth(dy: float, half: float, n: float = 4.6) -> float:
    v = 1 - abs(dy / half) ** n
    return half * (v ** (1 / n)) if v > 0 else 0.0


def body_color(y: float, top: str = APP_TOP, bottom: str = APP_BOTTOM) -> str:
    """The squircle's own colour at height y (its vertical gradient)."""
    return mix(top, bottom, (y - BODY0) / (BODY1 - BODY0))


# --------------------------------------------------------------- geometry
Pt = Tuple[float, float]


def _unit(dx: float, dy: float) -> Pt:
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def arc_points(cx: float, cy: float, rx: float, ry: Optional[float], a0: float, a1: float,
               n: Optional[int] = None) -> List[Pt]:
    """Points on an ellipse from angle a0 to a1 (degrees, counterclockwise as seen on screen)."""
    ry = rx if ry is None else ry
    n = n or max(8, int(abs(a1 - a0) / 3))
    out = []
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / n)
        out.append((cx + rx * math.cos(a), cy - ry * math.sin(a)))
    return out


def bezier(p0: Pt, p1: Pt, p2: Pt, p3: Pt, n: int = 40) -> List[Pt]:
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
                    u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]))
    return out


def polyline_normals(pts: Sequence[Pt]) -> List[Pt]:
    """Unit normals (tangent turned 90 degrees) at every point of a polyline."""
    n = len(pts)
    out = []
    for i in range(n):
        a = pts[max(0, i - 1)]
        b = pts[min(n - 1, i + 1)]
        tx, ty = _unit(b[0] - a[0], b[1] - a[1])
        out.append((-ty, tx))
    return out


def offset_polyline(pts: Sequence[Pt], d: float) -> List[Pt]:
    return [(x + nx * d, y + ny * d) for (x, y), (nx, ny) in zip(pts, polyline_normals(pts))]


def rrect_points(x0: float, y0: float, x1: float, y1: float, r: float, n: int = 10) -> List[Pt]:
    pts: List[Pt] = []
    for cx, cy, a0 in ((x1 - r, y0 + r, 0), (x0 + r, y0 + r, 90), (x0 + r, y1 - r, 180), (x1 - r, y1 - r, 270)):
        pts += arc_points(cx, cy, r, r, a0, a0 + 90, n)
    return pts


def rotate(pts: Iterable[Pt], cx: float, cy: float, deg: float) -> List[Pt]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * c + (y - cy) * s, cy - (x - cx) * s + (y - cy) * c) for x, y in pts]


def arrow_parts(pts: Sequence[Pt], width: float, arrow: str,
                shape: Optional[Tuple[float, float, float]] = None) -> Tuple[List[Pt], List[List[Pt]]]:
    """Shorten a polyline for its arrowheads and return (line points, head triangles).

    shape = (length, back, half-spread) of each head: the head is a triangle with its
    tip at the end of the line, `length` long, `half-spread` beyond the line's edge.
    The same geometry serves both painters, so arrows match exactly.
    """
    pts = list(pts)
    length, _back, spread = shape or (width * 2.3, width * 2.3, width * 0.95)
    heads = []
    ends = []
    if arrow in ("last", "both"):
        ends.append(("last", pts[-1], pts[-2]))
    if arrow in ("first", "both"):
        ends.append(("first", pts[0], pts[1]))
    for which, tip, prev in ends:
        ux, uy = _unit(tip[0] - prev[0], tip[1] - prev[1])
        px, py = -uy, ux
        base = (tip[0] - ux * length, tip[1] - uy * length)
        hw = width / 2 + spread
        heads.append([tip, (base[0] + px * hw, base[1] + py * hw), (base[0] - px * hw, base[1] - py * hw)])
        stop = (tip[0] - ux * length * 0.7, tip[1] - uy * length * 0.7)
        if which == "last":
            pts[-1] = stop
        else:
            pts[0] = stop
    return pts, heads


# --------------------------------------------------------------- painters
class _Painter:
    """What both painters share: arrows are drawn as polygons, identically.

    detail: device pixels per art unit (about 1 for the 1024-px SVG, about 0.1 for a
    50-point canvas on a retina screen); helpers use it to draw fewer, longer strokes
    when the icon is small.
    """

    detail = 1.0

    def line(self, pts: Sequence[Pt], color: str, width: float, smooth: bool = False, cap: str = "round",
             arrow: Optional[str] = None, arrowshape: Optional[Tuple[float, float, float]] = None) -> None:
        pts = list(pts)
        if arrow:
            pts, heads = arrow_parts(pts, width, arrow, arrowshape)
            self._line(pts, color, width, smooth, cap)
            for head in heads:
                self.poly(head, color, color, width * 0.22)
        else:
            self._line(pts, color, width, smooth, cap)

    def _line(self, pts, color, width, smooth, cap) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def poly(self, pts, fill, outline="", width=0.0, smooth=False) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # Finishing touches only the SVG painter can show (a Tk canvas has no blur or
    # transparency): no-ops here.
    def begin(self, effect: str) -> None:
        return

    def end(self) -> None:
        return


class CanvasPainter(_Painter):
    """Draws 1024-unit artwork onto a Tk canvas; the squircle fills `size`."""

    def __init__(self, canvas, size: float, ox: float = 0.0, oy: float = 0.0) -> None:
        self.c = canvas
        self.k = size / (BODY1 - BODY0)
        self.ox = ox - BODY0 * self.k
        self.oy = oy - BODY0 * self.k
        self.size = size
        self.detail = 2.0 * self.k  # retina: two pixels per point
        self._bg: Optional[Tuple[str, str]] = None
        self._glows: List[Tuple[float, float, float, str, float]] = []

    def X(self, x: float) -> float:
        return self.ox + x * self.k

    def Y(self, y: float) -> float:
        return self.oy + y * self.k

    def _flat(self, pts: Iterable[Pt]) -> List[float]:
        out: List[float] = []
        for x, y in pts:
            out += [self.X(x), self.Y(y)]
        return out

    def _rows(self) -> List[float]:
        rows = max(12, int((BODY1 - BODY0) * self.k * 2))
        return [BODY0 + (BODY1 - BODY0) * i / rows for i in range(rows + 1)]

    def _inner_halfwidth(self, y: float) -> float:
        """Half-width of a scanline at y that stays 0.6 pt inside the squircle on every
        side, so the outline drawn by _edge alone defines the edge."""
        half = (BODY1 - BODY0) / 2
        reach = (0.5 + 0.6) / self.k
        return min(squircle_halfwidth(y - 512 - reach, half), squircle_halfwidth(y - 512 + reach, half)) - 0.6 / self.k

    def _edge(self) -> None:
        # An outline in the gradient's own colour along the edge: it alone draws the
        # squircle's boundary, smoothly, over the scanlines' stepped ends.
        top, bottom = self._bg or (APP_TOP, APP_BOTTOM)
        w = 2.0  # points
        half = (BODY1 - BODY0) / 2 - (w / 2) / self.k
        pts = squircle_points(512, 512, half, steps=240)
        pts.append(pts[0])
        prev = None
        for i in range(len(pts) - 1):
            col = body_color((pts[i][1] + pts[i + 1][1]) / 2, top, bottom)
            if prev and prev[0] == col:
                prev[1].append(pts[i + 1])
                continue
            if prev:
                self.c.create_line(*self._flat(prev[1]), fill=prev[0], width=w, capstyle="round", joinstyle="round")
            prev = (col, [pts[i], pts[i + 1]])
        if prev:
            self.c.create_line(*self._flat(prev[1]), fill=prev[0], width=w, capstyle="round", joinstyle="round")

    def background(self, top: str, bottom: str) -> None:
        # Scanlines through the squircle give a crisp gradient without images.
        self._bg = (top, bottom)
        cx = (BODY0 + BODY1) / 2
        for y in self._rows():
            hw = self._inner_halfwidth(y)
            if hw <= 0.5:
                continue
            self.c.create_line(self.X(cx - hw), self.Y(y), self.X(cx + hw), self.Y(y), fill=body_color(y, top, bottom),
                               width=1.0, capstyle="butt")
        self._edge()

    def glow(self, cx: float, cy: float, r: float, color: str, opacity: float = 0.45) -> None:
        """The radial glow, baked into the scanlines it covers (no transparency on a canvas)."""
        if not self._bg:
            return
        self._glows.append((cx, cy, r, color, opacity))
        top, bottom = self._bg
        piece = 1.6 / self.k  # about 1.6 points per piece
        for y in self._rows():
            dy = y - cy
            if abs(dy) >= r:
                continue
            hw = self._inner_halfwidth(y)
            span = math.sqrt(r * r - dy * dy)
            x0, x1 = max(512 - hw, cx - span), min(512 + hw, cx + span)
            if x1 - x0 <= 0.5:
                continue
            base = body_color(y, top, bottom)
            n = max(1, int((x1 - x0) / piece))
            for j in range(n):
                xa = x0 + (x1 - x0) * j / n
                xb = x0 + (x1 - x0) * (j + 1) / n
                d = math.hypot((xa + xb) / 2 - cx, dy) / r
                col = mix(base, color, opacity * max(0.0, 1 - d))
                xe = min(x1, xb + 0.4 / self.k)
                self.c.create_line(self.X(xa), self.Y(y), self.X(xe), self.Y(y), fill=col, width=1.0, capstyle="butt")
        self._edge()

    def sheen(self) -> None:
        """Skipped on a canvas: at window sizes a rim highlight only reads as a seam."""
        return

    def _line(self, pts, color, width, smooth, cap) -> None:
        kw = dict(fill=color, width=max(0.6, width * self.k), capstyle=cap, joinstyle="round")
        if smooth:
            kw.update(smooth=True, splinesteps=16)
        self.c.create_line(*self._flat(pts), **kw)

    def poly(self, pts: Sequence[Pt], fill: str, outline: str = "", width: float = 0.0,
             smooth: bool = False) -> None:
        kw = dict(fill=fill, outline=outline, width=max(0.0, width * self.k) if outline else 0.0, joinstyle="round")
        if smooth:
            kw.update(smooth=True, splinesteps=16)
        self.c.create_polygon(*self._flat(pts), **kw)

    def circle(self, cx: float, cy: float, r: float, fill: str = "", outline: str = "", width: float = 0.0) -> None:
        self.c.create_oval(self.X(cx - r), self.Y(cy - r), self.X(cx + r), self.Y(cy + r), fill=fill,
                           outline=outline, width=max(0.0, width * self.k) if outline else 0.0)


_COLOR_RE = re.compile(r'(?:stroke|fill)="(#[0-9a-fA-F]{6})"')


class SvgPainter(_Painter):
    """The same artwork as SVG, with soft shadow, glow, bloom and sheen."""

    def __init__(self) -> None:
        self.items: List[str] = []
        self.defs: List[str] = []
        self._gid = 0
        self._body_d = ""
        self._stack: List[Tuple[str, int]] = []
        self._bg: Optional[Tuple[str, str]] = None
        self._glows: List[Tuple[float, float, float, str, float]] = []

    def _id(self, prefix: str) -> str:
        self._gid += 1
        return f"{prefix}{self._gid}"

    @staticmethod
    def _d(pts: Sequence[Pt], closed: bool = False, smooth: bool = False) -> str:
        pts = list(pts)
        if not smooth or len(pts) < 3:
            d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
            return d + (" Z" if closed else "")
        # Tk's smooth=true: a quadratic B-spline through the segment midpoints.
        if closed:
            ext = pts + [pts[0], pts[1]]
            mids = [((ext[i][0] + ext[i + 1][0]) / 2, (ext[i][1] + ext[i + 1][1]) / 2) for i in range(len(ext) - 1)]
            d = f"M {mids[0][0]:.1f} {mids[0][1]:.1f}"
            for i in range(1, len(ext) - 1):
                d += f" Q {ext[i][0]:.1f} {ext[i][1]:.1f} {mids[i][0]:.1f} {mids[i][1]:.1f}"
            return d + " Z"
        mids = [((pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2) for i in range(len(pts) - 1)]
        d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f} L {mids[0][0]:.1f} {mids[0][1]:.1f}"
        for i in range(1, len(pts) - 1):
            d += f" Q {pts[i][0]:.1f} {pts[i][1]:.1f} {mids[i][0]:.1f} {mids[i][1]:.1f}"
        return d + f" L {pts[-1][0]:.1f} {pts[-1][1]:.1f}"

    def background(self, top: str, bottom: str) -> None:
        self._bg = (top, bottom)
        gid = self._id("bg")
        self.defs.append(
            f'<linearGradient id="{gid}" x1="0" y1="{BODY0}" x2="0" y2="{BODY1}" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0" stop-color="{top}"/><stop offset="1" stop-color="{bottom}"/></linearGradient>')
        d = self._d(squircle_points(512, 512, (BODY1 - BODY0) / 2), closed=True)
        self.defs.append(
            '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
            '<feGaussianBlur in="SourceAlpha" stdDeviation="5"/><feOffset dy="4" result="s1"/>'
            '<feComponentTransfer in="s1" result="s1a"><feFuncA type="linear" slope="0.22"/></feComponentTransfer>'
            '<feGaussianBlur in="SourceAlpha" stdDeviation="18"/><feOffset dy="14" result="s2"/>'
            '<feComponentTransfer in="s2" result="s2a"><feFuncA type="linear" slope="0.30"/></feComponentTransfer>'
            '<feMerge><feMergeNode in="s2a"/><feMergeNode in="s1a"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>')
        self.items.append(f'<path d="{d}" fill="url(#{gid})" filter="url(#shadow)"/>')
        self.defs.append(f'<clipPath id="body"><path d="{d}"/></clipPath>')
        self._body_d = d

    def sheen(self) -> None:
        gid = self._id("sh")
        self.defs.append(
            f'<linearGradient id="{gid}" x1="0" y1="{BODY0}" x2="0" y2="560" gradientUnits="userSpaceOnUse">'
            '<stop offset="0" stop-color="#ffffff" stop-opacity="0.10"/>'
            '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient>')
        rid = self._id("rim")
        self.defs.append(
            f'<linearGradient id="{rid}" x1="0" y1="{BODY0}" x2="0" y2="{BODY1}" gradientUnits="userSpaceOnUse">'
            '<stop offset="0" stop-color="#ffffff" stop-opacity="0.30"/>'
            '<stop offset="0.35" stop-color="#ffffff" stop-opacity="0.06"/>'
            '<stop offset="1" stop-color="#ffffff" stop-opacity="0.03"/></linearGradient>')
        self.items.append(f'<path d="{self._body_d}" fill="url(#{gid})"/>')
        self.items.append(f'<path d="{self._body_d}" fill="none" stroke="url(#{rid})" '
                          'stroke-width="5" clip-path="url(#body)"/>')

    def glow(self, cx: float, cy: float, r: float, color: str, opacity: float = 0.45) -> None:
        self._glows.append((cx, cy, r, color, opacity))
        gid = self._id("gl")
        self.defs.append(
            f'<radialGradient id="{gid}" cx="{cx}" cy="{cy}" r="{r}" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0" stop-color="{color}" stop-opacity="{opacity}"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></radialGradient>')
        self.items.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#{gid})" clip-path="url(#body)"/>')

    def begin(self, effect: str) -> None:
        self._stack.append((effect, len(self.items)))

    def end(self) -> None:
        if not self._stack:
            return
        effect, start = self._stack.pop()
        body = self.items[start:]
        del self.items[start:]
        if effect == "bloom":
            # A blurred copy of the bright strokes, screened over the artwork: the strands
            # glow, and the glow also spills over the gaps they cut.
            bright = []
            for item in body:
                m = _COLOR_RE.search(item)
                if m and luminance(m.group(1)) > 0.10:
                    bright.append(item)
            fid = self._id("bloom")
            self.defs.append(f'<filter id="{fid}" x="-25%" y="-25%" width="150%" height="150%">'
                             '<feGaussianBlur stdDeviation="16"/></filter>')
            self.items.extend(body)
            self.items.append(f'<g clip-path="url(#body)" style="mix-blend-mode:screen">'
                              f'<g filter="url(#{fid})" opacity="0.42">{"".join(bright)}</g></g>')
        elif effect == "lift":
            # A soft drop shadow under a tool glyph, cast by its ink only: strokes in the
            # tile's own colours (knock-outs) cast none.
            ink = [item for item in body if not self._on_tile(item)]
            fid = self._id("lift")
            self.defs.append(f'<filter id="{fid}" x="-25%" y="-25%" width="150%" height="150%">'
                             '<feGaussianBlur in="SourceAlpha" stdDeviation="10"/><feOffset dy="9"/>'
                             '<feComponentTransfer><feFuncA type="linear" slope="0.20"/></feComponentTransfer>'
                             '</filter>')
            self.items.append(f'<g filter="url(#{fid})" clip-path="url(#body)">{"".join(ink)}</g>')
            self.items.extend(body)
        else:
            self.items.extend(body)

    def _on_tile(self, item: str) -> bool:
        """Whether an item is painted in the tile's own gradient colours (a knock-out)."""
        m = _COLOR_RE.search(item)
        if not m or not self._bg:
            return False
        c = _rgb(m.group(1))
        a, b = _rgb(self._bg[0]), _rgb(self._bg[1])
        span = sum((b[i] - a[i]) ** 2 for i in range(3)) or 1
        t = max(0.0, min(1.0, sum((c[i] - a[i]) * (b[i] - a[i]) for i in range(3)) / span))
        return all(abs(c[i] - (a[i] + (b[i] - a[i]) * t)) <= 2 for i in range(3))

    def _line(self, pts, color, width, smooth, cap) -> None:
        d = self._d(pts, smooth=smooth)
        cap_svg = {"round": "round", "butt": "butt", "projecting": "square"}.get(cap, "round")
        self.items.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width:.2f}" '
                          f'stroke-linecap="{cap_svg}" stroke-linejoin="round"/>')

    def poly(self, pts, fill, outline="", width=0.0, smooth=False) -> None:
        d = self._d(pts, closed=True, smooth=smooth)
        stroke = f' stroke="{outline}" stroke-width="{width:.2f}" stroke-linejoin="round"' if outline else ""
        self.items.append(f'<path d="{d}" fill="{fill or "none"}"{stroke}/>')

    def circle(self, cx, cy, r, fill="", outline="", width=0.0) -> None:
        stroke = f' stroke="{outline}" stroke-width="{width:.2f}"' if outline else ""
        self.items.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill or "none"}"{stroke}/>')

    def svg(self) -> str:
        while self._stack:
            self.end()
        # isolation keeps the screen-blended bloom inside the icon (the page stays transparent)
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">'
                f'<defs>{"".join(self.defs)}</defs><g style="isolation:isolate">{"".join(self.items)}</g></svg>')


# ------------------------------------------------------------- strands
DEEP = mix(BLACK, CHARCOAL, 0.43)  # #161616: what a strand fades toward as it turns away


def _merged(p, pts, colors, widths, cap, a, b) -> None:
    i = a
    while i < b:
        j = i + 1
        while j < b and colors[j] == colors[i] and abs(widths[j] - widths[i]) < 0.05:
            j += 1
        p.line(pts[i:j + 1], colors[i], widths[i], cap=cap)
        i = j


def stroke_runs(p, pts: Sequence[Pt], colors: Sequence[str], widths: Sequence[float], cap: str = "round",
                ends: Tuple[bool, bool] = (True, True)) -> None:
    """A polyline with a colour and width per segment; equal neighbours merge into one stroke.

    Pieces join with round caps (with finely stepped colours the caps' arcs do not
    show). ends: whether the polyline's own start / end are round. A square end is drawn
    as one single stroke over its last half-width, stopping exactly at its point, so a
    run that continues in another stroke joins it without anything spilling across.
    """
    n = len(pts) - 1
    if n < 1:
        return
    if cap != "round" or (ends[0] and ends[1]):
        _merged(p, pts, colors, widths, cap, 0, n)
        return
    e = max(widths) / 2
    cum = [0.0]
    for i in range(n):
        cum.append(cum[-1] + math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
    total = cum[-1]
    i0 = 0 if ends[0] else next((i for i in range(1, n + 1) if cum[i] >= e), n)
    i1 = n if ends[1] else next((i for i in range(n - 1, -1, -1) if total - cum[i] >= e), 0)
    if i0 >= i1:
        p.line(pts, colors[n // 2], max(widths), cap="butt")
        return
    if i0 > 0:
        p.line(pts[:i0 + 1], colors[i0 // 2], max(widths[:i0]), cap="butt")
    if i1 < n:
        p.line(pts[i1:], colors[(i1 + n) // 2], max(widths[i1:]), cap="butt")
    _merged(p, pts, colors, widths, "round", i0, i1)


def surface(p, x: float, y: float) -> str:
    """What the tile shows at (x, y) before any artwork: its gradient and glows."""
    top, bottom = getattr(p, "_bg", None) or (APP_TOP, APP_BOTTOM)
    color = body_color(y, top, bottom)
    for gx, gy, r, gc, op in getattr(p, "_glows", ()):
        d = math.hypot(x - gx, y - gy) / r
        if d < 1:
            color = mix(color, gc, op * (1 - d))
    return color


def cut_color(p, dark: float = 0.18) -> Callable[[float, float], str]:
    """The colour of the gap a strand cuts through what lies behind it: the surface, a
    little deeper, so the cut reads as a clean gap and not as an outline."""
    if getattr(p, "detail", 1.0) < 0.5:
        dark *= 0.25  # small: a darker cut would only show as a beaded outline
    return lambda x, y: mix(surface(p, x, y), BLACK, dark)


def cut_polyline(pts: Sequence[Pt], s0: float, s1: float) -> List[Pt]:
    """The part of a polyline between arc lengths s0 and s1."""
    out: List[Pt] = []
    acc = 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg <= 0:
            continue
        lo, hi = acc, acc + seg
        if hi >= s0 and lo <= s1:
            t0 = max(0.0, (s0 - lo) / seg)
            t1 = min(1.0, (s1 - lo) / seg)
            q0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
            q1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
            if not out:
                out.append(q0)
            out.append(q1)
        acc = hi
    return out


def polyline_length(pts: Sequence[Pt]) -> float:
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))


def halo_line(p, pts: Sequence[Pt], color, width: float, caps: Tuple[bool, bool] = (False, False),
              strips: int = 1) -> None:
    """A wide stroke under a strand that cuts it out of whatever lies behind.

    color: a colour or a function of (x, y) (so the cut can match the surface). The ends
    are square unless capped, so consecutive runs of one strand join without a seam; the
    middle has round joins.
    """
    colfn = color if callable(color) else (lambda _x, _y, c=color: c)
    if strips > 1:
        # side by side strips, each coloured where it lies: a wide cut then matches a
        # surface that changes across it (the glow)
        w = width / strips
        for k in range(strips):
            off = (k - (strips - 1) / 2) * w
            halo_line(p, offset_polyline(pts, off) if off else pts, colfn, w + 2.0, caps)
        return

    def mid(a, b):
        return colfn((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    total = polyline_length(pts)
    e = width / 2
    if total <= 2 * e + 1:
        cap = "round" if (caps[0] and caps[1]) else "butt"
        p.line(pts, mid(pts[0], pts[-1]), width, cap=cap)
        for flag, q in zip(caps, (pts[0], pts[-1])):
            if flag and cap == "butt":
                p.circle(q[0], q[1], e, fill=colfn(q[0], q[1]))
        return
    body = cut_polyline(pts, e, total - e)
    cols = [mid(body[i], body[i + 1]) for i in range(len(body) - 1)]
    stroke_runs(p, body, cols, [width] * len(cols))
    for flag, piece in ((caps[0], cut_polyline(pts, 0, e)), (caps[1], cut_polyline(pts, total - e, total))):
        if len(piece) >= 2:
            p.line(piece, mid(piece[0], piece[-1]), width, cap="round" if flag else "butt")


def _extend_open_ends(run, caps: Tuple[bool, bool], by: float):
    """Lengthen a run's open (square) ends a little along their tangents, so a run and
    its neighbour overlap instead of meeting edge to edge (which leaves a hairline)."""
    run = list(run)
    if not caps[0]:
        a, b = run[0], run[1]
        ux, uy = _unit(a[0] - b[0], a[1] - b[1])
        run.insert(0, (a[0] + ux * by, a[1] + uy * by) + tuple(a[2:]))
    if not caps[1]:
        a, b = run[-1], run[-2]
        ux, uy = _unit(a[0] - b[0], a[1] - b[1])
        run.append((a[0] + ux * by, a[1] + uy * by) + tuple(a[2:]))
    return run


def _toward_light(pts: Sequence[Pt], amounts: Sequence[float]) -> List[Pt]:
    """Shift each point sideways toward the light by amounts[i] (0 where the strand runs
    straight at the light)."""
    out = []
    for (x, y), (nx, ny), a in zip(pts, polyline_normals(pts), amounts):
        o = a * (nx * LIGHT[0] + ny * LIGHT[1])
        out.append((x + nx * o, y + ny * o))
    return out


def tube(p, run, color: str, width: float, back: str = DEEP, halo=None, gap: float = 9.0, shade: float = 0.40,
         thin: float = 0.82, hi: float = 0.55, rim: float = 0.42, caps: Tuple[bool, bool] = (False, False),
         levels: int = 48, halo_from: float = 0.0) -> None:
    """One run of a strand drawn as a lit tube.

    run: [(x, y, d)] with d in -1..1 (1 = nearest the viewer). Passes: a halo that cuts
    the strand cleanly out of anything drawn before it; the tube as a darker rim with a
    core set toward the light (so it reads round); and a highlight line. The tube darkens
    and thins as it turns away.
    halo: a colour, a function of (x, y) giving one, or None. caps: the run starts / ends
    a strand (round halo ends there). rim: how much darker the tube's edges are.
    halo_from: cut only where the tube is at least this near (0..1), e.g. to keep the
    cut away from the joints of a helix strand.
    """
    if len(run) < 2:
        return
    detail = getattr(p, "detail", 1.0)
    levels = max(4, min(levels, int(levels * detail * 1.5)))
    run = _extend_open_ends(run, caps, 3.0)
    pts = [(s[0], s[1]) for s in run]
    kk = [(s[2] + 1) / 2 for s in run]
    seg = [round((kk[i] + kk[i + 1]) / 2 * levels) / levels for i in range(len(run) - 1)]
    if callable(color):  # a colour along the run: color(f), f = 0..1
        n = len(seg)
        base = [color((i + 0.5) / n) for i in range(n)]
        cols = [mix(back, c, shade + (1 - shade) * k) for c, k in zip(base, seg)]
    else:
        cols = [mix(back, color, shade + (1 - shade) * k) for k in seg]
    wids = [width * (thin + (1 - thin) * k) for k in seg]
    pw = [width * (thin + (1 - thin) * k) for k in kk]
    if halo is not None:
        hw = sum(wids) / len(wids) + 2 * gap
        i, n = 0, len(pts)
        while i < n:
            if kk[i] < halo_from:
                i += 1
                continue
            j = i
            while j + 1 < n and kk[j + 1] >= halo_from:
                j += 1
            if j > i:
                halo_line(p, pts[i:j + 1], halo, hw, (caps[0] and i == 0, caps[1] and j == n - 1))
            i = j + 1
    if rim > 0:
        rims = [mix(c, BLACK, rim) for c in cols]
        if detail < 0.5:  # small: one stroke for the outline keeps the edge smooth
            rims = [rims[len(rims) // 2]] * len(rims)
            wids = [max(wids)] * len(wids)
        stroke_runs(p, pts, rims, wids, ends=caps)
        stroke_runs(p, _toward_light(pts, [w * 0.10 for w in pw]), cols, [w * 0.70 for w in wids], ends=caps)
    else:
        stroke_runs(p, pts, cols, wids, ends=caps)
    if hi > 0:
        hp = _toward_light(pts, [w * 0.22 for w in pw])
        i = 0
        n = len(seg)
        while i < n:
            if seg[i] <= 0.6:
                i += 1
                continue
            j = i
            while j < n and seg[j] > 0.6:
                j += 1
            hc = [mix(cols[m], WHITE, hi * ((seg[m] - 0.6) / 0.4) ** 1.2) for m in range(i, j)]
            stroke_runs(p, hp[i:j + 1], hc, [wids[m] * 0.22 for m in range(i, j)])
            i = j


def split_runs(samples, level: float = 0.0):
    """Split [(x, y, d)] where d crosses `level`: [(in_front, run), ...] with the crossing point shared."""
    runs = []
    cur = [samples[0]]
    front = samples[0][2] >= level
    for s in samples[1:]:
        f = s[2] >= level
        if f != front:
            a = cur[-1]
            t = (level - a[2]) / ((s[2] - a[2]) or 1e-9)
            m = (a[0] + (s[0] - a[0]) * t, a[1] + (s[1] - a[1]) * t, level)
            cur.append(m)
            runs.append((front, cur))
            cur = [m]
            front = f
        cur.append(s)
    runs.append((front, cur))
    return runs


def helix_samples(cx: float, y_top: float, y_bot: float, radius: float, turns: float, phase: float,
                  n: int = 160) -> List[Tuple[float, float, float]]:
    """One strand of an upright right-handed helix, bottom to top: [(x, y, depth)].

    depth = -sin(t) > 0 toward the viewer, so the strand runs in front while x grows
    and climbs: front segments slope "/" as a right-handed helix must.
    """
    out = []
    for i in range(n + 1):
        f = i / n
        t = phase + 2 * math.pi * turns * f
        out.append((cx + radius * math.cos(t), y_bot + (y_top - y_bot) * f, -math.sin(t)))
    return out


def draw_strands(p, strands, rungs=(), width: float = 52.0, halo=None, gap: float = 9.0,
                 middle: Optional[Callable] = None, rung_split: float = 0.0, level: float = 0.3, **kw) -> None:
    """Strands [(samples, colour)] on one cylinder of depth: every back run, then the
    rungs (and `middle`, anything at the axis), then every front run. Two strands of
    one helix only ever cross with one in front and one behind (at depth about +-0.9),
    so this order is exact. Runs split at depth `level` rather than 0: the turn at the
    helix's silhouette, where the projected strand bends tighter than the tube is wide,
    then stays inside one stroke. Only front runs need a halo.
    rungs: [(a, b, colour_a, colour_b, width)] with a, b = (x, y, d)."""
    halo = cut_color(p) if halo is None else halo
    fronts = []
    backs = []
    for samples, color in strands:
        runs = split_runs(samples, level)
        # a stub at a strand's end reads as a lump: fold it into its neighbour
        if len(runs) > 1 and polyline_length([q[:2] for q in runs[0][1]]) < width * 1.6:
            runs = [(runs[1][0], runs[0][1] + runs[1][1][1:])] + runs[2:]
        if len(runs) > 1 and polyline_length([q[:2] for q in runs[-1][1]]) < width * 1.6:
            runs = runs[:-2] + [(runs[-2][0], runs[-2][1] + runs[-1][1][1:])]
        for i, (front, run) in enumerate(runs):
            caps = (i == 0, i == len(runs) - 1)
            (fronts if front else backs).append((run, color, caps))
    for run, color, caps in backs:
        tube(p, run, color, width, halo=None, caps=caps, **kw)
    if middle:
        middle()
    for a, b, ca, cb, w in rungs:
        rung(p, a, b, ca, cb, w, trim=(_rung_trim(strands, a, width, w, level),
                                       _rung_trim(strands, b, width, w, level)), split=rung_split)
    for run, color, caps in fronts:
        tube(p, run, color, width, halo=halo, gap=gap, caps=caps, halo_from=0.75, **kw)


def _rung_trim(strands, q, width: float, rung_w: float, level: float) -> float:
    """How far a rung must stop short of strand point q: to the tube's edge, measured
    along the rung, when the strand there is drawn before the rung (behind it)."""
    best = None
    for samples, _c in strands:
        for i, s in enumerate(samples):
            d2 = (s[0] - q[0]) ** 2 + (s[1] - q[1]) ** 2
            if best is None or d2 < best[0]:
                best = (d2, samples, i)
    if best is None or best[0] > 400 or q[2] >= level:
        return width * 0.40
    _d2, samples, i = best
    a = samples[max(0, i - 1)]
    b = samples[min(len(samples) - 1, i + 1)]
    _tx, ty = _unit(b[0] - a[0], b[1] - a[1])
    return (width / 2) / max(0.3, abs(ty)) + rung_w / 2 + 3


def rung(p, a, b, ca: str, cb: str, width: float, trim=0.0, split: float = 0.0,
         back: str = DEEP, shade: float = 0.30) -> None:
    """A base pair between two strand points: two half-bars, shaded by depth.
    trim: how far each end stops short of its strand point (one value, or a pair)."""
    (x1, y1, d1), (x2, y2, d2) = a, b
    ta, tb = trim if isinstance(trim, tuple) else (trim, trim)
    length = math.hypot(x2 - x1, y2 - y1)
    if length < ta + tb + width:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    s = (x1 + ux * ta, y1 + uy * ta)
    e = (x2 - ux * tb, y2 - uy * tb)
    m = ((x1 + x2) / 2, (y1 + y2) / 2)
    k = ((d1 + d2) / 2 + 1) / 2
    p.line([s, (m[0] - ux * split, m[1] - uy * split)], mix(back, ca, shade + (1 - shade) * k), width)
    p.line([(m[0] + ux * split, m[1] + uy * split), e], mix(back, cb, shade + (1 - shade) * k), width)


# ---------------------------------------------------------------- knots
def trefoil(t: float, cx: float, cy: float, s: float):
    x = math.sin(t) + 2 * math.sin(2 * t)
    y = math.cos(t) - 2 * math.cos(2 * t)
    z = -math.sin(3 * t)
    return cx + s * x, cy + s * y, z


def _seg_hit(p1, p2, p3, p4):
    d = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p2[1] - p1[1]) * (p4[0] - p3[0])
    if abs(d) < 1e-12:
        return None
    ta = ((p3[0] - p1[0]) * (p4[1] - p3[1]) - (p3[1] - p1[1]) * (p4[0] - p3[0])) / d
    tb = ((p3[0] - p1[0]) * (p2[1] - p1[1]) - (p3[1] - p1[1]) * (p2[0] - p1[0])) / d
    if 0 <= ta < 1 and 0 <= tb < 1:
        return ta, tb
    return None


def find_crossings(pts3d) -> List[Tuple[float, float]]:
    """Self-crossings of a closed 3D polyline seen from +z: [(over_index, under_index)] as
    fractional indices (pts3d[-1] repeats pts3d[0])."""
    n = len(pts3d) - 1
    out = []
    for i in range(n):
        a, b = pts3d[i], pts3d[i + 1]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = pts3d[j], pts3d[j + 1]
            if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]):
                continue
            if max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
                continue
            hit = _seg_hit(a, b, c, d)
            if hit:
                ta, tb = hit
                za = a[2] + (b[2] - a[2]) * ta
                zb = c[2] + (d[2] - c[2]) * tb
                out.append((i + ta, j + tb) if za > zb else (j + tb, i + ta))
    return out


class Rope:
    """A closed centreline dressed as a duplex: two strands winding round it.

    The rope is drawn whole once, then each over-crossing is drawn again on top
    (with its halo), which gives exact over/under without any depth artefacts.
    """

    def __init__(self, pts3d, radius: float, pitch: float, phases=(0.0, 2.4)) -> None:
        self.pts = [(x, y) for x, y, _ in pts3d]
        self.z = [z for _, _, z in pts3d]
        self.n = len(pts3d) - 1
        self.nrm = polyline_normals(self.pts)
        # close the normals smoothly
        a, b = self.pts[-2], self.pts[1]
        tx, ty = _unit(b[0] - a[0], b[1] - a[1])
        self.nrm[0] = self.nrm[-1] = (-ty, tx)
        self.s = [0.0]
        for i in range(self.n):
            self.s.append(self.s[-1] + math.hypot(self.pts[i + 1][0] - self.pts[i][0],
                                                  self.pts[i + 1][1] - self.pts[i][1]))
        self.length = self.s[-1]
        # a whole number of turns so the closed rope has no seam
        turns = max(1, round(self.length / pitch))
        self.pitch = self.length / turns
        self.r = radius
        self.phases = phases
        zmin, zmax = min(self.z), max(self.z)
        self.zk = [(z - zmin) / ((zmax - zmin) or 1) for z in self.z]
        step = max(1, self.n // 160)
        coarse = [(self.pts[i][0], self.pts[i][1], self.z[i]) for i in range(0, self.n + 1, step)]
        if coarse[-1][:2] != (self.pts[-1][0], self.pts[-1][1]):
            coarse.append((self.pts[-1][0], self.pts[-1][1], self.z[-1]))
        self.crossings = [(a * step, b * step) for a, b in find_crossings(coarse)]

    def strand(self, j: int, idx: Sequence[int]):
        """Strand j as [(x, y, depth)]. Right-handed: with the normal (-ty, tx) on a
        y-down screen, depth = -sin turns the strand anticlockwise about the direction
        of travel, so front segments lean clockwise from the axis ("/" on an upright
        axis), as in helix_samples."""
        out = []
        for i in idx:
            th = 2 * math.pi * self.s[i] / self.pitch + self.phases[j]
            x, y = self.pts[i]
            nx, ny = self.nrm[i]
            u = self.r * math.cos(th)
            out.append((x + nx * u, y + ny * u, -math.sin(th)))
        return out

    def window(self, center: float, half: float) -> List[int]:
        """Indices within arc length `half` of fractional index `center` (wrapping)."""
        n = self.n
        c = int(round(center)) % n
        idx = [c]
        acc = 0.0
        i = c
        while acc < half:
            j = (i - 1) % n
            acc += math.hypot(self.pts[i][0] - self.pts[j][0], self.pts[i][1] - self.pts[j][1])
            idx.insert(0, j)
            i = j
        acc = 0.0
        i = c
        while acc < half:
            j = (i + 1) % n
            acc += math.hypot(self.pts[j][0] - self.pts[i][0], self.pts[j][1] - self.pts[i][1])
            idx.append(j)
            i = j
        return idx


# ------------------------------------------------------------- app art
def art_bnp_na(p) -> None:
    """A right-handed B-DNA double helix threaded on its axis, +Z above it."""
    p.background(APP_TOP, APP_BOTTOM)
    p.glow(512, 520, 400, lab("blue"), 0.36)
    cx, top, bot, R, turns = 512, 318, 788, 148, 1.5
    phi = 2.40  # the minor groove: the strands sit 138 degrees apart, not 180
    th0 = (math.pi - phi) / 2  # ends placed symmetrically about the axis
    red = helix_samples(cx, top, bot, R, turns, th0)
    blue = helix_samples(cx, top, bot, R, turns, th0 + phi)
    rungs = []
    count = 13
    for j in range(count):
        f = (j + 0.5) / count
        y = bot + (top - bot) * f
        t1 = th0 + 2 * math.pi * turns * f
        t2 = t1 + phi
        rungs.append(((cx + R * math.cos(t1), y, -math.sin(t1)), (cx + R * math.cos(t2), y, -math.sin(t2)),
                      lab("red", "T60"), lab("blue", "T60"), 16))
    frame = lab("neutral", "T20")

    def axis() -> None:  # inside the helix: behind the base pairs and the front strands
        p.line([(cx, bot + 4), (cx, top - 4)], mix(frame, DEEP, 0.5), 14)

    p.begin("bloom")
    draw_strands(p, [(red, lab("red")), (blue, lab("blue"))], rungs, width=56, middle=axis)
    p.end()
    # +Z: the axis carries on above the helix
    p.line([(cx, top - 34), (cx, 200)], frame, 14, arrow="last", arrowshape=(56, 56, 20))
    p.sheen()


def art_re_helix(p) -> None:
    """Two parallel duplexes exchanging strands in the middle (a Holliday / DX crossover).

    The right duplex is the left one turned 180 degrees about the centre, so both stay
    right-handed; each inner strand leaves its own duplex and finishes in the other.
    """
    p.background(APP_TOP, APP_BOTTOM)
    p.glow(512, 520, 400, lab("purple"), 0.40)
    top, bot = 236, 796
    mid = (top + bot) / 2
    R, left = 94, 512 - 156
    turns, phi, h = 1.25, 2.4, 80.0
    span = bot - top
    n = 240

    def left_pt(y, ph):
        t = 2 * math.pi * turns * (mid - y) / span + ph
        return left + R * math.cos(t), -math.sin(t)

    def right_pt(y, ph):
        x, d = left_pt(2 * mid - y, ph)
        return 1024 - x, d

    def smooth(v):
        v = max(0.0, min(1.0, v))
        return v * v * (3 - 2 * v)

    ys = [bot + (top - bot) * i / n for i in range(n + 1)]
    outer_l = [(left_pt(y, phi)[0], y, left_pt(y, phi)[1]) for y in ys]
    outer_r = [(right_pt(y, phi)[0], y, right_pt(y, phi)[1]) for y in ys]
    x1, x2 = [], []
    for y in ys:
        b = smooth((y - (mid - h)) / (2 * h))  # 0 above the junction, 1 below
        xl, dl = left_pt(y, 0.0)
        xr, dr = right_pt(y, 0.0)
        lift = 1 - 0.6 * math.exp(-((y - mid) / (2.2 * h)) ** 2)  # keep the crossover behind in view
        x1.append((xl + (xr - xl) * b, y, (dl + (dr - dl) * b) * lift))
        x2.append((xr + (xl - xr) * b, y, dr + (dl - dr) * b))
    P, Q = "purple", "magenta"
    rungs = []
    count = 10
    for j in range(count):
        f = (j + 0.5) / count
        y = bot + (top - bot) * f
        if abs(y - mid) < h * 1.15:
            continue
        i = int(round((bot - y) / (bot - top) * n))
        above = y < mid
        rungs.append((outer_l[i], x1[i] if above else x2[i], lab(P, "T60"), lab(P if above else Q, "T60"), 15))
        rungs.append((x2[i] if above else x1[i], outer_r[i], lab(Q if above else P, "T60"), lab(Q, "T60"), 15))
    p.begin("bloom")
    draw_strands(p, [(outer_l, lab(P)), (outer_r, lab(Q)), (x1, lab(P)), (x2, lab(Q))], rungs, width=46)
    p.end()
    p.sheen()


def art_curve_it(p) -> None:
    """A duplex sculpted along a trefoil knot: two strands twisted round the curve like
    a two-ply cord, cut cleanly at each over-crossing."""
    p.background(APP_TOP, APP_BOTTOM)
    p.glow(512, 520, 400, lab("cyan"), 0.34)
    n = 720
    pts = [trefoil(2 * math.pi * i / n, 512, 548, 100) for i in range(n + 1)]
    r, sw = 18, 38
    rope = Rope(pts, r, 170, (0.0, math.pi))
    cut = cut_color(p, 0.0)
    strand_cols = (lab("cyan"), lab("blue"))
    D = 2 * r + sw

    def piece(idx, over):
        if over:
            halo_line(p, [rope.pts[i] for i in idx], cut, D + 24, strips=5)
        layers = {True: [], False: []}
        for j in (0, 1):
            # the knot's own depth dims its far side a little
            samples = [(x, y, 0.7 * d + 0.3 * (2 * rope.zk[i] - 1))
                       for (x, y, d), i in zip(rope.strand(j, idx), idx)]
            for front, run in split_runs(samples, 0.0):
                layers[front].append((run, strand_cols[j]))
        for front in (False, True):
            for run, col in layers[front]:
                tube(p, run, col, sw, hi=0.5 if front else 0.0, rim=0.35, shade=0.45)

    p.begin("bloom")
    piece(list(range(rope.n + 1)) + [1, 2], False)
    for over, _under in rope.crossings:
        piece(rope.window(over, D * 1.3 + 30), True)
    p.end()
    p.sheen()


def art_azbmost(p) -> None:
    """The suite mark: an A drawn by one DNA strand folded into a hairpin, its base
    pairs the rungs; blue to purple along the strand, the crossbar cyan (the three
    module colours)."""
    p.background(APP_TOP, APP_BOTTOM)
    p.glow(512, 540, 410, lab("purple"), 0.36)
    p0, p3 = (300, 806), (512, 236)
    p2 = (512 - 62, 236)
    p1 = (p0[0] + (p2[0] - p0[0]) / 3, p0[1] + (p2[1] - p0[1]) / 3)
    left = bezier(p0, p1, p2, p3, 60)
    path = left + [(1024 - x, y) for x, y in reversed(left)][1:]

    def leg_x(y):
        return min(left, key=lambda q: abs(q[1] - y))[0]

    p.begin("bloom")
    for i, y in enumerate((600, 482, 364)):
        xl = leg_x(y)
        run = [(xl + (1024 - 2 * xl) * j / 20, y, 1.0) for j in range(21)]
        tube(p, run, lab("cyan") if i == 0 else lab("cyan", "T60"), 50 if i == 0 else 28,
             hi=0.5 if i == 0 else 0.3, rim=0.3)
    run = [(x, y, 1.0) for x, y in path]
    tube(p, run, lambda f: mix(lab("blue"), lab("purple"), f), 74, halo=cut_color(p, 0.18), gap=11,
         caps=(True, True), hi=0.40)
    p.end()
    p.sheen()


APP_ART: Dict[str, Callable] = {
    "azbmost": art_azbmost,
    "bnp_na": art_bnp_na,
    "re_helix": art_re_helix,
    "curve_it": art_curve_it,
}


# ------------------------------------------------------------ tool glyphs
class Inks:
    """What a glyph draws with on its tile.

    a: main ink (white on dark hues, charcoal on light ones)
    b: secondary ink (a lighter tint of the hue, or neutral T60)
    c: accent ink (the hue's T100, or its T20 tint)
    tile(y): the tile's own colour at height y, for knock-out gaps.
    """

    def __init__(self, hue: str) -> None:
        self.hue = hue
        self.a, self.b, self.c = tool_inks(hue)
        if hue == "neutral":
            self.top, self.bottom = LAB["neutral"]["T80"], LAB["neutral"]["T100"]
        else:
            self.top, self.bottom = lab(hue, "T80"), lab(hue, "T100")

    def tile(self, y: float) -> str:
        return body_color(y, self.top, self.bottom)


W1, W2, W3 = 52.0, 36.0, 26.0  # main, secondary and fine stroke weights
HEAD = (96, 96, 30)  # arrowhead for main strokes: length, (unused), spread beyond the stroke's edge
HEAD_S = (66, 66, 22)  # arrowhead for fine strokes


def _densify(path: Sequence[Pt], per: int = 24) -> List[Pt]:
    out: List[Pt] = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        for t in range(per):
            out.append((a[0] + (b[0] - a[0]) * t / per, a[1] + (b[1] - a[1]) * t / per))
    out.append(path[-1])
    return out


def _gaps(pts: Sequence[Pt], cuts: Sequence[Tuple[float, float]]) -> List[List[Pt]]:
    """The pieces of an open polyline left after removing [c - half, c + half] (arc
    lengths) for every (c, half): how an under-strand is drawn in a knot diagram."""
    total = polyline_length(pts)
    spans = sorted((max(0.0, c - h), min(total, c + h)) for c, h in cuts)
    pieces, s0 = [], 0.0
    for a, b in spans:
        if a > s0 + 1:
            pieces.append(cut_polyline(pts, s0, a))
        s0 = max(s0, b)
    if total > s0 + 1:
        pieces.append(cut_polyline(pts, s0, total))
    return [q for q in pieces if len(q) >= 2]


def _gap_half(under_w: float, over_w: float, gap: float, du: Pt, do: Pt) -> float:
    """Half-length to cut from an under-strand so it stops `gap` clear of the over-strand,
    allowing for its round cap and the angle between them."""
    a = _unit(*du)
    b = _unit(*do)
    sin = abs(a[0] * b[1] - a[1] * b[0])
    return (over_w / 2 + gap) / max(0.35, sin) + under_w / 2


def _arclen_at(pts: Sequence[Pt], index: float) -> float:
    i = int(index)
    s = polyline_length(pts[:i + 1])
    if i + 1 < len(pts):
        s += (index - i) * math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    return s


def _knot_glyph(p, k: Inks, pts3d, width: float, gap: float = 18.0, ink: Optional[str] = None) -> None:
    """A closed curve in one ink, broken where it passes under itself."""
    ink = ink or k.a
    crossings = find_crossings(pts3d)
    n = len(pts3d) - 1
    # start the open polyline at an over-crossing, so no gap wraps round the seam
    start = int(round(crossings[0][0])) % n if crossings else 0
    xy = [(pts3d[(start + i) % n][0], pts3d[(start + i) % n][1]) for i in range(n + 1)]

    def direction(index: float) -> Pt:
        i = int(index) % n
        return (pts3d[(i + 1) % n][0] - pts3d[i][0], pts3d[(i + 1) % n][1] - pts3d[i][1])

    cuts = [(_arclen_at(xy, (under - start) % n), _gap_half(width, width, gap, direction(under), direction(over)))
            for over, under in crossings]
    for piece in _gaps(xy, cuts):
        p.line(piece, ink, width)


def _glyph_helix(p, k: Inks, strands, cx, top, bot, radius, turns, width, gap=11, tilt=0.0):
    """Strands of a helix for a glyph: back runs in the secondary ink, broken where a
    front run passes over them; front runs whole. tilt > 0 shows the coil from a
    little above."""
    backs, fronts = [], []
    for phase, ink in strands:
        samples = [(x, y + tilt * radius * d, d) for x, y, d in helix_samples(cx, top, bot, radius, turns, phase, 96)]
        for front, run in split_runs(samples):
            (fronts if front else backs).append(([(x, y) for x, y, _ in run], ink))
    def box(q):
        xs = [a for a, _ in q]
        ys = [b for _, b in q]
        return min(xs), min(ys), max(xs), max(ys)

    for pts, ink in backs:
        cuts = []
        bb = box(pts)
        for fpts, _ in fronts:
            fb = box(fpts)
            if fb[0] > bb[2] or bb[0] > fb[2] or fb[1] > bb[3] or bb[1] > fb[3]:
                continue
            for i in range(len(pts) - 1):
                a0, a1 = pts[i], pts[i + 1]
                for j in range(len(fpts) - 1):
                    b0, b1 = fpts[j], fpts[j + 1]
                    if max(a0[0], a1[0]) < min(b0[0], b1[0]) or max(b0[0], b1[0]) < min(a0[0], a1[0]):
                        continue
                    if max(a0[1], a1[1]) < min(b0[1], b1[1]) or max(b0[1], b1[1]) < min(a0[1], a1[1]):
                        continue
                    hit = _seg_hit(a0, a1, b0, b1)
                    if hit:
                        du = (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                        do = (fpts[j + 1][0] - fpts[j][0], fpts[j + 1][1] - fpts[j][1])
                        cuts.append((_arclen_at(pts, i + hit[0]), _gap_half(width * 0.78, width, gap, du, do)))
        for piece in _gaps(pts, cuts):
            p.line(piece, k.b if ink == k.a else ink, width * 0.78)
    for pts, ink in fronts:
        p.line(pts, ink, width)


def g_generic(p, k: Inks) -> None:
    for y, x in ((350, 404), (512, 620), (674, 460)):
        p.line([(262, y), (762, y)], k.b, W3)
        p.circle(x, y, 54, fill=k.a)


def g_bz_builder(p, k: Inks) -> None:
    """B-DNA (smooth, below) joined to Z-DNA (zigzag, above) on one axis."""
    amp = 132
    lo, mid_y = 800, 520
    smooth = [(512 + amp * math.sin(2 * math.pi * (lo - y) / (lo - mid_y)), y)
              for y in [lo + (mid_y - lo) * i / 48 for i in range(49)]]
    zig = [(512 + amp, mid_y - 70), (512 - amp, mid_y - 210), (512, mid_y - 280)]
    for sign, ink, w in ((-1, k.b, 34), (1, k.a, 46)):
        p.line([(512 + sign * (x - 512), y) for x, y in smooth + zig], ink, w)


def g_triplex(p, k: Inks) -> None:
    _glyph_helix(p, k, [(0.0, k.b), (2.1, k.b), (4.2, k.a)], 512, 236, 788, 150, 1.25, 42)


def g_measure_angle(p, k: Inks) -> None:
    v, a1, a2 = (280, 728), (728, 728), (580, 322)
    ang = math.degrees(math.atan2(v[1] - a2[1], a2[0] - v[0]))
    p.line(arc_points(v[0], v[1], 208, 208, 4, ang - 4, 30), k.b, W2)
    p.line([a1, v, a2], k.a, W1)
    p.circle(a1[0], a1[1], 50, fill=k.a)
    p.circle(a2[0], a2[1], 50, fill=k.a)


def g_align_z(p, k: Inks) -> None:
    """A tilted helix axis swung up onto +Z."""
    pivot = (650, 790)
    tilt = 136.0
    end = (pivot[0] + 450 * math.cos(math.radians(tilt)), pivot[1] - 450 * math.sin(math.radians(tilt)))
    p.line([pivot, end], k.b, W1)
    p.line(arc_points(pivot[0], pivot[1], 330, 330, tilt - 7, 99, 30), k.a, 30, arrow="last", arrowshape=HEAD_S)
    p.line([pivot, (pivot[0], 226)], k.a, W1, arrow="last", arrowshape=HEAD)
    p.circle(pivot[0], pivot[1], 42, fill=k.a)


def g_axis_info(p, k: Inks) -> None:
    cx, ytop, ybot, rx, ry = 512, 340, 664, 160, 50
    p.line(arc_points(cx, ytop, rx, ry, 0, 360, 72), k.b, W3)
    p.line(arc_points(cx, ybot, rx, ry, 180, 360, 36), k.b, W3)
    p.line([(cx - rx, ytop), (cx - rx, ybot)], k.b, W3)
    p.line([(cx + rx, ytop), (cx + rx, ybot)], k.b, W3)
    p.line([(cx, 790), (cx, 226)], k.a, 46, arrow="last", arrowshape=HEAD)
    p.circle(cx, 790, 42, fill=k.a)


def g_add_phosphates(p, k: Inks) -> None:
    """A strand with a phosphate on one end and one being added at the other."""
    p.line(bezier((292, 674), (452, 674), (502, 364), (630, 364), 48), k.a, 42)
    p.circle(292, 674, 66, fill=k.a)
    p.circle(712, 364, 82, outline=k.a, width=32)
    p.line([(712, 326), (712, 402)], k.a, 28)
    p.line([(674, 364), (750, 364)], k.a, 28)


def g_xyz_bild(p, k: Inks) -> None:
    o = (420, 616)
    p.line([o, (420, 226)], k.a, 44, arrow="last", arrowshape=HEAD)
    p.line([o, (800, 616)], k.a, 44, arrow="last", arrowshape=HEAD)
    p.line([o, (240, 796)], k.b, 44, arrow="last", arrowshape=HEAD)
    p.circle(o[0], o[1], 44, fill=k.a)


def g_combine_pdb(p, k: Inks) -> None:
    p.line(bezier((300, 248), (300, 430), (512, 440), (512, 600), 40), k.b, W1)
    p.line(bezier((724, 248), (724, 430), (512, 440), (512, 600), 40), k.a, W1)
    p.line([(512, 590), (512, 800)], k.a, W1, arrow="last", arrowshape=HEAD)
    p.circle(300, 248, 46, fill=k.b)
    p.circle(724, 248, 46, fill=k.a)


def g_regularize(p, k: Inks) -> None:
    jitter = [(262, 330), (380, 280), (520, 352), (628, 296), (762, 344)]
    p.line(jitter, k.b, W3)
    for x, y in jitter:
        p.circle(x, y, 34, fill=k.b)
    xs = [262 + 125 * i for i in range(5)]
    p.line([(xs[0], 696), (xs[-1], 696)], k.a, W3)
    for x in xs:
        p.circle(x, 696, 46, fill=k.a)
    p.line([(512, 438), (512, 584)], k.a, 40, arrow="last", arrowshape=(74, 74, 26))


def g_bend_helix(p, k: Inks) -> None:
    """A two-chain helix (rails and rungs) bent at a pivot."""
    pv = (392, 500)
    ang = math.radians(40)
    ux, uy = math.cos(ang), -math.sin(ang)
    bend_r = 96.0
    c = (pv[0] + bend_r, pv[1])
    sweep = 90 - math.degrees(ang)
    arc = arc_points(c[0], c[1], bend_r, bend_r, 180, 180 - sweep, 16)
    a_end = arc[-1]
    tip = (a_end[0] + 330 * ux, a_end[1] + 330 * uy)
    path = _densify([(pv[0], 812), pv], 16)[:-1] + arc + _densify([a_end, tip], 16)[1:]
    gap = 56
    for y in (760, 640):
        p.line([(pv[0] - gap, y), (pv[0] + gap, y)], k.b, 26)
    for t in (140, 270):
        q = (a_end[0] + t * ux, a_end[1] + t * uy)
        p.line([(q[0] - uy * gap, q[1] + ux * gap), (q[0] + uy * gap, q[1] - ux * gap)], k.b, 26)
    for side in (-1, 1):
        p.line(offset_polyline(path, side * gap), k.a, 42)
    mid_arc = arc[len(arc) // 2]
    p.circle(mid_arc[0], mid_arc[1], 30, fill=k.a)


def g_symmetry(p, k: Inks) -> None:
    for i in range(3):
        a0 = 90 + 120 * i
        p.line(arc_points(512, 512, 196, 196, a0, a0 + 70, 24), k.a, W1)
        q = arc_points(512, 512, 196, 196, a0, a0, 1)[0]
        p.circle(q[0], q[1], 60, fill=k.a)
    p.circle(512, 512, 46, fill=k.b)


def g_link_record(p, k: Inks) -> None:
    def capsule(cx, cy):
        return rotate(rrect_points(cx - 186, cy - 90, cx + 186, cy + 90, 90, 14), cx, cy, 45)
    p.poly(capsule(424, 600), "", k.a, 48)
    p.poly(capsule(600, 424), "", k.a, 48)


def g_virtual_residue(p, k: Inks) -> None:
    y = 566
    p.line([(214, y), (400, y)], k.b, W3)
    p.line([(624, y), (810, y)], k.b, W3)
    for x in (256, 372, 652, 768):
        p.circle(x, y, 44, fill=k.a)
    p.circle(512, y, 76, outline=k.a, width=W3 + 4)
    p.line([(512, 248), (512, 398)], k.a, 40)
    p.line([(437, 323), (587, 323)], k.a, 40)


def g_permute_chain(p, k: Inks) -> None:
    for i, x in enumerate((300, 512, 724)):
        p.poly(rrect_points(x - 76, 566, x + 76, 718, 30, 6), k.a if i != 1 else k.b)
    p.line(arc_points(512, 566, 232, 214, 10, 170, 40), k.a, 40, arrow="last", arrowshape=(76, 76, 28))


def g_lattice(p, k: Inks) -> None:
    o = (266, 748)
    va, vb, vc = (330, 0), (0, -330), (166, -150)
    corners = {}
    for i in (0, 1):
        for j in (0, 1):
            for m in (0, 1):
                corners[(i, j, m)] = (o[0] + va[0] * i + vb[0] * j + vc[0] * m, o[1] + va[1] * i + vb[1] * j + vc[1] * m)
    for key, q in corners.items():
        for d in range(3):
            if key[d] == 0 and key != (0, 0, 0):
                k2 = list(key)
                k2[d] = 1
                p.line([q, corners[tuple(k2)]], k.b, W3)
    for key in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        p.line([o, corners[key]], k.a, 42, arrow="last", arrowshape=(84, 84, 26))
    p.circle(o[0], o[1], 40, fill=k.a)


def g_restraints(p, k: Inks) -> None:
    a, b = (300, 670), (724, 354)
    p.circle(a[0], a[1], 82, fill=k.a)
    p.circle(b[0], b[1], 82, fill=k.a)
    ux, uy = _unit(b[0] - a[0], b[1] - a[1])
    nx, ny = -uy, ux
    total = math.hypot(b[0] - a[0], b[1] - a[1])
    s0, s1 = 104, total - 104
    pts = []
    for i in range(121):
        f = i / 120
        sv = s0 + (s1 - s0) * f
        off = 56 * math.sin(f * math.pi * 8) * min(1.0, f * 8, (1 - f) * 8)
        pts.append((a[0] + ux * sv + nx * off, a[1] + uy * sv + ny * off))
    p.line([(a[0] + ux * 60, a[1] + uy * 60)] + pts + [(b[0] - ux * 60, b[1] - uy * 60)], k.a, 30)


def g_reverse_strand(p, k: Inks) -> None:
    y = 656
    p.line([(232, y), (792, y)], k.b, W3)
    for i, x in enumerate((262, 387, 512, 637, 762)):
        p.circle(x, y, 26 + 7 * i, fill=k.a)
    p.line(arc_points(512, 476, 250, 200, 0, 180, 40), k.a, 44, arrow="last", arrowshape=(88, 88, 30))


def g_clashes(p, k: Inks) -> None:
    """Two atoms too close: one sphere bumps into the other, impact marks above."""
    r = 150
    c1, c2 = (398, 576), (626, 576)
    p.circle(c1[0], c1[1], r, fill=k.b)
    ring = arc_points(c2[0], c2[1], r + 10, r + 10, 0, 360, 90)
    stroke_runs(p, ring, [k.tile((ring[i][1] + ring[i + 1][1]) / 2) for i in range(len(ring) - 1)],
                [24] * (len(ring) - 1))
    p.circle(c2[0], c2[1], r, fill=k.a)
    top = c1[1] - math.sqrt(r * r - ((c2[0] - c1[0]) / 2) ** 2)
    for a in (58, 90, 122):
        u = (math.cos(math.radians(a)), -math.sin(math.radians(a)))
        p.line([(512 + u[0] * 70, top - 34 + u[1] * 70), (512 + u[0] * 150, top - 34 + u[1] * 150)], k.a, 34)


def g_convert_xyz(p, k: Inks) -> None:
    r = 262
    p.line(arc_points(512, 512, r, r, 160, 20, 40), k.a, 46, arrow="last", arrowshape=(90, 90, 30))
    p.line(arc_points(512, 512, r, r, -20, -160, 40), k.a, 46, arrow="last", arrowshape=(90, 90, 30))
    p.line([(372 + 280 * i / 40, 512 - 70 * math.sin(2 * math.pi * i / 40)) for i in range(41)], k.b, W2)


def g_svg2xyz(p, k: Inks) -> None:
    a0, a1 = (252, 704), (772, 320)
    c0, c1 = (300, 330), (724, 694)
    p.line([a0, c0], k.b, 20)
    p.line([a1, c1], k.b, 20)
    p.line(bezier(a0, c0, c1, a1, 60), k.a, 46)
    p.circle(c0[0], c0[1], 34, fill=k.b)
    p.circle(c1[0], c1[1], 34, fill=k.b)
    for x, y in (a0, a1):
        p.poly(rrect_points(x - 50, y - 50, x + 50, y + 50, 14, 4), k.a)


def g_kp2xyz(p, k: Inks) -> None:
    n = 180
    _knot_glyph(p, k, [trefoil(2 * math.pi * i / n, 512, 540, 94) for i in range(n + 1)], W1, 20)


def g_generate_helix(p, k: Inks) -> None:
    _glyph_helix(p, k, [(0.0, k.a)], 512, 310, 730, 182, 3.0, 40, tilt=0.42)


def g_generate_sc(p, k: Inks) -> None:
    """A supercoiled loop: a closed duplex wound on itself (two crossings of one sign)."""
    n = 200
    pts = []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        pts.append((512 + 150 * math.sin(3 * t), 512 - 282 * math.cos(t), math.sin(2 * t)))
    _knot_glyph(p, k, pts, 46, 20)


def g_local_curvature(p, k: Inks) -> None:
    pts = bezier((226, 700), (380, 160), (640, 220), (800, 690), 60)
    x, y = pts[30]
    r = 168
    p.circle(x, y + r, r, outline=k.b, width=W3)
    p.line([(x, y), (x, y + r)], k.b, 22)
    p.line(pts, k.a, W1)
    p.circle(x, y, 42, fill=k.a)
    p.circle(x, y + r, 22, fill=k.b)


def g_plane_it(p, k: Inks) -> None:
    p.poly([(214, 690), (614, 690), (810, 548), (410, 548)], k.b, k.b, W3)
    for x, y, h in ((404, 300, 330), (560, 382, 236), (676, 268, 316)):
        p.line([(x, y), (x, y + h)], k.a, 18)
        p.circle(x, y + h, 22, fill=k.a)
        p.circle(x, y, 48, fill=k.a)


def g_curved_connector(p, k: Inks) -> None:
    """Two helix ends joined by a bent connector duplex: one curved ladder."""
    gap = 42
    path = _densify([(300, 812), (300, 600)], 12)[:-1] + bezier((300, 600), (300, 236), (724, 236), (724, 600), 60) \
        + _densify([(724, 600), (724, 812)], 12)[1:]
    total = polyline_length(path)
    nrm = polyline_normals(path)
    cum = [0.0]
    for i in range(len(path) - 1):
        cum.append(cum[-1] + math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]))
    sv = 58.0
    while sv < total - 40:
        i = min(range(len(path)), key=lambda j: abs(cum[j] - sv))
        (x, y), (nx, ny) = path[i], nrm[i]
        p.line([(x - nx * gap, y - ny * gap), (x + nx * gap, y + ny * gap)], k.b, 22)
        sv += 76
    for side in (-1, 1):
        p.line(offset_polyline(path, side * gap), k.a, 36)


def g_xyz2model(p, k: Inks) -> None:
    path = bezier((236, 720), (300, 300), (700, 240), (790, 560), 60)
    p.line(path[30:], k.b, 22)
    solid = path[:31]
    p.line(solid, k.a, 112)
    p.line(offset_polyline(solid, -26), k.c, 20)
    end = path[30]
    p.circle(end[0], end[1], 56, fill=k.b)


def g_multicolor_split(p, k: Inks) -> None:
    for i, hue in enumerate(("cyan", "magenta", "yellow")):
        cy = 330 + 176 * i
        top = [(512, cy - 96), (760, cy), (512, cy + 96), (264, cy)]
        p.poly([(264, cy), (512, cy + 96), (512, cy + 140), (264, cy + 44)], mix(lab(hue), BLACK, 0.25))
        p.poly([(512, cy + 96), (760, cy), (760, cy + 44), (512, cy + 140)], mix(lab(hue), BLACK, 0.45))
        p.poly(top, lab(hue))


def g_reverse_curve_it(p, k: Inks) -> None:
    """A helix wound on a curved axis, and the axis (centreline) recovered from it."""
    axis = arc_points(530, 512, 236, 236, 52, 308, 120)
    nrm = polyline_normals(axis)
    cum = [0.0]
    for i in range(len(axis) - 1):
        cum.append(cum[-1] + math.hypot(axis[i + 1][0] - axis[i][0], axis[i + 1][1] - axis[i][1]))
    coil = []
    for (x, y), (nx, ny), sv in zip(axis, nrm, cum):
        off = 58 * math.sin(2 * math.pi * sv / 150)
        coil.append((x + nx * off, y + ny * off))
    p.line(coil, k.b, 26)
    p.line(axis, k.a, 42)
    for q in (axis[0], axis[-1]):
        p.circle(q[0], q[1], 46, fill=k.a)


def g_get_phase(p, k: Inks) -> None:
    def wave(ph):
        return [(232 + 560 * i / 60, 548 - 150 * math.sin(2 * math.pi * 1.25 * i / 60 + ph)) for i in range(61)]
    p.line(wave(0.0), k.b, W2)
    p.line(wave(-1.0), k.a, W1)
    p.line([(330, 290), (500, 290)], k.a, 28, arrow="both", arrowshape=(56, 56, 20))


def g_view_curve(p, k: Inks) -> None:
    up = bezier((214, 512), (330, 300), (694, 300), (810, 512), 40)
    dn = bezier((810, 512), (694, 724), (330, 724), (214, 512), 40)
    p.poly(up + dn[1:-1], "", k.a, 46)
    p.circle(512, 512, 110, fill=k.a)
    p.circle(512, 512, 44, fill=k.c)


# Tool key -> (glyph, hue). "neutral" means a charcoal tile.
TOOLS: Dict[str, Tuple[Callable, str]] = {
    "generic": (g_generic, "blue"),
    # BNP_NA
    "bz_builder": (g_bz_builder, "red"),
    "triplex": (g_triplex, "magenta"),
    "measure_angle": (g_measure_angle, "orange"),
    "align_z": (g_align_z, "blue"),
    "axis_info": (g_axis_info, "cyan"),
    "add_phosphates": (g_add_phosphates, "yellow"),
    "xyz_bild": (g_xyz_bild, "purple"),
    "combine_pdb": (g_combine_pdb, "green"),
    "regularize_phosphates": (g_regularize, "mint"),
    # RE_Helix
    "bend_helix": (g_bend_helix, "blue"),
    "do_symmetry": (g_symmetry, "purple"),
    "add_link": (g_link_record, "orange"),
    "insert_virtual_resi": (g_virtual_residue, "cyan"),
    "permute_chain": (g_permute_chain, "magenta"),
    "generate_lattice": (g_lattice, "green"),
    "phenix_restraints": (g_restraints, "yellow"),
    "reverse_strand": (g_reverse_strand, "mint"),
    "check_clashes": (g_clashes, "red"),
    # Curve_It
    "convert_xyz": (g_convert_xyz, "blue"),
    "svg2xyz": (g_svg2xyz, "orange"),
    "kp2xyz": (g_kp2xyz, "purple"),
    "generate_helix": (g_generate_helix, "green"),
    "generate_sc": (g_generate_sc, "magenta"),
    "local_curvature": (g_local_curvature, "yellow"),
    "plane_it": (g_plane_it, "cyan"),
    "curved_connector": (g_curved_connector, "red"),
    "xyz2model": (g_xyz2model, "mint"),
    "multicolor_split": (g_multicolor_split, "neutral"),
    "reverse_curve_it": (g_reverse_curve_it, "blue"),
    "get_phase": (g_get_phase, "orange"),
    "view_curve": (g_view_curve, "cyan"),
}


def tool_inks(hue: str) -> Tuple[str, str, str]:
    if hue == "neutral":
        return WHITE, LAB["neutral"]["T20"], LAB["neutral"]["T80"]
    if hue in LIGHT_HUES:
        return CHARCOAL, LAB["neutral"]["T60"], lab(hue, "T20")
    return WHITE, lab(hue, "T40"), lab(hue, "T100")


def draw_tool(p, key: str) -> None:
    glyph, hue = TOOLS.get(key, TOOLS["generic"])
    k = Inks(hue)
    p.background(k.top, k.bottom)
    p.begin("lift")
    glyph(p, k)
    p.end()
    p.sheen()


def draw(p, key: str) -> None:
    """key: "bnp_na", "re_helix", "curve_it", "azbmost" or "tool:<name>"."""
    if key.startswith("tool:"):
        draw_tool(p, key[5:])
    else:
        APP_ART.get(key, art_azbmost)(p)


def icon_svg(key: str) -> str:
    painter = SvgPainter()
    draw(painter, key)
    return painter.svg()


def export_svg(out_dir: str) -> List[str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for key in APP_ART:
        path = os.path.join(out_dir, f"{key}.svg")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(icon_svg(key))
        written.append(path)
    for key in TOOLS:
        path = os.path.join(out_dir, f"tool_{key}.svg")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(icon_svg("tool:" + key))
        written.append(path)
    return written


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--svg":
        for path in export_svg(sys.argv[2]):
            print(path)
    else:
        print(__doc__)
