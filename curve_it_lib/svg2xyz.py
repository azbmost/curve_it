#!/usr/bin/env python3
"""
svg2xyz.py -- turn the curves drawn in a 2D SVG into 3D XYZ space curves.

Every path, polyline, polygon, line, rect, circle and ellipse in the drawing
becomes one component of a plain-coordinate XYZ file:

    x y z          <- component A
    x y z
                   <- a blank line separates components
    x y z          <- component B
    x y z

The drawing is two dimensional, so every z is the same constant, 0 by
default.  Curve It, view_xyzV3.py and xyz2model.py all read that file and
treat the blank-line-separated blocks as components A, B, C, and so on, so a
drawing holding three curves becomes a curve file holding three space curves.

Beziers and elliptical arcs are flattened by adaptive subdivision, so corners
stay sharp and arcs stay smooth at the requested tolerance.  Group and element
transforms are composed exactly.  The SVG y axis points DOWN the page, so it
is flipped by default and the exported curve then looks like the drawing when
plotted in the usual mathematical orientation.

    python3 svg2xyz.py                            # GUI
    python3 svg2xyz.py drawing.svg --info         # describe it, write nothing
    python3 svg2xyz.py drawing.svg
    python3 svg2xyz.py drawing.svg -s 0.25 --points 400
    python3 svg2xyz.py drawing.svg --exclude xy-plane,scale-bar

Requires numpy; matplotlib is needed only for --preview.  Run with no
arguments, or with --gui, for the graphical interface.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

__version__ = "1.0"

TOOL_NAME = "SVG to XYZ"


def resource_path(relative_path):
    """Return a resource path that also works from a PyInstaller bundle."""
    source_dir = os.path.dirname(os.path.abspath(__file__))
    source_root = (os.path.dirname(source_dir)
                   if os.path.basename(source_dir) == "curve_it_lib" else source_dir)
    base_dir = getattr(sys, "_MEIPASS", source_root)
    return os.path.join(base_dir, relative_path)


def set_optional_window_icon(root, tk_module, icon_filenames, image_attr):
    """Set a Tk window icon if one of the optional PNG assets is available."""
    for icon_filename in icon_filenames:
        icon_path = resource_path(os.path.join("assets", icon_filename))
        if not os.path.isfile(icon_path):
            continue
        try:
            icon_image = tk_module.PhotoImage(file=icon_path)
            root.iconphoto(True, icon_image)
            setattr(root, image_attr, icon_image)
            return
        except Exception:
            continue


DEFAULT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#808000", "#000075",
]

# Recursion cap for Bezier flattening.  The tolerance test normally stops
# long before this; the cap only bounds pathological input.
MAX_SUBDIVISION = 12
MIN_TOLERANCE = 1.0e-9

# Resampling reads arc length off the flattened polyline, and a chord always
# undercuts the curve it spans, most of all where the curvature is highest.
# Flattening this much finer than asked before resampling keeps that bias out
# of the point spacing; the extra points are thrown away immediately.
DENSE_FACTOR = 16

# Smallest turning angle, in degrees, that makes a vertex count as a corner.
# Corners are kept exactly through resampling; gentler vertices are not.
DEFAULT_MIN_CORNER_ANGLE = 20.0

# Decimal places per coordinate; matches curve_it.write_plain_xyz_curve.
DEFAULT_PRECISION = 6

# A <use> cycle is caught by the id stack, but an acyclic fan-out is not, so
# the total number of expansions is capped as well.
MAX_USE_DEPTH = 12
MAX_USE_EXPANSIONS = 20000


# --------------------------------------------------------------------------
# GUI help text: key -> (title, prose, example or None)
# --------------------------------------------------------------------------
HELP = {
    "file": (
        "Input SVG file",
        "A 2D vector drawing. Every <path>, <polyline>, <polygon>, <line>, "
        "<rect>, <circle> and <ellipse> that is actually drawn becomes one "
        "curve. Adobe Illustrator, Inkscape, Affinity Designer, Figma, "
        "CorelDRAW, matplotlib and Plane It output are all read the same way.\n\n"
        "Text, images, gradients, clip paths, markers and anything inside "
        "<defs> are ignored, as is anything hidden with display:none or "
        "visibility:hidden. A <use> reference is expanded in place.\n\n"
        "macOS Finder aliases are followed to the original file.",
        "one drawn star + one ellipse\n"
        "  <g id=\"Layer_1\"> <path d=\"M318,282 l10,-7 ... Z\"/> </g>\n"
        "  <g id=\"Layer_2\"> <ellipse cx=\"346\" cy=\"482\" rx=\"100\" ry=\"50\"/> </g>\n"
        "  ->  component A (star), component B (ellipse)"),

    "curves": (
        "Curves found",
        "One line per curve, in document order, labelled the way Curve It "
        "labels components: A, B, C, and so on.\n\n"
        "Each line names the SVG element it came from, its id / class / layer "
        "when the drawing has them, whether the curve is closed, how many "
        "points it will be written with, and its length in output units.\n\n"
        "Use the include / exclude fields to drop curves you do not want.",
        "A  path      id=star     layer=Layer_1   closed   214 pts   len 214.71\n"
        "B  ellipse   class=st0   layer=Layer_2   closed   198 pts   len 484.42"),

    "tolerance": (
        "Flattening tolerance",
        "Beziers and elliptical arcs are not polylines, so they are "
        "subdivided until every piece is straight to within this distance. "
        "The value is in the SVG's own coordinate units, before the scale "
        "factor.\n\n"
        "Smaller means more points and a closer fit. Corner points are always "
        "kept exactly, whatever the tolerance, because subdivision never "
        "removes an endpoint.\n\n"
        "A typical page is several hundred units across, so the 0.05 default "
        "is already far finer than the drawing's own accuracy.",
        "on a 612 x 792 page\n"
        "    0.5    coarse, visibly faceted arcs\n"
        "    0.05   default; error below one part in ten thousand\n"
        "    0.005  very fine, roughly three times as many points"),

    "points": (
        "Points per curve",
        "'auto' keeps the points that adaptive flattening produced. They are "
        "dense on tight bends and sparse on straight runs, and every corner "
        "sits exactly on a point.\n\n"
        "A number instead resamples each curve to exactly that many points, "
        "evenly spaced by arc length. Even spacing suits curvature and writhe "
        "work, but a resampled corner is only approximated, so prefer 'auto' "
        "for polygonal drawings.\n\n"
        "Closed curves are resampled without repeating the first point.",
        "auto   corners exact, spacing uneven\n"
        "400    400 evenly spaced points on every curve\n"
        "\n"
        "Curve It can also resample later: Interpolation mode 'n'."),

    "spacing": (
        "Point spacing",
        "An alternative to a fixed point count: resample every curve to this "
        "arc-length step, measured in FINAL output units, after the scale "
        "factor.\n\n"
        "Unlike a fixed count, this gives long and short curves the same point "
        "density, which is usually what you want when one drawing holds curves "
        "of very different sizes.\n\n"
        "Leave it blank to use the points-per-curve setting instead.",
        "two curves, 400 and 100 units long\n"
        "    points  = 400  ->  400 pts and 400 pts  (4x denser on the short one)\n"
        "    spacing = 1.0  ->  400 pts and 100 pts  (same density)"),

    "corner": (
        "Minimum corner angle",
        "The smallest turning angle that makes a vertex count as a corner. "
        "It is a threshold, not an angle the tool applies to anything.\n\n"
        "Only used when resampling. Evenly spaced samples walk straight past "
        "a sharp vertex, so a rectangle, a polygon or a star comes back with "
        "clipped corners. Every vertex that turns by at least this much is "
        "kept exactly instead, and each smooth stretch between two kept "
        "corners is resampled on its own, in proportion to its length.\n\n"
        "Set it to 0 to resample blindly. Raise it to keep only the very "
        "sharpest vertices; lower it to keep gentler ones too. It has no "
        "effect when points per curve is 'auto', because nothing is resampled "
        "then and every vertex is already exact.",
        "60 x 40 rectangle resampled to 137 points\n"
        "    min corner angle  0   corners clipped by 0.52 units\n"
        "    min corner angle 20   all four corners exact; a rectangle turns 90 deg\n"
        "\n"
        "more points does not fix clipping: at 301 points it is still 0.23"),

    "scale": (
        "Scale factor",
        "Multiplies every output coordinate. SVG user units are arbitrary, "
        "while Curve It works in Angstrom, so this is where the two are "
        "reconciled.\n\n"
        "It is applied after flattening, so it does not change how finely the "
        "curves were sampled.\n\n"
        "Curve It can also rescale a curve on its own with Scale mode, so an "
        "exact factor here matters only when you want the XYZ file itself to "
        "carry real units.",
        "a 200-unit-wide drawing\n"
        "    scale 1.0    ->  200 units across\n"
        "    scale 0.5    ->  100 units across\n"
        "    scale 1.7    ->  340 units across  (one DNA persistence length)"),

    "fit": (
        "Fit size",
        "Instead of giving a scale factor, give the size you want. The whole "
        "drawing is scaled uniformly so that its largest dimension, x or y, "
        "equals this number.\n\n"
        "All curves are scaled by the same factor, so their relative sizes and "
        "positions are preserved.\n\n"
        "Leave it blank to use the scale factor instead. When both are given, "
        "fit size wins.",
        "drawing spans 612 x 792\n"
        "    fit size 340  ->  spans 262.7 x 340  (factor 0.42929)"),

    "z": (
        "Z value",
        "The drawing is flat, so every point gets the same z. Zero is almost "
        "always what you want: the curve then lies in the z = 0 plane, which "
        "is where Curve It's default ring and Plane It's projection plane also "
        "live.\n\n"
        "A nonzero value only shifts the whole curve along z.",
        "z = 0     318.420  -282.440  0.000\n"
        "z = 10    318.420  -282.440  10.000"),

    "closed": (
        "Closed curves",
        "'auto' takes the answer from the drawing: a path ended with the Z "
        "closepath command, and every <circle>, <ellipse>, <rect> and "
        "<polygon>, is closed. A path whose last point simply lands back on "
        "its first point is also treated as closed.\n\n"
        "'yes' and 'no' force every curve either way.\n\n"
        "A closed curve is written WITHOUT repeating its first point at the "
        "end, which is the convention Curve It, its writhe calculation and "
        "Generate SC all use. Load such a file with Path type: closed.",
        "square drawn as  M 0 0 H 10 V 10 H 0 Z\n"
        "    written as 4 points, not 5\n"
        "    Curve It closes the loop from the last point back to the first"),

    "center": (
        "Centre on the origin",
        "SVG coordinates are measured from the top-left corner of the page, so "
        "a drawing usually sits at a large offset. Recentring moves it onto "
        "the origin.\n\n"
        "bbox     -- centre of the bounding box of all curves\n"
        "centroid -- mean of all points of all curves\n"
        "none     -- keep the SVG's own coordinates\n\n"
        "All curves are shifted together, so their relative positions survive. "
        "This pairs with Curve It's Scale anchor: a centred curve scales the "
        "same way about 'origin' and about 'centroid'.",
        "star drawn at x = 280..350 on the page\n"
        "    none  ->  x runs 280 .. 350\n"
        "    bbox  ->  x runs -35 .. 35"),

    "flip": (
        "Flip the y axis",
        "The SVG y axis points DOWN the page: a point lower on the screen has "
        "a LARGER y. Ordinary 3D and mathematical plots point y up.\n\n"
        "Flipping negates y so the exported curve looks like the drawing when "
        "viewed in Curve It's curve viewer, view_xyzV3.py, or any normal xy "
        "plot. Leave it on unless you specifically want raw SVG coordinates.\n\n"
        "Flipping mirrors the curve, so it reverses the sign of any signed "
        "quantity computed from it later, such as writhe.",
        "SVG point   (318.42,  282.44)\n"
        "  flipped   (318.42, -282.44)   <- drawing looks upright\n"
        "  raw       (318.42,  282.44)   <- drawing looks upside down"),

    "space": (
        "Coordinate space",
        "user     -- the SVG's own user units, the numbers you see inside the "
        "d= attribute and the viewBox. Predictable, and the default.\n\n"
        "viewport -- additionally applies the root viewBox to width/height "
        "mapping, including preserveAspectRatio, so the coordinates come out "
        "in rendered pixels. Use this when the file carries a physical width "
        "and height, such as width=\"210mm\", and you care about the printed "
        "size.\n\n"
        "The two are identical when there is no viewBox, or when the viewBox "
        "matches the width and height.",
        "<svg width=\"105mm\" height=\"105mm\" viewBox=\"0 0 100 100\">\n"
        "    user      ->  0 .. 100        (drawing units)\n"
        "    viewport  ->  0 .. 396.85     (px, at 96 dpi)"),

    "components": (
        "Components",
        "Which curves to keep, chosen by the labels in the listing above: A, "
        "or B,C, or A-C, or all.\n\n"
        "These are the same labels Curve It uses, so what you pick here is "
        "spelled the same way as its Select components... dialog and its "
        "--curve-components option.\n\n"
        "It is applied after every other filter, so the labels are the ones "
        "you can actually see. Size and position were already resolved over "
        "the whole drawing, so two components pulled out in separate runs "
        "still line up with each other. Use include / exclude instead when "
        "you want the leftover curves to reframe the output.\n\n"
        "'one file per curve' in Output writes all of them separately in a "
        "single pass, which is usually easier than converting one at a time.",
        "listing shows   A path (star)   B ellipse\n"
        "    all    both curves, as components A and B\n"
        "    B      only the ellipse, written as component A\n"
        "    A,C    the first and third\n"
        "    A-C    the first three"),

    "filters": (
        "Include and exclude",
        "Comma-separated names, matched case-insensitively against each "
        "curve's id, any of its class names, and the id or label of every "
        "group it sits inside.\n\n"
        "Include keeps only the curves that match. Exclude drops the curves "
        "that match. Exclude is applied after include.\n\n"
        "This is how you drop the decoration from a Plane It SVG and keep only "
        "the projected curve.",
        "keep one Illustrator layer\n"
        "    include: Layer_2\n\n"
        "drop Plane It's frame, scale bar and base-pair lines\n"
        "    exclude: xy-plane,scale-bar,base-pair-line"),

    "elements": (
        "Element types",
        "Which SVG shapes are read. 'all' takes every one of them.\n\n"
        "Narrow it when a drawing mixes real curves with markers or dots you "
        "do not want. Plane It, for instance, writes one <circle> per atom "
        "next to the <path> that connects them.",
        "all                     everything below\n"
        "path                    only <path> elements\n"
        "path,polyline,polygon   drawn outlines, no primitive shapes\n"
        "\n"
        "available: path, polyline, polygon, line, rect, circle, ellipse"),

    "minpoints": (
        "Smallest curve kept",
        "Curves with fewer points than the minimum, or shorter than the "
        "minimum length, are dropped.\n\n"
        "Illustrator and Inkscape both leave behind stray one- and two-point "
        "fragments when a drawing has been edited. Two points is the smallest "
        "curve Curve It can use at all, and a curve needs at least four "
        "distinct points before its writhe can be computed.\n\n"
        "The minimum length is in final output units, after the scale factor.",
        "min points 2     drop nothing but isolated points\n"
        "min points 4     drop fragments too short for a writhe calculation\n"
        "min length 1.0   drop curves under one output unit long"),

    "subpaths": (
        "Split subpaths",
        "One <path> element can hold several disconnected outlines, each "
        "started by its own M command. Illustrator writes a letter with a hole, "
        "or several shapes merged into one compound path, exactly this way.\n\n"
        "Splitting turns each outline into its own component, which is almost "
        "always right: they are separate closed loops in space.\n\n"
        "Turning it off concatenates every outline of one element into a "
        "single component, joined by a straight jump between them.",
        "<path d=\"M 0 0 H 10 V 10 Z  M 20 0 H 30 V 10 Z\"/>\n"
        "    split   ->  two closed square components\n"
        "    joined  ->  one component that jumps from square to square"),

    "output": (
        "Output file",
        "A plain-coordinate XYZ file: one 'x y z' row per point, with a blank "
        "line between components. That is the format Curve It's Curve XYZ/txt "
        "field, view_xyzV3.py and xyz2model.py all read.\n\n"
        "'one file per curve' additionally writes drawing_A.xyz, drawing_B.xyz "
        "and so on, each holding a single curve.\n\n"
        "The comment header records where every component came from. It starts "
        "with '#', which all of the readers skip, so it is safe to leave on.",
        "drawing.xyz\n"
        "    # SVG to XYZ v1.0 -- 2 component(s) from drawing.svg\n"
        "    # component A: path id=star layer=Layer_1 closed=yes points=214\n"
        "    318.420000 -282.440000 0.000000\n"
        "    ...\n"
        "    <blank line>\n"
        "    # component B: ellipse class=st0 layer=Layer_2 closed=yes ...\n"
        "    446.330000 -482.880000 0.000000"),
}


# --------------------------------------------------------------------------
# Input paths
# --------------------------------------------------------------------------
def resolve_path(path):
    """Follow symlinks and macOS Finder aliases to the real file."""
    path = os.path.expanduser(path)
    if os.path.islink(path):
        return os.path.realpath(path)
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"book":       # not an alias
                return path
    except OSError:
        return path
    err = None
    try:
        from Foundation import NSURL       # type: ignore

        url = NSURL.fileURLWithPath_(os.path.abspath(path))
        resolved, err = NSURL.URLByResolvingAliasFileAtURL_options_error_(
            url, 1 << 8, None)              # NSURLBookmarkResolutionWithoutUI
        if resolved is not None:
            return resolved.path()
        data, err = NSURL.bookmarkDataWithContentsOfURL_error_(url, None)
        if data is not None:
            resolved, _stale, err = (
                NSURL.URLByResolvingBookmarkData_options_relativeToURL_bookmarkDataIsStale_error_(
                    data, 1 << 8, None, None, None))
            if resolved is not None:
                return resolved.path()
    except Exception as exc:                # noqa: BLE001
        err = exc
    raise IOError("'%s' is a macOS alias that could not be resolved (%s).\n"
                  "Point the tool at the original file instead." % (path, err))


def component_label(index):
    """Spreadsheet-style labels A, B, ..., Z, AA, AB, matching Curve It."""
    if index < 0:
        raise ValueError("Component index must be non-negative.")
    label = ""
    n = index
    while True:
        label = chr(ord("A") + (n % 26)) + label
        n = n // 26 - 1
        if n < 0:
            break
    return label


def component_index(label):
    """Turn a label such as 'B' or 'AA' back into a zero-based index."""
    text = (label or "").strip().upper()
    if not text or not text.isalpha():
        raise ValueError("invalid component label %r" % label)
    value = 0
    for character in text:
        value = value * 26 + (ord(character) - ord("A") + 1)
    return value - 1


def parse_component_selection(spec, count):
    """Read 'A', 'B,C', 'A-C' or 'all' into zero-based indices.

    Deliberately the same spelling as Curve It's --curve-components, so one
    set of labels drives both tools.
    """
    text = (spec or "all").strip()
    if not text or text.lower() in ("all", "*"):
        return list(range(count))
    if count <= 0:
        raise ValueError("there are no components to select from")

    chosen = []
    for token in text.replace(";", ",").replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = [part.strip() for part in token.split("-", 1)]
            start, end = component_index(first), component_index(last)
            if end < start:
                start, end = end, start
            chosen.extend(range(start, end + 1))
        else:
            chosen.append(component_index(token))

    selected = []
    for index in sorted(chosen):
        if index < 0 or index >= count:
            raise ValueError("component %s is out of range A-%s"
                             % (component_label(max(index, 0)),
                                component_label(count - 1)))
        if index not in selected:
            selected.append(index)
    if not selected:
        raise ValueError("no components were selected")
    return selected


# --------------------------------------------------------------------------
# Numbers, lengths and affine transforms
# --------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_LENGTH_RE = re.compile(r"^\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)"
                        r"\s*([a-zA-Z%]*)\s*$")

# CSS reference pixels: 1in = 96px.
UNIT_TO_PX = {
    "": 1.0, "px": 1.0,
    "pt": 96.0 / 72.0, "pc": 16.0,
    "in": 96.0, "mm": 96.0 / 25.4, "cm": 96.0 / 2.54,
    "q": 96.0 / 101.6,
    "em": 16.0, "ex": 8.0, "rem": 16.0,
}

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def parse_numbers(text):
    """Every number in a whitespace/comma separated list."""
    return [float(m.group()) for m in _NUMBER_RE.finditer(text or "")]


def parse_length(text, percent_of=None, default=None):
    """Convert an SVG length such as '210mm' or '50%' to user units."""
    if text is None:
        return default
    match = _LENGTH_RE.match(str(text))
    if not match:
        return default
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "%":
        if percent_of is None:
            return default
        return value * 0.01 * percent_of
    factor = UNIT_TO_PX.get(unit)
    if factor is None:
        return default
    return value * factor


def mat_mul(m1, m2):
    """Return m1 followed by m2 in SVG order: apply m2 first, then m1."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (a1 * a2 + c1 * b2,
            b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2,
            b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1,
            b1 * e2 + d1 * f2 + f1)


def mat_point(m, x, y):
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def mat_array(m, points):
    a, b, c, d, e, f = m
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.column_stack([a * p[:, 0] + c * p[:, 1] + e,
                            b * p[:, 0] + d * p[:, 1] + f])


_TRANSFORM_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")


def parse_transform(text):
    """Compose an SVG transform list into one matrix.

    SVG applies a transform list left to right, so the leftmost transform is
    the outermost one: transform="translate(a) scale(b)" scales first, then
    translates.
    """
    matrix = IDENTITY
    if not text:
        return matrix
    for match in _TRANSFORM_RE.finditer(text):
        name = match.group(1)
        args = parse_numbers(match.group(2))
        step = IDENTITY
        if name == "matrix" and len(args) >= 6:
            step = tuple(args[:6])
        elif name == "translate" and args:
            tx = args[0]
            ty = args[1] if len(args) > 1 else 0.0
            step = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale" and args:
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            step = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate" and args:
            angle = math.radians(args[0])
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            step = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                step = mat_mul((1.0, 0.0, 0.0, 1.0, cx, cy), step)
                step = mat_mul(step, (1.0, 0.0, 0.0, 1.0, -cx, -cy))
        elif name == "skewX" and args:
            step = (1.0, 0.0, math.tan(math.radians(args[0])), 1.0, 0.0, 0.0)
        elif name == "skewY" and args:
            step = (1.0, math.tan(math.radians(args[0])), 0.0, 1.0, 0.0, 0.0)
        matrix = mat_mul(matrix, step)
    return matrix


def viewbox_transform(view_box, width, height, preserve_aspect_ratio):
    """The viewBox to viewport mapping of SVG 1.1 section 7.8."""
    if not view_box or len(view_box) != 4:
        return IDENTITY
    vb_x, vb_y, vb_w, vb_h = view_box
    if vb_w <= 0.0 or vb_h <= 0.0 or width is None or height is None:
        return IDENTITY
    if width <= 0.0 or height <= 0.0:
        return IDENTITY

    tokens = (preserve_aspect_ratio or "xMidYMid meet").split()
    tokens = [t for t in tokens if t != "defer"]
    align = tokens[0] if tokens else "xMidYMid"
    meet_or_slice = tokens[1] if len(tokens) > 1 else "meet"

    scale_x = width / vb_w
    scale_y = height / vb_h
    if align.lower() != "none":
        scale = min(scale_x, scale_y) if meet_or_slice != "slice" else max(scale_x, scale_y)
        scale_x = scale_y = scale

    tx = -vb_x * scale_x
    ty = -vb_y * scale_y
    if "xmid" in align.lower():
        tx += (width - vb_w * scale_x) / 2.0
    elif "xmax" in align.lower():
        tx += width - vb_w * scale_x
    if "ymid" in align.lower():
        ty += (height - vb_h * scale_y) / 2.0
    elif "ymax" in align.lower():
        ty += height - vb_h * scale_y
    return (scale_x, 0.0, 0.0, scale_y, tx, ty)


# --------------------------------------------------------------------------
# Path data
# --------------------------------------------------------------------------
_WSP = " \t\r\n\f\v,"


class _PathScanner:
    """A command-aware scanner for the 'd' attribute.

    Path data is not a plain number list: the elliptical-arc flags are single
    characters that may be written with no separator at all, as in
    'a1 1 0 011 1', so flags need their own reader.
    """

    def __init__(self, text):
        self.s = text or ""
        self.i = 0
        self.n = len(self.s)

    def _skip(self):
        while self.i < self.n and self.s[self.i] in _WSP:
            self.i += 1

    def eof(self):
        self._skip()
        return self.i >= self.n

    def peek_command(self):
        self._skip()
        if self.i < self.n and self.s[self.i].isalpha():
            return self.s[self.i]
        return None

    def take_command(self):
        command = self.peek_command()
        if command is not None:
            self.i += 1
        return command

    def number(self):
        self._skip()
        match = _NUMBER_RE.match(self.s, self.i)
        if not match:
            raise ValueError("expected a number at offset %d of the path data" % self.i)
        self.i = match.end()
        return float(match.group())

    def flag(self):
        self._skip()
        if self.i < self.n and self.s[self.i] in "01":
            value = self.s[self.i] == "1"
            self.i += 1
            return value
        return bool(self.number())      # a producer that wrote the flag in full


def _reflect(point, about):
    return (2.0 * about[0] - point[0], 2.0 * about[1] - point[1])


def elliptical_arc_cubics(cx, cy, rx, ry, phi, theta0, delta,
                          start=None, end=None):
    """Cubic segments for an elliptical arc in centre parameterisation.

    The arc is cut into pieces of at most 45 degrees and each piece is matched
    at both ends in position and tangent, which keeps the radial error below
    about 2e-5 of the radius.  The whole-quarter constant that is often used
    for circles instead leaves an error near 3e-4, large enough to show up in
    a curvature or writhe calculation, so every rounded shape in this module
    goes through here.
    """
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    def point_at(theta):
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        return (cx + rx * cos_t * cos_phi - ry * sin_t * sin_phi,
                cy + rx * cos_t * sin_phi + ry * sin_t * cos_phi)

    def derivative_at(theta):
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        return (-rx * sin_t * cos_phi - ry * cos_t * sin_phi,
                -rx * sin_t * sin_phi + ry * cos_t * cos_phi)

    if abs(delta) < 1e-15:
        return []
    count = max(1, int(math.ceil(abs(delta) / (math.pi / 4.0) - 1e-9)))
    step = delta / count
    alpha = 4.0 / 3.0 * math.tan(step / 4.0)

    segments = []
    current = start if start is not None else point_at(theta0)
    for k in range(count):
        theta_a = theta0 + k * step
        theta_b = theta_a + step
        target = point_at(theta_b)
        if k == count - 1 and end is not None:
            target = end
        da = derivative_at(theta_a)
        db = derivative_at(theta_b)
        c1 = (current[0] + alpha * da[0], current[1] + alpha * da[1])
        c2 = (target[0] - alpha * db[0], target[1] - alpha * db[1])
        segments.append(("C", current, c1, c2, target))
        current = target
    return segments


def arc_to_cubics(p0, rx, ry, rotation_deg, large_arc, sweep, p1):
    """Endpoint-parameterised elliptical arc as cubic Beziers.

    Implements the endpoint-to-centre conversion of SVG 1.1 appendix F.6.5
    together with the out-of-range corrections of F.6.6, then approximates the
    resulting arc with cubic segments of at most 45 degrees, whose radial
    error is below 2e-5 of the radius.
    """
    x0, y0 = p0
    x1, y1 = p1
    if abs(x0 - x1) < 1e-12 and abs(y0 - y1) < 1e-12:
        return []                                   # F.6.2: coincident, no arc
    rx, ry = abs(rx), abs(ry)
    if rx < 1e-12 or ry < 1e-12:
        return [("L", p0, p1)]                      # F.6.2: degenerate, a line

    phi = math.radians(rotation_deg % 360.0)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    dx2 = (x0 - x1) / 2.0
    dy2 = (y0 - y1) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:                                   # F.6.6.2: radii too small
        root = math.sqrt(lam)
        rx *= root
        ry *= root

    numerator = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    factor = math.sqrt(max(0.0, numerator / denominator)) if denominator > 0.0 else 0.0
    if bool(large_arc) == bool(sweep):
        factor = -factor
    cxp = factor * rx * y1p / ry
    cyp = -factor * ry * x1p / rx

    cx = cos_phi * cxp - sin_phi * cyp + (x0 + x1) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y0 + y1) / 2.0

    def angle_between(ux, uy, vx, vy):
        norm = math.hypot(ux, uy) * math.hypot(vx, vy)
        if norm <= 0.0:
            return 0.0
        cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / norm))
        angle = math.acos(cosine)
        return -angle if (ux * vy - uy * vx) < 0.0 else angle

    ux = (x1p - cxp) / rx
    uy = (y1p - cyp) / ry
    vx = (-x1p - cxp) / rx
    vy = (-y1p - cyp) / ry
    theta0 = angle_between(1.0, 0.0, ux, uy)
    delta = angle_between(ux, uy, vx, vy)
    if not sweep and delta > 0.0:
        delta -= 2.0 * math.pi
    elif sweep and delta < 0.0:
        delta += 2.0 * math.pi

    return elliptical_arc_cubics(cx, cy, rx, ry, phi, theta0, delta,
                                 start=p0, end=p1)


_IMPLICIT_AFTER_MOVE = {"M": "L", "m": "l"}


def parse_path_d(d, on_error=None):
    """Return [(segments, closed), ...], one entry per subpath.

    A segment is ('L', p0, p1) or ('C', p0, c1, c2, p1); quadratics are
    degree-elevated and arcs are converted while parsing, so flattening only
    ever sees straight lines and cubics.

    Malformed data stops the parse and keeps everything read so far, which is
    what a browser does and what salvages a truncated file.  When `on_error`
    is given it is called with the reason instead of the data being dropped
    silently.
    """
    scanner = _PathScanner(d)
    subpaths = []
    segments = []
    start = (0.0, 0.0)
    current = (0.0, 0.0)
    prev_cubic_ctrl = None
    prev_quad_ctrl = None
    command = None

    def flush(closed):
        if segments:
            subpaths.append((list(segments), bool(closed)))
        del segments[:]

    while not scanner.eof():
        try:
            letter = scanner.peek_command()
            if letter is not None:
                command = scanner.take_command()
            elif command is None:
                raise ValueError("path data does not begin with a command")
            elif command in _IMPLICIT_AFTER_MOVE:
                command = _IMPLICIT_AFTER_MOVE[command]     # extra M pairs are L
            elif command in "Zz":
                raise ValueError("path data has numbers after a closepath command")

            upper = command.upper()
            relative = command.islower()

            if upper == "M":
                x = scanner.number()
                y = scanner.number()
                if relative:
                    x += current[0]
                    y += current[1]
                flush(False)
                current = start = (x, y)
                prev_cubic_ctrl = prev_quad_ctrl = None

            elif upper == "Z":
                if segments and (abs(current[0] - start[0]) > 1e-12
                                 or abs(current[1] - start[1]) > 1e-12):
                    segments.append(("L", current, start))
                flush(True)
                current = start
                prev_cubic_ctrl = prev_quad_ctrl = None

            elif upper in ("L", "H", "V"):
                if upper == "L":
                    x = scanner.number()
                    y = scanner.number()
                    if relative:
                        x += current[0]
                        y += current[1]
                elif upper == "H":
                    x = scanner.number()
                    if relative:
                        x += current[0]
                    y = current[1]
                else:
                    y = scanner.number()
                    if relative:
                        y += current[1]
                    x = current[0]
                target = (x, y)
                segments.append(("L", current, target))
                current = target
                prev_cubic_ctrl = prev_quad_ctrl = None

            elif upper in ("C", "S"):
                if upper == "C":
                    c1 = (scanner.number(), scanner.number())
                    c2 = (scanner.number(), scanner.number())
                else:
                    c1 = (_reflect(prev_cubic_ctrl, current)
                          if prev_cubic_ctrl is not None else current)
                    c2 = (scanner.number(), scanner.number())
                target = (scanner.number(), scanner.number())
                if relative:
                    if upper == "C":
                        c1 = (c1[0] + current[0], c1[1] + current[1])
                    c2 = (c2[0] + current[0], c2[1] + current[1])
                    target = (target[0] + current[0], target[1] + current[1])
                segments.append(("C", current, c1, c2, target))
                current = target
                prev_cubic_ctrl = c2
                prev_quad_ctrl = None

            elif upper in ("Q", "T"):
                if upper == "Q":
                    q = (scanner.number(), scanner.number())
                else:
                    q = (_reflect(prev_quad_ctrl, current)
                         if prev_quad_ctrl is not None else current)
                target = (scanner.number(), scanner.number())
                if relative:
                    if upper == "Q":
                        q = (q[0] + current[0], q[1] + current[1])
                    target = (target[0] + current[0], target[1] + current[1])
                # degree elevation: an exact cubic form of the quadratic
                c1 = (current[0] + 2.0 / 3.0 * (q[0] - current[0]),
                      current[1] + 2.0 / 3.0 * (q[1] - current[1]))
                c2 = (target[0] + 2.0 / 3.0 * (q[0] - target[0]),
                      target[1] + 2.0 / 3.0 * (q[1] - target[1]))
                segments.append(("C", current, c1, c2, target))
                current = target
                prev_quad_ctrl = q
                prev_cubic_ctrl = None

            elif upper == "A":
                rx = scanner.number()
                ry = scanner.number()
                rotation = scanner.number()
                large_arc = scanner.flag()
                sweep = scanner.flag()
                x = scanner.number()
                y = scanner.number()
                if relative:
                    x += current[0]
                    y += current[1]
                target = (x, y)
                segments.extend(arc_to_cubics(current, rx, ry, rotation,
                                              large_arc, sweep, target))
                current = target
                prev_cubic_ctrl = prev_quad_ctrl = None

            else:
                raise ValueError("unknown path command %r" % command)

        except (ValueError, IndexError) as exc:
            if on_error is not None:
                on_error("path data: %s" % exc)
            break

    flush(False)
    return subpaths


# --------------------------------------------------------------------------
# Primitive shapes, expressed as the same segment lists
# --------------------------------------------------------------------------
def ellipse_segments(cx, cy, rx, ry):
    """A full ellipse, starting at (cx + rx, cy) and running clockwise.

    Clockwise on screen is the direction SVG 1.1 gives for the equivalent
    path of <circle> and <ellipse>, because the y axis points down.
    """
    return elliptical_arc_cubics(cx, cy, rx, ry, 0.0, 0.0, 2.0 * math.pi)


def rect_segments(x, y, w, h, rx, ry):
    """A rectangle, with the rounded-corner rules of SVG 1.1 section 9.2."""
    if w <= 0.0 or h <= 0.0:
        return []
    if rx is None and ry is None:
        rx = ry = 0.0
    elif rx is None:
        rx = ry
    elif ry is None:
        ry = rx
    rx = min(max(rx or 0.0, 0.0), w / 2.0)
    ry = min(max(ry or 0.0, 0.0), h / 2.0)
    x2, y2 = x + w, y + h
    if rx <= 0.0 or ry <= 0.0:
        corners = [(x, y), (x2, y), (x2, y2), (x, y2)]
        return [("L", corners[i], corners[(i + 1) % 4]) for i in range(4)]

    quarter = math.pi / 2.0
    segments = []
    segments.append(("L", (x + rx, y), (x2 - rx, y)))
    segments.extend(elliptical_arc_cubics(x2 - rx, y + ry, rx, ry, 0.0,
                                          -quarter, quarter))
    segments.append(("L", (x2, y + ry), (x2, y2 - ry)))
    segments.extend(elliptical_arc_cubics(x2 - rx, y2 - ry, rx, ry, 0.0,
                                          0.0, quarter))
    segments.append(("L", (x2 - rx, y2), (x + rx, y2)))
    segments.extend(elliptical_arc_cubics(x + rx, y2 - ry, rx, ry, 0.0,
                                          quarter, quarter))
    segments.append(("L", (x, y2 - ry), (x, y + ry)))
    segments.extend(elliptical_arc_cubics(x + rx, y + ry, rx, ry, 0.0,
                                          math.pi, quarter))
    return [s for s in segments
            if not (s[0] == "L" and abs(s[1][0] - s[2][0]) < 1e-12
                    and abs(s[1][1] - s[2][1]) < 1e-12)]


def polyline_segments(points):
    return [("L", points[i], points[i + 1]) for i in range(len(points) - 1)]


# --------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------
def _flatten_cubic(p0, p1, p2, p3, limit, out, depth):
    """Recursive de Casteljau subdivision to a flatness bound.

    The test is the standard bound on the distance from a cubic to its chord:
    with u = 3*P1 - 2*P0 - P3 and v = 3*P2 - P0 - 2*P3 the error is at most
    sqrt(max(ux^2, vx^2) + max(uy^2, vy^2)) / 16.  It stays correct when the
    curve returns to its own start, where a chord-distance test degenerates.
    """
    ux = 3.0 * p1[0] - 2.0 * p0[0] - p3[0]
    uy = 3.0 * p1[1] - 2.0 * p0[1] - p3[1]
    vx = 3.0 * p2[0] - p0[0] - 2.0 * p3[0]
    vy = 3.0 * p2[1] - p0[1] - 2.0 * p3[1]
    error = max(ux * ux, vx * vx) + max(uy * uy, vy * vy)
    if error <= limit or depth >= MAX_SUBDIVISION:
        out.append(p3)
        return
    p01 = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
    p12 = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    p23 = ((p2[0] + p3[0]) * 0.5, (p2[1] + p3[1]) * 0.5)
    p012 = ((p01[0] + p12[0]) * 0.5, (p01[1] + p12[1]) * 0.5)
    p123 = ((p12[0] + p23[0]) * 0.5, (p12[1] + p23[1]) * 0.5)
    mid = ((p012[0] + p123[0]) * 0.5, (p012[1] + p123[1]) * 0.5)
    _flatten_cubic(p0, p01, p012, mid, limit, out, depth + 1)
    _flatten_cubic(mid, p123, p23, p3, limit, out, depth + 1)


def flatten_segments(segments, tolerance):
    """Turn a segment list into an (N, 2) polyline."""
    if not segments:
        return np.zeros((0, 2), dtype=float)
    tolerance = max(float(tolerance), MIN_TOLERANCE)
    limit = 16.0 * tolerance * tolerance
    points = [segments[0][1]]
    for segment in segments:
        if segment[0] == "L":
            points.append(segment[2])
        else:
            _flatten_cubic(segment[1], segment[2], segment[3], segment[4],
                           limit, points, 0)
    return np.asarray(points, dtype=float)


# --------------------------------------------------------------------------
# Minimal CSS: only the two properties that can hide geometry
# --------------------------------------------------------------------------
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_CSS_RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}", re.S)
_CLASS_CHAIN_RE = re.compile(r"^(?:\.[A-Za-z_][-\w]*)+$")
# Only the properties that can remove geometry from the drawing.
_VISIBILITY_PROPERTIES = ("display", "visibility", "opacity")


def parse_style_sheet(root):
    """Collect display/visibility/opacity declarations from every <style>.

    A geometry converter does not need colours, so nothing else is kept:
    Illustrator's '.st0 { fill:#fff; stroke:#231f20; }' parses to nothing at
    all, which is the correct result.  What it must not miss is a hidden
    layer, which Illustrator delivers as '.st5 { display:none; }' on a class
    rather than as an inline style.

    Only simple selectors are understood: '*', a tag name, '#id', and a chain
    of one or more classes such as '.fil0.str0', which is how CorelDRAW
    writes them.  Combinators, attribute selectors, pseudo-classes and
    at-rules are ignored.
    """
    rules = {"universal": {}, "tag": {}, "class": {}, "compound": [], "id": {}}
    for element in root.iter():
        if local_name(element.tag) != "style":
            continue
        text = _CSS_COMMENT_RE.sub(" ", "".join(element.itertext()))
        for match in _CSS_RULE_RE.finditer(text):
            declarations = {k: v for k, v in
                            parse_style_attribute(match.group(2)).items()
                            if k in _VISIBILITY_PROPERTIES}
            if not declarations:
                continue
            for selector in match.group(1).split(","):
                selector = selector.strip()
                if not selector or selector.startswith("@"):
                    continue
                if selector == "*":
                    rules["universal"].update(declarations)
                elif selector.startswith("#") and " " not in selector:
                    rules["id"].setdefault(selector[1:], {}).update(declarations)
                elif _CLASS_CHAIN_RE.match(selector):
                    names = [n for n in selector.split(".") if n]
                    if len(names) == 1:
                        rules["class"].setdefault(names[0], {}).update(declarations)
                    else:
                        rules["compound"].append((frozenset(names), declarations))
                elif selector.replace("-", "").isalnum():
                    rules["tag"].setdefault(selector.lower(), {}).update(declarations)
    return rules


def parse_style_attribute(text):
    declarations = {}
    for chunk in (text or "").split(";"):
        if ":" not in chunk:
            continue
        name, _, value = chunk.partition(":")
        declarations[name.strip().lower()] = value.strip().lower()
    return declarations


def resolved_style(element, tag, classes, element_id, style_rules):
    """Effective display/visibility/opacity, in rising order of precedence."""
    style = dict(style_rules["universal"])
    style.update(style_rules["tag"].get(tag, {}))
    class_set = set(classes)
    for name in classes:
        style.update(style_rules["class"].get(name, {}))
    for names, declarations in style_rules["compound"]:
        if names <= class_set:
            style.update(declarations)
    if element_id:
        style.update(style_rules["id"].get(element_id, {}))
    for name in _VISIBILITY_PROPERTIES:
        value = element.get(name)
        if value:
            style[name] = value.strip().lower()
    style.update({k: v for k, v in parse_style_attribute(element.get("style")).items()
                  if k in _VISIBILITY_PROPERTIES})
    return style


def hides_subtree(style):
    """True when the element and everything under it is not drawn.

    display:none and a fully transparent group both remove the whole subtree.
    visibility:hidden does not: it is inherited but a descendant can set
    visibility:visible and reappear, so it is resolved per element instead.
    """
    if style.get("display", "").startswith("none"):
        return True
    opacity = style.get("opacity")
    if opacity is not None:
        try:
            return float(opacity) <= 0.0
        except ValueError:
            return False
    return False


# --------------------------------------------------------------------------
# Document walk
# --------------------------------------------------------------------------
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"

GEOMETRY_TAGS = ("path", "polyline", "polygon", "line", "rect", "circle", "ellipse")

# Subtrees that never contribute drawn geometry on their own.  <symbol> and
# the gradient/pattern elements are reachable only through <use> or a paint
# reference, so they are pruned here and expanded there.
SKIP_SUBTREE_TAGS = frozenset((
    "defs", "clippath", "mask", "marker", "pattern", "symbol", "filter",
    "lineargradient", "radialgradient", "meshgradient", "solidcolor",
    "metadata", "title", "desc", "style", "script", "text", "tspan",
    "textpath", "altglyph", "image", "foreignobject", "font", "font-face",
    "glyph", "missing-glyph", "hatch", "view", "cursor",
    "animate", "animatetransform", "animatemotion", "animatecolor", "set",
))

CONTAINER_TAGS = frozenset(("g", "a"))

# Attributes whose spelling an HTML serializer lower-cases.  Only relevant in
# a document that carries no namespace at all, which is what a copy out of a
# browser's innerHTML looks like.
CAMEL_CASE_ATTRIBUTES = ("viewBox", "preserveAspectRatio")


def local_name(tag):
    """Strip an ElementTree namespace, tolerating comments and unnamespaced files."""
    if not isinstance(tag, str):
        return ""
    return (tag.split("}", 1)[1] if "}" in tag else tag).lower()


def restore_camel_case_attributes(root):
    """Recover viewBox and friends from an all-lower-case serialization."""
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        for name in CAMEL_CASE_ATTRIBUTES:
            if element.get(name) is None:
                value = element.get(name.lower())
                if value is not None:
                    element.set(name, value)


def switch_child(element):
    """The one child a <switch> renders, per SVG 1.1 section 5.8.

    Illustrator's "Preserve Illustrator Editing Capabilities" export wraps
    the drawing in a <switch> beside a <foreignObject>; descending into every
    child would emit the fallback geometry twice.
    """
    for child in element:
        tag = local_name(child.tag)
        if not tag or tag in SKIP_SUBTREE_TAGS:
            continue
        blocked = False
        for name in ("requiredExtensions", "requiredFeatures"):
            value = child.get(name)
            if value is not None and value.strip():
                blocked = True
        if not blocked:
            return child
    return None


class Curve(object):
    """One flattened outline, with the drawing metadata it came from."""

    __slots__ = ("points", "closed", "tag", "element_id", "classes",
                 "layers", "subpath", "subpath_count")

    def __init__(self, points, closed, tag, element_id, classes, layers,
                 subpath=0, subpath_count=1):
        self.points = points
        self.closed = bool(closed)
        self.tag = tag
        self.element_id = element_id or ""
        self.classes = list(classes)
        self.layers = list(layers)
        self.subpath = subpath
        self.subpath_count = subpath_count

    @property
    def layer(self):
        return self.layers[-1] if self.layers else ""

    def names(self):
        """Every name this curve can be selected by."""
        names = [self.element_id] + self.classes + self.layers
        return [n.lower() for n in names if n]

    def source(self):
        parts = [self.tag]
        if self.element_id:
            parts.append("id=%s" % _flat(self.element_id))
        if self.classes:
            parts.append("class=%s" % ",".join(_flat(c) for c in self.classes))
        if self.layer:
            parts.append("layer=%s" % _flat(self.layer))
        if self.subpath_count > 1:
            parts.append("subpath=%d/%d" % (self.subpath + 1, self.subpath_count))
        return " ".join(parts)


def _flat(text):
    """Collapse whitespace so a name stays one token inside a comment line."""
    return "_".join(str(text).split()) or "-"


def _element_classes(element):
    return [c for c in (element.get("class") or "").replace(",", " ").split() if c]


def _group_name(element):
    return (element.get(INKSCAPE_LABEL) or element.get("id") or "").strip()


def element_subpaths(element, tag, tolerance, on_error=None):
    """Return [(points, closed), ...] in the element's own coordinate space."""
    if tag == "path":
        pieces = parse_path_d(element.get("d"), on_error=on_error)
    elif tag == "line":
        p0 = (parse_length(element.get("x1"), default=0.0) or 0.0,
              parse_length(element.get("y1"), default=0.0) or 0.0)
        p1 = (parse_length(element.get("x2"), default=0.0) or 0.0,
              parse_length(element.get("y2"), default=0.0) or 0.0)
        pieces = [([("L", p0, p1)], False)]
    elif tag in ("polyline", "polygon"):
        flat = parse_numbers(element.get("points"))
        pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
        if len(pts) < 2:
            return []
        closed = (tag == "polygon")
        segments = polyline_segments(pts)
        if closed and (abs(pts[0][0] - pts[-1][0]) > 1e-12
                       or abs(pts[0][1] - pts[-1][1]) > 1e-12):
            segments.append(("L", pts[-1], pts[0]))
        pieces = [(segments, closed)]
    elif tag == "rect":
        x = parse_length(element.get("x"), default=0.0) or 0.0
        y = parse_length(element.get("y"), default=0.0) or 0.0
        w = parse_length(element.get("width"), default=0.0) or 0.0
        h = parse_length(element.get("height"), default=0.0) or 0.0
        rx = parse_length(element.get("rx")) if element.get("rx") not in (None, "auto") else None
        ry = parse_length(element.get("ry")) if element.get("ry") not in (None, "auto") else None
        segments = rect_segments(x, y, w, h, rx, ry)
        pieces = [(segments, True)] if segments else []
    elif tag in ("circle", "ellipse"):
        cx = parse_length(element.get("cx"), default=0.0) or 0.0
        cy = parse_length(element.get("cy"), default=0.0) or 0.0
        if tag == "circle":
            r = parse_length(element.get("r"), default=0.0) or 0.0
            rx = ry = r
        else:
            rx = parse_length(element.get("rx"), default=0.0) or 0.0
            ry = parse_length(element.get("ry"), default=0.0) or 0.0
        if rx <= 0.0 or ry <= 0.0:
            return []
        pieces = [(ellipse_segments(cx, cy, rx, ry), True)]
    else:
        return []
    return pieces


def _root_viewport(root):
    """The root viewBox, and the width/height it is rendered into."""
    view_box = parse_numbers(root.get("viewBox"))
    view_box = view_box[:4] if len(view_box) >= 4 else None
    vb_w = view_box[2] if view_box else None
    vb_h = view_box[3] if view_box else None
    width = parse_length(root.get("width"), percent_of=vb_w, default=None)
    height = parse_length(root.get("height"), percent_of=vb_h, default=None)
    if width is None or width <= 0.0:
        width = vb_w
    if height is None or height <= 0.0:
        height = vb_h
    return view_box, width, height


def extract_curves(path, tolerance=0.05, space="user", split_subpaths=True,
                   elements="all"):
    """Read an SVG and return (curves, info).

    Curves are flattened polylines in the chosen coordinate space, still with
    the SVG's own y-down orientation and no scaling applied.
    """
    real = resolve_path(path)
    try:
        tree = ET.parse(real)
    except ET.ParseError as exc:
        raise ValueError("%s is not readable as XML (%s)" % (os.path.basename(real), exc))
    root = tree.getroot()
    if local_name(root.tag) != "svg":
        raise ValueError("%s has <%s> as its root element, not <svg>"
                         % (os.path.basename(real), local_name(root.tag)))
    if "}" not in str(root.tag):
        restore_camel_case_attributes(root)

    wanted = None
    if elements and str(elements).strip().lower() not in ("", "all", "*"):
        wanted = {t.strip().lower() for t in str(elements).replace(";", ",").split(",")
                  if t.strip()}
        unknown = wanted - set(GEOMETRY_TAGS)
        if unknown:
            raise ValueError("unknown element type(s): %s; choose from %s"
                             % (", ".join(sorted(unknown)), ", ".join(GEOMETRY_TAGS)))

    style_rules = parse_style_sheet(root)
    ids = {}
    for element in root.iter():
        element_id = element.get("id")
        if element_id and element_id not in ids:
            ids[element_id] = element

    view_box, width, height = _root_viewport(root)
    root_matrix = IDENTITY
    if space == "viewport":
        root_matrix = viewbox_transform(view_box, width, height,
                                        root.get("preserveAspectRatio"))
    root_matrix = mat_mul(root_matrix, parse_transform(root.get("transform")))

    curves = []
    counts = {"elements": 0, "hidden": 0, "empty": 0, "uses": 0}
    warnings = []

    def warn(message):
        if message not in warnings and len(warnings) < 40:
            warnings.append(message)

    def visit(element, matrix, layers, visibility, use_depth, seen):
        tag = local_name(element.tag)
        if not tag or tag in SKIP_SUBTREE_TAGS:
            return
        classes = _element_classes(element)
        style = resolved_style(element, tag, classes, element.get("id"), style_rules)
        if hides_subtree(style):
            counts["hidden"] += 1
            return
        visibility = style.get("visibility", visibility)
        matrix = mat_mul(matrix, parse_transform(element.get("transform")))

        if tag == "switch":
            chosen = switch_child(element)
            if chosen is not None:
                visit(chosen, matrix, layers, visibility, use_depth, seen)
            return

        if tag == "use":
            if use_depth >= MAX_USE_DEPTH or counts["uses"] >= MAX_USE_EXPANSIONS:
                warn("stopped expanding <use>: the reference nesting or count "
                     "hit the built-in limit")
                return
            counts["uses"] += 1
            href = (element.get("href") or element.get(XLINK_HREF) or "").strip()
            if not href.startswith("#"):
                if href:
                    warn("skipped a <use> pointing outside this file: %s" % href)
                return
            target = ids.get(href[1:])
            if target is None or id(target) in seen:
                return
            x = parse_length(element.get("x"), default=0.0) or 0.0
            y = parse_length(element.get("y"), default=0.0) or 0.0
            child_matrix = mat_mul(matrix, (1.0, 0.0, 0.0, 1.0, x, y))
            target_tag = local_name(target.tag)
            if target_tag in ("symbol", "svg"):
                inner_box = parse_numbers(target.get("viewBox"))
                inner_box = inner_box[:4] if len(inner_box) >= 4 else None
                use_w = parse_length(element.get("width"), default=None)
                use_h = parse_length(element.get("height"), default=None)
                if inner_box:
                    child_matrix = mat_mul(child_matrix, viewbox_transform(
                        inner_box,
                        use_w if use_w else inner_box[2],
                        use_h if use_h else inner_box[3],
                        target.get("preserveAspectRatio")))
                for child in target:
                    visit(child, child_matrix, layers + [_group_name(element)]
                          if _group_name(element) else layers,
                          visibility, use_depth + 1, seen | {id(target)})
            else:
                visit(target, child_matrix, layers, visibility,
                      use_depth + 1, seen | {id(target)})
            return

        if tag == "svg" and element is not root:
            x = parse_length(element.get("x"), default=0.0) or 0.0
            y = parse_length(element.get("y"), default=0.0) or 0.0
            inner_box = parse_numbers(element.get("viewBox"))
            inner_box = inner_box[:4] if len(inner_box) >= 4 else None
            matrix = mat_mul(matrix, (1.0, 0.0, 0.0, 1.0, x, y))
            if inner_box:
                inner_w = parse_length(element.get("width"), percent_of=inner_box[2],
                                       default=inner_box[2])
                inner_h = parse_length(element.get("height"), percent_of=inner_box[3],
                                       default=inner_box[3])
                matrix = mat_mul(matrix, viewbox_transform(
                    inner_box, inner_w, inner_h,
                    element.get("preserveAspectRatio")))

        if tag in GEOMETRY_TAGS:
            counts["elements"] += 1
            if visibility in ("hidden", "collapse"):
                counts["hidden"] += 1
                return
            if wanted is not None and tag not in wanted:
                return
            pieces = element_subpaths(element, tag, tolerance, on_error=warn)
            if not pieces:
                counts["empty"] += 1
                return
            flattened = []
            for segments, closed in pieces:
                # Beziers are affine invariant, so transforming the control
                # points is exact and lets the tolerance be measured in the
                # coordinates that are actually written.
                placed = []
                for seg in segments:
                    kind = seg[0]
                    moved = tuple(mat_point(matrix, p[0], p[1]) for p in seg[1:])
                    placed.append((kind,) + moved)
                points = flatten_segments(placed, tolerance)
                if len(points) >= 2:
                    flattened.append((points, closed))
            if not flattened:
                counts["empty"] += 1
                return
            if split_subpaths or len(flattened) == 1:
                total = len(flattened)
                for index, (points, closed) in enumerate(flattened):
                    curves.append(Curve(points, closed, tag, element.get("id"),
                                        classes, layers, index, total))
            else:
                merged = np.vstack([p for p, _c in flattened])
                closed = all(c for _p, c in flattened)
                curves.append(Curve(merged, closed, tag, element.get("id"),
                                    classes, layers))
            return

        if tag in CONTAINER_TAGS or element is root or tag == "svg":
            name = _group_name(element) if tag in ("g", "svg") else ""
            child_layers = layers + [name] if name else layers
            for child in element:
                visit(child, matrix, child_layers, visibility, use_depth, seen)

    for child in root:
        visit(child, root_matrix, [], "visible", 0, frozenset())

    info = {
        "path": real,
        "view_box": view_box,
        "width": width,
        "height": height,
        "space": space,
        "elements": counts["elements"],
        "hidden": counts["hidden"],
        "empty": counts["empty"],
        "warnings": warnings,
    }
    return curves, info


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------
def dedupe(points, closed, eps):
    """Drop consecutive duplicates; a repeated closing point is also removed.

    Curve It's writhe calculation rejects a polyline with a zero-length
    segment, and Illustrator does emit repeated points, so this is not
    optional cleanup.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep[1:] = step > eps
    pts = pts[keep]
    if closed and len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= eps:
        pts = pts[:-1]
    return pts


def polyline_length(points, closed):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    total = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    if closed:
        total += float(np.linalg.norm(pts[0] - pts[-1]))
    return total


def dedupe_rounded(points, closed, precision):
    """Drop points that would be written as an identical row.

    Coordinates are rounded on the way out, so two points closer together
    than the output quantum collapse into one row and leave a zero-length
    segment behind.  Curve It's writhe calculation rejects those outright and
    its discrete total curvature quietly loses the turning angle there, so
    the duplicate has to go before the file is written rather than after.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    grid = np.round(pts, int(precision))
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(grid[1:] != grid[:-1], axis=1)
    pts, grid = pts[keep], grid[keep]
    if closed and len(pts) > 1 and np.array_equal(grid[-1], grid[0]):
        pts = pts[:-1]
    return pts


def corner_flags(points, closed, angle_deg):
    """Mark vertices whose turning angle reaches `angle_deg`."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    flags = np.zeros(n, dtype=bool)
    if n < 3 or not angle_deg or float(angle_deg) <= 0.0:
        return flags
    before = np.roll(pts, 1, axis=0) if closed else np.vstack([pts[:1], pts[:-1]])
    after = np.roll(pts, -1, axis=0) if closed else np.vstack([pts[1:], pts[-1:]])
    incoming = pts - before
    outgoing = after - pts
    len_in = np.linalg.norm(incoming, axis=1)
    len_out = np.linalg.norm(outgoing, axis=1)
    usable = (len_in > 0.0) & (len_out > 0.0)
    cosine = np.ones(n)
    cosine[usable] = np.clip(
        np.einsum("ij,ij->i", incoming[usable], outgoing[usable])
        / (len_in[usable] * len_out[usable]), -1.0, 1.0)
    flags = np.degrees(np.arccos(cosine)) >= float(angle_deg)
    if not closed:
        flags[0] = flags[-1] = False        # the ends are kept regardless
    return flags


def _even_points(points, count):
    """`count` points evenly spaced by arc length, both ends included."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2 or count < 2:
        return pts[:1] if len(pts) else pts
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    if arc[-1] <= 0.0:
        return pts[:1]
    target = np.linspace(0.0, float(arc[-1]), int(count))
    return np.column_stack([np.interp(target, arc, pts[:, 0]),
                            np.interp(target, arc, pts[:, 1])])


def _allocate(lengths, total):
    """Split `total` points between spans in proportion to their length."""
    lengths = np.asarray(lengths, dtype=float)
    count = len(lengths)
    total = max(int(total), count)
    if lengths.sum() <= 0.0:
        share = np.ones(count, dtype=int)
    else:
        exact = lengths / lengths.sum() * total
        share = np.maximum(1, np.floor(exact).astype(int))
        order = np.argsort(-(exact - np.floor(exact)))
        while share.sum() < total:
            share[order[share.sum() % count]] += 1
        order = np.argsort(exact - np.floor(exact))
        index = 0
        while share.sum() > total and index < 100 * count:
            candidate = order[index % count]
            if share[candidate] > 1:
                share[candidate] -= 1
            index += 1
    return share


def resample(points, closed, n=None, spacing=None, min_corner_angle=None):
    """Evenly space points by arc length, keeping every corner exactly.

    Blind uniform resampling walks straight past a vertex and rounds the
    corners off a rectangle, a polygon or a star.  Curve It's discrete total
    curvature is exact for polygons, so a rounded corner is lost curvature.
    Corners are therefore pinned and each smooth span between them is
    resampled on its own, in proportion to its length.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    loop = np.vstack([pts, pts[0]]) if closed else pts
    step = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    total_length = float(step.sum())
    if total_length <= 0.0:
        return pts

    if n is None:
        if spacing is None or float(spacing) <= 0.0:
            return pts
        n = int(round(total_length / float(spacing)))
    n = max(int(n), 4 if closed else 2)

    corners = corner_flags(pts, closed, min_corner_angle)
    index = list(np.nonzero(corners)[0])
    if not index:
        target = (np.linspace(0.0, total_length, n, endpoint=False) if closed
                  else np.linspace(0.0, total_length, n))
        arc = np.concatenate([[0.0], np.cumsum(step)])
        return np.column_stack([np.interp(target, arc, loop[:, 0]),
                                np.interp(target, arc, loop[:, 1])])

    if closed:
        pts = np.roll(pts, -index[0], axis=0)
        index = [i - index[0] for i in index]
        loop = np.vstack([pts, pts[0]])
        bounds = index + [len(pts)]
        spans = [loop[bounds[k]:bounds[k + 1] + 1] for k in range(len(bounds) - 1)]
        unique_total = n
    else:
        bounds = [0] + index + [len(pts) - 1]
        spans = [pts[bounds[k]:bounds[k + 1] + 1] for k in range(len(bounds) - 1)]
        unique_total = n - 1

    spans = [s for s in spans if len(s) >= 2]
    if not spans:
        return pts
    lengths = [float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) for s in spans]
    share = _allocate(lengths, unique_total)

    out = [_even_points(span, int(count) + 1)[:-1]
           for span, count in zip(spans, share)]
    if not closed:
        out.append(spans[-1][-1:])
    return np.vstack(out)


def bounding_box(curves):
    stacked = np.vstack([c.points for c in curves])
    return stacked.min(axis=0), stacked.max(axis=0)


def _subdivide_to_four(points):
    """Split the longest edge of a triangle so a closed curve has 4 points.

    calculate_polyline_writhe needs four distinct points.  The new point sits
    on an existing edge, so the shape is untouched and a triangle drawn in
    Illustrator stays usable instead of being thrown away.
    """
    pts = np.asarray(points, dtype=float)
    edges = np.vstack([pts[1:], pts[:1]]) - pts
    longest = int(np.argmax(np.linalg.norm(edges, axis=1)))
    midpoint = pts[longest] + 0.5 * edges[longest]
    return np.insert(pts, longest + 1, midpoint, axis=0)


def prepare_curves(curves, scale=1.0, fit_size=None, flip_y=True,
                   center="bbox", closed_mode="auto", points=None,
                   spacing=None, min_points=2, min_length=0.0, z=0.0,
                   precision=DEFAULT_PRECISION,
                   min_corner_angle=DEFAULT_MIN_CORNER_ANGLE):
    """Apply the closed decision, y flip, scaling, centring and resampling.

    Returns a new list of Curve objects whose points are still (N, 2); the z
    column is added by write_xyz.
    """
    prepared = []
    for curve in curves:
        pts = np.array(curve.points, dtype=float)
        if closed_mode == "yes":
            closed = True
        elif closed_mode == "no":
            closed = False
        else:
            closed = curve.closed
            if not closed and len(pts) > 3:
                span = float(np.max(np.ptp(pts, axis=0)))
                if span > 0.0 and np.linalg.norm(pts[-1] - pts[0]) <= 1e-6 * span:
                    closed = True
        prepared.append(Curve(pts, closed, curve.tag, curve.element_id,
                              curve.classes, curve.layers,
                              curve.subpath, curve.subpath_count))

    for curve in prepared:
        span = float(np.max(np.ptp(curve.points, axis=0))) if len(curve.points) else 0.0
        curve.points = dedupe(curve.points, curve.closed,
                              max(1e-12, 1e-9 * span))

    prepared = [c for c in prepared if len(c.points) >= max(2, int(min_points))]
    if not prepared:
        return prepared

    if flip_y:
        for curve in prepared:
            curve.points = curve.points * np.array([1.0, -1.0])

    factor = float(scale)
    if fit_size is not None and float(fit_size) > 0.0:
        low, high = bounding_box(prepared)
        extent = float(np.max(high - low))
        factor = float(fit_size) / extent if extent > 0.0 else 1.0
    if factor != 1.0:
        for curve in prepared:
            curve.points = curve.points * factor

    if center in ("bbox", "centroid"):
        if center == "bbox":
            low, high = bounding_box(prepared)
            shift = (low + high) / 2.0
        else:
            shift = np.vstack([c.points for c in prepared]).mean(axis=0)
        for curve in prepared:
            curve.points = curve.points - shift

    if min_length and float(min_length) > 0.0:
        prepared = [c for c in prepared
                    if polyline_length(c.points, c.closed) >= float(min_length)]

    if points is not None or (spacing is not None and float(spacing) > 0.0):
        for curve in prepared:
            curve.points = resample(curve.points, curve.closed, n=points,
                                    spacing=spacing,
                                    min_corner_angle=min_corner_angle)

    # Last, because it is the written rows that must come out distinct.
    for curve in prepared:
        curve.points = dedupe_rounded(curve.points, curve.closed, precision)

    prepared = [c for c in prepared if len(c.points) >= max(2, int(min_points))]
    prepared = [c for c in prepared if not (c.closed and len(c.points) < 3)]
    for curve in prepared:
        if curve.closed and len(curve.points) == 3:
            curve.points = _subdivide_to_four(curve.points)
    return prepared


def filter_curves(curves, include=None, exclude=None):
    """Keep or drop curves by id, class name or enclosing group name."""
    def tokens(text):
        return [t.strip().lower() for t in str(text or "").replace(";", ",").split(",")
                if t.strip()]

    wanted = tokens(include)
    unwanted = tokens(exclude)
    out = curves
    if wanted:
        out = [c for c in out if any(n in wanted for n in c.names())]
    if unwanted:
        out = [c for c in out if not any(n in unwanted for n in c.names())]
    return out


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_xyz(path, curves, z=0.0, precision=DEFAULT_PRECISION,
              comments=True, source=None):
    """Write the plain-coordinate XYZ file Curve It reads."""
    if not curves:
        raise ValueError("there are no curves to write")
    fmt = "%%.%df %%.%df %%.%df\n" % (precision, precision, precision)
    # Guarantee the invariant here too: write_xyz is public, and a caller that
    # skipped prepare_curves must still not produce duplicate rows.
    blocks = [(curve, dedupe_rounded(curve.points, curve.closed, precision))
              for curve in curves]
    with open(path, "w") as fh:
        if comments:
            fh.write("# %s v%s -- %d component(s) from %s\n"
                     % (TOOL_NAME, __version__, len(blocks),
                        _flat(os.path.basename(source or path))))
            fh.write("# columns: x y z ; blank lines separate components\n")
        for index, (curve, pts) in enumerate(blocks):
            if index:
                fh.write("\n")
            if comments:
                fh.write("# component %s: %s closed=%s points=%d\n"
                         % (component_label(index), curve.source(),
                            "yes" if curve.closed else "no", len(pts)))
            for x, y in pts:
                fh.write(fmt % (x, y, z))
    return path


def write_xyz_split(prefix, curves, z=0.0, precision=DEFAULT_PRECISION,
                    comments=True, source=None):
    written = []
    for index, curve in enumerate(curves):
        out = "%s_%s.xyz" % (prefix, component_label(index))
        written.append(write_xyz(out, [curve], z, precision, comments, source))
    return written


def describe(curves, info=None, z=0.0):
    """A readable summary of what was found and what will be written."""
    lines = []
    if info:
        box = info.get("view_box")
        lines.append("SOURCE")
        lines.append("  file        %s" % os.path.basename(info.get("path", "")))
        lines.append("  viewBox     %s" % ("%g %g %g %g" % tuple(box) if box else "(none)"))
        lines.append("  width x h   %s" % (
            "%g x %g" % (info["width"], info["height"])
            if info.get("width") and info.get("height") else "(not given)"))
        lines.append("  space       %s" % info.get("space", "user"))
        found = "  shapes      %d drawable element(s)" % info.get("elements", 0)
        if info.get("hidden"):
            found += ", %d hidden" % info["hidden"]
        if info.get("empty"):
            found += ", %d with no geometry" % info["empty"]
        lines.append(found)
        lines.append("")

    if not curves:
        lines.append("CURVES\n  none -- nothing matched the current settings")
        return "\n".join(lines)

    total = sum(len(c.points) for c in curves)
    lines.append("CURVES  (%d component(s), %d points, z = %g)" % (len(curves), total, z))
    if info and info.get("selected"):
        picked = info["selected"]
        note = "  selected %s of %d" % (",".join(picked), info["selected_from"])
        rewritten = [component_label(i) for i in range(len(picked))]
        if rewritten != picked:
            note += "; written as %s" % ",".join(rewritten)
        lines.append(note)
    width = max(len(c.source()) for c in curves)
    width = min(max(width, 10), 58)
    for index, curve in enumerate(curves):
        lines.append("  %-3s %-*s  %-6s %6d pts  len %12.4f"
                     % (component_label(index), width, curve.source()[:width],
                        "closed" if curve.closed else "open",
                        len(curve.points),
                        polyline_length(curve.points, curve.closed)))
    low, high = bounding_box(curves)
    lines.append("")
    lines.append("EXTENT")
    lines.append("  x  %12.4f .. %12.4f   (%.4f)" % (low[0], high[0], high[0] - low[0]))
    lines.append("  y  %12.4f .. %12.4f   (%.4f)" % (low[1], high[1], high[1] - low[1]))
    lines.append("  z  %12.4f .. %12.4f   (0.0000)" % (z, z))

    if info and info.get("warnings"):
        lines.append("")
        lines.append("NOTES")
        for message in info["warnings"]:
            lines.append("  %s" % message)

    if len(curves) > 1:
        mixed = len({c.closed for c in curves}) > 1
        lines.append("")
        lines.append("USING THIS FILE IN CURVE IT")
        lines.append("  Curve It concatenates every component by default, which would")
        lines.append("  join these %d curves into one path. Select one at a time:" % len(curves))
        for index in range(min(len(curves), 3)):
            lines.append("    python3 curve_it.py in.pdb out.xyz --curve-components %s "
                         "--path-type %s"
                         % (component_label(index),
                            "closed" if curves[index].closed else "open"))
        if len(curves) > 3:
            lines.append("    ... and so on through component %s"
                         % component_label(len(curves) - 1))
        lines.append("  In the GUI, use Select components... beside the curve file.")
        if mixed:
            lines.append("  Path type is one setting for the whole run, and these")
            lines.append("  components are not all closed, so convert them separately.")
    return "\n".join(lines)


def render_preview(curves, path=None, colors=None, z=0.0):
    """A flat xy plot of the curves, enough to confirm the conversion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = colors or DEFAULT_COLORS
    if path is None:
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "svg2xyz_preview.png")

    fig = plt.figure(figsize=(7.6, 7.6))
    ax = fig.add_subplot(1, 1, 1)
    for index, curve in enumerate(curves):
        pts = curve.points
        drawn = np.vstack([pts, pts[0]]) if curve.closed else pts
        color = colors[index % len(colors)]
        ax.plot(drawn[:, 0], drawn[:, 1], lw=1.4, color=color,
                label="%s  %s" % (component_label(index),
                                  "closed" if curve.closed else "open"))
        ax.plot([pts[0, 0]], [pts[0, 1]], "o", ms=5, color=color)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("svg2xyz -- %d component(s) at z = %g" % (len(curves), z), fontsize=10)
    if len(curves) <= 14:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="svg2xyz.py",
        description="Convert the curves in a 2D SVG into 3D XYZ space curves "
                    "with a constant z.",
        epilog="Run with no arguments for the GUI.")
    p.add_argument("svg", nargs="?", help="input .svg file")
    p.add_argument("--gui", action="store_true", help="force the graphical interface")

    p.add_argument("-o", "--output",
                   help="output .xyz path (default: the input name with .xyz)")
    p.add_argument("--split", action="store_true",
                   help="also write one .xyz per component, named _A, _B, ...")

    p.add_argument("-t", "--tolerance", type=float, default=0.05,
                   help="Bezier/arc flattening tolerance in SVG units; smaller "
                        "means more points (default 0.05)")
    p.add_argument("-n", "--points", type=int,
                   help="resample every curve to this many evenly spaced points "
                        "(default: keep the adaptively flattened points)")
    p.add_argument("--spacing", type=float,
                   help="resample every curve to this arc-length step, in final "
                        "output units; an alternative to --points")
    p.add_argument("--min-corner-angle", type=float, default=DEFAULT_MIN_CORNER_ANGLE,
                   help="smallest turning angle, in degrees, that counts as a "
                        "corner; when resampling, corners are kept exactly and "
                        "gentler vertices are not. 0 keeps none (default %g)"
                        % DEFAULT_MIN_CORNER_ANGLE)

    p.add_argument("-s", "--scale", type=float, default=1.0,
                   help="multiply every output coordinate by this (default 1.0)")
    p.add_argument("--fit-size", type=float,
                   help="scale the whole drawing so its largest dimension "
                        "equals this; overrides --scale")
    p.add_argument("-z", "--z", type=float, default=0.0, dest="z",
                   help="z coordinate given to every point (default 0.0)")
    p.add_argument("--closed", choices=["auto", "yes", "no"], default="auto",
                   help="treat curves as closed loops (default: as drawn)")
    p.add_argument("--center", choices=["bbox", "centroid", "none"], default="bbox",
                   help="move the drawing onto the origin (default bbox)")
    p.add_argument("--no-flip-y", action="store_true",
                   help="keep the SVG's y-down coordinates instead of flipping "
                        "them so the drawing comes out upright")
    p.add_argument("--space", choices=["user", "viewport"], default="user",
                   help="user units, or rendered pixels after applying the root "
                        "viewBox mapping (default user)")

    p.add_argument("-c", "--components", default="all",
                   help="which components to keep, by the labels --info lists: "
                        "A, or B,C, or A-C, or all. Applied after every other "
                        "filter, so the labels are the ones you just saw "
                        "(default all)")
    p.add_argument("--include",
                   help="keep only curves whose id, class or group name is in "
                        "this comma-separated list")
    p.add_argument("--exclude",
                   help="drop curves whose id, class or group name is in this "
                        "comma-separated list")
    p.add_argument("--elements", default="all",
                   help="element types to read: all, or a comma-separated "
                        "subset of %s" % ", ".join(GEOMETRY_TAGS))
    p.add_argument("--min-points", type=int, default=2,
                   help="drop curves with fewer points than this (default 2)")
    p.add_argument("--min-length", type=float, default=0.0,
                   help="drop curves shorter than this, in output units")
    p.add_argument("--no-split-subpaths", action="store_true",
                   help="keep every subpath of one element in a single "
                        "component instead of splitting them")

    p.add_argument("--precision", type=int, default=DEFAULT_PRECISION,
                   help="decimal places written per coordinate (default 6)")
    p.add_argument("--no-comments", action="store_true",
                   help="omit the '#' header naming each component's source")
    p.add_argument("--preview", action="store_true", help="write a PNG preview")
    p.add_argument("--info", action="store_true",
                   help="describe the drawing and exit without writing")
    return p


def _convert(args):
    """Shared read/filter/prepare pipeline for the CLI and the GUI."""
    resampling = (args.points is not None
                  or (args.spacing is not None and float(args.spacing) > 0.0))
    # Resampling measures arc length off the flattened polyline, so flatten
    # finer than asked when it is going to be resampled anyway.
    tolerance = args.tolerance / DENSE_FACTOR if resampling else args.tolerance

    curves, info = extract_curves(
        args.svg,
        tolerance=tolerance,
        space=args.space,
        split_subpaths=not args.no_split_subpaths,
        elements=args.elements,
    )
    curves = filter_curves(curves, args.include, args.exclude)
    curves = prepare_curves(
        curves,
        scale=args.scale,
        fit_size=args.fit_size,
        flip_y=not args.no_flip_y,
        center=args.center,
        closed_mode=args.closed,
        points=args.points,
        spacing=args.spacing,
        min_points=args.min_points,
        min_length=args.min_length,
        z=args.z,
        precision=args.precision,
        min_corner_angle=args.min_corner_angle,
    )

    # Last of all, so the labels are exactly the ones the listing showed.
    # Scaling and centring have already been resolved over the whole drawing,
    # which is what keeps separately extracted components in register.
    selection = parse_component_selection(getattr(args, "components", None),
                                          len(curves))
    if len(selection) != len(curves):
        info["selected"] = [component_label(i) for i in selection]
        info["selected_from"] = len(curves)
        curves = [curves[i] for i in selection]
    return curves, info


def run_cli(args):
    if args.points is not None and args.spacing is not None:
        raise ValueError("use either --points or --spacing, not both")
    if args.points is not None and args.points < 2:
        raise ValueError("--points needs at least 2")
    if args.fit_size is not None and args.scale != 1.0:
        print("  note: --fit-size overrides --scale")

    curves, info = _convert(args)
    print("Loaded %s" % info["path"])
    print(describe(curves, info, args.z))
    if args.info:
        return 0
    if not curves:
        raise ValueError("no curves survived the current settings; "
                         "check --include / --exclude / --elements")

    prefix = os.path.splitext(args.output)[0] if args.output \
        else os.path.splitext(info["path"])[0]
    out = args.output or (prefix + ".xyz")

    written = [write_xyz(out, curves, args.z, args.precision,
                         not args.no_comments, info["path"])]
    if args.split:
        written += write_xyz_split(prefix, curves, args.z, args.precision,
                                   not args.no_comments, info["path"])
    if args.preview:
        written.append(render_preview(curves, prefix + "_preview.png", z=args.z))
    print("\nWROTE")
    for item in written:
        print("  %s" % item)
    return 0


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def run_gui(initial_file=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("svg2xyz  -  a 2D SVG drawing  ->  3D XYZ curves   v%s" % __version__)
    root.minsize(940, 600)
    set_optional_window_icon(
        root, tk, ["svg2xyz_icon.png", "icon.png"], "_svg2xyz_icon_image")

    state = {"curves": None, "info": None, "path": None}
    V = {
        "file":      tk.StringVar(value=initial_file or ""),
        "tolerance": tk.StringVar(value="0.05"),
        "points":    tk.StringVar(value="auto"),
        "spacing":   tk.StringVar(value=""),
        "corner":    tk.StringVar(value="%g" % DEFAULT_MIN_CORNER_ANGLE),
        "scale":     tk.StringVar(value="1.0"),
        "fit":       tk.StringVar(value=""),
        "z":         tk.StringVar(value="0.0"),
        "closed":    tk.StringVar(value="auto"),
        "center":    tk.StringVar(value="bbox"),
        "space":     tk.StringVar(value="user"),
        "flip":      tk.BooleanVar(value=True),
        "components": tk.StringVar(value="all"),
        "include":   tk.StringVar(value=""),
        "exclude":   tk.StringVar(value=""),
        "elements":  tk.StringVar(value="all"),
        "minpoints": tk.StringVar(value="2"),
        "minlength": tk.StringVar(value="0"),
        "subpaths":  tk.BooleanVar(value=True),
        "outdir":    tk.StringVar(value=""),
        "basename":  tk.StringVar(value="curves"),
        "split":     tk.BooleanVar(value=False),
        "comments":  tk.BooleanVar(value=True),
        "png":       tk.BooleanVar(value=False),
    }

    # ---- "?" help chips ---------------------------------------------------
    open_help = {"win": None}

    def show_help(key, near):
        title, prose, example = HELP[key]
        if open_help["win"] is not None:
            try:
                open_help["win"].destroy()
            except Exception:               # noqa: BLE001
                pass
        top = tk.Toplevel(root)
        open_help["win"] = top
        top.title(title)
        top.transient(root)
        top.configure(bg="#f4f9ff")
        frm = tk.Frame(top, bg="#f4f9ff", padx=16, pady=14)
        frm.pack(fill="both", expand=True)
        tk.Label(frm, text=title, font=("Helvetica", 14, "bold"),
                 bg="#f4f9ff", fg="#0b4d80", anchor="w",
                 justify="left").pack(fill="x", pady=(0, 8))
        tk.Label(frm, text=prose, wraplength=470, justify="left",
                 bg="#f4f9ff", fg="#1a1a1a", font=("Helvetica", 12),
                 anchor="w").pack(fill="x")
        if example:
            tk.Label(frm, text="Example", font=("Helvetica", 11, "bold"),
                     bg="#f4f9ff", fg="#0b4d80", anchor="w").pack(fill="x",
                                                                  pady=(12, 3))
            box = tk.Text(frm, font=("Menlo", 11), height=example.count("\n") + 1,
                          width=62, bg="#e8f1fa", relief="flat", padx=8, pady=6,
                          highlightthickness=0)
            box.insert("1.0", example)
            box.configure(state="disabled")
            box.pack(fill="x")
        tk.Button(frm, text="Close", command=top.destroy).pack(pady=(12, 0))
        top.bind("<Escape>", lambda e: top.destroy())
        try:
            x = near.winfo_rootx() + 26
            y = near.winfo_rooty() - 10
            top.geometry("+%d+%d" % (max(x, 0), max(y, 0)))
        except Exception:                   # noqa: BLE001
            pass
        top.focus_set()

    def chip(parent, key):
        """A small light-blue '?' that opens the explanation for `key`.

        Built from a Label rather than a Button because macOS's native button
        ignores background colour.
        """
        c = tk.Label(parent, text="?", font=("Helvetica", 11, "bold"),
                     bg="#bcdcf5", fg="#0b4d80", width=2, relief="raised",
                     borderwidth=1, cursor="hand2")
        c.bind("<Button-1>", lambda e, k=key, w=c: show_help(k, w))
        c.bind("<Enter>", lambda e, w=c: w.configure(bg="#8fc7ee"))
        c.bind("<Leave>", lambda e, w=c: w.configure(bg="#bcdcf5"))
        return c

    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(0, weight=1)
    left = ttk.Frame(main)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    right = ttk.Frame(main)
    right.grid(row=0, column=1, sticky="nsew")
    right.rowconfigure(1, weight=1)
    right.columnconfigure(0, weight=1)

    # input -----------------------------------------------------------------
    fi = ttk.LabelFrame(left, text="Input", padding=8)
    fi.pack(fill="x", pady=(0, 8))
    fi.columnconfigure(0, weight=1)
    ttk.Entry(fi, textvariable=V["file"], width=40).grid(row=0, column=0, sticky="ew")

    def browse():
        p = filedialog.askopenfilename(
            title="Select an SVG drawing",
            filetypes=[("SVG drawings", "*.svg"), ("All files", "*.*")])
        if p:
            V["file"].set(p)
            do_load()

    ttk.Button(fi, text="Browse...", command=browse).grid(row=0, column=1, padx=(6, 0))
    chip(fi, "file").grid(row=0, column=2, padx=(6, 0))
    lbl = ttk.Label(fi, text="no file loaded", foreground="#888")
    lbl.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
    chip(fi, "curves").grid(row=1, column=2, padx=(6, 0), pady=(6, 0))

    # sampling --------------------------------------------------------------
    fs = ttk.LabelFrame(left, text="Sampling", padding=8)
    fs.pack(fill="x", pady=(0, 8))
    sampling_rows = [("flattening tolerance", "tolerance", "SVG units, before scaling"),
                     ("points per curve", "points", "'auto' keeps corners exact"),
                     ("point spacing", "spacing", "output units; overrides count"),
                     ("min corner angle", "corner", "degrees; sharper vertices pinned")]
    for r, (label, key, hint) in enumerate(sampling_rows):
        ttk.Label(fs, text=label).grid(row=r, column=0, sticky="w")
        ttk.Entry(fs, textvariable=V[key], width=10).grid(row=r, column=1,
                                                          sticky="w", padx=(4, 0))
        chip(fs, key).grid(row=r, column=2, padx=(6, 0))
        tk.Label(fs, text=hint, font=("Helvetica", 10), fg="#8a929b").grid(
            row=r, column=3, sticky="w", padx=(8, 0))
    ttk.Checkbutton(fs, text="split subpaths into separate curves",
                    variable=V["subpaths"]).grid(row=4, column=0, columnspan=2,
                                                 sticky="w", pady=(4, 0))
    chip(fs, "subpaths").grid(row=4, column=2, padx=(6, 0), pady=(4, 0))

    # geometry --------------------------------------------------------------
    fg = ttk.LabelFrame(left, text="Geometry", padding=8)
    fg.pack(fill="x", pady=(0, 8))
    geometry_rows = [("scale factor", "scale", "x every coordinate"),
                     ("fit size", "fit", "largest dimension; wins over scale"),
                     ("z value", "z", "the plane the curve lies in")]
    for r, (label, key, hint) in enumerate(geometry_rows):
        ttk.Label(fg, text=label).grid(row=r, column=0, sticky="w")
        ttk.Entry(fg, textvariable=V[key], width=10).grid(row=r, column=1,
                                                          sticky="w", padx=(4, 0))
        chip(fg, {"fit": "fit"}.get(key, key)).grid(row=r, column=2, padx=(6, 0))
        tk.Label(fg, text=hint, font=("Helvetica", 10), fg="#8a929b").grid(
            row=r, column=3, sticky="w", padx=(8, 0))
    combo_rows = [("closed curves", "closed", ["auto", "yes", "no"]),
                  ("centre on origin", "center", ["bbox", "centroid", "none"]),
                  ("coordinate space", "space", ["user", "viewport"])]
    for r, (label, key, values) in enumerate(combo_rows, start=3):
        ttk.Label(fg, text=label).grid(row=r, column=0, sticky="w")
        ttk.Combobox(fg, textvariable=V[key], width=9, state="readonly",
                     values=values).grid(row=r, column=1, sticky="w", padx=(4, 0))
        chip(fg, key).grid(row=r, column=2, padx=(6, 0))
    ttk.Checkbutton(fg, text="flip the y axis (SVG y points down)",
                    variable=V["flip"]).grid(row=6, column=0, columnspan=2,
                                             sticky="w", pady=(4, 0))
    chip(fg, "flip").grid(row=6, column=2, padx=(6, 0), pady=(4, 0))

    # filters ---------------------------------------------------------------
    ff = ttk.LabelFrame(left, text="Which curves", padding=8)
    ff.pack(fill="x", pady=(0, 8))
    ff.columnconfigure(1, weight=1)
    filter_rows = [("components", "components", "components"),
                   ("include", "include", "filters"), ("exclude", "exclude", "filters"),
                   ("element types", "elements", "elements"),
                   ("min points", "minpoints", "minpoints"),
                   ("min length", "minlength", "minpoints")]
    for r, (label, key, topic) in enumerate(filter_rows):
        ttk.Label(ff, text=label).grid(row=r, column=0, sticky="w")
        ttk.Entry(ff, textvariable=V[key], width=24).grid(row=r, column=1,
                                                          sticky="ew", padx=(4, 0))
        chip(ff, topic).grid(row=r, column=2, padx=(6, 0))

    # output ----------------------------------------------------------------
    fo = ttk.LabelFrame(left, text="Output", padding=8)
    fo.pack(fill="x")
    fo.columnconfigure(1, weight=1)
    ttk.Label(fo, text="folder").grid(row=0, column=0, sticky="w")
    ttk.Entry(fo, textvariable=V["outdir"], width=24).grid(row=0, column=1, sticky="ew")
    ttk.Button(fo, text="...", width=3,
               command=lambda: V["outdir"].set(filedialog.askdirectory()
                                               or V["outdir"].get())
               ).grid(row=0, column=2, padx=(4, 0))
    chip(fo, "output").grid(row=0, column=3, padx=(6, 0))
    ttk.Label(fo, text="base name").grid(row=1, column=0, sticky="w", pady=(4, 0))
    ttk.Entry(fo, textvariable=V["basename"], width=24).grid(row=1, column=1,
                                                             sticky="ew", pady=(4, 0))
    opts = ttk.Frame(fo)
    opts.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
    for i, (label, key) in enumerate([("one file per curve", "split"),
                                      ("comment header", "comments"),
                                      ("PNG preview", "png")]):
        ttk.Checkbutton(opts, text=label, variable=V[key]).grid(
            row=i // 2, column=i % 2, sticky="w", padx=(0, 10))

    # right pane ------------------------------------------------------------
    bar = ttk.Frame(right)
    bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    ttk.Button(bar, text="Reload", command=lambda: do_load()).pack(side="left")
    ttk.Button(bar, text="Preview", command=lambda: do_preview()).pack(side="left", padx=6)
    ttk.Button(bar, text="Write XYZ", command=lambda: do_write()).pack(side="left")
    txt = tk.Text(right, wrap="none", font=("Menlo", 11), height=26)
    txt.grid(row=1, column=0, sticky="nsew")
    sb = ttk.Scrollbar(right, orient="vertical", command=txt.yview)
    sb.grid(row=1, column=1, sticky="ns")
    txt.configure(yscrollcommand=sb.set)
    status = ttk.Label(right, text="", foreground="#0a7")
    status.grid(row=2, column=0, sticky="w", pady=(4, 0))

    def show(s):
        txt.delete("1.0", "end")
        txt.insert("1.0", s)

    def fnum(key, default=None):
        text = V[key].get().strip()
        if text.lower() in ("", "auto", "none") and default is not None:
            return default
        try:
            return float(text)
        except ValueError:
            raise ValueError("%s: %r is not a number" % (key, text))

    def optional(key):
        text = V[key].get().strip()
        if text.lower() in ("", "auto", "none"):
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError("%s: %r is not a number" % (key, text))

    def gui_args():
        points = optional("points")
        spacing = optional("spacing")
        if points is not None and spacing is not None:
            raise ValueError("set either points per curve or point spacing, not both")
        return argparse.Namespace(
            svg=V["file"].get().strip(),
            tolerance=fnum("tolerance", 0.05),
            points=int(points) if points is not None else None,
            spacing=spacing,
            scale=fnum("scale", 1.0),
            fit_size=optional("fit"),
            z=fnum("z", 0.0),
            closed=V["closed"].get(),
            center=V["center"].get(),
            space=V["space"].get(),
            no_flip_y=not V["flip"].get(),
            components=V["components"].get().strip() or "all",
            include=V["include"].get().strip(),
            exclude=V["exclude"].get().strip(),
            elements=V["elements"].get().strip() or "all",
            min_points=int(fnum("minpoints", 2)),
            min_length=fnum("minlength", 0.0),
            no_split_subpaths=not V["subpaths"].get(),
            precision=DEFAULT_PRECISION,
            min_corner_angle=fnum("corner", DEFAULT_MIN_CORNER_ANGLE),
        )

    def do_load():
        path = V["file"].get().strip()
        if not path:
            return
        try:
            curves, info = _convert(gui_args())
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Load failed", str(exc))
            status.configure(text="load failed", foreground="#c00")
            return
        state.update(curves=curves, info=info, path=info["path"])
        lbl.configure(text="%d curve(s), %d points total"
                          % (len(curves), sum(len(c.points) for c in curves)),
                      foreground="#333")
        if not V["outdir"].get():
            V["outdir"].set(os.path.dirname(info["path"]))
        if V["basename"].get() in ("", "curves"):
            V["basename"].set(os.path.splitext(os.path.basename(info["path"]))[0])
        show(describe(curves, info, fnum("z", 0.0)))
        status.configure(text="loaded -- %d curve(s)" % len(curves), foreground="#0a7")

    def do_preview():
        if state["curves"] is None:
            do_load()
        if not state["curves"]:
            return
        try:
            png = render_preview(state["curves"], None, z=fnum("z", 0.0))
            top = tk.Toplevel(root)
            top.title("Preview")
            try:
                from PIL import Image, ImageTk
                im = Image.open(png)
                w = min(900, im.width)
                im = im.resize((w, int(im.height * w / im.width)))
                img = ImageTk.PhotoImage(im)
            except Exception:               # noqa: BLE001
                img = tk.PhotoImage(file=png)
            holder = tk.Label(top, image=img)
            holder.image = img
            holder.pack()
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Preview failed", str(exc))

    def do_write():
        do_load()
        curves = state["curves"]
        if not curves:
            messagebox.showwarning(
                "Nothing to write",
                "No curves survived the current settings.\n\n"
                "Check the include / exclude and element type fields.")
            return
        try:
            z = fnum("z", 0.0)
            folder = V["outdir"].get() or os.path.dirname(state["path"])
            prefix = os.path.join(folder, V["basename"].get() or "curves")
            comments = V["comments"].get()
            written = [write_xyz(prefix + ".xyz", curves, z, DEFAULT_PRECISION,
                                 comments, state["path"])]
            if V["split"].get():
                written += write_xyz_split(prefix, curves, z, DEFAULT_PRECISION,
                                           comments, state["path"])
            if V["png"].get():
                written.append(render_preview(curves, prefix + "_preview.png", z=z))
            report = describe(curves, state["info"], z)
            report += "\n\nWROTE\n" + "\n".join("  " + w for w in written)
            report += ("\n\nLoad %s in Curve It as the Curve XYZ/txt input.\n"
                       % os.path.basename(written[0]))
            report += ("Its %d component(s) appear as %s in the component selector."
                       % (len(curves),
                          ", ".join(component_label(i) for i in range(min(len(curves), 6)))
                          + (", ..." if len(curves) > 6 else "")))
            show(report)
            status.configure(text="done -- %d file(s)" % len(written), foreground="#0a7")
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Write failed", str(exc))
            status.configure(text="failed", foreground="#c00")

    if initial_file:
        root.after(100, do_load)
    else:
        show("Choose an SVG drawing to begin.\n\n"
             "Every drawn path, polyline, polygon, line, rect, circle and\n"
             "ellipse becomes one component of a plain-coordinate XYZ file:\n\n"
             "    x y z         <- component A\n"
             "    x y z\n"
             "                  <- a blank line separates components\n"
             "    x y z         <- component B\n\n"
             "The drawing is flat, so every z is the same, 0 by default.\n\n"
             "Order of operations:\n"
             "    SVG geometry\n"
             "      + transforms          <- groups and elements, composed exactly\n"
             "      -> flattened curves   <- adaptive, at the tolerance\n"
             "      x  y flip             <- SVG y points down the page\n"
             "      x  scale factor\n"
             "      -  centre             <- onto the origin\n"
             "      -> XYZ components\n\n"
             "Click any light-blue  ?  for an explanation and examples.\n")

    root.after(120, lambda: (root.lift(), root.attributes("-topmost", True),
                             root.after(400, lambda: root.attributes("-topmost", False))))
    root.mainloop()
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        return run_gui()
    args = parser.parse_args(argv)
    if args.gui:
        return run_gui(args.svg)
    if not args.svg:
        parser.error("an input .svg file is required (or use --gui)")
    try:
        return run_cli(args)
    except FileNotFoundError as exc:
        print("error: cannot open %s" % exc.filename, file=sys.stderr)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
