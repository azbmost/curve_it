#!/usr/bin/env python3
"""
xyz2model.py -- sweep a rod along the curves in an .xyz file and write STL / GLB.

Reads an .xyz file holding one or more space curves and turns each into a
round tube, then writes

    * a binary STL -- every component in one multi-shell file, ready to print
    * a binary GLB -- one separately coloured mesh per component, for rendering

Accepted input: blocks of "x y z" or "element x y z" separated by blank lines,
standard XYZ files with a count + comment header, and macOS Finder aliases to
any of those. Closed loops and open strands are both handled; open ends get
rounded caps.

Order of operations -- --scale is applied to the centre-line, then the rod is
swept, so --diameter is in FINAL output units and --scale does not change it:

    input coordinates  x  --scale  ->  centre-line  +  --diameter  ->  solid

    python3 xyz2model.py                                   # GUI
    python3 xyz2model.py curves.xyz --info                 # just describe it
    python3 xyz2model.py curves.xyz -d 2.0 -s 0.25
    python3 xyz2model.py curves.xyz -d 1.5 --colors "#e6194b,#3cb44b,#4363d8"

Requires numpy, scipy, trimesh (and matplotlib for --preview / colour names).
Run with no arguments, or with --gui, for the graphical interface.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

# Optional Geomview VECT support.  VECT is the one curve format that states
# per component whether it is a closed loop, so a .vect input answers --closed
# instead of leaving it to the looks_closed heuristic below.
try:
    from . import vect_io
except ImportError:
    try:
        import vect_io  # type: ignore[no-redef]
    except ImportError:
        vect_io = None  # type: ignore[assignment]

__version__ = "1.0"

TOOL_NAME = "XYZ to 3D Model"


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


# --------------------------------------------------------------------------
# GUI help text: key -> (title, prose, example or None)
# --------------------------------------------------------------------------
HELP = {
    "file": (
        "Input curve file",
        "One or more space curves. Components are separated by blank lines, or "
        "by the count + comment header of a standard XYZ frame. Each line may be "
        "\"x y z\" or \"element x y z\". macOS Finder aliases are followed to the "
        "original file.\n\n"
        "A Geomview VECT file (.vect) is also accepted, recognised by its header "
        "word rather than its extension. Each of its polylines is one component, "
        "and because VECT states which of them are closed loops, a VECT input "
        "answers Closed curves by itself instead of leaving it to the "
        "last-point-to-first measurement.\n\n"
        "An STL / OBJ / PLY / GLB / 3MF / OFF mesh is also accepted -- see the "
        "Input mode help. A mesh is only scaled.",
        "x y z form              element form           standard XYZ\n"
        "  -47.604 -36.487 -2.4    C  -47.604 ...        199\n"
        "  -47.755 -36.209 -4.1    C  -47.755 ...        frame 1\n"
        "  <blank line>            <blank line>          C -47.604 ...\n"
        "  12.964  -56.745  6.4    C   12.964 ...        ..."),

    "size": (
        "Model size",
        "How big the finished object actually is.\n\n"
        "centre-line extent -- the span of the curves themselves, after the "
        "scale factor. This is NOT the size of the printed part.\n\n"
        "MODEL SIZE -- the span of the solid. A rod of radius r around a curve "
        "reaches r beyond it in every direction, so the model is exactly one "
        "rod diameter larger than the centre-line on each of the three axes. "
        "Rounded end caps are already included. This is the number to compare "
        "against your build volume.\n\n"
        "measured size -- read back off the finished mesh after it is built. "
        "It comes out a hair under MODEL SIZE because a 24-sided tube is "
        "inscribed in the true circle rather than touching it.",
        "centre-line extent   28.496 x 28.109 x 11.038\n"
        "rod diameter          2.000\n"
        "MODEL SIZE           30.496 x 30.109 x 13.038\n"
        "                     ^ 28.496 + 2.000, and so on"),

    "mode": (
        "Input mode",
        "curves -- an .xyz file of space curves. A rod of the diameter you "
        "choose is swept along each one to make a solid. All the geometry "
        "fields apply.\n\n"
        "mesh -- an STL (or OBJ, PLY, GLB, 3MF, OFF). The solid already exists, "
        "so the only thing this tool does is SCALE it. Rod diameter, segments, "
        "facets and the closed-curve setting are all meaningless and are "
        "greyed out.\n\n"
        "The mode is picked from the file extension, then from the file's "
        "first bytes, so an STL saved as .dat is still recognised. Override "
        "with --as on the command line.",
        "curves in   ->  scale, sweep a rod  ->  STL + GLB\n"
        "mesh in     ->  scale               ->  STL + GLB"),

    "scale": (
        "Scale factor",
        "Multiplies every input coordinate. It is applied to the centre-line "
        "BEFORE the rod is swept, so it changes the overall size of the model "
        "but NOT the rod thickness.\n\n"
        "Use it when your coordinates are in some arbitrary or non-mm unit and "
        "you want millimetres out.\n\n"
        "For a mesh input this is the only thing that happens: every vertex is "
        "multiplied by it, so lengths scale by s, areas by s^2 and volume by "
        "s^3.",
        "this project's link is ~118 units across\n"
        "    scale 1.0      -> 118 units across (unchanged)\n"
        "    scale 0.24185  ->  31 mm across  (the size-7 ring)\n"
        "    scale 0.31672  ->  41 mm across  (the US 13.5 thumb ring)"),

    "diameter": (
        "Rod diameter  --  AFTER scaling",
        "The thickness of the rod swept along each curve, in FINAL OUTPUT "
        "UNITS. It is applied after the scale factor, so the scale factor does "
        "not change it: type 2.0 and you get a 2.0 mm rod at any scale.\n\n"
        "This is the one number you almost always want in real-world units.",
        "a straight 10-unit curve, rod diameter 2.0\n\n"
        "  scale   model length   rod thickness\n"
        "   0.5        7.0            2.0\n"
        "   1.0       12.0            2.0\n"
        "   2.0       22.0            2.0\n\n"
        "(length includes the two rounded end caps)"),

    "segments": (
        "Segments per curve",
        "How many points each curve is resampled to before the rod is swept. "
        "More segments means a smoother tube and a bigger file.\n\n"
        "'auto' picks about one segment per eighth of a rod diameter, which is "
        "smooth enough that facets are invisible at print resolution.\n\n"
        "Enter 0 to skip resampling entirely and sweep through your input "
        "points exactly as given -- use this when the coordinates are already "
        "final and must not be smoothed.",
        "auto   -> ~356 for this project's link at a 2 mm rod\n"
        "640    -> what the ring models were built with\n"
        "0      -> no resampling; the 199 input points are used as-is"),

    "sides": (
        "Facets around the rod",
        "How many flat faces make up the tube's circular cross-section. "
        "24 is plenty for printing: on a 2 mm rod the resulting bulge is under "
        "7 microns, far below any printer's resolution.\n\n"
        "Raise it only for close-up renders. Cost is linear in file size.",
        "rod 2.0 mm      error vs a true circle\n"
        "  12 sides          0.068 mm\n"
        "  24 sides          0.017 mm\n"
        "  48 sides          0.004 mm"),

    "closed": (
        "Closed curves",
        "Whether each curve is a loop that joins back to its start.\n\n"
        "auto  -- decide per component. A Geomview VECT input states closure in "
        "its own header and is believed; anything else is measured, and a curve "
        "is closed when its last point sits within about 2.5 median steps of its "
        "first. This is right almost always.\n"
        "yes   -- force every component closed (a seamless loop, no end caps).\n"
        "no    -- force every component open (rounded caps on both ends).\n\n"
        "Get this wrong on a loop and you will see a small gap with two caps "
        "where the ends should have met.",
        "closed loop  -> Euler characteristic 0 (a torus)\n"
        "open strand  -> Euler characteristic 2 (a capped sphere)"),

    "centre": (
        "Recentre on the origin",
        "Shifts the model so its centre of mass sits at (0, 0, 0). Handy "
        "because most slicers and viewers open a model centred.\n\n"
        "Turn it off when the absolute coordinates matter -- for example when "
        "you are exporting several files that have to stay registered to each "
        "other.",
        None),

    "colors": (
        "Component colours",
        "One colour per component, used by the GLB only. Click a swatch to "
        "change it. The swatches rebuild to match the component count when a "
        "file loads.\n\n"
        "STL has no concept of colour, so the .stl file is unaffected. If you "
        "want colours in a print, use the split STLs and assign a material per "
        "part in the slicer.",
        None),

    "output": (
        "Output files",
        "STL  -- every component in one file as separate shells. This is the "
        "print file: nothing touches, so interlocked parts stay movable.\n"
        "GLB  -- the same geometry with per-component colour, for viewing and "
        "sharing.\n"
        "one STL per component -- separate files named _01, _02, ... for "
        "multi-material printing or per-part editing.\n"
        "PNG preview -- quick centre-line projections to confirm the right file "
        "loaded.",
        "base name 'ring' produces\n"
        "    ring.stl\n"
        "    ring.glb\n"
        "    ring_01.stl, ring_02.stl, ...   (if split)\n"
        "    ring_preview.png                (if preview)"),
}


# --------------------------------------------------------------------------
# Input
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


def load_xyz(path):
    """Return [(N,3) float array, ...], one entry per component."""
    return load_curves(path)[0]


def load_curves(path):
    """Return (components, stated_closure).

    stated_closure is one bool per component for a VECT input, which records
    closure in the sign of each vertex count, and None for every other format,
    which leaves closure unsaid.  None means "nobody said", not "open", so the
    caller can fall back to looks_closed rather than assuming.
    """
    path = resolve_path(path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    if vect_io is not None and vect_io.looks_like_vect(text):
        curves = vect_io.read_vect_text(text, source=path)
        blocks = [b for b in curves.components if len(b) >= 2]
        if len(blocks) != len(curves.components):
            raise ValueError("%s has a component with fewer than 2 points" % path)
        if not blocks:
            raise ValueError("no coordinate blocks found in %s" % path)
        return blocks, list(curves.closed)
    if vect_io is None and text.lstrip()[:4].upper() == "VECT":
        raise ValueError(
            "%s is a Geomview VECT file, which needs vect_io.py from "
            "curve_it_lib. Make sure that file sits beside this one." % path)

    lines = text.splitlines()

    blocks, cur, i = [], [], 0
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if not s:
            if cur:
                blocks.append(np.asarray(cur, float))
                cur = []
            continue
        parts = s.split()
        if len(parts) == 1:                 # count line of a standard XYZ frame
            try:
                int(parts[0])
                i += 1                      # skip the comment line
                if cur:
                    blocks.append(np.asarray(cur, float))
                    cur = []
                continue
            except ValueError:
                pass
        try:
            cur.append([float(x) for x in (parts[1:4] if len(parts) >= 4
                                           else parts[:3])])
        except ValueError:
            continue                        # a stray non-numeric line
    if cur:
        blocks.append(np.asarray(cur, float))

    blocks = [b for b in blocks if len(b) >= 2]
    if not blocks:
        raise ValueError("no coordinate blocks found in %s" % path)
    return blocks, None


def closure_flags(components, closed="auto"):
    """One closed/open decision per component.

    `closed` is "auto", "yes" or "no" for the whole input, or one such value
    per component.  The per-component form is what a VECT input supplies: VECT
    states closure per polyline, and a link may mix closed loops with open
    arcs, which a single setting cannot express.
    """
    if isinstance(closed, str):
        settings = [closed] * len(components)
    else:
        settings = list(closed)
        if len(settings) != len(components):
            raise ValueError("got %d closure setting(s) for %d component(s)"
                             % (len(settings), len(components)))
    return [looks_closed(pts) if setting == "auto" else (setting == "yes")
            for pts, setting in zip(components, settings)]


def stated_closure_setting(stated):
    """Turn a per-component closure statement into a `closed` setting."""
    if not stated:
        return "auto"
    return ["yes" if flag else "no" for flag in stated]


def looks_closed(pts, factor=2.5):
    """True when the last point sits about one step from the first."""
    if len(pts) < 4:
        return False
    step = np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return bool(np.linalg.norm(pts[0] - pts[-1]) < factor * step)


def resample(pts, n, closed):
    """Cubic-spline resample to n points, near-uniform in arc length."""
    from scipy.interpolate import splprep, splev

    p = pts
    if closed and np.linalg.norm(p[0] - p[-1]) < 1e-9:
        p = p[:-1]
    ctrl = np.vstack([p, p[0]]) if closed else p
    k = min(3, len(ctrl) - 1)
    tck, _ = splprep(ctrl.T, s=0, per=bool(closed), k=k)
    u0 = np.linspace(0.0, 1.0, max(8 * n, 2000))
    dense = np.asarray(splev(u0, tck)).T
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    target = (np.linspace(0.0, arc[-1], n, endpoint=False) if closed
              else np.linspace(0.0, arc[-1], n))
    return np.asarray(splev(np.interp(target, arc, u0), tck)).T


# --------------------------------------------------------------------------
# Tube meshing
# --------------------------------------------------------------------------
def _rotate_min(v, t0, t1):
    """Rotate v by the smallest rotation carrying t0 onto t1."""
    axis = np.cross(t0, t1)
    sn = np.linalg.norm(axis)
    cs = float(np.dot(t0, t1))
    if sn < 1e-12:
        return v.copy() if cs > 0 else -v
    axis = axis / sn
    ang = math.atan2(sn, cs)
    ca, sa = math.cos(ang), math.sin(ang)
    return v * ca + np.cross(axis, v) * sa + axis * float(np.dot(axis, v)) * (1 - ca)


def frames(P, closed):
    """Rotation-minimising frame along the curve.

    A naive Frenet frame flips at inflection points and spins wildly where the
    curve is nearly straight; transporting one normal along the curve avoids
    both. On a closed curve the leftover twist is spread evenly so the tube
    joins itself seamlessly.
    """
    n = len(P)
    if closed:
        T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    else:
        T = np.gradient(P, axis=0) * 2.0
    T /= np.linalg.norm(T, axis=1)[:, None]

    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(T[0], seed))) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    N = np.empty_like(P)
    N[0] = np.cross(T[0], seed)
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, n):
        v = _rotate_min(N[i - 1], T[i - 1], T[i])
        v -= T[i] * float(np.dot(v, T[i]))
        N[i] = v / np.linalg.norm(v)

    if closed:
        v_end = _rotate_min(N[-1], T[-1], T[0])
        B0 = np.cross(T[0], N[0])
        ang = math.atan2(float(np.dot(v_end, B0)), float(np.dot(v_end, N[0])))
        th = -ang * np.arange(n) / n
        B = np.cross(T, N)
        N = np.cos(th)[:, None] * N + np.sin(th)[:, None] * B
    return T, N, np.cross(T, N)


def tube(P, radius, sides, closed, cap_rings=None):
    """Watertight tube of the given radius around P. Open ends get round caps."""
    if cap_rings is None:
        # a cap is a quarter turn, so match the tube's angular resolution --
        # a coarser cap would fall measurably inside the true hemisphere
        cap_rings = max(3, int(sides) // 4)
    T, N, B = frames(P, closed)
    th = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    ct, st = np.cos(th), np.sin(th)

    rings = [P + radius * (ct[j] * N + st[j] * B) for j in range(sides)]
    V = np.stack(rings, axis=1)             # (n, sides, 3)

    if not closed:                          # hemispherical caps
        pre, post = [], []
        for k in range(1, cap_rings + 1):
            a = (k / (cap_rings + 1)) * (np.pi / 2)
            r, off = radius * math.cos(a), radius * math.sin(a)
            pre.append(np.stack([P[0] - T[0] * off + r * (ct[j] * N[0] + st[j] * B[0])
                                 for j in range(sides)]))
            post.append(np.stack([P[-1] + T[-1] * off + r * (ct[j] * N[-1] + st[j] * B[-1])
                                  for j in range(sides)]))
        V = np.concatenate([np.array(pre[::-1]), V, np.array(post)], axis=0)

    n = len(V)
    verts = V.reshape(-1, 3)
    i = np.arange(n if closed else n - 1)[:, None]
    j = np.arange(sides)[None, :]
    a = i * sides + j
    b = ((i + 1) % n) * sides + j
    c = ((i + 1) % n) * sides + (j + 1) % sides
    d = i * sides + (j + 1) % sides
    faces = [np.stack([a, b, c], -1).reshape(-1, 3),
             np.stack([a, c, d], -1).reshape(-1, 3)]

    if not closed:                          # close each cap with a pole fan
        # the side quads above wind inward; the fans must match or the mesh
        # ends up with inconsistent winding and a meaningless volume
        for pole, ring0, flip in ((P[0] - T[0] * radius, 0, False),
                                  (P[-1] + T[-1] * radius, n - 1, True)):
            pi_ = len(verts)
            verts = np.vstack([verts, pole[None, :]])
            jj = np.arange(sides)
            r0 = ring0 * sides + jj
            r1 = ring0 * sides + (jj + 1) % sides
            fan = (np.stack([np.full(sides, pi_), r1, r0], -1) if flip
                   else np.stack([np.full(sides, pi_), r0, r1], -1))
            faces.append(fan)

    return verts, np.concatenate(faces).astype(np.int64)


def build_meshes(components, scale=1.0, diameter=1.0, segments=None, sides=24,
                 closed="auto", recentre=True):
    """One trimesh.Trimesh per component, in output units."""
    import trimesh

    centre = np.vstack(components).mean(axis=0) if recentre else np.zeros(3)
    out = []
    for pts, shut in zip(components, closure_flags(components, closed)):
        P = (pts - centre) * scale
        if segments:
            P = resample(P, int(segments), shut)
        m = trimesh.Trimesh(*tube(P, diameter / 2.0, int(sides), shut),
                            process=False)
        if not m.is_winding_consistent:
            m.fix_normals()                 # safety net
        elif m.volume < 0:
            m.invert()
        out.append(m)
    return out


def auto_segments(components, scale, diameter, closed="auto"):
    """Enough samples that facets stay well under the rod radius."""
    target = max(diameter / 8.0, 1e-6)
    n = 0
    for pts, shut in zip(components, closure_flags(components, closed)):
        p = np.vstack([pts, pts[0]]) if shut else pts
        length = np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * scale
        n = max(n, int(length / target))
    return int(min(max(n, 64), 6000))


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def _rgba(colour):
    h = str(colour).strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return [int(h[k:k + 2], 16) for k in (0, 2, 4)] + [255]


def export_stl(meshes, path):
    import trimesh
    trimesh.util.concatenate(meshes).export(path, file_type="stl")
    return path


def export_stl_split(meshes, prefix):
    paths = []
    for i, m in enumerate(meshes):
        p = "%s_%02d.stl" % (prefix, i + 1)
        m.export(p, file_type="stl")
        paths.append(p)
    return paths


def export_glb(meshes, path, colors=None):
    import trimesh
    colors = colors or DEFAULT_COLORS
    scene = trimesh.Scene()
    for i, m in enumerate(meshes):
        m = m.copy()
        m.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name="component_%d" % (i + 1),
                baseColorFactor=_rgba(colors[i % len(colors)]),
                metallicFactor=0.3, roughnessFactor=0.4))
        scene.add_geometry(m, node_name="component_%d" % (i + 1),
                           geom_name="component_%d" % (i + 1))
    scene.export(path, file_type="glb")
    return path


def measured_size(meshes):
    """Actual bounding-box dimensions of a built mesh list."""
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    return hi - lo


def describe(components, scale=1.0, diameter=None, closed="auto"):
    L = ["%d component(s)" % len(components)]
    allp = np.vstack(components) * scale
    lo, hi = allp.min(0), allp.max(0)
    ext = hi - lo
    if diameter:
        L.append("  scale %g, rod diameter %g" % (scale, diameter))
    L.append("")
    L.append("  centre-line extent %9.3f x %9.3f x %9.3f" % tuple(ext))
    if diameter:
        # A tube of radius r around a curve is the set of points within r of
        # it, so its bounding box is the centre-line's grown by r on all six
        # sides -- exactly one diameter on each axis. Rounded caps included.
        size = ext + diameter
        L.append("  MODEL SIZE         %9.3f x %9.3f x %9.3f"
                 % tuple(size))
        L.append("                     (centre-line + one rod diameter per axis)")
        L.append("  largest dimension  %9.3f" % size.max())
    L.append("  centre             %9.3f , %9.3f , %9.3f" % tuple((hi + lo) / 2))
    L.append("")
    total = 0.0
    for i, (pts, shut) in enumerate(zip(components,
                                        closure_flags(components, closed))):
        p = np.vstack([pts, pts[0]]) if shut else pts
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1) * scale
        total += seg.sum()
        L.append("  [%2d] %5d points  %-6s length %9.3f  step %.4f..%.4f"
                 % (i + 1, len(pts), "closed" if shut else "open",
                    seg.sum(), seg.min(), seg.max()))
    L.append("  total centre-line length %.3f" % total)
    if diameter:
        vol = total * math.pi * (diameter / 2.0) ** 2
        L.append("  rod %.3f -> volume %.3f  (%.3f cm3 if units are mm)"
                 % (diameter, vol, vol / 1000.0))
    return "\n".join(L)


def render_preview(components, scale=1.0, colors=None, path=None, closed="auto"):
    """Quick centre-line projections; enough to confirm the right file loaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = colors or DEFAULT_COLORS
    if path is None:
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "xyz2model_preview.png")

    curves = []
    for pts, shut in zip(components, closure_flags(components, closed)):
        curves.append((np.vstack([pts, pts[0]]) if shut else pts) * scale)
    mid = np.vstack(curves).mean(axis=0)
    curves = [c - mid for c in curves]

    fig = plt.figure(figsize=(15, 4.4))
    for k, (a, b, t) in enumerate([(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]):
        ax = fig.add_subplot(1, 4, k + 1)
        for i, c in enumerate(curves):
            ax.plot(c[:, a], c[:, b], lw=1.3, color=colors[i % len(colors)])
        ax.set_aspect("equal"); ax.set_title(t, fontsize=9); ax.grid(alpha=0.3)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    for i, c in enumerate(curves):
        ax.plot(c[:, 0], c[:, 1], c[:, 2], lw=1.0, color=colors[i % len(colors)])
    ax.set_title("3-D", fontsize=9)
    # The three flat panels are set_aspect("equal"); match that here, which
    # needs a cubic box as well as equal limits or z comes out at 0.75.
    stacked = np.vstack(curves)
    span = max(float(np.ptp(stacked[:, k])) for k in range(3)) or 1.0
    for k, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        middle = 0.5 * float(stacked[:, k].max() + stacked[:, k].min())
        setter(middle - 0.5 * span, middle + 0.5 * span)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:                   # noqa: BLE001
        pass
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Mesh input (STL and friends) -- scaling only
# --------------------------------------------------------------------------
MESH_EXTS = {".stl", ".obj", ".ply", ".off", ".glb", ".gltf", ".3mf"}
CURVE_EXTS = {".xyz", ".txt", ".dat", ".csv", ".vect"}
MAX_PARTS = 256          # beyond this, treat a mesh as one piece


def detect_kind(path):
    """'mesh' or 'curves', from the extension and then from the bytes."""
    path = resolve_path(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in MESH_EXTS:
        return "mesh"

    # Sniff the bytes before trusting a generic extension -- a .dat or .txt
    # holding an STL is far more likely than an .xyz that isn't coordinates.
    try:
        with open(path, "rb") as fh:
            head = fh.read(1024)
    except OSError:
        return "curves"
    if head[:5].lower() == b"solid" and b"facet" in head.lower():
        return "mesh"                       # ASCII STL
    if b"\0" in head:                       # binary payload, not coordinates
        return "mesh"
    return "curves"


def load_mesh(path, split=True):
    """Load a mesh file and return its connected components as a list.

    Splitting matches how the curve path works: one entry per component, so
    the GLB gets one colour each and --split writes one STL each. A mesh that
    is a single connected lump simply comes back as a one-item list.
    """
    import trimesh

    path = resolve_path(path)
    # process=True welds duplicate vertices. STL stores every triangle with its
    # own three vertices, so without welding each face is its own island and
    # split() would return one "component" per triangle.
    obj = trimesh.load(path, force="mesh", process=True)
    if obj is None or not hasattr(obj, "faces") or len(obj.faces) == 0:
        raise ValueError("no triangles found in %s" % path)
    obj.merge_vertices()
    if not split:
        return [obj]
    parts = obj.split(only_watertight=False)
    if not len(parts):
        return [obj]
    if len(parts) > MAX_PARTS:
        print("  note: %d connected components found; keeping the mesh whole "
              "(use --no-split-input to silence this)" % len(parts))
        return [obj]
    return list(parts)


def scale_meshes(meshes, scale=1.0, recentre=True):
    """Copy and resize. Recentres on the bounding box of the whole set."""
    out = [m.copy() for m in meshes]
    if recentre:
        lo = np.min([m.bounds[0] for m in out], axis=0)
        hi = np.max([m.bounds[1] for m in out], axis=0)
        centre = (lo + hi) / 2.0
    else:
        centre = np.zeros(3)
    for m in out:
        m.vertices = (np.asarray(m.vertices) - centre) * scale
    return out


def describe_meshes(meshes, scale=1.0):
    lo = np.min([m.bounds[0] for m in meshes], axis=0) * scale
    hi = np.max([m.bounds[1] for m in meshes], axis=0) * scale
    ext = hi - lo
    L = ["%d component(s)  [mesh input -- scaling only]" % len(meshes)]
    L.append("  scale %g" % scale)
    L.append("")
    L.append("  MODEL SIZE         %9.3f x %9.3f x %9.3f" % tuple(ext))
    L.append("  largest dimension  %9.3f" % ext.max())
    L.append("  centre             %9.3f , %9.3f , %9.3f" % tuple((hi + lo) / 2))
    L.append("")
    tf = tv = ta = 0
    tight = True
    for i, m in enumerate(meshes):
        v = float(m.volume) * scale ** 3
        a = float(m.area) * scale ** 2
        tf += len(m.faces); tv += v; ta += a
        tight &= bool(m.is_watertight)
        if len(meshes) <= 12:
            L.append("  [%2d] %7d faces  %-14s volume %10.3f  area %10.3f"
                     % (i + 1, len(m.faces),
                        "watertight" if m.is_watertight else "NOT watertight",
                        v, a))
    L.append("  total %d faces, volume %.3f, area %.3f" % (tf, tv, ta))
    L.append("  all watertight: %s" % ("yes" if tight else "no"))
    if not tight:
        L.append("  ! an open mesh has no meaningful volume and may not print")
    L.append("  volume is %.3f cm3 if the scaled units are mm" % (tv / 1000.0))
    return "\n".join(L)


def _zbuffer(meshes, colors, elev, azim, size=780, ambient=0.25, spec=0.3):
    """Small orthographic z-buffer rasteriser.

    Needed because a woven or nested model cannot be depth-sorted per triangle
    -- one part's triangles are both in front of and behind another's.
    """
    from matplotlib.colors import to_rgb

    cols = [np.array(to_rgb(c), float) for c in colors]
    e, a = math.radians(elev), math.radians(azim)
    fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    up0 = np.array([0.0, 0.0, 1.0])
    if abs(float(fwd @ up0)) > 0.999:
        up0 = np.array([0.0, 1.0, 0.0])
    right = np.cross(up0, fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    light = fwd * 0.55 + np.array([0.25, 0.35, 0.85]); light /= np.linalg.norm(light)
    hv = light + fwd; hv /= np.linalg.norm(hv)

    allv = np.vstack([m.vertices for m in meshes])
    centre = (allv.min(0) + allv.max(0)) / 2.0
    loc = allv - centre
    ext = max(np.abs(loc @ right).max(), np.abs(loc @ up).max(), 1e-9) * 1.06
    sc = size / (2 * ext)

    img = np.ones((size, size, 3))
    zb = np.full((size, size), -1e18)
    for mi, m in enumerate(meshes):
        V = np.asarray(m.vertices) - centre
        F = np.asarray(m.faces)
        tri = np.stack([(V @ right) * sc + size / 2,
                        size / 2 - (V @ up) * sc, V @ fwd], 1)[F]
        n = np.cross(V[F][:, 1] - V[F][:, 0], V[F][:, 2] - V[F][:, 0])
        ln = np.linalg.norm(n, axis=1); ln[ln == 0] = 1.0
        n /= ln[:, None]
        keep = (n @ fwd) > 0
        tri, n = tri[keep], n[keep]
        if not len(tri):
            continue
        shade = ambient + (1 - ambient) * np.clip(n @ light, 0, 1)
        col = np.clip(cols[mi % len(cols)][None, :] * shade[:, None]
                      + spec * np.clip(n @ hv, 0, 1)[:, None] ** 28, 0, 1)
        x0 = np.clip(np.floor(tri[:, :, 0].min(1)).astype(int), 0, size - 1)
        x1 = np.clip(np.ceil(tri[:, :, 0].max(1)).astype(int), 0, size - 1)
        y0 = np.clip(np.floor(tri[:, :, 1].min(1)).astype(int), 0, size - 1)
        y1 = np.clip(np.ceil(tri[:, :, 1].max(1)).astype(int), 0, size - 1)
        for k in np.nonzero((x1 >= x0) & (y1 >= y0))[0]:
            ax_, ay_, az_ = tri[k, 0]; bx, by, bz = tri[k, 1]; cx, cy, cz = tri[k, 2]
            det = (by - cy) * (ax_ - cx) + (cx - bx) * (ay_ - cy)
            if abs(det) < 1e-12:
                continue
            gx, gy = np.meshgrid(np.arange(x0[k], x1[k] + 1) + 0.5,
                                 np.arange(y0[k], y1[k] + 1) + 0.5)
            w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
            w1 = ((cy - ay_) * (gx - cx) + (ax_ - cx) * (gy - cy)) / det
            w2 = 1.0 - w0 - w1
            ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not ins.any():
                continue
            zz = w0 * az_ + w1 * bz + w2 * cz
            sub = zb[y0[k]:y1[k] + 1, x0[k]:x1[k] + 1]
            hit = ins & (zz > sub)
            if hit.any():
                sub[hit] = zz[hit]
                img[y0[k]:y1[k] + 1, x0[k]:x1[k] + 1][hit] = col[k]
    return img


def render_mesh_preview(meshes, colors=None, path=None):
    """Three shaded views of a mesh. Returns the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = colors or DEFAULT_COLORS
    if path is None:
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "xyz2model_preview.png")
    views = [(90, -90, "top  (down +Z)"), (0, -90, "front"), (25, -55, "three-quarter")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="white")
    for ax, (el, az, t) in zip(np.atleast_1d(axs), views):
        ax.imshow(_zbuffer(meshes, colors, el, az))
        ax.set_axis_off()
        ax.set_title(t, fontsize=11)
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0.01)
    fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="xyz2model.py",
        description="Sweep a rod along the curves in an .xyz file "
                    "and write STL / GLB.",
        epilog="Run with no arguments for the GUI.")
    p.add_argument("xyz", nargs="?", help="input .xyz file")
    p.add_argument("--gui", action="store_true", help="force the graphical interface")

    p.add_argument("-d", "--diameter", type=float, default=1.0,
                   help="rod diameter in FINAL output units, applied AFTER "
                        "--scale, so --scale does not change it (default 1.0)")
    p.add_argument("-s", "--scale", type=float, default=1.0,
                   help="multiply the input coordinates by this; stretches the "
                        "centre-line only, not the rod (default 1.0)")
    p.add_argument("--segments", type=int,
                   help="samples per curve (default: from the rod diameter; "
                        "0 keeps the input points untouched)")
    p.add_argument("--sides", type=int, default=24,
                   help="facets around the rod (default 24)")
    p.add_argument("--closed", choices=["auto", "yes", "no"], default="auto",
                   help="treat curves as closed loops. Default auto: a Geomview "
                        "VECT input states closure per component and is believed; "
                        "any other format is measured by the gap back to the "
                        "first point")
    p.add_argument("--as", dest="kind", choices=["auto", "curves", "mesh"],
                   default="auto",
                   help="how to read the input (default: from the extension). "
                        "A mesh input is only scaled -- no rod is swept.")
    p.add_argument("--no-split-input", action="store_true",
                   help="mesh input: keep it as one piece instead of separating "
                        "connected components")
    p.add_argument("--keep-origin", action="store_true",
                   help="do not recentre the model on its bounding box")

    p.add_argument("-o", "--output", help="output path prefix (default: input name)")
    p.add_argument("--stl", help="explicit .stl path")
    p.add_argument("--glb", help="explicit .glb path")
    p.add_argument("--no-stl", action="store_true")
    p.add_argument("--no-glb", action="store_true")
    p.add_argument("--split", action="store_true",
                   help="also write one STL per component")
    p.add_argument("--colors", help="comma separated hex colours for the GLB")
    p.add_argument("--preview", action="store_true", help="write a PNG preview")
    p.add_argument("--info", action="store_true",
                   help="describe the file and exit without meshing")
    return p


def run_cli(args):
    real = resolve_path(args.xyz)
    kind = detect_kind(args.xyz) if args.kind == "auto" else args.kind
    colors = ([c.strip() for c in args.colors.split(",")] if args.colors
              else DEFAULT_COLORS)
    prefix = args.output or os.path.splitext(real)[0]

    # ---------------- mesh input: scale only -----------------------------
    if kind == "mesh":
        parts = load_mesh(args.xyz, split=not args.no_split_input)
        print("Loaded %s  [mesh]" % real)
        ignored = [n for n, v, d in (("--diameter", args.diameter, 1.0),
                                     ("--segments", args.segments, None),
                                     ("--sides", args.sides, 24),
                                     ("--closed", args.closed, "auto"))
                   if v != d]
        if ignored:
            print("  note: %s ignored -- a mesh input is only scaled"
                  % ", ".join(ignored))
        print(describe_meshes(parts, args.scale))
        if args.info:
            return 0

        meshes = scale_meshes(parts, args.scale, recentre=not args.keep_origin)
        written = []
        if not args.no_stl:
            written.append(export_stl(meshes, args.stl or (prefix + ".stl")))
        if not args.no_glb:
            written.append(export_glb(meshes, args.glb or (prefix + ".glb"), colors))
        if args.split:
            written += export_stl_split(meshes, prefix)
        if args.preview:
            written.append(render_mesh_preview(meshes, colors,
                                               prefix + "_preview.png"))
        for w in written:
            print("  wrote %s" % w)
        return 0

    # ---------------- curve input: sweep a rod ---------------------------
    comps, stated = load_curves(args.xyz)
    closed = args.closed
    if stated and closed == "auto":
        # VECT states closure per polyline, so believe the file rather than
        # measuring the gap back to the first vertex.  An explicit --closed
        # still wins, which is why this only fires on the "auto" default.
        closed = stated_closure_setting(stated)
        shut = sum(1 for flag in stated if flag)
        print("  closure from the VECT file: %d closed, %d open"
              % (shut, len(stated) - shut))
    print("Loaded %s  [curves]" % real)
    print(describe(comps, args.scale, args.diameter, closed))
    if args.info:
        return 0

    seg = args.segments
    if seg is None:
        seg = auto_segments(comps, args.scale, args.diameter, closed)
        print("  segments per curve: %d (auto)" % seg)
    need_mesh = not (args.no_stl and args.no_glb and not args.split)
    meshes = []
    if need_mesh:
        print("\nMeshing ...")
        meshes = build_meshes(comps, args.scale, args.diameter, seg, args.sides,
                              closed, recentre=not args.keep_origin)
        tight = all(m.is_watertight for m in meshes)
        print("  %d triangles, watertight=%s"
              % (sum(len(m.faces) for m in meshes), tight))
        print("  measured size  %.3f x %.3f x %.3f" % tuple(measured_size(meshes)))
        if not tight:
            print("  ! not watertight -- try a different --closed setting")

    written = []
    if not args.no_stl:
        written.append(export_stl(meshes, args.stl or (prefix + ".stl")))
    if not args.no_glb:
        written.append(export_glb(meshes, args.glb or (prefix + ".glb"), colors))
    if args.split:
        written += export_stl_split(meshes, prefix)
    if args.preview:
        written.append(render_preview(comps, args.scale, colors,
                                      prefix + "_preview.png", closed))
    for w in written:
        print("  wrote %s" % w)
    return 0


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def run_gui(initial_file=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, colorchooser, messagebox

    root = tk.Tk()
    root.title("xyz2model  -  .xyz curves or a mesh  ->  STL / GLB   v%s"
               % __version__)
    root.minsize(880, 560)
    set_optional_window_icon(
        root, tk, ["xyz2model_icon.png", "icon.png"], "_xyz2model_icon_image")

    state = {"comps": None, "path": None, "kind": None, "stated_closure": None}

    def gui_closed():
        """The closure setting to use, letting a VECT file answer "auto"."""
        setting = V["closed"].get()
        if setting == "auto" and state.get("stated_closure"):
            return stated_closure_setting(state["stated_closure"])
        return setting
    colors = list(DEFAULT_COLORS)
    V = {
        "file":     tk.StringVar(value=initial_file or ""),
        "scale":    tk.StringVar(value="1.0"),
        "diameter": tk.StringVar(value="1.0"),
        "segments": tk.StringVar(value="auto"),
        "sides":    tk.StringVar(value="24"),
        "closed":   tk.StringVar(value="auto"),
        "outdir":   tk.StringVar(value=""),
        "basename": tk.StringVar(value="model"),
        "stl":      tk.BooleanVar(value=True),
        "glb":      tk.BooleanVar(value=True),
        "split":    tk.BooleanVar(value=False),
        "png":      tk.BooleanVar(value=False),
        "centre":   tk.BooleanVar(value=True),
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
                          width=56, bg="#e8f1fa", relief="flat", padx=8, pady=6,
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
    left = ttk.Frame(main); left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    right = ttk.Frame(main); right.grid(row=0, column=1, sticky="nsew")
    right.rowconfigure(1, weight=1); right.columnconfigure(0, weight=1)

    # input -----------------------------------------------------------------
    fi = ttk.LabelFrame(left, text="Input", padding=8); fi.pack(fill="x", pady=(0, 8))
    fi.columnconfigure(0, weight=1)
    ttk.Entry(fi, textvariable=V["file"], width=40).grid(row=0, column=0, sticky="ew")

    def browse():
        p = filedialog.askopenfilename(
            title="Select an .xyz or mesh file",
            filetypes=[("Curves and meshes",
                        "*.xyz *.vect *.stl *.obj *.ply *.glb *.3mf"),
                       ("XYZ curves", "*.xyz"),
                       ("Geomview VECT curves", "*.vect"),
                       ("Meshes", "*.stl *.obj *.ply *.off *.glb *.gltf *.3mf"),
                       ("All files", "*.*")])
        if p:
            V["file"].set(p)
            do_load()

    ttk.Button(fi, text="Browse...", command=browse).grid(row=0, column=1, padx=(6, 0))
    chip(fi, "file").grid(row=0, column=2, padx=(6, 0))
    lbl = ttk.Label(fi, text="no file loaded", foreground="#888")
    lbl.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
    modebar = tk.Frame(fi, bg="#f4f9ff")
    modebar.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
    mode_lbl = tk.Label(modebar, text="mode: -", bg="#f4f9ff", fg="#0b4d80",
                        font=("Helvetica", 11, "bold"), anchor="w", padx=6, pady=3)
    mode_lbl.pack(side="left", fill="x", expand=True)
    chip(modebar, "mode").pack(side="right", padx=(6, 4))

    # geometry --------------------------------------------------------------
    fg = ttk.LabelFrame(left, text="Geometry", padding=8); fg.pack(fill="x", pady=(0, 8))
    tk.Label(fg, text="Scale is applied to the centre-line first; the rod\n"
                      "diameter is then in final output units.",
             justify="left", anchor="w", font=("Helvetica", 10),
             fg="#55606b").grid(row=0, column=0, columnspan=3, sticky="w",
                                pady=(0, 6))
    rows = [("scale factor", "scale", "x input coords"),
            ("rod diameter", "diameter", "final units, after scaling"),
            ("segments / curve", "segments", "'auto', or 0 for raw points"),
            ("facets / rod", "sides", "24 is plenty for printing")]
    W, HINT = {}, {}
    for r, (lab, key, hint) in enumerate(rows, start=1):
        lw = ttk.Label(fg, text=lab); lw.grid(row=r, column=0, sticky="w")
        ew = ttk.Entry(fg, textvariable=V[key], width=10)
        ew.grid(row=r, column=1, sticky="w", padx=(4, 0))
        chip(fg, key).grid(row=r, column=2, padx=(6, 0))
        hw = tk.Label(fg, text=hint, font=("Helvetica", 10), fg="#8a929b")
        hw.grid(row=r, column=3, sticky="w", padx=(8, 0))
        W[key] = (lw, ew); HINT[key] = (hw, hint)
    lw = ttk.Label(fg, text="closed curves"); lw.grid(row=5, column=0, sticky="w")
    cb = ttk.Combobox(fg, textvariable=V["closed"], width=8, state="readonly",
                      values=["auto", "yes", "no"])
    cb.grid(row=5, column=1, sticky="w", padx=(4, 0))
    chip(fg, "closed").grid(row=5, column=2, padx=(6, 0))
    W["closed"] = (lw, cb)

    def set_mode(kind):
        """Grey out the curve-only fields when the input is already a solid."""
        curve_only = ("diameter", "segments", "sides", "closed")
        on = (kind != "mesh")
        for key in curve_only:
            lw_, ew_ = W[key]
            lw_.configure(state="normal" if on else "disabled")
            ew_.configure(state=("readonly" if key == "closed" else "normal")
                          if on else "disabled")
            if key in HINT:
                hw_, txt = HINT[key]
                hw_.configure(text=txt if on else "not used for a mesh input",
                              fg="#8a929b" if on else "#b0b6bc")
        if kind == "mesh":
            mode_lbl.configure(text="mode: MESH  --  scaling only",
                               fg="#8a4b00", bg="#fff2e0")
            modebar.configure(bg="#fff2e0")
        elif kind == "curves":
            mode_lbl.configure(text="mode: CURVES  --  a rod is swept along each",
                               fg="#0b4d80", bg="#f4f9ff")
            modebar.configure(bg="#f4f9ff")
        else:
            mode_lbl.configure(text="mode: -", fg="#0b4d80", bg="#f4f9ff")
            modebar.configure(bg="#f4f9ff")
    ttk.Checkbutton(fg, text="recentre on the origin", variable=V["centre"]
                    ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))
    set_mode(None)
    chip(fg, "centre").grid(row=6, column=2, padx=(6, 0), pady=(4, 0))

    # colours ---------------------------------------------------------------
    fc = ttk.LabelFrame(left, text="Component colours (GLB only)", padding=8)
    fc.pack(fill="x", pady=(0, 8))
    head = ttk.Frame(fc); head.pack(fill="x")
    tk.Label(head, text="click a swatch to change it", font=("Helvetica", 10),
             fg="#8a929b").pack(side="left")
    chip(head, "colors").pack(side="right")
    row = ttk.Frame(fc); row.pack(fill="x", pady=(4, 0))
    swatches = []

    def pick(i):
        _rgb, hx = colorchooser.askcolor(color=colors[i],
                                         title="Component %d" % (i + 1))
        if hx:
            colors[i] = hx
            swatches[i].configure(bg=hx)

    def build_swatches(n):
        for w in swatches:
            w.destroy()
        swatches.clear()
        while len(colors) < n:
            colors.append(DEFAULT_COLORS[len(colors) % len(DEFAULT_COLORS)])
        for i in range(n):
            b = tk.Button(row, text=" %d " % (i + 1), bg=colors[i], fg="white",
                          width=3, command=lambda i=i: pick(i))
            b.grid(row=i // 8, column=i % 8, padx=2, pady=2)
            swatches.append(b)
    build_swatches(5)

    # output ----------------------------------------------------------------
    fo = ttk.LabelFrame(left, text="Output", padding=8); fo.pack(fill="x")
    fo.columnconfigure(1, weight=1)
    ttk.Label(fo, text="folder").grid(row=0, column=0, sticky="w")
    ttk.Entry(fo, textvariable=V["outdir"], width=26).grid(row=0, column=1, sticky="ew")
    ttk.Button(fo, text="...", width=3,
               command=lambda: V["outdir"].set(filedialog.askdirectory()
                                               or V["outdir"].get())
               ).grid(row=0, column=2, padx=(4, 0))
    chip(fo, "output").grid(row=0, column=3, padx=(6, 0))
    ttk.Label(fo, text="base name").grid(row=1, column=0, sticky="w", pady=(4, 0))
    ttk.Entry(fo, textvariable=V["basename"], width=26).grid(row=1, column=1,
                                                             sticky="ew", pady=(4, 0))
    opts = ttk.Frame(fo); opts.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
    for i, (lab, key) in enumerate([("STL", "stl"), ("GLB", "glb"),
                                    ("one STL per component", "split"),
                                    ("PNG preview", "png")]):
        ttk.Checkbutton(opts, text=lab, variable=V[key]).grid(
            row=i // 2, column=i % 2, sticky="w", padx=(0, 10))

    # right pane ------------------------------------------------------------
    bar = ttk.Frame(right); bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    ttk.Button(bar, text="Reload", command=lambda: do_load()).pack(side="left")
    ttk.Button(bar, text="Preview", command=lambda: do_preview()).pack(side="left", padx=6)
    ttk.Button(bar, text="Generate", command=lambda: do_generate()).pack(side="left")
    txt = tk.Text(right, wrap="none", font=("Menlo", 11), height=24)
    txt.grid(row=1, column=0, sticky="nsew")
    sb = ttk.Scrollbar(right, orient="vertical", command=txt.yview)
    sb.grid(row=1, column=1, sticky="ns"); txt.configure(yscrollcommand=sb.set)
    status = ttk.Label(right, text="", foreground="#0a7")
    status.grid(row=2, column=0, sticky="w", pady=(4, 0))

    def show(s):
        txt.delete("1.0", "end"); txt.insert("1.0", s)

    def fnum(key, default=None):
        v = V[key].get().strip()
        if v.lower() in ("", "auto") and default is not None:
            return default
        try:
            return float(v)
        except ValueError:
            raise ValueError("%s: %r is not a number" % (key, v))

    def do_load():
        p = V["file"].get().strip()
        if not p:
            return
        try:
            kind = detect_kind(p)
            if kind == "mesh":
                comps, stated = load_mesh(p), None
            else:
                comps, stated = load_curves(p)
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Load failed", str(exc))
            status.configure(text="load failed", foreground="#c00")
            return
        state.update(comps=comps, path=resolve_path(p), kind=kind,
                     stated_closure=stated)
        set_mode(kind)
        if kind == "mesh":
            lbl.configure(text="%d component(s), %d triangles"
                              % (len(comps), sum(len(m.faces) for m in comps)),
                          foreground="#333")
        else:
            lbl.configure(text="%d component(s), %d points total"
                              % (len(comps), sum(len(c) for c in comps)),
                          foreground="#333")
        build_swatches(len(comps))
        if not V["outdir"].get():
            V["outdir"].set(os.path.dirname(state["path"]))
        if V["basename"].get() in ("", "model"):
            V["basename"].set(os.path.splitext(os.path.basename(state["path"]))[0])
        refresh()
        status.configure(text="loaded", foreground="#0a7")

    def refresh(*_):
        if state["comps"] is None:
            return
        try:
            if state["kind"] == "mesh":
                show(describe_meshes(state["comps"], fnum("scale", 1.0)))
            else:
                show(describe(state["comps"], fnum("scale", 1.0),
                              fnum("diameter", 1.0), gui_closed()))
        except Exception as exc:            # noqa: BLE001
            show("cannot evaluate:\n  %s" % exc)

    for key in ("scale", "diameter", "closed"):
        V[key].trace_add("write", lambda *a: refresh())

    def do_preview():
        if state["comps"] is None:
            do_load()
        if state["comps"] is None:
            return
        try:
            if state["kind"] == "mesh":
                status.configure(text="rendering ...", foreground="#888")
                root.update_idletasks()
                png = render_mesh_preview(
                    scale_meshes(state["comps"], fnum("scale", 1.0),
                                 V["centre"].get()), colors, None)
                status.configure(text="", foreground="#0a7")
            else:
                png = render_preview(state["comps"], fnum("scale", 1.0), colors,
                                     None, gui_closed())
            top = tk.Toplevel(root); top.title("Preview")
            try:
                from PIL import Image, ImageTk
                im = Image.open(png)
                w = min(1350, im.width)
                im = im.resize((w, int(im.height * w / im.width)))
                img = ImageTk.PhotoImage(im)
            except Exception:               # noqa: BLE001
                img = tk.PhotoImage(file=png)
            lab2 = tk.Label(top, image=img); lab2.image = img; lab2.pack()
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Preview failed", str(exc))

    def do_generate():
        if state["comps"] is None:
            do_load()
        if state["comps"] is None:
            return
        try:
            comps = state["comps"]
            scale = fnum("scale", 1.0)
            is_mesh = state["kind"] == "mesh"
            status.configure(text="working ...", foreground="#888")
            root.update_idletasks()
            if is_mesh:
                dia = seg = None
                meshes = scale_meshes(comps, scale, recentre=V["centre"].get())
            else:
                dia = fnum("diameter", 1.0)
                closed = gui_closed()
                seg_txt = V["segments"].get().strip().lower()
                seg = (auto_segments(comps, scale, dia, closed)
                       if seg_txt in ("", "auto") else int(float(seg_txt)))
                meshes = build_meshes(comps, scale, dia, seg,
                                      int(fnum("sides", 24)), closed,
                                      recentre=V["centre"].get())
            prefix = os.path.join(V["outdir"].get() or os.path.dirname(state["path"]),
                                  V["basename"].get() or "model")
            written = []
            if V["stl"].get():
                written.append(export_stl(meshes, prefix + ".stl"))
            if V["glb"].get():
                written.append(export_glb(meshes, prefix + ".glb", colors))
            if V["split"].get():
                written += export_stl_split(meshes, prefix)
            if V["png"].get():
                written.append(render_mesh_preview(meshes, colors,
                                                   prefix + "_preview.png")
                               if is_mesh else
                               render_preview(comps, scale, colors,
                                              prefix + "_preview.png", closed))
            msz = measured_size(meshes)
            if is_mesh:
                rep = describe_meshes(comps, scale)
                rep += ("\n\nRESULT\n  scaled by %g (lengths), %g (areas), "
                        "%g (volume)\n  measured size  %.3f x %.3f x %.3f"
                        "\n  %d triangles, watertight=%s\n\nWROTE\n"
                        % (scale, scale ** 2, scale ** 3, msz[0], msz[1], msz[2],
                           sum(len(m.faces) for m in meshes),
                           all(m.is_watertight for m in meshes)))
            else:
                rep = describe(comps, scale, dia, closed)
                rep += ("\n\nMESH\n  segments/curve %d, facets %d"
                        "\n  measured size  %.3f x %.3f x %.3f"
                        "\n  %d triangles, watertight=%s\n\nWROTE\n"
                        % (seg, int(fnum("sides", 24)), msz[0], msz[1], msz[2],
                           sum(len(m.faces) for m in meshes),
                           all(m.is_watertight for m in meshes)))
            rep += "\n".join("  " + w for w in written) or "  (nothing selected)"
            show(rep)
            status.configure(text="done -- %d file(s)" % len(written),
                             foreground="#0a7")
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Generate failed", str(exc))
            status.configure(text="failed", foreground="#c00")

    if initial_file:
        root.after(100, do_load)
    else:
        show("Choose a file to begin.\n\n"
             "CURVES  (.xyz)   each component becomes a round rod of the\n"
             "                 given diameter. Closed loops are detected\n"
             "                 automatically; open strands get rounded ends.\n"
             "MESH    (.stl,   the solid already exists, so it is only\n"
             "  .obj .ply       SCALED. The rod, segment, facet and closed\n"
             "  .glb .3mf)      settings grey out.\n\n"
             "Order of operations:\n"
             "    input coordinates\n"
             "      x  scale factor        <- changes the overall size\n"
             "      -> centre-line\n"
             "      +  rod diameter        <- in FINAL units, unaffected\n"
             "      -> solid model            by the scale factor\n\n"
             "Click any light-blue  ?  for an explanation and examples.\n")
    root.mainloop()
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        return run_gui()
    args = parser.parse_args(argv)
    if args.gui:
        return run_gui(args.xyz)
    if not args.xyz:
        parser.error("an input .xyz file is required (or use --gui)")
    try:
        return run_cli(args)
    except FileNotFoundError as exc:
        print("error: cannot open %s" % exc.filename, file=sys.stderr)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
