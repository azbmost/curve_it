#!/usr/bin/env python3
"""interpolate_xyz.py

Curve interpolation helper for XYZ-like point files.

This script reads a 3D polyline from an XYZ-like text file and writes an
interpolated version.

Two interpolation modes are supported:

1) Set total number of points (n):
   The output curve will contain exactly n points that are evenly spaced along
   the curve *by arc length*.

2) Insert p points per segment (p):
   The output curve will insert p equally spaced points between every pair of
   adjacent input points.

The default output filename is derived from the input filename by inserting
"_interpolated" before the extension.

Examples
--------
# 1) Uniform arc-length sampling to 400 points
python curve_it_lib/interpolate_xyz.py curve.xyz --n 400

# 2) Insert 5 points between each pair of adjacent points
python curve_it_lib/interpolate_xyz.py curve.xyz --p 5

# Treat input as a closed loop when interpolating (includes last->first segment)
python curve_it_lib/interpolate_xyz.py curve.xyz --n 400 --closed

Notes
-----
- The reader accepts either:
  * Standard XYZ (element + x y z per line, optionally preceded by atom-count
    and a comment line), or
  * Simple whitespace-separated x y z per line (with optional comments).
- Output is written as plain "x y z" lines (one point per line).
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np


# Optional Geomview VECT support.  VECT is the one curve format that states
# per component whether it is a closed loop, so a .vect input answers --closed
# instead of leaving it to the default.
try:
    from . import vect_io
except ImportError:
    try:
        import vect_io  # type: ignore[no-redef]
    except ImportError:
        vect_io = None  # type: ignore[assignment]


def first_token_is_integer(line: str) -> bool:
    """Return True if the first token in a line is an integer atom count."""
    parts = line.split()
    if not parts:
        return False
    try:
        int(parts[0])
        return len(parts) == 1
    except ValueError:
        return False


def parse_xyz_coordinate_line(line: str) -> Optional[List[float]]:
    """Return the first three numeric tokens in a coordinate line, if present."""
    floats: List[float] = []
    for tok in line.split():
        try:
            floats.append(float(tok))
        except ValueError:
            continue
        if len(floats) == 3:
            return floats
    return None


def is_vect_text(text: str) -> bool:
    """True when this text is a Geomview VECT file, by its header word."""
    return vect_io is not None and vect_io.looks_like_vect(text)


def read_vect_curves(text: str, source: Optional[str] = None):
    """Parse VECT text, raising ValueError so existing handlers keep working."""
    if vect_io is None:
        raise ValueError(
            "This is a Geomview VECT file, which needs vect_io.py from "
            "curve_it_lib. Make sure that file sits beside this one.")
    return vect_io.read_vect_text(text, source=source)


def read_xyz_curve_from_text(xyz_text: str) -> np.ndarray:
    """Read a 3D polyline from a generic XYZ-like text string, or from VECT.

    Accepts:
    - Geomview VECT, detected by its header word. Every polyline is
      concatenated in file order, matching how the coordinate reader below
      treats blank-line-separated components.
    - Standard XYZ: first line = number of atoms, second = comment, remaining
      lines 'Element x y z'
    - Or a simple whitespace separated 'x y z' per line (with optional comments
      starting with # or !).

    Strategy:
    - If the first non-empty line is an atom count, treat the file as molecular
      XYZ and skip the atom-count and comment header lines.
    - Otherwise, on each non-comment line, collect numeric tokens and use the
      first three as x,y,z.
    """
    if is_vect_text(xyz_text) or xyz_text.lstrip()[:4].upper() == "VECT":
        pts = read_vect_curves(xyz_text).points
        if pts.shape[0] < 2:
            raise ValueError("VECT file does not contain at least two 3D points.")
        return pts

    raw_lines = xyz_text.splitlines()
    nonempty_indices = [i for i, line in enumerate(raw_lines) if line.strip()]
    if not nonempty_indices:
        raise ValueError("XYZ file does not contain readable lines.")

    start_index = nonempty_indices[0]
    if first_token_is_integer(raw_lines[start_index]):
        atom_count = int(raw_lines[start_index].strip())
        data_lines = raw_lines[start_index + 2:start_index + 2 + atom_count]
    else:
        data_lines = raw_lines[start_index:]

    pts: List[List[float]] = []
    for line in data_lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        point = parse_xyz_coordinate_line(line)
        if point is not None:
            pts.append(point)
    if len(pts) < 2:
        raise ValueError("XYZ file does not contain at least two 3D points.")
    return np.array(pts, dtype=float)


def read_xyz_curve(path: str) -> np.ndarray:
    """Read curve points from a file path."""
    return read_curve_file(path)[0]


def read_curve_file(path: str):
    """Return (points, stated_closure, n_components).

    stated_closure is True/False for a VECT input, which records closure in the
    sign of each vertex count, and None for every other format, which leaves it
    unsaid.  A VECT file whose polylines disagree reports None too, since the
    concatenated curve this tool interpolates is then neither.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    if is_vect_text(txt) or txt.lstrip()[:4].upper() == "VECT":
        curves = read_vect_curves(txt, source=path)
        pts = curves.points
        if pts.shape[0] < 2:
            raise ValueError("VECT file does not contain at least two 3D points.")
        stated = curves.closed[0] if len(set(curves.closed)) == 1 else None
        return pts, stated, len(curves)
    return read_xyz_curve_from_text(txt), None, 1


def write_xyz_curve(path: str, points: np.ndarray) -> None:
    """Write curve points as plain whitespace-separated x y z lines."""
    with open(path, "w") as f:
        for p in np.asarray(points, dtype=float):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def compute_arc_lengths(points: np.ndarray) -> np.ndarray:
    """Compute cumulative arc-lengths along a polyline.

    points: (M,3)
    returns: (M,) s with s[0]=0 and s[i] = sum_{k<i} |p_{k+1}-p_k|
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Need at least two points to define a curve.")
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.zeros(pts.shape[0], dtype=float)
    s[1:] = np.cumsum(seg_lengths)
    return s


def sample_curve_position(points: np.ndarray,
                          arc_lengths: np.ndarray,
                          s_query: float,
                          eps: float = 1e-8) -> np.ndarray:
    """Sample a position on a polyline at a given arc-length."""
    total = float(arc_lengths[-1])
    if total <= eps:
        raise ValueError("Curve length is too small or degenerate.")

    if s_query <= 0.0:
        return np.asarray(points[0], dtype=float)
    if s_query >= total:
        return np.asarray(points[-1], dtype=float)

    idx = int(np.searchsorted(arc_lengths, s_query) - 1)
    idx = max(0, min(idx, len(points) - 2))

    s0, s1 = float(arc_lengths[idx]), float(arc_lengths[idx + 1])
    # A zero-length segment is guarded explicitly.  Adding an epsilon to this
    # denominator instead would bias every sample toward its segment start,
    # which is a systematic error rather than a safety net.
    t = 0.0 if s1 <= s0 else (s_query - s0) / (s1 - s0)

    p0 = np.asarray(points[idx], dtype=float)
    p1 = np.asarray(points[idx + 1], dtype=float)
    return (1.0 - t) * p0 + t * p1


def _strip_closing_duplicate(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """If last point duplicates first (within eps), drop the last point."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] >= 2 and np.linalg.norm(pts[-1] - pts[0]) <= eps:
        return pts[:-1].copy()
    return pts.copy()


# --------------------------------------------------------------------------
# Even arc-length resampling, shared with svg2xyz.py
#
# This is the one implementation of even resampling in the package.  It works
# in any number of dimensions: svg2xyz feeds it 2D drawing curves and Curve It
# feeds it 3D space curves through --interp-mode n.
# --------------------------------------------------------------------------

# Smallest turning angle, in degrees, that makes a vertex count as a corner.
# Corners are reproduced exactly through resampling; gentler vertices are not.
# Shared so svg2xyz and Curve It cannot drift apart.
DEFAULT_MIN_CORNER_ANGLE = 20.0


def corner_flags(points: np.ndarray,
                 closed: bool,
                 angle_deg: Optional[float],
                 min_segment_fraction: float = 0.0) -> np.ndarray:
    """Mark vertices whose turning angle reaches `angle_deg`.

    Dimension agnostic: the turning angle between two segments is defined by
    their dot product, which is the same in 2D and 3D.

    `min_segment_fraction` optionally rejects turns that only exist because
    of a stub segment.  A drawing program closing a path can leave a segment
    a thousandth the length of its neighbours, and both of its endpoints then
    look like sharp corners; pinning those preserves the stub and defeats
    even spacing.  Set it to, say, 0.25 to require both adjacent segments to
    reach a quarter of the median segment length.  It defaults to 0, meaning
    no filtering, so Curve It's --interp-mode n is unaffected.
    """
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
    if min_segment_fraction and float(min_segment_fraction) > 0.0:
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        steps = steps[steps > 0.0]
        if len(steps):
            shortest = float(min_segment_fraction) * float(np.median(steps))
            flags &= (len_in > shortest) & (len_out > shortest)
    if not closed:
        flags[0] = flags[-1] = False        # the ends are kept regardless
    return flags


def even_points(points: np.ndarray, count: int) -> np.ndarray:
    """`count` points evenly spaced by arc length, both ends included.

    Every coordinate column of the input survives, so a 3D curve stays 3D.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("even_points expects an (N, D) array, got shape %r"
                         % (pts.shape,))
    if len(pts) < 2 or count < 2:
        return pts[:1].copy() if len(pts) else pts.copy()
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    if arc[-1] <= 0.0:
        return pts[:1].copy()
    target = np.linspace(0.0, float(arc[-1]), int(count))
    return np.column_stack([np.interp(target, arc, pts[:, d])
                            for d in range(pts.shape[1])])


def allocate_points(lengths, total: int) -> np.ndarray:
    """Split `total` points between spans in proportion to their length."""
    lengths = np.asarray(lengths, dtype=float)
    count = len(lengths)
    total = max(int(total), count)
    if lengths.sum() <= 0.0:
        return np.ones(count, dtype=int)
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


def resample_curve(points: np.ndarray,
                   closed: bool,
                   n: Optional[int] = None,
                   spacing: Optional[float] = None,
                   min_corner_angle: Optional[float] = DEFAULT_MIN_CORNER_ANGLE,
                   min_segment_fraction: float = 0.0) -> np.ndarray:
    """Evenly space points by arc length, reproducing every corner exactly.

    Blind uniform resampling walks straight past a sharp vertex and clips the
    corners off a rectangle, a polygon or a star.  Corners are therefore
    pinned and each smooth span between them is resampled on its own, in
    proportion to its length.  Pass ``min_corner_angle=0`` for the older
    blind behaviour.

    Closed curves are returned without repeating the first point, the
    convention the rest of the package uses.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("resample_curve expects an (N, D) array, got shape %r"
                         % (pts.shape,))
    if len(pts) < 2:
        return pts.copy()
    loop = np.vstack([pts, pts[:1]]) if closed else pts
    step = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    total_length = float(step.sum())
    if total_length <= 0.0:
        return pts.copy()

    if n is None:
        if spacing is None or float(spacing) <= 0.0:
            return pts.copy()
        n = int(round(total_length / float(spacing)))
    n = max(int(n), 3 if closed else 2)

    corners = corner_flags(pts, closed, min_corner_angle, min_segment_fraction)
    index = list(np.nonzero(corners)[0])
    if not index:
        target = (np.linspace(0.0, total_length, n, endpoint=False) if closed
                  else np.linspace(0.0, total_length, n))
        arc = np.concatenate([[0.0], np.cumsum(step)])
        return np.column_stack([np.interp(target, arc, loop[:, d])
                                for d in range(loop.shape[1])])

    if closed:
        pts = np.roll(pts, -index[0], axis=0)
        index = [i - index[0] for i in index]
        loop = np.vstack([pts, pts[:1]])
        bounds = index + [len(pts)]
        spans = [loop[bounds[k]:bounds[k + 1] + 1] for k in range(len(bounds) - 1)]
        unique_total = n
    else:
        bounds = [0] + index + [len(pts) - 1]
        spans = [pts[bounds[k]:bounds[k + 1] + 1] for k in range(len(bounds) - 1)]
        unique_total = n - 1

    spans = [s for s in spans if len(s) >= 2]
    if not spans:
        return pts.copy()
    lengths = [float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) for s in spans]
    share = allocate_points(lengths, unique_total)

    out = [even_points(span, int(count) + 1)[:-1]
           for span, count in zip(spans, share)]
    if not closed:
        out.append(spans[-1][-1:])
    return np.vstack(out)


def interpolate_curve_n_points(points: np.ndarray,
                               n: int,
                               closed: bool = False,
                               eps: float = 1e-8,
                               min_corner_angle: Optional[float] =
                               DEFAULT_MIN_CORNER_ANGLE) -> np.ndarray:
    """Interpolate a curve to contain exactly n points evenly spaced by arc length.

    Vertices turning by at least ``min_corner_angle`` degrees are reproduced
    exactly, so a polygonal curve keeps its corners; pass 0 for the older
    blind behaviour.  Works in any number of dimensions.
    """
    if int(n) != n:
        raise ValueError("n must be an integer")
    n = int(n)
    pts0 = _strip_closing_duplicate(points) if closed else np.asarray(points, dtype=float)
    if closed:
        if n < 3:
            raise ValueError("For closed curves, n must be >= 3.")
    elif n < 2:
        raise ValueError("For open curves, n must be >= 2.")

    ext = np.vstack([pts0, pts0[:1]]) if closed else pts0
    if float(compute_arc_lengths(ext)[-1]) <= eps:
        raise ValueError("Curve length is too small or degenerate.")
    if not closed and n == 2:
        return np.vstack([pts0[0], pts0[-1]])

    return resample_curve(pts0, closed, n=n, min_corner_angle=min_corner_angle)


def interpolate_curve_insert_p(points: np.ndarray,
                               p: int,
                               closed: bool = False) -> np.ndarray:
    """Insert p equally spaced points between every pair of adjacent points."""
    if int(p) != p:
        raise ValueError("p must be an integer")
    p = int(p)
    if p < 0:
        raise ValueError("p must be >= 0")

    pts0 = np.asarray(points, dtype=float)
    if pts0.shape[0] < 2:
        raise ValueError("Need at least two points.")

    if closed:
        pts0 = _strip_closing_duplicate(pts0)
        M = pts0.shape[0]
        out: List[np.ndarray] = []
        denom = float(p + 1)
        for i in range(M):
            p0 = pts0[i]
            p1 = pts0[(i + 1) % M]
            for j in range(p + 1):
                t = float(j) / denom
                out.append((1.0 - t) * p0 + t * p1)
        return np.vstack(out)

    # Open
    M = pts0.shape[0]
    out = []
    denom = float(p + 1)
    for i in range(M - 1):
        p0 = pts0[i]
        p1 = pts0[i + 1]
        for j in range(p + 1):
            t = float(j) / denom
            out.append((1.0 - t) * p0 + t * p1)
    out.append(pts0[-1])
    return np.vstack(out)


def interpolate_curve(points: np.ndarray,
                      mode: str = "none",
                      n: Optional[int] = None,
                      p: Optional[int] = None,
                      closed: bool = False,
                      min_corner_angle: Optional[float] =
                      DEFAULT_MIN_CORNER_ANGLE) -> np.ndarray:
    """Convenience wrapper for curve interpolation.

    mode:
      - 'none'
      - 'n' : interpolate to n points by arc length
      - 'p' : insert p points between adjacent points
    """
    m = (mode or "none").strip().lower()
    if m in ("none", "off", "false", "0", ""):
        return np.asarray(points, dtype=float)
    if m in ("n", "npoints", "n_points", "num", "num_points"):
        if n is None:
            raise ValueError("mode='n' requires n")
        return interpolate_curve_n_points(points, int(n), closed=closed,
                                          min_corner_angle=min_corner_angle)
    if m in ("p", "pbetween", "p_between", "insert"):
        if p is None:
            raise ValueError("mode='p' requires p")
        return interpolate_curve_insert_p(points, int(p), closed=closed)
    raise ValueError("Unknown interpolation mode: %r" % mode)


def _default_output_path(input_path: str) -> str:
    """Return '<stem>_interpolated<ext>' in the same directory as input."""
    d = os.path.dirname(input_path)
    base = os.path.basename(input_path)
    stem, ext = os.path.splitext(base)
    if not ext:
        return os.path.join(d, stem + "_interpolated")
    # write_xyz_curve always writes plain coordinate rows, so a .vect input
    # must not hand its extension to the output: that would produce a file
    # named VECT that no VECT reader could open.
    if ext.lower() == ".vect":
        ext = ".xyz"
    return os.path.join(d, stem + "_interpolated" + ext)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate an XYZ-like curve file either by setting the total number "
            "of points (n) evenly spaced along arc length, or by inserting p points "
            "between each pair of adjacent points."
        )
    )
    parser.add_argument(
        "input_xyz",
        help="Input XYZ/txt file containing curve points.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--n",
        type=int,
        help="Interpolate to exactly n points evenly spaced along arc length.",
    )
    mode_group.add_argument(
        "--p",
        type=int,
        help="Insert p equally spaced points between each adjacent pair.",
    )

    closure_group = parser.add_mutually_exclusive_group()
    closure_group.add_argument(
        "--closed",
        action="store_true",
        help=(
            "Treat the input curve as a closed loop for interpolation (includes the "
            "segment from the last point back to the first)."
        ),
    )
    closure_group.add_argument(
        "--open",
        dest="open_curve",
        action="store_true",
        help=(
            "Treat the input curve as an open path. Only needed to override a "
            "Geomview VECT file that states it is closed; every other format "
            "leaves closure unsaid and is treated as open already."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output file path. If omitted, '<stem>_interpolated<ext>' is used in the "
            "same directory as the input."
        ),
    )

    args = parser.parse_args(argv)

    pts, stated, n_components = read_curve_file(args.input_xyz)

    # An explicit flag always wins; otherwise a VECT file is believed, because
    # it is the one input format that states closure rather than leaving it to
    # be guessed from the gap back to the first point.
    closed = bool(args.closed)
    if not args.closed and not args.open_curve and stated is not None:
        closed = stated
        print(f"[INFO] Curve file states {'a closed' if closed else 'an open'} "
              f"curve; interpolating that way.")
    if n_components > 1:
        print(f"[WARN] Input holds {n_components} components, joined end to end "
              f"into one curve of {pts.shape[0]} points. Split the file first "
              f"to interpolate them separately.")

    if args.n is not None:
        out_pts = interpolate_curve_n_points(pts, args.n, closed=closed)
    else:
        out_pts = interpolate_curve_insert_p(pts, args.p, closed=closed)

    out_path = args.output or _default_output_path(args.input_xyz)
    write_xyz_curve(out_path, out_pts)

    print(f"[INFO] Input points:  {pts.shape[0]}")
    print(f"[INFO] Output points: {out_pts.shape[0]}")
    print(f"[INFO] Wrote: {out_path}")


if __name__ == "__main__":
    main()
