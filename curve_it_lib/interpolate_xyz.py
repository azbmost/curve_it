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


def read_xyz_curve_from_text(xyz_text: str) -> np.ndarray:
    """Read a 3D polyline from a generic XYZ-like text string.

    Accepts:
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
    with open(path, "r") as f:
        txt = f.read()
    return read_xyz_curve_from_text(txt)


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
    t = (s_query - s0) / (s1 - s0 + eps)

    p0 = np.asarray(points[idx], dtype=float)
    p1 = np.asarray(points[idx + 1], dtype=float)
    return (1.0 - t) * p0 + t * p1


def _strip_closing_duplicate(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """If last point duplicates first (within eps), drop the last point."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] >= 2 and np.linalg.norm(pts[-1] - pts[0]) <= eps:
        return pts[:-1].copy()
    return pts.copy()


def interpolate_curve_n_points(points: np.ndarray,
                               n: int,
                               closed: bool = False,
                               eps: float = 1e-8) -> np.ndarray:
    """Interpolate a curve to contain exactly n points evenly spaced by arc length."""
    if int(n) != n:
        raise ValueError("n must be an integer")
    n = int(n)
    if closed:
        if n < 3:
            raise ValueError("For closed curves, n must be >= 3.")
        pts0 = _strip_closing_duplicate(points)
        # Build an explicit closing segment for arc-length parametrization.
        pts_ext = np.vstack([pts0, pts0[0]])
        arc = compute_arc_lengths(pts_ext)
        total = float(arc[-1])
        if total <= eps:
            raise ValueError("Curve length is too small or degenerate.")
        step = total / float(n)
        s_samples = (np.arange(n, dtype=float) * step)
        out = np.zeros((n, 3), dtype=float)
        for i, s in enumerate(s_samples):
            out[i] = sample_curve_position(pts_ext, arc, float(s), eps=eps)
        return out

    # Open curve
    if n < 2:
        raise ValueError("For open curves, n must be >= 2.")
    pts0 = np.asarray(points, dtype=float)
    arc = compute_arc_lengths(pts0)
    total = float(arc[-1])
    if total <= eps:
        raise ValueError("Curve length is too small or degenerate.")
    if n == 2:
        return np.vstack([pts0[0], pts0[-1]])
    step = total / float(n - 1)
    s_samples = (np.arange(n, dtype=float) * step)
    out = np.zeros((n, 3), dtype=float)
    for i, s in enumerate(s_samples):
        out[i] = sample_curve_position(pts0, arc, float(s), eps=eps)
    return out


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
                      closed: bool = False) -> np.ndarray:
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
        return interpolate_curve_n_points(points, int(n), closed=closed)
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

    parser.add_argument(
        "--closed",
        action="store_true",
        help=(
            "Treat the input curve as a closed loop for interpolation (includes the "
            "segment from the last point back to the first)."
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

    pts = read_xyz_curve(args.input_xyz)

    if args.n is not None:
        out_pts = interpolate_curve_n_points(pts, args.n, closed=args.closed)
    else:
        out_pts = interpolate_curve_insert_p(pts, args.p, closed=args.closed)

    out_path = args.output or _default_output_path(args.input_xyz)
    write_xyz_curve(out_path, out_pts)

    print(f"[INFO] Input points:  {pts.shape[0]}")
    print(f"[INFO] Output points: {out_pts.shape[0]}")
    print(f"[INFO] Wrote: {out_path}")


if __name__ == "__main__":
    main()
