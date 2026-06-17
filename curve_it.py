#!/usr/bin/env python3
"""
Curve It (CLI + GUI; holonomy compensation, group-wise mapping, scale-anchor
control, path-start control, and axial twist control).

This script takes:
    (1) a PDB file containing a roughly straight DNA/RNA helix, protein helix,
        or other filament-like PDB structure, and
    (2) optionally, an XYZ-like text file defining a 3D curve.

It produces a new PDB in which the structure has been "bent" so that its
principal axis follows the curve.

If the curve file is omitted, a default planar circular ring is used
as the curve.

Curve interpolation (optional):
    You can optionally interpolate/resample the input curve points *before*
    embedding the helix. Two modes are supported:
        (1) interp_mode='n': resample to exactly n points evenly spaced along
            the curve by arc length.
        (2) interp_mode='p': insert p equally spaced points between every pair
            of adjacent input points.
    (CLI: --interp-mode {none,n,p} with --interp-n / --interp-p;
     GUI: see the 'Curve parameters' section.)

Two modes:
    - CLI mode (default when enough arguments are provided):
        behaves like the original curve_na.py, controlled by command-line options.
    - GUI mode (when no arguments or only --gui/-g are given):
        opens a simple Tkinter GUI where you can:
            * Choose the helix PDB file and see the estimated helix axis length.
            * Choose the curve (XYZ/txt) file and see its length, total curvature,
              and writhe (if curvature tools are available).
            * View the 3D curve using the view_xyzV2.py plotting code (if importable).
            * Adjust all mapping parameters (scale-mode, scale-anchor, path-type,
              helix_phase, twist, path-start).
            * Run the embedding and write the output PDB.

Geometry:
- The straight helix is decomposed into an axis coordinate (along its principal
  component) plus a radial offset in an orthonormal basis.
- The input curve is parametrized by arc length and equipped with a
  rotation-minimizing (parallel transport / Bishop) frame.
- Each atom group is treated as a rigid body: nucleic acid residues use
  phosphate/sugar/base groups, while protein and other residues are grouped
  residue-by-residue. The group center along the input axis is mapped to the
  curve, and all atoms in the group are transformed using the same local frame,
  preserving their internal geometry.
- For closed paths with numeric --scale-mode, holonomy of the frame is
  compensated per wrap to avoid twist discontinuities at the seam.

Inputs (CLI mode):
    helix_pdb: PDB file with a straight DNA/RNA helix, protein helix, or other
               filament-like structure (required in CLI)
    curve_xyz: XYZ-like file with curve points (optional)
               If omitted, a default ring curve is used.
    output_pdb: Output PDB file (optional, via -o/--output-pdb)

Scaling options (via --scale-mode / GUI):
    - 'curve_to_helix' (default):
        scale the curve so its length matches the helix axis length.
    - 'none':
        do not scale the curve; the helix is distributed over the given curve.
    - 'helix_to_curve':
        alias for 'none'.
    - <positive number, in Å>:
        scale the curve so its length equals this number, but **do not remap
        the helix to fill that length**; helix spacing is preserved and
        mismatches appear as gaps/overlaps (possibly clashes).

Scaling anchor (via --scale-anchor / GUI):
    - 'centroid' (default):
        scale the curve about its own centroid (center of mass of the points).
        This preserves the overall position of the path while changing its size.
    - 'origin':
        scale the curve about (0, 0, 0).
    - 'x,y,z':
        scale the curve about an explicit point in Å, e.g. '0,0,0' or '10,0,0'.
        The chosen anchor point stays fixed in space during scaling.

Path start (via --path-start / GUI):
    - For closed curves, a value between 0.0 and 1.0 (default 0.0) that controls
      where along the loop the helix starts to be embedded, i.e. where the PDB
      connectivity break lies on the loop. 0.0 uses the first point in the
      XYZ file as the seam; 0.25 places the seam one quarter of the way
      around the loop, etc. For open paths this option is ignored.

Additional behavior:
    - If a user-supplied curve_xyz file is rescaled ('curve_to_helix' or a
      numeric --scale-mode that changes length), a rescaled XYZ file is written
      alongside the original, named '<curve_xyz_basename>_rescaled.ext'.
    - '--helix_phase ANGLE' (degrees) rotates the helix cross-sections about
      their own axis before embedding.
    - '--twist ANGLE' (degrees) applies an additional linear twist of the helix
      about its own axis along its length before embedding. Positive values
      twist right-handed, negative values twist left-handed.
    - For closed paths with numeric --scale-mode, holonomy of the frame is
      compensated per lap to eliminate twist jumps.
    - If curve_it_lib/cal_xyz_total_curvature_writheV2.py is importable, the total curvature
      and writhe of the (scaled) curve are reported for closed paths, and
      are also shown in the GUI for loaded curves (assuming closed).

CLI examples:
    python curve_it.py helix.pdb
    python curve_it.py helix.pdb curve.xyz
    python curve_it.py helix.pdb curve.xyz --scale-mode none
    python curve_it.py helix.pdb curve.xyz --scale-mode helix_to_curve
    python curve_it.py helix.pdb curve.xyz --path-type closed --scale-mode 340.0
    python curve_it.py helix.pdb curve.xyz --scale-mode curve_to_helix --scale-anchor origin
    python curve_it.py helix.pdb curve.xyz --scale-mode 340.0 --scale-anchor 0,0,0
    python curve_it.py helix.pdb curve.xyz --helix_phase 90.0
    python curve_it.py helix.pdb curve.xyz --twist 360.0
    python curve_it.py --gui
    python curve_it.py  # (GUI mode, equivalent to --gui)

GUI usage:
    Simply run without arguments:
        python curve_it.py
    or
        python curve_it.py --gui
"""

import argparse
import contextlib
import io
import os
import shlex
import sys
import textwrap
from typing import List, Tuple, Dict, Optional, Any

import numpy as np

APP_NAME = "curve_it"
APP_VERSION = "V2.4"
APP_TITLE = "re_helix is AZBMOST Package Module #3 - Fit PDB along Any Curve"


# Optional import of curve interpolation helper.
try:
    from curve_it_lib import interpolate_xyz  # noqa: F401
    HAVE_INTERPOLATE_XYZ = True
except Exception:
    HAVE_INTERPOLATE_XYZ = False


# Optional import of curvature & writhe utilities + smoothing support
try:
    from curve_it_lib.cal_xyz_total_curvature_writheV2 import (
        fit_spline_and_calculate_curvature,
        calculate_writhe,
    )
    from scipy.signal import savgol_filter
    HAVE_CURVATURE_WRITHE = True
except Exception:
    HAVE_CURVATURE_WRITHE = False

# The optional Matplotlib viewer is imported lazily by the GUI so simple CLI
# commands such as --version do not trigger font/cache setup warnings.


class AtomRecord:
    """Container for an atom line, coordinates, and basic PDB metadata."""
    def __init__(
        self,
        line: str,
        coord: np.ndarray,
        atom_name: str,
        atom_name_norm: str,
        res_name: str,
        chain_id: str,
        res_seq: int,
        i_code: str,
        element: str,
    ):
        self.line = line
        self.coord = coord
        self.atom_name = atom_name          # raw atom name (trimmed)
        self.atom_name_norm = atom_name_norm  # normalized name (C1*,C1' -> C1')
        self.res_name = res_name
        self.chain_id = chain_id
        self.res_seq = res_seq
        self.i_code = i_code
        self.element = element  # guessed if not present in PDB


def normalize_atom_name(atom_name: str) -> str:
    """
    Normalize atom names:
    - strip spaces
    - uppercase
    - convert old '*' or '`' primes to "'"
    """
    name = atom_name.strip().upper()
    name = name.replace("*", "'").replace("`", "'")
    return name


def guess_element(line: str, atom_name_norm: str) -> str:
    """
    Guess element symbol from PDB line or atom name.
    Uses columns 77-78 if available; otherwise first letter in atom_name_norm.
    """
    elem = ""
    if len(line) >= 78:
        elem = line[76:78].strip()
    if not elem:
        letters = [c for c in atom_name_norm if c.isalpha()]
        if letters:
            elem = letters[0]
    return elem.upper()


def resource_path(filename: str) -> str:
    """Return a project resource path, including when bundled by PyInstaller."""
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, filename)


def parse_pdb_atoms(pdb_text: str) -> List[AtomRecord]:
    """
    Parse ATOM/HETATM records from a PDB string, extracting coordinates and
    basic metadata (atom name, residue, chain, element).

    Coordinates are taken from standard fixed-width columns (30:38, 38:46, 46:54).
    """
    atoms: List[AtomRecord] = []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        coord = np.array([x, y, z], dtype=float)

        raw_name = line[12:16]
        atom_name = raw_name.strip()
        atom_name_norm = normalize_atom_name(atom_name)

        res_name = line[17:20].strip()
        chain_id = line[21].strip() if len(line) > 21 else ""
        res_seq_str = line[22:26] if len(line) > 26 else ""
        try:
            res_seq = int(res_seq_str.strip()) if res_seq_str.strip() else 0
        except ValueError:
            res_seq = 0
        i_code = line[26].strip() if len(line) > 26 else ""

        element = guess_element(line, atom_name_norm)

        atoms.append(
            AtomRecord(
                line.rstrip("\n"),
                coord,
                atom_name,
                atom_name_norm,
                res_name,
                chain_id,
                res_seq,
                i_code,
                element,
            )
        )
    return atoms


def read_xyz_curve_from_text(xyz_text: str) -> np.ndarray:
    """
    Read a 3D polyline from a generic XYZ-like text file.

    Accepts:
    - Standard XYZ: first line = number of atoms, second = comment, remaining
      lines 'Element x y z'
    - Or a simple whitespace separated 'x y z' per line (with optional comments
      starting with # or !).

    Strategy: on each line, collect all tokens that can be parsed as floats;
    if there are at least three, use the first three as x,y,z.
    """
    pts = []
    for line in xyz_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        parts = line.split()
        floats: List[float] = []
        for tok in parts:
            try:
                floats.append(float(tok))
            except ValueError:
                continue
        if len(floats) >= 3:
            pts.append(floats[:3])
    if len(pts) < 2:
        raise ValueError("XYZ file does not contain at least two 3D points.")
    return np.array(pts, dtype=float)


def generate_ring_curve(num_points: int = 200, radius: float = 10.0) -> np.ndarray:
    """
    Generate a default planar circular ring in the xy-plane.

    num_points: number of sample points along the ring
    radius: circle radius in Å
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros_like(x)
    pts = np.column_stack((x, y, z))
    return pts


def apply_curve_interpolation(points: np.ndarray,
                              interp_mode: str = "none",
                              interp_n: int = 200,
                              interp_p: int = 0,
                              closed: bool = False,
                              verbose: bool = False) -> np.ndarray:
    """Interpolate curve points before embedding.

    Two interpolation modes are supported (controlled by interp_mode):

      - 'n'   : resample the curve so the output contains exactly interp_n
               points evenly spaced by arc length.
      - 'p'   : insert interp_p equally spaced points between each adjacent
               input point pair.
      - 'none': do not change the input curve points (default).

    closed controls whether the curve is treated as a closed loop for the
    interpolation step (for example, whether the last->first segment is
    included).
    """
    mode = (interp_mode or "none").strip().lower()
    if mode in ("none", "off", "false", "0", ""):
        return np.asarray(points, dtype=float)

    if not HAVE_INTERPOLATE_XYZ:
        raise ImportError(
            "Curve interpolation requested but interpolate_xyz.py could not be imported. "
            "Make sure curve_it_lib/interpolate_xyz.py is available."
        )

    pts_in = np.asarray(points, dtype=float)
    if mode in ("n", "n_points", "npoints", "num", "num_points"):
        n = int(interp_n)
        pts_out = interpolate_xyz.interpolate_curve(
            pts_in, mode="n", n=n, closed=closed
        )
        if verbose:
            print(f"[INFO] Curve interpolation (n={n}, closed={closed}): "
                  f"{pts_in.shape[0]} -> {pts_out.shape[0]} points")
        return pts_out

    if mode in ("p", "p_between", "pbetween", "insert"):
        p = int(interp_p)
        pts_out = interpolate_xyz.interpolate_curve(
            pts_in, mode="p", p=p, closed=closed
        )
        if verbose:
            print(f"[INFO] Curve interpolation (p={p}, closed={closed}): "
                  f"{pts_in.shape[0]} -> {pts_out.shape[0]} points")
        return pts_out

    raise ValueError(f"Unknown interp_mode '{interp_mode}'. Use 'none', 'n', or 'p'.")


def compute_helix_local_coords(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray,
                                                           np.ndarray]:
    """
    Given an (N,3) array of atom coordinates for a (roughly) straight helix,
    compute:

        center : (3,) mean position
        axis   : (3,) unit vector along best-fit helix axis (principal component)
        basis  : (3,3) matrix whose columns are (x_axis, y_axis, axis)
        s_vals : (N,) projection of each atom along 'axis'
        local  : (N,3) radial offsets expressed in 'basis' coordinates

    For a perfectly straight helix, local[:,2] should be ~0 (all radial offsets
    lie in the plane perpendicular to the axis).
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be an (N,3) array")

    center = coords.mean(axis=0)
    rel = coords - center

    cov = rel.T @ rel / coords.shape[0]
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argmax(vals)]
    axis = axis / np.linalg.norm(axis)

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(tmp, axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    basis = np.column_stack((x_axis, y_axis, axis))

    s_vals = rel @ axis
    radial = rel - np.outer(s_vals, axis)
    local = radial @ basis

    return center, axis, basis, s_vals, local


def compute_arc_lengths(points: np.ndarray) -> np.ndarray:
    """
    Compute cumulative arc-lengths along a polyline.

    points: (M,3) array
    returns: (M,) array s with s[0]=0 and s[i] = sum_{k < i} |p_{k+1}-p_k|
    """
    if points.shape[0] < 2:
        raise ValueError("Need at least two points to define a curve.")
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.zeros(points.shape[0], dtype=float)
    s[1:] = np.cumsum(seg_lengths)
    return s


def get_scale_anchor_point(scale_anchor: str, pts: np.ndarray) -> np.ndarray:
    """
    Return the anchor point for scaling the curve.

    scale_anchor:
        - 'centroid' (default): use the mean of the curve points.
        - 'origin': use (0,0,0).
        - 'x,y,z': explicit coordinates in Å, e.g. '10.0,0.0,0.0'.

    If parsing fails, falls back to the centroid and prints a warning.
    """
    if scale_anchor is None:
        scale_anchor = "centroid"
    opt = scale_anchor.strip()

    low = opt.lower()
    if low in ("centroid", "center", "centre"):
        return pts.mean(axis=0)
    if low == "origin":
        return np.zeros(3, dtype=float)

    try:
        parts = opt.split(",")
        if len(parts) == 3:
            vals = [float(p) for p in parts]
            return np.array(vals, dtype=float)
    except Exception:
        pass

    print(f"[WARNING] Could not parse --scale-anchor='{scale_anchor}', "
          "falling back to centroid.")
    return pts.mean(axis=0)


def _strip_closing_duplicate(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """If last point duplicates first (within eps), drop the last point."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] >= 2 and np.linalg.norm(pts[-1] - pts[0]) <= eps:
        return pts[:-1].copy()
    return pts.copy()


def _turning_angles_deg_closed(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Turning angles (degrees) between consecutive segments of a closed polyline."""
    pts = _strip_closing_duplicate(points, eps=1e-6)
    n = pts.shape[0]
    if n < 3:
        return np.zeros(0, dtype=float)

    ang: List[float] = []
    for i in range(n):
        v_prev = pts[i] - pts[i - 1]
        v_next = pts[(i + 1) % n] - pts[i]
        n1 = float(np.linalg.norm(v_prev))
        n2 = float(np.linalg.norm(v_next))
        if n1 <= eps or n2 <= eps:
            continue
        t1 = v_prev / n1
        t2 = v_next / n2
        dot = float(np.clip(np.dot(t1, t2), -1.0, 1.0))
        ang.append(float(np.degrees(np.arccos(dot))))

    if not ang:
        return np.zeros(0, dtype=float)
    return np.array(ang, dtype=float)


def curve_looks_polygonal(points: np.ndarray,
                          angle_large_deg: float = 45.0,
                          angle_small_deg: float = 1.0,
                          frac_nontrivial_threshold: float = 0.25) -> bool:
    """Heuristic to detect mostly-straight polylines with a few sharp corners.

    This matters because Savitzky–Golay smoothing and periodic spline fitting
    can introduce 'ringing' near sharp corners, inflating total curvature.
    """
    angles = _turning_angles_deg_closed(points)
    if angles.size == 0:
        return False
    max_ang = float(np.max(angles))
    frac_nontrivial = float(np.mean(angles > angle_small_deg))
    return (max_ang >= angle_large_deg) and (frac_nontrivial <= frac_nontrivial_threshold)


def compute_discrete_total_curvature(points: np.ndarray,
                                     closed: bool = True,
                                     eps: float = 1e-12) -> float:
    """Compute total curvature of a polyline as the sum of turning angles (radians).

    For closed curves, this is exact for polygons (curvature is concentrated at vertices).
    """
    if not closed:
        raise ValueError("compute_discrete_total_curvature is defined for closed curves only.")

    pts = _strip_closing_duplicate(points, eps=1e-6)
    n = pts.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for a closed curve.")

    total = 0.0
    for i in range(n):
        v_prev = pts[i] - pts[i - 1]
        v_next = pts[(i + 1) % n] - pts[i]
        n1 = float(np.linalg.norm(v_prev))
        n2 = float(np.linalg.norm(v_next))
        if n1 <= eps or n2 <= eps:
            continue
        t1 = v_prev / n1
        t2 = v_next / n2
        dot = float(np.clip(np.dot(t1, t2), -1.0, 1.0))
        total += float(np.arccos(dot))
    return total


def smooth_closed_curve(points: np.ndarray) -> np.ndarray:
    """
    Smooth a closed curve using the same Savitzky–Golay filter strategy
    as in cal_xyz_total_curvature_writhe.read_xyz(), but operating on an
    in-memory (N,3) array of points instead of reading from file.

    IMPORTANT:
        For curves with sharp corners (polygon-like polylines), Savitzky–Golay
        smoothing can introduce oscillations ('ringing') near corners that
        artificially inflate total curvature. In that case we skip smoothing
        and return the input unchanged.

    If there are too few points, returns the input unchanged.
    """
    if not HAVE_CURVATURE_WRITHE:
        return points

    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 7:
        return pts

    # Avoid corner ringing for polygon-like curves.
    if curve_looks_polygonal(pts):
        return pts

    # Choose an odd window size ~ N/18 but at least 7.
    smooth_window_size = max(7, (n // 18) * 2 + 1)
    polyorder = 3
    if smooth_window_size <= polyorder:
        smooth_window_size = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3

    half_window = smooth_window_size // 2

    pts_ext = np.concatenate((pts[-half_window:], pts, pts[:half_window]))

    if smooth_window_size >= len(pts_ext):
        smooth_window_size = len(pts_ext) - 1
        if smooth_window_size % 2 == 0:
            smooth_window_size -= 1

    smoothed_ext = savgol_filter(
        pts_ext,
        window_length=smooth_window_size,
        polyorder=polyorder,
        axis=0,
    )
    smoothed = smoothed_ext[half_window:-half_window]
    return smoothed



def smooth_closed_curve_force(points: np.ndarray) -> np.ndarray:
    """
    Smooth a closed curve with Savitzky-Golay *without* polygon/corner guard.

    This is only used when the user explicitly requests spline-based curvature
    on a cornered/polyline-like curve. It reproduces the original smoothing
    behavior (pre-V2.2) that can introduce corner rounding/overshoot.

    If there are too few points, returns the input unchanged.
    """

    
    if not HAVE_CURVATURE_WRITHE:
        return points

    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 7:
        return pts

    # Choose an odd window size ~ N/18 but at least 7.
    smooth_window_size = max(7, (n // 18) * 2 + 1)
    polyorder = 3
    if smooth_window_size <= polyorder:
        smooth_window_size = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3

    half_window = smooth_window_size // 2

    pts_ext = np.concatenate((pts[-half_window:], pts, pts[:half_window]))

    if smooth_window_size >= len(pts_ext):
        smooth_window_size = len(pts_ext) - 1
        if smooth_window_size % 2 == 0:
            smooth_window_size -= 1

    smoothed_ext = savgol_filter(
        pts_ext,
        window_length=smooth_window_size,
        polyorder=polyorder,
        axis=0,
    )
    smoothed = smoothed_ext[half_window:-half_window]
    return smoothed

def compute_parallel_transport_frames(points: np.ndarray,
                                      eps: float = 1e-8
                                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a discrete parallel-transport (Bishop) frame along a polyline.

    Returns three (M,3) arrays: N, B, T at each point, where T is the unit
    tangent, and N,B span the normal plane. The frame is propagated by
    projecting the previous normal onto the new normal plane, which closely
    approximates a rotation-minimizing frame.
    """
    M = points.shape[0]
    if M < 2:
        raise ValueError("Need at least two points to define frames.")

    N = np.zeros((M, 3), dtype=float)
    B = np.zeros((M, 3), dtype=float)
    T = np.zeros((M, 3), dtype=float)

    v0 = points[1] - points[0]
    if np.linalg.norm(v0) < eps:
        raise ValueError("First two points of curve are coincident.")
    T0 = v0 / np.linalg.norm(v0)

    # Choose an arbitrary normal not parallel to T0.
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, T0)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    N0 = tmp - np.dot(tmp, T0) * T0
    N0 /= np.linalg.norm(N0)
    B0 = np.cross(T0, N0)
    B0 /= np.linalg.norm(B0)

    N[0], B[0], T[0] = N0, B0, T0

    for i in range(1, M):
        v = points[i] - points[i - 1]
        if np.linalg.norm(v) < eps:
            N[i] = N[i - 1]
            B[i] = B[i - 1]
            T[i] = T[i - 1]
            continue

        Ti = v / np.linalg.norm(v)
        T[i] = Ti

        n_prev = N[i - 1]
        n_i = n_prev - np.dot(n_prev, Ti) * Ti
        n_norm = np.linalg.norm(n_i)

        if n_norm < eps:
            b_prev = B[i - 1]
            n_i = b_prev - np.dot(b_prev, Ti) * Ti
            n_norm = np.linalg.norm(n_i)

            if n_norm < eps:
                tmp = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(tmp, Ti)) > 0.9:
                    tmp = np.array([0.0, 1.0, 0.0])
                n_i = tmp - np.dot(tmp, Ti) * Ti
                n_norm = np.linalg.norm(n_i)

        n_i /= n_norm
        b_i = np.cross(Ti, n_i)
        b_i /= np.linalg.norm(b_i)

        N[i], B[i] = n_i, b_i

    return N, B, T


def sample_curve_with_frame(points: np.ndarray,
                            arc_lengths: np.ndarray,
                            N: np.ndarray,
                            B: np.ndarray,
                            T: np.ndarray,
                            s_query: float,
                            eps: float = 1e-8
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample position and frame at a given arc-length along the curve.
    """
    total = arc_lengths[-1]
    if s_query <= 0:
        return points[0], N[0], B[0], T[0]
    if s_query >= total:
        return points[-1], N[-1], B[-1], T[-1]

    idx = int(np.searchsorted(arc_lengths, s_query) - 1)
    idx = max(0, min(idx, len(points) - 2))

    s0, s1 = arc_lengths[idx], arc_lengths[idx + 1]
    t = (s_query - s0) / (s1 - s0 + eps)

    p0, p1 = points[idx], points[idx + 1]
    pos = (1.0 - t) * p0 + t * p1

    T_interp = (1.0 - t) * T[idx] + t * T[idx + 1]
    t_norm = np.linalg.norm(T_interp)
    if t_norm < eps:
        T_interp = T[idx]
    else:
        T_interp /= t_norm

    N_interp_raw = (1.0 - t) * N[idx] + t * N[idx + 1]
    N_proj = N_interp_raw - np.dot(N_interp_raw, T_interp) * T_interp
    n_norm = np.linalg.norm(N_proj)

    if n_norm < eps:
        N_proj = N[idx] - np.dot(N[idx], T_interp) * T_interp
        n_norm = np.linalg.norm(N_proj)
        if n_norm < eps:
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, T_interp)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            N_proj = tmp - np.dot(tmp, T_interp) * T_interp
            n_norm = np.linalg.norm(N_proj)

    N_s = N_proj / n_norm
    B_s = np.cross(T_interp, N_s)
    B_s /= np.linalg.norm(B_s)

    return pos, N_s, B_s, T_interp


def sample_curve_position(points: np.ndarray,
                          arc_lengths: np.ndarray,
                          s_query: float,
                          eps: float = 1e-8) -> np.ndarray:
    """
    Sample only the position on the curve at a given arc-length (no frame).

    points      : (M,3) curve points
    arc_lengths : (M,) cumulative arc-lengths with arc_lengths[0]=0, arc_lengths[-1]=total
    s_query     : arc-length in [0,total]; values outside are clamped
    """
    total = arc_lengths[-1]
    if total <= eps:
        raise ValueError("Curve length is too small or degenerate.")

    if s_query <= 0.0:
        return points[0]
    if s_query >= total:
        return points[-1]

    idx = int(np.searchsorted(arc_lengths, s_query) - 1)
    idx = max(0, min(idx, len(points) - 2))

    s0, s1 = arc_lengths[idx], arc_lengths[idx + 1]
    t = (s_query - s0) / (s1 - s0 + eps)

    p0, p1 = points[idx], points[idx + 1]
    pos = (1.0 - t) * p0 + t * p1
    return pos


def reparameterize_closed_curve(points: np.ndarray,
                                arc_lengths: np.ndarray,
                                frac: float,
                                eps: float = 1e-8
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reparameterize a *closed* curve so that arc-length s=0 corresponds to
    a fractional position 'frac' ∈ [0,1] along the original closed loop.

    points      : (M,3) array, representing one traversal of a closed curve.
                  For closed paths we assume the representation already includes
                  the closing segment (either last==first or last->first segment).
    arc_lengths : (M,) cumulative arc-lengths with arc_lengths[0]=0
                  and arc_lengths[-1]=total length of one loop.
    frac        : desired starting position as a fraction of total length.

    Returns:
        new_points : (M,3) array of reparameterized points (still closed)
        new_arc    : (M,) cumulative arc-lengths for new_points
    """
    if points.shape[0] < 2:
        raise ValueError("Need at least two points to reparameterize a curve.")

    total = float(arc_lengths[-1])
    if total <= eps:
        raise ValueError("Curve length is too small or degenerate.")

    f = float(frac)
    if f < 0.0 or f > 1.0:
        # Clamp and warn: caller should already have validated
        print(f"[WARNING] reparameterize_closed_curve: frac={f:.3f} outside [0,1]; clamping.")
        f = max(0.0, min(1.0, f))

    if f <= eps or f >= 1.0 - eps:
        # Very close to 0 or 1: effectively no shift
        return points.copy(), arc_lengths.copy()

    s_start = f * total
    M = points.shape[0]

    new_points = np.zeros_like(points)
    for j in range(M):
        # Original parameter positions along the loop
        s_j = float(arc_lengths[j])
        s_shift = s_j + s_start
        if s_shift >= total:
            s_shift -= total
        new_points[j] = sample_curve_position(points, arc_lengths, s_shift)

    new_arc = compute_arc_lengths(new_points)
    return new_points, new_arc


def has_prime(name_norm: str) -> bool:
    """Return True if atom name contains a prime (new or old naming converted)."""
    return "'" in name_norm


def is_phosphate_oxygen(name_norm: str) -> bool:
    """
    Heuristic to detect phosphate oxygens (old and new naming):
    OP1, OP2, OP3, O1P, O2P, O3P, O4P, and generic O?P patterns.
    """
    n = name_norm
    if n in {"OP1", "OP2", "O1P", "O2P"}:
        return True
    if len(n) == 3 and n[0] == "O" and n[2] == "P":
        return True
    return False


STANDARD_PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
    "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "MSE",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL", "ASX", "GLX", "UNK",
}

STANDARD_NUCLEIC_ACID_RESIDUES = {
    "A", "C", "G", "U", "T", "I",
    "DA", "DC", "DG", "DT", "DI", "DU",
    "ADE", "CYT", "GUA", "URA", "THY",
    "RA", "RC", "RG", "RU",
}


def classify_residue_type(res_name: str) -> str:
    """Classify residue chemistry for grouping during curve mapping."""
    name = (res_name or "").strip().upper()
    if name in STANDARD_NUCLEIC_ACID_RESIDUES:
        return "nucleic_acid"
    if name in STANDARD_PROTEIN_RESIDUES:
        return "protein"
    return "other"


def classify_atom_group(atom: AtomRecord) -> str:
    """
    Classify a nucleic acid atom into a group within its residue:
        'P' : phosphate group
        'S' : sugar group
        'B' : base group (or "other")

    Heuristics (robust to old/new naming):
        - P and OP*/O?P oxygens -> phosphate
        - C/O with prime (C1', C2', O4', O5', etc.) -> sugar
        - N atoms -> base
        - H atoms with prime in name -> sugar, else base
        - Everything else -> base
    """
    name = atom.atom_name_norm
    elem = atom.element.upper() if atom.element else ""

    if elem == "P":
        return "P"
    if elem == "O":
        if is_phosphate_oxygen(name):
            return "P"
        if has_prime(name):
            return "S"
        return "B"
    if elem == "C":
        if has_prime(name):
            return "S"
        return "B"
    if elem == "N":
        return "B"
    if elem == "H":
        if has_prime(name):
            return "S"
        return "B"
    return "B"


def build_atom_groups(
    atoms: List[AtomRecord],
    s_vals: np.ndarray
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Build groups of atom indices.

    Nucleic acid residues are grouped by phosphate/sugar/base, matching the
    original Curve NA behavior. Protein and unknown residues are grouped as
    whole residues, which lets alpha-helical or otherwise elongated protein PDBs
    follow a target curve without using nucleic-acid-specific atom labels.

    Returns:
        group_atom_indices : list of list of atom indices
        group_s            : (G,) array of group center axis coordinate (s along helix)

    Every atom is assigned exactly one group; groups are keyed by
    (chain_id, res_seq, i_code, group_type).
    """
    groups_by_key: Dict[Tuple[str, int, str, str], int] = {}
    group_atom_indices: List[List[int]] = []

    for idx, atom in enumerate(atoms):
        residue_type = classify_residue_type(atom.res_name)
        if residue_type == "nucleic_acid":
            gtype = classify_atom_group(atom)
        else:
            gtype = "RES"
        key = (atom.chain_id, atom.res_seq, atom.i_code, gtype)
        if key not in groups_by_key:
            groups_by_key[key] = len(group_atom_indices)
            group_atom_indices.append([])
        gid = groups_by_key[key]
        group_atom_indices[gid].append(idx)

    group_s = np.zeros(len(group_atom_indices), dtype=float)
    for gid, idx_list in enumerate(group_atom_indices):
        group_s[gid] = float(s_vals[idx_list].mean())

    return group_atom_indices, group_s


def embed_helix_on_curve(atoms: List[AtomRecord],
                         curve_points: np.ndarray,
                         scale_mode: str = "curve_to_helix",
                         path_type: str = "open",
                         helix_phase: float = 0.0,
                         twist: float = 0.0,
                         scale_anchor: str = "centroid",
                         path_start: float = 0.0
                         ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Core geometry: map a straight helix onto a 3D curve.

    atoms       : list of AtomRecord (order defines output order)
    curve_points: (M,3) array of curve points
    scale_mode  :
        - 'curve_to_helix' (default): scale curve length to match helix length
          and distribute the helix evenly over the full curve.
        - 'none' or 'helix_to_curve': keep curve unchanged; helix is
          distributed over its length.
        - any positive number (as string or float): target curve length in Å.
          The curve is scaled to that length, but the helix axis coordinate
          itself is used for mapping, so its axial spacing is preserved.
    path_type   : 'open' (default) or 'closed'.
    helix_phase : global rotation (in degrees) of the helix cross-sections
                  about its own axis before embedding.
    twist       : total additional twist (in degrees) applied along the helix
                  axis before embedding. The twist is distributed linearly from
                  0° at the helix start to 'twist' at the helix end; positive
                  values correspond to right-handed twisting (looking along
                  the axis), negative to left-handed.
    scale_anchor: controls the anchor point used when scaling the curve
                  (centroid/origin/x,y,z).
    path_start  : for closed paths, a value in [0,1] choosing where along the
                  loop the helix starts (i.e. the location of the PDB seam).
                  Ignored for open paths.

    Returns:
        new_coords         : (N,3) array of curved helix coordinates.
        curve_pts_scaled   : (M',3) array of curve points after any scaling
                             and closure handling (and reparameterization).
        scaling_applied    : bool indicating whether a non-trivial scaling
                             (scale != 1) was applied to the curve.
    """
    # ---- Helix decomposition ----
    coords = np.vstack([a.coord for a in atoms])
    center, axis, basis, s_vals, local = compute_helix_local_coords(coords)
    s_min = float(s_vals.min())
    s_max = float(s_vals.max())
    helix_len = s_max - s_min
    if helix_len <= 1e-6:
        raise ValueError("Helix length is too small or degenerate.")

    print(f"[INFO] Helix axis length (Å): {helix_len:.3f}")
    print(f"[INFO] Helix axis direction: ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")

    # ---- Apply linear twist along the helix axis (pre-mapping) ----
    twist_deg_total = float(twist)
    if abs(twist_deg_total) > 1e-12:
        twist_rad_total = np.deg2rad(twist_deg_total)
        t_norm = (s_vals - s_min) / helix_len  # ∈ [0,1]
        phi_arr = twist_rad_total * t_norm
        cos_phi = np.cos(phi_arr)
        sin_phi = np.sin(phi_arr)
        x = local[:, 0].copy()
        y = local[:, 1].copy()
        local[:, 0] = cos_phi * x - sin_phi * y
        local[:, 1] = sin_phi * x + cos_phi * y
        print(
            "[INFO] Applying linear twist of "
            f"{twist_deg_total:.3f} degrees from helix start to end."
        )
        if twist_deg_total > 0:
            print("       (positive = right-handed twist)")
        elif twist_deg_total < 0:
            print("       (negative = left-handed twist)")
    else:
        print("[INFO] twist = 0 (no additional twist along the helix axis).")

    # ---- Apply global phase rotation to the helix cross-section in its own frame ----
    phase_deg = float(helix_phase) % 360.0
    if abs(phase_deg) > 1e-12:
        phi = np.deg2rad(phase_deg)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        local_xy = local[:, :2].copy()
        local[:, 0] = cos_phi * local_xy[:, 0] - sin_phi * local_xy[:, 1]
        local[:, 1] = sin_phi * local_xy[:, 0] + cos_phi * local_xy[:, 1]
        print(f"[INFO] Applying helix_phase rotation of {phase_deg:.3f} degrees.")
    else:
        print("[INFO] helix_phase = 0 (no additional rotation of helix cross-sections).")

    # ---- Prepare curve points and basic arc-lengths ----
    pts = np.asarray(curve_points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Curve must have at least two points.")

    closed = (path_type == "closed")
    if closed:
        dist_end = np.linalg.norm(pts[-1] - pts[0])
        if dist_end > 1e-6:
            pts = np.vstack([pts, pts[0]])
        print(f"[INFO] Path type: closed (points: {pts.shape[0]})")
    else:
        print(f"[INFO] Path type: open (points: {pts.shape[0]})")

    arc_raw = compute_arc_lengths(pts)
    L_curve = arc_raw[-1]
    if L_curve <= 1e-6:
        raise ValueError("Curve length is too small or degenerate.")

    # ---- Interpret scale_mode ----
    mode: Optional[str] = None
    target_length: Optional[float] = None
    if isinstance(scale_mode, str):
        sm_str = scale_mode.strip()
        try:
            target_length = float(sm_str)
            if target_length <= 0:
                raise ValueError
            mode = "target_length"
        except ValueError:
            mode = sm_str.lower()
    else:
        target_length = float(scale_mode)
        if target_length <= 0:
            raise ValueError("Numeric scale_mode must be > 0.")
        mode = "target_length"

    if mode == "target_length" and (not closed) and target_length is not None and target_length < helix_len:
        print(
            f"[WARNING] Numeric scale_mode ({target_length:.3f} Å) is smaller than "
            f"the estimated helix axis length ({helix_len:.3f} Å) with path_type='open'. "
            "Many atoms will accumulate near the end of the curve. Consider using "
            "--path-type closed in this situation."
        )

    scaling_applied = False

    # ---- Apply scaling (if any) ----
    if mode == "curve_to_helix":
        scale = helix_len / L_curve
        anchor_pt = get_scale_anchor_point(scale_anchor, pts)
        pts = anchor_pt + (pts - anchor_pt) * scale
        arc = compute_arc_lengths(pts)
        scaling_applied = abs(scale - 1.0) > 1e-8
        print(f"[INFO] Original curve length (Å): {L_curve:.3f}")
        print(f"[INFO] Scaling curve to helix length (factor {scale:.3f})")
        print(f"[INFO] Scale anchor: {anchor_pt[0]:.3f} {anchor_pt[1]:.3f} {anchor_pt[2]:.3f}")
        print(f"[INFO] Scaled curve length (Å): {arc[-1]:.3f}")
    elif mode in ("none", "helix_to_curve"):
        arc = arc_raw.copy()
        print(f"[INFO] Curve length (Å): {arc[-1]:.3f} (no scaling)")
    elif mode == "target_length" and target_length is not None:
        scale = target_length / L_curve
        anchor_pt = get_scale_anchor_point(scale_anchor, pts)
        pts = anchor_pt + (pts - anchor_pt) * scale
        arc = compute_arc_lengths(pts)
        scaling_applied = abs(scale - 1.0) > 1e-8
        print(f"[INFO] Original curve length (Å): {L_curve:.3f}")
        print(f"[INFO] Target curve length (Å): {target_length:.3f}")
        print(f"[INFO] Scaling curve by factor: {scale:.3f}")
        print(f"[INFO] Scale anchor: {anchor_pt[0]:.3f} {anchor_pt[1]:.3f} {anchor_pt[2]:.3f}")
        print(f"[INFO] Scaled curve length (Å): {arc[-1]:.3f}")
    else:
        raise ValueError(
            f"Unknown scale_mode '{scale_mode}'. "
            "Use 'curve_to_helix', 'none', 'helix_to_curve', or a positive number."
        )

    # ---- Reparameterize closed curve based on path_start ----
    total = arc[-1]
    if closed:
        ps = float(path_start)
        if ps < 0.0 or ps > 1.0:
            print(f"[WARNING] path_start={ps:.3f} is outside [0,1]; clamping.")
            ps = max(0.0, min(1.0, ps))

        if ps > 1e-6 and ps < 1.0 - 1e-6:
            print(f"[INFO] Reparameterizing closed curve for path_start={ps:.3f}.")
            pts, arc = reparameterize_closed_curve(pts, arc, ps)
            total = arc[-1]
            print(f"[INFO] After reparameterization, curve length (Å): {total:.3f}")
        else:
            if ps <= 1e-6:
                print("[INFO] path_start ≈ 0.0 → no reparameterization.")
            else:
                print("[INFO] path_start ≈ 1.0 → treated as 0.0 (no reparameterization).")
    else:
        if abs(path_start) > 1e-8:
            print("[INFO] path_start specified but path_type='open'; "
                  "path_start is ignored for open curves.")

    # ---- Build rotation-minimizing frame along (possibly scaled & reparameterized) curve ----
    Nf, Bf, Tf = compute_parallel_transport_frames(pts)
    total = arc[-1]
    closed = (path_type == "closed")  # unchanged, but keep semantics

    print(f"[INFO] Total arc length used for mapping (Å): {total:.3f}")
    if closed:
        print("[INFO] Closed curve will be treated as periodic (helix may wrap or overlap).")
    else:
        print("[INFO] Path is open; no periodic wrapping.")

    # ---- Holonomy per loop (for closed curves) ----
    holonomy_angle = 0.0
    if closed:
        R0 = np.column_stack((Nf[0], Bf[0], Tf[0]))
        Rend = np.column_stack((Nf[-1], Bf[-1], Tf[-1]))
        S = R0.T @ Rend  # rotation from start frame to end frame in body coords
        holonomy_angle = float(np.arctan2(S[1, 0], S[0, 0]))
        print(f"[INFO] Approximate holonomy per loop (deg): {np.degrees(holonomy_angle):.3f}")
    else:
        print("[INFO] Path is open; holonomy angle not used.")

    # ---- Build atom groups and group-level axis coordinates ----
    group_atom_indices, group_s = build_atom_groups(atoms, s_vals)
    G = len(group_atom_indices)
    residue_types = {classify_residue_type(atom.res_name) for atom in atoms}
    if residue_types == {"nucleic_acid"}:
        grouping_note = "phosphate/sugar/base per residue"
    elif "nucleic_acid" in residue_types:
        grouping_note = "mixed: nucleic acids use phosphate/sugar/base; proteins/other residues use whole-residue groups"
    else:
        grouping_note = "whole-residue groups for protein/other residues"
    print(f"[INFO] Built {G} atom groups ({grouping_note}).")

    group_s_query = np.zeros(G, dtype=float)
    group_phi_lap = np.zeros(G, dtype=float)

    for gid in range(G):
        s_axis = group_s[gid]

        if mode == "target_length":
            # Numeric scale_mode: preserve helix spacing and allow multi-wrap.
            s_local = s_axis - s_min  # >= 0
            if closed and total > 0.0:
                lap = int(np.floor(s_local / total))
                s_mod = s_local % total
                s_q = s_mod
                group_phi_lap[gid] = lap * holonomy_angle
            else:
                s_q = s_local
                if s_q < 0.0:
                    s_q = 0.0
                elif s_q > total:
                    s_q = total
        else:
            # Other modes: helix is distributed once over [0,total].
            t_norm = (s_axis - s_min) / helix_len  # ∈ [0,1]
            s_q = t_norm * total
            if closed and total > 0.0:
                s_q = s_q % total
            else:
                if s_q < 0.0:
                    s_q = 0.0
                elif s_q > total:
                    s_q = total

        group_s_query[gid] = s_q

    # ---- Sample curve/frame at each group center ----
    group_pos = np.zeros((G, 3), dtype=float)
    group_N = np.zeros((G, 3), dtype=float)
    group_B = np.zeros((G, 3), dtype=float)
    group_T = np.zeros((G, 3), dtype=float)

    for gid in range(G):
        pos_g, N_g, B_g, T_g = sample_curve_with_frame(pts, arc, Nf, Bf, Tf, group_s_query[gid])
        group_pos[gid] = pos_g
        group_N[gid] = N_g
        group_B[gid] = B_g
        group_T[gid] = T_g

    # ---- Map atoms using group frames (rigid group mapping) ----
    new_coords = np.zeros_like(coords, dtype=float)

    # Precompute group index per atom
    atom_group_index = np.full(len(atoms), -1, dtype=int)
    for gid, idx_list in enumerate(group_atom_indices):
        for idx in idx_list:
            atom_group_index[idx] = gid

    for i, (s_axis, loc) in enumerate(zip(s_vals, local)):
        gid = atom_group_index[i]

        # Fallback: if grouping failed for some atom (should not happen),
        # revert to single-atom mapping with holonomy (no path_start offset).
        if gid < 0:
            u, v, w = loc
            phi_lap = 0.0

            if mode == "target_length":
                s_local = s_axis - s_min
                if closed and total > 0.0:
                    lap = int(np.floor(s_local / total))
                    s_mod = s_local % total
                    s_query = s_mod
                    phi_lap = lap * holonomy_angle
                else:
                    s_query = s_local
                    if s_query < 0.0:
                        s_query = 0.0
                    elif s_query > total:
                        s_query = total
            else:
                t_norm = (s_axis - s_min) / helix_len
                s_query = t_norm * total
                if closed and total > 0.0:
                    s_query = s_query % total
                else:
                    if s_query < 0.0:
                        s_query = 0.0
                    elif s_query > total:
                        s_query = total

            pos, N_s, B_s, T_s = sample_curve_with_frame(pts, arc, Nf, Bf, Tf, s_query)

            if closed and abs(phi_lap) > 1e-12:
                cphi = np.cos(phi_lap)
                sphi = np.sin(phi_lap)
                u_rot = cphi * u - sphi * v
                v_rot = sphi * u + cphi * v
            else:
                u_rot, v_rot = u, v

            new_coords[i] = pos + u_rot * N_s + v_rot * B_s + w * T_s
            continue

        # Normal group-based mapping
        s_group = group_s[gid]
        u = loc[0]
        v = loc[1]
        w_axis = s_axis - s_group

        phi_lap = group_phi_lap[gid]
        if closed and abs(phi_lap) > 1e-12:
            cphi = np.cos(phi_lap)
            sphi = np.sin(phi_lap)
            u_rot = cphi * u - sphi * v
            v_rot = sphi * u + cphi * v
        else:
            u_rot, v_rot = u, v

        pos_g = group_pos[gid]
        N_g = group_N[gid]
        B_g = group_B[gid]
        T_g = group_T[gid]

        new_coords[i] = pos_g + u_rot * N_g + v_rot * B_g + w_axis * T_g

    return new_coords, pts, scaling_applied


def update_pdb_line_coords(line: str, coord: np.ndarray) -> str:
    """
    Replace the X,Y,Z coordinate fields in a PDB ATOM/HETATM line with new
    values, preserving all other columns.
    """
    x, y, z = coord.tolist()
    if len(line) < 54:
        line = line.ljust(54)
    return f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"


def make_pdb_remark_lines(messages: List[str], remark_number: int = 900) -> List[str]:
    """Format provenance messages as wrapped PDB REMARK records."""
    lines: List[str] = []
    prefix = f"REMARK {remark_number:3d} "
    width = max(20, 80 - len(prefix))
    for message in messages:
        text = " ".join(str(message).split())
        if not text:
            lines.append(prefix.rstrip())
            continue
        wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
        if not wrapped:
            lines.append(prefix.rstrip())
        else:
            for part in wrapped:
                lines.append(f"{prefix}{part}")
    return lines


def build_generation_remarks(
    helix_pdb_path: Optional[str],
    curve_xyz_path: Optional[str],
    scale_mode: str,
    path_type: str,
    helix_phase: float,
    twist: float,
    scale_anchor: str,
    path_start: float,
    interp_mode: str,
    interp_n: int,
    interp_p: int,
) -> List[str]:
    """Build PDB REMARK lines describing how Curve It generated the file."""
    curve_source = curve_xyz_path if curve_xyz_path else "default planar ring curve"
    messages = [
        f"Generated by {APP_NAME} {APP_VERSION}.",
        f"Input PDB: {helix_pdb_path if helix_pdb_path else 'GUI-loaded PDB'}.",
        f"Curve source: {curve_source}.",
        f"Options: scale_mode={scale_mode}; path_type={path_type}; scale_anchor={scale_anchor}.",
        (
            f"Options: helix_phase={float(helix_phase):.6g} deg; "
            f"twist={float(twist):.6g} deg; path_start={float(path_start):.6g}."
        ),
        f"Options: interp_mode={interp_mode}; interp_n={int(interp_n)}; interp_p={int(interp_p)}.",
        "Method: principal-axis coordinates mapped onto the target curve with a rotation-minimizing frame and rigid atom-group transforms.",
    ]
    return make_pdb_remark_lines(messages)


def estimate_helix_length(atoms: List[AtomRecord]) -> float:
    """
    Estimate helix axis length from a list of AtomRecords.
    """
    coords = np.vstack([a.coord for a in atoms])
    _, _, _, s_vals, _ = compute_helix_local_coords(coords)
    return float(s_vals.max() - s_vals.min())


def compute_curve_metrics(points: np.ndarray,
                          closed: bool = True,
                          curvature_mode: str = "auto"
                          ) -> Tuple[float, Optional[float], Optional[float]]:
    """Compute basic geometric metrics of a curve.

    Returns:
        length, total_curvature, writhe

    Args:
        points: (N,3) polyline points.
        closed: whether to treat the curve as closed.
        curvature_mode:
            - 'auto' (default): detect polygon-like curves and use a robust
              polyline total curvature for cornered curves; otherwise use the
              periodic spline curvature.
            - 'polyline': always use polyline total curvature (sum of turning angles).
            - 'spline': always use periodic spline curvature (requires curvature tools);
              for sharp corners this can overestimate total curvature.

    Notes:
        - Length is always computed from the *given* polyline points.
        - Total curvature and writhe are only meaningful for closed curves.
    """
    length = float(compute_arc_lengths(points)[-1])

    if not closed:
        return length, None, None

    mode = (curvature_mode or "auto").strip().lower()
    total_curv: Optional[float] = None
    wr: Optional[float] = None

    polygonal = curve_looks_polygonal(points)

    # Decide which curvature estimator to use.
    force_polyline = mode in ("polyline", "pline", "discrete")
    force_spline = mode in ("spline", "smooth", "periodic")

    if force_spline and (not HAVE_CURVATURE_WRITHE):
        # Spline tools unavailable: fall back to polyline.
        force_spline = False
        force_polyline = True

    use_polyline = force_polyline or ((not HAVE_CURVATURE_WRITHE) or (polygonal and not force_spline))
    use_spline = force_spline or (HAVE_CURVATURE_WRITHE and (not polygonal) and (not force_polyline))

    # --- Total curvature ---
    if use_polyline:
        try:
            total_curv = compute_discrete_total_curvature(points, closed=True)
        except Exception as e:
            print(f"[WARNING] compute_curve_metrics failed for polyline curvature: {e}")
            total_curv = None
    elif use_spline:
        try:
            # If the user explicitly requested spline mode, smooth even for polygon-like curves.
            pts_smooth = smooth_closed_curve_force(points) if force_spline else smooth_closed_curve(points)
            total_curv = fit_spline_and_calculate_curvature(pts_smooth)
        except Exception as e:
            print(f"[WARNING] compute_curve_metrics failed for spline curvature: {e}")
            total_curv = None

    # --- Writhe ---
    if HAVE_CURVATURE_WRITHE:
        try:
            if force_spline:
                pts_for_writhe = smooth_closed_curve_force(points)
            elif force_polyline or polygonal:
                pts_for_writhe = points
            else:
                pts_for_writhe = smooth_closed_curve(points)
            wr = calculate_writhe(pts_for_writhe, n_samples=400)
        except Exception as e:
            print(f"[WARNING] compute_curve_metrics failed for writhe: {e}")
            wr = None

    return length, total_curv, wr



def write_output_pdb(pdb_text: str,
                     new_coords: np.ndarray,
                     output_pdb: str,
                     remark_lines: Optional[List[str]] = None) -> None:
    """
    Write an output PDB by replacing coordinates in ATOM/HETATM records
    (in order) with new_coords.
    """
    print(f"[INFO] Writing output PDB: {output_pdb}")
    out_lines: List[str] = []
    if remark_lines:
        out_lines.extend(remark_lines)
    coord_idx = 0
    for line in pdb_text.splitlines():
        if (line.startswith("ATOM") or line.startswith("HETATM")) and coord_idx < len(new_coords):
            updated = update_pdb_line_coords(line.rstrip("\n"), new_coords[coord_idx])
            coord_idx += 1
            out_lines.append(updated)
        else:
            out_lines.append(line.rstrip("\n"))

    with open(output_pdb, "w") as f:
        f.write("\n".join(out_lines) + "\n")


def write_rescaled_curve_xyz(curve_xyz_path: Optional[str],
                             scaled_curve_pts: np.ndarray,
                             scaling_applied: bool) -> None:
    """
    If the curve was scaled and an original XYZ path is known, write a
    '_rescaled' XYZ alongside it.
    """
    if curve_xyz_path is None or not scaling_applied:
        return

    curve_dir = os.path.dirname(curve_xyz_path)
    base_curve = os.path.basename(curve_xyz_path)
    stem_c, ext_c = os.path.splitext(base_curve)
    if not ext_c:
        ext_c = ".xyz"
    rescaled_name = f"{stem_c}_rescaled{ext_c}"
    rescaled_path = os.path.join(curve_dir, rescaled_name)
    print(f"[INFO] Writing rescaled curve XYZ: {rescaled_path}")
    with open(rescaled_path, "w") as f:
        for p in scaled_curve_pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def launch_gui() -> None:
    """
    Launch a simple Tkinter GUI for interactive use of Curve It.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as e:
        raise SystemExit(f"Tkinter GUI not available: {e}")

    # Local state
    atoms: Optional[List[AtomRecord]] = None
    pdb_text: Optional[str] = None
    helix_pdb_path: Optional[str] = None

    curve_points_raw: Optional[np.ndarray] = None
    curve_points: Optional[np.ndarray] = None  # interpolated curve used for embedding/viewing
    curve_xyz_path: Optional[str] = None

    root = tk.Tk()
    root.title(f"{APP_NAME} {APP_VERSION} - {APP_TITLE}")
    icon_path = resource_path(os.path.join("assets", "icon.png"))
    if os.path.isfile(icon_path):
        try:
            icon_image = tk.PhotoImage(file=icon_path)
            root.iconphoto(True, icon_image)
            root._curve_it_icon_image = icon_image
        except Exception:
            pass

    # --- Variables ---
    helix_path_var = tk.StringVar()
    curve_path_var = tk.StringVar()
    output_path_var = tk.StringVar()

    helix_len_var = tk.StringVar(value="N/A")
    curve_len_var = tk.StringVar(value="N/A")
    total_curv_var = tk.StringVar(value="N/A")
    writhe_var = tk.StringVar(value="N/A")
    curve_n_raw_var = tk.StringVar(value="N/A")
    curve_n_used_var = tk.StringVar(value="N/A")
    interp_out_path_var = tk.StringVar(value="N/A")

    scale_mode_var = tk.StringVar(value="curve_to_helix")
    numeric_length_var = tk.StringVar(value="340.0")  # example default

    scale_anchor_mode_var = tk.StringVar(value="centroid")
    scale_anchor_custom_var = tk.StringVar(value="0,0,0")

    path_type_var = tk.StringVar(value="closed")  # default closed in GUI
    helix_phase_var = tk.StringVar(value="0.0")
    twist_var = tk.StringVar(value="0.0")
    path_start_var = tk.DoubleVar(value=0.0)

    # Curve interpolation (see interpolate_xyz.py)
    interp_mode_var = tk.StringVar(value="none")  # none | n | p
    interp_n_var = tk.StringVar(value="200")
    interp_p_var = tk.StringVar(value="0")


    # Total curvature estimation mode for GUI reporting
    #   auto    : detect polygon-like curves and choose polyline vs spline automatically
    #   polyline: always use polyline turning-angle sum (robust for sharp corners)
    #   spline  : always use smoothed periodic spline curvature (may overestimate at corners)
    curvature_mode_var = tk.StringVar(value="auto")
    curvature_used_var = tk.StringVar(value="N/A")

    section_font = ("TkDefaultFont", 10, "bold")
    help_button_kwargs = {
        "text": "?",
        "width": 2,
        "bg": "#cfefff",
        "activebackground": "#aee6ff",
        "relief": tk.RAISED,
        "borderwidth": 1,
    }

    help_topics = {
        "helix_pdb": (
            "Helix PDB",
            "Choose the input PDB to bend. The structure should have a meaningful roughly straight principal axis.\n\n"
            "Examples:\n"
            "- a straight DNA/RNA helix\n"
            "- an alpha helix or coiled-coil protein\n"
            "- an elongated filament-like PDB\n\n"
            "Compact globular proteins may not fit well because one principal axis is not a good shape description."
        ),
        "curve_xyz": (
            "Curve XYZ/txt",
            "Choose an optional curve file containing 3D points. Plain x y z rows and standard XYZ-like files are accepted.\n\n"
            "Example plain file:\n"
            "0 0 0\n"
            "5 0 2\n"
            "10 4 6\n\n"
            "If no curve file is selected, Curve It uses a default planar ring."
        ),
        "curve_metrics": (
            "Curve Metrics",
            "Curve length is the polyline arc length. Total curvature and writhe are reported for closed curves only.\n\n"
            "Use these as geometry checks before running the fit."
        ),
        "path_type": (
            "Path Type",
            "closed treats the curve as a periodic loop and can wrap the fitted PDB around the curve.\n\n"
            "open treats the curve as a path with distinct start and end points.\n\n"
            "Example: use closed for rings; use open for arcs, spirals, or drawn centerlines."
        ),
        "interp_mode": (
            "Interpolation Mode",
            "none uses the curve as-is.\n\n"
            "n resamples the curve to exactly n points, evenly spaced by arc length.\n\n"
            "p inserts p evenly spaced points between every adjacent pair of curve points."
        ),
        "interp_n": (
            "Interpolation n",
            "Used only when interpolation mode is n.\n\n"
            "Example: n = 400 creates a 400-point curve with approximately even arc-length spacing."
        ),
        "interp_p": (
            "Interpolation p",
            "Used only when interpolation mode is p.\n\n"
            "Example: p = 5 inserts five new points between every adjacent point pair."
        ),
        "curvature_mode": (
            "Curvature Mode",
            "auto detects polygon-like curves and chooses a robust estimator.\n\n"
            "polyline uses the sum of turning angles, good for sharp-corner curves.\n\n"
            "spline uses a smoothed periodic spline and may overestimate curvature near sharp corners."
        ),
        "interp_file": (
            "Interpolated Curve File",
            "When interpolation is enabled and a curve file is loaded, Curve It writes a sibling file named <curve>_interpolated.<ext>."
        ),
        "scale_mode": (
            "Scale Mode",
            "curve_to_helix scales the curve length to match the input PDB axis length.\n\n"
            "none or helix_to_curve leaves the curve length unchanged.\n\n"
            "numeric scales the curve to the numeric target length while preserving PDB axial spacing."
        ),
        "numeric_length": (
            "Numeric Length",
            "Used only when scale mode is numeric. Enter the target curve length in Angstrom.\n\n"
            "Example: 340.0"
        ),
        "scale_anchor": (
            "Scale Anchor",
            "centroid scales the curve around its own center.\n\n"
            "origin scales around 0,0,0.\n\n"
            "custom uses the x,y,z point in the custom anchor field."
        ),
        "custom_anchor": (
            "Custom Anchor",
            "Used only when scale anchor is custom. Enter three comma-separated coordinates in Angstrom.\n\n"
            "Examples:\n"
            "0,0,0\n"
            "10,0,0"
        ),
        "helix_phase": (
            "Helix Phase",
            "Rotates all cross-sections around the input structure axis before fitting.\n\n"
            "Example: 90 rotates the fitted structure by 90 degrees around its local axis."
        ),
        "twist": (
            "Twist",
            "Adds a linear axial twist from the start to the end of the input PDB before fitting.\n\n"
            "Example: 360 applies one full additional turn."
        ),
        "path_start": (
            "Path Start",
            "For closed curves only, chooses where the PDB starts on the loop.\n\n"
            "0.0 uses the first curve point. 0.5 starts halfway around the loop. Open curves ignore this setting."
        ),
        "output_pdb": (
            "Output PDB",
            "Choose where to write the curved PDB. If left blank after loading a PDB, Curve It suggests <input>_curved.pdb.\n\n"
            "Generated PDB files include REMARK 900 lines describing the input files and options."
        ),
    }

    def show_help(topic_key: str) -> None:
        title, body = help_topics[topic_key]
        messagebox.showinfo(title, body)

    def help_button(parent: Any, topic_key: str) -> Any:
        return tk.Button(parent, command=lambda: show_help(topic_key), **help_button_kwargs)

    def compute_interpolated_curve(points_in: np.ndarray, show_error: bool = True) -> Optional[np.ndarray]:
        """Return curve points after applying the current interpolation settings.

        If interpolation is disabled (mode="none"), returns points_in unchanged.
        On failure, returns None (and optionally shows a GUI error message).
        """
        mode = (interp_mode_var.get() or "none").strip().lower()
        closed_interp = (path_type_var.get() == "closed")

        # Parse numeric parameters only when required by the chosen mode.
        if mode in ("none", "off", "false", "0", ""):
            return np.asarray(points_in, dtype=float)

        n_val = 0
        p_val = 0

        if mode in ("n", "n_points", "npoints", "num", "num_points"):
            try:
                n_val = int(interp_n_var.get().strip())
            except ValueError:
                if show_error:
                    messagebox.showerror("Invalid interpolation n",
                                         "Interpolation n must be an integer.")
                return None

        elif mode in ("p", "p_between", "pbetween", "insert"):
            try:
                p_val = int(interp_p_var.get().strip())
            except ValueError:
                if show_error:
                    messagebox.showerror("Invalid interpolation p",
                                         "Interpolation p must be an integer.")
                return None

        else:
            if show_error:
                messagebox.showerror("Invalid interpolation mode",
                                     f"Unknown interpolation mode '{mode}'. Use none, n, or p.")
            return None

        try:
            pts_out = apply_curve_interpolation(
                points_in,
                interp_mode=mode,
                interp_n=n_val,
                interp_p=p_val,
                closed=closed_interp,
                verbose=False,
            )
            return pts_out
        except Exception as e:
            if show_error:
                messagebox.showerror("Interpolation error", f"Failed to interpolate curve:\n{e}")
            return None

    def refresh_curve_metrics_from_points(pts_used: np.ndarray) -> None:
        """Update the displayed curve metrics for the given curve points."""
        closed = (path_type_var.get() == "closed")

        mode = (curvature_mode_var.get() or "auto").strip().lower()
        polygonal = curve_looks_polygonal(pts_used) if closed else False

        # Decide what method is used (for display).
        if not closed:
            curvature_used_var.set("N/A (open curve)")
        elif mode in ("polyline", "pline", "discrete"):
            curvature_used_var.set("polyline (manual)")
        elif mode in ("spline", "smooth", "periodic"):
            if HAVE_CURVATURE_WRITHE:
                curvature_used_var.set("spline (manual)")
            else:
                curvature_used_var.set("polyline (fallback; spline unavailable)")
        else:
            # auto (default)
            if (not HAVE_CURVATURE_WRITHE) or polygonal:
                curvature_used_var.set("polyline (auto)")
            else:
                curvature_used_var.set("spline (auto)")

        length, total_curv, wr = compute_curve_metrics(
            pts_used,
            closed=closed,
            curvature_mode=mode,
        )
        curve_len_var.set(f"{length:.3f} Å")

        if not closed:
            total_curv_var.set("N/A (open curve)")
            writhe_var.set("N/A (open curve)")
            return

        if total_curv is not None:
            curv_pi = total_curv / np.pi
            curv_deg = np.degrees(total_curv)
            total_curv_var.set(f"{total_curv:.6f} ({curv_pi:.6f} * pi; {curv_deg:.3f}°)")
        else:
            if HAVE_CURVATURE_WRITHE:
                total_curv_var.set("N/A")
            else:
                total_curv_var.set("N/A (curvature tools not available)")

        if wr is not None:
            frac = wr - np.floor(wr)
            wr_deg = frac * 360.0
            writhe_var.set(f"{wr:.6f} ({wr_deg:.3f}° residual)")
        else:
            writhe_var.set("N/A")

    def _default_interpolated_curve_path(input_path: str) -> str:
        """Return '<stem>_interpolated<ext>' in the same directory as input."""
        d = os.path.dirname(input_path)
        base = os.path.basename(input_path)
        stem, ext = os.path.splitext(base)
        if not ext:
            return os.path.join(d, stem + "_interpolated")
        return os.path.join(d, stem + "_interpolated" + ext)

    def refresh_curve_after_interpolation(show_error: bool = False) -> None:
        """Recompute curve_points from curve_points_raw using current interpolation settings.

        Also updates point-count fields and (when interpolation is enabled) writes
        '<curve>_interpolated.ext' next to the input curve file.
        """
        nonlocal curve_points
        if curve_points_raw is None:
            curve_points = None
            curve_n_used_var.set("N/A")
            interp_out_path_var.set("N/A")
            return

        pts_used = compute_interpolated_curve(curve_points_raw, show_error=show_error)
        if pts_used is None:
            return

        curve_points = pts_used
        curve_n_used_var.set(str(int(pts_used.shape[0])))

        # Write interpolated curve file (only when a user-supplied curve file exists
        # and interpolation mode is not 'none').
        mode = (interp_mode_var.get() or "none").strip().lower()
        if curve_xyz_path is None:
            interp_out_path_var.set("N/A (default curve)")
        elif mode in ("none", "off", "false", "0", ""):
            interp_out_path_var.set("N/A (mode none)")
        else:
            out_path = _default_interpolated_curve_path(curve_xyz_path)
            try:
                if HAVE_INTERPOLATE_XYZ and hasattr(interpolate_xyz, "write_xyz_curve"):
                    interpolate_xyz.write_xyz_curve(out_path, pts_used)
                else:
                    # Fallback writer: plain whitespace-separated x y z
                    with open(out_path, "w") as f:
                        for p in np.asarray(pts_used, dtype=float):
                            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                interp_out_path_var.set(out_path)
            except Exception as e:
                interp_out_path_var.set(f"Write failed: {e}")
                if show_error:
                    messagebox.showerror(
                        "Write interpolated curve failed",
                        f"Failed to write interpolated curve:\n{e}",
                    )

        # Refresh displayed metrics (length/curvature/writhe) based on the
        # interpolated curve (the curve that will actually be used).
        refresh_curve_metrics_from_points(pts_used)

    # Keep interpolation/metrics in sync when path type changes (open vs closed).
    def on_path_type_changed_for_interp(*_args: Any) -> None:
        refresh_curve_after_interpolation(show_error=False)

    path_type_var.trace_add("write", on_path_type_changed_for_interp)

    title_label = tk.Label(
        root,
        text=APP_TITLE,
        font=("TkDefaultFont", 14, "bold"),
        anchor="w",
    )
    title_label.grid(row=0, column=0, sticky="we", padx=8, pady=(8, 2))

    # --- File selection frame ---
    file_frame = tk.LabelFrame(root, text="Input files", font=section_font)
    file_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)

    # Helix PDB
    tk.Label(file_frame, text="Helix PDB:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
    helix_entry = tk.Entry(file_frame, textvariable=helix_path_var, width=44)
    helix_entry.grid(row=0, column=1, columnspan=5, sticky="we", padx=4, pady=2)

    def browse_helix():
        nonlocal atoms, pdb_text, helix_pdb_path
        path = filedialog.askopenfilename(
            title="Select helix PDB file",
            filetypes=[("PDB files", "*.pdb *.PDB *.ent *.ENT"), ("All files", "*.*")]
        )
        if not path:
            return
        helix_pdb_path = path
        helix_path_var.set(path)
        try:
            with open(path, "r") as f:
                pdb_text_local = f.read()
            atoms_local = parse_pdb_atoms(pdb_text_local)
            if not atoms_local:
                raise ValueError("No ATOM/HETATM records found.")
            atoms = atoms_local
            pdb_text = pdb_text_local
            helix_len = estimate_helix_length(atoms)
            helix_len_var.set(f"{helix_len:.3f} Å")
            # Set a default output name if empty
            if not output_path_var.get().strip():
                base = os.path.basename(path)
                stem, ext = os.path.splitext(base)
                if not ext:
                    ext = ".pdb"
                out_name = f"{stem}_curved{ext}"
                out_path = os.path.join(os.path.dirname(path), out_name)
                output_path_var.set(out_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error reading PDB", f"Failed to read PDB:\n{e}")
            atoms = None
            pdb_text = None
            helix_len_var.set("N/A")

    tk.Button(file_frame, text="Browse...", command=browse_helix).grid(
        row=0, column=6, sticky="w", padx=4, pady=2
    )
    help_button(file_frame, "helix_pdb").grid(row=0, column=7, sticky="w", padx=(0, 4), pady=2)

    # Curve XYZ
    tk.Label(file_frame, text="Curve XYZ/txt:").grid(row=1, column=0, sticky="e", padx=4, pady=2)
    curve_entry = tk.Entry(file_frame, textvariable=curve_path_var, width=44)
    curve_entry.grid(row=1, column=1, columnspan=5, sticky="we", padx=4, pady=2)

    def load_curve_from_path(path: str) -> Optional[np.ndarray]:
        nonlocal curve_points_raw, curve_points, curve_xyz_path
        try:
            with open(path, "r") as f:
                xyz_text = f.read()
            pts_raw = read_xyz_curve_from_text(xyz_text)
            curve_n_raw_var.set(str(int(pts_raw.shape[0])))
            curve_points_raw = pts_raw
            curve_points = None
            curve_xyz_path = path

            # Apply interpolation according to the current settings and refresh
            # the displayed metrics.
            refresh_curve_after_interpolation(show_error=True)

            return curve_points
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error reading curve", f"Failed to read curve:\n{e}")
            curve_points_raw = None
            curve_points = None
            curve_xyz_path = None
            curve_n_raw_var.set("N/A")
            curve_n_used_var.set("N/A")
            interp_out_path_var.set("N/A")
            curve_len_var.set("N/A")
            total_curv_var.set("N/A")
            writhe_var.set("N/A")
            return None

    def browse_curve():
        path = filedialog.askopenfilename(
            title="Select curve XYZ/text file",
            filetypes=[("XYZ / text files", "*.xyz *.XYZ *.txt *.TXT"), ("All files", "*.*")]
        )
        if not path:
            return
        curve_path_var.set(path)
        load_curve_from_path(path)

    tk.Button(file_frame, text="Browse...", command=browse_curve).grid(
        row=1, column=6, sticky="w", padx=4, pady=2
    )
    help_button(file_frame, "curve_xyz").grid(row=1, column=7, sticky="w", padx=(0, 4), pady=2)

    tk.Label(file_frame, text="Helix length:").grid(row=2, column=0, sticky="e", padx=4, pady=2)
    helix_len_entry = ttk.Entry(file_frame, textvariable=helix_len_var, width=14, state="readonly")
    helix_len_entry.grid(row=2, column=1, sticky="w", padx=4, pady=2)

    tk.Label(file_frame, text="Curve points:").grid(row=2, column=2, sticky="e", padx=4, pady=2)
    curve_n_entry = ttk.Entry(file_frame, textvariable=curve_n_raw_var, width=10, state="readonly")
    curve_n_entry.grid(row=2, column=3, sticky="w", padx=4, pady=2)

    tk.Label(file_frame, text="Curve length:").grid(row=2, column=4, sticky="e", padx=4, pady=2)
    curve_len_entry = ttk.Entry(file_frame, textvariable=curve_len_var, width=14, state="readonly")
    curve_len_entry.grid(row=2, column=5, sticky="w", padx=4, pady=2)

    def view_curve():
        if curve_points_raw is None:
            messagebox.showwarning("No curve", "Load a curve file first.")
            return

        # Ensure we are viewing the curve *after interpolation* using the
        # current interpolation settings.
        refresh_curve_after_interpolation(show_error=True)
        if curve_points is None:
            return

        closed = (path_type_var.get() == "closed")
        try:
            try:
                # Use the plotting helper from view_xyzV2.py when available.
                from curve_it_lib import view_xyzV2 as view_xyz
                view_xyz.plot_curve(curve_points, closed=closed)
                return
            except Exception:
                pass

            try:
                # Simple fallback 3D plot.
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                pts = curve_points
                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="0.6", linewidth=1.0)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title("Curve")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                raise RuntimeError(f"Matplotlib curve viewer is not available: {e}") from e
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Plot error", f"Failed to plot curve:\n{e}")

    tk.Button(file_frame, text="View curve", command=view_curve).grid(
        row=2, column=6, sticky="w", padx=4, pady=2
    )
    help_button(file_frame, "curve_metrics").grid(row=2, column=7, sticky="w", padx=(0, 4), pady=2)

    tk.Label(file_frame, text="Total curvature:").grid(row=3, column=0, sticky="e", padx=4, pady=2)
    total_curv_entry = ttk.Entry(file_frame, textvariable=total_curv_var, width=32, state="readonly")
    total_curv_entry.grid(row=3, column=1, columnspan=3, sticky="we", padx=4, pady=2)

    tk.Label(file_frame, text="Writhe:").grid(row=3, column=4, sticky="e", padx=4, pady=2)
    writhe_entry = ttk.Entry(file_frame, textvariable=writhe_var, width=20, state="readonly")
    writhe_entry.grid(row=3, column=5, columnspan=2, sticky="we", padx=4, pady=2)

    for col in range(8):
        file_frame.grid_columnconfigure(col, weight=0)
    file_frame.grid_columnconfigure(1, weight=1)
    file_frame.grid_columnconfigure(5, weight=1)

    # --- Curve parameters frame ---
    curve_param_frame = tk.LabelFrame(root, text="Curve parameters", font=section_font)
    curve_param_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)

    tk.Label(curve_param_frame, text="Interpolation mode:").grid(
        row=0, column=0, sticky="e", padx=4, pady=2
    )
    interp_mode_menu = tk.OptionMenu(curve_param_frame, interp_mode_var, "none", "n", "p")
    interp_mode_menu.grid(row=0, column=1, sticky="w", padx=4, pady=2)
    help_button(curve_param_frame, "interp_mode").grid(row=0, column=2, sticky="w", padx=(0, 8), pady=2)

    tk.Label(curve_param_frame, text="n points (mode 'n'):").grid(
        row=0, column=3, sticky="e", padx=4, pady=2
    )
    interp_n_entry = tk.Entry(curve_param_frame, textvariable=interp_n_var, width=10)
    interp_n_entry.grid(row=0, column=4, sticky="w", padx=4, pady=2)
    help_button(curve_param_frame, "interp_n").grid(row=0, column=5, sticky="w", padx=(0, 8), pady=2)

    tk.Label(curve_param_frame, text="p inserted/segment:").grid(
        row=0, column=6, sticky="e", padx=4, pady=2
    )
    interp_p_entry = tk.Entry(curve_param_frame, textvariable=interp_p_var, width=10)
    interp_p_entry.grid(row=0, column=7, sticky="w", padx=4, pady=2)
    help_button(curve_param_frame, "interp_p").grid(row=0, column=8, sticky="w", padx=(0, 4), pady=2)

    tk.Label(curve_param_frame, text="Points after interpolation:").grid(
        row=1, column=0, sticky="e", padx=4, pady=2
    )
    curve_n_used_entry = ttk.Entry(curve_param_frame, textvariable=curve_n_used_var, width=10, state="readonly")
    curve_n_used_entry.grid(row=1, column=1, sticky="w", padx=4, pady=2)

    tk.Label(curve_param_frame, text="Curvature mode:").grid(
        row=1, column=3, sticky="e", padx=4, pady=2
    )
    curvature_mode_menu = tk.OptionMenu(curve_param_frame, curvature_mode_var, "auto", "polyline", "spline")
    curvature_mode_menu.grid(row=1, column=4, sticky="w", padx=4, pady=2)
    help_button(curve_param_frame, "curvature_mode").grid(row=1, column=5, sticky="w", padx=(0, 8), pady=2)

    tk.Label(curve_param_frame, text="Curvature used:").grid(
        row=1, column=6, sticky="e", padx=4, pady=2
    )
    curvature_used_entry = ttk.Entry(
        curve_param_frame, textvariable=curvature_used_var, width=26, state="readonly"
    )
    curvature_used_entry.grid(row=1, column=7, columnspan=2, sticky="we", padx=4, pady=2)

    tk.Label(curve_param_frame, text="Interpolated curve file:").grid(
        row=2, column=0, sticky="e", padx=4, pady=2
    )
    interp_out_entry = ttk.Entry(curve_param_frame, textvariable=interp_out_path_var, width=48, state="readonly")
    interp_out_entry.grid(row=2, column=1, columnspan=7, sticky="we", padx=4, pady=2)
    help_button(curve_param_frame, "interp_file").grid(row=2, column=8, sticky="w", padx=(0, 4), pady=2)

    def update_interp_widgets(*_args: Any):
        mode = (interp_mode_var.get() or "none").strip().lower()
        if mode == "n":
            interp_n_entry.config(state="normal")
            interp_p_entry.config(state="disabled")
        elif mode == "p":
            interp_n_entry.config(state="disabled")
            interp_p_entry.config(state="normal")
        else:
            interp_n_entry.config(state="disabled")
            interp_p_entry.config(state="disabled")

    interp_mode_var.trace_add("write", update_interp_widgets)
    update_interp_widgets()

    # Recompute interpolated curve + displayed metrics when parameters change.
    def on_interp_params_committed(event=None):
        refresh_curve_after_interpolation(show_error=False)

    def on_interp_mode_changed(*_args: Any):
        refresh_curve_after_interpolation(show_error=False)

    interp_n_entry.bind("<Return>", on_interp_params_committed)
    interp_n_entry.bind("<FocusOut>", on_interp_params_committed)
    interp_p_entry.bind("<Return>", on_interp_params_committed)
    interp_p_entry.bind("<FocusOut>", on_interp_params_committed)
    interp_mode_var.trace_add("write", on_interp_mode_changed)


    def on_curvature_mode_changed(*_args: Any) -> None:
        # Curvature mode affects only reporting; curve geometry is unchanged.
        if curve_points is not None:
            refresh_curve_metrics_from_points(curve_points)
        elif curve_points_raw is not None:
            pts_used = compute_interpolated_curve(curve_points_raw, show_error=False)
            if pts_used is not None:
                refresh_curve_metrics_from_points(pts_used)

    curvature_mode_var.trace_add("write", on_curvature_mode_changed)

    for col in range(9):
        curve_param_frame.grid_columnconfigure(col, weight=0)
    curve_param_frame.grid_columnconfigure(7, weight=1)

    # --- Parameters frame ---
    param_frame = tk.LabelFrame(root, text="Mapping parameters", font=section_font)
    param_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=6)

    # Scale mode and numeric target length
    tk.Label(param_frame, text="Scale mode:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
    scale_mode_menu = tk.OptionMenu(
        param_frame,
        scale_mode_var,
        "curve_to_helix",
        "none",
        "helix_to_curve",
        "numeric",
    )
    scale_mode_menu.grid(row=0, column=1, sticky="w", padx=4, pady=2)
    help_button(param_frame, "scale_mode").grid(row=0, column=2, sticky="w", padx=(0, 8), pady=2)

    tk.Label(param_frame, text="Numeric length (Å):").grid(row=0, column=3, sticky="e", padx=4, pady=2)
    numeric_length_entry = tk.Entry(param_frame, textvariable=numeric_length_var, width=10)
    numeric_length_entry.grid(row=0, column=4, sticky="w", padx=4, pady=2)
    help_button(param_frame, "numeric_length").grid(row=0, column=5, sticky="w", padx=(0, 8), pady=2)

    # Enable the numeric length entry only when scale_mode = 'numeric'
    def update_scale_mode_widgets(*_args: Any):
        if scale_mode_var.get() == "numeric":
            numeric_length_entry.config(state="normal")
        else:
            numeric_length_entry.config(state="disabled")

    scale_mode_var.trace_add("write", update_scale_mode_widgets)
    update_scale_mode_widgets()

    # Scale anchor
    tk.Label(param_frame, text="Scale anchor:").grid(row=1, column=0, sticky="e", padx=4, pady=2)
    scale_anchor_menu = tk.OptionMenu(
        param_frame,
        scale_anchor_mode_var,
        "centroid",
        "origin",
        "custom",
    )
    scale_anchor_menu.grid(row=1, column=1, sticky="w", padx=4, pady=2)
    help_button(param_frame, "scale_anchor").grid(row=1, column=2, sticky="w", padx=(0, 8), pady=2)

    tk.Label(param_frame, text="Custom anchor x,y,z:").grid(row=1, column=3, sticky="e", padx=4, pady=2)
    scale_anchor_entry = tk.Entry(param_frame, textvariable=scale_anchor_custom_var, width=14)
    scale_anchor_entry.grid(row=1, column=4, sticky="w", padx=4, pady=2)
    help_button(param_frame, "custom_anchor").grid(row=1, column=5, sticky="w", padx=(0, 8), pady=2)

    # Enable custom anchor entry only when scale_anchor = 'custom'
    def update_scale_anchor_widgets(*_args: Any):
        if scale_anchor_mode_var.get() == "custom":
            scale_anchor_entry.config(state="normal")
        else:
            scale_anchor_entry.config(state="disabled")

    scale_anchor_mode_var.trace_add("write", update_scale_anchor_widgets)
    update_scale_anchor_widgets()

    # Path type
    tk.Label(param_frame, text="Path type:").grid(row=2, column=6, sticky="e", padx=4, pady=2)
    path_type_menu = tk.OptionMenu(param_frame, path_type_var, "closed", "open")
    path_type_menu.grid(row=2, column=7, sticky="w", padx=4, pady=2)
    help_button(param_frame, "path_type").grid(row=2, column=8, sticky="w", padx=(0, 4), pady=2)

    # Helix phase & Twist
    tk.Label(param_frame, text="Helix phase (deg):").grid(row=2, column=0, sticky="e", padx=4, pady=2)
    helix_phase_entry = tk.Entry(param_frame, textvariable=helix_phase_var, width=10)
    helix_phase_entry.grid(row=2, column=1, sticky="w", padx=4, pady=2)
    help_button(param_frame, "helix_phase").grid(row=2, column=2, sticky="w", padx=(0, 8), pady=2)

    tk.Label(param_frame, text="Twist (deg):").grid(row=2, column=3, sticky="e", padx=4, pady=2)
    twist_entry = tk.Entry(param_frame, textvariable=twist_var, width=10)
    twist_entry.grid(row=2, column=4, sticky="w", padx=4, pady=2)
    help_button(param_frame, "twist").grid(row=2, column=5, sticky="w", padx=(0, 8), pady=2)

    # Path start (closed curves): slider + entry, coupled
    tk.Label(param_frame, text="Path start (0–1, closed only):").grid(
        row=3, column=0, sticky="e", padx=4, pady=2
    )

    path_start_scale = tk.Scale(
        param_frame,
        from_=0.0,
        to=1.0,
        orient=tk.HORIZONTAL,
        resolution=0.01,
        variable=path_start_var,
        length=180,
    )
    path_start_scale.grid(row=3, column=1, columnspan=5, sticky="we", padx=4, pady=2)

    # Entry field coupled to the slider
    path_start_entry_var = tk.StringVar(value="0.00")
    path_start_entry = tk.Entry(param_frame, textvariable=path_start_entry_var, width=6)
    path_start_entry.grid(row=3, column=6, sticky="w", padx=4, pady=2)
    help_button(param_frame, "path_start").grid(row=3, column=7, sticky="w", padx=(0, 8), pady=2)

    # When slider moves, update the entry text
    def on_path_start_var_changed(*_args: Any) -> None:
        val = path_start_var.get()
        path_start_entry_var.set(f"{val:.2f}")

    path_start_var.trace_add("write", on_path_start_var_changed)
    on_path_start_var_changed()

    # When entry is edited (Enter or focus-out), update the slider (clamped to [0,1])
    def on_path_start_entry_changed(event=None):
        txt = path_start_entry_var.get().strip()
        try:
            val = float(txt)
        except ValueError:
            return
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        path_start_var.set(val)
        path_start_entry_var.set(f"{val:.2f}")

    path_start_entry.bind("<Return>", on_path_start_entry_changed)
    path_start_entry.bind("<FocusOut>", on_path_start_entry_changed)

    # Enable path_start controls only when path_type = 'closed'
    def update_path_type_widgets(*_args: Any):
        closed = (path_type_var.get() == "closed")
        state = "normal" if closed else "disabled"
        path_start_scale.config(state=state)
        path_start_entry.config(state=state)

    path_type_var.trace_add("write", update_path_type_widgets)
    update_path_type_widgets()

    for col in range(9):
        param_frame.grid_columnconfigure(col, weight=0)
    param_frame.grid_columnconfigure(1, weight=1)
    param_frame.grid_columnconfigure(4, weight=1)

    # --- Output and run frame ---
    out_frame = tk.LabelFrame(root, text="Output and run", font=section_font)
    out_frame.grid(row=4, column=0, sticky="nsew", padx=8, pady=6)

    tk.Label(out_frame, text="Output PDB:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
    output_entry = tk.Entry(out_frame, textvariable=output_path_var, width=44)
    output_entry.grid(row=0, column=1, columnspan=2, sticky="we", padx=4, pady=2)

    def choose_output_path():
        initialdir = os.path.dirname(helix_path_var.get()) if helix_path_var.get() else "."
        path = filedialog.asksaveasfilename(
            title="Select output PDB file",
            defaultextension=".pdb",
            filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")],
            initialdir=initialdir,
            initialfile=os.path.basename(output_path_var.get() or ""),
        )
        if path:
            output_path_var.set(path)

    tk.Button(out_frame, text="Browse...", command=choose_output_path).grid(
        row=0, column=3, sticky="w", padx=4, pady=2
    )
    help_button(out_frame, "output_pdb").grid(row=0, column=4, sticky="w", padx=(0, 4), pady=2)

    status_var = tk.StringVar(value="Ready.")
    status_label = tk.Label(out_frame, textvariable=status_var, anchor="w")
    status_label.grid(row=1, column=0, columnspan=3, sticky="we", padx=4, pady=4)

    # --- Run log frame ---
    log_frame = tk.LabelFrame(root, text="Run log", font=section_font)
    log_frame.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 8))
    log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=(4, 0), pady=4)
    log_scroll.grid(row=0, column=3, sticky="ns", padx=(0, 4), pady=4)

    def append_log(text: str) -> None:
        if not text:
            return
        log_text.configure(state="normal")
        log_text.insert("end", text)
        log_text.see("end")
        log_text.configure(state="disabled")
        root.update_idletasks()

    def clear_log() -> None:
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")

    class TkLogWriter(io.StringIO):
        def write(self, text: str) -> int:
            append_log(text)
            return len(text)

        def flush(self) -> None:
            return None

    def format_cli_command(
        input_pdb: str,
        input_curve: Optional[str],
        output_pdb: str,
        scale_mode: str,
        path_type: str,
        helix_phase: float,
        twist: float,
        scale_anchor: str,
        path_start: float,
        interp_mode: str,
        interp_n: int,
        interp_p: int,
    ) -> str:
        """Return a copyable CLI equivalent for the current GUI run."""
        cmd = [
            "python3",
            "curve_it.py",
            input_pdb,
        ]
        if input_curve:
            cmd.append(input_curve)
        cmd.extend([
            "--scale-mode", scale_mode,
            "--path-type", path_type,
            "--helix_phase", f"{helix_phase:.12g}",
            "--twist", f"{twist:.12g}",
            "--scale-anchor", scale_anchor,
            "--path-start", f"{path_start:.12g}",
            "--interp-mode", interp_mode,
            "--interp-n", str(interp_n),
            "--interp-p", str(interp_p),
            "-o", output_pdb,
        ])
        quoted = [shlex.quote(str(part)) for part in cmd]
        if len(" ".join(quoted)) <= 100:
            return " ".join(quoted)
        first = " ".join(quoted[:3])
        rest = " \\\n    ".join(quoted[3:])
        return f"{first} \\\n    {rest}"

    tk.Button(log_frame, text="Clear", command=clear_log).grid(
        row=1, column=2, sticky="e", padx=4, pady=(0, 4)
    )
    log_frame.grid_columnconfigure(0, weight=1)
    log_frame.grid_rowconfigure(0, weight=1)

    def run_embedding():
        nonlocal atoms, pdb_text, curve_points_raw, curve_points, curve_xyz_path

        if atoms is None or pdb_text is None:
            messagebox.showwarning("No PDB", "Please load a helix PDB file first.")
            return

        pts: np.ndarray
        if curve_points_raw is None:
            # Use default ring
            pts_raw = generate_ring_curve()
            pts_used = compute_interpolated_curve(pts_raw, show_error=True)
            if pts_used is None:
                return
            pts = pts_used
            curve_xyz_path = None
            status_var.set("Using default ring curve (after interpolation).")
        else:
            # Ensure we embed along the curve after applying the current
            # interpolation settings.
            refresh_curve_after_interpolation(show_error=True)
            if curve_points is None:
                return
            pts = curve_points

        # Build scale_mode argument
        smode = scale_mode_var.get()
        if smode == "numeric":
            smode_arg = numeric_length_var.get().strip()
            if not smode_arg:
                messagebox.showerror("Missing numeric length",
                                     "Please enter a numeric target length in Å.")
                return
        else:
            smode_arg = smode

        # Build scale_anchor argument
        anch_mode = scale_anchor_mode_var.get()
        if anch_mode == "custom":
            anch_arg = scale_anchor_custom_var.get().strip()
            if not anch_arg:
                messagebox.showerror(
                    "Missing custom anchor",
                    "Please enter custom anchor coordinates as x,y,z."
                )
                return
        else:
            anch_arg = anch_mode

        # Path type
        ptype = path_type_var.get()

        # Helix phase
        try:
            hphase = float(helix_phase_var.get())
        except ValueError:
            messagebox.showerror("Invalid helix_phase",
                                 "helix_phase must be a number (degrees).")
            return

        # Twist
        try:
            twist_deg = float(twist_var.get())
        except ValueError:
            messagebox.showerror("Invalid twist",
                                 "Twist must be a number (degrees).")
            return

        pstart = float(path_start_var.get())

        out_path = output_path_var.get().strip()
        if not out_path:
            # Derive default from helix PDB path
            if helix_pdb_path:
                base = os.path.basename(helix_pdb_path)
                stem, ext = os.path.splitext(base)
                if not ext:
                    ext = ".pdb"
                out_name = f"{stem}_curved{ext}"
                out_path = os.path.join(os.path.dirname(helix_pdb_path), out_name)
                output_path_var.set(out_path)
            else:
                messagebox.showerror("No output path",
                                     "Please specify an output PDB file.")
                return

        try:
            interp_n_meta = int(interp_n_var.get().strip())
        except ValueError:
            interp_n_meta = 0
        try:
            interp_p_meta = int(interp_p_var.get().strip())
        except ValueError:
            interp_p_meta = 0
        cli_command = format_cli_command(
            input_pdb=helix_pdb_path or helix_path_var.get().strip(),
            input_curve=curve_xyz_path,
            output_pdb=out_path,
            scale_mode=smode_arg,
            path_type=ptype,
            helix_phase=hphase,
            twist=twist_deg,
            scale_anchor=anch_arg,
            path_start=pstart,
            interp_mode=interp_mode_var.get(),
            interp_n=interp_n_meta,
            interp_p=interp_p_meta,
        )

        try:
            clear_log()
            status_var.set("Running embedding...")
            append_log(f"[INFO] Starting {APP_NAME} {APP_VERSION} GUI run.\n")
            append_log("[INFO] CLI equivalent:\n")
            append_log(f"{cli_command}\n")
            if curve_xyz_path is None:
                append_log("[INFO] CLI note: no curve file argument means the default planar ring curve is used.\n")
            root.update_idletasks()

            log_writer = TkLogWriter()
            with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                new_coords, scaled_curve_pts, scaling_applied = embed_helix_on_curve(
                    atoms,
                    pts,
                    scale_mode=smode_arg,
                    path_type=ptype,
                    helix_phase=hphase,
                    twist=twist_deg,
                    scale_anchor=anch_arg,
                    path_start=pstart,
                )

                remark_lines = build_generation_remarks(
                    helix_pdb_path=helix_pdb_path,
                    curve_xyz_path=curve_xyz_path,
                    scale_mode=smode_arg,
                    path_type=ptype,
                    helix_phase=hphase,
                    twist=twist_deg,
                    scale_anchor=anch_arg,
                    path_start=pstart,
                    interp_mode=interp_mode_var.get(),
                    interp_n=interp_n_meta,
                    interp_p=interp_p_meta,
                )

                write_output_pdb(pdb_text, new_coords, out_path, remark_lines=remark_lines)
                write_rescaled_curve_xyz(curve_xyz_path, scaled_curve_pts, scaling_applied)
                print(f"[INFO] GUI run finished successfully. Output: {out_path}")

            status_var.set(f"Done. Wrote PDB to {out_path}")
            messagebox.showinfo("Success", f"Embedding finished.\nOutput: {out_path}")
        except Exception as e:
            import traceback
            append_log(traceback.format_exc())
            status_var.set("Error during embedding.")
            messagebox.showerror("Embedding error", f"An error occurred:\n{e}")

    tk.Button(out_frame, text="Run", command=run_embedding).grid(
        row=1, column=3, sticky="e", padx=4, pady=4
    )
    tk.Button(out_frame, text="Quit", command=root.destroy).grid(
        row=1, column=4, sticky="w", padx=4, pady=4
    )

    for col in range(5):
        out_frame.grid_columnconfigure(col, weight=0)
    out_frame.grid_columnconfigure(1, weight=1)

    # Allow window to resize
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=0)
    root.grid_rowconfigure(1, weight=0)
    root.grid_rowconfigure(2, weight=0)
    root.grid_rowconfigure(3, weight=0)
    root.grid_rowconfigure(4, weight=0)
    root.grid_rowconfigure(5, weight=1)

    root.mainloop()


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description=("Embed a roughly straight DNA/RNA helix, protein helix, "
                     "or other filament-like PDB structure along a 3D curve "
                     "defined by an XYZ-like file (or a default ring), using "
                     "a rotation-minimizing (parallel transport/Bishop) frame, "
                     "chemistry-aware residue/group mapping, and "
                     "holonomy compensation for closed multi-wrap cases. "
                     "If no arguments or only --gui are provided, a GUI is launched.")
    )
    parser.add_argument(
        "helix_pdb",
        nargs="?",
        help=("Input PDB file containing a roughly straight DNA/RNA helix, "
              "protein helix, or other filament-like structure (CLI mode)."),
    )
    parser.add_argument(
        "curve_xyz",
        nargs="?",
        default=None,
        help=("Optional XYZ-like file containing 3D points of the "
              "target curve (CLI mode). If omitted, a default planar ring "
              "curve is used."),
    )
    parser.add_argument(
        "-o", "--output-pdb",
        dest="output_pdb",
        default=None,
        help=("Output PDB file (CLI mode). "
              "If omitted, '<helix_pdb_basename>_curved.pdb' is used."),
    )
    parser.add_argument(
        "--scale-mode",
        default="curve_to_helix",
        help=("How to reconcile length mismatch between helix and curve. "
              "Use 'curve_to_helix' (default) to scale the curve length to "
              "match the helix, 'none' or 'helix_to_curve' to leave the curve "
              "unchanged, or provide a positive number in Å (e.g. "
              "'--scale-mode 340.0') to scale the curve to that target length "
              "without stretching the helix; in that numeric mode, any "
              "mismatch shows up as gaps or overlaps."),
    )
    parser.add_argument(
        "--path-type",
        choices=["open", "closed"],
        default="open",
        help=("Whether the input curve should be treated as an open path "
              "(default) or a closed periodic loop."),
    )
    parser.add_argument(
        "--helix_phase",
        type=float,
        default=0.0,
        help=("Global phase (in degrees) to rotate the helix cross-sections "
              "about its own axis before embedding (default: 0)."),
    )
    parser.add_argument(
        "--twist",
        type=float,
        default=0.0,
        help=("Additional linear twist (in degrees) applied along the helix axis "
              "before embedding. The twist is distributed from 0° at the helix "
              "start to the given angle at the helix end. Positive values "
              "correspond to right-handed twisting, negative values to "
              "left-handed twisting."),
    )
    parser.add_argument(
        "--scale-anchor",
        default="centroid",
        help=("Anchor point for scaling the curve. "
              "Use 'centroid' (default) to scale about the curve centroid, "
              "'origin' to scale about (0,0,0), or provide an explicit point "
              "as 'x,y,z' in Å, e.g. --scale-anchor 0,0,0 or --scale-anchor 10,0,0."),
    )
    parser.add_argument(
        "--path-start",
        type=float,
        default=0.0,
        help=("For closed curves, choose the starting position along the loop "
              "for the helix, as a fraction in [0,1]. 0.0 uses the first point "
              "in the XYZ file as the seam; 0.5 moves the break halfway around "
              "the loop. Ignored for open curves."),
    )
    parser.add_argument(
        "--interp-mode",
        choices=["none", "n", "p"],
        default="none",
        help=("Curve interpolation before embedding. 'none' (default) uses the curve as-is; "
              "'n' resamples the curve to exactly --interp-n points evenly spaced by arc length; "
              "'p' inserts --interp-p points between each pair of adjacent curve points."),
    )
    parser.add_argument(
        "--interp-n",
        type=int,
        default=200,
        help=("When --interp-mode n: total number of points in the interpolated curve (>=2 open, >=3 closed). "
              "Default: 200."),
    )
    parser.add_argument(
        "--interp-p",
        type=int,
        default=0,
        help=("When --interp-mode p: number of extra points to insert between each adjacent point pair (>=0). "
              "Default: 0."),
    )

    parser.add_argument(
        "--gui", "-g",
        action="store_true",
        help="Launch GUI mode instead of CLI.",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"{APP_NAME} {APP_VERSION}",
        help="Show the Curve It version and exit.",
    )

    args = parser.parse_args(argv)

    # Decide whether to use GUI:
    non_gui_tokens = [a for a in argv if a not in ("--gui", "-g")]
    use_gui = (len(non_gui_tokens) == 0)
    if use_gui:
        launch_gui()
        return

    # --- CLI mode ---
    if args.helix_pdb is None:
        parser.error("helix_pdb is required in CLI mode. "
                     "Run with no arguments or --gui for GUI mode.")

    helix_pdb_path = args.helix_pdb
    curve_xyz_path = args.curve_xyz

    # Determine output filename (possibly defaulted).
    if args.output_pdb is None:
        helix_dir = os.path.dirname(helix_pdb_path)
        base = os.path.basename(helix_pdb_path)
        stem, ext = os.path.splitext(base)
        if not ext:
            ext = ".pdb"
        out_name = f"{stem}_curved{ext}"
        output_pdb = os.path.join(helix_dir, out_name)
        print(f"[INFO] Output PDB not specified, using default: {output_pdb}")
    else:
        output_pdb = args.output_pdb

    # Read helix PDB.
    print(f"[INFO] Reading helix PDB: {helix_pdb_path}")
    with open(helix_pdb_path, "r") as f:
        pdb_text = f.read()
    atoms = parse_pdb_atoms(pdb_text)
    if not atoms:
        raise SystemExit("No valid ATOM/HETATM records found in input PDB.")
    print(f"[INFO] Parsed {len(atoms)} atoms from PDB.")

    # Read or generate curve.
    if curve_xyz_path is None:
        print("[INFO] No curve XYZ provided; using default planar ring curve.")
        curve_points = generate_ring_curve()
        print(f"[INFO] Generated ring with {curve_points.shape[0]} points, radius 10.0 Å.")
    else:
        print(f"[INFO] Reading curve XYZ: {curve_xyz_path}")
        with open(curve_xyz_path, "r") as f:
            xyz_text = f.read()
        curve_points = read_xyz_curve_from_text(xyz_text)
        print(f"[INFO] Parsed {curve_points.shape[0]} points from curve file.")

    # Optional curve interpolation (resample/insert points before embedding).
    interp_closed = (args.path_type == "closed")
    try:
        curve_points = apply_curve_interpolation(
            curve_points,
            interp_mode=args.interp_mode,
            interp_n=args.interp_n,
            interp_p=args.interp_p,
            closed=interp_closed,
            verbose=True,
        )
    except Exception as e:
        raise SystemExit(f"Interpolation failed: {e}")

    print(f"[INFO] Embedding helix along curve (scale_mode={args.scale_mode}, "
          f"path_type={args.path_type}, helix_phase={args.helix_phase}, "
          f"twist={args.twist}, path_start={args.path_start})")

    new_coords, scaled_curve_pts, scaling_applied = embed_helix_on_curve(
        atoms,
        curve_points,
        scale_mode=args.scale_mode,
        path_type=args.path_type,
        helix_phase=args.helix_phase,
        twist=args.twist,
        scale_anchor=args.scale_anchor,
        path_start=args.path_start,
    )

    # Report total curvature and writhe of the (scaled) curve when possible
    if HAVE_CURVATURE_WRITHE:
        if args.path_type == "closed":
            try:
                polygonal = curve_looks_polygonal(scaled_curve_pts)
                curve_for_invariants = scaled_curve_pts if polygonal else smooth_closed_curve(scaled_curve_pts)
                if polygonal:
                    # Polygon-like curves can yield spuriously large curvature after periodic spline fitting.
                    total_curv = compute_discrete_total_curvature(scaled_curve_pts, closed=True)
                else:
                    total_curv = fit_spline_and_calculate_curvature(curve_for_invariants)
                wr = calculate_writhe(curve_for_invariants, n_samples=400)
                curv_pi = total_curv / np.pi
                curv_deg = np.degrees(total_curv)
                frac = wr - np.floor(wr)
                wr_deg = frac * 360.0
                print("\n[INFO] Geometric invariants of (scaled) closed curve:")
                print(f"       Total curvature: {total_curv:.6f} "
                      f"(≈ {curv_pi:.6f} * pi; {curv_deg:.3f}°)")
                print(f"       Approximate writhe: {wr:.6f} "
                      f"(residual {wr_deg:.3f}°)")
            except Exception as e:
                print(f"[WARNING] Failed to compute curvature/writhe: {e}")
        else:
            print("[INFO] path_type='open' – skipping total curvature/writhe "
                  "computation (curvature/writhe script assumes closed curves).")
    else:
        print("[INFO] curve_it_lib/cal_xyz_total_curvature_writheV2.py not importable; "
              "skipping curvature/writhe report.")

    remark_lines = build_generation_remarks(
        helix_pdb_path=helix_pdb_path,
        curve_xyz_path=curve_xyz_path,
        scale_mode=args.scale_mode,
        path_type=args.path_type,
        helix_phase=args.helix_phase,
        twist=args.twist,
        scale_anchor=args.scale_anchor,
        path_start=args.path_start,
        interp_mode=args.interp_mode,
        interp_n=args.interp_n,
        interp_p=args.interp_p,
    )

    write_rescaled_curve_xyz(curve_xyz_path, scaled_curve_pts, scaling_applied)
    write_output_pdb(pdb_text, new_coords, output_pdb, remark_lines=remark_lines)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
