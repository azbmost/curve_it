#!/usr/bin/env python3
# Compute the Curve It --helix_phase angle that rotates a selected atom so its
# mapped radial direction faces a user-defined target direction along any
# sufficiently smooth sampled space curve.
#
# Inputs: a straight/roughly straight PDB, the XYZ/text curve used by Curve It,
# a selected atom, the same mapping options used in Curve It, and a target
# direction mode.
# Output: a small text report containing "Helix phase (deg):" and an equivalent
# Curve It command-line example. If no command-line arguments, or --gui, is used,
# the script opens a compact Tkinter GUI for entering these parameters.
#
# Example CLI, local curvature-normal target:
#   python get_curve_it_phaseV5.py BC220_oriented_placed.pdb curve.xyz \
#       --chain A --resseq 1 --atom-name P --scale-mode none --path-type open \
#       --target-mode curvature_angle --curvature-angle-deg 0
#
# Example CLI, target toward a point:
#   python get_curve_it_phaseV5.py BC220_oriented_placed.pdb curve.xyz \
#       --chain A --resseq 1 --atom-name P --target-mode toward_point \
#       --target-point 0,0,0
#
# Example GUI:
#   python get_curve_it_phaseV5.py
#   python get_curve_it_phaseV5.py --gui
#
# Notes:
#   V5 is self-contained for PDB/XYZ parsing and Curve-It-compatible geometry.
#   It does not require importing curve_it.py, but the final value is intended
#   for Curve It's --helix_phase option.
#   A phase rotation can only control the atom direction after projecting the
#   target vector onto the plane perpendicular to the local curve tangent.
#   In curvature_angle mode, the angle is measured in the cross-section
#   plane defined by the local curvature normal and curvature binormal:
#   0 deg = curvature normal, 90 deg = curvature binormal.

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def resource_path(relative_path: str) -> str:
    """Return a resource path that also works from a PyInstaller bundle."""
    source_dir = os.path.dirname(os.path.abspath(__file__))
    source_root = os.path.dirname(source_dir) if os.path.basename(source_dir) == "curve_it_lib" else source_dir
    base_dir = getattr(sys, "_MEIPASS", source_root)
    return os.path.join(base_dir, relative_path)


def set_optional_window_icon(
    root: Any,
    tk_module: Any,
    icon_filenames: List[str],
    image_attr: str,
    default_icon: bool = True,
) -> None:
    """Set a Tk window icon if one of the optional PNG assets is available."""
    for icon_filename in icon_filenames:
        icon_path = resource_path(os.path.join("assets", icon_filename))
        if not os.path.isfile(icon_path):
            continue
        try:
            icon_image = tk_module.PhotoImage(file=icon_path)
            root.iconphoto(default_icon, icon_image)
            setattr(root, image_attr, icon_image)
            return
        except Exception:
            continue


@dataclass
class AtomRecord:
    """Container for a PDB atom/HETATM record and metadata used by Curve It."""

    line: str
    coord: np.ndarray
    atom_name: str
    atom_name_norm: str
    res_name: str
    chain_id: str
    res_seq: int
    i_code: str
    element: str


STANDARD_NA = {
    "A", "C", "G", "U", "T", "I",
    "DA", "DC", "DG", "DT", "DI", "DU",
    "ADE", "CYT", "GUA", "URA", "THY",
    "RA", "RC", "RG", "RU",
}


def parse_vec3(text: str, option_name: str) -> np.ndarray:
    """Parse 'x,y,z' into a numpy vector."""
    try:
        parts = [float(x.strip()) for x in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("{} must be x,y,z".format(option_name))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("{} must be x,y,z".format(option_name))
    return np.array(parts, dtype=float)


def unit_vector(v: np.ndarray, name: str, eps: float = 1e-12) -> np.ndarray:
    """Return a normalized vector, raising a clear error for zero vectors."""
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("{} is too small to normalize".format(name))
    return v / n


def angle_0_360(angle_deg: float) -> float:
    """Normalize an angle to [0, 360)."""
    return angle_deg % 360.0


def angle_minus180_180(angle_deg: float) -> float:
    """Normalize an angle to (-180, 180]."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


def normalize_atom_name(atom_name: str) -> str:
    """Normalize PDB atom names to Curve-It style."""
    return atom_name.strip().upper().replace("*", "'").replace("`", "'")


def guess_element(line: str, atom_name_norm: str) -> str:
    """Guess element from PDB columns 77-78 or from the atom name."""
    elem = ""
    if len(line) >= 78:
        elem = line[76:78].strip()
    if not elem:
        letters = [c for c in atom_name_norm if c.isalpha()]
        elem = letters[0] if letters else ""
    return elem.upper()


def parse_pdb_atoms_from_text(pdb_text: str) -> List[AtomRecord]:
    """Parse ATOM/HETATM records from PDB text."""
    atoms: List[AtomRecord] = []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        try:
            coord = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
        except ValueError:
            continue
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


def first_token_is_integer(line: str) -> bool:
    """Return True if the first token in a line is a single integer atom count."""
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


def curve_component_label(index: int) -> str:
    """Return spreadsheet-style labels: A, B, ..., Z, AA, AB, ..."""
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


def curve_component_index(label: str) -> int:
    """Convert spreadsheet-style component labels to zero-based indices."""
    text = (label or "").strip().upper()
    if not text or not text.isalpha():
        raise ValueError("Invalid component label: {!r}".format(label))
    value = 0
    for char in text:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def combine_curve_components(components: List[np.ndarray], indices: Optional[List[int]] = None) -> np.ndarray:
    """Concatenate selected components in file order into one polyline."""
    if indices is None:
        indices = list(range(len(components)))
    if not indices:
        raise ValueError("At least one curve component must be selected.")
    selected = [components[i] for i in indices]
    combined = np.vstack(selected)
    if combined.shape[0] < 2:
        raise ValueError("Selected curve component(s) must contain at least two points total.")
    return combined


def parse_curve_component_selection(selection: Optional[str], n_components: int) -> List[int]:
    """Parse component labels such as 'A,C' or 'A-C' into zero-based indices."""
    if n_components <= 0:
        raise ValueError("No curve components are available.")
    text = (selection or "all").strip()
    if not text or text.lower() in {"all", "*"}:
        return list(range(n_components))
    chosen: List[int] = []
    for token in text.replace(";", ",").replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_label, end_label = [part.strip() for part in token.split("-", 1)]
            start = curve_component_index(start_label)
            end = curve_component_index(end_label)
            if end < start:
                start, end = end, start
            chosen.extend(range(start, end + 1))
        else:
            chosen.append(curve_component_index(token))
    unique: List[int] = []
    for idx in sorted(chosen):
        if idx < 0 or idx >= n_components:
            max_label = curve_component_label(n_components - 1)
            bad_label = str(idx) if idx < 0 else curve_component_label(idx)
            raise ValueError("Component {} is out of range A-{}.".format(bad_label, max_label))
        if idx not in unique:
            unique.append(idx)
    if not unique:
        raise ValueError("No valid curve components were selected.")
    return unique


def read_xyz_curve_components_from_text(xyz_text: str) -> List[np.ndarray]:
    """Read one or more curve components from XYZ-like text."""
    raw_lines = xyz_text.splitlines()
    nonempty_indices = [i for i, line in enumerate(raw_lines) if line.strip()]
    if not nonempty_indices:
        raise ValueError("XYZ file does not contain readable lines.")

    start_index = nonempty_indices[0]
    if first_token_is_integer(raw_lines[start_index]):
        atom_count = int(raw_lines[start_index].strip())
        data_lines = raw_lines[start_index + 2:start_index + 2 + atom_count]
        pts: List[List[float]] = []
        for line in data_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("!"):
                continue
            point = parse_xyz_coordinate_line(stripped)
            if point is not None:
                pts.append(point)
        if len(pts) < 2:
            raise ValueError("XYZ file does not contain at least two 3D points.")
        return [np.array(pts, dtype=float)]

    components: List[np.ndarray] = []
    current: List[List[float]] = []
    for line in raw_lines[start_index:]:
        stripped = line.strip()
        if not stripped:
            if current:
                components.append(np.array(current, dtype=float))
                current = []
            continue
        if stripped.startswith("#") or stripped.startswith("!"):
            continue
        point = parse_xyz_coordinate_line(stripped)
        if point is not None:
            current.append(point)
    if current:
        components.append(np.array(current, dtype=float))
    components = [component for component in components if component.shape[0] > 0]
    if not components or sum(component.shape[0] for component in components) < 2:
        raise ValueError("XYZ file does not contain at least two 3D points.")
    return components


def read_xyz_curve_from_text(xyz_text: str) -> np.ndarray:
    """Read a 3D polyline from a generic XYZ-like text file."""
    components = read_xyz_curve_components_from_text(xyz_text)
    return combine_curve_components(components)


def compute_helix_local_coords(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Curve-It-style principal-axis local coordinates for input atoms."""
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
    """Compute cumulative arc lengths along a polyline."""
    if points.shape[0] < 2:
        raise ValueError("Need at least two points to define a curve.")
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.zeros(points.shape[0], dtype=float)
    s[1:] = np.cumsum(seg_lengths)
    return s


def get_scale_anchor_point(scale_anchor: str, pts: np.ndarray) -> np.ndarray:
    """Return Curve-It-style scale anchor point."""
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
            return np.array([float(p) for p in parts], dtype=float)
    except Exception:
        pass
    print("[WARNING] Could not parse --scale-anchor='{}', falling back to centroid.".format(scale_anchor))
    return pts.mean(axis=0)


def resample_polyline_by_arc(points: np.ndarray, n_points: int, closed: bool = False) -> np.ndarray:
    """Resample a polyline to n equally spaced arc-length points."""
    if n_points < 2:
        raise ValueError("Interpolation n must be at least 2.")
    pts = np.asarray(points, dtype=float)
    if closed and np.linalg.norm(pts[-1] - pts[0]) > 1e-6:
        pts = np.vstack([pts, pts[0]])
    arc = compute_arc_lengths(pts)
    total = float(arc[-1])
    if total <= 1e-12:
        raise ValueError("Curve length is too small for interpolation.")
    if closed:
        queries = np.linspace(0.0, total, n_points + 1, endpoint=True)[:-1]
    else:
        queries = np.linspace(0.0, total, n_points, endpoint=True)
    out = np.zeros((len(queries), 3), dtype=float)
    for j, s_query in enumerate(queries):
        if s_query <= 0:
            out[j] = pts[0]
            continue
        if s_query >= total:
            out[j] = pts[-1]
            continue
        idx = int(np.searchsorted(arc, s_query) - 1)
        idx = max(0, min(idx, len(pts) - 2))
        s0, s1 = arc[idx], arc[idx + 1]
        u = (s_query - s0) / (s1 - s0) if s1 > s0 else 0.0
        out[j] = (1.0 - u) * pts[idx] + u * pts[idx + 1]
    return out


def insert_points_between_neighbors(points: np.ndarray, p_between: int, closed: bool = False) -> np.ndarray:
    """Insert p equally spaced points between each adjacent input point pair."""
    if p_between < 0:
        raise ValueError("Interpolation p must be non-negative.")
    pts = np.asarray(points, dtype=float)
    segments = len(pts) if closed else len(pts) - 1
    out: List[np.ndarray] = []
    for i in range(segments):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        out.append(a.copy())
        for k in range(1, p_between + 1):
            u = k / float(p_between + 1)
            out.append((1.0 - u) * a + u * b)
    if not closed:
        out.append(pts[-1].copy())
    return np.vstack(out)


def apply_curve_interpolation(points: np.ndarray, interp_mode: str = "none", interp_n: int = 200, interp_p: int = 0, closed: bool = False) -> np.ndarray:
    """Apply Curve-It-compatible optional curve interpolation."""
    mode = (interp_mode or "none").strip().lower()
    if mode in ("none", "off", "false", "0", ""):
        return np.asarray(points, dtype=float)
    if mode in ("n", "n_points", "npoints", "num", "num_points"):
        return resample_polyline_by_arc(points, int(interp_n), closed=closed)
    if mode in ("p", "p_between", "pbetween", "insert"):
        return insert_points_between_neighbors(points, int(interp_p), closed=closed)
    raise ValueError("Unknown interp_mode '{}'. Use 'none', 'n', or 'p'.".format(interp_mode))


def sample_point_by_arc(points: np.ndarray, arc_lengths: np.ndarray, s_query: float) -> np.ndarray:
    """Sample a point along a polyline by arc length."""
    total = float(arc_lengths[-1])
    if s_query <= 0:
        return points[0].copy()
    if s_query >= total:
        return points[-1].copy()
    idx = int(np.searchsorted(arc_lengths, s_query) - 1)
    idx = max(0, min(idx, len(points) - 2))
    s0, s1 = arc_lengths[idx], arc_lengths[idx + 1]
    u = (s_query - s0) / (s1 - s0) if s1 > s0 else 0.0
    return (1.0 - u) * points[idx] + u * points[idx + 1]


def reparameterize_closed_curve(points: np.ndarray, arc_lengths: np.ndarray, frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """Move the seam of a closed polyline to a fractional arc-length position."""
    pts = np.asarray(points, dtype=float)
    if np.linalg.norm(pts[-1] - pts[0]) > 1e-6:
        pts = np.vstack([pts, pts[0]])
        arc_lengths = compute_arc_lengths(pts)
    total = float(arc_lengths[-1])
    if total <= 1e-12:
        raise ValueError("Closed curve length is too small.")
    start_s = (float(frac) % 1.0) * total
    if start_s <= 1e-10:
        return pts, arc_lengths
    start_point = sample_point_by_arc(pts, arc_lengths, start_s)
    # Build samples at the original segment distances, shifted by start_s.
    segment_lengths = np.diff(arc_lengths)
    shifted_s = [start_s]
    acc = start_s
    for seg_len in segment_lengths:
        acc += float(seg_len)
        shifted_s.append(acc)
    new_pts: List[np.ndarray] = []
    for s in shifted_s:
        wrapped_s = s % total
        if s >= total + start_s - 1e-9:
            new_pts.append(start_point.copy())
        else:
            new_pts.append(sample_point_by_arc(pts, arc_lengths, wrapped_s))
    new_arr = np.vstack(new_pts)
    new_arr[0] = start_point
    new_arr[-1] = start_point
    new_arc = compute_arc_lengths(new_arr)
    return new_arr, new_arc


def compute_parallel_transport_frames(points: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Curve-It-style rotation-minimizing frames along a polyline."""
    M = points.shape[0]
    if M < 2:
        raise ValueError("Need at least two points")
    N = np.zeros((M, 3), dtype=float)
    B = np.zeros((M, 3), dtype=float)
    T = np.zeros((M, 3), dtype=float)

    v0 = points[1] - points[0]
    T0 = unit_vector(v0, "first curve segment")
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, T0)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    N0 = tmp - np.dot(tmp, T0) * T0
    N0 = unit_vector(N0, "initial normal")
    B0 = np.cross(T0, N0)
    B0 = unit_vector(B0, "initial binormal")
    N[0], B[0], T[0] = N0, B0, T0

    for i in range(1, M):
        v = points[i] - points[i - 1]
        if np.linalg.norm(v) < eps:
            N[i], B[i], T[i] = N[i - 1], B[i - 1], T[i - 1]
            continue
        Ti = v / np.linalg.norm(v)
        T[i] = Ti
        n_i = N[i - 1] - np.dot(N[i - 1], Ti) * Ti
        n_norm = np.linalg.norm(n_i)
        if n_norm < eps:
            n_i = B[i - 1] - np.dot(B[i - 1], Ti) * Ti
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


def sample_curve_with_frame(
    points: np.ndarray,
    arc_lengths: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    T: np.ndarray,
    s_query: float,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample position and frame at arc length s_query."""
    total = arc_lengths[-1]
    if s_query <= 0:
        return points[0], N[0], B[0], T[0]
    if s_query >= total:
        return points[-1], N[-1], B[-1], T[-1]
    idx = int(np.searchsorted(arc_lengths, s_query) - 1)
    idx = max(0, min(idx, len(points) - 2))
    s0, s1 = arc_lengths[idx], arc_lengths[idx + 1]
    u = (s_query - s0) / (s1 - s0 + eps)
    pos = (1.0 - u) * points[idx] + u * points[idx + 1]
    T_interp = (1.0 - u) * T[idx] + u * T[idx + 1]
    T_interp /= np.linalg.norm(T_interp)
    N_raw = (1.0 - u) * N[idx] + u * N[idx + 1]
    N_proj = N_raw - np.dot(N_raw, T_interp) * T_interp
    n_norm = np.linalg.norm(N_proj)
    if n_norm < eps:
        N_proj = N[idx] - np.dot(N[idx], T_interp) * T_interp
        n_norm = np.linalg.norm(N_proj)
    N_s = N_proj / n_norm
    B_s = np.cross(T_interp, N_s)
    B_s /= np.linalg.norm(B_s)
    return pos, N_s, B_s, T_interp



def sample_point_by_arc_maybe_closed(
    points: np.ndarray,
    arc_lengths: np.ndarray,
    s_query: float,
    closed: bool,
) -> np.ndarray:
    """Sample a point by arc length, wrapping the query for closed curves."""
    total = float(arc_lengths[-1])
    if closed and total > 1e-12:
        s_query = s_query % total
    return sample_point_by_arc(points, arc_lengths, s_query)


def estimate_local_tangent_from_points(
    points: np.ndarray,
    arc_lengths: np.ndarray,
    s_query: float,
    closed: bool,
    step: float,
    eps: float = 1e-10,
) -> np.ndarray:
    """Estimate a local tangent by centered finite difference in arc length."""
    total = float(arc_lengths[-1])
    if total <= eps:
        raise ValueError("Curve length is too small to estimate tangents.")
    ds = max(float(step), total * 1e-6, eps)
    if closed:
        p0 = sample_point_by_arc_maybe_closed(points, arc_lengths, s_query - ds, True)
        p1 = sample_point_by_arc_maybe_closed(points, arc_lengths, s_query + ds, True)
    else:
        s0 = max(0.0, s_query - ds)
        s1 = min(total, s_query + ds)
        if s1 - s0 < eps:
            s0 = max(0.0, min(total, s_query) - 2.0 * ds)
            s1 = min(total, max(0.0, s_query) + 2.0 * ds)
        p0 = sample_point_by_arc(points, arc_lengths, s0)
        p1 = sample_point_by_arc(points, arc_lengths, s1)
    return unit_vector(p1 - p0, "finite-difference tangent")


def default_curvature_step(points: np.ndarray, arc_lengths: np.ndarray) -> float:
    """Choose a conservative arc-length step for finite-difference curvature."""
    seg = np.diff(arc_lengths)
    seg = seg[seg > 1e-8]
    if seg.size == 0:
        total = float(arc_lengths[-1])
        return max(total * 1e-3, 1e-3)
    med = float(np.median(seg))
    total = float(arc_lengths[-1])
    return max(2.0 * med, total * 1e-4, 1e-3)


def discrete_vertex_curvature_normal(
    points: np.ndarray,
    index: int,
    tangent: np.ndarray,
    closed: bool,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, float]:
    """Estimate curvature normal at a polyline vertex from adjacent segment tangents."""
    pts = np.asarray(points, dtype=float)
    n_pts = pts.shape[0]
    if n_pts < 3:
        raise ValueError("At least three curve points are needed to estimate curvature normal.")

    duplicate_closure = closed and np.linalg.norm(pts[-1] - pts[0]) <= 1e-6
    n_unique = n_pts - 1 if duplicate_closure else n_pts
    if n_unique < 3:
        raise ValueError("At least three unique curve points are needed to estimate curvature normal.")

    T0 = unit_vector(tangent, "local tangent")

    if closed:
        i = int(index) % n_unique
        p_prev = pts[(i - 1) % n_unique]
        p_cur = pts[i]
        p_next = pts[(i + 1) % n_unique]
        v_prev = p_cur - p_prev
        v_next = p_next - p_cur
    else:
        i = max(0, min(int(index), n_pts - 1))
        if i <= 0:
            p0, p1, p2 = pts[0], pts[1], pts[2]
            v_prev = p1 - p0
            v_next = p2 - p1
        elif i >= n_pts - 1:
            p0, p1, p2 = pts[-3], pts[-2], pts[-1]
            v_prev = p1 - p0
            v_next = p2 - p1
        else:
            v_prev = pts[i] - pts[i - 1]
            v_next = pts[i + 1] - pts[i]

    t_prev = unit_vector(v_prev, "previous curve segment")
    t_next = unit_vector(v_next, "next curve segment")
    dT = t_next - t_prev
    normal = dT - np.dot(dT, T0) * T0
    norm = float(np.linalg.norm(normal))
    if norm < eps:
        raise ValueError("Discrete curvature normal is too small at this vertex.")
    avg_len = 0.5 * (float(np.linalg.norm(v_prev)) + float(np.linalg.norm(v_next)))
    curvature = norm / max(avg_len, eps)
    return normal / norm, curvature


def estimate_local_curvature_normal(
    points: np.ndarray,
    arc_lengths: np.ndarray,
    s_query: float,
    tangent: np.ndarray,
    closed: bool,
    curvature_step: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, float, float]:
    """Estimate the local principal normal from a sampled smooth curve.

    The primary estimate interpolates discrete vertex curvature normals from the
    two curve vertices bracketing s_query. This works well for densely sampled
    smooth polylines and avoids endpoint bias near the start of open curves. If
    the discrete estimate fails, the function falls back to a finite-difference
    estimate of dT/ds.

    Returns (curvature_normal, estimated_curvature, step_used). The normal is
    projected to be perpendicular to the supplied local tangent.
    """
    pts = np.asarray(points, dtype=float)
    total = float(arc_lengths[-1])
    if total <= eps:
        raise ValueError("Curve length is too small to estimate local curvature.")
    T0 = unit_vector(tangent, "local tangent")

    # First try vertex-normal interpolation, which is usually best for sampled
    # smooth curves and exactly captures the inward direction of a sampled helix.
    try:
        sq = float(s_query)
        if closed and total > eps:
            sq = sq % total
        else:
            sq = min(max(sq, 0.0), total)
        if sq <= 0.0:
            idx = 0
            u = 0.0
        elif sq >= total:
            idx = len(pts) - 2
            u = 1.0
        else:
            idx = int(np.searchsorted(arc_lengths, sq) - 1)
            idx = max(0, min(idx, len(pts) - 2))
            denom = float(arc_lengths[idx + 1] - arc_lengths[idx])
            u = (sq - float(arc_lengths[idx])) / denom if denom > eps else 0.0
        n0, k0 = discrete_vertex_curvature_normal(pts, idx, T0, closed, eps=eps)
        n1, k1 = discrete_vertex_curvature_normal(pts, idx + 1, T0, closed, eps=eps)
        normal = (1.0 - u) * n0 + u * n1
        normal = normal - np.dot(normal, T0) * T0
        n_norm = float(np.linalg.norm(normal))
        if n_norm > eps:
            step_used = float(arc_lengths[idx + 1] - arc_lengths[idx]) if idx + 1 < len(arc_lengths) else 0.0
            curvature = (1.0 - u) * k0 + u * k1
            return normal / n_norm, float(curvature), step_used
    except Exception:
        pass

    # Fallback: finite-difference dT/ds using an automatic or user-provided step.
    base_step = float(curvature_step) if float(curvature_step) > 0.0 else default_curvature_step(points, arc_lengths)
    base_step = min(max(base_step, total * 1e-6), max(total * 0.25, total * 1e-6))
    for mult in (1.0, 2.0, 4.0, 8.0, 16.0):
        ds = min(base_step * mult, max(total * 0.25, base_step))
        if ds <= eps:
            continue
        try:
            Tm = estimate_local_tangent_from_points(points, arc_lengths, s_query - ds, closed, ds * 0.5)
            Tp = estimate_local_tangent_from_points(points, arc_lengths, s_query + ds, closed, ds * 0.5)
        except Exception:
            continue
        dT = Tp - Tm
        normal = dT - np.dot(dT, T0) * T0
        n_norm = float(np.linalg.norm(normal))
        if n_norm > eps:
            curvature = n_norm / max(2.0 * ds, eps)
            return normal / n_norm, curvature, ds

    raise ValueError(
        "Local curvature normal is ill-defined here. The curve may be locally straight, "
        "near an inflection point, or too coarsely sampled. Try a different atom/position, "
        "increase curve sampling density, set --curvature-step, or use --target-mode "
        "toward_point/toward_axis/custom_vector."
    )


def project_target_to_cross_section(
    raw_direction: np.ndarray,
    tangent: np.ndarray,
    mode_name: str,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, float, float]:
    """Project a raw target vector into the plane perpendicular to the tangent."""
    raw_norm = float(np.linalg.norm(raw_direction))
    if raw_norm < eps:
        raise ValueError("Target direction for {} is too small.".format(mode_name))
    T0 = unit_vector(tangent, "local tangent")
    raw_unit = raw_direction / raw_norm
    proj = raw_unit - np.dot(raw_unit, T0) * T0
    proj_norm = float(np.linalg.norm(proj))
    if proj_norm < eps:
        raise ValueError(
            "Target direction for {} is parallel to the local curve tangent after projection. "
            "Curve It --helix_phase rotates cross-sections around the tangent, so it cannot "
            "make a radial atom vector point exactly along the tangent. Choose a target direction "
            "with a nonzero projection into the cross-section plane."
            .format(mode_name)
        )
    tangent_component = float(np.dot(raw_unit, T0))
    return proj / proj_norm, proj_norm, tangent_component


def scaled_reference_point(point: np.ndarray, scale: float, anchor_pt: np.ndarray, scaling_applied: bool) -> np.ndarray:
    """Apply the same curve scaling to a reference point, when relevant."""
    p = np.asarray(point, dtype=float).copy()
    if scaling_applied:
        p = anchor_pt + (p - anchor_pt) * scale
    return p


def compute_target_direction(
    args: argparse.Namespace,
    pts: np.ndarray,
    arc: np.ndarray,
    closed: bool,
    local_axis_point: np.ndarray,
    group_T: np.ndarray,
    scale: float,
    anchor_pt: np.ndarray,
    scaling_applied: bool,
    group_s_query: float,
    target_s_query: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute the target direction in 3D and project it into the cross-section plane."""
    original_mode = (args.target_mode or "curvature_angle").strip().lower()
    mode = original_mode
    normal_angle_alias = False
    if mode in ("curvature", "curvature-plane", "curvature_plane", "curvature-angle"):
        mode = "curvature_angle"
    if mode in ("normal", "normal-plane", "normal_plane", "normal-angle", "normal_angle"):
        # Backward-compatible alias from V4. In V5, curvature_angle already
        # means rotation in the curvature-normal/binormal cross-section plane.
        mode = "curvature_angle"
        normal_angle_alias = True
    if mode in ("axis", "toward-axis"):
        mode = "toward_axis"
    if mode in ("point", "toward-point"):
        mode = "toward_point"
    if mode in ("vector", "custom", "custom-vector"):
        mode = "custom_vector"

    info: Dict[str, Any] = {
        "mode": mode,
        "curvature_normal": None,
        "curvature_binormal": None,
        "estimated_curvature": None,
        "curvature_step_used": None,
        "raw_direction": None,
        "projection_norm": None,
        "tangent_component": None,
        "reference_point": None,
        "axis_point": None,
        "axis_dir": None,
    }

    T0 = unit_vector(group_T, "local curve tangent")

    # Backward compatibility: old V3.1 had --side outer. In V5, use an angle
    # offset instead, but this hidden option keeps older commands meaningful.
    legacy_side = getattr(args, "side", None)
    legacy_flip = legacy_side == "outer"

    if mode == "curvature_angle":
        curv_s = group_s_query if target_s_query is None else float(target_s_query)
        curv_N, curv_kappa, curv_step = estimate_local_curvature_normal(
            pts,
            arc,
            curv_s,
            T0,
            closed,
            float(args.curvature_step),
        )
        info["curvature_s_query"] = curv_s
        curv_B = np.cross(T0, curv_N)
        curv_B = unit_vector(curv_B, "curvature binormal")
        info["curvature_normal"] = curv_N
        info["curvature_binormal"] = curv_B
        info["estimated_curvature"] = curv_kappa
        info["curvature_step_used"] = curv_step

        # V5 convention requested by the user:
        # 0 deg = local curvature normal, 90 deg = local curvature binormal.
        # This is a true cross-section-plane direction, so no tangent projection
        # ambiguity occurs for 90 deg.
        angle_value = args.curvature_angle_deg
        if normal_angle_alias and getattr(args, "normal_angle_deg", None) is not None:
            angle_value = args.normal_angle_deg
        angle_deg = float(angle_value) + (180.0 if legacy_flip else 0.0)
        angle_rad = math.radians(angle_deg)
        raw = math.cos(angle_rad) * curv_N + math.sin(angle_rad) * curv_B
        info["angle_deg"] = angle_deg
        info["angle_definition"] = "cross-section plane: 0=local curvature normal, 90=local curvature binormal"

    elif mode == "toward_axis":
        axis_point = scaled_reference_point(args.axis_point, scale, anchor_pt, scaling_applied)
        axis_dir = unit_vector(args.axis_dir, "--axis-dir")
        closest_on_axis = axis_point + np.dot(local_axis_point - axis_point, axis_dir) * axis_dir
        raw = closest_on_axis - local_axis_point
        if legacy_flip:
            raw = -raw
        info["axis_point"] = axis_point
        info["axis_dir"] = axis_dir
        info["reference_point"] = closest_on_axis
        info["angle_definition"] = "toward nearest point on supplied axis"

    elif mode == "toward_point":
        target_point = scaled_reference_point(args.target_point, scale, anchor_pt, scaling_applied)
        raw = target_point - local_axis_point
        if legacy_flip:
            raw = -raw
        info["reference_point"] = target_point
        info["angle_definition"] = "toward supplied point"

    elif mode == "custom_vector":
        raw = np.asarray(args.custom_vector, dtype=float)
        if legacy_flip:
            raw = -raw
        info["angle_definition"] = "global custom vector, projected to cross-section plane"

    else:
        raise ValueError(
            "Unknown --target-mode '{}'. Use curvature_angle, "
            "toward_axis, toward_point, or custom_vector.".format(args.target_mode)
        )

    projected, proj_norm, tangent_component = project_target_to_cross_section(raw, T0, mode)
    info["raw_direction"] = np.asarray(raw, dtype=float)
    info["projected_direction"] = projected
    info["projection_norm"] = proj_norm
    info["tangent_component"] = tangent_component
    return info

def has_prime(name_norm: str) -> bool:
    """Return True if a normalized atom name contains a nucleic-acid prime."""
    return "'" in name_norm


def is_phosphate_oxygen(name_norm: str) -> bool:
    """Return True for OP/O?P phosphate oxygen names."""
    n = name_norm
    if n in {"OP1", "OP2", "O1P", "O2P"}:
        return True
    if len(n) == 3 and n[0] == "O" and n[2] == "P":
        return True
    return False


def classify_atom_group(atom: AtomRecord) -> str:
    """Classify an atom into Curve-It-style rigid groups."""
    name = atom.atom_name_norm
    elem = atom.element.upper() if atom.element else ""
    if atom.res_name.upper() not in STANDARD_NA:
        return "RES"
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
        return "S" if has_prime(name) else "B"
    return "B"


def build_atom_groups(atoms: List[AtomRecord], s_vals: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    """Build Curve-It-style rigid atom groups and each group axis coordinate."""
    groups_by_key: Dict[Tuple[str, int, str, str], int] = {}
    group_atom_indices: List[List[int]] = []
    for idx, atom in enumerate(atoms):
        gtype = classify_atom_group(atom)
        key = (atom.chain_id, atom.res_seq, atom.i_code, gtype)
        if key not in groups_by_key:
            groups_by_key[key] = len(group_atom_indices)
            group_atom_indices.append([])
        group_atom_indices[groups_by_key[key]].append(idx)
    group_s = np.zeros(len(group_atom_indices), dtype=float)
    for gid, idx_list in enumerate(group_atom_indices):
        group_s[gid] = float(s_vals[idx_list].mean())
    return group_atom_indices, group_s


def read_curve_points(curve_xyz: str, components: Optional[str]) -> np.ndarray:
    """Read a Curve It XYZ/text curve, honoring optional component selection."""
    with open(curve_xyz, "r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    if components:
        comps = read_xyz_curve_components_from_text(text)
        indices = parse_curve_component_selection(components, len(comps))
        return combine_curve_components(comps, indices)
    return read_xyz_curve_from_text(text)


def pdb_atom_serial(atom: AtomRecord) -> Optional[int]:
    """Return integer PDB atom serial from the original line, if present."""
    line = getattr(atom, "line", "")
    try:
        return int(line[6:11].strip())
    except Exception:
        return None


def atom_label(atom: AtomRecord, index0: int) -> str:
    """Return a compact human-readable atom label."""
    serial = pdb_atom_serial(atom)
    serial_text = "serial={}".format(serial) if serial is not None else "serial=?"
    return (
        "index0={} {} chain='{}' res={}{} atom='{}' resname='{}'".format(
            index0,
            serial_text,
            atom.chain_id,
            atom.res_seq,
            atom.i_code,
            atom.atom_name,
            atom.res_name,
        )
    )


def select_atom_index(atoms: List[AtomRecord], args: argparse.Namespace) -> int:
    """Select exactly one atom by serial, zero-based index, or metadata."""
    matches: List[int] = []

    if args.atom_serial is not None:
        for i, atom in enumerate(atoms):
            if pdb_atom_serial(atom) == args.atom_serial:
                matches.append(i)
    elif args.atom_index0 is not None:
        if args.atom_index0 < 0 or args.atom_index0 >= len(atoms):
            raise SystemExit("--atom-index0 is outside the atom list.")
        matches = [args.atom_index0]
    else:
        if args.atom_name is None or args.resseq is None:
            raise SystemExit(
                "Select an atom with --atom-serial, --atom-index0, or at least "
                "--atom-name and --resseq."
            )
        atom_name_norm = normalize_atom_name(args.atom_name)
        for i, atom in enumerate(atoms):
            if atom.atom_name_norm != atom_name_norm:
                continue
            if int(atom.res_seq) != int(args.resseq):
                continue
            if args.chain is not None and atom.chain_id != args.chain:
                continue
            if args.icode is not None and atom.i_code != args.icode:
                continue
            if args.resname is not None and atom.res_name.upper() != args.resname.upper():
                continue
            matches.append(i)

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise SystemExit("No atom matched the selection.")

    lines = ["Atom selection matched multiple atoms. Please refine the selection:"]
    for i in matches[:20]:
        lines.append("  " + atom_label(atoms[i], i))
    if len(matches) > 20:
        lines.append("  ... {} more".format(len(matches) - 20))
    raise SystemExit("\n".join(lines))


def parse_scale_mode(scale_mode: str) -> Tuple[str, Optional[float]]:
    """Parse Curve It scale mode into an internal mode and optional target length."""
    text = str(scale_mode).strip()
    try:
        value = float(text)
        if value <= 0.0:
            raise ValueError
        return "target_length", value
    except ValueError:
        return text.lower(), None


def prepare_curve_like_curve_it(
    curve_points: np.ndarray,
    helix_len: float,
    scale_mode: str,
    path_type: str,
    scale_anchor: str,
    path_start: float,
) -> Tuple[np.ndarray, np.ndarray, str, bool, float, np.ndarray, bool]:
    """Prepare/scale/reparameterize the curve using Curve-It-compatible logic."""
    pts = np.asarray(curve_points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Curve must have at least two points.")

    closed = (path_type.strip().lower() == "closed")
    if closed:
        dist_end = float(np.linalg.norm(pts[-1] - pts[0]))
        if dist_end > 1e-6:
            pts = np.vstack([pts, pts[0]])

    arc_raw = compute_arc_lengths(pts)
    l_curve = float(arc_raw[-1])
    if l_curve <= 1e-6:
        raise ValueError("Curve length is too small.")

    mode, target_length = parse_scale_mode(scale_mode)
    scaling_applied = False
    scale = 1.0
    anchor_pt = np.zeros(3, dtype=float)

    if mode == "none":
        if (not closed) and l_curve + 1e-6 < helix_len:
            raise ValueError(
                "scale_mode='none' with path_type='open' needs a curve at least as long as the PDB axis. "
                "Curve length={:.3f} A; PDB axis length={:.3f} A.".format(l_curve, helix_len)
            )
        target_length = l_curve
        mode = "target_length"

    if mode == "curve_to_helix":
        scale = helix_len / l_curve
        anchor_pt = get_scale_anchor_point(scale_anchor, pts)
        pts = anchor_pt + (pts - anchor_pt) * scale
        arc = compute_arc_lengths(pts)
        scaling_applied = abs(scale - 1.0) > 1e-8
    elif mode == "helix_to_curve":
        arc = arc_raw.copy()
    elif mode == "target_length" and target_length is not None:
        scale = float(target_length) / l_curve
        anchor_pt = get_scale_anchor_point(scale_anchor, pts)
        pts = anchor_pt + (pts - anchor_pt) * scale
        arc = compute_arc_lengths(pts)
        scaling_applied = abs(scale - 1.0) > 1e-8
    else:
        raise ValueError(
            "Unknown --scale-mode '{}'. Use curve_to_helix, none, helix_to_curve, or a positive number.".format(
                scale_mode
            )
        )

    if closed:
        ps = max(0.0, min(1.0, float(path_start)))
        if ps > 1e-6 and ps < 1.0 - 1e-6:
            pts, arc = reparameterize_closed_curve(pts, arc, ps)

    return pts, arc, mode, closed, scale, anchor_pt, scaling_applied


def group_query_like_curve_it(
    mode: str,
    closed: bool,
    total: float,
    helix_len: float,
    s_min: float,
    s_group: float,
    holonomy_angle: float,
) -> Tuple[float, float, int]:
    """Return group curve arc-length query and closed-loop holonomy phase."""
    if mode == "target_length":
        s_local = float(s_group - s_min)
        if closed and total > 0.0:
            lap = int(math.floor(s_local / total))
            return s_local % total, lap * holonomy_angle, lap
        return min(max(s_local, 0.0), total), 0.0, 0

    t_norm = float((s_group - s_min) / helix_len)
    s_query = t_norm * total
    if closed and total > 0.0:
        return s_query % total, 0.0, 0
    return min(max(s_query, 0.0), total), 0.0, 0


def rotate_xy(x: float, y: float, angle_rad: float) -> Tuple[float, float]:
    """Rotate a 2D vector by angle_rad."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return c * x - s * y, s * x + c * y


def shell_quote(text: str) -> str:
    """Small POSIX shell quoting helper without requiring shlex behavior details."""
    if text == "":
        return "''"
    safe = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./:-+=,@%"
    if all(ch in safe for ch in text):
        return text
    return "'" + text.replace("'", "'\\''") + "'"


def default_curved_output(input_pdb: str) -> str:
    """Derive a Curve It output PDB path by adding _curved_phase before .pdb."""
    folder = os.path.dirname(input_pdb)
    base = os.path.basename(input_pdb)
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".pdb"
    return os.path.join(folder, stem + "_curved_phase" + ext)



def format_vec3(vec: Optional[np.ndarray], digits: int = 6) -> str:
    """Format a vector or None for reports."""
    if vec is None:
        return "NA"
    arr = np.asarray(vec, dtype=float).reshape(3)
    fmt = "{{:.{}f}} {{:.{}f}} {{:.{}f}}".format(digits, digits, digits)
    return fmt.format(arr[0], arr[1], arr[2])


def build_phase_report(
    args: argparse.Namespace,
    atom: AtomRecord,
    atom_idx: int,
    phase_deg: float,
    phase_signed_deg: float,
    target_angle_deg: float,
    atom_angle_deg: float,
    holonomy_lap_deg: float,
    final_error_deg: float,
    helix_len: float,
    curve_len: float,
    group_s_query: float,
    local_axis_point: np.ndarray,
    target_info: Dict[str, Any],
    curve_command: str,
) -> str:
    """Format a text report."""
    lines = []
    lines.append("Helix phase (deg): {:.6f}".format(phase_deg))
    lines.append("Curve It --helix_phase (deg): {:.6f}".format(phase_deg))
    lines.append("Helix phase signed (deg): {:.6f}".format(phase_signed_deg))
    lines.append("")
    lines.append("Selected atom: {}".format(atom_label(atom, atom_idx)))
    lines.append("Target mode: {}".format(target_info.get("mode", "NA")))
    lines.append("Target definition: {}".format(target_info.get("angle_definition", "NA")))
    if "angle_deg" in target_info:
        lines.append("Target angle input (deg): {:.6f}".format(float(target_info["angle_deg"])))
    lines.append("Atom radial angle before helix_phase (deg): {:.6f}".format(atom_angle_deg))
    lines.append("Target projected angle in Curve-It frame (deg): {:.6f}".format(target_angle_deg))
    lines.append("Closed-loop holonomy/lap correction (deg): {:.6f}".format(holonomy_lap_deg))
    lines.append("Verification final angular error (deg): {:.6e}".format(final_error_deg))
    lines.append("")
    lines.append("PDB principal-axis length (A): {:.6f}".format(helix_len))
    lines.append("Prepared curve length used by mapping (A): {:.6f}".format(curve_len))
    lines.append("Selected group arc-length on curve (A): {:.6f}".format(group_s_query))
    lines.append("Mapped local-axis point for selected atom (A): {}".format(format_vec3(local_axis_point)))
    lines.append("")
    lines.append("Raw target direction before projection: {}".format(format_vec3(target_info.get("raw_direction"), digits=8)))
    lines.append("Projected target unit vector: {}".format(format_vec3(target_info.get("projected_direction"), digits=8)))
    lines.append("Projection norm before normalization: {:.8f}".format(float(target_info.get("projection_norm", float("nan")))))
    lines.append("Raw target tangent component: {:.8f}".format(float(target_info.get("tangent_component", float("nan")))))
    if target_info.get("curvature_normal") is not None:
        lines.append("Local curvature normal unit vector: {}".format(format_vec3(target_info.get("curvature_normal"), digits=8)))
        lines.append("Local curvature-binormal unit vector: {}".format(format_vec3(target_info.get("curvature_binormal"), digits=8)))
        lines.append("Estimated local curvature (1/A): {:.8e}".format(float(target_info.get("estimated_curvature"))))
        lines.append("Curvature arc-length query used (A): {:.6f}".format(float(target_info.get("curvature_s_query", float("nan")))))
        lines.append("Curvature finite-difference step used (A): {:.6f}".format(float(target_info.get("curvature_step_used"))))
    if target_info.get("axis_point") is not None:
        lines.append("Reference axis point after scaling, if any (A): {}".format(format_vec3(target_info.get("axis_point"))))
        lines.append("Reference axis direction unit vector: {}".format(format_vec3(target_info.get("axis_dir"), digits=8)))
    if target_info.get("reference_point") is not None:
        lines.append("Reference point used for target (A): {}".format(format_vec3(target_info.get("reference_point"))))
    lines.append("")
    lines.append("Equivalent Curve It command:")
    lines.append(curve_command)
    lines.append("")
    lines.append("Implementation note: V5 computes a target direction for any sufficiently smooth sampled space curve.")
    lines.append("In curvature_angle mode the target is cos(angle)*curvature_normal + sin(angle)*curvature_binormal.")
    lines.append("In axis/point/vector modes the target is projected into the plane perpendicular to the local tangent,")
    lines.append("matching Curve It's cross-section phase convention.")
    return "\n".join(lines) + "\n"


def compute_phase(args: argparse.Namespace) -> str:
    """Compute the Curve-It phase report."""
    with open(args.input_pdb, "r", encoding="utf-8", errors="ignore") as handle:
        pdb_text = handle.read()
    atoms = parse_pdb_atoms_from_text(pdb_text)
    if not atoms:
        raise SystemExit("No ATOM/HETATM records found in input PDB.")

    selected_idx = select_atom_index(atoms, args)
    selected_atom = atoms[selected_idx]

    coords = np.vstack([a.coord for a in atoms])
    center, axis, basis, s_vals, local = compute_helix_local_coords(coords)
    s_min = float(s_vals.min())
    s_max = float(s_vals.max())
    helix_len = s_max - s_min
    if helix_len <= 1e-6:
        raise SystemExit("PDB principal-axis length is too small.")

    curve_points = read_curve_points(args.curve_xyz, args.curve_components)
    closed_for_interp = args.path_type.strip().lower() == "closed"
    curve_points = apply_curve_interpolation(
        curve_points,
        args.interp_mode,
        args.interp_n,
        args.interp_p,
        closed_for_interp,
    )

    pts, arc, mode, closed, scale, anchor_pt, scaling_applied = prepare_curve_like_curve_it(
        curve_points,
        helix_len,
        args.scale_mode,
        args.path_type,
        args.scale_anchor,
        args.path_start,
    )
    total = float(arc[-1])

    Nf, Bf, Tf = compute_parallel_transport_frames(pts)

    holonomy_angle = 0.0
    if closed:
        r0 = np.column_stack((Nf[0], Bf[0], Tf[0]))
        rend = np.column_stack((Nf[-1], Bf[-1], Tf[-1]))
        s_mat = r0.T @ rend
        holonomy_angle = float(math.atan2(s_mat[1, 0], s_mat[0, 0]))

    group_atom_indices, group_s = build_atom_groups(atoms, s_vals)
    atom_to_group = np.full(len(atoms), -1, dtype=int)
    for gid_loop, idx_list in enumerate(group_atom_indices):
        for idx_loop in idx_list:
            atom_to_group[idx_loop] = gid_loop
    gid = int(atom_to_group[selected_idx])
    if gid < 0:
        raise SystemExit("Internal error: selected atom was not assigned to a Curve It atom group.")

    s_group = float(group_s[gid])
    group_s_query, group_phi_lap, lap = group_query_like_curve_it(
        mode, closed, total, helix_len, s_min, s_group, holonomy_angle
    )
    group_pos, group_N, group_B, group_T = sample_curve_with_frame(
        pts, arc, Nf, Bf, Tf, group_s_query
    )

    # Curve It places atoms in a rigid group at:
    #   group_pos + u*N + v*B + (s_atom - s_group)*T
    # The local-axis point corresponding to the selected atom is therefore the
    # group center shifted by w_axis along the tangent.
    w_axis = float(s_vals[selected_idx] - s_group)
    local_axis_point = group_pos + w_axis * group_T

    target_info = compute_target_direction(
        args,
        pts,
        arc,
        closed,
        local_axis_point,
        group_T,
        scale,
        anchor_pt,
        scaling_applied,
        group_s_query,
        (group_s_query + w_axis) % total if closed else min(max(group_s_query + w_axis, 0.0), total),
    )
    target_unit = target_info["projected_direction"]

    # Express the desired direction in the Curve-It normal/binormal frame.
    target_x = float(np.dot(target_unit, group_N))
    target_y = float(np.dot(target_unit, group_B))
    target_angle = math.atan2(target_y, target_x)

    # Curve It first applies linear twist, then global helix_phase, then closed-loop
    # lap/holonomy correction if applicable. We solve for the global helix_phase.
    u0 = float(local[selected_idx, 0])
    v0 = float(local[selected_idx, 1])
    if math.hypot(u0, v0) < 1e-8:
        raise SystemExit(
            "The selected atom is too close to the PDB principal axis; its radial direction is ill-defined."
        )

    twist_rad = math.radians(float(args.twist))
    t_norm_atom = float((s_vals[selected_idx] - s_min) / helix_len)
    u_twist, v_twist = rotate_xy(u0, v0, twist_rad * t_norm_atom)
    atom_angle = math.atan2(v_twist, u_twist)

    phase_rad = target_angle - atom_angle - group_phi_lap
    phase_deg = angle_0_360(math.degrees(phase_rad))
    phase_signed_deg = angle_minus180_180(phase_deg)

    # Verification in the same angle convention.
    final_angle = atom_angle + math.radians(phase_deg) + group_phi_lap
    final_error_deg = angle_minus180_180(math.degrees(final_angle - target_angle))

    helix_phase_for_command = "{:.6f}".format(phase_deg)
    command_parts = [
        "python curve_it.py",
        shell_quote(args.input_pdb),
        shell_quote(args.curve_xyz),
        "--scale-mode", shell_quote(str(args.scale_mode)),
        "--path-type", shell_quote(str(args.path_type)),
        "--helix_phase", helix_phase_for_command,
    ]
    if abs(float(args.twist)) > 1e-12:
        command_parts.extend(["--twist", "{:.6f}".format(float(args.twist))])
    if args.scale_anchor != "centroid":
        command_parts.extend(["--scale-anchor", shell_quote(str(args.scale_anchor))])
    if args.path_type.strip().lower() == "closed" and abs(float(args.path_start)) > 1e-12:
        command_parts.extend(["--path-start", "{:.6f}".format(float(args.path_start))])
    if args.curve_components:
        command_parts.extend(["--curve-components", shell_quote(args.curve_components)])
    if (args.interp_mode or "none").strip().lower() not in ("", "none"):
        command_parts.extend(["--interp-mode", shell_quote(args.interp_mode)])
        if args.interp_mode == "n":
            command_parts.extend(["--interp-n", str(args.interp_n)])
        elif args.interp_mode == "p":
            command_parts.extend(["--interp-p", str(args.interp_p)])
    command_parts.extend(["-o", shell_quote(default_curved_output(args.input_pdb))])
    curve_command = " ".join(command_parts)

    return build_phase_report(
        args=args,
        atom=selected_atom,
        atom_idx=selected_idx,
        phase_deg=phase_deg,
        phase_signed_deg=phase_signed_deg,
        target_angle_deg=math.degrees(target_angle),
        atom_angle_deg=math.degrees(atom_angle),
        holonomy_lap_deg=math.degrees(group_phi_lap),
        final_error_deg=final_error_deg,
        helix_len=helix_len,
        curve_len=total,
        group_s_query=group_s_query,
        local_axis_point=local_axis_point,
        target_info=target_info,
        curve_command=curve_command,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Curve It's --helix_phase angle so a selected atom faces a "
            "specified direction along any sufficiently smooth sampled space curve."
        )
    )
    parser.add_argument("input_pdb", nargs="?", help="Input PDB used by Curve It.")
    parser.add_argument("curve_xyz", nargs="?", help="XYZ/text curve file used by Curve It.")

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the Tkinter GUI. GUI mode is also used automatically when no arguments are given.",
    )

    # Kept for compatibility with V2/V3 GUIs/scripts. V5 does not need to import curve_it.py.
    parser.add_argument(
        "--curve-it-dir",
        default=None,
        help=argparse.SUPPRESS,
    )

    sel = parser.add_argument_group("atom selection")
    sel.add_argument("--atom-serial", type=int, default=None, help="PDB atom serial number, columns 7-11.")
    sel.add_argument("--atom-index0", type=int, default=None, help="Zero-based ATOM/HETATM index after PDB parsing.")
    sel.add_argument("--atom-name", default=None, help="Atom name, e.g. P, C1', N1, C6.")
    sel.add_argument("--chain", default=None, help="Chain ID for metadata selection.")
    sel.add_argument("--resseq", type=int, default=None, help="Residue number for metadata selection.")
    sel.add_argument("--icode", default=None, help="Insertion code for metadata selection.")
    sel.add_argument("--resname", default=None, help="Residue name for metadata selection, e.g. DA.")

    mapg = parser.add_argument_group("Curve It mapping options")
    mapg.add_argument("--scale-mode", default="none", help="Curve It scale mode: none, curve_to_helix, helix_to_curve, or numeric length.")
    mapg.add_argument("--path-type", choices=["open", "closed"], default="open", help="Curve It path type.")
    mapg.add_argument("--scale-anchor", default="centroid", help="Curve It scale anchor: centroid, origin, or x,y,z.")
    mapg.add_argument("--path-start", type=float, default=0.0, help="Curve It path_start for closed curves.")
    mapg.add_argument("--twist", type=float, default=0.0, help="Curve It total additional twist in degrees.")
    mapg.add_argument("--curve-components", default=None, help="Optional component selection such as A,C or A-C.")
    mapg.add_argument("--interp-mode", choices=["none", "n", "p"], default="none", help="Optional Curve It interpolation mode.")
    mapg.add_argument("--interp-n", type=int, default=200, help="Number of points for --interp-mode n.")
    mapg.add_argument("--interp-p", type=int, default=0, help="Inserted points between neighbors for --interp-mode p.")

    targetg = parser.add_argument_group("target direction options")
    targetg.add_argument(
        "--target-mode",
        choices=["curvature_angle", "normal_angle", "toward_axis", "toward_point", "custom_vector"],
        default="curvature_angle",
        help=(
            "Direction mode: curvature_angle, toward_axis, toward_point, or custom_vector. "
            "curvature_angle uses the local cross-section plane: "
            "0=local curvature normal, 90=local curvature binormal. "
            "normal_angle is accepted as a legacy alias."
        ),
    )
    targetg.add_argument(
        "--curvature-angle-deg",
        type=float,
        default=0.0,
        help="Angle in degrees for curvature_angle mode: 0=local curvature normal, 90=local curvature binormal.",
    )
    targetg.add_argument(
        "--normal-angle-deg",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    targetg.add_argument(
        "--curvature-step",
        type=float,
        default=0.0,
        help="Finite-difference step in A for estimating local curvature normal. 0 means automatic.",
    )
    targetg.add_argument(
        "--axis-point",
        type=lambda s: parse_vec3(s, "--axis-point"),
        default=np.array([0.0, 0.0, 0.0], dtype=float),
        help="A point on the reference axis for --target-mode toward_axis. Default: 0,0,0.",
    )
    targetg.add_argument(
        "--axis-dir",
        type=lambda s: parse_vec3(s, "--axis-dir"),
        default=np.array([0.0, 0.0, 1.0], dtype=float),
        help="Reference axis direction for --target-mode toward_axis. Default: 0,0,1.",
    )
    targetg.add_argument(
        "--target-point",
        type=lambda s: parse_vec3(s, "--target-point"),
        default=np.array([0.0, 0.0, 0.0], dtype=float),
        help="Reference point for --target-mode toward_point. Default: 0,0,0.",
    )
    targetg.add_argument(
        "--custom-vector",
        type=lambda s: parse_vec3(s, "--custom-vector"),
        default=np.array([1.0, 0.0, 0.0], dtype=float),
        help="Global direction vector for --target-mode custom_vector. Default: 1,0,0.",
    )
    # Hidden backward-compatibility option from V3.1.
    targetg.add_argument("--side", choices=["inner", "outer"], default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output report path. Default: <input_pdb>_curve_it_phase.txt.",
    )
    return parser


def default_output_path(input_pdb: str) -> str:
    """Derive the report path by adding _curve_it_phase before the extension."""
    folder = os.path.dirname(os.path.abspath(input_pdb))
    base = os.path.basename(input_pdb)
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".txt"
    return os.path.join(folder, stem + "_curve_it_phase.txt")


def extract_phase_deg(report: str) -> float:
    """Extract the computed phase from a Curve It phase report."""
    for line in report.splitlines():
        if line.startswith("Helix phase (deg):"):
            return float(line.split(":", 1)[1].strip())
    raise ValueError("Could not find 'Helix phase (deg)' in the phase report.")


def _namespace_value(initial: Optional[argparse.Namespace], name: str, default: Any) -> Any:
    """Return a value from an optional argparse namespace for GUI prefilling."""
    if initial is None:
        return default
    value = getattr(initial, name, default)
    if value is None:
        return default
    return value


def _vec3_to_text(value: Any, default: str) -> str:
    """Format a 3-vector as x,y,z for GUI entries."""
    try:
        arr = np.asarray(value, dtype=float).reshape(3)
        return "{:.6g},{:.6g},{:.6g}".format(arr[0], arr[1], arr[2])
    except Exception:
        return default


def _optional_int(text: str, name: str) -> Optional[int]:
    """Parse an optional integer GUI entry."""
    value = text.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError("{} must be an integer or blank.".format(name))


def _optional_float(text: str, name: str) -> Optional[float]:
    """Parse an optional float GUI entry."""
    value = text.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError("{} must be a number or blank.".format(name))


def _optional_text(text: str) -> Optional[str]:
    """Convert blank GUI text to None."""
    value = text.strip()
    return value if value else None


def _required_path(text: str, name: str) -> str:
    """Validate a required GUI path/string."""
    value = text.strip()
    if not value:
        raise ValueError("{} is required.".format(name))
    return value


def run_gui(
    initial: Optional[argparse.Namespace] = None,
    parent: Optional[Any] = None,
    on_phase: Optional[Callable[[float, str, str], None]] = None,
) -> Any:
    """Open a compact Tkinter GUI for computing the Curve It phase."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        from tkinter import scrolledtext
    except Exception as exc:
        raise SystemExit("Tkinter is not available in this Python environment: {}".format(exc))

    owns_mainloop = parent is None
    root = tk.Tk() if owns_mainloop else tk.Toplevel(parent)
    root.title("Curve It phase helper V5")
    set_optional_window_icon(
        root,
        tk,
        ["get_phase_icon.png", "icon.png"],
        "_curve_it_phase_icon_image",
        default_icon=owns_mainloop,
    )
    if parent is not None:
        root.transient(parent)

    vars_dict: Dict[str, Any] = {}
    widgets_dict: Dict[str, Any] = {}
    help_button_kwargs = {
        "text": "?",
        "width": 2,
        "bg": "#cfefff",
        "activebackground": "#aee6ff",
        "relief": tk.RAISED,
        "borderwidth": 1,
    }

    help_topics = {
        "input_pdb": (
            "Input PDB",
            "Input PDB file used by Curve It. The structure should be the same straight or roughly straight PDB that will be bent onto the curve."
        ),
        "curve_xyz": (
            "Curve XYZ/text",
            "Curve file used by Curve It. Plain x y z rows and standard molecular XYZ-like files are accepted. Coordinates are in Angstrom-like units."
        ),
        "output": (
            "Output report",
            "Text report written after computing the phase. If blank, the helper writes <input_pdb>_curve_it_phase.txt beside the input PDB."
        ),
        "chain": (
            "ChainID",
            "Optional PDB chain ID for the selected atom. Example: A. Leave blank only if atom name and residue number uniquely identify one atom."
        ),
        "resseq": (
            "ResNumber",
            "PDB residue sequence number for the selected atom. Example: 12. This is required when selecting by residue and atom name."
        ),
        "atom_name": (
            "Atom name",
            "PDB atom name whose radial direction should be aimed. Examples: P, C1', N1, C6."
        ),
        "scale_mode": (
            "Scale mode",
            "Curve It scale mode. Use none, curve_to_helix, helix_to_curve, or a positive numeric target curve length in Angstrom."
        ),
        "path_type": (
            "Path type",
            "Use closed for periodic loops and open for paths with distinct ends. Closed mode enables path start."
        ),
        "twist": (
            "Twist (deg)",
            "Total additional axial twist in degrees, matching the Curve It Twist (deg) field. Example: 360 adds one full turn."
        ),
        "scale_anchor": (
            "Scale anchor",
            "Anchor used when Curve It scales the curve. Use centroid, origin, or custom x,y,z coordinates in Angstrom. This is only relevant for modes that scale the curve."
        ),
        "path_start": (
            "Path start",
            "Closed curves only. Fraction from 0 to 1 describing where the PDB starts on the loop. Example: 0.25 starts one quarter around the loop."
        ),
        "curve_components": (
            "Components",
            "Optional curve component selection for multi-component coordinate files. Examples: A,C or A-C. Leave blank to use all components."
        ),
        "interp_mode": (
            "Interp mode",
            "Optional Curve It interpolation mode. none uses the original curve, n resamples to a total point count, and p inserts points per segment."
        ),
        "interp_n": (
            "Interp n",
            "Used only when Interp mode is n. Unit: number of total curve points after resampling. Example: 400."
        ),
        "interp_p": (
            "Interp p",
            "Used only when Interp mode is p. Unit: inserted points per neighboring point pair. Example: 5."
        ),
        "target_mode": (
            "Mode",
            "Target direction mode. curvature_angle uses the local curvature frame; toward_axis points toward a reference axis; toward_point points toward a point; custom_vector uses a global vector."
        ),
        "curvature_angle_deg": (
            "Angle deg",
            "Used only for curvature_angle mode. Unit: degrees. 0 points along local curvature normal; 90 points along local curvature binormal."
        ),
        "curvature_step": (
            "Curv step",
            "Used only for curvature_angle mode. Unit: Angstrom along the curve. 0 chooses an automatic finite-difference step."
        ),
        "axis_point": (
            "Axis point",
            "Used only for toward_axis mode. A point on the reference axis, entered as x,y,z in Angstrom. Example: 0,0,0."
        ),
        "axis_dir": (
            "Axis dir",
            "Used only for toward_axis mode. Direction vector of the reference axis. Unitless. Example: 0,0,1."
        ),
        "target_point": (
            "Target point",
            "Used only for toward_point mode. Reference point entered as x,y,z in Angstrom. Example: 0,0,0."
        ),
        "custom_vector": (
            "Custom vector",
            "Used only for custom_vector mode. Global direction vector; only its direction matters. Example: 1,0,0."
        ),
    }

    def show_help(topic_key: str) -> None:
        title, body = help_topics[topic_key]
        messagebox.showinfo(title, body, parent=root)

    def help_button(parent_widget: Any, topic_key: str) -> Any:
        return tk.Button(parent_widget, command=lambda: show_help(topic_key), **help_button_kwargs)

    def add_path_row(parent_widget, row, label, key, default="", browse="file", width=76):
        """Add a full-width path entry with a Browse button."""
        tk.Label(parent_widget, text=label, anchor="w").grid(row=row, column=0, sticky="w", padx=4, pady=2)
        var = tk.StringVar(value=str(default))
        vars_dict[key] = var
        entry = tk.Entry(parent_widget, textvariable=var, width=width)
        widgets_dict[key] = entry
        entry.grid(row=row, column=1, columnspan=5, sticky="ew", padx=4, pady=2)

        def do_browse():
            if browse == "file":
                path = filedialog.askopenfilename(parent=root)
            elif browse == "save":
                path = filedialog.asksaveasfilename(parent=root, defaultextension=".txt")
            else:
                path = ""
            if path:
                var.set(path)

        tk.Button(parent_widget, text="Browse", command=do_browse).grid(row=row, column=6, sticky="ew", padx=4, pady=2)
        help_button(parent_widget, key).grid(row=row, column=7, sticky="w", padx=(0, 4), pady=2)
        return row + 1

    def add_section(parent_widget, row, text):
        """Add a compact section label."""
        tk.Label(parent_widget, text=text, font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=9, sticky="w", padx=4, pady=(10, 2)
        )
        return row + 1

    def add_compact_row(parent_widget, row, specs):
        """Add several labeled widgets on one row.

        Each spec is (label, key, default, kind, choices, width, help_key).
        kind is 'entry' or 'option'.
        """
        col = 0
        for label, key, default, kind, choices, width, help_key in specs:
            tk.Label(parent_widget, text=label, anchor="w").grid(row=row, column=col, sticky="w", padx=(4, 2), pady=2)
            var = tk.StringVar(value=str(default))
            vars_dict[key] = var
            if kind == "option":
                widget = tk.OptionMenu(parent_widget, var, *choices)
                widget.grid(row=row, column=col + 1, sticky="ew", padx=(2, 8), pady=2)
            else:
                widget = tk.Entry(parent_widget, textvariable=var, width=width)
                widget.grid(row=row, column=col + 1, sticky="ew", padx=(2, 8), pady=2)
            widgets_dict[key] = widget
            help_button(parent_widget, help_key).grid(row=row, column=col + 2, sticky="w", padx=(0, 8), pady=2)
            col += 3
        return row + 1

    frame = tk.Frame(root)
    frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=0)
    for col in range(9):
        frame.columnconfigure(col, weight=1 if col in (1, 4, 7) else 0)

    row = 0
    row = add_path_row(frame, row, "Input PDB", "input_pdb", _namespace_value(initial, "input_pdb", ""), browse="file")
    row = add_path_row(frame, row, "Curve XYZ/text", "curve_xyz", _namespace_value(initial, "curve_xyz", ""), browse="file")
    row = add_path_row(frame, row, "Output report", "output", _namespace_value(initial, "output", ""), browse="save")

    row = add_section(frame, row, "Atom selection")
    row = add_compact_row(
        frame,
        row,
        [
            ("ChainID", "chain", _namespace_value(initial, "chain", ""), "entry", None, 8, "chain"),
            ("ResNumber", "resseq", _namespace_value(initial, "resseq", ""), "entry", None, 10, "resseq"),
            ("Atom name", "atom_name", _namespace_value(initial, "atom_name", ""), "entry", None, 12, "atom_name"),
        ],
    )

    row = add_section(frame, row, "Curve It mapping options")
    row = add_compact_row(
        frame,
        row,
        [
            ("Scale mode", "scale_mode", _namespace_value(initial, "scale_mode", "none"), "entry", None, 14, "scale_mode"),
            ("Path type", "path_type", _namespace_value(initial, "path_type", "open"), "option", ["open", "closed"], 10, "path_type"),
            ("Twist (deg)", "twist", _namespace_value(initial, "twist", 0.0), "entry", None, 10, "twist"),
        ],
    )
    row = add_compact_row(
        frame,
        row,
        [
            ("Scale anchor (A)", "scale_anchor", _namespace_value(initial, "scale_anchor", "centroid"), "entry", None, 14, "scale_anchor"),
            ("Path start (0-1)", "path_start", _namespace_value(initial, "path_start", 0.0), "entry", None, 10, "path_start"),
            ("Components", "curve_components", _namespace_value(initial, "curve_components", ""), "entry", None, 10, "curve_components"),
        ],
    )
    row = add_compact_row(
        frame,
        row,
        [
            ("Interp mode", "interp_mode", _namespace_value(initial, "interp_mode", "none"), "option", ["none", "n", "p"], 10, "interp_mode"),
            ("Interp n (points)", "interp_n", _namespace_value(initial, "interp_n", 200), "entry", None, 10, "interp_n"),
            ("Interp p (points/seg)", "interp_p", _namespace_value(initial, "interp_p", 0), "entry", None, 10, "interp_p"),
        ],
    )

    row = add_section(frame, row, "Target direction")
    row = add_compact_row(
        frame,
        row,
        [
            (
                "Mode",
                "target_mode",
                _namespace_value(initial, "target_mode", "curvature_angle"),
                "option",
                ["curvature_angle", "toward_axis", "toward_point", "custom_vector"],
                18,
                "target_mode",
            ),
            ("Angle (deg)", "curvature_angle_deg", _namespace_value(initial, "curvature_angle_deg", 0.0), "entry", None, 10, "curvature_angle_deg"),
            ("Curv step (A)", "curvature_step", _namespace_value(initial, "curvature_step", 0.0), "entry", None, 10, "curvature_step"),
        ],
    )
    row = add_compact_row(
        frame,
        row,
        [
            (
                "Axis point (A)",
                "axis_point",
                _vec3_to_text(_namespace_value(initial, "axis_point", np.array([0.0, 0.0, 0.0])), "0,0,0"),
                "entry",
                None,
                18,
                "axis_point",
            ),
            (
                "Axis dir",
                "axis_dir",
                _vec3_to_text(_namespace_value(initial, "axis_dir", np.array([0.0, 0.0, 1.0])), "0,0,1"),
                "entry",
                None,
                18,
                "axis_dir",
            ),
        ],
    )
    row = add_compact_row(
        frame,
        row,
        [
            (
                "Target point (A)",
                "target_point",
                _vec3_to_text(_namespace_value(initial, "target_point", np.array([0.0, 0.0, 0.0])), "0,0,0"),
                "entry",
                None,
                18,
                "target_point",
            ),
            (
                "Custom vector",
                "custom_vector",
                _vec3_to_text(_namespace_value(initial, "custom_vector", np.array([1.0, 0.0, 0.0])), "1,0,0"),
                "entry",
                None,
                18,
                "custom_vector",
            ),
        ],
    )

    def set_widget_enabled(key: str, enabled: bool) -> None:
        widget = widgets_dict.get(key)
        if widget is not None:
            widget.config(state="normal" if enabled else "disabled")

    def scale_mode_uses_anchor(mode_text: str) -> bool:
        mode = (mode_text or "").strip().lower()
        if mode == "curve_to_helix":
            return True
        try:
            return float(mode) > 0.0
        except ValueError:
            return False

    def update_dynamic_fields(*_args: Any) -> None:
        interp_mode = (vars_dict["interp_mode"].get() or "none").strip().lower()
        set_widget_enabled("interp_n", interp_mode == "n")
        set_widget_enabled("interp_p", interp_mode == "p")

        path_type = (vars_dict["path_type"].get() or "open").strip().lower()
        set_widget_enabled("path_start", path_type == "closed")

        set_widget_enabled("scale_anchor", scale_mode_uses_anchor(vars_dict["scale_mode"].get()))

        target_mode = (vars_dict["target_mode"].get() or "curvature_angle").strip().lower()
        curvature_mode = target_mode in {"curvature_angle", "normal_angle", "curvature-angle", "curvature"}
        set_widget_enabled("curvature_angle_deg", curvature_mode)
        set_widget_enabled("curvature_step", curvature_mode)
        set_widget_enabled("axis_point", target_mode == "toward_axis")
        set_widget_enabled("axis_dir", target_mode == "toward_axis")
        set_widget_enabled("target_point", target_mode == "toward_point")
        set_widget_enabled("custom_vector", target_mode == "custom_vector")

    for dynamic_key in ("interp_mode", "path_type", "scale_mode", "target_mode"):
        vars_dict[dynamic_key].trace_add("write", update_dynamic_fields)
    update_dynamic_fields()

    report_box = scrolledtext.ScrolledText(root, width=112, height=18)
    report_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
    root.rowconfigure(1, weight=1)

    def collect_args() -> argparse.Namespace:
        try:
            axis_point = parse_vec3(vars_dict["axis_point"].get(), "axis-point")
            axis_dir = parse_vec3(vars_dict["axis_dir"].get(), "axis-dir")
            target_point = parse_vec3(vars_dict["target_point"].get(), "target-point")
            custom_vector = parse_vec3(vars_dict["custom_vector"].get(), "custom-vector")
        except argparse.ArgumentTypeError as exc:
            raise ValueError(str(exc))
        return argparse.Namespace(
            gui=True,
            input_pdb=_required_path(vars_dict["input_pdb"].get(), "Input PDB"),
            curve_xyz=_required_path(vars_dict["curve_xyz"].get(), "Curve XYZ/text"),
            curve_it_dir=None,
            output=_optional_text(vars_dict["output"].get()),
            # The compact GUI intentionally uses only ChainID, ResNumber, and Atom name.
            # CLI mode still supports atom serial/index0/insertion-code/resname for batch scripts.
            atom_serial=None,
            atom_index0=None,
            atom_name=_optional_text(vars_dict["atom_name"].get()),
            chain=_optional_text(vars_dict["chain"].get()),
            resseq=_optional_int(vars_dict["resseq"].get(), "ResNumber"),
            icode=None,
            resname=None,
            scale_mode=vars_dict["scale_mode"].get().strip() or "none",
            path_type=vars_dict["path_type"].get().strip() or "open",
            scale_anchor=vars_dict["scale_anchor"].get().strip() or "centroid",
            path_start=float(vars_dict["path_start"].get().strip() or "0"),
            twist=float(vars_dict["twist"].get().strip() or "0"),
            curve_components=_optional_text(vars_dict["curve_components"].get()),
            interp_mode=vars_dict["interp_mode"].get().strip() or "none",
            interp_n=int(vars_dict["interp_n"].get().strip() or "200"),
            interp_p=int(vars_dict["interp_p"].get().strip() or "0"),
            target_mode=vars_dict["target_mode"].get().strip() or "curvature_angle",
            curvature_angle_deg=float(vars_dict["curvature_angle_deg"].get().strip() or "0"),
            normal_angle_deg=None,
            curvature_step=float(vars_dict["curvature_step"].get().strip() or "0"),
            axis_point=axis_point,
            axis_dir=axis_dir,
            target_point=target_point,
            custom_vector=custom_vector,
            side=None,
        )

    def on_compute():
        try:
            gui_args = collect_args()
            report = compute_phase(gui_args)
            phase_deg = extract_phase_deg(report)
            out_path = gui_args.output if gui_args.output else default_output_path(gui_args.input_pdb)
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(report)
            report_box.delete("1.0", tk.END)
            report_box.insert(tk.END, report)
            report_box.insert(tk.END, "\nWrote report: {}\n".format(out_path))
            if on_phase is not None:
                on_phase(phase_deg, report, out_path)
                report_box.insert(tk.END, "Transferred phase to Curve It: {:.6f} deg\n".format(phase_deg))
                message = (
                    "Curve It phase report written:\n{}\n\n"
                    "Transferred phase to Curve It: {:.6f} deg"
                ).format(out_path, phase_deg)
            else:
                message = "Curve It phase report written:\n{}".format(out_path)
            messagebox.showinfo("Done", message, parent=root)
        except (Exception, SystemExit) as exc:
            messagebox.showerror("Error", str(exc), parent=root)

    button_frame = tk.Frame(root)
    button_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
    tk.Button(button_frame, text="Compute phase", command=on_compute).pack(side="left")
    tk.Button(button_frame, text="Close" if parent is not None else "Quit", command=root.destroy).pack(side="right")

    help_text = (
        "curvature_angle: 0=local curvature normal, 90=local curvature binormal. "
        "toward_axis/toward_point/custom_vector are projected into the local cross-section."
    )
    tk.Label(button_frame, text=help_text, anchor="w").pack(side="left", padx=12)

    if owns_mainloop:
        root.mainloop()
    return root


def main() -> None:
    """Parse arguments, run GUI or CLI, and write the Curve It phase report."""
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        run_gui(None)
        return
    args = parser.parse_args()
    if args.gui:
        run_gui(args)
        return
    if not args.input_pdb or not args.curve_xyz:
        parser.error("input_pdb and curve_xyz are required in CLI mode. Use --gui or no arguments for GUI mode.")

    try:
        report = compute_phase(args)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit("Error: {}".format(exc))

    out_path = args.output if args.output else default_output_path(args.input_pdb)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report.rstrip())
    print("\nWrote report: {}".format(out_path))


if __name__ == "__main__":
    main()
