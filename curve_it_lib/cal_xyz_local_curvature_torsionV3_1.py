#!/usr/bin/env python3
"""
cal_xyz_local_curvature_torsionV3_1.py

Compute local curvature, regularized local Frenet torsion, and local writhe density
along an open or closed 3D curve stored in a plain coordinate file or
molecular XYZ file.

Inputs:
  - Plain coordinate file: x y z, one point per line
  - Molecular XYZ file: atom_count, comment line, then atom_symbol x y z

Outputs:
  - CSV table with normalized position from 0 to 1, coordinates, local
    curvature, local torsion, local writhe density, and diagnostic columns
  - Optional pop-up Matplotlib plot of curvature, torsion, and local writhe
    density versus normalized position

Example commands:
  python curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py
  python curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py --gui
  python curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py tk_200.txt
  python curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py tk_200.txt --open --out tk_200_local.csv
  python curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py tk_200.txt --no-smooth --nsamples 1000
  python curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py --example-trefoil --no-plot

Notes:
  - The curve is treated as closed by default.
  - No command-line arguments, or --gui, launches a Tkinter GUI.
  - Torsion is reported as a regularized Frenet torsion estimated from
    signed binormal rotation. The raw third-derivative torsion is also saved
    for diagnosis. Its integral divided by 2*pi is not generally equal to writhe.
  - Local writhe density is a distribution of the Gauss writhe double integral
    along the curve. It is local only as an assigned density; each value still
    depends on all other sampled curve positions.
"""

import argparse
import csv
import math
import os
import sys
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.signal import savgol_filter

try:
    # Reuse the parser and closed-curve smoothing behavior from the companion
    # Curve It helper when this file is placed in curve_it_lib/.
    try:
        from .cal_xyz_total_curvature_writheV2 import read_xyz_like_raw, smooth_closed_points
    except ImportError:
        from cal_xyz_total_curvature_writheV2 import read_xyz_like_raw, smooth_closed_points
except ImportError:
    # Fallback copies so this script can still run as a standalone small tool.
    def token_to_float(token: str) -> Optional[float]:
        """Return float(token), or None if token is not a plain numeric token."""
        try:
            return float(token)
        except ValueError:
            return None

    def parse_coordinate_line(line: str) -> Optional[List[float]]:
        """
        Parse one coordinate line.

        The line may be:
          x y z
          C x y z
          C x y z extra ...

        Any non-numeric tokens are ignored. The first three numeric values are
        returned. If fewer than three numeric values are present, return None.
        """
        values = []
        for token in line.split():
            value = token_to_float(token)
            if value is not None:
                values.append(value)
                if len(values) == 3:
                    return values
        return None

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

    def read_xyz_like_raw(filename: str, file_format: str = "auto") -> np.ndarray:
        """
        Read raw 3D points from a plain coordinate file or molecular XYZ file.

        file_format:
          auto      detect molecular XYZ if the first non-empty line is an atom count
          plain     treat all lines as potential x y z coordinate lines
          molecule  skip the first two lines as standard XYZ header
        """
        with open(filename, "r") as f:
            raw_lines = f.readlines()

        if not raw_lines:
            raise ValueError("The file is empty: {}".format(filename))

        lines = [line.rstrip("\n") for line in raw_lines]
        nonempty_indices = [i for i, line in enumerate(lines) if line.strip()]
        if not nonempty_indices:
            raise ValueError("The file contains no readable lines: {}".format(filename))

        start_index = nonempty_indices[0]

        if file_format not in ("auto", "plain", "molecule"):
            raise ValueError("file_format must be auto, plain, or molecule.")

        if file_format == "molecule":
            data_start = start_index + 2
        elif file_format == "plain":
            data_start = start_index
        else:
            if first_token_is_integer(lines[start_index]) and len(lines) >= start_index + 3:
                data_start = start_index + 2
            else:
                data_start = start_index

        coords = []
        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("!"):
                continue
            point = parse_coordinate_line(stripped)
            if point is not None:
                coords.append(point)

        if not coords:
            raise ValueError("No 3D points could be read from file: {}".format(filename))

        return np.asarray(coords, dtype=float)

    def smooth_closed_points(points: np.ndarray) -> np.ndarray:
        """
        Smooth points using a Savitzky-Golay filter with periodic extension.

        This mirrors the default behavior in cal_xyz_total_curvature_writheV2.py.
        """
        n_points = len(points)
        if n_points < 7:
            return points

        smooth_window_size = max(7, (n_points // 18) * 2 + 1)
        polyorder = 3
        if smooth_window_size <= polyorder:
            if (polyorder + 2) % 2 == 1:
                smooth_window_size = polyorder + 2
            else:
                smooth_window_size = polyorder + 3

        half_window = smooth_window_size // 2
        points_extended = np.concatenate((points[-half_window:], points, points[:half_window]))

        if smooth_window_size >= len(points_extended):
            smooth_window_size = len(points_extended) - 1
            if smooth_window_size % 2 == 0:
                smooth_window_size -= 1

        if smooth_window_size <= polyorder:
            return points

        smoothed_points = savgol_filter(
            points_extended,
            window_length=smooth_window_size,
            polyorder=polyorder,
            axis=0,
        )[half_window:-half_window]

        return smoothed_points


def strip_duplicate_endpoint(points: np.ndarray, closed: bool, tol: float = 1e-8) -> np.ndarray:
    """
    Remove a duplicated final point for closed curves.

    Periodic splines append the initial point internally. Removing an input
    duplicate avoids double-counting the same geometric point during smoothing
    and spline construction.
    """
    points = np.asarray(points, dtype=float)
    if closed and len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= tol:
        return points[:-1].copy()
    return points


def default_savgol_window(n_points: int, polyorder: int) -> int:
    """
    Return the default odd Savitzky-Golay window used by the V2 helper script.
    """
    if n_points < 7:
        return 0
    window = max(7, (n_points // 18) * 2 + 1)
    if window <= polyorder:
        if (polyorder + 2) % 2 == 1:
            window = polyorder + 2
        else:
            window = polyorder + 3
    if window > n_points:
        window = n_points if n_points % 2 == 1 else n_points - 1
    if window <= polyorder or window < 3:
        return 0
    return window


def validate_savgol_window(window: int, n_points: int, polyorder: int) -> int:
    """
    Validate a user-provided Savitzky-Golay window.
    """
    if window < 3:
        raise ValueError("--smooth-window must be at least 3.")
    if window % 2 == 0:
        raise ValueError("--smooth-window must be odd for scipy.signal.savgol_filter.")
    if window <= polyorder:
        raise ValueError("--smooth-window must be larger than --polyorder.")
    if window > n_points:
        raise ValueError("--smooth-window cannot exceed the number of points.")
    return window


def smooth_open_points(points: np.ndarray, window: Optional[int] = None, polyorder: int = 3) -> np.ndarray:
    """
    Smooth an open curve using Savitzky-Golay filtering.
    """
    n_points = len(points)
    if window is None:
        window = default_savgol_window(n_points, polyorder)
    else:
        window = validate_savgol_window(window, n_points, polyorder)

    if window == 0:
        return points

    return savgol_filter(points, window_length=window, polyorder=polyorder, axis=0, mode="interp")


def smooth_points(
    points: np.ndarray,
    closed: bool,
    smooth: bool,
    window: Optional[int] = None,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Smooth points for open or closed curves.
    """
    if not smooth:
        return points

    if closed:
        if window is None and polyorder == 3:
            # Keep the exact default behavior from cal_xyz_total_curvature_writheV2.py.
            return smooth_closed_points(points)

        # User-overridden closed smoothing, with the same periodic extension idea.
        n_points = len(points)
        if window is None:
            window = default_savgol_window(n_points, polyorder)
        else:
            window = validate_savgol_window(window, n_points, polyorder)
        if window == 0:
            return points
        half_window = window // 2
        points_extended = np.concatenate((points[-half_window:], points, points[:half_window]))
        return savgol_filter(
            points_extended,
            window_length=window,
            polyorder=polyorder,
            axis=0,
            mode="interp",
        )[half_window:-half_window]

    return smooth_open_points(points, window=window, polyorder=polyorder)


def chord_length_parameter(points: np.ndarray, closed: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return spline points and a normalized chord-length parameter from 0 to 1.
    """
    points = np.asarray(points, dtype=float)

    if closed:
        spline_points = np.vstack([points, points[0]])
    else:
        spline_points = points.copy()

    segment_lengths = np.linalg.norm(np.diff(spline_points, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 0.0:
        raise ValueError("The curve has zero total chord length.")

    u_values = np.empty(len(spline_points), dtype=float)
    u_values[0] = 0.0
    u_values[1:] = np.cumsum(segment_lengths) / total_length
    u_values[-1] = 1.0

    # CubicSpline requires strictly increasing x values. Remove zero-length
    # duplicate interior points if present. For closed curves, keep u=0 and u=1.
    keep = np.ones(len(u_values), dtype=bool)
    if len(u_values) > 2:
        keep[1:-1] = np.diff(u_values[:-1]) > 1e-14
    keep[-1] = True
    spline_points = spline_points[keep]
    u_values = u_values[keep]

    if len(spline_points) < 4:
        raise ValueError("At least 4 non-duplicated points are needed for spline derivatives.")

    return spline_points, u_values


def build_splines(points: np.ndarray, closed: bool) -> Tuple[CubicSpline, CubicSpline, CubicSpline]:
    """
    Build cubic splines x(u), y(u), z(u), where u is normalized chord length.
    """
    spline_points, u_values = chord_length_parameter(points, closed=closed)
    bc_type = "periodic" if closed else "not-a-knot"

    spline_x = CubicSpline(u_values, spline_points[:, 0], bc_type=bc_type)
    spline_y = CubicSpline(u_values, spline_points[:, 1], bc_type=bc_type)
    spline_z = CubicSpline(u_values, spline_points[:, 2], bc_type=bc_type)
    return spline_x, spline_y, spline_z


def evaluate_local_geometry(
    spline_x: CubicSpline,
    spline_y: CubicSpline,
    spline_z: CubicSpline,
    u_values: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Evaluate coordinates, curvature, and torsion at normalized positions.

    Formulas are valid for an arbitrary parameter u:
      curvature = |r'(u) x r''(u)| / |r'(u)|^3
      torsion   = det(r'(u), r''(u), r'''(u)) / |r'(u) x r''(u)|^2
    """
    x = spline_x(u_values)
    y = spline_y(u_values)
    z = spline_z(u_values)

    r1 = np.column_stack((spline_x(u_values, 1), spline_y(u_values, 1), spline_z(u_values, 1)))
    r2 = np.column_stack((spline_x(u_values, 2), spline_y(u_values, 2), spline_z(u_values, 2)))
    r3 = np.column_stack((spline_x(u_values, 3), spline_y(u_values, 3), spline_z(u_values, 3)))

    speed = np.linalg.norm(r1, axis=1)
    cross12 = np.cross(r1, r2)
    cross_norm = np.linalg.norm(cross12, axis=1)

    curvature = np.full(len(u_values), np.nan, dtype=float)
    torsion = np.full(len(u_values), np.nan, dtype=float)

    good_speed = speed > eps
    curvature[good_speed] = cross_norm[good_speed] / (speed[good_speed] ** 3)

    good_torsion = cross_norm > eps
    determinant = np.einsum("ij,ij->i", cross12, r3)
    torsion[good_torsion] = determinant[good_torsion] / (cross_norm[good_torsion] ** 2)

    ds_du = speed
    curvature_integrand = curvature * ds_du
    torsion_integrand = torsion * ds_du

    tangent = np.full_like(r1, np.nan, dtype=float)
    tangent[good_speed] = r1[good_speed] / speed[good_speed, None]

    return {
        "u": u_values,
        "x": x,
        "y": y,
        "z": z,
        "curvature": curvature,
        # These two columns will be overwritten by the binormal-rotation
        # estimate in add_regularized_torsion_from_binormal_rotation().
        # The third-derivative values are retained as torsion_raw.
        "torsion": torsion.copy(),
        "torsion_raw": torsion,
        "ds_du": ds_du,
        "curvature_integrand": curvature_integrand,
        "torsion_integrand": torsion_integrand.copy(),
        "torsion_raw_integrand": torsion_integrand,
        "torsion_reliable": np.isfinite(torsion).astype(float),
        "tangent_x": tangent[:, 0],
        "tangent_y": tangent[:, 1],
        "tangent_z": tangent[:, 2],
    }


def add_regularized_torsion_from_binormal_rotation(
    data: Dict[str, np.ndarray],
    spline_x: CubicSpline,
    spline_y: CubicSpline,
    spline_z: CubicSpline,
    closed: bool,
    rel_curvature_cutoff: float = 0.05,
    min_curvature: Optional[float] = None,
    eps: float = 1e-12,
) -> None:
    '''Replace data["torsion"] with a regularized torsion estimate.'''
    u_values = np.asarray(data["u"], dtype=float)
    n_values = len(u_values)
    if n_values < 2:
        return

    r1 = np.column_stack((spline_x(u_values, 1), spline_y(u_values, 1), spline_z(u_values, 1)))
    r2 = np.column_stack((spline_x(u_values, 2), spline_y(u_values, 2), spline_z(u_values, 2)))

    speed = np.linalg.norm(r1, axis=1)
    cross12 = np.cross(r1, r2)
    cross_norm = np.linalg.norm(cross12, axis=1)
    curvature = np.asarray(data["curvature"], dtype=float)

    finite_positive_curv = curvature[np.isfinite(curvature) & (curvature > 0.0)]
    if min_curvature is None:
        if len(finite_positive_curv) == 0:
            curvature_cutoff = 0.0
        else:
            curvature_cutoff = float(rel_curvature_cutoff) * float(np.median(finite_positive_curv))
    else:
        curvature_cutoff = float(min_curvature)

    reliable = (
        np.isfinite(speed)
        & np.isfinite(cross_norm)
        & np.isfinite(curvature)
        & (speed > eps)
        & (cross_norm > eps)
        & (curvature >= curvature_cutoff)
    )

    tangent = np.full_like(r1, np.nan, dtype=float)
    binormal = np.full_like(r1, np.nan, dtype=float)
    tangent[reliable] = r1[reliable] / speed[reliable, None]
    binormal[reliable] = cross12[reliable] / cross_norm[reliable, None]

    row_torsion_sum = np.zeros(n_values, dtype=float)
    row_weight_sum = np.zeros(n_values, dtype=float)
    segment_torsion_angles = []

    # The table includes both u=0 and u=1 for closed curves, so n-1 intervals
    # already cover the whole curve, including the closing interval.
    for i in range(n_values - 1):
        j = i + 1
        du = float(u_values[j] - u_values[i])
        if du <= 0.0 or not (reliable[i] and reliable[j]):
            continue

        ds = 0.5 * float(data["ds_du"][i] + data["ds_du"][j]) * du
        if ds <= eps or not math.isfinite(ds):
            continue

        t_mid = tangent[i] + tangent[j]
        t_mid_norm = float(np.linalg.norm(t_mid))
        if t_mid_norm <= eps or not math.isfinite(t_mid_norm):
            t_mid = tangent[i]
            t_mid_norm = float(np.linalg.norm(t_mid))
            if t_mid_norm <= eps or not math.isfinite(t_mid_norm):
                continue
        t_mid = t_mid / t_mid_norm

        dot_bb = float(np.clip(np.dot(binormal[i], binormal[j]), -1.0, 1.0))
        signed_sin = float(np.dot(t_mid, np.cross(binormal[i], binormal[j])))
        angle = math.atan2(signed_sin, dot_bb)
        if not math.isfinite(angle):
            continue

        tau_segment = angle / ds
        segment_torsion_angles.append(angle)
        row_torsion_sum[i] += tau_segment * ds
        row_torsion_sum[j] += tau_segment * ds
        row_weight_sum[i] += ds
        row_weight_sum[j] += ds

    torsion_regularized = np.full(n_values, np.nan, dtype=float)
    has_value = row_weight_sum > 0.0
    torsion_regularized[has_value] = row_torsion_sum[has_value] / row_weight_sum[has_value]

    if closed and n_values > 2 and has_value[0] and has_value[-1]:
        endpoint_value = 0.5 * (torsion_regularized[0] + torsion_regularized[-1])
        torsion_regularized[0] = endpoint_value
        torsion_regularized[-1] = endpoint_value

    data["torsion"] = torsion_regularized
    data["torsion_integrand"] = torsion_regularized * np.asarray(data["ds_du"], dtype=float)
    data["torsion_reliable"] = reliable.astype(float)
    data["_torsion_curvature_cutoff"] = curvature_cutoff
    data["_torsion_reliable_fraction"] = float(np.count_nonzero(reliable)) / float(n_values)
    data["_total_torsion_angle_binormal"] = float(np.sum(segment_torsion_angles)) if segment_torsion_angles else float("nan")


def evaluate_integrands_at_u(
    spline_x: CubicSpline,
    spline_y: CubicSpline,
    spline_z: CubicSpline,
    u: float,
    eps_speed: float = 1e-12,
    eps_cross: float = 1e-10,
) -> Tuple[float, float]:
    """
    Return curvature*ds/du and torsion*ds/du at one normalized position.

    This is used for adaptive quadrature summaries. Torsion is set to zero at
    positions where the Frenet frame is numerically ill-defined because the
    speed or curvature is too small.
    """
    r1 = np.array([spline_x(u, 1), spline_y(u, 1), spline_z(u, 1)], dtype=float)
    r2 = np.array([spline_x(u, 2), spline_y(u, 2), spline_z(u, 2)], dtype=float)
    r3 = np.array([spline_x(u, 3), spline_y(u, 3), spline_z(u, 3)], dtype=float)

    speed = float(np.linalg.norm(r1))
    if speed <= eps_speed or not math.isfinite(speed):
        return 0.0, 0.0

    cross12 = np.cross(r1, r2)
    cross_norm = float(np.linalg.norm(cross12))
    curvature_integrand = cross_norm / (speed ** 2)

    if cross_norm <= eps_cross or not math.isfinite(cross_norm):
        torsion_integrand = 0.0
    else:
        torsion = float(np.dot(cross12, r3) / (cross_norm ** 2))
        torsion_integrand = torsion * speed

    return curvature_integrand, torsion_integrand


def integrate_curvature_torsion_adaptive(
    spline_x: CubicSpline,
    spline_y: CubicSpline,
    spline_z: CubicSpline,
) -> Tuple[float, float]:
    """
    Integrate curvature*ds and torsion*ds by adaptive quadrature.

    Cubic splines have piecewise polynomial derivatives. Splitting at spline
    knots avoids integrating across third-derivative discontinuities and gives
    a more stable total torsion than a simple trapezoidal sum over sampled rows.
    """
    knots = np.asarray(spline_x.x, dtype=float)
    total_curvature = 0.0
    total_torsion_angle = 0.0

    for a, b in zip(knots[:-1], knots[1:]):
        if b <= a:
            continue

        curv_piece, _curv_err = quad(
            lambda uu: evaluate_integrands_at_u(spline_x, spline_y, spline_z, uu)[0],
            float(a),
            float(b),
            limit=200,
            epsabs=1e-8,
            epsrel=1e-6,
        )
        tors_piece, _tors_err = quad(
            lambda uu: evaluate_integrands_at_u(spline_x, spline_y, spline_z, uu)[1],
            float(a),
            float(b),
            limit=200,
            epsabs=1e-8,
            epsrel=1e-6,
        )
        total_curvature += float(curv_piece)
        total_torsion_angle += float(tors_piece)

    return total_curvature, total_torsion_angle


def trapezoid_weights(u_values: np.ndarray) -> np.ndarray:
    """
    Return trapezoidal integration weights for an increasing 1D grid.
    """
    u_values = np.asarray(u_values, dtype=float)
    n_values = len(u_values)
    if n_values < 2:
        raise ValueError("At least two sample positions are needed for integration weights.")

    weights = np.empty(n_values, dtype=float)
    weights[0] = 0.5 * (u_values[1] - u_values[0])
    weights[-1] = 0.5 * (u_values[-1] - u_values[-2])
    if n_values > 2:
        weights[1:-1] = 0.5 * (u_values[2:] - u_values[:-2])
    return weights


def add_local_writhe_density_from_tangents(
    data: Dict[str, np.ndarray],
    tangent: np.ndarray,
    eps: float = 1e-12,
) -> None:
    """
    Add a local writhe-density estimate to the data dictionary in place.

    The standard Gauss writhe is a double integral. This function distributes
    that double-integral contribution along the curve by defining

        local_writhe_density(s) = (1 / 4*pi) * integral K(s, s_prime) ds_prime

    where K is the Gauss writhe kernel. This is a density per unit arclength.
    Multiplying by ds/du gives local_writhe_times_dsdu, and integrating that
    column over u gives the approximate writhe.

    The calculation is O(N^2) in the number of sampled table rows.
    """
    u_values = np.asarray(data["u"], dtype=float)
    r = np.column_stack((data["x"], data["y"], data["z"]))
    ds_du = np.asarray(data["ds_du"], dtype=float)
    tangent = np.asarray(tangent, dtype=float)

    n_values = len(u_values)
    if n_values < 2:
        raise ValueError("At least two samples are needed for local writhe density.")
    if tangent.shape != r.shape:
        raise ValueError("tangent must have the same N x 3 shape as the coordinates.")

    weights_u = trapezoid_weights(u_values)
    ds_weights = ds_du * weights_u

    R = r[:, None, :] - r[None, :, :]
    dist = np.linalg.norm(R, axis=2)

    Ti = tangent[:, None, :]
    Tj = tangent[None, :, :]
    cross_tangent = np.cross(Ti, Tj)
    numerator = np.einsum("ijk,ijk->ij", R, cross_tangent)

    valid = dist > eps
    valid &= np.isfinite(dist)
    valid &= np.isfinite(numerator)
    valid &= np.isfinite(ds_weights)[None, :]

    kernel = np.zeros((n_values, n_values), dtype=float)
    kernel[valid] = numerator[valid] / (dist[valid] ** 3)

    local_density = (1.0 / (4.0 * math.pi)) * np.sum(kernel * ds_weights[None, :], axis=1)
    local_integrand = local_density * ds_du

    data["local_writhe_density"] = local_density
    data["local_writhe_integrand"] = local_integrand


def output_default_name(input_path: str) -> str:
    """
    Return default output path by adding _local_curvature_torsion_writhe before extension.
    """
    root, _ext = os.path.splitext(input_path)
    return root + "_local_curvature_torsion_writhe.csv"


def generate_trefoil_points(n_points: int = 300) -> np.ndarray:
    """
    Generate an equation-defined three-lobe trefoil, the (2,3) torus knot.

        x(t) = (2 + cos(3t)) cos(2t)
        y(t) = (2 + cos(3t)) sin(2t)
        z(t) = sin(3t), 0 <= t < 2*pi
    """
    if n_points < 12:
        raise ValueError("Trefoil example needs at least 12 points.")
    t = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False)
    x = (2.0 + np.cos(3.0 * t)) * np.cos(2.0 * t)
    y = (2.0 + np.cos(3.0 * t)) * np.sin(2.0 * t)
    z = np.sin(3.0 * t)
    return np.column_stack((x, y, z))


def write_trefoil_example(path: Optional[str] = None, n_points: int = 300) -> str:
    """Write the built-in trefoil example as plain x y z coordinates."""
    if path is None:
        example_dir = os.path.join(tempfile.gettempdir(), "curve_it_examples")
        os.makedirs(example_dir, exist_ok=True)
        path = os.path.join(example_dir, "trefoil_2_3_torus_knot.xyz")
    points = generate_trefoil_points(n_points=n_points)
    with open(path, "w") as f:
        f.write("# Three-lobe trefoil knot, the (2,3) torus knot\n")
        f.write("# x=(2+cos(3t))*cos(2t), y=(2+cos(3t))*sin(2t), z=sin(3t)\n")
        for point in points:
            f.write("{:.10f} {:.10f} {:.10f}\n".format(point[0], point[1], point[2]))
    return path


def write_csv_table(data: Dict[str, np.ndarray], output_path: str) -> None:
    """
    Write local curvature/torsion data to a CSV table.
    """
    fieldnames = [
        "u_0_to_1",
        "x",
        "y",
        "z",
        "curvature",
        "torsion",
        "torsion_raw",
        "torsion_reliable",
        "ds_du",
        "curvature_times_dsdu",
        "torsion_times_dsdu",
        "torsion_raw_times_dsdu",
        "local_writhe_density",
        "local_writhe_times_dsdu",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(data["u"])):
            writer.writerow({
                "u_0_to_1": "{:.10g}".format(data["u"][i]),
                "x": "{:.10g}".format(data["x"][i]),
                "y": "{:.10g}".format(data["y"][i]),
                "z": "{:.10g}".format(data["z"][i]),
                "curvature": "{:.10g}".format(data["curvature"][i]),
                "torsion": "{:.10g}".format(data["torsion"][i]),
                "torsion_raw": "{:.10g}".format(data.get("torsion_raw", np.full(len(data["u"]), np.nan))[i]),
                "torsion_reliable": "{:.0f}".format(data.get("torsion_reliable", np.full(len(data["u"]), np.nan))[i]),
                "ds_du": "{:.10g}".format(data["ds_du"][i]),
                "curvature_times_dsdu": "{:.10g}".format(data["curvature_integrand"][i]),
                "torsion_times_dsdu": "{:.10g}".format(data["torsion_integrand"][i]),
                "torsion_raw_times_dsdu": "{:.10g}".format(data.get("torsion_raw_integrand", np.full(len(data["u"]), np.nan))[i]),
                "local_writhe_density": "{:.10g}".format(data.get("local_writhe_density", np.full(len(data["u"]), np.nan))[i]),
                "local_writhe_times_dsdu": "{:.10g}".format(data.get("local_writhe_integrand", np.full(len(data["u"]), np.nan))[i]),
            })


def finite_trapz(y_values: np.ndarray, x_values: np.ndarray) -> float:
    """
    Integrate finite values by trapezoidal rule, ignoring NaNs/Infs.
    """
    finite = np.isfinite(y_values) & np.isfinite(x_values)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values[finite], x_values[finite]))
    return float(np.trapz(y_values[finite], x_values[finite]))


def summarize_geometry(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Return summary values from sampled local geometry.
    """
    sampled_total_curvature = finite_trapz(data["curvature_integrand"], data["u"])
    sampled_total_torsion_angle = finite_trapz(data["torsion_integrand"], data["u"])
    sampled_raw_torsion_angle = finite_trapz(data.get("torsion_raw_integrand", data["torsion_integrand"]), data["u"])

    total_curvature = float(data.get("_total_curvature_adaptive", sampled_total_curvature))
    total_torsion_angle = float(data.get("_total_torsion_angle_binormal", sampled_total_torsion_angle))
    torsion_turns = total_torsion_angle / (2.0 * math.pi) if math.isfinite(total_torsion_angle) else float("nan")
    approximate_writhe = finite_trapz(data["local_writhe_integrand"], data["u"]) if "local_writhe_integrand" in data else float("nan")
    return {
        "total_curvature": total_curvature,
        "total_curvature_over_pi": total_curvature / math.pi if math.isfinite(total_curvature) else float("nan"),
        "total_torsion_angle": total_torsion_angle,
        "torsion_turns": torsion_turns,
        "sampled_total_curvature": sampled_total_curvature,
        "sampled_total_torsion_angle": sampled_total_torsion_angle,
        "sampled_torsion_turns": sampled_total_torsion_angle / (2.0 * math.pi) if math.isfinite(sampled_total_torsion_angle) else float("nan"),
        "sampled_raw_torsion_angle": sampled_raw_torsion_angle,
        "sampled_raw_torsion_turns": sampled_raw_torsion_angle / (2.0 * math.pi) if math.isfinite(sampled_raw_torsion_angle) else float("nan"),
        "approximate_writhe_from_density": approximate_writhe,
        "torsion_curvature_cutoff": float(data.get("_torsion_curvature_cutoff", float("nan"))),
        "torsion_reliable_fraction": float(data.get("_torsion_reliable_fraction", float("nan"))),
    }


def format_summary(
    data: Dict[str, np.ndarray],
    output_path: str,
    closed: bool,
    smoothed: bool,
) -> str:
    """
    Return a concise numerical summary as text.
    """
    summary = summarize_geometry(data)
    lines = [
        "Output table:",
        "  {}".format(output_path),
        "",
        "Summary:",
        "  Curve type: {}".format("closed" if closed else "open"),
        "  Smoothing: {}".format("on" if smoothed else "off"),
        "  Number of table rows: {}".format(len(data["u"])),
        "  Approx. total curvature = {:.10g}".format(summary["total_curvature"]),
        "  Approx. total curvature / pi = {:.10g}".format(summary["total_curvature_over_pi"]),
        "  Approx. integral torsion ds = {:.10g} rad".format(summary["total_torsion_angle"]),
        "  Approx. torsion turns = {:.10g}".format(summary["torsion_turns"]),
        "  Approx. writhe from local density = {:.10g}".format(summary["approximate_writhe_from_density"]),
        "  Sampled-row torsion turns = {:.10g}".format(summary["sampled_torsion_turns"]),
        "  Raw third-derivative sampled torsion turns = {:.10g}".format(summary["sampled_raw_torsion_turns"]),
        "  Torsion curvature cutoff = {:.10g}".format(summary["torsion_curvature_cutoff"]),
        "  Reliable torsion rows = {:.1f}%".format(100.0 * summary["torsion_reliable_fraction"]),
        "",
        "Note: total torsion uses signed binormal rotation after masking very low-curvature rows.",
        "Note: torsion_raw is also saved because it is the direct third-derivative formula.",
        "Note: raw torsion can show artificial spikes where local curvature is very small.",
        "Note: local writhe density is assigned locally but depends on the whole curve.",
        "Note: torsion turns are not generally equal to writhe.",
    ]
    return "\n".join(lines)


def print_summary(data: Dict[str, np.ndarray], output_path: str, closed: bool, smoothed: bool) -> None:
    """
    Print a concise numerical summary.
    """
    print("\n" + format_summary(data, output_path, closed=closed, smoothed=smoothed))


def plot_local_geometry(data: Dict[str, np.ndarray], title: str) -> None:
    """
    Plot curvature, torsion, and local writhe density in a pop-up window.
    """
    import matplotlib.pyplot as plt

    has_writhe_density = "local_writhe_density" in data and np.any(np.isfinite(data["local_writhe_density"]))
    if has_writhe_density:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    else:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    axes[0].plot(data["u"], data["curvature"])
    axes[0].set_ylabel("curvature")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data["u"], data["torsion"])
    axes[1].set_ylabel("torsion")
    finite_torsion = data["torsion"][np.isfinite(data["torsion"])]
    if len(finite_torsion) >= 10:
        lo, hi = np.percentile(finite_torsion, [1, 99])
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            pad = 0.10 * (hi - lo)
            axes[1].set_ylim(lo - pad, hi + pad)
    axes[1].grid(True, alpha=0.3)

    if has_writhe_density:
        axes[2].plot(data["u"], data["local_writhe_density"])
        axes[2].set_ylabel("local writhe density")
        axes[2].set_xlabel("normalized position along curve, 0=start, 1=end")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[1].set_xlabel("normalized position along curve, 0=start, 1=end")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def run_calculation(
    xyz_file: str,
    file_format: str = "auto",
    closed: bool = True,
    nsamples: int = 500,
    output_path: Optional[str] = None,
    smooth: bool = True,
    smooth_window: Optional[int] = None,
    polyorder: int = 3,
    show_plot: bool = True,
    calculate_writhe_density: bool = True,
    torsion_rel_curvature_cutoff: float = 0.05,
    torsion_min_curvature: Optional[float] = None,
    log_func: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, np.ndarray], str, str]:
    """
    Run the full calculation and return data, output path, and summary text.
    """
    if log_func is None:
        def log_func(message: str) -> None:
            print(message)

    filename = xyz_file.strip().strip('"').strip("'")
    if not os.path.isfile(filename):
        raise FileNotFoundError("File not found: {}".format(filename))
    if file_format not in ("auto", "plain", "molecule"):
        raise ValueError("file_format must be auto, plain, or molecule.")
    if nsamples < 4:
        raise ValueError("nsamples must be at least 4.")
    if polyorder < 0:
        raise ValueError("polyorder must be non-negative.")

    output_path = output_path.strip() if output_path else output_default_name(filename)

    log_func("[INFO] Reading points from: {}".format(filename))
    log_func("[INFO] Input format: {}".format(file_format))
    log_func("[INFO] Curve type: {}".format("closed" if closed else "open"))
    log_func("[INFO] Smoothing: {}".format("on" if smooth else "off"))

    points = read_xyz_like_raw(filename, file_format=file_format)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an N x 3 coordinate array.")
    if len(points) < 4:
        raise ValueError("At least 4 points are needed.")

    log_func("[INFO] Loaded {} raw points.".format(len(points)))
    points = strip_duplicate_endpoint(points, closed=closed)
    points = smooth_points(
        points,
        closed=closed,
        smooth=smooth,
        window=smooth_window,
        polyorder=polyorder,
    )
    log_func("[INFO] Using {} points after duplicate-endpoint handling/smoothing.".format(len(points)))

    spline_x, spline_y, spline_z = build_splines(points, closed=closed)
    u_values = np.linspace(0.0, 1.0, nsamples)
    data = evaluate_local_geometry(spline_x, spline_y, spline_z, u_values)
    total_curvature_quad, total_torsion_angle_quad = integrate_curvature_torsion_adaptive(
        spline_x, spline_y, spline_z
    )
    data["_total_curvature_adaptive"] = total_curvature_quad
    data["_total_torsion_angle_adaptive_raw"] = total_torsion_angle_quad

    add_regularized_torsion_from_binormal_rotation(
        data,
        spline_x,
        spline_y,
        spline_z,
        closed=closed,
        rel_curvature_cutoff=torsion_rel_curvature_cutoff,
        min_curvature=torsion_min_curvature,
    )

    if calculate_writhe_density:
        log_func("[INFO] Calculating local writhe density using an O(N^2) Gauss-integral estimate.")
        tangent = np.column_stack((data["tangent_x"], data["tangent_y"], data["tangent_z"]))
        add_local_writhe_density_from_tangents(data, tangent=tangent)
    else:
        data["local_writhe_density"] = np.full(len(data["u"]), np.nan)
        data["local_writhe_integrand"] = np.full(len(data["u"]), np.nan)

    write_csv_table(data, output_path)
    summary_text = format_summary(data, output_path, closed=closed, smoothed=smooth)
    log_func("\n" + summary_text)

    if show_plot:
        plot_title = os.path.basename(filename)
        plot_local_geometry(data, plot_title)

    return data, output_path, summary_text


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate local curvature, torsion, and local writhe density along a "
            "3D curve from a plain coordinate file or molecular XYZ file. The curve is treated as closed "
            "by default. With no arguments, or with --gui, a Tkinter GUI opens."
        )
    )
    parser.add_argument(
        "xyz_file",
        nargs="?",
        help="Path to the coordinate/XYZ file containing x y z points.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI. This is also the default when no xyz_file is given.",
    )
    parser.add_argument(
        "--example-trefoil",
        action="store_true",
        help="Generate and analyze the built-in three-lobe trefoil example, a (2,3) torus knot.",
    )
    parser.add_argument(
        "--example-points",
        type=int,
        default=300,
        help="Number of points for --example-trefoil. Default: 300.",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "plain", "molecule"],
        default="auto",
        help="Input format. Default: auto.",
    )
    parser.add_argument(
        "-m",
        "--molecule",
        action="store_true",
        help="Shortcut for --format molecule.",
    )

    curve_group = parser.add_mutually_exclusive_group()
    curve_group.add_argument(
        "--closed",
        action="store_true",
        help="Treat the curve as closed. This is the default.",
    )
    curve_group.add_argument(
        "--open",
        action="store_true",
        help="Treat the curve as open.",
    )

    parser.add_argument(
        "--nsamples",
        type=int,
        default=500,
        help="Number of local positions in the output table. Default: 500.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: input name with _local_curvature_torsion_writhe.csv.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable Savitzky-Golay smoothing and use raw input points.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="Optional odd Savitzky-Golay smoothing window. Default follows the V2 helper script.",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order. Default: 3.",
    )
    parser.add_argument(
        "--torsion-rel-curvature-cutoff",
        type=float,
        default=0.05,
        help=(
            "Relative curvature cutoff for regularized torsion. Rows with curvature below "
            "this fraction of the median positive curvature are marked unreliable. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--torsion-min-curvature",
        type=float,
        default=None,
        help="Absolute curvature cutoff for regularized torsion. Overrides --torsion-rel-curvature-cutoff if given.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not show the pop-up plot window.",
    )
    parser.add_argument(
        "--no-writhe-density",
        action="store_true",
        help="Skip the O(N^2) local writhe-density calculation.",
    )
    return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> None:
    """Run command-line mode."""
    xyz_file = args.xyz_file
    output_path = args.out
    if args.example_trefoil:
        xyz_file = write_trefoil_example(n_points=args.example_points)
        if output_path is None:
            output_path = output_default_name(xyz_file)
        print("[INFO] Wrote trefoil example: {}".format(xyz_file))

    if xyz_file is None:
        raise SystemExit("Please provide an xyz_file, or run with --gui.")

    file_format = "molecule" if args.molecule else args.format
    closed = not args.open

    try:
        run_calculation(
            xyz_file=xyz_file,
            file_format=file_format,
            closed=closed,
            nsamples=args.nsamples,
            output_path=output_path,
            smooth=(not args.no_smooth),
            smooth_window=args.smooth_window,
            polyorder=args.polyorder,
            show_plot=(not args.no_plot),
            calculate_writhe_density=(not args.no_writhe_density),
            torsion_rel_curvature_cutoff=args.torsion_rel_curvature_cutoff,
            torsion_min_curvature=args.torsion_min_curvature,
        )
    except Exception as exc:
        raise SystemExit(str(exc))


def launch_gui() -> None:
    """Launch a small Tkinter GUI for the calculation."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        raise SystemExit("Could not import Tkinter: {}".format(exc))

    root = tk.Tk()
    root.title("Local Curvature, Torsion, and Writhe Density")
    root.geometry("860x700")

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    format_var = tk.StringVar(value="auto")
    curve_type_var = tk.StringVar(value="closed")
    nsamples_var = tk.StringVar(value="500")
    smooth_var = tk.BooleanVar(value=True)
    smooth_window_var = tk.StringVar(value="")
    polyorder_var = tk.StringVar(value="3")
    show_plot_var = tk.BooleanVar(value=True)
    writhe_density_var = tk.BooleanVar(value=True)
    torsion_rel_cutoff_var = tk.StringVar(value="0.05")
    torsion_min_curv_var = tk.StringVar(value="")

    def append_log(message: str) -> None:
        log_text.configure(state="normal")
        log_text.insert("end", message + "\n")
        log_text.see("end")
        log_text.configure(state="disabled")
        root.update_idletasks()

    def browse_input() -> None:
        filename = filedialog.askopenfilename(
            title="Select coordinate/XYZ file",
            filetypes=[
                ("Coordinate or XYZ files", "*.txt *.xyz *.dat *.csv"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            input_var.set(filename)
            if not output_var.get().strip():
                output_var.set(output_default_name(filename))

    def browse_output() -> None:
        filename = filedialog.asksaveasfilename(
            title="Save output CSV as",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filename:
            output_var.set(filename)

    def clear_log() -> None:
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")

    def load_trefoil_example(run_after: bool = False) -> None:
        try:
            path = write_trefoil_example(n_points=300)
            input_var.set(path)
            output_var.set(output_default_name(path))
            format_var.set("plain")
            curve_type_var.set("closed")
            nsamples_var.set("500")
            smooth_var.set(True)
            smooth_window_var.set("")
            polyorder_var.set("3")
            show_plot_var.set(True)
            writhe_density_var.set(True)
            torsion_rel_cutoff_var.set("0.05")
            torsion_min_curv_var.set("")
            clear_log()
            append_log("Loaded trefoil example: {}".format(path))
            append_log("Equation: x=(2+cos(3t))*cos(2t), y=(2+cos(3t))*sin(2t), z=sin(3t)")
            append_log("This is the three-lobe trefoil knot, the (2,3) torus knot.")
            if run_after:
                root.after(50, run_from_gui)
        except Exception as exc:
            append_log("[ERROR] {}".format(exc))
            messagebox.showerror("Trefoil example error", str(exc))

    def run_from_gui() -> None:
        clear_log()
        xyz_file = input_var.get().strip()
        if not xyz_file:
            messagebox.showerror("Missing input", "Please choose an input coordinate/XYZ file.")
            return

        try:
            nsamples = int(nsamples_var.get().strip())
            polyorder = int(polyorder_var.get().strip())
            smooth_window_text = smooth_window_var.get().strip()
            smooth_window = int(smooth_window_text) if smooth_window_text else None
            output_path = output_var.get().strip() or output_default_name(xyz_file)
            output_var.set(output_path)
            torsion_rel_cutoff = float(torsion_rel_cutoff_var.get().strip() or "0.05")
            torsion_min_text = torsion_min_curv_var.get().strip()
            torsion_min_curvature = float(torsion_min_text) if torsion_min_text else None

            run_calculation(
                xyz_file=xyz_file,
                file_format=format_var.get(),
                closed=(curve_type_var.get() == "closed"),
                nsamples=nsamples,
                output_path=output_path,
                smooth=smooth_var.get(),
                smooth_window=smooth_window,
                polyorder=polyorder,
                show_plot=show_plot_var.get(),
                calculate_writhe_density=writhe_density_var.get(),
                torsion_rel_curvature_cutoff=torsion_rel_cutoff,
                torsion_min_curvature=torsion_min_curvature,
                log_func=append_log,
            )
            messagebox.showinfo("Finished", "Calculation finished.\n\nOutput:\n{}".format(output_path))
        except Exception as exc:
            append_log("[ERROR] {}".format(exc))
            messagebox.showerror("Error", str(exc))

    main_frame = ttk.Frame(root, padding=12)
    main_frame.pack(fill="both", expand=True)

    for col in range(3):
        main_frame.columnconfigure(col, weight=1 if col == 1 else 0)
    main_frame.rowconfigure(12, weight=1)

    ttk.Label(main_frame, text="Input XYZ/coordinate file:").grid(row=0, column=0, sticky="w", pady=4)
    ttk.Entry(main_frame, textvariable=input_var).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
    ttk.Button(main_frame, text="Browse...", command=browse_input).grid(row=0, column=2, sticky="ew", pady=4)

    ttk.Label(main_frame, text="Output CSV:").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Entry(main_frame, textvariable=output_var).grid(row=1, column=1, sticky="ew", padx=6, pady=4)
    ttk.Button(main_frame, text="Browse...", command=browse_output).grid(row=1, column=2, sticky="ew", pady=4)

    ttk.Label(main_frame, text="Input format:").grid(row=2, column=0, sticky="w", pady=4)
    ttk.Combobox(
        main_frame,
        textvariable=format_var,
        values=["auto", "plain", "molecule"],
        state="readonly",
        width=15,
    ).grid(row=2, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(main_frame, text="Curve type:").grid(row=3, column=0, sticky="w", pady=4)
    curve_frame = ttk.Frame(main_frame)
    curve_frame.grid(row=3, column=1, sticky="w", padx=6, pady=4)
    ttk.Radiobutton(curve_frame, text="Closed", variable=curve_type_var, value="closed").pack(side="left")
    ttk.Radiobutton(curve_frame, text="Open", variable=curve_type_var, value="open").pack(side="left", padx=(12, 0))

    ttk.Label(main_frame, text="Number of samples:").grid(row=4, column=0, sticky="w", pady=4)
    ttk.Entry(main_frame, textvariable=nsamples_var, width=12).grid(row=4, column=1, sticky="w", padx=6, pady=4)

    ttk.Checkbutton(main_frame, text="Use Savitzky-Golay smoothing", variable=smooth_var).grid(
        row=5, column=1, sticky="w", padx=6, pady=4
    )

    ttk.Label(main_frame, text="Smoothing window:").grid(row=6, column=0, sticky="w", pady=4)
    smooth_frame = ttk.Frame(main_frame)
    smooth_frame.grid(row=6, column=1, sticky="w", padx=6, pady=4)
    ttk.Entry(smooth_frame, textvariable=smooth_window_var, width=12).pack(side="left")
    ttk.Label(smooth_frame, text="blank = default; must be odd").pack(side="left", padx=(8, 0))

    ttk.Label(main_frame, text="Smoothing polyorder:").grid(row=7, column=0, sticky="w", pady=4)
    ttk.Entry(main_frame, textvariable=polyorder_var, width=12).grid(row=7, column=1, sticky="w", padx=6, pady=4)

    ttk.Checkbutton(main_frame, text="Show pop-up plot after calculation", variable=show_plot_var).grid(
        row=8, column=1, sticky="w", padx=6, pady=4
    )

    ttk.Checkbutton(
        main_frame,
        text="Calculate local writhe density (O(N^2))",
        variable=writhe_density_var,
    ).grid(row=9, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(main_frame, text="Torsion rel. curvature cutoff:").grid(row=10, column=0, sticky="w", pady=4)
    torsion_frame = ttk.Frame(main_frame)
    torsion_frame.grid(row=10, column=1, sticky="w", padx=6, pady=4)
    ttk.Entry(torsion_frame, textvariable=torsion_rel_cutoff_var, width=12).pack(side="left")
    ttk.Label(torsion_frame, text="default 0.05; masks near-zero curvature").pack(side="left", padx=(8, 0))

    ttk.Label(main_frame, text="Torsion min curvature:").grid(row=11, column=0, sticky="w", pady=4)
    torsion_min_frame = ttk.Frame(main_frame)
    torsion_min_frame.grid(row=11, column=1, sticky="w", padx=6, pady=4)
    ttk.Entry(torsion_min_frame, textvariable=torsion_min_curv_var, width=12).pack(side="left")
    ttk.Label(torsion_min_frame, text="blank = use relative cutoff").pack(side="left", padx=(8, 0))

    log_text = tk.Text(main_frame, height=16, wrap="word", state="disabled")
    log_text.grid(row=12, column=0, columnspan=3, sticky="nsew", pady=(10, 6))
    log_scroll = ttk.Scrollbar(main_frame, orient="vertical", command=log_text.yview)
    log_scroll.grid(row=12, column=3, sticky="ns", pady=(10, 6))
    log_text.configure(yscrollcommand=log_scroll.set)

    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=13, column=0, columnspan=3, sticky="e", pady=4)
    ttk.Button(
        button_frame,
        text="Load trefoil example",
        command=lambda: load_trefoil_example(False),
    ).pack(side="left", padx=4)
    ttk.Button(
        button_frame,
        text="Run trefoil example",
        command=lambda: load_trefoil_example(True),
    ).pack(side="left", padx=4)
    ttk.Button(button_frame, text="Clear Log", command=clear_log).pack(side="left", padx=4)
    ttk.Button(button_frame, text="Run", command=run_from_gui).pack(side="left", padx=4)
    ttk.Button(button_frame, text="Close", command=root.destroy).pack(side="left", padx=4)

    append_log("Choose an input file and click Run. Default curve type is closed; local writhe density is enabled.")
    root.mainloop()


def main(argv: Optional[List[str]] = None) -> None:
    """Program entry point."""
    args = parse_args(argv)
    if args.gui or (args.xyz_file is None and not args.example_trefoil):
        launch_gui()
    else:
        run_cli(args)


if __name__ == "__main__":
    main(sys.argv[1:])
