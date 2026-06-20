#!/usr/bin/env python3
"""
cal_xyz_total_curvature_writheV2.py

Compute the total curvature and approximate writhe of a closed 3D curve.

This script can read both:

1) Plain coordinate files:
       x y z
       x y z
       ...

2) Standard molecular XYZ files:
       number_of_atoms
       comment line
       atom_symbol x y z
       atom_symbol x y z
       ...

Input:
  - a plain coordinate file or a molecular XYZ file

Output:
  - total curvature
  - total curvature divided by pi
  - approximate writhe

Examples:
  python curve_it_lib/cal_xyz_total_curvature_writheV2.py curve_coords.txt
  python curve_it_lib/cal_xyz_total_curvature_writheV2.py curve.xyz --molecule
  python curve_it_lib/cal_xyz_total_curvature_writheV2.py curve.xyz --format auto --nsamples 800
"""

import argparse
import math
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.signal import savgol_filter


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

    The smoothing behavior follows the original script. If too few points are
    present, the raw points are returned.
    """
    n = len(points)
    if n < 7:
        return points

    smooth_window_size = max(7, (n // 18) * 2 + 1)
    polyorder = 3
    if smooth_window_size <= polyorder:
        smooth_window_size = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3

    half_window = smooth_window_size // 2

    points_extended = np.concatenate(
        (points[-half_window:], points, points[:half_window])
    )

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


def strip_duplicate_endpoint(points: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Remove a duplicated final point from a closed curve.

    Periodic smoothing/spline code closes the curve internally. If the input
    already repeats the first point as the final row, keeping both rows can
    overweight that same geometric location.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= tol:
        return pts[:-1].copy()
    return pts


def read_xyz_like(filename: str, file_format: str = "auto", smooth: bool = True) -> np.ndarray:
    """
    Read 3D points and optionally smooth them as a closed curve.
    """
    points = read_xyz_like_raw(filename, file_format=file_format)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an N x 3 coordinate array.")

    if len(points) < 4:
        raise ValueError("At least 4 points are needed for a closed 3D curve.")

    points = strip_duplicate_endpoint(points)

    if smooth:
        points = smooth_closed_points(points)

    return points


def build_periodic_splines(points: np.ndarray) -> Tuple[CubicSpline, CubicSpline, CubicSpline]:
    """
    Build periodic cubic splines x(t), y(t), z(t) for a closed curve.

    The spline parameter t is normalized chord length, which is more stable
    than uniform point-index spacing for irregularly sampled curves. If the
    input repeats the first point as the final row, the duplicate is removed
    before the curve is closed internally.
    """
    points = strip_duplicate_endpoint(np.asarray(points, dtype=float))
    if len(points) < 4:
        raise ValueError("At least 4 non-duplicated points are needed for a closed 3D curve.")

    points = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 0.0:
        raise ValueError("The curve has zero total chord length.")

    t = np.empty(len(points), dtype=float)
    t[0] = 0.0
    t[1:] = np.cumsum(segment_lengths) / total_length
    t[-1] = 1.0

    keep = np.ones(len(t), dtype=bool)
    if len(t) > 2:
        keep[1:-1] = np.diff(t[:-1]) > 1e-14
    keep[-1] = True
    points = points[keep]
    t = t[keep]
    if len(points) < 4:
        raise ValueError("At least 4 non-duplicated points are needed for a closed 3D curve.")

    spline_x = CubicSpline(t, points[:, 0], bc_type="periodic")
    spline_y = CubicSpline(t, points[:, 1], bc_type="periodic")
    spline_z = CubicSpline(t, points[:, 2], bc_type="periodic")
    return spline_x, spline_y, spline_z


def curvature_function(spline_x: CubicSpline,
                       spline_y: CubicSpline,
                       spline_z: CubicSpline,
                       t: float) -> float:
    """
    Return kappa(t) * ds/dt so integration over t gives total curvature.
    """
    dx_dt = spline_x(t, 1)
    dy_dt = spline_y(t, 1)
    dz_dt = spline_z(t, 1)
    ddx_dt = spline_x(t, 2)
    ddy_dt = spline_y(t, 2)
    ddz_dt = spline_z(t, 2)

    ds_dt = np.sqrt(dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2)
    eps = 1e-12

    if np.isscalar(ds_dt):
        if ds_dt < eps:
            return 0.0
    else:
        ds_dt = np.where(ds_dt < eps, eps, ds_dt)

    cross_product = np.cross([dx_dt, dy_dt, dz_dt],
                             [ddx_dt, ddy_dt, ddz_dt])
    curvature = np.linalg.norm(cross_product) / (ds_dt ** 3)
    return curvature * ds_dt


def fit_spline_and_calculate_curvature(points: np.ndarray) -> float:
    """Fit periodic cubic splines and compute total curvature."""
    spline_x, spline_y, spline_z = build_periodic_splines(points)

    total_curvature, _ = quad(
        lambda t: curvature_function(spline_x, spline_y, spline_z, t),
        0.0,
        1.0,
        limit=1000,
        epsabs=1e-5,
        epsrel=1e-3,
    )
    return total_curvature


def calculate_writhe(points: np.ndarray, n_samples: int = 400) -> float:
    """
    Approximate writhe using the Gauss double integral on a spline sample.

    This is O(N^2) in n_samples.
    """
    if n_samples < 10:
        raise ValueError("n_samples should be at least 10.")

    spline_x, spline_y, spline_z = build_periodic_splines(points)

    ts = np.linspace(0.0, 1.0, n_samples, endpoint=False)

    x = spline_x(ts)
    y = spline_y(ts)
    z = spline_z(ts)

    dx_dt = spline_x(ts, 1)
    dy_dt = spline_y(ts, 1)
    dz_dt = spline_z(ts, 1)

    ds_dt = np.sqrt(dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2)
    eps = 1e-12
    ds_dt = np.where(ds_dt < eps, eps, ds_dt)

    tangents = np.stack((dx_dt, dy_dt, dz_dt), axis=1) / ds_dt[:, None]

    dt = 1.0 / n_samples
    ds = ds_dt * dt

    r = np.stack((x, y, z), axis=1)

    R = r[:, None, :] - r[None, :, :]
    dist = np.linalg.norm(R, axis=-1)

    np.fill_diagonal(dist, np.inf)

    Ti = tangents[:, None, :]
    Tj = tangents[None, :, :]
    cross_T = np.cross(Ti, Tj)

    numerator = np.einsum("ijk,ijk->ij", R, cross_T)

    dist_safe = np.where(dist < eps, eps, dist)
    inv_dist3 = 1.0 / (dist_safe ** 3)

    integrand = numerator * inv_dist3
    ds_outer = np.outer(ds, ds)

    wr = (1.0 / (4.0 * math.pi)) * np.sum(integrand * ds_outer)
    return wr


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute total curvature and approximate writhe of a closed 3D curve "
            "from a plain coordinate or molecular XYZ file."
        )
    )
    parser.add_argument(
        "xyz_file",
        nargs="?",
        help="Path to the coordinate/XYZ file containing x y z points."
    )
    parser.add_argument(
        "--format",
        choices=["auto", "plain", "molecule"],
        default="auto",
        help="Input format. Default: auto."
    )
    parser.add_argument(
        "-m",
        "--molecule",
        action="store_true",
        help="Shortcut for --format molecule."
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=400,
        help="Number of samples used for writhe approximation. Default: 400."
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable Savitzky-Golay smoothing and use raw input points."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.xyz_file is None:
        filename_raw = input("Enter the path to the coordinate/XYZ file: ").strip()
    else:
        filename_raw = args.xyz_file

    filename = filename_raw.strip().strip('"').strip("'")

    if not os.path.isfile(filename):
        raise SystemExit("File not found: {}".format(filename))

    file_format = "molecule" if args.molecule else args.format

    print("[INFO] Reading points from: {}".format(filename))
    print("[INFO] Input format: {}".format(file_format))
    print("[INFO] Smoothing: {}".format("off" if args.no_smooth else "on"))

    points = read_xyz_like(filename, file_format=file_format, smooth=(not args.no_smooth))
    print("[INFO] Loaded {} points.".format(len(points)))

    total_curvature = fit_spline_and_calculate_curvature(points)
    print("\nTotal Curvature of the Curve:")
    print(total_curvature)
    print("({} * pi)".format(total_curvature / math.pi))

    writhe = calculate_writhe(points, n_samples=args.nsamples)
    print("\nApproximate Writhe of the Curve:")
    print(writhe)


if __name__ == "__main__":
    main()
