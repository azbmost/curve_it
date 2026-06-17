#!/usr/bin/env python3
"""
view_xyzV2.py

Simple 3D viewer for curve coordinate files.

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
  - an interactive 3D matplotlib plot
  - optionally, XY/XZ/YZ projection plots

Example:
  python curve_it_lib/view_xyzV2.py curve_coords.txt
  python curve_it_lib/view_xyzV2.py curve.xyz --molecule
  python curve_it_lib/view_xyzV2.py curve.xyz --format auto --projections
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  needed for 3D projection


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


def read_xyz_like(filename: str, file_format: str = "auto") -> np.ndarray:
    """
    Read points from a plain coordinate file or a molecular XYZ file.

    file_format:
      auto      detect molecular XYZ if the first non-empty line is an atom count
      plain     treat all lines as potential x y z coordinate lines
      molecule  skip the first two lines as standard XYZ header
    """
    with open(filename, "r") as f:
        raw_lines = f.readlines()

    if not raw_lines:
        raise ValueError("The file is empty: {}".format(filename))

    # Preserve line order, but ignore leading/trailing blank lines for detection.
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
        # Auto-detect standard molecular XYZ:
        # first non-empty line is a single integer atom count.
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


def set_equal_aspect_3d(ax, points: np.ndarray) -> None:
    """Set a 3D axis to have equal aspect ratio based on the given points."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
    if max_range <= 0:
        max_range = 1.0

    mid_x = 0.5 * (x.max() + x.min())
    mid_y = 0.5 * (y.max() + y.min())
    mid_z = 0.5 * (z.max() + z.min())

    half = 0.5 * max_range
    ax.set_xlim(mid_x - half, mid_x + half)
    ax.set_ylim(mid_y - half, mid_y + half)
    ax.set_zlim(mid_z - half, mid_z + half)


def plot_curve(points: np.ndarray, closed: bool = True) -> None:
    """
    Plot the 3D curve.

    The curve is shown as a line, with points colored by their order along
    the curve. The start point is highlighted in red.
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(points[:, 0], points[:, 1], points[:, 2],
            color="0.6", linewidth=1.0, label="curve")

    if closed:
        ax.plot(
            [points[-1, 0], points[0, 0]],
            [points[-1, 1], points[0, 1]],
            [points[-1, 2], points[0, 2]],
            color="0.6",
            linewidth=1.0,
        )

    t = np.linspace(0.0, 1.0, len(points))
    scat = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=t,
        cmap="viridis",
        s=12,
        depthshade=True,
        label="points",
    )
    cbar = fig.colorbar(scat, ax=ax, pad=0.08)
    cbar.set_label("Curve parameter")

    ax.scatter(
        points[0, 0],
        points[0, 1],
        points[0, 2],
        color="red",
        s=40,
        edgecolor="k",
        label="start",
    )
    if not closed:
        ax.scatter(
            points[-1, 0],
            points[-1, 1],
            points[-1, 2],
            color="black",
            s=40,
            edgecolor="k",
            label="end",
        )

    set_equal_aspect_3d(ax, points)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Curve")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_projections(points: np.ndarray, closed: bool = True) -> None:
    """Plot XY, XZ, and YZ projections."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    t = np.linspace(0.0, 1.0, len(points))

    datasets = [
        (x, y, "X", "Y", "XY projection"),
        (x, z, "X", "Z", "XZ projection"),
        (y, z, "Y", "Z", "YZ projection"),
    ]

    last_scatter = None
    for ax, (a, b, xlabel, ylabel, title) in zip(axes, datasets):
        last_scatter = ax.scatter(a, b, c=t, cmap="viridis", s=8)
        ax.plot(a, b, color="0.7", linewidth=0.8)
        if closed:
            ax.plot([a[-1], a[0]], [b[-1], b[0]], color="0.7", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)

    cbar = fig.colorbar(last_scatter, ax=axes.tolist(), pad=0.02)
    cbar.set_label("Curve parameter")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="View a 3D curve from a plain coordinate or molecular XYZ file."
    )
    parser.add_argument(
        "xyz_file",
        nargs="?",
        help="Path to the coordinate/XYZ file containing x y z points.",
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
    parser.add_argument(
        "--open",
        dest="closed",
        action="store_false",
        help="Treat the curve as open and do not connect last point to first.",
    )
    parser.set_defaults(closed=True)
    parser.add_argument(
        "--projections",
        action="store_true",
        help="Also show XY, XZ, and YZ projections in a separate figure.",
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
    points = read_xyz_like(filename, file_format=file_format)

    print("[INFO] Loaded {} points.".format(len(points)))
    plot_curve(points, closed=args.closed)

    if args.projections:
        plot_projections(points, closed=args.closed)


if __name__ == "__main__":
    main()
