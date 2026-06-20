#!/usr/bin/env python3
"""
view_xyzV3.py

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
  python curve_it_lib/view_xyzV3.py curve_coords.txt
  python curve_it_lib/view_xyzV3.py curve.xyz --molecule
  python curve_it_lib/view_xyzV3.py curve.xyz --format auto --projections
  python curve_it_lib/view_xyzV3.py multi_component.txt --components A,C
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
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


def component_label(index: int) -> str:
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


def component_index(label: str) -> int:
    """Convert spreadsheet-style component labels to zero-based indices."""
    text = (label or "").strip().upper()
    if not text or not text.isalpha():
        raise ValueError("Invalid component label: {!r}".format(label))
    value = 0
    for char in text:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def parse_component_selection(selection: Optional[str], n_components: int) -> List[int]:
    """Parse component labels such as all, A, A,C, or A-C."""
    if n_components <= 0:
        raise ValueError("No components are available.")
    text = (selection or "all").strip()
    if not text or text.lower() in ("all", "*"):
        return list(range(n_components))

    chosen: List[int] = []
    for token in text.replace(";", ",").replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_label, end_label = [part.strip() for part in token.split("-", 1)]
            start = component_index(start_label)
            end = component_index(end_label)
            if end < start:
                start, end = end, start
            chosen.extend(range(start, end + 1))
        else:
            chosen.append(component_index(token))

    unique: List[int] = []
    for idx in sorted(chosen):
        if idx < 0 or idx >= n_components:
            max_label = component_label(n_components - 1)
            bad_label = str(idx) if idx < 0 else component_label(idx)
            raise ValueError("Component {} is out of range A-{}.".format(bad_label, max_label))
        if idx not in unique:
            unique.append(idx)
    if not unique:
        raise ValueError("No valid components were selected.")
    return unique


def read_xyz_like_components(filename: str, file_format: str = "auto") -> List[np.ndarray]:
    """
    Read one or more components from a plain coordinate file or molecular XYZ file.

    In plain/auto coordinate mode, blank lines separate components A, B, C...
    In molecular XYZ mode, the file is treated as one molecule-like component.
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

    is_molecule = file_format == "molecule" or (
        file_format == "auto"
        and first_token_is_integer(lines[start_index])
        and len(lines) >= start_index + 3
    )

    if is_molecule:
        data_start = start_index + 2
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
        return [np.asarray(coords, dtype=float)]

    components: List[np.ndarray] = []
    current = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped:
            if current:
                components.append(np.asarray(current, dtype=float))
                current = []
            continue
        if stripped.startswith("#") or stripped.startswith("!"):
            continue
        point = parse_coordinate_line(stripped)
        if point is not None:
            current.append(point)
    if current:
        components.append(np.asarray(current, dtype=float))

    components = [component for component in components if component.shape[0] > 0]
    if not components:
        raise ValueError("No 3D points could be read from file: {}".format(filename))
    return components


def read_xyz_like(filename: str, file_format: str = "auto") -> np.ndarray:
    """
    Read points from a plain coordinate file or a molecular XYZ file.

    file_format:
      auto      detect molecular XYZ if the first non-empty line is an atom count
      plain     treat all lines as potential x y z coordinate lines
      molecule  skip the first two lines as standard XYZ header
    """
    components = read_xyz_like_components(filename, file_format=file_format)
    return np.vstack(components)


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


def plot_curve_components(
    components: List[np.ndarray],
    selected_indices: Optional[List[int]] = None,
    closed: bool = True,
) -> None:
    """
    Plot multiple curve components with checkboxes and quick all/selected buttons.

    selected_indices controls which components are visible when the window opens
    and which components the Selected button restores.
    """
    clean_components = [
        np.asarray(component, dtype=float)
        for component in components
        if np.asarray(component).ndim == 2 and np.asarray(component).shape[1] == 3
    ]
    if not clean_components:
        raise ValueError("No valid curve components to plot.")

    n_components = len(clean_components)
    if selected_indices is None:
        selected_set = set(range(n_components))
    else:
        selected_set = {idx for idx in selected_indices if 0 <= idx < n_components}
    if not selected_set:
        selected_set = set(range(n_components))

    all_points = np.vstack(clean_components)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.24, right=0.96, top=0.92, bottom=0.08)

    color_map = plt.get_cmap("tab10")
    artists_by_component = []
    labels = []

    for i, points in enumerate(clean_components):
        label = "{} ({} pts)".format(component_label(i), points.shape[0])
        labels.append(label)
        color = color_map(i % 10)
        visible = i in selected_set
        artists = []

        if points.shape[0] >= 2:
            line, = ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                linewidth=1.2,
                label=label,
                visible=visible,
            )
            artists.append(line)
            if closed and points.shape[0] > 2:
                close_line, = ax.plot(
                    [points[-1, 0], points[0, 0]],
                    [points[-1, 1], points[0, 1]],
                    [points[-1, 2], points[0, 2]],
                    color=color,
                    linestyle=":",
                    linewidth=0.9,
                    visible=visible,
                )
                artists.append(close_line)

        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=[color],
            s=14,
            depthshade=True,
            visible=visible,
        )
        artists.append(scatter)

        start_marker = ax.scatter(
            points[0, 0],
            points[0, 1],
            points[0, 2],
            color="red",
            s=28,
            edgecolor="k",
            visible=visible,
        )
        artists.append(start_marker)
        if not closed and points.shape[0] > 1:
            end_marker = ax.scatter(
                points[-1, 0],
                points[-1, 1],
                points[-1, 2],
                color="black",
                s=28,
                edgecolor="k",
                visible=visible,
            )
            artists.append(end_marker)

        artists_by_component.append(artists)

    set_equal_aspect_3d(ax, all_points)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Curve Components")
    ax.legend(loc="best")

    check_ax = fig.add_axes([0.03, 0.24, 0.16, 0.62])
    check = CheckButtons(check_ax, labels, [i in selected_set for i in range(n_components)])
    check_ax.set_title("Show")

    def on_check_clicked(_label: str) -> None:
        statuses = check.get_status()
        for idx, artists in enumerate(artists_by_component):
            visible = statuses[idx]
            for artist in artists:
                artist.set_visible(visible)
        fig.canvas.draw_idle()

    check.on_clicked(on_check_clicked)

    def apply_visibility(indices) -> None:
        visible_set = set(indices)
        statuses = list(check.get_status())
        for idx in range(n_components):
            wanted = idx in visible_set
            if statuses[idx] != wanted:
                check.set_active(idx)
        fig.canvas.draw_idle()

    all_button_ax = fig.add_axes([0.03, 0.15, 0.16, 0.055])
    selected_button_ax = fig.add_axes([0.03, 0.08, 0.16, 0.055])
    all_button = Button(all_button_ax, "All components")
    selected_button = Button(selected_button_ax, "Selected")

    all_button.on_clicked(lambda _event: apply_visibility(range(n_components)))
    selected_button.on_clicked(lambda _event: apply_visibility(selected_set))

    fig._curve_component_widgets = (check, all_button, selected_button)
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
    parser.add_argument(
        "--components",
        default="all",
        help="For blank-line-separated coordinate files, show components such as A, B,C, A-C, or all.",
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
    components = read_xyz_like_components(filename, file_format=file_format)
    selected_indices = parse_component_selection(args.components, len(components))
    points = np.vstack([components[i] for i in selected_indices])

    if len(components) > 1:
        summary = ", ".join(
            "{}:{}".format(component_label(i), component.shape[0])
            for i, component in enumerate(components)
        )
        selected = ",".join(component_label(i) for i in selected_indices)
        print("[INFO] Loaded {} component(s): {}".format(len(components), summary))
        print("[INFO] Showing selected component(s): {} ({} points).".format(selected, len(points)))
        plot_curve_components(components, selected_indices=selected_indices, closed=args.closed)
    else:
        print("[INFO] Loaded {} points.".format(len(points)))
        plot_curve(points, closed=args.closed)

    if args.projections:
        plot_projections(points, closed=args.closed)


if __name__ == "__main__":
    main()
