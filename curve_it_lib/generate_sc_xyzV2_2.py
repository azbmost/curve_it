#!/usr/bin/env python3
"""Generate a closed plectonemic supercoiled-DNA axis curve.

The generated curve is a single closed centerline made from two antipodal
superhelical arms and two smooth end loops.  It is intended as an educational
"textbook plectoneme" rather than an elastic-rod energy minimizer.

Two related writhe quantities are kept explicit:

* ``--writhe`` is fitted to the continuous Gauss-integral writhe used by
  Curve It.
* When the requested writhe is an integer, the phase sweep is constrained so
  that the fixed xz projection has exactly ``abs(writhe)`` visible crossings.
  The sign selects the mirror image/handedness; a crossing count itself is
  unsigned.  The unconstrained PCA plane and its crossing count are reported
  separately because PCA can select yz for compact plectonemes.

Generate SC V2_2 can choose the opening angle automatically or retain a
user-provided angle.  Automatic mode offers four geometric objectives:
minimize the largest local curvature (the default), minimize total curvature,
minimize the reduced bending energy ``integral kappa(s)^2 ds``, or make the
projected terminal-lobe z-height equal to the middle-lobe z-height.  Curvature,
bending-energy, and projected-lobe metrics are evaluated on the final-length
centerline and reported whenever they are defined.

The canonical dimensionless curve is fitted first, periodically smoothed once
with Curve It's Savitzky-Golay convention, resampled, and then uniformly
scaled.  Consequently the geometry written to disk is the same smoothed
geometry whose writhe is fitted and verified, and its closed-polyline contour
length matches ``--total-length``.  Peak local curvature is evaluated only
after this length scaling because it has inverse-length units.

Examples
--------
Open the GUI::

    python generate_sc_xyzV2_2.py

Generate a 340-Angstrom curve with writhe -3::

    python generate_sc_xyzV2_2.py -L 340 -w -3 --angle-objective max-local -n 2000 -o sc.xyz

Retain a user-provided 25-degree opening angle::

    python generate_sc_xyzV2_2.py -L 340 -w -3 -a 25 -n 2000 -o sc_25deg.xyz

The output is plain coordinate XYZ: one ``x y z`` row per point, without an
atom-count header.  Load it into Curve It as a *closed* curve.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from curve_it_lib.cal_xyz_total_curvature_writheV2 import (
        build_periodic_splines,
        calculate_polyline_writhe,
        calculate_writhe,
        fit_spline_and_calculate_curvature,
        smooth_closed_points,
    )
    _WRITHE_IMPORT_ERROR: Optional[Exception] = None
except Exception as package_exc:
    try:
        # Support running this file directly from inside curve_it_lib.
        from cal_xyz_total_curvature_writheV2 import (  # type: ignore
            build_periodic_splines,
            calculate_polyline_writhe,
            calculate_writhe,
            fit_spline_and_calculate_curvature,
            smooth_closed_points,
        )
        _WRITHE_IMPORT_ERROR = None
    except Exception as local_exc:
        calculate_polyline_writhe = None  # type: ignore
        calculate_writhe = None  # type: ignore
        build_periodic_splines = None  # type: ignore
        fit_spline_and_calculate_curvature = None  # type: ignore
        smooth_closed_points = None  # type: ignore
        _WRITHE_IMPORT_ERROR = local_exc if local_exc is not None else package_exc


PointArray = np.ndarray

TOOL_NAME = "Generate SC"
TOOL_VERSION = "V2_2"

DEFAULT_TOTAL_LENGTH = 340.0
DEFAULT_WRITHE = -3.0
DEFAULT_NUM_POINTS = 2000
DEFAULT_PRECISION = 8
DEFAULT_OUTPUT = "supercoiled_DNA.xyz"

CANONICAL_RADIUS = 1.0
DEFAULT_OPENING_ANGLE_DEG = 25.0
CANONICAL_STEM_EXTENSION = 2.0
DEFAULT_LOOP_CONTROL = 4.0 / 3.0

CURVATURE_OBJECTIVE_TOTAL = "total"
CURVATURE_OBJECTIVE_MAX_LOCAL = "max-local"
CURVATURE_OBJECTIVE_BENDING_ENERGY = "bending-energy"
OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES = "equal-lobes"
OPENING_ANGLE_MODE_MANUAL = "manual"
CURVATURE_OBJECTIVES = (
    CURVATURE_OBJECTIVE_TOTAL,
    CURVATURE_OBJECTIVE_MAX_LOCAL,
    CURVATURE_OBJECTIVE_BENDING_ENERGY,
    OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES,
)
DEFAULT_CURVATURE_OBJECTIVE = CURVATURE_OBJECTIVE_MAX_LOCAL

# A strictly positive lower bound prevents curvature-based objectives from
# approaching the degenerate zero-angle limit.  The upper bound avoids the
# equally degenerate 90-degree limit.  Infeasible angle/writhe combinations
# inside this documented interval are skipped by the optimizer.
MIN_AUTO_OPENING_ANGLE_DEG = 5.0
MAX_AUTO_OPENING_ANGLE_DEG = 85.0
OPENING_ANGLE_GRID_SIZE = 17
OPENING_ANGLE_REFINEMENT_LEVELS = 3
OPENING_ANGLE_REFINEMENT_SAMPLES = 9
CURVATURE_EVALUATION_SAMPLES = 8000

WRITHE_SEARCH_SAMPLES = 400
WRITHE_TOLERANCE = 1.0e-3
EQUAL_LOBE_WRITHE_FIT_TOLERANCE = 1.0e-4
BENDING_ENERGY_WRITHE_FIT_TOLERANCE = 1.0e-4
EQUAL_LOBE_RELATIVE_TOLERANCE = 2.0e-3
MAX_ABS_WRITHE = 10.0
# Far beyond molecular coordinate scales, downstream spline, Gauss-integral,
# and fixed-width PDB operations cease to be numerically meaningful.
MAX_TOTAL_LENGTH = 1.0e12


@dataclass
class ContourLandmark:
    """An output point and its closed-contour position from the first XYZ row."""

    point_index: int
    contour_percent: float


@dataclass
class ContourLandmarks:
    """Reproducible structural landmarks on the generated plectoneme."""

    top_terminus: ContourLandmark
    bottom_terminus: ContourLandmark
    middle_segment_peaks: Tuple[Tuple[ContourLandmark, ContourLandmark], ...]


@dataclass
class SCGenerationResult:
    """Generated coordinates plus the quantities shown by the CLI/GUI."""

    points: PointArray
    requested_length: float
    achieved_length: float
    requested_writhe: float
    achieved_writhe: float
    xz_crossings: int
    pca_crossings: int
    pca_plane: str
    superhelical_turns: float
    opening_angle_deg: float
    curvature_objective: str
    total_curvature: float
    maximum_local_curvature: float
    bending_energy_integral: float
    top_terminal_lobe_height_xz: Optional[float]
    bottom_terminal_lobe_height_xz: Optional[float]
    terminal_lobe_height_xz: Optional[float]
    middle_lobe_height_xz: Optional[float]
    lobe_height_mismatch_xz: Optional[float]
    opening_angle_evaluations: int
    loop_control_canonical: float
    radius_scaled: float
    stem_height_scaled: float
    solver_iterations: int
    landmarks: ContourLandmarks


@dataclass
class _AngleCandidate:
    """One feasible opening-angle candidate and its fitted canonical curve."""

    angle_deg: float
    loop_control: float
    canonical_points: PointArray
    achieved_writhe: float
    stem_height: float
    writhe_solver_iterations: int
    total_curvature: float
    maximum_local_curvature: float
    bending_energy_integral: float
    top_terminal_lobe_height_xz: Optional[float]
    bottom_terminal_lobe_height_xz: Optional[float]
    terminal_lobe_height_xz: Optional[float]
    middle_lobe_height_xz: Optional[float]
    lobe_height_mismatch_xz: Optional[float]


def resource_path(relative_path: str) -> Path:
    """Return a resource path that also works from a PyInstaller bundle."""
    source_dir = Path(__file__).resolve().parent
    source_root = source_dir.parent if source_dir.name == "curve_it_lib" else source_dir
    base_dir = Path(getattr(sys, "_MEIPASS", source_root))
    return base_dir / relative_path


def set_optional_window_icon(root, tk_module, icon_filenames: Sequence[str], image_attr: str) -> None:
    """Set a Tk window icon when one of the optional PNG assets is present."""
    for icon_filename in icon_filenames:
        icon_path = resource_path("assets/{0}".format(icon_filename))
        if not icon_path.exists():
            continue
        try:
            icon_image = tk_module.PhotoImage(file=str(icon_path))
            root.iconphoto(True, icon_image)
            setattr(root, image_attr, icon_image)
            return
        except Exception:
            continue


def minimum_num_points(target_writhe: float) -> int:
    """Return the sampling floor needed for a stable crossing diagram."""
    return max(240, 80 * int(math.ceil(abs(float(target_writhe)))))


def validate_inputs(
    total_length: float,
    target_writhe: float,
    num_points: int,
    precision: int = DEFAULT_PRECISION,
    curvature_objective: str = DEFAULT_CURVATURE_OBJECTIVE,
    opening_angle_deg: Optional[float] = None,
) -> None:
    """Validate user-facing inputs and raise ``ValueError`` when invalid."""
    if not math.isfinite(total_length) or total_length <= 0.0:
        raise ValueError("Total length L must be a finite positive number.")
    if total_length > MAX_TOTAL_LENGTH:
        raise ValueError(
            "Total length L is outside the supported numeric range "
            "(maximum {0:.0e}).".format(MAX_TOTAL_LENGTH)
        )
    if not math.isfinite(target_writhe):
        raise ValueError("Writhe must be finite.")
    if abs(target_writhe) > MAX_ABS_WRITHE + 1.0e-12:
        raise ValueError(
            "For reliable educational xz crossing diagrams, |writhe| must be "
            "at most {0:g}.".format(MAX_ABS_WRITHE)
        )
    if isinstance(num_points, bool) or int(num_points) != num_points:
        raise ValueError("Number of points must be an integer.")
    required = minimum_num_points(target_writhe)
    if int(num_points) < required:
        raise ValueError(
            "Number of points must be at least {0} for writhe {1:g}; this keeps "
            "the plectoneme and xz crossings adequately sampled.".format(required, target_writhe)
        )
    if isinstance(precision, bool) or int(precision) != precision:
        raise ValueError("Precision must be an integer.")
    if int(precision) < 0 or int(precision) > 15:
        raise ValueError("Precision must be between 0 and 15 decimal places.")
    if str(curvature_objective) not in CURVATURE_OBJECTIVES:
        raise ValueError(
            "Automatic opening-angle objective must be one of: {0}.".format(
                ", ".join(CURVATURE_OBJECTIVES)
            )
        )
    if opening_angle_deg is not None and (
        not math.isfinite(float(opening_angle_deg))
        or not 0.0 < float(opening_angle_deg) < 90.0
    ):
        raise ValueError(
            "A user-provided opening angle must be finite and strictly between "
            "0 and 90 degrees."
        )
    if (
        opening_angle_deg is None
        and str(curvature_objective) == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES
    ):
        nearest_integer = int(round(float(target_writhe)))
        if (
            abs(float(target_writhe) - nearest_integer) > 1.0e-10
            or abs(nearest_integer) < 2
        ):
            raise ValueError(
                "The equal-lobes objective requires an integer writhe with "
                "|W| >= 2 so the fixed xz projection has at least one middle lobe."
            )
    if (
        calculate_polyline_writhe is None
        or calculate_writhe is None
        or build_periodic_splines is None
        or fit_spline_and_calculate_curvature is None
        or smooth_closed_points is None
    ):
        detail = "" if _WRITHE_IMPORT_ERROR is None else " ({0})".format(_WRITHE_IMPORT_ERROR)
        raise RuntimeError(
            "Generate SC needs SciPy for curvature and mapped-polyline writhe support. "
            "Install the packages in requirements.txt." + detail
        )


def _unit(vector: PointArray) -> PointArray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        raise ValueError("Cannot normalize a zero tangent vector.")
    return np.asarray(vector, dtype=float) / norm


def _cubic_bezier(
    p0: PointArray,
    p1: PointArray,
    p2: PointArray,
    p3: PointArray,
    t: PointArray,
) -> PointArray:
    """Evaluate a cubic Bezier segment for a one-dimensional ``t`` array."""
    tt = np.asarray(t, dtype=float)[:, None]
    uu = 1.0 - tt
    return (
        uu ** 3 * p0
        + 3.0 * uu ** 2 * tt * p1
        + 3.0 * uu * tt ** 2 * p2
        + tt ** 3 * p3
    )


def _canonical_geometry(
    target_writhe: float,
    opening_angle_deg: float = DEFAULT_OPENING_ANGLE_DEG,
) -> Tuple[float, float]:
    """Return signed phase sweep and canonical stem height.

    A phase sweep of pi per requested writhe unit makes every integer unit one
    crossing in the fixed xz projection.  Each full superhelical turn
    therefore contributes two diagram crossings.
    """
    theta_total = math.pi * float(target_writhe)
    alpha = math.radians(float(opening_angle_deg))
    stem_height = (
        CANONICAL_STEM_EXTENSION
        + abs(theta_total) * CANONICAL_RADIUS / math.tan(alpha)
    )
    return theta_total, stem_height


def _build_dense_canonical_curve(
    target_writhe: float,
    loop_control: float,
    num_points: int,
    opening_angle_deg: float = DEFAULT_OPENING_ANGLE_DEG,
) -> Tuple[PointArray, float]:
    """Build a dense, C1 closed plectoneme before arc-length resampling."""
    theta_total, stem_height = _canonical_geometry(target_writhe, opening_angle_deg)
    radius = CANONICAL_RADIUS

    dense_per_segment = max(
        600,
        2 * int(num_points),
        int(math.ceil(240.0 * max(1.0, abs(target_writhe)))),
    )
    t = np.linspace(0.0, 1.0, dense_per_segment, endpoint=False)

    def arm_a(u: PointArray) -> PointArray:
        theta = theta_total * u
        return np.column_stack(
            (
                radius * np.cos(theta),
                radius * np.sin(theta),
                stem_height * (0.5 - u),
            )
        )

    def arm_b(v: PointArray) -> PointArray:
        theta = theta_total * (1.0 - v) + math.pi
        return np.column_stack(
            (
                radius * np.cos(theta),
                radius * np.sin(theta),
                stem_height * (-0.5 + v),
            )
        )

    def tangent_a(u: float) -> PointArray:
        theta = theta_total * u
        return _unit(
            np.array(
                (
                    -radius * theta_total * math.sin(theta),
                    radius * theta_total * math.cos(theta),
                    -stem_height,
                ),
                dtype=float,
            )
        )

    def tangent_b(v: float) -> PointArray:
        theta = theta_total * (1.0 - v) + math.pi
        return _unit(
            np.array(
                (
                    radius * theta_total * math.sin(theta),
                    -radius * theta_total * math.cos(theta),
                    stem_height,
                ),
                dtype=float,
            )
        )

    a_top = arm_a(np.array((0.0,)))[0]
    a_bottom = arm_a(np.array((1.0,)))[0]
    b_bottom = arm_b(np.array((0.0,)))[0]
    b_top = arm_b(np.array((1.0,)))[0]

    ta_top = tangent_a(0.0)
    ta_bottom = tangent_a(1.0)
    tb_bottom = tangent_b(0.0)
    tb_top = tangent_b(1.0)

    # The Bezier controls continue the arm tangents, turn outside the stem's
    # z-range, and arrive tangent to the antipodal return arm.  Varying the
    # common control distance adjusts the end-loop contribution to Gauss
    # writhe without changing the integer number of stem crossings.
    bottom_loop = _cubic_bezier(
        a_bottom,
        a_bottom + loop_control * ta_bottom,
        b_bottom - loop_control * tb_bottom,
        b_bottom,
        t,
    )
    top_loop = _cubic_bezier(
        b_top,
        b_top + loop_control * tb_top,
        a_top - loop_control * ta_top,
        a_top,
        t,
    )

    dense = np.vstack((arm_a(t), bottom_loop, arm_b(t), top_loop))
    return dense, stem_height


def closed_polyline_length(points: PointArray) -> float:
    """Return contour length including the final-to-first closing segment."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError("A closed curve needs an N x 3 array with at least 3 points.")
    return float(np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))


def analyze_contour_landmarks(
    points: PointArray,
    target_writhe: float,
    radius_scaled: float,
    stem_height_scaled: float,
) -> ContourLandmarks:
    """Locate termini and all interior-lobe peaks in the fixed xz diagram.

    The top and bottom termini are the output vertices with maximum and minimum
    z.  For integer W, the |W| crossings divide the xz diagram into two terminal
    lobes and |W|-1 middle lobes.  Each middle lobe has two centerline peaks at
    opposing x extrema; the nearest serialized output vertices are reported.
    Percentages start at the first XYZ row and include the closing seam.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise ValueError("Contour landmarks need an N x 3 curve with at least 4 points.")

    segment_lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    total = float(np.sum(segment_lengths))
    if not math.isfinite(total) or total <= 1.0e-14 or np.any(segment_lengths <= 0.0):
        raise ValueError("Contour landmarks need a finite closed curve without duplicate points.")
    vertex_arc = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))

    top_index = int(np.argmax(pts[:, 2]))
    bottom_index = int(np.argmin(pts[:, 2]))

    def landmark(index: int) -> ContourLandmark:
        return ContourLandmark(
            point_index=int(index),
            contour_percent=100.0 * float(vertex_arc[index]) / total,
        )

    middle_segment_peaks = []
    nearest_integer = int(round(float(target_writhe)))
    if abs(float(target_writhe) - nearest_integer) <= 1.0e-10:
        crossing_count = abs(nearest_integer)
        center = np.mean(pts, axis=0)
        for middle_index in range(1, crossing_count):
            z_offset = float(stem_height_scaled) * (
                0.5 - float(middle_index) / float(crossing_count)
            )
            arm_a_x = float(radius_scaled) * (-1.0 if middle_index % 2 else 1.0)
            target_a = center + np.array((arm_a_x, 0.0, z_offset), dtype=float)
            target_b = center + np.array((-arm_a_x, 0.0, z_offset), dtype=float)
            index_a = int(np.argmin(np.linalg.norm(pts - target_a, axis=1)))
            index_b = int(np.argmin(np.linalg.norm(pts - target_b, axis=1)))
            middle_segment_peaks.append((landmark(index_a), landmark(index_b)))

    return ContourLandmarks(
        top_terminus=landmark(top_index),
        bottom_terminus=landmark(bottom_index),
        middle_segment_peaks=tuple(middle_segment_peaks),
    )


def evaluate_xz_lobe_heights(
    points: PointArray,
    target_writhe: float,
    stem_height_scaled: float,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """Measure terminal and middle lobe z-spans in the fixed xz projection.

    For integer ``m = |W| >= 2``, the analytic arm crossings are evenly spaced
    along z.  Their common middle-lobe height is ``H / m``.  Each terminal-lobe
    height runs from the outermost arm crossing to the corresponding z-extreme
    tip of the final curve.  The returned terminal height is the mean of the
    independently measured top and bottom values, and mismatch is terminal
    minus middle.  The x coordinate does not enter a z-span measurement, but
    the lobe partition itself is defined by the fixed xz crossing diagram.
    """
    nearest_integer = int(round(float(target_writhe)))
    if (
        abs(float(target_writhe) - nearest_integer) > 1.0e-10
        or abs(nearest_integer) < 2
    ):
        return None, None, None, None, None

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise ValueError("Lobe-height evaluation needs at least four 3D points.")
    if not np.all(np.isfinite(pts)) or not math.isfinite(stem_height_scaled):
        raise ValueError("Lobe-height evaluation received non-finite geometry.")

    lobe_count = abs(nearest_integer)
    center_z = float(np.mean(pts[:, 2]))
    outer_crossing_offset = float(stem_height_scaled) * (
        0.5 - 0.5 / float(lobe_count)
    )
    top_crossing_z = center_z + outer_crossing_offset
    bottom_crossing_z = center_z - outer_crossing_offset
    top_height = float(np.max(pts[:, 2]) - top_crossing_z)
    bottom_height = float(bottom_crossing_z - np.min(pts[:, 2]))
    terminal_height = 0.5 * (top_height + bottom_height)
    middle_height = float(stem_height_scaled) / float(lobe_count)
    mismatch = terminal_height - middle_height
    if not all(
        math.isfinite(value)
        for value in (top_height, bottom_height, terminal_height, middle_height, mismatch)
    ):
        raise ValueError("Lobe-height evaluation produced a non-finite value.")
    return top_height, bottom_height, terminal_height, middle_height, mismatch


def quantize_points_for_xyz(points: PointArray, precision: int) -> PointArray:
    """Round coordinates exactly as the plain-XYZ writer will serialize them."""
    precision = int(precision)
    value_format = "{{:.{0}f}}".format(precision)
    pts = np.asarray(points, dtype=float)
    return np.asarray(
        [[float(value_format.format(value)) for value in row] for row in pts],
        dtype=float,
    )


def resample_closed_curve(points: PointArray, num_points: int) -> PointArray:
    """Return unique periodic samples evenly spaced by polyline arc length."""
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= 1.0e-12:
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError("At least three distinct points are needed for a closed curve.")

    extended = np.vstack((pts, pts[0]))
    segment_lengths = np.linalg.norm(np.diff(extended, axis=0), axis=1)
    keep_segments = segment_lengths > 1.0e-14
    if not np.all(keep_segments):
        keep_points = np.concatenate(([True], keep_segments[:-1]))
        pts = pts[keep_points]
        extended = np.vstack((pts, pts[0]))
        segment_lengths = np.linalg.norm(np.diff(extended, axis=0), axis=1)

    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1.0e-14:
        raise ValueError("The canonical curve has zero contour length.")

    queries = np.arange(int(num_points), dtype=float) * (total / float(num_points))
    return np.column_stack(
        [np.interp(queries, cumulative, extended[:, axis]) for axis in range(3)]
    )


def smooth_curve_for_output(points: PointArray, num_points: int) -> PointArray:
    """Smooth a closed curve once and return arc-length-spaced output samples."""
    if smooth_closed_points is None:
        raise RuntimeError("Curve It smoothing support is unavailable; install SciPy.")
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= 1.0e-8:
        pts = pts[:-1]
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise ValueError("Smoothing needs at least four closed-curve points.")
    if not np.all(np.isfinite(pts)):
        raise ValueError("Smoothing received non-finite coordinates.")
    smoothed = np.asarray(smooth_closed_points(pts), dtype=float)
    return resample_closed_curve(smoothed, int(num_points))


def evaluate_curve_it_writhe(points: PointArray, smooth: bool = False) -> float:
    """Evaluate the mapped polyline's writhe, optionally smoothing it once."""
    if calculate_polyline_writhe is None or smooth_closed_points is None:
        raise RuntimeError("Curve It writhe support is unavailable; install SciPy.")
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= 1.0e-8:
        pts = pts[:-1]
    evaluation_points = (
        smooth_curve_for_output(pts, len(pts)) if smooth else pts
    )
    writhe = float(calculate_polyline_writhe(evaluation_points))
    if not math.isfinite(writhe):
        raise ValueError("Writhe evaluation produced a non-finite value.")
    return writhe


def evaluate_search_writhe(points: PointArray) -> float:
    """Return a fast writhe estimate used only inside shape optimization."""
    if calculate_writhe is None:
        raise RuntimeError("Spline writhe support is unavailable; install SciPy.")
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= 1.0e-8:
        pts = pts[:-1]
    evaluation_points = smooth_curve_for_output(pts, len(pts))
    return float(calculate_writhe(evaluation_points, n_samples=WRITHE_SEARCH_SAMPLES))


def evaluate_curvature_metrics(
    points: PointArray,
    n_samples: int = CURVATURE_EVALUATION_SAMPLES,
    adaptive_total: bool = True,
    smooth: bool = True,
) -> Tuple[float, float, float]:
    """Return total curvature, peak curvature, and reduced bending energy.

    When ``smooth`` is true, Generate SC first uses the same closed
    Savitzky-Golay smoothing as Curve It.  Final output verification passes
    ``smooth=False`` because the written curve has already been smoothed once.
    By default, total curvature uses Curve It's adaptive spline integral; the
    angle search uses a deterministic dense-grid approximation to avoid
    quadrature noise between neighboring candidates.  The local maximum uses
    a dense grid plus peak refinement.  Total curvature is dimensionless
    (radians); local curvature and ``integral kappa(s)^2 ds`` have
    inverse-coordinate units.  For constant bending rigidity ``A``, the
    physical bending energy is ``A/2`` times the latter integral.
    """
    if build_periodic_splines is None or smooth_closed_points is None:
        raise RuntimeError("Curve It curvature support is unavailable; install SciPy.")

    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= 1.0e-8:
        pts = pts[:-1]
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise ValueError("Curvature evaluation needs at least four closed-curve points.")
    if not np.all(np.isfinite(pts)):
        raise ValueError("Curvature evaluation received non-finite coordinates.")

    evaluation_points = smooth_closed_points(pts) if smooth else pts
    spline_x, spline_y, spline_z = build_periodic_splines(evaluation_points)
    sample_count = max(
        int(n_samples),
        min(32000, 8 * len(evaluation_points)),
    )
    parameter = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    velocity = np.column_stack(
        (
            spline_x(parameter, 1),
            spline_y(parameter, 1),
            spline_z(parameter, 1),
        )
    )
    acceleration = np.column_stack(
        (
            spline_x(parameter, 2),
            spline_y(parameter, 2),
            spline_z(parameter, 2),
        )
    )
    speed = np.linalg.norm(velocity, axis=1)
    if not np.all(np.isfinite(speed)) or np.any(speed <= 1.0e-14):
        raise ValueError("Curvature evaluation found a zero or non-finite spline speed.")
    local_curvature = np.linalg.norm(
        np.cross(velocity, acceleration), axis=1
    ) / (speed ** 3)
    curvature_integrand = local_curvature * speed
    bending_energy_integrand = local_curvature ** 2 * speed
    # The spline parameter spans the unit periodic interval.  Its uniform
    # periodic trapezoidal rule is therefore the arithmetic mean of the
    # sampled integrand, with ds = speed du.
    bending_energy_integral = float(np.mean(bending_energy_integrand))
    if adaptive_total:
        if fit_spline_and_calculate_curvature is None:
            raise RuntimeError("Curve It total-curvature support is unavailable; install SciPy.")
        total_curvature = float(fit_spline_and_calculate_curvature(evaluation_points))
    else:
        # The deterministic dense-grid integral is used inside the angle search
        # to avoid adaptive-quadrature noise between neighboring candidates.
        total_curvature = float(np.mean(curvature_integrand))

    # Very shallow plectonemes can concentrate curvature into narrow end-loop
    # neighborhoods.  Refine around the largest coarse samples so the reported
    # maximum is not sensitive to the phase of the uniform evaluation grid.
    peak_candidate_count = min(24, sample_count)
    peak_indices = np.argpartition(
        local_curvature,
        -peak_candidate_count,
    )[-peak_candidate_count:]
    local_offsets = np.linspace(-1.0, 1.0, 33) / float(sample_count)
    refined_parameter = (
        parameter[peak_indices, None] + local_offsets[None, :]
    ).reshape(-1) % 1.0
    refined_velocity = np.column_stack(
        (
            spline_x(refined_parameter, 1),
            spline_y(refined_parameter, 1),
            spline_z(refined_parameter, 1),
        )
    )
    refined_acceleration = np.column_stack(
        (
            spline_x(refined_parameter, 2),
            spline_y(refined_parameter, 2),
            spline_z(refined_parameter, 2),
        )
    )
    refined_speed = np.linalg.norm(refined_velocity, axis=1)
    refined_curvature = np.linalg.norm(
        np.cross(refined_velocity, refined_acceleration), axis=1
    ) / (refined_speed ** 3)
    maximum_local_curvature = float(
        max(np.max(local_curvature), np.max(refined_curvature))
    )
    if (
        not math.isfinite(total_curvature)
        or not math.isfinite(maximum_local_curvature)
        or not math.isfinite(bending_energy_integral)
    ):
        raise ValueError("Curvature evaluation produced a non-finite value.")
    return total_curvature, maximum_local_curvature, bending_energy_integral


def _count_projected_crossings(projected: PointArray) -> int:
    """Count proper nonadjacent segment crossings in a 2D projection."""
    xy = np.asarray(projected, dtype=float)
    if len(xy) > 1 and np.linalg.norm(xy[-1] - xy[0]) <= 1.0e-10:
        xy = xy[:-1]
    if len(xy) < 4:
        return 0
    span = float(np.max(np.ptp(xy, axis=0)))
    if span <= 1.0e-14:
        return 0
    xy = xy / span

    starts = xy
    ends = np.roll(xy, -1, axis=0)
    directions = ends - starts
    n = len(xy)
    crossing_count = 0
    endpoint_tol = 1.0e-7
    parallel_tol = 1.0e-12

    for i in range(n):
        first_j = i + 2
        last_j_exclusive = n
        if i == 0:
            # Segment n-1 is adjacent to segment 0 across the periodic seam.
            last_j_exclusive = n - 1
        if first_j >= last_j_exclusive:
            continue

        js = np.arange(first_j, last_j_exclusive)
        r = directions[i]
        s = directions[js]
        delta = starts[js] - starts[i]
        denominator = r[0] * s[:, 1] - r[1] * s[:, 0]
        valid = np.abs(denominator) > parallel_tol
        if not np.any(valid):
            continue

        t_num = delta[:, 0] * s[:, 1] - delta[:, 1] * s[:, 0]
        u_num = delta[:, 0] * r[1] - delta[:, 1] * r[0]
        t_param = np.zeros_like(denominator)
        u_param = np.zeros_like(denominator)
        t_param[valid] = t_num[valid] / denominator[valid]
        u_param[valid] = u_num[valid] / denominator[valid]
        proper = (
            valid
            & (t_param > endpoint_tol)
            & (t_param < 1.0 - endpoint_tol)
            & (u_param > endpoint_tol)
            & (u_param < 1.0 - endpoint_tol)
        )
        crossing_count += int(np.count_nonzero(proper))

    return crossing_count


def analyze_pca_projection(points: PointArray) -> Tuple[int, str]:
    """Return the crossing count and nearest Cartesian name of the PC1-PC2 plane."""
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= 1.0e-10:
        pts = pts[:-1]
    if len(pts) < 4:
        return 0, "degenerate"

    centered = pts - np.mean(pts, axis=0)
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] <= 1.0e-14:
        return 0, "degenerate"

    # PC3 is normal to the PC1-PC2 plane.  Name the plane by the Cartesian
    # axis most closely aligned with that normal: x -> yz, y -> xz, z -> xy.
    normal_axis = int(np.argmax(np.abs(vh[2])))
    plane_name = ("yz", "xz", "xy")[normal_axis]
    projected = centered @ vh[:2].T
    return _count_projected_crossings(projected), plane_name


def count_pca_projection_crossings(points: PointArray) -> int:
    """Count crossings in the unconstrained first-two-PC projection."""
    return analyze_pca_projection(points)[0]


def count_xz_projection_crossings(points: PointArray) -> int:
    """Count crossings in the fixed educational xz projection.

    Coordinates are ordered z then x so the diagram orientation follows the
    default PCA plot (PC1 is approximately z and PC2 is approximately x).
    """
    pts = np.asarray(points, dtype=float)
    return _count_projected_crossings(pts[:, (2, 0)])


def _fit_loop_control(
    target_writhe: float,
    num_points: int,
    opening_angle_deg: float = DEFAULT_OPENING_ANGLE_DEG,
    tolerance: float = WRITHE_TOLERANCE,
) -> Tuple[float, PointArray, float, float, int]:
    """Fit end-loop control distance while phase/crossing count stays fixed."""
    target_magnitude = abs(float(target_writhe))
    if target_magnitude <= 1.0e-12:
        dense, stem_height = _build_dense_canonical_curve(
            0.0, DEFAULT_LOOP_CONTROL, num_points, opening_angle_deg
        )
        points = resample_closed_curve(dense, num_points)
        return DEFAULT_LOOP_CONTROL, points, evaluate_search_writhe(points), stem_height, 0

    cache: Dict[float, Tuple[float, PointArray, float, float]] = {}

    def evaluate(control: float) -> Tuple[float, PointArray, float, float]:
        key = round(float(control), 12)
        if key not in cache:
            dense, stem_height = _build_dense_canonical_curve(
                target_writhe, float(control), num_points, opening_angle_deg
            )
            candidate = resample_closed_curve(dense, num_points)
            achieved = evaluate_search_writhe(candidate)
            cache[key] = (
                abs(achieved) - target_magnitude,
                candidate,
                achieved,
                stem_height,
            )
        return cache[key]

    lower = 0.02
    upper = max(4.0, 3.0 * target_magnitude + 3.0)
    controls = np.linspace(lower, upper, 21)
    bracket: Optional[Tuple[float, float]] = None
    previous_control = float(controls[0])
    previous_value = evaluate(previous_control)[0]

    if abs(previous_value) <= tolerance * 0.25:
        value = evaluate(previous_control)
        return previous_control, value[1], value[2], value[3], 1

    evaluations = 1
    for control in controls[1:]:
        current_control = float(control)
        current_value = evaluate(current_control)[0]
        evaluations += 1
        if previous_value <= 0.0 <= current_value:
            bracket = (previous_control, current_control)
            break
        previous_control = current_control
        previous_value = current_value

    if bracket is None:
        raise RuntimeError(
            "Could not fit this writhe within the canonical plectoneme family. "
            "This opening angle is infeasible for the requested writhe and is "
            "excluded from the automatic search."
        )

    lo, hi = bracket
    lo_value = evaluate(lo)[0]
    best_control = lo
    best_data = evaluate(lo)

    for _iteration in range(40):
        mid = 0.5 * (lo + hi)
        mid_data = evaluate(mid)
        evaluations += 1
        if abs(mid_data[0]) < abs(best_data[0]):
            best_control = mid
            best_data = mid_data
        if abs(mid_data[0]) <= tolerance * 0.2:
            break
        if mid_data[0] < 0.0:
            lo = mid
            lo_value = mid_data[0]
        else:
            hi = mid
        if hi - lo <= 1.0e-8 * max(1.0, mid):
            break

    # Keep the variable live as a solver-invariant assertion for maintenance.
    if lo_value > tolerance:
        raise RuntimeError("Internal writhe bracket lost its lower bound.")

    return best_control, best_data[1], best_data[2], best_data[3], evaluations


def _fit_angle_candidate(
    total_length: float,
    target_writhe: float,
    num_points: int,
    opening_angle_deg: float,
    writhe_tolerance: float = WRITHE_TOLERANCE,
) -> _AngleCandidate:
    """Fit writhe and measure curvature/bending metrics at one opening angle."""
    (
        loop_control,
        canonical,
        achieved_writhe,
        stem_height,
        solver_iterations,
    ) = _fit_loop_control(
        target_writhe,
        num_points,
        float(opening_angle_deg),
        tolerance=float(writhe_tolerance),
    )
    smoothed_canonical = smooth_curve_for_output(canonical, num_points)
    smoothed_length = closed_polyline_length(smoothed_canonical)
    scale = float(total_length) / smoothed_length
    scaled_points = smoothed_canonical * scale
    scaled_points -= np.mean(scaled_points, axis=0)
    (
        total_curvature,
        maximum_local_curvature,
        bending_energy_integral,
    ) = evaluate_curvature_metrics(
        scaled_points,
        adaptive_total=False,
        smooth=False,
    )
    (
        top_terminal_lobe_height_xz,
        bottom_terminal_lobe_height_xz,
        terminal_lobe_height_xz,
        middle_lobe_height_xz,
        lobe_height_mismatch_xz,
    ) = evaluate_xz_lobe_heights(
        scaled_points,
        target_writhe,
        stem_height * scale,
    )
    return _AngleCandidate(
        angle_deg=float(opening_angle_deg),
        loop_control=loop_control,
        canonical_points=canonical,
        achieved_writhe=achieved_writhe,
        stem_height=stem_height,
        writhe_solver_iterations=solver_iterations,
        total_curvature=total_curvature,
        maximum_local_curvature=maximum_local_curvature,
        bending_energy_integral=bending_energy_integral,
        top_terminal_lobe_height_xz=top_terminal_lobe_height_xz,
        bottom_terminal_lobe_height_xz=bottom_terminal_lobe_height_xz,
        terminal_lobe_height_xz=terminal_lobe_height_xz,
        middle_lobe_height_xz=middle_lobe_height_xz,
        lobe_height_mismatch_xz=lobe_height_mismatch_xz,
    )


def _optimize_opening_angle(
    total_length: float,
    target_writhe: float,
    num_points: int,
    curvature_objective: str,
) -> Tuple[_AngleCandidate, int]:
    """Choose the best feasible opening angle for the requested objective.

    A coarse global grid prevents a local search from becoming trapped on the
    wrong feasible branch.  Three deterministic local grid refinements then
    resolve the selected angle to substantially better than 0.1 degree.  The
    bending-energy objective minimizes ``integral kappa(s)^2 ds`` after every
    candidate is scaled to the requested common contour length.  Equal-lobes
    minimizes the absolute relative difference between terminal and middle
    z-spans in the fixed xz projection.  Any angle at which the inner writhe
    fit fails is treated as infeasible.
    """
    objective = str(curvature_objective)
    if objective not in CURVATURE_OBJECTIVES:
        raise ValueError("Unknown curvature objective: {0}".format(objective))

    cache: Dict[float, Optional[_AngleCandidate]] = {}

    def evaluate(angle_deg: float) -> Optional[_AngleCandidate]:
        clipped_angle = min(
            MAX_AUTO_OPENING_ANGLE_DEG,
            max(MIN_AUTO_OPENING_ANGLE_DEG, float(angle_deg)),
        )
        key = round(clipped_angle, 10)
        if key in cache:
            return cache[key]
        try:
            candidate = _fit_angle_candidate(
                total_length,
                target_writhe,
                num_points,
                clipped_angle,
                writhe_tolerance=(
                    EQUAL_LOBE_WRITHE_FIT_TOLERANCE
                    if objective == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES
                    else (
                        BENDING_ENERGY_WRITHE_FIT_TOLERANCE
                        if objective == CURVATURE_OBJECTIVE_BENDING_ENERGY
                        else WRITHE_TOLERANCE
                    )
                ),
            )
        except (RuntimeError, ValueError, FloatingPointError):
            candidate = None
        cache[key] = candidate
        return candidate

    def objective_value(candidate: _AngleCandidate) -> float:
        if objective == CURVATURE_OBJECTIVE_TOTAL:
            return candidate.total_curvature
        if objective == CURVATURE_OBJECTIVE_MAX_LOCAL:
            return candidate.maximum_local_curvature
        if objective == CURVATURE_OBJECTIVE_BENDING_ENERGY:
            return candidate.bending_energy_integral
        if (
            candidate.lobe_height_mismatch_xz is None
            or candidate.middle_lobe_height_xz is None
            or candidate.middle_lobe_height_xz <= 0.0
        ):
            return math.inf
        return abs(candidate.lobe_height_mismatch_xz) / candidate.middle_lobe_height_xz

    # W = 0 has no helical phase sweep, so the opening-angle term vanishes
    # identically.  Keep the historical 25-degree reference rather than
    # claiming that a numerically arbitrary angle is better.
    if abs(float(target_writhe)) <= 1.0e-12:
        zero_candidate = evaluate(DEFAULT_OPENING_ANGLE_DEG)
        if zero_candidate is None:
            raise RuntimeError("Could not construct the zero-writhe reference curve.")
        return zero_candidate, len(cache)

    initial_angles = np.linspace(
        MIN_AUTO_OPENING_ANGLE_DEG,
        MAX_AUTO_OPENING_ANGLE_DEG,
        OPENING_ANGLE_GRID_SIZE,
    )
    feasible = [candidate for candidate in (evaluate(angle) for angle in initial_angles) if candidate]
    if not feasible:
        raise RuntimeError(
            "No feasible opening angle was found in the supported automatic "
            "search interval {0:g}-{1:g} degrees.".format(
                MIN_AUTO_OPENING_ANGLE_DEG,
                MAX_AUTO_OPENING_ANGLE_DEG,
            )
        )

    def candidate_key(candidate: _AngleCandidate) -> Tuple[float, float, float]:
        # The second term gives deterministic, visually familiar tie-breaking
        # if the sampled objective is numerically flat.
        return (
            objective_value(candidate),
            abs(candidate.angle_deg - DEFAULT_OPENING_ANGLE_DEG),
            candidate.angle_deg,
        )

    best = min(feasible, key=candidate_key)
    step = (MAX_AUTO_OPENING_ANGLE_DEG - MIN_AUTO_OPENING_ANGLE_DEG) / float(
        OPENING_ANGLE_GRID_SIZE - 1
    )
    for _level in range(OPENING_ANGLE_REFINEMENT_LEVELS):
        lower = max(MIN_AUTO_OPENING_ANGLE_DEG, best.angle_deg - step)
        upper = min(MAX_AUTO_OPENING_ANGLE_DEG, best.angle_deg + step)
        refinement_angles = np.linspace(
            lower,
            upper,
            OPENING_ANGLE_REFINEMENT_SAMPLES,
        )
        refined = [
            candidate
            for candidate in (evaluate(angle) for angle in refinement_angles)
            if candidate
        ]
        if refined:
            best = min((best, *refined), key=candidate_key)
        step = max(
            1.0e-4,
            (upper - lower) / float(OPENING_ANGLE_REFINEMENT_SAMPLES - 1),
        )

    # Equality is a root condition rather than merely a shallow minimum.
    # When feasible samples bracket a sign change, finish with bisection on
    # terminal-minus-middle height so "equal" is accurate beyond the generic
    # angular grid resolution.
    if objective == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES:
        signed_candidates = sorted(
            (
                candidate
                for candidate in cache.values()
                if candidate is not None
                and candidate.lobe_height_mismatch_xz is not None
            ),
            key=lambda candidate: candidate.angle_deg,
        )
        brackets = []
        for lower_candidate, upper_candidate in zip(
            signed_candidates[:-1],
            signed_candidates[1:],
        ):
            lower_mismatch = float(lower_candidate.lobe_height_mismatch_xz)
            upper_mismatch = float(upper_candidate.lobe_height_mismatch_xz)
            if lower_mismatch == 0.0 or lower_mismatch * upper_mismatch <= 0.0:
                brackets.append((lower_candidate, upper_candidate))
        if brackets:
            lo_candidate, hi_candidate = min(
                brackets,
                key=lambda pair: min(candidate_key(pair[0]), candidate_key(pair[1])),
            )
            lo_mismatch = float(lo_candidate.lobe_height_mismatch_xz)
            for _iteration in range(16):
                mid_candidate = evaluate(
                    0.5 * (lo_candidate.angle_deg + hi_candidate.angle_deg)
                )
                if mid_candidate is None or mid_candidate.lobe_height_mismatch_xz is None:
                    break
                best = min((best, mid_candidate), key=candidate_key)
                mid_mismatch = float(mid_candidate.lobe_height_mismatch_xz)
                if objective_value(mid_candidate) <= 1.0e-6:
                    break
                if lo_mismatch * mid_mismatch <= 0.0:
                    hi_candidate = mid_candidate
                else:
                    lo_candidate = mid_candidate
                    lo_mismatch = mid_mismatch

    return best, len(cache)


def generate_sc_points(
    total_length: float,
    target_writhe: float,
    num_points: int = DEFAULT_NUM_POINTS,
    precision: int = DEFAULT_PRECISION,
    curvature_objective: str = DEFAULT_CURVATURE_OBJECTIVE,
    opening_angle_deg: Optional[float] = None,
) -> SCGenerationResult:
    """Generate an automatically optimized or manually angled plectoneme.

    ``num_points`` is the number of unique periodic samples written.  The first
    row is not duplicated at the end; closed length always includes the seam.
    When ``opening_angle_deg`` is supplied, it overrides automatic curvature
    optimization.  Curvature and bending-energy metrics are still measured
    and reported.
    """
    validate_inputs(
        total_length,
        target_writhe,
        num_points,
        precision,
        curvature_objective,
        opening_angle_deg,
    )
    num_points = int(num_points)

    effective_objective = str(curvature_objective)
    if opening_angle_deg is None:
        angle_candidate, angle_evaluations = _optimize_opening_angle(
            total_length,
            target_writhe,
            num_points,
            curvature_objective,
        )
    else:
        angle_candidate = _fit_angle_candidate(
            total_length,
            target_writhe,
            num_points,
            float(opening_angle_deg),
        )
        angle_evaluations = 1
        effective_objective = OPENING_ANGLE_MODE_MANUAL
    loop_control = angle_candidate.loop_control
    canonical = angle_candidate.canonical_points
    stem_height = angle_candidate.stem_height
    iterations = angle_candidate.writhe_solver_iterations
    # Write the same once-smoothed geometry used by the writhe search. Rescale
    # after smoothing because smoothing shortens the closed polyline slightly.
    smoothed_canonical = smooth_curve_for_output(canonical, num_points)
    smoothed_length = closed_polyline_length(smoothed_canonical)
    scale = float(total_length) / smoothed_length
    points = smoothed_canonical * scale
    points -= np.mean(points, axis=0)

    # Verify the exact decimal coordinates that will be written, not only the
    # full-precision in-memory fit. Low precision can otherwise collapse
    # neighboring samples or change length/crossings after a false PASS.
    points = quantize_points_for_xyz(points, precision)
    segment_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    unique_count = int(len(np.unique(points, axis=0)))
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(segment_lengths)):
        raise ValueError("Serialized curve coordinates are outside the numeric range.")
    if unique_count != num_points or np.any(segment_lengths <= 0.0):
        raise ValueError(
            "Output precision collapses distinct curve samples ({0} of {1} remain unique). "
            "Increase --precision or use a larger contour length.".format(
                unique_count, num_points
            )
        )

    achieved_length = closed_polyline_length(points)
    # The serialized points have already been smoothed once. Verify the exact
    # closed polyline that Curve It will use for downstream mapping.
    achieved_writhe = evaluate_curve_it_writhe(points, smooth=False)
    (
        total_curvature,
        maximum_local_curvature,
        bending_energy_integral,
    ) = evaluate_curvature_metrics(
        points,
        smooth=False,
    )
    (
        top_terminal_lobe_height_xz,
        bottom_terminal_lobe_height_xz,
        terminal_lobe_height_xz,
        middle_lobe_height_xz,
        lobe_height_mismatch_xz,
    ) = evaluate_xz_lobe_heights(
        points,
        target_writhe,
        stem_height * scale,
    )
    xz_crossings = count_xz_projection_crossings(points)
    pca_crossings, pca_plane = analyze_pca_projection(points)
    landmarks = analyze_contour_landmarks(
        points,
        target_writhe=target_writhe,
        radius_scaled=CANONICAL_RADIUS * scale,
        stem_height_scaled=stem_height * scale,
    )

    if not math.isfinite(achieved_length) or not math.isfinite(achieved_writhe):
        raise ValueError("Serialized curve metrics are outside the numeric range.")

    length_tolerance = max(1.0e-9, total_length * 1.0e-6)
    if abs(achieved_length - total_length) > length_tolerance:
        raise ValueError(
            "Output precision changes the requested contour length too much: "
            "target={0:.10g}, serialized={1:.10g}. Increase --precision.".format(
                total_length, achieved_length
            )
        )
    if abs(achieved_writhe - target_writhe) > WRITHE_TOLERANCE:
        raise ValueError(
            "Serialized-coordinate writhe is outside tolerance: "
            "target={0:.6g}, achieved={1:.6g}. Increase --precision.".format(
                target_writhe, achieved_writhe
            )
        )

    nearest_integer = int(round(target_writhe))
    is_integer_request = abs(target_writhe - nearest_integer) <= 1.0e-10
    if is_integer_request and xz_crossings != abs(nearest_integer):
        raise ValueError(
            "Serialized-coordinate xz crossing verification failed: expected {0}, "
            "found {1}. Increase --precision.".format(
                abs(nearest_integer), xz_crossings
            )
        )

    if effective_objective == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES:
        if (
            lobe_height_mismatch_xz is None
            or middle_lobe_height_xz is None
            or middle_lobe_height_xz <= 0.0
        ):
            raise ValueError("Equal-lobes verification could not measure the xz lobe heights.")
        relative_lobe_mismatch = abs(lobe_height_mismatch_xz) / middle_lobe_height_xz
        if relative_lobe_mismatch > EQUAL_LOBE_RELATIVE_TOLERANCE:
            raise ValueError(
                "Equal-lobes verification failed: terminal and middle fixed-xz "
                "z-heights differ by {0:.4%}, above the {1:.4%} tolerance.".format(
                    relative_lobe_mismatch,
                    EQUAL_LOBE_RELATIVE_TOLERANCE,
                )
            )

    return SCGenerationResult(
        points=points,
        requested_length=float(total_length),
        achieved_length=achieved_length,
        requested_writhe=float(target_writhe),
        achieved_writhe=achieved_writhe,
        xz_crossings=xz_crossings,
        pca_crossings=pca_crossings,
        pca_plane=pca_plane,
        superhelical_turns=abs(float(target_writhe)) / 2.0,
        opening_angle_deg=angle_candidate.angle_deg,
        curvature_objective=effective_objective,
        total_curvature=total_curvature,
        maximum_local_curvature=maximum_local_curvature,
        bending_energy_integral=bending_energy_integral,
        top_terminal_lobe_height_xz=top_terminal_lobe_height_xz,
        bottom_terminal_lobe_height_xz=bottom_terminal_lobe_height_xz,
        terminal_lobe_height_xz=terminal_lobe_height_xz,
        middle_lobe_height_xz=middle_lobe_height_xz,
        lobe_height_mismatch_xz=lobe_height_mismatch_xz,
        opening_angle_evaluations=angle_evaluations,
        loop_control_canonical=loop_control,
        radius_scaled=CANONICAL_RADIUS * scale,
        stem_height_scaled=stem_height * scale,
        solver_iterations=iterations,
        landmarks=landmarks,
    )


def write_plain_xyz(points: PointArray, output_path: str, precision: int = DEFAULT_PRECISION) -> None:
    """Write one plain ``x y z`` coordinate row per point."""
    if int(precision) < 0 or int(precision) > 15:
        raise ValueError("Precision must be between 0 and 15 decimal places.")
    path = Path(output_path)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "{{:.{0}f}} {{:.{0}f}} {{:.{0}f}}\n".format(int(precision))
    with path.open("w", encoding="utf-8") as handle:
        for x, y, z in np.asarray(points, dtype=float):
            handle.write(fmt.format(x, y, z))


def generation_summary(result: SCGenerationResult) -> str:
    """Return a compact report for the command line and GUI."""
    residual = result.achieved_writhe - result.requested_writhe
    nearest_integer = int(round(result.requested_writhe))
    is_integer_request = abs(result.requested_writhe - nearest_integer) <= 1.0e-10
    if is_integer_request:
        crossing_check = "PASS (required |W| = {0})".format(abs(nearest_integer))
    else:
        crossing_check = "not constrained for fractional W"
    if result.pca_plane == "yz":
        pca_note = "NOTE: PCA selected yz; its crossing count may differ from fixed xz."
    elif result.pca_plane == "xz":
        pca_note = "PCA selected xz, the same plane as the educational check."
    else:
        pca_note = "PCA selected {0}; it is independent of the fixed xz check.".format(
            result.pca_plane
        )
    handedness = "planar / no supercoiling"
    if result.requested_writhe > 0.0:
        handedness = "positive-writhe mirror"
    elif result.requested_writhe < 0.0:
        handedness = "negative-writhe mirror"
    if result.curvature_objective == CURVATURE_OBJECTIVE_TOTAL:
        objective_label = "automatic: minimize total curvature"
        angle_search_interval = "{0:g} to {1:g} deg".format(
            MIN_AUTO_OPENING_ANGLE_DEG,
            MAX_AUTO_OPENING_ANGLE_DEG,
        )
    elif result.curvature_objective == CURVATURE_OBJECTIVE_MAX_LOCAL:
        objective_label = "automatic: minimize largest local curvature (default)"
        angle_search_interval = "{0:g} to {1:g} deg".format(
            MIN_AUTO_OPENING_ANGLE_DEG,
            MAX_AUTO_OPENING_ANGLE_DEG,
        )
    elif result.curvature_objective == CURVATURE_OBJECTIVE_BENDING_ENERGY:
        objective_label = "automatic: minimize reduced bending energy (integral kappa^2 ds)"
        angle_search_interval = "{0:g} to {1:g} deg".format(
            MIN_AUTO_OPENING_ANGLE_DEG,
            MAX_AUTO_OPENING_ANGLE_DEG,
        )
    elif result.curvature_objective == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES:
        objective_label = "automatic: equal terminal/middle lobe z-heights in fixed xz"
        angle_search_interval = "{0:g} to {1:g} deg".format(
            MIN_AUTO_OPENING_ANGLE_DEG,
            MAX_AUTO_OPENING_ANGLE_DEG,
        )
    else:
        objective_label = "manual: use user-provided opening angle"
        angle_search_interval = "not used for manual angle"
    if result.curvature_objective == OPENING_ANGLE_MODE_MANUAL:
        angle_note = "manual angle retained; automatic optimization bypassed"
    elif abs(result.requested_writhe) <= 1.0e-12:
        angle_note = "W = 0 makes angle irrelevant; using 25-deg reference"
    elif result.opening_angle_deg <= MIN_AUTO_OPENING_ANGLE_DEG + 0.02:
        angle_note = "lower supported bound selected; zero angle is excluded"
    elif result.opening_angle_deg >= MAX_AUTO_OPENING_ANGLE_DEG - 0.02:
        angle_note = "upper supported bound selected; 90 degrees is excluded"
    elif result.curvature_objective == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES:
        angle_note = "minimized projected terminal-minus-middle z-height mismatch"
    else:
        angle_note = "best feasible angle found by deterministic refinement"
    minimum_bend_radius = math.inf
    if result.maximum_local_curvature > 0.0:
        minimum_bend_radius = 1.0 / result.maximum_local_curvature

    if (
        result.top_terminal_lobe_height_xz is not None
        and result.bottom_terminal_lobe_height_xz is not None
        and result.terminal_lobe_height_xz is not None
        and result.middle_lobe_height_xz is not None
        and result.lobe_height_mismatch_xz is not None
    ):
        relative_lobe_mismatch = (
            result.lobe_height_mismatch_xz / result.middle_lobe_height_xz
            if result.middle_lobe_height_xz > 0.0
            else math.nan
        )
        if result.curvature_objective == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES:
            lobe_check = "PASS (relative mismatch <= {0:.4%})".format(
                EQUAL_LOBE_RELATIVE_TOLERANCE
            )
        else:
            lobe_check = "diagnostic only for this opening-angle mode"
        lobe_height_lines = (
            "Fixed-xz top terminal lobe z-height    = {0:.10g}".format(
                result.top_terminal_lobe_height_xz
            ),
            "Fixed-xz bottom terminal lobe z-height = {0:.10g}".format(
                result.bottom_terminal_lobe_height_xz
            ),
            "Fixed-xz mean terminal lobe z-height   = {0:.10g}".format(
                result.terminal_lobe_height_xz
            ),
            "Fixed-xz middle lobe z-height          = {0:.10g}".format(
                result.middle_lobe_height_xz
            ),
            "Fixed-xz terminal-minus-middle height  = {0:+.6g} ({1:+.4%})".format(
                result.lobe_height_mismatch_xz,
                relative_lobe_mismatch,
            ),
            "Fixed-xz equal-lobes check              = {0}".format(lobe_check),
        )
    else:
        lobe_height_lines = (
            "Fixed-xz lobe z-heights = N/A (requires integer |W| >= 2)",
        )

    def landmark_line(label: str, landmark: ContourLandmark) -> str:
        return "{0:<31} = {1:.6f}% (XYZ row {2})".format(
            label,
            landmark.contour_percent,
            landmark.point_index + 1,
        )

    landmark_lines = [
        "Contour landmarks: percent of closed length from XYZ row 1",
        "Landmark convention: termini are z tips; middle segments run top-to-bottom",
        landmark_line("Top terminus (+z tip)", result.landmarks.top_terminus),
        landmark_line("Bottom terminus (-z tip)", result.landmarks.bottom_terminus),
    ]
    if not is_integer_request:
        landmark_lines.append("Middle-segment peaks = not enumerated for fractional W")
    elif not result.landmarks.middle_segment_peaks:
        landmark_lines.append("Middle-segment peaks = none (|W| <= 1)")
    else:
        for segment_number, (peak_a, peak_b) in enumerate(
            result.landmarks.middle_segment_peaks, start=1
        ):
            landmark_lines.append(
                landmark_line("Middle segment {0} peak A".format(segment_number), peak_a)
            )
            landmark_lines.append(
                landmark_line("Middle segment {0} peak B".format(segment_number), peak_b)
            )

    return "\n".join(
        (
            "Requested closed contour length = {0:.10g}".format(result.requested_length),
            "Achieved closed contour length  = {0:.10g}".format(result.achieved_length),
            "Requested Gauss writhe = {0:.10g}".format(result.requested_writhe),
            "Achieved Gauss writhe  = {0:.10g}".format(result.achieved_writhe),
            "Writhe residual        = {0:+.3e}".format(residual),
            "Fixed xz projection crossings = {0}".format(result.xz_crossings),
            "Integer xz crossing check      = {0}".format(crossing_check),
            "Projection convention = fixed xz has |W| crossings for integer W",
            "PCA principal plane    = {0}".format(result.pca_plane),
            "PCA-plane crossings    = {0}".format(result.pca_crossings),
            "PCA plane note         = {0}".format(pca_note),
            "Superhelical turns       = {0:.8g}".format(result.superhelical_turns),
            "Opening-angle selection  = {0}".format(objective_label),
            "Selected opening angle   = {0:.8g} deg".format(result.opening_angle_deg),
            "Angle search interval    = {0}".format(angle_search_interval),
            "Angle candidates evaluated = {0}".format(result.opening_angle_evaluations),
            "Angle optimization note    = {0}".format(angle_note),
            "Total curvature           = {0:.10g} rad ({1:.8g} * pi)".format(
                result.total_curvature,
                result.total_curvature / math.pi,
            ),
            "Largest local curvature   = {0:.10g} inverse length".format(
                result.maximum_local_curvature
            ),
            "Reduced bending energy    = {0:.10g} inverse length (integral kappa^2 ds)".format(
                result.bending_energy_integral
            ),
            "Minimum local bend radius = {0:.10g} length units".format(
                minimum_bend_radius
            ),
            "Output geometry           = periodically smoothed once, then rescaled",
            "Writhe convention         = exact written closed polyline used by Curve It",
            "Curvature convention       = periodic spline of written coordinates",
            *lobe_height_lines,
            "Scaled superhelix radius = {0:.8g}".format(result.radius_scaled),
            "Scaled stem height       = {0:.8g}".format(result.stem_height_scaled),
            "Mirror / handedness      = {0}".format(handedness),
            "Writhe solver evaluations = {0}".format(result.solver_iterations),
            *landmark_lines,
        )
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a closed textbook-style plectonemic DNA-axis curve with "
            "a requested contour length and Gauss writhe. Integer |writhe| also "
            "sets the exact fixed-xz projection crossing count."
        )
    )
    parser.add_argument(
        "-L",
        "--total-length",
        type=float,
        default=DEFAULT_TOTAL_LENGTH,
        help="Closed contour length. Default: {0:g}".format(DEFAULT_TOTAL_LENGTH),
    )
    parser.add_argument(
        "-w",
        "--writhe",
        type=float,
        default=DEFAULT_WRITHE,
        help=(
            "Target signed Gauss writhe (|W| <= {0:g}). Integer |W| also gives "
            "that many fixed-xz projection crossings. Default: {1:g}"
        ).format(MAX_ABS_WRITHE, DEFAULT_WRITHE),
    )
    parser.add_argument(
        "--angle-objective",
        "--curvature-objective",
        dest="curvature_objective",
        choices=CURVATURE_OBJECTIVES,
        default=DEFAULT_CURVATURE_OBJECTIVE,
        help=(
            "Automatic opening-angle objective: 'max-local' minimizes the "
            "largest local curvature (default); 'total' minimizes integrated "
            "curvature; 'bending-energy' minimizes integral kappa(s)^2 ds; "
            "'equal-lobes' matches terminal and middle lobe z-heights in the "
            "fixed xz projection and requires integer |W| >= 2. The older name "
            "--curvature-objective remains an alias. Ignored when --opening-angle "
            "is supplied."
        ),
    )
    parser.add_argument(
        "-a",
        "--opening-angle",
        type=float,
        default=None,
        help=(
            "Use this opening angle in degrees instead of automatic optimization; "
            "must be strictly between 0 and 90. Curvature and bending-energy "
            "metrics are still reported."
        ),
    )
    parser.add_argument(
        "-n",
        "--num-points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help="Number of unique periodic output samples. Default: {0}".format(DEFAULT_NUM_POINTS),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output plain-coordinate XYZ file. Default: {0}".format(DEFAULT_OUTPUT),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal places written to XYZ. Default: {0}".format(DEFAULT_PRECISION),
    )
    parser.add_argument("--gui", action="store_true", help="Open the graphical user interface.")
    parser.add_argument("--version", action="version", version="{0} {1}".format(TOOL_NAME, TOOL_VERSION))
    return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> SCGenerationResult:
    """Generate, write, and report a curve from parsed CLI arguments."""
    result = generate_sc_points(
        total_length=args.total_length,
        target_writhe=args.writhe,
        num_points=args.num_points,
        precision=args.precision,
        curvature_objective=args.curvature_objective,
        opening_angle_deg=args.opening_angle,
    )
    write_plain_xyz(result.points, args.output, args.precision)
    print("Wrote {0} unique periodic points to: {1}".format(len(result.points), args.output))
    print("Load this file in Curve It with path type 'closed'.")
    print(generation_summary(result))
    return result


def run_gui() -> None:
    """Open the Curve It-style Tkinter interface."""
    try:
        import tkinter as tk
        from tkinter import filedialog, font, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError("Tkinter is not available in this Python installation.") from exc

    root = tk.Tk()
    root.title("{0} {1}".format(TOOL_NAME, TOOL_VERSION))
    root.geometry("1020x790")
    root.minsize(900, 710)
    set_optional_window_icon(root, tk, ("icon.png",), "_generate_sc_icon_image")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    title_font = font.Font(root, family="Helvetica", size=16, weight="bold")
    mono_font = font.Font(root, family="Menlo", size=10)
    section_font = ("TkDefaultFont", 10, "bold")
    style.configure("Tool.TLabelframe.Label", font=section_font)
    style.configure("Hint.TLabel", foreground="gray35")

    help_button_kwargs = {
        "text": "?",
        "width": 2,
        "bg": "#cfefff",
        "activebackground": "#aee6ff",
        "relief": tk.RAISED,
        "borderwidth": 1,
    }
    help_texts = {
        "overview": (
            "Generate SC",
            "Generate one closed plectonemic centerline for a supercoiled DNA axis. "
            "The file can be loaded directly into Curve It using path type closed.",
        ),
        "length": (
            "Closed Contour Length",
            "Total polyline arc length including the last-to-first closing segment. "
            "Coordinates are normally interpreted as Angstrom by Curve It.",
        ),
        "writhe": (
            "Writhe And Crossings",
            "The signed value is verified from the exact closed polyline written "
            "to disk, using the same segment-pair Gauss-integral calculation that "
            "Curve It reports for mapping. For an integer W, the fixed xz diagram "
            "is verified to contain exactly |W| crossings. Sign selects the mirror "
            "image because a crossing count itself cannot be negative. The "
            "unconstrained PCA plane and its crossing count are reported separately.",
        ),
        "objective": (
            "Opening-Angle Selection",
            "The default automatically minimizes the largest local curvature and "
            "therefore reduces the sharpest bend. Total-curvature mode usually "
            "favors a shallower, more elongated plectoneme. Bending-energy mode "
            "minimizes integral kappa(s)^2 ds, which is proportional to elastic "
            "bending energy for constant bending rigidity. Equal-lobes mode matches "
            "the projected terminal and middle lobe z-spans and requires integer "
            "|W| >= 2. Automatic modes search 5 through 85 degrees and skip "
            "infeasible candidates. Manual mode retains an entered angle between 0 "
            "and 90 degrees. Every mode reports curvature, reduced bending energy, "
            "and applicable lobe metrics.",
        ),
        "points": (
            "Number Of Points",
            "Unique periodic samples written to the file; the first row is not "
            "duplicated at the end. Higher |writhe| needs more points.",
        ),
        "model": (
            "Canonical Plectoneme",
            "Length and writhe do not uniquely determine a shape. Generate SC uses "
            "a textbook-style two-arm plectoneme, automatically optimizes or retains "
            "a user-provided opening angle, adjusts its smooth end loops to fit writhe, "
            "and scales the whole curve to length. The equal-lobes metric uses z-spans "
            "between crossings and tips in the fixed xz projection. "
            "It is a geometry teaching model, not an elastic-energy simulation.",
        ),
        "output": (
            "Output File",
            "Plain x y z rows with no molecular-XYZ header. In Curve It, select "
            "path type closed so the seam is included. The report locates both "
            "termini and both centerline peaks of every interior xz lobe by XYZ "
            "row and percentage of the closed contour length from the first row.",
        ),
    }

    def help_button(parent, title: str, body: str):
        return tk.Button(parent, command=lambda: messagebox.showinfo(title, body), **help_button_kwargs)

    main = ttk.Frame(root, padding=14)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(2, weight=1)

    header_row = ttk.Frame(main)
    header_row.grid(row=0, column=0, sticky="ew")
    ttk.Label(header_row, text=TOOL_NAME, font=title_font).pack(side="left")
    help_button(header_row, *help_texts["overview"]).pack(side="left", padx=(8, 0))

    intro = (
        "Closed textbook plectoneme: two antipodal superhelical arms joined by smooth end loops.\n"
        "Integer W: |W|/2 turns -> exactly |W| crossings in fixed xz; PCA is reported separately."
    )
    ttk.Label(main, text=intro, justify="left").grid(row=1, column=0, sticky="ew", pady=(8, 10))

    body = ttk.Frame(main)
    body.grid(row=2, column=0, sticky="nsew")
    body.columnconfigure(0, weight=0)
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)

    inputs = ttk.LabelFrame(body, text="Inputs", style="Tool.TLabelframe", padding=10)
    inputs.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    inputs.columnconfigure(1, weight=1)

    length_var = tk.StringVar(value=str(DEFAULT_TOTAL_LENGTH))
    writhe_var = tk.StringVar(value=str(DEFAULT_WRITHE))
    curvature_objective_var = tk.StringVar(value=DEFAULT_CURVATURE_OBJECTIVE)
    opening_angle_var = tk.StringVar(value=str(DEFAULT_OPENING_ANGLE_DEG))
    points_var = tk.StringVar(value=str(DEFAULT_NUM_POINTS))
    precision_var = tk.StringVar(value=str(DEFAULT_PRECISION))
    output_var = tk.StringVar(value=DEFAULT_OUTPUT)
    field_specs = (
        ("Length L:", length_var, "length", "closed contour length"),
        ("Writhe W:", writhe_var, "writhe", "signed; integer |W| = crossings"),
        ("Points:", points_var, "points", "unique periodic samples"),
        ("Precision:", precision_var, None, "decimal places"),
    )
    for row, (label, variable, help_key, hint) in enumerate(field_specs):
        ttk.Label(inputs, text=label).grid(row=row, column=0, sticky="e", padx=(0, 6), pady=5)
        ttk.Entry(inputs, textvariable=variable, width=16).grid(row=row, column=1, sticky="ew", pady=5)
        if help_key is not None:
            help_button(inputs, *help_texts[help_key]).grid(row=row, column=2, padx=(5, 0), pady=5)
        ttk.Label(inputs, text=hint, style="Hint.TLabel").grid(
            row=row, column=3, sticky="w", padx=(6, 0), pady=5
        )

    objective_frame = ttk.LabelFrame(
        inputs,
        text="Opening-angle selection",
        style="Tool.TLabelframe",
        padding=8,
    )
    objective_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))
    ttk.Radiobutton(
        objective_frame,
        text="Minimize largest local curvature (default)",
        variable=curvature_objective_var,
        value=CURVATURE_OBJECTIVE_MAX_LOCAL,
    ).grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(
        objective_frame,
        text="Minimize total curvature",
        variable=curvature_objective_var,
        value=CURVATURE_OBJECTIVE_TOTAL,
    ).grid(row=1, column=0, sticky="w", pady=(3, 0))
    ttk.Radiobutton(
        objective_frame,
        text="Equal terminal and middle lobe heights in fixed xz",
        variable=curvature_objective_var,
        value=OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES,
    ).grid(row=3, column=0, sticky="w", pady=(3, 0))
    ttk.Radiobutton(
        objective_frame,
        text="Minimize bending energy (integral curvature squared)",
        variable=curvature_objective_var,
        value=CURVATURE_OBJECTIVE_BENDING_ENERGY,
    ).grid(row=2, column=0, sticky="w", pady=(3, 0))
    ttk.Radiobutton(
        objective_frame,
        text="Use provided angle:",
        variable=curvature_objective_var,
        value=OPENING_ANGLE_MODE_MANUAL,
    ).grid(row=4, column=0, sticky="w", pady=(3, 0))
    manual_angle_entry = ttk.Entry(
        objective_frame,
        textvariable=opening_angle_var,
        width=9,
    )
    manual_angle_entry.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(3, 0))
    ttk.Label(objective_frame, text="degrees").grid(
        row=4,
        column=2,
        sticky="w",
        padx=(5, 0),
        pady=(3, 0),
    )
    help_button(objective_frame, *help_texts["objective"]).grid(
        row=0,
        column=3,
        rowspan=5,
        padx=(8, 0),
        sticky="n",
    )

    def update_manual_angle_state(*_args) -> None:
        state = "normal" if curvature_objective_var.get() == OPENING_ANGLE_MODE_MANUAL else "disabled"
        manual_angle_entry.configure(state=state)

    curvature_objective_var.trace_add("write", update_manual_angle_state)
    update_manual_angle_state()

    model_frame = ttk.LabelFrame(inputs, text="Canonical model", style="Tool.TLabelframe", padding=8)
    model_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(10, 0))
    ttk.Label(
        model_frame,
        text="Opening angle: four automatic objectives or user-provided\n"
        "Gauss-writhe-fitted end loops\n|W| <= 10",
        justify="left",
    ).pack(side="left")
    help_button(model_frame, *help_texts["model"]).pack(side="left", padx=(8, 0), anchor="n")

    output_frame = ttk.LabelFrame(body, text="Output and verification", style="Tool.TLabelframe", padding=10)
    output_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
    output_frame.columnconfigure(0, weight=1)
    output_frame.rowconfigure(3, weight=1)

    output_header = ttk.Frame(output_frame)
    output_header.grid(row=0, column=0, sticky="ew")
    ttk.Label(output_header, text="Output file").pack(side="left")
    help_button(output_header, *help_texts["output"]).pack(side="left", padx=(6, 0))

    output_row = ttk.Frame(output_frame)
    output_row.grid(row=1, column=0, sticky="ew", pady=(4, 8))
    output_row.columnconfigure(0, weight=1)
    ttk.Entry(output_row, textvariable=output_var).grid(row=0, column=0, sticky="ew")

    def browse_output() -> None:
        filename = filedialog.asksaveasfilename(
            title="Save plectonemic curve",
            defaultextension=".xyz",
            filetypes=(("XYZ files", "*.xyz"), ("Text files", "*.txt"), ("All files", "*.*")),
        )
        if filename:
            output_var.set(filename)

    ttk.Button(output_row, text="Browse...", command=browse_output).grid(row=0, column=1, padx=(6, 0))

    ttk.Label(output_frame, text="Derived values and checks").grid(row=2, column=0, sticky="w")
    summary_frame = ttk.Frame(output_frame)
    summary_frame.grid(row=3, column=0, sticky="nsew", pady=(4, 0))
    summary_frame.columnconfigure(0, weight=1)
    summary_frame.rowconfigure(0, weight=1)
    summary_box = tk.Text(
        summary_frame,
        wrap="none",
        relief="groove",
        padx=8,
        pady=8,
        bg="white",
        font=mono_font,
        cursor="xterm",
        takefocus=True,
        exportselection=True,
    )
    summary_box.grid(row=0, column=0, sticky="nsew")
    summary_y_scroll = ttk.Scrollbar(
        summary_frame,
        orient="vertical",
        command=summary_box.yview,
    )
    summary_y_scroll.grid(row=0, column=1, sticky="ns")
    summary_x_scroll = ttk.Scrollbar(
        summary_frame,
        orient="horizontal",
        command=summary_box.xview,
    )
    summary_x_scroll.grid(row=1, column=0, sticky="ew")
    summary_box.configure(
        yscrollcommand=summary_y_scroll.set,
        xscrollcommand=summary_x_scroll.set,
    )

    def set_summary(text: str) -> None:
        """Replace the read-only report while preserving selectable text."""
        summary_box.configure(state="normal")
        summary_box.delete("1.0", "end")
        summary_box.insert("1.0", text)
        summary_box.configure(state="disabled")
        summary_box.yview_moveto(0.0)
        summary_box.xview_moveto(0.0)

    def select_all_summary(_event=None):
        summary_box.tag_add(tk.SEL, "1.0", "end-1c")
        summary_box.mark_set(tk.INSERT, "1.0")
        summary_box.see(tk.INSERT)
        return "break"

    def copy_summary(_event=None):
        try:
            selected_text = summary_box.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return "break"
        root.clipboard_clear()
        root.clipboard_append(selected_text)
        root.update_idletasks()
        return "break"

    summary_box.bind("<Command-a>", select_all_summary)
    summary_box.bind("<Control-a>", select_all_summary)
    summary_box.bind("<Command-c>", copy_summary)
    summary_box.bind("<Control-c>", copy_summary)
    set_summary("Choose Preview or Generate XYZ to fit and verify the curve.")

    status_var = tk.StringVar(value="Ready.")
    ttk.Label(main, textvariable=status_var, style="Hint.TLabel").grid(
        row=3, column=0, sticky="sw", pady=(10, 0)
    )

    cached_key: Optional[Tuple[float, float, str, Optional[float], int, int]] = None
    cached_result: Optional[SCGenerationResult] = None

    def parse_gui_values() -> Tuple[float, float, str, Optional[float], int, int, str]:
        try:
            total_length = float(length_var.get().strip())
        except ValueError:
            raise ValueError("Length L must be a number.")
        try:
            target_writhe = float(writhe_var.get().strip())
        except ValueError:
            raise ValueError("Writhe W must be a number.")
        selection = curvature_objective_var.get().strip()
        curvature_objective = selection
        opening_angle_deg: Optional[float] = None
        if selection == OPENING_ANGLE_MODE_MANUAL:
            curvature_objective = DEFAULT_CURVATURE_OBJECTIVE
            try:
                opening_angle_deg = float(opening_angle_var.get().strip())
            except ValueError:
                raise ValueError("The provided opening angle must be a number.")
        try:
            num_points = int(points_var.get().strip())
        except ValueError:
            raise ValueError("Points must be an integer.")
        try:
            precision = int(precision_var.get().strip())
        except ValueError:
            raise ValueError("Precision must be an integer.")
        output = output_var.get().strip()
        if not output:
            raise ValueError("Please specify an output file.")
        validate_inputs(
            total_length,
            target_writhe,
            num_points,
            precision,
            curvature_objective,
            opening_angle_deg,
        )
        return (
            total_length,
            target_writhe,
            curvature_objective,
            opening_angle_deg,
            num_points,
            precision,
            output,
        )

    def obtain_result() -> Tuple[SCGenerationResult, str, int]:
        nonlocal cached_key, cached_result
        (
            total_length,
            target_writhe,
            curvature_objective,
            opening_angle_deg,
            num_points,
            precision,
            output,
        ) = parse_gui_values()
        key = (
            total_length,
            target_writhe,
            curvature_objective,
            opening_angle_deg,
            num_points,
            precision,
        )
        if cached_result is None or cached_key != key:
            if opening_angle_deg is None:
                status_var.set(
                    "Optimizing opening angle, fitting writhe, and verifying geometry..."
                )
            else:
                status_var.set(
                    "Using provided opening angle, fitting writhe, and verifying geometry..."
                )
            root.update_idletasks()
            cached_result = generate_sc_points(
                total_length=total_length,
                target_writhe=target_writhe,
                num_points=num_points,
                precision=precision,
                curvature_objective=curvature_objective,
                opening_angle_deg=opening_angle_deg,
            )
            cached_key = key
        set_summary(generation_summary(cached_result))
        status_var.set("Curve fitted and verified.")
        return cached_result, output, precision

    def preview() -> None:
        try:
            obtain_result()
        except Exception as exc:
            status_var.set("Input or generation issue: {0}".format(exc))
            set_summary("Input or generation issue:\n{0}".format(exc))
            messagebox.showerror("Generate SC error", str(exc))

    def generate_file() -> None:
        try:
            result, output, precision = obtain_result()
            write_plain_xyz(result.points, output, precision)
            status_var.set("Wrote {0} points to {1}".format(len(result.points), output))
            messagebox.showinfo(
                "Generate SC complete",
                "Wrote {0} points to:\n{1}\n\nFixed xz crossings: {2}\n"
                "PCA plane: {3}; PCA crossings: {4}\nAchieved writhe: {5:.6f}\n"
                "Selected angle: {6:.4f} deg\nTotal curvature: {7:.6f} rad\n"
                "Largest local curvature: {8:.6g} inverse length\n"
                "Reduced bending energy: {9:.6g} inverse length\n"
                "Load it in Curve It as a closed path.".format(
                    len(result.points),
                    output,
                    result.xz_crossings,
                    result.pca_plane,
                    result.pca_crossings,
                    result.achieved_writhe,
                    result.opening_angle_deg,
                    result.total_curvature,
                    result.maximum_local_curvature,
                    result.bending_energy_integral,
                ),
            )
        except Exception as exc:
            status_var.set("Input or generation issue: {0}".format(exc))
            set_summary("Input or generation issue:\n{0}".format(exc))
            messagebox.showerror("Generate SC error", str(exc))

    buttons = ttk.Frame(main)
    buttons.grid(row=4, column=0, sticky="e", pady=(12, 0))
    ttk.Button(buttons, text="Preview / verify", command=preview).pack(side="left", padx=(0, 8))
    ttk.Button(buttons, text="Generate XYZ", command=generate_file).pack(side="left")

    root.bind("<Escape>", lambda _event: root.destroy())
    root.mainloop()


def main() -> None:
    args = parse_args(sys.argv[1:])
    if args.gui or len(sys.argv) == 1:
        run_gui()
    else:
        try:
            run_cli(args)
        except Exception as exc:
            raise SystemExit("{0} error: {1}".format(TOOL_NAME, exc)) from exc


if __name__ == "__main__":
    main()
