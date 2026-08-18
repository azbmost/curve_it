#!/usr/bin/env python3
"""Generate a radius-aware, projection-robust closed plectonemic DNA-axis curve.

Generate SC V3_5 is a standalone successor to generate_sc_xyzV2_2.py and
Generate SC V3.  Only this SC generator script is required in ``curve_it_lib``;
it does NOT import or monkey-patch an older Generate SC module.

The generated curve is a single closed centerline made from two antipodal
superhelical arms and two smooth end loops.  It is intended as an educational
"textbook plectoneme" rather than an elastic-rod energy minimizer.

Two related writhe concepts are kept explicit:

* ``--writhe`` is fitted to the continuous Gauss-integral writhe used by
  Curve It.
* For nonzero integer writhe, V3_5 also evaluates the *signed* crossing sum in
  many deterministic generic orthographic projections.  The default search
  seeks a geometry for which at least 55% of sampled viewing directions have
  signed crossing sum exactly equal to the requested writhe.

Projection-robust integer geometry
----------------------------------
V2.2 used an arm phase sweep ``theta = pi * W``.  V3_5 shortens the phase before
attaching the end loops:

    theta = sign(W) * pi * (|W| - phase_trim),  0 <= phase_trim < 1

The default trim is 0.40. The final symmetry rotation centers the shortened
phase interval, leaving its two ends ``phase_trim*pi/2`` from the legacy ends.
All ``|W|`` intended xz crossings therefore remain inside the interval while
``phase_trim < 1``. The end-loop Bezier control distance is then refitted so
that the Gauss writhe remains W.

Finally, the whole curve is rotated around z by half the phase removed from the
legacy sweep.  This restores V2.2's fixed-xz symmetry: odd integer writhe has
mirror symmetry under ``z -> -z``, while even integer writhe has central
symmetry under ``(x, z) -> (-x, -z)``.

Trimming is enabled by default and may be disabled with ``--no-trim``. For
fractional writhe and W=0, V3_5 uses ``phase_trim = 0`` because a projection
crossing sum is integer-valued. The equal-lobes formulas account for the active
centered phase trim, so equal-lobes mode can use the same trimming search.

V3_5 uses ``--minimum-final-radius`` as the default alternative to a requested
qualifying-view percentage. In that mode, the largest phase trim compatible
with the minimum radius measured from the final serialized central-arm
coordinates is selected, maximizing the qualifying-view fraction for this
phase-trim family. The default minimum final measured radius is 13.

Automatic opening-angle objectives
----------------------------------
* ``max-local``: minimize the largest local curvature.
* ``total``: minimize total curvature.
* ``bending-energy`` (default): minimize total bending energy, proportional to
  integral kappa(s)^2 ds for constant bending rigidity.
* ``equal-lobes``: match terminal and middle lobe z-heights in fixed xz;
  requires integer |W| >= 2 and supports both trimmed and untrimmed geometry.

The canonical dimensionless curve is fitted first, periodically smoothed once
with Curve It's smoothing convention, resampled, and uniformly scaled to the
requested closed contour length.  The exact decimal coordinates written to
disk are used for final length, writhe, and crossing verification.

Dependencies
------------
This script relies on ``cal_xyz_total_curvature_writheV2.py`` from Curve It for
periodic splines, smoothing, curvature, and Gauss-writhe calculations.  It does
not depend on any older ``generate_sc_xyz*.py`` file.

Examples
--------
Open the GUI::

    python generate_sc_xyzV3_5.py

Generate the default 1071-Angstrom curve with writhe -3 and final radius >= 13::

    python generate_sc_xyzV3_5.py -L 1071 -w -3 -n 2000 -o sc_Wm3.xyz

Retain a user-provided 25-degree opening angle::

    python generate_sc_xyzV3_5.py -L 1071 -w -3 -a 25 -n 2000 -o sc_Wm3_25deg.xyz

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
TOOL_VERSION = "V3_5"

DEFAULT_TOTAL_LENGTH = 1071.0
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
    CURVATURE_OBJECTIVE_BENDING_ENERGY,
    CURVATURE_OBJECTIVE_MAX_LOCAL,
    CURVATURE_OBJECTIVE_TOTAL,
    OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES,
)
DEFAULT_CURVATURE_OBJECTIVE = CURVATURE_OBJECTIVE_BENDING_ENERGY

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
MAX_TOTAL_LENGTH = 1.0e12

# V3/V3.5 projection-robustness settings.
DEFAULT_QUALIFYING_VIEWS_PERCENT = 55.0
DEFAULT_MINIMUM_FINAL_RADIUS = 13.0
DEFAULT_PHASE_TRIM = 0.40
FALLBACK_PHASE_TRIMS = (0.42, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90)
MANUAL_FALLBACK_PHASE_TRIMS = (0.35, 0.30, 0.20, 0.0)
FINAL_PROJECTION_DIRECTIONS = 256
SEARCH_PROJECTION_DIRECTIONS = FINAL_PROJECTION_DIRECTIONS
MIN_PROJECTION_CURVE_SAMPLES = 220
MAX_PROJECTION_CURVE_SAMPLES = 420
MAX_RADIUS_SEARCH_PHASE_TRIM = 0.90
RADIUS_SEARCH_BISECTION_STEPS = 12

SCREENING_MODE_QUALIFYING_VIEWS = "qualifying-views"
SCREENING_MODE_MINIMUM_RADIUS = "minimum-final-radius"

_ACTIVE_PHASE_TRIM = 0.0


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
    """Generated coordinates plus CLI/GUI report quantities."""

    points: PointArray
    requested_length: float
    achieved_length: float
    requested_writhe: float
    achieved_writhe: float
    xz_crossings: int
    pca_crossings: int
    pca_plane: str
    plectoneme_phase_turns: float
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
    final_superhelix_radius: float
    stem_height_scaled: float
    solver_iterations: int
    landmarks: ContourLandmarks
    phase_trim: float
    phase_sweep_rad: float
    phase_sweep_pi: float
    phase_factor: float
    xz_symmetry_rotation_rad: float
    trim_enabled: bool
    projection_stats: Dict[str, object]
    search_projection_stats: Dict[str, object]
    projection_majority_target: float
    screening_mode: str
    minimum_final_radius: Optional[float]
    projection_search_evaluations: int


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
    trim_enabled: bool = True,
    qualifying_views_percent: float = DEFAULT_QUALIFYING_VIEWS_PERCENT,
    minimum_final_radius: Optional[float] = None,
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
            "For reliable educational crossing diagrams, |writhe| must be "
            "at most {0:g}.".format(MAX_ABS_WRITHE)
        )
    if isinstance(num_points, bool) or int(num_points) != num_points:
        raise ValueError("Number of points must be an integer.")
    required = minimum_num_points(target_writhe)
    if int(num_points) < required:
        raise ValueError(
            "Number of points must be at least {0} for writhe {1:g}; this keeps "
            "the plectoneme and crossings adequately sampled.".format(required, target_writhe)
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
            "A user-provided opening angle must be finite and strictly between 0 and 90 degrees."
        )
    if not isinstance(trim_enabled, (bool, np.bool_)):
        raise ValueError("Trim selection must be true or false.")
    if (
        not math.isfinite(float(qualifying_views_percent))
        or not 0.0 <= float(qualifying_views_percent) <= 100.0
    ):
        raise ValueError("Qualifying views must be a percentage between 0 and 100.")
    if minimum_final_radius is not None:
        minimum_radius = float(minimum_final_radius)
        if not math.isfinite(minimum_radius) or minimum_radius <= 0.0:
            raise ValueError("Minimum final measured superhelix radius must be positive and finite.")
        is_integer, nearest = _integer_request(target_writhe)
        if not bool(trim_enabled):
            raise ValueError(
                "Minimum-radius screening requires arm-phase trimming to be enabled."
            )
        if not is_integer or nearest == 0:
            raise ValueError(
                "Minimum-radius screening requires a nonzero integer writhe."
            )
    if opening_angle_deg is None and str(curvature_objective) == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES:
        nearest_integer = int(round(float(target_writhe)))
        if (
            abs(float(target_writhe) - nearest_integer) > 1.0e-10
            or abs(nearest_integer) < 2
        ):
            raise ValueError(
                "The equal-lobes objective requires an integer writhe with |W| >= 2 "
                "so the fixed xz projection has at least one middle lobe."
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
    return uu ** 3 * p0 + 3.0 * uu ** 2 * tt * p1 + 3.0 * uu * tt ** 2 * p2 + tt ** 3 * p3


def _set_active_phase_trim(value: float) -> None:
    global _ACTIVE_PHASE_TRIM
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError("V3_5 phase trim must satisfy 0 <= trim < 1.")
    _ACTIVE_PHASE_TRIM = value


def _loop_control_upper_limit(target_writhe: float) -> float:
    """Return a practical loop-control bound for the active shortened phase."""

    base = max(4.0, 3.0 * abs(float(target_writhe)) + 3.0)
    trim_extension = 120.0 * max(0.0, _ACTIVE_PHASE_TRIM - DEFAULT_PHASE_TRIM)
    return base + trim_extension


def _integer_request(target_writhe: float) -> Tuple[bool, int]:
    nearest = int(round(float(target_writhe)))
    return abs(float(target_writhe) - nearest) <= 1.0e-10, nearest


def _phase_sweep(target_writhe: float) -> float:
    """Return the current signed arm phase sweep in radians."""

    w = float(target_writhe)
    if abs(w) <= 1.0e-12 or _ACTIVE_PHASE_TRIM <= 1.0e-12:
        return math.pi * w
    is_integer, nearest = _integer_request(w)
    if not is_integer or nearest == 0:
        return math.pi * w
    return math.copysign(math.pi * (abs(nearest) - _ACTIVE_PHASE_TRIM), w)


def _xz_symmetry_rotation(target_writhe: float) -> float:
    """Return the z-axis rotation that restores the legacy symmetric xz view.

    V3.5 removes equal phase from the two ends of the V2.2 arm sweep. Rotating
    by half of that removed phase places both terminal loops symmetrically
    about their legacy fixed-xz directions. As in V2.2, the projected symmetry
    is z-reflection for odd integer writhe and central inversion for even
    integer writhe. Untrimmed and fractional curves receive zero rotation.
    """

    return 0.5 * (math.pi * float(target_writhe) - _phase_sweep(target_writhe))


def _rotate_points_about_z(points: PointArray, angle_rad: float) -> PointArray:
    """Rotate an N x 3 coordinate array around the z axis."""

    pts = np.asarray(points, dtype=float)
    angle = float(angle_rad)
    if abs(angle) <= 1.0e-15:
        return pts.copy()
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated = pts.copy()
    rotated[:, 0] = cosine * pts[:, 0] - sine * pts[:, 1]
    rotated[:, 1] = sine * pts[:, 0] + cosine * pts[:, 1]
    return rotated


def _canonical_geometry(
    target_writhe: float,
    opening_angle_deg: float = DEFAULT_OPENING_ANGLE_DEG,
) -> Tuple[float, float]:
    """Return current signed phase sweep and canonical stem height."""

    theta_total = _phase_sweep(target_writhe)
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
    """Build a dense C1 closed plectoneme before arc-length resampling."""

    theta_total, stem_height = _canonical_geometry(target_writhe, opening_angle_deg)
    radius = CANONICAL_RADIUS
    dense_per_segment = max(
        600,
        2 * int(num_points),
        int(math.ceil(240.0 * max(1.0, abs(float(target_writhe))))),
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

    bottom_loop = _cubic_bezier(
        a_bottom,
        a_bottom + float(loop_control) * ta_bottom,
        b_bottom - float(loop_control) * tb_bottom,
        b_bottom,
        t,
    )
    top_loop = _cubic_bezier(
        b_top,
        b_top + float(loop_control) * tb_top,
        a_top - float(loop_control) * ta_top,
        a_top,
        t,
    )
    dense = np.vstack((arm_a(t), bottom_loop, arm_b(t), top_loop))
    dense = _rotate_points_about_z(dense, _xz_symmetry_rotation(target_writhe))
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
    """Locate reproducible tips and interior fixed-xz lobe peaks.

    The V2.2 landmarks occur where each arm reaches an x extremum between two
    neighboring fixed-xz crossings. For phase-trimmed V3.5 geometry, account
    for both the shortened arm sweep and the final symmetry rotation before
    finding the nearest serialized output vertices.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise ValueError("Contour landmarks need an N x 3 curve with at least 4 points.")
    segment_lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    total = float(np.sum(segment_lengths))
    if not math.isfinite(total) or total <= 1.0e-14 or np.any(segment_lengths <= 0.0):
        raise ValueError("Contour landmarks need a finite closed curve without duplicate points.")
    vertex_arc = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))

    def landmark(index: int) -> ContourLandmark:
        return ContourLandmark(
            point_index=int(index),
            contour_percent=100.0 * float(vertex_arc[index]) / total,
        )

    middle_segment_peaks = []
    nearest_integer = int(round(float(target_writhe)))
    if abs(float(target_writhe) - nearest_integer) <= 1.0e-10 and nearest_integer != 0:
        crossing_count = abs(nearest_integer)
        center = np.mean(pts, axis=0)
        phase_sweep = _phase_sweep(target_writhe)
        symmetry_rotation = _xz_symmetry_rotation(target_writhe)
        phase_sign = math.copysign(1.0, float(nearest_integer))
        for middle_index in range(1, crossing_count):
            arm_parameter = (
                phase_sign * float(middle_index) * math.pi - symmetry_rotation
            ) / phase_sweep
            z_offset = float(stem_height_scaled) * (
                0.5 - arm_parameter
            )
            arm_a_x = float(radius_scaled) * (-1.0 if middle_index % 2 else 1.0)
            target_a = center + np.array((arm_a_x, 0.0, z_offset), dtype=float)
            target_b = center + np.array((-arm_a_x, 0.0, z_offset), dtype=float)
            index_a = int(np.argmin(np.linalg.norm(pts - target_a, axis=1)))
            index_b = int(np.argmin(np.linalg.norm(pts - target_b, axis=1)))
            middle_segment_peaks.append((landmark(index_a), landmark(index_b)))

    return ContourLandmarks(
        top_terminus=landmark(int(np.argmax(pts[:, 2]))),
        bottom_terminus=landmark(int(np.argmin(pts[:, 2]))),
        middle_segment_peaks=tuple(middle_segment_peaks),
    )


def evaluate_xz_lobe_heights(
    points: PointArray,
    target_writhe: float,
    stem_height_scaled: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Measure fixed-xz lobe heights for centered trimmed or untrimmed phases."""

    nearest_integer = int(round(float(target_writhe)))
    if abs(float(target_writhe) - nearest_integer) > 1.0e-10 or abs(nearest_integer) < 2:
        return None, None, None, None, None
    pts = np.asarray(points, dtype=float)
    phase_span_pi = abs(_phase_sweep(target_writhe)) / math.pi
    if phase_span_pi <= 1.0e-12:
        return None, None, None, None, None
    first_crossing_parameter = (0.5 - 0.5 * _ACTIVE_PHASE_TRIM) / phase_span_pi
    center_z = float(np.mean(pts[:, 2]))
    outer_crossing_offset = float(stem_height_scaled) * (
        0.5 - first_crossing_parameter
    )
    top_crossing_z = center_z + outer_crossing_offset
    bottom_crossing_z = center_z - outer_crossing_offset
    top_height = float(np.max(pts[:, 2]) - top_crossing_z)
    bottom_height = float(bottom_crossing_z - np.min(pts[:, 2]))
    terminal_height = 0.5 * (top_height + bottom_height)
    middle_height = float(stem_height_scaled) / phase_span_pi
    mismatch = terminal_height - middle_height
    return top_height, bottom_height, terminal_height, middle_height, mismatch


def measure_final_superhelix_radius(
    points: PointArray,
    axis_center: PointArray,
    stem_height_scaled: float,
) -> float:
    """Measure the median arm radius from the serialized final curve.

    The central 90% of the canonical stem-height interval excludes the two end
    loops and the smoothing transition near each arm/loop join. Radial distance
    is measured from the translated canonical z axis after final centering.
    """

    pts = np.asarray(points, dtype=float)
    axis = np.asarray(axis_center, dtype=float)
    half_window = 0.45 * float(stem_height_scaled)
    arm_mask = np.abs(pts[:, 2] - axis[2]) <= half_window
    if int(np.count_nonzero(arm_mask)) < 4:
        raise ValueError("Final superhelix-radius measurement found too few arm points.")
    radial_distance = np.linalg.norm(pts[arm_mask, :2] - axis[None, :2], axis=1)
    radius = float(np.median(radial_distance))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Final superhelix-radius measurement is non-finite or zero.")
    return radius


def quantize_points_for_xyz(points: PointArray, precision: int) -> PointArray:
    """Round coordinates exactly as the plain-XYZ writer will serialize them."""

    value_format = "{{:.{0}f}}".format(int(precision))
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
    smoothed = np.asarray(smooth_closed_points(pts), dtype=float)
    return resample_closed_curve(smoothed, int(num_points))


def evaluate_curve_it_writhe(points: PointArray, smooth: bool = False) -> float:
    """Evaluate the mapped polyline's writhe, optionally smoothing it once."""

    if calculate_polyline_writhe is None or smooth_closed_points is None:
        raise RuntimeError("Curve It writhe support is unavailable; install SciPy.")
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= 1.0e-8:
        pts = pts[:-1]
    evaluation_points = smooth_curve_for_output(pts, len(pts)) if smooth else pts
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
    """Return total curvature, peak curvature, and integral kappa^2 ds."""

    if build_periodic_splines is None or smooth_closed_points is None:
        raise RuntimeError("Curve It curvature support is unavailable; install SciPy.")
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= 1.0e-8:
        pts = pts[:-1]
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise ValueError("Curvature evaluation needs at least four closed-curve points.")
    evaluation_points = smooth_closed_points(pts) if smooth else pts
    spline_x, spline_y, spline_z = build_periodic_splines(evaluation_points)
    sample_count = max(int(n_samples), min(32000, 8 * len(evaluation_points)))
    parameter = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    velocity = np.column_stack(
        (spline_x(parameter, 1), spline_y(parameter, 1), spline_z(parameter, 1))
    )
    acceleration = np.column_stack(
        (spline_x(parameter, 2), spline_y(parameter, 2), spline_z(parameter, 2))
    )
    speed = np.linalg.norm(velocity, axis=1)
    if not np.all(np.isfinite(speed)) or np.any(speed <= 1.0e-14):
        raise ValueError("Curvature evaluation found a zero or non-finite spline speed.")
    local_curvature = np.linalg.norm(np.cross(velocity, acceleration), axis=1) / (speed ** 3)
    curvature_integrand = local_curvature * speed
    bending_energy_integrand = local_curvature ** 2 * speed
    bending_energy_integral = float(np.mean(bending_energy_integrand))
    if adaptive_total:
        if fit_spline_and_calculate_curvature is None:
            raise RuntimeError("Curve It total-curvature support is unavailable; install SciPy.")
        total_curvature = float(fit_spline_and_calculate_curvature(evaluation_points))
    else:
        total_curvature = float(np.mean(curvature_integrand))

    peak_candidate_count = min(24, sample_count)
    peak_indices = np.argpartition(local_curvature, -peak_candidate_count)[-peak_candidate_count:]
    local_offsets = np.linspace(-1.0, 1.0, 33) / float(sample_count)
    refined_parameter = (parameter[peak_indices, None] + local_offsets[None, :]).reshape(-1) % 1.0
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
    maximum_local_curvature = float(max(np.max(local_curvature), np.max(refined_curvature)))
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
        last_j_exclusive = n - 1 if i == 0 else n
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
    """Return crossing count and nearest Cartesian name of the PC1-PC2 plane."""

    pts = np.asarray(points, dtype=float)
    centered = pts - np.mean(pts, axis=0)
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] <= 1.0e-14:
        return 0, "degenerate"
    normal_axis = int(np.argmax(np.abs(vh[2])))
    plane_name = ("yz", "xz", "xy")[normal_axis]
    projected = centered @ vh[:2].T
    return _count_projected_crossings(projected), plane_name


def count_pca_projection_crossings(points: PointArray) -> int:
    return analyze_pca_projection(points)[0]


def count_xz_projection_crossings(points: PointArray) -> int:
    pts = np.asarray(points, dtype=float)
    return _count_projected_crossings(pts[:, (2, 0)])


def _projection_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    helper = np.array((1.0, 0.0, 0.0), dtype=float)
    if abs(float(normal[0])) > 0.8:
        helper = np.array((0.0, 1.0, 0.0), dtype=float)
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def _fibonacci_directions(count: int) -> np.ndarray:
    """Return deterministic, approximately uniform, y-reflection-paired views.

    Positive- and negative-writhe outputs are related by y reflection. Pairing
    every sampled normal with its y-reflected counterpart ensures those mirror
    curves receive exactly the same finite-sample projection statistics.
    """

    count = int(count)
    if count < 8:
        raise ValueError("At least 8 projection directions are required.")
    if count % 2:
        raise ValueError("Projection direction count must be even for reflection pairing.")
    half_count = count // 2
    k = np.arange(half_count, dtype=float)
    golden = 0.5 * (1.0 + math.sqrt(5.0))
    z = 1.0 - 2.0 * (k + 0.5) / float(half_count)
    azimuth = 2.0 * math.pi * k / golden
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    base = np.column_stack(
        (radius * np.cos(azimuth), np.abs(radius * np.sin(azimuth)), z)
    )
    mirrored = base.copy()
    mirrored[:, 1] *= -1.0
    return np.vstack((base, mirrored))


def signed_projection_crossing_sum(points: np.ndarray, normal: np.ndarray) -> int:
    """Return the oriented signed crossing sum for one orthographic projection."""

    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= 1.0e-10:
        pts = pts[:-1]
    if len(pts) < 4:
        return 0
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    e1, e2 = _projection_basis(normal)
    projected = np.column_stack((pts @ e1, pts @ e2))
    span = float(np.max(np.ptp(projected, axis=0)))
    if span <= 1.0e-14:
        return 0
    projected /= span

    starts = projected
    ends = np.roll(projected, -1, axis=0)
    directions_2d = ends - starts
    directions_3d = np.roll(pts, -1, axis=0) - pts
    n_seg = len(pts)
    endpoint_tol = 1.0e-7
    parallel_tol = 1.0e-12
    depth_tol = 1.0e-12
    crossing_sum = 0

    for i in range(n_seg):
        first_j = i + 2
        last_j_exclusive = n_seg - 1 if i == 0 else n_seg
        if first_j >= last_j_exclusive:
            continue
        js = np.arange(first_j, last_j_exclusive)
        r = directions_2d[i]
        s = directions_2d[js]
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
        if not np.any(proper):
            continue
        crossed_js = js[proper]
        ti = t_param[proper]
        uj = u_param[proper]
        point_i = pts[i] + ti[:, None] * directions_3d[i]
        point_j = pts[crossed_js] + uj[:, None] * directions_3d[crossed_js]
        depth = (point_i - point_j) @ normal
        cross_normal = np.cross(
            np.repeat(directions_3d[i][None, :], len(crossed_js), axis=0),
            directions_3d[crossed_js],
        ) @ normal
        nondegenerate = (np.abs(depth) > depth_tol) & (np.abs(cross_normal) > depth_tol)
        if np.any(nondegenerate):
            crossing_sum += int(np.sum(np.sign(depth[nondegenerate] * cross_normal[nondegenerate])))
    return crossing_sum


def _projection_curve_sample_count(target_writhe: float) -> int:
    desired = 160 + 24 * int(math.ceil(abs(float(target_writhe))))
    return max(MIN_PROJECTION_CURVE_SAMPLES, min(MAX_PROJECTION_CURVE_SAMPLES, desired))


def projection_statistics(
    points: np.ndarray,
    target_writhe: float,
    direction_count: int,
) -> Dict[str, object]:
    """Evaluate signed crossing sums over deterministic generic projections."""

    is_integer, nearest = _integer_request(target_writhe)
    if not is_integer or nearest == 0:
        return {"applicable": False, "direction_count": int(direction_count)}
    sample_count = _projection_curve_sample_count(target_writhe)
    sampled_points = resample_closed_curve(np.asarray(points, dtype=float), sample_count)
    directions = _fibonacci_directions(direction_count)
    values = np.array(
        [signed_projection_crossing_sum(sampled_points, normal) for normal in directions],
        dtype=int,
    )
    unique_values, counts = np.unique(values, return_counts=True)
    histogram = {int(value): int(count) for value, count in zip(unique_values, counts)}
    target_fraction = float(np.mean(values == nearest))
    mode_index = int(np.argmax(counts))
    return {
        "applicable": True,
        "direction_count": int(direction_count),
        "curve_samples": int(sample_count),
        "target": int(nearest),
        "target_fraction": target_fraction,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "mode": int(unique_values[mode_index]),
        "minimum": int(np.min(values)),
        "maximum": int(np.max(values)),
        "histogram": histogram,
    }


def _fit_loop_control(
    target_writhe: float,
    num_points: int,
    opening_angle_deg: float = DEFAULT_OPENING_ANGLE_DEG,
    tolerance: float = WRITHE_TOLERANCE,
) -> Tuple[float, PointArray, float, float, int]:
    """Fit end-loop control distance while the arm phase stays fixed."""

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
    upper = _loop_control_upper_limit(target_writhe)
    controls = np.linspace(lower, upper, 21)
    bracket: Optional[Tuple[float, float]] = None
    previous_control = float(controls[0])
    previous_value = evaluate(previous_control)[0]
    evaluations = 1
    if abs(previous_value) <= tolerance * 0.25:
        value = evaluate(previous_control)
        return previous_control, value[1], value[2], value[3], evaluations

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
            "This opening angle/phase-trim combination is infeasible."
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

    loop_control, canonical, achieved_writhe, stem_height, solver_iterations = _fit_loop_control(
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
    total_curvature, maximum_local_curvature, bending_energy_integral = evaluate_curvature_metrics(
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
    ) = evaluate_xz_lobe_heights(scaled_points, target_writhe, stem_height * scale)
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


def _refine_candidate_exact_writhe(
    candidate: _AngleCandidate,
    total_length: float,
    target_writhe: float,
    num_points: int,
) -> _AngleCandidate:
    """Refit loop control against the exact smoothed output-polyline writhe.

    The opening-angle search uses the faster periodic-spline writhe estimate.
    At high |W| that estimate can differ from the exact segment-pair writhe by
    more than the serialized-output tolerance.  This final one-dimensional
    correction keeps the selected angle and phase trim fixed while refitting
    only the loop control against the exact geometry Curve It will map.
    """

    target_magnitude = abs(float(target_writhe))
    if target_magnitude <= 1.0e-12:
        return candidate

    cache: Dict[float, Tuple[float, PointArray, float, float]] = {}

    def evaluate(control: float) -> Tuple[float, PointArray, float, float]:
        key = round(float(control), 12)
        if key not in cache:
            dense, stem_height = _build_dense_canonical_curve(
                target_writhe,
                float(control),
                num_points,
                candidate.angle_deg,
            )
            canonical = resample_closed_curve(dense, num_points)
            smoothed = smooth_curve_for_output(canonical, num_points)
            exact_writhe = evaluate_curve_it_writhe(smoothed, smooth=False)
            cache[key] = (
                abs(exact_writhe) - target_magnitude,
                canonical,
                exact_writhe,
                stem_height,
            )
        return cache[key]

    start_control = float(candidate.loop_control)
    start_data = evaluate(start_control)
    exact_target_tolerance = min(1.0e-5, 0.1 * WRITHE_TOLERANCE)
    if abs(start_data[0]) <= exact_target_tolerance:
        return candidate

    lower_limit = 0.02
    upper_limit = _loop_control_upper_limit(target_writhe)
    direction = -1.0 if start_data[0] > 0.0 else 1.0
    step = max(0.01, 0.02 * start_control)
    previous_control = start_control
    previous_data = start_data
    bracket: Optional[Tuple[float, float]] = None

    for _iteration in range(18):
        trial_control = min(
            upper_limit,
            max(lower_limit, start_control + direction * step),
        )
        trial_data = evaluate(trial_control)
        if previous_data[0] * trial_data[0] <= 0.0:
            bracket = tuple(sorted((previous_control, trial_control)))
            break
        if trial_control in (lower_limit, upper_limit):
            break
        previous_control = trial_control
        previous_data = trial_data
        step *= 1.7

    if bracket is None:
        # The exact correction is normally tiny.  If no local bracket exists,
        # preserve the selected candidate and let serialized verification emit
        # the normal actionable tolerance error.
        return candidate

    lo, hi = bracket
    lo_data = evaluate(lo)
    hi_data = evaluate(hi)
    best_control, best_data = min(
        ((lo, lo_data), (hi, hi_data)),
        key=lambda item: abs(item[1][0]),
    )
    for _iteration in range(28):
        mid = 0.5 * (lo + hi)
        mid_data = evaluate(mid)
        if abs(mid_data[0]) < abs(best_data[0]):
            best_control, best_data = mid, mid_data
        if abs(mid_data[0]) <= exact_target_tolerance:
            break
        if lo_data[0] * mid_data[0] <= 0.0:
            hi = mid
            hi_data = mid_data
        else:
            lo = mid
            lo_data = mid_data

    canonical = best_data[1]
    exact_writhe = best_data[2]
    stem_height = best_data[3]
    smoothed = smooth_curve_for_output(canonical, num_points)
    scale = float(total_length) / closed_polyline_length(smoothed)
    scaled_points = smoothed * scale
    scaled_points -= np.mean(scaled_points, axis=0)
    total_curvature, maximum_local_curvature, bending_energy_integral = evaluate_curvature_metrics(
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
        angle_deg=candidate.angle_deg,
        loop_control=float(best_control),
        canonical_points=canonical,
        achieved_writhe=float(exact_writhe),
        stem_height=float(stem_height),
        writhe_solver_iterations=candidate.writhe_solver_iterations + len(cache),
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
    """Choose the best feasible opening angle for the requested objective."""

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
            "No feasible opening angle was found in the supported automatic search interval."
        )

    def candidate_key(candidate: _AngleCandidate) -> Tuple[float, float, float]:
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
        refinement_angles = np.linspace(lower, upper, OPENING_ANGLE_REFINEMENT_SAMPLES)
        refined = [candidate for candidate in (evaluate(angle) for angle in refinement_angles) if candidate]
        if refined:
            best = min((best, *refined), key=candidate_key)
        step = max(1.0e-4, (upper - lower) / float(OPENING_ANGLE_REFINEMENT_SAMPLES - 1))

    if objective == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES:
        signed_candidates = sorted(
            (
                candidate
                for candidate in cache.values()
                if candidate is not None and candidate.lobe_height_mismatch_xz is not None
            ),
            key=lambda candidate: candidate.angle_deg,
        )
        brackets = []
        for lower_candidate, upper_candidate in zip(signed_candidates[:-1], signed_candidates[1:]):
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
                mid_candidate = evaluate(0.5 * (lo_candidate.angle_deg + hi_candidate.angle_deg))
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


def _candidate_points(candidate: _AngleCandidate, total_length: float) -> np.ndarray:
    smoothed = smooth_curve_for_output(candidate.canonical_points, len(candidate.canonical_points))
    scale = float(total_length) / closed_polyline_length(smoothed)
    points = smoothed * scale
    points -= np.mean(points, axis=0)
    return points


def _candidate_final_measured_radius(
    candidate: _AngleCandidate,
    total_length: float,
    precision: int,
) -> float:
    """Estimate the final serialized central-arm radius for a search candidate.

    This applies the same smoothing, length scaling, centering, decimal
    quantization, axis translation, stem window, and median-radius convention
    used by ``_finalize_candidate``. Search candidates may still receive a
    final exact-writhe refinement before acceptance.
    """

    smoothed = smooth_curve_for_output(
        candidate.canonical_points,
        len(candidate.canonical_points),
    )
    scale = float(total_length) / closed_polyline_length(smoothed)
    scaled = smoothed * scale
    centering_offset = np.mean(scaled, axis=0)
    points = quantize_points_for_xyz(scaled - centering_offset, precision)
    return measure_final_superhelix_radius(
        points,
        axis_center=-centering_offset,
        stem_height_scaled=float(candidate.stem_height) * scale,
    )


def _candidate_projection_stats(
    candidate: _AngleCandidate,
    total_length: float,
    target_writhe: float,
) -> Optional[Dict[str, object]]:
    points = _candidate_points(candidate, total_length)
    is_integer, nearest = _integer_request(target_writhe)
    if is_integer and count_xz_projection_crossings(points) != abs(nearest):
        return None
    return projection_statistics(points, target_writhe, SEARCH_PROJECTION_DIRECTIONS)


def _projection_target_met(
    stats: Optional[Dict[str, object]], target_fraction: float
) -> bool:
    """Return whether deterministic projection screening meets its target."""

    return bool(
        stats is not None
        and bool(stats.get("applicable", False))
        and float(stats["target_fraction"]) >= float(target_fraction)
    )


def _objective_value(candidate: _AngleCandidate, objective: str) -> float:
    if objective == CURVATURE_OBJECTIVE_TOTAL:
        return float(candidate.total_curvature)
    if objective == CURVATURE_OBJECTIVE_MAX_LOCAL:
        return float(candidate.maximum_local_curvature)
    if objective == CURVATURE_OBJECTIVE_BENDING_ENERGY:
        return float(candidate.bending_energy_integral)
    return float(candidate.maximum_local_curvature)


def _writhe_tolerance_for_objective(objective: str) -> float:
    if objective == CURVATURE_OBJECTIVE_BENDING_ENERGY:
        return float(BENDING_ENERGY_WRITHE_FIT_TOLERANCE)
    return float(WRITHE_TOLERANCE)


def _fit_at_angle(
    total_length: float,
    target_writhe: float,
    num_points: int,
    opening_angle_deg: float,
    objective: str,
) -> Optional[_AngleCandidate]:
    try:
        return _fit_angle_candidate(
            total_length,
            target_writhe,
            num_points,
            float(opening_angle_deg),
            writhe_tolerance=_writhe_tolerance_for_objective(objective),
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def _find_projection_robust_auto_candidate(
    total_length: float,
    target_writhe: float,
    num_points: int,
    objective: str,
    qualifying_fraction: float,
) -> Tuple[float, _AngleCandidate, Dict[str, object], int]:
    """Keep each objective-optimal angle while seeking a robust phase trim."""

    tested = []
    _set_active_phase_trim(DEFAULT_PHASE_TRIM)
    candidate, optimizer_evaluations = _optimize_opening_angle(
        total_length, target_writhe, num_points, objective
    )
    evaluations = int(optimizer_evaluations)
    objective_angle = float(candidate.angle_deg)

    reoptimize_each_trim = objective == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES
    for trim_index, trim in enumerate((DEFAULT_PHASE_TRIM,) + FALLBACK_PHASE_TRIMS):
        _set_active_phase_trim(trim)
        if trim_index > 0:
            if reoptimize_each_trim:
                try:
                    candidate, optimizer_evaluations = _optimize_opening_angle(
                        total_length, target_writhe, num_points, objective
                    )
                    evaluations += int(optimizer_evaluations)
                except (RuntimeError, ValueError, FloatingPointError):
                    candidate = None
            else:
                candidate = _fit_at_angle(
                    total_length,
                    target_writhe,
                    num_points,
                    objective_angle,
                    objective,
                )
                evaluations += 1

        if candidate is not None:
            stats = _candidate_projection_stats(candidate, total_length, target_writhe)
            evaluations += 1
            if stats is not None and bool(stats.get("applicable", False)):
                entry = (float(trim), candidate, stats)
                tested.append(entry)
                if _projection_target_met(stats, qualifying_fraction):
                    return float(trim), candidate, stats, evaluations

    if not tested:
        raise RuntimeError(
            "V3_5 could not find a feasible shortened-phase plectoneme for this integer writhe."
        )
    best = min(
        tested,
        key=lambda item: (
            -float(item[2]["target_fraction"]),
            _objective_value(item[1], objective),
            abs(float(item[1].angle_deg) - DEFAULT_OPENING_ANGLE_DEG),
        ),
    )
    return best[0], best[1], best[2], evaluations


def _find_projection_robust_manual_candidate(
    total_length: float,
    target_writhe: float,
    num_points: int,
    opening_angle_deg: float,
    objective: str,
    qualifying_fraction: float,
) -> Tuple[float, _AngleCandidate, Dict[str, object], int]:
    """Retain the provided angle while choosing the best feasible phase trim."""

    trims = (DEFAULT_PHASE_TRIM,) + FALLBACK_PHASE_TRIMS + MANUAL_FALLBACK_PHASE_TRIMS
    tested = []
    evaluations = 0
    for trim in trims:
        _set_active_phase_trim(trim)
        candidate = _fit_at_angle(
            total_length,
            target_writhe,
            num_points,
            opening_angle_deg,
            objective,
        )
        evaluations += 1
        if candidate is None:
            continue
        stats = _candidate_projection_stats(candidate, total_length, target_writhe)
        if stats is None or not bool(stats.get("applicable", False)):
            continue
        tested.append((float(trim), candidate, stats))
        if float(stats["target_fraction"]) >= float(qualifying_fraction):
            return float(trim), candidate, stats, evaluations
    if not tested:
        raise RuntimeError("The provided opening angle is infeasible for all tested V3_5 phase trims.")
    best = min(
        tested,
        key=lambda item: (
            -float(item[2]["target_fraction"]),
            abs(float(item[0]) - DEFAULT_PHASE_TRIM),
        ),
    )
    return best[0], best[1], best[2], evaluations


def _find_radius_constrained_candidate(
    total_length: float,
    target_writhe: float,
    num_points: int,
    precision: int,
    objective: str,
    minimum_final_radius: float,
    opening_angle_deg: Optional[float] = None,
) -> Tuple[float, _AngleCandidate, Dict[str, object], int]:
    """Maximize phase trim while retaining the requested final measured radius.

    Increasing the centered phase trim removes poorly viewed terminal arm
    phase and monotonically improves the projection robustness of this curve
    family. The search therefore finds the largest feasible trim in [0, 0.90]
    whose serialized central-arm median radius remains at least the requested
    minimum. Candidate and final measurements use the same smoothing, scaling,
    centering, quantization, translated-axis, and central-90-percent convention.
    The final candidate is exact-writhe refined before acceptance.
    """

    minimum_radius = float(minimum_final_radius)
    evaluations = 0
    cache: Dict[float, Optional[_AngleCandidate]] = {}

    _set_active_phase_trim(DEFAULT_PHASE_TRIM)
    if opening_angle_deg is None:
        default_candidate, optimizer_evaluations = _optimize_opening_angle(
            total_length,
            target_writhe,
            num_points,
            objective,
        )
        evaluations += int(optimizer_evaluations)
        objective_angle = float(default_candidate.angle_deg)
        cache[DEFAULT_PHASE_TRIM] = default_candidate
    else:
        objective_angle = float(opening_angle_deg)

    reoptimize_each_trim = (
        opening_angle_deg is None
        and objective == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES
    )

    def evaluate(trim: float) -> Optional[Tuple[_AngleCandidate, float]]:
        nonlocal evaluations
        key = round(float(trim), 12)
        if key not in cache:
            _set_active_phase_trim(float(trim))
            if reoptimize_each_trim:
                try:
                    candidate, optimizer_evaluations = _optimize_opening_angle(
                        total_length,
                        target_writhe,
                        num_points,
                        objective,
                    )
                    evaluations += int(optimizer_evaluations)
                except (RuntimeError, ValueError, FloatingPointError):
                    candidate = None
            else:
                candidate = _fit_at_angle(
                    total_length,
                    target_writhe,
                    num_points,
                    objective_angle,
                    objective,
                )
                evaluations += 1
            cache[key] = candidate
        candidate = cache[key]
        if candidate is None:
            return None
        _set_active_phase_trim(float(trim))
        return candidate, _candidate_final_measured_radius(
            candidate,
            total_length,
            precision,
        )

    lower_trim = 0.0
    lower = evaluate(lower_trim)
    if lower is None:
        raise RuntimeError("No feasible untrimmed candidate was found for radius screening.")
    lower_candidate, lower_radius = lower
    if lower_radius < minimum_radius:
        _set_active_phase_trim(lower_trim)
        lower_candidate = _refine_candidate_exact_writhe(
            lower_candidate,
            total_length,
            target_writhe,
            num_points,
        )
        lower_radius = _candidate_final_measured_radius(
            lower_candidate,
            total_length,
            precision,
        )
        if lower_radius < minimum_radius:
            raise ValueError(
                "Minimum final measured superhelix radius {0:.10g} is unattainable; "
                "the largest available final measured radius is {1:.10g}.".format(
                    minimum_radius,
                    lower_radius,
                )
            )

    upper_trim = MAX_RADIUS_SEARCH_PHASE_TRIM
    upper = evaluate(upper_trim)
    if upper is not None and upper[1] >= minimum_radius:
        selected_trim = upper_trim
        selected_candidate = upper[0]
    else:
        selected_trim = lower_trim
        selected_candidate = lower_candidate
        for _iteration in range(RADIUS_SEARCH_BISECTION_STEPS):
            middle_trim = 0.5 * (lower_trim + upper_trim)
            middle = evaluate(middle_trim)
            if middle is not None and middle[1] >= minimum_radius:
                lower_trim = middle_trim
                selected_trim = middle_trim
                selected_candidate = middle[0]
            else:
                upper_trim = middle_trim

        # Stay one resolved interval inside the feasible side so the subsequent
        # exact-writhe correction cannot push a boundary candidate below the
        # user's measured-radius constraint through a small geometry change.
        guarded_trim = max(0.0, selected_trim - (upper_trim - lower_trim))
        guarded = evaluate(guarded_trim)
        if guarded is not None and guarded[1] >= minimum_radius:
            selected_trim = guarded_trim
            selected_candidate = guarded[0]

    _set_active_phase_trim(selected_trim)
    selected_candidate = _refine_candidate_exact_writhe(
        selected_candidate,
        total_length,
        target_writhe,
        num_points,
    )
    selected_radius = _candidate_final_measured_radius(
        selected_candidate,
        total_length,
        precision,
    )
    if selected_radius < minimum_radius:
        correction_lower_trim = 0.0
        correction_lower = evaluate(correction_lower_trim)
        if correction_lower is None:
            raise RuntimeError(
                "No feasible untrimmed candidate was found during exact radius correction."
            )
        _set_active_phase_trim(correction_lower_trim)
        correction_candidate = _refine_candidate_exact_writhe(
            correction_lower[0],
            total_length,
            target_writhe,
            num_points,
        )
        correction_radius = _candidate_final_measured_radius(
            correction_candidate,
            total_length,
            precision,
        )
        if correction_radius < minimum_radius:
            raise ValueError(
                "Minimum final measured superhelix radius {0:.10g} is unattainable; "
                "the largest available final measured radius is {1:.10g}.".format(
                    minimum_radius,
                    correction_radius,
                )
            )
        correction_upper_trim = selected_trim
        selected_trim = correction_lower_trim
        selected_candidate = correction_candidate
        for _iteration in range(RADIUS_SEARCH_BISECTION_STEPS):
            middle_trim = 0.5 * (correction_lower_trim + correction_upper_trim)
            middle = evaluate(middle_trim)
            if middle is None:
                correction_upper_trim = middle_trim
                continue
            _set_active_phase_trim(middle_trim)
            middle_candidate = _refine_candidate_exact_writhe(
                middle[0],
                total_length,
                target_writhe,
                num_points,
            )
            evaluations += 1
            middle_radius = _candidate_final_measured_radius(
                middle_candidate,
                total_length,
                precision,
            )
            if middle_radius >= minimum_radius:
                correction_lower_trim = middle_trim
                selected_trim = middle_trim
                selected_candidate = middle_candidate
            else:
                correction_upper_trim = middle_trim
    stats = _candidate_projection_stats(selected_candidate, total_length, target_writhe)
    evaluations += 1
    if stats is None or not bool(stats.get("applicable", False)):
        raise RuntimeError(
            "The radius-constrained candidate failed integer projection verification."
        )
    return selected_trim, selected_candidate, stats, evaluations


def _finalize_candidate(
    candidate: _AngleCandidate,
    total_length: float,
    target_writhe: float,
    num_points: int,
    precision: int,
    effective_objective: str,
    opening_angle_evaluations: int,
    selected_trim: float,
    trim_enabled: bool,
    qualifying_fraction: float,
    screening_mode: str,
    minimum_final_radius: Optional[float],
    search_stats: Dict[str, object],
    search_evaluations: int,
) -> SCGenerationResult:
    """Scale, quantize, verify, and package one already-selected candidate."""

    _set_active_phase_trim(selected_trim)
    candidate = _refine_candidate_exact_writhe(
        candidate,
        total_length,
        target_writhe,
        num_points,
    )
    smoothed_canonical = smooth_curve_for_output(candidate.canonical_points, num_points)
    smoothed_length = closed_polyline_length(smoothed_canonical)
    scale = float(total_length) / smoothed_length
    points = smoothed_canonical * scale
    centering_offset = np.mean(points, axis=0)
    points -= centering_offset
    points = quantize_points_for_xyz(points, precision)
    radius_scaled = CANONICAL_RADIUS * scale
    stem_height_scaled = candidate.stem_height * scale
    segment_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    unique_count = int(len(np.unique(points, axis=0)))
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(segment_lengths)):
        raise ValueError("Serialized curve coordinates are outside the numeric range.")
    if unique_count != num_points or np.any(segment_lengths <= 0.0):
        raise ValueError(
            "Output precision collapses distinct curve samples ({0} of {1} remain unique). "
            "Increase --precision or use a larger contour length.".format(unique_count, num_points)
        )
    final_superhelix_radius = measure_final_superhelix_radius(
        points,
        axis_center=-centering_offset,
        stem_height_scaled=stem_height_scaled,
    )
    if (
        minimum_final_radius is not None
        and final_superhelix_radius < float(minimum_final_radius)
    ):
        raise RuntimeError(
            "Final measured superhelix radius {0:.10g} is below the requested "
            "minimum {1:.10g}.".format(
                final_superhelix_radius,
                minimum_final_radius,
            )
        )

    achieved_length = closed_polyline_length(points)
    achieved_writhe = evaluate_curve_it_writhe(points, smooth=False)
    total_curvature, maximum_local_curvature, bending_energy_integral = evaluate_curvature_metrics(
        points,
        adaptive_total=False,
        smooth=False,
    )
    (
        top_terminal_lobe_height_xz,
        bottom_terminal_lobe_height_xz,
        terminal_lobe_height_xz,
        middle_lobe_height_xz,
        lobe_height_mismatch_xz,
    ) = evaluate_xz_lobe_heights(points, target_writhe, stem_height_scaled)
    xz_crossings = count_xz_projection_crossings(points)
    pca_crossings, pca_plane = analyze_pca_projection(points)
    landmarks = analyze_contour_landmarks(
        points,
        target_writhe=target_writhe,
        radius_scaled=radius_scaled,
        stem_height_scaled=stem_height_scaled,
    )

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
    is_integer, nearest = _integer_request(target_writhe)
    if is_integer and xz_crossings != abs(nearest):
        raise ValueError(
            "Serialized-coordinate xz crossing verification failed: expected {0}, found {1}.".format(
                abs(nearest), xz_crossings
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
                "Equal-lobes verification failed: relative lobe mismatch {0:.4%} exceeds {1:.4%}.".format(
                    relative_lobe_mismatch, EQUAL_LOBE_RELATIVE_TOLERANCE
                )
            )

    final_stats = (
        projection_statistics(points, target_writhe, FINAL_PROJECTION_DIRECTIONS)
        if is_integer
        else {"applicable": False, "direction_count": FINAL_PROJECTION_DIRECTIONS}
    )
    phase_sweep = _phase_sweep(target_writhe)
    phase_sweep_pi = phase_sweep / math.pi
    xz_symmetry_rotation = _xz_symmetry_rotation(target_writhe)
    phase_factor = (
        abs(phase_sweep_pi) / abs(float(target_writhe))
        if abs(float(target_writhe)) > 1.0e-12
        else 1.0
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
        plectoneme_phase_turns=abs(float(phase_sweep)) / (2.0 * math.pi),
        opening_angle_deg=candidate.angle_deg,
        curvature_objective=effective_objective,
        total_curvature=total_curvature,
        maximum_local_curvature=maximum_local_curvature,
        bending_energy_integral=bending_energy_integral,
        top_terminal_lobe_height_xz=top_terminal_lobe_height_xz,
        bottom_terminal_lobe_height_xz=bottom_terminal_lobe_height_xz,
        terminal_lobe_height_xz=terminal_lobe_height_xz,
        middle_lobe_height_xz=middle_lobe_height_xz,
        lobe_height_mismatch_xz=lobe_height_mismatch_xz,
        opening_angle_evaluations=int(opening_angle_evaluations),
        loop_control_canonical=candidate.loop_control,
        radius_scaled=radius_scaled,
        final_superhelix_radius=final_superhelix_radius,
        stem_height_scaled=stem_height_scaled,
        solver_iterations=candidate.writhe_solver_iterations,
        landmarks=landmarks,
        phase_trim=float(selected_trim),
        phase_sweep_rad=float(phase_sweep),
        phase_sweep_pi=float(phase_sweep_pi),
        phase_factor=float(phase_factor),
        xz_symmetry_rotation_rad=float(xz_symmetry_rotation),
        trim_enabled=bool(trim_enabled),
        projection_stats=final_stats,
        search_projection_stats=dict(search_stats),
        projection_majority_target=float(qualifying_fraction),
        screening_mode=str(screening_mode),
        minimum_final_radius=(
            None if minimum_final_radius is None else float(minimum_final_radius)
        ),
        projection_search_evaluations=int(search_evaluations),
    )


def generate_sc_points(
    total_length: float,
    target_writhe: float,
    num_points: int = DEFAULT_NUM_POINTS,
    precision: int = DEFAULT_PRECISION,
    curvature_objective: str = DEFAULT_CURVATURE_OBJECTIVE,
    opening_angle_deg: Optional[float] = None,
    trim_enabled: bool = True,
    qualifying_views_percent: float = DEFAULT_QUALIFYING_VIEWS_PERCENT,
    minimum_final_radius: Optional[float] = None,
) -> SCGenerationResult:
    """Generate an optimized/manual plectoneme with V3_5 multi-view screening."""

    validate_inputs(
        total_length,
        target_writhe,
        num_points,
        precision,
        curvature_objective,
        opening_angle_deg,
        trim_enabled,
        qualifying_views_percent,
        minimum_final_radius,
    )
    num_points = int(num_points)
    qualifying_fraction = float(qualifying_views_percent) / 100.0
    is_integer, nearest = _integer_request(target_writhe)
    legacy_geometry = (
        not bool(trim_enabled)
        or not is_integer
        or nearest == 0
    )

    if legacy_geometry:
        _set_active_phase_trim(0.0)
        if opening_angle_deg is None:
            candidate, angle_evaluations = _optimize_opening_angle(
                total_length,
                target_writhe,
                num_points,
                curvature_objective,
            )
            effective_objective = str(curvature_objective)
        else:
            candidate = _fit_angle_candidate(
                total_length,
                target_writhe,
                num_points,
                float(opening_angle_deg),
            )
            angle_evaluations = 1
            effective_objective = OPENING_ANGLE_MODE_MANUAL
        selected_trim = 0.0
        search_stats = (
            _candidate_projection_stats(candidate, total_length, target_writhe)
            if is_integer
            else {"applicable": False, "direction_count": SEARCH_PROJECTION_DIRECTIONS}
        )
        if search_stats is None:
            search_stats = {"applicable": False, "direction_count": SEARCH_PROJECTION_DIRECTIONS}
        search_evaluations = angle_evaluations
    elif minimum_final_radius is not None:
        selected_trim, candidate, search_stats, search_evaluations = (
            _find_radius_constrained_candidate(
                total_length,
                target_writhe,
                num_points,
                int(precision),
                curvature_objective,
                float(minimum_final_radius),
                opening_angle_deg=opening_angle_deg,
            )
        )
        effective_objective = (
            str(curvature_objective)
            if opening_angle_deg is None
            else OPENING_ANGLE_MODE_MANUAL
        )
        angle_evaluations = search_evaluations
    else:
        if opening_angle_deg is None:
            selected_trim, candidate, search_stats, search_evaluations = (
                _find_projection_robust_auto_candidate(
                    total_length,
                    target_writhe,
                    num_points,
                    curvature_objective,
                    qualifying_fraction,
                )
            )
            effective_objective = str(curvature_objective)
            angle_evaluations = search_evaluations
        else:
            selected_trim, candidate, search_stats, search_evaluations = (
                _find_projection_robust_manual_candidate(
                    total_length,
                    target_writhe,
                    num_points,
                    float(opening_angle_deg),
                    curvature_objective,
                    qualifying_fraction,
                )
            )
            effective_objective = OPENING_ANGLE_MODE_MANUAL
            angle_evaluations = search_evaluations

    return _finalize_candidate(
        candidate=candidate,
        total_length=float(total_length),
        target_writhe=float(target_writhe),
        num_points=num_points,
        precision=int(precision),
        effective_objective=effective_objective,
        opening_angle_evaluations=angle_evaluations,
        selected_trim=selected_trim,
        trim_enabled=bool(trim_enabled),
        qualifying_fraction=qualifying_fraction,
        screening_mode=(
            SCREENING_MODE_MINIMUM_RADIUS
            if minimum_final_radius is not None
            else SCREENING_MODE_QUALIFYING_VIEWS
        ),
        minimum_final_radius=minimum_final_radius,
        search_stats=search_stats,
        search_evaluations=search_evaluations,
    )


def write_plain_xyz(points: PointArray, output_path: str, precision: int = DEFAULT_PRECISION) -> None:
    """Write one plain ``x y z`` coordinate row per point."""

    path = Path(output_path)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "{{:.{0}f}} {{:.{0}f}} {{:.{0}f}}\n".format(int(precision))
    with path.open("w", encoding="utf-8") as handle:
        for x, y, z in np.asarray(points, dtype=float):
            handle.write(fmt.format(x, y, z))


def _landmark_line(label: str, landmark: ContourLandmark) -> str:
    return "{0:<31} = {1:.6f}% (XYZ row {2})".format(
        label, landmark.contour_percent, landmark.point_index + 1
    )


def generation_summary(result: SCGenerationResult) -> str:
    """Return a compact V3_5 report for the command line and GUI."""

    residual = result.achieved_writhe - result.requested_writhe
    is_integer, nearest = _integer_request(result.requested_writhe)
    crossing_check = (
        "PASS (required |W| = {0})".format(abs(nearest))
        if is_integer
        else "not constrained for fractional W"
    )
    screening_suffix = (
        " at the default-trim optimum; angle held fixed during phase-trim screening"
        if is_integer
        and nearest != 0
        and result.trim_enabled
        and result.curvature_objective != OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES
        else ""
    )
    if result.curvature_objective == OPENING_ANGLE_MODE_MANUAL:
        objective_label = "manual: use user-provided opening angle"
    elif result.curvature_objective == CURVATURE_OBJECTIVE_TOTAL:
        objective_label = "automatic: minimize total curvature" + screening_suffix
    elif result.curvature_objective == CURVATURE_OBJECTIVE_BENDING_ENERGY:
        objective_label = (
            "automatic: minimize total bending energy (integral kappa^2 ds; default)"
            + screening_suffix
        )
    elif result.curvature_objective == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES:
        objective_label = "automatic: equal fixed-xz terminal/middle lobe heights"
    else:
        objective_label = "automatic: minimize largest local curvature" + screening_suffix

    minimum_bend_radius = math.inf
    if result.maximum_local_curvature > 0.0:
        minimum_bend_radius = 1.0 / result.maximum_local_curvature
    handedness = "planar / no supercoiling"
    if result.requested_writhe > 0.0:
        handedness = "positive-writhe mirror"
    elif result.requested_writhe < 0.0:
        handedness = "negative-writhe mirror"

    lines = [
        "Requested closed contour length = {0:.10g}".format(result.requested_length),
        "Achieved closed contour length  = {0:.10g}".format(result.achieved_length),
        "Requested Gauss writhe = {0:.10g}".format(result.requested_writhe),
        "Achieved Gauss writhe  = {0:.10g}".format(result.achieved_writhe),
        "Writhe residual        = {0:+.3e}".format(residual),
        "Fixed xz projection crossings = {0}".format(result.xz_crossings),
        "Integer xz crossing check      = {0}".format(crossing_check),
        "PCA principal plane    = {0}".format(result.pca_plane),
        "PCA-plane crossings    = {0}".format(result.pca_crossings),
        "plectoneme_phase_turns = {0:.8g}  [|theta_total| / (2*pi)]".format(
            result.plectoneme_phase_turns
        ),
        "Opening-angle selection = {0}".format(objective_label),
        "Selected opening angle  = {0:.8g} deg".format(result.opening_angle_deg),
        "Candidates/search evaluations = {0}".format(result.opening_angle_evaluations),
        "Canonical loop control  = {0:.10g}".format(result.loop_control_canonical),
        "Nominal scaled superhelix radius = {0:.10g}".format(result.radius_scaled),
        (
            "Minimum final measured superhelix radius = not set"
            if result.minimum_final_radius is None
            else "Minimum final measured superhelix radius = {0:.10g} (PASS)".format(
                result.minimum_final_radius
            )
        ),
        "Final measured superhelix radius = {0:.10g}".format(
            result.final_superhelix_radius
        ),
        "Scaled stem height                = {0:.10g}".format(result.stem_height_scaled),
        "Total curvature          = {0:.10g} rad".format(result.total_curvature),
        "Largest local curvature  = {0:.10g} inverse length".format(result.maximum_local_curvature),
        "Minimum local bend radius = {0:.10g}".format(minimum_bend_radius),
        "Reduced bending energy integral = {0:.10g} inverse length".format(
            result.bending_energy_integral
        ),
        "Output geometry          = periodically smoothed once, then rescaled",
        "Final-radius convention  = median serialized arm distance from axis (central 90%)",
        "Writhe convention        = exact written closed polyline used by Curve It",
        "Curvature convention     = periodic spline of written coordinates",
        "Mirror / handedness      = {0}".format(handedness),
    ]

    if result.top_terminal_lobe_height_xz is not None:
        lines.extend(
            [
                "Fixed-xz top terminal lobe z-height    = {0:.10g}".format(
                    result.top_terminal_lobe_height_xz
                ),
                "Fixed-xz bottom terminal lobe z-height = {0:.10g}".format(
                    result.bottom_terminal_lobe_height_xz
                ),
                "Fixed-xz middle lobe z-height          = {0:.10g}".format(
                    result.middle_lobe_height_xz
                ),
                "Fixed-xz terminal-minus-middle height  = {0:+.6g}".format(
                    result.lobe_height_mismatch_xz
                ),
            ]
        )
    else:
        lines.append("Fixed-xz lobe z-heights = N/A (requires integer |W| >= 2)")

    lines.extend(
        [
            "Contour landmarks: percent of closed length from XYZ row 1",
            _landmark_line("Top terminus (+z tip)", result.landmarks.top_terminus),
            _landmark_line("Bottom terminus (-z tip)", result.landmarks.bottom_terminus),
        ]
    )
    if result.landmarks.middle_segment_peaks:
        for segment_number, (peak_a, peak_b) in enumerate(
            result.landmarks.middle_segment_peaks, start=1
        ):
            lines.append(_landmark_line("Middle segment {0} peak A".format(segment_number), peak_a))
            lines.append(_landmark_line("Middle segment {0} peak B".format(segment_number), peak_b))
    else:
        lines.append("Middle-segment peaks = N/A (requires integer |W| >= 2)")

    lines.extend(
        [
            "",
            "V3_5 projection-robustness diagnostics",
            "Arm phase trimming           = {0}".format(
                "enabled (default)" if result.trim_enabled else "disabled"
            ),
            "Projection screening mode     = {0}".format(
                "maximize qualifying views subject to minimum final measured radius"
                if result.screening_mode == SCREENING_MODE_MINIMUM_RADIUS
                else "meet requested qualifying-views percentage"
            ),
            "Arm phase trim              = {0:.8g} * pi".format(result.phase_trim),
            (
                "Opening angle in trim search = reoptimized to preserve equal lobes"
                if result.trim_enabled
                and result.curvature_objective == OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES
                else (
                    "Opening angle in trim search = held fixed"
                    if result.trim_enabled and is_integer and nearest != 0
                    else "Opening angle in trim search = trimming disabled/not applicable"
                )
            ),
            "Actual signed arm phase     = {0:+.10g} * pi".format(result.phase_sweep_pi),
            "Arm phase / (pi*W) factor   = {0:.8g}".format(result.phase_factor),
            "Final xz-symmetry z rotation = {0:+.8g} deg".format(
                math.degrees(result.xz_symmetry_rotation_rad)
            ),
            "Fixed-xz symmetry type       = {0}".format(
                "z-reflection (odd integer W)"
                if is_integer and abs(nearest) % 2 == 1
                else (
                    "central inversion (even integer W)"
                    if is_integer and nearest != 0
                    else "legacy/untrimmed orientation"
                )
            ),
        ]
    )
    stats = result.projection_stats
    if bool(stats.get("applicable", False)):
        fraction = float(stats["target_fraction"])
        if result.screening_mode == SCREENING_MODE_MINIMUM_RADIUS:
            screening_result = (
                "V3_5 final-radius-constrained qualifying views = {0:.2%} "
                "(maximum at feasible phase trim)".format(fraction)
            )
        else:
            screening_result = "V3_5 qualifying-views target (>={0:.6g}%) = {1}".format(
                100.0 * result.projection_majority_target,
                "diagnostic only because trimming is disabled"
                if not result.trim_enabled
                else (
                    "PASS"
                    if fraction >= result.projection_majority_target
                    else "best tested candidate; target not reached"
                ),
            )
        lines.extend(
            [
                "Generic projection directions = {0}".format(stats["direction_count"]),
                "Projection curve samples      = {0}".format(stats["curve_samples"]),
                "Target signed crossing sum    = {0:+d}".format(int(stats["target"])),
                "Views with target signed sum  = {0:.2%}".format(fraction),
                "Strict majority (>50%)        = {0}".format("PASS" if fraction > 0.5 else "FAIL"),
                screening_result,
                "Mean signed crossing sum      = {0:+.8g}".format(float(stats["mean"])),
                "Median / mode                 = {0:+.8g} / {1:+d}".format(
                    float(stats["median"]), int(stats["mode"])
                ),
                "Observed signed-sum range     = {0:+d} to {1:+d}".format(
                    int(stats["minimum"]), int(stats["maximum"])
                ),
                "Signed-sum histogram          = {0}".format(stats["histogram"]),
                "Projection note               = spherical mean should approach Gauss writhe",
            ]
        )
    else:
        lines.extend(
            [
                "Generic projection majority   = N/A for fractional writhe or W = 0",
                "Projection note               = nonzero integer W is required for screening",
            ]
        )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a closed projection-robust textbook-style plectonemic DNA-axis "
            "curve with requested contour length and Gauss writhe."
        )
    )
    parser.add_argument(
        "-L", "--total-length", type=float, default=DEFAULT_TOTAL_LENGTH,
        help="Closed contour length. Default: {0:g}".format(DEFAULT_TOTAL_LENGTH),
    )
    parser.add_argument(
        "-w", "--writhe", type=float, default=DEFAULT_WRITHE,
        help="Target signed Gauss writhe (|W| <= {0:g}). Default: {1:g}".format(
            MAX_ABS_WRITHE, DEFAULT_WRITHE
        ),
    )
    parser.add_argument(
        "--angle-objective", "--curvature-objective", dest="curvature_objective",
        choices=CURVATURE_OBJECTIVES, default=DEFAULT_CURVATURE_OBJECTIVE,
        help=(
            "Automatic opening-angle objective: bending-energy (default), "
            "max-local, total, or equal-lobes."
        ),
    )
    parser.add_argument(
        "-a", "--opening-angle", type=float, default=None,
        help="Use this opening angle in degrees instead of automatic optimization.",
    )
    trim_group = parser.add_mutually_exclusive_group()
    trim_group.add_argument(
        "--trim",
        dest="trim_enabled",
        action="store_true",
        help="Enable arm-phase trimming and projection screening (default).",
    )
    trim_group.add_argument(
        "--no-trim",
        dest="trim_enabled",
        action="store_false",
        help="Use the untrimmed pi*W arm phase; qualifying views are diagnostic only.",
    )
    parser.set_defaults(trim_enabled=True)
    screening_group = parser.add_mutually_exclusive_group()
    screening_group.add_argument(
        "--qualifying-views",
        type=float,
        default=None,
        metavar="PERCENT",
        help=(
            "Use qualifying-view screening instead of the default minimum-final-radius "
            "mode; required percentage of generic views (0-100)."
        ),
    )
    screening_group.add_argument(
        "--minimum-final-radius", "--minimum-final-measured-radius",
        dest="minimum_final_radius",
        type=float,
        default=None,
        metavar="LENGTH",
        help=(
            "Maximize qualifying views while keeping the final measured central-arm "
            "superhelix radius at least this large. Default for eligible integer-W "
            "trimmed designs: {0:g}.".format(DEFAULT_MINIMUM_FINAL_RADIUS)
        ),
    )
    parser.add_argument(
        "-n", "--num-points", type=int, default=DEFAULT_NUM_POINTS,
        help="Number of unique periodic output samples. Default: {0}".format(DEFAULT_NUM_POINTS),
    )
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT,
        help="Output plain-coordinate XYZ file. Default: {0}".format(DEFAULT_OUTPUT),
    )
    parser.add_argument(
        "--precision", type=int, default=DEFAULT_PRECISION,
        help="Decimal places written to XYZ. Default: {0}".format(DEFAULT_PRECISION),
    )
    parser.add_argument("--gui", action="store_true", help="Open the graphical user interface.")
    parser.add_argument("--version", action="version", version="{0} {1}".format(TOOL_NAME, TOOL_VERSION))
    args = parser.parse_args(argv)
    if args.qualifying_views is None:
        args.qualifying_views = DEFAULT_QUALIFYING_VIEWS_PERCENT
        is_integer, nearest = _integer_request(args.writhe)
        if (
            args.minimum_final_radius is None
            and args.trim_enabled
            and is_integer
            and nearest != 0
        ):
            args.minimum_final_radius = DEFAULT_MINIMUM_FINAL_RADIUS
    return args


def run_cli(args: argparse.Namespace) -> SCGenerationResult:
    """Generate, write, and report a curve from parsed CLI arguments."""

    result = generate_sc_points(
        total_length=args.total_length,
        target_writhe=args.writhe,
        num_points=args.num_points,
        precision=args.precision,
        curvature_objective=args.curvature_objective,
        opening_angle_deg=args.opening_angle,
        trim_enabled=args.trim_enabled,
        qualifying_views_percent=args.qualifying_views,
        minimum_final_radius=args.minimum_final_radius,
    )
    write_plain_xyz(result.points, args.output, args.precision)
    print("Wrote {0} unique periodic points to: {1}".format(len(result.points), args.output))
    print("Load this file in Curve It with path type 'closed'.")
    print(generation_summary(result))
    return result


def run_gui() -> None:
    """Open a compact standalone Tkinter interface."""

    try:
        import tkinter as tk
        from tkinter import filedialog, font, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError("Tkinter is not available in this Python installation.") from exc

    root = tk.Tk()
    root.title("{0} {1}".format(TOOL_NAME, TOOL_VERSION))
    root.geometry("1040x800")
    root.minsize(900, 700)
    set_optional_window_icon(root, tk, ("icon.png",), "_generate_sc_icon_image")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        "Help.TButton",
        background="#c9ecff",
        foreground="#075985",
        padding=(3, 1),
        relief="raised",
    )
    style.map(
        "Help.TButton",
        background=(("active", "#9fddff"), ("pressed", "#7dccf5")),
        foreground=(("disabled", "#7a8d99"),),
    )
    title_font = font.Font(root, family="Helvetica", size=16, weight="bold")
    mono_font = font.Font(root, family="Menlo", size=10)

    main = ttk.Frame(root, padding=14)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(2, weight=1)

    ttk.Label(main, text="{0} {1}".format(TOOL_NAME, TOOL_VERSION), font=title_font).grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(
        main,
        text=(
            "Standalone projection-robust plectoneme generator. Integer W is fitted to Gauss writhe "
            "and screened over generic projections."
        ),
        wraplength=980,
        justify="left",
    ).grid(row=1, column=0, sticky="ew", pady=(6, 10))

    body = ttk.Frame(main)
    body.grid(row=2, column=0, sticky="nsew")
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)

    inputs = ttk.LabelFrame(body, text="Inputs", padding=10)
    inputs.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    inputs.columnconfigure(1, weight=1)

    def add_help_button(parent, row: int, column: int, title: str, explanation: str):
        return ttk.Button(
            parent,
            text="?",
            width=2,
            style="Help.TButton",
            command=lambda: messagebox.showinfo(title, explanation, parent=root),
        ).grid(row=row, column=column, padx=(5, 0), sticky="w")

    length_var = tk.StringVar(value=str(DEFAULT_TOTAL_LENGTH))
    writhe_var = tk.StringVar(value=str(DEFAULT_WRITHE))
    curvature_objective_var = tk.StringVar(value=DEFAULT_CURVATURE_OBJECTIVE)
    opening_angle_var = tk.StringVar(value=str(DEFAULT_OPENING_ANGLE_DEG))
    trim_enabled_var = tk.BooleanVar(value=True)
    qualifying_views_var = tk.StringVar(value=str(DEFAULT_QUALIFYING_VIEWS_PERCENT))
    minimum_final_radius_var = tk.StringVar(value=str(DEFAULT_MINIMUM_FINAL_RADIUS))
    points_var = tk.StringVar(value=str(DEFAULT_NUM_POINTS))
    precision_var = tk.StringVar(value=str(DEFAULT_PRECISION))
    output_var = tk.StringVar(value=DEFAULT_OUTPUT)

    fields = (
        ("Length L:", length_var),
        ("Writhe W:", writhe_var),
        ("Minimum final measured radius:", minimum_final_radius_var),
        ("Qualifying views (%):", qualifying_views_var),
        ("Points:", points_var),
        ("Precision:", precision_var),
    )
    field_help = {
        "Length L:": (
            "Length L (--total-length)",
            "The requested closed contour length, including the last-to-first seam. "
            "Generate SC smooths the curve once and uniformly rescales it so the written "
            "closed polyline has this length.\n\nExample: use 1071 for a 1071-A centerline.",
        ),
        "Writhe W:": (
            "Writhe W (--writhe)",
            "The requested signed Gauss writhe, supported for |W| <= 10. Positive and "
            "negative values produce mirror-related curves. Projection screening requires "
            "a nonzero integer W.\n\nExample: 3 or -3 requests three signed crossings in "
            "the qualifying projections.",
        ),
        "Qualifying views (%):": (
            "Qualifying views (--qualifying-views)",
            "The minimum percentage of 256 deterministic generic views whose signed "
            "crossing sum must equal W. Entering a percentage selects this mode after "
            "the minimum final measured radius is cleared. This field is disabled when "
            "trimming is off or a minimum final measured radius is set.\n\nExample: "
            "clear Radius, then enter 60 to request at least 60% qualifying views.",
        ),
        "Minimum final measured radius:": (
            "Minimum final measured radius (--minimum-final-radius)",
            "The default screening mode. Generate SC chooses the largest feasible phase "
            "trim whose radius, measured from the final serialized central 90% of the "
            "arms, remains at least this value. This maximizes qualifying views within "
            "the phase-trim family. Clear the field to use Qualifying views instead.\n\n"
            "Default: {0:g} length units.".format(DEFAULT_MINIMUM_FINAL_RADIUS),
        ),
        "Points:": (
            "Points (--num-points)",
            "The number of unique periodic XYZ samples. More points improve discrete frame "
            "transport and mapping accuracy but increase generation and Curve It runtime. "
            "The default is 2000; the final point is not duplicated.\n\nExample: 2000.",
        ),
        "Precision:": (
            "Precision (--precision)",
            "The number of decimal places written for each x, y, and z coordinate. Final "
            "length and writhe are verified from these rounded coordinates. Too little "
            "precision can collapse samples or move writhe outside tolerance.\n\nDefault: 8.",
        ),
    }
    qualifying_views_label = None
    qualifying_views_entry = None
    minimum_radius_label = None
    minimum_radius_entry = None
    for row, (label, variable) in enumerate(fields):
        label_widget = ttk.Label(inputs, text=label)
        label_widget.grid(row=row, column=0, sticky="e", padx=(0, 6), pady=5)
        entry_widget = ttk.Entry(inputs, textvariable=variable, width=17)
        entry_widget.grid(row=row, column=1, sticky="ew", pady=5)
        help_title, help_text = field_help[label]
        add_help_button(inputs, row, 2, help_title, help_text)
        if variable is qualifying_views_var:
            qualifying_views_label = label_widget
            qualifying_views_entry = entry_widget
        elif variable is minimum_final_radius_var:
            minimum_radius_label = label_widget
            minimum_radius_entry = entry_widget

    objective_frame = ttk.LabelFrame(inputs, text="Opening-angle selection", padding=8)
    objective_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    options = (
        (
            "Minimize total bending energy (default)",
            CURVATURE_OBJECTIVE_BENDING_ENERGY,
            "Chooses the opening angle that minimizes integral kappa(s)^2 ds. For a "
            "homogeneous isotropic rod with constant bending rigidity A, physical bending "
            "energy is A/2 times this integral.",
        ),
        (
            "Minimize largest local curvature",
            CURVATURE_OBJECTIVE_MAX_LOCAL,
            "Chooses the opening angle that minimizes max kappa(s), equivalently maximizing "
            "the minimum local bend radius 1/max(kappa).",
        ),
        (
            "Minimize total curvature",
            CURVATURE_OBJECTIVE_TOTAL,
            "Chooses the opening angle that minimizes integral kappa(s) ds over the closed "
            "curve. This differs from bending energy, which integrates kappa squared.",
        ),
        (
            "Equal terminal/middle lobe heights",
            OPENING_ANGLE_OBJECTIVE_EQUAL_LOBES,
            "Chooses the angle that matches terminal and middle fixed-xz lobe heights. It "
            "requires integer |W| >= 2 and re-solves the equality for every tested trim.",
        ),
        (
            "Use provided angle",
            OPENING_ANGLE_MODE_MANUAL,
            "Keeps the entered opening angle fixed while fitting exact writhe and screening "
            "phase trim. The angle must be strictly between 0 and 90 degrees.\n\nExample: 25.",
        ),
    )
    for row, (label, value, explanation) in enumerate(options):
        ttk.Radiobutton(
            objective_frame,
            text=label,
            variable=curvature_objective_var,
            value=value,
        ).grid(row=row, column=0, sticky="w", pady=2)
        add_help_button(
            objective_frame,
            row,
            3,
            "Opening-angle option",
            explanation,
        )
    manual_angle_entry = ttk.Entry(objective_frame, textvariable=opening_angle_var, width=9)
    manual_angle_entry.grid(row=4, column=1, sticky="w", padx=(8, 0))
    ttk.Label(objective_frame, text="deg").grid(row=4, column=2, sticky="w", padx=(4, 0))
    ttk.Checkbutton(
        objective_frame,
        text="Enable arm-phase trimming (default)",
        variable=trim_enabled_var,
    ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
    add_help_button(
        objective_frame,
        5,
        3,
        "Arm-phase trimming (--trim / --no-trim)",
        "Shortens the centered arm phase below pi*|W| and refits the end loops to preserve "
        "exact writhe. Larger trim generally decreases the final measured radius and "
        "increases the fraction of qualifying views. The final z rotation preserves the symmetric xz "
        "presentation and |W| fixed-xz crossings for trim < 1. Disable trimming to retain "
        "the legacy pi*W arm phase; projection statistics then become diagnostic only.",
    )

    def update_manual_state(*_args) -> None:
        manual_angle_entry.configure(
            state="normal" if curvature_objective_var.get() == OPENING_ANGLE_MODE_MANUAL else "disabled"
        )

    def update_screening_state(*_args) -> None:
        trimming_enabled = bool(trim_enabled_var.get())
        radius_mode = bool(minimum_final_radius_var.get().strip())
        qualifying_state = "!disabled" if trimming_enabled and not radius_mode else "disabled"
        radius_state = "!disabled" if trimming_enabled else "disabled"
        qualifying_views_label.state((qualifying_state,))
        qualifying_views_entry.state((qualifying_state,))
        minimum_radius_label.state((radius_state,))
        minimum_radius_entry.state((radius_state,))

    curvature_objective_var.trace_add("write", update_manual_state)
    trim_enabled_var.trace_add("write", update_screening_state)
    minimum_final_radius_var.trace_add("write", update_screening_state)
    update_manual_state()
    update_screening_state()

    output_frame = ttk.LabelFrame(body, text="Output and verification", padding=10)
    output_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
    output_frame.columnconfigure(0, weight=1)
    output_frame.rowconfigure(2, weight=1)

    output_row = ttk.Frame(output_frame)
    output_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    output_row.columnconfigure(0, weight=1)
    ttk.Entry(output_row, textvariable=output_var).grid(row=0, column=0, sticky="ew")
    add_help_button(
        output_row,
        0,
        2,
        "Output file (--output)",
        "Destination for the plain-coordinate XYZ file. Each row contains x y z; there is "
        "no atom-count header and the first point is not repeated at the end. Load this file "
        "in Curve It with Path type: closed.\n\nExample: sc_W3_ABend.xyz.",
    )

    def browse_output() -> None:
        filename = filedialog.asksaveasfilename(
            title="Save plectonemic curve",
            defaultextension=".xyz",
            filetypes=(("XYZ files", "*.xyz"), ("Text files", "*.txt"), ("All files", "*.*")),
        )
        if filename:
            output_var.set(filename)

    ttk.Button(output_row, text="Browse...", command=browse_output).grid(row=0, column=1, padx=(6, 0))
    ttk.Label(output_frame, text="Derived values and checks").grid(row=1, column=0, sticky="w")

    summary_box = tk.Text(output_frame, wrap="none", font=mono_font, bg="white", padx=8, pady=8)
    summary_box.grid(row=2, column=0, sticky="nsew", pady=(4, 0))
    y_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=summary_box.yview)
    y_scroll.grid(row=2, column=1, sticky="ns", pady=(4, 0))
    x_scroll = ttk.Scrollbar(output_frame, orient="horizontal", command=summary_box.xview)
    x_scroll.grid(row=3, column=0, sticky="ew")
    summary_box.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    def set_summary(text: str) -> None:
        summary_box.configure(state="normal")
        summary_box.delete("1.0", "end")
        summary_box.insert("1.0", text)
        summary_box.configure(state="disabled")
        summary_box.yview_moveto(0.0)
        summary_box.xview_moveto(0.0)

    set_summary(
        "Preview / verify function:\n"
        "  Validates the current inputs, constructs the complete curve in memory, and runs\n"
        "  the same final length, writhe, crossing, radius, curvature, and projection checks\n"
        "  used by Generate XYZ. It displays the full report here without writing a file.\n\n"
        "Generate XYZ function:\n"
        "  Performs the same preview/verification (or reuses its cached result when the\n"
        "  geometry inputs are unchanged), then writes the selected XYZ output file."
    )
    status_var = tk.StringVar(value="Ready.")
    ttk.Label(main, textvariable=status_var).grid(row=3, column=0, sticky="w", pady=(10, 0))

    cached_key = None
    cached_result: Optional[SCGenerationResult] = None

    def parse_gui_values():
        total_length = float(length_var.get().strip())
        target_writhe = float(writhe_var.get().strip())
        selection = curvature_objective_var.get().strip()
        objective = selection
        manual_angle: Optional[float] = None
        if selection == OPENING_ANGLE_MODE_MANUAL:
            objective = DEFAULT_CURVATURE_OBJECTIVE
            manual_angle = float(opening_angle_var.get().strip())
        num_points = int(points_var.get().strip())
        precision = int(precision_var.get().strip())
        trim_enabled = bool(trim_enabled_var.get())
        minimum_radius_text = minimum_final_radius_var.get().strip()
        minimum_final_radius = (
            float(minimum_radius_text)
            if trim_enabled and minimum_radius_text
            else None
        )
        qualifying_views = (
            float(qualifying_views_var.get().strip())
            if trim_enabled and minimum_final_radius is None
            else DEFAULT_QUALIFYING_VIEWS_PERCENT
        )
        output = output_var.get().strip()
        if not output:
            raise ValueError("Please specify an output file.")
        validate_inputs(
            total_length,
            target_writhe,
            num_points,
            precision,
            objective,
            manual_angle,
            trim_enabled,
            qualifying_views,
            minimum_final_radius,
        )
        return (
            total_length,
            target_writhe,
            objective,
            manual_angle,
            trim_enabled,
            qualifying_views,
            minimum_final_radius,
            num_points,
            precision,
            output,
        )

    def obtain_result():
        nonlocal cached_key, cached_result
        values = parse_gui_values()
        (
            total_length,
            target_writhe,
            objective,
            manual_angle,
            trim_enabled,
            qualifying_views,
            minimum_final_radius,
            num_points,
            precision,
            output,
        ) = values
        key = (
            total_length,
            target_writhe,
            objective,
            manual_angle,
            trim_enabled,
            qualifying_views,
            minimum_final_radius,
            num_points,
            precision,
        )
        if cached_result is None or cached_key != key:
            status_var.set("Fitting Gauss writhe and screening generic projections...")
            root.update_idletasks()
            cached_result = generate_sc_points(
                total_length=total_length,
                target_writhe=target_writhe,
                num_points=num_points,
                precision=precision,
                curvature_objective=objective,
                opening_angle_deg=manual_angle,
                trim_enabled=trim_enabled,
                qualifying_views_percent=qualifying_views,
                minimum_final_radius=minimum_final_radius,
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
            stats = result.projection_stats
            projection_line = "N/A"
            if bool(stats.get("applicable", False)):
                projection_line = "{0:.1%} of {1} views".format(
                    float(stats["target_fraction"]), int(stats["direction_count"])
                )
            messagebox.showinfo(
                "Generate SC complete",
                "Wrote {0} points to:\n{1}\n\nAchieved writhe: {2:.6f}\n"
                "Fixed xz crossings: {3}\nGeneric target projections: {4}\n"
                "Selected angle: {5:.4f} deg\nPhase trim: {6:.3f} pi\n"
                "Load it in Curve It as a closed path.".format(
                    len(result.points),
                    output,
                    result.achieved_writhe,
                    result.xz_crossings,
                    projection_line,
                    result.opening_angle_deg,
                    result.phase_trim,
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
            raise SystemExit("{0} {1} error: {2}".format(TOOL_NAME, TOOL_VERSION, exc)) from exc


if __name__ == "__main__":
    main()
