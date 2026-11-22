"""Which estimator Curve It's auto mode uses for total curvature (V3_16).

A periodic spline through a polygon rings at its corners. Before V3_16, auto
mode took the turning-angle sum only for corners of 45 degrees or more, so
KnotPlot's polygons, whose edges kp2xyz cuts into collinear points and whose
corners are shallower, were reported from the spline: 1313.9 degrees for
`torus 2 6` against a turning-angle sum of 867.6. These tests pin down both
sides. Subdivided polygons and samples kinked between points use the
turning-angle sum; sampled smooth curves keep the spline value they always
had. Every curve is generated here, so the test needs no local fixtures.
"""

import math
import unittest

import numpy as np

import curve_it


def _turning_sum(points):
    return curve_it.compute_discrete_total_curvature(points, closed=True)


def _spline(points):
    return float(curve_it.fit_spline_and_calculate_curvature(points))


def _torus_13(n, R=8.25, r=2.75):
    """The (1,3) torus curve, one component of the (2,6) torus link."""
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack((
        (R + r * np.cos(3.0 * t)) * np.cos(t),
        (R + r * np.cos(3.0 * t)) * np.sin(t),
        r * np.sin(3.0 * t),
    ))


def _subdivide(corners, pieces):
    """Cut every edge of a closed polygon into `pieces` collinear pieces."""
    out = []
    for i in range(len(corners)):
        a, b = corners[i], corners[(i + 1) % len(corners)]
        for k in range(pieces):
            out.append(a + (b - a) * (k / pieces))
    return np.array(out)


def _resample_evenly(points, n):
    """n evenly spaced samples along a closed polyline; corners get split."""
    loop = np.vstack((points, points[:1]))
    cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))))
    s = np.linspace(0.0, cum[-1], n, endpoint=False)
    return np.column_stack([np.interp(s, cum, loop[:, d]) for d in range(3)])


def _regular_polygon(n, radius=30.0):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(t), radius * np.sin(t), np.zeros(n)))


def _star10(outer=50.0, inner=18.0):
    """A five-pointed star given by its ten corners only."""
    t = np.radians(-90.0 + 36.0 * np.arange(10))
    r = np.where(np.arange(10) % 2 == 0, outer, inner)
    return np.column_stack((r * np.cos(t), r * np.sin(t), np.zeros(10)))


def _smooth_curves():
    rng = np.random.default_rng(7)
    coef = rng.normal(size=(3, 4, 2)) / np.arange(1, 5)[None, :, None] ** 1.5

    def fourier(t):
        cols = []
        for d in range(3):
            v = np.zeros_like(t)
            for k in range(4):
                v += coef[d, k, 0] * np.cos((k + 1) * t) + coef[d, k, 1] * np.sin((k + 1) * t)
            cols.append(v)
        return np.column_stack(cols)

    return {
        "circle": lambda t: np.column_stack((np.cos(t), np.sin(t), 0.0 * t)),
        "ellipse": lambda t: np.column_stack((3.0 * np.cos(t), np.sin(t), 0.5 * np.sin(t))),
        "trefoil": lambda t: np.column_stack(((2 + np.cos(3 * t)) * np.cos(2 * t),
                                              (2 + np.cos(3 * t)) * np.sin(2 * t), np.sin(3 * t))),
        "torus_knot_2_5": lambda t: np.column_stack(((2 + np.cos(5 * t)) * np.cos(2 * t),
                                                     (2 + np.cos(5 * t)) * np.sin(2 * t), np.sin(5 * t))),
        "figure_eight": lambda t: np.column_stack(((2 + np.cos(2 * t)) * np.cos(3 * t),
                                                   (2 + np.cos(2 * t)) * np.sin(3 * t), np.sin(4 * t))),
        "torus_1_3": lambda t: _torus_13(len(t)),
        "fourier": fourier,
    }


def _kinked_hexagram(n=600, arc_points=40, fillet=0.5, amp=0.18, phase_deg=-75.2):
    """A rounded triangle drawn with coarse 40-point arcs, then resampled
    linearly to n points and given a three-period weave in z. The samples sit
    on the chords of the coarse arcs, so the curve is kinked between them."""
    verts = [np.array([math.cos(k * 2 * math.pi / 3 + math.pi / 2),
                       math.sin(k * 2 * math.pi / 3 + math.pi / 2)]) for k in range(3)]
    rf = fillet * 0.5
    half = math.pi / 6
    d = rf / math.tan(half)
    dense = []
    for k in range(3):
        p, pm, pn = verts[k], verts[(k - 1) % 3], verts[(k + 1) % 3]
        u = (pm - p) / np.linalg.norm(pm - p)
        v = (pn - p) / np.linalg.norm(pn - p)
        a0, b0 = p + u * d, p + v * d
        bis = (u + v) / np.linalg.norm(u + v)
        c = p + bis * (rf / math.sin(half))
        ang0 = math.atan2(a0[1] - c[1], a0[0] - c[0])
        ang1 = math.atan2(b0[1] - c[1], b0[0] - c[0])
        dang = (ang1 - ang0 + math.pi) % (2 * math.pi) - math.pi
        for t in np.linspace(0, 1, arc_points):
            dense.append(c + rf * np.array([math.cos(ang0 + dang * t), math.sin(ang0 + dang * t)]))
        pn_start = pn + ((p - pn) / np.linalg.norm(p - pn)) * d
        for t in np.linspace(0, 1, arc_points)[1:-1]:
            dense.append(b0 + (pn_start - b0) * t)
    dense = np.column_stack((np.array(dense), np.zeros(len(dense))))
    xy = _resample_evenly(dense, n)
    s = np.arange(n) / n
    xy[:, 2] = amp * np.sin(3 * 2 * math.pi * s + math.radians(phase_deg))
    return xy


class RunLengthTests(unittest.TestCase):

    def test_runs_wrap_round_the_seam(self):
        runs = curve_it._circular_run_lengths(np.array([1, 0, 0, 1, 1], dtype=bool))
        self.assertEqual(runs, [3])

    def test_edge_cases(self):
        self.assertEqual(curve_it._circular_run_lengths(np.zeros(4, dtype=bool)), [])
        self.assertEqual(curve_it._circular_run_lengths(np.ones(4, dtype=bool)), [4])
        self.assertEqual(sorted(curve_it._circular_run_lengths(
            np.array([0, 1, 1, 0, 1, 0, 0], dtype=bool))), [1, 2])


class PolygonTests(unittest.TestCase):

    def test_subdivided_polygon_with_shallow_corners(self):
        # 79 corners turning about 11 degrees each, like a component of
        # KnotPlot's torus 2 6, with every edge cut into five pieces.
        points = _subdivide(_torus_13(79), 5)
        self.assertLess(float(np.max(curve_it._turning_angles_deg_closed(points))), 45.0)
        self.assertTrue(curve_it.curve_looks_polygonal(points))
        value, method = curve_it.estimate_total_curvature(points)
        self.assertEqual(method, "polyline (auto: polygon-like curve)")
        self.assertAlmostEqual(value, _turning_sum(points), places=12)
        # The spline it used to report rings at the corners.
        self.assertGreater(_spline(points), 1.2 * value)

    def test_corners_split_between_samples(self):
        # Evenly spaced samples, written to six decimals as kp2xyz does: most
        # corners now fall between two samples and are split over both.
        points = np.round(_resample_evenly(_torus_13(79), 400), 6)
        self.assertTrue(curve_it.curve_looks_polygonal(points))
        self.assertEqual(curve_it.estimate_total_curvature(points)[1],
                         "polyline (auto: polygon-like curve)")

    def test_sharp_corners_as_before(self):
        square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        points = _subdivide(square, 40)
        self.assertTrue(curve_it.curve_looks_polygonal(points))
        value, _method = curve_it.estimate_total_curvature(points)
        self.assertAlmostEqual(math.degrees(value), 360.0, places=9)

    def test_corners_alone_keep_the_spline(self):
        # A polygon given by its corners only cannot be told from a coarse
        # sample of a smooth curve; it is not called polygon-like.
        for name, points in (("12-gon", _regular_polygon(12)), ("24-gon", _regular_polygon(24)),
                             ("ten-corner star", _star10())):
            with self.subTest(name=name):
                self.assertFalse(curve_it.curve_looks_polygonal(points))
                value, method = curve_it.estimate_total_curvature(points)
                self.assertEqual(method, "spline (auto)")
                self.assertEqual(value, _spline(points))

    def test_long_straight_runs_between_long_arcs(self):
        # A densely sampled stadium is mostly straight vertices, but its
        # turning vertices form two long arcs, not isolated corners.
        arc = np.linspace(-math.pi / 2, math.pi / 2, 120, endpoint=False)
        right = np.column_stack((5 + np.cos(arc), np.sin(arc), 0 * arc))
        left = np.column_stack((-5 - np.cos(arc), -np.sin(arc), 0 * arc))
        top = np.column_stack((np.linspace(5, -5, 300, endpoint=False), np.ones(300), np.zeros(300)))
        bottom = np.column_stack((np.linspace(-5, 5, 300, endpoint=False), -np.ones(300), np.zeros(300)))
        points = np.vstack((right, top, left, bottom))
        self.assertGreater(float(np.mean(curve_it._turning_angles_deg_closed(points) < 1e-6)), 0.5)
        self.assertFalse(curve_it.curve_looks_polygonal(points))


class SplineCheckTests(unittest.TestCase):

    def test_kinked_sample_uses_the_turning_angle_sum(self):
        points = _kinked_hexagram()
        self.assertFalse(curve_it.curve_looks_polygonal(points))
        value, method = curve_it.estimate_total_curvature(points)
        self.assertTrue(method.startswith("polyline (auto: spline overshoot "), method)
        self.assertAlmostEqual(value, _turning_sum(points), places=12)
        excess = _spline(points) / value - 1.0
        self.assertGreater(excess, 3.0 * curve_it.spline_excess_allowance(points))

    def test_smooth_curves_keep_the_spline(self):
        # Coarse or dense, a sampled smooth curve is reported exactly as
        # before V3_16: from the periodic spline.
        for name, curve in _smooth_curves().items():
            for n in (8, 16, 24, 48, 100, 400, 800):
                points = curve(np.linspace(0.0, 2.0 * math.pi, n, endpoint=False))
                with self.subTest(curve=name, n=n):
                    self.assertFalse(curve_it.curve_looks_polygonal(points))
                    value, method = curve_it.estimate_total_curvature(points)
                    self.assertEqual(method, "spline (auto)")
                    self.assertEqual(value, _spline(points))

    def test_p_densified_curve_is_polygon_like(self):
        # --interp-mode p inserts collinear points, so a smooth sample turns
        # into a subdivided polygon; the spline through it used to ring at
        # every original vertex (680 degrees for a planar ellipse).
        t = np.linspace(0.0, 2.0 * math.pi, 183, endpoint=False)
        ellipse = np.column_stack((40.0 * np.cos(t), 25.0 * np.sin(t), np.zeros_like(t)))
        dense = curve_it.apply_curve_interpolation(ellipse, "p", 0, 3, closed=True)
        self.assertTrue(curve_it.curve_looks_polygonal(dense))
        value, method = curve_it.estimate_total_curvature(dense)
        self.assertEqual(method, "polyline (auto: polygon-like curve)")
        self.assertAlmostEqual(math.degrees(value), 360.0, delta=1e-3)


class InterfaceTests(unittest.TestCase):

    def test_compute_curve_metrics_return_shapes(self):
        points = _subdivide(_torus_13(79), 5)
        three = curve_it.compute_curve_metrics(points, closed=True)
        four = curve_it.compute_curve_metrics(points, closed=True, return_method=True)
        self.assertEqual(len(three), 3)
        self.assertEqual(four[:3], three)
        self.assertEqual(four[3], "polyline (auto: polygon-like curve)")
        self.assertEqual(curve_it.compute_curve_metrics(points, closed=False)[1:], (None, None))
        self.assertEqual(curve_it.compute_curve_metrics(points, closed=False, return_method=True)[3],
                         "N/A (open curve)")

    def test_manual_modes(self):
        points = _kinked_hexagram()
        value, method = curve_it.estimate_total_curvature(points, "polyline")
        self.assertEqual(method, "polyline (manual)")
        self.assertAlmostEqual(value, _turning_sum(points), places=12)
        self.assertEqual(curve_it.estimate_total_curvature(points, "spline")[1], "spline (manual)")


if __name__ == "__main__":
    unittest.main()
