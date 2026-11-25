"""Corner Curve It (curve_it_lib/corner_curve_it.py, Curve It V3_18): a helix with corners on arcs in one plane.

Every input is a synthetic duplex built here from one base-pair template and a screw (3.4 A rise, 36 deg twist,
so ten base pairs make a whole turn), with rigid kinks after chosen base pairs: no local fixtures are needed.
A kink bends about the base pair's long axis, which turns with the helix, so sides of whole turns put every
corner on the same side of the plane (all in, or all out, without extra twist) and sides of one and a half turns
alternate them (in, out, in, out).  Z-DNA sides repeat every two base pairs (C anti, G syn) with alternating
steps.  The tests check that

* the corners are found from the helix axis where they were built;
* a closed ring closes flat with every corner bending the way it was asked to, for in, out and mixed patterns,
  and needs no extra twist when the lengths suit the pattern;
* the wrong pattern for the lengths is reported with the twist it needs and a length or phase that fixes it;
* an open path with straight sides puts every corner in one plane;
* a Z-DNA side is mapped about the screw that carries each base pair two steps on, with that screw's twist per
  step, and its suggested lengths change by whole repeats;
* the window's commands and the command line run end to end.
"""

import io
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

import curve_it
from curve_it_lib import corner_curve_it as cc

RISE, TWIST = 3.4, 36.0

# One base pair in its own frame: z along the helix axis, the guide strand (chain A) on -x running up, its
# partner (chain B) the pseudo-dyad image on +x running down.  Two base N/O pairs 2.8 A apart pair them.
GUIDE = [("P", (-6.2, 6.6, 2.0)), ("O5'", (-6.6, 5.6, 1.2)), ("C5'", (-7.4, 5.0, 0.4)), ("C4'", (-7.0, 3.8, -0.2)),
         ("O4'", (-5.7, 3.4, 0.0)), ("C3'", (-7.6, 3.4, -1.5)), ("O3'", (-8.6, 4.3, -2.0)), ("C1'", (-5.4, 2.0, 0.0)),
         ("N9", (-4.1, 1.5, 0.0)), ("C4", (-3.4, 0.4, 0.0)), ("C6", (-2.4, -1.3, 0.0)), ("N1", (-1.5, -0.5, 0.0)),
         ("N6", (-1.6, -2.7, 0.0))]
PARTNER = [("P", (6.2, 6.6, -2.0)), ("O5'", (6.6, 5.6, -1.2)), ("C5'", (7.4, 5.0, -0.4)), ("C4'", (7.0, 3.8, 0.2)),
           ("O4'", (5.7, 3.4, 0.0)), ("C3'", (7.6, 3.4, 1.5)), ("O3'", (8.6, 4.3, 2.0)), ("C1'", (5.4, 2.0, 0.0)),
           ("N1", (4.1, 1.5, 0.0)), ("C2", (3.4, 0.4, 0.0)), ("C4", (2.2, -1.3, 0.0)), ("N3", (1.3, -0.5, 0.0)),
           ("O4", (1.2, -2.7, 0.0))]


def _rz(a):
    a = math.radians(a)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def _rx(a):
    a = math.radians(a)
    return np.array([[1.0, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def duplex_pdb(lengths, bend=-30.0, skews=None):
    """PDB text of a duplex of sum(lengths) base pairs with a kink of `bend` degrees (about the base pair's long
    axis, halfway to the next pair) after each segment but the last.  Chain A runs 5'->3' up the helix, chain B
    back down it, so B1 pairs with the last residue of A.  skews[c] shifts everything after kink c along that
    axis, so the two helix axes there pass that far apart instead of meeting."""
    n = sum(lengths)
    kink_at = list(np.cumsum(lengths)[:-1])
    kinks = set(kink_at)
    skews = list(skews or [0.0] * len(kink_at))
    R, o = np.eye(3), np.zeros(3)
    frames = []
    for j in range(1, n + 1):
        frames.append((R.copy(), o.copy()))
        step_o = np.array([0.0, 0.0, RISE])
        if j in kinks:
            m = np.array([0.0, 0.0, RISE / 2])
            o = o + R @ (_rx(bend) @ (step_o - m) + m + np.array([skews[kink_at.index(j)], 0.0, 0.0]))
            R = R @ _rx(bend) @ _rz(TWIST)
        else:
            o = o + R @ step_o
            R = R @ _rz(TWIST)
    return _duplex_text(frames, [BASE_PAIRS["B"]] * n)


def _duplex_text(frames, pairs):
    """PDB text of base pairs placed by frames: chain A, the guide residues, 5'->3' up the helix; chain B, their
    partners, back down it.  pairs[j] = ((residue name, template) of the guide residue, the same of its partner)."""
    n = len(frames)
    lines, serial = [], 1

    def atom(name, resn, chain, resi, xyz):
        nonlocal serial
        nm = f" {name:<3s}" if len(name) < 4 else name
        el = name[0]
        lines.append(f"ATOM  {serial:5d} {nm} {resn:>3s} {chain}{resi:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                     f"  1.00  0.00          {el:>2s}")
        serial += 1
    for j, (R_, o_) in enumerate(frames, 1):
        resn, template = pairs[j - 1][0]
        for name, p in template:
            atom(name, resn, "A", j, R_ @ np.array(p) + o_)
    lines.append("TER")
    for b in range(1, n + 1):
        R_, o_ = frames[n - b]
        resn, template = pairs[n - b][1]
        for name, p in template:
            atom(name, resn, "B", b, R_ @ np.array(p) + o_)
    lines.append("TER")
    return "\n".join(lines) + "\nEND\n"


# Z-DNA repeats every two base pairs, 5'-CG-3': C anti, then G syn.  The synthetic Z-DNA here uses the same
# base-pair template, with the syn residue's sugar and phosphate turned 180 deg about its glycosidic bond, and
# alternating left-handed steps (CpG -9 deg, 3.55 A; GpC -51 deg, 3.70 A): -60 deg and 7.25 A per two steps, as in
# BNP_NA's Z-DNA.  No one screw carries each of its base pairs onto the next, and the residue-to-residue screws
# alternate, as they do on real Z-DNA.
Z_STEPS = {"CpG": (-9.0, 3.55), "GpC": (-51.0, 3.70)}
BACKBONE = ("P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'")


def _syn(template, base_atom):
    """The template with its sugar and phosphate turned 180 deg about the glycosidic bond (C1' to base_atom):
    the base keeps its pairing, but the residue no longer superposes on an anti one without a large turn."""
    pos = dict(template)
    c1 = np.array(pos["C1'"])
    a = np.array(pos[base_atom]) - c1
    a /= np.linalg.norm(a)
    flip = 2.0 * np.outer(a, a) - np.eye(3)
    return [(name, tuple(c1 + flip @ (np.array(p) - c1)) if name in BACKBONE else p) for name, p in template]


# (residue name, template) of the guide residue and of its partner, for each kind of base pair
BASE_PAIRS = {
    "B": (("DA", GUIDE), ("DT", PARTNER)),
    "C": (("DC", GUIDE), ("DG", _syn(PARTNER, "N1"))),        # Z-DNA: C anti on the guide strand, G syn opposite
    "G": (("DG", _syn(GUIDE, "N9")), ("DC", PARTNER)),        # Z-DNA: G syn on the guide strand, C anti opposite
}


def bz_duplex_pdb(segments, bend=-30.0, frames_out=None):
    """PDB text of a duplex of B-DNA-like segments ("B": 36 deg, 3.4 A per step) and Z-DNA-like ones ("Z":
    5'-(CG)n-3' on the guide strand), given as (kind, base pairs), with a kink of `bend` degrees after each
    segment but the last, as in duplex_pdb.  The step into a base pair is its own segment's (into a Z segment's
    first C, a GpC step), so B sides of 20 bp and Z sides of 24 bp (two turns each) put every corner on the same
    side.  frames_out, if a list, receives every base pair's frame (rotation, origin)."""
    kinds = []
    for kind, n in segments:
        kinds += ["B"] * n if kind == "B" else ["C" if p % 2 == 0 else "G" for p in range(n)]
    kink_after = set(int(k) for k in np.cumsum([n for _, n in segments])[:-1])
    R, o = np.eye(3), np.zeros(3)
    frames = []
    for j, kind in enumerate(kinds, 1):
        if j > 1:
            twist, rise = (TWIST, RISE) if kind == "B" else Z_STEPS["GpC" if kind == "C" else "CpG"]
            step_o = np.array([0.0, 0.0, rise])
            if j - 1 in kink_after:
                m = np.array([0.0, 0.0, rise / 2])
                o = o + R @ (_rx(bend) @ (step_o - m) + m)
                R = R @ _rx(bend) @ _rz(twist)
            else:
                o = o + R @ step_o
                R = R @ _rz(twist)
        frames.append((R.copy(), o.copy()))
    if frames_out is not None:
        frames_out.extend(frames)
    return _duplex_text(frames, [BASE_PAIRS[k] for k in kinds])


class CornerCase(unittest.TestCase):
    """Runs a case once per test class through cc.run() on a temporary PDB."""

    def run_case(self, lengths, phases, path="closed", radius=None, check=False, corners="auto", skews=None,
                 text=None):
        tmp = tempfile.mkdtemp()
        pdb = os.path.join(tmp, "duplex.pdb")
        with open(pdb, "w") as f:
            f.write(duplex_pdb(lengths, skews=skews) if text is None else text)
        out = os.path.join(tmp, "duplex_curved.pdb")
        curve = os.path.join(tmp, "duplex_axis.xyz")
        buf = io.StringIO()
        with redirect_stdout(buf):
            res = cc.run(pdb, corners, phases, path, radius, out, None if check else curve, check_only=check)
        res["log"] = buf.getvalue()
        res["paths"] = (pdb, out, curve)
        return res

    def assertFlatAndPhased(self, res, closure=0.1, plane=0.5):
        diag, plan = res["diag"], res["plan"]
        if plan.closed:
            self.assertLess(diag["closure"], closure, res["log"])
        self.assertLess(max(diag["tilts"], default=0.0), plane, res["log"])
        for cd in diag["corners"]:
            self.assertLess(cd["axis"], plane, res["log"])

    def max_twist_per_bp(self, res):
        return max(abs(p[2]) / s.steps for p, s in zip(res["params"], res["plan"].sides))


class TestCornerDetection(CornerCase):
    def test_corners_found_where_built(self):
        X_lengths = [5, 20, 20, 20, 15]
        atoms = curve_it.parse_pdb_atoms(duplex_pdb(X_lengths))
        L = cc.layout(atoms, "auto")
        self.assertEqual(L.guide, "A")
        self.assertEqual([cc.res_label(L.guide_keys[p]) for p in L.corner_pos], ["A5", "A25", "A45", "A65"])
        self.assertEqual(len(L.partner), 2 * sum(X_lengths))      # every base pair, both ways

    def test_listed_corners_and_partner_assignment(self):
        atoms = curve_it.parse_pdb_atoms(duplex_pdb([5, 20, 20, 20, 15]))
        L = cc.layout(atoms, "A5,A25,A45,A65")
        self.assertFalse(L.detected)
        # chain B follows its partners: B1 pairs with A80, so it sits in the last segment
        self.assertEqual(L.seg_of[("B", 1, "")], 4)
        self.assertEqual(L.seg_of[("B", 80, "")], 0)

    def test_bad_specs(self):
        atoms = curve_it.parse_pdb_atoms(duplex_pdb([5, 20, 15]))
        with self.assertRaises(ValueError):
            cc.layout(atoms, "A25,A5")              # out of order
        with self.assertRaises(ValueError):
            cc.layout(atoms, "A5,B10")              # two chains
        with self.assertRaises(ValueError):
            cc.parse_phase_spec("in,out,in", 2)     # wrong count
        self.assertEqual(cc.parse_phase_spec("out", 3), [-1, -1, -1])


class TestWholeTurnSides(CornerCase):
    """Sides of two whole turns: every corner on the same side, so in and out both need no extra twist."""

    @classmethod
    def setUpClass(cls):
        cls.in_ = cls().run_case([10, 20, 20, 20, 10], "in")
        cls.out = cls().run_case([10, 20, 20, 20, 10], "out")

    def test_in_closes_flat(self):
        self.assertFlatAndPhased(self.in_)
        self.assertLess(self.max_twist_per_bp(self.in_), 0.5, self.in_["log"])
        self.assertEqual(self.in_["checks"]["contacts"], [])

    def test_out_closes_flat(self):
        self.assertFlatAndPhased(self.out)
        self.assertLess(self.max_twist_per_bp(self.out), 0.5, self.out["log"])

    def test_out_ring_is_tighter(self):
        r_in = [1 / p[0] for p in self.in_["params"]]
        r_out = [1 / p[0] for p in self.out["params"]]
        self.assertGreater(min(r_in), max(r_out))
        # in: the arcs turn 360 - 4 x 30 = 240 deg; out: 360 + 120 = 480 deg
        arc_in = sum(p[0] * s.arc_length for p, s in zip(self.in_["params"], self.in_["plan"].sides))
        arc_out = sum(p[0] * s.arc_length for p, s in zip(self.out["params"], self.out["plan"].sides))
        self.assertAlmostEqual(math.degrees(arc_in), 240.0, delta=3.0)
        self.assertAlmostEqual(math.degrees(arc_out), 480.0, delta=3.0)

    def test_corners_kept_exactly(self):
        # each corner's two end base pairs (rigid groups) keep their relative pose atom for atom
        res, L = self.in_, self.in_["layout"]
        for J in res["plan"].junctions:
            A, B = res["plan"].sides[J.a], res["plan"].sides[J.b]
            idx = np.r_[A.atom_idx[A.last], B.atom_idx[B.first]]
            _, _, rmsd = cc.kabsch(L.X[idx], res["X_out"][idx])
            self.assertLess(rmsd, 1e-3)

    def test_output_files(self):
        pdb, out, curve = self.in_["paths"]
        text = open(out).read()
        self.assertIn("REMARK 900", text)
        self.assertIn("corner mode", text)
        atoms_in = curve_it.parse_pdb_atoms(open(pdb).read())
        atoms_out = curve_it.parse_pdb_atoms(text)
        self.assertEqual([a.atom_name for a in atoms_in], [a.atom_name for a in atoms_out])
        X = np.array([a.coord for a in atoms_out])
        self.assertLess(np.abs(X[:, 2]).max(), 12.0)            # flat: the ring lies in the xy plane
        self.assertLess(np.linalg.norm(X.mean(0)), 1.0)         # centred
        comps = curve_it.read_xyz_curve_components_from_text(open(curve).read())
        self.assertEqual(len(comps), 4)

    def test_closing_steps(self):
        ends = {ch: d for ch, _, d in self.in_["checks"]["ends"]}
        # the chain ends meet as an ordinary step of the template (O3'-P 2.98 A in this coarse template)
        straight = curve_it.parse_pdb_atoms(duplex_pdb([40]))
        ix = {(a.chain_id, a.res_seq, a.atom_name_norm): a.coord for a in straight}
        ideal = float(np.linalg.norm(ix[("A", 1, "O3'")] - ix[("A", 2, "P")]))
        self.assertAlmostEqual(ends["A"], ideal, delta=0.3)


class TestAlternatingSides(CornerCase):
    """Sides of one and a half turns: the corners alternate, so in,out,in,out needs no extra twist and all-in
    needs about half a turn per side, which the report flags with a fix."""

    @classmethod
    def setUpClass(cls):
        cls.mixed = cls().run_case([8, 15, 15, 15, 7], "in,out,in,out")
        cls.check_in = cls().run_case([8, 15, 15, 15, 7], "in", check=True)

    def test_mixed_closes_flat(self):
        self.assertFlatAndPhased(self.mixed)
        self.assertLess(self.max_twist_per_bp(self.mixed), 0.5, self.mixed["log"])
        signs = [J.sign for J in self.mixed["plan"].junctions]
        self.assertEqual(signs, [1, -1, 1, -1])
        # in and out cancel, so the arcs turn a full circle
        arc = sum(p[0] * s.arc_length for p, s in zip(self.mixed["params"], self.mixed["plan"].sides))
        self.assertAlmostEqual(math.degrees(arc), 360.0, delta=3.0)

    def test_wrong_pattern_is_reported(self):
        misses = self.check_in["misses"]
        self.assertEqual(len(misses), 3)
        for m in misses:
            self.assertGreater(abs(m["twist"]), 150.0)
        sug = cc.suggestions(self.check_in["plan"], misses)
        for s in sug:
            self.assertLess(abs(s["flip"]), 30.0)            # flipping the phase fixes it
            self.assertIn(abs(s["dn"]), (4, 5, 6))          # or half a turn more or less
        self.assertIn("Suggestion", self.check_in["log"])


class TestSkewedCorners(CornerCase):
    """Helix axes that pass beside each other at the corners: each corner then steps the next side along the
    plane normal.  The steps cancel or add up around a ring, depending on their signs and the phases."""

    def test_steps_that_add_up_tilt_the_ring(self):
        res = self.run_case([10, 20, 20, 20, 10], "in", skews=[2.0, 2.0, 2.0, 2.0])
        plan = res["plan"]
        for J in plan.junctions:
            self.assertAlmostEqual(abs(J.offset), 2.0, delta=0.05)
        self.assertAlmostEqual(abs(plan.rise()), 8.0, delta=0.2)
        self.assertIn("cannot lie flat", res["log"])
        self.assertLess(res["diag"]["closure"], 0.1, res["log"])            # it still closes
        self.assertGreater(max(res["diag"]["tilts"]), 0.5, res["log"])      # but not flat

    def test_steps_that_cancel_keep_it_flat(self):
        res = self.run_case([10, 20, 20, 20, 10], "in", skews=[2.0, -2.0, 2.0, -2.0])
        self.assertLess(abs(res["plan"].rise()), 0.2)
        self.assertNotIn("cannot lie flat", res["log"])
        self.assertFlatAndPhased(res)


class TestOpenPath(CornerCase):
    def test_straight_sides_phased_into_a_plane(self):
        # sides of one and a half turns alternate their corners: a flat zigzag without extra twist
        res = self.run_case([12, 15, 15, 12], "in,out,in", path="open", radius="straight")
        self.assertFlatAndPhased(res)
        self.assertLess(self.max_twist_per_bp(res), 0.5, res["log"])
        X = res["X_out"]
        self.assertLess(np.abs(X[:, 2]).max(), 12.0)
        for p in res["params"]:
            self.assertEqual(p[0], 0.0)

    def test_open_whole_turns_turn_one_way(self):
        # sides of two whole turns: every corner turns the same way, a flat open polygon
        res = self.run_case([12, 20, 20, 12], "in", path="open")
        self.assertFlatAndPhased(res)
        self.assertLess(self.max_twist_per_bp(res), 0.5, res["log"])

    def test_open_path_on_arcs(self):
        res = self.run_case([12, 20, 12], "out", path="open", radius="80")
        self.assertFlatAndPhased(res)
        for p in res["params"]:
            self.assertAlmostEqual(1 / p[0], 80.0)


class TestZDNASides(CornerCase):
    """A B-Z-B-Z ring whose Z-DNA sides repeat every two base pairs: no one screw carries each of their base pairs
    onto the next (they used to fall back to their principal axes), so each Z side is mapped about the screw that
    carries each pair onto the pair two steps on, with half its twist (-30 deg) as the twist per step."""

    SEGMENTS = [("B", 10), ("Z", 24), ("B", 20), ("Z", 24), ("B", 10)]
    CORNERS = "A10,A34,A54,A78"

    @classmethod
    def setUpClass(cls):
        cls.frames = []
        cls.text = bz_duplex_pdb(cls.SEGMENTS, frames_out=cls.frames)
        cls.ring = cls().run_case(None, "in", corners=cls.CORNERS, text=cls.text)

    def test_z_axis_is_the_two_step_screw(self):
        L = cc.layout(curve_it.parse_pdb_atoms(self.text), self.CORNERS)
        self.assertEqual(len(L.partner), 2 * 88)

        def coord(i):
            return L.X[i]
        z = L.seg_keys[1]                                         # Z24, A11-A34
        self.assertEqual(cc.res_label(z[0]), "A11")
        self.assertIsNone(cc._screw_fit(L, z, coord, 1, 0.5))     # no one-step screw fits
        point, u, twist, rise, repeat = cc.helix_axis(L, z, coord)
        self.assertEqual(repeat, 2)
        self.assertAlmostEqual(twist, -30.0, delta=0.01)
        self.assertAlmostEqual(rise, 3.625, delta=0.001)
        R, o = self.frames[10]                                    # A11's frame: its z axis is the helix axis
        self.assertLess(cc._angle(u, R[:, 2]), 0.01)
        off = (point - o) - float((point - o) @ u) * u
        self.assertLess(np.linalg.norm(off), 0.01)
        # a B side keeps the screw from each base pair to the next
        b = cc.helix_axis(L, L.seg_keys[2], coord)
        self.assertEqual(b[4], 1)
        self.assertAlmostEqual(b[2], TWIST, delta=0.01)

    def test_sides_twist_and_suggestions(self):
        plan = self.ring["plan"]
        self.assertEqual([S.repeat for S in plan.sides], [1, 2, 1, 2])
        for S in plan.sides:
            self.assertEqual(S.axis_source, "helical")
            self.assertAlmostEqual(S.omega, -30.0 if S.repeat == 2 else TWIST, delta=0.05)
        self.assertEqual(self.ring["log"].count("repeating every 2 base pairs"), 2)
        # 30 deg of twist: a B side one base pair longer leaves -6 deg; a Z side changes by whole repeats, so not
        # by one base pair (which would change its end base pairs from G syn to C anti)
        sug = cc.suggestions(plan, [dict(corner=1, side=1, miss=-30.0, twist=30.0),
                                    dict(corner=2, side=2, miss=-30.0, twist=30.0)])
        self.assertEqual(sug[0]["dn"], -2)
        self.assertAlmostEqual(sug[0]["left"], -30.0, delta=0.5)
        self.assertEqual(sug[1]["dn"], 1)
        self.assertAlmostEqual(sug[1]["left"], -6.0, delta=0.5)

    def test_ring_closes_flat(self):
        # sides of two whole turns: every corner in, with no extra twist
        self.assertFlatAndPhased(self.ring)
        self.assertLess(self.max_twist_per_bp(self.ring), 0.05, self.ring["log"])
        arc = sum(p[0] * s.arc_length for p, s in zip(self.ring["params"], self.ring["plan"].sides))
        self.assertAlmostEqual(math.degrees(arc), 240.0, delta=3.0)
        L = self.ring["layout"]
        for J in self.ring["plan"].junctions:
            A, B = self.ring["plan"].sides[J.a], self.ring["plan"].sides[J.b]
            idx = np.r_[A.atom_idx[A.last], B.atom_idx[B.first]]
            _, _, rmsd = cc.kabsch(L.X[idx], self.ring["X_out"][idx])
            self.assertLess(rmsd, 1e-3)
        self.assertIn("(the screw that carries each base pair 2 steps on)", open(self.ring["paths"][1]).read()
                      .replace("\nREMARK 900 ", " "))

    def test_auto_corners(self):
        # Z-DNA's residue-to-residue screws alternate, so its stretches are found from the two-step screws; a
        # B-Z junction is a corner however little it bends (one side cannot hold both), and the step from B onto
        # the first C anti, which lines up with the B-DNA when the junction is straight, does not take that C
        for bend in (-30.0, -4.0, 0.0):
            L = cc.layout(curve_it.parse_pdb_atoms(bz_duplex_pdb(self.SEGMENTS, bend=bend)), "auto")
            self.assertEqual([cc.res_label(L.guide_keys[p]) for p in L.corner_pos], ["A10", "A34", "A54", "A78"],
                             f"bend {bend}")


class TestZDNASeam(CornerCase):
    """A ring whose chain ends lie inside a Z-DNA side: the two pieces of that side are joined by the step from
    each base pair to the one two on (no one step fits Z-DNA), and the corners are found automatically."""

    SEGMENTS = [("Z", 12), ("B", 20), ("Z", 24), ("B", 20), ("Z", 12)]

    @classmethod
    def setUpClass(cls):
        cls.text = bz_duplex_pdb(cls.SEGMENTS)
        cls.ring = cls().run_case(None, "in", corners="auto", text=cls.text)

    def test_corners_and_seam(self):
        L, plan = self.ring["layout"], self.ring["plan"]
        self.assertEqual([cc.res_label(L.guide_keys[p]) for p in L.corner_pos], ["A12", "A32", "A56", "A76"])
        self.assertLess(L.seam_rmsd, 0.01)
        self.assertEqual(plan.sides[0].span, "A77-A88 + A1-A12")
        self.assertEqual([S.repeat for S in plan.sides], [2, 1, 2, 1])
        for S in plan.sides:
            self.assertEqual(S.axis_source, "helical")

    def test_ring_closes_flat(self):
        self.assertFlatAndPhased(self.ring)
        self.assertLess(self.max_twist_per_bp(self.ring), 0.05, self.ring["log"])
        # each chain end meets the other as the same GpC step does inside the Z-DNA
        ix = {(a.chain_id, a.res_seq, a.atom_name_norm): a.coord for a in curve_it.parse_pdb_atoms(self.text)}
        ends = {ch: d for ch, _, d in self.ring["checks"]["ends"]}
        for ch in "AB":
            ideal = float(np.linalg.norm(ix[(ch, 2, "O3'")] - ix[(ch, 3, "P")]))
            self.assertAlmostEqual(ends[ch], ideal, delta=0.3)


class TestCommandLine(unittest.TestCase):
    def test_curve_it_corner_options(self):
        tmp = tempfile.mkdtemp()
        pdb = os.path.join(tmp, "kinked.pdb")
        with open(pdb, "w") as f:
            f.write(duplex_pdb([10, 20, 20, 20, 10]))
        out = os.path.join(tmp, "ring.pdb")
        buf = io.StringIO()
        with redirect_stdout(buf):
            curve_it.main([pdb, "--corners", "auto", "--corner-phase", "in", "-o", out])
        self.assertTrue(os.path.isfile(out), buf.getvalue())
        self.assertIn("Solved", buf.getvalue())
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                curve_it.main([pdb, os.path.join(tmp, "curve.xyz"), "--corners", "auto"])

    def test_module_check(self):
        tmp = tempfile.mkdtemp()
        pdb = os.path.join(tmp, "kinked.pdb")
        with open(pdb, "w") as f:
            f.write(duplex_pdb([8, 15, 15, 15, 7]))
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cc.main([pdb, "--phase", "in,out,in,out", "--check"])
        self.assertEqual(code, 0)
        self.assertIn("corner 4", buf.getvalue())
        self.assertFalse(os.path.exists(cc.default_output_path(pdb)))


if __name__ == "__main__":
    unittest.main()
