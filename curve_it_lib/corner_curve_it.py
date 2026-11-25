#!/usr/bin/env python3
"""Corner Curve It: curve a helix that already has corners onto arcs in one plane, each corner in or out of phase.

A helix built with corners, such as a duplex bent at a pivot by RE_Helix's Bend Helix or D- and L-RNA pieces
joined by bent mirror junctions, keeps every corner fixed in the helix frame: each corner bends about a direction
that turns with the helix, so the lengths between the corners decide where the bends point.  This tool splits
the helix at its corners into straight sides, curves every side on its own circular arc with Curve It's mapping
(rotation-minimizing frame, rigid phosphate/sugar/base groups), keeps every corner exactly as it is in the input,
and solves for each side's curvature, helix phase and extra twist so that

* every arc lies in one plane and every corner bends in that plane, towards the inside of the curve ("in", in
  phase with the curve) or away from it ("out", out of phase), chosen corner by corner;
* a closed path closes: the last corner meets the first side as it does in the input;
* the bending is spread as evenly as the closure allows (least bending energy), and the extra twist is as small
  as the side lengths allow.

The corners are given as the last residue before each corner on one chain, or found from the helix axis.
Before solving, the tool reports how far each corner misses its in/out direction when no side is twisted, the
twist that cancels it, and the side lengths that would make that twist small.  A closed path needs its chain
ends in the middle of a side, joined by an ordinary step: that side is curved as one arc across the seam.

Two things differ from running Curve It on each side by hand.  Each side is mapped about its helical axis (the
screw axis of its base-pair steps), not its principal axis: the principal axis of a finite piece of helix is
tilted and shifted off the helix axis, so two sides would meet in skew lines and no flat ring could close.
Z-DNA repeats every two base pairs (C anti, G syn), so no one screw carries each of its base pairs onto the
next; a Z-DNA side takes the screw that carries each pair onto the pair two steps on instead.  And each side's
end base pairs, with any linker residue on them (an X33 phosphate, say), move as rigid groups, so the bonds
across every corner stay as built; the curving shows up one step further in instead.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Also support `python curve_it_lib/corner_curve_it.py` from another folder.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import curve_it as CI  # noqa: E402

VERSION = "1.1.0"
ResKey = Tuple[str, int, str]

# Atom names shared by every nucleotide or amino acid, used to superpose residues of different types.
BACKBONE_NAMES = {"P", "OP1", "OP2", "O1P", "O2P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'",
                  "C1'", "N", "CA", "C", "O", "CB"}
PHOSPHATE_NAMES = {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P", "O5'"}
PAIR_CUTOFF = 3.35        # base N/O atoms closer than this count as one hydrogen bond (A)
STEP_RMSD_MAX = 0.5       # residue-to-residue superposition worse than this is not a helical step (A)
RUN_TOLERANCE = 5.0       # helical steps whose axes agree within this belong to one straight stretch (deg)
DEFAULT_MIN_CORNER = 10.0  # straight stretches whose axes differ by at least this meet at a corner (deg)
REPEAT_TWIST = 30.0       # a residue-to-residue screw whose twist is within this of a two-step stretch's twist per
                          # step belongs to an ordinary helix, so the stretch does not repeat every two residues (deg)
MAX_HELIX_TWIST = 60.0    # a two-step stretch that turns more than this per residue is no duplex (Z-DNA: -30) (deg)
TWIST_WARN = 0.5          # extra twist above this (deg per step) is reported as a warning
SMALL_BEND = 2.0          # a corner bending less than this has no direction to phase (deg)

# Weights of the least-squares terms.
W_CLOSE = 100.0           # per Angstrom of closure shift (and per ROT_LENGTH A x radian of closure rotation)
ROT_LENGTH = 10.0         # a closure rotation counts as the shift it gives 10 A from the base pair's centre
W_PLANE = 300.0           # per unit of deviation of an arc normal or a corner axis from the plane normal
                          # (at 30 the bending-energy term tilted the arcs of a D/L ring by up to 1.6 deg to
                          # shave its curvature; at 300 the plane holds within 0.02 deg)
W_ENERGY = 30.0           # bending energy: sqrt(L_k / L) kappa_k / kappa_0 per side
W_TWIST = 0.3             # per degree of extra twist per step


# ----------------------------------------------------------------------------- small helpers
def _heavy(atom) -> bool:
    return (atom.element or "").upper() not in ("H", "D")


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two vectors (deg), accurate for small angles."""
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), float(u @ v))))


def wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rotation R and translation t with p @ R.T + t ~ q for the rows of P and Q, and the RMSD left."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = qc - R @ pc
    rmsd = float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, axis=1))))
    return R, t, rmsd


def screw_of(R: np.ndarray, t: np.ndarray):
    """Screw axis direction (pointing along the translation), signed rotation about it (deg) and rise (A)."""
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(w) / 2.0
    if s < 1e-9:
        return None
    ang = math.degrees(math.atan2(s, (np.trace(R) - 1.0) / 2.0))
    u = w / (2.0 * s)
    rise = float(u @ t)
    if rise < 0:
        u, ang, rise = -u, -ang, -rise
    return u, ang, rise


def solve1(f, lo: float, hi: float, step: float = 5.0) -> float:
    """Minimum of a one-parameter function: a grid, then a bounded refinement around the best grid point."""
    from scipy.optimize import minimize_scalar
    grid = np.arange(lo, hi, step)
    x0 = float(grid[int(np.argmin([f(x) for x in grid]))])
    r = minimize_scalar(f, bounds=(x0 - step, x0 + step), method="bounded", options=dict(xatol=1e-7))
    return float(r.x)


def curve_map(local, s, sg, s_min, length, kappa, phase_deg, twist_deg, s_twist=None):
    """Curve It's open-path, scale-mode-none mapping onto a circular arc of curvature kappa that starts at the
    origin along +x and turns towards +y (Curve It's own frame for such an arc), written out for the arc.
    local, s and sg are the atoms' offsets from the axis, axial positions and group positions, as
    embed_helix_on_curve computes them; the result equals Curve It's output on a finely sampled arc.
    s_twist, if given, is the axial position at which each atom takes its share of the twist (Curve It uses s;
    an atom group that must stay rigid takes its group position instead)."""
    loc = np.array(local, dtype=float, copy=True)
    ph = np.radians(twist_deg) * ((s if s_twist is None else s_twist) - s_min) / length
    x, y = loc[:, 0].copy(), loc[:, 1].copy()
    loc[:, 0], loc[:, 1] = np.cos(ph) * x - np.sin(ph) * y, np.sin(ph) * x + np.cos(ph) * y
    p = np.radians(phase_deg % 360.0)
    x, y = loc[:, 0].copy(), loc[:, 1].copy()
    loc[:, 0], loc[:, 1] = np.cos(p) * x - np.sin(p) * y, np.sin(p) * x + np.cos(p) * y
    sg = sg - s_min
    th = kappa * sg
    # sin(k s)/k and (1 - cos(k s))/k written with sinc, smooth through k = 0 (a straight side)
    pos = np.column_stack([sg * np.sinc(th / np.pi), 0.5 * kappa * sg ** 2 * np.sinc(th / (2 * np.pi)) ** 2, 0 * th])
    T = np.column_stack([np.cos(th), np.sin(th), 0 * th])
    N = np.column_stack([-np.sin(th), np.cos(th), 0 * th])
    return pos + loc[:, :1] * N + loc[:, 1:2] * np.array([0.0, 0.0, 1.0]) + ((s - s_min) - sg)[:, None] * T


def axis_local_coords(X: np.ndarray, point: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Axial positions and offsets from a given axis line, in the same form as Curve It's
    compute_helix_local_coords (which uses the principal axis): the line through `point` along `u`, the origin of
    s at the atoms' centroid projected onto it, and the same choice of perpendicular basis."""
    u = u / np.linalg.norm(u)
    centre = point + float((X.mean(0) - point) @ u) * u
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(tmp @ u)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(tmp, u)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(u, x_axis)
    rel = X - centre
    s = rel @ u
    radial = rel - np.outer(s, u)
    return s, radial @ np.column_stack((x_axis, y_axis, u))


def arc_points(kappa: float, sigma: np.ndarray) -> np.ndarray:
    th = kappa * sigma
    return np.column_stack([sigma * np.sinc(th / np.pi), 0.5 * kappa * sigma ** 2 * np.sinc(th / (2 * np.pi)) ** 2,
                            0 * sigma])


# ----------------------------------------------------------------------------- residues, pairs, steps
def residue_table(atoms) -> Tuple[Dict[ResKey, List[int]], Dict[str, List[ResKey]]]:
    """Atom indices of every residue, and every chain's residues, in file order."""
    res_atoms: Dict[ResKey, List[int]] = {}
    chains: Dict[str, List[ResKey]] = {}
    for i, a in enumerate(atoms):
        key = (a.chain_id, a.res_seq, a.i_code)
        if key not in res_atoms:
            res_atoms[key] = []
            chains.setdefault(a.chain_id, []).append(key)
        res_atoms[key].append(i)
    return res_atoms, chains


def res_label(key: ResKey) -> str:
    chain, num, icode = key
    return f"{chain or '_'}{num}{icode}"


def residue_kind(atoms, res_atoms, key) -> str:
    return CI.classify_residue_type(atoms[res_atoms[key][0]].res_name)


def is_core(atoms, res_atoms, key) -> bool:
    """A standard nucleotide or amino acid (linkers such as an X33 phosphate are carried, not fitted)."""
    return residue_kind(atoms, res_atoms, key) in ("nucleic_acid", "protein")


def residue_match(atoms, res_atoms, k1, k2) -> List[Tuple[int, int]]:
    """Index pairs of the heavy atoms two residues share by name, for superposing one residue on the next
    (backbone and sugar only between residue types).  The phosphate and O5' are left out: they are what a
    junction rebuilds (a 5'-5' link moves both O5' and re-places one phosphate), and the steps measured here
    are the helix's own."""
    a1 = {atoms[i].atom_name_norm: i for i in res_atoms[k1] if _heavy(atoms[i])}
    a2 = {atoms[i].atom_name_norm: i for i in res_atoms[k2] if _heavy(atoms[i])}
    same = atoms[res_atoms[k1][0]].res_name == atoms[res_atoms[k2][0]].res_name
    return [(a1[n], a2[n]) for n in a1 if n in a2 and n not in PHOSPHATE_NAMES and (same or n in BACKBONE_NAMES)]


def find_base_pairs(atoms, res_atoms, X) -> Dict[ResKey, ResKey]:
    """Base-pair partners: the residue whose base shares the most hydrogen bonds (base N/O atoms within 3.35 A,
    at least two) with this one, kept only when the choice is mutual."""
    from scipy.spatial import cKDTree
    keys = list(res_atoms)
    kpos = {k: n for n, k in enumerate(keys)}
    idx, owner = [], []
    for k in keys:
        if residue_kind(atoms, res_atoms, k) != "nucleic_acid":
            continue
        for i in res_atoms[k]:
            a = atoms[i]
            if (a.element or "").upper() in ("N", "O") and CI.classify_atom_group(a) == "B":
                idx.append(i)
                owner.append(kpos[k])
    if len(idx) < 2:
        return {}
    P = X[idx]
    count: Dict[Tuple[int, int], int] = {}
    dmin: Dict[Tuple[int, int], float] = {}
    for p, q in cKDTree(P).query_pairs(PAIR_CUTOFF):
        r1, r2 = owner[p], owner[q]
        if r1 == r2:
            continue
        k1, k2 = keys[r1], keys[r2]
        if k1[0] == k2[0] and abs(k1[1] - k2[1]) <= 1:
            continue                                   # sequence neighbours stack; they do not pair
        pair = (min(r1, r2), max(r1, r2))
        count[pair] = count.get(pair, 0) + 1
        dmin[pair] = min(dmin.get(pair, 99.0), float(np.linalg.norm(P[p] - P[q])))
    best: Dict[int, Tuple[int, int, float]] = {}
    for (r1, r2), c in count.items():
        if c < 2:
            continue
        for a, b in ((r1, r2), (r2, r1)):
            cur = best.get(a)
            if cur is None or (c, -dmin[(r1, r2)]) > (cur[1], -cur[2]):
                best[a] = (b, c, dmin[(r1, r2)])
    return {keys[a]: keys[b] for a, (b, _, _) in best.items() if best.get(b, (None,))[0] == a}


def guide_steps(atoms, res_atoms, keys: Sequence[ResKey], X, n: int = 1) -> List[Optional[dict]]:
    """The screw that carries each residue of the guide chain onto the next one, or onto the one n residues on
    (None where that is not a helical step: too few shared atoms, a poor superposition such as a change of hand,
    or no rise).  Twist and rise are per residue."""
    steps: List[Optional[dict]] = []
    for k1, k2 in zip(keys, keys[n:]):
        pairs = residue_match(atoms, res_atoms, k1, k2)
        if len(pairs) < 4:
            steps.append(None)
            continue
        R, t, rmsd = kabsch(X[[p for p, _ in pairs]], X[[q for _, q in pairs]])
        sc = screw_of(R, t)
        if rmsd > STEP_RMSD_MAX or sc is None or sc[2] < 0.5 * n:
            steps.append(None)
            continue
        steps.append({"u": sc[0], "twist": sc[1] / n, "rise": sc[2] / n, "rmsd": rmsd})
    return steps


def straight_runs(steps, tol: float = RUN_TOLERANCE, min_steps: int = 2) -> List[dict]:
    """Stretches of consecutive helical steps whose screw axes agree (one straight piece of helix each)."""
    runs: List[dict] = []
    cur = None
    for j, st in enumerate(steps):
        if st is None:
            if cur is not None:
                runs.append(cur)
                cur = None
            continue
        if cur is not None and _angle(cur["u"], st["u"]) <= tol:
            cur["last"] = j
            cur["u"] = cur["u"] + st["u"]
            cur["twists"].append(st["twist"])
            continue
        if cur is not None:
            runs.append(cur)
        cur = {"first": j, "last": j, "u": st["u"].copy(), "twists": [st["twist"]]}
    if cur is not None:
        runs.append(cur)
    return [r for r in runs if r["last"] - r["first"] + 1 >= min_steps]


def two_step_runs(atoms, res_atoms, keys: Sequence[ResKey], X, steps, runs: Sequence[dict],
                  partner: Dict[ResKey, ResKey], tol: float = RUN_TOLERANCE, min_steps: int = 2) -> List[dict]:
    """Straight stretches of a duplex that repeats every two residues, such as Z-DNA (C anti, G syn), in the gaps
    between the runs of residue-to-residue steps: along Z-DNA those screws alternate (about -135 and +131 deg)
    and make no run, but the screws that carry each residue onto the one two on agree.  A stretch counts only if
    at least half its residues are base-paired, it turns by at most MAX_HELIX_TWIST per residue, and its
    residue-to-residue screws (steps) do not describe the same helix: where half of them or more have its twist
    per step, within REPEAT_TWIST, it is an ordinary helix that merely curves or wobbles there, and its
    residue-to-residue steps decide.  Each run gives the position of its first two-step screw ("first") and of
    its last ("last"), and n = 2."""
    out: List[dict] = []
    starts = [r["first"] for r in runs] + [len(keys)]
    for lo, hi in zip([0] + [r["last"] + 2 for r in runs], starts):   # residues lo .. hi - 1 lie between runs
        if hi - lo < min_steps + 2:
            continue
        for r in straight_runs(guide_steps(atoms, res_atoms, keys[lo:hi], X, 2), tol, min_steps):
            first, last = lo + r["first"], lo + r["last"]
            twist = float(np.mean(r["twists"]))
            paired = sum(k in partner for k in keys[first:last + 3])
            inside = steps[first:last + 2]                  # residue-to-residue screws over residues first..last+2
            same = sum(st is not None and abs(wrap180(st["twist"] - twist)) <= REPEAT_TWIST for st in inside)
            if 2 * paired >= last - first + 3 and abs(twist) <= MAX_HELIX_TWIST and 2 * same < len(inside):
                out.append(dict(r, first=first, last=last, n=2))
    return out


def _settle_repeat_boundaries(atoms, res_atoms, keys: Sequence[ResKey], X, steps, runs: Sequence[dict],
                              tol: float = RUN_TOLERANCE, min_steps: int = 2) -> None:
    """Where a run of residue-to-residue steps meets a two-step run (B- and Z-DNA), the step across the junction
    can point along the one-step run, with any twist, and take the first or last residue of the Z-DNA into it.
    Give such a residue back to the two-step run while the screw that carries it two residues on matches that
    run, in axis and in twist.  runs are in chain order and are changed in place."""
    for ra, rb in zip(runs, runs[1:]):
        if ra["n"] == rb["n"]:
            continue
        one, two = (ra, rb) if ra["n"] == 1 else (rb, ra)
        while one["last"] - one["first"] + 1 > min_steps:
            if two is rb:            # one-step run first: its last residue would be the two-step run's first
                k = two["first"] - 1
                if one["last"] + 1 != k:
                    break
            else:                    # two-step run first: the one-step run's first residue would be its last
                k = two["last"] + 1
                if one["first"] != k + 2:
                    break
            st = guide_steps(atoms, res_atoms, keys[k:k + 3], X, 2)[0] if 0 <= k and k + 2 < len(keys) else None
            if (st is None or _angle(two["u"], st["u"]) > tol
                    or abs(st["twist"] - float(np.mean(two["twists"]))) > tol):
                break
            if two is rb:
                one["u"] = one["u"] - steps[one["last"]]["u"]
                one["twists"].pop()
                one["last"] -= 1
                two["first"] = k
                two["twists"].insert(0, st["twist"])
            else:
                one["u"] = one["u"] - steps[one["first"]]["u"]
                one["twists"].pop(0)
                one["first"] += 1
                two["last"] = k
                two["twists"].append(st["twist"])
            two["u"] = two["u"] + st["u"]


def longest_chain(atoms, res_atoms, chains) -> str:
    return max(chains, key=lambda c: (sum(is_core(atoms, res_atoms, k) for k in chains[c]), -list(chains).index(c)))


def detect_corners(atoms, res_atoms, chains, X, chain: Optional[str] = None,
                   min_angle: float = DEFAULT_MIN_CORNER, partner: Optional[Dict[ResKey, ResKey]] = None) -> dict:
    """Corners of the guide chain: where two straight stretches of helix meet at an angle of min_angle or more,
    and, whatever the angle, where a helix that repeats every residue meets one that repeats every two (B- and
    Z-DNA), since one side cannot hold both.  partner: the base pairs, found here if not given.  Returns the
    chain, its residues, the positions of the last residue before each corner and the angles."""
    if chain is None:
        chain = longest_chain(atoms, res_atoms, chains)
    if chain not in chains:
        raise ValueError(f"Chain {chain or '_'} is not in the PDB (chains: {', '.join(c or '_' for c in chains)}).")
    keys = chains[chain]
    steps = guide_steps(atoms, res_atoms, keys, X)
    runs = [dict(r, n=1) for r in straight_runs(steps)]
    if partner is None:
        partner = find_base_pairs(atoms, res_atoms, X)
    runs = sorted(runs + two_step_runs(atoms, res_atoms, keys, X, steps, runs, partner), key=lambda r: r["first"])
    _settle_repeat_boundaries(atoms, res_atoms, keys, X, steps, runs)
    corners, angles = [], []
    for ra, rb in zip(runs, runs[1:]):
        ang = _angle(ra["u"], rb["u"])
        if ang >= min_angle or ra["n"] != rb["n"]:
            corners.append(ra["last"] + ra["n"])            # the run's last residue
            angles.append(ang)
    return {"chain": chain, "keys": keys, "corners": corners, "angles": angles, "steps": steps, "runs": runs}


_RES_RE = (re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)$"), re.compile(r"^([^:]?):(-?\d+)([A-Za-z]?)$"))


def parse_corner_spec(spec: Optional[str]):
    """'auto', 'auto:B', or residues such as 'A28,A79' / 'A:28, A:79' -> ('auto', chain) or ('list', keys)."""
    text = (spec or "auto").strip()
    if text.lower().startswith("auto"):
        rest = text[4:].lstrip(":").strip()
        return "auto", (None if not rest else ("" if rest == "_" else rest))
    keys: List[ResKey] = []
    for word in re.split(r"[,;\s]+", text):
        if not word:
            continue
        m = _RES_RE[0].match(word) or _RES_RE[1].match(word)
        if not m:
            raise ValueError(f"Cannot read the corner residue '{word}': write the chain and residue number, "
                             f"as A28 or A:28 (a blank chain is _:28).")
        chain = "" if m.group(1) in ("", "_") else m.group(1)
        keys.append((chain, int(m.group(2)), m.group(3).upper()))
    if not keys:
        raise ValueError("No corner residues given.")
    return "list", keys


def parse_phase_spec(spec: Optional[str], n: int) -> List[int]:
    """'in', 'out' or one word per corner ('in,out,in,out') -> +1 / -1 per corner."""
    words = [w for w in re.split(r"[,;\s]+", (spec or "in").strip().lower()) if w]
    vals = []
    for w in words:
        if w in ("in", "i", "+", "+1", "1"):
            vals.append(+1)
        elif w in ("out", "o", "-", "-1"):
            vals.append(-1)
        else:
            raise ValueError(f"Corner phase '{w}' is not in or out.")
    if not vals:
        vals = [+1]
    if len(vals) == 1:
        vals = vals * n
    if len(vals) != n:
        raise ValueError(f"The helix has {n} corner{'s' if n != 1 else ''}, so give {n} phases "
                         f"(or one for all of them); got {len(vals)}.")
    return vals


def parse_radius(text) -> float:
    """Arc radius of an open path -> curvature (0 for a straight side)."""
    if text is None:
        return 0.0
    word = str(text).strip().lower()
    if word in ("", "straight", "none", "inf", "infinity", "0", "0.0"):
        return 0.0
    try:
        r = float(word)
    except ValueError:
        raise ValueError(f"Arc radius '{text}' is not a number of Angstrom or 'straight'.") from None
    if r == 0:
        return 0.0
    return 1.0 / r


# ----------------------------------------------------------------------------- sides and corners
class Side:
    """One straight side of the helix, prepared as Curve It prepares its input (offsets from the axis, atom
    groups), but about the side's helical axis, with its first and last base pairs (the atoms the corners hold
    on to).  Each end base pair, with any linker residue on it (an X33 phosphate, say), is mapped as one rigid
    group, so the corners and the bonds across them stay exactly as built; Curve It's usual phosphate, sugar
    and base groups take every other residue.  repeat is the number of base pairs after which the side's
    helix repeats: 1, or 2 for Z-DNA, whose length can only change by whole repeats."""

    def __init__(self, number: int, span: str, atoms_all, atom_idx, X, first_idx, last_idx, n_res: int,
                 omega: Optional[float], segments: Sequence[int], axis=None, rigid: Sequence[Sequence[int]] = (),
                 repeat: int = 1):
        self.number, self.span, self.n_res, self.omega = number, span, n_res, omega
        self.repeat = int(repeat)
        self.segments = list(segments)
        self.atom_idx = np.asarray(atom_idx, dtype=int)
        self.X = np.asarray(X, dtype=float)
        if len(self.atom_idx) < 4:
            raise ValueError(f"Side {number} ({span}) has too few atoms to define a helix axis.")
        if axis is not None:
            # the helical (screw) axis of the side's base-pair steps (of every second step on Z-DNA): a principal
            # axis of a finite piece of helix is tilted and shifted off it (2.4 A between two 20-bp sides of a
            # 10-bp-per-turn helix), so sides joined at a corner would meet in skew lines and no flat ring could
            # close
            self.s, self.local = axis_local_coords(self.X, *axis)
            self.axis_source = "helical"
        else:
            _, _, _, self.s, self.local = CI.compute_helix_local_coords(self.X)
            self.axis_source = "principal"
        self.s_min = float(self.s.min())
        self.length = float(self.s.max() - self.s.min())
        groups, gs = CI.build_atom_groups([atoms_all[i] for i in self.atom_idx], self.s)
        self.gs = np.empty(len(self.atom_idx))
        for g, members in enumerate(groups):
            self.gs[members] = gs[g]
        where = {int(g): n for n, g in enumerate(self.atom_idx)}
        self.s_tw = self.s.copy()                   # where each atom takes its share of the twist
        for members in rigid:
            m = np.array([where[int(i)] for i in members if int(i) in where], dtype=int)
            if len(m):
                self.gs[m] = self.s_tw[m] = float(self.s[m].mean())
        self.first = np.array([where[int(i)] for i in first_idx], dtype=int)
        self.last = np.array([where[int(i)] for i in last_idx], dtype=int)
        self.sig_first = float(self.gs[self.first].mean() - self.s_min)
        self.sig_last = float(self.gs[self.last].mean() - self.s_min)
        self.dir = 1.0 if self.sig_last >= self.sig_first else -1.0
        self.sub = np.unique(np.r_[self.first, self.last])
        pos = {int(v): n for n, v in enumerate(self.sub)}
        self.sub_first = np.array([pos[int(v)] for v in self.first], dtype=int)
        self.sub_last = np.array([pos[int(v)] for v in self.last], dtype=int)

    @property
    def arc_length(self) -> float:
        return abs(self.sig_last - self.sig_first)

    def phase_twist(self, phi: float, tau: float) -> Tuple[float, float]:
        """Curve It's --helix_phase and --twist for a phase phi at the side's first base pair and an extra twist
        tau between its first and last base pairs.  Curve It spreads --twist evenly over the whole axial extent
        of the atoms, from 0 at the lowest atom to the full angle at the highest, so the end base pairs, a few
        Angstrom inside that extent, see a little less of it; the first base pair keeps phi whichever way the side
        runs.  (A twist turns either end the same way about the direction of travel, so tau needs no sign.)"""
        twist = tau * self.length / max(self.arc_length, 1e-9)
        return ((phi - twist * self.sig_first / self.length) % 360.0, twist)

    @property
    def steps(self) -> int:
        """Base-pair steps between the side's first and last base pairs."""
        return max(1, self.n_res - 1)

    def curve(self, kappa: float, phi: float, tau: float, full: bool = False) -> np.ndarray:
        idx = slice(None) if full else self.sub
        hp, tw = self.phase_twist(phi, tau)
        return curve_map(self.local[idx], self.s[idx], self.gs[idx], self.s_min, self.length, kappa, hp, tw,
                         self.s_tw[idx])

    def tangent(self, kappa: float, sigma: float) -> np.ndarray:
        return np.array([math.cos(kappa * sigma), math.sin(kappa * sigma), 0.0])


@dataclass
class Junction:
    """A corner: side a's last base pair and side b's first base pair, held as they are in the input."""
    a: int
    b: int
    exit_ref: np.ndarray
    entry_ref: np.ndarray
    sign: int
    label: str
    bend: float = 0.0
    offset: float = 0.0


class Plan:
    """The sides joined by their corners.  Side 0 stays in Curve It's frame for an arc (it starts at the origin
    along +x and turns about +z), so its arc fixes the plane; every later side is curved and then placed so
    that its first base pair sits on the previous side's last base pair as in the input."""

    def __init__(self, sides: List[Side], junctions: List[Junction], closed: bool, kappa_open: float = 0.0):
        self.sides, self.junctions, self.closed = sides, junctions, closed
        self.K = len(sides)
        self.kappa_open = float(kappa_open)
        self.z = sides[0].dir * np.array([0.0, 0.0, 1.0])     # the plane normal: the curve turns about +z
        total = sum(s.arc_length for s in sides)
        self.kappa0 = 2 * math.pi / total if total > 0 else 0.0
        self.lw = np.sqrt(np.array([s.arc_length for s in sides]) / max(total, 1e-9))
        self.nres = np.array([s.steps for s in sides], dtype=float)

    # -------------------------------------------------------------- geometry
    def place(self, params, full: bool = False, upto: Optional[int] = None):
        last = self.K - 1 if upto is None else upto
        Xs = [self.sides[0].curve(*params[0], full=full)]
        Rs, Ts, fits = [np.eye(3)], [np.zeros(3)], []
        for k in range(1, last + 1):
            J = self.junctions[k - 1]
            A, B = self.sides[J.a], self.sides[J.b]
            Rm, tm, _ = kabsch(J.exit_ref, Xs[J.a][A.last if full else A.sub_last])
            target = J.entry_ref @ Rm.T + tm
            C = B.curve(*params[k], full=full)
            R, t, rms = kabsch(C[B.first if full else B.sub_first], target)
            Xs.append(C @ R.T + t)
            Rs.append(R)
            Ts.append(t)
            fits.append(rms)
        return Xs, Rs, Ts, fits

    def closure(self, Xs, full: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """Where the closing corner puts side 0's first base pair against where side 0 has it, as a pose:
        the shift of the base pair's centre (A), the rotation vector (rad) and the RMSD left after both.  Curving
        distorts a base pair slightly, so an atom-by-atom error could never reach zero and would reward flatter
        sides; the pose can."""
        J = self.junctions[-1]
        A, B = self.sides[J.a], self.sides[0]
        Rm, tm, _ = kabsch(J.exit_ref, Xs[J.a][A.last if full else A.sub_last])
        target = J.entry_ref @ Rm.T + tm
        actual = Xs[0][B.first if full else B.sub_first]
        R, _, rmsd = kabsch(actual, target)
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
        sin_a = float(np.linalg.norm(w))
        ang = math.atan2(sin_a, (np.trace(R) - 1.0) / 2.0)
        rotvec = w * (ang / sin_a) if sin_a > 1e-12 else w
        return target.mean(0) - actual.mean(0), rotvec, rmsd

    def tangents(self, k: int, kappa: float, R: np.ndarray):
        """Direction of travel at side k's first and last base pairs."""
        S = self.sides[k]
        return S.dir * (R @ S.tangent(kappa, S.sig_first)), S.dir * (R @ S.tangent(kappa, S.sig_last))

    def normal(self, k: int, R: np.ndarray) -> np.ndarray:
        """The normal about which side k turns when its curvature is positive."""
        return self.sides[k].dir * R[:, 2]

    def closing(self, c: int) -> bool:
        """Whether corner c is the one that closes a ring (from the last side back onto side 0)."""
        return self.closed and c == len(self.junctions) - 1

    def corner(self, c: int, params, placed, full: bool = False):
        """Directions of travel just before and just after corner c.  Every side after side 0 is placed by the
        corner before it, so its own placement gives the direction after the corner.  Side 0 stays where it is,
        so for the corner that closes a ring the direction after it is taken from where that corner would put
        side 0: the corner's phase is then measured even while the ring is still open."""
        Xs, Rs = placed[0], placed[1]
        J = self.junctions[c]
        te = self.tangents(J.a, params[J.a][0], Rs[J.a])[1]
        if not self.closing(c):
            return te, self.tangents(J.b, params[J.b][0], Rs[J.b])[0]
        A, B = self.sides[J.a], self.sides[0]
        Rm, tm, _ = kabsch(J.exit_ref, Xs[J.a][A.last if full else A.sub_last])
        R0, _, _ = kabsch(Xs[0][B.first if full else B.sub_first], J.entry_ref @ Rm.T + tm)
        return te, self.tangents(0, params[0][0], R0)[0]

    def miss(self, c: int, params, placed, full: bool = False) -> float:
        """How far corner c bends from its wanted direction (deg, signed about the direction of travel)."""
        te, ts = self.corner(c, params, placed, full)
        n_in = np.cross(self.z, te)
        n_in /= max(np.linalg.norm(n_in), 1e-12)
        d = ts - (ts @ te) * te
        ang = math.degrees(math.atan2(float(te @ np.cross(n_in, d)), float(n_in @ d)))
        return wrap180(ang - (0.0 if self.junctions[c].sign > 0 else 180.0))

    def input_bends(self) -> List[float]:
        """The corners' bend angles (deg) between the sides' axes at their end base pairs, as built."""
        params = [[0.0, 0.0, 0.0] for _ in range(self.K)]
        placed = self.place(params)
        return [_angle(*self.corner(c, params, placed)) for c in range(len(self.junctions))]

    def axis_offsets(self) -> List[float]:
        """How far the two sides' axes pass each other at every corner (A), measured along the corner's bend
        axis: zero when they meet.  On a flat curve every bend axis is the plane normal (in) or its opposite
        (out), so each corner lifts the next side by sign x offset; around a ring these must cancel, or the
        ring cannot lie flat."""
        out = []
        ex = np.array([1.0, 0.0, 0.0])
        for J in self.junctions:
            A, B = self.sides[J.a], self.sides[J.b]
            XA = A.curve(0.0, 0.0, 0.0)                    # straight, in its own frame: the axis is the x axis
            Rm, tm, _ = kabsch(J.exit_ref, XA[A.sub_last])
            R, t, _ = kabsch(B.curve(0.0, 0.0, 0.0)[B.sub_first], J.entry_ref @ Rm.T + tm)
            da, db = A.dir * ex, B.dir * (R @ ex)
            n = np.cross(da, db)
            nn = float(np.linalg.norm(n))
            out.append(float(t @ n) / nn if nn > 1e-9 else 0.0)
        return out

    def rise(self) -> float:
        """How far the corners carry the path along the plane normal (A): sum of phase x axis offset."""
        return float(sum(J.sign * J.offset for J in self.junctions))

    # -------------------------------------------------------------- solving
    def residuals(self, params, E: float) -> np.ndarray:
        placed = self.place(params)
        Xs, Rs = placed[0], placed[1]
        out = []
        if self.closed:
            shift, rotvec, _ = self.closure(Xs)
            out.append(W_CLOSE * np.r_[shift, ROT_LENGTH * rotvec])
        for k in range(1, self.K):
            out.append(W_PLANE * (self.normal(k, Rs[k]) - self.z))
        for c, J in enumerate(self.junctions):
            if J.bend < SMALL_BEND:
                continue
            te, ts = self.corner(c, params, placed)
            b = np.cross(te, ts)
            out.append(W_PLANE * (J.sign * b / max(np.linalg.norm(b), 1e-12) - self.z))
        if self.closed:
            out.append(E * self.lw * np.array([p[0] for p in params]) / self.kappa0)
        out.append(W_TWIST * np.array([p[2] for p in params]) / self.nres)
        return np.concatenate(out)

    def start_kappa(self) -> float:
        if not self.closed:
            return self.kappa_open
        turn = 2 * math.pi - sum(math.radians(J.sign * J.bend) for J in self.junctions)
        return turn / sum(s.arc_length for s in self.sides)

    def sequential(self, kappa: float):
        """Untwisted start, corner by corner.  Side 0's phase points corner 1 exactly; every later side's phase
        lays its arc in the plane, the angle by which the corner after it then misses is recorded, and its far
        end is turned by the twist that cancels the miss.  So each miss belongs to one side."""
        K = self.K
        P = [[kappa, 0.0, 0.0] for _ in range(K)]
        misses = []

        def upto_for(c):
            """Sides that must be placed to see corner c."""
            return c if self.closing(c) else min(c + 1, K - 1)

        if self.junctions and self.junctions[0].bend >= SMALL_BEND:
            def f0(ph):
                Pt = [p[:] for p in P]
                Pt[0][1] = ph
                return self.miss(0, Pt, self.place(Pt, upto=upto_for(0))) ** 2
            P[0][1] = solve1(f0, 0.0, 360.0)
        for k in range(1, K):
            def arc_err(ph, k=k):
                Pt = [p[:] for p in P]
                Pt[k][1], Pt[k][2] = ph, 0.0
                return float(np.sum((self.normal(k, self.place(Pt, upto=k)[1][k]) - self.z) ** 2))
            P[k][1], P[k][2] = solve1(arc_err, 0.0, 360.0), 0.0
            if k < len(self.junctions) and self.junctions[k].bend >= SMALL_BEND:
                upto = upto_for(k)
                e = self.miss(k, P, self.place(P, upto=upto))

                def twist_err(tw, k=k, upto=upto):
                    Pt = [p[:] for p in P]
                    Pt[k][2] = tw
                    return self.miss(k, Pt, self.place(Pt, upto=upto)) ** 2
                P[k][2] = solve1(twist_err, -180.0, 180.0)
                misses.append({"corner": k, "side": k, "miss": e, "twist": P[k][2]})
        return P, misses

    def solve(self, P0):
        from scipy.optimize import least_squares
        K = self.K
        opts = dict(diff_step=1e-7, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
        if self.closed:
            def unpack(v):
                return [[v[k] * self.kappa0, v[K + k], v[2 * K + k]] for k in range(K)]
            v = np.r_[[p[0] / self.kappa0 for p in P0], [p[1] for p in P0], [p[2] for p in P0]]
            # one curvature for every side first (a rounded polygon), then each side its own
            shared = least_squares(lambda x: self.residuals(unpack(np.r_[np.full(K, x[0]), x[1:]]), 1.0),
                                   np.r_[np.mean(v[:K]), v[K:]], x_scale=np.r_[0.05, np.ones(2 * K)], **opts).x
            v1 = np.r_[np.full(K, np.clip(shared[0], -4.9, 4.9)), shared[1:]]
            lo = np.r_[np.full(K, -5.0), np.full(2 * K, -np.inf)]
            hi = np.r_[np.full(K, 5.0), np.full(2 * K, np.inf)]
            best = least_squares(lambda x: self.residuals(unpack(x), W_ENERGY), v1, bounds=(lo, hi),
                                 x_scale=np.r_[np.full(K, 0.05), np.ones(2 * K)], **opts).x
            return unpack(best)

        def unpack_open(v):
            return [[self.kappa_open, v[k], v[K + k]] for k in range(K)]
        v = np.r_[[p[1] for p in P0], [p[2] for p in P0]]
        best = least_squares(lambda x: self.residuals(unpack_open(x), 0.0), v, x_scale=np.ones(2 * K), **opts).x
        return unpack_open(best)

    def diagnose(self, params) -> dict:
        placed = self.place(params, full=True)
        Xs, Rs, Ts, fits = placed
        out = {"Xs": Xs, "Rs": Rs, "Ts": Ts, "fits": fits}
        if self.closed:
            shift, rotvec, rmsd = self.closure(Xs, full=True)
            out["closure"] = float(np.linalg.norm(shift))
            out["closure_deg"] = math.degrees(float(np.linalg.norm(rotvec)))
            out["closure_rmsd"] = rmsd
        else:
            out["closure"] = out["closure_deg"] = out["closure_rmsd"] = None
        out["tilts"] = [_angle(self.normal(k, Rs[k]), self.z) for k in range(1, self.K)]
        corners = []
        for c, J in enumerate(self.junctions):
            te, ts = self.corner(c, params, placed, full=True)
            b = np.cross(te, ts)
            corners.append({"bend": _angle(te, ts), "axis": _angle(J.sign * b, self.z) if J.bend >= SMALL_BEND
                            else float("nan")})
        out["corners"] = corners
        return out


# ----------------------------------------------------------------------------- building the plan
@dataclass
class Layout:
    """Everything worked out from the input before solving."""
    atoms: list
    X: np.ndarray
    res_atoms: Dict[ResKey, List[int]]
    chains: Dict[str, List[ResKey]]
    partner: Dict[ResKey, ResKey]
    guide: str
    guide_keys: List[ResKey]
    corner_pos: List[int]
    detected: bool
    steps: List[Optional[dict]]
    seg_of: Dict[ResKey, int]
    seg_keys: List[List[ResKey]]            # core guide residues of every segment, in order
    seam_rmsd: Optional[float] = None


def _bp_atoms(atoms, res_atoms, partner, keys: Sequence[ResKey], from_end: bool) -> List[int]:
    """Heavy atoms of the first (or last) base pair of a run of guide residues: the residue and its partner,
    extended by the next residue (and partner) while fewer than 8 atoms are held."""
    order = list(reversed(keys)) if from_end else list(keys)
    out: List[int] = []
    for k in order:
        for r in (k, partner.get(k)):
            if r is not None:
                out.extend(i for i in res_atoms[r] if _heavy(atoms[i]))
        if len(out) >= 8:
            break
    return out


def _bp_pairs(atoms, res_atoms, partner, k1: ResKey, k2: ResKey) -> List[Tuple[int, int]]:
    pairs = residue_match(atoms, res_atoms, k1, k2)
    p1, p2 = partner.get(k1), partner.get(k2)
    if p1 is not None and p2 is not None:
        pairs += residue_match(atoms, res_atoms, p1, p2)
    return pairs


def _screw_fit(L: Layout, keys: Sequence[ResKey], coord_of, n: int, max_rmsd: float):
    """The screw that carries every base pair (guide residue and partner) of a run onto the pair n steps on,
    fitted over the whole run: (point on the axis, unit direction along the run, twist per step in deg, rise per
    step in A), or None when the run is too short or the screw does not fit."""
    P, Q = [], []
    for k1, k2 in zip(keys, keys[n:]):
        for i, j in _bp_pairs(L.atoms, L.res_atoms, L.partner, k1, k2):
            P.append(coord_of(i))
            Q.append(coord_of(j))
    if len(P) < 6:
        return None
    R, t, rmsd = kabsch(np.array(P), np.array(Q))
    sc = screw_of(R, t)
    if sc is None or rmsd > max_rmsd or sc[2] < 0.5 * n:
        return None
    u, ang, rise = sc
    p = np.linalg.lstsq(np.eye(3) - R, t - (u @ t) * u, rcond=None)[0]
    return p - (p @ u) * u, u, ang / n, rise / n


def helix_axis(L: Layout, keys: Sequence[ResKey], coord_of, max_rmsd: float = 0.5):
    """The helical axis of a run of guide residues: the screw that carries every base pair (guide residue and
    partner) onto the next one, fitted over the whole run.  Z-DNA repeats every two base pairs (C anti, G syn),
    so no one screw fits its steps (about 4 A RMSD on a BNP_NA Z-DNA side); there the screw that carries every
    pair onto the pair two steps on gives the axis (-60.0 deg and 7.25 A, 0.001 A RMSD), and half its twist and
    rise are the twist and rise per step.  Returns (point on the axis, unit direction along the run, twist per
    step in deg, rise per step in A, base pairs per repeat: 1, or 2 for Z-DNA), or None when the run is too short
    or not helical, in which case the side keeps Curve It's principal axis."""
    for n in (1, 2):
        fit = _screw_fit(L, keys, coord_of, n, max_rmsd)
        if fit is not None:
            return (*fit, n)
    return None


def _seam_transform(L: Layout):
    """Rigid motion that puts the first segment right after the last one as one straight helix: the helical
    step of the last segment's base pairs (or the first segment's, if the last one is too short) applied to
    its last base pair gives where the first segment's first base pair belongs.  Where no one step fits the
    last segment (Z-DNA, which repeats every two base pairs), the step from each base pair to the one two on,
    applied to its last base pair but one, gives it instead."""
    atoms, res_atoms, partner, X = L.atoms, L.res_atoms, L.partner, L.X
    first, last = L.seg_keys[0], L.seg_keys[-1]
    src = last if len(last) >= 2 else first
    if len(src) < 2:
        raise ValueError("The two sides that meet at the chain ends are too short to give the helical step "
                         "that closes the ring.")

    def step(n: int):
        """The rigid motion that carries each base pair onto the one n on, over the last (or first) six such
        pairs of src, with the RMSD it leaves; None when too few atoms match."""
        w = min(6, len(src) - n)
        run = src[-(w + n):] if src is last else src[:w + n]
        P, Q = [], []
        for k1, k2 in zip(run, run[n:]):
            for i, j in _bp_pairs(atoms, res_atoms, partner, k1, k2):
                P.append(X[i])
                Q.append(X[j])
        return kabsch(np.array(P), np.array(Q)) if len(P) >= 4 else None

    one = step(1)
    if one is None:
        raise ValueError("Cannot measure the helical step at the chain ends.")
    H, h, fit = one
    n = 1
    if fit > STEP_RMSD_MAX and src is last and len(last) >= 3:
        two = step(2)
        if two is not None and two[2] <= STEP_RMSD_MAX:
            (H, h, _), n = two, 2
    pairs = _bp_pairs(atoms, res_atoms, partner, last[-n], first[0])
    if len(pairs) < 4:
        raise ValueError(f"The chain ends {res_label(last[-1])} and {res_label(first[0])} share too few atoms "
                         f"to be joined by an ordinary step.")
    target = np.array([X[i] for i, _ in pairs]) @ H.T + h        # the base pair after the last one
    R, t, rmsd = kabsch(np.array([X[j] for _, j in pairs]), target)
    return R, t, rmsd


def layout(atoms, corners: str = "auto", min_angle: float = DEFAULT_MIN_CORNER) -> Layout:
    X = np.array([a.coord for a in atoms], dtype=float)
    res_atoms, chains = residue_table(atoms)
    partner = find_base_pairs(atoms, res_atoms, X)
    mode, value = parse_corner_spec(corners)
    if mode == "auto":
        det = detect_corners(atoms, res_atoms, chains, X, value, min_angle, partner)
        guide, guide_keys, corner_pos, steps = det["chain"], det["keys"], det["corners"], det["steps"]
        if not corner_pos:
            raise ValueError(f"No corners found on chain {guide or '_'}: its straight stretches all point within "
                             f"{min_angle:g} deg of each other. List the corner residues instead (for example "
                             f"A28,A79).")
        detected = True
    else:
        keys = value
        guide = keys[0][0]
        if any(k[0] != guide for k in keys):
            raise ValueError("List every corner residue on one chain (the guide chain).")
        if guide not in chains:
            raise ValueError(f"Chain {guide or '_'} is not in the PDB.")
        guide_keys = chains[guide]
        pos = {k: n for n, k in enumerate(guide_keys)}
        loose = {(k[0], k[1]): n for n, k in enumerate(guide_keys)}
        corner_pos = []
        for k in keys:
            n = pos.get(k, loose.get((k[0], k[1])) if not k[2] else None)
            if n is None:
                raise ValueError(f"Residue {res_label(k)} is not in chain {guide or '_'}.")
            corner_pos.append(n)
        if any(b <= a for a, b in zip(corner_pos, corner_pos[1:])):
            raise ValueError("List the corner residues in the order of the chain.")
        if corner_pos[-1] >= len(guide_keys) - 1:
            raise ValueError(f"{res_label(guide_keys[corner_pos[-1]])} is the chain's last residue; a corner needs "
                             f"residues after it.")
        steps = guide_steps(atoms, res_atoms, guide_keys, X)
        detected = False
    # segments of the guide chain, then every other residue
    bounds = [-1] + list(corner_pos) + [len(guide_keys) - 1]
    seg_of: Dict[ResKey, int] = {}
    for s in range(len(bounds) - 1):
        for p in range(bounds[s] + 1, bounds[s + 1] + 1):
            seg_of[guide_keys[p]] = s
    for k in guide_keys:
        q = partner.get(k)
        if q is not None and q not in seg_of:
            seg_of[q] = seg_of[k]
    for ch, keys in chains.items():
        known = [i for i, k in enumerate(keys) if k in seg_of]
        if not known:
            continue
        for i, k in enumerate(keys):
            if k not in seg_of:
                j = min(known, key=lambda q: (abs(q - i), q > i))
                seg_of[k] = seg_of[keys[j]]
    rest = [k for k in res_atoms if k not in seg_of]
    if rest:
        from scipy.spatial import cKDTree
        ref = [i for k in seg_of for i in res_atoms[k]]
        owner = [seg_of[k] for k in seg_of for _ in res_atoms[k]]
        tree = cKDTree(X[ref])
        for k in rest:
            _, j = tree.query(X[res_atoms[k]].mean(0))
            seg_of[k] = owner[int(j)]
    seg_keys = [[guide_keys[p] for p in range(bounds[s] + 1, bounds[s + 1] + 1)
                 if is_core(atoms, res_atoms, guide_keys[p])] for s in range(len(bounds) - 1)]
    for s, ks in enumerate(seg_keys):
        if len(ks) < 2:
            raise ValueError(f"Segment {s + 1} of chain {guide or '_'} holds fewer than two residues; check the "
                             f"corner residues.")
    return Layout(atoms, X, res_atoms, chains, partner, guide, guide_keys, list(corner_pos), detected, steps,
                  seg_of, seg_keys)


def _segment_omega(L: Layout, segs: Sequence[int]) -> Optional[float]:
    """Helix twist per step of the segments, for the length suggestions: the mean twist of the guide residues'
    screws onto the next residue, or, on a segment that repeats every two base pairs (Z-DNA), the twist per step
    of its two-step screw, since there the residue-to-residue screws alternate (about +131 and -135 deg)."""
    pos = {k: n for n, k in enumerate(L.guide_keys)}
    tw = []
    for s in segs:
        ks = L.seg_keys[s]
        axis = helix_axis(L, ks, lambda i: L.X[i])
        if axis is not None and axis[4] > 1:
            tw.extend([axis[2]] * (len(ks) - 1))
            continue
        for k1, k2 in zip(ks, ks[1:]):
            n1, n2 = pos[k1], pos[k2]
            if n2 == n1 + 1 and L.steps[n1] is not None:
                tw.append(L.steps[n1]["twist"])
    return float(np.mean(tw)) if tw else None


def _span(L: Layout, segs: Sequence[int]) -> str:
    parts = []
    for s in segs:
        ks = L.seg_keys[s]
        parts.append(f"{res_label(ks[0])}-{res_label(ks[-1])}")
    return " + ".join(parts)


def build_plan(L: Layout, phases: Sequence[int], closed: bool, kappa_open: float = 0.0) -> Plan:
    atoms, res_atoms, partner, X = L.atoms, L.res_atoms, L.partner, L.X
    nseg = len(L.seg_keys)
    seg_atoms = [[] for _ in range(nseg)]
    for k, idx in res_atoms.items():
        seg_atoms[L.seg_of[k]].extend(idx)
    seg_atoms = [sorted(a) for a in seg_atoms]
    first_bp = [_bp_atoms(atoms, res_atoms, partner, ks, False) for ks in L.seg_keys]
    last_bp = [_bp_atoms(atoms, res_atoms, partner, ks, True) for ks in L.seg_keys]
    sides: List[Side] = []

    res_of = {i: k for k, idx in res_atoms.items() for i in idx}
    chain_pos = {k: (ch, n) for ch, keys in L.chains.items() for n, k in enumerate(keys)}

    def end_group(bp: Sequence[int], seg: int) -> List[int]:
        """Every atom of an end base pair's residues and of the linker residues (not standard nucleotides or
        amino acids) joined to them along a chain within the same segment: one rigid group."""
        res = {res_of[i] for i in bp}
        grow = True
        while grow:
            grow = False
            for k in list(res):
                ch, n = chain_pos[k]
                for m in (n - 1, n + 1):
                    if 0 <= m < len(L.chains[ch]):
                        q = L.chains[ch][m]
                        if q not in res and L.seg_of[q] == seg and not is_core(atoms, res_atoms, q):
                            res.add(q)
                            grow = True
        return [i for k in res for i in res_atoms[k]]

    def make_side(number, segs, Xside):
        """Side from segments segs (in order), with Xside the coordinates of their atoms in that order."""
        idx = [i for s in segs for i in seg_atoms[s]]
        row = {i: n for n, i in enumerate(idx)}
        keys = [k for s in segs for k in L.seg_keys[s]]
        axis = helix_axis(L, keys, lambda i: Xside[row[i]])
        n = sum(len(L.seg_keys[s]) for s in segs)
        rigid = [end_group(first_bp[segs[0]], segs[0]), end_group(last_bp[segs[-1]], segs[-1])]
        return Side(number, _span(L, segs), atoms, idx, Xside, first_bp[segs[0]], last_bp[segs[-1]], n,
                    _segment_omega(L, segs), segs, None if axis is None else axis[:2], rigid,
                    1 if axis is None else axis[4])

    if closed:
        if nseg == 1:
            raise ValueError("A closed path with corners needs at least one corner.")
        R, t, rmsd = _seam_transform(L)
        L.seam_rmsd = rmsd
        if rmsd > 0.6:
            raise ValueError(f"The chain ends ({res_label(L.seg_keys[-1][-1])} and {res_label(L.seg_keys[0][0])}) do "
                             f"not continue each other as one straight helix (step fit {rmsd:.2f} A). A closed "
                             f"path needs the chain ends in the middle of a side, joined by an ordinary step.")
        sides.append(make_side(1, [nseg - 1, 0], np.vstack([X[seg_atoms[-1]], X[seg_atoms[0]] @ R.T + t])))
        for s in range(1, nseg - 1):
            sides.append(make_side(s + 1, [s], X[seg_atoms[s]]))
        side_of_seg = {nseg - 1: 0, 0: 0, **{s: s for s in range(1, nseg - 1)}}
    else:
        for s in range(nseg):
            sides.append(make_side(s + 1, [s], X[seg_atoms[s]]))
        side_of_seg = {s: s for s in range(nseg)}
    # Corner c joins segment c to segment c + 1, so junctions[k - 1] runs from side k - 1 to side k; on a ring
    # the last corner runs from the last side back onto side 0, the seam side, and closes it.
    junctions = []
    for c in range(nseg - 1):
        a, b = side_of_seg[c], side_of_seg[c + 1]
        label = f"{res_label(L.seg_keys[c][-1])} | {res_label(L.seg_keys[c + 1][0])}"
        junctions.append(Junction(a, b, X[last_bp[c]], X[first_bp[c + 1]], phases[c], label))
    plan = Plan(sides, junctions, closed, kappa_open)
    for J, bend, off in zip(plan.junctions, plan.input_bends(), plan.axis_offsets()):
        J.bend, J.offset = bend, off
    return plan


# ----------------------------------------------------------------------------- checks and output
def suggestions(plan: Plan, misses: Sequence[dict]) -> List[dict]:
    """For every side that needs extra twist to point the corner after it: the length change (up to six
    residues either way, in whole repeats: two base pairs at a time on Z-DNA, so that the side keeps the
    conformations of its end base pairs) that leaves the least twist, and the twist if that corner's phase were
    flipped."""
    out = []
    for m in misses:
        S = plan.sides[m["side"]]
        need = wrap180(m["twist"])
        best = None
        if S.omega is not None:
            for dn in range(-6, 7):
                if dn == 0 or dn % S.repeat or S.n_res + dn < 3:
                    continue
                left = wrap180(need - dn * S.omega)
                if best is None or abs(left) < abs(best[1]) - 1e-9:
                    best = (dn, left)
        out.append(dict(corner=m["corner"], side=m["side"], need=need, per_res=need / S.steps,
                        dn=None if best is None else best[0], left=None if best is None else best[1],
                        flip=wrap180(need + 180.0)))
    return out


def model_checks(L: Layout, X_out: np.ndarray, closed: bool) -> dict:
    """Bonds that cross a corner (atoms of different segments bonded in the input), the bonds that would close
    the chains, and new close contacts (heavy atoms under 2.2 A that were not that close in the input)."""
    from scipy.spatial import cKDTree
    atoms, res_atoms, X = L.atoms, L.res_atoms, L.X
    seg_atom = np.empty(len(atoms), dtype=int)
    res_of = [None] * len(atoms)
    for k, idx in res_atoms.items():
        seg_atom[idx] = L.seg_of[k]
        for i in idx:
            res_of[i] = k
    heavy = np.array([i for i, a in enumerate(atoms) if _heavy(a)], dtype=int)
    cross = []
    for p, q in cKDTree(X[heavy]).query_pairs(1.9):
        i, j = heavy[p], heavy[q]
        if seg_atom[i] != seg_atom[j]:
            cross.append((float(np.linalg.norm(X[i] - X[j])), float(np.linalg.norm(X_out[i] - X_out[j]))))
    ends = []
    if closed:
        for ch, keys in L.chains.items():
            if len(keys) < 3:
                continue
            first, last = keys[0], keys[-1]
            names_last = {atoms[i].atom_name_norm: i for i in res_atoms[last]}
            names_first = {atoms[i].atom_name_norm: i for i in res_atoms[first]}
            for a1, a2 in (("O3'", "P"), ("C", "N")):
                if a1 in names_last and a2 in names_first:
                    ends.append((ch, f"{a1}({res_label(last)})-{a2}({res_label(first)})",
                                 float(np.linalg.norm(X_out[names_last[a1]] - X_out[names_first[a2]]))))
                    break
    end_pairs = {frozenset((keys[0], keys[-1])) for keys in L.chains.values()} if closed else set()
    new = []
    for p, q in cKDTree(X_out[heavy]).query_pairs(2.2):
        i, j = heavy[p], heavy[q]
        ki, kj = res_of[i], res_of[j]
        if ki == kj or (ki[0] == kj[0] and abs(ki[1] - kj[1]) <= 1) or frozenset((ki, kj)) in end_pairs:
            continue
        if np.linalg.norm(X[i] - X[j]) < 2.2:
            continue
        new.append((float(np.linalg.norm(X_out[i] - X_out[j])), res_label(ki), atoms[i].atom_name_norm,
                    res_label(kj), atoms[j].atom_name_norm))
    return {"cross": cross, "ends": ends, "contacts": sorted(new)}


def global_frame(plan: Plan, Xs_full) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation and translation applied to the result: the plane normal along +z (the curve turns
    counter-clockwise seen from +z, so 'in' corners turn left) and, for a closed path, the centre at the origin."""
    Rg = np.eye(3) if plan.z[2] > 0 else np.diag([1.0, -1.0, -1.0])
    if plan.closed:
        allx = np.vstack(Xs_full) @ Rg.T
        return Rg, -allx.mean(0)
    return Rg, np.zeros(3)


def centerline(plan: Plan, params, diag, Rg, tg, step: float = 1.0) -> List[np.ndarray]:
    comps = []
    for k, S in enumerate(plan.sides):
        n = max(2, int(math.ceil(S.length / step)) + 1)
        sig = np.linspace(0.0, S.length, n)
        pts = arc_points(params[k][0], sig) @ diag["Rs"][k].T + diag["Ts"][k]
        pts = pts @ Rg.T + tg
        comps.append(pts if S.dir > 0 else pts[::-1])
    return comps


def write_curve(path: str, comps: List[np.ndarray], header: List[str]) -> None:
    lines = [f"# {h}" for h in header]
    for n, c in enumerate(comps):
        if n:
            lines.append("")
        lines.extend(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in c)
    Path(path).write_text("\n".join(lines) + "\n")


def default_output_path(pdb_path: str) -> str:
    stem, ext = os.path.splitext(pdb_path)
    return f"{stem}_curved{ext or '.pdb'}"


# ----------------------------------------------------------------------------- the run
def describe_layout(L: Layout, plan: Optional[Plan] = None) -> List[str]:
    lines = []
    how = "found from the helix axis" if L.detected else "as given"
    corners = ", ".join(res_label(L.guide_keys[p]) for p in L.corner_pos)
    lines.append(f"Guide chain {L.guide or '_'}: {len(L.corner_pos)} corner{'s' if len(L.corner_pos) != 1 else ''} "
                 f"after {corners} ({how}); {len(L.partner) // 2} base pairs found.")
    if L.detected:
        lines.append(f"  Corner residues for --corners: {','.join(res_label(L.guide_keys[p]) for p in L.corner_pos)}")
    if plan is not None:
        for S in plan.sides:
            om = f", helix twist {S.omega:+.1f} deg per step" if S.omega is not None else ""
            if S.repeat > 1:
                om += f", repeating every {S.repeat} base pairs"
            if S.axis_source != "helical":
                om += "; no helical screw fits its base pairs, so it is mapped about its principal axis"
            lines.append(f"  side {S.number}: {S.span} ({S.n_res} residues on chain {L.guide or '_'}, axis "
                         f"{S.length:.1f} A{om})")
        for c, J in enumerate(plan.junctions):
            lines.append(f"  corner {c + 1} ({J.label}): bends {J.bend:.1f} deg, between side {J.a + 1} and side "
                         f"{J.b + 1}{' (closes the ring)' if plan.closing(c) else ''}; the two axes pass "
                         f"{abs(J.offset):.1f} A apart")
        if plan.closed and L.seam_rmsd is not None:
            lines.append(f"  The chain ends meet in side 1 at an ordinary step (step fit {L.seam_rmsd:.2f} A).")
    return lines


def phase_report(plan: Plan, misses: Sequence[dict], phases: Sequence[int]) -> List[str]:
    lines = []
    word = {+1: "in", -1: "out"}
    lines.append("Corner phases: " + ", ".join(word[J.sign] for J in plan.junctions)
                 + " (in: the corner bends towards the inside of the curve; out: away from it).")
    if plan.closed:
        turn = sum(J.sign * J.bend for J in plan.junctions)
        lines.append(f"The corners turn {turn:+.1f} deg in all, so the arcs turn {360 - turn:+.1f} deg.")
    rise = plan.rise()
    if abs(rise) >= 1.0:
        steps = ", ".join(f"{J.sign * J.offset:+.1f}" for J in plan.junctions)
        if plan.closed:
            tilt = math.degrees(math.atan2(abs(rise), sum(S.arc_length for S in plan.sides)))
            lines.append(f"The corners step the sides by {steps} A along the plane normal, {rise:+.1f} A per lap: "
                         f"with these phases the ring cannot lie flat, and its arcs will tilt by about "
                         f"{tilt:.1f} deg.")
        else:
            lines.append(f"The corners step the sides by {steps} A along the plane normal ({rise:+.1f} A in all).")
    lines.append("With no extra twist (each arc laid in the plane, each earlier corner pointed exactly):")
    first = plan.junctions[0]
    lines.append(f"  corner 1 ({first.label}): pointed by side 1's phase.")
    sug = {s["corner"]: s for s in suggestions(plan, misses)}
    for m in misses:
        J = plan.junctions[m["corner"]]
        S = plan.sides[m["side"]]
        s = sug[m["corner"]]
        text = (f"  corner {m['corner'] + 1} ({J.label}): misses {word[J.sign]} by {m['miss']:+.1f} deg; side "
                f"{S.number} needs {s['need']:+.1f} deg of extra twist ({s['per_res']:+.2f} deg per step)")
        if plan.closed and J.b == 0:
            text += " (shared with side 1)"
        lines.append(text + ".")
        if abs(s["per_res"]) > TWIST_WARN or abs(s["need"]) > 30.0:
            hints = []
            if s["dn"] is not None and abs(s["left"]) < abs(s["need"]) - 10.0:
                hints.append(f"side {S.number} {abs(s['dn'])} residue{'s' if abs(s['dn']) != 1 else ''} "
                             f"{'longer' if s['dn'] > 0 else 'shorter'} ({S.n_res + s['dn']}) would leave "
                             f"{s['left']:+.1f} deg")
            if abs(s["flip"]) < abs(s["need"]) - 10.0:
                hints.append(f"making corner {m['corner'] + 1} {word[-J.sign]} would leave {s['flip']:+.1f} deg")
            if hints:
                lines.append("    Suggestion: " + "; ".join(hints) + ".")
    return lines


def run(pdb_path: str, corners: str = "auto", phases: str = "in", path_type: str = "closed",
        radius=None, output: Optional[str] = None, curve_out: Optional[str] = None, check_only: bool = False,
        min_angle: float = DEFAULT_MIN_CORNER) -> dict:
    """Curve a helix with corners.  Prints a report; returns the solution and the output paths."""
    text = Path(pdb_path).read_text()
    atoms = CI.parse_pdb_atoms(text)
    if not atoms:
        raise ValueError("No ATOM/HETATM records in the input PDB.")
    closed = (path_type or "closed").lower() == "closed"
    kappa_open = 0.0 if closed else parse_radius(radius)
    L = layout(atoms, corners, min_angle)
    signs = parse_phase_spec(phases, len(L.corner_pos))
    plan = build_plan(L, signs, closed, kappa_open)
    print(f"[CORNERS] Corner Curve It {VERSION} ({CI.APP_NAME} {CI.APP_VERSION}): {Path(pdb_path).name}, "
          f"{'closed ring' if closed else 'open path'}.")
    for line in describe_layout(L, plan):
        print("[CORNERS] " + line)
    if not closed:
        print(f"[CORNERS] Open path: every side on an arc of radius "
              f"{('%.1f A' % (1 / kappa_open)) if kappa_open else 'infinity (straight sides)'}.")
    P0, misses = plan.sequential(plan.start_kappa())
    for line in phase_report(plan, misses, signs):
        print("[CORNERS] " + line)
    result = {"layout": L, "plan": plan, "misses": misses, "start": P0}
    if check_only:
        return result
    params = plan.solve(P0)
    diag = plan.diagnose(params)
    Rg, tg = global_frame(plan, diag["Xs"])
    X_out = np.empty_like(L.X)
    for k, S in enumerate(plan.sides):
        X_out[S.atom_idx] = diag["Xs"][k] @ Rg.T + tg
    checks = model_checks(L, X_out, closed)
    word = {+1: "in", -1: "out"}
    print("[CORNERS] Solved:")
    if closed:
        print(f"[CORNERS]   closure: corner {len(plan.junctions)} puts side 1's first base pair {diag['closure']:.3f} A "
              f"and {diag['closure_deg']:.3f} deg from where side 1 has it")
    for k, S in enumerate(plan.sides):
        kap, phi, tau = params[k]
        radius_txt = f"radius {1 / kap:7.1f} A" if abs(kap) > 1e-12 else "straight"
        tilt = f", arc {diag['tilts'][k - 1]:.2f} deg out of the plane" if k > 0 else ""
        print(f"[CORNERS]   side {S.number}: {radius_txt}, extra twist {tau:+7.2f} deg over {S.steps} steps "
              f"({tau / S.steps:+.3f} deg per step){tilt}")
    for c, (J, cd) in enumerate(zip(plan.junctions, diag["corners"])):
        axis = "" if math.isnan(cd["axis"]) else f", bend axis {cd['axis']:.2f} deg from the plane normal"
        print(f"[CORNERS]   corner {c + 1} ({J.label}): {word[J.sign]}, bends {cd['bend']:.1f} deg{axis}")
    if diag["fits"]:
        print(f"[CORNERS]   corners kept as built (each side's first base pair on its place within "
              f"{max(diag['fits']):.3f} A)")
    if checks["cross"]:
        d0 = [a for a, _ in checks["cross"]]
        d1 = [b for _, b in checks["cross"]]
        print(f"[CORNERS]   bonds across the corners: {min(d1):.2f}-{max(d1):.2f} A "
              f"(input {min(d0):.2f}-{max(d0):.2f} A, {len(d1)} bonds)")
    for ch, what, d in checks["ends"]:
        print(f"[CORNERS]   chain {ch or '_'} closing step {what}: {d:.2f} A")
    nc = len(checks["contacts"])
    print(f"[CORNERS]   new contacts under 2.2 A: {nc}"
          + (f" (closest {checks['contacts'][0][0]:.2f} A, {checks['contacts'][0][1]} "
             f"{checks['contacts'][0][2]} - {checks['contacts'][0][3]} {checks['contacts'][0][4]})" if nc else ""))
    worst = max(abs(params[k][2]) / S.steps for k, S in enumerate(plan.sides))
    if worst > TWIST_WARN:
        print(f"[WARN] The extra twist reaches {worst:.2f} deg per step (more than {TWIST_WARN} deg); "
              f"see the suggestions above for lengths that need less.")
    tilt_max = max(diag["tilts"], default=0.0)
    axis_max = max((cd["axis"] for cd in diag["corners"] if not math.isnan(cd["axis"])), default=0.0)
    if closed and diag["closure"] > 0.5:
        print("[WARN] The ring does not close; check the corners and the phases.")
    elif tilt_max > 2.0 or axis_max > 2.0:
        if closed and abs(plan.rise()) >= 1.0:
            print(f"[NOTE] The ring is not flat, as expected: with these phases the corners' axis offsets add up "
                  f"to {plan.rise():+.1f} A per lap.")
        else:
            print("[WARN] The solution is not flat; check the corners and the phases.")
    output = output or default_output_path(pdb_path)
    remarks = [
        f"Generated by {CI.APP_NAME} {CI.APP_VERSION}, corner mode (Corner Curve It {VERSION}).",
        f"Input PDB: {pdb_path}.",
        f"Corners: {','.join(res_label(L.guide_keys[p]) for p in L.corner_pos)} on chain {L.guide or '_'}"
        f" ({'found from the helix axis' if L.detected else 'as given'}); phases "
        f"{','.join(word[J.sign] for J in plan.junctions)}; path_type={'closed' if closed else 'open'}.",
    ]
    for k, S in enumerate(plan.sides):
        kap, phi, tau = params[k]
        hp, tw = S.phase_twist(phi, tau)
        rad = f"{1 / kap:.3f} A" if abs(kap) > 1e-12 else "straight"
        screw = f" (the screw that carries each base pair {S.repeat} steps on)" if S.repeat > 1 else ""
        remarks.append(f"Side {S.number} ({S.span}): arc radius {rad}; extra twist {tau:+.3f} deg between its end "
                       f"base pairs; mapped about its {S.axis_source} axis{screw} with helix_phase={hp:.4f} deg, "
                       f"twist={tw:.4f} deg.")
    if closed:
        remarks.append(f"Closure {diag['closure']:.3f} A, {diag['closure_deg']:.3f} deg; arcs within {tilt_max:.2f} deg and corner axes within "
                       f"{axis_max:.2f} deg of the plane normal.")
    remarks.append("Method: each straight side mapped about its helical axis onto its own circular arc, all in "
                   "one plane, with Curve It's rotation-minimizing frame and rigid atom-group transforms; each "
                   "side's end base pairs (with any linker on them) move as rigid groups, so the corners are kept "
                   "as in the input.")
    CI.write_output_pdb(text, X_out, output, remark_lines=CI.make_pdb_remark_lines(remarks))
    print(f"[CORNERS] Wrote {output}")
    if curve_out:
        comps = centerline(plan, params, diag, Rg, tg)
        write_curve(curve_out, comps, [f"Mapped axis of {Path(output).name}: one component per side "
                                       f"(Corner Curve It {VERSION}).",
                                       f"path_type={'closed' if closed else 'open'}; the components meet at "
                                       f"the corners, not end to end."])
        print(f"[CORNERS] Wrote {curve_out}")
    result.update(params=params, diag=diag, checks=checks, output=output, curve=curve_out, X_out=X_out)
    return result


# ----------------------------------------------------------------------------- command line and window
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="corner_curve_it.py",
        description="Curve a helix that already has corners onto arcs in one plane, every corner bending "
                    "into the curve (in) or out of it (out).")
    p.add_argument("pdb", nargs="?", help="Helix PDB with its corners already built.")
    p.add_argument("--corners", default="auto",
                   help="auto (find them on the longest chain), auto:B (on chain B), or the last residue before "
                        "each corner on one chain, e.g. A28,A79,A129,A180. Default: auto.")
    p.add_argument("--phase", default="in",
                   help="in or out for every corner, or one per corner in the chain's order, e.g. in,out,in,out. "
                        "Default: in.")
    p.add_argument("--path-type", choices=["closed", "open"], default="closed",
                   help="closed: a ring whose curvatures are solved so that it closes (default); open: every "
                        "side on an arc of --radius.")
    p.add_argument("--radius", default=None,
                   help="Open path only: arc radius of every side in A, or straight (default).")
    p.add_argument("-o", "--output", default=None, help="Output PDB. Default: <input>_curved.pdb.")
    p.add_argument("--curve", default=None, help="Also write the mapped axis as an XYZ curve, one component per side.")
    p.add_argument("--check", action="store_true",
                   help="Only report the corners, the twist each side needs and the suggested lengths.")
    p.add_argument("--min-corner", type=float, default=DEFAULT_MIN_CORNER,
                   help=f"auto: smallest turn between straight stretches that counts as a corner (deg); a "
                        f"junction between B- and Z-DNA always counts. Default: {DEFAULT_MIN_CORNER:g}.")
    p.add_argument("--gui", action="store_true", help="Open the window.")
    p.add_argument("-v", "--version", action="version", version=f"Corner Curve It {VERSION}")
    return p


def run_gui(args=None) -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    try:
        from curve_it_lib import azbmost_ui as ui
    except ImportError:
        import azbmost_ui as ui  # type: ignore[no-redef]

    args = args or build_parser().parse_args([])
    root = tk.Tk()
    root.title("Corner Curve It - curve a helix with corners")
    header = ui.tool_window(
        root, "Corner Curve It", module="curve_it", version=VERSION, pack=False,
        subtitle="Curve a helix with corners onto arcs in one plane, each corner in or out of phase.")
    header.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 15))
    ui.hairline(root).grid(row=0, column=0, columnspan=4, sticky="sew", pady=(0, 14))
    root.columnconfigure(1, weight=1)
    root.rowconfigure(10, weight=1)
    pdb0 = args.pdb or ""
    v = {
        "pdb": tk.StringVar(value=pdb0),
        "corners": tk.StringVar(value=args.corners or "auto"),
        "phase": tk.StringVar(value=args.phase or "in"),
        "path": tk.StringVar(value=args.path_type or "closed"),
        "radius": tk.StringVar(),
        "output": tk.StringVar(value=args.output or (default_output_path(pdb0) if pdb0 else "")),
        "curve": tk.StringVar(value=args.curve or ""),
    }
    radius_open = tk.StringVar(value=str(args.radius) if args.radius else "straight")
    help_text = {
        "pdb": "The helix with its corners already built: straight sides joined at bends, such as a duplex "
               "bent at a pivot by RE_Helix's Bend Helix, or D- and L-RNA pieces joined by bent mirror "
               "junctions. Each side is mapped onto its own arc; the corners stay exactly as they are in this "
               "file. For a ring, the chain ends must meet in the middle of a side, at an ordinary step, not at "
               "a corner.\n\nExample: D28_L50_D50_L50_D22_straight.pdb",
        "corners": "auto finds the corners on the longest chain: places where two straight stretches of helix "
                   "meet at 10 degrees or more, and every junction between B- and Z-DNA, however small its bend. "
                   "auto:B looks on chain B instead.\n\nOr list the last residue "
                   "before each corner, all on one chain, in the chain's order.\n\nExamples: auto; auto:B; "
                   "A28,A79,A129,A180\n\nDetect fills this in and lists the sides and their bends.",
        "phase": "One word per corner, in the order of the guide chain, or one word for all of them.\n\n"
                 "in: the corner bends towards the inside of the curve (in phase with it).\nout: the corner "
                 "bends away from it (out of phase).\n\nExamples: in; out; in,out,in,out\n\nCheck shows how "
                 "much extra twist each side needs for these phases, and which lengths would need less.",
        "path": "closed: a ring. Every side's arc radius, phase and twist are solved so that the ring closes, "
                "the arcs and the corners lie in one plane and the bending is spread evenly.\n\nopen: an open "
                "path. Every side gets the arc radius below, and the phases and twists put the corners in one "
                "plane.",
        "radius": "Open path only: the arc radius of every side, in Angstrom. straight (or 0) keeps the sides "
                  "straight and only turns the corners into one plane.\n\nExamples: straight; 150\n\nA closed "
                  "path solves its radii, so this shows solved.",
        "output": "The curved model: the input's records in the input's order, with new coordinates and REMARK "
                  "900 lines giving the corners, the phases and each side's radius, helix phase and twist. The "
                  "ring lies in the xy plane with its centre at the origin.\n\nDefault: <input>_curved.pdb",
        "curve": "Optional. The mapped axis of every side, one curve component per side (separated by blank "
                 "lines), in the output model's frame. Leave blank to skip.\n\nExample: ring_axis.xyz",
    }

    def help_button(key, row):
        ui.HelpButton(root, command=lambda: messagebox.showinfo("Corner Curve It", help_text[key],
                                                                parent=root)).grid(
            row=row, column=3, sticky="w", padx=(8, 20), pady=3)

    def browse(key):
        if key in ("output", "curve"):
            ext = ".pdb" if key == "output" else ".xyz"
            sel = filedialog.asksaveasfilename(parent=root, defaultextension=ext,
                                               filetypes=[("PDB files" if key == "output" else "Curve XYZ",
                                                           "*" + ext), ("All files", "*.*")])
        else:
            sel = filedialog.askopenfilename(parent=root, title="Select the helix PDB",
                                             filetypes=[("PDB files", "*.pdb *.ent"), ("All files", "*.*")])
        if sel:
            if key == "pdb":
                old = default_output_path(v["pdb"].get()) if v["pdb"].get() else ""
                if v["output"].get() in ("", old):
                    v["output"].set(default_output_path(sel))
            v[key].set(sel)

    rows = (("pdb", "Helix PDB", "browse"), ("corners", "Corners", "detect"), ("phase", "Corner phases", None),
            ("path", "Path type", None), ("radius", "Arc radius (Å)", None), ("output", "Output PDB", "browse"),
            ("curve", "Axis curve XYZ", "browse"))
    widgets = {}
    for row, (key, label, button) in enumerate(rows, 1):
        ttk.Label(root, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", padx=(20, 12), pady=3)
        if key == "path":
            w = ttk.Combobox(root, textvariable=v[key], state="readonly", values=("closed", "open"), width=12)
            w.grid(row=row, column=1, sticky="w", pady=3)
        elif key in ("radius", "phase"):
            w = ttk.Entry(root, textvariable=v[key], width=24)
            w.grid(row=row, column=1, sticky="w", pady=3)
        else:
            w = ttk.Entry(root, textvariable=v[key], width=60)
            w.grid(row=row, column=1, sticky="ew", pady=3)
        widgets[key] = w
        if button == "browse":
            ttk.Button(root, text="Browse...", command=lambda k=key: browse(k)).grid(row=row, column=2,
                                                                                     padx=(8, 0), pady=3)
        elif button == "detect":
            ttk.Button(root, text="Detect", command=lambda: go("detect")).grid(row=row, column=2, padx=(8, 0),
                                                                               pady=3)
        help_button(key, row)

    def update_path(*_):
        if v["path"].get() == "closed":
            if v["radius"].get() != "solved":
                radius_open.set(v["radius"].get() or "straight")
            v["radius"].set("solved")
            widgets["radius"].configure(state="disabled")
        else:
            widgets["radius"].configure(state="normal")
            v["radius"].set(radius_open.get())

    v["path"].trace_add("write", update_path)
    update_path()
    buttons = ttk.Frame(root)
    buttons.grid(row=9, column=0, columnspan=4, sticky="w", padx=20, pady=(10, 0))
    log_view = ui.LogView(root, wrap="word", height=18)
    log_view.grid(row=10, column=0, columnspan=4, sticky="nsew", padx=20, pady=(10, 16))
    log = log_view.text
    log.configure(width=100, state="disabled")
    status = tk.StringVar(value="Ready.")

    def write_log(text):
        log.configure(state="normal")
        log.insert("end", text)
        log.see("end")
        log.configure(state="disabled")

    def go(what):
        pdb = v["pdb"].get().strip()
        if not pdb or not os.path.isfile(pdb):
            messagebox.showerror("Corner Curve It", "Select the helix PDB first.", parent=root)
            return
        buf = io.StringIO()
        status.set({"detect": "Finding corners...", "check": "Checking phases...", "run": "Solving..."}[what])
        root.update_idletasks()
        try:
            with redirect_stdout(buf):
                if what == "detect":
                    atoms = CI.parse_pdb_atoms(Path(pdb).read_text())
                    spec = v["corners"].get().strip() or "auto"
                    if not spec.lower().startswith("auto"):
                        spec = "auto"
                    L = layout(atoms, spec)
                    signs = parse_phase_spec("in", len(L.corner_pos))
                    plan = build_plan(L, signs, v["path"].get() == "closed", 0.0)
                    for line in describe_layout(L, plan):
                        print(line)
                    v["corners"].set(",".join(res_label(L.guide_keys[p]) for p in L.corner_pos))
                else:
                    radius = None if v["path"].get() == "closed" else v["radius"].get()
                    run(pdb, v["corners"].get(), v["phase"].get(), v["path"].get(), radius,
                        v["output"].get().strip() or None, v["curve"].get().strip() or None,
                        check_only=(what == "check"))
            write_log(buf.getvalue() + "\n")
            status.set("Done.")
        except Exception as exc:  # report every failure in the log and a dialog
            write_log(buf.getvalue() + f"error: {exc}\n\n")
            status.set("Failed.")
            messagebox.showerror("Corner Curve It", str(exc), parent=root)

    ttk.Button(buttons, text="Check phases", command=lambda: go("check")).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(buttons, text="Run", style="Accent.TButton", command=lambda: go("run")).grid(row=0, column=1)
    ttk.Label(buttons, textvariable=status, style="Hint.TLabel").grid(row=0, column=2, padx=(16, 0))
    root.mainloop()
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not argv or args.gui:
        return run_gui(args)
    if not args.pdb:
        parser.error("give the helix PDB, or use --gui")
    try:
        run(args.pdb, args.corners, args.phase, args.path_type, args.radius, args.output, args.curve,
            args.check, args.min_corner)
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
