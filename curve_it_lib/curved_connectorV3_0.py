#!/usr/bin/env python3
"""
curved_connectorV3_0.py

Screen and build curved nucleic-acid connectors between two helical end base-pairs.

Main features in V3_0
---------------------
1. Free residue matching at both source and destination base-pairs:
   swapping A33,B1 <-> B1,A33 or E1,F33 <-> F33,E1 gives the same result.
2. Screen only one canonical template direction by default to avoid redundant
   reverse-direction duplicates.
3. Preserve *all* atoms from the first input PDB in the final output assemblies.
4. Renumber connector residues chain-by-chain from 5' -> 3'.
5. Report the maximum local curvature of the centerline in connector_summary.tsv.
6. Provide an optional Tk GUI; if run with no arguments or with --gui, the GUI
   opens and can suggest likely helix-end residues after loading the target PDB.

Geometry note
-------------
The connector centerline is a practical clamped Euler-elastica proxy. It matches
the two endpoint positions and tangent directions, then screens template lengths
using an efficient cubic clamped centerline, optionally fairing-refined. The
reported twist_mismatch_deg is an endpoint base-pair orientation mismatch, not
integrated curve torsion and not material twist energy.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares, minimize, minimize_scalar
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


SUGAR_ATOMS = ["C1'", "C2'", "C3'", "C4'", "O5'", "C5'"]
CHAIN_ID_POOL = list("XYZUVWQRSTMNOPLKJIHGFEDCBA0123456789")
EPS = 1.0e-8
TOOL_NAME = "Curved Connector"
TOOL_VERSION = "V3_0"
DEFAULT_OUTDIR = "curved_connectorV3_out"


def resource_path(relative_path: str) -> str:
    """Return a resource path that also works from a PyInstaller bundle."""
    source_dir = os.path.dirname(os.path.abspath(__file__))
    source_root = os.path.dirname(source_dir) if os.path.basename(source_dir) == "curve_it_lib" else source_dir
    base_dir = getattr(sys, "_MEIPASS", source_root)
    return os.path.join(base_dir, relative_path)


def set_optional_window_icon(root, tk_module, icon_filenames: List[str], image_attr: str) -> None:
    """Set a Tk window icon if one of the optional PNG assets is available."""
    for icon_filename in icon_filenames:
        icon_path = resource_path(os.path.join("assets", icon_filename))
        if not os.path.isfile(icon_path):
            continue
        try:
            icon_image = tk_module.PhotoImage(file=icon_path)
            root.iconphoto(True, icon_image)
            setattr(root, image_attr, icon_image)
            return
        except Exception:
            continue


def eprint(*args: object, **kwargs: object) -> None:
    print(*args, file=sys.stderr, **kwargs)


@dataclass(frozen=True)
class EndpointSpec:
    chain1: str
    res1: int
    chain2: str
    res2: int

    def label(self) -> str:
        return f"{self.chain1}{self.res1},{self.chain2}{self.res2}"


@dataclass
class AtomRecord:
    serial: int
    atom_name: str
    atom_name_norm: str
    res_name: str
    chain_id: str
    res_seq: int
    i_code: str
    coord: np.ndarray
    element: str
    occupancy: float
    temp_factor: float
    original_line: str

    def copy_with_coord(self, coord: np.ndarray) -> "AtomRecord":
        return AtomRecord(
            serial=self.serial,
            atom_name=self.atom_name,
            atom_name_norm=self.atom_name_norm,
            res_name=self.res_name,
            chain_id=self.chain_id,
            res_seq=self.res_seq,
            i_code=self.i_code,
            coord=np.asarray(coord, dtype=float),
            element=self.element,
            occupancy=self.occupancy,
            temp_factor=self.temp_factor,
            original_line=self.original_line,
        )


@dataclass
class Frame:
    origin: np.ndarray
    n: np.ndarray
    b: np.ndarray
    t: np.ndarray

    def matrix(self) -> np.ndarray:
        return np.column_stack((self.n, self.b, self.t))


@dataclass
class StructureIndex:
    atoms: List[AtomRecord]
    residues: Dict[Tuple[str, int, str], List[int]]
    ranges: Dict[str, Tuple[int, int]]
    chains: List[str]

    def key_exists(self, chain: str, res_seq: int, i_code: str = "") -> bool:
        return (chain, res_seq, i_code) in self.residues

    def residue_atom_indices(self, chain: str, res_seq: int, i_code: str = "") -> List[int]:
        key = (chain, res_seq, i_code)
        if key not in self.residues:
            raise KeyError(f"Residue {chain}{res_seq}{i_code or ''} not found.")
        return self.residues[key]

    def residue_atoms(self, chain: str, res_seq: int, i_code: str = "") -> List[AtomRecord]:
        return [self.atoms[i] for i in self.residue_atom_indices(chain, res_seq, i_code)]


@dataclass
class Fragment:
    atoms: List[AtomRecord]
    pair_series: List[EndpointSpec]
    start_endpoint: EndpointSpec
    end_endpoint: EndpointSpec
    old_to_new_resid: Dict[Tuple[str, int], Tuple[str, int]]
    chain_order: Tuple[str, str]
    start_label: str


# ---------------------------------------------------------------------------
# PDB parsing and indexing
# ---------------------------------------------------------------------------

def normalize_atom_name(atom_name: str) -> str:
    name = atom_name.strip().upper()
    return name.replace("*", "'").replace("`", "'")


def guess_element(line: str, atom_name_norm: str) -> str:
    elem = ""
    if len(line) >= 78:
        elem = line[76:78].strip()
    if not elem:
        letters = [c for c in atom_name_norm if c.isalpha()]
        if letters:
            elem = letters[0]
    return elem.upper()


def parse_pdb_atoms(path: str) -> List[AtomRecord]:
    atoms: List[AtomRecord] = []
    with open(path, "r") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atom_name = line[12:16].strip()
            atom_name_norm = normalize_atom_name(atom_name)
            res_name = line[17:20].strip()
            chain_id = line[21].strip() if len(line) > 21 else ""
            res_seq_str = line[22:26].strip() if len(line) > 26 else "0"
            try:
                res_seq = int(res_seq_str)
            except ValueError:
                res_seq = 0
            i_code = line[26].strip() if len(line) > 26 else ""
            serial_str = line[6:11].strip() if len(line) > 11 else "0"
            try:
                serial = int(serial_str)
            except ValueError:
                serial = len(atoms) + 1
            occupancy = 1.0
            temp_factor = 0.0
            if len(line) >= 60:
                try:
                    occupancy = float(line[54:60])
                except ValueError:
                    occupancy = 1.0
            if len(line) >= 66:
                try:
                    temp_factor = float(line[60:66])
                except ValueError:
                    temp_factor = 0.0
            atoms.append(
                AtomRecord(
                    serial=serial,
                    atom_name=atom_name,
                    atom_name_norm=atom_name_norm,
                    res_name=res_name,
                    chain_id=chain_id,
                    res_seq=res_seq,
                    i_code=i_code,
                    coord=np.array([x, y, z], dtype=float),
                    element=guess_element(line, atom_name_norm),
                    occupancy=occupancy,
                    temp_factor=temp_factor,
                    original_line=line.rstrip("\n"),
                )
            )
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    return atoms


def build_structure_index(atoms: List[AtomRecord]) -> StructureIndex:
    residues: Dict[Tuple[str, int, str], List[int]] = defaultdict(list)
    ranges_map: Dict[str, List[int]] = defaultdict(list)
    for i, atom in enumerate(atoms):
        residues[(atom.chain_id, atom.res_seq, atom.i_code)].append(i)
        ranges_map[atom.chain_id].append(atom.res_seq)
    ranges = {chain: (min(vals), max(vals)) for chain, vals in ranges_map.items() if vals}
    chains = sorted(ranges)
    idx = StructureIndex(atoms=atoms, residues=dict(residues), ranges=ranges, chains=chains)
    setattr(idx, "_chain_order_5to3", {})
    setattr(idx, "_chain_pos_5to3", {})
    return idx


def parse_endpoint_spec(text: str) -> EndpointSpec:
    m = re.match(r"^\s*([^,\s])(\-?\d+)\s*,\s*([^,\s])(\-?\d+)\s*$", text)
    if not m:
        raise ValueError(f"Could not parse endpoint specification '{text}'. Use e.g. A33,B1")
    return EndpointSpec(m.group(1), int(m.group(2)), m.group(3), int(m.group(4)))


def validate_endpoint(index: StructureIndex, ep: EndpointSpec) -> None:
    for chain, resi in ((ep.chain1, ep.res1), (ep.chain2, ep.res2)):
        if chain not in index.ranges:
            raise ValueError(f"Chain '{chain}' not found.")
        if not index.key_exists(chain, resi):
            raise ValueError(f"Residue {chain}{resi} not found.")


def residue_atom_map(index: StructureIndex, chain: str, res_seq: int) -> Dict[str, AtomRecord]:
    atoms = index.residue_atoms(chain, res_seq)
    amap: Dict[str, AtomRecord] = {}
    for atom in atoms:
        if atom.atom_name_norm not in amap:
            amap[atom.atom_name_norm] = atom
    return amap


def residue_atom_coord(index: StructureIndex, chain: str, res_seq: int, atom_name: str) -> Optional[np.ndarray]:
    amap = residue_atom_map(index, chain, res_seq)
    atom_name = normalize_atom_name(atom_name)
    atom = amap.get(atom_name)
    return None if atom is None else atom.coord


def chain_residue_numbers(index: StructureIndex, chain: str) -> List[int]:
    vals = sorted({res for (ch, res, _ic) in index.residues if ch == chain})
    if not vals:
        raise ValueError(f"Chain '{chain}' has no residues.")
    return vals


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    n = np.linalg.norm(arr)
    if n < EPS:
        raise ValueError("Encountered near-zero vector during normalization.")
    return arr / n


def orthogonalize(vec: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.asarray(vec, dtype=float) - np.dot(vec, axis) * np.asarray(axis, dtype=float)


def axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis.tolist()
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def direct_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return pts @ R.T + t


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, _S, Wt = np.linalg.svd(C)
    d = np.linalg.det(V @ Wt)
    D = np.diag([1.0, 1.0, np.sign(d)])
    return V @ D @ Wt


def rigid_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    R = kabsch_rotation(P, Q)
    t = Q.mean(axis=0) - (P.mean(axis=0) @ R.T)
    moved = transform_points(P, R, t)
    return R, t, direct_rmsd(moved, Q)


# ---------------------------------------------------------------------------
# Chain orientation, residue directions, endpoint series
# ---------------------------------------------------------------------------

def infer_chain_5to3_order(index: StructureIndex, chain: str) -> List[int]:
    cache: Dict[str, List[int]] = getattr(index, "_chain_order_5to3")
    if chain in cache:
        return list(cache[chain])

    residues = chain_residue_numbers(index, chain)
    if len(residues) <= 1:
        cache[chain] = list(residues)
        getattr(index, "_chain_pos_5to3")[chain] = {r: i for i, r in enumerate(residues)}
        return list(residues)

    d_forward: List[float] = []
    d_reverse: List[float] = []
    for a, b in zip(residues[:-1], residues[1:]):
        o3_a = residue_atom_coord(index, chain, a, "O3'")
        p_b = residue_atom_coord(index, chain, b, "P")
        if o3_a is not None and p_b is not None:
            d_forward.append(float(np.linalg.norm(o3_a - p_b)))
        o3_b = residue_atom_coord(index, chain, b, "O3'")
        p_a = residue_atom_coord(index, chain, a, "P")
        if o3_b is not None and p_a is not None:
            d_reverse.append(float(np.linalg.norm(o3_b - p_a)))

    score_f = float(np.median(d_forward)) if d_forward else float("inf")
    score_r = float(np.median(d_reverse)) if d_reverse else float("inf")

    if score_r + 0.5 < score_f:
        order = list(reversed(residues))
    else:
        order = list(residues)

    cache[chain] = list(order)
    getattr(index, "_chain_pos_5to3")[chain] = {r: i for i, r in enumerate(order)}
    return list(order)


def residue_pos_5to3(index: StructureIndex, chain: str, res_seq: int) -> int:
    infer_chain_5to3_order(index, chain)
    pos_cache: Dict[str, Dict[int, int]] = getattr(index, "_chain_pos_5to3")
    if res_seq not in pos_cache[chain]:
        raise ValueError(f"Residue {chain}{res_seq} not present in inferred 5'->3' order.")
    return int(pos_cache[chain][res_seq])


def chain_termini(index: StructureIndex, chain: str) -> Tuple[int, int]:
    order = infer_chain_5to3_order(index, chain)
    return order[0], order[-1]


def residue_local_5to3_direction(index: StructureIndex, chain: str, res_seq: int) -> np.ndarray:
    c3 = residue_atom_coord(index, chain, res_seq, "C3'")
    c5 = residue_atom_coord(index, chain, res_seq, "C5'")
    if c3 is not None and c5 is not None:
        return normalize(c3 - c5)
    o5 = residue_atom_coord(index, chain, res_seq, "O5'")
    if c3 is not None and o5 is not None:
        return normalize(c3 - o5)

    order = infer_chain_5to3_order(index, chain)
    pos = residue_pos_5to3(index, chain, res_seq)
    def sugar_cent(r: int) -> np.ndarray:
        return residue_sugar_centroid(index, chain, r)

    if len(order) == 1:
        raise ValueError(f"Cannot infer local 5'->3' direction for singleton chain {chain}.")
    if pos == 0:
        return normalize(sugar_cent(order[1]) - sugar_cent(order[0]))
    if pos == len(order) - 1:
        return normalize(sugar_cent(order[-1]) - sugar_cent(order[-2]))
    return normalize(sugar_cent(order[pos + 1]) - sugar_cent(order[pos - 1]))


def endpoint_inward_step(index: StructureIndex, chain: str, res_seq: int) -> int:
    order = infer_chain_5to3_order(index, chain)
    pos = residue_pos_5to3(index, chain, res_seq)
    if pos == 0:
        return +1
    if pos == len(order) - 1:
        return -1
    raise ValueError(
        f"Residue {chain}{res_seq} is not clearly a chain terminus in 5'->3' order."
    )


def endpoint_series(index: StructureIndex, ep: EndpointSpec, n_bp: int) -> List[EndpointSpec]:
    if n_bp < 1:
        raise ValueError("n_bp must be >= 1")
    order1 = infer_chain_5to3_order(index, ep.chain1)
    order2 = infer_chain_5to3_order(index, ep.chain2)
    pos1 = residue_pos_5to3(index, ep.chain1, ep.res1)
    pos2 = residue_pos_5to3(index, ep.chain2, ep.res2)
    step1 = endpoint_inward_step(index, ep.chain1, ep.res1)
    step2 = endpoint_inward_step(index, ep.chain2, ep.res2)

    out: List[EndpointSpec] = []
    for k in range(n_bp):
        i1 = pos1 + k * step1
        i2 = pos2 + k * step2
        if not (0 <= i1 < len(order1) and 0 <= i2 < len(order2)):
            raise ValueError(f"Requested n_bp={n_bp} extends beyond a chain terminus.")
        p = EndpointSpec(ep.chain1, order1[i1], ep.chain2, order2[i2])
        validate_endpoint(index, p)
        out.append(p)
    return out


def max_n_bp_from_endpoint(index: StructureIndex, ep: EndpointSpec) -> int:
    order1 = infer_chain_5to3_order(index, ep.chain1)
    order2 = infer_chain_5to3_order(index, ep.chain2)
    pos1 = residue_pos_5to3(index, ep.chain1, ep.res1)
    pos2 = residue_pos_5to3(index, ep.chain2, ep.res2)
    step1 = endpoint_inward_step(index, ep.chain1, ep.res1)
    step2 = endpoint_inward_step(index, ep.chain2, ep.res2)
    span1 = (len(order1) - 1 - pos1) if step1 > 0 else pos1
    span2 = (len(order2) - 1 - pos2) if step2 > 0 else pos2
    return int(min(span1, span2) + 1)


def ordered_sugar_coords(index: StructureIndex, chain: str, res_seq: int) -> np.ndarray:
    amap: Dict[str, np.ndarray] = {}
    for atom in index.residue_atoms(chain, res_seq):
        if atom.atom_name_norm in SUGAR_ATOMS and atom.atom_name_norm not in amap:
            amap[atom.atom_name_norm] = atom.coord
    missing = [name for name in SUGAR_ATOMS if name not in amap]
    if missing:
        raise ValueError(f"Missing sugar atoms in {chain}{res_seq}: {', '.join(missing)}")
    return np.vstack([amap[name] for name in SUGAR_ATOMS])


def bp_sugar_blocks(index: StructureIndex, ep: EndpointSpec) -> List[np.ndarray]:
    return [
        ordered_sugar_coords(index, ep.chain1, ep.res1),
        ordered_sugar_coords(index, ep.chain2, ep.res2),
    ]


def stack_bp_blocks(blocks: List[np.ndarray], perm: Tuple[int, int] = (0, 1)) -> np.ndarray:
    return np.vstack([blocks[perm[0]], blocks[perm[1]]])


def ordered_bp_sugar_coords(index: StructureIndex, ep: EndpointSpec) -> np.ndarray:
    return stack_bp_blocks(bp_sugar_blocks(index, ep), (0, 1))


def bp_sugar_centroid(index: StructureIndex, ep: EndpointSpec) -> np.ndarray:
    return ordered_bp_sugar_coords(index, ep).mean(axis=0)


def residue_sugar_centroid(index: StructureIndex, chain: str, res_seq: int) -> np.ndarray:
    return ordered_sugar_coords(index, chain, res_seq).mean(axis=0)


def fit_direction(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Need at least two points to fit a direction.")
    centered = pts - pts.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    chord = pts[-1] - pts[0]
    if np.dot(direction, chord) < 0.0:
        direction = -direction
    return normalize(direction)


def endpoint_tangent(index: StructureIndex, ep: EndpointSpec, mode: str, k: int = 11) -> np.ndarray:
    max_n = max_n_bp_from_endpoint(index, ep)
    k_use = max(1, min(k, max_n - 1))
    centers = np.vstack([bp_sugar_centroid(index, p) for p in endpoint_series(index, ep, k_use + 1)])
    inward = fit_direction(centers)
    if mode.lower() == "inward":
        return inward
    if mode.lower() == "outward":
        return -inward
    raise ValueError("mode must be 'inward' or 'outward'")


def endpoint_frame_with_reference(index: StructureIndex, ep: EndpointSpec, mode: str, ref_residue_index: int, k: int = 11) -> Frame:
    origin = bp_sugar_centroid(index, ep)
    t = endpoint_tangent(index, ep, mode=mode, k=k)
    residue_centers = [
        residue_sugar_centroid(index, ep.chain1, ep.res1),
        residue_sugar_centroid(index, ep.chain2, ep.res2),
    ]
    ref = orthogonalize(residue_centers[ref_residue_index] - origin, t)
    if np.linalg.norm(ref) < 1.0e-6:
        other = 1 - ref_residue_index
        ref = orthogonalize(residue_centers[other] - origin, t)
    if np.linalg.norm(ref) < 1.0e-6:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, t)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        ref = orthogonalize(tmp, t)
    n = normalize(ref)
    b = normalize(np.cross(t, n))
    n = normalize(np.cross(b, t))
    return Frame(origin=origin, n=n, b=b, t=t)


def frame_alignment_transform(src: Frame, dst: Frame) -> Tuple[np.ndarray, np.ndarray]:
    R = dst.matrix() @ src.matrix().T
    t = dst.origin - (R @ src.origin)
    return R, t


def twist_refine_about_axis(moving: np.ndarray, target: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    axis = normalize(axis)
    moving = np.asarray(moving, dtype=float)
    target = np.asarray(target, dtype=float)

    def rmsd_for_angle(phi: float) -> float:
        R = axis_angle_matrix(axis, phi)
        moved = origin + (moving - origin) @ R.T
        return direct_rmsd(moved, target)

    if HAVE_SCIPY:
        opt = minimize_scalar(rmsd_for_angle, bounds=(-math.pi, math.pi), method="bounded")
        return float(opt.x), float(opt.fun)

    angles = np.linspace(-math.pi, math.pi, 721)
    values = [rmsd_for_angle(a) for a in angles]
    idx = int(np.argmin(values))
    return float(angles[idx]), float(values[idx])

# ---------------------------------------------------------------------------
# Template handling and fragment extraction
# ---------------------------------------------------------------------------

def fragment_index(fragment: Fragment) -> StructureIndex:
    return build_structure_index(fragment.atoms)


def choose_canonical_template_start(index: StructureIndex) -> Tuple[str, EndpointSpec]:
    if len(index.chains) != 2:
        raise ValueError(
            f"Template PDB must contain exactly 2 chains; found {len(index.chains)}: {index.chains}"
        )
    c1, c2 = index.chains
    c1_5, c1_3 = chain_termini(index, c1)
    c2_5, c2_3 = chain_termini(index, c2)
    ep = EndpointSpec(c1, c1_5, c2, c2_3)
    return ep.label(), ep


def extract_fragment(index: StructureIndex, start_ep: EndpointSpec, n_bp: int, start_label: str) -> Fragment:
    series = endpoint_series(index, start_ep, n_bp)
    selected_by_chain: Dict[str, set[int]] = defaultdict(set)
    for bp in series:
        selected_by_chain[bp.chain1].add(bp.res1)
        selected_by_chain[bp.chain2].add(bp.res2)

    keys = {(ch, res) for ch, vals in selected_by_chain.items() for res in vals}
    frag_atoms = [a.copy_with_coord(a.coord.copy()) for a in index.atoms if (a.chain_id, a.res_seq) in keys]

    old_to_new: Dict[Tuple[str, int], Tuple[str, int]] = {}
    for chain, residues in selected_by_chain.items():
        order_5to3 = infer_chain_5to3_order(index, chain)
        ordered_selected = [r for r in order_5to3 if r in residues]
        for new_res, old_res in enumerate(ordered_selected, start=1):
            old_to_new[(chain, old_res)] = (chain, new_res)

    return Fragment(
        atoms=frag_atoms,
        pair_series=series,
        start_endpoint=series[0],
        end_endpoint=series[-1],
        old_to_new_resid=old_to_new,
        chain_order=(start_ep.chain1, start_ep.chain2),
        start_label=start_label,
    )


# ---------------------------------------------------------------------------
# Alignment with free residue matching
# ---------------------------------------------------------------------------

def _apply_column_rotation(v: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.asarray(R @ np.asarray(v, dtype=float), dtype=float)


def _transform_atom_coords_col(x: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(R @ np.asarray(x, dtype=float) + t, dtype=float)


def _bp_match_labels(ep: EndpointSpec, perm: Tuple[int, int]) -> str:
    residues = [f"{ep.chain1}{ep.res1}", f"{ep.chain2}{ep.res2}"]
    return f"0->{residues[perm[0]]};1->{residues[perm[1]]}"


def align_fragment_to_source_free(
    fragment: Fragment,
    source_index: StructureIndex,
    source_endpoint: EndpointSpec,
    tangent_k: int,
    min_direction_dot: float,
) -> Tuple[Fragment, Dict[str, float], Frame]:
    frag_idx = fragment_index(fragment)

    frag_start_blocks = bp_sugar_blocks(frag_idx, fragment.start_endpoint)
    frag_start_coords = stack_bp_blocks(frag_start_blocks, (0, 1))
    frag_inward_tangent = endpoint_tangent(frag_idx, fragment.start_endpoint, mode="inward", k=tangent_k)
    frag_local_dirs = [
        residue_local_5to3_direction(frag_idx, fragment.start_endpoint.chain1, fragment.start_endpoint.res1),
        residue_local_5to3_direction(frag_idx, fragment.start_endpoint.chain2, fragment.start_endpoint.res2),
    ]
    template_frame = endpoint_frame_with_reference(
        frag_idx, fragment.start_endpoint, mode="inward", ref_residue_index=0, k=tangent_k
    )

    src_blocks = bp_sugar_blocks(source_index, source_endpoint)
    src_local_dirs = [
        residue_local_5to3_direction(source_index, source_endpoint.chain1, source_endpoint.res1),
        residue_local_5to3_direction(source_index, source_endpoint.chain2, source_endpoint.res2),
    ]

    candidates: List[Tuple[float, Dict[str, float], Fragment, Frame]] = []
    for perm in ((0, 1), (1, 0)):
        target_coords = stack_bp_blocks(src_blocks, perm)
        target_frame = endpoint_frame_with_reference(
            source_index, source_endpoint, mode="outward", ref_residue_index=perm[0], k=tangent_k
        )
        R0, t0 = frame_alignment_transform(template_frame, target_frame)
        moved0 = np.vstack([_transform_atom_coords_col(x, R0, t0) for x in frag_start_coords])
        phi, start_rmsd = twist_refine_about_axis(moved0, target_coords, target_frame.origin, target_frame.t)
        Rphi = axis_angle_matrix(target_frame.t, phi)

        transformed_atoms: List[AtomRecord] = []
        for atom in fragment.atoms:
            x = _transform_atom_coords_col(atom.coord, R0, t0)
            x = target_frame.origin + (Rphi @ (x - target_frame.origin))
            transformed_atoms.append(atom.copy_with_coord(x))

        t_aligned = normalize(Rphi @ (_apply_column_rotation(frag_inward_tangent, R0)))
        axis_dot = float(np.dot(t_aligned, target_frame.t))
        dir0 = normalize(Rphi @ (_apply_column_rotation(frag_local_dirs[0], R0)))
        dir1 = normalize(Rphi @ (_apply_column_rotation(frag_local_dirs[1], R0)))
        dir_dot0 = float(np.dot(dir0, src_local_dirs[perm[0]]))
        dir_dot1 = float(np.dot(dir1, src_local_dirs[perm[1]]))
        if min(axis_dot, dir_dot0, dir_dot1) < min_direction_dot:
            continue

        aligned_fragment = Fragment(
            atoms=transformed_atoms,
            pair_series=list(fragment.pair_series),
            start_endpoint=fragment.start_endpoint,
            end_endpoint=fragment.end_endpoint,
            old_to_new_resid=dict(fragment.old_to_new_resid),
            chain_order=fragment.chain_order,
            start_label=fragment.start_label,
        )

        n = normalize(Rphi @ (_apply_column_rotation(template_frame.n, R0)))
        b = normalize(Rphi @ (_apply_column_rotation(template_frame.b, R0)))
        t = normalize(Rphi @ (_apply_column_rotation(template_frame.t, R0)))
        aligned_frame = Frame(origin=target_frame.origin.copy(), n=n, b=b, t=t)

        meta = {
            "start_rmsd": float(start_rmsd),
            "start_twist_deg": float(math.degrees(phi)),
            "start_axis_dot": axis_dot,
            "start_dir_chain1_dot": dir_dot0,
            "start_dir_chain2_dot": dir_dot1,
            "start_dir_min_dot": float(min(dir_dot0, dir_dot1)),
            "start_match": _bp_match_labels(source_endpoint, perm),
        }
        candidates.append((float(start_rmsd), meta, aligned_fragment, aligned_frame))

    if not candidates:
        raise ValueError(
            "No valid source alignment remained after free matching and direction filtering."
        )

    candidates.sort(key=lambda item: (item[0], -item[1]["start_dir_min_dot"], -item[1]["start_axis_dot"]))
    _score, meta_best, frag_best, frame_best = candidates[0]
    return frag_best, meta_best, frame_best


def fragment_end_length(fragment: Fragment, start_frame: Frame) -> float:
    idx = fragment_index(fragment)
    c_end = bp_sugar_centroid(idx, fragment.end_endpoint)
    return float(np.dot(c_end - start_frame.origin, start_frame.t))


# ---------------------------------------------------------------------------
# Centerline / elastica proxy
# ---------------------------------------------------------------------------

def bezier_points(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    omt = 1.0 - t
    return (
        (omt ** 3)[:, None] * P0
        + (3.0 * omt * omt * t)[:, None] * P1
        + (3.0 * omt * t * t)[:, None] * P2
        + (t ** 3)[:, None] * P3
    )


def bezier_derivatives(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    omt = 1.0 - t
    d1 = (
        (3.0 * omt * omt)[:, None] * (P1 - P0)
        + (6.0 * omt * t)[:, None] * (P2 - P1)
        + (3.0 * t * t)[:, None] * (P3 - P2)
    )
    d2 = (6.0 * omt)[:, None] * (P2 - 2.0 * P1 + P0) + (6.0 * t)[:, None] * (P3 - 2.0 * P2 + P1)
    return d1, d2


def compute_arc_lengths(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.zeros(points.shape[0], dtype=float)
    s[1:] = np.cumsum(seg_lengths)
    return s


def resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        raise ValueError("Need at least two points to resample a polyline.")
    arc = compute_arc_lengths(points)
    total = arc[-1]
    if total < EPS:
        return np.repeat(points[:1], n_points, axis=0)
    q = np.linspace(0.0, total, n_points)
    out = np.zeros((n_points, 3), dtype=float)
    j = 0
    for i, qq in enumerate(q):
        while j < len(arc) - 2 and arc[j + 1] < qq:
            j += 1
        s0, s1 = arc[j], arc[j + 1]
        if s1 - s0 < EPS:
            out[i] = points[j]
        else:
            a = (qq - s0) / (s1 - s0)
            out[i] = (1.0 - a) * points[j] + a * points[j + 1]
    return out


def bezier_length_and_energy(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray, n_eval: int = 250) -> Tuple[float, float, np.ndarray]:
    t = np.linspace(0.0, 1.0, n_eval)
    pts = bezier_points(P0, P1, P2, P3, t)
    d1, d2 = bezier_derivatives(P0, P1, P2, P3, t)
    speed = np.linalg.norm(d1, axis=1)
    speed = np.maximum(speed, 1.0e-10)
    cross = np.cross(d1, d2)
    k2_ds_dt = np.sum(cross * cross, axis=1) / (speed ** 5)
    length = float(np.trapz(speed, t))
    energy = float(np.trapz(k2_ds_dt, t))
    return length, energy, pts


def solve_bezier_centerline(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    length_target: float,
    n_points: int = 240,
) -> Tuple[np.ndarray, Dict[str, float]]:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    t0 = normalize(t0)
    t1 = normalize(t1)
    chord = np.linalg.norm(p1 - p0)
    if length_target < chord - 1.0e-3:
        raise ValueError(
            f"Requested centerline length {length_target:.3f} Å is shorter than the endpoint distance {chord:.3f} Å"
        )

    a0 = max(chord / 3.0, min(length_target / 2.0, 8.0))
    b0 = a0

    def objective(log_ab: np.ndarray) -> float:
        a = float(np.exp(log_ab[0]))
        b = float(np.exp(log_ab[1]))
        P1 = p0 + a * t0
        P2 = p1 - b * t1
        length, energy, _ = bezier_length_and_energy(p0, P1, P2, p1)
        rel = (length - length_target) / max(length_target, 1.0)
        return energy + 2.0e4 * rel * rel

    if HAVE_SCIPY:
        res = minimize(
            objective,
            x0=np.log([a0, b0]),
            method="L-BFGS-B",
            bounds=[
                (math.log(1.0e-3), math.log(max(5.0 * length_target, 10.0))),
                (math.log(1.0e-3), math.log(max(5.0 * length_target, 10.0))),
            ],
            options={"maxiter": 200},
        )
        log_ab = res.x
    else:
        grid = np.linspace(math.log(0.5), math.log(max(length_target, 10.0)), 40)
        best = None
        best_val = float("inf")
        for ga in grid:
            for gb in grid:
                val = objective(np.array([ga, gb]))
                if val < best_val:
                    best_val = val
                    best = np.array([ga, gb])
        if best is None:
            raise RuntimeError("Could not optimize cubic centerline.")
        log_ab = best

    a = float(np.exp(log_ab[0]))
    b = float(np.exp(log_ab[1]))
    P1 = p0 + a * t0
    P2 = p1 - b * t1
    length, energy, dense = bezier_length_and_energy(p0, P1, P2, p1, n_eval=600)
    curve = resample_polyline(dense, n_points)
    meta = {
        "method": "bezier",
        "a": a,
        "b": b,
        "curve_length": float(compute_arc_lengths(curve)[-1]),
        "length_error": float(length - length_target),
        "bend_energy": float(energy),
    }
    return curve, meta


def refine_discrete_centerline(
    points_init: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    length_target: float,
    max_nfev: int = 80,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy is required for discrete centerline refinement.")

    pts0 = np.asarray(points_init, dtype=float)
    K = pts0.shape[0] - 1
    h = length_target / max(K, 1)
    x0 = pts0[1:-1].reshape(-1)

    def unpack(x: np.ndarray) -> np.ndarray:
        P = np.zeros((K + 1, 3), dtype=float)
        P[0] = p0
        P[-1] = p1
        P[1:-1] = x.reshape(-1, 3)
        return P

    def residuals(x: np.ndarray) -> np.ndarray:
        P = unpack(x)
        seg = P[1:] - P[:-1]
        lens = np.linalg.norm(seg, axis=1)
        lens = np.maximum(lens, 1.0e-8)
        T = seg / lens[:, None]
        curv = (P[2:] - 2.0 * P[1:-1] + P[:-2]) / max(h * h, 1.0e-8)
        r_curv = 0.25 * curv.reshape(-1)
        r_len = 40.0 * (lens - h)
        r_t0 = 18.0 * (T[0] - t0)
        r_t1 = 18.0 * (T[-1] - t1)
        return np.concatenate([r_curv, r_len, r_t0, r_t1])

    res = least_squares(residuals, x0, method="trf", max_nfev=max_nfev, verbose=0)
    P = unpack(res.x)
    arc = compute_arc_lengths(P)
    seg = P[1:] - P[:-1]
    lens = np.linalg.norm(seg, axis=1)
    T = seg / np.maximum(lens[:, None], 1.0e-8)
    bend = float(np.sum(np.linalg.norm(P[2:] - 2.0 * P[1:-1] + P[:-2], axis=1) ** 2))
    meta = {
        "method": "discrete",
        "curve_length": float(arc[-1]),
        "length_error": float(arc[-1] - length_target),
        "bend_energy": bend,
        "dot0": float(np.dot(T[0], t0)),
        "dot1": float(np.dot(T[-1], t1)),
        "opt_cost": float(res.cost),
        "opt_success": 1.0 if res.success else 0.0,
    }
    return P, meta


def solve_centerline(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    length_target: float,
    method: str = "auto",
    n_points: int = 240,
) -> Tuple[np.ndarray, Dict[str, float]]:
    method = method.lower()
    curve_bez, meta_bez = solve_bezier_centerline(p0, p1, t0, t1, length_target, n_points=n_points)
    if method in {"bezier", "auto"}:
        return curve_bez, meta_bez
    if method != "discrete":
        raise ValueError("centerline method must be one of: auto, discrete, bezier")
    if not HAVE_SCIPY:
        return curve_bez, meta_bez
    try:
        curve_dis, meta_dis = refine_discrete_centerline(curve_bez, p0, p1, t0, t1, length_target)
        return curve_dis, meta_dis
    except Exception:
        return curve_bez, meta_bez


def estimate_local_curvature(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n < 3:
        return np.zeros(n, dtype=float)
    kappa = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        a = pts[i] - pts[i - 1]
        b = pts[i + 1] - pts[i]
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        if la < 1.0e-10 or lb < 1.0e-10:
            continue
        ta = a / la
        tb = b / lb
        cosang = float(np.clip(np.dot(ta, tb), -1.0, 1.0))
        theta = math.acos(cosang)
        havg = 0.5 * (la + lb)
        if havg > 1.0e-10:
            kappa[i] = 2.0 * math.sin(0.5 * theta) / havg
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa

# ---------------------------------------------------------------------------
# Bending the template fragment onto the centerline
# ---------------------------------------------------------------------------

def compute_parallel_transport_frames(points: np.ndarray, init_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 points for frame construction.")
    T = np.zeros_like(pts)
    N = np.zeros_like(pts)
    B = np.zeros_like(pts)

    v0 = pts[1] - pts[0]
    t0 = normalize(v0)
    n0 = orthogonalize(init_normal, t0)
    if np.linalg.norm(n0) < 1.0e-8:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, t0)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        n0 = orthogonalize(tmp, t0)
    n0 = normalize(n0)
    b0 = normalize(np.cross(t0, n0))
    n0 = normalize(np.cross(b0, t0))
    T[0], N[0], B[0] = t0, n0, b0

    for i in range(1, pts.shape[0]):
        v = pts[i] - pts[i - 1]
        if np.linalg.norm(v) < 1.0e-10:
            T[i], N[i], B[i] = T[i - 1], N[i - 1], B[i - 1]
            continue
        ti = normalize(v)
        T[i] = ti
        ni = orthogonalize(N[i - 1], ti)
        if np.linalg.norm(ni) < 1.0e-8:
            ni = orthogonalize(B[i - 1], ti)
        if np.linalg.norm(ni) < 1.0e-8:
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, ti)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            ni = orthogonalize(tmp, ti)
        ni = normalize(ni)
        bi = normalize(np.cross(ti, ni))
        ni = normalize(np.cross(bi, ti))
        N[i], B[i] = ni, bi
    return N, B, T


def sample_curve_with_frame(
    points: np.ndarray,
    arc: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    T: np.ndarray,
    s_query: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = float(arc[-1])
    if s_query <= 0.0:
        pos = points[0] + s_query * T[0]
        return pos, N[0], B[0], T[0]
    if s_query >= total:
        pos = points[-1] + (s_query - total) * T[-1]
        return pos, N[-1], B[-1], T[-1]
    idx = int(np.searchsorted(arc, s_query) - 1)
    idx = max(0, min(idx, len(points) - 2))
    s0, s1 = arc[idx], arc[idx + 1]
    alpha = 0.0 if abs(s1 - s0) < 1.0e-12 else (s_query - s0) / (s1 - s0)
    pos = (1.0 - alpha) * points[idx] + alpha * points[idx + 1]
    t_interp = normalize((1.0 - alpha) * T[idx] + alpha * T[idx + 1])
    n_interp = orthogonalize((1.0 - alpha) * N[idx] + alpha * N[idx + 1], t_interp)
    if np.linalg.norm(n_interp) < 1.0e-10:
        n_interp = orthogonalize(N[idx], t_interp)
    n_interp = normalize(n_interp)
    b_interp = normalize(np.cross(t_interp, n_interp))
    n_interp = normalize(np.cross(b_interp, t_interp))
    return pos, n_interp, b_interp, t_interp


def bend_fragment_on_curve(fragment: Fragment, start_frame: Frame, curve_points: np.ndarray) -> List[AtomRecord]:
    arc = compute_arc_lengths(curve_points)
    Nf, Bf, Tf = compute_parallel_transport_frames(curve_points, start_frame.n)
    frag_idx = fragment_index(fragment)

    pair_centers: List[np.ndarray] = []
    pair_s: List[float] = []
    pair_atom_sets: List[List[int]] = []
    for bp in fragment.pair_series:
        center = bp_sugar_centroid(frag_idx, bp)
        pair_centers.append(center)
        pair_s.append(float(np.dot(center - start_frame.origin, start_frame.t)))
        keys = {(bp.chain1, bp.res1), (bp.chain2, bp.res2)}
        idxs = [j for j, atom in enumerate(fragment.atoms) if (atom.chain_id, atom.res_seq) in keys]
        pair_atom_sets.append(idxs)

    out_coords = [atom.coord.copy() for atom in fragment.atoms]
    for bp_idx, atom_idxs in enumerate(pair_atom_sets):
        center = pair_centers[bp_idx]
        s_bp = pair_s[bp_idx]
        pos, n, b, t = sample_curve_with_frame(curve_points, arc, Nf, Bf, Tf, float(s_bp))
        for j in atom_idxs:
            atom = fragment.atoms[j]
            d = atom.coord - center
            uu = float(np.dot(d, start_frame.n))
            vv = float(np.dot(d, start_frame.b))
            ww = float(np.dot(d, start_frame.t))
            out_coords[j] = pos + uu * n + vv * b + ww * t

    return [atom.copy_with_coord(coord) for atom, coord in zip(fragment.atoms, out_coords)]


# ---------------------------------------------------------------------------
# Relabeling, scoring, output
# ---------------------------------------------------------------------------

def relabel_connector_atoms(atoms: List[AtomRecord], fragment: Fragment, new_chain_ids: Tuple[str, str]) -> List[AtomRecord]:
    c1_old, c2_old = fragment.chain_order
    c1_new, c2_new = new_chain_ids
    remapped: List[AtomRecord] = []
    for atom in atoms:
        key = (atom.chain_id, atom.res_seq)
        if key not in fragment.old_to_new_resid:
            continue
        old_chain, new_res = fragment.old_to_new_resid[key]
        new_chain = c1_new if old_chain == c1_old else c2_new
        remapped.append(
            AtomRecord(
                serial=atom.serial,
                atom_name=atom.atom_name,
                atom_name_norm=atom.atom_name_norm,
                res_name=atom.res_name,
                chain_id=new_chain,
                res_seq=new_res,
                i_code="",
                coord=atom.coord.copy(),
                element=atom.element,
                occupancy=atom.occupancy,
                temp_factor=atom.temp_factor,
                original_line=atom.original_line,
            )
        )
    return remapped


def atom_sort_key(atom: AtomRecord) -> Tuple[str, int, str, int]:
    return (atom.chain_id, atom.res_seq, atom.atom_name, atom.serial)


def format_pdb_atom(serial: int, atom: AtomRecord) -> str:
    atom_name = atom.atom_name[:4]
    element = (atom.element or "")[:2].rjust(2)
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {atom.res_name:>3s} {atom.chain_id:1s}"
        f"{atom.res_seq:4d}{atom.i_code:1s}   "
        f"{atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}"
        f"{atom.occupancy:6.2f}{atom.temp_factor:6.2f}          {element:>2s}"
    )


def choose_unused_chain_ids(existing: Iterable[str], n: int = 2) -> List[str]:
    used = {c for c in existing if c}
    out: List[str] = []
    for c in CHAIN_ID_POOL:
        if c not in used:
            out.append(c)
            if len(out) == n:
                return out
    raise ValueError("Could not find enough unused chain IDs.")


def collect_target_atoms_for_output(atoms: List[AtomRecord]) -> List[AtomRecord]:
    return [a.copy_with_coord(a.coord.copy()) for a in atoms]


def score_connector_bp_free(
    connector_atoms: List[AtomRecord],
    connector_ep: EndpointSpec,
    target_index: StructureIndex,
    target_ep: EndpointSpec,
    tangent_k: int,
    min_direction_dot: float,
    connector_mode: str = "inward",
    target_mode: str = "inward",
) -> Dict[str, float]:
    conn_idx = build_structure_index(connector_atoms)
    conn_blocks = bp_sugar_blocks(conn_idx, connector_ep)
    targ_blocks = bp_sugar_blocks(target_index, target_ep)

    conn_cent = bp_sugar_centroid(conn_idx, connector_ep)
    targ_cent = bp_sugar_centroid(target_index, target_ep)
    cent_dist = float(np.linalg.norm(conn_cent - targ_cent))

    conn_t = endpoint_tangent(conn_idx, connector_ep, mode=connector_mode, k=tangent_k)
    targ_t = endpoint_tangent(target_index, target_ep, mode=target_mode, k=tangent_k)
    axis_dot = float(np.dot(conn_t, targ_t))

    conn_dirs = [
        residue_local_5to3_direction(conn_idx, connector_ep.chain1, connector_ep.res1),
        residue_local_5to3_direction(conn_idx, connector_ep.chain2, connector_ep.res2),
    ]
    targ_dirs = [
        residue_local_5to3_direction(target_index, target_ep.chain1, target_ep.res1),
        residue_local_5to3_direction(target_index, target_ep.chain2, target_ep.res2),
    ]
    conn_res_cent = [
        residue_sugar_centroid(conn_idx, connector_ep.chain1, connector_ep.res1),
        residue_sugar_centroid(conn_idx, connector_ep.chain2, connector_ep.res2),
    ]
    targ_res_cent = [
        residue_sugar_centroid(target_index, target_ep.chain1, target_ep.res1),
        residue_sugar_centroid(target_index, target_ep.chain2, target_ep.res2),
    ]

    candidates: List[Tuple[float, Dict[str, float]]] = []
    conn_coords = stack_bp_blocks(conn_blocks, (0, 1))
    for perm in ((0, 1), (1, 0)):
        targ_coords = stack_bp_blocks(targ_blocks, perm)
        rmsd = direct_rmsd(conn_coords, targ_coords)
        dir_dot0 = float(np.dot(conn_dirs[0], targ_dirs[perm[0]]))
        dir_dot1 = float(np.dot(conn_dirs[1], targ_dirs[perm[1]]))
        if min(axis_dot, dir_dot0, dir_dot1) < min_direction_dot:
            continue

        targ_ref = orthogonalize(targ_res_cent[perm[0]] - targ_cent, targ_t)
        if np.linalg.norm(targ_ref) < 1.0e-8:
            targ_ref = orthogonalize(targ_res_cent[perm[1]] - targ_cent, targ_t)
        conn_ref = orthogonalize(conn_res_cent[0] - conn_cent, targ_t)
        if np.linalg.norm(conn_ref) < 1.0e-8:
            conn_ref = orthogonalize(conn_res_cent[1] - conn_cent, targ_t)
        if np.linalg.norm(targ_ref) < 1.0e-8:
            targ_ref = np.array([1.0, 0.0, 0.0])
        if np.linalg.norm(conn_ref) < 1.0e-8:
            conn_ref = np.array([1.0, 0.0, 0.0])
        targ_ref = normalize(targ_ref)
        conn_ref = normalize(conn_ref)
        twist = math.degrees(
            math.atan2(
                np.dot(targ_t, np.cross(targ_ref, conn_ref)),
                np.dot(targ_ref, conn_ref),
            )
        )
        meta = {
            "rmsd": float(rmsd),
            "centroid_distance": cent_dist,
            "twist_mismatch_deg": float(twist),
            "end_axis_dot": axis_dot,
            "end_dir_chain1_dot": dir_dot0,
            "end_dir_chain2_dot": dir_dot1,
            "end_dir_min_dot": float(min(dir_dot0, dir_dot1)),
            "end_match": _bp_match_labels(target_ep, perm),
        }
        candidates.append((float(rmsd), meta))

    if not candidates:
        raise ValueError(
            "No valid destination alignment remained after free matching and direction filtering."
        )
    candidates.sort(key=lambda item: (item[0], abs(item[1]["twist_mismatch_deg"])))
    return candidates[0][1]


def write_assembly_pdb(path: str, target_atoms: List[AtomRecord], connector_atoms: List[AtomRecord]) -> None:
    serial = 1
    lines: List[str] = []
    last_chain = None
    for atom in target_atoms + connector_atoms:
        if last_chain is not None and atom.chain_id != last_chain:
            lines.append(f"TER   {serial:5d}")
            serial += 1
        lines.append(format_pdb_atom(serial, atom))
        serial += 1
        last_chain = atom.chain_id
    lines.append("END")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def summary_header() -> List[str]:
    return [
        "rank",
        "template_start_bp",
        "n_bp",
        "axis_length_A",
        "curve_method",
        "curve_length_A",
        "curve_length_error_A",
        "bend_energy",
        "max_local_curvature_Ainv",
        "start_rmsd_A",
        "start_twist_deg",
        "start_axis_dot",
        "start_dir_chain1_dot",
        "start_dir_chain2_dot",
        "start_dir_min_dot",
        "start_match",
        "end_rmsd_A",
        "end_centroid_distance_A",
        "twist_mismatch_deg",
        "end_axis_dot",
        "end_dir_chain1_dot",
        "end_dir_chain2_dot",
        "end_dir_min_dot",
        "end_match",
        "pdb_file",
    ]


def write_summary_tsv(path: str, rows: List[Dict[str, object]]) -> None:
    cols = summary_header()
    with open(path, "w") as handle:
        handle.write("\t".join(cols) + "\n")
        for i, row in enumerate(rows, start=1):
            values = [row.get(col, "") for col in cols]
            values[0] = i
            handle.write("\t".join(str(v) for v in values) + "\n")


# ---------------------------------------------------------------------------
# Screening workflow
# ---------------------------------------------------------------------------

def screen_connectors(args: argparse.Namespace, log: Callable[..., None] = eprint) -> int:
    target_atoms = parse_pdb_atoms(args.target_pdb)
    template_atoms = parse_pdb_atoms(args.template_pdb)
    target_idx = build_structure_index(target_atoms)
    template_idx = build_structure_index(template_atoms)

    source_ep = parse_endpoint_spec(args.source_bp)
    dest_ep = parse_endpoint_spec(args.dest_bp)
    validate_endpoint(target_idx, source_ep)
    validate_endpoint(target_idx, dest_ep)

    source_origin = bp_sugar_centroid(target_idx, source_ep)
    dest_origin = bp_sugar_centroid(target_idx, dest_ep)
    source_tangent = endpoint_tangent(target_idx, source_ep, mode="outward", k=args.tangent_k)
    dest_tangent = endpoint_tangent(target_idx, dest_ep, mode="inward", k=args.tangent_k)
    endpoint_distance = float(np.linalg.norm(dest_origin - source_origin))

    log(f"[INFO] Source endpoint : {source_ep.label()}")
    log(f"[INFO] Destination    : {dest_ep.label()}")
    log(f"[INFO] Endpoint distance (Å): {endpoint_distance:.3f}")
    log(f"[INFO] Tangent dot(source,dest): {float(np.dot(source_tangent, dest_tangent)):.3f}")

    template_label, template_start_ep = choose_canonical_template_start(template_idx)
    log(f"[INFO] Canonical template start bp: {template_label}")

    max_bp_available = max_n_bp_from_endpoint(template_idx, template_start_ep)
    max_bp = max_bp_available if args.max_bp is None else min(args.max_bp, max_bp_available)
    min_bp = max(2, args.min_bp)
    if min_bp > max_bp:
        log(f"[ERROR] Requested bp range {min_bp}-{max_bp} is empty for the template.")
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    target_atoms_out = collect_target_atoms_for_output(target_atoms)
    used_chain_ids = {a.chain_id for a in target_atoms_out}
    connector_chain_ids = tuple(choose_unused_chain_ids(used_chain_ids, n=2))

    results: List[Dict[str, object]] = []
    log(f"[INFO] Screening bp lengths {min_bp}..{max_bp} in the canonical template direction only")

    for n_bp in range(min_bp, max_bp + 1):
        try:
            fragment = extract_fragment(template_idx, template_start_ep, n_bp, start_label=template_label)
            aligned_fragment, align_meta, aligned_frame = align_fragment_to_source_free(
                fragment,
                target_idx,
                source_ep,
                tangent_k=args.tangent_k,
                min_direction_dot=args.min_direction_dot,
            )
            axis_length = fragment_end_length(aligned_fragment, aligned_frame)
            if axis_length <= 0.0:
                raise ValueError("Non-positive connector axis length after alignment.")
            if axis_length < endpoint_distance - args.length_slack:
                log(
                    f"[SKIP] n={n_bp:2d} axis_len={axis_length:7.3f} < endpoint distance {endpoint_distance:7.3f}"
                )
                continue

            curve_points, curve_meta = solve_centerline(
                source_origin,
                dest_origin,
                source_tangent,
                dest_tangent,
                axis_length,
                method=args.centerline_method,
                n_points=max(80, args.curve_points),
            )
            max_curv = float(np.max(estimate_local_curvature(curve_points)))

            bent_atoms = bend_fragment_on_curve(aligned_fragment, aligned_frame, curve_points)
            start_score = score_connector_bp_free(
                bent_atoms,
                aligned_fragment.start_endpoint,
                target_idx,
                source_ep,
                tangent_k=args.tangent_k,
                min_direction_dot=args.min_direction_dot,
                connector_mode="inward",
                target_mode="outward",
            )
            end_score = score_connector_bp_free(
                bent_atoms,
                aligned_fragment.end_endpoint,
                target_idx,
                dest_ep,
                tangent_k=args.tangent_k,
                min_direction_dot=args.min_direction_dot,
                connector_mode="outward",
                target_mode="inward",
            )
            bent_atoms_relab = relabel_connector_atoms(bent_atoms, aligned_fragment, connector_chain_ids)
            bent_atoms_relab.sort(key=atom_sort_key)

            result: Dict[str, object] = {
                "template_start_bp": template_label,
                "n_bp": n_bp,
                "axis_length_A": f"{axis_length:.3f}",
                "curve_method": str(curve_meta.get("method", args.centerline_method)),
                "curve_length_A": f"{float(curve_meta.get('curve_length', np.nan)):.3f}",
                "curve_length_error_A": f"{float(curve_meta.get('length_error', np.nan)):.3f}",
                "bend_energy": f"{float(curve_meta.get('bend_energy', np.nan)):.6f}",
                "max_local_curvature_Ainv": f"{max_curv:.6f}",
                "start_rmsd_A": f"{float(start_score['rmsd']):.3f}",
                "start_twist_deg": f"{float(align_meta['start_twist_deg']):.3f}",
                "start_axis_dot": f"{float(align_meta['start_axis_dot']):.4f}",
                "start_dir_chain1_dot": f"{float(align_meta['start_dir_chain1_dot']):.4f}",
                "start_dir_chain2_dot": f"{float(align_meta['start_dir_chain2_dot']):.4f}",
                "start_dir_min_dot": f"{float(align_meta['start_dir_min_dot']):.4f}",
                "start_match": str(align_meta['start_match']),
                "end_rmsd_A": f"{float(end_score['rmsd']):.3f}",
                "end_centroid_distance_A": f"{float(end_score['centroid_distance']):.3f}",
                "twist_mismatch_deg": f"{float(end_score['twist_mismatch_deg']):.3f}",
                "end_axis_dot": f"{float(end_score['end_axis_dot']):.4f}",
                "end_dir_chain1_dot": f"{float(end_score['end_dir_chain1_dot']):.4f}",
                "end_dir_chain2_dot": f"{float(end_score['end_dir_chain2_dot']):.4f}",
                "end_dir_min_dot": f"{float(end_score['end_dir_min_dot']):.4f}",
                "end_match": str(end_score['end_match']),
                "pdb_file": "",
                "_sort_end_rmsd": float(end_score['rmsd']),
                "_sort_twist_abs": abs(float(end_score['twist_mismatch_deg'])),
                "_connector_atoms": bent_atoms_relab,
            }
            results.append(result)
            log(
                f"[OK]   n={n_bp:2d}  L={axis_length:7.3f}  method={result['curve_method']:>8s}  "
                f"startRMSD={float(start_score['rmsd']):6.3f}  endRMSD={float(end_score['rmsd']):6.3f}  "
                f"twist_mismatch={float(end_score['twist_mismatch_deg']):7.3f}  kmax={max_curv:8.5f}"
            )
        except Exception as exc:
            log(f"[FAIL] n={n_bp:2d}  {exc}")

    if not results:
        log("[ERROR] No valid connector candidates were generated.")
        return 1

    results.sort(key=lambda row: (row["_sort_end_rmsd"], row["_sort_twist_abs"]))

    top_k = min(args.top_k, len(results))
    for rank, row in enumerate(results[:top_k], start=1):
        out_name = (
            f"rank{rank:02d}_{str(row['template_start_bp']).replace(',', '_')}_bp{int(row['n_bp']):02d}_"
            f"rmsd{float(row['_sort_end_rmsd']):.3f}.pdb"
        )
        out_path = os.path.join(args.outdir, out_name)
        write_assembly_pdb(out_path, target_atoms_out, row["_connector_atoms"])
        row["pdb_file"] = out_name

    summary_path = os.path.join(args.outdir, "connector_summary.tsv")
    write_summary_tsv(summary_path, results)

    log("\n[RESULT] Top candidates")
    for rank, row in enumerate(results[:top_k], start=1):
        log(
            f"  {rank:2d}. start={row['template_start_bp']:>10s}  n_bp={int(row['n_bp']):2d}  "
            f"endRMSD={float(row['_sort_end_rmsd']):6.3f} Å  twist_mismatch={float(row['twist_mismatch_deg']):7.3f} deg  "
            f"file={row['pdb_file']}"
        )
    log(f"[RESULT] Summary TSV: {summary_path}")
    return 0

# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------

def detect_chain_numbering_direction(index: StructureIndex, chain: str) -> str:
    order = infer_chain_5to3_order(index, chain)
    if len(order) < 2:
        return "singleton"
    return "increasing" if order[0] < order[-1] else "decreasing"


def suggest_terminal_basepairs(index: StructureIndex, max_suggestions: int = 24) -> List[Tuple[str, float]]:
    suggestions: List[Tuple[str, float]] = []
    chains = list(index.chains)
    termini: Dict[Tuple[str, str], int] = {}
    for chain in chains:
        five, three = chain_termini(index, chain)
        termini[(chain, "5'")] = five
        termini[(chain, "3'")] = three
    for i, c1 in enumerate(chains):
        for c2 in chains[i + 1:]:
            for e1, e2 in (("5'", "3'"), ("3'", "5'"), ("5'", "5'"), ("3'", "3'")):
                r1 = termini[(c1, e1)]
                r2 = termini[(c2, e2)]
                d = float(
                    np.linalg.norm(
                        residue_sugar_centroid(index, c1, r1) - residue_sugar_centroid(index, c2, r2)
                    )
                )
                suggestions.append((f"{c1}{r1},{c2}{r2}    [{e1} - {e2}]", d))
    suggestions.sort(key=lambda x: x[1])
    return suggestions[:max_suggestions]


def build_target_hints_text(target_pdb: str) -> str:
    atoms = parse_pdb_atoms(target_pdb)
    index = build_structure_index(atoms)
    lines: List[str] = []
    lines.append(f"Target PDB: {target_pdb}")
    lines.append("")
    lines.append("Detected chain termini (in inferred 5' -> 3' order):")
    for chain in index.chains:
        order = infer_chain_5to3_order(index, chain)
        direction = detect_chain_numbering_direction(index, chain)
        lines.append(
            f"  Chain {chain}: 5'={order[0]}  3'={order[-1]}  length={len(order)}  numbering={direction}"
        )
    lines.append("")
    lines.append("Likely end-base-pair hints (sorted by terminal sugar-centroid distance):")
    for label, dist in suggest_terminal_basepairs(index):
        lines.append(f"  {label}    distance={dist:.2f} Å")
    lines.append("")
    lines.append("Tip: for duplexes, the most useful terminal combinations are usually 5'-3' and 3'-5'.")
    return "\n".join(lines)


GUI_PARAM_HELP: Dict[str, str] = {
    "target_pdb": "Target PDB containing the helices to be connected. After loading it, the GUI shows likely helix-end residue hints.",
    "template_pdb": "Straight duplex template PDB used to cut candidate connector lengths before bending.",
    "source_bp": "First target end base-pair to attach the connector to, written like A33,B1.",
    "dest_bp": "Second target end base-pair to connect to, written like E1,F33.",
    "min_bp": "Smallest connector length, in base pairs, to screen from the template.",
    "max_bp": "Largest connector length to screen. Leave blank to use the full available template length.",
    "top_k": "How many best-ranked connector PDB files to write at the end.",
    "outdir": "Directory where the ranked PDBs and connector_summary.tsv will be saved.",
    "centerline_method": "Centerline generator: auto uses the robust cubic clamped-elastica proxy, bezier forces that cubic proxy, and discrete tries the refined discrete solver.",
    "curve_points": "Number of sampled points used to represent the centerline during bending.",
    "tangent_k": "How many inward base-pairs are used to estimate the local helix direction at each endpoint. Larger values average over a longer helical segment; 11 bp is about one turn for B-form DNA.",
    "length_slack": "Reject a candidate if its straightened axis is shorter than the endpoint distance by more than this tolerance in angstrom.",
    "min_direction_dot": "Minimum allowed dot product for strand and helix direction agreement. 0 rejects opposite directions.",
}


def namespace_to_cli_command(ns: argparse.Namespace, script_name: str = "curved_connectorV3_0.py") -> str:
    parts: List[str] = [
        "python",
        script_name,
        ns.target_pdb,
        ns.template_pdb,
        "--source-bp",
        ns.source_bp,
        "--dest-bp",
        ns.dest_bp,
        "--min-bp",
        str(ns.min_bp),
        "--top-k",
        str(ns.top_k),
        "--outdir",
        ns.outdir,
        "--centerline-method",
        ns.centerline_method,
        "--curve-points",
        str(ns.curve_points),
        "--tangent-k",
        str(ns.tangent_k),
        "--length-slack",
        str(ns.length_slack),
        "--min-direction-dot",
        str(ns.min_direction_dot),
    ]
    if ns.max_bp is not None:
        parts.extend(["--max-bp", str(ns.max_bp)])
    return " ".join(shlex.quote(str(part)) for part in parts)


def launch_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk
    except Exception as exc:
        eprint(f"[ERROR] GUI mode requires tkinter: {exc}")
        return 1

    root = tk.Tk()
    root.title(f"{TOOL_NAME} {TOOL_VERSION}")
    root.geometry("980x700")
    root.minsize(920, 620)
    set_optional_window_icon(root, tk, ["curved_connector_icon.png", "icon.png"], "_curved_connector_icon_image")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    vars_s: Dict[str, tk.StringVar] = {
        "target_pdb": tk.StringVar(value=""),
        "template_pdb": tk.StringVar(value=""),
        "source_bp": tk.StringVar(value="A33,B1"),
        "dest_bp": tk.StringVar(value="E1,F33"),
        "min_bp": tk.StringVar(value="2"),
        "max_bp": tk.StringVar(value=""),
        "top_k": tk.StringVar(value="5"),
        "outdir": tk.StringVar(value=DEFAULT_OUTDIR),
        "centerline_method": tk.StringVar(value="auto"),
        "curve_points": tk.StringVar(value="240"),
        "tangent_k": tk.StringVar(value="11"),
        "length_slack": tk.StringVar(value="0.25"),
        "min_direction_dot": tk.StringVar(value="0.0"),
    }

    main = ttk.Frame(root, padding=10)
    main.grid(row=0, column=0, sticky="nsew")
    main.columnconfigure(0, weight=1)
    main.rowconfigure(3, weight=1)

    intro = ttk.Label(
        main,
        text=(
            "Choose the target/template PDBs, fill in the end base-pairs and screening options, "
            "then click Run. Click any ? button for a short parameter explanation."
        ),
        justify="left",
        wraplength=940,
    )
    intro.grid(row=0, column=0, sticky="ew", pady=(0, 8))

    top = ttk.Frame(main)
    top.grid(row=1, column=0, sticky="ew")
    top.columnconfigure(0, weight=1)
    top.columnconfigure(1, weight=1)

    files_box = ttk.LabelFrame(top, text="Files and end base-pairs", padding=8)
    files_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    files_box.columnconfigure(1, weight=1)

    options_box = ttk.LabelFrame(top, text="Screening options", padding=8)
    options_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
    options_box.columnconfigure(1, weight=1)

    def show_help_dialog(label: str, key: str) -> None:
        messagebox.showinfo(label, GUI_PARAM_HELP[key])

    def add_row(
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        key: str,
        browse: bool = False,
        is_dir: bool = False,
        width: int = 32,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=3)
        entry = ttk.Entry(parent, textvariable=vars_s[key], width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=(0, 6), pady=3)
        col = 2
        if browse:
            def _choose() -> None:
                if is_dir:
                    path = filedialog.askdirectory()
                else:
                    path = filedialog.askopenfilename(
                        filetypes=[("PDB files", "*.pdb"), ("All files", "*")]
                    )
                if path:
                    vars_s[key].set(path)
                    if key == "target_pdb":
                        try:
                            hints = build_target_hints_text(path)
                            hints_text.delete("1.0", tk.END)
                            hints_text.insert(tk.END, hints)
                            notebook.select(hints_tab)
                        except Exception as exc:
                            messagebox.showerror("Target PDB parse error", str(exc))
            ttk.Button(parent, text="Browse", width=8, command=_choose).grid(
                row=row, column=2, sticky="w", padx=(0, 6), pady=3
            )
            col = 3
        ttk.Button(
            parent,
            text="?",
            width=3,
            command=lambda lbl=label, kk=key: show_help_dialog(lbl, kk),
        ).grid(row=row, column=col, sticky="w", pady=3)

    add_row(files_box, 0, "Target PDB", "target_pdb", browse=True, width=34)
    add_row(files_box, 1, "Template PDB", "template_pdb", browse=True, width=34)
    add_row(files_box, 2, "Source base-pair", "source_bp", width=18)
    add_row(files_box, 3, "Destination base-pair", "dest_bp", width=18)
    add_row(files_box, 4, "Output directory", "outdir", browse=True, is_dir=True, width=34)

    add_row(options_box, 0, "Min bp", "min_bp", width=12)
    add_row(options_box, 1, "Max bp", "max_bp", width=12)
    add_row(options_box, 2, "Top K", "top_k", width=12)
    add_row(options_box, 3, "Centerline method", "centerline_method", width=12)
    add_row(options_box, 4, "Curve points", "curve_points", width=12)
    add_row(options_box, 5, "Tangent k", "tangent_k", width=12)
    add_row(options_box, 6, "Length slack (Å)", "length_slack", width=12)
    add_row(options_box, 7, "Min direction dot", "min_direction_dot", width=12)

    buttons = ttk.Frame(main)
    buttons.grid(row=2, column=0, sticky="w", pady=(10, 0))

    notebook = ttk.Notebook(main)
    notebook.grid(row=3, column=0, sticky="nsew", pady=(10, 0))

    hints_tab = ttk.Frame(notebook, padding=4)
    hints_tab.rowconfigure(0, weight=1)
    hints_tab.columnconfigure(0, weight=1)
    hints_text = scrolledtext.ScrolledText(hints_tab, wrap="word", height=14)
    hints_text.grid(row=0, column=0, sticky="nsew")
    notebook.add(hints_tab, text="Target hints")

    log_tab = ttk.Frame(notebook, padding=4)
    log_tab.rowconfigure(0, weight=1)
    log_tab.columnconfigure(0, weight=1)
    log_text = scrolledtext.ScrolledText(log_tab, wrap="word", height=14)
    log_text.grid(row=0, column=0, sticky="nsew")
    notebook.add(log_tab, text="Run log")

    run_counter = [0]

    def gui_log(*parts: object) -> None:
        msg = " ".join(str(p) for p in parts)
        log_text.insert(tk.END, msg + "\n")
        log_text.see(tk.END)
        notebook.select(log_tab)
        root.update_idletasks()

    def load_target_hints() -> None:
        target_pdb = vars_s["target_pdb"].get().strip()
        if not target_pdb:
            messagebox.showwarning("Missing target PDB", "Please choose a target PDB first.")
            return
        try:
            hints = build_target_hints_text(target_pdb)
            hints_text.delete("1.0", tk.END)
            hints_text.insert(tk.END, hints)
            notebook.select(hints_tab)
        except Exception as exc:
            messagebox.showerror("Target PDB parse error", str(exc))

    def run_now() -> None:
        try:
            target_pdb = vars_s["target_pdb"].get().strip()
            template_pdb = vars_s["template_pdb"].get().strip()
            if not target_pdb or not template_pdb:
                raise ValueError("Please choose both the target PDB and the template PDB.")
            ns = argparse.Namespace(
                target_pdb=target_pdb,
                template_pdb=template_pdb,
                source_bp=vars_s["source_bp"].get().strip(),
                dest_bp=vars_s["dest_bp"].get().strip(),
                min_bp=int(vars_s["min_bp"].get().strip()),
                max_bp=None if not vars_s["max_bp"].get().strip() else int(vars_s["max_bp"].get().strip()),
                top_k=int(vars_s["top_k"].get().strip()),
                outdir=vars_s["outdir"].get().strip(),
                centerline_method=vars_s["centerline_method"].get().strip().lower(),
                curve_points=int(vars_s["curve_points"].get().strip()),
                tangent_k=int(vars_s["tangent_k"].get().strip()),
                length_slack=float(vars_s["length_slack"].get().strip()),
                min_direction_dot=float(vars_s["min_direction_dot"].get().strip()),
            )
            run_counter[0] += 1
            gui_log("")
            gui_log("=" * 88)
            gui_log(f"[GUI] Run {run_counter[0]}")
            gui_log(f"[GUI] CLI: {namespace_to_cli_command(ns)}")
            rc = screen_connectors(ns, log=gui_log)
            if rc == 0:
                messagebox.showinfo("Done", f"Finished. Results were written to:\n{ns.outdir}")
            else:
                messagebox.showwarning("Finished with no valid result", f"Return code: {rc}")
        except Exception as exc:
            messagebox.showerror("Run failed", str(exc))

    ttk.Button(buttons, text="Load target hints", command=load_target_hints).pack(side="left", padx=(0, 6))
    ttk.Button(buttons, text="Run", command=run_now).pack(side="left", padx=(0, 6))
    ttk.Button(buttons, text="Quit", command=root.destroy).pack(side="left")

    root.mainloop()
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Screen curved nucleic-acid connectors between two target helical end base-pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("target_pdb", help="PDB with many helices.")
    p.add_argument("template_pdb", help="PDB with one straight duplex helix to use as the connector template.")
    p.add_argument("--source-bp", required=True, help="Source end base-pair in the target PDB, e.g. A33,B1")
    p.add_argument("--dest-bp", required=True, help="Destination end base-pair in the target PDB, e.g. E1,F33")
    p.add_argument("--min-bp", type=int, default=2, help="Minimum connector length to screen (bp)")
    p.add_argument("--max-bp", type=int, default=None, help="Maximum connector length to screen (bp)")
    p.add_argument("--top-k", type=int, default=5, help="How many top-ranked PDBs to write")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    p.add_argument(
        "--centerline-method",
        choices=["auto", "discrete", "bezier"],
        default="auto",
        help="Centerline generator: auto uses the robust cubic clamped-elastica proxy; discrete tries a SciPy-refined centerline.",
    )
    p.add_argument("--curve-points", type=int, default=240, help="Number of centerline points used for bending")
    p.add_argument("--tangent-k", type=int, default=11, help="How many inward base-pairs to use when estimating endpoint tangents")
    p.add_argument(
        "--length-slack",
        type=float,
        default=0.25,
        help="Skip candidates whose straightened axis length is smaller than the endpoint distance by more than this slack (Å).",
    )
    p.add_argument(
        "--min-direction-dot",
        type=float,
        default=0.0,
        help="Minimum allowed dot product for strand-direction and helix-direction agreement. 0 rejects opposite directions.",
    )
    p.add_argument("--gui", action="store_true", help="Launch the GUI.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or "--gui" in argv:
        return launch_gui()
    parser = build_argparser()
    args = parser.parse_args(argv)
    return screen_connectors(args)


if __name__ == "__main__":
    raise SystemExit(main())
