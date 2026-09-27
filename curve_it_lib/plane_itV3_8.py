#!/usr/bin/env python3
"""
plane_itV3_8.py

Plane It projects selected atoms/points from a PDB or XYZ file into 2D and writes an SVG.

Versioned implementation: plane_itV3_8.py
User-facing launcher: plane_it.py

Inputs:
  - PDB file, standard molecular XYZ file, or coordinate-only XYZ file
  - one or more atom selectors, usually PDB atom names such as P, CA, C1', O3'
  - optional per-atom-type drawing controls
  - optional per-atom-type straight-line or smooth-curve neighbor connections
  - optional DSSR-derived base-pair interaction lines
  - optional projection mode:
      pca        : project onto the first two PCA directions of the selected atoms
      current-xy : use the current input X/Y coordinates directly, without PCA fitting
  - optional pre-projection flip about the Y axis, x -> -x and z -> -z

Outputs:
  - SVG projection file, default: <input_stem>_<atom_types>_projection.svg
  - JSON file with projection metadata, default: <input_stem>_<atom_types>_plane.json
  - Optional CSV coordinate table if --csv-output is provided
  - Optional projection-basis PDB/XYZ/text file if --write-pca-pdb or --write-projection-basis is provided; PDB input writes PDB, and XYZ/coordinate text keeps the original coordinate-row layout

Examples:
  python plane_it.py input.pdb --atom-type P
  python plane_it.py input.pdb --atom-types P,C1' --draw-lines
  python plane_it.py input.pdb --atom-type P --draw-lines --connection-mode smooth
  python plane_it.py input.pdb --atom-type P --style "P draw_lines=true connection_mode=straight line_width=1.5"
  python plane_it.py input.pdb --atom-type P --draw-lines --closed-chains A,H
  python plane_it.py input.pdb --atom-type P --projection-mode current-xy
  python plane_it.py input.pdb --atom-type P --color-by chain
  python plane_it.py input.pdb --atom-type P --color-by chain --palette DiLiuLab-T80
  python plane_it.py input.pdb --atom-type P --color-by chain --chain-colors "A=RedT80,B=#4c79e6"
  python plane_it.py input.pdb --atom-types "P,C1'" --style "C1' fill=MintGreenT80" --xy-plane-fill GrayT20
  python plane_it.py input.pdb --atom-type P --draw-lines --depth-order-lines --line-underlay
  python plane_it.py input.pdb --atom-type P --style "P draw_lines=true extend_3prime=true"
  python plane_it.py input.pdb --atom-type P --flip-about-y --write-projection-basis
  python plane_it.py input.pdb --atom-type P --draw-base-pairs
  python plane_it.py input.pdb --atom-type P --draw-base-pairs --base-pair-atom "C4'"
  python plane_it.py input.pdb --atom-type P --draw-xy-plane --depth-order-circles
  python plane_it.py --gui

GUI behavior:
  If the script is run with no arguments, or with --gui, a Tkinter GUI opens.

Color names:
  Every color option (fill, stroke and line_stroke in --style, --chain-colors,
  --xy-plane-fill, --scale-bar-stroke, ...) takes a hex code, an SVG color name,
  or a DiLiuLab color by name with its tint, as the lab's color lists spell it:
  RedT80, MintGreenT60, GrayT40. A name is turned into its #rrggbb before
  anything is written, so no DiLiuLab name reaches the SVG; the bare name red
  stays SVG's #ff0000. The palette's own name, DiLiuLab or DiLiuLab-T80, names
  nine colors rather than one and is refused where one color is expected.

PCA convention:
  In PCA mode, the plane is spanned by the first two principal components of
  the selected atom coordinates. This is a covariance/least-squares projection,
  not a true convex-hull-area maximization algorithm.

Current-XY convention:
  In current-xy mode, no PCA operation is performed. The script uses the input X
  coordinate as proj_x, the input Y coordinate as proj_y, and the input Z coordinate
  as the depth value for optional SVG front/back ordering. If --flip-about-y is used,
  current-xy mode uses x -> -x and z -> -z before projection.

DSSR base-pair convention:
  If --draw-base-pairs is used, the script reads x3dna-dssr output from the
  default path <input_folder>/tmp_file/<input_filename>.out. If that file is
  missing, the script tries to run:

      x3dna-dssr -i=<input.pdb> --more -o=<that tmp_file output>

  DSSR is launched with <input_folder>/tmp_file as its working directory so
  any sidecar files produced by x3dna-dssr stay with the DSSR output.

  Each base-pair line connects the selected projected anchor atom of the two
  DSSR-listed residues. The default anchor is C3', recommended for B-DNA.
  C4' is recommended for A-RNA.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import html
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Optional Geomview VECT support.  VECT is the one coordinate format that
# states per component whether it is a closed loop, which is exactly what
# --closed-chains needs and what a PDB gets from its LINK records.
# plane_it.py loads this file by path, so the package import comes last.
try:
    from . import vect_io
except ImportError:
    try:
        import vect_io  # type: ignore[no-redef]
    except ImportError:
        try:
            from curve_it_lib import vect_io  # type: ignore[no-redef]
        except ImportError:
            vect_io = None  # type: ignore[assignment]

# Optional DiLiuLab palette for --palette; lab_colors.py reads it from
# assets/diliulab_colors.json.  plane_it.py loads this file by path, with the
# repository rather than curve_it_lib on sys.path, so the package import is
# tried last.  Without lab_colors only the default palette is offered.
try:
    from . import lab_colors
except ImportError:
    try:
        import lab_colors  # type: ignore[no-redef]
    except ImportError:
        try:
            from curve_it_lib import lab_colors  # type: ignore[no-redef]
        except ImportError:
            lab_colors = None  # type: ignore[assignment]


TOOL_NAME = "Plane It"
TOOL_VERSION = "V3.8"


def resource_path(relative_path: str) -> Path:
    """Return a resource path that also works from a PyInstaller bundle."""
    source_dir = Path(__file__).resolve().parent
    source_root = source_dir.parent if source_dir.name == "curve_it_lib" else source_dir
    base_dir = Path(getattr(sys, "_MEIPASS", source_root))
    return base_dir / relative_path


TWO_LETTER_ELEMENTS = {
    "AC", "AG", "AL", "AM", "AR", "AS", "AT", "AU", "BA", "BE", "BH",
    "BI", "BK", "BR", "CA", "CD", "CE", "CF", "CL", "CM", "CN", "CO",
    "CR", "CS", "CU", "DB", "DS", "DY", "ER", "ES", "EU", "FE", "FL",
    "FM", "FR", "GA", "GD", "GE", "HE", "HF", "HG", "HO", "HS", "IN",
    "IR", "KR", "LA", "LI", "LR", "LU", "LV", "MC", "MD", "MG", "MN",
    "MO", "MT", "NA", "NB", "ND", "NE", "NH", "NI", "NO", "NP", "OG",
    "OS", "PB", "PD", "PM", "PO", "PR", "PT", "PU", "RB", "RE", "RF",
    "RG", "RH", "RN", "RU", "SB", "SC", "SE", "SG", "SI", "SM", "SN",
    "SR", "TA", "TB", "TC", "TE", "TH", "TI", "TL", "TM", "TS", "XE",
    "YB", "ZN", "ZR",
}

DEFAULT_COLOR_SATURATION = 0.67
DEFAULT_COLOR_VALUE = 0.90
GOLDEN_RATIO_CONJUGATE = 0.618033988749895
DEFAULT_PALETTE = "default"
COLOR_STRIP_MAX = 12  # the most swatches the GUI shows beside the Palette, as many as fit, before "+N more"
XY_PLANE_MARGIN_FRACTION = 0.08
DEFAULT_SCALE_BAR_LENGTH = 10.0
DEFAULT_SCALE_BAR_UNIT_LABEL = "\u00c5"
DEFAULT_SCALE_BAR_STROKE = "#111827"
DEFAULT_SCALE_BAR_STROKE_WIDTH = 2.5
DEFAULT_SCALE_BAR_TEXT_SIZE = 14.0
DEFAULT_SCALE_BAR_MARGIN = 32.0
DEFAULT_SCALE_BAR_BACKGROUND = "#ffffff"
DEFAULT_SCALE_BAR_BACKGROUND_OPACITY = 0.78


@dataclass
class AtomRecord:
    model: int
    record: str
    serial: str
    atom_name: str
    element: str
    altloc: str
    resname: str
    chain: str
    resseq: str
    icode: str
    x: float
    y: float
    z: float
    line_number: int


@dataclass
class SelectedAtom:
    atom: AtomRecord
    atom_type: str


@dataclass
class AtomStyle:
    fill: str
    stroke: str
    stroke_width: float
    radius: float
    opacity: float
    draw_lines: bool
    connection_mode: str
    line_stroke: str
    line_width: float
    line_opacity: float
    extend_3prime: bool




@dataclass
class DssrBasePair:
    index: int
    nt1: str
    nt2: str
    bp: str
    name: str
    saenger: str
    lw: str
    dssr: str
    source_line: str


@dataclass(frozen=True)
class ResidueKey:
    model: int
    chain: str
    resseq: str
    icode: str


@dataclass
class BasePairDrawable:
    base_pair: DssrBasePair
    key1: ResidueKey
    key2: ResidueKey
    x1: float
    y1: float
    depth1: float
    x2: float
    y2: float
    depth2: float
    segment_line1: int
    segment_line2: int
    skipped_reason: str = ""


@dataclass
class ConnectionSegment:
    entries: Sequence[Tuple[SelectedAtom, float, float, float]]
    index: int
    closed: bool
    connection_mode: str
    terminal_extension_point: Optional[Tuple[float, float, float, int, str]] = None


@dataclass
class TerminalExtensionSegment:
    start_entry: Tuple[SelectedAtom, float, float, float]
    previous_entry: Optional[Tuple[SelectedAtom, float, float, float]]
    end_x: float
    end_y: float
    end_depth: float
    end_line_number: int
    end_atom_name: str
    connection_mode: str


@dataclass
class ProjectionResult:
    mode: str
    centroid: np.ndarray
    basis_x: np.ndarray
    basis_y: np.ndarray
    normal: np.ndarray
    plane_coefficients: np.ndarray
    projected_xy: np.ndarray
    projected_depth: np.ndarray
    eigenvalues: Optional[np.ndarray]
    explained_variance_ratio: Optional[np.ndarray]
    rank: Optional[int]
    method_name: str
    method_description: str
    pre_flip_about_y: bool = False


def _safe_slice(line: str, start: int, end: int) -> str:
    return line[start:end] if len(line) >= start else ""


def infer_element_from_atom_field(raw_atom_field: str) -> str:
    raw = raw_atom_field.rstrip("\n")
    if not raw.strip():
        return ""
    if len(raw) >= 2 and raw[0] == " " and raw[1].isalpha():
        return raw[1].upper()
    letters = "".join(ch for ch in raw.strip() if ch.isalpha())
    if not letters:
        return ""
    first_two = letters[:2].upper()
    if len(first_two) == 2 and first_two in TWO_LETTER_ELEMENTS:
        return first_two
    return letters[0].upper()


def parse_model_number(line: str, fallback: int) -> int:
    text = line[10:14].strip() if len(line) >= 14 else ""
    if text:
        try:
            return int(text)
        except ValueError:
            pass
    return fallback


def altloc_matches(altloc_value: str, requested_altloc: str) -> bool:
    requested = requested_altloc.strip()
    altloc = altloc_value.strip()
    if requested.lower() == "all":
        return True
    if requested == "":
        return altloc == ""
    return altloc == "" or altloc.upper() == requested.upper()


def parse_pdb_atoms(
    pdb_file: Path,
    records: str,
    model: str,
    chain: Optional[str],
    resname: Optional[str],
    altloc: str,
) -> List[AtomRecord]:
    allowed_records = {"ATOM", "HETATM"} if records == "all" else {records.upper()}

    if model.lower() in {"all", "first"}:
        requested_model: Optional[int] = None
    else:
        try:
            requested_model = int(model)
        except ValueError as exc:
            raise ValueError("--model must be 'first', 'all', or an integer model number") from exc

    atoms: List[AtomRecord] = []
    current_model: Optional[int] = None
    next_model_fallback = 1
    first_model_seen: Optional[int] = None
    implicit_model = 1

    with pdb_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            rec = line[0:6].strip().upper()
            if rec == "MODEL":
                current_model = parse_model_number(line, next_model_fallback)
                next_model_fallback = current_model + 1
                if first_model_seen is None:
                    first_model_seen = current_model
                continue
            if rec == "ENDMDL":
                current_model = None
                continue
            if rec not in allowed_records:
                continue

            atom_model = current_model if current_model is not None else implicit_model
            first_model_for_filter = first_model_seen if first_model_seen is not None else implicit_model
            if model.lower() == "first" and atom_model != first_model_for_filter:
                continue
            if requested_model is not None and atom_model != requested_model:
                continue

            raw_atom_name = _safe_slice(line, 12, 16)
            atom_name = raw_atom_name.strip()
            altloc_value = _safe_slice(line, 16, 17).strip()
            atom_resname = _safe_slice(line, 17, 20).strip()
            atom_chain = _safe_slice(line, 21, 22).strip()
            atom_resseq = _safe_slice(line, 22, 26).strip()
            atom_icode = _safe_slice(line, 26, 27).strip()
            serial = _safe_slice(line, 6, 11).strip()

            if chain is not None and atom_chain != chain:
                continue
            if resname is not None and atom_resname.upper() != resname.upper():
                continue
            if not altloc_matches(altloc_value, altloc):
                continue

            try:
                x = float(_safe_slice(line, 30, 38).strip())
                y = float(_safe_slice(line, 38, 46).strip())
                z = float(_safe_slice(line, 46, 54).strip())
            except ValueError:
                continue

            element = _safe_slice(line, 76, 78).strip().upper()
            if not element:
                element = infer_element_from_atom_field(raw_atom_name)

            atoms.append(
                AtomRecord(
                    model=atom_model,
                    record=rec,
                    serial=serial,
                    atom_name=atom_name,
                    element=element,
                    altloc=altloc_value,
                    resname=atom_resname,
                    chain=atom_chain,
                    resseq=atom_resseq,
                    icode=atom_icode,
                    x=x,
                    y=y,
                    z=z,
                    line_number=line_number,
                )
            )
    return atoms



def _is_float_token(text: str) -> bool:
    try:
        float(str(text))
        return True
    except ValueError:
        return False


def infer_element_from_xyz_label(label: str) -> str:
    """Infer a chemical element-like label from the first column of an XYZ file."""
    letters = "".join(ch for ch in str(label).strip() if ch.isalpha())
    if not letters:
        return "X"
    first_two = letters[:2].upper()
    if len(first_two) == 2 and first_two in TWO_LETTER_ELEMENTS:
        return first_two
    return letters[0].upper()


def split_xyz_line(line: str) -> List[str]:
    """Split an XYZ/coordinate line, allowing commas and # comments."""
    text = line.split("#", 1)[0].strip()
    if not text:
        return []
    if "," in text:
        return [part.strip() for part in next(csv.reader([text])) if part.strip()]
    return [part.strip() for part in re.split(r"\s+", text) if part.strip()]


def chain_id_from_index(index: int) -> str:
    """Return spreadsheet-style chain labels: A..Z, AA..AZ, BA..."""
    if index < 0:
        index = 0
    letters = []
    value = index
    while True:
        letters.append(chr(ord("A") + (value % 26)))
        value = value // 26 - 1
        if value < 0:
            break
    return "".join(reversed(letters))


def resolve_input_format(input_file: Path, requested_format: str = "auto") -> str:
    """Return 'pdb' or 'xyz' for the input file.

    XYZ includes both standard molecular XYZ files and coordinate-only XYZ files.
    """
    requested = (requested_format or "auto").strip().lower().replace("_", "-")
    aliases = {
        "pdb": "pdb",
        "ent": "pdb",
        "xyz": "xyz",
        "molecular-xyz": "xyz",
        "molecular_xyz": "xyz",
        "coordinate-xyz": "xyz",
        "coordinate_xyz": "xyz",
        "coord-xyz": "xyz",
        "coord_xyz": "xyz",
        "vect": "vect",
        "geomview": "vect",
        "geomview-vect": "vect",
        "geomview_vect": "vect",
        "auto": "auto",
    }
    if requested not in aliases:
        raise ValueError("--input-format must be auto, pdb, xyz, molecular-xyz, coordinate-xyz, or vect")
    requested = aliases[requested]
    if requested != "auto":
        return requested

    suffix = input_file.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return "pdb"

    # VECT is checked by content before any extension is trusted, because a
    # VECT file read as coordinates yields its counts line and its trailing
    # colours as if they were atoms -- a wrong answer rather than an error.
    if file_is_vect(input_file):
        return "vect"
    if suffix == ".xyz":
        return "xyz"

    # Content fallback for extensionless files.
    try:
        with input_file.open("r", encoding="utf-8", errors="replace") as handle:
            for _ in range(40):
                line = handle.readline()
                if not line:
                    break
                rec = line[0:6].strip().upper()
                if rec in {"ATOM", "HETATM", "MODEL", "LINK"}:
                    return "pdb"
    except OSError:
        pass
    return "xyz"


def file_is_vect(input_file: Path) -> bool:
    """True when this file is a Geomview VECT file, by its header word."""
    if vect_io is None:
        return False
    try:
        with input_file.open("r", encoding="utf-8", errors="replace") as handle:
            return vect_io.looks_like_vect(handle.read(4096))
    except OSError:
        return False


def parse_vect_atoms(vect_file: Path) -> Tuple[List[AtomRecord], List[bool]]:
    """Parse a Geomview VECT file into AtomRecords plus per-chain closure.

    Each polyline becomes one chain, labelled A, B, C... in file order, exactly
    as blank-line-separated components do in a coordinate XYZ file. The second
    return value is the file's own statement of which of those chains are
    closed loops, which is what --closed-chains would otherwise have to guess.
    """
    if vect_io is None:
        raise ValueError(
            "{0} is a Geomview VECT file, which needs vect_io.py from "
            "curve_it_lib. Make sure that file sits beside this one.".format(vect_file))
    text = vect_file.read_text(encoding="utf-8", errors="replace")
    curves = vect_io.read_vect_text(text, source=str(vect_file))

    atoms: List[AtomRecord] = []
    serial = 1
    for component_index, points in enumerate(curves.components):
        chain = chain_id_from_index(component_index)
        for x, y, z in points:
            atoms.append(
                AtomRecord(
                    model=1,
                    record="XYZ",
                    serial=str(serial),
                    atom_name="X",
                    element="X",
                    altloc="",
                    resname="XYZ",
                    chain=chain,
                    resseq=str(serial),
                    icode="",
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    line_number=serial,
                )
            )
            serial += 1
    if not atoms:
        raise ValueError("No coordinates could be parsed from: {0}".format(vect_file))
    return atoms, list(curves.closed)


def parse_xyz_atoms(xyz_file: Path, input_format: str = "auto") -> List[AtomRecord]:
    """Parse molecular XYZ or coordinate-only XYZ into AtomRecord objects.

    Supported coordinate line forms after optional XYZ count/comment lines:
      C 1.0 2.0 3.0       molecular XYZ
      1.0 2.0 3.0         coordinate-only XYZ
      C,1.0,2.0,3.0       comma-separated molecular XYZ
      1.0,2.0,3.0         comma-separated coordinate XYZ

    For coordinate-only input, atom_name/element are assigned as 'X'. Empty
    lines separate components, and components are assigned chain labels A, B,
    C, ... in input order. This also works for .txt files containing XYZ-like
    coordinates.
    """
    raw_lines = xyz_file.read_text(encoding="utf-8", errors="replace").splitlines()
    nonempty_indices = [i for i, line in enumerate(raw_lines) if line.strip() and not line.lstrip().startswith("#")]
    if not nonempty_indices:
        raise ValueError("XYZ input file is empty: {0}".format(xyz_file))

    requested = (input_format or "auto").strip().lower().replace("_", "-")
    start_index = nonempty_indices[0]
    expected_count: Optional[int] = None
    first_tokens = split_xyz_line(raw_lines[start_index])
    if requested in {"auto", "xyz", "molecular-xyz"} and len(first_tokens) == 1:
        try:
            expected_count = int(first_tokens[0])
            start_index += 2  # standard XYZ has one comment line after atom count
        except ValueError:
            expected_count = None

    atoms: List[AtomRecord] = []
    serial = 1
    component_index = 0
    current_component_has_atoms = False

    for line_number, line in enumerate(raw_lines[start_index:], start=start_index + 1):
        tokens = split_xyz_line(line)
        if not tokens:
            # A truly blank line separates coordinate components. Comment-only
            # lines are ignored without starting a new chain/component.
            if not line.strip() and current_component_has_atoms:
                component_index += 1
                current_component_has_atoms = False
            continue

        label = "X"
        coord_tokens: Optional[List[str]] = None
        if len(tokens) >= 3 and all(_is_float_token(tok) for tok in tokens[:3]):
            coord_tokens = tokens[:3]
            label = "X"
        elif len(tokens) >= 4 and all(_is_float_token(tok) for tok in tokens[1:4]):
            label = tokens[0]
            coord_tokens = tokens[1:4]
        if coord_tokens is None:
            continue
        try:
            x, y, z = (float(coord_tokens[0]), float(coord_tokens[1]), float(coord_tokens[2]))
        except ValueError:
            continue

        element = infer_element_from_xyz_label(label)
        atom_name = label.strip() if label.strip() else element
        chain = chain_id_from_index(component_index)
        atoms.append(
            AtomRecord(
                model=1,
                record="XYZ",
                serial=str(serial),
                atom_name=atom_name,
                element=element,
                altloc="",
                resname="XYZ",
                chain=chain,
                resseq=str(serial),
                icode="",
                x=x,
                y=y,
                z=z,
                line_number=line_number,
            )
        )
        serial += 1
        current_component_has_atoms = True
        if expected_count is not None and len(atoms) >= expected_count:
            break

    if expected_count is not None and len(atoms) != expected_count:
        raise ValueError(
            "Standard XYZ header expected {0} atoms, but parsed {1} coordinate line(s) from {2}".format(
                expected_count, len(atoms), xyz_file
            )
        )
    if not atoms:
        raise ValueError("No XYZ coordinates could be parsed from: {0}".format(xyz_file))
    return atoms


def parse_structure_atoms(input_file: Path, args: argparse.Namespace) -> Tuple[List[AtomRecord], str]:
    """Parse PDB or XYZ input and return (atoms, resolved_input_format)."""
    input_format = resolve_input_format(input_file, getattr(args, "input_format", "auto"))
    if input_format == "pdb":
        atoms = parse_pdb_atoms(
            pdb_file=input_file,
            records=args.records,
            model=args.model,
            chain=args.chain,
            resname=args.resname,
            altloc=args.altloc,
        )
    elif input_format == "vect":
        atoms, stated_closure = parse_vect_atoms(input_file)
        # Stash it for the caller: this is the VECT counterpart of the LINK
        # records a PDB uses to say which chains close.
        args.vect_closed_chains = [chain_id_from_index(i)
                                   for i, closed in enumerate(stated_closure) if closed]
    else:
        atoms = parse_xyz_atoms(input_file, getattr(args, "input_format", "auto"))
    return atoms, input_format


def atom_matches(atom: AtomRecord, atom_type: str, select_by: str) -> bool:
    query = atom_type.strip().upper()
    if query in {"ALL", "*", "ANY"}:
        return True
    atom_name = atom.atom_name.strip().upper()
    element = atom.element.strip().upper()
    if select_by == "name":
        return atom_name == query
    if select_by == "element":
        return element == query
    if select_by == "auto":
        return atom_name == query or element == query
    raise ValueError("Unknown select_by mode: {0}".format(select_by))


def matching_atom_type(atom: AtomRecord, atom_types: Sequence[str], select_by: str) -> Optional[str]:
    for atom_type in atom_types:
        if atom_matches(atom, atom_type, select_by):
            return atom_type
    return None


def select_atoms(atoms: Sequence[AtomRecord], atom_types: Sequence[str], select_by: str) -> List[SelectedAtom]:
    selected: List[SelectedAtom] = []
    for atom in atoms:
        atom_type = matching_atom_type(atom, atom_types, select_by)
        if atom_type is not None:
            selected.append(SelectedAtom(atom=atom, atom_type=atom_type))
    return selected


def orient_vector_stably(vec: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(np.abs(vec)))
    if vec[idx] < 0:
        return -vec
    return vec


def apply_pre_projection_transform(points: np.ndarray, flip_about_y: bool = False) -> np.ndarray:
    """Apply optional coordinate transforms before projection calculations.

    flip_about_y applies a 180-degree flip/rotation about the Y axis:
    x -> -x, y -> y, z -> -z. This is useful when the desired 2D image
    should be mirrored before PCA or current-XY projection.
    """
    transformed = np.array(points, dtype=float, copy=True)
    if flip_about_y:
        transformed[:, 0] *= -1.0
        transformed[:, 2] *= -1.0
    return transformed


def compute_pca_projection(points: np.ndarray, flip_about_y: bool = False) -> ProjectionResult:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    points = apply_pre_projection_transform(points, flip_about_y)
    if points.shape[0] < 3:
        raise ValueError("PCA mode needs at least 3 selected atoms to define a useful projection plane")

    centroid = points.mean(axis=0)
    centered = points - centroid
    rank = int(np.linalg.matrix_rank(centered))
    if rank < 2:
        raise ValueError("Selected points are nearly collinear or identical; the PCA plane is not uniquely defined")

    covariance = (centered.T @ centered) / max(points.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    basis_x = orient_vector_stably(eigenvectors[:, 0])
    basis_y = orient_vector_stably(eigenvectors[:, 1])
    normal = np.cross(basis_x, basis_y)
    normal = normal / np.linalg.norm(normal)
    basis_y = np.cross(normal, basis_x)
    basis_y = basis_y / np.linalg.norm(basis_y)

    proj_x = centered @ basis_x
    proj_y = centered @ basis_y
    proj_z = centered @ normal
    a, b, c = normal.tolist()
    d = -float(np.dot(normal, centroid))

    total = float(np.sum(eigenvalues))
    if total > 0:
        explained = eigenvalues / total
    else:
        explained = np.zeros(3, dtype=float)

    return ProjectionResult(
        mode="pca",
        centroid=centroid,
        basis_x=basis_x,
        basis_y=basis_y,
        normal=normal,
        plane_coefficients=np.array([a, b, c, d], dtype=float),
        projected_xy=np.column_stack((proj_x, proj_y)),
        projected_depth=proj_z,
        eigenvalues=eigenvalues,
        explained_variance_ratio=explained,
        rank=rank,
        method_name="PCA/SVD-equivalent covariance projection",
        method_description=(
            "The projection plane is spanned by the first two principal components of the selected atom coordinates. "
            "The depth coordinate is the third principal-component coordinate."
        ),
        pre_flip_about_y=bool(flip_about_y),
    )


def compute_current_xy_projection(points: np.ndarray, flip_about_y: bool = False) -> ProjectionResult:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    points = apply_pre_projection_transform(points, flip_about_y)
    if points.shape[0] < 1:
        raise ValueError("No points available for current-xy projection")

    centroid = points.mean(axis=0)
    projected_xy = points[:, :2].copy()
    projected_depth = points[:, 2].copy()
    return ProjectionResult(
        mode="current-xy",
        centroid=centroid,
        basis_x=np.array([1.0, 0.0, 0.0], dtype=float),
        basis_y=np.array([0.0, 1.0, 0.0], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        plane_coefficients=np.array([0.0, 0.0, 1.0, 0.0], dtype=float),
        projected_xy=projected_xy,
        projected_depth=projected_depth,
        eigenvalues=None,
        explained_variance_ratio=None,
        rank=None,
        method_name="Current input X/Y projection without PCA",
        method_description=(
            "No PCA fitting was performed. proj_x is the input X coordinate, proj_y is the input Y coordinate, "
            "and depth is the input Z coordinate after the optional pre-projection transform."
        ),
        pre_flip_about_y=bool(flip_about_y),
    )


def transform_points_to_projection_basis(points: np.ndarray, projection: ProjectionResult) -> np.ndarray:
    transformed = apply_pre_projection_transform(points, getattr(projection, "pre_flip_about_y", False))
    if projection.mode == "current-xy":
        return transformed.copy()
    centered = transformed - projection.centroid
    return np.column_stack((centered @ projection.basis_x, centered @ projection.basis_y, centered @ projection.normal))


def transform_points_from_projection_basis(points: np.ndarray, projection: ProjectionResult) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if projection.mode == "current-xy":
        transformed = pts.copy()
    else:
        transformed = (
            projection.centroid
            + pts[:, 0, None] * projection.basis_x
            + pts[:, 1, None] * projection.basis_y
            + pts[:, 2, None] * projection.normal
        )
    if getattr(projection, "pre_flip_about_y", False):
        transformed = transformed.copy()
        transformed[:, 0] *= -1.0
        transformed[:, 2] *= -1.0
    return transformed


def convex_hull_area_2d(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 3:
        return 0.0
    pts = sorted(set((float(x), float(y)) for x, y in points_xy))
    if len(pts) < 3:
        return 0.0

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    area2 = 0.0
    for (x1, y1), (x2, y2) in zip(hull, hull[1:] + hull[:1]):
        area2 += x1 * y2 - x2 * y1
    return abs(area2) / 2.0


def sanitize_for_filename(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe or "atoms"


def atom_types_label(atom_types: Sequence[str]) -> str:
    return "_".join(sanitize_for_filename(atom_type) for atom_type in atom_types) or "atoms"


def default_output_paths(
    input_file: Path,
    atom_types: Sequence[str],
    resolved_input_format: Optional[str] = None,
    name_tags: Optional[Sequence[str]] = None,
) -> Tuple[Path, Path, Path]:
    safe_atoms = atom_types_label(atom_types)
    safe_tags = [sanitize_for_filename(tag) for tag in (name_tags or []) if str(tag).strip()]
    label_parts = [safe_atoms] + safe_tags
    label = "_".join(part for part in label_parts if part) or "points"
    stem = input_file.with_suffix("").name
    svg_path = Path("{0}_{1}_projection.svg".format(stem, label))
    json_path = Path("{0}_{1}_plane.json".format(stem, label))

    fmt = (resolved_input_format or "").strip().lower()
    if not fmt:
        suffix = input_file.suffix.lower()
        fmt = "pdb" if suffix in {".pdb", ".ent"} else "xyz"
    if fmt == "pdb":
        basis_suffix = ".pdb"
    else:
        basis_suffix = input_file.suffix or ".xyz"
    basis_path = input_file.with_name("{0}_{1}_PCA{2}".format(input_file.stem, label, basis_suffix))
    return svg_path, json_path, basis_path


def default_name_tags_from_args(args: argparse.Namespace, styles: Optional[Dict[str, AtomStyle]] = None) -> List[str]:
    """Build short default-filename tags from options that visibly change output."""
    tags: List[str] = []
    if getattr(args, "projection_mode", "pca") == "current-xy":
        tags.append("currentXY")

    any_lines = False
    modes = set()
    any_extend = False
    if styles:
        for style in styles.values():
            if style.draw_lines:
                any_lines = True
                modes.add(normalize_connection_mode(style.connection_mode))
                if style.extend_3prime:
                    any_extend = True
    else:
        any_lines = bool(getattr(args, "draw_lines", False))
        if any_lines:
            modes.add(normalize_connection_mode(getattr(args, "connection_mode", "smooth")))
        any_extend = bool(getattr(args, "extend_3prime", False))

    if any_lines:
        if modes == {"straight"}:
            tags.append("straight")
        elif modes == {"smooth"}:
            tags.append("smooth")
        else:
            tags.append("lines")
    if any_extend:
        tags.append("3prime")
    if bool(getattr(args, "flip_about_y", False)):
        tags.append("flipY")
    if bool(getattr(args, "draw_base_pairs", False)):
        tags.append("bp")
    if bool(getattr(args, "draw_xy_plane", False)):
        tags.append("xyplane")
    if bool(getattr(args, "depth_order_circles", False) or getattr(args, "depth_order_lines", False) or getattr(args, "depth_order_base_pairs", False)):
        tags.append("depth")
    if bool(getattr(args, "line_underlay", False) and getattr(args, "depth_order_lines", False) and any_lines):
        tags.append("underlay")
    return tags


def split_atom_type_text(text: str) -> List[str]:
    parts = re.split(r"[,;\n\r\t]+", text)
    return [part.strip() for part in parts if part.strip()]


def collect_atom_types(args: argparse.Namespace) -> List[str]:
    raw_values: List[str] = []
    for value in getattr(args, "atom_type", None) or []:
        raw_values.extend(split_atom_type_text(value))
    if getattr(args, "atom_types", None):
        raw_values.extend(split_atom_type_text(args.atom_types))
    unique: List[str] = []
    seen = set()
    for value in raw_values:
        key = value.upper()
        if key not in seen:
            unique.append(value)
            seen.add(key)
    return unique


def color_for_index(index: int) -> str:
    hue = (index * GOLDEN_RATIO_CONJUGATE) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, DEFAULT_COLOR_SATURATION, DEFAULT_COLOR_VALUE)
    return "#{0:02x}{1:02x}{2:02x}".format(int(red * 255), int(green * 255), int(blue * 255))


def normalize_palette_name(value: object) -> str:
    """Return "default" or the DiLiuLab label, e.g. "DiLiuLab T80", that a --palette value names."""
    if lab_colors is not None:
        return lab_colors.check_palette(value)
    if value is None or str(value).strip().lower() in {"", "default", "none", "auto"}:
        return DEFAULT_PALETTE
    raise ValueError(
        "palette {0!r} needs lab_colors.py from curve_it_lib, which could not be imported; "
        "only the default palette is available".format(value)
    )


def argparse_palette(value: str) -> str:
    try:
        return normalize_palette_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def palette_from_args(args: argparse.Namespace) -> str:
    return normalize_palette_name(getattr(args, "palette", DEFAULT_PALETTE))


def palette_choices() -> List[str]:
    if lab_colors is None:
        return [DEFAULT_PALETTE]
    return lab_colors.palette_choices(DEFAULT_PALETTE)


def palette_colors(palette: str) -> List[str]:
    """The DiLiuLab colors a palette names, in lab order; empty for the default palette."""
    palette = normalize_palette_name(palette)
    if palette == DEFAULT_PALETTE:
        return []
    return lab_colors.resolve_palette(palette, [])


def palette_color_for_index(index: int, palette: str = DEFAULT_PALETTE) -> str:
    """Categorical color number index: golden-ratio HSV for the default palette,
    otherwise the palette's colors in lab order, repeating after the last one."""
    colors = palette_colors(palette)
    if not colors:
        return color_for_index(index)
    return colors[index % len(colors)]


class ColorOptionError(ValueError):
    """A color value Plane It cannot use: a DiLiuLab name with a tint the palette
    lacks (RedT75), "DiLiuLab <not a lab color>", the palette's own name where
    one color is expected (DiLiuLab-T80), or a malformed --chain-colors value.
    main() reports it as a usage error, exit 2, the way argparse reports any
    other bad option value; the GUI shows it in the run log."""


# The single-color options, each resolved once by resolve_color_options.
COLOR_OPTIONS = (
    ("line_underlay_stroke", "--line-underlay-stroke"),
    ("xy_plane_fill", "--xy-plane-fill"),
    ("xy_plane_stroke", "--xy-plane-stroke"),
    ("scale_bar_stroke", "--scale-bar-stroke"),
    ("scale_bar_background", "--scale-bar-background"),
    ("base_pair_stroke", "--base-pair-stroke"),
)
STYLE_COLOR_KEYS = {"fill", "stroke", "line_stroke"}
BLANK_CHAIN_WORDS = {"blank", "none", "_blank_"}
LAB_NAME_HINT = "; a DiLiuLab name such as MintGreenT80 also works" if lab_colors is not None else ""
LAB_NAMES_HELP = (
    " DiLiuLab names: Red, Blue, Magenta, Cyan, Orange, Purple, Green, Yellow, MintGreen and Black/Gray, "
    "each with its tint T100, T80, T60, T40 or T20, e.g. RedT80, MintGreenT60, GrayT40; the bare name red "
    "stays SVG's #ff0000, and the palette's own name, DiLiuLab-T80, is refused, since it names nine colors."
) if lab_colors is not None else ""
# The 147 color keywords of SVG 1.1 (the CSS3 names), with the #rrggbb each
# stands for, for the window's squares and swatches. Tk also knows the X11
# names (gray50, wheat1, navyblue, VioletRed), which an SVG viewer does not draw,
# and gives gray, green, maroon and purple other values than SVG does.
SVG_COLOR_KEYWORDS = {
    "aliceblue": "#f0f8ff", "antiquewhite": "#faebd7", "aqua": "#00ffff", "aquamarine": "#7fffd4",
    "azure": "#f0ffff", "beige": "#f5f5dc", "bisque": "#ffe4c4", "black": "#000000", "blanchedalmond": "#ffebcd",
    "blue": "#0000ff", "blueviolet": "#8a2be2", "brown": "#a52a2a", "burlywood": "#deb887", "cadetblue": "#5f9ea0",
    "chartreuse": "#7fff00", "chocolate": "#d2691e", "coral": "#ff7f50", "cornflowerblue": "#6495ed",
    "cornsilk": "#fff8dc", "crimson": "#dc143c", "cyan": "#00ffff", "darkblue": "#00008b", "darkcyan": "#008b8b",
    "darkgoldenrod": "#b8860b", "darkgray": "#a9a9a9", "darkgreen": "#006400", "darkgrey": "#a9a9a9",
    "darkkhaki": "#bdb76b", "darkmagenta": "#8b008b", "darkolivegreen": "#556b2f", "darkorange": "#ff8c00",
    "darkorchid": "#9932cc", "darkred": "#8b0000", "darksalmon": "#e9967a", "darkseagreen": "#8fbc8f",
    "darkslateblue": "#483d8b", "darkslategray": "#2f4f4f", "darkslategrey": "#2f4f4f", "darkturquoise": "#00ced1",
    "darkviolet": "#9400d3", "deeppink": "#ff1493", "deepskyblue": "#00bfff", "dimgray": "#696969",
    "dimgrey": "#696969", "dodgerblue": "#1e90ff", "firebrick": "#b22222", "floralwhite": "#fffaf0",
    "forestgreen": "#228b22", "fuchsia": "#ff00ff", "gainsboro": "#dcdcdc", "ghostwhite": "#f8f8ff",
    "gold": "#ffd700", "goldenrod": "#daa520", "gray": "#808080", "green": "#008000", "greenyellow": "#adff2f",
    "grey": "#808080", "honeydew": "#f0fff0", "hotpink": "#ff69b4", "indianred": "#cd5c5c", "indigo": "#4b0082",
    "ivory": "#fffff0", "khaki": "#f0e68c", "lavender": "#e6e6fa", "lavenderblush": "#fff0f5",
    "lawngreen": "#7cfc00", "lemonchiffon": "#fffacd", "lightblue": "#add8e6", "lightcoral": "#f08080",
    "lightcyan": "#e0ffff", "lightgoldenrodyellow": "#fafad2", "lightgray": "#d3d3d3", "lightgreen": "#90ee90",
    "lightgrey": "#d3d3d3", "lightpink": "#ffb6c1", "lightsalmon": "#ffa07a", "lightseagreen": "#20b2aa",
    "lightskyblue": "#87cefa", "lightslategray": "#778899", "lightslategrey": "#778899", "lightsteelblue": "#b0c4de",
    "lightyellow": "#ffffe0", "lime": "#00ff00", "limegreen": "#32cd32", "linen": "#faf0e6", "magenta": "#ff00ff",
    "maroon": "#800000", "mediumaquamarine": "#66cdaa", "mediumblue": "#0000cd", "mediumorchid": "#ba55d3",
    "mediumpurple": "#9370db", "mediumseagreen": "#3cb371", "mediumslateblue": "#7b68ee",
    "mediumspringgreen": "#00fa9a", "mediumturquoise": "#48d1cc", "mediumvioletred": "#c71585",
    "midnightblue": "#191970", "mintcream": "#f5fffa", "mistyrose": "#ffe4e1", "moccasin": "#ffe4b5",
    "navajowhite": "#ffdead", "navy": "#000080", "oldlace": "#fdf5e6", "olive": "#808000", "olivedrab": "#6b8e23",
    "orange": "#ffa500", "orangered": "#ff4500", "orchid": "#da70d6", "palegoldenrod": "#eee8aa",
    "palegreen": "#98fb98", "paleturquoise": "#afeeee", "palevioletred": "#db7093", "papayawhip": "#ffefd5",
    "peachpuff": "#ffdab9", "peru": "#cd853f", "pink": "#ffc0cb", "plum": "#dda0dd", "powderblue": "#b0e0e6",
    "purple": "#800080", "red": "#ff0000", "rosybrown": "#bc8f8f", "royalblue": "#4169e1", "saddlebrown": "#8b4513",
    "salmon": "#fa8072", "sandybrown": "#f4a460", "seagreen": "#2e8b57", "seashell": "#fff5ee", "sienna": "#a0522d",
    "silver": "#c0c0c0", "skyblue": "#87ceeb", "slateblue": "#6a5acd", "slategray": "#708090",
    "slategrey": "#708090", "snow": "#fffafa", "springgreen": "#00ff7f", "steelblue": "#4682b4", "tan": "#d2b48c",
    "teal": "#008080", "thistle": "#d8bfd8", "tomato": "#ff6347", "turquoise": "#40e0d0", "violet": "#ee82ee",
    "wheat": "#f5deb3", "white": "#ffffff", "whitesmoke": "#f5f5f5", "yellow": "#ffff00", "yellowgreen": "#9acd32",
}


def resolve_color_value(value, where: str):
    """The #rrggbb a DiLiuLab color name such as MintGreenT80 stands for. Any
    other value, a hex code, an SVG name such as red or anything else, comes
    back exactly as given, as every value does when lab_colors is missing.
    `where` names the option in the error for a lab name the palette lacks,
    and for the palette's own name, DiLiuLab-T80, which names nine colors
    where one is expected and would reach the SVG as an invalid color."""
    if lab_colors is None or value is None:
        return value
    try:
        tint = lab_colors.lab_tint_of(value)
    except ValueError:
        tint = "T80"                        # the palette's name with a tint it lacks, DiLiuLab-T75
    if tint is not None:
        try:
            example = lab_colors.lab_color_name("red", tint)
        except ValueError:
            example = "RedT80"
        raise ColorOptionError(
            "{0}: {1} names the whole palette, not one color; give one color, such as {2}".format(
                where, str(value).strip(), example
            )
        )
    try:
        code = lab_colors.resolve_lab_color(value)
    except ValueError as exc:
        raise ColorOptionError("{0}: {1}".format(where, exc)) from exc
    return value if code is None else code


def svg_color_code(value) -> Optional[str]:
    """The #rrggbb an SVG viewer draws for a color value, or None for a value it
    cannot draw as a color, for the window's squares and swatches. Accepted:
    #rgb, #rgba, #rrggbb and #rrggbbaa (the alpha left out), the 147 SVG color
    keywords, in any case, and a DiLiuLab name, which the run writes as its hex
    code. Everything else is None: the X11 names Tk knows and SVG does not
    (gray50, wheat1, navyblue, VioletRed), rgb(), which a style token cannot
    carry since a spec splits at its commas and spaces, none, the palette's own
    name, and a lab name with a tint the palette lacks."""
    text = str(value or "").strip()
    if not text:
        return None
    if lab_colors is not None:
        try:
            if lab_colors.lab_tint_of(text) is not None:
                return None                 # DiLiuLab-T80: nine colors, not one
            code = lab_colors.resolve_lab_color(text)
        except ValueError:
            return None
        if code is not None:
            return code.lower()
    match = re.fullmatch(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})", text)
    if match:
        digits = match.group(1)
        if len(digits) <= 4:
            digits = "".join(digit * 2 for digit in digits)
        return "#" + digits[:6].lower()
    return SVG_COLOR_KEYWORDS.get(text.lower())


def style_color_token(text: str, where: str) -> str:
    """A window field's color as a --style token carries it. The value is checked
    here, under the field's own name (where, e.g. "P Fill"), so that a bad one
    is reported the same way with or without spaces; one with spaces, commas or
    semicolons, at which a style spec splits, such as "Mint Green T80", goes as
    its hex code, and every other value goes as typed."""
    code = resolve_color_value(text, where)
    if re.search(r"[\s,;]", text):
        return str(code)
    return text


def resolve_color_options(args: argparse.Namespace) -> None:
    """Replace a DiLiuLab name in any single-color option by its #rrggbb, once,
    before anything is read or written, so no DiLiuLab name reaches the SVG."""
    for attr, option in COLOR_OPTIONS:
        if hasattr(args, attr):
            setattr(args, attr, resolve_color_value(getattr(args, attr), option))


def parse_chain_colors(text: object) -> Dict[str, Tuple[str, str]]:
    """{chain ID: (color, color as given)} from a --chain-colors value such as
    "A=RedT80,B=#4c79e6": CHAIN=COLOR pairs separated by commas, semicolons or
    whitespace, with blank (or none) naming the blank chain as --closed-chains
    does. Names are resolved to #rrggbb here; a pair that is not CHAIN=COLOR is
    a ColorOptionError. A chain listed twice takes its last color."""
    specs: Dict[str, Tuple[str, str]] = {}
    text = str(text or "").strip()
    if not text:
        return specs
    for part in re.split(r"[,;\s]+", re.sub(r"\s*=\s*", "=", text)):
        if not part:
            continue
        chain, sep, color = part.partition("=")
        if not sep or not chain or not color or "=" in color:
            raise ColorOptionError(
                "--chain-colors takes CHAIN=COLOR pairs separated by commas, e.g. A=RedT80,B=#4c79e6 "
                "(blank=... for a blank chain ID); {0!r} is not one".format(part)
            )
        if chain.lower() in BLANK_CHAIN_WORDS:
            chain = ""
        specs[chain] = (resolve_color_value(color, "--chain-colors {0}".format(part)), color)
    return specs


def chain_color_overrides(args: argparse.Namespace) -> Dict[str, str]:
    """The resolved --chain-colors of a run, {chain ID: #rrggbb or color}."""
    specs = getattr(args, "chain_color_specs", None) or {}
    return {chain: color for chain, (color, _given) in specs.items()}


def format_chain_colors(colors: Dict[str, str]) -> str:
    """A=#eb7070,B=#4c79e6: the --chain-colors spelling, blank for the blank chain."""
    return ",".join("{0}={1}".format(chain_label(chain), color) for chain, color in colors.items())


def chain_color_order(selected_atoms: Sequence[SelectedAtom]) -> List[str]:
    """Chain IDs in the order the SVG gives them palette colors, the order they
    first appear among the selected atoms, found exactly as write_projection_svg
    groups them (the projected positions play no part in the order)."""
    count = len(selected_atoms)
    grouped = group_selected_atoms(selected_atoms, np.zeros((count, 2)), np.zeros(count))
    return [chain for chain, _type_groups in sorted_chain_items(grouped)]


def assign_chain_colors(chains: Sequence[str], palette: str, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Each chain's color in chain mode: its --chain-colors color, or else the
    palette's color for its place in `chains`, so a chain given its own color
    does not shift the colors of the others."""
    colors = {chain: palette_color_for_index(index, palette) for index, chain in enumerate(chains)}
    for chain, color in (overrides or {}).items():
        if chain in colors:
            colors[chain] = color
    return colors


def explicit_chain_mode_colors(
    args: argparse.Namespace,
    atom_types: Sequence[str],
    selected_atoms: Sequence[SelectedAtom],
    styles: Dict[str, AtomStyle],
) -> List[Tuple[str, str]]:
    """[(atom type, "fill" or "line_stroke"), ...]: the colors set explicitly in a
    style that replace chain colors in this chain-mode run. A line_stroke counts
    only for an atom type that draws lines."""
    if getattr(args, "color_by", "chain") != "chain":
        return []
    explicit = getattr(args, "explicit_style_keys", None) or {}
    counts = atom_type_counts(selected_atoms)
    found: List[Tuple[str, str]] = []
    for atom_type in atom_types:
        keys = explicit.get(atom_type.upper(), set())
        if not counts.get(atom_type):
            continue
        if "fill" in keys:
            found.append((atom_type, "fill"))
        if "line_stroke" in keys and styles[atom_type.upper()].draw_lines:
            found.append((atom_type, "line_stroke"))
    return found


def chain_color_report(args: argparse.Namespace, selected_atoms: Sequence[SelectedAtom]) -> dict:
    """What --chain-colors did in this run: the colors given, the chains they
    colored ("used"), the chains in the selection whose color nothing drawn
    took ("unused": every circle and line there has a fill or line_stroke set
    explicitly), the chains missing from the selection, and whether the option
    was ignored because color_by is atom-type. Which chains some drawn element
    took its chain color for is what write_projection_svg records as
    args.chain_color_takers; before an SVG is written every listed chain in
    the selection counts as used."""
    specs = getattr(args, "chain_color_specs", None) or {}
    chain_mode = getattr(args, "color_by", "chain") == "chain"
    present = chain_color_order(selected_atoms) if specs else []
    takers = getattr(args, "chain_color_takers", None)
    listed = [chain for chain in present if chain_mode and chain in specs]
    used = [chain for chain in listed if takers is None or chain in takers]
    return {
        "given": specs,
        "used": used,
        "unused": [chain for chain in listed if chain not in used],
        "missing": [chain for chain in specs if chain_mode and chain not in present],
        "present": present,
        "ignored": bool(specs) and not chain_mode,
    }


def parse_bool_text(value: str) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError("Expected true/false value, got {0!r}".format(value))


def normalize_style_key(key: str) -> str:
    key = key.strip().lower().replace("-", "_")
    aliases = {
        "r": "radius",
        "alpha": "opacity",
        "circle_opacity": "opacity",
        "sw": "stroke_width",
        "stroke_w": "stroke_width",
        "line": "line_stroke",
        "line_color": "line_stroke",
        "line_stroke_width": "line_width",
        "lw": "line_width",
        "line_alpha": "line_opacity",
        "connect": "draw_lines",
        "lines": "draw_lines",
        "extend3": "extend_3prime",
        "extend_3": "extend_3prime",
        "extend_3prime": "extend_3prime",
        "extend_3_prime": "extend_3prime",
        "three_prime_extension": "extend_3prime",
        "o3_extension": "extend_3prime",
    }
    return aliases.get(key, key)


def parse_style_token(token: str) -> Tuple[str, str]:
    if "=" not in token:
        raise ValueError("Style token must use key=value format: {0}".format(token))
    key, value = token.split("=", 1)
    return normalize_style_key(key), value.strip()


def parse_style_specs(
    style_specs: Sequence[str],
    atom_types: Sequence[str],
    default_radius: float,
    default_line_width: float,
    default_line_opacity: float,
    default_draw_lines: bool,
    default_connection_mode: str,
    default_extend_3prime: bool = False,
    palette: str = DEFAULT_PALETTE,
    explicit_keys: Optional[Dict[str, set]] = None,
) -> Dict[str, AtomStyle]:
    """Per-atom-type styles from the defaults and the --style specs. A DiLiuLab
    name in fill, stroke or line_stroke becomes its #rrggbb here. When given,
    explicit_keys collects {ATOM TYPE: {key, ...}} for every key a spec set, so
    that chain mode can tell an explicit fill= from the palette's default."""
    styles: Dict[str, AtomStyle] = {}
    for index, atom_type in enumerate(atom_types):
        # The palette sets only the starting fill/line_stroke; fill= and
        # line_stroke= in a style spec below still replace them.
        color = palette_color_for_index(index, palette)
        default_opacity = 0.2 if index == 0 else 1.0
        styles[atom_type.upper()] = AtomStyle(
            fill=color,
            stroke="#222222",
            stroke_width=0.6,
            radius=float(default_radius),
            opacity=default_opacity,
            draw_lines=bool(default_draw_lines),
            connection_mode=normalize_connection_mode(default_connection_mode),
            line_stroke=color,
            line_width=float(default_line_width),
            line_opacity=float(default_line_opacity),
            extend_3prime=bool(default_extend_3prime),
        )

    chunks: List[str] = []
    for spec in style_specs:
        for raw_line in spec.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            for chunk in line.split(";"):
                chunk = chunk.strip()
                if chunk:
                    chunks.append(chunk)

    allowed = {
        "fill", "stroke", "stroke_width", "radius", "opacity", "draw_lines",
        "connection_mode", "line_stroke", "line_width", "line_opacity",
        "extend_3prime",
    }
    numeric = {"stroke_width", "radius", "opacity", "line_width", "line_opacity"}
    bool_keys = {"draw_lines", "extend_3prime"}

    for chunk in chunks:
        if ":" in chunk:
            atom_type_part, rest = chunk.split(":", 1)
        else:
            fields = chunk.split(None, 1)
            if len(fields) != 2:
                raise ValueError("Style spec must start with an atom type followed by key=value pairs: {0}".format(chunk))
            atom_type_part, rest = fields[0], fields[1]
        atom_type_key = atom_type_part.strip().upper()
        if atom_type_key not in styles:
            raise ValueError(
                "Style provided for atom type {0!r}, but selected atom types are: {1}".format(
                    atom_type_part.strip(), ", ".join(atom_types)
                )
            )
        tokens = [tok for tok in re.split(r"[\s,]+", rest.strip()) if tok]
        style = styles[atom_type_key]
        for token in tokens:
            key, value = parse_style_token(token)
            if key not in allowed:
                raise ValueError("Unknown style key {0!r}; allowed keys are {1}".format(key, ", ".join(sorted(allowed))))
            if key in numeric:
                try:
                    number = float(value)
                except ValueError as exc:
                    raise ValueError("Style key {0} requires a number, got {1!r}".format(key, value)) from exc
                if key in {"radius", "stroke_width", "line_width"} and number < 0:
                    raise ValueError("Style key {0} must be non-negative".format(key))
                if key in {"opacity", "line_opacity"} and not (0.0 <= number <= 1.0):
                    raise ValueError("Style key {0} must be between 0 and 1".format(key))
                setattr(style, key, number)
            elif key in bool_keys:
                setattr(style, key, parse_bool_text(value))
            elif key == "connection_mode":
                setattr(style, key, normalize_connection_mode(value))
            elif key in STYLE_COLOR_KEYS:
                setattr(style, key, resolve_color_value(value, "--style {0} {1}".format(atom_type_part.strip(), token)))
            else:
                setattr(style, key, value)
            if explicit_keys is not None:
                explicit_keys.setdefault(atom_type_key, set()).add(key)
    return styles


def vector_to_dict(vec: Sequence[float]) -> dict:
    return {"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])}


def atom_type_counts(selected_atoms: Sequence[SelectedAtom]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for selected in selected_atoms:
        counts[selected.atom_type] = counts.get(selected.atom_type, 0) + 1
    return counts


def chain_label(chain: str) -> str:
    return chain if chain else "blank"


def chain_counts(selected_atoms: Sequence[SelectedAtom]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for selected in selected_atoms:
        label = chain_label(selected.atom.chain)
        counts[label] = counts.get(label, 0) + 1
    return counts


def style_to_dict(style: AtomStyle) -> dict:
    return {
        "fill": style.fill,
        "stroke": style.stroke,
        "stroke_width": style.stroke_width,
        "radius": style.radius,
        "opacity": style.opacity,
        "draw_lines": style.draw_lines,
        "connection_mode": style.connection_mode,
        "line_stroke": style.line_stroke,
        "line_width": style.line_width,
        "line_opacity": style.line_opacity,
        "extend_3prime": style.extend_3prime,
    }


def write_projection_json(
    output_json: Path,
    args: argparse.Namespace,
    atom_types: Sequence[str],
    selected_atoms: Sequence[SelectedAtom],
    projection: ProjectionResult,
    hull_area: float,
    output_svg: Path,
    output_csv: Optional[Path],
    output_pca_pdb: Optional[Path],
    styles: Dict[str, AtomStyle],
    base_pair_info: Optional[dict] = None,
    projection_scale: Optional[float] = None,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    coeff = projection.plane_coefficients
    projection_section = {
        "mode": projection.mode,
        "method_name": projection.method_name,
        "description": projection.method_description,
        "not_a_true_hull_area_optimization": True if projection.mode == "pca" else None,
        "depth_coordinate": "PCA-normal coordinate" if projection.mode == "pca" else "input Z coordinate after optional pre-projection transform",
        "pre_flip_about_y": bool(getattr(projection, "pre_flip_about_y", False)),
        "coordinate_convention": "Coordinates are flipped about the Y axis before projection when pre_flip_about_y is true: x -> -x, y -> y, z -> -z.",
        "convex_hull_area_of_written_2d_projection_diagnostic_only": float(hull_area),
        "basis_x": vector_to_dict(projection.basis_x),
        "basis_y": vector_to_dict(projection.basis_y),
        "normal_or_depth_axis": vector_to_dict(projection.normal),
        "centroid_of_selected_atoms": vector_to_dict(projection.centroid),
        "plane_A_B_C_D": [float(v) for v in coeff],
    }
    if projection_scale is not None:
        projection_section["svg_scale_units_per_projection_unit"] = float(projection_scale)
        projection_section["svg_distance_formula"] = "svg_distance = projected_distance * svg_scale_units_per_projection_unit"
        projection_section["projected_distance_formula"] = "projected_distance = svg_distance / svg_scale_units_per_projection_unit"
    if projection.eigenvalues is not None:
        projection_section["pca"] = {
            "coordinate_rank": int(projection.rank) if projection.rank is not None else None,
            "eigenvalues_descending": [float(v) for v in projection.eigenvalues],
            "explained_variance_ratio": [float(v) for v in projection.explained_variance_ratio],
            "projected_covariance_spread_sqrt_lambda1_lambda2": float(
                math.sqrt(max(projection.eigenvalues[0], 0.0) * max(projection.eigenvalues[1], 0.0))
            ),
        }

    base_pair_depth_order_active = bool(
        getattr(args, "depth_order_base_pairs", False)
        and base_pair_info
        and base_pair_info.get("base_pairs_drawn", 0)
    )
    xy_plane_depth_ordered = bool(
        getattr(args, "draw_xy_plane", False)
        and (
            getattr(args, "depth_order_circles", False)
            or getattr(args, "depth_order_lines", False)
            or getattr(args, "pdb_order_circles", False)
            or getattr(args, "pdb_order_lines", False)
            or base_pair_depth_order_active
        )
    )

    metadata = {
        "input_file": str(Path(args.pdb_file)),
        "requested_input_format": getattr(args, "input_format", "auto"),
        "resolved_input_format": getattr(args, "resolved_input_format", getattr(args, "input_format", "auto")),
        "atom_types": list(atom_types),
        "select_by": args.select_by,
        "records": args.records,
        "model": args.model,
        "chain": args.chain,
        "resname": args.resname,
        "altloc": args.altloc,
        "n_points": len(selected_atoms),
        "pre_flip_about_y": bool(getattr(args, "flip_about_y", False)),
        "default_color_saturation": DEFAULT_COLOR_SATURATION,
        "default_color_value": DEFAULT_COLOR_VALUE,
        "color_by": args.color_by,
    }
    # The palette is recorded only when it is not the default, so default
    # metadata stays exactly as before.
    palette = palette_from_args(args)
    if palette != DEFAULT_PALETTE:
        metadata["palette"] = palette
        metadata["palette_colors"] = palette_colors(palette)
    # --chain-colors, and colors set explicitly in a style, are recorded only
    # when they are given, for the same reason.
    chain_report = chain_color_report(args, selected_atoms)
    if chain_report["given"]:
        metadata["chain_colors"] = {chain_label(chain): color for chain, (color, _given) in chain_report["given"].items()}
    # The chain-mode note keeps its exact words unless a style sets a fill or
    # line_stroke that wins over the chain colors, which those words deny.
    explicit_colors = explicit_chain_mode_colors(args, atom_types, selected_atoms, styles)
    if explicit_colors:
        chain_mode_note = (
            "When color_by is chain, chain colors override the per-atom-type fill and line_stroke not set explicitly in a style, "
            "but radius, opacity, stroke_width, line_width, and line_opacity remain per atom type; a fill or line_stroke set "
            "explicitly ({0}) replaces the chain colors for that atom type's circles or lines.".format(
                ", ".join("{0} {1}".format(atom_type, key) for atom_type, key in explicit_colors)
            )
        )
    else:
        chain_mode_note = (
            "When color_by is chain, chain colors override per-atom-type fill and line_stroke, but radius, opacity, "
            "stroke_width, line_width, and line_opacity remain per atom type."
        )
    metadata.update({
        "default_connection_mode": getattr(args, "connection_mode", "smooth"),
        "closed_chains": getattr(args, "closed_chains", ""),
        "close_all_chains": bool(getattr(args, "close_all_chains", False)),
        "depth_order_circles": bool(args.depth_order_circles or getattr(args, "pdb_order_circles", False)),
        "depth_order_lines": bool(args.depth_order_lines or getattr(args, "pdb_order_lines", False)),
        "depth_order_base_pairs": bool(getattr(args, "depth_order_base_pairs", False)),
        "line_underlay": bool(getattr(args, "line_underlay", False)),
        "depth_front": args.depth_front,
        "base_pairs": base_pair_info or {"enabled": False},
        "xy_plane": {
            "drawn": bool(getattr(args, "draw_xy_plane", False)),
            "layer_id": "xy-plane",
            "shape_id": "xy-plane-shape",
            "plane": "projection-basis-depth=0",
            "finite_patch": "selected atom/point projected x/y bounds plus margin",
            "depth_ordered": xy_plane_depth_ordered,
            "depth_sort": "mean projected corner depth; zero for the projection-basis xy plane",
            "fill": getattr(args, "xy_plane_fill", "#7dd3fc"),
            "stroke": getattr(args, "xy_plane_stroke", "#0284c7"),
            "stroke_width": float(getattr(args, "xy_plane_stroke_width", 1.5)),
            "opacity": float(getattr(args, "xy_plane_opacity", 0.18)),
        },
        "scale_bar": {
            "drawn": not bool(getattr(args, "no_scale_bar", False)),
            "length_projection_units": float(getattr(args, "scale_bar_length", DEFAULT_SCALE_BAR_LENGTH)),
            "unit_label": str(getattr(args, "scale_bar_unit_label", DEFAULT_SCALE_BAR_UNIT_LABEL) or DEFAULT_SCALE_BAR_UNIT_LABEL),
            "svg_units_per_projection_unit": float(projection_scale) if projection_scale is not None else None,
            "definition": "Scale bar length in SVG units is length_projection_units multiplied by svg_units_per_projection_unit.",
        },
        "counts_by_atom_type": atom_type_counts(selected_atoms),
        "counts_by_chain": chain_counts(selected_atoms),
        "styles_by_atom_type": {atom_type: style_to_dict(styles[atom_type.upper()]) for atom_type in atom_types},
        "outputs": {
            "svg": str(output_svg),
            "csv": str(output_csv) if output_csv is not None else None,
            "projection_basis_structure": str(output_pca_pdb) if output_pca_pdb is not None else None,
        },
        "projection": projection_section,
        "notes": [
            "Neighbor connections, if enabled for an atom type, connect consecutive selected atoms with the same chain, model, and atom type in input order.",
            "Each atom type can use straight or smooth Catmull-Rom/cubic-Bezier neighbor connections.",
            "Closed chains add one additional neighbor segment from the last selected atom in that chain/model/type group back to the first.",
            "Base-pair interaction lines, if enabled, are read from DSSR and connect the selected projected anchor atom of the paired residues.",
            "When SVG depth ordering is enabled, circles, neighbor segments, and/or base-pair lines are written back-to-front using the selected projection depth coordinate.",
            chain_mode_note,
        ],
    })
    if palette != DEFAULT_PALETTE:
        metadata["notes"].append(
            "The palette gives chain colors in the order chains first appear and the default per-atom-type fill and line_stroke in atom-type order, "
            "taking palette_colors in order and repeating after the last one; fill and line_stroke set in a style still win."
        )
    if chain_report["used"]:
        used = chain_report["used"]
        metadata["notes"].append(
            "In this run, chain_colors gives chain{0} {1} {2} own color{0}, and the other chains keep the palette's chain colors. "
            "Precedence when color_by is chain: an atom type's explicit fill or line_stroke, then chain_colors, then the "
            "palette's chain color.".format(
                "s" if len(used) > 1 else "", ", ".join(chain_label(chain) for chain in used), "their" if len(used) > 1 else "its"
            )
        )
    if chain_report["unused"]:
        unused = chain_report["unused"]
        metadata["notes"].append(
            "chain_colors for chain{0} {1} {2} not used: every circle and line drawn in {3} takes a fill or line_stroke set "
            "explicitly in its atom type's style.".format(
                "s" if len(unused) > 1 else "", ", ".join(chain_label(chain) for chain in unused),
                "were" if len(unused) > 1 else "was", "those chains" if len(unused) > 1 else "that chain",
            )
        )
    if chain_report["missing"]:
        metadata["notes"].append(
            "chain_colors names chains that are not in the selection and were not used: {0}.".format(
                ", ".join(chain_label(chain) for chain in chain_report["missing"])
            )
        )
    if chain_report["ignored"]:
        metadata["notes"].append("chain_colors is not used because color_by is atom-type.")
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def write_projection_csv(
    output_csv: Path,
    selected_atoms: Sequence[SelectedAtom],
    projected_xy: np.ndarray,
    projected_depth: np.ndarray,
    xy_only: bool,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if xy_only:
            writer.writerow(["proj_x", "proj_y"])
            for x2, y2 in projected_xy:
                writer.writerow(["{0:.10f}".format(x2), "{0:.10f}".format(y2)])
        else:
            writer.writerow([
                "selected_atom_type", "model", "record", "serial", "atom_name", "element", "altloc",
                "resname", "chain", "resseq", "icode", "line_number",
                "x", "y", "z", "proj_x", "proj_y", "projection_depth",
            ])
            for selected, (x2, y2), depth in zip(selected_atoms, projected_xy, projected_depth):
                atom = selected.atom
                writer.writerow([
                    selected.atom_type,
                    atom.model,
                    atom.record,
                    atom.serial,
                    atom.atom_name,
                    atom.element,
                    atom.altloc,
                    atom.resname,
                    atom.chain,
                    atom.resseq,
                    atom.icode,
                    atom.line_number,
                    "{0:.6f}".format(atom.x),
                    "{0:.6f}".format(atom.y),
                    "{0:.6f}".format(atom.z),
                    "{0:.10f}".format(x2),
                    "{0:.10f}".format(y2),
                    "{0:.10f}".format(depth),
                ])


def svg_escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def svg_float(value: float) -> str:
    return "{0:.6f}".format(float(value)).rstrip("0").rstrip(".")


def svg_id_part(text: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    if not safe:
        safe = fallback
    if not re.match(r"^[A-Za-z_]", safe):
        safe = "_" + safe
    return safe


def svg_group_id_for_chain(chain: str) -> str:
    return "chain_{0}".format(svg_id_part(chain_label(chain), "blank"))


def svg_group_id_for_type(atom_type: str) -> str:
    return "type_{0}".format(svg_id_part(atom_type, "atom"))


def compute_svg_transform(points_xy: np.ndarray, width: float, height: float, padding: float, invert_y: bool) -> Tuple[float, float, float]:
    min_x = float(np.min(points_xy[:, 0]))
    max_x = float(np.max(points_xy[:, 0]))
    min_y = float(np.min(points_xy[:, 1]))
    max_y = float(np.max(points_xy[:, 1]))
    span_x = max(max_x - min_x, 1.0e-12)
    span_y = max(max_y - min_y, 1.0e-12)
    inner_width = max(width - 2.0 * padding, 1.0)
    inner_height = max(height - 2.0 * padding, 1.0)
    scale = min(inner_width / span_x, inner_height / span_y)
    plot_width = span_x * scale
    plot_height = span_y * scale
    left = (width - plot_width) / 2.0
    top = (height - plot_height) / 2.0
    x_offset = left - min_x * scale
    if invert_y:
        y_offset = top + max_y * scale
    else:
        y_offset = top - min_y * scale
    return scale, x_offset, y_offset


def to_svg_xy(proj_x: float, proj_y: float, scale: float, x_offset: float, y_offset: float, invert_y: bool) -> Tuple[float, float]:
    svg_x = x_offset + proj_x * scale
    svg_y = y_offset - proj_y * scale if invert_y else y_offset + proj_y * scale
    return svg_x, svg_y


def atom_title(selected: SelectedAtom, proj_x: float, proj_y: float, depth: float) -> str:
    atom = selected.atom
    chain_text = atom.chain if atom.chain else "blank chain"
    residue = "{0} {1}{2}".format(atom.resname, atom.resseq, atom.icode).strip()
    return (
        "selected type {selected_type}; chain {chain}; model {model}; {residue}; atom {atom_name}; "
        "serial {serial}; proj_x {proj_x:.6f}; proj_y {proj_y:.6f}; depth {depth:.6f}"
    ).format(
        selected_type=selected.atom_type,
        chain=chain_text,
        model=atom.model,
        residue=residue,
        atom_name=atom.atom_name,
        serial=atom.serial,
        proj_x=proj_x,
        proj_y=proj_y,
        depth=depth,
    )


def atom_object_name(selected: SelectedAtom) -> str:
    atom = selected.atom
    chain = atom.chain.strip() or "blank"
    residue = "{0}{1}".format(atom.resseq.strip(), atom.icode.strip())
    atom_name = re.sub(r"\s+", "", atom.atom_name.strip())
    raw_name = "{0}{1}{2}".format(chain, residue, atom_name)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "", raw_name)
    if not name:
        name = "atom{0}".format(atom.line_number)
    if not re.match(r"^[A-Za-z_]", name):
        name = "atom_{0}".format(name)
    return name


def group_selected_atoms(
    selected_atoms: Sequence[SelectedAtom],
    projected_xy: np.ndarray,
    projected_depth: np.ndarray,
) -> Dict[str, Dict[str, List[Tuple[SelectedAtom, float, float, float]]]]:
    grouped: Dict[str, Dict[str, List[Tuple[SelectedAtom, float, float, float]]]] = {}
    for selected, (proj_x, proj_y), depth in zip(selected_atoms, projected_xy, projected_depth):
        grouped.setdefault(selected.atom.chain, {}).setdefault(selected.atom_type, []).append(
            (selected, float(proj_x), float(proj_y), float(depth))
        )
    return grouped


def sorted_chain_items(grouped: Dict[str, Dict[str, List[Tuple[SelectedAtom, float, float, float]]]]) -> List[Tuple[str, Dict[str, List[Tuple[SelectedAtom, float, float, float]]]]]:
    return list(grouped.items())


def entries_by_model(entries: Sequence[Tuple[SelectedAtom, float, float, float]]) -> Dict[int, List[Tuple[SelectedAtom, float, float, float]]]:
    grouped: Dict[int, List[Tuple[SelectedAtom, float, float, float]]] = {}
    for entry in entries:
        grouped.setdefault(entry[0].atom.model, []).append(entry)
    return grouped


def polyline_points(entries: Sequence[Tuple[SelectedAtom, float, float, float]], scale: float, x_offset: float, y_offset: float, invert_y: bool) -> List[str]:
    parts: List[str] = []
    for _selected, proj_x, proj_y, _depth in entries:
        svg_x, svg_y = to_svg_xy(proj_x, proj_y, scale, x_offset, y_offset, invert_y)
        parts.append("{0},{1}".format(svg_float(svg_x), svg_float(svg_y)))
    return parts




def normalize_connection_mode(value: str) -> str:
    """Normalize connection mode aliases to 'straight' or 'smooth'."""
    text = str(value).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'line': 'straight',
        'lines': 'straight',
        'straight_line': 'straight',
        'straight_lines': 'straight',
        'polyline': 'straight',
        'straight': 'straight',
        'curve': 'smooth',
        'curved': 'smooth',
        'smooth_curve': 'smooth',
        'spline': 'smooth',
        'bezier': 'smooth',
        'catmull_rom': 'smooth',
        'catmullrom': 'smooth',
        'smooth': 'smooth',
    }
    if text not in aliases:
        raise ValueError("Connection mode must be straight or smooth")
    return aliases[text]


def parse_chain_list(text: str) -> set:
    """Parse a comma/semicolon/whitespace-separated list of chain labels."""
    if not text:
        return set()
    labels = set()
    for part in re.split(r"[,;\s]+", text.strip()):
        if not part:
            continue
        if part.lower() in {"blank", "none", "_blank_"}:
            labels.add("blank")
            labels.add("")
        else:
            labels.add(part)
    return labels


def chain_is_closed(chain: str, args: argparse.Namespace) -> bool:
    if bool(getattr(args, "close_all_chains", False)):
        return True
    labels = parse_chain_list(getattr(args, "closed_chains", ""))
    return chain in labels or chain_label(chain) in labels


def connection_mode_for_atom_type(atom_type: str, styles: Dict[str, AtomStyle], default_mode: str) -> str:
    style = styles.get(atom_type.upper())
    if style is not None:
        return normalize_connection_mode(style.connection_mode)
    return normalize_connection_mode(default_mode)


def svg_points_from_entries(
    entries: Sequence[Tuple[SelectedAtom, float, float, float]],
    scale: float,
    x_offset: float,
    y_offset: float,
    invert_y: bool,
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for _selected, proj_x, proj_y, _depth in entries:
        points.append(to_svg_xy(proj_x, proj_y, scale, x_offset, y_offset, invert_y))
    return points


def catmull_bezier_control_points(
    points: Sequence[Tuple[float, float]],
    segment_index: int,
    closed: bool = False,
    tension: float = 1.0,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Return cubic Bezier points p0,c1,c2,p1 for a Catmull-Rom segment."""
    n_points = len(points)
    if n_points < 2:
        raise ValueError("At least two points are needed for a curve segment")
    if closed:
        segment_index = segment_index % n_points
        previous_point = points[(segment_index - 1) % n_points]
        current_point = points[segment_index]
        next_point = points[(segment_index + 1) % n_points]
        following_point = points[(segment_index + 2) % n_points]
    else:
        if segment_index < 0 or segment_index + 1 >= n_points:
            raise ValueError("Open curve segment index is out of range")
        previous_point = points[segment_index - 1] if segment_index > 0 else points[segment_index]
        current_point = points[segment_index]
        next_point = points[segment_index + 1]
        following_point = points[segment_index + 2] if segment_index + 2 < n_points else next_point

    factor = float(tension) / 6.0
    c1 = (
        current_point[0] + (next_point[0] - previous_point[0]) * factor,
        current_point[1] + (next_point[1] - previous_point[1]) * factor,
    )
    c2 = (
        next_point[0] - (following_point[0] - current_point[0]) * factor,
        next_point[1] - (following_point[1] - current_point[1]) * factor,
    )
    return current_point, c1, c2, next_point


def cubic_bezier_point(
    p0: Tuple[float, float],
    c1: Tuple[float, float],
    c2: Tuple[float, float],
    p1: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    u = 1.0 - t
    x = u * u * u * p0[0] + 3.0 * u * u * t * c1[0] + 3.0 * u * t * t * c2[0] + t * t * t * p1[0]
    y = u * u * u * p0[1] + 3.0 * u * u * t * c1[1] + 3.0 * u * t * t * c2[1] + t * t * t * p1[1]
    return x, y


def smooth_path_from_svg_points(points: Sequence[Tuple[float, float]], closed: bool = False, tension: float = 1.0) -> str:
    """Return a Catmull-Rom-style cubic Bezier SVG path through the points."""
    if not points:
        return ""
    if len(points) == 1:
        return "M {0},{1}".format(svg_float(points[0][0]), svg_float(points[0][1]))

    parts = ["M {0},{1}".format(svg_float(points[0][0]), svg_float(points[0][1]))]
    n_segments = len(points) if closed else len(points) - 1
    for index in range(n_segments):
        p0, c1, c2, p1 = catmull_bezier_control_points(points, index, closed=closed, tension=tension)
        parts.append(
            "C {0},{1} {2},{3} {4},{5}".format(
                svg_float(c1[0]),
                svg_float(c1[1]),
                svg_float(c2[0]),
                svg_float(c2[1]),
                svg_float(p1[0]),
                svg_float(p1[1]),
            )
        )
    return " ".join(parts)


def smooth_path_points(
    entries: Sequence[Tuple[SelectedAtom, float, float, float]],
    scale: float,
    x_offset: float,
    y_offset: float,
    invert_y: bool,
    closed: bool = False,
) -> str:
    return smooth_path_from_svg_points(svg_points_from_entries(entries, scale, x_offset, y_offset, invert_y), closed=closed)


def smooth_segment_path(
    segment: ConnectionSegment,
    scale: float,
    x_offset: float,
    y_offset: float,
    invert_y: bool,
) -> str:
    """Return one cubic segment from the same Catmull-Rom curve used for a full group.

    If a terminal O3' extension follows the last selected atom, append that
    O3' point to the control-point context for the last selected-selected
    segment. This makes the regular final segment and the O3' extension share
    a continuous tangent when both are drawn as smooth paths.
    """
    points = svg_points_from_entries(segment.entries, scale, x_offset, y_offset, invert_y)
    if segment.terminal_extension_point is not None and not segment.closed:
        end_x, end_y, _end_depth, _end_line, _end_atom_name = segment.terminal_extension_point
        points.append(to_svg_xy(float(end_x), float(end_y), scale, x_offset, y_offset, invert_y))
    p0, c1, c2, p1 = catmull_bezier_control_points(points, segment.index, closed=segment.closed, tension=1.0)
    return "M {0},{1} C {2},{3} {4},{5} {6},{7}".format(
        svg_float(p0[0]),
        svg_float(p0[1]),
        svg_float(c1[0]),
        svg_float(c1[1]),
        svg_float(c2[0]),
        svg_float(c2[1]),
        svg_float(p1[0]),
        svg_float(p1[1]),
    )


def project_arbitrary_points(points: np.ndarray, projection: ProjectionResult) -> Tuple[np.ndarray, np.ndarray]:
    transformed = transform_points_to_projection_basis(points, projection)
    return transformed[:, :2].copy(), transformed[:, 2].copy()


def build_xy_plane_patch(
    selected_atoms: Sequence[SelectedAtom],
    projection: ProjectionResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return input, projected, and depth coordinates for the projection-basis xy plane.

    The mathematical projection-basis xy plane is infinite, so the SVG layer
    draws a finite depth=0 patch spanning selected atoms' projected x/y bounds
    with a small margin. In PCA mode, this is the PC1/PC2 plane through the
    selected-atom centroid. In current-xy mode, this is the current coordinate
    xy plane after any optional pre-projection transform.
    """
    if not selected_atoms:
        raise ValueError("Cannot draw xy plane without selected atoms or points")

    xy = np.asarray(projection.projected_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
        raise ValueError("Cannot draw xy plane without projected points")
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    x_span = float(x_max - x_min)
    y_span = float(y_max - y_min)

    fallback_span = max(x_span, y_span, 1.0)
    if x_span <= 1.0e-12:
        x_min -= 0.5 * fallback_span
        x_max += 0.5 * fallback_span
        x_span = float(x_max - x_min)
    if y_span <= 1.0e-12:
        y_min -= 0.5 * fallback_span
        y_max += 0.5 * fallback_span
        y_span = float(y_max - y_min)

    x_margin = max(x_span * XY_PLANE_MARGIN_FRACTION, 1.0e-6)
    y_margin = max(y_span * XY_PLANE_MARGIN_FRACTION, 1.0e-6)
    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin

    basis_corners = np.array(
        [
            [x_min, y_min, 0.0],
            [x_max, y_min, 0.0],
            [x_max, y_max, 0.0],
            [x_min, y_max, 0.0],
        ],
        dtype=float,
    )
    input_corners = transform_points_from_projection_basis(basis_corners, projection)
    projected_corners = basis_corners[:, :2].copy()
    depths = basis_corners[:, 2].copy()
    return input_corners, projected_corners, depths


def default_dssr_output_path(pdb_file: Path) -> Path:
    return pdb_file.parent / "tmp_file" / (pdb_file.name + ".out")


def residue_tuple_from_atom(atom: AtomRecord) -> Tuple[str, str, str]:
    """Return a chain/resseq/icode tuple for LINK-based chain closure checks."""
    return (atom.chain, atom.resseq, atom.icode)


def parse_link_residue_pairs(pdb_file: Path) -> List[Tuple[Tuple[str, str, str], Tuple[str, str, str]]]:
    """Parse PDB LINK records and return residue pairs as (chain, resseq, icode)."""
    pairs: List[Tuple[Tuple[str, str, str], Tuple[str, str, str]]] = []
    if not pdb_file.exists():
        return pairs
    with pdb_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line[0:6].strip().upper() != "LINK":
                continue
            chain1 = _safe_slice(line, 21, 22).strip()
            resseq1 = _safe_slice(line, 22, 26).strip()
            icode1 = _safe_slice(line, 26, 27).strip()
            chain2 = _safe_slice(line, 51, 52).strip()
            resseq2 = _safe_slice(line, 52, 56).strip()
            icode2 = _safe_slice(line, 56, 57).strip()

            # Some LINK records produced by modeling tools are slightly shifted
            # from the strict PDB fixed-column format. If the fixed-column chain
            # IDs are blank but token parsing is clear, use the tokenized form:
            # LINK atom1 resname1 chain1 resseq1 atom2 resname2 chain2 resseq2 ...
            tokens = line.split()
            if len(tokens) >= 9 and (not chain1 and not chain2):
                if tokens[0].upper() == "LINK":
                    chain1, resseq1, icode1 = tokens[3], tokens[4], ""
                    chain2, resseq2, icode2 = tokens[7], tokens[8], ""

            if not resseq1 or not resseq2:
                continue
            pairs.append(((chain1, resseq1, icode1), (chain2, resseq2, icode2)))
    return pairs


def detect_closed_chains_from_link(pdb_file: Path, atoms: Optional[Sequence[AtomRecord]] = None) -> List[str]:
    """Detect chains whose first and last residues are directly connected by a LINK record.

    The function compares LINK-record residue pairs with the first and last
    coordinate residues observed in each chain. It returns chain IDs in the
    order they first appear in the PDB. A blank chain is returned as an empty
    string; use chain_label() when displaying it.
    """
    if atoms is None:
        try:
            atoms = parse_pdb_atoms(
                pdb_file=pdb_file,
                records="all",
                model="first",
                chain=None,
                resname=None,
                altloc="all",
            )
        except Exception:
            return []

    residues_by_chain: Dict[str, List[Tuple[str, str, str]]] = {}
    seen_by_chain: Dict[str, set] = {}
    for atom in atoms:
        residue = residue_tuple_from_atom(atom)
        chain = residue[0]
        if chain not in residues_by_chain:
            residues_by_chain[chain] = []
            seen_by_chain[chain] = set()
        if residue not in seen_by_chain[chain]:
            residues_by_chain[chain].append(residue)
            seen_by_chain[chain].add(residue)

    terminal_pairs: Dict[str, Tuple[Tuple[str, str, str], Tuple[str, str, str]]] = {}
    chain_order: List[str] = []
    for chain, residues in residues_by_chain.items():
        if len(residues) < 2:
            continue
        terminal_pairs[chain] = (residues[0], residues[-1])
        chain_order.append(chain)

    link_pairs = parse_link_residue_pairs(pdb_file)
    closed = set()
    for first, second in link_pairs:
        chain1, _resseq1, _icode1 = first
        chain2, _resseq2, _icode2 = second
        if chain1 != chain2:
            continue
        terminals = terminal_pairs.get(chain1)
        if terminals is None:
            continue
        start, end = terminals
        if (first == start and second == end) or (first == end and second == start):
            closed.add(chain1)

    return [chain for chain in chain_order if chain in closed]


def format_chain_list(chains: Sequence[str]) -> str:
    """Format chain IDs for the --closed-chains text field."""
    labels = [chain_label(chain) for chain in chains]
    return ",".join(labels)


def merge_closed_chain_text(existing: str, detected_chains: Sequence[str]) -> str:
    """Merge user-entered closed chains with LINK-detected chains."""
    labels = parse_chain_list(existing)
    ordered: List[str] = []
    seen = set()
    for part in re.split(r"[,;\s]+", existing.strip() if existing else ""):
        if not part:
            continue
        label = "blank" if part.lower() in {"blank", "none", "_blank_"} else part
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    for chain in detected_chains:
        label = chain_label(chain)
        if chain in labels or label in labels:
            continue
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    return ",".join(ordered)


def ensure_dssr_output(pdb_file: Path, args: argparse.Namespace) -> Path:
    """Return the default DSSR output path, running x3dna-dssr if needed."""
    output_path = default_dssr_output_path(pdb_file)
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_path = pdb_file.resolve()
    output_path = output_path.resolve()
    executable = "x3dna-dssr"
    command = [executable, "-i={0}".format(str(input_path)), "--more", "-o={0}".format(str(output_path))]
    try:
        completed = subprocess.run(
            command,
            cwd=str(output_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "Could not find 'x3dna-dssr'. Install x3dna-dssr, or place an existing DSSR output file at the default path: {0}".format(output_path)
        ) from exc
    if completed.returncode != 0:
        raise ValueError(
            "x3dna-dssr failed with return code {0}. stderr:\n{1}".format(completed.returncode, completed.stderr.strip())
        )
    if not output_path.exists():
        raise ValueError("x3dna-dssr finished but the expected output file was not created: {0}".format(output_path))
    return output_path


def parse_dssr_base_pairs(dssr_output: Path) -> List[DssrBasePair]:
    lines = dssr_output.read_text(encoding="utf-8", errors="replace").splitlines()
    start_index = None
    for i, line in enumerate(lines):
        if re.search(r"List of\s+\d+\s+base pairs", line):
            start_index = i + 1
            break
    if start_index is None:
        raise ValueError("Could not find the 'List of ... base pairs' section in DSSR output: {0}".format(dssr_output))

    base_pairs: List[DssrBasePair] = []
    for line in lines[start_index:]:
        if line.startswith("****************************************************************************"):
            break
        match = re.match(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(.*)$", line)
        if not match:
            continue
        index = int(match.group(1))
        nt1 = match.group(2)
        nt2 = match.group(3)
        rest_tokens = match.group(4).split()
        bp = rest_tokens[0] if len(rest_tokens) >= 1 else ""
        name = rest_tokens[1] if len(rest_tokens) >= 2 else ""
        saenger = rest_tokens[2] if len(rest_tokens) >= 3 else ""
        lw = rest_tokens[3] if len(rest_tokens) >= 4 else ""
        dssr = rest_tokens[4] if len(rest_tokens) >= 5 else ""
        base_pairs.append(
            DssrBasePair(
                index=index,
                nt1=nt1,
                nt2=nt2,
                bp=bp,
                name=name,
                saenger=saenger,
                lw=lw,
                dssr=dssr,
                source_line=line.rstrip(),
            )
        )
    return base_pairs


def parse_dssr_nt_token(token: str) -> Optional[Tuple[str, str, str, str]]:
    """Return chain, resname, resseq, icode from a DSSR nucleotide token such as G.DT42."""
    cleaned = token.strip().strip(",;[]()")
    if not cleaned:
        return None
    if "." in cleaned:
        chain, residue_text = cleaned.split(".", 1)
    else:
        chain, residue_text = "", cleaned
    residue_text = residue_text.strip()
    match = re.search(r"(-?\d+)([A-Za-z]?)$", residue_text)
    if not match:
        return None
    resname = residue_text[: match.start()].strip()
    resseq = match.group(1).strip()
    icode = match.group(2).strip()
    return chain, resname, resseq, icode


def residue_key_from_atom(atom: AtomRecord) -> ResidueKey:
    return ResidueKey(model=atom.model, chain=atom.chain, resseq=atom.resseq, icode=atom.icode)


def residue_key_sort_key(key: ResidueKey) -> Tuple[int, str, int, str]:
    try:
        resseq_int = int(key.resseq)
    except ValueError:
        resseq_int = 0
    return (key.model, key.chain, resseq_int, key.icode)


DEFAULT_BASE_PAIR_ATOM = "C3'"


def normalize_base_pair_atom_name(atom_name: str) -> str:
    """Normalize a base-pair anchor atom name, accepting legacy * primes."""
    return (atom_name or DEFAULT_BASE_PAIR_ATOM).strip().upper().replace("*", "'").replace("`", "'")


def base_pair_atom_matches(atom: AtomRecord, anchor_atom_name: str) -> bool:
    return normalize_base_pair_atom_name(atom.atom_name) == normalize_base_pair_atom_name(anchor_atom_name)


def build_base_pair_anchor_points(
    all_atoms: Sequence[AtomRecord],
    projection: ProjectionResult,
    anchor_atom_name: str,
) -> Dict[ResidueKey, Tuple[float, float, float, int]]:
    """Map each residue to the projected position of the selected anchor atom."""
    anchor_atoms = [atom for atom in all_atoms if base_pair_atom_matches(atom, anchor_atom_name)]
    points: Dict[ResidueKey, Tuple[float, float, float, int]] = {}
    if not anchor_atoms:
        return points

    xyz = np.array([[atom.x, atom.y, atom.z] for atom in anchor_atoms], dtype=float)
    xy, depths = project_arbitrary_points(xyz, projection)
    for atom, (proj_x, proj_y), depth in zip(anchor_atoms, xy, depths):
        key = residue_key_from_atom(atom)
        # Keep the first matching anchor encountered for a residue. The altloc filter has
        # already been applied while reading atoms, so this normally avoids
        # duplicate alternate conformations.
        if key not in points:
            points[key] = (float(proj_x), float(proj_y), float(depth), atom.line_number)
    return points




def o3_prime_atom_matches(atom: AtomRecord) -> bool:
    """Return True for nucleic-acid O3' atoms, accepting legacy O3*."""
    name = atom.atom_name.strip().upper()
    return name in {"O3'", "O3*"}


def build_o3_prime_points(
    all_atoms: Sequence[AtomRecord],
    projection: ProjectionResult,
) -> Dict[ResidueKey, Tuple[float, float, float, int, str]]:
    """Map each residue to the projected position of its O3' atom."""
    o3_atoms = [atom for atom in all_atoms if o3_prime_atom_matches(atom)]
    points: Dict[ResidueKey, Tuple[float, float, float, int, str]] = {}
    if not o3_atoms:
        return points

    xyz = np.array([[atom.x, atom.y, atom.z] for atom in o3_atoms], dtype=float)
    xy, depths = project_arbitrary_points(xyz, projection)
    for atom, (proj_x, proj_y), depth in zip(o3_atoms, xy, depths):
        key = residue_key_from_atom(atom)
        if key not in points:
            points[key] = (float(proj_x), float(proj_y), float(depth), atom.line_number, atom.atom_name)
    return points


def find_o3_prime_point_for_entry(
    entry: Tuple[SelectedAtom, float, float, float],
    o3_points: Dict[ResidueKey, Tuple[float, float, float, int, str]],
) -> Optional[Tuple[float, float, float, int, str]]:
    selected = entry[0]
    key = residue_key_from_atom(selected.atom)
    return o3_points.get(key)

def find_base_pair_anchor_point(
    nt_token: str,
    anchor_points: Dict[ResidueKey, Tuple[float, float, float, int]],
) -> Optional[Tuple[ResidueKey, Tuple[float, float, float, int]]]:
    parsed = parse_dssr_nt_token(nt_token)
    if parsed is None:
        return None
    chain, _resname, resseq, icode = parsed
    candidates: List[Tuple[ResidueKey, Tuple[float, float, float, int]]] = []
    for key, point in anchor_points.items():
        if key.chain != chain:
            continue
        if key.resseq != resseq:
            continue
        if icode and key.icode != icode:
            continue
        candidates.append((key, point))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (residue_key_sort_key(item[0]), item[1][3]))
    return candidates[0]


def prepare_base_pair_drawables(
    pdb_file: Path,
    args: argparse.Namespace,
    all_atoms: Sequence[AtomRecord],
    projection: ProjectionResult,
    styles: Dict[str, AtomStyle],
) -> Tuple[List[BasePairDrawable], dict]:
    if not bool(getattr(args, "draw_base_pairs", False)):
        return [], {
            "enabled": False,
            "dssr_output": None,
            "base_pairs_in_dssr": 0,
            "base_pairs_drawn": 0,
            "base_pairs_skipped": 0,
            "skipped_examples": [],
        }

    anchor_atom_name = normalize_base_pair_atom_name(getattr(args, "base_pair_atom", DEFAULT_BASE_PAIR_ATOM))
    dssr_output = ensure_dssr_output(pdb_file, args)
    base_pairs = parse_dssr_base_pairs(dssr_output)
    anchor_points = build_base_pair_anchor_points(all_atoms, projection, anchor_atom_name)
    drawables: List[BasePairDrawable] = []
    skipped: List[str] = []
    for bp in base_pairs:
        first = find_base_pair_anchor_point(bp.nt1, anchor_points)
        second = find_base_pair_anchor_point(bp.nt2, anchor_points)
        if first is None or second is None:
            if len(skipped) < 10:
                missing = []
                if first is None:
                    missing.append(bp.nt1)
                if second is None:
                    missing.append(bp.nt2)
                skipped.append("base pair {0}: missing {1} atom for {2}".format(bp.index, anchor_atom_name, ", ".join(missing)))
            continue
        key1, point1 = first
        key2, point2 = second
        drawables.append(
            BasePairDrawable(
                base_pair=bp,
                key1=key1,
                key2=key2,
                x1=point1[0],
                y1=point1[1],
                depth1=point1[2],
                x2=point2[0],
                y2=point2[1],
                depth2=point2[2],
                segment_line1=point1[3],
                segment_line2=point2[3],
            )
        )

    info = {
        "enabled": True,
        "dssr_output": str(dssr_output),
        "dssr_default_output_folder": str(default_dssr_output_path(pdb_file).parent),
        "base_pairs_in_dssr": len(base_pairs),
        "base_pairs_drawn": len(drawables),
        "base_pairs_skipped": len(base_pairs) - len(drawables),
        "skipped_examples": skipped,
        "anchor_atom_name": anchor_atom_name,
        "anchor_definition": "projected {0} atom of each DSSR-listed residue".format(anchor_atom_name),
    }
    return drawables, info


def write_projection_svg(
    output_svg: Path,
    args: argparse.Namespace,
    atom_types: Sequence[str],
    selected_atoms: Sequence[SelectedAtom],
    projection: ProjectionResult,
    styles: Dict[str, AtomStyle],
    base_pair_drawables: Optional[Sequence[BasePairDrawable]] = None,
    all_atoms: Optional[Sequence[AtomRecord]] = None,
) -> float:
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    projected_xy = projection.projected_xy
    projected_depth = projection.projected_depth
    base_pair_drawables = list(base_pair_drawables or [])

    width = float(args.width)
    height = float(args.height)
    padding = float(args.padding)
    invert_y = not args.no_invert_y
    draw_scale_bar = not bool(getattr(args, "no_scale_bar", False))
    scale_bar_length = float(getattr(args, "scale_bar_length", DEFAULT_SCALE_BAR_LENGTH))
    scale_bar_unit_label = str(getattr(args, "scale_bar_unit_label", DEFAULT_SCALE_BAR_UNIT_LABEL) or DEFAULT_SCALE_BAR_UNIT_LABEL)
    scale_bar_stroke = getattr(args, "scale_bar_stroke", DEFAULT_SCALE_BAR_STROKE)
    scale_bar_stroke_width = float(getattr(args, "scale_bar_stroke_width", DEFAULT_SCALE_BAR_STROKE_WIDTH))
    scale_bar_text_size = float(getattr(args, "scale_bar_text_size", DEFAULT_SCALE_BAR_TEXT_SIZE))
    scale_bar_margin = float(getattr(args, "scale_bar_margin", DEFAULT_SCALE_BAR_MARGIN))
    scale_bar_background = getattr(args, "scale_bar_background", DEFAULT_SCALE_BAR_BACKGROUND)
    scale_bar_background_opacity = float(getattr(args, "scale_bar_background_opacity", DEFAULT_SCALE_BAR_BACKGROUND_OPACITY))
    draw_xy_plane = bool(getattr(args, "draw_xy_plane", False))
    xy_plane_raw_corners = None
    xy_plane_projected_corners = None
    xy_plane_depths = None
    transform_points = projected_xy
    if draw_xy_plane:
        xy_plane_raw_corners, xy_plane_projected_corners, xy_plane_depths = build_xy_plane_patch(selected_atoms, projection)
        transform_points = np.vstack([projected_xy, xy_plane_projected_corners])

    scale, x_offset, y_offset = compute_svg_transform(transform_points, width, height, padding, invert_y)
    grouped = group_selected_atoms(selected_atoms, projected_xy, projected_depth)
    selected_entries = [
        (selected, float(proj_x), float(proj_y), float(depth))
        for selected, (proj_x, proj_y), depth in zip(selected_atoms, projected_xy, projected_depth)
    ]
    o3_prime_points = build_o3_prime_points(all_atoms or [], projection)

    color_by = getattr(args, "color_by", "chain")
    if color_by not in {"atom-type", "chain"}:
        raise ValueError("--color-by must be 'atom-type' or 'chain'")
    depth_order_circles = bool(args.depth_order_circles or getattr(args, "pdb_order_circles", False))
    depth_order_lines = bool(args.depth_order_lines or getattr(args, "pdb_order_lines", False))
    depth_order_base_pairs = bool(getattr(args, "depth_order_base_pairs", False)) and bool(base_pair_drawables)
    depth_front = getattr(args, "depth_front", "positive")
    if depth_front not in {"positive", "negative"}:
        raise ValueError("--depth-front must be 'positive' or 'negative'")
    line_underlay = bool(getattr(args, "line_underlay", False)) and depth_order_lines
    palette = palette_from_args(args)

    # Chain mode: explicit atom-type fill/line_stroke, then --chain-colors, then
    # the palette color for the chain's place in first-appearance order.
    chain_order = [chain for chain, _type_groups in sorted_chain_items(grouped)]
    chain_overrides = chain_color_overrides(args) if color_by == "chain" else {}
    chain_colors = assign_chain_colors(chain_order, palette, chain_overrides)
    explicit_keys = getattr(args, "explicit_style_keys", None) or {}
    # The chains whose chain color a drawn circle or line takes, gathered as
    # the elements are written: a --chain-colors color counts as used only
    # there, not in a chain whose every element has an explicit color.
    chain_color_takers: set = set()

    def takes_chain_color(atom_type: str, key: str) -> bool:
        return color_by == "chain" and key not in explicit_keys.get(atom_type.upper(), set())

    def point_fill_for(selected: SelectedAtom) -> str:
        style = styles[selected.atom_type.upper()]
        if not takes_chain_color(selected.atom_type, "fill"):
            return style.fill
        chain_color_takers.add(selected.atom.chain)
        return chain_colors[selected.atom.chain]

    def line_color_for(chain: str, atom_type: str) -> str:
        style = styles[atom_type.upper()]
        if not takes_chain_color(atom_type, "line_stroke"):
            return style.line_stroke
        chain_color_takers.add(chain)
        return chain_colors[chain]

    def line_stroke_for(selected: SelectedAtom) -> str:
        return line_color_for(selected.atom.chain, selected.atom_type)

    def depth_sort_value(depth: float) -> float:
        return depth if depth_front == "positive" else -depth

    def selected_entry_key(entry: Tuple[SelectedAtom, float, float, float]) -> Tuple[int, int, str]:
        selected = entry[0]
        return (selected.atom.model, selected.atom.line_number, selected.atom_type.upper())

    def circle_depth_sort_key(entry: Tuple[SelectedAtom, float, float, float]) -> Tuple[float, int, int]:
        selected = entry[0]
        # Sort tuple layout is (depth_score, element_layer, source_order).
        # Element layer keeps circles after line/base-pair elements at the same depth.
        return (depth_sort_value(entry[3]), 2, selected.atom.line_number)

    def segment_endpoints(segment: ConnectionSegment):
        first = segment.entries[segment.index]
        second = segment.entries[(segment.index + 1) % len(segment.entries)]
        return first, second

    def segment_depth(segment: ConnectionSegment) -> float:
        first, second = segment_endpoints(segment)
        return 0.5 * (first[3] + second[3])

    def segment_depth_sort_key(segment: ConnectionSegment) -> Tuple[float, int, int]:
        first, second = segment_endpoints(segment)
        order_line = max(first[0].atom.line_number, second[0].atom.line_number)
        return (depth_sort_value(segment_depth(segment)), 0, order_line)

    def extension_depth(extension: TerminalExtensionSegment) -> float:
        return 0.5 * (extension.start_entry[3] + extension.end_depth)

    def extension_depth_sort_key(extension: TerminalExtensionSegment) -> Tuple[float, int, int]:
        start_line = extension.start_entry[0].atom.line_number
        order_line = max(start_line, extension.end_line_number)
        return (depth_sort_value(extension_depth(extension)), 0, order_line)

    def extension_endpoint_key(extension: TerminalExtensionSegment) -> Tuple[int, int, str]:
        return selected_entry_key(extension.start_entry)

    def base_pair_depth(drawable: BasePairDrawable) -> float:
        return 0.5 * (drawable.depth1 + drawable.depth2)

    def base_pair_depth_sort_key(drawable: BasePairDrawable) -> Tuple[float, int, int]:
        # Use the frontmost endpoint for sorting so the white underlay of either
        # corresponding neighbor segment is drawn before this base-pair line.
        # This prevents an underlay from hiding the base-pair line at its own anchors.
        order_line = max(drawable.segment_line1, drawable.segment_line2)
        frontmost_score = max(depth_sort_value(drawable.depth1), depth_sort_value(drawable.depth2))
        return (frontmost_score, 1, order_line)

    def xy_plane_depth_mean() -> float:
        if xy_plane_depths is None or len(xy_plane_depths) == 0:
            return 0.0
        return float(np.mean(xy_plane_depths))

    def xy_plane_depth_sort_key() -> Tuple[float, int, int]:
        # The projection-basis xy-plane is drawn at depth 0; keep the mean-based
        # form so the ordering code stays parallel with other drawable items.
        return (depth_sort_value(xy_plane_depth_mean()), 0, -1)

    circle_id_counts: Dict[str, int] = {}
    segment_id_counts: Dict[str, int] = {}

    def unique_svg_object_id(object_name: str, counts: Dict[str, int]) -> str:
        count = counts.get(object_name, 0)
        counts[object_name] = count + 1
        return object_name if count == 0 else "{0}_{1}".format(object_name, count + 1)

    def circle_lines(selected: SelectedAtom, proj_x: float, proj_y: float, depth: float, indent: str) -> List[str]:
        atom = selected.atom
        style = styles[selected.atom_type.upper()]
        svg_x, svg_y = to_svg_xy(proj_x, proj_y, scale, x_offset, y_offset, invert_y)
        title = atom_title(selected, proj_x, proj_y, depth)
        object_name = atom_object_name(selected)
        object_id = unique_svg_object_id(object_name, circle_id_counts)
        out: List[str] = []
        out.append(
            indent + '<circle id="{object_id}" class="point" cx="{cx}" cy="{cy}" r="{r}" '
            'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}" '
            'inkscape:label="{object_name}" data-name="{object_name}" '
            'data-selected-atom-type="{selected_type}" data-model="{model}" data-record="{record}" '
            'data-serial="{serial}" data-atom-name="{atom_name}" data-element="{element}" '
            'data-altloc="{altloc}" data-resname="{resname}" data-chain="{chain}" '
            'data-resseq="{resseq}" data-icode="{icode}" data-line-number="{line_number}" '
            'data-x="{x}" data-y="{y}" data-z="{z}" data-proj-x="{proj_x}" data-proj-y="{proj_y}" '
            'data-projection-depth="{depth}" data-projection-mode="{mode}" data-depth-front="{depth_front}">'.format(
                cx=svg_float(svg_x),
                cy=svg_float(svg_y),
                r=svg_float(style.radius),
                object_id=svg_escape(object_id),
                object_name=svg_escape(object_name),
                fill=svg_escape(point_fill_for(selected)),
                stroke=svg_escape(style.stroke),
                stroke_width=svg_float(style.stroke_width),
                opacity=svg_float(style.opacity),
                selected_type=svg_escape(selected.atom_type),
                model=svg_escape(atom.model),
                record=svg_escape(atom.record),
                serial=svg_escape(atom.serial),
                atom_name=svg_escape(atom.atom_name),
                element=svg_escape(atom.element),
                altloc=svg_escape(atom.altloc),
                resname=svg_escape(atom.resname),
                chain=svg_escape(atom.chain),
                resseq=svg_escape(atom.resseq),
                icode=svg_escape(atom.icode),
                line_number=svg_escape(atom.line_number),
                x=svg_float(atom.x),
                y=svg_float(atom.y),
                z=svg_float(atom.z),
                proj_x=svg_float(proj_x),
                proj_y=svg_float(proj_y),
                depth=svg_float(depth),
                mode=svg_escape(projection.mode),
                depth_front=svg_escape(depth_front),
            )
        )
        out.append(indent + "  <title>{0}</title>".format(svg_escape(title)))
        out.append(indent + "</circle>")
        return out

    def connection_geometry_lines(segment: ConnectionSegment, indent: str, underlay: bool = False) -> List[str]:
        first, second = segment_endpoints(segment)
        selected1, proj_x1, proj_y1, depth1 = first
        selected2, proj_x2, proj_y2, depth2 = second
        style = styles[selected2.atom_type.upper()]
        atom1 = selected1.atom
        atom2 = selected2.atom
        seg_depth = 0.5 * (depth1 + depth2)
        label = chain_label(atom2.chain)
        out: List[str] = []
        stroke = getattr(args, "line_underlay_stroke", "#ffffff") if underlay else line_stroke_for(selected2)
        width_value = style.line_width + float(getattr(args, "line_underlay_extra_width", 8.0)) if underlay else style.line_width
        opacity_value = float(getattr(args, "line_underlay_opacity", 1.0)) if underlay else style.line_opacity
        class_name = "neighbor-line-underlay" if underlay else "neighbor-line"
        mode = normalize_connection_mode(segment.connection_mode)
        segment_name = "{0}_{1}".format(atom_object_name(selected1), atom_object_name(selected2))
        object_name = "{0}_underlay".format(segment_name) if underlay else segment_name
        object_id = unique_svg_object_id(object_name, segment_id_counts)

        if mode == "smooth":
            d_attr = smooth_segment_path(segment, scale, x_offset, y_offset, invert_y)
            out.append(
                indent + '<path id="{object_id}" class="{class_name} smooth-curve" d="{d}" '
                'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" '
                'inkscape:label="{object_name}" data-name="{object_name}" '
                'data-connection-mode="smooth" data-chain-closed="{closed}" data-model="{model}" data-chain="{chain}" data-atom-type="{atom_type}" '
                'data-from-serial="{from_serial}" data-to-serial="{to_serial}" '
                'data-from-line-number="{from_line}" data-to-line-number="{to_line}" '
                'data-from-depth="{from_depth}" data-to-depth="{to_depth}" data-segment-depth="{segment_depth}" '
                'data-projection-mode="{mode_name}" data-depth-front="{depth_front}">'.format(
                    object_id=svg_escape(object_id),
                    object_name=svg_escape(object_name),
                    class_name=svg_escape(class_name),
                    d=svg_escape(d_attr),
                    stroke=svg_escape(stroke),
                    width=svg_float(width_value),
                    opacity=svg_float(opacity_value),
                    closed=str(segment.closed).lower(),
                    model=svg_escape(atom2.model),
                    chain=svg_escape(atom2.chain),
                    atom_type=svg_escape(selected2.atom_type),
                    from_serial=svg_escape(atom1.serial),
                    to_serial=svg_escape(atom2.serial),
                    from_line=svg_escape(atom1.line_number),
                    to_line=svg_escape(atom2.line_number),
                    from_depth=svg_float(depth1),
                    to_depth=svg_float(depth2),
                    segment_depth=svg_float(seg_depth),
                    mode_name=svg_escape(projection.mode),
                    depth_front=svg_escape(depth_front),
                )
            )
            title_prefix = "Neighbor smooth-curve underlay segment" if underlay else "Neighbor smooth-curve segment"
            close_tag = "</path>"
        else:
            x1, y1 = to_svg_xy(proj_x1, proj_y1, scale, x_offset, y_offset, invert_y)
            x2, y2 = to_svg_xy(proj_x2, proj_y2, scale, x_offset, y_offset, invert_y)
            out.append(
                indent + '<line id="{object_id}" class="{class_name} straight-line" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" '
                'inkscape:label="{object_name}" data-name="{object_name}" '
                'data-connection-mode="straight" data-chain-closed="{closed}" data-model="{model}" data-chain="{chain}" data-atom-type="{atom_type}" '
                'data-from-serial="{from_serial}" data-to-serial="{to_serial}" '
                'data-from-line-number="{from_line}" data-to-line-number="{to_line}" '
                'data-from-depth="{from_depth}" data-to-depth="{to_depth}" data-segment-depth="{segment_depth}" '
                'data-projection-mode="{mode_name}" data-depth-front="{depth_front}">'.format(
                    object_id=svg_escape(object_id),
                    object_name=svg_escape(object_name),
                    class_name=svg_escape(class_name),
                    x1=svg_float(x1),
                    y1=svg_float(y1),
                    x2=svg_float(x2),
                    y2=svg_float(y2),
                    stroke=svg_escape(stroke),
                    width=svg_float(width_value),
                    opacity=svg_float(opacity_value),
                    closed=str(segment.closed).lower(),
                    model=svg_escape(atom2.model),
                    chain=svg_escape(atom2.chain),
                    atom_type=svg_escape(selected2.atom_type),
                    from_serial=svg_escape(atom1.serial),
                    to_serial=svg_escape(atom2.serial),
                    from_line=svg_escape(atom1.line_number),
                    to_line=svg_escape(atom2.line_number),
                    from_depth=svg_float(depth1),
                    to_depth=svg_float(depth2),
                    segment_depth=svg_float(seg_depth),
                    mode_name=svg_escape(projection.mode),
                    depth_front=svg_escape(depth_front),
                )
            )
            title_prefix = "Neighbor straight underlay segment" if underlay else "Neighbor straight segment"
            close_tag = "</line>"

        out.append(
            indent + "  <title>{0}: consecutive selected {1} atoms in chain {2}; serial {3} to {4}</title>".format(
                svg_escape(title_prefix), svg_escape(selected2.atom_type), svg_escape(label), svg_escape(atom1.serial), svg_escape(atom2.serial)
            )
        )
        out.append(indent + close_tag)
        return out

    def connection_segment_lines(segment: ConnectionSegment, indent: str) -> List[str]:
        out: List[str] = []
        if line_underlay:
            out.extend(connection_geometry_lines(segment, indent, underlay=True))
        out.extend(connection_geometry_lines(segment, indent, underlay=False))
        return out

    def extension_geometry_lines(extension: TerminalExtensionSegment, indent: str, underlay: bool = False) -> List[str]:
        selected, proj_x1, proj_y1, depth1 = extension.start_entry
        atom = selected.atom
        style = styles[selected.atom_type.upper()]
        x1, y1 = to_svg_xy(proj_x1, proj_y1, scale, x_offset, y_offset, invert_y)
        x2, y2 = to_svg_xy(extension.end_x, extension.end_y, scale, x_offset, y_offset, invert_y)
        seg_depth = extension_depth(extension)
        stroke = getattr(args, "line_underlay_stroke", "#ffffff") if underlay else line_stroke_for(selected)
        width_value = style.line_width + float(getattr(args, "line_underlay_extra_width", 8.0)) if underlay else style.line_width
        opacity_value = float(getattr(args, "line_underlay_opacity", 1.0)) if underlay else style.line_opacity
        class_name = "neighbor-line-underlay terminal-extension-underlay" if underlay else "neighbor-line terminal-extension"
        title_prefix = "3-prime O3' extension underlay" if underlay else "3-prime O3' extension"
        mode = normalize_connection_mode(extension.connection_mode)
        out: List[str] = []

        if mode == "smooth" and extension.previous_entry is not None:
            prev_x, prev_y = to_svg_xy(extension.previous_entry[1], extension.previous_entry[2], scale, x_offset, y_offset, invert_y)
            control_points = [(prev_x, prev_y), (x1, y1), (x2, y2)]
            p0, c1, c2, p1 = catmull_bezier_control_points(control_points, 1, closed=False, tension=1.0)
            d_attr = "M {0},{1} C {2},{3} {4},{5} {6},{7}".format(
                svg_float(p0[0]), svg_float(p0[1]),
                svg_float(c1[0]), svg_float(c1[1]),
                svg_float(c2[0]), svg_float(c2[1]),
                svg_float(p1[0]), svg_float(p1[1]),
            )
            out.append(
                indent + '<path class="{class_name} smooth-curve" d="{d}" '
                'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" '
                'data-connection-mode="terminal-3prime-smooth" data-chain-closed="false" data-model="{model}" data-chain="{chain}" data-atom-type="{atom_type}" '
                'data-from-serial="{from_serial}" data-from-line-number="{from_line}" data-to-atom-name="{to_atom_name}" data-to-line-number="{to_line}" '
                'data-from-depth="{from_depth}" data-to-depth="{to_depth}" data-segment-depth="{segment_depth}" '
                'data-projection-mode="{mode_name}" data-depth-front="{depth_front}">'.format(
                    class_name=svg_escape(class_name),
                    d=svg_escape(d_attr),
                    stroke=svg_escape(stroke),
                    width=svg_float(width_value),
                    opacity=svg_float(opacity_value),
                    model=svg_escape(atom.model),
                    chain=svg_escape(atom.chain),
                    atom_type=svg_escape(selected.atom_type),
                    from_serial=svg_escape(atom.serial),
                    from_line=svg_escape(atom.line_number),
                    to_atom_name=svg_escape(extension.end_atom_name),
                    to_line=svg_escape(extension.end_line_number),
                    from_depth=svg_float(depth1),
                    to_depth=svg_float(extension.end_depth),
                    segment_depth=svg_float(seg_depth),
                    mode_name=svg_escape(projection.mode),
                    depth_front=svg_escape(depth_front),
                )
            )
            close_tag = "</path>"
        else:
            out.append(
                indent + '<line class="{class_name}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" '
                'data-connection-mode="terminal-3prime-straight" data-chain-closed="false" data-model="{model}" data-chain="{chain}" data-atom-type="{atom_type}" '
                'data-from-serial="{from_serial}" data-from-line-number="{from_line}" data-to-atom-name="{to_atom_name}" data-to-line-number="{to_line}" '
                'data-from-depth="{from_depth}" data-to-depth="{to_depth}" data-segment-depth="{segment_depth}" '
                'data-projection-mode="{mode_name}" data-depth-front="{depth_front}">'.format(
                    class_name=svg_escape(class_name),
                    x1=svg_float(x1),
                    y1=svg_float(y1),
                    x2=svg_float(x2),
                    y2=svg_float(y2),
                    stroke=svg_escape(stroke),
                    width=svg_float(width_value),
                    opacity=svg_float(opacity_value),
                    model=svg_escape(atom.model),
                    chain=svg_escape(atom.chain),
                    atom_type=svg_escape(selected.atom_type),
                    from_serial=svg_escape(atom.serial),
                    from_line=svg_escape(atom.line_number),
                    to_atom_name=svg_escape(extension.end_atom_name),
                    to_line=svg_escape(extension.end_line_number),
                    from_depth=svg_float(depth1),
                    to_depth=svg_float(extension.end_depth),
                    segment_depth=svg_float(seg_depth),
                    mode_name=svg_escape(projection.mode),
                    depth_front=svg_escape(depth_front),
                )
            )
            close_tag = "</line>"

        out.append(
            indent + "  <title>{0}: chain {1}; serial {2} to terminal {3}</title>".format(
                svg_escape(title_prefix), svg_escape(chain_label(atom.chain)), svg_escape(atom.serial), svg_escape(extension.end_atom_name)
            )
        )
        out.append(indent + close_tag)
        return out

    def extension_segment_lines(extension: TerminalExtensionSegment, indent: str) -> List[str]:
        out: List[str] = []
        if line_underlay:
            out.extend(extension_geometry_lines(extension, indent, underlay=True))
        out.extend(extension_geometry_lines(extension, indent, underlay=False))
        return out

    def base_pair_line_lines(drawable: BasePairDrawable, indent: str) -> List[str]:
        x1, y1 = to_svg_xy(drawable.x1, drawable.y1, scale, x_offset, y_offset, invert_y)
        x2, y2 = to_svg_xy(drawable.x2, drawable.y2, scale, x_offset, y_offset, invert_y)
        bp = drawable.base_pair
        depth = base_pair_depth(drawable)
        anchor_atom_name = normalize_base_pair_atom_name(getattr(args, "base_pair_atom", DEFAULT_BASE_PAIR_ATOM))
        out: List[str] = []
        out.append(
            indent + '<line class="base-pair-line" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" '
            'data-base-pair-index="{bp_index}" data-nt1="{nt1}" data-nt2="{nt2}" data-bp="{bp}" data-name="{name}" '
            'data-lw="{lw}" data-dssr="{dssr}" data-chain1="{chain1}" data-resseq1="{resseq1}" '
            'data-chain2="{chain2}" data-resseq2="{resseq2}" data-anchor-atom="{anchor_atom}" '
            'data-depth="{depth}" data-depth-front="{depth_front}">'.format(
                x1=svg_float(x1),
                y1=svg_float(y1),
                x2=svg_float(x2),
                y2=svg_float(y2),
                stroke=svg_escape(getattr(args, "base_pair_stroke", "#444444")),
                width=svg_float(float(getattr(args, "base_pair_width", 3.0))),
                opacity=svg_float(float(getattr(args, "base_pair_opacity", 0.75))),
                bp_index=svg_escape(bp.index),
                nt1=svg_escape(bp.nt1),
                nt2=svg_escape(bp.nt2),
                bp=svg_escape(bp.bp),
                name=svg_escape(bp.name),
                lw=svg_escape(bp.lw),
                dssr=svg_escape(bp.dssr),
                chain1=svg_escape(drawable.key1.chain),
                resseq1=svg_escape(drawable.key1.resseq),
                chain2=svg_escape(drawable.key2.chain),
                resseq2=svg_escape(drawable.key2.resseq),
                anchor_atom=svg_escape(anchor_atom_name),
                depth=svg_float(depth),
                depth_front=svg_escape(depth_front),
            )
        )
        out.append(
            indent + "  <title>Base pair {0}: {1} - {2}; line connects projected {3} atoms of paired residues</title>".format(
                svg_escape(bp.index), svg_escape(bp.nt1), svg_escape(bp.nt2), svg_escape(anchor_atom_name)
            )
        )
        out.append(indent + "</line>")
        return out

    def append_xy_plane_layer(svg_lines: List[str], indent: str) -> None:
        if not draw_xy_plane or xy_plane_raw_corners is None or xy_plane_projected_corners is None or xy_plane_depths is None:
            return

        svg_points = [
            to_svg_xy(float(proj_x), float(proj_y), scale, x_offset, y_offset, invert_y)
            for proj_x, proj_y in xy_plane_projected_corners
        ]
        points_attr = " ".join("{0},{1}".format(svg_float(x), svg_float(y)) for x, y in svg_points)
        raw_attr = ";".join(
            "{0},{1},{2}".format(svg_float(x), svg_float(y), svg_float(z))
            for x, y, z in xy_plane_raw_corners
        )
        proj_attr = ";".join(
            "{0},{1},{2}".format(svg_float(x), svg_float(y), svg_float(depth))
            for (x, y), depth in zip(xy_plane_projected_corners, xy_plane_depths)
        )
        depth_mean = xy_plane_depth_mean()
        svg_lines.append(
            indent + '<g id="xy-plane" class="xy-plane-layer" inkscape:groupmode="layer" inkscape:label="xy-plane" data-layer-name="xy-plane" data-plane="projection-basis-depth=0" data-depth-sort-key="{depth_key}">'.format(
                depth_key=svg_float(depth_sort_value(depth_mean))
            )
        )
        svg_lines.append(indent + "  <title>xy-plane: projection-basis depth=0 patch</title>")
        svg_lines.append(
            indent + '  <polygon id="xy-plane-shape" class="xy-plane" points="{points}" '
            'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}" '
            'data-name="xy-plane" data-plane="projection-basis-depth=0" data-raw-corners="{raw}" data-projected-corners="{proj}" '
            'data-depth-mean="{depth_mean}" data-projection-mode="{mode}">'.format(
                points=svg_escape(points_attr),
                fill=svg_escape(getattr(args, "xy_plane_fill", "#7dd3fc")),
                stroke=svg_escape(getattr(args, "xy_plane_stroke", "#0284c7")),
                stroke_width=svg_float(float(getattr(args, "xy_plane_stroke_width", 1.5))),
                opacity=svg_float(float(getattr(args, "xy_plane_opacity", 0.18))),
                raw=svg_escape(raw_attr),
                proj=svg_escape(proj_attr),
                depth_mean=svg_float(depth_mean),
                mode=svg_escape(projection.mode),
            )
        )
        svg_lines.append(indent + "    <title>xy-plane shape: finite projection-basis depth=0 patch spanning selected projected x/y bounds</title>")
        svg_lines.append(indent + "  </polygon>")
        svg_lines.append(indent + "</g>")

    def append_scale_bar_layer(svg_lines: List[str], indent: str) -> None:
        if not draw_scale_bar:
            return
        bar_length_svg = scale_bar_length * scale
        margin = scale_bar_margin
        if bar_length_svg + 2.0 * margin <= width:
            x1 = margin
        else:
            x1 = max(0.0, (width - bar_length_svg) / 2.0)
        x2 = x1 + bar_length_svg
        length_label = "{0} {1}".format(svg_float(scale_bar_length), scale_bar_unit_label)
        scale_label = "scale: 1 {0} = {1} SVG units".format(scale_bar_unit_label, svg_float(scale))
        available_text_width = max(width - 2.0 * margin, 24.0)
        length_font_size = max(6.0, min(scale_bar_text_size, available_text_width / max(len(length_label) * 0.58, 1.0)))
        scale_font_size = max(6.0, min(max(8.0, scale_bar_text_size * 0.82), available_text_width / max(len(scale_label) * 0.58, 1.0)))
        tick = max(4.0, length_font_size * 0.45)
        y_bar = max(margin + length_font_size + tick + scale_font_size + 14.0, height - margin - length_font_size - 6.0)
        y_label = y_bar + length_font_size + 5.0
        y_scale = y_bar - tick - 8.0
        approx_text_width = max(len(length_label) * length_font_size * 0.58, len(scale_label) * scale_font_size * 0.58)
        bg_width = max(bar_length_svg, approx_text_width) + 18.0
        bg_height = (y_label - y_scale) + max(length_font_size, scale_font_size) + 10.0
        bg_x = max(0.0, min(x1 - 9.0, width - bg_width))
        bg_y = max(0.0, y_scale - scale_font_size - 5.0)
        svg_lines.append(
            indent + '<g id="scale-bar" class="scale-bar-layer" data-scale="{scale}" '
            'data-scale-definition="SVG units per projected coordinate unit" '
            'data-length-projection-units="{length_units}" data-length-svg-units="{length_svg}" '
            'data-unit-label="{unit_label}">'.format(
                scale=svg_float(scale),
                length_units=svg_float(scale_bar_length),
                length_svg=svg_float(bar_length_svg),
                unit_label=svg_escape(scale_bar_unit_label),
            )
        )
        svg_lines.append(
            indent + "  <title>Scale bar: {length}; {scale_label}</title>".format(
                length=svg_escape(length_label),
                scale_label=svg_escape(scale_label),
            )
        )
        svg_lines.append(
            indent + '  <rect class="scale-bar-background" x="{x}" y="{y}" width="{w}" height="{h}" '
            'rx="4" ry="4" fill="{fill}" opacity="{opacity}"/>'.format(
                x=svg_float(bg_x),
                y=svg_float(bg_y),
                w=svg_float(bg_width),
                h=svg_float(bg_height),
                fill=svg_escape(scale_bar_background),
                opacity=svg_float(scale_bar_background_opacity),
            )
        )
        svg_lines.append(
            indent + '  <line class="scale-bar-line" x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
            'stroke="{stroke}" stroke-width="{stroke_width}"/>'.format(
                x1=svg_float(x1),
                y=svg_float(y_bar),
                x2=svg_float(x2),
                stroke=svg_escape(scale_bar_stroke),
                stroke_width=svg_float(scale_bar_stroke_width),
            )
        )
        for tick_x in (x1, x2):
            svg_lines.append(
                indent + '  <line class="scale-bar-tick" x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
                'stroke="{stroke}" stroke-width="{stroke_width}"/>'.format(
                    x=svg_float(tick_x),
                    y1=svg_float(y_bar - tick),
                    y2=svg_float(y_bar + tick),
                    stroke=svg_escape(scale_bar_stroke),
                    stroke_width=svg_float(scale_bar_stroke_width),
                )
            )
        svg_lines.append(
            indent + '  <text class="scale-bar-label" x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
            'font-size="{font_size}" fill="{fill}">{text}</text>'.format(
                x=svg_float(x1),
                y=svg_float(y_label),
                font_size=svg_float(length_font_size),
                fill=svg_escape(scale_bar_stroke),
                text=svg_escape(length_label),
            )
        )
        svg_lines.append(
            indent + '  <text class="scale-bar-scale-label" x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
            'font-size="{font_size}" fill="{fill}">{text}</text>'.format(
                x=svg_float(x1),
                y=svg_float(y_scale),
                font_size=svg_float(scale_font_size),
                fill=svg_escape(scale_bar_stroke),
                text=svg_escape(scale_label),
            )
        )
        svg_lines.append(indent + "</g>")

    def connection_segments() -> List[ConnectionSegment]:
        segments: List[ConnectionSegment] = []
        for chain, type_groups in sorted_chain_items(grouped):
            for atom_type in atom_types:
                entries = type_groups.get(atom_type, [])
                if not entries:
                    continue
                style = styles[atom_type.upper()]
                if not style.draw_lines:
                    continue
                for _model, model_entries in entries_by_model(entries).items():
                    if len(model_entries) < 2:
                        continue
                    closed = chain_is_closed(chain, args)
                    n_segments = len(model_entries) if closed else len(model_entries) - 1
                    for index in range(n_segments):
                        terminal_extension_point = None
                        if (
                            not closed
                            and index == n_segments - 1
                            and style.extend_3prime
                            and normalize_connection_mode(style.connection_mode) == "smooth"
                        ):
                            o3_point = find_o3_prime_point_for_entry(model_entries[-1], o3_prime_points)
                            if o3_point is not None and model_entries[-1][0].atom.line_number != int(o3_point[3]):
                                terminal_extension_point = o3_point
                        segments.append(
                            ConnectionSegment(
                                entries=model_entries,
                                index=index,
                                closed=closed,
                                connection_mode=normalize_connection_mode(style.connection_mode),
                                terminal_extension_point=terminal_extension_point,
                            )
                        )
        return segments

    def terminal_extension_segments() -> List[TerminalExtensionSegment]:
        extensions: List[TerminalExtensionSegment] = []
        if not o3_prime_points:
            return extensions
        for chain, type_groups in sorted_chain_items(grouped):
            if chain_is_closed(chain, args):
                continue
            for atom_type in atom_types:
                entries = type_groups.get(atom_type, [])
                if not entries:
                    continue
                style = styles[atom_type.upper()]
                if not style.draw_lines or not style.extend_3prime:
                    continue
                for _model, model_entries in entries_by_model(entries).items():
                    if len(model_entries) < 1:
                        continue
                    last_entry = model_entries[-1]
                    o3_point = find_o3_prime_point_for_entry(last_entry, o3_prime_points)
                    if o3_point is None:
                        continue
                    end_x, end_y, end_depth, end_line, end_atom_name = o3_point
                    # Avoid adding a zero-length/duplicate extension if the selected
                    # terminal atom is itself the same O3' record.
                    if last_entry[0].atom.line_number == end_line:
                        continue
                    previous_entry = model_entries[-2] if len(model_entries) >= 2 else None
                    extensions.append(
                        TerminalExtensionSegment(
                            start_entry=last_entry,
                            previous_entry=previous_entry,
                            end_x=float(end_x),
                            end_y=float(end_y),
                            end_depth=float(end_depth),
                            end_line_number=int(end_line),
                            end_atom_name=str(end_atom_name),
                            connection_mode=normalize_connection_mode(style.connection_mode),
                        )
                    )
        return extensions

    def append_grouped_elements(svg_lines: List[str], include_lines: bool, include_points: bool, layer_id: str, layer_label: str) -> None:
        svg_lines.append('    <g id="{0}" class="grouped-layer" data-layer-label="{1}">'.format(svg_escape(layer_id), svg_escape(layer_label)))
        for chain, type_groups in sorted_chain_items(grouped):
            chain_id = svg_group_id_for_chain(chain)
            label = chain_label(chain)
            suffix = "" if layer_id == "chain_grouped" else "_" + svg_id_part(layer_id, "layer")
            svg_lines.append(
                '      <g id="{0}{1}" class="chain" data-chain="{2}" data-chain-label="{3}" data-chain-closed="{4}">'.format(
                    svg_escape(chain_id), svg_escape(suffix), svg_escape(chain), svg_escape(label), str(chain_is_closed(chain, args)).lower()
                )
            )
            svg_lines.append("        <title>Chain {0}</title>".format(svg_escape(label)))
            for atom_type in atom_types:
                entries = type_groups.get(atom_type, [])
                if not entries:
                    continue
                style = styles[atom_type.upper()]
                mode = normalize_connection_mode(style.connection_mode)
                type_group_id = "{0}_{1}".format(chain_id, svg_group_id_for_type(atom_type))
                if layer_id != "chain_grouped":
                    type_group_id = "{0}_{1}".format(type_group_id, svg_id_part(layer_id, "layer"))
                svg_lines.append(
                    '        <g id="{0}" class="atom-type" data-atom-type="{1}" data-draw-lines="{2}" data-connection-mode="{3}">'.format(
                        svg_escape(type_group_id), svg_escape(atom_type), str(style.draw_lines).lower(), svg_escape(mode)
                    )
                )
                svg_lines.append("          <title>Chain {0}, atom type {1}</title>".format(svg_escape(label), svg_escape(atom_type)))
                if include_lines and style.draw_lines:
                    for model, model_entries in entries_by_model(entries).items():
                        if len(model_entries) >= 2:
                            closed = chain_is_closed(chain, args)
                            if mode == "smooth":
                                terminal_o3_included = False
                                smooth_points = svg_points_from_entries(model_entries, scale, x_offset, y_offset, invert_y)
                                if (not closed) and style.extend_3prime and len(model_entries) >= 2:
                                    o3_point = find_o3_prime_point_for_entry(model_entries[-1], o3_prime_points)
                                    if o3_point is not None and model_entries[-1][0].atom.line_number != int(o3_point[3]):
                                        smooth_points.append(to_svg_xy(float(o3_point[0]), float(o3_point[1]), scale, x_offset, y_offset, invert_y))
                                        terminal_o3_included = True
                                path_attr = smooth_path_from_svg_points(smooth_points, closed=closed)
                                svg_lines.append(
                                    '          <path class="neighbor-line smooth-curve" d="{path}" '
                                    'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" '
                                    'data-connection-mode="smooth" data-chain-closed="{closed}" data-terminal-o3-extension-included="{terminal_o3}" data-model="{model}" data-chain="{chain}" data-atom-type="{atom_type}">'.format(
                                        path=svg_escape(path_attr),
                                        stroke=svg_escape(line_color_for(chain, atom_type)),
                                        width=svg_float(style.line_width),
                                        opacity=svg_float(style.line_opacity),
                                        closed=str(closed).lower(),
                                        terminal_o3=str(terminal_o3_included).lower(),
                                        model=svg_escape(model),
                                        chain=svg_escape(chain),
                                        atom_type=svg_escape(atom_type),
                                    )
                                )
                                title_extra = " with terminal O3' extension" if terminal_o3_included else ""
                                svg_lines.append(
                                    "            <title>Smooth neighbor curve{3}: consecutive selected {0} atoms in chain {1}, model {2}</title>".format(
                                        svg_escape(atom_type), svg_escape(label), svg_escape(model), title_extra
                                    )
                                )
                                svg_lines.append("          </path>")
                            else:
                                point_list = polyline_points(model_entries, scale, x_offset, y_offset, invert_y)
                                if closed and point_list:
                                    point_list.append(point_list[0])
                                points_attr = " ".join(point_list)
                                svg_lines.append(
                                    '          <polyline class="neighbor-line straight-line" points="{points}" '
                                    'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" '
                                    'data-connection-mode="straight" data-chain-closed="{closed}" data-model="{model}" data-chain="{chain}" data-atom-type="{atom_type}">'.format(
                                        points=svg_escape(points_attr),
                                        stroke=svg_escape(line_color_for(chain, atom_type)),
                                        width=svg_float(style.line_width),
                                        opacity=svg_float(style.line_opacity),
                                        closed=str(closed).lower(),
                                        model=svg_escape(model),
                                        chain=svg_escape(chain),
                                        atom_type=svg_escape(atom_type),
                                    )
                                )
                                svg_lines.append(
                                    "            <title>Straight neighbor line: consecutive selected {0} atoms in chain {1}, model {2}</title>".format(
                                        svg_escape(atom_type), svg_escape(label), svg_escape(model)
                                    )
                                )
                                svg_lines.append("          </polyline>")
                if include_points:
                    for selected, proj_x, proj_y, depth in entries:
                        svg_lines.extend(circle_lines(selected, proj_x, proj_y, depth, "          "))
                svg_lines.append("        </g>")
            svg_lines.append("      </g>")
        if include_lines:
            extensions = terminal_extension_segments()
            drawable_extensions = [
                extension for extension in extensions
                if not (normalize_connection_mode(extension.connection_mode) == "smooth" and not depth_order_lines)
            ]
            if drawable_extensions:
                svg_lines.append('      <g id="terminal_3prime_annotations_{0}" class="terminal-extension-layer">'.format(svg_escape(svg_id_part(layer_id, "layer"))))
                svg_lines.append("        <title>Optional terminal 3-prime O3' extensions</title>")
                for extension in drawable_extensions:
                    svg_lines.extend(extension_segment_lines(extension, "        "))
                svg_lines.append("      </g>")
        svg_lines.append("    </g>")

    def append_base_pair_layer(svg_lines: List[str], layer_id: str, ordered: bool) -> None:
        if not base_pair_drawables:
            return
        order_text = "back-to-front by projection depth" if ordered else "DSSR order"
        svg_lines.append(
            '    <g id="{0}" class="base-pair-layer" data-order="{1}" data-front-side="{2}">'.format(
                svg_escape(layer_id), svg_escape(order_text), svg_escape(depth_front)
            )
        )
        anchor_atom_name = normalize_base_pair_atom_name(getattr(args, "base_pair_atom", DEFAULT_BASE_PAIR_ATOM))
        svg_lines.append(
            "      <title>Base-pair interaction lines from DSSR, anchored at {0}</title>".format(
                svg_escape(anchor_atom_name)
            )
        )
        items = sorted(base_pair_drawables, key=base_pair_depth_sort_key) if ordered else list(base_pair_drawables)
        for drawable in items:
            svg_lines.extend(base_pair_line_lines(drawable, "      "))
        svg_lines.append("    </g>")

    def append_depth_ordered_elements(svg_lines: List[str], include_lines: bool, include_points: bool, include_base_pairs: bool) -> None:
        svg_lines.append(
            '    <g id="depth_ordered_elements" class="depth-order-layer" data-order="back-to-front by projection depth" data-front-side="{0}">'.format(
                svg_escape(depth_front)
            )
        )
        svg_lines.append("      <title>Depth-ordered circles, neighbor connection segments, base-pair lines, and/or xy-plane. Anchor circles are kept above their incident neighbor/base-pair segments.</title>")
        drawable_items: List[Tuple[Tuple[float, int, int], str, object]] = []
        segments = connection_segments() if include_lines else []
        extensions = terminal_extension_segments() if include_lines else []

        if draw_xy_plane:
            drawable_items.append((xy_plane_depth_sort_key(), "xy_plane", None))

        # Record the highest depth score of line/base-pair elements touching each circle.
        # A circle is then drawn at least at that score, with a later element layer, so
        # its own connector segments cannot partially cover the atom marker.
        incident_circle_scores: Dict[Tuple[int, int, str], float] = {}
        line_number_to_entry_keys: Dict[int, List[Tuple[int, int, str]]] = {}
        for entry in selected_entries:
            line_number_to_entry_keys.setdefault(entry[0].atom.line_number, []).append(selected_entry_key(entry))

        if include_lines:
            for segment in segments:
                key = segment_depth_sort_key(segment)
                drawable_items.append((key, "line", segment))
                first, second = segment_endpoints(segment)
                for endpoint in (first, second):
                    atom_key = selected_entry_key(endpoint)
                    incident_circle_scores[atom_key] = max(incident_circle_scores.get(atom_key, -float("inf")), key[0])
            for extension in extensions:
                key = extension_depth_sort_key(extension)
                drawable_items.append((key, "extension", extension))
                atom_key = extension_endpoint_key(extension)
                incident_circle_scores[atom_key] = max(incident_circle_scores.get(atom_key, -float("inf")), key[0])
        if include_base_pairs:
            for drawable in base_pair_drawables:
                key = base_pair_depth_sort_key(drawable)
                drawable_items.append((key, "base_pair", drawable))
                for line_number in (drawable.segment_line1, drawable.segment_line2):
                    for atom_key in line_number_to_entry_keys.get(line_number, []):
                        incident_circle_scores[atom_key] = max(incident_circle_scores.get(atom_key, -float("inf")), key[0])
        if include_points:
            for entry in selected_entries:
                key = circle_depth_sort_key(entry)
                incident_score = incident_circle_scores.get(selected_entry_key(entry))
                if incident_score is not None and incident_score > key[0]:
                    key = (incident_score, key[1], key[2])
                drawable_items.append((key, "circle", entry))
        drawable_items.sort(key=lambda item: item[0])
        for _key, kind, item in drawable_items:
            if kind == "line":
                svg_lines.extend(connection_segment_lines(item, "      "))
            elif kind == "extension":
                svg_lines.extend(extension_segment_lines(item, "      "))
            elif kind == "base_pair":
                svg_lines.extend(base_pair_line_lines(item, "      "))
            elif kind == "xy_plane":
                append_xy_plane_layer(svg_lines, "      ")
            else:
                selected, proj_x, proj_y, depth = item
                svg_lines.extend(circle_lines(selected, proj_x, proj_y, depth, "      "))
        svg_lines.append("    </g>")

    metadata = {
        "input_file": str(Path(args.pdb_file)),
        "requested_input_format": getattr(args, "input_format", "auto"),
        "resolved_input_format": getattr(args, "resolved_input_format", getattr(args, "input_format", "auto")),
        "atom_types": list(atom_types),
        "projection_mode": projection.mode,
        "color_by": color_by,
        "default_color_saturation": DEFAULT_COLOR_SATURATION,
        "default_color_value": DEFAULT_COLOR_VALUE,
        "n_points": len(selected_atoms),
        "pre_flip_about_y": bool(getattr(args, "flip_about_y", False)),
        "method": projection.method_name,
        "neighbor_connection_definition": "Per atom type; consecutive selected atoms with same chain, model, and atom type in PDB order; each atom type can use straight or smooth connections",
        "depth_order_circles": depth_order_circles,
        "depth_order_lines": depth_order_lines,
        "depth_order_base_pairs": depth_order_base_pairs,
        "depth_order_xy_plane": bool(draw_xy_plane and (depth_order_circles or depth_order_lines or depth_order_base_pairs)),
        "line_underlay": line_underlay,
        "draw_xy_plane": draw_xy_plane,
        "depth_front": depth_front,
        "closed_chains": getattr(args, "closed_chains", ""),
        "close_all_chains": bool(getattr(args, "close_all_chains", False)),
        "base_pair_lines": len(base_pair_drawables),
        "depth_coordinate": "PCA-normal coordinate" if projection.mode == "pca" else "input Z coordinate after optional pre-projection transform",
        "projection_scale_pixels_per_projection_unit": scale,
        "projection_scale_svg_units_per_projection_unit": scale,
        "scale_bar": {
            "drawn": draw_scale_bar,
            "length_projection_units": scale_bar_length,
            "unit_label": scale_bar_unit_label,
            "length_svg_units": scale_bar_length * scale,
        },
    }
    # The elements are written first, into body: which chains took a
    # --chain-colors color is known only once they are, and the projection
    # group ahead of them records it.
    body: List[str] = []
    any_depth_order = depth_order_circles or depth_order_lines or depth_order_base_pairs
    if not any_depth_order:
        append_xy_plane_layer(body, "    ")
        append_grouped_elements(body, include_lines=True, include_points=False, layer_id="chain_grouped_lines", layer_label="grouped neighbor lines")
        append_base_pair_layer(body, layer_id="base_pairs", ordered=False)
        append_grouped_elements(body, include_lines=False, include_points=True, layer_id="chain_grouped_points", layer_label="grouped points")
    else:
        if not depth_order_lines:
            append_grouped_elements(body, include_lines=True, include_points=False, layer_id="chain_grouped_lines", layer_label="grouped neighbor lines")
        append_depth_ordered_elements(
            body,
            include_lines=depth_order_lines,
            include_points=depth_order_circles,
            include_base_pairs=depth_order_base_pairs,
        )
        if base_pair_drawables and not depth_order_base_pairs:
            append_base_pair_layer(body, layer_id="base_pairs", ordered=False)
        if not depth_order_circles:
            append_grouped_elements(body, include_lines=False, include_points=True, layer_id="chain_grouped_points", layer_label="grouped points")
    append_scale_bar_layer(body, "    ")
    # Kept for the JSON and the run log, written after this SVG: the chains
    # whose chain color some drawn circle or line took.
    args.chain_color_takers = [chain for chain in chain_order if chain in chain_color_takers]

    # A non-default palette is recorded here and as data-palette on the
    # projection group; default SVGs carry neither.
    palette_attr = ""
    if palette != DEFAULT_PALETTE:
        metadata["palette"] = palette
        palette_attr = ' data-palette="{0}"'.format(svg_escape(palette))
    # Likewise the --chain-colors this SVG used, as data-chain-colors beside
    # it: a listed chain's color counts only where a circle or line took it.
    used_chain_colors = {
        chain: chain_overrides[chain] for chain in chain_order if chain in chain_overrides and chain in chain_color_takers
    }
    if used_chain_colors:
        metadata["chain_colors"] = {chain_label(chain): color for chain, color in used_chain_colors.items()}
        palette_attr += ' data-chain-colors="{0}"'.format(svg_escape(format_chain_colors(used_chain_colors)))

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(svg_float(width), svg_float(height)))
    title = "Structure PCA projection" if projection.mode == "pca" else "Structure current-XY projection"
    lines.append("  <title>{0}</title>".format(svg_escape(title)))
    if projection.mode == "pca":
        desc = "Selected atoms projected onto the first two PCA axes. This is not a convex-hull-area optimization."
    else:
        desc = "Selected atoms drawn using the current PDB X/Y coordinates without PCA fitting."
    lines.append("  <desc>{0}</desc>".format(svg_escape(desc)))
    lines.append("  <metadata>{0}</metadata>".format(svg_escape(json.dumps(metadata, sort_keys=True))))
    lines.append("  <style>")
    lines.append("    .point { vector-effect: non-scaling-stroke; }")
    lines.append("    .neighbor-line { fill: none; vector-effect: non-scaling-stroke; stroke-linecap: butt; stroke-linejoin: round; }")
    lines.append("    .neighbor-line-underlay { fill: none; vector-effect: non-scaling-stroke; stroke-linecap: butt; stroke-linejoin: round; }")
    lines.append("    .base-pair-line { fill: none; vector-effect: non-scaling-stroke; stroke-linecap: butt; }")
    lines.append("    .xy-plane { vector-effect: non-scaling-stroke; stroke-linejoin: round; }")
    lines.append("    .scale-bar-line, .scale-bar-tick { vector-effect: non-scaling-stroke; stroke-linecap: square; }")
    lines.append("    .smooth-curve { fill: none; }")
    lines.append("  </style>")
    lines.append(
        '  <g id="projection" data-projection-mode="{0}" data-color-by="{1}" data-invert-y="{2}" data-scale="{3}" data-depth-order-circles="{4}" data-depth-order-lines="{5}" data-depth-order-base-pairs="{6}" data-depth-front="{7}"{8}>'.format(
            svg_escape(projection.mode),
            svg_escape(color_by),
            str(invert_y).lower(),
            svg_float(scale),
            str(depth_order_circles).lower(),
            str(depth_order_lines).lower(),
            str(depth_order_base_pairs).lower(),
            svg_escape(depth_front),
            palette_attr,
        )
    )
    lines.extend(body)
    lines.append("  </g>")
    lines.append("</svg>")
    with output_svg.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return scale

def format_pdb_coord_line(line: str, x: float, y: float, z: float) -> str:
    newline = "\n" if line.endswith("\n") else ""
    body = line[:-1] if newline else line
    if len(body) < 54:
        body = body.ljust(54)
    coord_text = "{0:8.3f}{1:8.3f}{2:8.3f}".format(float(x), float(y), float(z))
    return body[:30] + coord_text + body[54:] + newline


def write_projection_basis_pdb(input_pdb: Path, output_pdb: Path, projection: ProjectionResult) -> int:
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    n_transformed = 0
    with input_pdb.open("r", encoding="utf-8", errors="replace") as inp, output_pdb.open("w", encoding="utf-8") as out:
        out.write("REMARK PROJECTION-BASIS PDB CREATED BY Plane It (plane_itV3_8.py)\n")
        out.write("REMARK PROJECTION MODE: {0}\n".format(projection.mode.upper()))
        if bool(getattr(projection, "pre_flip_about_y", False)):
            out.write("REMARK INPUT COORDINATES WERE FIRST FLIPPED ABOUT THE Y AXIS: X -> -X, Z -> -Z\n")
        if projection.mode == "pca":
            out.write("REMARK NEW X,Y ARE PC1,PC2; NEW Z IS THE PCA-NORMAL COORDINATE\n")
            out.write("REMARK THIS IS PCA/COVARIANCE ALIGNMENT, NOT HULL-AREA OPTIMIZATION\n")
        else:
            if bool(getattr(projection, "pre_flip_about_y", False)):
                out.write("REMARK CURRENT-XY MODE: OUTPUT IS THE Y-AXIS-FLIPPED INPUT COORDINATE SYSTEM\n")
            else:
                out.write("REMARK CURRENT-XY MODE: ORIGINAL COORDINATES WERE KEPT\n")
        for line in inp:
            rec = line[0:6].strip().upper()
            if rec in {"ATOM", "HETATM"}:
                try:
                    x = float(_safe_slice(line, 30, 38).strip())
                    y = float(_safe_slice(line, 38, 46).strip())
                    z = float(_safe_slice(line, 46, 54).strip())
                except ValueError:
                    out.write(line)
                    continue
                new_xyz = transform_points_to_projection_basis(np.array([[x, y, z]], dtype=float), projection)[0]
                out.write(format_pdb_coord_line(line, new_xyz[0], new_xyz[1], new_xyz[2]))
                n_transformed += 1
            else:
                out.write(line)
    return n_transformed


def write_projection_basis_xyz(input_xyz: Path, output_xyz: Path, atoms: Sequence[AtomRecord], projection: ProjectionResult) -> int:
    """Write XYZ/coordinate input points in the projection basis, preserving input text layout.

    For coordinate-only text such as rows of ``x y z`` separated by blank lines,
    the output keeps coordinate-only rows and keeps the blank lines. For molecular
    XYZ rows such as ``C x y z``, the label is kept and only x/y/z are replaced.
    Header, comment, blank, and unparsed lines are copied unchanged.
    """
    output_xyz.parent.mkdir(parents=True, exist_ok=True)
    if not atoms:
        raise ValueError("No XYZ atoms/points are available for projection-basis XYZ output")

    points = np.array([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)
    transformed = transform_points_to_projection_basis(points, projection)
    transformed_by_line = {atom.line_number: (atom, coords) for atom, coords in zip(atoms, transformed)}

    def format_transformed_line(original_line: str, atom: AtomRecord, coords: np.ndarray) -> str:
        newline = "\n" if original_line.endswith("\n") else ""
        body = original_line[:-1] if newline else original_line
        comment = ""
        main = body
        if "#" in body:
            main, comment_part = body.split("#", 1)
            comment = " #" + comment_part
        stripped = main.strip()
        comma_mode = "," in stripped
        tokens = split_xyz_line(body)
        x, y, z = (float(coords[0]), float(coords[1]), float(coords[2]))
        if len(tokens) >= 3 and all(_is_float_token(tok) for tok in tokens[:3]):
            if comma_mode:
                return "{0:.10f},{1:.10f},{2:.10f}{3}{4}".format(x, y, z, comment, newline)
            return "{0:.10f} {1:.10f} {2:.10f}{3}{4}".format(x, y, z, comment, newline)
        if len(tokens) >= 4 and all(_is_float_token(tok) for tok in tokens[1:4]):
            label = tokens[0]
            if comma_mode:
                return "{0},{1:.10f},{2:.10f},{3:.10f}{4}{5}".format(label, x, y, z, comment, newline)
            return "{0} {1:.10f} {2:.10f} {3:.10f}{4}{5}".format(label, x, y, z, comment, newline)
        label = atom.atom_name.strip() or atom.element.strip() or "X"
        return "{0} {1:.10f} {2:.10f} {3:.10f}{4}".format(label, x, y, z, newline)

    n_transformed = 0
    with input_xyz.open("r", encoding="utf-8", errors="replace") as inp, output_xyz.open("w", encoding="utf-8") as out:
        for line_number, line in enumerate(inp, start=1):
            item = transformed_by_line.get(line_number)
            if item is None:
                out.write(line)
                continue
            atom, coords = item
            out.write(format_transformed_line(line, atom, coords))
            n_transformed += 1
    return n_transformed


def validate_svg_args(args: argparse.Namespace) -> None:
    if args.width <= 0:
        raise ValueError("--width must be positive")
    if args.height <= 0:
        raise ValueError("--height must be positive")
    if args.padding < 0:
        raise ValueError("--padding must be non-negative")
    if args.padding * 2 >= min(args.width, args.height):
        raise ValueError("--padding is too large for the requested SVG size")
    if args.radius <= 0:
        raise ValueError("--radius must be positive")
    if args.line_width < 0:
        raise ValueError("--line-width must be non-negative")
    if not (0.0 <= args.line_opacity <= 1.0):
        raise ValueError("--line-opacity must be between 0 and 1")
    if args.projection_mode not in {"pca", "current-xy"}:
        raise ValueError("--projection-mode must be pca or current-xy")
    normalize_connection_mode(getattr(args, "connection_mode", "smooth"))
    if not (0.0 <= float(getattr(args, "line_underlay_opacity", 1.0)) <= 1.0):
        raise ValueError("--line-underlay-opacity must be between 0 and 1")
    if float(getattr(args, "line_underlay_extra_width", 8.0)) < 0:
        raise ValueError("--line-underlay-extra-width must be non-negative")
    if float(getattr(args, "base_pair_width", 3.0)) < 0:
        raise ValueError("--base-pair-width must be non-negative")
    if not (0.0 <= float(getattr(args, "base_pair_opacity", 0.75)) <= 1.0):
        raise ValueError("--base-pair-opacity must be between 0 and 1")
    if not normalize_base_pair_atom_name(getattr(args, "base_pair_atom", DEFAULT_BASE_PAIR_ATOM)):
        raise ValueError("--base-pair-atom must not be blank")
    if float(getattr(args, "xy_plane_stroke_width", 1.5)) < 0:
        raise ValueError("--xy-plane-stroke-width must be non-negative")
    if not (0.0 <= float(getattr(args, "xy_plane_opacity", 0.18)) <= 1.0):
        raise ValueError("--xy-plane-opacity must be between 0 and 1")
    if float(getattr(args, "scale_bar_length", DEFAULT_SCALE_BAR_LENGTH)) <= 0:
        raise ValueError("--scale-bar-length must be positive")
    if not str(getattr(args, "scale_bar_unit_label", DEFAULT_SCALE_BAR_UNIT_LABEL) or "").strip():
        raise ValueError("--scale-bar-unit-label must not be blank")
    if float(getattr(args, "scale_bar_stroke_width", DEFAULT_SCALE_BAR_STROKE_WIDTH)) < 0:
        raise ValueError("--scale-bar-stroke-width must be non-negative")
    if float(getattr(args, "scale_bar_text_size", DEFAULT_SCALE_BAR_TEXT_SIZE)) <= 0:
        raise ValueError("--scale-bar-text-size must be positive")
    if float(getattr(args, "scale_bar_margin", DEFAULT_SCALE_BAR_MARGIN)) < 0:
        raise ValueError("--scale-bar-margin must be non-negative")
    if not (0.0 <= float(getattr(args, "scale_bar_background_opacity", DEFAULT_SCALE_BAR_BACKGROUND_OPACITY)) <= 1.0):
        raise ValueError("--scale-bar-background-opacity must be between 0 and 1")


def format_summary(
    args: argparse.Namespace,
    atom_types: Sequence[str],
    selected_atoms: Sequence[SelectedAtom],
    projection: ProjectionResult,
    output_svg: Path,
    output_json: Path,
    output_csv: Optional[Path],
    output_pdb: Optional[Path],
    n_pdb_atoms_transformed: Optional[int],
    styles: Dict[str, AtomStyle],
    base_pair_info: Optional[dict] = None,
    projection_scale: Optional[float] = None,
) -> str:
    chains = sorted(chain_label(selected.atom.chain) for selected in selected_atoms)
    unique_chains: List[str] = []
    seen = set()
    for chain in chains:
        if chain not in seen:
            unique_chains.append(chain)
            seen.add(chain)

    out: List[str] = []
    out.append("Resolved input format: {0}".format(getattr(args, "resolved_input_format", getattr(args, "input_format", "auto"))))
    out.append("Selected atoms/points: {0}".format(len(selected_atoms)))
    out.append("Selected atom types: {0}".format(", ".join(atom_types)))
    out.append("Counts by atom type: {0}".format(", ".join("{0}={1}".format(k, v) for k, v in atom_type_counts(selected_atoms).items())))
    out.append("Chains represented: {0}".format(", ".join(unique_chains)))
    out.append("Projection mode: {0}".format(projection.mode))
    if bool(getattr(projection, "pre_flip_about_y", False)):
        out.append("Pre-projection transform: flipped about Y axis (x -> -x, y -> y, z -> -z)")
    out.append("Method: {0}".format(projection.method_name))
    if projection.mode == "pca":
        coeff = projection.plane_coefficients
        out.append("Plane equation: A*x + B*y + C*z + D = 0")
        out.append("A B C D: {0:.10g} {1:.10g} {2:.10g} {3:.10g}".format(coeff[0], coeff[1], coeff[2], coeff[3]))
        out.append("Centroid: {0:.10g} {1:.10g} {2:.10g}".format(*projection.centroid))
        out.append("Normal:   {0:.10g} {1:.10g} {2:.10g}".format(*projection.normal))
        if projection.eigenvalues is not None and projection.explained_variance_ratio is not None:
            out.append("Eigenvalues descending: {0:.10g} {1:.10g} {2:.10g}".format(*projection.eigenvalues))
            out.append("Explained variance ratio: {0:.6f} {1:.6f} {2:.6f}".format(*projection.explained_variance_ratio))
    else:
        if bool(getattr(projection, "pre_flip_about_y", False)):
            out.append("Current-XY mode uses Y-axis-flipped input coordinates: proj_x=-input_x, proj_y=input_y, and depth=-input_z.")
        else:
            out.append("Current-XY mode uses original input X/Y as proj_x/proj_y and original input Z as depth.")
    out.append("Color by: {0}".format(args.color_by))
    palette = palette_from_args(args)
    if palette != DEFAULT_PALETTE:
        used_for = "chain colors" if args.color_by == "chain" else "atom-type fill and line colors not set explicitly"
        out.append("Palette: {0} ({1})".format(palette, used_for))
    chain_report = chain_color_report(args, selected_atoms)
    if chain_report["given"]:
        pairs = ", ".join(
            "{0}={1}".format(chain_label(chain), color) if given.strip().lower() == color.lower()
            else "{0}={1} ({2})".format(chain_label(chain), given, color)
            for chain, (color, given) in chain_report["given"].items()
        )
        if chain_report["ignored"]:
            out.append("Note: --chain-colors {0} is ignored with --color-by atom-type".format(pairs))
        else:
            out.append("Chain colors: {0}".format(pairs))
        if chain_report["missing"]:
            out.append("Warning: chain colors for {0} were not used; the selection has chains {1}".format(
                ", ".join(chain_label(chain) for chain in chain_report["missing"]),
                ", ".join(chain_label(chain) for chain in chain_report["present"]),
            ))
        if chain_report["unused"]:
            unused = chain_report["unused"]
            out.append("Warning: chain colors for {0} were not used; every circle and line drawn in {1} takes a fill or "
                       "line color set explicitly for its atom type".format(
                           ", ".join(chain_label(chain) for chain in unused),
                           "those chains" if len(unused) > 1 else "that chain",
                       ))
    explicit_colors = explicit_chain_mode_colors(args, atom_types, selected_atoms, styles)
    if explicit_colors:
        out.append("Atom-type colors set explicitly win over the chain colors: {0}".format(
            ", ".join("{0} {1}".format(atom_type, key) for atom_type, key in explicit_colors)
        ))
    out.append("Default connection mode: {0}".format(normalize_connection_mode(getattr(args, "connection_mode", "smooth"))))
    line_enabled = ["{0}({1})".format(atom_type, styles[atom_type.upper()].connection_mode) for atom_type in atom_types if styles[atom_type.upper()].draw_lines]
    out.append("Neighbor connections enabled for atom types: {0}".format(", ".join(line_enabled) if line_enabled else "none"))
    if getattr(args, "close_all_chains", False):
        out.append("Closed chains: all")
    elif getattr(args, "closed_chains", ""):
        out.append("Closed chains: {0}".format(getattr(args, "closed_chains", "")))
    if base_pair_info and base_pair_info.get("enabled"):
        out.append("Base-pair lines: {0} drawn from {1} DSSR base pairs; skipped {2}".format(
            base_pair_info.get("base_pairs_drawn", 0),
            base_pair_info.get("base_pairs_in_dssr", 0),
            base_pair_info.get("base_pairs_skipped", 0),
        ))
        out.append("Base-pair line atom: {0}".format(base_pair_info.get("anchor_atom_name", DEFAULT_BASE_PAIR_ATOM)))
        out.append("DSSR output: {0}".format(base_pair_info.get("dssr_output")))
    out.append("Projection SVG: {0}".format(output_svg))
    out.append("Projection JSON: {0}".format(output_json))
    if output_csv is not None:
        out.append("Projection CSV: {0}".format(output_csv))
    if output_pdb is not None:
        out.append("Projection-basis PDB/XYZ: {0}".format(output_pdb))
        if n_pdb_atoms_transformed is not None:
            out.append("Coordinate records transformed: {0}".format(n_pdb_atoms_transformed))
    if projection_scale is not None:
        unit_label = str(getattr(args, "scale_bar_unit_label", DEFAULT_SCALE_BAR_UNIT_LABEL) or DEFAULT_SCALE_BAR_UNIT_LABEL)
        out.append("SVG projection scale: {0:.10g} SVG units per projected {1}".format(projection_scale, unit_label))
        if not bool(getattr(args, "no_scale_bar", False)):
            scale_bar_length = float(getattr(args, "scale_bar_length", DEFAULT_SCALE_BAR_LENGTH))
            out.append("SVG scale bar: {0:g} {1} = {2:.10g} SVG units".format(scale_bar_length, unit_label, scale_bar_length * projection_scale))
    if args.depth_order_circles:
        out.append("SVG circle stacking: depth order; front side '{0}' is drawn last and covers the back".format(args.depth_front))
    if args.depth_order_lines:
        out.append("SVG connection stacking: depth order; front side '{0}' is drawn last and covers the back".format(args.depth_front))
        if getattr(args, "line_underlay", False):
            out.append("SVG connection underlay: enabled")
    if getattr(args, "depth_order_base_pairs", False):
        out.append("SVG base-pair stacking: depth order; front side '{0}' is drawn last and covers the back".format(args.depth_front))
    if getattr(args, "draw_xy_plane", False):
        base_pair_depth_order_active = bool(
            getattr(args, "depth_order_base_pairs", False)
            and base_pair_info
            and base_pair_info.get("base_pairs_drawn", 0)
        )
        depth_order_xy_plane = bool(
            args.depth_order_circles
            or args.depth_order_lines
            or getattr(args, "pdb_order_circles", False)
            or getattr(args, "pdb_order_lines", False)
            or base_pair_depth_order_active
        )
        if depth_order_xy_plane:
            out.append("SVG xy-plane: enabled as layer 'xy-plane' and depth-ordered by mean projected patch depth")
        else:
            out.append("SVG xy-plane: enabled as background layer 'xy-plane'")
    return "\n".join(out)


def run_processing(args: argparse.Namespace) -> str:
    pdb_file = Path(args.pdb_file)
    if not pdb_file.exists():
        raise ValueError("Input structure file does not exist: {0}".format(pdb_file))

    atom_types = collect_atom_types(args)
    if not atom_types:
        raise ValueError("Please provide at least one atom type with --atom-type, --atom-types, or the GUI atom-type rows")

    validate_svg_args(args)
    args.palette = palette_from_args(args)
    # Color names are resolved here, once, before anything is read or written;
    # the style colors follow in parse_style_specs, still before any output.
    resolve_color_options(args)
    args.chain_color_specs = parse_chain_colors(getattr(args, "chain_colors", ""))

    output_csv = Path(args.csv_output) if args.csv_output else None

    all_atoms, resolved_input_format = parse_structure_atoms(pdb_file, args)
    args.resolved_input_format = resolved_input_format

    output_pdb = None

    if resolved_input_format != "pdb":
        if getattr(args, "draw_base_pairs", False):
            raise ValueError("DSSR base-pair lines require PDB input; XYZ input does not contain residue/chain records for DSSR.")
        detected_closed_chains = list(getattr(args, "vect_closed_chains", []) or [])
        if detected_closed_chains and not bool(getattr(args, "close_all_chains", False)):
            args.closed_chains = merge_closed_chain_text(getattr(args, "closed_chains", ""), detected_closed_chains)
    else:
        detected_closed_chains = detect_closed_chains_from_link(pdb_file)
        if detected_closed_chains and not bool(getattr(args, "close_all_chains", False)):
            args.closed_chains = merge_closed_chain_text(getattr(args, "closed_chains", ""), detected_closed_chains)

    selected_atoms = select_atoms(all_atoms, atom_types, args.select_by)
    if not selected_atoms:
        raise ValueError(
            "No atoms matched atom_types={0!r} with select_by={1!r}. Try --select-by element, --select-by auto, or check chain/model filters.".format(
                atom_types, args.select_by
            )
        )

    points = np.array([[selected.atom.x, selected.atom.y, selected.atom.z] for selected in selected_atoms], dtype=float)
    if args.projection_mode == "current-xy":
        projection = compute_current_xy_projection(points, flip_about_y=bool(getattr(args, "flip_about_y", False)))
    else:
        projection = compute_pca_projection(points, flip_about_y=bool(getattr(args, "flip_about_y", False)))
    hull_area = convex_hull_area_2d(projection.projected_xy)

    style_specs = getattr(args, "style", None) or []
    explicit_style_keys: Dict[str, set] = {}
    styles = parse_style_specs(
        style_specs,
        atom_types,
        args.radius,
        args.line_width,
        args.line_opacity,
        args.draw_lines,
        getattr(args, "connection_mode", "smooth"),
        bool(getattr(args, "extend_3prime", False)),
        args.palette,
        explicit_keys=explicit_style_keys,
    )
    # Kept beside the styles, which record the same values as before: in chain
    # mode an explicit fill= or line_stroke= wins over the chain colors.
    args.explicit_style_keys = explicit_style_keys

    filename_tags = default_name_tags_from_args(args, styles)
    default_svg, default_json, default_basis = default_output_paths(pdb_file, atom_types, resolved_input_format, filename_tags)
    output_svg = Path(args.output) if args.output else default_svg
    output_json = Path(args.plane_output) if args.plane_output else default_json
    if args.write_pca_pdb:
        output_pdb = Path(args.pca_pdb_output) if args.pca_pdb_output else default_basis

    base_pair_drawables, base_pair_info = prepare_base_pair_drawables(pdb_file, args, all_atoms, projection, styles)

    projection_scale = write_projection_svg(output_svg, args, atom_types, selected_atoms, projection, styles, base_pair_drawables, all_atoms=all_atoms)
    write_projection_json(
        output_json,
        args,
        atom_types,
        selected_atoms,
        projection,
        hull_area,
        output_svg,
        output_csv,
        output_pdb,
        styles,
        base_pair_info,
        projection_scale=projection_scale,
    )
    if output_csv is not None:
        write_projection_csv(output_csv, selected_atoms, projection.projected_xy, projection.projected_depth, args.xy_only)

    n_transformed = None
    if output_pdb is not None:
        if resolved_input_format == "pdb":
            n_transformed = write_projection_basis_pdb(pdb_file, output_pdb, projection)
        else:
            n_transformed = write_projection_basis_xyz(pdb_file, output_pdb, all_atoms, projection)

    return format_summary(
        args,
        atom_types,
        selected_atoms,
        projection,
        output_svg,
        output_json,
        output_csv,
        output_pdb,
        n_transformed,
        styles,
        base_pair_info,
        projection_scale=projection_scale,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plane It projects selected PDB/XYZ atoms or points into 2D and writes an SVG. The default projection mode uses PCA; "
            "--projection-mode current-xy uses the input X/Y coordinates directly."
        )
    )
    parser.add_argument("pdb_file", nargs="?", help="Input PDB, molecular XYZ, or coordinate XYZ/.txt file")
    parser.add_argument("--gui", action="store_true", help="Open the Tkinter GUI")
    parser.add_argument("-a", "--atom-type", action="append", default=None, help="Atom selector value. Can be repeated or comma-separated.")
    parser.add_argument("--atom-types", default=None, help="Comma-, semicolon-, or newline-separated atom selector values, e.g. P,C1',O3'")
    parser.add_argument("--select-by", choices=["name", "element", "auto"], default="name", help="Select by PDB atom name, element, or either. Default: name")
    parser.add_argument("--input-format", choices=["auto", "pdb", "xyz", "molecular-xyz", "coordinate-xyz", "vect"], default="auto", help="Input file format. Default: auto, which detects Geomview VECT by its header word. A VECT polyline becomes one chain, and the closure it states fills --closed-chains the way PDB LINK records do.")
    parser.add_argument("--records", choices=["all", "ATOM", "HETATM"], default="all", help="Which PDB coordinate records to use. Default: all")
    parser.add_argument("--model", default="first", help="Model to use: first, all, or an integer. Default: first")
    parser.add_argument("--chain", default=None, help="Optional chain ID filter")
    parser.add_argument("--resname", default=None, help="Optional residue name filter")
    parser.add_argument("--altloc", default="A", help="Alternate location filter. Default A means blank or A. Use all for all altlocs, or '' for blank only")
    parser.add_argument("--projection-mode", choices=["pca", "current-xy"], default="pca", help="Projection mode. Default: pca")
    parser.add_argument("--flip-about-y", "--flip-y-axis", "--mirror-about-y", dest="flip_about_y", action="store_true", help="Flip coordinates about the Y axis, x -> -x and z -> -z, before projection and projection-basis output")
    parser.add_argument("-o", "--output", default=None, help="Output SVG path")
    parser.add_argument("--plane-output", default=None, help="Output JSON path for projection metadata")
    parser.add_argument("--csv-output", default=None, help="Optional CSV path for projected coordinates")
    parser.add_argument("--xy-only", action="store_true", help="When --csv-output is used, write only proj_x,proj_y columns")
    parser.add_argument("--width", type=float, default=1000.0, help="SVG width in pixels. Default: 1000")
    parser.add_argument("--height", type=float, default=1000.0, help="SVG height in pixels. Default: 1000")
    parser.add_argument("--padding", type=float, default=50.0, help="SVG padding in pixels. Default: 50")
    parser.add_argument("--radius", type=float, default=3.0, help="Default circle radius. Can be overridden by --style. Default: 3")
    parser.add_argument("--no-invert-y", action="store_true", help="Do not invert SVG y coordinates")
    parser.add_argument("--draw-lines", action="store_true", help="Default to drawing neighbor connections for all selected atom types")
    parser.add_argument("--connection-mode", "--line-mode", dest="connection_mode", choices=["straight", "smooth"], default="smooth", help="Default neighbor-connection mode for atom types unless overridden by --style connection_mode=. Default: smooth")
    parser.add_argument("--extend-3prime", action="store_true", help="Default to drawing a terminal segment from the last selected atom of each open chain to the same residue's O3' atom; can be overridden per atom type with --style extend_3prime=true")
    parser.add_argument("--closed-chains", default="", help="Comma/semicolon/space-separated chain IDs whose neighbor connections should be closed, e.g. A,B,H. Use blank for blank chain.")
    parser.add_argument("--close-all-chains", action="store_true", help="Close neighbor connections for every chain")
    parser.add_argument("--depth-order-circles", action="store_true", help="Draw circles back-to-front using projection depth")
    parser.add_argument("--depth-order-lines", action="store_true", help="Draw neighbor line segments back-to-front using projection depth")
    parser.add_argument("--depth-order-base-pairs", action="store_true", help="Draw DSSR base-pair lines back-to-front using projection depth")
    parser.add_argument("--depth-front", choices=["positive", "negative"], default="positive", help="Which depth side is front for SVG depth ordering. Default: positive")
    parser.add_argument("--line-underlay", dest="line_underlay", action="store_true", default=True, help="When --depth-order-lines is used, draw a wider line under each neighbor segment, usually white, to make depth separation clearer. Default: on")
    parser.add_argument("--no-line-underlay", dest="line_underlay", action="store_false", help="Disable the wider depth-ordered neighbor-line underlay")
    parser.add_argument("--line-underlay-stroke", default="#ffffff", help="Stroke color for the depth-order neighbor-line underlay" + LAB_NAME_HINT + ". Default: white")
    parser.add_argument("--line-underlay-extra-width", type=float, default=8.0, help="Additional width added to each underlay line relative to the visible neighbor line. Default: 8.0")
    parser.add_argument("--line-underlay-opacity", type=float, default=1.0, help="Opacity of the neighbor-line underlay. Default: 1")
    parser.add_argument("--draw-xy-plane", dest="draw_xy_plane", action="store_true", default=True, help="Draw the projection-basis xy plane (projection depth=0) as an SVG layer/group named xy-plane. In PCA mode this is the PC1/PC2 plane through the selected-atom centroid. Default: on")
    parser.add_argument("--no-xy-plane", dest="draw_xy_plane", action="store_false", help="Do not draw the projection-basis xy-plane layer")
    parser.add_argument("--xy-plane-fill", default="#7dd3fc", help="Fill color for --draw-xy-plane" + LAB_NAME_HINT + ". Default: #7dd3fc")
    parser.add_argument("--xy-plane-stroke", default="#0284c7", help="Stroke color for --draw-xy-plane" + LAB_NAME_HINT + ". Default: #0284c7")
    parser.add_argument("--xy-plane-stroke-width", type=float, default=1.5, help="Stroke width for --draw-xy-plane. Default: 1.5")
    parser.add_argument("--xy-plane-opacity", type=float, default=0.18, help="Opacity for --draw-xy-plane. Default: 0.18")
    parser.add_argument("--no-scale-bar", action="store_true", help="Do not draw the default SVG scale bar")
    parser.add_argument("--scale-bar-length", type=float, default=DEFAULT_SCALE_BAR_LENGTH, help="Scale-bar length in projected coordinate units. Default: 10")
    parser.add_argument("--scale-bar-unit-label", default=DEFAULT_SCALE_BAR_UNIT_LABEL, help="Unit label for the scale bar. Default: Angstrom symbol")
    parser.add_argument("--scale-bar-stroke", default=DEFAULT_SCALE_BAR_STROKE, help="Scale-bar and label color" + LAB_NAME_HINT + ". Default: #111827")
    parser.add_argument("--scale-bar-stroke-width", type=float, default=DEFAULT_SCALE_BAR_STROKE_WIDTH, help="Scale-bar stroke width. Default: 2.5")
    parser.add_argument("--scale-bar-text-size", type=float, default=DEFAULT_SCALE_BAR_TEXT_SIZE, help="Scale-bar text size. Default: 14")
    parser.add_argument("--scale-bar-margin", type=float, default=DEFAULT_SCALE_BAR_MARGIN, help="Scale-bar margin from the lower-left SVG edge. Default: 32")
    parser.add_argument("--scale-bar-background", default=DEFAULT_SCALE_BAR_BACKGROUND, help="Scale-bar background fill" + LAB_NAME_HINT + ". Default: white")
    parser.add_argument("--scale-bar-background-opacity", type=float, default=DEFAULT_SCALE_BAR_BACKGROUND_OPACITY, help="Scale-bar background opacity. Default: 0.78")
    parser.add_argument("--pdb-order-circles", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pdb-order-lines", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--color-by", choices=["atom-type", "chain"], default="chain",
        help=(
            "Default SVG color mode. chain gives each chain one color: its --chain-colors color, or else the --palette "
            "color for its place in the order the chains first appear; atom-type gives each atom type its own fill and line "
            "color. In either mode fill= and line_stroke= in --style win for their atom type. Default: chain"
        ),
    )
    parser.add_argument(
        "--palette", type=argparse_palette, default=DEFAULT_PALETTE,
        help=(
            "Color palette for the chain colors (--color-by chain) and for the per-atom-type fill and line colors "
            "that --style does not set (--color-by atom-type). default keeps Plane It's own golden-ratio HSV colors, unchanged. "
            "DiLiuLab uses the lab's nine gr_colors figure colors in the order red, blue, magenta, cyan, orange, purple, "
            "green, yellow, mint green, repeating after nine: the first chain in the input, or the first atom type, is red, "
            "the second blue. DiLiuLab-T80, -T60, -T40 or -T20 are lighter tints mixed toward white; T40 and T20 are pale "
            "and can be hard to see on white. fill= and line_stroke= in --style still win in either mode, and --chain-colors "
            "sets single chains. The palette's neutral is not used: circle strokes, base-pair lines, the xy-plane, "
            "the underlay and the scale bar keep their own colors"
            + (", though each takes a DiLiuLab name such as GrayT40" if lab_colors is not None else "")
            + ". The colors are read from assets/diliulab_colors.json. Example: --palette DiLiuLab-T80. Default: default"
        ),
    )
    parser.add_argument(
        "--chain-colors", default="", metavar="CHAIN=COLOR,...",
        help=(
            "Colors for single chains under --color-by chain, as CHAIN=COLOR pairs separated by commas, semicolons or spaces, "
            "e.g. --chain-colors \"A=RedT80,B=#4c79e6\"; write blank for a blank chain ID, as in --closed-chains. A listed chain "
            "takes its color for circles and neighbor lines; the other chains keep their --palette colors, in the order chains "
            "first appear. fill= and line_stroke= in --style still win for their atom type. A listed chain that is not in the "
            "selection is named in a warning in the run log, and so is one whose color nothing takes because every circle and "
            "line in it has an explicit fill= or line_stroke=; under --color-by atom-type the option is ignored, with a note in "
            "the log. The colors used are recorded in the SVG (data-chain-colors on the projection group), and all of them in "
            "the JSON and the log." + LAB_NAMES_HELP
        ),
    )
    parser.add_argument("--line-width", type=float, default=1.0, help="Default neighbor line width. Default: 1")
    parser.add_argument("--line-opacity", type=float, default=0.65, help="Default neighbor line opacity. Default: 0.65")
    parser.add_argument(
        "--style", action="append", default=None,
        help=(
            "Per-atom-type style. Repeat as needed. Format examples: "
            "'P radius=3 opacity=0.9 draw_lines=true connection_mode=smooth line_width=1.5 line_opacity=0.7'; "
            "'C1\' fill=#377eb8 radius=2 draw_lines=false connection_mode=straight'. "
            "fill, stroke and line_stroke take a hex code, an SVG color name"
            + (" or a DiLiuLab name written without spaces, such as MintGreenT80" if lab_colors is not None else "")
            + ". A fill= or line_stroke= given here wins in either --color-by mode: under --color-by chain it replaces the "
            "chain color for that atom type's circles or lines, e.g. --color-by chain --style \"C1' fill=#000000\"."
            + LAB_NAMES_HELP
        ),
    )
    parser.add_argument("--draw-base-pairs", action="store_true", help="Draw base-pair interaction lines parsed from the default x3dna-dssr output")
    parser.add_argument("--base-pair-atom", default=DEFAULT_BASE_PAIR_ATOM, help="Atom used as each residue's base-pair line anchor. Default: C3'. Recommended: C3' for B-DNA, C4' for A-RNA")
    parser.add_argument("--base-pair-stroke", default="#444444", help="Base-pair line color" + LAB_NAME_HINT + ". Default: #444444")
    parser.add_argument("--base-pair-width", type=float, default=3.0, help="Base-pair line width. Default: 3.0")
    parser.add_argument("--base-pair-opacity", type=float, default=0.75, help="Base-pair line opacity. Default: 0.75")
    parser.add_argument("--write-pca-pdb", "--write-projection-basis", dest="write_pca_pdb", action="store_true", help="Write a projection-basis PDB/XYZ file. PDB input writes PDB; XYZ/coordinate input writes XYZ")
    parser.add_argument("--pca-pdb-output", "--projection-basis-output", dest="pca_pdb_output", default=None, help="Output projection-basis PDB/XYZ path. Default: <input_stem>_<atom/options>_PCA.pdb for PDB input, or _PCA before the original XYZ/text extension for XYZ/coordinate input")
    return parser


def namespace_from_gui_values(values: dict) -> argparse.Namespace:
    return argparse.Namespace(
        pdb_file=values["pdb_file"],
        gui=False,
        atom_type=None,
        atom_types=values["atom_types"],
        input_format=values.get("input_format", "auto"),
        select_by=values["select_by"],
        records=values["records"],
        model=values["model"],
        chain=values["chain"] or None,
        resname=values["resname"] or None,
        altloc=values["altloc"],
        projection_mode=values["projection_mode"],
        flip_about_y=values.get("flip_about_y", False),
        output=values["output"] or None,
        plane_output=values["plane_output"] or None,
        csv_output=values["csv_output"] or None,
        xy_only=values["xy_only"],
        width=float(values["width"]),
        height=float(values["height"]),
        padding=float(values["padding"]),
        radius=float(values["radius"]),
        no_invert_y=not values["invert_y"],
        draw_lines=False,
        connection_mode=values.get("connection_mode", "smooth"),
        extend_3prime=False,
        closed_chains=values.get("closed_chains", ""),
        close_all_chains=values.get("close_all_chains", False),
        depth_order_circles=values.get("depth_order_circles", False),
        depth_order_lines=values.get("depth_order_lines", False),
        depth_order_base_pairs=values.get("depth_order_base_pairs", False),
        depth_front=values.get("depth_front", "positive"),
        line_underlay=values.get("line_underlay", True),
        line_underlay_stroke=values.get("line_underlay_stroke", "#ffffff"),
        line_underlay_extra_width=float(values.get("line_underlay_extra_width", 8.0)),
        line_underlay_opacity=float(values.get("line_underlay_opacity", 1.0)),
        draw_xy_plane=values.get("draw_xy_plane", True),
        xy_plane_fill=values.get("xy_plane_fill", "#7dd3fc"),
        xy_plane_stroke=values.get("xy_plane_stroke", "#0284c7"),
        xy_plane_stroke_width=float(values.get("xy_plane_stroke_width", 1.5)),
        xy_plane_opacity=float(values.get("xy_plane_opacity", 0.18)),
        no_scale_bar=values.get("no_scale_bar", False),
        scale_bar_length=float(values.get("scale_bar_length", DEFAULT_SCALE_BAR_LENGTH)),
        scale_bar_unit_label=values.get("scale_bar_unit_label", DEFAULT_SCALE_BAR_UNIT_LABEL),
        scale_bar_stroke=values.get("scale_bar_stroke", DEFAULT_SCALE_BAR_STROKE),
        scale_bar_stroke_width=float(values.get("scale_bar_stroke_width", DEFAULT_SCALE_BAR_STROKE_WIDTH)),
        scale_bar_text_size=float(values.get("scale_bar_text_size", DEFAULT_SCALE_BAR_TEXT_SIZE)),
        scale_bar_margin=float(values.get("scale_bar_margin", DEFAULT_SCALE_BAR_MARGIN)),
        scale_bar_background=values.get("scale_bar_background", DEFAULT_SCALE_BAR_BACKGROUND),
        scale_bar_background_opacity=float(values.get("scale_bar_background_opacity", DEFAULT_SCALE_BAR_BACKGROUND_OPACITY)),
        pdb_order_circles=False,
        pdb_order_lines=False,
        color_by=values.get("color_by", "chain"),
        palette=values.get("palette", DEFAULT_PALETTE),
        chain_colors=values.get("chain_colors", ""),
        line_width=float(values["line_width"]),
        line_opacity=float(values["line_opacity"]),
        style=values.get("style_specs", []),
        draw_base_pairs=values.get("draw_base_pairs", False),
        base_pair_atom=values.get("base_pair_atom", DEFAULT_BASE_PAIR_ATOM),
        base_pair_stroke=values.get("base_pair_stroke", "#444444"),
        base_pair_width=float(values.get("base_pair_width", 3.0)),
        base_pair_opacity=float(values.get("base_pair_opacity", 0.75)),
        write_pca_pdb=values["write_pca_pdb"],
        pca_pdb_output=values["pca_pdb_output"] or None,
    )


def strip_fit_count(widths: Sequence[int], total: int, room: int, more_width) -> int:
    """How many swatches the strip beside the window's Palette shows: the most
    of the first ones, `widths` pixels each with its pad (at most
    COLOR_STRIP_MAX of them), that fit in `room` pixels together with a
    "+N more" label, more_width(N) pixels, for the other N = total - count
    items, or with no label when none is left over. None when not even the
    first fits beside its "+N more"."""
    best = used = 0
    for count, width in enumerate(list(widths)[:COLOR_STRIP_MAX], start=1):
        used += width
        if used > room:
            break
        rest = total - count
        if used + (more_width(rest) if rest > 0 else 0) <= room:
            best = count
    return best


def fit_text(text: str, room: int, measure) -> str:
    """`text` if measure(text) fits in `room` pixels, or else as much of its
    start as fits followed by an ellipsis."""
    if measure(text) <= room:
        return text
    for end in range(len(text) - 1, 0, -1):
        shorter = text[:end].rstrip() + "\u2026"
        if measure(shorter) <= room:
            return shorter
    return "\u2026"


def popup_row_length(count: int) -> int:
    """Swatches per row in the window "+N more" opens: COLOR_STRIP_MAX, or more
    for a file with very many chains, so that the rows stay about as many as
    the swatches in each."""
    return max(COLOR_STRIP_MAX, math.ceil(math.sqrt(max(count, 0))))


def run_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import colorchooser, filedialog, messagebox, ttk
    except Exception as exc:
        print("ERROR: Tkinter GUI is not available in this Python environment: {0}".format(exc), file=sys.stderr)
        return 1

    root = tk.Tk()
    root.title(f"{TOOL_NAME} {TOOL_VERSION}")
    # Prefer the Plane It icon in assets, then fall back to the generic repo icon.
    for icon_path in (
        resource_path("assets/plane_it_icon.png"),
        Path(__file__).with_name("plane_it_icon.png"),
        resource_path("assets/icon.png"),
    ):
        if icon_path.exists():
            try:
                icon_image = tk.PhotoImage(file=str(icon_path))
                root.iconphoto(True, icon_image)
                root._plane_it_icon_image = icon_image
                break
            except Exception:
                pass
    root.geometry("1240x900")
    root.minsize(1000, 720)

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=0)          # the page's horizontal scrollbar, shown when needed
    root.rowconfigure(2, weight=0)          # Run, Quit and the tip

    help_bg = "#d8eefc"
    help_active = "#c5e3f5"

    def show_help(title: str, text: str) -> None:
        messagebox.showinfo(title, text)

    def help_button(parent, title: str, text: str):
        return tk.Button(
            parent,
            text="?",
            width=2,
            padx=2,
            pady=0,
            bg=help_bg,
            activebackground=help_active,
            relief="raised",
            command=lambda: show_help(title, text),
        )

    def label_with_help(parent, text: str, title: str, help_text: str):
        frame = ttk.Frame(parent)
        ttk.Label(frame, text=text).pack(side="left")
        help_button(frame, title, help_text).pack(side="left", padx=(4, 0))
        return frame

    def set_widget_state(widget, enabled: bool, readonly: bool = False) -> None:
        if widget is None:
            return
        try:
            if enabled:
                widget.configure(state="readonly" if readonly else "normal")
            else:
                widget.configure(state="disabled")
        except Exception:
            pass

    def set_label_state(widget, enabled: bool) -> None:
        """Grey/ungrey label-like widgets, including labels inside small help frames."""
        if widget is None:
            return
        try:
            widget.state(["!disabled"] if enabled else ["disabled"])
        except Exception:
            pass
        try:
            widget.configure(foreground="" if enabled else "#888888")
        except Exception:
            pass
        try:
            children = widget.winfo_children()
        except Exception:
            children = []
        for child in children:
            set_label_state(child, enabled)

    # Without lab_colors no color can be given by name, and the help offers none.
    lab_names_note = (
        " or a DiLiuLab name with its tint, as the lab's color lists spell it: RedT100, BlueT80, MintGreenT60, GrayT40 "
        "(Red, Blue, Magenta, Cyan, Orange, Purple, Green, Yellow, MintGreen, and Black/Gray for the neutral; tints T100, T80, "
        "T60, T40, T20). The bare name red is SVG's pure red; RedT100 is the lab's red"
    ) if lab_colors is not None else ""
    named = lab_colors is not None
    help_texts = {
        "select_by": (
            "How atom types are matched. name uses the PDB atom-name field, which is usually best for P, C1', O3', "
            "or CA alpha carbon. element uses the chemical element field. auto accepts either."
        ),
        "model": "For multi-model PDB files, choose first, all, or a model number such as 1. Single-model PDBs usually work with first.",
        "altloc": (
            "Alternate-location filter. Default A means blank altloc plus altloc A. Use all to include all alternate locations, "
            "or leave empty to include only blank altloc atoms."
        ),
        "projection_mode": (
            "pca fits a PCA projection using selected atoms. current-xy skips PCA and draws the existing input X/Y coordinates directly, "
            "using input Z as depth. If enabled, the Y-axis flip is applied before either mode as x -> -x and z -> -z."
        ),
        "color_by": (
            "chain gives each chain its own color: the Palette's colors in the order the chains first appear "
            "(golden-ratio HSV colors under default), or a color you pick for a chain by clicking its swatch beside the Palette. "
            "atom-type gives each atom-type row its own color, from its Fill and line Color fields, which start at the Palette's color for that row.\n\n"
            "In either mode a Fill or line Color you set in an atom-type row wins for that atom type. A field left at its default "
            "shows chain in chain mode and the row's palette color in atom-type mode, and follows the Palette."
        ),
        "palette": (
            "Colors for the chains when Color by is chain, and the default Fill and line Color of each atom-type row when Color by is atom-type. "
            "The swatches to the right show the colors a run will use.\n\n"
            "default keeps Plane It's own golden-ratio HSV colors, unchanged.\n\n"
            "DiLiuLab uses the lab's nine figure colors from gr_colors in the order red, blue, magenta, cyan, orange, purple, green, yellow, mint green, "
            "repeating after nine: the first chain in the input, or Atom type 1, is red, the second is blue, and so on.\n\n"
            "T80, T60, T40 and T20 are lighter tints mixed toward white; T40 and T20 are pale and can be hard to see on a white background.\n\n"
            "A Fill or line Color set in an atom-type row wins in either mode, and a chain color picked on a swatch wins over the Palette in chain mode; "
            "fields left at their default follow the Palette. The palette's neutral is not used" + (" unless you name it" if named else "")
            + ": circle strokes, base-pair lines, the xy-plane, the underlay and the scale bar keep their own colors"
            + (", though each of those fields takes a DiLiuLab name such as GrayT40" if named else "") + ".\n\n"
            "The colors are read from assets/diliulab_colors.json. A palette other than default is written to the SVG (data-palette on the projection group), "
            "the JSON and the run log. Command line: --palette DiLiuLab-T80."
        ),
        "color_strip": (
            "The swatches beside the Palette show the colors a run will use.\n\n"
            "Color by chain: one swatch per chain of the input file and atom types above, labelled with the chain ID, in the order the SVG gives out "
            "the colors, the order the chains first appear. Click one to give that chain its own color. "
            "It is sent to the run as a chain color, recorded in the SVG (data-chain-colors), the JSON and the run log; "
            "on the command line: --chain-colors A={1},B=#4c79e6. An atom type whose Fill or line Color you set keeps that color. "
            "Chain colors belong to the input file they were picked for, and loading another file clears them; a chain the atom types "
            "or the filters leave out keeps its color, unsent, until it is listed again.\n\n"
            "Color by atom-type: one swatch per atom-type row, showing its Fill. Click one to set that row's Fill, and its line Color as well "
            "unless you set that one yourself.\n\n"
            "A swatch whose color you set has a thicker border. Up to {0} swatches are shown, as many as fit beside the Palette, then +N more. "
            "Click +N more for a small window holding a swatch for every chain, or every atom-type row, each clicked just as the ones here; "
            "Escape or Close closes it. Reset clears the chain colors "
            "you picked (chain mode), or returns every row's Fill and line Color to the Palette's colors (atom-type mode). Before an input "
            "file is loaded there are no chains to list, and the strip says so."
        ).format(COLOR_STRIP_MAX, "RedT80" if named else "#eb7070"),
        "color_fields": (
            "Fill colors an atom type's circles and line Color its neighbor lines. line Color is available only while Draw lines is on, "
            "since without lines it colors nothing.\n\n"
            "Default: each field starts at the Palette's assignment. With Color by chain it shows chain, meaning this atom type takes the chain colors. "
            "With Color by atom-type it shows the row's palette color, by its DiLiuLab name (such as RedT80) under a DiLiuLab palette and as a hex code "
            "under default. A field at its default follows the Palette and Color by, and nothing is sent to the run for it.\n\n"
            "To set a color, type a hex code (#1f77b4), an SVG color name (black)" + lab_names_note + ". Or click the small square beside the field "
            "to pick one; the square shows the color the field gives, stripes for the chain colors and ? for a value an SVG viewer cannot "
            "draw as a color, such as gray50 or wheat1, names that Tk knows but SVG does not. "
            "A color you set is kept when the Palette or Color by changes. Clear the field, or type auto or chain, to return it to its default.\n\n"
            "Precedence with Color by chain: a Fill or line Color set here, then a chain color picked on the swatches beside the Palette, then the "
            "Palette's chain color. Command line: --style \"C1' fill=" + ("MintGreenT80" if named else "#70ebad") + " line_stroke=#1f77b4\"."
        ),
        "connection_mode": (
            "Per-atom-type connection mode. straight draws ordinary straight neighbor connections. smooth draws a cubic-Bezier curve "
            "that passes through the selected points in PDB order. The smooth curve uses Catmull-Rom style interpolation with built-in default settings."
        ),
        "closed_chains": (
            "Optional comma-separated chain IDs to close. Closing a chain adds a final connection from the last selected atom in that chain "
            "back to the first. Use blank for a blank chain ID."
        ),
        "write_pdb": (
            "Write a projection-basis PDB/XYZ/text file. In PCA mode, coordinates are transformed into the PCA projection basis. In current-xy mode, the current coordinate basis is kept. If Flip Y is enabled, that transform is applied before writing. PDB input writes PDB; XYZ/coordinate input keeps the original row layout and blank lines."
        ),
        "depth_circles": (
            "Draw circles back-to-front using projection depth, so front circles cover back circles. "
            "In PCA mode depth is PCA-normal coordinate; in current-xy mode depth is input Z after any optional pre-projection transform."
        ),
        "depth_lines": (
            "Draw neighbor line or curve segments back-to-front using projection depth. This is enabled only when at least one atom type has Draw lines checked."
        ),
        "depth_base_pairs": "Draw DSSR base-pair lines back-to-front using projection depth.",
        "depth_front": "Which depth side is treated as the front. If the visible order looks reversed, switch positive to negative or negative to positive.",
        "line_underlay": (
            "When depth-ordering neighbor lines, draw a wider line underneath each segment, usually white. This can mask lines behind it and make front/back order clearer."
        ),
        "xy_plane": (
            "Draw a finite patch of the projection-basis xy plane, where projection depth is 0. "
            "In PCA mode this is the PC1/PC2 plane through the selected-atom centroid. "
            "The SVG group/layer is named xy-plane and contains a shape named xy-plane-shape. "
            "When depth ordering is enabled for circles, neighbor lines, or base-pair lines, this plane patch is sorted with those items at projected depth 0."
        ),
        "scale_bar": (
            "Draw a projected-length scale bar in the final SVG. The label reports the bar length and the conversion factor: "
            "1 projected coordinate unit equals data-scale SVG units. For PDB input, projected coordinate units are normally Angstroms."
        ),
        "base_pairs": (
            "Draw base-pair interaction lines from x3dna-dssr output. The script uses or creates tmp_file/<input_filename>.out next to the input PDB and runs x3dna-dssr from that tmp_file folder when needed. "
            "Choose the residue atom used as each line anchor with Line atom. C3' is recommended for B-DNA; C4' is recommended for A-RNA. C1' and P are also available for comparison or custom workflows."
        ),
    }

    # A single scrollable settings page. This avoids hidden tabs while still fitting small screens.
    # It scrolls sideways as well when the window is narrower than the page,
    # rather than cutting off the page's right edge; the horizontal bar shows
    # only then.
    main_canvas = tk.Canvas(root, highlightthickness=0)
    main_scroll = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
    main_hscroll = ttk.Scrollbar(root, orient="horizontal", command=main_canvas.xview)

    def _set_xscroll(first, last) -> None:
        main_hscroll.set(first, last)
        if float(first) <= 0.0 and float(last) >= 1.0:
            main_hscroll.grid_remove()
        else:
            main_hscroll.grid()

    main_canvas.configure(yscrollcommand=main_scroll.set, xscrollcommand=_set_xscroll)
    main_canvas.grid(row=0, column=0, sticky="nsew")
    main_scroll.grid(row=0, column=1, sticky="ns")
    main_hscroll.grid(row=1, column=0, sticky="ew")
    main_hscroll.grid_remove()

    content = ttk.Frame(main_canvas, padding=(12, 10, 12, 8))
    content_window = main_canvas.create_window((0, 0), window=content, anchor="nw")
    content.columnconfigure(0, weight=1)

    def _fit_content_width(canvas_width: int) -> None:
        # The page fills the canvas, and keeps the width it asks for when the
        # canvas is narrower, scrolling sideways then instead of losing its
        # right edge.
        width = max(int(canvas_width), content.winfo_reqwidth())
        if round(float(main_canvas.itemcget(content_window, "width") or 0)) != width:
            main_canvas.itemconfigure(content_window, width=width)

    def _content_configure(_event=None) -> None:
        _fit_content_width(main_canvas.winfo_width())
        main_canvas.configure(scrollregion=main_canvas.bbox("all"))

    def _canvas_configure(event) -> None:
        _fit_content_width(event.width)

    content.bind("<Configure>", _content_configure)
    main_canvas.bind("<Configure>", _canvas_configure)

    def _on_main_mousewheel(event) -> None:
        try:
            if sys.platform == "darwin":
                main_canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _on_main_shift_mousewheel(event) -> None:
        # Shift with the wheel, and a trackpad's sideways swipe, scroll sideways.
        try:
            if sys.platform == "darwin":
                main_canvas.xview_scroll(int(-1 * event.delta), "units")
            else:
                main_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _bind_main_wheel(_event=None) -> None:
        root.bind_all("<MouseWheel>", _on_main_mousewheel)
        root.bind_all("<Shift-MouseWheel>", _on_main_shift_mousewheel)

    def _unbind_main_wheel(_event=None) -> None:
        root.unbind_all("<MouseWheel>")
        root.unbind_all("<Shift-MouseWheel>")

    main_canvas.bind("<Enter>", _bind_main_wheel)
    main_canvas.bind("<Leave>", _unbind_main_wheel)

    title = ttk.Label(content, text=f"{TOOL_NAME} {TOOL_VERSION}", font=("TkDefaultFont", 14, "bold"))
    title.grid(row=0, column=0, sticky="w", pady=(0, 4))
    ttk.Label(
        content,
        text=(
            "Project selected PDB/XYZ atoms or coordinate points to SVG. All settings are on this page; scroll if your screen is small. "
            "Fields are greyed out when they do not apply."
        ),
        wraplength=1100,
    ).grid(row=1, column=0, sticky="w", pady=(0, 10))

    pdb_var = tk.StringVar()
    input_format_var = tk.StringVar(value="auto")
    select_by_var = tk.StringVar(value="name")
    records_var = tk.StringVar(value="all")
    model_var = tk.StringVar(value="first")
    chain_var = tk.StringVar(value="")
    resname_var = tk.StringVar(value="")
    altloc_var = tk.StringVar(value="A")
    projection_mode_var = tk.StringVar(value="pca")
    flip_about_y_var = tk.BooleanVar(value=False)
    output_var = tk.StringVar(value="")
    json_output_var = tk.StringVar(value="")
    csv_output_var = tk.StringVar(value="")
    pca_pdb_output_var = tk.StringVar(value="")
    width_var = tk.StringVar(value="1000")
    height_var = tk.StringVar(value="1000")
    padding_var = tk.StringVar(value="50")
    color_by_var = tk.StringVar(value="chain")
    palette_var = tk.StringVar(value=DEFAULT_PALETTE)
    connection_mode_var = tk.StringVar(value="smooth")
    closed_chains_var = tk.StringVar(value="")
    close_all_chains_var = tk.BooleanVar(value=False)
    invert_y_var = tk.BooleanVar(value=True)
    depth_order_circles_var = tk.BooleanVar(value=True)
    depth_order_lines_var = tk.BooleanVar(value=True)
    depth_order_base_pairs_var = tk.BooleanVar(value=True)
    depth_front_var = tk.StringVar(value="positive")
    line_underlay_var = tk.BooleanVar(value=True)
    line_underlay_stroke_var = tk.StringVar(value="#ffffff")
    line_underlay_extra_width_var = tk.StringVar(value="8.0")
    line_underlay_opacity_var = tk.StringVar(value="1.0")
    draw_xy_plane_var = tk.BooleanVar(value=True)
    xy_plane_fill_var = tk.StringVar(value="#7dd3fc")
    xy_plane_stroke_var = tk.StringVar(value="#0284c7")
    xy_plane_stroke_width_var = tk.StringVar(value="1.5")
    xy_plane_opacity_var = tk.StringVar(value="0.18")
    draw_scale_bar_var = tk.BooleanVar(value=True)
    scale_bar_length_var = tk.StringVar(value=svg_float(DEFAULT_SCALE_BAR_LENGTH))
    scale_bar_unit_label_var = tk.StringVar(value=DEFAULT_SCALE_BAR_UNIT_LABEL)
    scale_bar_stroke_var = tk.StringVar(value=DEFAULT_SCALE_BAR_STROKE)
    scale_bar_stroke_width_var = tk.StringVar(value=svg_float(DEFAULT_SCALE_BAR_STROKE_WIDTH))
    scale_bar_text_size_var = tk.StringVar(value=svg_float(DEFAULT_SCALE_BAR_TEXT_SIZE))
    draw_base_pairs_var = tk.BooleanVar(value=False)
    base_pair_atom_var = tk.StringVar(value=DEFAULT_BASE_PAIR_ATOM)
    base_pair_stroke_var = tk.StringVar(value="#444444")
    base_pair_width_var = tk.StringVar(value="3.0")
    base_pair_opacity_var = tk.StringVar(value="0.75")
    xy_only_var = tk.BooleanVar(value=False)
    write_pca_pdb_var = tk.BooleanVar(value=True)
    n_types_var = tk.IntVar(value=1)

    atom_type_rows: List[dict] = []
    state_widgets: Dict[str, object] = {}
    updating_state = False
    rebuild_after_id = None
    last_default_paths = {"output": "", "json": "", "basis": "", "csv": ""}
    # Colors. A row's Fill and line Color hold an "auto" value, the Palette's
    # assignment ("chain" in chain mode), until the user types or picks one;
    # auto values are shown but never sent, so a window left at its defaults
    # runs exactly as before. color_guard marks the program's own writes of
    # those values, which do not count as edits. chain_overrides holds the
    # colors picked on the chain swatches, sent to the run as --chain-colors,
    # for the input file whose path strip_state["input_key"] holds.
    # color_picker holds the open picker and the atom-type row or chain it
    # sets; strip_popup the window "+N more" opens, a swatch for every item.
    color_guard = {"on": False}
    chain_overrides: Dict[str, str] = {}
    strip_state: Dict[str, object] = {
        "after": None, "read_key": None, "atoms": None, "read_error": False, "select_key": None, "chains": [],
        "items": [], "items_mode": None, "input_key": "",
    }
    color_picker: Dict[str, object] = {"win": None, "row": None, "chain": None}
    strip_popup: Dict[str, object] = {"win": None, "grid": None, "status": None}

    def current_atom_types() -> List[str]:
        values: List[str] = []
        for row_info in atom_type_rows:
            value = row_info["atom_type"].get().strip()
            if value:
                values.append(value)
        return values or ["atoms"]

    def gui_default_name_tags() -> List[str]:
        tags: List[str] = []
        if projection_mode_var.get() == "current-xy":
            tags.append("currentXY")
        any_lines = False
        modes = set()
        any_extend = False
        for row_info in atom_type_rows:
            if bool(row_info.get("draw_lines").get()):
                any_lines = True
                modes.add(normalize_connection_mode(row_info.get("connection_mode").get() or connection_mode_var.get()))
                if bool(row_info.get("extend_3prime").get()):
                    any_extend = True
        if any_lines:
            if modes == {"straight"}:
                tags.append("straight")
            elif modes == {"smooth"}:
                tags.append("smooth")
            else:
                tags.append("lines")
        if any_extend:
            tags.append("3prime")
        if bool(flip_about_y_var.get()):
            tags.append("flipY")
        if bool(draw_base_pairs_var.get()):
            tags.append("bp")
        if bool(draw_xy_plane_var.get()):
            tags.append("xyplane")
        if bool(depth_order_circles_var.get() or depth_order_lines_var.get() or depth_order_base_pairs_var.get()):
            tags.append("depth")
        if bool(line_underlay_var.get() and depth_order_lines_var.get() and any_lines):
            tags.append("underlay")
        return tags

    def update_default_paths(force: bool = False, update_closed_chains: bool = False) -> None:
        path = pdb_var.get().strip()
        if not path:
            return
        p = Path(path)
        default_svg, default_json, default_basis = default_output_paths(
            p, current_atom_types(), gui_resolved_input_format(), gui_default_name_tags()
        )
        new_paths = {
            "output": str(p.with_name(default_svg.name)),
            "json": str(p.with_name(default_json.name)),
            "basis": str(default_basis),
            "csv": str(p.with_name(default_svg.name.replace("_projection.svg", "_projection.csv"))),
        }

        def maybe_update(var: tk.StringVar, key: str, set_when_blank: bool = True) -> None:
            current = var.get().strip()
            previous = last_default_paths.get(key, "")
            should_update = False
            if force:
                should_update = set_when_blank or bool(current)
            elif (set_when_blank and not current) or (current and current == previous):
                should_update = True
            if should_update:
                var.set(new_paths[key])
            last_default_paths[key] = new_paths[key]

        maybe_update(output_var, "output", set_when_blank=True)
        maybe_update(json_output_var, "json", set_when_blank=True)
        maybe_update(pca_pdb_output_var, "basis", set_when_blank=True)
        maybe_update(csv_output_var, "csv", set_when_blank=False)
        if update_closed_chains:
            closed = detect_closed_chains_from_link(p)
            closed_chains_var.set(format_chain_list(closed))

    def browse_pdb() -> None:
        path = filedialog.askopenfilename(title="Choose structure or coordinate file", filetypes=[("Structure/coordinate files", "*.pdb *.ent *.xyz *.txt *.csv *.tsv *.dat *.vect"), ("PDB files", "*.pdb *.ent"), ("XYZ/coordinate files", "*.xyz *.txt *.csv *.tsv *.dat"), ("Geomview VECT", "*.vect"), ("All files", "*.*")])
        if path:
            pdb_var.set(path)
            update_default_paths(force=True, update_closed_chains=True)

    def browse_save(var: tk.StringVar, title: str, default_ext: str, filetypes: list) -> None:
        path = filedialog.asksaveasfilename(title=title, defaultextension=default_ext, filetypes=filetypes)
        if path:
            var.set(path)

    def browse_projection_basis() -> None:
        fmt = gui_resolved_input_format()
        if fmt == "pdb":
            default_ext = ".pdb"
        else:
            default_ext = Path(pdb_var.get().strip()).suffix or ".xyz"
        path = filedialog.asksaveasfilename(
            title="Choose projection-basis PDB/XYZ output",
            defaultextension=default_ext,
            filetypes=[("PDB files", "*.pdb"), ("XYZ files", "*.xyz"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            pca_pdb_output_var.set(path)

    def trace_state(var) -> None:
        try:
            var.trace_add("write", lambda *_args: update_gui_state())
        except Exception:
            pass

    def numeric_entry(parent, variable, width: int = 8):
        return ttk.Entry(parent, textvariable=variable, width=width)

    def gui_resolved_input_format() -> str:
        requested = input_format_var.get().strip() or "auto"
        if requested != "auto":
            return "pdb" if requested == "pdb" else "xyz"
        path = pdb_var.get().strip()
        if not path:
            return "pdb"
        suffix = Path(path).suffix.lower()
        if suffix in {".pdb", ".ent"}:
            return "pdb"
        if suffix == ".xyz":
            return "xyz"
        if Path(path).exists():
            try:
                return resolve_input_format(Path(path), "auto")
            except Exception:
                return "pdb"
        return "pdb"

    page_row = 2

    # ---------- Input and filters ----------
    input_frame = ttk.LabelFrame(content, text="Input and atom filters", padding=10)
    input_frame.grid(row=page_row, column=0, sticky="ew", pady=(0, 8))
    page_row += 1
    input_frame.columnconfigure(1, weight=1)
    input_frame.columnconfigure(5, weight=1)

    ttk.Label(input_frame, text="Input file").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(input_frame, textvariable=pdb_var).grid(row=0, column=1, columnspan=4, sticky="ew", pady=4)
    ttk.Button(input_frame, text="Browse", command=browse_pdb).grid(row=0, column=5, sticky="e", padx=(6, 0), pady=4)

    ttk.Label(input_frame, text="Input format").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
    input_format_combo = ttk.Combobox(input_frame, textvariable=input_format_var, values=["auto", "pdb", "xyz", "molecular-xyz", "coordinate-xyz", "vect"], width=14, state="readonly")
    input_format_combo.grid(row=1, column=1, sticky="w", pady=4)

    label_with_help(input_frame, "Select by", "Select by", help_texts["select_by"]).grid(row=1, column=2, sticky="w", padx=(18, 6), pady=4)
    ttk.Combobox(input_frame, textvariable=select_by_var, values=["name", "element", "auto"], width=10, state="readonly").grid(row=1, column=3, sticky="w", pady=4)

    records_label = ttk.Label(input_frame, text="Records")
    records_label.grid(row=1, column=4, sticky="w", padx=(18, 6), pady=4)
    records_combo = ttk.Combobox(input_frame, textvariable=records_var, values=["all", "ATOM", "HETATM"], width=10, state="readonly")
    records_combo.grid(row=1, column=5, sticky="w", pady=4)
    state_widgets["records"] = records_combo
    state_widgets["records_label"] = records_label

    model_label_frame = label_with_help(input_frame, "Model", "Model", help_texts["model"])
    model_label_frame.grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
    model_entry = ttk.Entry(input_frame, textvariable=model_var, width=10)
    model_entry.grid(row=2, column=1, sticky="w", pady=4)
    state_widgets["model"] = model_entry
    state_widgets["model_label"] = model_label_frame

    chain_label_widget = ttk.Label(input_frame, text="Chain filter")
    chain_label_widget.grid(row=2, column=2, sticky="w", padx=(18, 6), pady=4)
    chain_entry = ttk.Entry(input_frame, textvariable=chain_var, width=10)
    chain_entry.grid(row=2, column=3, sticky="w", pady=4)
    state_widgets["chain"] = chain_entry
    state_widgets["chain_label"] = chain_label_widget

    resname_label_widget = ttk.Label(input_frame, text="Residue filter")
    resname_label_widget.grid(row=2, column=4, sticky="w", padx=(18, 6), pady=4)
    resname_entry = ttk.Entry(input_frame, textvariable=resname_var, width=10)
    resname_entry.grid(row=2, column=5, sticky="w", pady=4)
    state_widgets["resname"] = resname_entry
    state_widgets["resname_label"] = resname_label_widget

    altloc_label_frame = label_with_help(input_frame, "Altloc", "Altloc", help_texts["altloc"])
    altloc_label_frame.grid(row=3, column=0, sticky="w", padx=(0, 6), pady=4)
    altloc_entry = ttk.Entry(input_frame, textvariable=altloc_var, width=10)
    altloc_entry.grid(row=3, column=1, sticky="w", pady=4)
    state_widgets["altloc"] = altloc_entry
    state_widgets["altloc_label"] = altloc_label_frame

    # Coordinate-only XYZ note is shown in the Atom type 1 title line.

    # ---------- Output paths ----------
    output_frame = ttk.LabelFrame(content, text="Output files", padding=10)
    output_frame.grid(row=page_row, column=0, sticky="ew", pady=(0, 8))
    page_row += 1
    output_frame.columnconfigure(1, weight=1)

    ttk.Label(output_frame, text="SVG output").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(output_frame, textvariable=output_var).grid(row=0, column=1, sticky="ew", pady=3)
    ttk.Button(output_frame, text="Save as", command=lambda: browse_save(output_var, "Choose SVG output", ".svg", [("SVG files", "*.svg"), ("All files", "*")])).grid(row=0, column=2, padx=(6, 0), pady=3)

    ttk.Label(output_frame, text="JSON output").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(output_frame, textvariable=json_output_var).grid(row=1, column=1, sticky="ew", pady=3)
    ttk.Button(output_frame, text="Save as", command=lambda: browse_save(json_output_var, "Choose JSON output", ".json", [("JSON files", "*.json"), ("All files", "*")])).grid(row=1, column=2, padx=(6, 0), pady=3)

    ttk.Label(output_frame, text="Optional CSV").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=3)
    ttk.Entry(output_frame, textvariable=csv_output_var).grid(row=2, column=1, sticky="ew", pady=3)
    ttk.Button(output_frame, text="Save as", command=lambda: browse_save(csv_output_var, "Choose CSV output", ".csv", [("CSV files", "*.csv"), ("All files", "*")])).grid(row=2, column=2, padx=(6, 0), pady=3)

    pca_label_frame = label_with_help(output_frame, "Projection-basis PDB/XYZ", "Projection-basis PDB/XYZ", help_texts["write_pdb"])
    pca_label_frame.grid(row=3, column=0, sticky="w", padx=(0, 6), pady=3)
    state_widgets["pca_pdb_label"] = pca_label_frame
    pca_entry = ttk.Entry(output_frame, textvariable=pca_pdb_output_var)
    pca_entry.grid(row=3, column=1, sticky="ew", pady=3)
    pca_button = ttk.Button(output_frame, text="Save as", command=browse_projection_basis)
    pca_button.grid(row=3, column=2, padx=(6, 0), pady=3)
    state_widgets["pca_pdb_entry"] = pca_entry
    state_widgets["pca_pdb_button"] = pca_button

    ttk.Label(
        output_frame,
        text="Leave output paths blank to use default filenames based on the input file and selected atom types.",
        wraplength=1000,
    ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))

    # ---------- Atom types ----------
    atom_frame = ttk.LabelFrame(content, text="Atom types and drawing style", padding=10)
    atom_frame.grid(row=page_row, column=0, sticky="ew", pady=(0, 8))
    page_row += 1
    atom_frame.columnconfigure(0, weight=1)

    atom_control = ttk.Frame(atom_frame)
    atom_control.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(atom_control, text="Number of atom types").pack(side="left")
    spin = ttk.Spinbox(atom_control, from_=1, to=20, textvariable=n_types_var, width=5)
    spin.pack(side="left", padx=(6, 12))
    ttk.Label(atom_control, text="Rows update automatically. Up to 20 atom types.").pack(side="left")
    label_with_help(atom_control, "Fill and line Color", "Fill and line Color", help_texts["color_fields"]).pack(side="left", padx=(18, 0))

    type_rows_frame = ttk.Frame(atom_frame)
    type_rows_frame.grid(row=1, column=0, sticky="ew")
    type_rows_frame.columnconfigure(0, weight=1)

    default_type_names = ["P", "C1'", "O3'", "CA", "N", "O", "C", "S"]

    def make_row_defaults(index: int) -> dict:
        # GUI defaults are tuned for the common nucleic-acid backbone case.
        # Only the first default atom type (P) starts with connections enabled.
        is_first_default_p = index == 0
        return {
            "atom_type": tk.StringVar(value=default_type_names[index] if index < len(default_type_names) else ""),
            "fill": tk.StringVar(value=""),
            "radius": tk.StringVar(value="3"),
            "opacity": tk.StringVar(value="0.2" if is_first_default_p else "1.0"),
            "stroke": tk.StringVar(value="#222222"),
            "stroke_width": tk.StringVar(value="0.6"),
            "draw_lines": tk.BooleanVar(value=is_first_default_p),
            "connection_mode": tk.StringVar(value="smooth"),
            "line_width": tk.StringVar(value="12" if is_first_default_p else "1.0"),
            "line_opacity": tk.StringVar(value="1.0" if is_first_default_p else "0.65"),
            "line_stroke": tk.StringVar(value=""),
            "extend_3prime": tk.BooleanVar(value=is_first_default_p),
            # Fill and line Color start at their auto value; line_stroke_linked
            # marks a line Color set along with the Fill from the swatch strip.
            "fill_auto": True,
            "line_stroke_auto": True,
            "line_stroke_linked": False,
            "widgets": {},
        }

    def bind_row_traces(row_info: dict) -> None:
        for key in ["atom_type", "radius", "opacity", "stroke", "stroke_width", "draw_lines", "connection_mode", "line_width", "line_opacity", "extend_3prime"]:
            trace_state(row_info[key])
        for key in ["fill", "line_stroke"]:
            row_info[key].trace_add("write", lambda *_args, key=key: color_field_written(row_info, key))
        try:
            row_info["atom_type"].trace_add("write", lambda *_args: update_default_paths())
        except Exception:
            pass

    def label_entry(parent, label: str, variable, width: int, row_num: int, col_num: int, widget_key: Optional[str] = None, row_info: Optional[dict] = None):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row_num, column=col_num, sticky="w", padx=(0, 4), pady=3)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row_num, column=col_num + 1, sticky="w", padx=(0, 14), pady=3)
        if widget_key and row_info is not None:
            row_info["widgets"][widget_key] = entry
            row_info["widgets"][widget_key + "_label"] = label_widget
        return entry

    def color_entry(parent, label: str, row_info: dict, key: str, row_num: int, col_num: int):
        """label_entry for a Fill or line Color field, with a small square beside
        it that shows the color the field gives and opens the picker."""
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row_num, column=col_num, sticky="w", padx=(0, 4), pady=3)
        holder = ttk.Frame(parent)
        holder.grid(row=row_num, column=col_num + 1, sticky="w", padx=(0, 14), pady=3)
        # Wide enough for the longest lab name, MintGreenT100.
        entry = ttk.Entry(holder, textvariable=row_info[key], width=12)
        entry.pack(side="left")
        # A Canvas rather than a Button, whose background macOS ignores.
        square = tk.Canvas(holder, width=16, height=16, highlightthickness=1, highlightbackground="#9ca3af", borderwidth=0)
        square.pack(side="left", padx=(3, 0))
        square.bind("<Button-1>", lambda _event: open_field_picker(row_info, key, square))
        # A cleared field returns to its default when the user leaves it, not while typing.
        entry.bind("<FocusOut>", lambda _event: leave_color_field(row_info, key), add="+")
        row_info["widgets"][key] = entry
        row_info["widgets"][key + "_label"] = label_widget
        row_info["widgets"][key + "_square"] = square
        return entry

    # ---------- Colors: field defaults, squares, the swatch strip and the picker ----------
    auto_color_words = {"", "auto", "chain"}

    def gui_palette() -> str:
        try:
            return normalize_palette_name(palette_var.get())
        except ValueError:
            return DEFAULT_PALETTE

    def palette_color_text(index: int) -> str:
        """A row's palette color as its field shows it: the lab name, e.g. RedT80,
        under a DiLiuLab palette, the hex code under default."""
        palette = gui_palette()
        if palette != DEFAULT_PALETTE:
            names = lab_colors.lab_color_names(palette.split()[-1])
            return names[index % len(names)]
        return palette_color_for_index(index, palette)

    def row_palette_indexes() -> List[int]:
        """Each row's place in the run's atom-type order, which picks its palette
        color: a repeated type shares the first one's place, since the run keeps
        it once, and a blank row shows the place a new type would take."""
        seen: Dict[str, int] = {}
        indexes: List[int] = []
        for row_info in atom_type_rows:
            key = row_info["atom_type"].get().strip().upper()
            if key and key not in seen:
                seen[key] = len(seen)
            indexes.append(seen[key] if key else len(seen))
        return indexes

    def auto_color_text(index: int) -> str:
        return "chain" if color_by_var.get() == "chain" else palette_color_text(index)

    def color_is_auto(row_info: dict, key: str) -> bool:
        return bool(row_info[key + "_auto"]) or row_info[key].get().strip().lower() in auto_color_words

    def write_color_field(row_info: dict, key: str, text: str) -> None:
        """Set a field without it counting as the user's edit."""
        color_guard["on"] = True
        try:
            row_info[key].set(text)
        finally:
            color_guard["on"] = False

    def set_field_auto(row_info: dict, key: str) -> None:
        """Return a field to the Palette's assignment."""
        row_info[key + "_auto"] = True
        if key == "line_stroke":
            row_info["line_stroke_linked"] = False
        write_color_field(row_info, key, auto_color_text(row_palette_indexes()[atom_type_rows.index(row_info)]))

    def set_field_color(row_info: dict, key: str, text: str) -> None:
        """A color the user picked: kept until they change it or ask for the default."""
        row_info[key + "_auto"] = False
        if key == "line_stroke":
            row_info["line_stroke_linked"] = False
        write_color_field(row_info, key, text)

    def color_field_written(row_info: dict, key: str) -> None:
        # Typing into a field makes it the user's; clearing it, or typing auto
        # or chain, returns it to its default. The program's own writes refresh
        # through their callers.
        if color_guard["on"]:
            return
        row_info[key + "_auto"] = row_info[key].get().strip().lower() in auto_color_words
        if key == "line_stroke":
            row_info["line_stroke_linked"] = False
        # After this trace returns: a field rewritten from inside its own trace
        # would keep showing the old text, since Tk then skips the entry's update.
        root.after_idle(update_gui_state)

    def row_alive(row_info: dict) -> bool:
        """Whether a row is still one of the window's rows: a picker or swatch
        left from a row the atom-type count removed must not set it."""
        return any(row is row_info for row in atom_type_rows)

    def leave_color_field(row_info: dict, key: str) -> None:
        if not row_alive(row_info):
            return                          # a row just removed
        try:
            if row_info[key + "_auto"]:
                set_field_auto(row_info, key)
            update_gui_state()
        except tk.TclError:
            pass                            # the window is closing

    def override_codes() -> Dict[str, str]:
        codes = {chain: svg_color_code(text) for chain, text in chain_overrides.items()}
        return {chain: code for chain, code in codes.items() if code}

    def strip_chain_colors() -> List[str]:
        """The chain colors in order, for the stripes of a field that follows the chains."""
        chains = list(strip_state["chains"] or [])
        palette = gui_palette()
        if not chains:
            return [palette_color_for_index(index, palette) for index in range(2)]
        colors = assign_chain_colors(chains, palette, override_codes())
        return [colors[chain] for chain in chains]

    def field_color(row_info: dict, key: str, index: int) -> Tuple[str, object]:
        """What a field gives, as its square shows it: ("color", "#rrggbb"),
        ("chain", [chain colors]), ("unknown", None) for a value the SVG cannot
        draw as a color, or ("off", None) for a line Color without lines."""
        if key == "line_stroke" and not bool(row_info["draw_lines"].get()):
            return "off", None
        if color_is_auto(row_info, key):
            if color_by_var.get() == "chain":
                return "chain", strip_chain_colors()
            return "color", palette_color_for_index(index, gui_palette())
        code = svg_color_code(row_info[key].get())
        return ("color", code) if code else ("unknown", None)

    def paint_color_square(square, look: str, value) -> None:
        square.delete("all")
        if look == "color":
            square.create_rectangle(-2, -2, 20, 20, fill=value, outline="")
        elif look == "chain":
            colors = list(value) or ["#d1d5db"]
            for stripe in range(4):         # thin stripes of the chain colors, repeating
                left = 1 + 4 * stripe           # the square's inside runs from 1 to 17, inside its 1-pixel ring
                square.create_rectangle(-2 if stripe == 0 else left, -2, 20 if stripe == 3 else left + 4, 20,
                                        fill=colors[stripe % len(colors)], outline="")
        elif look == "unknown":
            square.create_rectangle(-2, -2, 20, 20, fill="#e5e7eb", outline="")
            square.create_text(9, 9, text="?", fill="#6b7280", font=("TkDefaultFont", 11, "bold"))
        else:
            square.create_rectangle(-2, -2, 20, 20, fill="#f3f4f6", outline="")
        square.configure(cursor="" if look == "off" else "hand2")

    def draw_field_square(row_info: dict, key: str, index: int) -> None:
        square = row_info.get("widgets", {}).get(key + "_square")
        if square is None:
            return
        try:
            look, value = field_color(row_info, key, index)
            paint_color_square(square, look, value)
        except tk.TclError:
            pass                            # a square of a row being rebuilt

    def refresh_color_fields() -> None:
        """Show every auto field's current default and redraw the squares. A
        field being typed in keeps its text until the user leaves it."""
        try:
            focused = root.focus_get()
        except (KeyError, tk.TclError):
            focused = None
        for row_info, index in zip(atom_type_rows, row_palette_indexes()):
            for key in ("fill", "line_stroke"):
                entry = row_info.get("widgets", {}).get(key)
                typing = entry is not None and entry is focused
                if row_info[key + "_auto"] and not typing:
                    text = auto_color_text(index)
                    if row_info[key].get() != text:
                        write_color_field(row_info, key, text)
                draw_field_square(row_info, key, index)

    def strip_chain_order() -> Tuple[List[str], str]:
        """(chains in the order the SVG gives them colors, "") for the current
        input and selection, or ([], a short hint). The atoms are read with the
        run's own reader and filters, cached by path, mtime and those fields,
        and the selection is cached by the atom types as well."""
        path = pdb_var.get().strip()
        try:
            stat = Path(path).stat() if path and Path(path).is_file() else None
        except OSError:
            stat = None
        if stat is None:
            return [], "load an input to list its chains"
        reader = argparse.Namespace(
            input_format=input_format_var.get(),
            records=records_var.get(),
            model=model_var.get().strip() or "first",
            chain=chain_var.get().strip() or None,
            resname=resname_var.get().strip() or None,
            altloc=altloc_var.get(),
        )
        read_key = (path, stat.st_mtime_ns, stat.st_size, reader.input_format, reader.records,
                    reader.model, reader.chain, reader.resname, reader.altloc)
        if strip_state["read_key"] != read_key:
            strip_state.update(read_key=read_key, atoms=None, read_error=False, select_key=None, chains=[])
            try:
                strip_state["atoms"], _resolved_format = parse_structure_atoms(Path(path), reader)
            except Exception:
                strip_state["read_error"] = True
        if strip_state["read_error"]:
            return [], "cannot read the input to list its chains"
        typed = ",".join(row_info["atom_type"].get().strip() for row_info in atom_type_rows if row_info["atom_type"].get().strip())
        atom_types = collect_atom_types(argparse.Namespace(atom_type=None, atom_types=typed))
        select_key = (read_key, tuple(atom_types), select_by_var.get())
        if strip_state["select_key"] != select_key:
            try:
                chains = chain_color_order(select_atoms(strip_state["atoms"], atom_types, select_by_var.get()))
            except Exception:
                chains = []
            strip_state.update(select_key=select_key, chains=chains)
        if not strip_state["chains"]:
            # The atom types, the chain, model, residue or record filters, or
            # a file without the records asked for: any of them can leave none.
            return [], "no atoms selected; check the atom types and filters"
        return list(strip_state["chains"]), ""

    def input_path_key() -> str:
        path = pdb_var.get().strip()
        if not path:
            return ""
        try:
            return str(Path(path).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            return path

    def input_file_changed(*_args) -> None:
        # Chain colors belong to the file they were picked for: chain A of the
        # next file is another chain. So another file clears them, and closes a
        # chain's picker, or the list of chains, still open on the old one.
        key = input_path_key()
        if key == strip_state["input_key"]:
            return
        strip_state["input_key"] = key
        chain_overrides.clear()
        if color_picker["chain"] is not None:
            close_color_picker()
        if strip_popup["win"] is not None and strip_state["items_mode"] == "chain":
            close_strip_popup()

    def chain_colors_to_send() -> str:
        """--chain-colors for the run, in chain mode only: the colors picked on
        the chain swatches for the chains the strip lists for this input and
        selection. One the atom types or filters now leave out is kept, unsent."""
        if color_by_var.get() != "chain" or not chain_overrides:
            return ""
        chains, _hint = strip_chain_order()
        return format_chain_colors({chain: color for chain, color in chain_overrides.items() if chain in chains})

    def readable_text_color(code: str) -> str:
        red, green, blue = (int(code[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        return "#ffffff" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.55 else "#000000"

    def strip_items() -> Tuple[List[Tuple[str, Optional[str], bool, object]], str]:
        """(items, hint) for the strip and its "+N more" window: one (label,
        color, set by the user, on click) per chain in chain mode, in the order
        the SVG colors them, or per atom-type row, showing its Fill; or no
        items and a short hint saying why."""
        items: List[Tuple[str, Optional[str], bool, object]] = []
        hint = ""
        if color_by_var.get() == "chain":
            chains, hint = strip_chain_order()
            colors = assign_chain_colors(chains, gui_palette(), override_codes())
            for chain in chains:
                items.append((chain_label(chain), colors[chain], chain in chain_overrides,
                              lambda anchor, chain=chain: open_chain_picker(chain, anchor)))
        else:
            for row_info, index in zip(atom_type_rows, row_palette_indexes()):
                atom_type = row_info["atom_type"].get().strip()
                if not atom_type:
                    continue
                look, value = field_color(row_info, "fill", index)
                items.append((atom_type, value if look == "color" else None, not color_is_auto(row_info, "fill"),
                              lambda anchor, row_info=row_info: open_row_picker(row_info, anchor)))
            if not items:
                hint = "type an atom type to see its color"
        return items, hint

    def color_swatch(parent, label: str, color: Optional[str], user_set: bool, on_click):
        """One swatch: the chain ID or atom type, cut to five characters, on its
        color, with a thicker border when the user set the color. A click
        opens its picker, anchored at the swatch."""
        text = label if len(label) <= 5 else label[:4] + "\u2026"
        swatch = tk.Label(
            parent, text=text, width=max(3, len(text)), font=("TkDefaultFont", 11),
            bg=color or "#e5e7eb", fg=readable_text_color(color) if color else "#6b7280",
            relief="solid", borderwidth=2 if user_set else 1, cursor="hand2",
        )
        swatch.bind("<Button-1>", lambda _event: on_click(swatch))
        return swatch

    def strip_room() -> int:
        """The pixels the strip's swatches, "+N more" and hint may take: from the
        strip's grid cell to the right edge the drawing frame's other rows give
        it, less Reset, ? and the pads. Kept within that, the strip never
        widens the frame or the page, and Reset and ? always show."""
        def measure() -> int:
            inset = draw_frame.grid_bbox(0, 0)[0]
            right = draw_frame.winfo_reqwidth() - inset     # the frame's padding and border are alike on both sides
            fixed = strip_reset.winfo_reqwidth() + 4 + strip_help.winfo_reqwidth() + 4
            return right - draw_frame.grid_bbox(4, 1)[0] - 10 - fixed

        room = measure()
        if room <= 0:
            draw_frame.update_idletasks()   # not laid out yet
            room = measure()
        return room

    def redraw_color_strip() -> None:
        """Rebuild the swatches beside the Palette: the chains in chain mode, in
        the order the SVG colors them, or the atom-type rows' Fill. As many as
        fit show, at most COLOR_STRIP_MAX, and "+N more" opens the rest."""
        for child in strip_box.winfo_children():
            child.destroy()
        chain_mode = color_by_var.get() == "chain"
        items, hint = strip_items()
        strip_state.update(items=items, items_mode=color_by_var.get())
        room = strip_room()
        more_widths: Dict[int, int] = {}

        def more_width(count: int) -> int:
            if count not in more_widths:
                probe = ttk.Label(strip_box, text="+{0} more".format(count))
                more_widths[count] = probe.winfo_reqwidth() + 2
                probe.destroy()
            return more_widths[count]

        candidates = [color_swatch(strip_box, *item) for item in items[:COLOR_STRIP_MAX]]
        shown = strip_fit_count([swatch.winfo_reqwidth() + 2 for swatch in candidates], len(items), room, more_width)
        for swatch in candidates[:shown]:
            swatch.pack(side="left", padx=(0, 2))
        for swatch in candidates[shown:]:
            swatch.destroy()
        if len(items) > shown:
            more = ttk.Label(strip_box, text="+{0} more".format(len(items) - shown), foreground="#1d4ed8", cursor="hand2")
            more.pack(side="left", padx=(2, 0))
            more.bind("<Button-1>", lambda _event: open_strip_popup(more))
        if hint:
            hint_label = ttk.Label(strip_box, text=hint, foreground="#666666")

            def hint_width(text: str) -> int:
                hint_label.configure(text=text)
                return hint_label.winfo_reqwidth()

            hint_label.configure(text=fit_text(hint, room, hint_width))
            hint_label.pack(side="left")
        if chain_mode:
            resettable = any(user_set for _label, _color, user_set, _on_click in items)
        else:
            resettable = any(not (row_info["fill_auto"] and row_info["line_stroke_auto"]) for row_info in atom_type_rows)
        strip_reset.configure(fg="#111111" if resettable else "#9ca3af", cursor="hand2" if resettable else "")
        if strip_popup["win"] is not None:
            try:
                if items:
                    fill_strip_popup()
                else:
                    close_strip_popup()
            except tk.TclError:
                close_strip_popup()         # gone some other way

    def schedule_color_strip(delay: int = 300) -> None:
        after_id = strip_state["after"]
        if after_id is not None:
            try:
                root.after_cancel(after_id)
            except Exception:
                pass
        strip_state["after"] = root.after(delay, run_color_strip)

    def run_color_strip() -> None:
        """The strip, and the squares' chain stripes, once the input or selection settles."""
        strip_state["after"] = None
        try:
            redraw_color_strip()
            for row_info, index in zip(atom_type_rows, row_palette_indexes()):
                for key in ("fill", "line_stroke"):
                    draw_field_square(row_info, key, index)
        except tk.TclError:
            pass                            # the window is closing

    def place_popup(top, anchor) -> None:
        """Below the square or swatch, kept on the screen like the help popups
        of the other tools. Tk measures only the main display, so an anchor on
        another display keeps the plain placement rather than being pulled back."""
        try:
            top.update_idletasks()
            screen_w, screen_h = top.winfo_screenwidth(), top.winfo_screenheight()
            anchor_x, anchor_y = anchor.winfo_rootx(), anchor.winfo_rooty()
            x, y = anchor_x, anchor_y + anchor.winfo_height() + 4
            if 0 <= anchor_x < screen_w and 0 <= anchor_y < screen_h:
                width, height = top.winfo_reqwidth(), top.winfo_reqheight()
                x = max(min(x, screen_w - width - 8), 0)
                if y + height > screen_h - 100:     # title bar and Dock: open above instead
                    y = max(anchor_y - height - 34, 0)
            top.geometry("+{0}+{1}".format(x, y))
        except tk.TclError:
            pass

    def close_color_picker() -> None:
        win = color_picker["win"]
        color_picker.update(win=None, row=None, chain=None)
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

    def open_color_picker(anchor, title: str, current: str, on_choose, row_info: Optional[dict] = None,
                          chain: Optional[str] = None) -> None:
        """A small window of the DiLiuLab colors, one row per tint and one column
        per color with the neutral last. A swatch hands its name, MintGreenT80,
        to on_choose; Other color... hands the hex code from the system color
        chooser; Palette default hands None; Escape and Close cancel. row_info
        or chain names what it sets, so that the rules below can close it."""
        close_color_picker()
        top = tk.Toplevel(root)
        top.withdraw()                      # placed first, then shown: no flash at 0,0
        color_picker.update(win=top, row=row_info, chain=chain)
        top.title(title)
        top.transient(root)
        body = ttk.Frame(top, padding=10)
        body.pack(fill="both", expand=True)
        idle_text = "Current: {0}".format(current.strip() or "Palette default")
        status = ttk.Label(body, text=idle_text)

        def choose(value: Optional[str]) -> None:
            close_color_picker()
            on_choose(value)

        if lab_colors is not None:
            grid = ttk.Frame(body)
            grid.pack(anchor="w")
            for row_number, tint in enumerate(lab_colors.available_tints()):
                ttk.Label(grid, text=tint).grid(row=row_number, column=0, sticky="w", padx=(0, 6))
                named = zip(lab_colors.lab_color_names(tint, neutral=True), lab_colors.lab_colors(tint, neutral=True))
                for column, (name, code) in enumerate(named, start=1):
                    swatch = tk.Label(grid, bg=code, width=2, relief="solid", borderwidth=1, cursor="hand2")
                    swatch.grid(row=row_number, column=column, padx=1, pady=1)
                    swatch.bind("<Enter>", lambda _event, name=name, code=code: status.configure(text="{0}  {1}".format(name, code)))
                    swatch.bind("<Leave>", lambda _event: status.configure(text=idle_text))
                    swatch.bind("<Button-1>", lambda _event, name=name: choose(name))
        status.pack(anchor="w", pady=(6, 6))

        def other_color() -> None:
            try:
                _rgb, code = colorchooser.askcolor(color=svg_color_code(current), title=title, parent=top)
            except tk.TclError:
                code = None
            if code:
                choose(str(code).lower())

        buttons = ttk.Frame(body)
        buttons.pack(anchor="w")
        ttk.Button(buttons, text="Other color...", command=other_color).pack(side="left")
        ttk.Button(buttons, text="Palette default", command=lambda: choose(None)).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="Close", command=close_color_picker).pack(side="left", padx=(6, 0))
        top.bind("<Escape>", lambda _event: close_color_picker())
        top.protocol("WM_DELETE_WINDOW", close_color_picker)
        place_popup(top, anchor)
        if root.winfo_viewable():           # a hidden window keeps its popup hidden too
            top.deiconify()
        top.focus_set()

    def open_field_picker(row_info: dict, key: str, anchor) -> None:
        if key == "line_stroke" and not bool(row_info["draw_lines"].get()):
            return                          # no lines: nothing for the color to color
        atom_type = row_info["atom_type"].get().strip() or "Atom type {0}".format(atom_type_rows.index(row_info) + 1)

        def chosen(value: Optional[str]) -> None:
            if not row_alive(row_info):
                return                      # the row went while the picker was open
            if value is None:
                set_field_auto(row_info, key)
            else:
                set_field_color(row_info, key, value)
            update_gui_state()
            run_color_strip()

        open_color_picker(anchor, "{0}: {1}".format("Fill" if key == "fill" else "Line color", atom_type), row_info[key].get(), chosen,
                          row_info=row_info)

    def open_row_picker(row_info: dict, anchor) -> None:
        """A strip swatch in atom-type mode: the row's Fill, and its line Color
        too while that still follows its default, or follows the Fill set here."""
        if not row_alive(row_info):
            return                          # a swatch left from a row just removed

        def chosen(value: Optional[str]) -> None:
            if not row_alive(row_info):
                return                      # the row went while the picker was open
            follows = row_info["line_stroke_auto"] or row_info["line_stroke_linked"]
            if value is None:
                set_field_auto(row_info, "fill")
                if row_info["line_stroke_linked"]:
                    set_field_auto(row_info, "line_stroke")
            else:
                set_field_color(row_info, "fill", value)
                if follows:
                    set_field_color(row_info, "line_stroke", value)
                    row_info["line_stroke_linked"] = True
            update_gui_state()
            run_color_strip()

        open_color_picker(anchor, "Fill: {0}".format(row_info["atom_type"].get().strip()), row_info["fill"].get(), chosen,
                          row_info=row_info)

    def open_chain_picker(chain: str, anchor) -> None:
        def chosen(value: Optional[str]) -> None:
            if value is None:
                chain_overrides.pop(chain, None)
            else:
                chain_overrides[chain] = value
            run_color_strip()

        open_color_picker(anchor, "Chain {0}".format(chain_label(chain)), chain_overrides.get(chain, ""), chosen, chain=chain)

    def close_strip_popup() -> None:
        win = strip_popup["win"]
        strip_popup.update(win=None, grid=None, status=None)
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

    def fill_strip_popup() -> None:
        """The "+N more" window's swatches: every item of the strip, wrapping
        into rows, each clicked just as the strip's. Redrawn with the strip."""
        top, grid, status = strip_popup["win"], strip_popup["grid"], strip_popup["status"]
        chain_mode = strip_state["items_mode"] == "chain"
        top.title("Chain colors" if chain_mode else "Atom-type colors")
        idle_text = "Click a swatch to set that {0}.".format("chain's color" if chain_mode else "row's Fill")
        status.configure(text=idle_text)
        for child in grid.winfo_children():
            child.destroy()
        items = strip_state["items"]
        per_row = popup_row_length(len(items))
        for index, (label, color, user_set, on_click) in enumerate(items):
            swatch = color_swatch(grid, label, color, user_set, on_click)
            swatch.grid(row=index // per_row, column=index % per_row, padx=1, pady=1, sticky="w")
            about = "{0} {1}  {2}".format("Chain" if chain_mode else "Atom type", label, color or "?")
            swatch.bind("<Enter>", lambda _event, about=about: status.configure(text=about))
            swatch.bind("<Leave>", lambda _event: status.configure(text=idle_text))

    def open_strip_popup(anchor) -> None:
        """What "+N more" opens: a small window with a swatch for every chain,
        or every atom-type row, kept on the screen like the picker. Escape and
        Close close it, as do opening it again, removing the atom-type rows it
        shows and loading another file while it shows chains."""
        close_color_picker()
        close_strip_popup()
        if not strip_state["items"]:
            return
        top = tk.Toplevel(root)
        top.withdraw()                      # placed first, then shown: no flash at 0,0
        top.transient(root)
        body = ttk.Frame(top, padding=10)
        body.pack(fill="both", expand=True)
        grid = ttk.Frame(body)
        grid.pack(anchor="w")
        status = ttk.Label(body)
        status.pack(anchor="w", pady=(6, 6))
        ttk.Button(body, text="Close", command=close_strip_popup).pack(anchor="w")
        top.bind("<Escape>", lambda _event: close_strip_popup())
        top.protocol("WM_DELETE_WINDOW", close_strip_popup)
        strip_popup.update(win=top, grid=grid, status=status)
        fill_strip_popup()
        place_popup(top, anchor)
        if root.winfo_viewable():           # a hidden window keeps its popup hidden too
            top.deiconify()
        top.focus_set()

    def reset_colors() -> None:
        if color_by_var.get() == "chain":
            chain_overrides.clear()
        else:
            for row_info in atom_type_rows:
                set_field_auto(row_info, "fill")
                set_field_auto(row_info, "line_stroke")
        update_gui_state()
        run_color_strip()

    def update_gui_state() -> None:
        nonlocal updating_state
        if updating_state:
            return
        updating_state = True
        try:
            input_is_pdb = gui_resolved_input_format() == "pdb"
            projection_is_pca = projection_mode_var.get() == "pca"
            any_lines = any(bool(row_info["draw_lines"].get()) for row_info in atom_type_rows)
            csv_enabled = bool(csv_output_var.get().strip())
            base_pairs_requested = bool(draw_base_pairs_var.get())
            base_pairs_available = input_is_pdb
            base_pairs_on = base_pairs_requested and base_pairs_available
            if not input_is_pdb:
                draw_base_pairs_var.set(False)
            if not csv_enabled:
                xy_only_var.set(False)
            depth_enabled = bool(depth_order_circles_var.get() or (any_lines and depth_order_lines_var.get()) or (base_pairs_on and depth_order_base_pairs_var.get()))

            def set_with_label(key: str, enabled: bool, readonly: bool = False) -> None:
                set_widget_state(state_widgets.get(key), enabled, readonly=readonly)
                set_label_state(state_widgets.get(key + "_label"), enabled)

            # PDB-only filters and outputs.
            for key in ["records", "model", "chain", "resname", "altloc"]:
                set_with_label(key, input_is_pdb, readonly=(key == "records"))

            pca_pdb_allowed = True
            set_widget_state(state_widgets.get("write_pca_pdb"), pca_pdb_allowed)
            set_label_state(state_widgets.get("write_pca_pdb_label"), pca_pdb_allowed)
            set_widget_state(state_widgets.get("pca_pdb_entry"), pca_pdb_allowed and bool(write_pca_pdb_var.get()))
            set_label_state(state_widgets.get("pca_pdb_label"), pca_pdb_allowed and bool(write_pca_pdb_var.get()))
            set_widget_state(state_widgets.get("pca_pdb_button"), pca_pdb_allowed and bool(write_pca_pdb_var.get()))
            set_widget_state(state_widgets.get("xy_only"), csv_enabled)
            set_label_state(state_widgets.get("xy_only_label"), csv_enabled)

            # Depth ordering and underlay. Disabled controls keep their check state so defaults remain visible.
            set_widget_state(state_widgets.get("depth_order_lines"), any_lines)
            set_label_state(state_widgets.get("depth_order_lines_label"), any_lines)
            set_widget_state(state_widgets.get("line_underlay"), any_lines and bool(depth_order_lines_var.get()))
            set_label_state(state_widgets.get("line_underlay_label"), any_lines and bool(depth_order_lines_var.get()))
            underlay_enabled = any_lines and bool(depth_order_lines_var.get()) and bool(line_underlay_var.get())
            for key in ["line_underlay_stroke", "line_underlay_extra_width", "line_underlay_opacity"]:
                set_with_label(key, underlay_enabled)

            set_widget_state(state_widgets.get("draw_base_pairs"), base_pairs_available)
            set_label_state(state_widgets.get("draw_base_pairs_label"), base_pairs_available)
            set_widget_state(state_widgets.get("depth_order_base_pairs"), base_pairs_on)
            set_label_state(state_widgets.get("depth_order_base_pairs_label"), base_pairs_on)
            for key in ["base_pair_atom", "base_pair_stroke", "base_pair_width", "base_pair_opacity"]:
                set_with_label(key, base_pairs_on)
            set_widget_state(state_widgets.get("depth_front"), depth_enabled, readonly=True)
            set_label_state(state_widgets.get("depth_front_label"), depth_enabled)

            xy_plane_on = bool(draw_xy_plane_var.get())
            for key in ["xy_plane_fill", "xy_plane_stroke", "xy_plane_stroke_width", "xy_plane_opacity"]:
                set_with_label(key, xy_plane_on)

            scale_bar_on = bool(draw_scale_bar_var.get())
            for key in ["scale_bar_length", "scale_bar_unit_label", "scale_bar_stroke", "scale_bar_stroke_width", "scale_bar_text_size"]:
                set_with_label(key, scale_bar_on)

            for row_info in atom_type_rows:
                widgets = row_info.get("widgets", {})
                line_on = bool(row_info["draw_lines"].get())

                def row_set(key: str, enabled: bool, readonly: bool = False) -> None:
                    set_widget_state(widgets.get(key), enabled, readonly=readonly)
                    set_label_state(widgets.get(key + "_label"), enabled)

                # Fill is open in either Color by mode, since a color set there
                # wins in both; line Color needs lines to color.
                row_set("fill", True)
                set_label_state(widgets.get("circle_title"), True)
                set_label_state(widgets.get("connection_title"), line_on)
                row_set("connection_mode", line_on, readonly=True)
                row_set("line_width", line_on)
                row_set("line_opacity", line_on)
                row_set("line_stroke", line_on)
                row_set("extend_3prime", line_on)
            refresh_color_fields()
            schedule_color_strip()
            update_default_paths(force=False)
        finally:
            updating_state = False

    def rebuild_type_rows() -> None:
        nonlocal rebuild_after_id
        rebuild_after_id = None
        try:
            n = int(n_types_var.get())
        except Exception:
            return
        n = max(1, min(20, n))
        try:
            if int(n_types_var.get()) != n:
                n_types_var.set(n)
        except Exception:
            pass
        # A picker open on a row about to go would set a row that no longer
        # exists, and so would the "+N more" window's swatches in atom-type
        # mode: they go with the row. A rebuild that keeps the rows, as when
        # the spinbox loses the focus to a picker just opened, keeps both.
        dropped = atom_type_rows[n:]
        if dropped:
            if any(color_picker["row"] is row for row in dropped):
                close_color_picker()
            if strip_popup["win"] is not None and strip_state["items_mode"] == "atom-type":
                close_strip_popup()
        while len(atom_type_rows) < n:
            info = make_row_defaults(len(atom_type_rows))
            bind_row_traces(info)
            atom_type_rows.append(info)
        while len(atom_type_rows) > n:
            atom_type_rows.pop()

        for child in type_rows_frame.winfo_children():
            child.destroy()

        for i, row_info in enumerate(atom_type_rows, start=1):
            row_info["widgets"] = {}
            box = ttk.LabelFrame(type_rows_frame, padding=8)
            box.grid(row=i - 1, column=0, sticky="ew", padx=2, pady=5)
            for col in range(12):
                box.columnconfigure(col, weight=0)
            box.columnconfigure(11, weight=1)

            header = ttk.Frame(box)
            header.grid(row=0, column=0, columnspan=12, sticky="ew", pady=(0, 5))
            header_text = "Atom type 1  (coordinate-only XYZ: use atom type all or X)" if i == 1 else "Atom type {0}".format(i)
            header_label = ttk.Label(header, text=header_text, font=("TkDefaultFont", 10, "bold"))
            header_label.pack(side="left")
            ttk.Label(header, text="Type").pack(side="left", padx=(18, 4))
            type_entry = ttk.Entry(header, textvariable=row_info["atom_type"], width=12)
            type_entry.pack(side="left")
            row_info["widgets"]["atom_type"] = type_entry
            row_info["widgets"]["atom_type_label"] = header_label

            circle_title = ttk.Label(box, text="Circle", font=("TkDefaultFont", 9, "bold"))
            circle_title.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=3)
            row_info["widgets"]["circle_title"] = circle_title
            label_entry(box, "Radius", row_info["radius"], 7, 1, 1, "radius", row_info)
            label_entry(box, "Opacity", row_info["opacity"], 7, 1, 3, "opacity", row_info)
            color_entry(box, "Fill", row_info, "fill", 1, 5)
            label_entry(box, "Stroke", row_info["stroke"], 10, 1, 7, "stroke", row_info)
            label_entry(box, "Stroke width", row_info["stroke_width"], 7, 1, 9, "stroke_width", row_info)

            connection_title = ttk.Label(box, text="Connection", font=("TkDefaultFont", 9, "bold"))
            connection_title.grid(row=2, column=0, sticky="w", padx=(0, 10), pady=3)
            row_info["widgets"]["connection_title"] = connection_title
            draw_check = ttk.Checkbutton(box, text="Draw lines", variable=row_info["draw_lines"])
            draw_check.grid(row=2, column=1, columnspan=2, sticky="w", padx=(0, 14), pady=3)
            row_info["widgets"]["draw_lines"] = draw_check
            mode_label = ttk.Label(box, text="Mode")
            mode_label.grid(row=2, column=3, sticky="w", padx=(0, 4), pady=3)
            conn_combo = ttk.Combobox(box, textvariable=row_info["connection_mode"], values=["smooth", "straight"], width=9, state="readonly")
            conn_combo.grid(row=2, column=4, sticky="w", padx=(0, 14), pady=3)
            row_info["widgets"]["connection_mode"] = conn_combo
            row_info["widgets"]["connection_mode_label"] = mode_label
            label_entry(box, "Width", row_info["line_width"], 7, 2, 5, "line_width", row_info)
            label_entry(box, "Opacity", row_info["line_opacity"], 7, 2, 7, "line_opacity", row_info)
            color_entry(box, "Color", row_info, "line_stroke", 2, 9)

            extend_check = ttk.Checkbutton(box, text="3' to O3'", variable=row_info["extend_3prime"])
            extend_check.grid(row=3, column=1, columnspan=2, sticky="w", padx=(0, 14), pady=3)
            row_info["widgets"]["extend_3prime"] = extend_check
            row_info["widgets"]["extend_3prime_label"] = extend_check
            ttk.Label(
                box,
                text="Connections use consecutive selected atoms with the same model, chain, and atom type. For open chains, optional 3' to O3' adds a terminal O3' segment. Line caps are butt in the SVG.",
                wraplength=1030,
                foreground="#555555",
            ).grid(row=4, column=0, columnspan=12, sticky="w", pady=(4, 0))

        update_default_paths()
        update_gui_state()
        root.after(10, _content_configure)

    def schedule_rebuild_type_rows() -> None:
        nonlocal rebuild_after_id
        if rebuild_after_id is not None:
            try:
                root.after_cancel(rebuild_after_id)
            except Exception:
                pass
        rebuild_after_id = root.after(250, rebuild_type_rows)

    spin.configure(command=schedule_rebuild_type_rows)
    spin.bind("<Return>", lambda _event: rebuild_type_rows())
    spin.bind("<FocusOut>", lambda _event: rebuild_type_rows())
    try:
        n_types_var.trace_add("write", lambda *_args: schedule_rebuild_type_rows())
    except Exception:
        pass

    # Reserve visual rows so base-pair controls appear above SVG drawing controls.
    basepair_grid_row = page_row
    page_row += 1
    draw_grid_row = page_row
    page_row += 1

    # ---------- Projection and SVG options ----------
    draw_frame = ttk.LabelFrame(content, text="Projection and SVG drawing", padding=10)
    draw_frame.grid(row=draw_grid_row, column=0, sticky="ew", pady=(0, 8))
    for col in range(10):
        draw_frame.columnconfigure(col, weight=0)
    draw_frame.columnconfigure(9, weight=1)

    label_with_help(draw_frame, "Projection", "Projection mode", help_texts["projection_mode"]).grid(row=0, column=0, sticky="w", padx=(0, 4), pady=4)
    ttk.Combobox(draw_frame, textvariable=projection_mode_var, values=["pca", "current-xy"], width=10, state="readonly").grid(row=0, column=1, sticky="w", pady=4)

    label_with_help(draw_frame, "Color", "Color by", help_texts["color_by"]).grid(row=0, column=2, sticky="w", padx=(10, 4), pady=4)
    ttk.Combobox(draw_frame, textvariable=color_by_var, values=["chain", "atom-type"], width=10, state="readonly").grid(row=0, column=3, sticky="w", pady=4)

    ttk.Label(draw_frame, text="Width").grid(row=0, column=4, sticky="w", padx=(10, 3), pady=4)
    numeric_entry(draw_frame, width_var, 6).grid(row=0, column=5, sticky="w", pady=4)
    ttk.Label(draw_frame, text="Height").grid(row=0, column=6, sticky="w", padx=(10, 3), pady=4)
    numeric_entry(draw_frame, height_var, 6).grid(row=0, column=7, sticky="w", pady=4)
    ttk.Label(draw_frame, text="Pad").grid(row=0, column=8, sticky="w", padx=(10, 3), pady=4)
    numeric_entry(draw_frame, padding_var, 6).grid(row=0, column=9, sticky="w", pady=4)

    # The palette sits directly under Color by, the other control it colors.
    label_with_help(draw_frame, "Palette", "Palette", help_texts["palette"]).grid(row=1, column=2, sticky="w", padx=(10, 4), pady=4)
    ttk.Combobox(draw_frame, textvariable=palette_var, values=palette_choices(), width=12, state="readonly").grid(row=1, column=3, sticky="w", pady=4)
    # To its right, the colors the chains or the atom-type rows will get.
    color_strip = ttk.Frame(draw_frame)
    color_strip.grid(row=1, column=4, columnspan=6, sticky="w", padx=(10, 0), pady=4)
    strip_box = ttk.Frame(color_strip)
    strip_box.pack(side="left")
    strip_reset = tk.Label(color_strip, text="Reset", relief="raised", borderwidth=1, padx=4)
    strip_reset.pack(side="left", padx=(4, 0))
    strip_reset.bind("<Button-1>", lambda _event: reset_colors())
    strip_help = help_button(color_strip, "Color swatches", help_texts["color_strip"])
    strip_help.pack(side="left", padx=(4, 0))

    ttk.Checkbutton(draw_frame, text="Flip Y: x,z -> -x,-z", variable=flip_about_y_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=(0, 10), pady=4)
    ttk.Checkbutton(draw_frame, text="Invert SVG y so +proj_y appears upward", variable=invert_y_var).grid(row=2, column=2, columnspan=3, sticky="w", padx=(10, 0), pady=4)
    xy_check = ttk.Checkbutton(draw_frame, text="CSV xy only", variable=xy_only_var)
    xy_check.grid(row=2, column=5, sticky="w", padx=(10, 0), pady=4)
    state_widgets["xy_only"] = xy_check
    state_widgets["xy_only_label"] = xy_check

    write_frame = ttk.Frame(draw_frame)
    write_frame.grid(row=2, column=6, columnspan=4, sticky="w", padx=(10, 0), pady=4)
    write_pdb_check = ttk.Checkbutton(write_frame, text="Write projection-basis PDB/XYZ", variable=write_pca_pdb_var)
    write_pdb_check.pack(side="left")
    help_button(write_frame, "Projection-basis PDB/XYZ", help_texts["write_pdb"]).pack(side="left", padx=(4, 0))
    state_widgets["write_pca_pdb"] = write_pdb_check
    state_widgets["write_pca_pdb_label"] = write_pdb_check

    label_with_help(draw_frame, "Closed chains", "Closed chains", help_texts["closed_chains"]).grid(row=3, column=0, sticky="w", padx=(0, 6), pady=4)
    ttk.Entry(draw_frame, textvariable=closed_chains_var, width=22).grid(row=3, column=1, sticky="w", pady=4)
    ttk.Checkbutton(draw_frame, text="Close all chains", variable=close_all_chains_var).grid(row=3, column=2, columnspan=2, sticky="w", padx=(18, 0), pady=4)

    depth_circle_frame = ttk.Frame(draw_frame)
    depth_circle_frame.grid(row=4, column=0, columnspan=2, sticky="w", pady=4)
    ttk.Checkbutton(depth_circle_frame, text="Depth-order circles", variable=depth_order_circles_var).pack(side="left")
    help_button(depth_circle_frame, "Depth-order circles", help_texts["depth_circles"]).pack(side="left", padx=(4, 0))

    depth_line_frame = ttk.Frame(draw_frame)
    depth_line_frame.grid(row=4, column=2, columnspan=2, sticky="w", padx=(18, 0), pady=4)
    depth_lines_check = ttk.Checkbutton(depth_line_frame, text="Depth-order neighbor lines", variable=depth_order_lines_var)
    depth_lines_check.pack(side="left")
    help_button(depth_line_frame, "Depth-order neighbor lines", help_texts["depth_lines"]).pack(side="left", padx=(4, 0))
    state_widgets["depth_order_lines"] = depth_lines_check
    state_widgets["depth_order_lines_label"] = depth_lines_check

    depth_front_label = label_with_help(draw_frame, "Front side", "Front side", help_texts["depth_front"])
    depth_front_label.grid(row=4, column=4, sticky="w", padx=(18, 6), pady=4)
    state_widgets["depth_front_label"] = depth_front_label
    depth_front_combo = ttk.Combobox(draw_frame, textvariable=depth_front_var, values=["positive", "negative"], width=12, state="readonly")
    depth_front_combo.grid(row=4, column=5, sticky="w", pady=4)
    state_widgets["depth_front"] = depth_front_combo

    underlay_frame = ttk.Frame(draw_frame)
    underlay_frame.grid(row=5, column=0, columnspan=10, sticky="ew", pady=4)
    underlay_check = ttk.Checkbutton(underlay_frame, text="Draw wider underlay below depth-ordered neighbor lines", variable=line_underlay_var)
    underlay_check.pack(side="left")
    help_button(underlay_frame, "Line underlay", help_texts["line_underlay"]).pack(side="left", padx=(4, 12))
    state_widgets["line_underlay"] = underlay_check
    state_widgets["line_underlay_label"] = underlay_check
    underlay_color_label = ttk.Label(underlay_frame, text="Color")
    underlay_color_label.pack(side="left")
    state_widgets["line_underlay_stroke_label"] = underlay_color_label
    underlay_color_entry = ttk.Entry(underlay_frame, textvariable=line_underlay_stroke_var, width=10)
    underlay_color_entry.pack(side="left", padx=(4, 12))
    state_widgets["line_underlay_stroke"] = underlay_color_entry
    underlay_width_label = ttk.Label(underlay_frame, text="Extra width")
    underlay_width_label.pack(side="left")
    state_widgets["line_underlay_extra_width_label"] = underlay_width_label
    underlay_width_entry = ttk.Entry(underlay_frame, textvariable=line_underlay_extra_width_var, width=7)
    underlay_width_entry.pack(side="left", padx=(4, 12))
    state_widgets["line_underlay_extra_width"] = underlay_width_entry
    underlay_opacity_label = ttk.Label(underlay_frame, text="Opacity")
    underlay_opacity_label.pack(side="left")
    state_widgets["line_underlay_opacity_label"] = underlay_opacity_label
    underlay_opacity_entry = ttk.Entry(underlay_frame, textvariable=line_underlay_opacity_var, width=7)
    underlay_opacity_entry.pack(side="left", padx=(4, 12))
    state_widgets["line_underlay_opacity"] = underlay_opacity_entry

    xy_plane_frame = ttk.Frame(draw_frame)
    xy_plane_frame.grid(row=6, column=0, columnspan=10, sticky="ew", pady=4)
    xy_plane_check = ttk.Checkbutton(xy_plane_frame, text="Draw xy-plane (depth=0)", variable=draw_xy_plane_var)
    xy_plane_check.pack(side="left")
    help_button(xy_plane_frame, "xy-plane layer", help_texts["xy_plane"]).pack(side="left", padx=(4, 12))
    xy_fill_label = ttk.Label(xy_plane_frame, text="Fill")
    xy_fill_label.pack(side="left")
    state_widgets["xy_plane_fill_label"] = xy_fill_label
    xy_fill_entry = ttk.Entry(xy_plane_frame, textvariable=xy_plane_fill_var, width=10)
    xy_fill_entry.pack(side="left", padx=(4, 12))
    state_widgets["xy_plane_fill"] = xy_fill_entry
    xy_stroke_label = ttk.Label(xy_plane_frame, text="Stroke")
    xy_stroke_label.pack(side="left")
    state_widgets["xy_plane_stroke_label"] = xy_stroke_label
    xy_stroke_entry = ttk.Entry(xy_plane_frame, textvariable=xy_plane_stroke_var, width=10)
    xy_stroke_entry.pack(side="left", padx=(4, 12))
    state_widgets["xy_plane_stroke"] = xy_stroke_entry
    xy_stroke_width_label = ttk.Label(xy_plane_frame, text="Stroke width")
    xy_stroke_width_label.pack(side="left")
    state_widgets["xy_plane_stroke_width_label"] = xy_stroke_width_label
    xy_stroke_width_entry = ttk.Entry(xy_plane_frame, textvariable=xy_plane_stroke_width_var, width=7)
    xy_stroke_width_entry.pack(side="left", padx=(4, 12))
    state_widgets["xy_plane_stroke_width"] = xy_stroke_width_entry
    xy_opacity_label = ttk.Label(xy_plane_frame, text="Opacity")
    xy_opacity_label.pack(side="left")
    state_widgets["xy_plane_opacity_label"] = xy_opacity_label
    xy_opacity_entry = ttk.Entry(xy_plane_frame, textvariable=xy_plane_opacity_var, width=7)
    xy_opacity_entry.pack(side="left", padx=(4, 12))
    state_widgets["xy_plane_opacity"] = xy_opacity_entry

    scale_bar_frame = ttk.Frame(draw_frame)
    scale_bar_frame.grid(row=7, column=0, columnspan=10, sticky="ew", pady=4)
    scale_bar_check = ttk.Checkbutton(scale_bar_frame, text="Draw scale bar", variable=draw_scale_bar_var)
    scale_bar_check.pack(side="left")
    help_button(scale_bar_frame, "Scale bar", help_texts["scale_bar"]).pack(side="left", padx=(4, 12))
    scale_bar_length_label = ttk.Label(scale_bar_frame, text="Length")
    scale_bar_length_label.pack(side="left")
    state_widgets["scale_bar_length_label"] = scale_bar_length_label
    scale_bar_length_entry = ttk.Entry(scale_bar_frame, textvariable=scale_bar_length_var, width=7)
    scale_bar_length_entry.pack(side="left", padx=(4, 12))
    state_widgets["scale_bar_length"] = scale_bar_length_entry
    scale_bar_unit_label = ttk.Label(scale_bar_frame, text="Unit")
    scale_bar_unit_label.pack(side="left")
    state_widgets["scale_bar_unit_label_label"] = scale_bar_unit_label
    scale_bar_unit_entry = ttk.Entry(scale_bar_frame, textvariable=scale_bar_unit_label_var, width=5)
    scale_bar_unit_entry.pack(side="left", padx=(4, 12))
    state_widgets["scale_bar_unit_label"] = scale_bar_unit_entry
    scale_bar_color_label = ttk.Label(scale_bar_frame, text="Color")
    scale_bar_color_label.pack(side="left")
    state_widgets["scale_bar_stroke_label"] = scale_bar_color_label
    scale_bar_color_entry = ttk.Entry(scale_bar_frame, textvariable=scale_bar_stroke_var, width=10)
    scale_bar_color_entry.pack(side="left", padx=(4, 12))
    state_widgets["scale_bar_stroke"] = scale_bar_color_entry
    scale_bar_stroke_width_label = ttk.Label(scale_bar_frame, text="Stroke width")
    scale_bar_stroke_width_label.pack(side="left")
    state_widgets["scale_bar_stroke_width_label"] = scale_bar_stroke_width_label
    scale_bar_stroke_width_entry = ttk.Entry(scale_bar_frame, textvariable=scale_bar_stroke_width_var, width=7)
    scale_bar_stroke_width_entry.pack(side="left", padx=(4, 12))
    state_widgets["scale_bar_stroke_width"] = scale_bar_stroke_width_entry
    scale_bar_text_size_label = ttk.Label(scale_bar_frame, text="Text size")
    scale_bar_text_size_label.pack(side="left")
    state_widgets["scale_bar_text_size_label"] = scale_bar_text_size_label
    scale_bar_text_size_entry = ttk.Entry(scale_bar_frame, textvariable=scale_bar_text_size_var, width=7)
    scale_bar_text_size_entry.pack(side="left", padx=(4, 12))
    state_widgets["scale_bar_text_size"] = scale_bar_text_size_entry

    ttk.Label(
        draw_frame,
        text=(
            "Notes: connection mode is controlled inside each atom-type row. Fill and line Color start at the Palette's assignment, chain in Color by chain mode "
            "and the row's palette color in atom-type mode; a color typed or picked there wins in either mode, and clearing it follows the Palette again. "
            "Radius, opacity, stroke width, and line settings are per atom type."
        ),
        wraplength=1080,
    ).grid(row=8, column=0, columnspan=10, sticky="w", pady=(6, 0))

    # ---------- DSSR base-pair options ----------
    basepair_frame = ttk.LabelFrame(content, text="DSSR base-pair interaction lines", padding=10)
    basepair_frame.grid(row=basepair_grid_row, column=0, sticky="ew", pady=(0, 8))
    basepair_frame.columnconfigure(1, weight=1)
    basepair_frame.columnconfigure(3, weight=1)
    basepair_frame.columnconfigure(7, weight=1)

    draw_bp_frame = ttk.Frame(basepair_frame)
    draw_bp_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=4)
    draw_bp_check = ttk.Checkbutton(draw_bp_frame, text="Draw base-pair lines", variable=draw_base_pairs_var)
    draw_bp_check.pack(side="left")
    state_widgets["draw_base_pairs"] = draw_bp_check
    state_widgets["draw_base_pairs_label"] = draw_bp_check
    help_button(draw_bp_frame, "Base-pair lines", help_texts["base_pairs"]).pack(side="left", padx=(4, 0))

    depth_bp_frame = ttk.Frame(basepair_frame)
    depth_bp_frame.grid(row=0, column=2, columnspan=4, sticky="w", padx=(18, 0), pady=4)
    depth_bp_check = ttk.Checkbutton(depth_bp_frame, text="Depth-order base-pair lines", variable=depth_order_base_pairs_var)
    depth_bp_check.pack(side="left")
    help_button(depth_bp_frame, "Depth-order base-pair lines", help_texts["depth_base_pairs"]).pack(side="left", padx=(4, 0))
    state_widgets["depth_order_base_pairs"] = depth_bp_check
    state_widgets["depth_order_base_pairs_label"] = depth_bp_check

    bp_atom_label = label_with_help(basepair_frame, "Line atom", "Base-pair line atom", help_texts["base_pairs"])
    bp_atom_label.grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
    state_widgets["base_pair_atom_label"] = bp_atom_label
    bp_atom_combo = ttk.Combobox(
        basepair_frame,
        textvariable=base_pair_atom_var,
        values=[DEFAULT_BASE_PAIR_ATOM, "C4'", "C1'", "P"],
        width=8,
    )
    bp_atom_combo.grid(row=1, column=1, sticky="w", pady=3)
    state_widgets["base_pair_atom"] = bp_atom_combo

    bp_color_label = ttk.Label(basepair_frame, text="Line color")
    bp_color_label.grid(row=1, column=2, sticky="w", padx=(18, 6), pady=3)
    state_widgets["base_pair_stroke_label"] = bp_color_label
    bp_color_entry = ttk.Entry(basepair_frame, textvariable=base_pair_stroke_var, width=10)
    bp_color_entry.grid(row=1, column=3, sticky="w", pady=3)
    state_widgets["base_pair_stroke"] = bp_color_entry
    bp_width_label = ttk.Label(basepair_frame, text="Width")
    bp_width_label.grid(row=1, column=4, sticky="w", padx=(18, 6), pady=3)
    state_widgets["base_pair_width_label"] = bp_width_label
    bp_width_entry = ttk.Entry(basepair_frame, textvariable=base_pair_width_var, width=7)
    bp_width_entry.grid(row=1, column=5, sticky="w", pady=3)
    state_widgets["base_pair_width"] = bp_width_entry
    bp_opacity_label = ttk.Label(basepair_frame, text="Opacity")
    bp_opacity_label.grid(row=1, column=6, sticky="w", padx=(18, 6), pady=3)
    state_widgets["base_pair_opacity_label"] = bp_opacity_label
    bp_opacity_entry = ttk.Entry(basepair_frame, textvariable=base_pair_opacity_var, width=7)
    bp_opacity_entry.grid(row=1, column=7, sticky="w", pady=3)
    state_widgets["base_pair_opacity"] = bp_opacity_entry
    ttk.Label(
        basepair_frame,
        text=(
            "Default DSSR path: tmp_file/<input_filename>.out in the same folder as the input PDB. "
            "If the file is missing, x3dna-dssr is run from that tmp_file folder so sidecar files stay there. "
            "Base-pair lines use the selected projected atom of the paired residues. "
            "C3' is recommended for B-DNA; C4' is recommended for A-RNA."
        ),
        wraplength=1080,
    ).grid(row=2, column=0, columnspan=8, sticky="w", pady=(6, 0))

    # ---------- Run log ----------
    log_frame = ttk.LabelFrame(content, text="Run log", padding=10)
    log_frame.grid(row=page_row, column=0, sticky="ew", pady=(0, 8))
    page_row += 1
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(1, weight=1)
    ttk.Label(log_frame, text="Click Run to process the input file. Messages and output paths will appear below.").grid(row=0, column=0, sticky="w", pady=(0, 5))
    output_text = tk.Text(log_frame, width=120, height=9, wrap="word")
    output_text.grid(row=1, column=0, sticky="ew")
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=output_text.yview)
    log_scroll.grid(row=1, column=1, sticky="ns")
    output_text.configure(yscrollcommand=log_scroll.set)

    def gui_style_color(row_info: dict, key: str, where: str) -> str:
        """The fill= or line_stroke= value a row sends: nothing for a field at its
        default; a bad value is reported by the row and field, "P Fill: ..."."""
        if color_is_auto(row_info, key):
            return ""
        return style_color_token(row_info[key].get().strip(), where)

    def collect_gui_styles() -> Tuple[str, List[str], float, float, float]:
        atom_types: List[str] = []
        specs: List[str] = []
        first_radius = 3.0
        first_line_width = 1.0
        first_line_opacity = 0.65
        for i, row_info in enumerate(atom_type_rows):
            atom_type = row_info["atom_type"].get().strip()
            if not atom_type:
                continue
            atom_types.append(atom_type)
            fill = gui_style_color(row_info, "fill", "{0} Fill".format(atom_type))
            radius = row_info["radius"].get().strip() or "3"
            opacity = row_info["opacity"].get().strip() or "1"
            stroke = style_color_token(row_info["stroke"].get().strip() or "#222222", "{0} Stroke".format(atom_type))
            stroke_width = row_info["stroke_width"].get().strip() or "0.6"
            line_width = row_info["line_width"].get().strip() or "1"
            line_opacity = row_info["line_opacity"].get().strip() or "0.65"
            line_stroke = gui_style_color(row_info, "line_stroke", "{0} line Color".format(atom_type))
            connection_mode = row_info["connection_mode"].get().strip() or "smooth"
            draw_lines = "true" if bool(row_info["draw_lines"].get()) else "false"
            extend_3prime = "true" if bool(row_info["extend_3prime"].get()) else "false"
            tokens = ["{0}".format(atom_type)]
            if fill:
                tokens.append("fill={0}".format(fill))
            tokens.extend([
                "radius={0}".format(radius),
                "opacity={0}".format(opacity),
                "stroke={0}".format(stroke),
                "stroke_width={0}".format(stroke_width),
                "draw_lines={0}".format(draw_lines),
                "connection_mode={0}".format(connection_mode),
                "line_width={0}".format(line_width),
                "line_opacity={0}".format(line_opacity),
                "extend_3prime={0}".format(extend_3prime),
            ])
            if line_stroke:
                tokens.append("line_stroke={0}".format(line_stroke))
            specs.append(" ".join(tokens))
            if i == 0:
                first_radius = float(radius)
                first_line_width = float(line_width)
                first_line_opacity = float(line_opacity)
        return ",".join(atom_types), specs, first_radius, first_line_width, first_line_opacity

    def namespace_from_current_gui() -> argparse.Namespace:
        atom_types_text, style_specs, first_radius, first_line_width, first_line_opacity = collect_gui_styles()
        values = {
            "pdb_file": pdb_var.get().strip(),
            "atom_types": atom_types_text,
            "input_format": input_format_var.get(),
            "select_by": select_by_var.get(),
            "records": records_var.get(),
            "model": model_var.get().strip() or "first",
            "chain": chain_var.get().strip(),
            "resname": resname_var.get().strip(),
            "altloc": altloc_var.get(),
            "projection_mode": projection_mode_var.get(),
            "flip_about_y": bool(flip_about_y_var.get()),
            "output": output_var.get().strip(),
            "plane_output": json_output_var.get().strip(),
            "csv_output": csv_output_var.get().strip(),
            "pca_pdb_output": pca_pdb_output_var.get().strip(),
            "width": width_var.get().strip(),
            "height": height_var.get().strip(),
            "padding": padding_var.get().strip(),
            "radius": str(first_radius),
            "line_width": str(first_line_width),
            "line_opacity": str(first_line_opacity),
            "invert_y": bool(invert_y_var.get()),
            "connection_mode": connection_mode_var.get(),
            "closed_chains": closed_chains_var.get().strip(),
            "close_all_chains": bool(close_all_chains_var.get()),
            "depth_order_circles": bool(depth_order_circles_var.get()),
            "depth_order_lines": bool(depth_order_lines_var.get()),
            "depth_order_base_pairs": bool(depth_order_base_pairs_var.get()),
            "depth_front": depth_front_var.get(),
            "line_underlay": bool(line_underlay_var.get()),
            "line_underlay_stroke": line_underlay_stroke_var.get().strip() or "#ffffff",
            "line_underlay_extra_width": line_underlay_extra_width_var.get().strip() or "8.0",
            "line_underlay_opacity": line_underlay_opacity_var.get().strip() or "1.0",
            "draw_xy_plane": bool(draw_xy_plane_var.get()),
            "xy_plane_fill": xy_plane_fill_var.get().strip() or "#7dd3fc",
            "xy_plane_stroke": xy_plane_stroke_var.get().strip() or "#0284c7",
            "xy_plane_stroke_width": xy_plane_stroke_width_var.get().strip() or "1.5",
            "xy_plane_opacity": xy_plane_opacity_var.get().strip() or "0.18",
            "no_scale_bar": not bool(draw_scale_bar_var.get()),
            "scale_bar_length": scale_bar_length_var.get().strip() or svg_float(DEFAULT_SCALE_BAR_LENGTH),
            "scale_bar_unit_label": scale_bar_unit_label_var.get().strip() or DEFAULT_SCALE_BAR_UNIT_LABEL,
            "scale_bar_stroke": scale_bar_stroke_var.get().strip() or DEFAULT_SCALE_BAR_STROKE,
            "scale_bar_stroke_width": scale_bar_stroke_width_var.get().strip() or svg_float(DEFAULT_SCALE_BAR_STROKE_WIDTH),
            "scale_bar_text_size": scale_bar_text_size_var.get().strip() or svg_float(DEFAULT_SCALE_BAR_TEXT_SIZE),
            "scale_bar_margin": DEFAULT_SCALE_BAR_MARGIN,
            "scale_bar_background": DEFAULT_SCALE_BAR_BACKGROUND,
            "scale_bar_background_opacity": DEFAULT_SCALE_BAR_BACKGROUND_OPACITY,
            "color_by": color_by_var.get(),
            "palette": palette_var.get(),
            # Only in chain mode, where the chain swatches show them, and only
            # for the chains they show.
            "chain_colors": chain_colors_to_send(),
            "xy_only": bool(xy_only_var.get()),
            "write_pca_pdb": bool(write_pca_pdb_var.get()),
            "draw_base_pairs": bool(draw_base_pairs_var.get()),
            "base_pair_atom": base_pair_atom_var.get().strip() or DEFAULT_BASE_PAIR_ATOM,
            "base_pair_stroke": base_pair_stroke_var.get().strip() or "#444444",
            "base_pair_width": base_pair_width_var.get().strip() or "3.0",
            "base_pair_opacity": base_pair_opacity_var.get().strip() or "0.75",
            "style_specs": style_specs,
        }
        return namespace_from_gui_values(values)

    def run_from_gui() -> None:
        try:
            args = namespace_from_current_gui()
            summary = run_processing(args)
        except Exception as exc:
            messagebox.showerror("Plane It error", str(exc))
            output_text.delete("1.0", "end")
            output_text.insert("1.0", "ERROR: {0}\n".format(exc))
            return
        output_text.delete("1.0", "end")
        output_text.insert("1.0", summary + "\n")
        messagebox.showinfo("Plane It", "Finished. See the run log for details.")

    def handle_projection_mode_change(*_args) -> None:
        desired = projection_mode_var.get() == "pca"
        if bool(write_pca_pdb_var.get()) != desired:
            write_pca_pdb_var.set(desired)
        else:
            update_gui_state()

    for var in [
        flip_about_y_var,
        input_format_var,
        color_by_var,
        palette_var,
        connection_mode_var,
        closed_chains_var,
        close_all_chains_var,
        csv_output_var,
        write_pca_pdb_var,
        depth_order_circles_var,
        depth_order_lines_var,
        depth_order_base_pairs_var,
        depth_front_var,
        line_underlay_var,
        draw_xy_plane_var,
        draw_scale_bar_var,
        draw_base_pairs_var,
    ]:
        trace_state(var)

    projection_mode_var.trace_add("write", handle_projection_mode_change)

    pdb_var.trace_add("write", lambda *_args: update_default_paths(force=True, update_closed_chains=True))
    pdb_var.trace_add("write", input_file_changed)
    input_format_var.trace_add("write", lambda *_args: update_default_paths(force=True, update_closed_chains=True))
    # The fields that decide which chains the run colors refresh the swatch
    # strip, debounced, since the input is read again when they change.
    for var in [pdb_var, input_format_var, records_var, model_var, chain_var, resname_var, altloc_var, select_by_var]:
        var.trace_add("write", lambda *_args: schedule_color_strip())

    rebuild_type_rows()
    update_gui_state()

    footer = ttk.Frame(root, padding=(12, 6, 12, 10))
    footer.grid(row=2, column=0, columnspan=2, sticky="ew")
    ttk.Button(footer, text="Run", command=run_from_gui).pack(side="left")
    ttk.Button(footer, text="Quit", command=root.destroy).pack(side="left", padx=(8, 0))
    ttk.Label(footer, text="Tip: use command-line mode for batch processing, or run with no arguments for this GUI.").pack(side="left", padx=(18, 0))

    # Open as wide as the page asks, 1240 px or not, and never wider than the
    # screen: at 1240 px the drawing frame asked for 18 px more, and on a
    # narrower screen the page, still as wide as it asks, scrolls sideways.
    try:
        root.update_idletasks()
        needed = content.winfo_reqwidth() + main_scroll.winfo_reqwidth()
        root.geometry("{0}x900".format(min(needed, root.winfo_screenwidth() - 40)))
    except Exception:
        pass
    # The swatches beside the Palette, now that the drawing frame is laid out
    # and its width, which they must stay within, is known.
    run_color_strip()

    root.mainloop()
    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0 or "--gui" in argv:
        return run_gui()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.pdb_file:
        parser.error("input file is required in command-line mode; use --gui or no arguments for GUI mode")
    try:
        summary = run_processing(args)
        print(summary)
        return 0
    except ColorOptionError as exc:
        parser.error(str(exc))  # a usage error, exit 2, like any other bad option value
    except Exception as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
