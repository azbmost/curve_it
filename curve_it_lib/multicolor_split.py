#!/usr/bin/env python3
"""
multicolor_split.py -- split a PDB/mmCIF model's chains into separate molmap
solids that tile the whole molmap surface EXACTLY, for multicolour printing
(one filament per part on an AMS / multi-material printer).

    python3 multicolor_split.py                      # no arguments -> GUI
    python3 multicolor_split.py --gui [options]      # GUI, pre-filled from options
    python3 multicolor_split.py MODEL.pdb            # every chain its own colour
    python3 multicolor_split.py MODEL.pdb --parts A C
    python3 multicolor_split.py MODEL.pdb --parts A,B C          # A+B = one colour
    python3 multicolor_split.py MODEL.pdb --resolution 3.7       # K rescaled + checked
    python3 multicolor_split.py MODEL.pdb --lay-flat             # fewer layers, less purge
    python3 multicolor_split.py MODEL.pdb --palette DiLiuLab     # lab colours in the previews
    python3 multicolor_split.py MODEL.pdb --part 'bbA=/A & backbone' \
        --part 'bbB=/B & backbone' --part rest     # atom selections as parts (ChimeraX's)
    python3 multicolor_split.py MODEL.pdb --selection-syntax chimera \
        --part "bbA=:.A@P,OP1,OP2,O5',C5',C4',O4',C3',O3',C2',C1'" --part rest   # Chimera's
    python3 multicolor_split.py MODEL.pdb --blobs --part "phosA=/A@P,OP1,OP2,O5',O3'" \
        --part "phosB=/B@P,OP1,OP2,O5',O3'" --part rest   # blobs, each its own molmap

This is Curve It's "Multicolor Split..." tool; it also runs on its own.  In the
GUI every field, checkbox and group of buttons has a light-blue ? that opens an
explanation of the setting, with an example for every entry field.

Output (default): ./<structure stem>_molmap<resolution>_split/ holding
    <tag>_p1_chainA.stl, <tag>_p2_chainC.stl, ...   one per part (tag = the folder name;
                                                    <tag>_p1_bbA.stl for a part --part names)
    <tag>_whole.stl                                 reference only
    <tag>_report.txt / .json                        scan + validation
    <tag>_preview.png / .glb                        local previews
    <tag>_work/                                     ChimeraX jobs + logs (<tag>_work_failed/
                                                    for a run that did not finish)
Outputs are written to a staging folder; only once the report is complete
are the previous run's files for the same tag moved into _previous_<timestamp>/
and the new ones moved in, so the folder holds one complete set per tag
(a run that ends with an error or is stopped leaves the previous set in place).

Exit codes: 0 all checks pass (warnings allowed) / 3 exported but a check
FAILED / 1 error / 2 usage error / 130 interrupted or stopped.

Needs UCSF ChimeraX (molmap + marching cubes, run headless) and a python3 with
numpy, scipy, trimesh and manifold3d (tested with /opt/anaconda3/bin/python3);
matplotlib for the preview PNG, tkinter for the GUI, and UCSF Chimera 1.x only
for selections written in Chimera's convention.  STL units are Angstroms;
slicers read them as mm, so 1:1 is 1 A -> 1 mm (--scale changes that).

METHOD  (the two-colour BR strand-split procedure, generalised to N parts)
--------------------------------------------------------------------------
0. Parts.  A part is one or more whole chains (--parts), or an atom selection
   (--part) in ChimeraX's convention or UCSF Chimera's (--selection-syntax),
   evaluated once, before any map is made, by the program whose convention it
   is: Chimera names the atoms it selects by chain, residue and atom name and
   ChimeraX takes exactly those, so both conventions give the same parts from
   the same atoms.  An atom two selections share goes to the earlier part
   (--overlap error refuses the run instead), rest takes every atom no other
   part takes, and every later ChimeraX job maps exactly those atoms, by index,
   checked against a fingerprint of their names.  All below is the same for
   chains and selections.

   Blob parts (--blobs) replace steps 2-3's ownership fields for parts that
   are separate blobs (phosphate groups): every part but the last is cut by
   its OWN molmap M_i -- its atoms' density alone -- at the contour level,
   clipped to the whole (an overlap goes to the earlier blob), and the last
   part is the whole minus the blobs.  A blob's own surface lies a hair inside
   the whole's where it faces out (the other atoms add a trace of density), so
   cut exactly the last part would film over most of each blob (on the
   switchback666 duplex at 3.7 A, 96 % of the phosphates' outward surface,
   80 % by more than 0.0001 A, with one blob sealed in; about half of the film
   is under 10 um at 1 mm per A, which slicers should drop on walls, not
   tested, but the files, previews and near-flat tops show it); so where the
   other atoms' density M_o is below blob_skin * |grad T| and M_i > M_o, the
   blob field is T itself and the blob reaches the surface (--blob-skin,
   default 0.2 A; 0 = the exact molmap).  M_o / |grad T| estimates the skin,
   low for thicker ones: at 0.2 A the blob takes skins up to about 0.3 A.
   No K; the level scan stops at the nearest clean level (the one a full scan
   would pick); dust anchors for the last part (and for exact blobs) come from
   where each part's density dominates.

1. Maps.  One molmap of all selected chains, T, at gridSpacing 0.5 A, and one
   molmap per part on the SAME grid (onGrid).  molmap Gaussians are additive,
   so T == sum of the part maps (checked every run, ~3e-7).  At molmap's
   default grid (resolution/3) the colour interface cannot be resolved finer
   than ~1 mm, whatever else is done.  Every surface is contoured at voxel
   step 1 (every grid point) -- forced and verified.

2. Ownership field.  For part k against a set of rival parts:
        F = T - K * max( max_rivals M_j - M_k , 0 )      inside the whole (T >= L)
        F = T                                             outside it
   F is T wherever part k is the densest, and drops steeply below the contour
   level across the medial surface to a rival, so marching cubes interpolates
   a smooth sub-voxel cut face.  Inside the whole this is bit-identical to
   ChimeraX's `volume subtract` / `volume threshold minimum 0 set 0` /
   `volume add scaleFactors 1,-K` chain.  Restricting it to the inside is a
   deliberate departure from that chain: applied everywhere, the penalty also
   pulls the field-built skin inside the whole's skin wherever the part owns
   the surface, and the complement inherits a wrong-colour film there (2-20
   A^3 per model at 4 A; detached flakes at 3 A and below).  Masked, a part's
   skin coincides with the whole's except within about half a voxel of a
   colour edge, where a feather-edge film under 0.1 A thick can remain.
   K = 16 is a slope PER VOXEL, valid at gridSpacing 0.5 and resolution 4:
   in the sweep it was validated on, below ~12 field-built parts interfered
   and at >= 20 marching cubes tore; another model can differ (the
   switchback666 example gives clean fields for every K from 8 to 24).
   For another grid or resolution K is rescaled to its per-voxel equivalent
   K0 = 16 * (res/4) * (0.5/grid) and confirmed by a scan (the clean candidate
   nearest K0 wins).  K only moves the colour boundary by hundredths of a mm.

3. Exact tiling by peeling.  Two field-built solids leave a ~0.25 A void
   between them, which prints as a visible groove.  So only ONE side of every
   colour boundary comes from a field; the other side is its exact complement
   (manifold3d booleans):
        R0 = whole
        P_k = R_(k-1) & F_k ,   R_k = R_(k-1) - F_k      k = 1 .. N-1
        P_N = R_(N-1)
   where F_k is part k's field against the LATER parts only.  Ignoring the
   earlier parts hands every void strip to the neighbour whose territory it
   lies in.  The booleans can still leave tiny fragments along the
   boundaries, not only where three parts meet (the two-part switchback666
   example at 4 A leaves 3, 0.0035 A^3 in all; anything under 4 voxels counts
   as one); each is handed to the neighbouring part it shares the most
   surface with, so the tiling stays exact and no pits are left (a fragment
   touching nothing is dropped and reported).  Junction lines can shift by
   up to about one voxel (~0.4 A).  For two parts this is
   exactly "strand X from the field, strand Y = duplex - X".  The earlier
   part of each pair carries the field-built face, so the colour boundary
   sits ~0.1-0.2 A into it (reported).  Sealed cavities are kept as cavities.

4. Contour level: pinned by scan, never inherited.  Candidates: guess +/-
   0.008 in 0.002 steps (guess = molmap's own auto level -- the contour that
   encloses 95 % of the map's density -- rounded to 3 decimals, unless
   given).  A level is accepted only when the whole AND every part's own
   field-built solid are closed 2-manifolds (0 open, 0 non-manifold edges)
   with at most one piece per cluster of the part's own atoms (see "pieces"
   below), and the peeled parts come out likewise, summing to the whole to
   +/-0.001 % with no overlap and no void.  The accepted level nearest the guess wins; if
   none, the scan is widened once, then the run stops and prints the table.
     * surface dust (default 20 A) runs on every contour, after the surface
       is actually built.  On the whole it removes specks and fills enclosed
       cavities smaller than the dust size, but never removes the whole's
       largest piece or a piece that carries a part's main piece (a small
       ligand survives when its chain is a part of its own, not when it is
       joined to another chain in one part).  On the part fields it only
       fills small cavities -- it never removes a piece, so a part's detached
       bit (the tail after a chain break, a small part anywhere in the order)
       keeps its colour wherever the whole keeps it.  (At a two-part boundary
       a filled cavity goes to the later part.)
     * zero-area triangles (two corners welded onto one point where the
       contour passes exactly through a grid point) are dropped, as slicers
       do, and counted -- they are not treated as defects.
     * pieces: heavy atoms within 4 A of each other form one molecule (a
       base-paired duplex is one).  The whole must have one piece per
       molecule, so separate molecules (three rings) are never accepted
       welded through a hair-line neck, and a connected model is never
       accepted pinched apart; if no clean level matches, the whole's count
       at the level guess is used.  Each part (and its own field solid) may
       have at most one piece per cluster of its own atoms -- normally 1, 2
       for a chain with a break.

5. Validation (report + JSON): parts sum vs whole (+0.0000 % expected), pieces,
   cavities, manifoldness, pinch edges after vertex welding (where the
   complement's reclaimed shell closes to zero thickness, the boolean result
   keeps one vertex copy per sheet at a single point; each copy is moved
   1/1000 of the median edge (or 32 float32 steps if larger; about 0.5 um at
   1 mm per A) into its own solid before writing, leaving the copies up to
   twice that apart, so the files have none and slicers that weld equal
   coordinates find no non-manifold edge -- --keep-pinch-edges writes them as
   v1.4.1 did), real pairwise overlap, void, the exported files
   re-read from disk, thin-neck and separate-piece warnings, and the
   colour-boundary offset.

Slicing (Bambu Studio): import all part STLs AT ONCE -> one object with one
sub-model per part, already registered; give each part a filament; check
"First layer filament sequence".  Lay long models flat (--lay-flat): purge
scales with layer count.  Enable flush into infill / support.  When the whole
is one piece the colours fuse into one solid -- nothing comes apart.
"""
import argparse
import collections
import glob
import gzip
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

__version__ = "1.6.0"
TOOL_NAME = "Multicolor Split"

try:
    import numpy as np
except ImportError:                                   # pragma: no cover
    if "--version" in sys.argv[1:]:
        print(__version__)
        sys.exit(0)
    sys.stderr.write("multicolor_split.py needs a python3 with numpy, scipy, trimesh and "
                     "manifold3d (e.g. /opt/anaconda3/bin/python3); this one has no numpy.\n")
    sys.exit(1)

# The DiLiuLab figure colours for --palette; without lab_colors.py only the
# tool's own colours are offered.
try:
    from . import lab_colors
except ImportError:
    try:
        import lab_colors  # type: ignore[no-redef]
    except ImportError:
        lab_colors = None  # type: ignore[assignment]

# The documented, validated operating point (the BR wrapping B-DNA sweep)
DOC_RESOLUTION = 4.0
DOC_GRID = 0.5
DOC_K = 16.0
DEFAULT_DUST = 20.0
DEFAULT_SPAN = 0.008
DEFAULT_STEP = 0.002
K_SCAN_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)
DEFAULT_BLOB_SKIN = 0.2   # A: with --blobs, a skin of the last part estimated thinner than this goes to
                          # the blob where the blob's density exceeds the other atoms' together
MAX_BLOB_SKIN = 1.0

SLIVER_VOXELS = 4.0  # a boolean body under this many voxels (0.5 A^3 at grid 0.5) is a fragment
TOL_REL = 1e-5      # sum / overlap / void tolerance as a fraction of the whole (0.001 %)
PINCH_WARN = 20     # pinch edges per part worth a look: max(20, PINCH_WARN_REL * triangles)
PINCH_WARN_REL = 2e-4
LAYER_H = 0.2       # mm, for the layer-count estimate
AMS_SLOTS = 4
CONTACT_CUTOFF = 4.0  # A: atoms closer than this belong to one molecule (piece of the whole)

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_FAIL, EXIT_STOPPED = 0, 1, 2, 3, 130

SOLVENT_NAMES = {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL", "T3P", "SPC"}

PALETTE = [("blue", (0.30, 0.47, 0.66)), ("orange", (0.96, 0.52, 0.09)),
           ("green", (0.33, 0.64, 0.29)), ("red", (0.89, 0.34, 0.34)),
           ("purple", (0.70, 0.47, 0.64)), ("teal", (0.45, 0.72, 0.70)),
           ("yellow", (0.93, 0.79, 0.23)), ("brown", (0.62, 0.46, 0.36))]
SWATCH_MAX = 8      # parts the GUI's Palette row shows a swatch for


def check_palette(spec):
    """A --palette value or the GUI's Palette choice -> "default" or
    "DiLiuLab T80" style; ValueError for anything else, and for DiLiuLab
    when lab_colors.py is missing."""
    if lab_colors is not None:
        return lab_colors.check_palette(spec)
    if spec is None or str(spec).strip().lower() in ("", "default", "none", "auto"):
        return "default"
    raise ValueError(f"palette {spec!r} needs lab_colors.py from curve_it_lib, which was not "
                     f"found; only default is available")


def palette_arg(value):
    """argparse type= for --palette: a bad value is a usage error."""
    try:
        return check_palette(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def palette_option(label):
    """The command-line spelling of a palette: --palette=DiLiuLab-T80."""
    return "--palette=" + check_palette(label).replace(" ", "-")


def palette_choices():
    """What the GUI's Palette menu offers: default, then DiLiuLab at every tint."""
    return lab_colors.palette_choices("default") if lab_colors is not None else ["default"]


def part_colors(spec="default"):
    """[(name, rgb 0-1), ...] for the parts, P1 first, repeating: PALETTE for
    "default", else the nine DiLiuLab colours at the tint `spec` names, in the
    lab's order (the neutral is not used: every part is a colour)."""
    label = check_palette(spec)
    if label == "default":
        return PALETTE
    return [(name, lab_colors.hex_to_rgb01(code))
            for name, code in lab_colors.lab_named_colors(lab_colors.lab_tint_of(label))]


SCRIPT_PATH = Path(__file__).resolve()


def resource_path(relative_path):
    """Return a resource path that also works from a PyInstaller bundle."""
    source_dir = os.path.dirname(os.path.abspath(__file__))
    source_root = (os.path.dirname(source_dir)
                   if os.path.basename(source_dir) == "curve_it_lib" else source_dir)
    base_dir = getattr(sys, "_MEIPASS", source_root)
    return os.path.join(base_dir, relative_path)


def set_optional_window_icon(root, tk_module, icon_filenames, image_attr):
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


def self_command(unbuffered=False):
    """The argv prefix that runs this tool again: the GUI's Run split and
    Show command, and the report's REPRODUCE line.

    In a frozen Curve It build sys.executable is the Curve It app itself,
    which cannot run a .py file; Curve It forwards everything after
    --multicolor-split to this module's main(), so the app is called with
    that flag instead.  Otherwise it is this python running this file
    (with -u for unbuffered output when asked).
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--multicolor-split"]
    return [sys.executable] + (["-u"] if unbuffered else []) + [str(SCRIPT_PATH)]


def log(msg=""):
    print(msg, flush=True)


class SplitError(Exception):
    """A user-facing failure: printed without a traceback."""


# ======================================================== parameter checking
def parse_k_list(text):
    ks = []
    for tok in re.split(r"[,\s]+", str(text).strip()):
        if not tok:
            continue
        k = float(tok)                      # ValueError for junk
        if not math.isfinite(k) or k <= 0:
            raise ValueError(f"K candidate {tok!r} must be a positive number")
        ks.append(k)
    if not ks:
        raise ValueError("empty K candidate list")
    return sorted(set(ks))


def check_numbers(v):
    """Shared by the CLI and the GUI.  v: name -> float or None.  Returns a
    list of plain-language problems (empty = fine)."""
    errs = []
    pos = ("resolution", "grid", "span", "step", "scale", "k", "level", "level_guess")
    for name, x in v.items():
        if x is None:
            continue
        if not math.isfinite(x):
            errs.append(f"{name} must be a finite number (got {x})")
        elif name in pos and x <= 0:
            errs.append(f"{name} must be positive (got {x:g})")
        elif name == "dust" and x < 0:
            errs.append(f"dust must be >= 0 (got {x:g})")
        elif name == "blob_skin" and not 0 <= x <= MAX_BLOB_SKIN:
            errs.append(f"blob skin must be from 0 to {MAX_BLOB_SKIN:g} A (got {x:g})")
    s, st = v.get("span"), v.get("step")
    if s and st and math.isfinite(s) and math.isfinite(st) and 0 < s < st:
        errs.append(f"level scan step {st:g} is larger than the span {s:g}: only the guess "
                    f"would be tried (the scan step is in contour units, not the voxel step)")
    return errs


# ============================================================ structure input
def _read_text(path):
    path = Path(path)
    if path.is_dir():
        raise SplitError(f"{path} is a folder, not a structure file")
    try:
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rt", errors="replace") as fh:
                return fh.read()
        return path.read_text(errors="replace")
    except OSError as e:
        raise SplitError(f"cannot read {path}: {e}")


def read_structure(path, exclude_solvent=True):
    """Chains of the FIRST model, in order of appearance.

    Returns dict(fmt, n_models, chains=[dict(id, atoms, residues, first, last,
    resnames, xyz, is_h)]).  Used for chain detection (CLI default parts, GUI
    table) and for counting separate molecules; the authoritative atom
    selection for the maps is done by ChimeraX itself.
    """
    path = Path(path)
    text = _read_text(path)
    name = path.name.lower()[:-3] if path.name.lower().endswith(".gz") else path.name.lower()
    head = text.lstrip()[:200].lower()
    if name.endswith((".cif", ".mmcif")) or head.startswith("data_"):
        chains, nmod = _read_cif(text, exclude_solvent)
        fmt = "mmcif"
    else:
        chains, nmod = _read_pdb(text, exclude_solvent)
        fmt = "pdb"
    out = []
    for cid, c in chains.items():
        res = c["res"]
        out.append(dict(id=cid, atoms=c["atoms"], residues=len(set(res)),
                        first=res[0].strip() if res else "",
                        last=res[-1].strip() if res else "",
                        resnames=dict(c["resnames"]),
                        xyz=np.array(c["xyz"], dtype=np.float64).reshape(-1, 3),
                        is_h=np.array(c["is_h"], dtype=bool)))
    return dict(fmt=fmt, n_models=nmod, chains=out)


def _add_atom(chains, cid, reskey, resn, xyz=None, is_h=False):
    c = chains.setdefault(cid, dict(atoms=0, res=[], resnames=collections.Counter(),
                                    xyz=[], is_h=[]))
    c["atoms"] += 1
    if not c["res"] or c["res"][-1] != reskey:
        c["res"].append(reskey)
        c["resnames"][resn] += 1
    if xyz is not None:
        c["xyz"].append(xyz)
        c["is_h"].append(bool(is_h))


def _xyz(a, b, c):
    try:
        return (float(a), float(b), float(c))
    except ValueError:
        return None


def _hydrogen(elem, name):
    e = (elem or "").strip().upper()
    if e:
        return e in ("H", "D")
    n = (name or "").strip().upper()
    return bool(re.match(r"^\d?[HD]", n))


def _read_pdb(text, exclude_solvent):
    chains = collections.OrderedDict()
    nmodels = 0
    in_first = True
    for line in text.splitlines():
        rec = line[:6]
        if rec.startswith("MODEL"):
            nmodels += 1
            in_first = nmodels == 1
            continue
        if rec not in ("ATOM  ", "HETATM") or not in_first:
            continue
        resn = line[17:20].strip()
        if exclude_solvent and resn in SOLVENT_NAMES:
            continue
        cid = line[21] if len(line) > 21 else " "
        _add_atom(chains, cid, line[22:27], resn, _xyz(line[30:38], line[38:46], line[46:54]),
                  _hydrogen(line[76:78] if len(line) > 76 else "", line[12:16]))
    return chains, max(nmodels, 1)


_CIF_TOKEN = re.compile(r"'[^']*'(?=\s|$)|\"[^\"]*\"(?=\s|$)|\S+")


def _read_cif(text, exclude_solvent):
    lines = text.splitlines()
    cols, start = [], None
    for i, ln in enumerate(lines):
        if ln.strip() == "loop_" and i + 1 < len(lines) \
                and lines[i + 1].strip().startswith("_atom_site."):
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                cols.append(lines[j].strip().split(".", 1)[1].split()[0])
                j += 1
            start = j
            break
    if start is None:
        raise SplitError("no _atom_site loop found in the mmCIF file")
    idx = {c: n for n, c in enumerate(cols)}
    ch = idx.get("auth_asym_id", idx.get("label_asym_id"))
    if ch is None:
        raise SplitError("mmCIF _atom_site has no auth_asym_id / label_asym_id")
    resn_i = idx.get("auth_comp_id", idx.get("label_comp_id"))
    seq_i = idx.get("auth_seq_id", idx.get("label_seq_id"))
    ins_i = idx.get("pdbx_PDB_ins_code")
    mod_i = idx.get("pdbx_PDB_model_num")
    grp_i = idx.get("group_PDB")
    xi, yi, zi = idx.get("Cartn_x"), idx.get("Cartn_y"), idx.get("Cartn_z")
    el_i = idx.get("type_symbol")
    nm_i = idx.get("auth_atom_id", idx.get("label_atom_id"))
    toks = []
    for ln in lines[start:]:
        s = ln.strip()
        if s.startswith(("loop_", "_", "data_")) or s.startswith("#"):
            if toks or s.startswith(("loop_", "_", "data_")):
                break
            continue
        toks.extend(t.strip("'\"") for t in _CIF_TOKEN.findall(s))
    n = len(cols)
    chains = collections.OrderedDict()
    first_model, models = None, set()
    for r in range(len(toks) // n):
        row = toks[r * n:(r + 1) * n]
        if grp_i is not None and row[grp_i] not in ("ATOM", "HETATM"):
            continue
        if mod_i is not None:
            models.add(row[mod_i])
            if first_model is None:
                first_model = row[mod_i]
            if row[mod_i] != first_model:
                continue
        resn = row[resn_i] if resn_i is not None else "?"
        if exclude_solvent and resn in SOLVENT_NAMES:
            continue
        key = (row[seq_i] if seq_i is not None else "?") + \
              (row[ins_i] if ins_i is not None and row[ins_i] not in ("?", ".") else "")
        xyz = _xyz(row[xi], row[yi], row[zi]) if None not in (xi, yi, zi) else None
        _add_atom(chains, row[ch], key, resn, xyz,
                  _hydrogen(row[el_i] if el_i is not None else "",
                            row[nm_i] if nm_i is not None else ""))
    return chains, max(len(models), 1)


def analyse_molecules(info, parts, resolution, dust):
    """Molecules and clusters from the atoms.  Heavy atoms within CONTACT_CUTOFF
    of each other belong to one molecule (hydrogens never join molecules).

    Returns dict(n, n_all, closest, removed, clusters):
      n         molecules the whole should show as separate pieces; a molecule
                whose molmap blob is smaller than the dust size is removed by
                dust -- unless it is a part's main molecule;
      removed   [(chains, heavy atoms)] of the molecules dust will remove;
      clusters  per part: its own heavy-atom clusters inside kept molecules, i.e.
                the most pieces that part's solid may have (a chain break gives 2);
      closest   closest heavy-atom approach between two separate molecules.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    def components(X):
        if len(X) == 1:
            return 1, np.zeros(1, dtype=np.int64)
        pairs = cKDTree(X).query_pairs(CONTACT_CUTOFF, output_type="ndarray")
        A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(X), len(X)))
        return connected_components(A, directed=False)

    X, owner, chain = [], [], []
    for pi, p in enumerate(parts):
        for c in info["chains"]:
            if c["id"] in p and len(c["xyz"]):
                heavy = ~c["is_h"]
                xyz = c["xyz"][heavy] if heavy.any() else c["xyz"]
                X.append(xyz)
                owner.append(np.full(len(xyz), pi))
                chain += [c["id"]] * len(xyz)
    if not X:
        return None
    X, owner, chain = np.concatenate(X), np.concatenate(owner), np.array(chain)
    n_all, lab = components(X)
    main = {int(np.bincount(lab[owner == pi]).argmax()) for pi in range(len(parts))
            if (owner == pi).any()}
    kept, removed = set(), []
    for k in range(n_all):
        sel = lab == k
        if float(np.ptp(X[sel], axis=0).max()) + resolution >= dust or k in main:
            kept.add(k)
        else:
            removed.append((sorted(set(chain[sel].tolist())), int(sel.sum())))
    clusters = []
    for pi in range(len(parts)):
        idx = np.where(owner == pi)[0]
        if not len(idx):
            clusters.append(1)
            continue
        nc, cl = components(X[idx])
        clusters.append(max(1, sum(1 for k in range(nc) if int(lab[idx[cl == k][0]]) in kept)))
    closest = None
    if 1 < n_all <= 64:
        best = np.inf
        for k in range(n_all):
            d, _ = cKDTree(X[lab == k]).query(X[lab != k], k=1)
            best = min(best, float(d.min()))
        closest = best
    return dict(n=len(kept), n_all=int(n_all), closest=closest, removed=removed, clusters=clusters)


def parse_parts(tokens, available):
    """['A', 'C'] / ['A,B', 'C'] / 'A,B C' / 'A, B C' / 'A+B | C'  ->  [['A','B'], ['C']].

    Whitespace or '|' separates parts; ',' or '+' joins chains into one part
    (spaces around a joiner are allowed).
    """
    if isinstance(tokens, str):
        tokens = [tokens]
    text = " ".join(tokens).replace("|", " ")
    text = re.sub(r"\s*([,+])\s*", r"\1", text)
    parts = []
    for tok in text.split():
        if tok[0] in ",+" or tok[-1] in ",+" or re.search(r"[,+]{2}", tok):
            raise SplitError(f"part '{tok}' has an empty chain name -- join chains like A,B")
        parts.append(re.split(r"[,+]", tok))
    seen = {}
    for p in parts:
        for c in p:
            if c not in available:
                hint = ""
                if re.search(r"\.(pdb|ent|cif|mmcif)(\.gz)?$", c, re.I):
                    hint = " (put the structure file BEFORE --parts)"
                raise SplitError(f"chain '{c}' is not in the structure "
                                 f"(has: {' '.join(available)}){hint}")
            if c in seen:
                raise SplitError(f"chain '{c}' is listed in more than one part")
            seen[c] = True
    if len(parts) < 2:
        raise SplitError("need at least two parts (colours) to split; got "
                         f"{parts or 'none'}")
    return parts


SELECTION_SYNTAXES = ("chimerax", "chimera")
OVERLAP_RULES = ("first", "error")
REST_WORD = "rest"
_PART_NAME = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_+-]*)\s*=\s*(.+?)\s*$", re.S)
_NAME_LIKE = re.compile(r"^([\w\s+-]*?)\s*=")        # text before '=' with no selection symbols


def selection_label(spec):
    """A file-name label for an unnamed selection, from its words: /A &
    backbone -> A_backbone, /A & ~backbone -> A_not_backbone."""
    words = [{"~": "not", "|": "or"}.get(w, w) for w in re.findall(r"~|\||[A-Za-z0-9]+", spec)]
    label = ""
    for w in words:
        if len(label) + len(w) + bool(label) > 24:
            break
        label += ("_" if label else "") + w
    return label or (words[0][:24] if words else "sel")


def selection_line_label(text):
    """The label a --part value gets: its name, rest, or selection_label()."""
    m = _PART_NAME.match(text)
    if m:
        return safe_tag(m.group(1))
    return REST_WORD if text.strip().lower() == REST_WORD else selection_label(text)


def parse_selection_parts(values, syntax="chimerax"):
    """--part values -> [dict(label, spec, kind, syntax, name)], in peeling order.

    Each value is one part: an atom specification in ChimeraX's or Chimera's
    convention, optionally named as NAME=SPEC (the name goes into the file
    name), or the word rest, for every atom no other part takes.  An
    unnamed selection is labelled from its words (selection_label)."""
    syntax = str(syntax or "chimerax").strip().lower()
    if syntax not in SELECTION_SYNTAXES:
        raise SplitError(f"unknown selection syntax {syntax!r}: use chimerax or chimera")
    parts = []
    for k, raw in enumerate(values or [], 1):
        text = str(raw).strip()
        name = None
        m = _PART_NAME.match(text)
        if m:
            name, text = m.group(1), m.group(2).strip()
        else:
            bad = _NAME_LIKE.match(text)
            if bad and bad.group(1).strip():
                word = bad.group(1).strip()
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]*", word):
                    raise SplitError(f"part {k} ({raw!r}) has no selection after its name")
                raise SplitError(f"part {k} ({raw!r}): the name {word!r} before '=' must be one "
                                 f"word that starts with a letter, using letters, digits and _ + - "
                                 f"(e.g. bbA=/A & backbone)")
        if not text:
            raise SplitError(f"part {k} ({raw!r}) has no selection")
        if ";" in text or "\n" in text or "\r" in text:
            raise SplitError(f"part {k} ({raw!r}): a selection may not contain ';' or a line break "
                             f"-- give each part as its own --part")
        kind = "rest" if text.lower() == REST_WORD else "spec"
        label = selection_line_label(str(raw).strip())
        parts.append(dict(label=label, spec=REST_WORD if kind == "rest" else text, kind=kind,
                          syntax=syntax, name=name))
    if sum(p["kind"] == "rest" for p in parts) > 1:
        raise SplitError("rest may be given only once: it already takes every atom no other part takes")
    if len(parts) < 2:
        raise SplitError("need at least two parts (colours) to split; got "
                         f"{len(parts)} --part value(s)")
    return parts


_CHIMERA_CHAIN = re.compile(r":[^\s@&|~()]*\.[A-Za-z0-9]")                  # :.A  :12.A
_CHIMERAX_CHAIN = re.compile(r"(^|[\s&|~(])/[A-Za-z0-9]|#[\d.]+/[A-Za-z0-9]")   # /A  #1/A
_CHIMERAX_ONLY_WORDS = re.compile(r"\b(backbone|mainchain|sidechain|sideonly|side|"
                                  r"nucleic(?!\s+acid))\b", re.I)


def convention_hint(spec, syntax):
    """For a selection that fails: a pointer to the other convention when it
    looks written in that one, else ""."""
    if syntax == "chimerax" and _CHIMERA_CHAIN.search(spec):
        return ("; :.A and :12.A are Chimera's convention (ChimeraX writes /A and /A:12) -- or "
                "choose the Chimera convention")
    if syntax == "chimera":
        if _CHIMERAX_CHAIN.search(spec):
            return ("; /A is ChimeraX's convention (Chimera writes :.A) -- or choose the ChimeraX "
                    "convention")
        m = _CHIMERAX_ONLY_WORDS.search(spec)
        if m:
            return (f"; {m.group(1)} is a ChimeraX keyword Chimera does not have -- list the atom "
                    f"names instead, e.g. :.A{CHIMERA_DNA_BACKBONE} for a DNA backbone and "
                    f":.A & ~@P,OP1,... for its bases, or choose the ChimeraX convention")
    return ""


def selection_part_arg(p):
    """One part as the --part value that rebuilds it."""
    return (f"{p['name']}={p['spec']}" if p.get("name") else p["spec"])


# The atom names ChimeraX's backbone keyword covers (ChimeraX 1.x, hydrogens
# and older or other programs' names included), for writing it in Chimera's
# convention, which has no such keyword and whose * does not match a primed
# name.  H1, H2 and H3 are a protein's N-terminal hydrogens but a nucleotide's
# base hydrogens, so they are listed only for a chain with no nucleotide.
NA_BACKBONE_NAMES = ("P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P", "HP", "O5'", "C5'", "C4'",
                     "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "H5'", "H5''", "H4'", "H3'",
                     "H2'", "H2''", "HO2'", "H1'", "HO3'", "HO5'", "H5T", "H3T")
AA_BACKBONE_NAMES = ("N", "CA", "C", "O", "OXT", "OT1", "OT2", "H", "HN", "HA", "HA1", "HA2",
                     "HA3", "HT1", "HT2", "HT3")
AA_TERMINAL_H = ("H1", "H2", "H3")
CHIMERA_DNA_BACKBONE = "@P,OP1,OP2,O5',C5',C4',O4',C3',O3',C2',C1'"   # its heavy atoms only
_NUCLEOTIDE = re.compile(r"^(D?[ACGTUI]|R[ACGU]|ADE|CYT|GUA|THY|URA)[35N]?$")
_AMINO_ACIDS = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
                "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "SEC",
                "PYL", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH", "LYN"}


def backbone_parts(chains, syntax="chimerax"):
    """--part values for "each chain's backbone its own colour, then the
    rest": one bb<chain> line per chain with an ID and a nucleotide or
    amino-acid residue, in file order, then rest.  chains is
    read_structure()'s list; both conventions select the same atoms."""
    lines = []
    for c in chains:
        cid = str(c["id"]).strip()
        names = set(c.get("resnames") or ())
        na = any(_NUCLEOTIDE.match(n) for n in names)
        aa = bool(names & _AMINO_ACIDS)
        if not cid or not (na or aa):
            continue
        if syntax == "chimera":
            atoms = (NA_BACKBONE_NAMES if na else ()) + \
                    (AA_BACKBONE_NAMES + (() if na else AA_TERMINAL_H) if aa else ())
            spec = f":.{cid}@" + ",".join(atoms)
        else:
            spec = f"/{cid} & backbone"
        lines.append(f"bb{re.sub(r'[^A-Za-z0-9_+-]', '_', cid)} = {spec}")
    return lines + [REST_WORD] if lines else []


def part_label(chains):
    return ("chain" if len(chains) == 1 else "chains") + "+".join(chains)


def safe_tag(s):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", s).strip("_") or "model"


def structure_stem(structure):
    name = Path(structure).name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    return Path(name).stem


def default_tag(structure, resolution):
    """The naming convention: <structure file stem>_molmap<resolution>_split,
    used for the output folder AND as the prefix of every file in it."""
    return safe_tag(f"{structure_stem(structure)}_molmap{float(resolution):g}_split")


# ================================================================== ChimeraX
def _exe_ok(p):
    return p and Path(p).is_file() and os.access(p, os.X_OK)


def _normalise_exe(p):
    p = str(Path(p).expanduser())
    if p.rstrip("/").endswith(".app"):                 # a macOS app bundle
        p = str(Path(p) / "Contents" / "MacOS" / "ChimeraX")
    return p


def find_chimerax(user=None):
    """An explicitly given path (or $CHIMERAX) must work; only when neither is
    given is ChimeraX searched for."""
    for given, what in ((user, "--chimerax"), (os.environ.get("CHIMERAX"), "$CHIMERAX")):
        if given:
            p = _normalise_exe(given)
            if not _exe_ok(p):
                raise SplitError(f"ChimeraX not found at {given} ({what}); give the executable, "
                                 f"e.g. /Applications/ChimeraX.app/Contents/MacOS/ChimeraX")
            return p
    cands = ["/Applications/ChimeraX.app/Contents/MacOS/ChimeraX"]
    cands += sorted(glob.glob("/Applications/ChimeraX*.app/Contents/MacOS/ChimeraX"), reverse=True)
    cands += sorted(glob.glob(os.path.expanduser(
        "~/Applications/ChimeraX*.app/Contents/MacOS/ChimeraX")), reverse=True)
    cands += [shutil.which("chimerax"), shutil.which("ChimeraX"),
              "/usr/bin/chimerax", "/usr/local/bin/chimerax"]
    cands += sorted(glob.glob(r"C:\Program Files\ChimeraX*\bin\ChimeraX-console.exe"), reverse=True)
    for c in cands:
        if _exe_ok(c):
            return str(c)
    raise SplitError("ChimeraX not found -- install it or pass --chimerax PATH "
                     "(or set the CHIMERAX environment variable)")


def _chimera_exe(p):
    p = str(Path(p).expanduser())
    if p.rstrip("/").endswith(".app"):                 # a macOS app bundle
        p = str(Path(p) / "Contents" / "MacOS" / "chimera")
    return p


def _is_chimerax(p):
    # macOS paths ignore case, so ChimeraX.app/Contents/MacOS/chimera exists
    # too; ChimeraX must never stand in for Chimera
    return "chimerax" in str(p).lower()


def find_chimera(user=None):
    """UCSF Chimera, which reads selections written in Chimera's own
    atom-specification convention.  An explicitly given path (or $CHIMERA)
    must work; only when neither is given is Chimera searched for."""
    for given, what in ((user, "--chimera"), (os.environ.get("CHIMERA"), "$CHIMERA")):
        if given:
            p = _chimera_exe(given)
            if _is_chimerax(p):
                raise SplitError(f"{given} ({what}) is ChimeraX: selections in Chimera's convention "
                                 f"need UCSF Chimera 1.x, e.g. /Applications/Chimera.app/Contents/MacOS/"
                                 f"chimera -- or use --selection-syntax chimerax")
            if not _exe_ok(p):
                raise SplitError(f"UCSF Chimera not found at {given} ({what}); give the executable, "
                                 f"e.g. /Applications/Chimera.app/Contents/MacOS/chimera")
            return p
    cands = ["/Applications/Chimera.app/Contents/MacOS/chimera"]
    cands += sorted(glob.glob("/Applications/Chimera*.app/Contents/MacOS/chimera"), reverse=True)
    cands += sorted(glob.glob(os.path.expanduser(
        "~/Applications/Chimera*.app/Contents/MacOS/chimera")), reverse=True)
    cands += [shutil.which("chimera"), "/usr/local/bin/chimera"]
    cands += sorted(glob.glob("/opt/UCSF/Chimera*/bin/chimera"), reverse=True)
    cands += sorted(glob.glob(r"C:\Program Files\Chimera*\bin\chimera.exe"), reverse=True)
    for c in cands:
        if c and _exe_ok(c) and not _is_chimerax(c):
            return str(c)
    raise SplitError("UCSF Chimera not found -- selections in Chimera's convention are read by "
                     "Chimera itself: install it (https://www.cgl.ucsf.edu/chimera/download.html), "
                     "pass --chimera PATH or set the CHIMERA environment variable, or write the "
                     "selections in ChimeraX's convention (--selection-syntax chimerax)")


# The ChimeraX side: maps, fields, marching cubes.  Runs INSIDE ChimeraX's
# python; parameters come from a JSON file whose path replaces @@PARAMS@@.
CHIMERAX_JOB = r'''
# Generated by multicolor_split.py @@VERSION@@ -- a ChimeraX Python job.
#   ChimeraX --nogui --exit --script <this file>
# Builds the whole map T and the per-part maps on one grid, forms the
# ownership fields (masked to the inside of the whole at each level), and
# exports every requested contour as STL at voxel step 1, de-dusted.
import json, math, os, traceback
import numpy as np
from chimerax.core.commands import run
from chimerax.map import Volume, volume_from_grid_data
from chimerax.map_data import ArrayGridData
from chimerax.atomic import AtomicStructure, selected_atoms
from chimerax.surface import connected_pieces

PARAMS = @@PARAMS@@


class UserError(Exception):
    """A problem with the input: reported without a traceback."""


def say(msg):
    print("[chimerax] " + msg, flush=True)


def open_structure(P, man):
    run(session, "close", log=False)
    run(session, 'open "%s"' % P["pdb"], log=False)
    structs = [m for m in session.models if isinstance(m, AtomicStructure)]
    if not structs:
        raise UserError("no atomic structure could be read from " + P["pdb"])
    s0 = structs[0]
    base = s0.atomspec
    man["structure_spec"] = base
    man["n_structures"] = len(structs)
    if len(structs) > 1:
        say("%d structures/models opened -- using the first, %s" % (len(structs), base))
    extra = ""
    if P["exclude_solvent"]:
        extra += " & ~solvent"
    if P["exclude_hydrogens"]:
        extra += " & ~H"
    return s0, base, extra


def atom_keys(atoms):
    """chain:residue[insertion]@name for every atom, the names Chimera's
    selections come back as."""
    r = atoms.residues
    return ["%s:%d%s@%s" % (c.strip(), n, ic.strip(), a)
            for c, n, ic, a in zip(r.chain_ids, r.numbers, r.insertion_codes, atoms.names)]


def fingerprint(keys, idx):
    import hashlib
    h = hashlib.sha1()
    for i in idx:
        h.update(keys[i].encode("utf-8") + b"\n")
    return h.hexdigest()


def select_job(P, man):
    """Selection mode: turn each part's atom specification into the atoms it
    takes, once, so every later job maps exactly the same atoms.  An atom two
    parts select goes to the earlier part; rest takes every remaining atom."""
    s0, base, extra = open_structure(P, man)
    atoms = s0.atoms
    keys = atom_keys(atoms)
    run(session, "select clear", log=False)
    run(session, "select %s%s" % (base, extra), log=False)
    considered = np.array(atoms.selected, dtype=bool)
    sels = P["selections"]
    masks, raws, not_found = [], [], []
    for k, p in enumerate(sels, 1):
        if p["kind"] == "rest":
            masks.append(None)
            raws.append(0)
            not_found.append(0)
            continue
        if p["syntax"] == "chimerax":
            run(session, "select clear", log=False)
            try:
                run(session, "select (%s) & %s" % (p["spec"], base), log=False)
            except Exception as e:
                text = str(e).strip()
                raise UserError("P%d (%s): ChimeraX cannot read the selection %r: %s%s" % (
                    k, p["label"], p["spec"], text.splitlines()[0] if text else type(e).__name__,
                    p.get("hint", "")))
            raw = np.array(atoms.selected, dtype=bool)
            not_found.append(0)
        else:                                   # Chimera named its atoms; take those
            want = set(p["keys"])
            raw = np.array([key in want for key in keys], dtype=bool)
            not_found.append(len(want - set(keys)))
        raws.append(int(raw.sum()))
        masks.append(raw & considered)
    run(session, "select clear", log=False)
    counts = {}
    for i in np.nonzero(considered)[0]:
        counts[keys[i]] = counts.get(keys[i], 0) + 1
    twice = sorted(key for key, n in counts.items() if n > 1)
    owner = np.full(len(atoms), -1, dtype=np.int64)
    shared, lost = [], [dict() for _ in sels]
    for k, m in enumerate(masks):
        if m is None:
            continue
        for j in range(k):
            if masks[j] is not None:
                both = m & masks[j]
                if both.any():
                    shared.append(dict(earlier=j, later=k, count=int(both.sum()),
                                       examples=[keys[i] for i in np.nonzero(both)[0][:5]]))
        taken = m & (owner >= 0)
        for j, n in zip(*np.unique(owner[taken], return_counts=True)):
            lost[k][str(int(j))] = int(n)
        owner[m & (owner < 0)] = k
    if shared and P.get("overlap") == "error":
        raise UserError("the part selections share atoms: " + "; ".join(
            "P%d and P%d share %d (%s%s)" % (s["earlier"] + 1, s["later"] + 1, s["count"],
                                            ", ".join(s["examples"]), ", ..." if s["count"] > 5 else "")
            for s in shared) + " -- write them so no atom is in two parts, or leave out "
                               "--overlap error to let the earlier part keep them")
    for k, m in enumerate(masks):
        if m is None:                           # rest: what no other part took
            owner[considered & (owner < 0)] = k
    out = []
    for k, p in enumerate(sels):
        idx = np.nonzero(owner == k)[0]
        if not len(idx):
            if masks[k] is not None and masks[k].any():
                why = "every atom it selects belongs to an earlier part"
            elif p["kind"] == "rest":
                why = "the other parts take every atom"
            elif raws[k]:
                why = ("the %d atom(s) it selects are all solvent or hydrogens, which are left out"
                       % raws[k])
            elif not_found[k]:
                why = ("Chimera selected %d atom(s), but none of them is in the structure as ChimeraX "
                       "reads it (e.g. %s)" % (len(p["keys"]), ", ".join(p["keys"][:3])))
            else:
                chains = sorted({str(c).strip() for c in s0.residues.chain_ids} - {""})
                why = "it selects no atoms (the structure%s has chain(s) %s)%s" % (
                    ", %s," % base if p["syntax"] == "chimerax" else "",
                    " ".join(chains) or "with blank IDs only", p.get("hint", ""))
            raise UserError("P%d (%s) has no atoms: %s" % (k + 1, p["label"], why))
        out.append(dict(selected=int(masks[k].sum()) if masks[k] is not None else int(len(idx)),
                        n_atoms=int(len(idx)), lost=lost[k], not_found=int(not_found[k]),
                        indices=idx.tolist(), fingerprint=fingerprint(keys, idx)))
    left = np.nonzero(considered & (owner < 0))[0]
    part_idx = np.nonzero(owner >= 0)[0]
    elements = atoms.element_names
    man["selection"] = dict(
        n_considered=int(considered.sum()), unassigned=int(len(left)),
        unassigned_examples=[keys[i] for i in left[:5]], shared=shared, parts=out,
        names_twice=len(twice), names_twice_examples=twice[:5],
        atoms=dict(part=owner[part_idx].tolist(),
                   coords=np.asarray(atoms.coords, dtype=float)[part_idx].round(4).tolist(),
                   is_h=[str(elements[i]) in ("H", "D") for i in part_idx],
                   chain=[keys[i].split(":", 1)[0] for i in part_idx]))
    say("parts selected: " + ", ".join("P%d %d atoms" % (k + 1, o["n_atoms"])
                                       for k, o in enumerate(out)))


def job(P, man):
    s0, base, extra = open_structure(P, man)

    def select(chains):
        run(session, "select %s/%s%s" % (base, ",".join(chains), extra), log=False)
        return len(selected_atoms(session))

    def molmap_sel(opts):
        v = run(session, "molmap sel %.10g %s replace false" % (P["resolution"], opts), log=False)
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        if not isinstance(v, Volume):
            v = [m for m in session.models if isinstance(m, Volume)][-1]
        return v

    parts = P["parts"]
    part_atoms = P.get("part_atoms")            # selection mode: each part's atoms, by index
    if part_atoms is not None:
        atoms = s0.atoms
        keys = atom_keys(atoms)
        for k, (idx, fp) in enumerate(zip(part_atoms, P["part_fingerprints"]), 1):
            if max(idx) >= len(atoms) or fingerprint(keys, idx) != fp:
                raise UserError("P%d: the structure's atoms no longer match the ones selected for "
                                "it -- was the file changed during the run?" % k)

        def select_part(k):
            run(session, "select clear", log=False)
            atoms[np.asarray(part_atoms[k], dtype=np.int64)].selected = True
            return len(part_atoms[k])

        run(session, "select clear", log=False)
        union = sorted({i for idx in part_atoms for i in idx})
        atoms[np.asarray(union, dtype=np.int64)].selected = True
        n_all = len(union)
    else:
        allch = [c for p in parts for c in p]
        n_all = select(allch)
    if n_all == 0:
        raise UserError("the selected chains contain no atoms")
    T = molmap_sel("gridSpacing %.10g" % P["grid"])
    Ta = np.array(T.full_matrix(), dtype=np.float32, copy=True)
    man["n_atoms_all"] = n_all
    man["grid"] = dict(size=[int(x) for x in T.data.size],
                       origin=[float(x) for x in T.data.origin],
                       step=[float(x) for x in T.data.step])
    man["T_max"] = float(Ta.max())
    say("whole map: %d atoms, grid %s at %.3g A" % (n_all, "x".join(map(str, T.data.size)), P["grid"]))

    M, natoms = [], []
    for k, chains in enumerate(parts):
        n = select_part(k) if part_atoms is not None else select(chains)
        if n == 0:
            raise UserError("part %s selects no atoms%s" % (
                "+".join(chains), " (after excluding solvent / hydrogens)" if extra else ""))
        v = molmap_sel("onGrid #%s" % T.id_string)
        M.append(np.array(v.full_matrix(), dtype=np.float32, copy=True))
        natoms.append(n)
        run(session, "close #%s" % v.id_string, log=False)
    run(session, "select clear", log=False)
    man["n_atoms_parts"] = natoms
    man["additivity_max_abs"] = float(np.abs(np.sum(M, axis=0, dtype=np.float32) - Ta).max())

    auto = None
    try:
        if T.surfaces:
            auto = float(T.surfaces[0].level)
    except Exception:
        auto = None
    if auto is None:     # fallback: the level enclosing 95 % of the map mass (molmap's rule)
        s = np.sort(Ta.ravel())[::-1]
        cs = np.cumsum(s, dtype=np.float64)
        auto = float(s[min(np.searchsorted(cs, 0.95 * cs[-1]), len(s) - 1)])
    man["auto_level"] = auto
    guess = P["level_guess"] if P["level_guess"] is not None else round(auto, 3)
    man["guess"] = guess
    if P["levels"]:
        levels = [float(x) for x in P["levels"]]
    elif P.get("guess_only"):
        levels = [guess]
    else:
        n = int(math.floor(P["span"] / P["step"] + 1e-9))
        levels = [round(guess + i * P["step"], 6) for i in range(-n, n + 1)]
    levels = sorted(set(L for L in levels if L > 0))
    man["levels"] = levels
    if not levels:
        raise UserError("no positive contour level to try (guess %.4g, span %.4g)" % (guess, P["span"]))
    say("auto level %.5f, guess %.4f, %d candidate level(s)" % (auto, guess, len(levels)))

    owner = np.argmax(np.stack(M), axis=0)
    man["voxel_counts"] = {"%.6f" % L: np.bincount(owner[Ta >= L], minlength=len(M)).tolist()
                           for L in levels}
    del owner

    out = P["out_dir"]
    os.makedirs(out, exist_ok=True)
    grid = T.data
    origin = np.asarray(grid.origin, dtype=np.float64)
    steps_seen, protected = set(), []

    def vq(xyz):
        """Vertex positions quantised to 0.001 A -- to recognise shared vertices."""
        return np.round((np.asarray(xyz, dtype=np.float64) - origin) * 1000.0).astype(np.int64)

    def piece_volume(verts, tris):
        t = np.asarray(verts, dtype=np.float64)[tris]
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    def export(v, level, path, label, anchors_out=None, keep_rows=None, keep_outer=False, save=True):
        # step 1 = every voxel.  Headless ChimeraX keeps full maps at step 1,
        # but force it AND verify it before anything is written.
        run(session, "volume #%s level %.10g style surface step 1 region all" % (v.id_string, level), log=False)
        st = tuple(int(x) for x in v.region[2])
        if st != (1, 1, 1):
            raise RuntimeError("volume #%s is at step %s, not 1 -- refusing to export" % (v.id_string, st))
        steps_seen.add(st)
        # Build the contour NOW: with log=False no 'command finished' trigger
        # fires, so the surface would otherwise only be computed inside `save`
        # -- after `surface dust`, which would then see no triangles.
        v.update_drawings()
        s = v.surfaces[0] if v.surfaces else None
        if s is None or s.vertices is None or abs(float(s.level) - level) > 1e-6 * max(1.0, abs(level)):
            raise RuntimeError("surface of #%s was not built at level %.10g" % (v.id_string, level))
        run(session, "surface dust #%s size %.10g" % (v.id_string, P["dust"]), log=False)
        tris = s.triangles
        m = s.triangle_mask
        pieces = connected_pieces(tris) if len(tris) and (m is not None or anchors_out is not None) else []
        if pieces:
            verts = s.vertices
            vols = [piece_volume(verts, tris[ti]) for vi, ti in pieces]
            big = max(range(len(pieces)), key=lambda i: len(pieces[i][1]))
            if m is not None:
                m = np.array(m, dtype=bool, copy=True)
                hit = None
                if keep_rows is not None and len(keep_rows):
                    rows = np.vstack([keep_rows, vq(verts)])
                    _, inv = np.unique(rows, axis=0, return_inverse=True)
                    inv = inv.reshape(-1)
                    hit = np.isin(inv[len(keep_rows):], inv[:len(keep_rows)])
                changed = False
                for i, (vi, ti) in enumerate(pieces):
                    # dust may FILL small cavities (negative volume) everywhere, but
                    # never removes: any piece of a part field (a part's chain-break
                    # tail, a small part), the whole's largest piece, or a piece of
                    # the whole that carries a part's main piece
                    if m[ti].all() or vols[i] <= 0:
                        continue
                    why = "field" if keep_outer else ("largest" if i == big else
                          ("part piece" if hit is not None and hit[vi].any() else None))
                    if why:
                        m[ti] = True
                        changed = True
                        if why != "field":
                            protected.append("%s(%s)@%.6f" % (label, why, level))
                if changed:
                    s.triangle_mask = m
            if anchors_out is not None:
                vi = pieces[big][0]
                sel = vi[np.linspace(0, len(vi) - 1, num=min(512, len(vi))).astype(np.int64)]
                anchors_out.append(vq(verts[sel]))
        if save:
            run(session, 'save "%s" models #%s' % (path, v.id_string), log=False)

    files = {"ref": {}, "fields": {}}
    anchors = {L: [] for L in levels}
    N = len(M)
    if P.get("blobs"):
        # blob parts: part i (all but the last) is its OWN molmap at the level.
        # Where the other atoms' density M_o covers the blob with a skin of the
        # remainder thinner than blob_skin (estimated as M_o / |grad T|) and the
        # blob's own density exceeds the other atoms' together (M_i > M_o), the
        # field is T itself, so the blob
        # reaches the surface there instead of lying a hair beneath it.
        skin = float(P.get("blob_skin") or 0.0)
        if skin > 0:
            gz, gy, gx = np.gradient(Ta, *[float(s) for s in grid.step[::-1]])   # z, y, x order
            reach = (np.float32(skin) * np.sqrt(gz * gz + gy * gy + gx * gx)).astype(np.float32)
            del gz, gy, gx
        files["fields"]["blob"] = {}
        for i in range(N - 1):
            label = "B%d" % (i + 1)
            other = Ta - M[i]
            F = M[i] if skin <= 0 else \
                np.where((other < reach) & (M[i] > other), Ta, M[i]).astype(np.float32)
            del other
            for L in levels:
                g = ArrayGridData(F, origin=grid.origin, step=grid.step,
                                  name="blob_%s_L%.6f" % (label, L))
                v = volume_from_grid_data(g, session, style="surface", show_dialog=False)
                v.position = T.position
                p = "%s/%s_L%.6f.stl" % (out, label, L)
                export(v, L, p, label, anchors_out=anchors[L], keep_outer=True)
                files["fields"]["blob"].setdefault("%.6f" % L, {})[label] = p
                run(session, "close #%s" % v.id_string, log=False)
            del F
        # Dust spares a piece of the whole only through anchor vertices it
        # shares with a part's field.  A blob field shares the whole's
        # vertices only where the skin rule made it T, and the last part has
        # no field at all, so these get an anchor-only contour of a field that
        # is T wherever the part's density beats every other part's -- what
        # the S_i field of the ownership mode is there.
        for i in [N - 1] + (list(range(N - 1)) if skin <= 0 else []):
            rival = np.maximum.reduce([M[j] for j in range(N) if j != i])
            mine = M[i] >= rival
            del rival
            for L in levels:
                A = np.where(Ta >= np.float32(L), np.where(mine, Ta, np.float32(0)), Ta)
                g = ArrayGridData(A.astype(np.float32), origin=grid.origin, step=grid.step,
                                  name="anchor_P%d_L%.6f" % (i + 1, L))
                v = volume_from_grid_data(g, session, style="surface", show_dialog=False)
                v.position = T.position
                export(v, L, None, "P%d" % (i + 1), anchors_out=anchors[L], keep_outer=True,
                       save=False)
                run(session, "close #%s" % v.id_string, log=False)
                del A
            del mine
        say("blob parts: %d field(s) x %d level(s) exported" % (N - 1, len(levels)))
    # (label, own part, rivals): S_i = part i against all others (the part's
    # own field-built solid), H_k = part k against LATER parts (peeling, k>=2)
    specs = [("S%d" % (i + 1), i, [j for j in range(N) if j != i]) for i in range(N)]
    specs += [("H%d" % (k + 1), k, list(range(k + 1, N))) for k in range(1, N - 1)]
    for K in ([] if P.get("blobs") else P["ks"]):
        kk = "%.12g" % K
        files["fields"][kk] = {}
        for label, own, rivals in specs:
            R = M[rivals[0]] if len(rivals) == 1 else np.maximum.reduce([M[j] for j in rivals])
            F = (Ta - np.float32(K) * np.maximum(R - M[own], np.float32(0))).astype(np.float32)
            del R
            for L in levels:
                # the penalty acts only INSIDE the whole; outside, the field is
                # T itself, so the part's skin coincides with the whole's
                Fm = np.where(Ta >= np.float32(L), F, Ta).astype(np.float32)
                g = ArrayGridData(Fm, origin=grid.origin, step=grid.step,
                                  name="field_%s_K%s_L%.6f" % (label, kk, L))
                v = volume_from_grid_data(g, session, style="surface", show_dialog=False)
                v.position = T.position
                p = "%s/%s_K%s_L%.6f.stl" % (out, label, kk, L)
                export(v, L, p, label, anchors_out=anchors[L] if label.startswith("S") else None,
                       keep_outer=True)
                files["fields"][kk].setdefault("%.6f" % L, {})[label] = p
                run(session, "close #%s" % v.id_string, log=False)
                del Fm
            del F
        say("K=%s: %d field(s) x %d level(s) exported" % (kk, len(specs), len(levels)))
    # the whole LAST, so dust can spare every piece that carries a part's main piece
    if P["export_ref"]:
        for L in levels:
            p = "%s/ref_L%.6f.stl" % (out, L)
            keep = np.vstack(anchors[L]) if anchors[L] else None
            export(T, L, p, "whole", keep_rows=keep)
            files["ref"]["%.6f" % L] = p
    man["files"] = files
    man["export_steps"] = sorted(list(x) for x in steps_seen)
    man["dust_protected"] = protected


man = {"ok": False}
P = None
try:
    P = json.load(open(PARAMS))
    try:
        from chimerax.core import version as _v
        man["chimerax_version"] = str(_v)
    except Exception:
        pass
    if P.get("mode") == "select":
        select_job(P, man)
    else:
        job(P, man)
    man["ok"] = True
except UserError as e:
    man["error"] = str(e)
    man["user_error"] = True
    say("ERROR " + str(e))
except Exception as e:
    man["error"] = "%s: %s" % (type(e).__name__, e)
    man["traceback"] = traceback.format_exc()
    say("ERROR " + man["error"])
with open(P["manifest"] if P else PARAMS + ".manifest.json", "w") as fh:
    json.dump(man, fh)
'''


def run_chimerax(exe, params, work, phase):
    """Write params + job into the (space-free) work dir, run ChimeraX headless."""
    scan_dir = work / f"scan_{phase}"
    scan_dir.mkdir(parents=True, exist_ok=True)
    params = dict(params, out_dir=str(scan_dir),
                  manifest=str(work / f"manifest_{phase}.json"))
    pfile = work / f"params_{phase}.json"
    pfile.write_text(json.dumps(params, indent=1))
    jfile = work / f"chimerax_job_{phase}.py"
    jfile.write_text(CHIMERAX_JOB.replace("@@PARAMS@@", repr(str(pfile)))
                     .replace("@@VERSION@@", __version__))
    t0 = time.time()
    try:
        proc = subprocess.Popen([exe, "--nogui", "--exit", "--script", jfile.name],
                                cwd=str(work), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    except OSError as e:
        raise SplitError(f"cannot start ChimeraX ({exe}): {e}")
    lines = []
    try:
        for line in proc.stdout:
            lines.append(line)
            if line.startswith("[chimerax]"):
                log("   " + line.rstrip())
        proc.wait()
    finally:
        if proc.poll() is None:                 # interrupted: do not leave ChimeraX running
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        (work / f"chimerax_{phase}.log").write_text("".join(lines))
    mpath = Path(params["manifest"])
    if not mpath.exists():
        mpath = Path(str(pfile) + ".manifest.json")
    if not mpath.exists():
        tail = "".join(lines[-25:])
        raise SplitError(f"ChimeraX exited ({proc.returncode}) without a result:\n{tail}")
    man = json.loads(mpath.read_text())
    if not man.get("ok"):
        if man.get("user_error"):
            raise SplitError(man.get("error", "?"))
        raise SplitError("ChimeraX job failed: " + man.get("error", "?") + "\n" +
                         man.get("traceback", "") + f"(full log: work/chimerax_{phase}.log)")
    log(f"   ChimeraX {phase} job done in {time.time() - t0:.0f} s")
    return man


# The Chimera side, for selections in Chimera's convention: UCSF Chimera reads
# the structure and evaluates each part's atom specification itself, so its
# own rules apply, and names the atoms it picks by chain, residue and atom
# name; ChimeraX then takes exactly those atoms.  Chimera runs Python 2.7.
CHIMERA_JOB = r'''
# Generated by multicolor_split.py @@VERSION@@ -- a UCSF Chimera job.
#   chimera --nogui --silent --script "<this file> <params.json>"
import json, re, sys, traceback
out = {"ok": False}
params = None
try:
    params = json.load(open(sys.argv[1]))
    import chimera
    from chimera.specifier import evalSpec
    out["chimera_version"] = str(chimera.version.release)
    models = chimera.openModels.open(params["pdb"])
    if not models:
        out["user_error"] = True
        raise ValueError("Chimera could not read " + params["pdb"])
    models = models[:1]                          # the first model, as the ChimeraX side uses
    keys = []
    for name, spec, hint in zip(params["names"], params["specs"], params["hints"]):
        try:
            atoms = evalSpec(spec, models=models).atoms()
        except Exception as e:                   # (%r would print u'...' in Python 2)
            out["user_error"] = True
            why = re.sub(r"\s*\(, line \d+\)$", "", " ".join(str(e).split()))
            raise ValueError("%s: Chimera cannot read the selection '%s': %s%s" % (
                name, spec, why or type(e).__name__, hint))
        found = set()
        for a in atoms:
            rid = a.residue.id
            found.add("%s:%d%s@%s" % (rid.chainId.strip(), rid.position,
                                      rid.insertionCode.strip(), a.name))
        keys.append(sorted(found))
    out["keys"] = keys
    out["ok"] = True
except Exception as e:
    if out.get("user_error"):
        out["error"] = e.args[0] if e.args else str(e)
    else:
        out["error"] = "%s: %s" % (type(e).__name__, e)
        out["traceback"] = traceback.format_exc()
with open(params["manifest"] if params else sys.argv[1] + ".manifest.json", "w") as fh:
    json.dump(out, fh)
'''


def run_chimera(exe, pdb, specs, work, names=None):
    """Evaluate `specs` in UCSF Chimera; returns its manifest, whose "keys"
    list the atoms each spec picks as chain:residue[insertion]@name.  names
    (default: the specs themselves) say which part an error is about."""
    pfile = work / "params_chimera.json"
    manifest = work / "manifest_chimera.json"
    specs = list(specs)
    pfile.write_text(json.dumps(dict(pdb=str(pdb), specs=specs, manifest=str(manifest),
                                     names=list(names) if names is not None else specs,
                                     hints=[convention_hint(x, "chimera") for x in specs]),
                                indent=1))
    jfile = work / "chimera_job_select.py"
    jfile.write_text(CHIMERA_JOB.replace("@@VERSION@@", __version__))
    t0 = time.time()
    try:                                        # an exception here also stops Chimera
        proc = subprocess.run([exe, "--nogui", "--silent", "--script", f"{jfile.name} {pfile.name}"],
                              cwd=str(work), capture_output=True, text=True, timeout=900)
    except OSError as e:
        raise SplitError(f"cannot start Chimera ({exe}): {e}")
    except subprocess.TimeoutExpired:
        raise SplitError("Chimera did not finish evaluating the selections within 15 minutes")
    (work / "chimera_select.log").write_text((proc.stdout or "") + (proc.stderr or ""))
    if not manifest.exists():
        tail = "".join(((proc.stdout or "") + (proc.stderr or "")).splitlines(True)[-25:])
        raise SplitError(f"Chimera exited ({proc.returncode}) without a result:\n{tail}")
    man = json.loads(manifest.read_text())
    if not man.get("ok"):
        if man.get("user_error"):
            raise SplitError(man.get("error", "?"))
        raise SplitError("Chimera job failed: " + man.get("error", "?") + "\n" +
                         man.get("traceback", "") + "(full log: work/chimera_select.log)")
    log(f"   Chimera selection job done in {time.time() - t0:.0f} s")
    return man


# ================================================================ mesh tools
def _import_mesh_libs():
    try:
        import trimesh            # noqa: F401
        import manifold3d as mf   # noqa: F401
        import scipy              # noqa: F401
    except ImportError as e:
        raise SplitError(f"missing python package ({e.name}); run this with a python3 that has "
                         f"numpy, scipy, trimesh and manifold3d, e.g. /opt/anaconda3/bin/python3 "
                         f"(or `pip install trimesh manifold3d scipy`)")
    globals()["trimesh"] = trimesh
    globals()["mf"] = mf


def weld(V, F):
    """Exact float32 vertex welding -- what a slicer does when it reads an STL
    -- then drop the zero-area triangles that welding collapses (a corner
    repeated), exactly as slicers do.  Returns (V, F, n_dropped)."""
    V32 = np.ascontiguousarray(np.asarray(V, dtype=np.float32))
    if len(V32) == 0:
        return V32.reshape(0, 3), np.zeros((0, 3), np.int64), 0
    uniq, inv = np.unique(V32, axis=0, return_inverse=True)
    G = inv.reshape(-1)[np.asarray(F, dtype=np.int64)]
    keep = (G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2])
    n_deg = int((~keep).sum())
    G = G[keep]
    used, remap = np.unique(G, return_inverse=True)
    return uniq[used], remap.reshape(-1, 3).astype(np.int64), n_deg


def unpinch(V, F):
    """Pull apart the vertex copies a boolean result keeps at one point.

    manifold3d represents two sheets of a surface that touch along a line --
    where the complement closes to zero thickness -- with a separate vertex
    for each sheet at the same position, so its mesh is a true 2-manifold.  An
    STL has no vertex identities, only coordinates: a slicer that welds equal
    coordinates, as Bambu Studio and PrusaSlicer do, fuses the copies, each
    such line then has edges shared by four triangles, and the slicer reports
    them as non-manifold edges.  Each copy is moved toward the centroid of its
    own triangles and into its own solid, by a step far below any printer's
    resolution -- 1/1000 of the median edge, and at least 32 float32 steps at
    the mesh's size, so the copies stay apart once written as float32 -- which
    keeps the file a 2-manifold after welding.

    Returns (V, n_moved, step, n_left): n_left copies still share a position
    (0 unless two copies' own triangles point the same way)."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if len(V) == 0 or len(F) == 0:
        return V, 0, 0.0, 0

    def shared(P):
        _, inv, counts = np.unique(P.astype(np.float32), axis=0, return_inverse=True,
                                   return_counts=True)
        return np.nonzero(counts[inv.reshape(-1)] > 1)[0]

    idx = shared(V)
    if not len(idx):
        return V, 0, 0.0, 0
    tri = V[F]
    ring = np.zeros_like(V)
    np.add.at(ring, F.reshape(-1), np.repeat(tri.mean(axis=1), 3, axis=0))
    normal = np.zeros_like(V)
    np.add.at(normal, F.reshape(-1), np.repeat(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                               3, axis=0))
    count = np.bincount(F.reshape(-1), minlength=len(V)).astype(np.float64)

    def unit(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return np.divide(X, n, out=np.zeros_like(X), where=n > 0)

    toward = unit(ring[idx] / np.maximum(count[idx], 1.0)[:, None] - V[idx])
    inward = unit(-normal[idx])                 # manifold3d winds its triangles outward
    direction = unit(toward + inward)
    ok = np.linalg.norm(direction, axis=1) > 0
    edges = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2)
    step = max(1e-3 * float(np.median(edges)),
               32.0 * float(np.spacing(np.float32(np.abs(V).max()))))
    out = V.copy()
    out[idx[ok]] += direction[ok] * step
    left = shared(out)
    return out, int(ok.sum()), step, int(len(np.intersect1d(left, idx)))


def read_stl(path):
    m = trimesh.load(str(path), force="mesh", process=False)
    if len(m.faces) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), 0
    return weld(m.vertices, m.faces)


def mesh_stats(V, F, n_deg=0):
    """Topology of a welded mesh: open / pinch (>2 faces) edges, shells split
    into outer pieces and cavities, volume, genus (None when not a 2-manifold)."""
    F = np.asarray(F, dtype=np.int64)
    if len(F) == 0:
        return dict(faces=0, degenerate=n_deg, open_edges=0, pinch_edges=0, bodies=0, outer=0,
                    cavities=0, volume=0.0, genus=None, clean=False)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    open_e, pinch = int((cnt == 1).sum()), int((cnt > 2).sum())
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(V)
    A = coo_matrix((np.ones(len(ue)), (ue[:, 0], ue[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    used = np.unique(F)
    tri = np.asarray(V, dtype=np.float64)[F]
    fvol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    flab = lab[F[:, 0]]
    comps = np.unique(flab)
    cvol = np.bincount(flab, weights=fvol, minlength=int(lab.max()) + 1)[comps]
    bodies = int(len(comps))
    euler = len(used) - len(ue) + len(F)
    clean = bool(open_e == 0 and pinch == 0)
    return dict(faces=int(len(F)), degenerate=int(n_deg), open_edges=open_e,
                pinch_edges=pinch, bodies=bodies, outer=int((cvol > 0).sum()),
                cavities=int((cvol < 0).sum()), volume=float(fvol.sum()),
                genus=int(round(bodies - euler / 2)) if clean else None, clean=clean)


def to_manifold(V, F, what):
    mesh = mf.Mesh(vert_properties=np.ascontiguousarray(V, dtype=np.float32),
                   tri_verts=np.ascontiguousarray(F, dtype=np.uint32))
    M = mf.Manifold(mesh)
    if M.status() != mf.Error.NoError:
        raise SplitError(f"{what} is not a valid manifold ({M.status()})")
    return M


def arrays(M):
    m = M.to_mesh()
    V = np.asarray(m.vert_properties, dtype=np.float32)[:, :3]
    F = np.asarray(m.tri_verts, dtype=np.int64)
    return V, F


def concat_manifolds(bodies):
    """One Manifold from disjoint bodies -- outer shells AND inverted cavity
    shells.  (Manifold.compose drops cavity shells: a hollow part would come
    out filled.)"""
    if len(bodies) == 1:
        return bodies[0]
    Vs, Fs, off = [], [], 0
    for b in bodies:
        V, F = arrays(b)
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    M = mf.Manifold(mf.Mesh(vert_properties=np.ascontiguousarray(np.concatenate(Vs), dtype=np.float32),
                            tri_verts=np.ascontiguousarray(np.concatenate(Fs), dtype=np.uint32)))
    if M.status() != mf.Error.NoError:
        raise SplitError(f"could not reassemble a part from its bodies ({M.status()})")
    return M


def _bbox_touch(a, b, pad=1e-3):
    A, B = a.bounding_box(), b.bounding_box()
    return all(A[i] <= B[i + 3] + pad and B[i] <= A[i + 3] + pad for i in range(3))


def _bbox_contains(outer, inner, pad=1e-3):
    return all(outer[i] - pad <= inner[i] and inner[i + 3] <= outer[i + 3] + pad for i in range(3))


def whole_solids(M):
    """The whole's pieces as solids: each outer shell together with the cavities
    it encloses.  A body floating inside a cavity (cargo in a cage) is its own
    piece."""
    bodies = M.decompose()
    pos = [b for b in bodies if b.volume() > 0]
    neg = [b for b in bodies if b.volume() < 0]
    if not neg:
        return pos
    groups = [[p] for p in pos]
    for n in neg:
        nb = n.bounding_box()
        inside = [i for i, p in enumerate(pos)
                  if p.volume() > -n.volume() and _bbox_contains(p.bounding_box(), nb)]
        if inside:
            groups[min(inside, key=lambda i: pos[i].volume())].append(n)
    return [concat_manifolds(g) for g in groups]


def _pinch_edges(M):
    return mesh_stats(*weld(*arrays(M))[:2])["pinch_edges"]


FINISH_KEYS = ("bodies", "cavities", "slivers", "sliver_volume", "received", "received_volume",
               "dropped_volume")


def finish_parts(raw, sliver):
    """Clean the peeled parts.  Bodies below `sliver` (SLIVER_VOXELS voxels,
    0.5 A^3 at gridSpacing 0.5) are boolean fragments (they form where three parts meet):
    each positive fragment goes to the neighbouring part it shares the most
    surface with -- preferring one it merges into without a new pinch edge --
    so the tiling stays exact; a fragment touching nothing is dropped (counted);
    a fragment-sized cavity is filled.  Every other body is kept, cavities
    included (a real ion or ligand piece is far larger than a few voxels)."""
    parts, infos, loose = [], [], []
    for idx, P in enumerate(raw):
        info = dict(bodies=0, cavities=0, slivers=0, sliver_volume=0.0,
                    received=0, received_volume=0.0, dropped_volume=0.0)
        bodies = P.decompose()
        if not bodies:
            parts.append(P)
            infos.append(info)
            continue
        vols = [b.volume() for b in bodies]
        keep = [(b, v) for b, v in zip(bodies, vols) if abs(v) >= sliver]
        tiny = [(b, v) for b, v in zip(bodies, vols) if abs(v) < sliver]
        parts.append(concat_manifolds([b for b, _ in keep]) if tiny and keep else
                     (P if not tiny else mf.Manifold()))
        info.update(bodies=int(sum(v > 0 for _, v in keep)), cavities=int(sum(v < 0 for _, v in keep)),
                    slivers=len(tiny), sliver_volume=float(sum(abs(v) for _, v in tiny)))
        infos.append(info)
        loose += [(idx, b, v) for b, v in tiny if v > 0]
    for own, b, v in loose:
        ab = b.surface_area()
        cands = []
        for j, Pj in enumerate(parts):
            if j == own or Pj.is_empty() or not _bbox_touch(b, Pj):
                continue
            U = Pj + b
            contact = (ab + Pj.surface_area() - U.surface_area()) / 2.0
            if contact > 1e-9:
                cands.append((contact, j, U))
        if not cands:
            infos[own]["dropped_volume"] += v
            continue
        cands.sort(key=lambda t: -t[0])
        choice = cands[0]
        if len(cands) > 1:
            for cand in cands:
                if cand[0] < 0.5 * cands[0][0]:
                    break
                if _pinch_edges(cand[2]) <= _pinch_edges(parts[cand[1]]):
                    choice = cand
                    break
        _, j, U = choice
        parts[j] = U
        infos[j]["received"] += 1
        infos[j]["received_volume"] += v
    return parts, infos


def outer_bodies(M):
    return [b for b in M.decompose() if b.volume() > 0]


def surface_is_on(V, refV, tol=1e-4):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(refV).query(V, distance_upper_bound=tol * 10)
    return d <= tol


def tri_areas(V, F):
    t = np.asarray(V, dtype=np.float64)[F]
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


# ============================================================== construction
def blob_names(N):
    """'P1 is its own molmap surface, and P2 is' / 'P1-P3 are each ... P4 is'."""
    return ("P1 is its own molmap surface, and P2 is" if N == 2 else
            f"P1-P{N - 1} are each their own molmap surface, and P{N} is")


def field_key(K):
    """The key of one K's fields in a ChimeraX job's files; blob parts have
    no K."""
    return "blob" if K is None else f"{K:.12g}"


def peel(ref, fields):
    """P_k = R_(k-1) & F_k, R_k = R_(k-1) - F_k; the last part is the remainder."""
    R, parts = ref, []
    for F in fields:
        parts.append(R ^ F)
        R = R - F
    parts.append(R)
    return parts


def fmt_mesh(s, expected=None, at_most=False):
    """Compact scan-table cell for one mesh's stats."""
    if s["faces"] == 0:
        return "empty"
    bits = []
    if s["open_edges"]:
        bits.append(f"{s['open_edges']}open")
    if s["pinch_edges"]:
        bits.append(f"{s['pinch_edges']}NM")
    pcs = s.get("pieces_in_whole", s["outer"])
    if expected is not None and (pcs > expected if at_most else pcs != expected) or pcs == 0:
        bits.append(f"{pcs}pc")
    return ",".join(bits) or "ok"


_STL_CACHE = {}


def read_stl_cached(path):
    if path not in _STL_CACHE:
        _STL_CACHE[path] = read_stl(path)
    return _STL_CACHE[path]


def whole_counts(files, levels):
    """Pieces (outer shells) of the clean whole at each level."""
    counts = {}
    for L in levels:
        s = mesh_stats(*read_stl_cached(files["ref"][f"{L:.6f}"]))
        if s["clean"] and s["faces"]:
            counts[L] = s["outer"]
    return counts


def expected_pieces(counts, guess, n_mol, pinned):
    """(pieces the whole must have, why)."""
    if pinned:
        return (next(iter(counts.values())) if counts else None), "the pinned level"
    if n_mol and n_mol in counts.values():
        return n_mol, (f"{n_mol} separate molecule(s)" if n_mol > 1 else "one connected molecule") \
            + f" (heavy atoms within {CONTACT_CUTOFF:g} A)"
    near = [L for L in counts if abs(L - round(guess, 6)) < 1e-9]
    if near:
        return counts[near[0]], "the whole at the level guess (no clean level matches the molecule count)"
    if counts:
        return min(counts.values()), "the fewest pieces in the scan"
    return None, "no clean level"


def evaluate(files, L, K, N, strict, expected_whole, clusters, sliver, construct=True, force=False):
    """Check one (K, level) candidate.  Returns a JSON-able record.

    clusters: per part, the most pieces its solid may have (its own heavy-atom
    clusters).  force: build the parts even if a check fails (a pinned level),
    as long as the meshes the construction needs are valid manifolds.
    """
    Ls = f"{L:.6f}"
    rec = dict(level=L, k=K, accepted=False, reasons=[])
    ref_V, ref_F, ref_deg = read_stl_cached(files["ref"][Ls])
    rs = mesh_stats(ref_V, ref_F, ref_deg)
    rec["whole"] = rs
    rec["fields"] = {}
    if rs["faces"] == 0:
        rec["reasons"].append("whole: empty (level above the map maximum?)")
        rec["fields_ok"] = False
        return rec
    whole_ok = rs["clean"] and (expected_whole is None or rs["outer"] == expected_whole)
    if not whole_ok:
        rec["reasons"].append("whole: " + fmt_mesh(rs, expected_whole)
                              + (f" (expected {expected_whole})" if rs["clean"] and
                                 expected_whole is not None else ""))
    ref = None
    if rs["clean"]:
        try:
            ref = to_manifold(ref_V, ref_F, "whole")
        except SplitError as e:
            whole_ok = False
            rec["reasons"].append(str(e))
    flds = files["fields"][field_key(K)][Ls]
    blobs = "B1" in flds                    # blob parts: B_i, no S_N, no H_k
    meshes = {}
    fields_ok = True
    for label in sorted(flds, key=lambda s: (s[0] != "S", int(s[1:]))):
        V, F, nd = read_stl(flds[label])
        s = mesh_stats(V, F, nd)
        rec["fields"][label] = s
        meshes[label] = (V, F)
        if not s["clean"] or s["faces"] == 0:
            fields_ok = False
            rec["reasons"].append(f"{label}: " + fmt_mesh(s))
    # the documented rule: each part's own field-built solid is ONE piece -- here,
    # inside the whole, at most one piece per cluster of the part's own atoms
    if strict and ref is not None:
        for i in range(1, N + 1):
            lab = f"B{i}" if blobs else f"S{i}"
            s = rec["fields"].get(lab)          # (the last blob-mode part has no field)
            if s is None or not s["clean"] or not s["faces"]:
                continue
            try:
                pcs = len(outer_bodies(to_manifold(*meshes[lab], lab) ^ ref))
            except SplitError as e:
                fields_ok = False
                rec["reasons"].append(f"{lab}: {e}")
                continue
            s["pieces_in_whole"], s["expected_bodies"] = pcs, clusters[i - 1]
            if pcs < 1 or pcs > clusters[i - 1]:
                fields_ok = False
                rec["reasons"].append(f"{lab}: {pcs}pc (at most {clusters[i - 1]})")
    rec["fields_ok"] = whole_ok and fields_ok
    if not construct or ref is None or (not rec["fields_ok"] and not force):
        return rec
    try:
        order = [f"B{k}" for k in range(1, N)] if blobs else ["S1"] + [f"H{k}" for k in range(2, N)]
        fields = [to_manifold(*meshes[lab], lab) for lab in order]
        parts, pinfo = finish_parts(peel(ref, fields), sliver)
    except SplitError as e:
        rec["reasons"].append(str(e))
        rec["buildable"] = False
        return rec
    rec["buildable"] = True
    Vw = ref.volume()
    for P, info, cl in zip(parts, pinfo, clusters):
        V, F = arrays(P)
        ws = mesh_stats(*weld(V, F)) if len(F) else mesh_stats(V, F)
        info.update(volume=P.volume(), genus=P.genus(), triangles=int(P.num_tri()),
                    welded_pinch_edges=ws["pinch_edges"], welded_open_edges=ws["open_edges"],
                    expected_bodies=cl)
    total = sum(p["volume"] for p in pinfo)
    rec["parts"] = pinfo
    rec["whole_volume"] = Vw
    rec["sum_err"] = (total - Vw) / Vw if Vw > 0 else float("inf")
    union = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    rec["void"] = max(0.0, float((ref - union).volume()))
    ov = {}
    for i in range(N):
        for j in range(i + 1, N):
            ov[f"{i + 1}-{j + 1}"] = max(0.0, float((parts[i] ^ parts[j]).volume()))
    rec["overlap"] = ov
    rec["pinch_total"] = int(sum(p["welded_pinch_edges"] for p in pinfo))
    for i, p in enumerate(pinfo, 1):
        if p["volume"] < sliver:
            rec["reasons"].append(f"part {i}: empty")
        elif strict and p["bodies"] > p["expected_bodies"]:
            rec["reasons"].append(f"part {i}: {p['bodies']} pieces (at most {p['expected_bodies']})")
        if p["welded_open_edges"]:
            rec["reasons"].append(f"part {i}: {p['welded_open_edges']} open edges")
    if abs(rec["sum_err"]) > TOL_REL:
        rec["reasons"].append(f"sum {100 * rec['sum_err']:+.4f}%")
    if max(ov.values(), default=0.0) > TOL_REL * Vw:
        rec["reasons"].append(f"overlap {max(ov.values()):.3g}")
    if rec["void"] > TOL_REL * Vw:
        rec["reasons"].append(f"void {rec['void']:.3g}")
    rec["accepted"] = not rec["reasons"]
    rec["_built"] = (ref, (ref_V, ref_F, ref_deg), parts,
                     [{k: v for k, v in p.items() if k in FINISH_KEYS} for p in pinfo])
    return rec


def build_final(files, L, K, N, sliver, built=None):
    """Rebuild the chosen candidate, keeping the manifold objects.  S (each
    part's own field solid inside the whole) feeds only the report's boundary
    statistics: if one is not a valid manifold S is returned as None.
    built: what evaluate() made for this candidate (the same calls on the
    same files), reused instead of peeling again."""
    Ls = f"{L:.6f}"
    flds = files["fields"][field_key(K)][Ls]
    if built is not None:
        ref, ref_mesh, parts, infos = built
    else:
        ref_mesh = read_stl_cached(files["ref"][Ls])
        ref = to_manifold(ref_mesh[0], ref_mesh[1], "whole")
    if "B1" in flds:                        # blob parts: no own-field statistics
        if built is None:
            fields = [to_manifold(*read_stl(flds[f"B{k}"])[:2], f"B{k}") for k in range(1, N)]
            parts, infos = finish_parts(peel(ref, fields), sliver)
        return ref, ref_mesh, None, parts, infos
    S1 = to_manifold(*read_stl(flds["S1"])[:2], "S1")
    if built is None:
        fields = [S1] + [to_manifold(*read_stl(flds[f"H{k}"])[:2], f"H{k}") for k in range(2, N)]
    S = [S1 ^ ref]
    for i in range(2, N + 1):
        try:
            S.append(to_manifold(*read_stl(flds[f"S{i}"])[:2], f"S{i}") ^ ref)
        except SplitError:
            S = None
            break
    if built is None:
        parts, infos = finish_parts(peel(ref, fields), sliver)
    return ref, ref_mesh, S, parts, infos


# ============================================================ outputs & report
def placement(refV, lay_flat, scale):
    """4x4 transform applied to every exported mesh (identical for all parts)."""
    A = np.eye(4)
    if lay_flat:
        X = np.asarray(refV, dtype=np.float64)
        c = X.mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov((X - c).T))
        R = vecs[:, ::-1].T.copy()          # rows: longest -> x, middle -> y, shortest -> z
        if np.linalg.det(R) < 0:
            R[2] *= -1
        Y = (X - c) @ R.T
        t = -np.array([(Y[:, 0].min() + Y[:, 0].max()) / 2,
                       (Y[:, 1].min() + Y[:, 1].max()) / 2, Y[:, 2].min()])
        A[:3, :3] = R
        A[:3, 3] = t - R @ c
    S = np.diag([scale, scale, scale, 1.0])
    return S @ A


def apply(A, V):
    return np.asarray(V, dtype=np.float64) @ A[:3, :3].T + A[:3, 3]


def export_stl(V, F, path):
    trimesh.Trimesh(vertices=np.asarray(V, dtype=np.float64), faces=F,
                    process=False).export(str(path))


def splat_views(meshes, colors, views, width=760, height=520):
    """Tiny software renderer: dense surface samples, z-buffered per pixel."""
    rng = np.random.default_rng(0)
    pts, nrm, col = [], [], []
    allV = np.concatenate([V for V, _ in meshes])
    ext = allV.max(0) - allV.min(0)
    tot_area = sum(tri_areas(V, F).sum() for V, F in meshes)
    px = min(width, height) / max(ext.max(), 1e-6) * 0.9
    nsamp = int(min(8e6, max(3e5, 3.0 * tot_area * px * px)))
    for (V, F), c in zip(meshes, colors):
        a = tri_areas(V, F)
        if a.sum() == 0:
            continue
        n = max(1, int(nsamp * a.sum() / tot_area))
        fi = rng.choice(len(F), n, p=a / a.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        t = np.asarray(V, dtype=np.float64)[F[fi]]
        pts.append(t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0]))
        fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        nrm.append(fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12))
        col.append(np.tile(np.asarray(c, dtype=np.float64), (n, 1)))
    P, N, C = np.concatenate(pts), np.concatenate(nrm), np.concatenate(col)
    P = P - allV.mean(axis=0)
    images = []
    for elev, azim in views:
        el, az = math.radians(elev), math.radians(azim)
        Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, math.cos(el - math.pi / 2), -math.sin(el - math.pi / 2)],
                       [0, math.sin(el - math.pi / 2), math.cos(el - math.pi / 2)]])
        Rv = Rx @ Rz
        Q, Nq = P @ Rv.T, N @ Rv.T
        lo, hi = Q[:, :2].min(0), Q[:, :2].max(0)
        s = min((width - 20) / max(hi[0] - lo[0], 1e-6), (height - 20) / max(hi[1] - lo[1], 1e-6))
        x = ((Q[:, 0] - lo[0]) * s + 10).astype(np.int64)
        y = ((hi[1] - Q[:, 1]) * s + 10).astype(np.int64)
        light = np.array([-0.35, 0.45, 0.82])
        light /= np.linalg.norm(light)
        shade = 0.28 + 0.72 * np.abs(Nq @ light)
        rgb = np.clip(C * shade[:, None], 0, 1)
        img = np.ones((height, width, 3))
        zbuf = np.full(height * width, -np.inf)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            xx, yy = np.clip(x + dx, 0, width - 1), np.clip(y + dy, 0, height - 1)
            idx = yy * width + xx
            order = np.lexsort((-Q[:, 2], idx))
            first = np.ones(len(order), bool)
            first[1:] = idx[order][1:] != idx[order][:-1]
            sel = order[first]
            better = Q[sel, 2] > zbuf[idx[sel]]
            sel = sel[better]
            zbuf[idx[sel]] = Q[sel, 2]
            img.reshape(-1, 3)[idx[sel]] = rgb[sel]
        # depth-aware hole fill: a pixel far BEHIND its 3x3 neighbourhood is a
        # gap in the front surface showing a back layer -- take the front colour
        Z = zbuf.reshape(height, width)
        tol = 1.5 / s            # ~1.5 px of depth, in model units
        for _ in range(2):
            Zp = np.pad(Z, 1, constant_values=-np.inf)
            Ip = np.pad(img, ((1, 1), (1, 1), (0, 0)), constant_values=1.0)
            stack_z = np.stack([Zp[dy:dy + height, dx:dx + width]
                                for dy in range(3) for dx in range(3)])
            stack_c = np.stack([Ip[dy:dy + height, dx:dx + width]
                                for dy in range(3) for dx in range(3)])
            k = np.argmax(stack_z, axis=0)
            zfront = np.take_along_axis(stack_z, k[None], 0)[0]
            leak = np.isfinite(zfront) & (Z < zfront - tol)
            if not leak.any():
                break
            cfront = np.take_along_axis(stack_c, k[None, :, :, None], 0)[0]
            img[leak] = cfront[leak]
            Z = np.where(leak, zfront, Z)
        images.append(img)
    return images


def render_preview(meshes, labels, colors, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    views = [(90, -90), (25, -60)]
    imgs = splat_views(meshes, colors, views)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), dpi=110)
    for ax, img, lab in zip(axs, imgs, ("top (print bed = page)", "oblique")):
        ax.imshow(img)
        ax.set_title(lab, fontsize=10)
        ax.axis("off")
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)],
               loc="lower center", ncol=min(len(labels), 6), frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(str(path))
    plt.close(fig)


def srgb_to_linear(c):
    """One sRGB channel in 0-1 as the linear value glTF colours are read as."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def export_glb(meshes, labels, colors, path):
    """One matte material per part, in the PNG preview's colours.

    glTF reads a material's baseColorFactor, like vertex colours, as LINEAR
    values, and a primitive with no material gets the specification's default,
    which is fully metallic.  Writing the sRGB colours as vertex colours on
    bare primitives therefore showed the parts as tinted metal, paler than the
    PNG.  Each part now carries a non-metallic material whose colour is the
    sRGB colour converted to linear, set exactly on the written tree (trimesh
    keeps material colours as 8-bit values), and vertex normals, so viewers
    shade the surface smoothly as the PNG does."""
    import trimesh                          # already loaded by _import_mesh_libs
    scene = trimesh.Scene()
    exact = {}
    for (V, F), lab, c in zip(meshes, labels, colors):
        tm = trimesh.Trimesh(vertices=np.asarray(V, dtype=np.float64), faces=F, process=False)
        linear = [srgb_to_linear(float(x)) for x in c[:3]] + [1.0]
        exact[lab] = linear
        tm.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name=lab, baseColorFactor=linear, metallicFactor=0.0, roughnessFactor=0.8))
        scene.add_geometry(tm, node_name=lab, geom_name=lab)

    def exact_colors(tree):
        for material in tree.get("materials", []):
            if material.get("name") in exact:
                material.setdefault("pbrMetallicRoughness", {})["baseColorFactor"] = exact[material["name"]]

    scene.export(str(path), file_type="glb", include_normals=True, tree_postprocessor=exact_colors)


def pct(x):
    return f"{100 * x:+.4f} %"


# ================================================================== pipeline
def nearest_first(levels, guess, check):
    """Blob mode's level scan: check(L) -> record for the levels nearest the
    guess first, stopping at the first distance where one is accepted, since
    choose() takes the accepted level nearest the guess.  Returns (records in
    level order, how many levels were left unchecked)."""
    order = sorted(levels, key=lambda x: (round(abs(x - guess), 9), x))
    recs = []
    for n_done, L in enumerate(order):
        if any(r["accepted"] for r in recs) and \
                round(abs(L - guess), 9) > round(abs(recs[-1]["level"] - guess), 9):
            return sorted(recs, key=lambda r: r["level"]), len(order) - n_done
        recs.append(check(L))
    return sorted(recs, key=lambda r: r["level"]), 0


def choose(records, guess):
    ok = [r for r in records if r.get("accepted")]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (round(abs(r["level"] - guess), 9),
                                     r.get("pinch_total", 0), r["level"]))[0]


def scan_table(records, N, expected_whole, blobs=False):
    if not records:
        return "  (no candidate levels)"
    own = [f"B{i}" for i in range(1, N)] if blobs else [f"S{i}" for i in range(1, N + 1)]
    peeling = [] if blobs else [f"H{k}" for k in range(2, N)]
    heads = ["level", "whole"] + own + peeling + ["parts", "pinch", "sum", "verdict"]
    rows = []
    for r in records:
        row = [f"{r['level']:.4f}", fmt_mesh(r["whole"], expected_whole)]
        for lab in own:
            s = r["fields"].get(lab)
            row.append(fmt_mesh(s, s.get("expected_bodies")) if s else "-")
        for lab in peeling:
            s = r["fields"].get(lab)
            row.append(fmt_mesh(s) if s else "-")
        if "parts" in r:
            row += ["/".join(str(p["bodies"]) for p in r["parts"]), str(r["pinch_total"]),
                    f"{100 * r['sum_err']:+.4f}%"]
        else:
            row += ["-", "-", "-"]
        row.append("ACCEPT" if r["accepted"] else "reject")
        rows.append(row)
    w = [max([len(h)] + [len(x[i]) for x in rows]) for i, h in enumerate(heads)]
    out = ["  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(heads))]
    out += ["  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(row)) for row in rows]
    return "\n".join(out)


def no_level_hint(records, selections=False):
    reasons = " ".join(x for r in records for x in r["reasons"])
    tips = []
    if re.search(r"part \d+: empty", reasons):
        tips.append("a smaller --dust (a part is so small that the whole loses it)")
    if re.search(r"\b(sum|void|overlap)\b|part \d+: \d+ pieces", reasons):
        tips.append("a finer grid (--grid 0.25, slower): at fine molmap resolutions the colour "
                    "interface needs more voxels")
    if "whole:" in reasons:
        tips.append("--level-guess near a clean row, or a finer --level-step 0.001 (the whole "
                    "must have one piece per separate molecule)")
    if re.search(r"[SB]\d+: \d+pc", reasons):
        tips.append("--allow-multi-shell if a part is genuinely in several pieces")
    tips.append("a different part order (the order of the --part values)" if selections
                else "a different part order (--parts)")
    return "no contour level passed every check.  Try " + "; or ".join(tips) + "."


def software_versions():
    """Python's and the mesh libraries' versions: the same STLs are only
    promised for the same ones (and the same ChimeraX)."""
    from importlib import metadata
    found = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "trimesh", "manifold3d"):
        try:
            found[name] = metadata.version(name)
        except Exception:
            found[name] = getattr(sys.modules.get(name), "__version__", "?")
    return found


def _tool_logdir(d):
    return d.is_dir() and any(d.glob("params_*.json")) and any(d.glob("chimerax_job_*.py"))


def previous_outputs(out, tag, pdb, names=()):
    """This tool's earlier outputs for this tag in `out` -- nothing else: the
    files listed by the previous report, the exact output-name patterns, the
    `names` this run is about to write (a part named by --part has no fixed
    pattern), this tag's log folders, and (v1.1 layout) a work/ folder
    holding this tool's jobs for the same structure."""
    listed = set()
    rep = out / f"{tag}_report.json"
    if rep.is_file():
        try:
            listed = {Path(f).name for f in json.loads(rep.read_text()).get("files", [])
                      if Path(f).parent == out}
        except Exception:
            listed = set()
    listed |= set(names)
    pat = re.compile(rf"^{re.escape(tag)}_(p\d+_chains?[^_/]+\.stl|whole\.stl|"
                     rf"preview\.(png|glb)|report\.(txt|json))$")
    items = [p for p in sorted(out.iterdir()) if p.is_file() and (p.name in listed or pat.match(p.name))]
    items += [out / d for d in (f"{tag}_work", f"{tag}_work_failed") if _tool_logdir(out / d)]
    for d in ("work", "work_failed"):
        q = out / d
        if _tool_logdir(q):
            try:
                if any(json.loads(f.read_text()).get("pdb") == str(pdb) for f in q.glob("params_*.json")):
                    items.append(q)
            except Exception:
                pass
    return items


def aside_name(out, stamp):
    dest, k = out / f"_previous_{stamp}", 1
    while dest.exists():
        dest, k = out / f"_previous_{stamp}_{k}", k + 1
    return dest


def save_work(work, dest, keep_scan, stamp):
    """Copy the ChimeraX jobs/params/logs, re-pointed at their new home so a
    job can be re-run from there.  Only a folder this tool made is replaced."""
    if dest.exists():
        if _tool_logdir(dest):
            shutil.rmtree(dest)
        else:
            dest = dest.with_name(f"{dest.name}_{stamp}")
    dest.mkdir(parents=True, exist_ok=True)
    for f in work.glob("*"):
        if f.is_file():
            if f.suffix in (".py", ".json"):
                (dest / f.name).write_text(f.read_text().replace(str(work), str(dest)))
            else:
                shutil.copy2(f, dest / f.name)
        elif keep_scan and f.is_dir():
            shutil.copytree(f, dest / f.name, dirs_exist_ok=True)
    return dest


LAST_LOG_DIR = None


def pipeline(a):
    global LAST_LOG_DIR
    t_start = time.time()
    _import_mesh_libs()
    given = Path(a.structure).expanduser().absolute()      # names follow the path as given
    pdb = given.resolve()
    if not pdb.exists():
        raise SplitError(f"structure file not found: {given}")
    exclude_solvent = not a.keep_solvent
    info = read_structure(pdb, exclude_solvent)
    all_ids = [c["id"] for c in info["chains"]]
    if not all_ids:
        raise SplitError(f"no atoms found in {pdb.name}")
    available = [c for c in all_ids if c.strip()]
    notes = []
    selections = None
    if getattr(a, "part", None):
        if a.parts or a.chains:
            raise SplitError("give the parts either as chains (--parts / --chains) or as atom "
                             "selections (--part), not both")
        selections = parse_selection_parts(a.part, getattr(a, "selection_syntax", "chimerax"))
        parts = [[f"P{i}"] for i in range(1, len(selections) + 1)]   # one id per part
    elif a.parts:
        parts = parse_parts(a.parts, available)
    elif a.chains:
        parts = parse_parts([c for c in re.split(r"[,\s]+", a.chains) if c], available)
    else:
        if len(available) < 2:
            raise SplitError("need at least two chains with an ID to split"
                             + (" (atoms with a blank chain ID cannot be selected)"
                                if len(available) < len(all_ids) else ""))
        parts = parse_parts(available, available)
    if selections is None:
        blank_atoms = sum(c["atoms"] for c in info["chains"] if not c["id"].strip())
        if blank_atoms:
            notes.append(f"{blank_atoms} atom(s) with a BLANK chain ID are left out "
                         f"(ChimeraX cannot select a blank chain)")
        used = {c for p in parts for c in p}
        left = [c for c in available if c not in used]
        if left:
            notes.append(f"chain(s) {' '.join(left)} are in no part and are LEFT OUT of the print")
    N = len(parts)
    if N > AMS_SLOTS:
        notes.append(f"{N} parts need {N} filaments -- more than one {AMS_SLOTS}-slot AMS")
    tag = safe_tag(a.tag) if a.tag else default_tag(given, a.resolution)
    out = Path(a.out).expanduser().resolve() if a.out else Path.cwd() / tag
    if out.exists() and not out.is_dir():
        raise SplitError(f"output path {out} exists and is not a folder")
    exe = find_chimerax(a.chimerax)
    chimera_exe = (find_chimera(getattr(a, "chimera", None))
                   if selections and selections[0]["syntax"] == "chimera" else None)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SplitError(f"cannot create the output folder {out}: {e}")
    res, grid, dust = float(a.resolution), float(a.grid), float(a.dust)
    strict = not a.allow_multi_shell
    try:                                    # the preview colours; the STLs carry none
        palette = check_palette(getattr(a, "palette", None))
    except ValueError as e:
        raise SplitError(str(e))
    pal = part_colors(palette)
    doc_point = abs(res - DOC_RESOLUTION) < 1e-9 and abs(grid - DOC_GRID) < 1e-9
    k0 = DOC_K * (res / DOC_RESOLUTION) * (DOC_GRID / grid)
    k_list = parse_k_list(a.k_list) if a.k_list else None
    blobs = bool(getattr(a, "blobs", False))
    if getattr(a, "blob_skin", None) is not None and not blobs:
        notes.append("--blob-skin is used only with --blobs")
    a.blob_skin = DEFAULT_BLOB_SKIN if getattr(a, "blob_skin", None) is None else float(a.blob_skin)
    if blobs:                               # blob parts are their own molmaps: no K
        k_mode, ks = "none", []
        if a.k is not None or a.k_scan or k_list:
            notes.append("--k, --k-scan and --k-list are not used with --blobs: blob parts need no K")
    elif a.k is not None:
        k_mode, ks = "given", [float(a.k)]
    elif a.k_scan or k_list or not doc_point:
        k_mode, ks = "scanned", (k_list or sorted({round(k0 * f, 2) for f in K_SCAN_FACTORS}))
    else:
        k_mode, ks = "documented", [DOC_K]
    # selection mode learns its atoms from ChimeraX first; _pipeline analyses them
    mol = analyse_molecules(info, parts, res, dust) if selections is None else None
    n_mol = mol["n"] if mol else None
    clusters = mol["clusters"] if mol else [1] * N
    closest = mol["closest"] if mol else None
    gone = collections.OrderedDict()
    for chs, n_at in (mol["removed"] if mol else []):
        g = gone.setdefault("+".join(chs), [0, 0])
        g[0] += 1
        g[1] += n_at
    for chs, (n_m, n_at) in gone.items():
        notes.append(f"{n_m} separate molecule(s) of chain(s) {chs} ({n_at} heavy atoms) "
                     f"{'is' if n_m == 1 else 'are'} smaller than the dust size and will not print")
    dusted_chains = {c for chs, _ in (mol["removed"] if mol else []) for c in chs}
    for i, (p, cl) in enumerate(zip(parts, clusters), 1):
        if cl > 1:
            notes.append(f"P{i} ({'+'.join(p)}) has {cl} separate atom clusters (a chain break?) -- "
                         f"it may print as up to {cl} pieces")
    sliver = SLIVER_VOXELS * grid ** 3
    for d in (f"{tag}_work", f"{tag}_work_failed"):
        if (out / d).exists() and not _tool_logdir(out / d):
            raise SplitError(f"{out / d} exists and was not made by this tool -- rename it, or use "
                             f"another --tag / --out")
    if (out / f".{tag}_incoming").exists() and not (out / f".{tag}_incoming").is_dir():
        raise SplitError(f"remove the file {out / f'.{tag}_incoming'} (the tool stages its outputs there)")
    try:
        with tempfile.NamedTemporaryFile(dir=out):
            pass
    except OSError as e:
        raise SplitError(f"cannot write into the output folder {out}: {e}")

    log(f"multicolor_split {__version__}")
    log(f"structure  {given}" + (f"  (-> {pdb})" if pdb != given else ""))
    log(f"           {info['fmt']}, {info['n_models']} model(s); chains "
        + ", ".join(f"{c['id'] if c['id'].strip() else repr(c['id'])} ({c['atoms']} atoms, "
                    f"{c['residues']} res)" for c in info["chains"]))
    if selections is None:
        log(f"parts      " + "  ".join(f"P{i}={'+'.join(p)} ({pal[(i - 1) % len(pal)][0]})"
                                     for i, p in enumerate(parts, 1)))
    else:
        log("parts      " + "  ".join(f"P{i}={s['label']} ({pal[(i - 1) % len(pal)][0]})"
                                     for i, s in enumerate(selections, 1)))
        for i, s in enumerate(selections, 1):
            log(f"           P{i} = {s['spec']}")
        log(f"selections {'Chimera' if chimera_exe else 'ChimeraX'} convention"
            + (f", read by Chimera ({chimera_exe})" if chimera_exe else "")
            + ("; atoms two parts share go to the earlier part"
               if getattr(a, "overlap", "first") == "first" else "; atoms shared by two parts are an error"))
    if palette != "default":
        log(f"palette    {palette}")
    if n_mol:
        log(f"molecules  {n_mol} separate piece(s) expected (heavy atoms within {CONTACT_CUTOFF:g} A "
            f"form one)" + (f"; closest approach between separate molecules {closest:.2f} A"
                            if closest else ""))
    for n_ in notes:
        log(f"note       {n_}")
    log(f"tag        {tag}")
    log(f"output     {out}")
    log(f"chimerax   {exe}")
    log(f"molmap     resolution {res:g} A, gridSpacing {grid:g} A, dust {dust:g} A, "
        f"surfaces at voxel step 1")
    if grid > DOC_GRID + 1e-9:
        log(f"WARNING    gridSpacing {grid:g} > {DOC_GRID:g} A: the colour interface cannot "
            f"be resolved finer than one voxel")

    base = dict(pdb=str(pdb), parts=parts, resolution=res, grid=grid, dust=dust,
                exclude_solvent=exclude_solvent, exclude_hydrogens=bool(a.no_hydrogens),
                level_guess=a.level_guess, span=float(a.span), step=float(a.step),
                levels=None, ks=None, export_ref=True, guess_only=False,
                blobs=blobs, blob_skin=a.blob_skin)
    work = Path(tempfile.mkdtemp(prefix="multicolor_"))
    state = dict(retired=False, stamp=time.strftime("%Y%m%d-%H%M%S"), stage=None)
    ctx = dict(pdb=pdb, given=given, info=info, parts=parts, N=N, tag=tag, out=out, exe=exe,
               res=res, grid=grid, dust=dust, strict=strict, work=work, base=base,
               k_mode=k_mode, ks=ks, k0=k0, notes=notes, t_start=t_start, state=state,
               n_mol=n_mol, mol=mol, clusters=clusters, closest=closest, sliver=sliver,
               dusted_chains=dusted_chains, palette=palette, pal=pal,
               selections=selections, chimera_exe=chimera_exe, blobs=blobs)
    try:
        return _pipeline(a, ctx)
    finally:
        if state["stage"] is not None:              # unfinished: never leave half a set
            shutil.rmtree(state["stage"], ignore_errors=True)
        try:
            LAST_LOG_DIR = save_work(work, out / (f"{tag}_work" if state["retired"]
                                                  else f"{tag}_work_failed"),
                                     a.keep_scan, state["stamp"])
        except Exception as e:                            # never mask the real outcome
            log(f"(could not save the logs: {e})")
        shutil.rmtree(work, ignore_errors=True)


def resolve_selections(a, c):
    """Selection mode: fix every part's atoms once, before any map is made.
    Chimera evaluates Chimera-convention specs; ChimeraX evaluates its own and
    applies the rules shared by both -- solvent and hydrogens excluded as
    asked, an atom two parts select given to the earlier part, rest filling
    in -- and reports each part's atom indices, which every later ChimeraX
    job maps exactly.  The molecule analysis then runs on those atoms."""
    sels, work, notes = c["selections"], c["work"], c["notes"]
    first_note = len(notes)                 # the notes before this step are already logged
    chimera = c.get("chimera_exe")
    log(f"\n[0] selecting each part's atoms ({'Chimera' if chimera else 'ChimeraX'} convention)")
    payload = [dict(label=s["label"], spec=s["spec"], kind=s["kind"], syntax=s["syntax"],
                    hint=convention_hint(s["spec"], s["syntax"]) if s["kind"] == "spec" else "")
               for s in sels]
    if chimera:
        cman = run_chimera(chimera, c["pdb"], [s["spec"] for s in sels if s["kind"] == "spec"], work,
                           [f"P{k} ({s['label']})" for k, s in enumerate(sels, 1) if s["kind"] == "spec"])
        found = iter(cman["keys"])
        for p in payload:
            if p["kind"] == "spec":
                p["keys"] = next(found)
        c["chimera_version"] = cman.get("chimera_version")
    man = run_chimerax(c["exe"], dict(c["base"], mode="select", selections=payload,
                                      overlap=getattr(a, "overlap", "first")), work, "select")
    sel = man["selection"]
    for k, (s, r) in enumerate(zip(sels, sel["parts"]), 1):
        s.update(n_selected=r["selected"], n_atoms=r["n_atoms"], lost=r["lost"],
                 not_found=r["not_found"])
        log(f"    P{k} {s['label']:10s} {r['n_atoms']:7d} atoms" +
            (f"  ({r['selected']} selected; " + ", ".join(
                f"{n} kept by P{int(j) + 1}" for j, n in sorted(r["lost"].items(), key=lambda t: int(t[0])))
             + ")" if r["lost"] else ""))
    for sh in sel["shared"]:
        notes.append(f"P{sh['earlier'] + 1} ({sels[sh['earlier']]['label']}) and P{sh['later'] + 1} "
                     f"({sels[sh['later']]['label']}) both select {sh['count']} atom(s), e.g. "
                     f"{', '.join(sh['examples'][:3])}: the earlier part, P{sh['earlier'] + 1}, keeps them")
    for k, s in enumerate(sels, 1):
        if s.get("not_found"):
            notes.append(f"P{k} ({s['label']}): {s['not_found']} atom(s) Chimera selected are not in "
                         f"the structure ChimeraX read and are left out")
    if chimera and sel.get("names_twice"):
        notes.append(f"{sel['names_twice']} chain:residue@atom name(s) occur more than once in the "
                     f"structure, e.g. {', '.join(sel['names_twice_examples'][:3])}: a part whose "
                     f"Chimera selection names one takes every atom of that name")
    if sel["unassigned"]:
        notes.append(f"{sel['unassigned']} atom(s) are in no part and are LEFT OUT of the print, e.g. "
                     f"{', '.join(sel['unassigned_examples'][:3])} -- add a rest part to take them")
    c["base"] = dict(c["base"], part_atoms=[r["indices"] for r in sel["parts"]],
                     part_fingerprints=[r["fingerprint"] for r in sel["parts"]])
    at = sel["atoms"]
    owner = np.asarray(at["part"], dtype=np.int64)
    xyz = np.asarray(at["coords"], dtype=np.float64).reshape(-1, 3)
    is_h = np.asarray(at["is_h"], dtype=bool)
    ids = [f"P{k}" for k in range(1, len(sels) + 1)]
    info = dict(chains=[dict(id=ids[k], atoms=int((owner == k).sum()), xyz=xyz[owner == k],
                             is_h=is_h[owner == k]) for k in range(len(sels))])
    mol = analyse_molecules(info, c["parts"], c["res"], c["dust"])
    c.update(mol=mol, n_mol=mol["n"] if mol else None, clusters=mol["clusters"] if mol else [1] * len(sels),
             closest=mol["closest"] if mol else None,
             dusted_chains={x for chs, _ in (mol["removed"] if mol else []) for x in chs})
    gone = collections.OrderedDict()
    for chs, n_at in (mol["removed"] if mol else []):
        g = gone.setdefault("+".join(chs), [0, 0])
        g[0] += 1
        g[1] += n_at
    for chs, (n_m, n_at) in gone.items():
        notes.append(f"{n_m} separate molecule(s) of part(s) {chs} ({n_at} heavy atoms) "
                     f"{'is' if n_m == 1 else 'are'} smaller than the dust size and will not print")
    for k, cl in enumerate(c["clusters"], 1):
        if cl > 1:
            notes.append(f"P{k} ({sels[k - 1]['label']}) has {cl} separate atom clusters -- "
                         f"it may print as up to {cl} pieces")
    if c["n_mol"]:
        log(f"molecules  {c['n_mol']} separate piece(s) expected (heavy atoms within "
            f"{CONTACT_CUTOFF:g} A form one)" + (f"; closest approach between separate molecules "
                                                 f"{c['closest']:.2f} A" if c["closest"] else ""))
    for n_ in notes[first_note:]:
        log(f"note       {n_}")


def _pipeline(a, c):
    if c.get("selections"):
        resolve_selections(a, c)
    pdb, info, parts, N, tag, out, exe = (c[k] for k in ("pdb", "info", "parts", "N", "tag", "out", "exe"))
    res, grid, dust, strict, work, base = (c[k] for k in ("res", "grid", "dust", "strict", "work", "base"))
    k_mode, ks, k0, notes, t_start, state = (c[k] for k in ("k_mode", "ks", "k0", "notes", "t_start", "state"))
    n_mol, mol, clusters, closest, sliver = (c[k] for k in ("n_mol", "mol", "clusters", "closest", "sliver"))
    palette, pal = c["palette"], c["pal"]
    guess = a.level_guess
    k_records = []
    cx_version = None

    # ---- phase 1 (only when K has to be found): fields at the guess level only
    blobs = c.get("blobs")
    if blobs:
        K = None
        log(f"\n[1] blob parts: {blob_names(N)} the whole minus "
            f"{'it' if N == 2 else 'them'} -- no K" + (f"; a skin of P{N} estimated thinner than "
                                                        f"{a.blob_skin:g} A over a blob goes to the "
                                                        f"blob where its density exceeds the other "
                                                        f"atoms' together" if a.blob_skin > 0 else ""))
    elif k_mode == "scanned":
        log(f"\n[1] K scan at the level guess: K = {', '.join(f'{k:g}' for k in ks)}"
            f"   (K0 = {k0:.3g}, the per-voxel equivalent of K=16 @ res 4 / grid 0.5)")
        lv = [a.level] if a.level is not None else None
        man = run_chimerax(exe, dict(base, levels=lv, guess_only=True, ks=ks), work, "kscan")
        cx_version = man.get("chimerax_version")
        if a.level is not None and a.level >= man["T_max"]:
            raise SplitError(f"pinned level {a.level:.10g} is above the map maximum "
                             f"({man['T_max']:.4g}): there is no surface.  Drop --level to scan.")
        guess = man["guess"] if a.level is None else a.level
        Lk = man["levels"][0]
        for K in ks:
            r = evaluate(man["files"], Lk, K, N, strict, None, clusters, sliver, construct=False)
            r["field_reasons"] = [x for x in r["reasons"] if not x.startswith("whole")]
            k_records.append(r)
            whole_bad = len(r["field_reasons"]) < len(r["reasons"])
            log(f"    K={K:<6g} fields: " + ("clean" if not r["field_reasons"]
                                              else "; ".join(r["field_reasons"]))
                + ("   (the whole itself is not clean at this level)" if whole_bad else ""))
        good = [r["k"] for r in k_records if not r["field_reasons"]]
        if good:
            # K0 is the per-voxel equivalent of the validated K = 16; take it
            # if its fields are clean, else the nearest clean candidate (lower
            # wins a tie: tearing is the high-K failure).  Every extra unit of
            # K only widens the clearance the complement inherits.
            K = min(good, key=lambda k: (abs(k - k0), k))
            log(f"    -> K = {K:g} (clean candidate nearest K0 = {k0:.3g})")
        else:
            K = round(k0, 2)
            log(f"    -> no K gave clean fields at this level; using K0 = {K:g} and "
                f"letting the level scan decide")
    else:
        K = ks[0]
        log(f"\n[1] K = {K:g} ({'given' if k_mode == 'given' else 'documented value for res 4 / grid 0.5'})")
    k_text = "blob parts" if K is None else f"K = {K:g}"

    # ---- phase 2: level scan (or the pinned level)
    pinned = a.level is not None
    if pinned:
        log(f"\n[2] pinned level {a.level:.10g} -- checking it")
        man = run_chimerax(exe, dict(base, levels=[a.level], ks=[K]), work, "level")
        guess = a.level
    else:
        g = f"{guess:.4f}" if guess is not None else "molmap auto level"
        log(f"\n[2] level scan: {g} +/- {a.span:g} in {a.step:g} steps, {k_text}")
        man = run_chimerax(exe, dict(base, level_guess=guess, ks=[K]), work, "level")
        guess = man["guess"]
    cx_version = man.get("chimerax_version") or cx_version
    auto_level = man["auto_level"]
    steps = "/".join("x".join(map(str, s)) for s in man.get("export_steps", [])) or "?"
    log(f"    molmap auto level {auto_level:.5f}; additivity |sum(parts) - whole| max "
        f"{man['additivity_max_abs']:.1e}; every surface exported at step {steps}")
    files, vox = man["files"], dict(man["voxel_counts"])
    levels = list(man["levels"])
    protected = list(man.get("dust_protected", []))
    for p_ in protected:
        what, at = p_.rsplit("@", 1)
        if "part piece" in what:
            log(f"    note: at {float(at):.4f} dust would have removed a piece of the whole that "
                f"carries a part -- kept")

    held = {}                               # the best candidate's built objects so far

    def scan(levels):
        counts = whole_counts(files, levels)
        n_whole, basis = expected_pieces(counts, guess, n_mol, pinned)
        def check(L):
            r = evaluate(files, L, K, N, strict, n_whole, clusters, sliver, force=pinned)
            built = r.pop("_built", None)
            if built is not None and (r["accepted"] or pinned):
                rank = (round(abs(L - guess), 9), r.get("pinch_total", 0), L)   # = choose()
                if "rank" not in held or rank < held["rank"]:
                    held.update(rank=rank, level=L, built=built)
            log(f"    level {L:.4f}: " + ("ACCEPT" if r["accepted"] else "reject")
                + ("" if r["accepted"] else "  (" + "; ".join(r["reasons"]) + ")")
                + (f"  [{r['pinch_total']} pinch edges after welding]" if "parts" in r else ""))
            return r

        if blobs:                           # nearest the guess first: the nearest clean level wins
            recs, skipped = nearest_first(levels, guess, check)
            if skipped:
                log(f"    ({skipped} level(s) further from the guess not checked: the nearest clean "
                    f"level wins)")
        else:
            recs = [check(L) for L in levels]
        return recs, n_whole, basis

    records, n_whole, basis = scan(levels)
    log(f"    the whole must be {n_whole} piece(s): {basis}" if n_whole else
        "    the whole is not clean at any level")
    best = choose(records, guess)
    if best is None and not pinned and not a.no_widen:
        n2 = int(math.floor(2 * a.span / a.step + 1e-9))
        tried = {round(L, 6) for L in levels}
        more = [round(guess + i * a.step, 6) for i in range(-n2, n2 + 1)]
        more = [L for L in more if L > 0 and round(L, 6) not in tried]
        if more:
            log(f"\n[2b] nothing accepted; widening the scan to +/- {2 * a.span:g} "
                f"({len(more)} more level(s))")
            man2 = run_chimerax(exe, dict(base, levels=more, ks=[K]), work, "widen")
            files["ref"].update(man2["files"]["ref"])
            files["fields"][field_key(K)].update(man2["files"]["fields"][field_key(K)])
            vox.update(man2["voxel_counts"])
            protected += man2.get("dust_protected", [])
            levels = sorted(levels + list(man2["levels"]))
            log("    re-checking the levels against the wider scan, nearest the guess first:"
                if blobs else "    re-checking every level against the wider scan:")
            records, n_whole, basis = scan(levels)
            best = choose(records, guess)
    pinned_fail = False
    if best is None:
        if pinned:
            r = records[0]
            if r.get("parts"):
                best, pinned_fail = r, True
                log("\n    the pinned level does NOT pass every check -- exporting anyway, "
                    "flagged FAIL in the report")
            elif r["whole"]["faces"] == 0:
                raise SplitError(f"pinned level {a.level:.10g} is above the map maximum "
                                 f"({man['T_max']:.4g}): there is no surface.  Drop --level to scan.")
            else:
                raise SplitError(f"pinned level {a.level:.10g} cannot be built "
                                 f"({'; '.join(r['reasons'])}).  Drop --level to let the tool "
                                 f"scan for a clean level.")
        else:
            log("\n" + scan_table(records, N, n_whole, blobs=bool(blobs)))
            raise SplitError(no_level_hint(records, selections=bool(c.get("selections"))))
    L = best["level"]
    log(f"\n    -> level {L:.4f}" + ("  (pinned)" if pinned else
                                    f"  (nearest clean level to the guess {guess:.4f})"))

    # ---- phase 3: build, validate and write everything to a staging folder;
    # it replaces the previous set only once the report is complete
    log("\n[3] building the parts at the chosen level")
    ref, (refV, refF, ref_deg), S, parts_m, pinfo = build_final(
        files, L, K, N, sliver, built=held.get("built") if held.get("level") == L else None)
    held.clear()
    Vw = ref.volume()
    pieces = whole_solids(ref)
    sels = c.get("selections")
    plabels = [x["label"] for x in sels] if sels else [part_label(p) for p in parts]
    labels = [f"P{i}_{lab}" for i, lab in enumerate(plabels, 1)]
    colors = [pal[i % len(pal)][1] for i in range(N)]
    A = placement(refV, a.lay_flat, float(a.scale))
    names = [f"{tag}_p{i}_{lab}.stl" for i, lab in enumerate(plabels, 1)]
    whole_name = f"{tag}_whole.stl"
    stage = out / f".{tag}_incoming"
    if stage.exists():
        shutil.rmtree(stage)                    # this tool's own leftover staging folder
    stage.mkdir()
    state["stage"] = stage

    pvol = [P.volume() for P in parts_m]
    empty = [v < sliver for v in pvol]
    part_arrays = [arrays(P) for P in parts_m]
    ex_meshes, ex_deg, written, separated = [], [], [], []
    for (V, F), nm, is_empty in zip(part_arrays, names, empty):
        Vt = apply(A, V)
        if not getattr(a, "keep_pinch_edges", False):
            # vertex copies at one point stay apart in the file, so a slicer
            # that welds equal coordinates finds no non-manifold edge
            Vt, moved, step, _left = unpinch(Vt, F)
            separated.append((moved, step))
        else:
            separated.append((0, 0.0))
        Vx, Fx, nd = weld(Vt, F)
        ex_meshes.append((Vx, Fx))
        ex_deg.append(nd)
        if not is_empty:
            export_stl(Vx, Fx, stage / nm)
            written.append(nm)
    Wx, WFx, wd = weld(apply(A, refV), refF)
    export_stl(Wx, WFx, stage / whole_name)

    # re-read what was written: the files, not the objects, are what gets printed
    reread = [mesh_stats(*read_stl(stage / nm)) if not e else mesh_stats(np.zeros((0, 3)), np.zeros((0, 3), int))
              for nm, e in zip(names, empty)]
    reread_whole = mesh_stats(*read_stl(stage / whole_name))

    # field-built solids (each part from its own field): the two-material /
    # flexible-workflow numbers, and the void a naive field split would leave
    if S is not None:
        S_union = mf.Manifold.batch_boolean(S, mf.OpType.Add)
        S_sum = sum(x.volume() for x in S)
        S_ov = max((max(0.0, (S[i] ^ S[j]).volume()) for i in range(N) for j in range(i + 1, N)),
                   default=0.0)
        gap_vol = max(0.0, Vw - S_union.volume())
    iface, shown = [], []
    for V, F in part_arrays:
        if len(F) == 0:
            iface.append(0.0)
            shown.append(0.0)
            continue
        on = surface_is_on(V, refV)[F].all(axis=1)
        areas = tri_areas(V, F)
        iface.append(float(areas[~on].sum()))
        shown.append(float(areas[on].sum()))
    iface_total = sum(iface) / 2.0
    gap_t = gap_vol / iface_total if (S is not None and iface_total > 1.0) else 0.0

    vc = vox.get(f"{L:.6f}")
    union = mf.Manifold.batch_boolean(parts_m, mf.OpType.Add)
    void = max(0.0, (ref - union).volume())
    # an empty boolean can report -0.0 / -1e-13: clamp numerical zeros
    overlaps = {f"P{i + 1}&P{j + 1}": max(0.0, (parts_m[i] ^ parts_m[j]).volume())
                for i in range(N) for j in range(i + 1, N)}
    sum_err = (sum(pvol) - Vw) / Vw
    part_pieces = [[j + 1 for j, B in enumerate(pieces) if len(pieces) > 1 and
                    (P ^ B).volume() > 1e-6 * B.volume()] for P in parts_m]
    necks = [i for i in range(N) if len(pieces) == 1 and not empty[i] and iface[i] < 2.0]

    ext_pdb = refV.max(0) - refV.min(0)
    ext_out = Wx.max(0) - Wx.min(0)
    flat_h = float(np.ptp(apply(placement(refV, True, float(a.scale)), refV)[:, 2]))

    preview_png, glb = f"{tag}_preview.png", f"{tag}_preview.glb"
    if not a.no_preview:
        try:
            render_preview(ex_meshes, labels, colors, stage / preview_png,
                           f"{tag}  --  molmap {res:g} A, level {L:.4f}, "
                           + ("blob parts" if K is None else f"K {K:g}"))
        except Exception as e:
            log(f"    (preview PNG skipped: {e})")
        try:
            export_glb(ex_meshes, labels, colors, stage / glb)
        except Exception as e:
            log(f"    (preview GLB skipped: {e})")

    # ---- verdicts
    checks = []

    def check(name, ok, detail, soft=False):
        checks.append(dict(name=name, status="PASS" if ok else ("WARN" if soft else "FAIL"),
                           detail=detail))

    check("parts sum to the whole", abs(sum_err) <= TOL_REL, pct(sum_err))
    check("no overlap between parts", max(overlaps.values(), default=0) <= TOL_REL * Vw,
          f"max {max(overlaps.values(), default=0):.4f} A^3")
    check("no void between parts", void <= TOL_REL * Vw, f"{void:.4f} A^3")
    for i, (P, rr) in enumerate(zip(parts_m, reread), 1):
        bi = best["parts"][i - 1]
        fi = pinfo[i - 1]
        exp = bi["expected_bodies"]
        if empty[i - 1]:
            check(f"P{i} is not empty", False, f"volume {pvol[i - 1]:.3g} A^3 -- no STL written")
            continue
        detail = f"{fi['bodies']} piece(s)" + (f", {fi['cavities']} cavity(ies)" if fi["cavities"] else "") \
            + (f", genus {P.genus()}" if fi["bodies"] == 1 and not fi["cavities"] else "")
        frag = []
        if fi["received"]:
            frag.append(f"took {fi['received']} fragment(s) ({fi['received_volume']:.2g} A^3) from neighbours")
        if fi["slivers"]:
            frag.append(f"{fi['slivers']} of its own handed on")
        if fi["dropped_volume"]:
            frag.append(f"{fi['dropped_volume']:.2g} A^3 dropped (touched nothing)")
        if not strict and fi["bodies"] != exp:  # the count is only reported, not required
            detail += (f"; its atoms form {exp} cluster(s), and --allow-multi-shell allows "
                       f"any number of pieces")
        check(f"P{i} boolean result: manifold" + (f", {exp} piece(s)" if strict else ""),
              P.status() == mf.Error.NoError and (fi["bodies"] == exp or not strict),
              detail + ("; " + "; ".join(frag) if frag else ""))
        check(f"P{i} file: no open edges", rr["open_edges"] == 0, f"{rr['open_edges']} open edges")
        moved, step = separated[i - 1]
        check(f"P{i} file: pinch edges",
              rr["pinch_edges"] <= max(PINCH_WARN, PINCH_WARN_REL * rr["faces"]),
              f"{rr['pinch_edges']} after welding" +
              (f" ({moved} vertex copies at pinch points each moved {step:.2g} mm into its own "
               f"solid, up to {2 * step:.2g} mm apart, before writing)" if moved else "") +
              (" -- slicers such as Bambu Studio report these as non-manifold edges; "
               "leave out --keep-pinch-edges to separate them" if rr["pinch_edges"] else ""),
              soft=True)
    check("whole file clean", reread_whole["clean"] and
          (n_whole is None or reread_whole["outer"] == n_whole),
          f"{reread_whole['outer']} piece(s)" + (f", {reread_whole['cavities']} cavity(ies)"
                                                  if reread_whole["cavities"] else "")
          + f", {reread_whole['pinch_edges']} NM edges, {ref_deg + wd} zero-area triangle(s) dropped")
    if len(pieces) > 1:
        check("the whole is one piece", False,
              f"{len(pieces)} separate pieces: they print as {len(pieces)} loose (or "
              f"interlocked) objects", soft=True)
    if n_mol and len(pieces) != n_mol:
        check("one piece per molecule", False,
              f"{n_mol} separate molecule(s) but the whole is {len(pieces)} piece(s) at this level "
              f"-- " + ("molecules are joined by the molmap" if len(pieces) < n_mol
                        else "a molecule is split by the contour"), soft=True)
    if vc and sum(vc) and not blobs:
        for i in range(N):
            share, dens = 100 * pvol[i] / Vw, 100 * vc[i] / sum(vc)
            # (a part that owns molecules dust removes cannot be compared: its
            # density share counts them)
            if not empty[i] and abs(share - dens) > max(2.0, 0.1 * dens) and \
                    not (set(parts[i]) & c["dusted_chains"]):
                check(f"P{i + 1} volume matches its density share", False,
                      f"{share:.2f} % of the whole vs {dens:.2f} % density-dominant voxels -- look at "
                      f"the preview: territory may have gone to another part", soft=True)
    for i in necks:
        check(f"P{i + 1} is well attached", False,
              f"it touches the rest over only {iface[i]:.2f} A^2 -- it will likely come apart",
              soft=True)
    check("molmap additivity", man["additivity_max_abs"] < 1e-4 * man["T_max"],
          f"{man['additivity_max_abs']:.1e}")
    if pinned_fail:
        check("pinned level passes the scan criteria", False, "; ".join(best["reasons"]))
    n_fail = sum(x["status"] == "FAIL" for x in checks)
    n_warn = sum(x["status"] == "WARN" for x in checks)

    prev_items = previous_outputs(out, tag, pdb, names)
    prev = aside_name(out, state["stamp"]) if prev_items else None

    # ---- report
    if sels:                                    # selection mode: the parts as given
        how = [f"--part={selection_part_arg(x)}" for x in sels]
        if c.get("chimera_exe"):
            how += ["--selection-syntax=chimera", f"--chimera={c['chimera_exe']}"]
        if getattr(a, "overlap", "first") != "first":
            how.append(f"--overlap={a.overlap}")
    else:
        how = ["--parts"] + [",".join(p) for p in parts]
    repro = self_command() + [str(c["given"])] + how + \
            [f"--resolution={res:.12g}", f"--grid={grid:.12g}"] + \
            (["--blobs", f"--blob-skin={a.blob_skin:.12g}"] if blobs else [f"--k={K:.12g}"]) + \
            [f"--level={L:.12g}", f"--dust={dust:.12g}", f"--out={out}", f"--tag={tag}",
             f"--chimerax={exe}"]
    if a.scale != 1.0:
        repro.append(f"--scale={a.scale:.12g}")
    for flag in ("lay_flat", "allow_multi_shell", "keep_solvent", "no_hydrogens", "keep_pinch_edges"):
        if getattr(a, flag):
            repro.append("--" + flag.replace("_", "-"))
    if palette != "default":                    # a default run's report stays as it was
        repro.append(palette_option(palette))
    R = []
    P_ = R.append
    P_(f"multicolor_split {__version__} -- {tag}")
    P_("=" * max(40, len(tag) + 28))
    P_(f"structure   {c['given']}" + (f"  (-> {pdb})" if pdb != c["given"] else ""))
    P_(f"chains      " + ", ".join(f"{x['id'] if x['id'].strip() else repr(x['id'])}: {x['atoms']} "
                                  f"atoms, {x['residues']} residues {x['first']}-{x['last']}"
                                  for x in info["chains"]))
    if sels:
        P_(f"parts       atom selections, {'Chimera' if c.get('chimera_exe') else 'ChimeraX'} "
           f"convention" + (f" (Chimera {c['chimera_version']})" if c.get("chimera_version") else "")
           + ("; atoms two parts select go to the earlier part"
              if getattr(a, "overlap", "first") == "first" else "; shared atoms refused"))
        for i, x in enumerate(sels, 1):
            P_(f"            P{i} = {x['label']}" + (f": {x['spec']}" if x["spec"] != x["label"] else "")
               + f"   ({man['n_atoms_parts'][i - 1]} atoms, "
               f"{pal[(i - 1) % len(pal)][0]})" +
               (f"; {x['n_selected'] - x['n_atoms']} of its {x['n_selected']} selected atoms kept "
                f"by an earlier part" if x.get("lost") else ""))
    else:
        P_(f"parts       " + "   ".join(f"P{i} = {'+'.join(p)} ({man['n_atoms_parts'][i - 1]} atoms, "
                                        f"{pal[(i - 1) % len(pal)][0]})"
                                        for i, p in enumerate(parts, 1)))
    if blobs:
        P_(f"            blob parts: {blob_names(N)} the whole minus {'it' if N == 2 else 'them'}; "
           f"a blob is its atoms' density alone at the contour level, clipped to the whole and "
           f"to the blobs before it")
    else:
        P_(f"            P1 is peeled first; each boundary's field-built face belongs to "
           f"the earlier part")
    if n_mol:
        P_(f"molecules   {n_mol} (heavy atoms within {CONTACT_CUTOFF:g} A form one)"
           + (f"; closest approach between separate molecules {closest:.2f} A" if closest else "")
           + (f"; atom clusters per part {'/'.join(map(str, clusters))}" if max(clusters) > 1 else ""))
    for n_ in notes:
        P_(f"note        {n_}")
    P_("")
    P_("PARAMETERS")
    P_(f"  molmap resolution   {res:g} A")
    P_(f"  gridSpacing         {grid:g} A      (grid {' x '.join(map(str, man['grid']['size']))}; "
       f"surfaces at voxel step {steps} = every grid point, verified)")
    if blobs:
        P_("  blob parts          no K; " + (f"a skin of P{N} estimated thinner than {a.blob_skin:g} A "
                                             f"over a blob goes to the blob where its density "
                                             f"exceeds the other atoms' together" if a.blob_skin > 0 else
                                             "blob skin 0: each blob is exactly its own molmap "
                                             "surface"))
    else:
        P_(f"  K (penalty slope)   {K:g}          ({k_mode}"
           + (f"; K0 = {k0:.3g}" if k_mode == 'scanned' else "") + ")")
    P_(f"  contour level       {L:.6g}      ({'pinned' if pinned else 'scanned'}; "
       f"molmap auto {auto_level:.5f}, guess {guess:.4f})")
    P_(f"  whole must be       {n_whole} piece(s)  ({basis})")
    spared = sum(1 for p_ in protected if "part piece" in p_ and abs(float(p_.rsplit("@", 1)[1]) - L) < 1e-9)
    P_(f"  surface dust        {dust:g} A      (removes specks / fills small cavities; never removes the "
       f"whole's largest piece, one carrying a part's main piece, or a piece of a part's field"
       + (f"; spared {spared} part piece(s) here" if spared else "") + ")")
    P_(f"  solvent / H         {'kept' if a.keep_solvent else 'excluded'} / "
       f"{'excluded' if a.no_hydrogens else 'as in the file'}")
    P_(f"  export              scale {a.scale:g} (1 A -> {a.scale:g} mm), "
       + ("laid flat (PCA; longest axis along X)" if a.lay_flat else "PDB frame"))
    if palette != "default":
        P_(f"  palette             {palette}   (the part colours of the previews and the colour "
           f"names above; the STLs carry no colour)")
    P_(f"  ChimeraX            {exe}" + (f"  ({cx_version})" if cx_version else ""))
    software = software_versions()
    P_(f"  python              {software['python']}  (" + ", ".join(
        f"{name} {software[name]}" for name in ("numpy", "scipy", "trimesh", "manifold3d")) + ")")
    if k_records:
        P_("")
        P_("K SCAN (part fields at the level guess)")
        for r in k_records:
            P_(f"  K {r['k']:<7g} " + ("clean" if not r["field_reasons"]
                                       else "; ".join(r["field_reasons"])))
    P_("")
    P_("LEVEL SCAN  (" + ("B_i = part i's blob, its own molmap" if blobs else
                          "S_i = part i's own field-built solid, H_k = peeling field")
       + "; cells: ok / Nopen / NNM = non-manifold edges / Npc = unexpected number of pieces)")
    P_(scan_table(records, N, n_whole, blobs=bool(blobs)))
    P_("")
    P_(f"VALIDATION at level {L:.6g}, " + ("blob parts" if blobs else f"K {K:g}")
       + "   (volumes in A^3 = mm^3 at 1:1)")
    P_(f"  whole molmap            {Vw:12,.1f}" + (f"   ({len(pieces)} separate pieces)"
                                                    if len(pieces) > 1 else ""))
    for i, (P, lab) in enumerate(zip(parts_m, labels), 1):
        share = pvol[i - 1] / Vw
        exp = (vc[i - 1] / sum(vc)) if vc else float("nan")
        P_(f"  {lab:24s}{pvol[i - 1]:12,.1f}   {100 * share:5.2f} %  (density-dominant voxel "
           f"share before dust {100 * exp:5.2f} %)   triangles {P.num_tri():,}"
           + (f"   on piece(s) {','.join(map(str, part_pieces[i - 1]))}" if len(pieces) > 1 else ""))
    P_(f"  sum of parts vs whole   {pct(sum_err)}")
    for kx, v in overlaps.items():
        P_(f"  real overlap {kx:10s} {v:.4f}")
    P_(f"  void (whole - parts)    {void:.4f}")
    P_("")
    for x in checks:
        P_(f"  {x['status']:4s}  {x['name']:40s} {x['detail']}")
    P_("")
    P_("COLOUR BOUNDARY")
    if iface_total <= 1.0 and len(pieces) > 1:
        P_("  n/a -- the parts do not touch (separate pieces)")
    else:
        P_(f"  interface area          {iface_total:,.1f} A^2")
        if blobs:
            P_("  on the whole's surface  " + "   ".join(f"P{i} {x:,.1f} A^2" for i, x in
                                                        enumerate(shown, 1)))
            if a.blob_skin == 0:
                P_(f"  (skin 0: where P{N}'s film is thinner than 0.0001 A both its faces count, and so "
                   f"does the blob under it, so these can add up to more than the whole's surface)")
            P_(f"  -> each blob's face inside the model is its own molmap surface; P{N} fills "
               f"around it")
        elif S is None:
            P_("  field-built statistics  n/a (a part's own field solid is not a valid manifold)")
        else:
            P_(f"  field-built solids      sum {pct((S_sum - Vw) / Vw)} vs whole, max pairwise "
               f"interference {S_ov:.3f} A^3  (the separate-print / flexible pair)")
            P_(f"  void they would leave   {gap_vol:,.1f} A^3 = {gap_t:.3f} A mean clearance")
            P_(f"  -> the complement fills it; the colour boundaries sit on average ~{gap_t / 2:.2f} A "
               f"(estimate; typically 0.1-0.2 A = {gap_t / 2 * a.scale:.2f} mm printed) off the "
               f"mid-surface, into the earlier part")
    P_("")
    P_("PRINTING")
    P_(f"  model size (PDB frame)  {ext_pdb[0]:.1f} x {ext_pdb[1]:.1f} x {ext_pdb[2]:.1f} A")
    P_(f"  exported size           {ext_out[0]:.1f} x {ext_out[1]:.1f} x {ext_out[2]:.1f} mm   "
       f"height {ext_out[2]:.1f} mm = ~{int(math.ceil(ext_out[2] / LAYER_H))} layers at {LAYER_H} mm")
    if not a.lay_flat and flat_h < 0.8 * ext_out[2]:
        P_(f"  laid flat it would be   {flat_h:.1f} mm tall (~{int(math.ceil(flat_h / LAYER_H))} layers) "
           f"-- purge scales with layer count: re-run with --lay-flat or rotate the WHOLE object")
    P_(f"  filaments               {N}" + (f"  (more than one {AMS_SLOTS}-slot AMS)" if N > AMS_SLOTS else ""))
    P_("  slicer: import all part STLs at once (one object, one sub-model per part, already")
    P_("  registered -- never move them apart); assign one filament per part; check 'First")
    P_("  layer filament sequence'; enable flush into infill/support.")
    if len(pieces) > 1:
        P_(f"  The whole is {len(pieces)} separate pieces: they come off the bed as "
           f"{len(pieces)} loose (or interlocked) objects.")
    elif necks:
        P_("  Some parts are attached only through a tiny contact -- see the WARN lines.")
    else:
        P_("  The parts fuse into one solid -- nothing comes apart.")
    P_("")
    P_("FILES  (only these belong to this run" + (f"; the earlier set was moved to {prev.name}/ "
                                                  f"when this one was published" if prev else "") + ")")
    for nm, e in zip(names, empty):
        P_(f"  {nm}" + ("   (EMPTY -- not written)" if e else ""))
    P_(f"  {whole_name}   (reference -- do not print together with the parts)")
    if not a.no_preview:
        P_(f"  {preview_png}, {glb}   (local previews)")
    P_(f"  {tag}_work/   ChimeraX jobs, parameters and logs of this run")
    P_("")
    P_("REPRODUCE (everything pinned)")
    P_("  " + " ".join(shlex.quote(x) for x in repro))
    P_("  (paste into a terminal to rebuild this set: the same STLs with the same ChimeraX and "
       "libraries; the files now here move into _previous_<timestamp>/ first)")
    P_("")
    verdict = "ALL CHECKS PASS" if not n_fail and not n_warn else \
        (f"{n_fail} FAIL, {n_warn} WARN" if n_fail else f"PASS with {n_warn} warning(s)")
    P_(f"VERDICT: {verdict}   ({time.time() - t_start:.0f} s)")
    text = "\n".join(R)
    (stage / f"{tag}_report.txt").write_text(text + "\n")
    rep = dict(version=__version__, structure=str(c["given"]), structure_resolved=str(pdb),
               tag=tag, parts=parts, labels=labels,
               selections=[dict(label=x["label"], spec=x["spec"], kind=x["kind"], syntax=x["syntax"],
                                selected=x.get("n_selected"), atoms=x.get("n_atoms"),
                                kept_by_earlier={f"P{int(j) + 1}": n for j, n in (x.get("lost") or {}).items()},
                                not_found=x.get("not_found"))
                           for x in sels] if sels else None,
               overlap=getattr(a, "overlap", "first") if sels else None,
               chimera=c.get("chimera_exe"), chimera_version=c.get("chimera_version"),
               colors={l: pal[i % len(pal)][0] for i, l in enumerate(labels)},
               resolution=res, grid=grid, k=K, k_mode=k_mode, level=L, blobs=bool(blobs),
               blob_skin=float(a.blob_skin) if blobs else None, part_surface_shown=shown,
               level_mode="pinned" if pinned else "scanned", auto_level=auto_level,
               guess=guess, dust=dust, dust_protected=protected, scale=a.scale,
               lay_flat=bool(a.lay_flat), transform=A.tolist(), whole_volume=Vw,
               whole_pieces=len(pieces), expected_pieces=n_whole, expected_basis=basis,
               molecules=n_mol, molecules_all=mol["n_all"] if mol else None,
               molecules_removed=mol["removed"] if mol else None, part_clusters=clusters,
               closest_molecule_approach=closest,
               part_volumes=pvol,
               part_pieces=part_pieces, part_interface_area=iface, part_info=pinfo,
               sum_err=sum_err, overlaps=overlaps, void=void, interface_area=iface_total,
               field_gap_volume=gap_vol if S is not None else None,
               field_gap_thickness=gap_t if S is not None else None,
               boundary_offset=gap_t / 2 if S is not None else None,
               checks=checks, verdict=verdict, scan=records, kscan=k_records,
               files=[str(out / nm) for nm in written] + [str(out / whole_name)],
               zero_area_dropped=dict(whole=ref_deg + wd, parts=ex_deg),
               pinch_vertices_separated=[m for m, _step in separated],
               keep_pinch_edges=bool(getattr(a, "keep_pinch_edges", False)),
               reread=reread, reread_whole=reread_whole, reproduce=repro, notes=notes,
               chimerax=exe, chimerax_version=cx_version, software=software, grid_info=man["grid"],
               export_steps=man.get("export_steps"), additivity=man["additivity_max_abs"],
               n_atoms_parts=man["n_atoms_parts"])
    if palette != "default":
        rep["palette"] = palette
    (stage / f"{tag}_report.json").write_text(json.dumps(rep, indent=1, default=float))

    # ---- publish: retire the previous set, then move the new one in
    if prev_items:
        prev.mkdir()
        for p in prev_items:
            shutil.move(str(p), str(prev / p.name))
        log(f"    an earlier run's files were moved to {prev.name}/")
    for f in sorted(stage.iterdir()):
        shutil.move(str(f), str(out / f.name))
    stage.rmdir()
    state["stage"] = None
    state["retired"] = True
    log("\n" + text)
    return EXIT_FAIL if n_fail else EXIT_OK


# ============================================================ GUI help text
# key -> (title, prose, example or None).  Every light-blue ? in the GUI opens
# one of these; a chip for a key that is missing here fails when it is built.
# The Convention ?'s table: what a selection selects, then how ChimeraX and
# how Chimera write it.  Each pair selects the same atoms; a test runs every
# one in both programs on the switchback666 duplex.
CONVENTION_EXAMPLES = (
    ("chain A", "/A", ":.A"),
    ("residues 12-20 of chain A", "/A:12-20", ":12-20.A"),
    ("atom P of residue 12 of A", "/A:12@P", ":12.A@P"),
    ("the P atoms of A and of C", "/A@P | /C@P", ":.A@P | :.C@P"),
    ("residues 12-20 of A and C", "/A,C:12-20", ":12-20.A | :12-20.C"),
    ("every DG residue", ":DG", ":DG"),
    ("all DNA and RNA", "nucleic", "nucleic acid"),
    ("all protein", "protein", "protein"),
    ("ligands", "ligand", "ligand"),
)
# ... and the rows too long for a table row: (meaning, ChimeraX's ways, Chimera's)
CONVENTION_BLOCKS = (
    ("chain A's DNA backbone, its sugar and phosphate atoms", ("/A & backbone",),
     ":.A" + CHIMERA_DNA_BACKBONE),
    ("chain A's bases, all but the backbone", ("/A & ~backbone", "/A & sideonly"),
     ":.A & ~" + CHIMERA_DNA_BACKBONE),
)


def _convention_table():
    """CONVENTION_EXAMPLES and CONVENTION_BLOCKS as the Convention ?'s
    example, the code marked for example_segments()."""
    rows = [f"{'What it selects':<28}{'ChimeraX':<14}Chimera"]
    rows += [f"{what:<28}`{cx}`{' ' * max(1, 14 - len(cx))}`{ch}`"
             for what, cx, ch in CONVENTION_EXAMPLES]
    for what, cxs, ch in CONVENTION_BLOCKS:
        rows += ["", what + ":", " ChimeraX " + "   or   ".join(f"`{cx}`" for cx in cxs),
                 f" Chimera  `{ch}`"]
    return "\n".join(rows)


HELP = {
    "structure": (
        "Structure file",
        "The PDB or mmCIF model whose chains are split into colours. It is "
        "required; there is no default.\n\n"
        "Accepted: .pdb and .ent (PDB), also gzip-compressed (.pdb.gz, "
        ".ent.gz), and .cif and .mmcif (mmCIF), uncompressed only. ChimeraX's "
        "mmCIF reader refuses a compressed file, so for a .cif.gz Detect "
        "chains still lists the chains but Run split stops with an error; "
        "gunzip it first.\n\n"
        "Only the first model counts: the chain list is read from the file's "
        "first model, and when ChimeraX opens several structures the maps are "
        "built from the first one. Waters are left out unless Keep solvent is "
        "ticked. "
        "Output names follow the path as given, so a symbolic link's own name, "
        "not its target's, becomes the default tag.\n\n"
        "Browse… also runs Detect chains and fills Parts; after typing a path, "
        "click Detect chains to list its chains (a structure passed in at "
        "start-up is detected by itself).\n\n"
        "Command line: the first argument, e.g. multicolor_split.py model.pdb "
        "--parts A C; a file written last, after the --parts chains, is "
        "recognised too.",
        "model.pdb      PDB\n"
        "model.ent      PDB\n"
        "model.cif      mmCIF (also .mmcif)\n"
        "model.pdb.gz   gzip-compressed PDB (also .ent.gz)\n"
        "model.cif.gz   not accepted: gunzip it to model.cif first",
    ),
    "chains": (
        "Detect chains and the chain table",
        "Detect chains reads the structure's first model, lists its chains in "
        "the order they appear, and fills Parts with every chain that has an "
        "ID, one colour each, in that order (replacing what Parts held).\n\n"
        "Columns: chain is the chain ID, shown as (blank) when the ID is a "
        "space; atoms counts the chain's atom records, hydrogens included; "
        "residues is the number of residues; numbering is the first and last "
        "residue number as written in the file; residue types lists each "
        "residue name with its count. The line beside the button gives the "
        "number of chains and models and the format read (pdb or mmcif). In "
        "mmCIF the chain ID is auth_asym_id (label_asym_id when there is none).\n\n"
        "Waters (residue names HOH, WAT, DOD, H2O, TIP, TIP3, SOL, T3P, SPC) "
        "are not listed unless Keep solvent is ticked; click Detect chains "
        "again after changing that box.\n\n"
        "Atoms with a blank chain ID cannot be selected in ChimeraX, so they "
        "can never be a part: a run leaves them out of the print and notes how "
        "many atoms that is. A chain that is in no part is left out too, and "
        "named in the log and the report.\n\n"
        "Command line: there is no separate step; a run prints each chain with "
        "its atom and residue counts when it starts.",
        "chain  atoms  residues  numbering  residue types\n"
        "A      342    18        8–25       DC×18\n"
        "C      396    18        9–26       DG×18\n"
        "2 chain(s), 1 model(s), pdb",
    ),
    "parts": (
        "Parts (colour order)",
        "Which chains go into which colour, and in what order, when Parts by "
        "is chains (for parts made of atom selections, see Selections). Each "
        "part becomes one STL and one filament. "
        "Separate parts with spaces (or |) and join chains into one part with "
        "a comma (or +). Chain IDs must match the file exactly, a chain may be "
        "in only one part, and at least two parts are needed.\n\n"
        "Blank means every chain with an ID is its own part, in file order; "
        "Detect chains fills that in. Chains left out of every part are not "
        "printed, and the log and report name them.\n\n"
        "The preview and the report colour the parts in order, in the "
        "Palette's colours: by default blue, orange, green, red, purple, teal, "
        "yellow, brown, then the list repeats (the STLs carry no colour; "
        "filaments are assigned in the slicer). More "
        "than 4 parts is noted, as that needs more than one 4-slot AMS.\n\n"
        "The order matters: parts are peeled off one by one, P1 first, and the "
        "last part is the exact remainder. At each colour boundary the "
        "field-built face belongs to the earlier part, so the boundary sits "
        "about 0.1–0.2 Å into the earlier part; reorder the parts to choose "
        "which side takes it. With Blob parts, every part but the last is "
        "instead its own molmap surface and the last part is the remainder, so "
        "the order decides which parts are blobs.\n\n"
        "Files are named <tag>_p1_chainA.stl, and <tag>_p1_chainsA+B.stl for a "
        "joined part.\n\n"
        "Command line: --parts A C (quote a |, e.g. --parts 'A+B|C', or the "
        "shell reads it as a pipe), or --chains A,C as a shorthand for one "
        "chain per part.",
        "`A C`        two colours: chain A, then chain C\n"
        "`A,B C`      chains A and B share one colour\n"
        "`A+B | C`    the same, written another way\n"
        "`C A`        C peeled first: the boundary sits in C",
    ),
    "parts_by": (
        "Parts by: chains or atom selections",
        "How the parts are given. chains (the default): each part is one or "
        "more whole chains, typed in Parts. atom selections: each part is a "
        "set of atoms written as an atom selection in ChimeraX's or UCSF "
        "Chimera's convention, one line per part in Selections, so a part can "
        "be one chain's sugar-phosphate backbone, say, and one chain's atoms "
        "can go to several parts.\n\n"
        "Everything after the parts is the same either way: the maps, the K "
        "and level scans, the peeling in part order, the checks and the "
        "files. Only the rows of the chosen way are shown; switching keeps "
        "what the other rows hold, and a run uses only the chosen one.\n\n"
        "Command line: --parts A C for chains; --part once per part for atom "
        "selections, e.g. --part 'bbA=/A & backbone' --part rest. The two "
        "cannot be combined.",
        "Parts by chains, typed in Parts:\n"
        "  `A C`\n"
        "  P1 = chain A, P2 = chain C\n"
        "\n"
        "Parts by atom selections, typed in Selections:\n"
        "  `bbA = /A & backbone`\n"
        "  `bbC = /C & backbone`\n"
        "  `rest`\n"
        "  P1 = chain A's backbone, P2 = chain C's backbone,\n"
        "  P3 = every other atom: the bases of both strands",
    ),
    "part": (
        "Selections (one part per line)",
        "One part per line, in colour order: the first line is P1, which is "
        "peeled first. A line is either an atom selection, written in the "
        "chosen Convention, or the word rest.\n\n"
        "A selection may have a name in front, as in \"bbA = /A & backbone\": "
        "the name goes into the file name, <tag>_p1_bbA.stl, and must start "
        "with a letter and use only letters, digits and _ + -. Without a "
        "name, the file name is made from the selection's words, so "
        "\"/A & backbone\" is named A_backbone.\n\n"
        "rest is this tool's word, not a selection: it takes every atom no "
        "other line takes, and may be named too (\"bases = rest\"). Give it at "
        "most once, anywhere in the order. At least two lines are needed, and "
        "a line may not contain ;.\n\n"
        "Selections are made in the structure's first model, after waters "
        "are left out (unless Keep solvent is ticked), and hydrogens too when "
        "Exclude hydrogens is ticked; a line that selects nothing else stops "
        "the run with the reason. An atom two lines select goes to the "
        "earlier part (see Shared atoms). Atoms no line selects, when there "
        "is no rest line, are left out of the print; the log and the report "
        "say how many, with examples.\n\n"
        "How to write a selection in each convention: the Convention ? sets "
        "them side by side. In short, chain A is \"/A\" in ChimeraX and "
        "\":.A\" in Chimera. ChimeraX's keywords, such as \"backbone\", have "
        "no Chimera equivalent; there Backbone per chain writes the atom "
        "names for you. In ChimeraX, \"sidechain\" includes the sugar ring "
        "atoms C1', C2', C3', C4' and O4' that \"backbone\" also has; for "
        "the bases alone write \"sideonly\" or \"~backbone\".\n\n"
        "The log and the report give each part's atom count, and say what the "
        "Shared atoms rule did to it.\n\n"
        "Command line: one --part per line, in order, e.g. --part "
        "'bbA=/A & backbone' --part 'bbC=/C & backbone' --part rest. Quote "
        "each, as the shell reads & | ~ ( ), and use double quotes for a "
        "Chimera line with primes (C1'). The report's REPRODUCE line writes "
        "them as NAME=selection.",
        "Typed in Selections, ChimeraX convention:\n"
        "  `bbA = /A & backbone`\n"
        "  `bbC = /C & backbone`\n"
        "  `rest`\n"
        "The parts, on a duplex of 18-nt strands A and C:\n"
        "  P1 bbA   chain A's backbone              198 atoms\n"
        "  P2 bbC   chain C's backbone              198 atoms\n"
        "  P3 rest  everything else (the bases)     342 atoms\n"
        "\n"
        "The same parts typed in Chimera's convention (for a model\n"
        "without hydrogens):\n"
        f"  `bbA = :.A{CHIMERA_DNA_BACKBONE}`\n"
        f"  `bbC = :.C{CHIMERA_DNA_BACKBONE}`\n"
        "  `rest`\n"
        "\n"
        "In a line, `bbA` is the part's name, `=` joins it to the\n"
        "selection, and `/A & backbone` is the selection;\n"
        "`rest` is this tool's word, not a selection.",
    ),
    "selection_syntax": (
        "Convention (selection syntax)",
        "The rules the Selections are written in, and so which program reads "
        "them. ChimeraX (the default): ChimeraX reads them, the same ChimeraX "
        "that builds the maps. Chimera: UCSF Chimera 1.x reads them, run "
        "without a window for a few seconds before the maps are made, so it "
        "must be installed (see the Chimera field, shown for this "
        "convention).\n\n"
        "Both conventions name atoms with @ (\"@P\"), list with a comma "
        "(\"@P,OP1\"), and combine selections with & (and), | (or) and ~ "
        "(not). They differ in how chains and residues are written and in "
        "their keywords; the table below sets them side by side, and every "
        "row was checked in both programs on a duplex of two 18-nt strands, "
        "A and C, selecting the same atoms in each. In Chimera a comma lists "
        "within one level only: \":12-20.A,C\" does not add chain C, so the "
        "table writes \":12-20.A | :12-20.C\".\n\n"
        "Chimera names the atoms it selects by chain, residue number, "
        "insertion code and atom name; ChimeraX then takes exactly those "
        "atoms, so the same atoms written in either convention give "
        "byte-identical STLs. Atoms Chimera names that ChimeraX does not have "
        "(the two can read a file's chain or residue names differently) are "
        "left out with a note, and a part left with none stops the run. Where "
        "a file has two atoms with the same chain, residue and name, a part "
        "whose Chimera selection names them takes both, also noted.\n\n"
        "The names before =, rest and the Shared atoms rule belong to this "
        "tool and work the same in both conventions.\n\n"
        "Command line: --selection-syntax chimera (default chimerax).",
        _convention_table(),
    ),
    "overlap": (
        "Shared atoms",
        "What happens to an atom that two Selections both select. earlier "
        "part keeps them (the default): it goes to the part listed first; the "
        "log, the report and its JSON say how many atoms each pair shares, "
        "with examples, and how many each later part lost. refuse the run: "
        "the run stops before any map is made and lists the shared atoms, so "
        "each atom may be in at most one selection.\n\n"
        "With the default, order does the work: put the specific part first "
        "and a broad one after it. \"bbA = /A & backbone\" followed by "
        "\"A = /A\" gives chain A's backbone to P1 and the rest of chain A to "
        "P2, without writing \"~backbone\". rest never shares: it takes only "
        "the atoms no line takes. A part whose every atom an earlier part "
        "keeps stops the run.\n\n"
        "A frequent surprise in ChimeraX's convention: \"sidechain\" includes "
        "the sugar ring atoms C1', C2', C3', C4' and O4', which \"backbone\" "
        "includes too, so \"/A & backbone\" and \"/A & sidechain\" share five "
        "atoms per nucleotide; the bases alone are \"sideonly\".\n\n"
        "Command line: --overlap error (default first).",
        "Typed in Selections:\n"
        "  `bbA = /A & backbone`\n"
        "  `scA = /A & sidechain`\n"
        "  `rest`\n"
        "What each part gets, on a duplex of 18-nt strands A and C:\n"
        "  P1 bbA   chain A's backbone                  198 atoms\n"
        "  P2 scA   chain A's sidechain (bases and sugar\n"
        "           rings), less the 90 sugar atoms P1\n"
        "           keeps                               144 atoms\n"
        "  P3 rest  everything else: chain C            396 atoms\n"
        "The note in the log and the report:\n"
        "  P1 (bbA) and P2 (scA) both select 90 atom(s), e.g.\n"
        "  A:8@C4', A:8@O4', A:8@C3': the earlier part, P1, keeps them",
    ),
    "fill_backbone": (
        "Backbone per chain",
        "Fills Selections with one line per chain holding that chain's "
        "backbone, named bb<chain>, in file order, then rest for everything "
        "else: the bases, side chains and ligands. It is the usual start for "
        "giving each strand's backbone its own colour and the bases one more; "
        "edit, reorder or delete lines afterwards. Selections that already "
        "hold other lines are replaced after a question.\n\n"
        "It takes the chains Detect chains found (running it when needed), "
        "and skips a chain with a blank ID or with no nucleotide or "
        "amino-acid residue, which then falls to rest.\n\n"
        "In ChimeraX's convention a line is \"/A & backbone\". Chimera's has "
        "no backbone keyword, so the line lists the atom names ChimeraX's "
        "backbone covers, with hydrogens and other programs' names: for DNA "
        "and RNA P, OP1, OP2, OP3, O1P, O2P, O3P, HP, O5', C5', C4', O4', "
        "C3', O3', C2', O2', C1', H1', H2', H2'', H3', H4', H5', H5'', HO2', "
        "HO3', HO5', H5T and H3T; for a protein N, CA, C, O, OXT, OT1, OT2, "
        "H, HN, HA, HA1, HA2, HA3, HT1, HT2 and HT3, with H1, H2 and H3 only "
        "in a chain with no nucleotide (in a nucleotide those are base "
        "hydrogens). With the usual atom names both select the same atoms; "
        "ChimeraX decides a hydrogen by the atom it is bonded to, so one with "
        "an unusual name can differ.\n\n"
        "Command line: write the --part values; Show command prints the ones "
        "in this window.",
        "For a DNA duplex with chains A and C, the button writes,\n"
        "in ChimeraX's convention:\n"
        "  `bbA = /A & backbone`\n"
        "  `bbC = /C & backbone`\n"
        "  `rest`\n"
        "and in Chimera's convention (a long line wraps here):\n"
        f"  `bbA = :.A@{','.join(NA_BACKBONE_NAMES)}`\n"
        f"  `bbC = :.C@{','.join(NA_BACKBONE_NAMES)}`\n"
        "  `rest`",
    ),
    "palette": (
        "Palette (part colours)",
        "The colours the parts are shown in: the preview PNG, the coloured "
        "meshes of the GLB preview and the swatches beside this menu, and the "
        "colour name each part gets in the log, the report and its JSON. The "
        "STLs carry no colour, so they are the same whatever is chosen; "
        "filaments are assigned in the slicer. P1 takes the first colour, P2 "
        "the second, and so on.\n\n"
        "default is this tool's own colours, unchanged: blue, orange, green, "
        "red, purple, teal, yellow, brown, then the list repeats.\n\n"
        "DiLiuLab is the lab's nine figure colours from gr_colors, in the order "
        "red, blue, magenta, cyan, orange, purple, green, yellow, mint green, "
        "repeating after nine. T100 is the full colour; T80, T60, T40 and T20 "
        "are lighter tints mixed toward white, and T40 and T20 are pale enough "
        "to be hard to see on the preview's white background. The palette's "
        "neutral (black, gray at the lighter tints) is not used: every part is "
        "a colour. The colours are read from assets/diliulab_colors.json.\n\n"
        "A palette other than default is recorded in the report's PARAMETERS "
        "block, in the JSON (palette) and on the REPRODUCE line; default adds "
        "nothing there.\n\n"
        "Command line: --palette DiLiuLab, or --palette DiLiuLab-T80 for a "
        "tint.",
        "default          P1 blue, P2 orange, P3 green ...\n"
        "DiLiuLab T100    P1 red, P2 blue, P3 magenta ...\n"
        "DiLiuLab T80     the same, one step lighter\n"
        "--palette DiLiuLab, from the report:\n"
        "  parts       P1 = A (342 atoms, red)   P2 = C (396 atoms, blue)",
    ),
    "out": (
        "Output folder",
        "The folder the STLs, report, previews and logs are written to; it is "
        "created if needed.\n\n"
        "Blank means ./<tag>: a folder named after the tag, which with a blank "
        "tag is <structure stem>_molmap<resolution>_split, in the current "
        "working directory. That is the folder the tool was started from, not "
        "necessarily the structure's or the script's folder. When Curve It "
        "opens this tool for a structure, it starts it in that structure's "
        "folder; opened with no structure loaded, it starts it in Curve It's "
        "own working folder, or in your home folder when that cannot be "
        "written (an app opened from the Finder runs in /). A relative path "
        "typed here is also taken from the current working directory. The "
        "line under Tag shows the folder the run will use.\n\n"
        "Several runs can share one folder as long as their tags differ: a "
        "re-run moves aside only the files of its own tag, and other files in "
        "the folder are never touched.\n\n"
        "Command line: --out FOLDER.",
        "blank, started in ~/prints, structure duplex.pdb:\n"
        "  ~/prints/duplex_molmap4_split/\n"
        "    duplex_molmap4_split_p1_chainA.stl\n"
        "    duplex_molmap4_split_p2_chainC.stl\n"
        "    duplex_molmap4_split_whole.stl\n"
        "    duplex_molmap4_split_report.txt, .json\n"
        "    duplex_molmap4_split_preview.png, .glb\n"
        "    duplex_molmap4_split_work/\n"
        "~/prints/run2   ->  ~/prints/run2/duplex_molmap4_split_...",
    ),
    "tag": (
        "Tag (file prefix)",
        "The name every output file starts with.\n\n"
        "Blank means <structure stem>_molmap<resolution>_split: the file name "
        "without .gz and its extension, then the resolution without trailing "
        "zeros (4.0 gives molmap4, 3.7 gives molmap3.7). With the output folder "
        "blank, the tag, given or default, also names the folder.\n\n"
        "Characters other than A–Z, a–z, 0–9 and . _ + - are replaced by _ "
        "(a run of them by one _), and every leading or trailing _ is dropped "
        "(a tag with nothing left becomes model).\n\n"
        "Give a tag to keep two runs of one structure side by side, e.g. at "
        "two contour levels: a re-run with the same tag in the same folder "
        "moves the previous set of that tag into _previous_<timestamp>/ when "
        "it puts its own files in place.\n\n"
        "Command line: --tag NAME.",
        "duplex.pdb, resolution 4, tag blank:\n"
        "  duplex_molmap4_split_p1_chainA.stl\n"
        "  duplex_molmap4_split_p2_chainC.stl\n"
        "  duplex_molmap4_split_whole.stl\n"
        "1abc.pdb.gz, resolution 3.7:  1abc_molmap3.7_split\n"
        "tag  my run   ->  my_run_p1_chainA.stl",
    ),
    "chimerax": (
        "ChimeraX",
        "The UCSF ChimeraX program; the tool runs it without a window to build "
        "the molmaps and contour them. Give the executable or, on macOS, the "
        ".app: a path ending in .app is read as <app>/Contents/MacOS/ChimeraX.\n\n"
        "Blank means the $CHIMERAX environment variable if it is set, else a "
        "search: /Applications/ChimeraX*.app, ~/Applications/ChimeraX*.app, "
        "chimerax or ChimeraX on the PATH, /usr/bin/chimerax, "
        "/usr/local/bin/chimerax and "
        "C:\\Program Files\\ChimeraX*\\bin\\ChimeraX-console.exe.\n\n"
        "The window fills this in at start-up with the path it held the last "
        "time Run split started a run (kept in ~/.multicolor_split_gui.json) "
        "if that path still works, or else with $CHIMERAX or what the search "
        "finds, so a remembered path that works wins over $CHIMERAX. A "
        "remembered path that no longer works, e.g. after ChimeraX was moved, "
        "is dropped with a line in the log.\n\n"
        "A path in this field when Run split starts, or in $CHIMERAX, must "
        "work: it is never replaced by a search. Clear the field to have "
        "ChimeraX searched for again.\n\n"
        "Command line: --chimerax PATH.",
        "/Applications/ChimeraX.app\n"
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX\n"
        "/usr/bin/chimerax",
    ),
    "chimera": (
        "Chimera",
        "The UCSF Chimera program, 1.x (not ChimeraX), used only to read "
        "Selections written in Chimera's convention; the maps are always "
        "built by ChimeraX. The row is shown for that convention only. Give "
        "the executable or, on macOS, the .app: a path ending in .app is read "
        "as <app>/Contents/MacOS/chimera. A ChimeraX path is refused.\n\n"
        "Blank means the $CHIMERA environment variable if it is set, else a "
        "search: /Applications/Chimera*.app, ~/Applications/Chimera*.app, "
        "chimera on the PATH, /usr/local/bin/chimera, "
        "/opt/UCSF/Chimera*/bin/chimera and "
        "C:\\Program Files\\Chimera*\\bin\\chimera.exe.\n\n"
        "The window fills it in at start-up with the path it held the last "
        "time Run split started a run (kept in ~/.multicolor_split_gui.json "
        "with the ChimeraX path) if that path still works, or else with "
        "$CHIMERA or what the search finds. A path in this field or in "
        "$CHIMERA must work when a run needs it: it is never replaced by a "
        "search. Chimera is available from "
        "https://www.cgl.ucsf.edu/chimera/download.html.\n\n"
        "Command line: --chimera PATH.",
        "/Applications/Chimera.app\n"
        "/Applications/Chimera.app/Contents/MacOS/chimera\n"
        "/opt/UCSF/Chimera64-1.19/bin/chimera",
    ),
    "resolution": (
        "Resolution (Å)",
        "The molmap resolution: how much the atoms are blurred before the "
        "surface is drawn. Default 4 Å. A smaller value follows the atoms more "
        "closely; a larger one gives a smoother blob.\n\n"
        "4 Å is the default and the safest choice. 3.7 Å is worth trying when "
        "you want more detail: it follows the atoms more closely and usually "
        "still gives a surface without many holes, while a much smaller value "
        "opens more holes and tunnels through the surface.\n\n"
        "4 Å with a grid of 0.5 Å is the documented, validated operating "
        "point, where K = 16 is used directly. At any other resolution, with K "
        "blank, K is rescaled to K0 = 16 × (resolution/4) × (0.5/grid) and "
        "confirmed by a K scan, which adds a ChimeraX pass. The contour level "
        "is scanned around molmap's own level for this resolution.\n\n"
        "The resolution is part of the default tag (molmap4, molmap3.7). When "
        "no level passes because of sum, void, overlap or part-piece failures, "
        "the run suggests a finer grid.\n\n"
        "Command line: --resolution 3.7.",
        "4      K = 16 as documented, no K scan\n"
        "3.7    K0 = 14.8; K scan 7.4, 11.1, 14.8, 18.5, 22.2\n"
        "3      K0 = 12; K scan 6, 9, 12, 15, 18",
    ),
    "grid": (
        "Grid spacing (Å)",
        "The spacing of the molmap grid. Every map, the whole and each part, "
        "is built on this one grid, and every surface is contoured at every "
        "grid point. Default 0.5 Å.\n\n"
        "0.5 Å is what resolves the colour interface: at molmap's own default "
        "grid (resolution/3, 1.33 Å at 4 Å) the boundary cannot be placed "
        "finer than about one voxel, about 1.3 mm at Scale 1. A spacing above "
        "0.5 Å gets a warning in the log. A finer grid such as 0.25 Å has 8 "
        "times the voxels, so the run is slower and the STLs have more "
        "triangles; the run suggests it when no level passes because of sum, "
        "void, overlap or part-piece failures.\n\n"
        "Away from 0.5 Å, with K blank, K is rescaled to K0 = 16 × "
        "(resolution/4) × (0.5/grid) and confirmed by a K scan. The fragment "
        "size follows the grid too: a boolean piece under 4 voxels in volume "
        "(0.5 Å³ at 0.5 Å) is handed to the neighbouring part it shares the "
        "most surface with, and a part under that volume counts as empty.\n\n"
        "Command line: --grid 0.25.",
        "0.5    default; K = 16 at resolution 4\n"
        "0.25   8x the voxels; K0 = 32, K scan 16, 24, 32, 40, 48",
    ),
    "dust": (
        "Dust size (Å)",
        "ChimeraX's surface dust, applied to every contour after it is built: "
        "a separate surface piece no bigger than this across (the longest side "
        "of its bounding box) is removed, and a sealed cavity no bigger than "
        "this is filled. Default 20 Å; 0 removes nothing.\n\n"
        "On the whole it removes loose specks and fills small cavities, but "
        "never removes the whole's largest piece or a piece that carries a "
        "part's main piece. So a small ligand whose chain is a part of its own "
        "survives, but one whose chain is joined to another in one part (A,L) "
        "is removed like any other speck. On the part fields it only fills "
        "small cavities and never removes a piece, so a part's detached bit, "
        "such as the tail after a chain break, keeps its colour wherever the "
        "whole keeps it.\n\n"
        "Separate molecules (heavy atoms within 4 Å form one) whose atoms span "
        "less than the dust size minus the resolution, and that are no part's "
        "main molecule, are named in a note at the start of the run: they "
        "will not print.\n\n"
        "Lower it, or make the chain a part of its own, to keep small separate "
        "pieces such as ions; raise it to clear more specks. When no level "
        "passes because a part comes out empty, the run suggests a smaller "
        "dust.\n\n"
        "Command line: --dust 10.",
        "20   a loose blob up to 20 Å across is removed and\n"
        "     a sealed cavity up to 20 Å across is filled\n"
        "0    nothing is removed, no cavity is filled",
    ),
    "k": (
        "K penalty slope",
        "How steeply a part's ownership field drops where a rival part's "
        "density is higher than its own; it shapes the cut face between two "
        "colours. K is a slope per voxel.\n\n"
        "Blank means automatic. At resolution 4 Å and grid 0.5 Å the "
        "documented K = 16 is used without a scan, unless K candidates are "
        "given or Force a K scan is ticked. (In the sweep K = 16 was validated "
        "on, below about 12 the field-built parts interfered and at 20 or more "
        "marching cubes tore; another model can differ: the switchback "
        "example gives clean fields for every K from 8 to 24.) "
        "Elsewhere K0 = 16 × (resolution/4) × (0.5/grid), and a K scan at the "
        "level guess (or the pinned level) tries K0 × 0.5, 0.75, 1, 1.25 and "
        "1.5, rounded to 2 decimals; the candidate with clean fields nearest "
        "K0 wins (the lower one on a tie), and if none is clean, K0 itself is "
        "used and the level scan decides.\n\n"
        "A number typed here is used as it is, with no K scan, even when K "
        "candidates are given or Force a K scan is ticked. Blob parts use no K. "
        "K moves the colour boundary only by hundredths of a mm.\n\n"
        "Command line: --k 16.",
        "blank, 4 Å / 0.5 Å     K = 16 (documented)\n"
        "blank, 3.7 Å / 0.5 Å   K scan 7.4 ... 22.2 -> K = 14.8\n"
        "14                     K = 14, no scan",
    ),
    "k_list": (
        "K candidates",
        "Your own K values for the K scan, separated by commas or spaces. Each "
        "must be a positive number; duplicates are dropped. Blank means none: "
        "a K scan, when there is one, tries K0 × 0.5, 0.75, 1, 1.25 and 1.5.\n\n"
        "Giving a list forces a K scan, even at the documented 4 Å / 0.5 Å. "
        "The choice rule stays the same: the candidate with clean fields "
        "nearest K0 = 16 × (resolution/4) × (0.5/grid) wins, and if none is "
        "clean, K0 itself is used even when it is not in the list.\n\n"
        "Ignored when K is set, though it must still be a valid list or the "
        "run does not start.\n\n"
        "Command line: --k-list 12,16,20.",
        "12,16,20 at 4 Å / 0.5 Å (K0 = 16), from the log:\n"
        "    K=12     fields: clean\n"
        "    K=16     fields: clean\n"
        "    K=20     fields: clean\n"
        "    -> K = 16 (clean candidate nearest K0 = 16)",
    ),
    "level": (
        "Contour level",
        "Pins the contour level, in the map's contour units, instead of "
        "scanning for one. Blank means scan: candidate levels around the level "
        "guess are checked and the accepted one nearest the guess is used.\n\n"
        "A pinned level is still checked by the same tests. If it fails one "
        "but the parts can still be built, they are exported anyway, the "
        "report adds a FAIL line for it and the run exits with 3. If the level "
        "is at or above the map's maximum there is no surface, and the run "
        "stops with an error, as it does when the parts cannot be built at "
        "all.\n\n"
        "A pinned level is never widened. The whole's expected piece count is "
        "then what the whole has at that level; a mismatch with the molecule "
        "count is only a warning. A K scan, if one runs, uses the pinned level. "
        "Every report's REPRODUCE line pins the level and K that run used "
        "(with Blob parts, the level, --blobs and its skin).\n\n"
        "Command line: --level 0.112.",
        "0.112, from the log:\n"
        "    level 0.1120: ACCEPT  [2 pinch edges after welding]",
    ),
    "level_guess": (
        "Level scan centre",
        "The level the scan is centred on (and the level a K scan is run at). "
        "Blank means molmap's own automatic level, the contour enclosing 95 % "
        "of the map's density, rounded to 3 decimals; the log prints it as "
        "\"auto level\".\n\n"
        "The scan tries the guess ± span in steps, keeps the levels above 0, "
        "and uses the accepted level nearest the guess (on a tie, fewer pinch "
        "edges, then the lower level). The whole must have one piece per "
        "separate molecule; if no clean level in the scan has that, the "
        "whole's piece count at the guess is the target instead (or, if the "
        "whole is not clean at the guess, the fewest pieces any clean level "
        "has).\n\n"
        "Move it to get a fuller (lower level) or thinner (higher level) model "
        "that still passes the checks; the run suggests it when no level "
        "passes and the whole is one of the reasons. Ignored when the level is "
        "pinned.\n\n"
        "Command line: --level-guess 0.12.",
        "blank   auto level 0.11203, so the centre is 0.112\n"
        "0.104, from the log:\n"
        "[2] level scan: 0.1040 +/- 0.008 in 0.002 steps, K = 16\n"
        "    -> level 0.1040  (nearest clean level to the guess 0.1040)",
    ),
    "span": (
        "Level scan span (±)",
        "How far either side of the guess the level scan reaches, in contour "
        "units. Default 0.008, so with the default step of 0.002 the scan "
        "tries 9 levels: the guess and 4 on each side.\n\n"
        "If none of them is accepted, the scan is widened once to ± twice the "
        "span (only the new levels are computed) and every level is checked "
        "again; if still none passes, the run prints the scan table with hints "
        "and exits with 1. Do not widen skips that second pass.\n\n"
        "A wider span costs time: every level contours the whole and every "
        "part field. The span must be at least the step. Not used with a "
        "pinned level.\n\n"
        "Command line: --level-span 0.008.",
        "0.008   guess 0.112: 0.104, 0.106 ... 0.120 (9 levels)\n"
        "        widened: 0.096 ... 0.128 (8 more)\n"
        "0.004   5 levels; widened to ± 0.008 (4 more)",
    ),
    "step": (
        "Level scan step",
        "The spacing between the contour levels the scan tries, in contour "
        "units. Default 0.002. This is not the voxel step: surfaces are always "
        "contoured at every grid point.\n\n"
        "A smaller step tries more levels in the same span, which takes "
        "longer, and can find a clean level between two rejected ones; the run "
        "suggests 0.001 when no level passes and the whole is one of the "
        "reasons. It must not be larger than the span.\n\n"
        "Command line: --level-step 0.001.",
        "0.002   default: 9 levels over ± 0.008\n"
        "0.001   17 levels over the same span",
    ),
    "scale": (
        "Scale (mm per Å)",
        "How many millimetres one ångström becomes in the exported files "
        "(slicers read STL units as mm). Default 1: 1 Å becomes 1 mm.\n\n"
        "It is applied only when the files are written. The split itself, "
        "the maps, the level and K, and the shapes of the parts, is computed "
        "in Å and does not depend on it, and the report's volumes stay in Å³. "
        "The part STLs, the whole STL and the GLB preview are scaled, and the "
        "report's PRINTING section gives the exported size in mm and the layer "
        "count.\n\n"
        "Command line: --scale 2.",
        "1    24.9 x 67.2 x 24.7 Å  ->  24.9 x 67.2 x 24.7 mm\n"
        "2    24.9 x 67.2 x 24.7 Å  ->  49.8 x 134.5 x 49.3 mm",
    ),
    "lay_flat": (
        "Lay flat for printing",
        "Rotates everything, every part and the whole by one shared transform, "
        "so the model's longest principal axis runs along X, the middle one "
        "along Y and the shortest one is vertical, then centres it in X and Y "
        "and puts its lowest point at Z = 0. The axes come from a principal "
        "component analysis of the whole's surface vertices. "
        "Off (the default) keeps the PDB frame.\n\n"
        "A lower model needs fewer layers, and on a multi-material printer the "
        "purge scales with the layer count. When the model is not laid flat "
        "and lying flat would make it less than 80 % as tall, the report's "
        "PRINTING section says so, with the layer counts at 0.2 mm layers. The "
        "parts stay registered either way; the alternative is to rotate the "
        "whole object in the slicer, never the parts on their own.\n\n"
        "Command line: --lay-flat.",
        "a triplex, from the report:\n"
        "off   67.1 mm tall, ~336 layers at 0.2 mm\n"
        "on    27.4 mm tall, ~138 layers",
    ),
    "k_scan": (
        "Force a K scan",
        "Scans K even at the documented point (resolution 4 Å, grid 0.5 Å), "
        "where K = 16 is otherwise used without a scan. The candidates are "
        "16 × 0.5, 0.75, 1, 1.25 and 1.5, i.e. 8, 12, 16, 20 and 24, or the K "
        "candidates if given, each tested at the level guess (or the pinned "
        "level); the candidate with clean fields nearest K0 = 16 wins.\n\n"
        "Off (the default): a K scan runs only away from the documented point "
        "or when K candidates are given. Ignored when K is set. "
        "It confirms K = 16 for an unusual model at the cost of an extra "
        "ChimeraX pass.\n\n"
        "Command line: --k-scan.",
        "at 4 Å / 0.5 Å the scan tries 8, 12, 16, 20, 24\n"
        "and keeps 16 whenever its fields are clean",
    ),
    "allow_multi_shell": (
        "Allow parts with several shells",
        "Drops the piece-count limits on the parts. Off (the default), each "
        "part's own field-built solid inside the whole, and each finished "
        "part, may have at most one piece per cluster of the part's own atoms "
        "(heavy atoms within 4 Å form a cluster): 1 normally, 2 for a chain "
        "with a break. A level where a part has more pieces is rejected.\n\n"
        "The report's boolean-result check for each part, though, wants "
        "exactly that many pieces. So a part with a chain break that comes out "
        "in one piece is accepted by the scan but gets a FAIL line there, and "
        "the run exits with 3 although the part is intact; ticking this "
        "clears that FAIL.\n\n"
        "Ticked, a part may come out in any number of pieces. The whole's own "
        "piece-count rule and every other check still apply.\n\n"
        "Use it also when a part really is in several pieces and the scan "
        "rejects every level for the piece count of that part's field (Npc "
        "under S1, S2 ..., or B1, B2 ... with Blob parts); the run suggests it then. With it, a part cut "
        "into islands by its neighbours is no longer caught.\n\n"
        "Command line: --allow-multi-shell.",
        None,
    ),
    "blobs": (
        "Blob parts",
        "Another way to cut the parts, for parts that are separate blobs, such as "
        "each chain's phosphate groups. Ticked, every part but the last is its "
        "own molmap surface: the surface its atoms' density alone has at the "
        "contour level, the molmap ChimeraX draws for those atoms on their own, "
        "clipped to the whole (where two blobs overlap, the earlier part keeps the "
        "overlap). The last part is the whole minus the blobs, so the blobs sit in "
        "sockets in it, and the parts still tile the whole exactly.\n\n"
        "Off (the default), each part is the territory where its own density beats "
        "the others' (ownership fields with the K penalty), which cuts a part "
        "midway between its atoms and the neighbouring ones. A blob's face inside "
        "the model is instead its own rounded molmap surface, so it reaches as far "
        "into its neighbours as its own density does. On the switchback666 duplex "
        "at 3.7 Å, chain A's phosphate groups (P, OP1, OP2, O5′, O3′) make "
        "1,474 Å³ as blobs against 1,274 Å³ as ownership territory, and show "
        "954 Å² of surface against 1,042 Å².\n\n"
        "Put the part that should fill around the blobs last (rest, for "
        "instance): the order decides which parts are blobs. No K is used, so "
        "there is no K scan, and the level scan checks the levels nearest the "
        "guess first and stops at the nearest clean one, which is the level a "
        "full scan would pick; on a 210 bp supercoil the phosphate split took 184 s this way, against 496 s by ownership with v1.5.0. "
        "Each level it checks gets the same checks, except that the last part "
        "has no field of its own, and the report leaves out the density-share "
        "warning, which a blob is not meant to meet. Blob skin sets what happens "
        "where a blob would lie a hair beneath the surface.\n\n"
        "Command line: --blobs.",
        "Parts by atom selections, Blob parts ticked:\n"
        "  `phosA = /A@P,OP1,OP2,OP3,O5',O3'`\n"
        "  `phosC = /C@P,OP1,OP2,OP3,O5',O3'`\n"
        "  `rest`\n"
        "P1, P2 = each phosphate group's own molmap blob\n"
        "P3 = the whole minus the blobs",
    ),
    "blob_skin": (
        "Blob skin (Å)",
        "Used only with Blob parts. Where a blob faces outwards, its own molmap "
        "surface lies a hair inside the whole's, because the neighbouring atoms "
        "add a trace of density there. Cut exactly, the last part would wrap each "
        "blob in a film, about half of it under 0.01 Å thick and up to a few "
        "tenths of an Å near the blob's edge.\n\n"
        "Printed together at 1 mm per Å, that half is under 10 µm, far below "
        "what a slicer prints, so a slicer should drop it on the blob's sides "
        "and show the blob's colour there (not tested). What the film does "
        "change is the files. The last part's STL covers the blobs (and can seal "
        "one in completely), its mesh grows by about 70 %, the preview PNG and GLB and the slicer's view before slicing "
        "show the blobs in the last part's colour, and the thicker band near a "
        "blob's edge, a near-flat blob top, or a large Scale can print as strips "
        "or patches of the last part's colour.\n\n"
        "Where the last part's skin over a blob would be thinner than this and "
        "the blob's own density exceeds that of all the other atoms together, the blob takes the "
        "skin, so the files come closer to what should print. The skin is estimated from the "
        "maps: the other atoms' density over the slope of the whole map. The "
        "estimate runs low for thicker skins, so at 0.2 Å skins up to about "
        "0.3 Å thick go to the blob. Blank means 0.2 Å; 0 keeps every blob "
        "exactly its own molmap surface, films included. Above about 0.3 Å it "
        "changes little, since only places where the blob's own density exceeds "
        "the other atoms' together are ever taken. Allowed 0 to 1.\n\n"
        "Command line: --blob-skin 0.2.",
        "switchback666 at 3.7 Å, chain A's phosphate blobs:\n"
        "`0.2`   1,474 Å³; 954 Å² on the whole's surface\n"
        "`0`     1,437 Å³; 208 Å² within 0.0001 Å of the\n"
        "        surface, 168 Å² of it still under the rest's\n"
        "        film, which covers the rest of each blob and\n"
        "        seals one blob in",
    ),
    "no_widen": (
        "Do not widen a failed level scan",
        "Stops after the first level scan when no level in it is accepted. "
        "Off (the default), the scan is then widened once to ± twice the span; "
        "only the new levels are computed, and every level is checked again "
        "before the run gives up.\n\n"
        "Either way, when nothing passes, the run prints the scan table and "
        "hints and exits with 1. Ticked, it saves the time of the second pass, "
        "but a level that only the wider scan reaches is never tried, so a run "
        "the widening would have rescued exits with 1 instead.\n\n"
        "A pinned level is never widened.\n\n"
        "Command line: --no-widen.",
        None,
    ),
    "keep_solvent": (
        "Keep solvent (waters)",
        "Includes waters in the maps. Off (the default), ChimeraX's solvent is "
        "left out of every map, and the chain table and the molecule count "
        "skip the residue names HOH, WAT, DOD, H2O, TIP, TIP3, SOL, T3P and "
        "SPC.\n\n"
        "Ticked, a water counts as part of the chain it is in and adds to that "
        "part's density, and a chain of waters becomes a chain of its own. "
        "With Parts blank, or filled by Detect chains, that chain then becomes "
        "a colour of its own, which is rarely wanted (and as dust spares a "
        "part's main molecule, one of its waters standing apart can print as a "
        "loose blob); list the parts without it.\n\n"
        "Any other water standing apart from everything is a separate small "
        "molecule: unless it is a part's main molecule, the default dust "
        "removes it, and the run names such molecules in a note. Click Detect "
        "chains again after changing this box.\n\n"
        "Command line: --keep-solvent.",
        None,
    ),
    "no_hydrogens": (
        "Exclude hydrogens",
        "Leaves hydrogen atoms out of the maps. Off (the default), hydrogens "
        "in the file go into the maps like any other atom. "
        "Tick it for a map of the heavy atoms only, e.g. to match a model that "
        "has no hydrogens.\n\n"
        "It does not change which atoms form a molecule or a cluster "
        "(hydrogens never count there) or the chain table's atom counts; the "
        "atom counts on the report's parts line are the atoms in the maps, so "
        "they drop by the hydrogens. The report's \"solvent / H\" line says "
        "which was used.\n\n"
        "Command line: --no-hydrogens.",
        None,
    ),
    "keep_scan": (
        "Keep scan STLs",
        "Also keeps every candidate surface the scans exported, the whole at "
        "each level and each part field at each K and level, in scan_kscan/, "
        "scan_level/ and scan_widen/ inside <tag>_work/ (or "
        "<tag>_work_failed/). S1, S2 ... are the parts' own fields and H2 ... "
        "the peeling fields of 3 or more parts; with Blob parts, B1, B2 ... "
        "are the blobs (B1_L0.112000.stl, no K and no K scan). Off (the default), only the "
        "ChimeraX jobs, parameters and logs are kept.\n\n"
        "Useful to look at why a level was rejected; each file is a full-size "
        "STL, so the folder grows with every level and K tried.\n\n"
        "Command line: --keep-scan.",
        "<tag>_work/scan_level/ref_L0.112000.stl     whole\n"
        "<tag>_work/scan_level/S1_K16_L0.112000.stl  part 1 field\n"
        "<tag>_work/scan_kscan/S2_K12_L0.112000.stl  K scan",
    ),
    "no_preview": (
        "Skip preview PNG/GLB",
        "Skips the two local previews. Off (the default), a run writes "
        "<tag>_preview.png, a top view (the print bed is the page) and an "
        "oblique view with each part in its colour, and <tag>_preview.glb, the "
        "parts as coloured meshes for a 3D viewer. A preview that cannot be "
        "made, e.g. without matplotlib, is skipped with a line in the log, not "
        "an error.\n\n"
        "After a run that exits with 0 or 3, the window opens the new PNG by "
        "itself unless this is ticked. Tick it to save the rendering time.\n\n"
        "Command line: --no-preview.",
        None,
    ),
    "voxel_step": (
        "Voxel step 1 (fixed)",
        "Every surface, the whole and each part field at every level and K, is "
        "contoured at voxel step 1, i.e. at every grid point. The tool sets "
        "step 1 on each surface and checks it before anything is exported; a "
        "surface at any other step stops the run instead of being written. The "
        "log says \"every surface exported at step 1x1x1\", and the report "
        "repeats it.\n\n"
        "It is not adjustable because a coarser step contours only every "
        "second or third grid point, which is the same as a coarser grid: the "
        "colour interface could no longer be placed finer than that. "
        "To trade detail for speed, change Grid spacing instead; the K rule "
        "accounts for it. Level scan step is a different thing: a step in "
        "contour level, not in voxels.\n\n"
        "Command line: none; there is no option for it.",
        None,
    ),
    "run": (
        "Run split, Stop and the other buttons",
        "Run split checks the fields, remembers the ChimeraX and Chimera "
        "paths, and runs "
        "the split as a separate process with the options Show command "
        "prints; its output streams into the log below and the status line at "
        "the right says how it ended.\n\n"
        "A run writes into the output folder: one STL per part (a part that "
        "comes out empty is not written, and fails), <tag>_whole.stl for "
        "reference only (do not print it with the parts), <tag>_report.txt "
        "and .json (scans, validation, printing notes and a REPRODUCE command "
        "with everything pinned), the two previews, and <tag>_work/ with the "
        "ChimeraX jobs, parameters and logs.\n\n"
        "The STLs, report and previews are first written to a staging folder, "
        ".<tag>_incoming. Once the report is complete, the previous set of the "
        "same tag in that folder (the files its report listed, files named "
        "like this tool's outputs, and <tag>_work/ or <tag>_work_failed/) is "
        "moved into _previous_<timestamp>/, the new files are moved in, and "
        "the logs follow into <tag>_work/, so the folder holds one set per "
        "tag; other files are never touched. A run that exits with 3 replaces "
        "the previous set too, so read its report before printing (the "
        "earlier set is in _previous_<timestamp>/).\n\n"
        "A run that ends with an error or is stopped puts nothing in place and "
        "leaves the previous set where it was; its logs go to "
        "<tag>_work_failed/. An error found before ChimeraX starts, such as a "
        "chain that is not in the file or ChimeraX not found, leaves no logs: "
        "its message is in the log below.\n\n"
        "Exit codes: 0 all checks pass (warnings allowed); 3 exported, but a "
        "check FAILED; 1 error; 2 invalid options; 130 stopped.\n\n"
        "Stop sends SIGINT to the run: it stops ChimeraX, removes the staging "
        "folder, keeps the logs and leaves nothing half-written. Closing the "
        "window during a run asks first.\n\n"
        "Show command prints the command in the log and copies it to the "
        "clipboard. Open output folder and Show preview use the folder and tag "
        "of the last run, or those of the current fields; the preview also "
        "opens by itself after a run that exits with 0 or 3.\n\n"
        "Command line: the command Show command prints runs the same split in "
        "a terminal, with the same exit codes. With Output folder blank it "
        "writes to ./<tag> of the folder the terminal is in, and a relative "
        "structure path is read from there too, so run it from the folder "
        "this window was started in, or fill Output folder first.",
        "exit  status line\n"
        "0     done — all checks pass\n"
        "3     done — some checks FAILED (see the report)\n"
        "1     FAILED (exit 1) — read the log\n"
        "2     not run — invalid options (see the log)\n"
        "130   stopped\n"
        "re-run, same tag: old set -> _previous_20260925-231500/",
    ),
}


def example_segments(text):
    """A HELP example as (piece, is_code) runs.  `...` marks what is typed,
    a selection, a Parts entry, which the popup shows apart from the words
    saying what it does."""
    pieces = text.split("`")
    if len(pieces) % 2 == 0:
        raise ValueError(f"unbalanced ` in a help example: {text[:40]!r}")
    return [(piece, bool(i % 2)) for i, piece in enumerate(pieces) if piece]


# ======================================================================= GUI
GUI_PREFS = Path.home() / ".multicolor_split_gui.json"   # remembers the ChimeraX and Chimera paths
# Parts by atom selections: the menus' labels for the options' values
SYNTAX_LABELS = {"chimerax": "ChimeraX", "chimera": "Chimera"}
OVERLAP_LABELS = {"first": "earlier part keeps them", "error": "refuse the run"}

# The numeric fields: key, label, default, hint
GUI_NUMBERS = [
    ("resolution", "Resolution (Å)", f"{DOC_RESOLUTION:g}", "default 4.0; try 3.7 for more detail, few holes"),
    ("grid", "Grid spacing (Å)", f"{DOC_GRID:g}", "0.5 resolves the colour interface"),
    ("dust", "Dust size (Å)", f"{DEFAULT_DUST:g}", "specks below this go; a part's main body never does"),
    ("k", "K penalty slope", "", "blank = 16 at 4 Å / 0.5; otherwise rescaled + checked"),
    ("k_list", "K candidates", "", "optional, e.g. 8,12,16 (implies a K scan)"),
    ("level", "Contour level", "", "blank = scan for a clean level"),
    ("level_guess", "Level scan centre", "", "blank = molmap's auto level"),
    ("span", "Level scan span (±)", f"{DEFAULT_SPAN:g}", "contour units"),
    ("step", "Level scan step", f"{DEFAULT_STEP:g}", "contour increment -- NOT the voxel step"),
    ("scale", "Scale (mm per Å)", "1", "1 = 1 Å → 1 mm"),
    ("blob_skin", "Blob skin (Å)", "", f"blank = {DEFAULT_BLOB_SKIN:g}; Blob parts: thinner skins go to the blob"),
]
# The CLI flag of a GUI_NUMBERS key when listed here; otherwise it is
# "--" + key.replace("_", "-").
GUI_FLAG_NAMES = {"span": "--level-span", "step": "--level-step", "level_guess": "--level-guess",
                  "k_list": "--k-list"}
# The checkboxes: key, label, default.  The CLI flag is "--" + key.replace("_", "-").
GUI_OPTIONS = [
    ("lay_flat", "Lay flat for printing (fewer layers → less purge)", False),
    ("k_scan", "Force a K scan", False),
    ("allow_multi_shell", "Allow parts with several shells", False),
    ("blobs", "Blob parts (each but the last is its own molmap)", False),
    ("no_widen", "Do not widen a failed level scan", False),
    ("keep_solvent", "Keep solvent (waters)", False),
    ("no_hydrogens", "Exclude hydrogens", False),
    ("keep_scan", "Keep scan STLs (<tag>_work/)", False),
    ("no_preview", "Skip preview PNG/GLB", False),
]


def run_gui(initial_file=None, prefill=None, selftest=None):
    """The window.  prefill: option name -> value, as main() collects them;
    initial_file becomes the structure unless prefill already names one.
    selftest ("build", "fill", "run" or "help") keeps the window hidden,
    prints what it checked and closes it ("fill" clicks Backbone per chain
    first).  Returns the exit code: with selftest "run" the
    split's own (2 when the fields give no command), with "help" 1 when a
    popup failed, otherwise 0."""
    import queue
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    prefill = dict(prefill or {})
    if initial_file and not prefill.get("structure"):
        prefill["structure"] = str(initial_file)

    # ---- "?" help chips ---------------------------------------------------
    open_help = {"win": None}
    chip_keys = []              # the key of every chip built, in build order
    chip_widgets = {}           # key -> its first chip (the self test opens popups there)

    def show_help(key, near):
        title, prose, example = HELP[key]
        if open_help["win"] is not None:
            try:
                open_help["win"].destroy()
            except Exception:               # noqa: BLE001
                pass
        top = tk.Toplevel(root)
        top.withdraw()                  # placed first, then shown: no flash at 0,0
        open_help["win"] = top
        top.title(title)
        top.transient(root)
        top.configure(bg="#f4f9ff")
        frm = tk.Frame(top, bg="#f4f9ff", padx=16, pady=14)
        frm.pack(fill="both", expand=True)
        tk.Label(frm, text=title, font=("Helvetica", 14, "bold"),
                 bg="#f4f9ff", fg="#0b4d80", anchor="w",
                 justify="left").pack(fill="x", pady=(0, 8))
        text = tk.Label(frm, text=prose, wraplength=470, justify="left",
                        bg="#f4f9ff", fg="#1a1a1a", font=("Helvetica", 12),
                        anchor="w")
        text.pack(fill="x")
        if example:
            segments = example_segments(example)
            plain = "".join(piece for piece, _ in segments)
            head = tk.Frame(frm, bg="#f4f9ff")
            head.pack(fill="x", pady=(12, 3))
            tk.Label(head, text="Example", font=("Helvetica", 11, "bold"),
                     bg="#f4f9ff", fg="#0b4d80").pack(side="left")
            if any(code for _, code in segments):
                tk.Label(head, text="    bold on white: what you type;  the rest says what it does",
                         font=("Helvetica", 11), bg="#f4f9ff", fg="#555").pack(side="left")
            # Height counted in DISPLAY rows, not in newlines: a line
            # longer than the box wraps, and a height counted in newlines
            # then clips the tail of the block off the bottom.
            width = 62
            rows = sum(max(1, -(-len(line) // width))
                       for line in plain.split("\n"))
            box = tk.Text(frm, font=("Menlo", 11), height=rows,
                          width=width, bg="#e8f1fa", relief="flat",
                          padx=8, pady=6, highlightthickness=0)
            box.tag_configure("code", font=("Menlo", 11, "bold"), background="#ffffff",
                              foreground="#0b3d66")
            for piece, code in segments:
                box.insert(tk.END, piece, ("code",) if code else ())
            box.configure(state="disabled")
            box.pack(fill="x")
        tk.Button(frm, text="Close", command=top.destroy).pack(pady=(12, 0))
        top.bind("<Escape>", lambda e: top.destroy())
        # Beside the chip, but kept on the screen: the lowest chips sit far
        # down the window and carry the longest texts, whose Close button
        # would otherwise land below the screen's bottom edge.  Tk measures
        # only the main display, so a chip on another display (a window moved
        # to an external monitor) keeps the plain beside-the-chip placement
        # rather than having its popup pulled back onto the main one.
        try:
            top.update_idletasks()
            sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
            cx, cy = near.winfo_rootx(), near.winfo_rooty()
            x, y = cx + 26, cy - 10
            if 0 <= cx < sw and 0 <= cy < sh:
                if top.winfo_reqheight() > sh - 130:    # a short screen: go wider
                    text.configure(wraplength=max(470, min(760, sw - 120)))
                    top.update_idletasks()
                w, h = top.winfo_reqwidth(), top.winfo_reqheight()
                x = max(min(x, sw - w - 8), 0)
                y = max(min(y, sh - h - 100), 0)        # title bar and Dock
            top.geometry("+%d+%d" % (x, y))
        except Exception:                   # noqa: BLE001
            pass
        if not selftest:                    # the self test lays it out unseen
            top.deiconify()
        top.focus_set()

    def chip(parent, key):
        """A small light-blue '?' that opens the explanation for `key`.

        Built from a Label rather than a Button because macOS's native button
        ignores background colour.  A key with no HELP entry fails here, when
        the window is built, not later when someone clicks it.
        """
        if key not in HELP:
            raise KeyError(f"no HELP entry for the ? chip {key!r}")
        c = tk.Label(parent, text="?", font=("Helvetica", 11, "bold"),
                     bg="#bcdcf5", fg="#0b4d80", width=2, relief="raised",
                     borderwidth=1, cursor="hand2")
        c.bind("<Button-1>", lambda e, k=key, w=c: show_help(k, w))
        c.bind("<Enter>", lambda e, w=c: w.configure(bg="#8fc7ee"))
        c.bind("<Leave>", lambda e, w=c: w.configure(bg="#bcdcf5"))
        chip_keys.append(key)
        chip_widgets.setdefault(key, c)
        return c

    class App:
        GEOMETRY = (1060, 900)          # default window size; grown to fit
        MINSIZE = (880, 660)
        LOG_LINES = 12                  # the log's natural height; it fills any extra
        LOG_MIN_LINES = 4               # the log shrinks to this many lines first

        def __init__(self, root):
            self.root = root
            self.proc = None
            self.q = queue.Queue()
            self.chains = []
            self.last_out = None
            self.last_tag = None
            self.run_started = 0.0
            self.stopped = False
            self.size = self.GEOMETRY
            root.title(f"{TOOL_NAME} v{__version__} — molmap chains → multicolour STLs")
            root.geometry("%dx%d" % self.GEOMETRY)
            root.minsize(*self.MINSIZE)
            root.protocol("WM_DELETE_WINDOW", self.on_close)
            frm = ttk.Frame(root, padding=10)
            frm.pack(fill=tk.BOTH, expand=True)
            frm.columnconfigure(1, weight=1)
            self.v = {}
            self.rows = {}              # key -> the widgets of its rows, to show or hide them
            self.chain_rows = []        # read_structure()'s chains, for Backbone per chain
            r = 0

            # columns: 0 label, 1 entry, 2 Browse…, 3 the ? chip
            def row_entry(label, key, browse=None, width=70):
                nonlocal r
                ws = [ttk.Label(frm, text=label)]
                ws[-1].grid(row=r, column=0, sticky=tk.W, pady=2)
                var = tk.StringVar()
                self.v[key] = var
                ws.append(ttk.Entry(frm, textvariable=var, width=width))
                ws[-1].grid(row=r, column=1, sticky=tk.EW, pady=2, padx=4)
                if browse:
                    ws.append(ttk.Button(frm, text="Browse…", command=browse))
                    ws[-1].grid(row=r, column=2, sticky=tk.W)
                ws.append(chip(frm, key))
                ws[-1].grid(row=r, column=3, sticky=tk.W, padx=(6, 0))
                self.rows[key] = ws
                r += 1
                return var

            row_entry("Structure (PDB / mmCIF)", "structure", self.browse_pdb)
            bf = ttk.Frame(frm)
            bf.grid(row=r, column=1, sticky=tk.W, padx=4)
            ttk.Button(bf, text="Detect chains", command=self.detect).pack(side=tk.LEFT)
            chip(bf, "chains").pack(side=tk.LEFT, padx=(6, 0))
            self.chain_info = tk.StringVar(value="")
            ttk.Label(bf, textvariable=self.chain_info, foreground="#666").pack(side=tk.LEFT, padx=8)
            r += 1
            tf = ttk.Frame(frm)
            tf.grid(row=r, column=0, columnspan=4, sticky=tk.EW, pady=4)
            self.tree = ttk.Treeview(tf, columns=("atoms", "res", "range", "types"),
                                     height=4, show="tree headings")
            for col, t, w in (("#0", "chain", 70), ("atoms", "atoms", 70), ("res", "residues", 80),
                              ("range", "numbering", 100), ("types", "residue types", 440)):
                self.tree.heading(col, text=t)
                self.tree.column(col, width=w, anchor=tk.W)
            tsb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=self.tree.yview)
            self.tree.configure(yscrollcommand=tsb.set)
            self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tsb.pack(side=tk.LEFT, fill=tk.Y)
            r += 1
            # the parts: whole chains, or atom selections (one of the two rows below)
            ttk.Label(frm, text="Parts by").grid(row=r, column=0, sticky=tk.W, pady=2)
            mf = ttk.Frame(frm)
            mf.grid(row=r, column=1, columnspan=2, sticky=tk.W, padx=4)
            self.v["parts_by"] = tk.StringVar(value="chains")
            for value, text in (("chains", "chains"),
                                ("selections", "atom selections (ChimeraX or Chimera convention)")):
                ttk.Radiobutton(mf, text=text, value=value, variable=self.v["parts_by"]).pack(
                    side=tk.LEFT, padx=(0, 12))
            chip(frm, "parts_by").grid(row=r, column=3, sticky=tk.W, padx=(6, 0))
            r += 1
            row_entry("Parts (colour order)", "parts")
            hint = ttk.Label(frm, text="one token per colour, space-separated; join chains with ','   "
                                       "e.g.  A C   or   A,B C", foreground="#666")
            hint.grid(row=r, column=1, sticky=tk.W, padx=4)
            self.rows["parts"].append(hint)
            r += 1
            # one selection per line, then its convention, the rule for atoms
            # two parts share, and a fill for the usual case
            ws = [ttk.Label(frm, text="Selections\n(colour order)")]
            ws[-1].grid(row=r, column=0, sticky=tk.NW, pady=2)
            sf = ttk.Frame(frm)
            sf.grid(row=r, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=2)
            ws.append(sf)
            self.sel_text = tk.Text(sf, height=4, width=70, wrap=tk.NONE, undo=True,
                                    font=("Menlo", 11))
            ssb = ttk.Scrollbar(sf, orient=tk.VERTICAL, command=self.sel_text.yview)
            self.sel_text.configure(yscrollcommand=ssb.set)
            self.sel_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ssb.pack(side=tk.LEFT, fill=tk.Y)
            self.sel_text.bind("<<Modified>>", self.on_selections_edited)
            ws.append(chip(frm, "part"))
            ws[-1].grid(row=r, column=3, sticky=tk.NW, padx=(6, 0), pady=2)
            r += 1
            of = ttk.Frame(frm)
            of.grid(row=r, column=1, columnspan=3, sticky=tk.W, padx=4, pady=(0, 2))
            ws.append(of)
            ttk.Label(of, text="Convention").pack(side=tk.LEFT)
            self.v["selection_syntax"] = tk.StringVar(value=SYNTAX_LABELS["chimerax"])
            ttk.Combobox(of, textvariable=self.v["selection_syntax"],
                         values=list(SYNTAX_LABELS.values()), state="readonly",
                         width=9).pack(side=tk.LEFT, padx=(4, 0))
            chip(of, "selection_syntax").pack(side=tk.LEFT, padx=(6, 14))
            ttk.Label(of, text="Shared atoms").pack(side=tk.LEFT)
            self.v["overlap"] = tk.StringVar(value=OVERLAP_LABELS["first"])
            ttk.Combobox(of, textvariable=self.v["overlap"], values=list(OVERLAP_LABELS.values()),
                         state="readonly", width=21).pack(side=tk.LEFT, padx=(4, 0))
            chip(of, "overlap").pack(side=tk.LEFT, padx=(6, 14))
            ttk.Button(of, text="Backbone per chain", command=self.fill_backbone).pack(side=tk.LEFT)
            chip(of, "fill_backbone").pack(side=tk.LEFT, padx=(6, 0))
            r += 1
            ws.append(ttk.Label(frm, text="one part per line, P1 first:   NAME = selection   or   "
                                          "rest      e.g.  bbA = /A & backbone", foreground="#666"))
            ws[-1].grid(row=r, column=1, columnspan=2, sticky=tk.W, padx=4)
            self.rows["part"] = ws
            r += 1
            # the part colours: a menu, then one swatch per part in its colour
            ttk.Label(frm, text="Palette").grid(row=r, column=0, sticky=tk.W, pady=2)
            pf = ttk.Frame(frm)
            pf.grid(row=r, column=1, columnspan=2, sticky=tk.W, padx=4, pady=2)
            self.v["palette"] = tk.StringVar(value="default")
            ttk.Combobox(pf, textvariable=self.v["palette"], values=palette_choices(),
                         state="readonly", width=14).pack(side=tk.LEFT)
            self.swatch_box = ttk.Frame(pf)
            self.swatch_box.pack(side=tk.LEFT, padx=(8, 0))
            self.swatches = []          # ("P1 A red", #rrggbb) per swatch, for the self test
            chip(frm, "palette").grid(row=r, column=3, sticky=tk.W, padx=(6, 0))
            r += 1
            row_entry("Output folder", "out", self.browse_out)
            row_entry("Tag (file prefix)", "tag", width=40)
            self.auto_name = tk.StringVar(value="")
            # A long output path wraps within its cell instead of widening the
            # window: the label asks for a fixed 40 characters of width and
            # wraps at whatever width its cell is given.
            names = ttk.Label(frm, textvariable=self.auto_name, foreground="#666",
                              width=40, wraplength=600)
            names.grid(row=r, column=1, columnspan=2, sticky=tk.EW, padx=4)
            names.bind("<Configure>", lambda e, w=names: w.configure(wraplength=max(200, e.width)))
            r += 1

            box = ttk.Frame(frm)
            box.grid(row=r, column=0, columnspan=4, sticky=tk.EW, pady=6)
            r += 1
            num = ttk.LabelFrame(box, text="molmap & split parameters", padding=6)
            num.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            # columns: 0 label, 1 entry, 2 the ? chip, 3 hint
            for i, (key, label, default, hint) in enumerate(GUI_NUMBERS):
                ttk.Label(num, text=label).grid(row=i, column=0, sticky=tk.W, pady=1)
                var = tk.StringVar(value=default)
                self.v[key] = var
                ttk.Entry(num, textvariable=var, width=10).grid(row=i, column=1, sticky=tk.W, padx=4)
                chip(num, key).grid(row=i, column=2, sticky=tk.W, padx=(0, 6))
                ttk.Label(num, text=hint, foreground="#666").grid(row=i, column=3, sticky=tk.W)
            vf = ttk.Frame(num)
            vf.grid(row=len(GUI_NUMBERS), column=0, columnspan=4, sticky=tk.W, pady=(4, 0))
            ttk.Label(vf, text="molmap surfaces are always computed at voxel step 1 (every grid "
                               "point) -- not adjustable", foreground="#2a6").pack(side=tk.LEFT)
            chip(vf, "voxel_step").pack(side=tk.LEFT, padx=(6, 0))
            opt = ttk.LabelFrame(box, text="options", padding=6)
            opt.pack(side=tk.LEFT, fill=tk.BOTH, padx=(8, 0))
            # columns: 0 the checkbox, 1 its ? chip
            for i, (key, label, default) in enumerate(GUI_OPTIONS):
                var = tk.BooleanVar(value=default)
                self.v[key] = var
                ttk.Checkbutton(opt, text=label, variable=var).grid(row=i, column=0, sticky=tk.W)
                chip(opt, key).grid(row=i, column=1, sticky=tk.W, padx=(6, 0), pady=1)

            row_entry("ChimeraX", "chimerax", self.browse_cx)
            row_entry("Chimera", "chimera", self.browse_chimera)   # the Chimera convention only
            bar = ttk.Frame(frm)
            bar.grid(row=r, column=0, columnspan=4, sticky=tk.EW, pady=6)
            r += 1
            self.run_btn = ttk.Button(bar, text="Run split", command=self.run)
            self.run_btn.pack(side=tk.LEFT)
            self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state=tk.DISABLED)
            self.stop_btn.pack(side=tk.LEFT, padx=4)
            chip(bar, "run").pack(side=tk.LEFT, padx=(2, 8))
            ttk.Button(bar, text="Show command", command=self.show_cmd).pack(side=tk.LEFT, padx=4)
            ttk.Button(bar, text="Open output folder", command=self.open_out).pack(side=tk.LEFT, padx=4)
            ttk.Button(bar, text="Show preview", command=self.show_preview).pack(side=tk.LEFT, padx=4)
            self.status = tk.StringVar(value="ready")
            ttk.Label(bar, textvariable=self.status).pack(side=tk.RIGHT)

            logf = ttk.Frame(frm)
            logf.grid(row=r, column=0, columnspan=4, sticky=tk.NSEW)
            frm.rowconfigure(r, weight=1)
            self.log = tk.Text(logf, wrap=tk.NONE, height=self.LOG_LINES, font=("Menlo", 11))
            sb = ttk.Scrollbar(logf, orient=tk.VERTICAL, command=self.log.yview)
            self.log.configure(yscrollcommand=sb.set, state=tk.DISABLED)
            self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.LEFT, fill=tk.Y)

            if not selftest:            # a self test must not depend on / touch ~/ prefs
                self.load_prefs()
            for k, val in prefill.items():
                if k in self.v and val not in (None, ""):
                    if isinstance(self.v[k], tk.BooleanVar):
                        self.v[k].set(bool(val))
                    else:
                        self.v[k].set(f"{val:.12g}" if isinstance(val, float) else str(val))
            if prefill.get("part"):
                self.set_selections(str(prefill["part"]).splitlines())
            for key, labels in (("selection_syntax", SYNTAX_LABELS), ("overlap", OVERLAP_LABELS)):
                given = self.v[key].get().strip()        # an option's value -> its label
                self.v[key].set(labels.get(given.lower(), given))
            if not self.v["chimerax"].get():
                try:
                    self.v["chimerax"].set(find_chimerax())
                except SplitError:
                    pass
            if not self.v["chimera"].get():
                try:
                    self.v["chimera"].set(find_chimera())
                except SplitError:
                    pass
            for key in ("structure", "resolution", "tag", "out"):
                self.v[key].trace_add("write", lambda *_: self.update_names())
            for key in ("parts", "palette"):
                self.v[key].trace_add("write", lambda *_: self.update_swatches())
            for key in ("parts_by", "selection_syntax"):
                self.v[key].trace_add("write", lambda *_: self.show_parts_mode())
            if self.v["structure"].get():
                self.detect(fill_parts=not self.v["parts"].get())
            self.update_names()
            self.show_parts_mode(fit=False)
            self.say("Click any light-blue  ?  for an explanation and examples.\n")
            self.fit_window()
            root.after(120, self.poll)

        # ---------------------------------------------------------- helpers
        def fit_window(self, initial=True):
            """Grow the minimum window size (and, when the window is first
            built, its default size) to what the widgets ask for, so nothing
            is clipped.  Only the log may get less height than it asks for,
            down to LOG_MIN_LINES lines: its grid row gives up any shortfall
            first, and it scrolls.  Called again after Detect chains, whose
            table grows with the number of chains.  Neither size is ever
            larger than the screen: on a short screen the chain table, which
            scrolls too, gives up rows next, down to 3, and on a screen too
            short even for that (1440x900 with ten chains) the log takes
            what height is left, below LOG_MIN_LINES."""
            root = self.root
            root.update_idletasks()
            sw = root.winfo_screenwidth() - 40
            sh = root.winfo_screenheight() - 90         # menu bar, title bar, Dock
            rw, rh = root.winfo_reqwidth(), root.winfo_reqheight()
            log_h = self.log.winfo_reqheight()
            spare = max(0, log_h - log_h * self.LOG_MIN_LINES // self.LOG_LINES)
            rows = int(self.tree.cget("height"))
            while rh - spare > sh and rows > 3:
                rows -= 1
                self.tree.configure(height=rows)
                root.update_idletasks()
                rw, rh = root.winfo_reqwidth(), root.winfo_reqheight()
            root.minsize(min(max(self.MINSIZE[0], rw), sw),
                         min(max(self.MINSIZE[1], rh - spare), sh))
            if initial:
                gw = min(max(self.GEOMETRY[0], rw), sw)
                gh = min(max(self.GEOMETRY[1], rh), sh)
                root.geometry(f"{gw}x{gh}")
                self.size = (gw, gh)

        def error(self, msg):
            if selftest:
                print("[selftest] error:", msg, flush=True)
            else:
                messagebox.showerror(TOOL_NAME, msg)

        def say(self, text):
            self.log.configure(state=tk.NORMAL)
            self.log.insert(tk.END, text)
            self.log.see(tk.END)
            self.log.configure(state=tk.DISABLED)

        def load_prefs(self):
            """The remembered ChimeraX and Chimera paths, if they still work.
            One that no longer does (the program moved or was updated) is
            left out, so the field is filled by $CHIMERAX / $CHIMERA or the
            search instead, not with a path every run would fail on."""
            try:
                d = json.loads(GUI_PREFS.read_text())
            except Exception:
                return
            if not isinstance(d, dict):
                return
            for key, name, works in (
                    ("chimerax", "ChimeraX", lambda q: _exe_ok(_normalise_exe(q))),
                    ("chimera", "Chimera", lambda q: _exe_ok(_chimera_exe(q)) and not _is_chimerax(
                        _chimera_exe(q)))):
                p = d.get(key)
                if not p or not isinstance(p, str):
                    continue
                if works(p):
                    self.v[key].set(p)
                else:
                    self.say(f"(the remembered {name} path {p} no longer works; "
                             f"searching for {name} instead)\n")

        def save_prefs(self):
            try:
                GUI_PREFS.write_text(json.dumps({"chimerax": self.v["chimerax"].get(),
                                                 "chimera": self.v["chimera"].get()}))
            except Exception:
                pass

        def browse_pdb(self):
            p = filedialog.askopenfilename(
                title="Structure", filetypes=[("PDB / mmCIF", "*.pdb *.ent *.cif *.mmcif *.gz"),
                                              ("all files", "*")])
            if p:
                self.v["structure"].set(p)
                self.detect(fill_parts=True)

        def browse_out(self):
            p = filedialog.askdirectory(title="Output folder")
            if p:
                self.v["out"].set(p)

        def browse_cx(self):
            p = filedialog.askopenfilename(title="ChimeraX (the .app or its Contents/MacOS/ChimeraX)")
            if p:
                self.v["chimerax"].set(_normalise_exe(p))

        def browse_chimera(self):
            p = filedialog.askopenfilename(title="UCSF Chimera (the .app or its Contents/MacOS/chimera)")
            if p:
                self.v["chimera"].set(_chimera_exe(p))

        def detect(self, fill_parts=True):
            p = self.v["structure"].get().strip()
            if not p:
                return
            p = str(Path(p).expanduser())
            if p != self.v["structure"].get():
                self.v["structure"].set(p)
            try:
                info = read_structure(p, not self.v["keep_solvent"].get())
            except Exception as e:
                self.error(f"Could not read {p}:\n{e}")
                return
            self.chains = [c["id"] for c in info["chains"]]
            self.chain_rows = info["chains"]
            self.tree.delete(*self.tree.get_children())
            for c in info["chains"]:
                types = ", ".join(f"{k}×{n}" for k, n in sorted(c["resnames"].items()))
                self.tree.insert("", tk.END, text=c["id"] if c["id"].strip() else "(blank)",
                                 values=(c["atoms"], c["residues"], f"{c['first']}–{c['last']}",
                                         types))
            self.tree.configure(height=min(max(len(self.chains), 3), 10))
            self.chain_info.set(f"{len(self.chains)} chain(s), {info['n_models']} model(s), "
                                f"{info['fmt']}")
            if fill_parts:
                self.v["parts"].set(" ".join(c for c in self.chains if c.strip()))
            self.update_names()
            self.fit_window(initial=False)

        def resolved_names(self):
            """(tag, out) the CLI will use: blank fields follow the naming
            convention <stem>_molmap<res>_split in the current folder."""
            st = self.v["structure"].get().strip()
            try:
                res = float(self.v["resolution"].get())
            except ValueError:
                res = None
            tag = self.v["tag"].get().strip()
            tag = safe_tag(tag) if tag else (default_tag(st, res) if st and res else "")
            out = self.v["out"].get().strip()
            out = str(Path(out).expanduser()) if out else (str(Path.cwd() / tag) if tag else "")
            return tag, out

        def update_names(self):
            tag, out = self.resolved_names()
            blank = not (self.v["tag"].get().strip() and self.v["out"].get().strip())
            self.auto_name.set((f"{'blank = default:  ' if blank else ''}{out}/{tag}_p1_… .stl"
                                if tag else "blank tag / folder = <stem>_molmap<res>_split"))

        def update_swatches(self):
            """One swatch per part (the Parts field, or the detected chains
            while it is blank; each line of Selections when the parts are
            atom selections) in the colour the previews give it."""
            for w in self.swatch_box.winfo_children():
                w.destroy()
            self.swatches = []
            try:
                pal = part_colors(self.v["palette"].get())
            except ValueError:
                pal = PALETTE
            if self.v["parts_by"].get() == "selections":
                tokens = [selection_line_label(x) for x in self.selection_lines()]
            else:
                text = re.sub(r"\s*([,+])\s*", r"\1", self.v["parts"].get().replace("|", " "))
                tokens = text.split() or [c for c in self.chains if c.strip()]
            if not tokens:
                ttk.Label(self.swatch_box, text="each part's colour in the previews "
                                                "(the STLs carry none)",
                          foreground="#666").pack(side=tk.LEFT)
                return
            for i, tok in enumerate(tokens[:SWATCH_MAX], 1):
                name, rgb = pal[(i - 1) % len(pal)]
                code = "#%02x%02x%02x" % tuple(int(255 * x) for x in rgb)
                dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.55
                label = f"P{i} {tok if len(tok) <= 8 else tok[:7] + '…'}"
                tk.Label(self.swatch_box, text=label, bg=code, fg="#fff" if dark else "#000",
                         relief="solid", borderwidth=1, padx=4).pack(side=tk.LEFT, padx=(0, 3))
                self.swatches.append((f"{label} {name}", code))
            if len(tokens) > SWATCH_MAX:
                ttk.Label(self.swatch_box, text=f"+{len(tokens) - SWATCH_MAX} more",
                          foreground="#666").pack(side=tk.LEFT)

        def build_args(self):
            g = {k: v.get() for k, v in self.v.items()}
            g = {k: (x.strip() if isinstance(x, str) else x) for k, x in g.items()}
            if not g["structure"]:
                raise ValueError("choose a structure file first")
            nums = {}
            for key, _, _, _ in GUI_NUMBERS:
                if key == "k_list" or g[key] == "":
                    continue
                try:
                    nums[key] = float(g[key])
                except ValueError:
                    raise ValueError(f"'{g[key]}' is not a number ({key})")
            if g["k_list"]:
                parse_k_list(g["k_list"])
            errs = check_numbers(nums)
            if errs:
                raise ValueError("\n".join(errs))
            args = [str(Path(g["structure"]).expanduser())]
            if g["parts_by"] == "selections":
                lines = self.selection_lines()
                try:
                    parse_selection_parts(lines, self.syntax())
                except SplitError as e:
                    raise ValueError(f"Selections: {e}")
                args += [f"--part={x}" for x in lines]
                if self.syntax() != "chimerax":
                    args.append(f"--selection-syntax={self.syntax()}")
                    if g["chimera"]:
                        args.append(f"--chimera={g['chimera']}")
                if self.overlap() != "first":
                    args.append(f"--overlap={self.overlap()}")
            elif g["parts"]:
                args += ["--parts"] + g["parts"].replace("|", " ").split()
            # --flag=VALUE so a value that starts with '-' still parses
            for key, flag in (("out", "--out"), ("tag", "--tag"), ("chimerax", "--chimerax")):
                if g[key]:
                    args.append(f"{flag}={g[key]}")
            for key, _, _, _ in GUI_NUMBERS:
                if g[key] != "" and (key != "blob_skin" or g["blobs"]):
                    args.append(f"{GUI_FLAG_NAMES.get(key, '--' + key.replace('_', '-'))}={g[key]}")
            for key, _, _ in GUI_OPTIONS:
                if g[key]:
                    args.append("--" + key.replace("_", "-"))
            if check_palette(g["palette"]) != "default":     # a menu, not a number field
                args.append(palette_option(g["palette"]))
            return args

        # ------------------------------------------------ atom-selection parts
        def syntax(self):
            s = self.v["selection_syntax"].get().strip().lower()
            return s if s in SELECTION_SYNTAXES else "chimerax"

        def overlap(self):
            given = self.v["overlap"].get().strip()
            return next((k for k, text in OVERLAP_LABELS.items() if given in (k, text)), "first")

        def selection_lines(self):
            return [x.strip() for x in self.sel_text.get("1.0", tk.END).splitlines() if x.strip()]

        def set_selections(self, lines):
            self.sel_text.delete("1.0", tk.END)
            self.sel_text.insert("1.0", "\n".join(lines))
            self.update_swatches()

        def on_selections_edited(self, _event=None):
            if self.sel_text.edit_modified():
                self.sel_text.edit_modified(False)
                self.update_swatches()

        def show_parts_mode(self, fit=True):
            """Show the Parts row for chains, the Selections rows for atom
            selections, and the Chimera row for the Chimera convention."""
            selections = self.v["parts_by"].get() == "selections"
            shown = {"parts": not selections, "part": selections,
                     "chimera": selections and self.syntax() == "chimera"}
            for key, show in shown.items():
                for w in self.rows[key]:
                    if show:
                        w.grid()
                    else:
                        w.grid_remove()
            self.update_swatches()
            if fit:
                self.fit_window(initial=False)

        def fill_backbone(self):
            """Selections: each chain's backbone, bb<chain>, then rest."""
            if not self.chain_rows and self.v["structure"].get().strip():
                self.detect(fill_parts=not self.v["parts"].get())
            if not self.chain_rows:
                self.error("Choose a structure first: Backbone per chain takes its chains.")
                return
            lines = backbone_parts(self.chain_rows, self.syntax())
            if not lines:
                self.error("No chain with an ID has a nucleotide or amino-acid residue, so there "
                           "is no backbone to select.")
                return
            old = self.selection_lines()
            if old and old != lines and not selftest and not messagebox.askyesno(
                    TOOL_NAME, "Replace the selections with each chain's backbone, then rest?"):
                return
            self.set_selections(lines)
            self.v["parts_by"].set("selections")

        def show_cmd(self):
            try:
                args = self.build_args()
            except ValueError as e:
                self.error(str(e))
                return
            cmd = " ".join(shlex.quote(x) for x in self_command() + args)
            self.say("\n$ " + cmd + "\n")
            self.root.clipboard_clear()
            self.root.clipboard_append(cmd)
            self.status.set("command copied to the clipboard")

        # ---------------------------------------------------------- running
        def run(self):
            if self.proc is not None:
                return
            try:
                args = self.build_args()
            except ValueError as e:
                self.error(str(e))
                return
            if not selftest:
                self.save_prefs()
            cmd = self_command(unbuffered=True) + args
            # echo the command as Show command gives it (without -u)
            self.say("\n$ " + " ".join(shlex.quote(x) for x in self_command() + args) + "\n")
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, bufsize=1, env=env,
                                             start_new_session=True)
            except Exception as e:
                self.say(f"[failed to start: {e}]\n")
                self.proc = None
                return
            self.last_out = self.resolved_names()[1] or None
            self.last_tag = None
            self.run_started = time.time()
            self.stopped = False
            self.run_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self.status.set("running …")
            threading.Thread(target=self.pump, args=(self.proc,), daemon=True).start()

        def pump(self, p):              # worker thread: only touches the queue
            for line in p.stdout:
                self.q.put(("line", line))
            p.wait()
            self.q.put(("done", p.returncode))

        def poll(self):
            try:
                while True:
                    kind, payload = self.q.get_nowait()
                    if kind == "line":
                        self.say(payload)
                        m = re.match(r"(output|tag)\s+(.*)$", payload.strip())
                        if m and m.group(1) == "output":
                            self.last_out = m.group(2).strip()
                        elif m:
                            self.last_tag = m.group(2).strip()
                    else:
                        self.proc = None
                        self.run_btn.configure(state=tk.NORMAL)
                        self.stop_btn.configure(state=tk.DISABLED)
                        self.say(f"[exit code {payload}]\n")
                        if self.stopped or payload in (EXIT_STOPPED, -2, -15, 143):
                            msg = "stopped"
                        else:
                            msg = {EXIT_OK: "done — all checks pass",
                                   EXIT_FAIL: "done — some checks FAILED (see the report)",
                                   EXIT_USAGE: "not run — invalid options (see the log)"}.get(
                                payload, f"FAILED (exit {payload}) — read the log")
                        self.status.set(msg)
                        self.on_done(payload)
            except queue.Empty:
                pass
            self.root.after(120, self.poll)

        def on_done(self, rc):
            if rc in (EXIT_OK, EXIT_FAIL) and not self.stopped and \
                    not self.v["no_preview"].get() and not selftest:
                self.show_preview(fresh_only=True)

        def stop(self):
            if self.proc is not None:
                self.stopped = True
                try:                    # SIGINT: the pipeline cleans up and keeps its logs
                    os.killpg(self.proc.pid, signal.SIGINT)
                except Exception:
                    self.proc.send_signal(signal.SIGINT)
                self.say("[stopping …]\n")

        def on_close(self):
            if self.proc is not None:
                if not messagebox.askyesno(TOOL_NAME,
                                           "A split is running. Stop it and quit?"):
                    return
                self.stop()
                try:
                    self.proc.wait(timeout=15)
                except Exception:
                    pass
            self.root.destroy()

        def out_dir(self):
            return Path(self.last_out or self.resolved_names()[1] or ".")

        def open_out(self):
            d = self.out_dir()
            if d.exists():
                opener = "open" if sys.platform == "darwin" else (
                    "explorer" if os.name == "nt" else "xdg-open")
                subprocess.Popen([opener, str(d)])
            else:
                self.status.set(f"{d} does not exist yet")

        def show_preview(self, fresh_only=False):
            tag = self.last_tag or self.resolved_names()[0]
            png = self.out_dir() / f"{tag}_preview.png"
            if not png.exists() or (fresh_only and png.stat().st_mtime < self.run_started):
                self.status.set("no preview for this run")
                return
            top = tk.Toplevel(self.root)
            top.title(png.name)
            img = tk.PhotoImage(file=str(png))
            f = max(1, math.ceil(img.width() / (self.root.winfo_screenwidth() * 0.9)))
            if f > 1:
                img = img.subsample(f)
            lab = ttk.Label(top, image=img)
            lab.image = img
            lab.pack()

    root = tk.Tk()
    set_optional_window_icon(root, tk, ["multicolor_split_icon.png", "icon.png"],
                             "_multicolor_split_icon_image")
    if selftest:
        root.withdraw()                 # before anything can pop up
    try:
        ttk.Style().theme_use("aqua" if sys.platform == "darwin" else "clam")
    except tk.TclError:
        pass
    app = App(root)
    res = {"rc": 0}
    if selftest:
        root.update_idletasks()
        print("[selftest] chips: " + " ".join(sorted(set(chip_keys))), flush=True)
        print("[selftest] palette: %s; swatches: %s"
              % (app.v["palette"].get(),
                 " | ".join(f"{t} {c}" for t, c in app.swatches) or "none"), flush=True)
        print("[selftest] parts by: %s; convention %s; shared atoms %s; rows shown: %s"
              % (app.v["parts_by"].get(), app.syntax(), app.overlap(),
                 " ".join(k for k in ("parts", "part", "chimera") if app.rows[k][0].winfo_manager())),
              flush=True)
        m = re.match(r"(\d+)x(\d+)", root.geometry())
        gw, gh = (int(m.group(1)), int(m.group(2))) if m else app.size
        print("[selftest] window: %dx%d geometry %dx%d"
              % (root.winfo_reqwidth(), root.winfo_reqheight(), gw, gh), flush=True)

        def help_popups():
            n, key = 0, None
            try:
                for key in HELP:
                    show_help(key, chip_widgets.get(key, root))
                    open_help["win"].withdraw()     # built and laid out, never shown
                    root.update_idletasks()
                    open_help["win"].destroy()
                    open_help["win"] = None
                    n += 1
            except Exception as e:
                print(f"[selftest] help popup {key!r} failed: {type(e).__name__}: {e}", flush=True)
                res["rc"] = 1
            else:
                print(f"[selftest] help popups: {n} ok", flush=True)
            root.destroy()

        def go():
            if selftest == "help":
                help_popups()
                return
            if selftest == "fill":
                app.fill_backbone()
                print("[selftest] selections: " + " | ".join(app.selection_lines()), flush=True)
                print("[selftest] swatches: " + (" | ".join(f"{t} {c}" for t, c in app.swatches)
                                                 or "none"), flush=True)
            try:
                args = app.build_args()
                print("[selftest] args:", " ".join(shlex.quote(x) for x in args), flush=True)
            except Exception as e:
                print("[selftest] build_args failed:", e, flush=True)
                if selftest == "run":           # nothing was run: invalid options
                    res["rc"] = EXIT_USAGE
                root.destroy()
                return
            if selftest == "run":
                app.run()
                wait()
            else:
                root.destroy()

        def wait():
            if app.proc is None and app.run_btn.instate(["!disabled"]):
                text = app.log.get("1.0", tk.END)
                m = re.findall(r"\[exit code (-?\d+)\]", text)
                rc = int(m[-1]) if m else None
                print("[selftest] log tail:\n" + "\n".join(text.strip().splitlines()[-12:]), flush=True)
                print(f"[selftest] status: {app.status.get()}", flush=True)
                print(f"[selftest] exit code {rc}", flush=True)
                # no exit code: it never started; a negative one: a signal
                res["rc"] = EXIT_ERROR if rc is None else (EXIT_STOPPED if rc < 0 else rc)
                root.destroy()
            else:
                root.after(500, wait)

        root.after(300, go)
    root.mainloop()
    return res["rc"]


# ======================================================================= CLI
def build_parser():
    ap = argparse.ArgumentParser(
        prog="multicolor_split.py",
        description="Split a structure's chains into molmap solids that tile the whole "
                    "molmap surface exactly, for multicolour 3D printing.  Run with no "
                    "arguments (or --gui) for the GUI.",
        epilog="Exit codes: 0 pass, 3 exported but a check FAILED, 1 error, 2 usage error, "
               "130 stopped.  Method, defaults and validation: see the top of this file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("structure", nargs="?",
                    help="PDB or mmCIF file (a PDB may be gzipped; an mmCIF may not)")
    ap.add_argument("--gui", action="store_true", help="open the GUI (pre-filled from the other options)")
    g = ap.add_argument_group("what to split")
    g.add_argument("--parts", nargs="+", metavar="CHAINS",
                   help="one token per colour part, in peeling order; join chains with ',' "
                        "(e.g. --parts A C, --parts A,B C).  Default: every chain its own part")
    g.add_argument("--chains", help="shorthand: comma-separated chains, one part each")
    g.add_argument("--part", action="append", metavar="SPEC", default=None,
                   help="one part as an atom selection instead of whole chains, given once per "
                        "colour in peeling order: an atom specification in ChimeraX's convention "
                        "(or Chimera's, with --selection-syntax chimera), optionally named as "
                        "NAME=SPEC for the file name, or rest for every atom no other part takes; "
                        "e.g. --part 'bbA=/A & backbone' --part 'bbB=/B & backbone' --part rest")
    g.add_argument("--selection-syntax", choices=SELECTION_SYNTAXES, default="chimerax",
                   help="the convention --part is written in: chimerax (/A:12@P, read by "
                        "ChimeraX) or chimera (:12.A@P, read by UCSF Chimera, which must be "
                        "installed)")
    g.add_argument("--overlap", choices=OVERLAP_RULES, default="first",
                   help="atoms two --part selections share: first gives each to the earlier part "
                        "and says so in the log and report; error refuses the run")
    g.add_argument("--keep-solvent", action="store_true", help="include waters in the maps")
    g.add_argument("--no-hydrogens", action="store_true", help="leave hydrogens out of the maps")
    g = ap.add_argument_group("molmap")
    g.add_argument("--resolution", type=float, default=DOC_RESOLUTION, help="molmap resolution, A")
    g.add_argument("--grid", type=float, default=DOC_GRID, help="molmap gridSpacing, A")
    g.add_argument("--dust", type=float, default=DEFAULT_DUST,
                   help="'surface dust' size, A: removes specks and fills enclosed cavities smaller "
                        "than this; never removes the whole's largest piece, a piece carrying a "
                        "part's main piece, or a piece of a part's field")
    g = ap.add_argument_group("split")
    g.add_argument("--k", type=float, default=None,
                   help="penalty slope K (default 16 at res 4 / grid 0.5; otherwise "
                        "K0 = 16*(res/4)*(0.5/grid), checked by a scan)")
    g.add_argument("--k-scan", action="store_true", help="scan K even at the documented point")
    g.add_argument("--k-list", default=None, help="comma-separated K candidates (implies --k-scan)")
    g.add_argument("--level", type=float, default=None, help="pin the contour level (no scan)")
    g.add_argument("--level-guess", type=float, default=None,
                   help="centre of the level scan (default: molmap auto level, 3 decimals)")
    g.add_argument("--level-span", dest="span", type=float, default=DEFAULT_SPAN,
                   help="half-width of the contour-level scan (contour units)")
    g.add_argument("--level-step", dest="step", type=float, default=DEFAULT_STEP,
                   help="contour-level scan increment (NOT the voxel step: surfaces are "
                        "always computed at voxel step 1)")
    g.add_argument("--span", dest="span", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    g.add_argument("--step", dest="step", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    g.add_argument("--no-widen", action="store_true", help="do not widen a failed scan")
    g.add_argument("--allow-multi-shell", action="store_true",
                   help="accept parts made of several separate pieces")
    g.add_argument("--blobs", action="store_true",
                   help="blob parts: every part but the last is its own molmap surface at the "
                        "contour level (its atoms' density alone), clipped to the whole, and the "
                        "last part is the whole minus them -- for parts that are separate blobs, "
                        "such as phosphate groups.  No K is used (--k, --k-scan and --k-list are "
                        "ignored)")
    g.add_argument("--blob-skin", type=float, default=None,
                   help="with --blobs: where the last part would cover a blob with a skin estimated "
                        "thinner than this (A) and the blob's own density exceeds that of all the other atoms together, "
                        "the blob takes the skin, so it reaches the surface instead of lying a hair beneath it; 0 "
                        "keeps each blob exactly its own molmap surface (0 to 1; default "
                        f"{DEFAULT_BLOB_SKIN:g})")
    g = ap.add_argument_group("output")
    g.add_argument("-o", "--out", default=None,
                   help="output folder (default ./<tag>, i.e. ./<stem>_molmap<res>_split)")
    g.add_argument("--tag", default=None,
                   help="file prefix (default <structure stem>_molmap<resolution>_split)")
    g.add_argument("--scale", type=float, default=1.0, help="mm per Angstrom in the STLs")
    g.add_argument("--lay-flat", action="store_true",
                   help="rotate everything so the shortest principal axis is vertical")
    g.add_argument("--keep-scan", action="store_true",
                   help="keep every candidate STL in <tag>_work/ (or <tag>_work_failed/)")
    g.add_argument("--no-preview", action="store_true", help="skip the preview PNG / GLB")
    g.add_argument("--keep-pinch-edges", action="store_true",
                   help="write the vertex copies at pinch points as one point, as v1.4.1 did, so "
                        "an earlier set is rebuilt byte for byte; slicers that weld equal "
                        "coordinates, Bambu Studio among them, then report non-manifold edges")
    g.add_argument("--palette", type=palette_arg, default="default",
                   help="part colours of the preview PNG / GLB, named in the log and report "
                        "(the STLs carry no colour): default (this tool's own: blue, orange, "
                        "green, red, purple, teal, yellow, brown) or DiLiuLab, the lab's nine "
                        "figure colours from gr_colors (red, blue, magenta, cyan, orange, "
                        "purple, green, yellow, mint green, repeating after nine; the neutral "
                        "is not used), optionally at a lighter tint: DiLiuLab-T80, -T60, -T40 "
                        "or -T20 (default tint T100; T40 and T20 are pale on white). Read from "
                        "assets/diliulab_colors.json")
    g.add_argument("--chimerax", default=None, help="path to the ChimeraX executable (or its .app)")
    g.add_argument("--chimera", default=None,
                   help="path to UCSF Chimera (or its .app), for --selection-syntax chimera")
    ap.add_argument("--gui-selftest", choices=("build", "fill", "run", "help"), help=argparse.SUPPRESS)
    ap.add_argument("--version", action="version", version=__version__)
    return ap


def run_cli(args):
    """Run the split for parsed options (from build_parser); returns the exit
    code.  SIGTERM is handled like Ctrl-C: a clean shutdown that removes the
    staging folder and keeps the logs."""
    def _stop(signum, frame):           # SIGTERM -> clean shutdown (temp dir, logs)
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _stop)
    except (ValueError, OSError):
        pass
    try:
        return pipeline(args)
    except SplitError as e:
        log(f"\nERROR: {e}")
        return EXIT_ERROR
    except OSError as e:
        log(f"\nERROR: {e}")
        return EXIT_ERROR
    except KeyboardInterrupt:
        log(f"\nstopped -- nothing half-written was left; logs are in "
            f"{LAST_LOG_DIR or '<output>/<tag>_work_failed/'}")
        return EXIT_STOPPED


def main(argv=None):
    """Command-line entry point; returns the exit code.  No arguments, --gui
    or --gui-selftest open the GUI, pre-filled from the other options."""
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    a = ap.parse_args(argv)
    # the common slip `--parts A C model.pdb`: argparse puts the file into --parts
    if a.structure is None and a.parts and \
            (Path(a.parts[-1]).expanduser().is_file() or
             re.search(r"\.(pdb|ent|cif|mmcif)(\.gz)?$", a.parts[-1], re.I)):
        a.structure = a.parts.pop()
        if not a.parts:
            a.parts = None
    nums = {k: getattr(a, k) for k in ("resolution", "grid", "dust", "k", "level",
                                       "level_guess", "span", "step", "scale", "blob_skin")}
    errs = check_numbers(nums)
    if a.k_list:
        try:
            parse_k_list(a.k_list)
        except ValueError as e:
            errs.append(f"--k-list: {e}")
    if errs:
        ap.error("; ".join(errs))
    if not argv or a.gui or a.gui_selftest:
        pre = {k: getattr(a, k) for k in ("structure", "out", "tag", "resolution", "grid",
                                          "dust", "k", "k_list", "level", "level_guess", "span",
                                          "step", "scale", "chimerax", "lay_flat", "k_scan",
                                          "allow_multi_shell", "no_widen", "keep_solvent",
                                          "no_hydrogens", "keep_scan", "no_preview", "palette",
                                          "selection_syntax", "overlap", "chimera", "blobs",
                                          "blob_skin")}
        if a.part:
            pre["part"] = "\n".join(a.part)
            pre["parts_by"] = "selections"
        if a.parts:
            pre["parts"] = " ".join(a.parts)
        elif a.chains:
            pre["parts"] = " ".join(c for c in re.split(r"[,\s]+", a.chains) if c)
        return run_gui(prefill=pre, selftest=a.gui_selftest)
    if not a.structure:
        ap.error("a structure file is required (or run with no arguments for the GUI)")
    return run_cli(a)


if __name__ == "__main__":
    sys.exit(main())
