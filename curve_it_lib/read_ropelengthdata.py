#!/usr/bin/env python3
"""
read_ropelengthdata.py -- read tight knots and links from Jason Cantarella's
ropelength data set and write them in Curve It's curve formats.

The data set is a folder of tight, ropelength-minimised knots and links: one
<name>_best.vect per entry, a Geomview VECT whose negative vertex counts mark
every component closed, one <name>_curvature.txt per entry, and an index,
ropelength_data.txt.  The index header names the two papers the data come
from, "Knot Tightening By Constrained Gradient Descent" by Ashton, Cantarella,
Piatek and Rawdon, and "The Shapes of Tight Composite Knots" by Cantarella,
La Pointe and Rawdon.  Each index row gives an entry's polygonal ropelength,
ropelength bound, number of edges, residual and writhe.

NAMES.  3_1 is a knot (crossings_index), 6_3_2 a link
(crossings_components_index), and 3_1_#_3_1_m a connected sum, _m marking a
summand drawn as its mirror image.  Entries are listed in crossing-number
order, a sum counting the crossings of all its summands.

UNITS.  The data set is scaled to tube radius 0.5: the tight rope around a
curve is 1 thick, twice a curve's length is its polygonal ropelength, and the
strands come as close as 1.0 to one another.  --scale multiplies every
coordinate, so --scale 20 makes the rope 20 thick.

The shapes are written as the data set holds them: no vertex is moved,
resampled or smoothed.  Closure comes from the VECT file, never from a guess.
Each entry is centred on the mean of its vertices unless --keep-origin is
given, since the data set leaves some curves hundreds of units out.

The folder is not bundled.  It is given with --data, $ROPELENGTHDATA, or
Locate data folder... in the window, which remembers it for next time.

    python3 read_ropelengthdata.py                                    # GUI
    python3 read_ropelengthdata.py --data DIR --check                 # is the folder usable?
    python3 read_ropelengthdata.py --data DIR --list --prime          # names, by crossings
    python3 read_ropelengthdata.py --data DIR 3_1 6_3_2 -o out
    python3 read_ropelengthdata.py --data DIR '3_1_#_3_1_m' --to coordinate,vect,fake-pdb -o out
    python3 read_ropelengthdata.py --data DIR 5_2_1 --to glb -d 0.618 -o out
    python3 read_ropelengthdata.py --data DIR 3_1 --scale 20 -o out
    python3 read_ropelengthdata.py --data DIR 6_3_2 --preview --palette DiLiuLab

Requires numpy; the rod models (STL, GLB) also need trimesh, as XYZ to 3D
Model does.  Run with no arguments, or with --gui, for the graphical
interface.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

# Every curve in the data set is a Geomview VECT, the one curve format that
# states closure, so vect_io.py is what reads it.
try:
    from . import vect_io
except ImportError:
    try:
        import vect_io  # type: ignore[no-redef]
    except ImportError:
        vect_io = None  # type: ignore[assignment]

# Optional DiLiuLab palette for a link's component colours.  Only --palette
# and the window's palette menu read it; without lab_colors.py each format
# keeps its own colours and nothing else is affected.
try:
    from . import lab_colors
except ImportError:
    try:
        import lab_colors  # type: ignore[no-redef]
    except ImportError:
        lab_colors = None  # type: ignore[assignment]

__version__ = "1.0"

TOOL_NAME = "Read ropelengthdata"

# The key the tool goes by in Curve It: the help topic, the icon and the
# hidden --read-ropelengthdata-gui flag all use it.
TOOL_KEY = "read_ropelengthdata"


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


# Decimal places per coordinate; matches curve_it.write_plain_xyz_curve.  The
# data set's shortest edge is 0.00635, so six decimals keep every vertex
# distinct at any scale of 1 or more.
DEFAULT_PRECISION = 6
MAX_PRECISION = 12

DEFAULT_OUTDIR = "ropelength_xyz"

ENV_VAR = "ROPELENGTHDATA"
INDEX_FILE = "ropelength_data.txt"
CURVE_SUFFIX = "_best.vect"

# The data set is scaled to tube radius 0.5.  Measured on all 943 entries:
# the closed polygon length is ropelength / 2 to within 4.2e-5, and the
# closest approach of non-neighbouring strands is 0.99992-1.00000.  So the
# tight rope has diameter 1 in the data set's own units, and a scale factor s
# makes it s thick.
TUBE_RADIUS = 0.5
ROPE_DIAMETER = 2.0 * TUBE_RADIUS

# Where Locate data folder... remembers its folder: the user's own home, never
# anything inside Curve It, since the folder is a path on one machine.
PREFS_PATH = os.path.join(os.path.expanduser("~"), ".read_ropelengthdata_gui.json")

# Format names follow curve_it.py's --convert-to choices, plus the two rod
# models XYZ to 3D Model makes.
FORMATS = ("coordinate", "vect", "fake-pdb", "molecular", "glb", "stl")
FORMAT_LABELS = {
    "coordinate": "coordinate XYZ (.xyz)",
    "vect": "Geomview VECT (.vect)",
    "fake-pdb": "fake PDB (.pdb)",
    "molecular": "molecular XYZ (_molecular.xyz)",
    "glb": "rod model GLB (.glb)",
    "stl": "rod model STL (.stl)",
}
FORMAT_ALIASES = {"xyz": "coordinate", "plain": "coordinate", "pdb": "fake-pdb",
                  "fake_pdb": "fake-pdb", "fakepdb": "fake-pdb"}
DEFAULT_FORMATS = ("coordinate",)
MODEL_FORMATS = ("glb", "stl")

# The reflection negates exactly one coordinate.  z leaves the view down z
# alone and swaps every crossing over for under, as KnotPlot to XYZ does.
MIRROR_AXIS = "z"

PREVIEW_LIMIT = 6
REPORT_LIMIT = 40

# A fake PDB coordinate is %8.3f: from -999.999 to 9999.999.
PDB_LOW, PDB_HIGH = -999.9995, 9999.9995

# How the window and the command line refer to the same settings.  A message
# shared between them is phrased in its reader's vocabulary: the window has
# fields, not flags.
LOCATED_LABEL = "the located folder"
CLI_LABEL = "--data"
CLI_TERMS = {
    "scale": "--scale",
    "diameter": "--diameter",
    "precision": "--precision",
    "outdir_fix": "give -o another folder",
    "locate": "give the folder with --data DIR, or set $%s" % ENV_VAR,
}
GUI_TERMS = {
    "scale": "scale",
    "diameter": "rod diameter",
    "precision": "precision",
    "outdir_fix": "put another folder in the output field",
    "locate": "use Locate data folder... and pick it",
}

VECT_MISSING = ("reading the data set needs vect_io.py from curve_it_lib; make "
                "sure that file sits beside this one")


# --------------------------------------------------------------------------
# GUI help text: key -> (title, prose, example or None)
# --------------------------------------------------------------------------
HELP = {
    "data": (
        "The data set's folder",
        "This tool reads the ropelength data set Jason Cantarella shares: a "
        "folder of tight knots and links, one <name>_best.vect file each, "
        "with an index called ropelength_data.txt and one "
        "<name>_curvature.txt per entry. The data set is not bundled with "
        "Curve It, so the tool has to be told where the folder is.\n\n"
        "It looks first at the folder picked with Locate data folder..., "
        "then at the ROPELENGTHDATA environment variable, then at the folder "
        "it remembered the last time one was located. Locate data folder... "
        "remembers its folder in .read_ropelengthdata_gui.json in your home "
        "folder; nothing about it is written into Curve It.\n\n"
        "A folder counts as the data set when it holds at least one "
        "<name>_best.vect. The folder above it works too, when that holds a "
        "folder called ropelengthdata. The index is optional: without it "
        "every curve still reads, and only the ropelength figures are "
        "missing.",
        "in this window\n"
        "  Locate data folder...  -> the folder holding 3_1_best.vect\n"
        "\n"
        "or from a shell\n"
        "  export ROPELENGTHDATA=/path/to/ropelengthdata\n"
        "  python3 read_ropelengthdata.py --check"),
    "names": (
        "What to read",
        "A list of entry names, separated by spaces, spelled as the data set "
        "spells its files without _best.vect.\n\n"
        "3_1 is a knot: 3 crossings, number 1 in the knot table. 6_3_2 is a "
        "link: 6 crossings, 3 components, number 2. Names joined by _#_ are "
        "connected sums, such as 3_1_#_3_1_m, the square knot; _m marks a "
        "summand drawn as its mirror image. _r marks a summand with its "
        "orientation reversed, and _rm both. They appear only on 8_17, 9_32 "
        "and 9_33, the knots here that are not invertible, which is why "
        "reversing one gives a different sum. TreeA and TreeB tell apart two "
        "ways of joining the same links into one sum.\n\n"
        "A dotted name such as 3.1, KnotPlot's spelling, is read as 3_1. Use "
        "Browse data set... to pick names from the list instead of typing "
        "them.",
        "3_1 6_3_2 3_1_#_3_1_m\n"
        "  3_1          the trefoil, 2400 points\n"
        "  6_3_2        the Borromean rings, 3 components\n"
        "  3_1_#_3_1_m  the square knot"),
    "catalogue": (
        "Browse data set..., and the order",
        "Lists every entry whose <name>_best.vect is in the folder, with its "
        "kind, crossing number, number of components and polygonal "
        "ropelength.\n\n"
        "The list is in crossing-number order. A connected sum counts the "
        "crossings of all its summands, so 3_1_#_3_1_m sits with the "
        "6-crossing entries. Within one crossing number come the knots, then "
        "the links by their number of components, and within each of those "
        "the prime entries before the sums.\n\n"
        "The figures come from ropelength_data.txt: the polygonal "
        "ropelength, the ropelength bound, the number of edges (here always "
        "the number of vertices), the residual and the writhe. For a link "
        "the writhe is that of the whole link, the components' own writhes "
        "plus twice their linking numbers. 2_2_1, 8_4_3 and 9_2_38 have no "
        "writhe in the index.",
        "2_2_1    link   2 crossings  2 components  ropelength 25.1415\n"
        "3_1      knot   3 crossings  1 component   ropelength 32.7437\n"
        "4_1      knot   4 crossings  1 component   ropelength 42.0971\n"
        "4_2_1    link   4 crossings  2 components\n"
        "..."),
    "prime": (
        "Prime knots and links only",
        "Keeps the prime entries, the ones whose name holds no _#_, and "
        "drops the connected sums. In the data set as shared that leaves "
        "the 249 prime knots up to 10 crossings and the 130 prime links up "
        "to 9 crossings, 379 of its 943 entries.\n\n"
        "It narrows both the Browse list and Entire data set.",
        None),
    "all": (
        "Entire data set",
        "Read every entry in the folder in one go, in the list's order, and "
        "write each one in every format ticked. With prime knots and links "
        "only ticked, just the prime entries are read.\n\n"
        "The whole data set is 943 entries and about 690,000 vertices. Coordinate XYZ "
        "for all of them takes a few seconds and the window waits until it "
        "is done. Rod models keep every vertex, so a GLB of every entry runs "
        "to hundreds of megabytes; tick them for a few entries rather than "
        "for all.",
        None),
    "scale": (
        "Scale, and the thickness of the rope",
        "The data set is scaled to tube radius 0.5, so the tight rope around "
        "every curve is 1 thick and twice a curve's length is its polygonal "
        "ropelength: 3_1 is 16.3715 long, ropelength 32.7437. Its strands "
        "come as close as 1.0 to one another, which is what tight means "
        "here.\n\n"
        "Scale multiplies every coordinate, so it is also the thickness of "
        "the rope in the output's units. To model a tight knot in DNA, scale "
        "by the diameter you want the duplex to have, in Å: about 20 for "
        "B-DNA itself, more to allow for the ions around it. The shape, "
        "the ropelength and the writhe do not change; the length grows with "
        "the scale.",
        "scale 1    the rope is 1 thick, 3_1 is 16.37 long\n"
        "scale 20   the rope is 20 Å thick, 3_1 is 327.4 Å long,\n"
        "           about 96 base pairs of B-DNA at 3.4 Å per pair"),
    "centre": (
        "Centre each entry",
        "The data set keeps each curve where its tightening left it, and "
        "some lie far from the origin: the mean of 10_113's vertices is 877 "
        "units out. Centring moves each entry so the mean of its vertices is "
        "at the origin, before scaling. It is a translation, so the shape, "
        "the length and the writhe are unchanged, and the header records "
        "the shift.\n\n"
        "Untick it to keep the data set's own coordinates, for example to "
        "lay an output over the original file. Every format of one entry is "
        "written in the same frame either way, rod models included.",
        None),
    "mirror": (
        "Mirror image",
        "Writes the reflection of each entry instead of the entry itself: z "
        "becomes -z, which leaves the view down z alone and swaps every "
        "crossing over for under. A reflection negates the writhe and every "
        "linking number, so it gives the mirror image of a knot or link.\n\n"
        "The data set holds each prime knot in one handedness only: 3_1's "
        "writhe in the index is -3.416633, and its mirror image measures "
        "+3.416633. Many sums come in both forms already, under names that "
        "differ only in _m, such as 3_1_#_3_1 and 3_1_m_#_3_1_m.\n\n"
        "Mirrored files are named <name>_mirror, and their header says "
        "so.",
        None),
    "formats": (
        "Formats",
        "Each ticked format is written for every entry read. Coordinate XYZ "
        "is Curve It's own curve file: x y z rows, a blank line between "
        "components, and a # header naming the entry, its ropelength and "
        "the scale. Load it as Curve It's Curve XYZ/txt input. Geomview VECT "
        "holds the same components and also states that each one is closed, "
        "which Curve It, View curve and XYZ to 3D Model all read.\n\n"
        "Fake PDB has one atom per vertex and one chain per component, with "
        "a LINK record closing each chain, for ChimeraX and PyMOL. Molecular "
        "XYZ holds one curve only, so a link's components are joined into "
        "one, and the report says so. These two are written by the same "
        "functions as Curve It's Convert XYZ.\n\n"
        "The rod models are tubes of the rod diameter around the curve, made "
        "by XYZ to 3D Model's own functions: GLB with one coloured material "
        "per component, STL as one solid. They keep every vertex of the data "
        "set, as XYZ to 3D Model's --segments 0 does, and lie in the same "
        "frame as the curve files.",
        "3_1 with coordinate XYZ, VECT and GLB ticked\n"
        "  3_1.xyz\n"
        "  3_1.vect\n"
        "  3_1-D1.0.glb    rod diameter 1.0, the tight rope"),
    "diameter": (
        "Rod diameter",
        "The thickness of the GLB and STL rods, in the output's units, after "
        "scaling. Leave it blank for the tight rope itself: 1 times the "
        "scale, the thickness the data set was tightened at.\n\n"
        "At that diameter the rods touch wherever the tight rope touches "
        "itself, and a link's components touch one another, so a print comes "
        "out as one piece. A smaller diameter, such as 0.618 times the "
        "scale, leaves gaps, so the strands can be seen apart. The diameter "
        "goes into the file name, so models of one entry at two diameters do "
        "not overwrite each other.",
        "blank            3_1-D1.0.glb    the tight rope\n"
        "0.618            3_1-D0.618.glb  thinner, strands apart\n"
        "scale 20, blank  3_1-D20.0.glb"),
    "split": (
        "One file per component",
        "A link is written to one file with a blank line between its "
        "components, and Curve It joins every component of a curve file "
        "into one path unless you select one. Tick this to write each "
        "component of a link to its own coordinate XYZ or molecular XYZ "
        "file instead, named _A, _B, _C after the entry.\n\n"
        "VECT, fake PDB and the rod models keep a link in one file either "
        "way, since each of them holds components apart. A knot has one "
        "component and is written as one file either way.",
        "6_3_2 with one file per component\n"
        "  6_3_2_A.xyz\n"
        "  6_3_2_B.xyz\n"
        "  6_3_2_C.xyz"),
    "precision": (
        "Decimal places",
        "How many decimals each coordinate is written with: 6 by default, as "
        "in the rest of Curve It. The data set's shortest edge is 0.00635, "
        "so six decimals keep every vertex distinct.\n\n"
        "Fake PDB always has three, which is what its columns hold.",
        None),
    "output": (
        "Output folder and file names",
        "Files are written to this folder, which is created if it does not "
        "exist. Each is named after its entry as the data set spells it, "
        "with the #: 3_1_#_3_1_m.xyz. The data set's own folder is never "
        "written to.\n\n"
        "Untick comment header to write coordinate XYZ and VECT without "
        "their # lines. Every Curve It reader skips those lines.",
        None),
    "palette": (
        "Component colours",
        "Colours a link's components in the Preview windows, in VECT files "
        "and on GLB rods. default leaves each one its own colours: the "
        "viewer's, VECT's opaque white, and XYZ to 3D Model's list.\n\n"
        "DiLiuLab is the lab's nine figure colours in the order red, blue, "
        "magenta, cyan, orange, purple, green, yellow, mint green, repeating "
        "after nine; DiLiuLab T80, T60, T40 and T20 are lighter tints mixed "
        "toward white. The colours come from lab_colors.py and "
        "assets/diliulab_colors.json. Coordinate XYZ, fake PDB, molecular "
        "XYZ and STL carry no colour.",
        "6_3_2 with DiLiuLab T80\n"
        "  component A  light red\n"
        "  component B  light blue\n"
        "  component C  light magenta"),
}


# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
PALETTE_MISSING = ("palette %r needs lab_colors.py from curve_it_lib; only "
                   "the default colours are available without it")


def check_palette(spec):
    """"default" or the lab palette's label, "DiLiuLab T80" and so on.

    Raises ValueError for an unknown palette, and for anything but the
    default when lab_colors.py is missing.
    """
    if lab_colors is not None:
        return lab_colors.check_palette(spec)
    if spec is None or str(spec).strip().lower() in ("", "default", "none", "auto"):
        return "default"
    raise ValueError(PALETTE_MISSING % spec)


def palette_argument(value):
    """argparse type for --palette: a bad palette is a usage error."""
    try:
        return check_palette(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def palette_choices():
    """What the window's palette menu offers."""
    if lab_colors is None:
        return ["default"]
    return lab_colors.palette_choices("default")


def _palette_token(label):
    """"DiLiuLab T80" as the viewer's and vect_io's spelling, DiLiuLab-T80."""
    return label.replace(" ", "-")


# --------------------------------------------------------------------------
# Names, and the order of the catalogue
# --------------------------------------------------------------------------
# One summand: a knot crossings_index, or a link crossings_components_index,
# with _m (mirror), _r (reversed) or _rm (both).  All 943 names of the data
# set parse, and the component count each implies matches its VECT file.
SUMMAND_RE = re.compile(r"^(?P<c>\d+)_(?:(?P<k>\d+)_)?(?P<i>\d+)(?:_(?P<flag>m|r|rm))?$")
TREE_RE = re.compile(r"^(?P<body>.+)_Tree(?P<tree>[A-Za-z0-9]+)$")
SUM_SEPARATOR = "_#_"


@dataclasses.dataclass(frozen=True)
class Summand:
    crossings: int
    components: int
    index: int
    mirror: bool = False
    reverse: bool = False


@dataclasses.dataclass(frozen=True)
class NameInfo:
    """What an entry's name says about it."""
    name: str
    summands: Tuple[Summand, ...]
    tree: str = ""

    @property
    def crossings(self):
        # A sum's crossing number is taken as the sum of its summands'.
        return sum(s.crossings for s in self.summands)

    @property
    def components(self):
        # Each join of a connected sum merges two components into one.
        return sum(s.components for s in self.summands) - (len(self.summands) - 1)

    @property
    def prime(self):
        return len(self.summands) == 1

    @property
    def kind(self):
        noun = "knot" if self.components == 1 else "link"
        return noun if self.prime else "composite " + noun


def parse_name(name):
    """NameInfo for a data-set name, or None for a name outside the grammar."""
    text = str(name).strip()
    body, tree = text, ""
    match = TREE_RE.match(text)
    if match:
        body, tree = match.group("body"), match.group("tree")
    summands = []
    for part in body.split(SUM_SEPARATOR):
        found = SUMMAND_RE.match(part)
        if not found:
            return None
        flag = found.group("flag") or ""
        summands.append(Summand(
            crossings=int(found.group("c")),
            components=int(found.group("k")) if found.group("k") else 1,
            index=int(found.group("i")),
            mirror="m" in flag,
            reverse="r" in flag))
    if tree and len(summands) < 2:
        return None                         # a tree only ever qualifies a sum
    return NameInfo(text, tuple(summands), tree)


def _natural(text):
    """Digits compared as numbers: 10_2 after 10_10 would read wrongly."""
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part)
                 for part in re.split(r"(\d+)", text) if part)


def catalogue_key(name):
    """Crossings, then components, then prime before sum, then the summands.

    Names outside the grammar go last, in natural order, so a stray file in
    the folder never hides among the table.
    """
    info = parse_name(name)
    if info is None:
        return (1, 0, 0, 0, (), "", _natural(str(name)))
    return (0, info.crossings, info.components, len(info.summands),
            tuple((s.crossings, s.components, s.index, s.reverse, s.mirror)
                  for s in info.summands),
            info.tree, _natural(info.name))


def is_prime(name):
    info = parse_name(name)
    return bool(info and info.prime)


def filter_catalogue(names, text="", prime_only=False):
    """The names that contain `text` (any case) and, if asked, are prime."""
    needle = str(text or "").strip().lower()
    return [name for name in names
            if (not needle or needle in name.lower())
            and (not prime_only or is_prime(name))]


def resolve_name(raw, known):
    """The data set's spelling of a typed name, or None.

    Accepts the file name itself (3_1_best.vect) and KnotPlot's dotted
    spelling (3.1), since both are what people have to hand.
    """
    text = str(raw).strip().strip('"').strip("'")
    if text in known:
        return text
    for suffix in (CURVE_SUFFIX, ".vect"):
        if text.endswith(suffix) and text[:-len(suffix)] in known:
            return text[:-len(suffix)]
    dotted = text.replace(".", "_")
    if dotted in known:
        return dotted
    return None


def split_names(text):
    """A typed name list: spaces or commas between names."""
    return [token for token in re.split(r"[\s,]+", str(text or "").strip()) if token]


# --------------------------------------------------------------------------
# The index and the catalogue
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Entry:
    """One row of ropelength_data.txt.  None where the index gives nan."""
    name: str
    ropelength: Optional[float] = None
    bound: Optional[float] = None
    edges: Optional[int] = None
    residual: Optional[float] = None
    writhe: Optional[float] = None


def _number(text):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_index(path):
    """(citation lines, entries by name, problems) from ropelength_data.txt.

    The header's own sentences are kept, without their '#', so every file
    written can say where its data come from; the line naming the columns is
    dropped.  Rows are name and five numbers, separated by whitespace.
    """
    citation, entries, problems = [], {}, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for number, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                body = " ".join(text.lstrip("#").split())
                # " Knot Tightening ... " -> "Knot Tightening ..."
                body = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', body)
                if body and not body.lower().startswith("knot type"):
                    citation.append(body)
                continue
            tokens = text.split()
            if len(tokens) != 6:
                problems.append("%s line %d has %d fields, not 6"
                                % (os.path.basename(path), number, len(tokens)))
                continue
            edges = _number(tokens[3])
            entries[tokens[0]] = Entry(
                name=tokens[0], ropelength=_number(tokens[1]),
                bound=_number(tokens[2]),
                edges=int(edges) if edges is not None else None,
                residual=_number(tokens[4]), writhe=_number(tokens[5]))
    return citation, entries, problems


def curve_path(folder, name):
    return os.path.join(folder, name + CURVE_SUFFIX)


def curve_names(folder):
    """Every <name>_best.vect in the folder, in catalogue order."""
    try:
        listing = os.listdir(folder)
    except OSError:
        return []
    names = [entry[:-len(CURVE_SUFFIX)] for entry in listing
             if entry.endswith(CURVE_SUFFIX) and len(entry) > len(CURVE_SUFFIX)
             and os.path.isfile(os.path.join(folder, entry))]
    return sorted(names, key=catalogue_key)


@dataclasses.dataclass
class Catalogue:
    """What the folder holds: the curves, and the index read beside them."""
    folder: str
    names: List[str]
    entries: Dict[str, Entry]
    citation: List[str]
    problems: List[str]
    has_index: bool

    def entry(self, name):
        return self.entries.get(name) or Entry(name)

    @property
    def index_only(self):
        """Index rows whose curve file is missing."""
        present = set(self.names)
        return sorted((n for n in self.entries if n not in present), key=catalogue_key)

    @property
    def unindexed(self):
        """Curve files the index has no row for."""
        return [n for n in self.names if n not in self.entries]

    def counts(self):
        """Entries by kind, in a fixed order."""
        out = {"knot": 0, "link": 0, "composite knot": 0, "composite link": 0, "other": 0}
        for name in self.names:
            info = parse_name(name)
            out[info.kind if info else "other"] += 1
        return out


def load_catalogue(folder):
    names = curve_names(folder)
    index = os.path.join(folder, INDEX_FILE)
    citation, entries, problems = [], {}, []
    has_index = os.path.isfile(index)
    if has_index:
        try:
            citation, entries, problems = read_index(index)
        except OSError as exc:
            problems.append("%s could not be read: %s" % (INDEX_FILE, exc))
            has_index = False
    return Catalogue(folder, names, entries, citation, problems, has_index)


# --------------------------------------------------------------------------
# Finding the data folder
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class RopelengthData:
    """Where the data set was found, and which setting found it."""
    folder: str = ""
    source: str = ""

    def __bool__(self):
        return bool(self.folder)


def source_label(source, explicit_label=CLI_LABEL):
    return {"explicit": explicit_label,
            "environment": "$%s" % ENV_VAR,
            "remembered": "remembered from the last time it was located"}.get(source, source)


def is_data_folder(folder):
    """True when the folder holds at least one <name>_best.vect."""
    try:
        return any(entry.endswith(CURVE_SUFFIX) and len(entry) > len(CURVE_SUFFIX)
                   for entry in os.listdir(folder))
    except OSError:
        return False


def _expand_candidate(path):
    """A picked path, and the ropelengthdata folder inside it."""
    text = os.path.expanduser(str(path).strip().strip('"').strip("'"))
    if not text:
        return []
    folder = os.path.abspath(text)
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    return [folder, os.path.join(folder, "ropelengthdata")]


def load_prefs(path=None):
    """The remembered settings, or {} when there are none or they are bad."""
    try:
        with open(path or PREFS_PATH, encoding="utf-8") as fh:
            prefs = json.load(fh)
    except (OSError, ValueError):
        return {}
    return prefs if isinstance(prefs, dict) else {}


def remembered_folder(path=None):
    folder = load_prefs(path).get("data")
    return folder if isinstance(folder, str) and folder.strip() else None


def save_prefs(folder, path=None):
    """Remember the data folder.  Returns False when it could not be saved;
    remembering is a convenience, so failing is never an error."""
    prefs = load_prefs(path)
    prefs["data"] = os.path.abspath(folder)
    try:
        with open(path or PREFS_PATH, "w", encoding="utf-8") as fh:
            json.dump(prefs, fh, indent=1)
            fh.write("\n")
    except OSError:
        return False
    return True


def _candidates(explicit=None):
    """(folder, source) pairs in the order they are tried."""
    out = []
    if explicit:
        out += [(p, "explicit") for p in _expand_candidate(explicit)]
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        out += [(p, "environment") for p in _expand_candidate(env)]
    remembered = remembered_folder()
    if remembered:
        out += [(p, "remembered") for p in _expand_candidate(remembered)]
    return out


def find_data(explicit=None):
    """The first candidate folder that holds the data set, or None."""
    for folder, source in _candidates(explicit):
        if is_data_folder(folder):
            return RopelengthData(folder, source)
    return None


def override_problems(explicit=None, explicit_label=CLI_LABEL):
    """Settings that are in force but lead nowhere, said on every run."""
    problems = []
    if explicit and not any(is_data_folder(p) for p in _expand_candidate(explicit)):
        problems.append("%s %s holds no <name>%s files" % (explicit_label, explicit, CURVE_SUFFIX))
    env = os.environ.get(ENV_VAR, "").strip()
    if env and not any(is_data_folder(p) for p in _expand_candidate(env)):
        problems.append("$%s=%s holds no <name>%s files" % (ENV_VAR, env, CURVE_SUFFIX))
    remembered = remembered_folder()
    if remembered and not any(is_data_folder(p) for p in _expand_candidate(remembered)):
        problems.append("the folder remembered from the last time one was "
                        "located, %s, no longer holds the data set" % remembered)
    return problems


def data_message(data, catalogue=None, explicit=None, explicit_label=CLI_LABEL):
    """Where the data set is and what it holds, or why it was not found."""
    terms = GUI_TERMS if explicit_label == LOCATED_LABEL else CLI_TERMS
    if not data:
        lines = ["The ropelength data set was not found.",
                 "",
                 "It is a folder holding one <name>%s per knot or link, with "
                 "an index called %s." % (CURVE_SUFFIX, INDEX_FILE),
                 "The data set is not bundled with Curve It: %s." % terms["locate"]]
        problems = override_problems(explicit, explicit_label)
        if problems:
            lines += [""] + ["  " + p for p in problems]
        return "\n".join(lines)
    catalogue = catalogue or load_catalogue(data.folder)
    counts = catalogue.counts()
    # Settings that point elsewhere and lead nowhere are named even when the
    # data set was found, since the search fell through them to get here.
    unused = [p for p in override_problems(explicit, explicit_label)
              if not (data.source == "explicit" and p.startswith(explicit_label))]
    lines = ["Ropelength data found at %s  (%s)" % (data.folder, source_label(data.source, explicit_label)),
             "  %d curve file(s): %d knot(s), %d link(s), %d composite knot(s), "
             "%d composite link(s)%s"
             % (len(catalogue.names), counts["knot"], counts["link"],
                counts["composite knot"], counts["composite link"],
                ", %d other" % counts["other"] if counts["other"] else ""),
             "  prime knots and links: %d" % (counts["knot"] + counts["link"])]
    if catalogue.has_index:
        lines.append("  %s: %d row(s)" % (INDEX_FILE, len(catalogue.entries)))
        if catalogue.unindexed:
            lines.append("  %d curve(s) have no index row, e.g. %s"
                         % (len(catalogue.unindexed), catalogue.unindexed[0]))
        if catalogue.index_only:
            lines.append("  %d index row(s) have no curve file, e.g. %s"
                         % (len(catalogue.index_only), catalogue.index_only[0]))
    else:
        lines.append("  no %s here: the curves read, without their ropelength figures" % INDEX_FILE)
    lines += ["  " + p for p in catalogue.problems[:5]]
    if unused:
        lines += ["", "Not used:"] + ["  " + p for p in unused]
    if catalogue.citation:
        lines += ["", "From the index:"] + ["  " + c for c in catalogue.citation]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Placing and writing the curves
# --------------------------------------------------------------------------
def component_label(index):
    """Spreadsheet-style labels A, B, ..., Z, AA, AB, matching Curve It."""
    label = ""
    n = index
    while True:
        label = chr(ord("A") + (n % 26)) + label
        n = n // 26 - 1
        if n < 0:
            return label


def place(components, scale=1.0, centre=True, mirror=False):
    """The components as they will be written, and the centring shift.

    Centring is a translation onto the mean of all the entry's vertices, so
    a link's components keep their places relative to one another; scaling
    is about the origin, after centring; the reflection negates z last.
    """
    comps = [np.asarray(c, dtype=float).reshape(-1, 3) for c in components]
    shift = -np.vstack(comps).mean(axis=0) if centre else np.zeros(3)
    flip = np.array([1.0, 1.0, -1.0]) if mirror else np.ones(3)
    return [(pts + shift) * float(scale) * flip for pts in comps], shift


def _rounded_rows(points, closed, precision):
    """The rows a file will hold: rounded, with a repeated row dropped.

    Repeats are judged on the rounded rows, not by distance, since a pair
    that differs only beyond the last decimal written becomes a zero-length
    segment in the file; a closed curve also loses a final row equal to its
    first, because closed curves never repeat their first vertex.  At six
    decimals and scale 1 the data set has neither.
    """
    rows = np.round(np.asarray(points, dtype=float), precision) + 0.0   # no -0.0
    if len(rows) > 1:
        keep = np.ones(len(rows), dtype=bool)
        keep[1:] = np.any(rows[1:] != rows[:-1], axis=1)
        rows = rows[keep]
    if closed and len(rows) > 2 and np.array_equal(rows[0], rows[-1]):
        rows = rows[:-1]
    return rows


def _fmt(value, digits):
    return "n/a" if value is None else ("%." + str(digits) + "g") % value


def index_line(entry):
    """The index's figures for one entry, as one header line."""
    if entry is None or entry.ropelength is None and entry.writhe is None and entry.edges is None:
        return "index: no row for this entry in %s" % INDEX_FILE
    return ("index: polygonal ropelength %s, ropelength bound %s, edges %s, "
            "residual %s, writhe %s"
            % (_fmt(entry.ropelength, 9), _fmt(entry.bound, 9),
               "n/a" if entry.edges is None else entry.edges,
               _fmt(entry.residual, 6),
               "not given" if entry.writhe is None else "%.6f" % entry.writhe))


def header_lines(name, entry, citation, n_components, scale=1.0, shift=None,
                 mirror=False):
    """The # lines that open a coordinate XYZ file, without their '#'.

    Each line's first word is a word, never a number: XYZ to 3D Model reads
    any line whose second to fourth tokens parse as numbers as a point, and
    names such as 3_1 parse as numbers (float("3_1") is 31.0).
    """
    info = parse_name(name)
    kind = info.kind if info else "curve"
    lines = ["%s v%s -- %s %s from Jason Cantarella's ropelength data set, %d component(s)"
             % (TOOL_NAME, __version__, kind, name, n_components)]
    lines += ["data set: %s" % c for c in citation]
    lines.append(index_line(entry))
    if scale == 1.0:
        lines.append("units: the data set's tube radius 0.5, so the tight rope is 1 thick")
    else:
        lines.append("units: scaled by %s from the data set's tube radius 0.5, so the "
                     "tight rope is %s thick" % (_fmt(scale, 9), _fmt(ROPE_DIAMETER * scale, 9)))
    if shift is not None:
        lines.append("frame: centred on the mean of its vertices; the data set's "
                     "coordinates were moved by %.6f %.6f %.6f before scaling"
                     % tuple(shift))
    else:
        lines.append("frame: the data set's own coordinates")
    if mirror:
        lines.append("MIRROR IMAGE: reflected through %s = 0 (%s -> -%s), so the "
                     "writhe and linking numbers are the negatives of the index's"
                     % (MIRROR_AXIS, MIRROR_AXIS, MIRROR_AXIS))
    lines.append("columns: x y z ; blank lines separate components ; a closed "
                 "component does not repeat its first vertex")
    return lines


def write_coordinate_xyz(path, components, closed, header, precision=DEFAULT_PRECISION,
                         comments=True, labels=None):
    """Curve It's plain-coordinate curve file, one block per component."""
    fmt = "%%.%df %%.%df %%.%df\n" % (precision, precision, precision)
    labels = labels or [component_label(i) for i in range(len(components))]
    with open(path, "w") as fh:
        if comments:
            for line in header:
                fh.write("# %s\n" % line)
        for index, (pts, shut) in enumerate(zip(components, closed)):
            rows = _rounded_rows(pts, shut, precision)
            if index:
                fh.write("\n")
            if comments:
                fh.write("# component %s: closed=%s points=%d\n"
                         % (labels[index], "yes" if shut else "no", len(rows)))
            for x, y, z in rows:
                fh.write(fmt % (x, y, z))
    return path


def write_vect_file(path, name, components, closed, precision=DEFAULT_PRECISION,
                    comments=True, palette="default"):
    """Geomview VECT through vect_io, closure written as the vertex count's sign."""
    rows = [_rounded_rows(pts, shut, precision) for pts, shut in zip(components, closed)]
    colors = None if palette == "default" else _palette_token(palette)
    return vect_io.write_vect(path, rows, closed=list(closed), colors=colors,
                              comments=comments, precision=precision,
                              source=name + CURVE_SUFFIX,
                              tool="%s v%s" % (TOOL_NAME, __version__))


def _curve_it():
    """curve_it.py itself, for the writers its Convert XYZ uses.

    Imported only when a fake PDB or molecular XYZ is asked for: it loads
    scipy, about half a second, which nothing else here needs.
    """
    try:
        import curve_it                     # noqa: F401
    except ImportError:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        import curve_it                     # noqa: F811
    return curve_it


def _xyz2model():
    try:
        from . import xyz2model
    except ImportError:
        import xyz2model                    # type: ignore[no-redef]
    return xyz2model


def write_fake_pdb(path, components, closed):
    """One atom per vertex, one chain per component, a LINK closing each loop."""
    rows = [_rounded_rows(pts, shut, 3) for pts, shut in zip(components, closed)]
    low = min(float(r.min()) for r in rows)
    high = max(float(r.max()) for r in rows)
    if low < PDB_LOW or high > PDB_HIGH:
        raise ValueError("a fake PDB holds coordinates from -999.999 to 9999.999, "
                         "and this one reaches %.3f; leave centring on, or use "
                         "a smaller scale" % (low if low < PDB_LOW else high))
    ci = _curve_it()
    ci.write_fake_pdb_from_components(
        path, rows, closed_chain_indices=[i for i, shut in enumerate(closed) if shut])
    return path


def write_molecular(path, components, closed, comment, precision=DEFAULT_PRECISION):
    """Molecular XYZ holds one curve: several components are joined, in order."""
    rows = [_rounded_rows(pts, shut, precision) for pts, shut in zip(components, closed)]
    ci = _curve_it()
    ci.write_molecular_xyz_curve(path, np.vstack(rows), element="X", comment=comment)
    return path


def trimesh_missing():
    """Why the rod models cannot be made here, or "" when they can."""
    try:
        import trimesh                      # noqa: F401
    except ImportError:
        return ("the rod models need trimesh, as XYZ to 3D Model does: "
                "python3 -m pip install trimesh")
    return ""


def write_models(prefix, components, closed, diameter, formats, palette="default"):
    """GLB and STL rods through XYZ to 3D Model, every vertex kept.

    The components arrive placed and scaled already, so the meshes are built
    at scale 1 and not recentred: they lie in the curve files' frame.
    """
    x2m = _xyz2model()
    meshes = x2m.build_meshes(components, scale=1.0, diameter=float(diameter),
                              segments=None, sides=24,
                              closed=x2m.stated_closure_setting(list(closed)),
                              recentre=False)
    base = x2m.tagged_prefix(prefix, "D", diameter)
    paths = []
    if "glb" in formats:
        paths.append(x2m.export_glb(meshes, base + ".glb", x2m.palette_colors(palette)))
    if "stl" in formats:
        paths.append(x2m.export_stl(meshes, base + ".stl"))
    return paths


def parse_formats(text):
    """A comma list of format names, in FORMATS order; "all" is every one."""
    picked = set()
    tokens = [t for t in re.split(r"[\s,]+", str(text or "").strip().lower()) if t]
    for token in tokens:
        if token == "all":
            picked.update(FORMATS)
            continue
        fmt = FORMAT_ALIASES.get(token, token)
        if fmt not in FORMATS:
            raise ValueError("unknown format %r; choose from %s"
                             % (token, ", ".join(FORMATS)))
        picked.add(fmt)
    if not picked:
        raise ValueError("no format given; choose from %s" % ", ".join(FORMATS))
    return tuple(f for f in FORMATS if f in picked)


def formats_argument(value):
    """argparse type for --to: an unknown format is a usage error."""
    try:
        return parse_formats(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def check_settings(scale, diameter, precision, terms=CLI_TERMS):
    if not (isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0):
        raise ValueError("%s must be a positive number, not %r" % (terms["scale"], scale))
    if diameter is not None and not (math.isfinite(diameter) and diameter > 0):
        raise ValueError("%s must be a positive number, not %r" % (terms["diameter"], diameter))
    if not (isinstance(precision, int) and 0 <= precision <= MAX_PRECISION):
        raise ValueError("%s must be a whole number from 0 to %d, not %r"
                         % (terms["precision"], MAX_PRECISION, precision))


def write_entry(prefix, name, components, closed, header, formats, split=False,
                precision=DEFAULT_PRECISION, comments=True, diameter=ROPE_DIAMETER,
                palette="default"):
    """Every requested file for one placed entry.  Returns one record per file."""
    labels = [component_label(i) for i in range(len(components))]
    parts = [{"label": label, "closed": bool(shut)} for label, shut in zip(labels, closed)]
    one_each = split and len(components) > 1
    records = []

    def record(path, fmt, indices, joined=False):
        records.append({"path": path, "name": name, "format": fmt, "joined": joined,
                        "components": [parts[i] for i in indices]})

    every = list(range(len(components)))
    if "coordinate" in formats:
        if one_each:
            for i in every:
                path = "%s_%s.xyz" % (prefix, labels[i])
                part = ["part: component %s of the %d in %s, written alone"
                        % (labels[i], len(components), name)]
                write_coordinate_xyz(path, [components[i]], [closed[i]],
                                     header[:-1] + part + header[-1:], precision,
                                     comments, labels=[labels[i]])
                record(path, "coordinate", [i])
        else:
            path = prefix + ".xyz"
            write_coordinate_xyz(path, components, closed, header, precision, comments)
            record(path, "coordinate", every)
    if "vect" in formats:
        path = prefix + ".vect"
        write_vect_file(path, name, components, closed, precision, comments, palette)
        record(path, "vect", every)
    if "fake-pdb" in formats:
        path = prefix + ".pdb"
        write_fake_pdb(path, components, closed)
        record(path, "fake-pdb", every)
    if "molecular" in formats:
        comment = header[0]
        if one_each:
            for i in every:
                path = "%s_%s_molecular.xyz" % (prefix, labels[i])
                write_molecular(path, [components[i]], [closed[i]],
                                "%s; component %s" % (comment, labels[i]), precision)
                record(path, "molecular", [i])
        else:
            path = prefix + "_molecular.xyz"
            write_molecular(path, components, closed, comment, precision)
            record(path, "molecular", every, joined=len(components) > 1)
    models = [f for f in MODEL_FORMATS if f in formats]
    if models:
        for path in write_models(prefix, components, closed, diameter, models, palette):
            record(path, os.path.splitext(path)[1].lstrip("."), every)
    return records


def extract(names, outdir, folder, formats=DEFAULT_FORMATS, scale=1.0, centre=True,
            mirror=False, split=False, precision=DEFAULT_PRECISION, comments=True,
            diameter=None, palette="default", catalogue=None, terms=None):
    """Read each named entry and write it in every format asked for.

    Returns {"written", "failed", "files", "joined", "read", "diameter",
    "outdir"}: the paths written, (name, reason) pairs for what could not be
    read, one record per file, the molecular files that joined components,
    the names read, and the rod diameter used.  Nothing is written for a
    name that does not resolve, and an entry that fails part-way keeps the
    files written before the failure.
    """
    terms = terms or CLI_TERMS
    if vect_io is None:
        raise ValueError(VECT_MISSING)
    unknown = sorted(set(formats) - set(FORMATS))
    if unknown:
        raise ValueError("unknown format %s; choose from %s"
                         % (", ".join(unknown), ", ".join(FORMATS)))
    formats = tuple(f for f in FORMATS if f in set(formats))
    if not formats:
        raise ValueError("no format was asked for")
    check_settings(scale, diameter, precision, terms)
    palette = check_palette(palette)
    if any(f in MODEL_FORMATS for f in formats):
        missing = trimesh_missing()
        if missing:
            raise ValueError(missing)
    catalogue = catalogue or load_catalogue(folder)
    outdir = os.path.abspath(os.path.expanduser(outdir or "."))
    if os.path.exists(outdir) and not os.path.isdir(outdir):
        raise ValueError("the output folder %s is a file; %s" % (outdir, terms["outdir_fix"]))
    if os.path.isdir(outdir) and os.path.isdir(folder) and os.path.samefile(outdir, folder):
        raise ValueError("the output folder is the data set's own folder, %s, which is "
                         "never written to; %s" % (folder, terms["outdir_fix"]))
    os.makedirs(outdir, exist_ok=True)
    rod = float(diameter) if diameter is not None else ROPE_DIAMETER * float(scale)
    known = set(catalogue.names)
    sorted_known = sorted(known)
    written, failed, files, joined, read = [], [], [], [], []
    seen = set()
    for raw in names:
        name = resolve_name(raw, known)
        if name is None:
            near = difflib.get_close_matches(str(raw), sorted_known, n=3)
            failed.append((str(raw), "not in the data set"
                           + ("; did you mean %s?" % ", ".join(near) if near else "")))
            continue
        if name in seen:
            continue
        seen.add(name)
        try:
            curves = vect_io.read_vect(curve_path(folder, name))
            comps, shift = place(curves.components, scale, centre, mirror)
            closed = [bool(flag) for flag in curves.closed]
            header = header_lines(name, catalogue.entry(name), catalogue.citation,
                                  len(comps), scale, shift if centre else None, mirror)
            stem = name + ("_mirror" if mirror else "")
            records = write_entry(os.path.join(outdir, stem), name, comps, closed, header,
                                  formats, split, precision, comments, rod, palette)
        except (OSError, ValueError) as exc:
            failed.append((name, str(exc)))
            continue
        read.append(name)
        for rec in records:
            written.append(rec["path"])
            files.append(rec)
            if rec["joined"]:
                joined.append(rec["path"])
    return {"written": written, "failed": failed, "files": files, "joined": joined,
            "read": read, "outdir": outdir,
            "diameter": rod if any(f in MODEL_FORMATS for f in formats) else None}


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def entry_rows(names, catalogue):
    """One aligned line per name: kind, crossings, components, figures."""
    width = max([len(n) for n in names] + [4])
    rows = []
    for name in names:
        info = parse_name(name)
        entry = catalogue.entry(name)
        kind = info.kind if info else "other"
        crossings = "%3d" % info.crossings if info else "  ?"
        comps = info.components if info else 0
        rows.append("%-*s  %-14s %s cr  %d comp  ropelength %9s  writhe %10s"
                    % (width, name, kind, crossings, comps,
                       _fmt(entry.ropelength, 6),
                       "n/a" if entry.writhe is None else "%.6f" % entry.writhe))
    return rows


def component_guidance(files):
    """The "USING THIS FILE IN CURVE IT" block, or [] when nothing needs it.

    Curve It joins every component of a curve file into one path unless one
    is selected, and a link here is one file, so the exact command is spelt
    out.  Every component of the data set is closed, so the path type is too.
    """
    # One file per entry is enough to name: the coordinate XYZ when there is
    # one, else the VECT, which Curve It joins the same way.
    multi = ([f for f in files if f["format"] == "coordinate" and len(f["components"]) > 1]
             or [f for f in files if f["format"] == "vect" and len(f["components"]) > 1])
    if not multi:
        return []
    record = multi[0]
    parts = record["components"]
    name = os.path.basename(record["path"])
    lines = ["",
             "USING THIS FILE IN CURVE IT",
             "  Curve It joins every component of a curve file into one path by",
             "  default, which would join the %d curves in %s. Select one at a time:"
             % (len(parts), name)]
    for part in parts[:3]:
        lines.append("    python3 curve_it.py in.pdb '%s' --curve-components %s --path-type %s"
                     % (name, part["label"], "closed" if part["closed"] else "open"))
    if len(parts) > 3:
        lines.append("    ... and so on through component %s" % parts[-1]["label"])
    lines.append("  In the GUI, use Select components... beside the curve file.")
    if len(multi) > 1:
        lines.append("  %d other file(s) from this run also hold several components."
                     % (len(multi) - 1))
    return lines


def describe(result, catalogue, scale=1.0, centre=True, mirror=False):
    """The summary printed after a run, before the list of files."""
    lines = []
    read = result["read"]
    if read:
        lines.append("READ %d entr%s from %s" % (len(read), "y" if len(read) == 1 else "ies",
                                                 catalogue.folder))
        shown = read[:REPORT_LIMIT]
        lines += ["  " + row for row in entry_rows(shown, catalogue)]
        if len(read) > len(shown):
            lines.append("  ... and %d more" % (len(read) - len(shown)))
        how = ["scale %s, so the tight rope is %s thick" % (_fmt(scale, 9), _fmt(ROPE_DIAMETER * scale, 9)),
               "centred on each entry's vertex mean" if centre else "the data set's own coordinates"]
        if mirror:
            how.append("MIRROR IMAGES, %s -> -%s" % (MIRROR_AXIS, MIRROR_AXIS))
        lines.append("  " + "; ".join(how))
    if read and result.get("diameter") is not None:
        rod = result["diameter"]
        lines.append("  rods %s thick%s" % (_fmt(rod, 9),
                     "; they touch where the tight rope touches itself"
                     if rod >= 0.999 * ROPE_DIAMETER * scale else ""))
    if read and result["joined"]:
        lines.append("  molecular XYZ holds one curve, so the components of %d entr%s were "
                     "joined into one; tick one file per component to keep them apart"
                     % (len(result["joined"]), "y" if len(result["joined"]) == 1 else "ies"))
    return "\n".join(lines)


def written_lines(result):
    written = result["written"]
    if not written:
        return []
    shown = written[:REPORT_LIMIT]
    lines = ["", "WROTE %d file(s) to %s" % (len(written), result["outdir"])]
    lines += ["  " + os.path.basename(path) for path in shown]
    if len(written) > len(shown):
        lines.append("  ... and %d more" % (len(written) - len(shown)))
    return lines


def failed_lines(result):
    if not result["failed"]:
        return []
    return ["", "FAILED"] + ["  %s  --  %s" % pair for pair in result["failed"][:REPORT_LIMIT]]


# --------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------
def preview(paths, palette="default"):
    """Open Curve It's curve viewer on each file, one process per window.

    view_xyzV3 ends in a blocking plt.show(), so it runs as its own process:
    the window that asked stays usable and several previews can be open at
    once.  The files are VECT, which states closure, so the viewer is told
    nothing about it.  Returns the paths opened.
    """
    label = check_palette(palette)
    if getattr(sys, "frozen", False):
        # sys.executable is the Curve It app itself there, which would take
        # the viewer's path for a structure to curve.
        raise ValueError("Preview runs Curve It's curve viewer with Python, which "
                         "the one-file app does not carry; open the written files "
                         "with View curve in Curve It instead")
    viewer = resource_path(os.path.join("curve_it_lib", "view_xyzV3.py"))
    if not os.path.isfile(viewer):
        raise ValueError("the curve viewer is missing: %s" % viewer)
    opened = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        command = [sys.executable, viewer, path]
        if label != "default":
            command += ["--palette", _palette_token(label)]
        try:
            subprocess.Popen(command)
        except OSError as exc:
            raise ValueError("could not start the curve viewer: %s" % exc)
        opened.append(path)
    if not opened:
        raise ValueError("there is nothing to preview")
    return opened


def preview_entries(names, folder, scale=1.0, centre=True, mirror=False,
                    palette="default", catalogue=None, terms=None):
    """Write the first few names as VECT into a scratch folder and view them.

    The output folder is never touched: this is for looking at a curve before
    keeping it.  Returns (opened paths, note about any names left out).
    """
    unique = list(dict.fromkeys(names))
    chosen = unique[:PREVIEW_LIMIT]
    note = ("previewing the first %d of %d entries" % (len(chosen), len(unique))
            if len(unique) > len(chosen) else "")
    scratch = tempfile.mkdtemp(prefix="read_ropelengthdata-preview-")
    result = extract(chosen, scratch, folder, formats=("vect",), scale=scale,
                     centre=centre, mirror=mirror, palette=palette,
                     catalogue=catalogue, terms=terms)
    if not result["written"]:
        reasons = "; ".join("%s: %s" % pair for pair in result["failed"]) or "nothing was read"
        raise ValueError("nothing to preview -- %s" % reasons)
    return preview(result["written"], palette=palette), note, result


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="read_ropelengthdata.py",
        description="Read tight knots and links from Jason Cantarella's "
                    "ropelength data set and write them as Curve It curve "
                    "files.",
        epilog="Run with no arguments for the GUI. The exit status is 0 only "
               "when every name was read and written, so a scripted --all can "
               "tell a partial run from a complete one.")
    p.add_argument("names", nargs="*",
                   help="entry names as the data set spells them: 3_1, 6_3_2, "
                        "3_1_#_3_1_m (quote a name with # in a shell that "
                        "treats it as a comment)")
    p.add_argument("--gui", action="store_true", help="force the graphical interface")
    p.add_argument("--data", metavar="DIR",
                   help="the data set's folder, the one holding <name>_best.vect "
                        "and ropelength_data.txt; overrides $%s and the folder "
                        "the window remembers" % ENV_VAR)
    p.add_argument("--check", action="store_true",
                   help="report where the data set was found and what it "
                        "holds, and exit")
    p.add_argument("--list", action="store_true", dest="list_names",
                   help="print every entry name, in crossing-number order, and exit")
    p.add_argument("--prime", action="store_true",
                   help="with --list or --all, only the prime knots and links "
                        "(names with no _#_)")
    p.add_argument("--all", action="store_true",
                   help="read every entry in the folder")
    p.add_argument("-o", "--outdir", default=DEFAULT_OUTDIR,
                   help="output folder (default %s)" % DEFAULT_OUTDIR)
    p.add_argument("--to", type=formats_argument, default=DEFAULT_FORMATS,
                   metavar="FORMATS",
                   help="comma-separated formats to write: %s, or all "
                        "(default coordinate)" % ", ".join(FORMATS))
    p.add_argument("-s", "--scale", type=float, default=1.0,
                   help="multiply every coordinate; the tight rope is 1 thick "
                        "in the data set, so this is also the rope's thickness "
                        "(default 1)")
    p.add_argument("-d", "--diameter", type=float, default=None,
                   help="rod diameter for glb and stl, in output units (default: "
                        "the tight rope, 1 x the scale, where the rods touch)")
    p.add_argument("--keep-origin", action="store_true",
                   help="keep the data set's own coordinates instead of centring "
                        "each entry on the mean of its vertices")
    p.add_argument("--mirror", action="store_true",
                   help="write the mirror image, %s -> -%s, which negates the "
                        "writhe and linking numbers" % (MIRROR_AXIS, MIRROR_AXIS))
    p.add_argument("--split", action="store_true",
                   help="write each component of a link to its own coordinate "
                        "or molecular XYZ file")
    p.add_argument("--precision", type=int, default=DEFAULT_PRECISION,
                   help="decimal places per coordinate (default 6)")
    p.add_argument("--no-comments", action="store_true",
                   help="omit the '#' header lines from coordinate XYZ and VECT")
    p.add_argument("--preview", action="store_true",
                   help="open the first %d entries in Curve It's 3D viewer after "
                        "writing" % PREVIEW_LIMIT)
    p.add_argument("--palette", type=palette_argument, default="default",
                   help="colours for a link's components in the preview, VECT "
                        "and GLB: default, or DiLiuLab, DiLiuLab-T80, -T60, "
                        "-T40, -T20 (read from assets/diliulab_colors.json)")
    p.add_argument("--info", action="store_true",
                   help="list what would be read, with its figures, and exit "
                        "without writing")
    return p


def resolve_names(args, catalogue):
    names = list(args.names)
    if args.all:
        names += filter_catalogue(catalogue.names, prime_only=args.prime)
    return names


def run_cli(args):
    # --data pins the run to one folder, so a bad one is fatal: falling through
    # to $ROPELENGTHDATA or the remembered folder would quietly read another
    # copy of the data than the one asked for.
    if args.data and not any(is_data_folder(p) for p in _expand_candidate(args.data)):
        raise ValueError("--data %s holds no <name>%s files.\n--data pins this run "
                         "to that folder, so no other is used instead.\nGive the "
                         "folder holding 3_1%s and %s."
                         % (args.data, CURVE_SUFFIX, CURVE_SUFFIX, INDEX_FILE))
    data = find_data(args.data)
    # Without --data the search falls through a bad $ROPELENGTHDATA or a stale
    # remembered folder, so a folder found elsewhere must not look as though
    # those settings had been honoured.  --check says it in its own message.
    if not args.check and not args.data:
        for problem in override_problems(None):
            print("warning: %s; ignored" % problem, file=sys.stderr)
    if args.check:
        catalogue = load_catalogue(data.folder) if data else None
        print(data_message(data, catalogue, args.data))
        return 0 if data else 1
    if not data:
        raise ValueError(data_message(None, None, args.data))
    if vect_io is None:
        raise ValueError(VECT_MISSING)
    catalogue = load_catalogue(data.folder)

    if args.list_names:
        for name in filter_catalogue(catalogue.names, prime_only=args.prime):
            print(name)
        return 0

    names = resolve_names(args, catalogue)
    if not names:
        raise ValueError("give at least one name, or --all")
    print("data:  %s  (%s)" % (data.folder, source_label(data.source)))
    if args.info:
        known = set(catalogue.names)
        resolved = [resolve_name(n, known) or n for n in dict.fromkeys(names)]
        print("names: %d" % len(resolved))
        for row in entry_rows(resolved[:REPORT_LIMIT], catalogue):
            print("  " + row)
        if len(resolved) > REPORT_LIMIT:
            print("  ... and %d more" % (len(resolved) - REPORT_LIMIT))
        return 0

    result = extract(names, args.outdir, data.folder, formats=args.to,
                     scale=args.scale, centre=not args.keep_origin,
                     mirror=args.mirror, split=args.split,
                     precision=args.precision, comments=not args.no_comments,
                     diameter=args.diameter, palette=args.palette,
                     catalogue=catalogue)
    report = describe(result, catalogue, args.scale, not args.keep_origin, args.mirror)
    if report:
        print(report)
    for line in written_lines(result) + component_guidance(result["files"]) + failed_lines(result):
        print(line)
    if args.preview and result["read"]:
        try:
            opened, note, _ = preview_entries(result["read"], data.folder, args.scale,
                                              not args.keep_origin, args.mirror,
                                              args.palette, catalogue)
            print("\npreview: %d window(s)%s" % (len(opened), "; " + note if note else ""))
        except ValueError as exc:
            # A failed preview must not fail the run that succeeded.
            print("preview unavailable: %s" % exc, file=sys.stderr)
    return 0 if result["written"] and not result["failed"] else 1


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def default_gui_outdir():
    """An absolute folder to prefill the output field with: an app started
    from the Finder has / as its working directory, so "here" is no place."""
    home = os.path.expanduser("~")
    documents = os.path.join(home, "Documents")
    return os.path.join(documents if os.path.isdir(documents) else home, DEFAULT_OUTDIR)


def _register_icon(ui):
    """Give the window its own glyph and purpose line, when they can be had."""
    try:
        try:
            from . import curve_it_icons
        except ImportError:
            import curve_it_icons           # type: ignore[no-redef]
        curve_it_icons.register(ui)
    except Exception:                       # noqa: BLE001  the icon is decoration
        pass


def run_gui(initial_data=None, initial_outdir=None, initial_palette="default"):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    try:
        from curve_it_lib import azbmost_ui as ui
    except ImportError:
        import azbmost_ui as ui
    _register_icon(ui)

    root = tk.Tk()
    root.title("read_ropelengthdata  -  tight knots and links  ->  curve files   v%s"
               % __version__)
    root.minsize(940, 620)
    set_optional_window_icon(
        root, tk, ["read_ropelengthdata_icon.png", "icon.png"],
        "_read_ropelengthdata_icon_image")
    ui.tool_window(root, TOOL_NAME, tool=TOOL_KEY, version="v%s" % __version__)
    C = ui.C

    # "explicit" is the folder picked with Locate data folder...; it is fed
    # back into every search, and forgotten as soon as it stops holding the
    # data set.  "catalogue" is the listing of the folder in force.
    state = {"data": None, "explicit": initial_data, "catalogue": None}
    V = {
        "names":     tk.StringVar(value=""),
        "all":       tk.BooleanVar(value=False),
        "prime":     tk.BooleanVar(value=False),
        "scale":     tk.StringVar(value="1"),
        "centre":    tk.BooleanVar(value=True),
        "mirror":    tk.BooleanVar(value=False),
        "diameter":  tk.StringVar(value=""),
        "split":     tk.BooleanVar(value=False),
        "precision": tk.StringVar(value="%d" % DEFAULT_PRECISION),
        "comments":  tk.BooleanVar(value=True),
        "outdir":    tk.StringVar(value=initial_outdir or default_gui_outdir()),
        "palette":   tk.StringVar(value=check_palette(initial_palette)),
    }
    for fmt in FORMATS:
        V["fmt:" + fmt] = tk.BooleanVar(value=fmt in DEFAULT_FORMATS)

    # ---- "?" help chips ---------------------------------------------------
    open_help = {"win": None}

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
        top.configure(bg=C.BG)
        frm = tk.Frame(top, bg=C.BG, padx=20, pady=16)
        frm.pack(fill="both", expand=True)
        tk.Label(frm, text=title, font="AzbTitle", bg=C.BG, fg=C.TEXT,
                 anchor="w", justify="left").pack(fill="x", pady=(0, 8))
        tk.Label(frm, text=prose, wraplength=470, justify="left", bg=C.BG,
                 fg=C.TEXT, font="AzbBody", anchor="w").pack(fill="x")
        if example:
            tk.Label(frm, text="Example", font="AzbSmallBold", bg=C.BG,
                     fg=C.TEXT_2, anchor="w").pack(fill="x", pady=(12, 4))
            # Height in DISPLAY rows: a line longer than the box wraps.
            width = 62
            rows = sum(max(1, -(-len(line) // width)) for line in example.split("\n"))
            box = tk.Text(frm, font="AzbMono", height=rows, width=width,
                          bg=C.SUBTLE, fg=C.TEXT, relief="flat", padx=10, pady=8,
                          highlightthickness=1, highlightbackground=C.BORDER,
                          highlightcolor=C.BORDER)
            box.insert("1.0", example)
            box.configure(state="disabled")
            box.pack(fill="x")
        ttk.Button(frm, text="Close", command=top.destroy).pack(anchor="e", pady=(14, 0))
        top.bind("<Escape>", lambda e: top.destroy())
        # Beside the chip, but kept on the main screen; Tk measures only the
        # main display, so a chip on another one keeps the plain placement.
        try:
            top.update_idletasks()
            sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
            cx, cy = near.winfo_rootx(), near.winfo_rooty()
            x, y = cx + 26, cy - 10
            if 0 <= cx < sw and 0 <= cy < sh:
                x = min(x, sw - top.winfo_reqwidth() - 8)
                y = min(y, sh - top.winfo_reqheight() - 100)
            top.geometry("+%d+%d" % (max(x, 0), max(y, 0)))
        except Exception:                   # noqa: BLE001
            pass
        top.deiconify()
        top.focus_set()

    def chip(parent, key):
        """The theme's round '?' that opens the explanation for `key`."""
        c = ui.HelpButton(parent)
        c.configure(command=lambda k=key, w=c: show_help(k, w))
        return c

    # The run buttons and the status line sit in the footer; packed first,
    # they keep their place when a short screen cuts the window's height.
    footer = ui.Footer(root)
    footer.pack(side="bottom", fill="x")
    ui.hairline(root).pack(side="bottom", fill="x")

    # ---- data folder banner ----------------------------------------------
    head = ttk.Frame(root, padding=(20, 14, 20, 0))
    head.pack(fill="x")
    head.columnconfigure(0, weight=1)
    data_lbl = tk.Label(head, text="looking for the ropelength data ...", justify="left",
                        anchor="w", font="AzbSmall", fg="#888", wraplength=700)
    data_lbl.grid(row=0, column=0, sticky="ew")
    data_lbl.bind("<Configure>", lambda e: data_lbl.configure(wraplength=max(300, e.width - 4)))
    locate_btn = ttk.Button(head, text="Locate data folder...")
    locate_btn.grid(row=0, column=1, padx=(12, 0))
    chip(head, "data").grid(row=0, column=2, padx=(8, 0))

    main = ttk.Frame(root, padding=(20, 14, 20, 16))
    main.pack(fill="both", expand=True)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(0, weight=1)
    form = ui.ScrollFrame(main)
    form.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    left = form.body
    left.configure(padding=(0, 0, 10, 0))
    right = ttk.Frame(main)
    right.grid(row=0, column=1, sticky="nsew")
    right.rowconfigure(0, weight=1)
    right.columnconfigure(0, weight=1)

    # what to read ------------------------------------------------------------
    ft = ttk.LabelFrame(left, text="What to read")
    ft.pack(fill="x", pady=(0, 12))
    ft.columnconfigure(0, weight=1)
    ttk.Entry(ft, textvariable=V["names"], width=34).grid(row=0, column=0, columnspan=2,
                                                         sticky="ew")
    chip(ft, "names").grid(row=0, column=2, padx=(8, 0))
    ttk.Label(ft, text="names as the data set spells them: 3_1 6_3_2 3_1_#_3_1_m",
              style="Hint.TLabel").grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
    browse_btn = ttk.Button(ft, text="Browse data set...")
    browse_btn.grid(row=2, column=0, sticky="w", pady=(8, 0))
    chip(ft, "catalogue").grid(row=2, column=2, padx=(8, 0), pady=(8, 0))
    all_chk = ttk.Checkbutton(ft, text="Entire data set", variable=V["all"])
    all_chk.grid(row=3, column=0, sticky="w", pady=(8, 0))
    chip(ft, "all").grid(row=3, column=2, padx=(8, 0), pady=(8, 0))
    prime_chk = ttk.Checkbutton(ft, text="prime knots and links only", variable=V["prime"])
    prime_chk.grid(row=4, column=0, sticky="w", pady=(2, 0))
    chip(ft, "prime").grid(row=4, column=2, padx=(8, 0), pady=(2, 0))

    # geometry ----------------------------------------------------------------
    fg = ttk.LabelFrame(left, text="Geometry")
    fg.pack(fill="x", pady=(0, 12))
    field_labels = [ttk.Label(fg, text="scale", style="Field.TLabel")]
    field_labels[-1].grid(row=0, column=0, sticky="w", pady=3)
    ttk.Entry(fg, textvariable=V["scale"], width=10).grid(row=0, column=1, sticky="w",
                                                          padx=(12, 0), pady=3)
    chip(fg, "scale").grid(row=0, column=2, padx=(8, 0), pady=3)
    ttk.Label(fg, text="1: the rope is 1 thick; 20 for 20 Å",
              style="Hint.TLabel").grid(row=0, column=3, sticky="w", padx=(10, 0), pady=3)
    ttk.Checkbutton(fg, text="centre on the vertex mean", variable=V["centre"]).grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
    chip(fg, "centre").grid(row=1, column=2, padx=(8, 0), pady=(6, 0))
    ttk.Checkbutton(fg, text="mirror image (z -> -z)", variable=V["mirror"]).grid(
        row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
    chip(fg, "mirror").grid(row=2, column=2, padx=(8, 0), pady=(4, 0))

    # formats -----------------------------------------------------------------
    fe = ttk.LabelFrame(left, text="Formats")
    fe.pack(fill="x", pady=(0, 12))
    for i, fmt in enumerate(FORMATS):
        ttk.Checkbutton(fe, text=FORMAT_LABELS[fmt], variable=V["fmt:" + fmt]).grid(
            row=i, column=0, columnspan=4, sticky="w", pady=1)
    chip(fe, "formats").grid(row=0, column=4, padx=(8, 0), pady=1)
    rows = len(FORMATS)
    field_labels.append(ttk.Label(fe, text="rod diameter", style="Field.TLabel"))
    field_labels[-1].grid(row=rows, column=0, sticky="w", pady=(8, 3))
    ttk.Entry(fe, textvariable=V["diameter"], width=10).grid(row=rows, column=1, sticky="w",
                                                             padx=(12, 0), pady=(8, 3))
    chip(fe, "diameter").grid(row=rows, column=4, padx=(8, 0), pady=(8, 3))
    ttk.Label(fe, text="blank = the tight rope (1 x scale)", style="Hint.TLabel").grid(
        row=rows, column=2, columnspan=2, sticky="w", padx=(10, 0), pady=(8, 3))
    field_labels.append(ttk.Label(fe, text="precision", style="Field.TLabel"))
    field_labels[-1].grid(row=rows + 1, column=0, sticky="w", pady=3)
    ttk.Entry(fe, textvariable=V["precision"], width=10).grid(row=rows + 1, column=1,
                                                              sticky="w", padx=(12, 0), pady=3)
    chip(fe, "precision").grid(row=rows + 1, column=4, padx=(8, 0), pady=3)
    ttk.Label(fe, text="decimals per coordinate", style="Hint.TLabel").grid(
        row=rows + 1, column=2, columnspan=2, sticky="w", padx=(10, 0), pady=3)
    ttk.Checkbutton(fe, text="one file per component", variable=V["split"]).grid(
        row=rows + 2, column=0, columnspan=4, sticky="w", pady=(6, 0))
    chip(fe, "split").grid(row=rows + 2, column=4, padx=(8, 0), pady=(6, 0))

    # output ------------------------------------------------------------------
    fo = ttk.LabelFrame(left, text="Output")
    fo.pack(fill="x")
    fo.columnconfigure(1, weight=1)
    field_labels.append(ttk.Label(fo, text="folder", style="Field.TLabel"))
    field_labels[-1].grid(row=0, column=0, sticky="w", pady=3)
    ttk.Entry(fo, textvariable=V["outdir"], width=24).grid(row=0, column=1, sticky="ew",
                                                           padx=(12, 0), pady=3)
    ttk.Button(fo, text="...", width=3,
               command=lambda: V["outdir"].set(filedialog.askdirectory()
                                               or V["outdir"].get())
               ).grid(row=0, column=2, padx=(8, 0), pady=3)
    chip(fo, "output").grid(row=0, column=3, padx=(8, 0), pady=3)
    ttk.Checkbutton(fo, text="comment header", variable=V["comments"]).grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
    label_width = max(label.winfo_reqwidth() for label in field_labels)
    for frame in (fg, fe, fo):
        frame.columnconfigure(0, minsize=label_width)

    # right pane: the report; the run buttons are in the footer -------------------
    report = ui.LogView(right, wrap="none", height=26)
    report.grid(row=0, column=0, sticky="nsew")
    txt = report.text
    status = footer.status
    status.configure(foreground="#0a7")
    bar = ttk.Frame(footer.buttons, style="Footer.TFrame")
    bar.pack(side="right")
    palette_row = ttk.Frame(bar, style="Footer.TFrame")
    palette_row.pack(side="bottom", fill="x", pady=(8, 0))
    ttk.Label(palette_row, text="component colours", style="Footer.TLabel").pack(side="left")
    ttk.Combobox(palette_row, textvariable=V["palette"], values=palette_choices(),
                 width=14, state="readonly").pack(side="left", padx=(8, 0))
    chip(palette_row, "palette").pack(side="left", padx=(8, 0))
    recheck_btn = ttk.Button(bar, text="Re-check", style="Footer.TButton")
    recheck_btn.pack(side="left")
    list_btn = ttk.Button(bar, text="List data set", style="Footer.TButton")
    list_btn.pack(side="left", padx=8)
    preview_btn = ttk.Button(bar, text="Preview", style="Footer.TButton")
    preview_btn.pack(side="left", padx=(0, 8))
    read_btn = ttk.Button(bar, text="Read", style="Accent.TButton")
    read_btn.pack(side="left")

    def show(s):
        txt.delete("1.0", "end")
        txt.insert("1.0", s)

    def fnum(key, default=None):
        text = V[key].get().strip()
        if text.lower() in ("", "auto", "none") and default is not None:
            return default
        try:
            return float(text)
        except ValueError:
            raise ValueError("%s: %r is not a number" % (GUI_TERMS.get(key, key), text))

    def optional(key):
        text = V[key].get().strip()
        if text.lower() in ("", "auto", "none"):
            return None
        return fnum(key)

    def precision():
        text = V["precision"].get().strip() or "%d" % DEFAULT_PRECISION
        try:
            return int(text)
        except ValueError:
            raise ValueError("precision: %r is not a whole number" % text)

    def chosen_formats():
        return tuple(fmt for fmt in FORMATS if V["fmt:" + fmt].get())

    # ---- data folder state -----------------------------------------------
    def refresh_data(announce=True):
        """Search again, list the folder, and set the banner and buttons.

        The located folder is fed back into every search and forgotten once
        it stops holding the data set, so a folder since moved is not named
        for the rest of the session as though it were in force.  The banner,
        the status line and the buttons all follow from the one answer here,
        so they cannot disagree; the window opens whatever the answer.
        """
        explicit = state["explicit"]
        data = find_data(explicit)
        dropped = ""
        if explicit and (not data or data.source != "explicit"):
            dropped = explicit
            state["explicit"] = explicit = None
            data = find_data(None)
        state["data"] = data
        state["catalogue"] = load_catalogue(data.folder) if data else None
        message = data_message(data, state["catalogue"], explicit,
                               explicit_label=LOCATED_LABEL)
        if dropped:
            message = ("The folder picked earlier with Locate data folder... no "
                       "longer holds the data set,\nso it has been forgotten:\n  %s\n\n%s"
                       % (dropped, message))
        ready = "normal" if data else "disabled"
        for widget in (read_btn, preview_btn, list_btn, browse_btn, all_chk):
            widget.configure(state=ready)
        if data:
            catalogue = state["catalogue"]
            data_lbl.configure(
                text="Ropelength data at %s  (%s) - %d entries%s"
                     % (data.folder, source_label(data.source, LOCATED_LABEL),
                        len(catalogue.names),
                        "" if catalogue.has_index else ", no %s" % INDEX_FILE),
                fg="#0a7")
            status.configure(text="data set ready", foreground="#0a7")
        else:
            V["all"].set(False)
            data_lbl.configure(
                text="The ropelength data set was not found - nothing can be read until "
                     "it is. Use Locate data folder... and pick the folder holding "
                     "<name>%s." % CURVE_SUFFIX, fg="#c00")
            status.configure(text="data set not found", foreground="#c00")
        if announce:
            show(message + "\n")
        return data

    def do_locate():
        start = state["data"].folder if state["data"] else os.path.expanduser("~")
        path = filedialog.askdirectory(
            title="Locate the ropelength data folder (it holds <name>_best.vect)",
            initialdir=start)
        if not path:
            return
        previous = state["explicit"]
        state["explicit"] = path
        data = refresh_data()
        if not data or data.source != "explicit":
            # Do not keep a folder that leads nowhere: Re-check would go on
            # reporting it.  Revert, so the report describes the folder in force.
            state["explicit"] = previous
            data = refresh_data()
            lines = ["No <name>%s files in:" % CURVE_SUFFIX, "  %s" % path, "",
                     "Pick the folder that holds 3_1_best.vect and %s," % INDEX_FILE,
                     "or the folder above it."]
            if data:
                lines += ["", "Still using %s" % data.folder]
            messagebox.showerror("Not the ropelength data", "\n".join(lines))
            return
        if save_prefs(data.folder):
            status.configure(text="data set ready - remembered for next time",
                             foreground="#0a7")

    locate_btn.configure(command=do_locate)
    recheck_btn.configure(command=lambda: refresh_data())

    # ---- listing and browsing ------------------------------------------------
    def gui_names():
        names = split_names(V["names"].get())
        if V["all"].get() and state["catalogue"]:
            names += filter_catalogue(state["catalogue"].names, prime_only=V["prime"].get())
        return names

    def do_list():
        catalogue = state["catalogue"]
        if not catalogue:
            return
        names = filter_catalogue(catalogue.names, prime_only=V["prime"].get())
        show("%d of %d entries in %s, in crossing-number order%s\n\n"
             % (len(names), len(catalogue.names), catalogue.folder,
                " (prime knots and links only)" if V["prime"].get() else "")
             + "\n".join(entry_rows(names, catalogue)) + "\n")
        status.configure(text="listed %d entries" % len(names), foreground="#0a7")

    def do_browse():
        catalogue = state["catalogue"]
        if not catalogue:
            return
        top = tk.Toplevel(root)
        top.title("Ropelength data set")
        top.transient(root)
        top.minsize(420, 480)
        frm = ttk.Frame(top, padding=(16, 14, 16, 14))
        frm.pack(fill="both", expand=True)
        frm.rowconfigure(2, weight=1)
        frm.columnconfigure(0, weight=1)
        query = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=query).grid(row=0, column=0, sticky="ew")
        ttk.Label(frm, text="filter", style="Hint.TLabel").grid(row=0, column=1, padx=(8, 0))
        ttk.Checkbutton(frm, text="prime knots and links only", variable=V["prime"],
                        command=lambda: repopulate()).grid(row=1, column=0, columnspan=3,
                                                           sticky="w", pady=(6, 0))
        box = tk.Listbox(frm, selectmode="extended", font="AzbMono", height=22,
                         width=58, activestyle="none")
        box.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        bsb = ttk.Scrollbar(frm, orient="vertical", command=box.yview)
        bsb.grid(row=2, column=2, sticky="ns", pady=(8, 0))
        box.configure(yscrollcommand=bsb.set)
        count = ttk.Label(frm, text="", style="Hint.TLabel")
        count.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))
        width = max([len(n) for n in catalogue.names] + [4])
        shown = []

        def repopulate(*_args):
            names = filter_catalogue(catalogue.names, query.get(), V["prime"].get())
            box.delete(0, "end")
            shown[:] = names
            for name in names:
                info = parse_name(name)
                entry = catalogue.entry(name)
                box.insert("end", "%-*s  %2s cr  %s  %9s"
                           % (width, name, info.crossings if info else "?",
                              "%d comp" % info.components if info else "      ",
                              _fmt(entry.ropelength, 6)))
            count.configure(text="%d of %d shown" % (len(names), len(catalogue.names)))

        query.trace_add("write", repopulate)
        repopulate()

        def add_selected():
            picked = [shown[i] for i in box.curselection()]
            if not picked:
                return
            current = V["names"].get().strip()
            V["names"].set((current + " " + " ".join(picked)).strip())
            top.destroy()

        row = ttk.Frame(frm)
        row.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Button(row, text="Add selected", command=add_selected,
                   style="Accent.TButton").pack(side="right")
        ttk.Button(row, text="Cancel", command=top.destroy).pack(side="right", padx=8)
        box.bind("<Double-Button-1>", lambda e: add_selected())
        top.bind("<Escape>", lambda e: top.destroy())
        top.focus_set()

    list_btn.configure(command=do_list)
    browse_btn.configure(command=do_browse)

    # ---- preview and read --------------------------------------------------
    def settings():
        scale = fnum("scale", 1.0)
        diameter = optional("diameter")
        digits = precision()
        check_settings(scale, diameter, digits, GUI_TERMS)
        return scale, diameter, digits

    def do_preview():
        data = state["data"]
        if not data:
            return
        try:
            names = gui_names()
            scale, _diameter, _digits = settings()
            palette = check_palette(V["palette"].get())
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Preview failed", str(exc))
            return
        if not names:
            messagebox.showwarning("Nothing to preview",
                                   "No names given.\n\nType a name such as 3_1, or pick "
                                   "some with Browse data set...")
            return
        status.configure(text="writing a preview ...", foreground="#888")
        root.update_idletasks()
        try:
            opened, note, result = preview_entries(
                names, data.folder, scale, V["centre"].get(), V["mirror"].get(),
                palette, state["catalogue"], GUI_TERMS)
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Preview failed", str(exc))
            status.configure(text="preview failed", foreground="#c00")
            return
        lines = ["PREVIEW", "",
                 "%d viewer window(s) opening. These files are temporary and were NOT"
                 % len(opened),
                 "written to the output folder - use Read to keep them.", ""]
        lines += ["  " + os.path.basename(path) for path in opened]
        if note:
            lines += ["", note]
        lines += failed_lines(result)
        show("\n".join(lines) + "\n")
        status.configure(text="preview open -- nothing written to the output folder",
                         foreground="#0a7")

    def do_read():
        data = state["data"]
        if not data:
            return
        try:
            names = gui_names()
            scale, diameter, digits = settings()
            formats = chosen_formats()
            palette = check_palette(V["palette"].get())
            outdir = V["outdir"].get().strip() or default_gui_outdir()
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Read failed", str(exc))
            status.configure(text="failed", foreground="#c00")
            return
        if not names:
            messagebox.showwarning("Nothing to read",
                                   "No names given.\n\nType a name such as 3_1, pick some "
                                   "with Browse data set..., or tick Entire data set.")
            return
        if not formats:
            messagebox.showwarning("No format", "Tick at least one format to write.")
            return
        status.configure(text="reading %d name(s) - the window is busy until it is done"
                              % len(names), foreground="#888")
        show("reading %d name(s) from %s\n  writing to %s\n"
             % (len(names), data.folder, os.path.abspath(outdir)))
        root.update_idletasks()
        try:
            result = extract(names, outdir, data.folder, formats=formats, scale=scale,
                             centre=V["centre"].get(), mirror=V["mirror"].get(),
                             split=V["split"].get(), precision=digits,
                             comments=V["comments"].get(), diameter=diameter,
                             palette=palette, catalogue=state["catalogue"],
                             terms=GUI_TERMS)
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Read failed", str(exc))
            status.configure(text="failed", foreground="#c00")
            show("read failed:\n  %s\n" % exc)
            return
        text = describe(result, state["catalogue"], scale, V["centre"].get(),
                        V["mirror"].get())
        lines = [text] if text else []
        lines += written_lines(result)
        if result["written"]:
            first = next((f["path"] for f in result["files"]
                          if f["format"] in ("coordinate", "vect")), None)
            if first:
                lines += ["", "Load %s in Curve It as the Curve XYZ/txt input."
                          % os.path.basename(first)]
        lines += component_guidance(result["files"]) + failed_lines(result)
        show("\n".join(lines) + "\n")
        written = result["written"]
        if result["failed"] and written:
            status.configure(text="done -- %d file(s), %d name(s) failed"
                                  % (len(written), len(result["failed"])), foreground="#c00")
        elif written:
            status.configure(text="done -- %d file(s)" % len(written), foreground="#0a7")
        else:
            status.configure(text="nothing written", foreground="#c00")

    preview_btn.configure(command=do_preview)
    read_btn.configure(command=do_read)

    refresh_data(announce=False)
    show(data_message(state["data"], state["catalogue"], state["explicit"],
                      explicit_label=LOCATED_LABEL) + "\n\n"
         "Type names such as 3_1 6_3_2, or pick them with Browse data set...\n"
         "Read writes each one in every format ticked under Formats.\n\n"
         "Order of operations:\n"
         "    <name>_best.vect in the data folder\n"
         "      -> read with vect_io      <- closure from the file: all closed\n"
         "      -  centre on vertex mean  <- optional; a translation\n"
         "      x  scale                  <- 1: the tight rope is 1 thick\n"
         "      -  mirror z -> -z         <- optional\n"
         "      -> coordinate XYZ, VECT, fake PDB, molecular XYZ, GLB, STL\n\n"
         "Click any  ?  for an explanation and examples.\n")

    # Open as large as the content asks, within the screen; the settings on
    # the left scroll in a shorter window.
    root.update_idletasks()
    form.canvas.configure(width=left.winfo_reqwidth(), height=left.winfo_reqheight())
    status.grid_configure(sticky="nwe", pady=(
        max(0, (read_btn.winfo_reqheight() - status.winfo_reqheight()) // 2), 0))
    root.update_idletasks()
    root.geometry("%dx%d" % (min(root.winfo_reqwidth(), root.winfo_screenwidth() - 40),
                             min(root.winfo_reqheight(), 900, root.winfo_screenheight() - 90)))
    root.after(120, lambda: (root.lift(), root.attributes("-topmost", True),
                             root.after(400, lambda: root.attributes("-topmost", False))))
    root.mainloop()
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        return run_gui()
    args = parser.parse_args(argv)
    if args.gui:
        return run_gui(args.data, args.outdir if args.outdir != DEFAULT_OUTDIR else None,
                       args.palette)
    if not (args.names or args.all or args.list_names or args.check):
        parser.error("give at least one name, or --all, --list, --check (or use --gui)")
    try:
        return run_cli(args)
    except FileNotFoundError as exc:
        print("error: cannot open %s" % exc.filename, file=sys.stderr)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
