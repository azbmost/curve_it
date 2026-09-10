#!/usr/bin/env python3
"""
kp2xyz.py -- pull knots and links out of KnotPlot and write them as XYZ curves.

KnotPlot ships a catalogue of several hundred knots and links in its own
binary format.  This tool drives KnotPlot headlessly, asks it to dump the
coordinates of each entry, and rewrites them in the plain-coordinate
convention the rest of Curve It reads:

    x y z          <- component A
    x y z
                   <- a blank line separates components
    x y z          <- component B
    x y z

A target is either a catalogue name in KnotPlot's own dotted Rolfsen style
(4.1, 6.3.2) or a KnotPlot generator command in quotes ("torus 2 3").  A
closed curve is written without repeating its first vertex.  Closure is
measured per component rather than assumed, so a target that produces an
open arc is written, and labelled, as an open arc.

MIRROR IMAGES.  The catalogue holds most entries under two spellings, 4.1
and 4_1.  Files with identical contents collapse to one entry by content
hash, which is what turns 585 files into 522, but the two spellings are NOT
simply aliases: of the 93 pairs, 30 differ, and 28 of those 30 are the two
mirror-image (chiral) forms of the same knot type.  Both forms are kept.
8.19 has writhe -8.626478 and 8_19 +8.626478; 7.3 is -7.123896 and 7_3
+7.045895.  Picking between 8.19 and 8_19 is picking a handedness, not a
duplicate.  The two remaining pairs, 9.20 / 9_20 (+6.346644 and +6.346636)
and 9.35 / 9_35 (+7.539209 and +7.539219), share a handedness: they are two
slightly DIFFERENT same-handed conformations, not one conformation stored
twice under two names, which is why their files differ by content hash and
their writhes differ in the fifth decimal.

Every writhe quoted in this file was measured on the shipped conformations
with Curve It's own cal_xyz_total_curvature_writheV2.py at its defaults.  A
different estimator or discretisation gives visibly different figures for
the same curve, so figures from two of them are never listed side by side.

Any target can also be written reflected, with --mirror, --mirror-axis and
--both.  Negating exactly one coordinate is a reflection, so it produces the
true mirror image and negates writhe and linking number; z is the default
axis because it leaves the x-y projection alone while swapping every
crossing.

The conformations are the ones KnotPlot ships.  This tool exports them as
they are and does NOT relax, tighten or otherwise modify the geometry, so
they are not ideal or ropelength-minimising knots.

KnotPlot itself is a separate program and is not bundled.  It is looked for
in the usual place for the platform; $KNOTPLOT, --knotplot and the GUI's
Locate button override that.  Get it from https://knotplot.com/download/ .

    python3 kp2xyz.py                                  # GUI
    python3 kp2xyz.py --check                          # is KnotPlot there?
    python3 kp2xyz.py --list                           # catalogue names
    python3 kp2xyz.py 4.1 6.3.2 "torus 2 3" -o out
    python3 kp2xyz.py 6.3.2 --nbeads 300 --split -o out
    python3 kp2xyz.py --all -o catalogue

Requires nothing but the standard library and a KnotPlot installation.  Run
with no arguments, or with --gui, for the graphical interface.
"""

import argparse
import dataclasses
import hashlib
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile

__version__ = "1.0"

TOOL_NAME = "KnotPlot to XYZ"


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


# Decimal places per coordinate; matches curve_it.write_plain_xyz_curve.
DEFAULT_PRECISION = 6

# One KnotPlot run handles every target, and the whole 522-entry catalogue
# takes about 0.26 s end to end here, so this ceiling only catches a hung
# process.
DEFAULT_TIMEOUT = 900.0

# The GUI blocks its own event loop while KnotPlot runs, so a hang there
# freezes the window.  The entire catalogue takes about 0.26 s; two minutes is
# already absurdly generous, and it bounds how long the window can lock up.
GUI_TIMEOUT = 120.0

# `--check`, and the GUI's own banner, launch the binary and do nothing but
# quit, which costs about ten milliseconds.  This only catches a binary that
# never answers at all.
PROBE_TIMEOUT = 20.0

DEFAULT_OUTDIR = "kp_xyz"

DOWNLOAD_URL = "https://knotplot.com/download/"

# The first line KnotPlot prints on startup.  Being executable does not make
# a file KnotPlot, so this is what --check actually looks for.
BANNER = "KnotPlot: Hypnagogic Software"

# A component is closed when the gap from its last vertex back to its first
# is about one ordinary step.  Measured over the 777 components of the
# shipped catalogue, every closed one lands between 0.85 and 1.26 median
# steps and the one genuinely open arc (component A of n3.1s) at 4.15, so
# this threshold sits in the empty band between the two populations.
CLOSURE_FACTOR = 2.0

# Reflecting a curve means negating exactly ONE coordinate: that has
# determinant -1, so it is a genuine reflection and gives the mirror image.
# Negating two coordinates is a rotation and hands back the same knot in a
# new pose, so only one named axis is ever touched.  z is the default because
# it leaves x and y alone: the picture you get looking down z is identical
# while every crossing swaps over for under, which is the textbook mirror
# diagram.  Reflection negates writhe, linking number and every other
# chirality-sensitive invariant.
MIRROR_AXES = ("x", "y", "z")
DEFAULT_MIRROR_AXIS = "z"

# How the GUI refers to a path chosen with Locate KnotPlot... in its
# messages.  The GUI has no --knotplot flag, so its messages must not name
# one; the command line passes its own label instead.
LOCATED_LABEL = "the located path"
CLI_LABEL = "--knotplot"

# The two surfaces call the same settings different things, and a message
# shared between them has to be phrased in the vocabulary its reader
# actually has.  The GUI offers fields and buttons, not flags: telling its
# reader to pass --nbeads is the same mistake as telling them to pass
# --knotplot, since neither exists anywhere they can type it.
CLI_TERMS = {
    "nbeads": "--nbeads",
    "precision": "--precision",
    "outdir_fix": "give -o a folder name",
}
GUI_TERMS = {
    "nbeads": "points per curve",
    "precision": "precision",
    "outdir_fix": "put a folder name in the output field",
}

# `coords` writes one of these before each component of a link.
COMPONENT_RE = re.compile(r"^\s*Component\s+(\d+)\s+of\s+(\d+)\s*:", re.I)

# KnotPlot narrates every dump and every mistake, which is how a failed
# target is given a reason instead of a shrug.
DUMP_RE = re.compile(r"data output to file [`'\"]?raw(\d+)\.txt")
ERROR_RE = re.compile(r"\*\*\*\s*(.+?)\s*$")

# Every shipped knot/link file starts with this ascii tag.  The catalogue
# directory also holds scripts and notes, and this is what tells them apart.
MAGIC = b"KnotPlot 1.0"

# Names the KnotPlot binary goes by, in the order they are tried on PATH.
PATH_NAMES = ("knotplot", "KnotPlot", "knotplot.exe")

# Mach-O cpu types, for the architecture sanity note below.
CPU_TYPES = {
    7: "i386", 0x01000007: "x86_64",
    12: "arm", 0x0100000C: "arm64",
    18: "ppc", 0x01000012: "ppc64",
}

# Which tool every writhe figure in this file came from, and at what
# settings.  Naming it is not politeness: a different Gauss estimator, or the
# same one with a different discretisation, gives visibly different numbers
# for the same curve, so a figure is only comparable to another figure from
# the same tool.  These were all measured here, on the shipped
# conformations, with this repo's own writhe tool at its defaults.
WRITHE_TOOL = "cal_xyz_total_curvature_writheV2.py"
WRITHE_SOURCE = ("writhe measured with Curve It's own %s at its defaults"
                 % WRITHE_TOOL)

# Said wherever the 585 -> 522 reduction is mentioned.  The two spellings
# are not interchangeable, and in DNA topology the difference is the whole
# point: 8.19 and 8_19 are opposite handednesses of one knot type.
CHIRALITY_NOTE = (
    "The 585 files of basic/ hold 522 distinct contents. Files that are "
    "byte-identical collapse to one entry by content hash. That is NOT the "
    "same as dropping aliases: of the 93 names that exist in both the "
    "dotted and the underscored spelling, 30 pairs DIFFER and both forms "
    "survive and are listed. 28 of those 30 differ because they are the two "
    "MIRROR-IMAGE (chiral) forms of one knot type: 8.19 has writhe "
    "-8.626478 and 8_19 +8.626478, 7.3 is -7.123896 and 7_3 +7.045895, and "
    "choosing between them chooses handedness. The remaining 2, "
    "9.20 / 9_20 (+6.346644 and +6.346636) and 9.35 / 9_35 (+7.539209 and "
    "+7.539219), have the SAME handedness. Those two are not one "
    "conformation stored twice under two names: their files differ by "
    "content hash and their writhes differ in the fifth decimal, so they "
    "are two slightly different conformations of one handedness. All "
    + WRITHE_SOURCE + ".")

CHIRALITY_ONE_LINER = (
    "note: 585 files -> 522 entries by content hash. The dotted and "
    "underscored spellings are not\n"
    "      plain aliases: 30 of the 93 pairs differ and both are kept. 28 "
    "of those 30 are the two\n"
    "      MIRROR-IMAGE (chiral) forms of one knot type, so 8.19 (writhe "
    "-8.626478) and 8_19\n"
    "      (+8.626478) both appear below; the other 2, 9.20 / 9_20 and "
    "9.35 / 9_35, are two slightly\n"
    "      different same-handed conformations rather than a chiral pair.\n"
    "      " + WRITHE_SOURCE + ".")


# --------------------------------------------------------------------------
# GUI help text: key -> (title, prose, example or None)
# --------------------------------------------------------------------------
HELP = {
    "install": (
        "Where KnotPlot is",
        "KnotPlot is a separate program by Rob Scharein and is not bundled "
        "with Curve It. This tool only drives it, so it has to find the "
        "KnotPlot binary first.\n\n"
        "It is looked for in this order: the path you give with Locate "
        "KnotPlot..., then the KNOTPLOT environment variable (the full path "
        "of the binary), then KNOTPLOT_HOME (the install folder), then "
        "knotplot / KnotPlot / knotplot.exe anywhere on PATH, and finally "
        "the usual install location for this platform.\n\n"
        "Being executable does not make a file KnotPlot, so finding a file "
        "is not the end of it: the banner at the top of this window actually "
        "launches the binary, asks it to quit, and looks for KnotPlot's own "
        "startup banner. That takes about ten milliseconds, and it runs "
        "again on every Re-check and every Locate. A file that runs but does "
        "not answer as KnotPlot is reported as exactly that, and Extract "
        "stays disabled.\n\n"
        "Only the macOS location has been confirmed here. The Windows and "
        "Linux guesses may well be wrong for your machine, so if it is not "
        "found, point at it yourself with Locate KnotPlot... - that always "
        "works, and it is the only thing this window needs.\n\n"
        "Download KnotPlot from " + DOWNLOAD_URL,
        "in this window\n"
        "  Locate KnotPlot...  -> KnotPlot.app, or the folder holding it\n"
        "\n"
        "or from a shell, before Curve It is started\n"
        "  export KNOTPLOT=/Applications/KnotPlot/KnotPlot.app/Contents/MacOS/KnotPlot\n"
        "  export KNOTPLOT_HOME=/Applications/KnotPlot/KnotPlot.app/Contents/Resources\n"
        "  python3 kp2xyz.py --knotplot /path/to/KnotPlot --check"),
    "targets": (
        "What to extract",
        "A space-separated list. Each item is either a catalogue name in "
        "KnotPlot's own dotted style, or a KnotPlot command that builds a "
        "curve.\n\n"
        "Names are passed to KnotPlot's `load`, so they look the way "
        "KnotPlot names them: 3.1 is the trefoil, 4.1 the figure eight, "
        "6.3.2 a three-component link. Use Browse catalogue... to pick them "
        "from a list rather than typing.\n\n"
        "A dotted and an underscored name are not always the same knot. For "
        "28 entries they are the two mirror images of each other, so 8.19 "
        "and 8_19 are the left- and right-handed forms and you have to pick "
        "the one you mean. Click the ? beside Browse catalogue... for the "
        "detail.\n\n"
        "Anything containing a space is treated as a KnotPlot command "
        "instead and passed through untouched, so generators work too. On "
        "the command line such an item has to be quoted. For torus, note "
        "that the order of its two numbers is not the one every text "
        "uses - click the ? beside it.",
        "4.1 6.3.2 \"torus 2 3\"\n"
        "  4.1          -> load 4.1\n"
        "  6.3.2        -> load 6.3.2      (a 3-component link)\n"
        "  torus 2 3    -> the trefoil, built on the fly; see the\n"
        "                  torus  ?  about the index order\n"
        "  8.19 / 8_19  -> the two mirror images,\n"
        "                  writhe -8.626478 / +8.626478"),
    "torus": (
        "torus p q, and which number is which",
        "KnotPlot's `torus p q` winds the curve p times around the z axis, "
        "the axis of the torus, and q times around the tube itself. "
        "Measured here: torus 2 3 goes twice around the axis and three "
        "times around the tube.\n\n"
        "That index order is not universal. The common parametrisation, and "
        "so most software, puts the axis winding first and calls this the "
        "(2,3) torus knot. Other sources, including Adams' The Knot Book, "
        "put the winding around the tube first, and in that convention the "
        "same curve is the (3,2) torus knot. So a (p,q) quoted in a paper "
        "may be KnotPlot's `torus q p`.\n\n"
        "The knot type is safe either way, because T(p,q) and T(q,p) are the "
        "same knot. What differs is the shape you get: the number of lobes, "
        "and the crossings in the view down z. If you are reproducing a "
        "figure and the shape looks wrong, swap the two numbers.",
        "torus 2 3   2 turns about the axis, 3 about the tube\n"
        "            trefoil, 3 crossings seen down z\n"
        "torus 3 2   3 turns about the axis, 2 about the tube\n"
        "            the same trefoil, 4 crossings seen down z\n"
        "torus 2 5   the 5.1 knot, 5 crossings seen down z"),
    "catalogue": (
        "KnotPlot's shipped catalogue, and chirality",
        "The basic/ folder inside KnotPlot's Resources holds every knot and "
        "link it ships, in KnotPlot's own binary format. This tool reads the "
        "folder only to list the names; the coordinates always come out of "
        "KnotPlot itself.\n\n"
        + CHIRALITY_NOTE + "\n\n"
        "So the reduction removes duplicated files, never a knot. Turn it "
        "off to see every filename, including the byte-identical ones.\n\n"
        "If KnotPlot is found but its catalogue folder is not, single "
        "targets and generator commands still work; only the listing and "
        "Entire catalogue are unavailable.",
        "585 files -> 522 distinct entries\n"
        "  93 names exist in both spellings, e.g. 8.19 and 8_19\n"
        "    63 pairs byte-identical  -> one entry kept\n"
        "    30 pairs DIFFER          -> both kept\n"
        "      28 are mirror images    8.19 -8.626478   8_19 +8.626478\n"
        "      2 are same-handed       9.20 +6.346644   9_20 +6.346636\n"
        "        (two conformations)   9.35 +7.539209   9_35 +7.539219\n"
        "  " + WRITHE_SOURCE),
    "all": (
        "Entire catalogue",
        "Extract every entry in basic/ in one go: one KnotPlot run, one .xyz "
        "file per entry, written into the output folder.\n\n"
        "It is fast - the whole catalogue is roughly 522 files and about "
        "60000 points, and the whole run takes about 0.26 s. Any "
        "entry that produces no coordinates is listed at the end and also "
        "written to _failed.txt, with KnotPlot's own output in "
        "_knotplot.log.\n\n"
        "Both spellings of a chiral pair are extracted, since for those 28 "
        "pairs the two files are different knots and not duplicates.\n\n"
        "Mirror image is never implied here. Ticking write both as well "
        "would reflect all 522 entries too and write 1044 files, so it has "
        "to be asked for on purpose.\n\n"
        "Names that would collide after being cleaned up for the filesystem "
        "get -2, -3 suffixes, so nothing is silently overwritten.",
        None),
    "nbeads": (
        "Points per curve",
        "Leave this blank to keep the resolution the conformation was "
        "shipped at, which is usually what you want.\n\n"
        "Fill it in and the curve is resampled with KnotPlot's `refine "
        "nbeads` before the coordinates are dumped. On a link the number is "
        "the TOTAL over all components, split between them roughly in "
        "proportion to their length: 300 on the three-component link 6.3.2 "
        "comes back as 107 + 97 + 95 points.\n\n"
        "This only redistributes points along the same curve. It does not "
        "change the shape, and it is not a relaxation.",
        "6.3.2 with --nbeads 300\n"
        "  component A: 107 points\n"
        "  component B:  97 points\n"
        "  component C:  95 points"),
    "conformations": (
        "The shapes KnotPlot ships",
        "Every curve is the conformation KnotPlot ships for that knot or "
        "link, exported as it is. This tool does not relax, tighten or "
        "smooth the geometry.\n\n"
        "They are clean, smooth embeddings with the right topology, but they "
        "are not tight or ropelength-minimising, so read their length and "
        "curvature as properties of this shape rather than of the knot type. "
        "If you need ideal conformations, load published ideal-knot "
        "coordinates as ordinary .xyz files instead.",
        None),
    "split": (
        "One file per component",
        "A link comes out of KnotPlot as several closed curves. By default "
        "they all go into one .xyz file, separated by blank lines, which is "
        "what Curve It's component selector expects - they arrive as "
        "components A, B, C.\n\n"
        "Curve It concatenates every component of a curve file by default, "
        "so a multi-component file has to be used one component at a time. "
        "The report after a run prints the exact --curve-components command "
        "for each one.\n\n"
        "Switch this on to write each component to its own file instead, "
        "named _A, _B, _C after the entry. Knots with a single component are "
        "written as one file either way.",
        "6.3.2 --split\n"
        "  6.3.2_A.xyz\n"
        "  6.3.2_B.xyz\n"
        "  6.3.2_C.xyz"),
    "mirror": (
        "Mirror image",
        "Writes the reflection of the curve instead of the curve itself, or "
        "alongside it with write both.\n\n"
        "Exactly one coordinate is negated. That has determinant -1, so it "
        "is a genuine reflection and gives the mirror image; negating two "
        "coordinates would be a rotation and would hand back the same knot "
        "in a new pose, so only one axis is ever touched.\n\n"
        "Reflection negates writhe and, on a link, the linking number, along "
        "with every other chirality-sensitive invariant. Measured here with "
        "Curve It's own " + WRITHE_TOOL + " at its defaults: 8.19 as shipped "
        "is -8.626478, its z-mirror +8.626478 exactly, and KnotPlot's own "
        "chiral partner 8_19 +8.626478 as well - the partner is a separate "
        "conformation, so it agrees to six decimals rather than to the "
        "last bit.\n\n"
        "z is the default axis because it leaves x and y untouched: the "
        "picture you get looking down z is identical while every crossing "
        "swaps over for under, which is the textbook mirror diagram. x or y "
        "give the same knot type, just presented differently.\n\n"
        "For an amphichiral knot such as 4.1 the mirror image is the SAME "
        "knot type, so what comes out is another conformation of 4.1 rather "
        "than a new knot.\n\n"
        "KnotPlot already ships both forms of the 28 chiral pairs, so for "
        "those you can either mirror one spelling or simply ask for the "
        "other name: 8.19 and 8_19 are already each other's mirror.\n\n"
        "A link is reflected as a whole, in one reflection, so its "
        "components stay in register and the linking number flips "
        "consistently with the writhe. Reflected files are named "
        "name_mirror.xyz, and their comment header records that they were "
        "reflected and on which axis, so a file on disk is never ambiguous "
        "about its handedness.",
        "8.19 --both\n"
        "  8.19.xyz          writhe -8.626478   as shipped\n"
        "  8.19_mirror.xyz   writhe +8.626478   reflected, z -> -z\n"
        "  8_19              writhe +8.626478   KnotPlot's own partner\n"
        "write both is never implied by Entire catalogue: that would\n"
        "turn 522 entries into 1044 files."),
    "precision": (
        "Decimal places",
        "How many decimals each coordinate is written with. Six matches the "
        "rest of Curve It and is far finer than anything KnotPlot's "
        "conformations actually resolve.\n\n"
        "Lower it only to make the files smaller; the geometry is unchanged.",
        None),
    "output": (
        "Output folder and file names",
        "One .xyz per target, named after the target with anything awkward "
        "for a filename replaced by an underscore, so 6.3.2 stays 6.3.2 and "
        "\"torus 2 3\" becomes torus_2_3.xyz.\n\n"
        "Each file carries a comment header naming the tool, the KnotPlot "
        "target and each component's point count and closure, then plain "
        "x y z rows with a blank line between components. Closure is "
        "measured, not assumed: almost everything KnotPlot ships is closed "
        "and is written without repeating its first vertex, but a target "
        "that yields an open arc is written as closed=no. Load one in Curve "
        "It as the Curve XYZ/txt input, or feed it straight to xyz2model.py."
        "\n\nUntick the comment header to write bare coordinates only.",
        "# KnotPlot to XYZ v1.0 -- 3 component(s) from KnotPlot `6.3.2`\n"
        "  # columns: x y z ; blank lines separate components ; ...\n"
        "  # component A: closed=yes points=107\n"
        "  -1.234567 0.891011 0.000000\n"
        "  ..."),
}


# --------------------------------------------------------------------------
# Finding the KnotPlot installation
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class KnotPlotInstall:
    """One usable KnotPlot installation.

    Truthy when the executable is really there and really executable, so
    callers can write `if install:` and get a meaningful answer.
    """

    executable: str = ""
    resources: str = ""
    catalogue: str = ""
    source: str = ""

    def __bool__(self):
        return bool(self.executable) and _is_executable(self.executable)

    @property
    def has_catalogue(self):
        return bool(self.catalogue) and os.path.isdir(self.catalogue)


def _is_executable(path):
    """True when `path` is a file this platform could actually launch.

    On Windows os.access(path, os.X_OK) degrades to an existence test, so
    every readable file would pass and a $KNOTPLOT pointing at a readme
    would be accepted as the program.  There the name has to look like
    KnotPlot as well; --check then confirms it really is.
    """
    if not path or not os.path.isfile(path) or not os.access(path, os.X_OK):
        return False
    if os.name == "nt":
        lowered = os.path.basename(path).lower()
        return lowered in tuple(name.lower() for name in PATH_NAMES)
    return True


def _expand_candidate(path):
    """Every binary a user-supplied path could plausibly mean.

    A file chooser on macOS hands back KnotPlot.app itself, which is a
    directory, and people type the install folder as often as the binary, so
    a directory is expanded into the binaries it might contain.  bin/ is in
    the list because that is how a Linux tarball unpacked into /opt/knotplot
    lays itself out.
    """
    path = os.path.expanduser(str(path).strip())
    if not path:
        return []
    return [
        path,
        os.path.join(path, "Contents", "MacOS", "KnotPlot"),
        os.path.join(path, "KnotPlot.app", "Contents", "MacOS", "KnotPlot"),
        os.path.join(path, "KnotPlot", "KnotPlot.app", "Contents", "MacOS", "KnotPlot"),
        os.path.join(path, "knotplot"),
        os.path.join(path, "KnotPlot"),
        os.path.join(path, "knotplot.exe"),
        os.path.join(path, "bin", "knotplot"),
        os.path.join(path, "bin", "KnotPlot"),
        os.path.join(path, "bin", "knotplot.exe"),
    ]


def _default_locations():
    """Per-platform guesses at where KnotPlot was installed.

    Only the first macOS entry has actually been confirmed; the Windows and
    Linux paths are best-effort, which is why $KNOTPLOT and --knotplot exist
    and why every message mentions them.
    """
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        return [
            "/Applications/KnotPlot/KnotPlot.app/Contents/MacOS/KnotPlot",
            os.path.join(home, "Applications", "KnotPlot", "KnotPlot.app",
                         "Contents", "MacOS", "KnotPlot"),
            "/Applications/KnotPlot.app/Contents/MacOS/KnotPlot",
            os.path.join(home, "Applications", "KnotPlot.app",
                         "Contents", "MacOS", "KnotPlot"),
        ]
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        roots = [os.environ.get("ProgramFiles", "C:\\Program Files"),
                 os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                 local,
                 os.path.join(local, "Programs") if local else "",
                 "C:\\"]
        return [os.path.join(root, "KnotPlot", "knotplot.exe")
                for root in roots if root]
    # A tarball unpacked into one of these roots usually puts the binary in
    # bin/, so both layouts are probed for every root.
    roots = [os.path.join(home, "knotplot"),
             os.path.join(home, "KnotPlot"),
             "/usr/local/knotplot",
             "/usr/local/KnotPlot",
             "/opt/knotplot",
             "/opt/KnotPlot"]
    out = []
    for root in roots:
        out.append(os.path.join(root, "knotplot"))
        out.append(os.path.join(root, "bin", "knotplot"))
    return out


def _candidates(explicit=None):
    """(path, source) pairs in the order find_knotplot() tries them."""
    out = []
    if explicit:
        out += [(p, "explicit") for p in _expand_candidate(explicit)]
    env_binary = os.environ.get("KNOTPLOT", "").strip()
    if env_binary:
        out += [(p, "KNOTPLOT env") for p in _expand_candidate(env_binary)]
    env_home = os.environ.get("KNOTPLOT_HOME", "").strip()
    if env_home:
        out += [(p, "KNOTPLOT_HOME env") for p in _expand_candidate(env_home)]
    for name in PATH_NAMES:
        found = shutil.which(name)
        if found:
            out.append((found, "PATH"))
    out += [(p, "default location") for p in _default_locations()]
    return out


def _derive_resources(executable):
    """Walk up from the binary to Resources/ and Resources/basic.

    On macOS that is Contents/MacOS/KnotPlot -> Contents/Resources; elsewhere
    the layout is anyone's guess, so the test is simply "does this directory
    contain basic/".
    """
    exe_dir = os.path.dirname(os.path.abspath(executable))
    parent = os.path.dirname(exe_dir)
    candidates = []
    if os.path.basename(exe_dir) == "MacOS" and os.path.basename(parent) == "Contents":
        candidates.append(os.path.join(parent, "Resources"))
    env_home = os.environ.get("KNOTPLOT_HOME", "").strip()
    if env_home:
        candidates.append(os.path.expanduser(env_home))
    candidates += [
        exe_dir,
        os.path.join(exe_dir, "Resources"),
        parent,
        os.path.join(parent, "Resources"),
        os.path.join(parent, "share", "knotplot"),
        os.path.join(parent, "knotplot"),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, "basic")):
            return candidate, os.path.join(candidate, "basic")
    # No catalogue, but a Resources directory is still worth reporting: the
    # install is perfectly usable for generator commands such as `torus 2 3`.
    for candidate in candidates:
        if candidate and os.path.isdir(candidate) \
                and os.path.basename(candidate) == "Resources":
            return candidate, ""
    return "", ""


def find_knotplot(explicit=None):
    """First usable KnotPlot installation, or None.

    Order: the explicit path, $KNOTPLOT, $KNOTPLOT_HOME, PATH, then the
    per-platform default locations.  "Usable" here means only that the file
    exists and can be launched; probe_knotplot() is what confirms that the
    thing launched is actually KnotPlot.
    """
    for path, source in _candidates(explicit):
        if _is_executable(path):
            resources, catalogue = _derive_resources(path)
            return KnotPlotInstall(os.path.abspath(path), resources, catalogue, source)
    return None


def searched_locations():
    """Every path find_knotplot() probes, for the "not installed" message.

    Listed in the order they are really tried, PATH included: PATH comes
    before the per-platform defaults, and a list that said otherwise would
    misdescribe which install wins.
    """
    out = []
    if not os.environ.get("KNOTPLOT", "").strip():
        out.append("$KNOTPLOT  (not set)")
    if not os.environ.get("KNOTPLOT_HOME", "").strip():
        out.append("$KNOTPLOT_HOME  (not set)")
    path_note = "%s  (anywhere on PATH)" % ", ".join(PATH_NAMES)
    noted_path = False
    seen = set()
    for path, source in _candidates(None):
        if source == "default location" and not noted_path:
            out.append(path_note)
            noted_path = True
        text = path if source == "default location" else "%s  (%s)" % (path, source)
        if text not in seen:
            seen.add(text)
            out.append(text)
    if not noted_path:
        out.append(path_note)
    return out


def _macho_architectures(path):
    """Best-effort list of the CPU architectures inside a Mach-O binary.

    Used only to turn a launch failure into a sentence. KnotPlot ships stale
    helpers next to the app - utilities/kpfsnoop dies with "bad CPU type in
    executable" - and the same could happen to the app itself on a machine
    whose architecture it was never built for.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return []
    if len(head) < 8:
        return []
    magic = struct.unpack(">I", head[:4])[0]
    if magic in (0xCAFEBABE, 0xCAFEBABF):       # universal ("fat") binary
        count = struct.unpack(">I", head[4:8])[0]
        step = 20 if magic == 0xCAFEBABE else 32
        archs = []
        for index in range(min(count, 32)):
            offset = 8 + index * step
            if offset + 4 > len(head):
                break
            cpu = struct.unpack(">i", head[offset:offset + 4])[0]
            archs.append(CPU_TYPES.get(cpu, "cpu%d" % cpu))
        return archs
    for endian in ("<", ">"):
        if struct.unpack(endian + "I", head[:4])[0] in (0xFEEDFACE, 0xFEEDFACF):
            cpu = struct.unpack(endian + "i", head[4:8])[0]
            return [CPU_TYPES.get(cpu, "cpu%d" % cpu)]
    return []


def architecture_note(executable):
    """A sentence about this binary's architecture, or "" when it is fine."""
    if sys.platform != "darwin" or not executable:
        return ""
    archs = _macho_architectures(executable)
    if not archs:
        return ""
    import platform

    machine = platform.machine()
    native = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    if native in archs:
        return ""
    if native == "arm64" and "x86_64" in archs:
        return ("note: this is an %s build, so it runs through Rosetta 2 on "
                "Apple silicon." % "/".join(archs))
    return ("warning: this is an %s build and this Mac is %s, so it will "
            "probably refuse to launch with 'bad CPU type in executable'."
            % ("/".join(archs), machine))


def probe_knotplot(install, timeout=PROBE_TIMEOUT):
    """Launch the binary and confirm that what launched really is KnotPlot.

    Returns (ok, detail).  An executable file is not evidence of anything:
    KNOTPLOT=/bin/ls used to be reported as a KnotPlot installation and then
    fail much later, on the extraction, with nothing to explain it.  So the
    check runs a session that does nothing but quit and looks for KnotPlot's
    own startup banner.  That session costs about ten milliseconds, which is
    cheap enough for the GUI to run it on every Re-check and every Locate;
    the full 522-entry catalogue takes about 0.26 s.
    """
    if not install:
        return False, "there is no binary to run"
    try:
        with tempfile.TemporaryDirectory() as workdir:
            # -keepdir keeps KnotPlot in the working directory it is given,
            # so a throwaway one guarantees the probe writes nothing anywhere
            # the user would notice.
            proc = subprocess.run(
                [install.executable, "-keepdir", "-nographics", "-stdin"],
                input="quit\n", cwd=workdir, capture_output=True, text=True,
                timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "it did not answer within %gs" % timeout
    except OSError as exc:
        note = architecture_note(install.executable)
        return False, "%s%s" % (exc, "; " + note if note else "")
    output = (proc.stdout or "") + (proc.stderr or "")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if BANNER.lower() in output.lower():
        return True, next((line for line in lines
                           if BANNER.lower() in line.lower()), BANNER)
    detail = "it printed no %s banner" % BANNER.split(":")[0]
    if lines:
        detail += "; its first line was %r" % lines[0][:120]
    else:
        detail += " and in fact printed nothing at all"
    return False, detail


def _gui_facing(explicit_label):
    """True when this message is being written for the GUI's own reader.

    The GUI passes LOCATED_LABEL because it has no --knotplot to name, so
    that one label already identifies the audience: everything else written
    for the same reader can key off it and offer the button they have
    instead of the flag they do not.  Getting this wrong is not cosmetic -
    a report pane that says "use --knotplot" is advice its reader cannot
    act on.
    """
    return explicit_label != CLI_LABEL


def _terms(explicit_label):
    """The GUI's or the command line's name for each shared setting."""
    return GUI_TERMS if _gui_facing(explicit_label) else CLI_TERMS


def override_problems(explicit=None, explicit_label=CLI_LABEL):
    """Complaints about overrides that are set but hold no KnotPlot.

    A wrong $KNOTPLOT still falls through to the rest of the search, which
    is the documented order, but quietly using some other install instead
    would be a nasty surprise, so it is always said out loud - on every run,
    not only under --check.  `explicit` is the --knotplot path or the GUI's
    Locate choice; `explicit_label` names it the way the reader of the
    message can act on, so the GUI does not quote a flag it does not have.
    """
    checks = [("$KNOTPLOT is set to %s, which holds no usable KnotPlot binary",
               os.environ.get("KNOTPLOT", "").strip()),
              ("$KNOTPLOT_HOME is set to %s, which holds no usable KnotPlot "
               "binary", os.environ.get("KNOTPLOT_HOME", "").strip())]
    if explicit:
        checks.insert(0, (explicit_label + " %s holds no usable KnotPlot "
                          "binary", str(explicit).strip()))
    out = []
    for template, value in checks:
        if not value:
            continue
        if not any(_is_executable(p) for p in _expand_candidate(value)):
            out.append(template % value)
    return out


def installation_message(install, explicit=None, probe=None,
                         explicit_label=CLI_LABEL):
    """Ready-to-show text describing what was or was not found.

    `explicit` is the --knotplot path or the GUI's Locate choice, so that a
    bad one is named in the body rather than only in a separate warning, and
    `explicit_label` is how the caller's own user refers to it -- which also
    says which surface is going to show this text, so the advice below
    offers remedies that surface actually has.  `probe` is the (ok, detail)
    pair from probe_knotplot() when the binary was actually launched, and
    None when it was not.
    """
    problems = override_problems(explicit, explicit_label)
    gui = _gui_facing(explicit_label)
    if not install:
        lines = ["KnotPlot was not found, so nothing can be extracted yet.",
                 ""]
        lines += ["  %s" % problem for problem in problems]
        if problems:
            lines.append("")
        lines += [
            "KnotPlot is a separate program and is not bundled with Curve It.",
            "Download it from %s" % DOWNLOAD_URL,
            ""]
        if gui:
            # Only what this window can actually do.  The environment
            # variables are read when the program starts, so they are real
            # advice here too, but they belong under their own caption
            # rather than inline beside a button.
            lines += [
                "Point this tool at an existing install:",
                "  use the Locate KnotPlot... button at the top of the window",
                "",
                "or, in a shell before starting Curve It:",
                "  export KNOTPLOT=/path/to/the/KnotPlot/binary",
                "  export KNOTPLOT_HOME=/path/to/the/install/folder"]
        else:
            lines += [
                "Point this tool at an existing install in any of these ways:",
                "  --knotplot /path/to/KnotPlot on the command line",
                "  export KNOTPLOT=/path/to/the/KnotPlot/binary",
                "  export KNOTPLOT_HOME=/path/to/the/install/folder",
                "  the Locate KnotPlot... button in the graphical interface"]
        lines += [
            "",
            "Looked in:"]
        lines += ["  %s" % item for item in searched_locations()]
        return "\n".join(lines)

    if probe is not None and not probe[0]:
        # "KnotPlot found at /bin/ls" was the old first line here, and it was
        # the wrong thing to say about a file that had just failed to answer.
        lines = ["A binary was found at %s  (%s), but it is NOT KnotPlot"
                 % (install.executable, install.source)]
    else:
        lines = ["KnotPlot found at %s  (%s)"
                 % (install.executable, install.source)]
    # An override that points nowhere is ignored by the search, so say so
    # rather than let it look as though it had been honoured.
    lines += ["  note: %s; ignored" % problem for problem in problems]
    if probe is not None:
        ok, detail = probe
        if ok:
            lines.append("  it runs, and answers as KnotPlot: %s" % detail)
        else:
            lines.append("  FOUND BUT DID NOT RESPOND AS KNOTPLOT: %s" % detail)
            lines.append("  Being executable does not make a file KnotPlot, "
                         "so nothing can be")
            if gui:
                lines.append("  extracted with this one. Use Locate "
                             "KnotPlot... at the top of the")
                lines.append("  window to point at the real binary.")
                lines.append("  ($KNOTPLOT and $KNOTPLOT_HOME do the same "
                             "thing, set in a shell")
                lines.append("  before Curve It starts.)")
            else:
                lines.append("  extracted with this one. Point at the real "
                             "binary with --knotplot")
                lines.append("  or $KNOTPLOT, or with Locate KnotPlot... in "
                             "the graphical interface.")
            lines.append("  Or download KnotPlot from %s" % DOWNLOAD_URL)
    note = architecture_note(install.executable)
    if note:
        lines.append("  %s" % note)
    lines.append("  resources: %s" % (install.resources or "not found"))
    if install.has_catalogue:
        lines.append("  catalogue: %s" % install.catalogue)
    else:
        lines.append("  catalogue: not found - single targets and generator "
                     "commands still work,")
        lines.append("             but the catalogue listing and Entire "
                     "catalogue do not.")
        lines.append("             Set KNOTPLOT_HOME to the folder holding "
                     "basic/ to fix that.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The catalogue
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


def catalogue_names(install, dedupe=True):
    """Every entry in the catalogue that is really a KnotPlot knot/link file.

    With dedupe on, files with identical contents collapse to the first name
    in sort order, which for a byte-identical pair is the dotted form that
    KnotPlot's own `load` syntax uses.  On a stock install that turns 585
    files into 522 entries.

    That is deduplication by content, not alias removal, and the difference
    matters.  93 names exist in both the dotted and the underscored
    spelling; only 63 of those pairs are byte-identical.  The other 30 pairs
    DIFFER and both forms are listed.  28 of those 30 differ because they
    are the two mirror-image (chiral) forms of one knot type: 8.19 has
    writhe -8.626478 and 8_19 +8.626478.  The remaining 2, 9.20 / 9_20
    (+6.346644 and +6.346636) and 9.35 / 9_35 (+7.539209 and +7.539219),
    have the same handedness, and they too hold different geometry: their
    writhes differ in the fifth decimal, so they are two slightly different
    same-handed conformations rather than one conformation under two names.
    Nothing here ever drops a distinct knot.  (Writhe throughout from
    cal_xyz_total_curvature_writheV2.py at its defaults.)

    `install` is a KnotPlotInstall; a catalogue directory path is accepted
    too, so a caller that already knows where basic/ is can say so.
    """
    if isinstance(install, str):
        directory = install
    elif install:
        directory = install.catalogue
    else:
        raise ValueError(installation_message(None))
    if not directory or not os.path.isdir(directory):
        raise ValueError(
            "KnotPlot's knot catalogue (the basic/ folder inside its "
            "Resources) was not found.\nSingle targets and generator "
            "commands such as 'torus 2 3' still work.\nSet KNOTPLOT_HOME to "
            "the folder that holds basic/ to enable the listing.")

    names = []
    seen = {}
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if not os.path.isfile(path) or entry.endswith(".kps"):
            continue
        try:
            with open(path, "rb") as fh:
                blob = fh.read()
        except OSError:
            continue
        if not blob.startswith(MAGIC):
            continue
        if dedupe:
            digest = hashlib.sha1(blob).hexdigest()
            if digest in seen:
                continue
            seen[digest] = entry
        names.append(entry)
    return names


# --------------------------------------------------------------------------
# Driving KnotPlot
# --------------------------------------------------------------------------
def _flat(text):
    """Collapse whitespace so a name stays one token inside a comment line."""
    return "_".join(str(text).split()) or "-"


def clean_target(target):
    """One target, guaranteed to be a single line, or a ValueError.

    A target is written verbatim into the .kps script, where a line break
    would smuggle an extra command into the KnotPlot session, and again into
    the file's `#` header, where it would split the comment and leave a
    trailing fragment the next reader tries to parse as coordinates.
    """
    text = str(target).strip()
    if "\n" in text or "\r" in text:
        raise ValueError("a target cannot contain a line break: %r" % target)
    return text


def build_script(targets, nbeads, workdir):
    """Write the .kps that loads each target and dumps its coordinates."""
    lines = []
    for index, target in enumerate(targets):
        target = clean_target(target)
        # Empty the workspace first.  A failed `load` leaves the PREVIOUS
        # knot in place and `coords` then happily dumps it again, which would
        # silently write one knot's coordinates under another knot's name.
        # After `delete all` a failed target dumps an empty file instead, and
        # that is what marks it failed below.
        lines.append("delete all")
        # A bare catalogue name is loaded; anything with a space (`torus 2 3`,
        # `lissajous ...`) is passed through as a KnotPlot command.
        lines.append(target if " " in target.strip() else "load %s" % target)
        if nbeads:
            # Plain `nbeads` is deprecated and prints a nag; `refine nbeads`
            # is the supported spelling.
            lines.append("refine nbeads %d" % nbeads)
        # `coords` refuses a path containing `/`, so write bare names and
        # run KnotPlot with its working directory set to workdir.
        lines.append("coords raw%04d.txt" % index)
    lines.append("quit")
    script = os.path.join(workdir, "dump.kps")
    with open(script, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return script


def run_knotplot(install, script, workdir, timeout):
    """Run one headless KnotPlot session and return everything it printed."""
    try:
        with open(script) as fh:
            # -keepdir stops KnotPlot from chdir-ing to its own home
            # directory, which is where the output would otherwise land.
            proc = subprocess.run(
                [install.executable, "-keepdir", "-nographics", "-stdin"],
                stdin=fh, cwd=workdir, capture_output=True, text=True,
                timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ValueError("KnotPlot did not finish within %gs; raise the "
                         "timeout or extract fewer targets at once" % timeout)
    except OSError as exc:
        # A stale or wrong-architecture binary fails here, and the raw
        # OSError ("bad CPU type in executable") is not self-explanatory.
        note = architecture_note(install.executable)
        raise ValueError("could not run KnotPlot at %s\n  %s%s"
                         % (install.executable, exc,
                            "\n  " + note if note else ""))
    return proc.stdout + proc.stderr


def knotplot_errors(log):
    """Target index -> the last `*** ...` KnotPlot printed before its dump.

    KnotPlot names each output file as it writes it, so the dump lines pin
    every complaint to the target that caused it.
    """
    errors = {}
    pending = []
    for line in log.splitlines():
        match = ERROR_RE.search(line)
        if match:
            pending.append(match.group(1))
        match = DUMP_RE.search(line)
        if match:
            if pending:
                errors[int(match.group(1))] = pending[-1]
            pending = []
    return errors


def parse_coords(path):
    """Split a `coords` dump into one point list per component."""
    components = []
    with open(path) as fh:
        for line in fh:
            if COMPONENT_RE.match(line):
                components.append([])
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                xyz = (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError:
                continue
            if not components:
                components.append([])
            components[-1].append(xyz)
    return [c for c in components if c]


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def _distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                     + (a[2] - b[2]) ** 2)


def _median(values):
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        return 0.0
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def component_closure(points):
    """(points, closed) for one component, any repeated first vertex gone.

    Nothing guarantees a target is closed.  `load 4.1` is, but a target is
    any KnotPlot command, and even the shipped catalogue is not uniform:
    component A of n3.1s is a genuine open arc whose two ends sit 4.15
    median steps apart and are each nearer to the middle of the chain than
    to one another.  Claiming closed=yes for it would hand Curve It a
    spurious closing segment.

    So closure is measured: a curve is closed when the gap from its last
    vertex back to its first is about one ordinary step.  Over the 777
    components of the shipped catalogue every closed one lands between 0.85
    and 1.26 median steps, and the single open arc at 4.15, so CLOSURE_FACTOR
    sits in the empty band between them.  An already-repeated first vertex
    is dropped instead, because Curve It's writhe calculation rejects the
    zero-length segment it would otherwise create.
    """
    points = list(points)
    if len(points) < 3:
        return points, False
    steps = [_distance(points[i], points[i + 1])
             for i in range(len(points) - 1)]
    median = _median(steps)
    if median <= 0.0:
        return points, False
    gap = _distance(points[-1], points[0])
    if gap <= 1e-6 * median:
        return points[:-1], True
    return points, gap <= CLOSURE_FACTOR * median


def component_report(components):
    """[{"label", "closed", "points"}] describing what write_xyz would write."""
    out = []
    for index, raw in enumerate(components):
        points, closed = component_closure(raw)
        out.append({"label": component_label(index), "closed": closed,
                    "points": len(points)})
    return out


def mirror_points(points, axis=DEFAULT_MIRROR_AXIS):
    """A reflected copy of `points`: exactly the named coordinate negated.

    Negating one coordinate has determinant -1, so it is a reflection and
    gives the true mirror image; negating two is a rotation and hands back
    the same knot in a new pose, which is why only one axis is ever touched.
    Reflection negates writhe, linking number and every other
    chirality-sensitive invariant.  z is the default because it leaves x and
    y alone: the projection down z is unchanged while every crossing swaps
    over for under.

    Either shape the module passes around is accepted, and the same shape
    comes back: one component, a list of (x, y, z), or a whole file's worth
    of components, a list of those lists.  A multi-component target must be
    reflected in ONE call so that every component gets the same reflection
    and a link's components stay in register.
    """
    if axis not in MIRROR_AXES:
        raise ValueError("the mirror axis must be one of %s, not %r"
                         % (", ".join(MIRROR_AXES), axis))
    index = MIRROR_AXES.index(axis)
    items = list(points)
    if not items:
        return []
    first = items[0]
    nested = isinstance(first, (list, tuple)) and (
        not first or isinstance(first[0], (list, tuple)))
    if nested:
        return [mirror_points(component, axis) for component in items]
    out = []
    for point in items:
        xyz = list(point)
        if len(xyz) < 3:
            raise ValueError("a point needs three coordinates, not %r"
                             % (point,))
        # Negating a zero gives -0.0, which prints as "-0.000000"; adding
        # 0.0 turns it back into 0.0.  This cleans up the NEGATED column
        # only.  The other two are copied through untouched, so a
        # "-0.000000" that KnotPlot already wrote in them survives into the
        # reflected file -- which is right, since reflection is not licence
        # to edit the coordinates it does not touch.
        xyz[index] = -xyz[index] + 0.0
        out.append((xyz[0], xyz[1], xyz[2]))
    return out


def write_xyz(path, components, name, precision=DEFAULT_PRECISION,
              comments=True, mirrored=False,
              mirror_axis=DEFAULT_MIRROR_AXIS):
    """Write the plain-coordinate XYZ file Curve It reads.

    Each component's closure is measured rather than asserted, so the header
    describes what is actually in the file; see component_closure.

    `mirrored` only records in the header that these coordinates were
    reflected and on which axis; mirror_points is what does the reflecting.
    It is recorded because handedness is invisible in a column of numbers,
    and a file on disk must never be ambiguous about which form it holds.
    """
    if not components:
        raise ValueError("there are no components to write")
    fmt = "%%.%df %%.%df %%.%df\n" % (precision, precision, precision)
    blocks = [component_closure(pts) for pts in components]
    with open(path, "w") as fh:
        if comments:
            # The target is flattened first: it can be an arbitrary KnotPlot
            # command, and whitespace in it must not break the comment line.
            fh.write("# %s v%s -- %d component(s) from KnotPlot `%s`\n"
                     % (TOOL_NAME, __version__, len(blocks), _flat(name)))
            fh.write("# columns: x y z ; blank lines separate components ; "
                     "a closed component does not repeat its first vertex\n")
            if mirrored:
                fh.write("# MIRROR IMAGE: reflected through the %s = 0 plane "
                         "(%s -> -%s), so writhe and\n"
                         "# linking number are negated relative to KnotPlot's "
                         "own conformation\n"
                         % (mirror_axis, mirror_axis, mirror_axis))
        for index, (pts, closed) in enumerate(blocks):
            if index:
                fh.write("\n")
            if comments:
                fh.write("# component %s: closed=%s points=%d\n"
                         % (component_label(index),
                            "yes" if closed else "no", len(pts)))
            for x, y, z in pts:
                fh.write(fmt % (x, y, z))
    return path


def safe_stem(name):
    """Filename stem that still reads like the KnotPlot name (`6.3.2`, `3_1`)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_") or "knot"


def unique_stem(stem, used):
    """Never let two different catalogue names overwrite one output file."""
    candidate, suffix = stem, 2
    while candidate in used:
        candidate = "%s-%d" % (stem, suffix)
        suffix += 1
    used.add(candidate)
    return candidate


def extract(targets, outdir, install=None, nbeads=None, split=False,
            precision=DEFAULT_PRECISION, timeout=DEFAULT_TIMEOUT,
            comments=True, mirror=False, mirror_axis=DEFAULT_MIRROR_AXIS,
            both=False, explicit_label=CLI_LABEL):
    """Extract every target into `outdir` and report what happened.

    Returns {"written": [paths], "failed": [targets], "log": knotplot output,
    "files": [{"path", "target", "components", "mirrored", "mirror_axis"}]},
    where "components" is the per-component report the guidance block is
    built from and "mirrored" says whether that file holds the reflection.
    Everything is done in one KnotPlot session, which is why the whole
    catalogue costs about as much as a single knot.

    `mirror` writes the reflection instead of the original and `both` writes
    the pair, but the naming does not depend on which: the reflected file is
    <stem>_mirror.xyz either way, so a mirror-only run writes
    <stem>_mirror.xyz and no <stem>.xyz at all.  The suffix is what makes a
    file's handedness readable from its name, so it is never dropped just
    because the unreflected form was not also written.  `both` is never
    implied by anything, because with the whole catalogue it turns 522 files
    into 1044.  A multi-component target is reflected in ONE reflection
    covering every component, so a link's components stay in register and
    its linking number flips consistently with its writhe.

    `explicit_label` says which surface will read the complaints raised
    here, exactly as it does in installation_message: the GUI passes
    LOCATED_LABEL and its settings are then named the way its own fields
    are, rather than after flags it does not have.
    """
    term = _terms(explicit_label)
    targets = [clean_target(t) for t in targets if str(t).strip()]
    if not targets:
        raise ValueError("give at least one target")
    if install is None:
        install = find_knotplot()
    if not install:
        raise ValueError(installation_message(
            None, explicit_label=explicit_label))
    if nbeads is not None and int(nbeads) < 3:
        raise ValueError("%s needs at least 3 points" % term["nbeads"])
    precision = int(precision)
    if not 0 <= precision <= 17:
        raise ValueError("%s must be between 0 and 17" % term["precision"])
    if mirror_axis not in MIRROR_AXES:
        raise ValueError("the mirror axis must be one of %s, not %r"
                         % (", ".join(MIRROR_AXES), mirror_axis))
    # `both` asks for the pair; `mirror` on its own replaces the original.
    variants = [False, True] if both else ([True] if mirror else [False])

    outdir = str(outdir)
    # os.makedirs on an existing file raises a bare "[Errno 17] File exists",
    # which reads as though the folder were already there and fine.
    if os.path.exists(outdir) and not os.path.isdir(outdir):
        raise ValueError(
            "the output folder %s is an existing file, not a folder;\n"
            "%s, or move that file out of the way"
            % (os.path.abspath(outdir), term["outdir_fix"]))
    os.makedirs(outdir, exist_ok=True)
    written = []
    failed = []
    files = []
    used_stems = set()

    with tempfile.TemporaryDirectory() as workdir:
        script = build_script(targets, nbeads, workdir)
        log = run_knotplot(install, script, workdir, timeout)

        for index, target in enumerate(targets):
            raw = os.path.join(workdir, "raw%04d.txt" % index)
            if not os.path.exists(raw):
                failed.append(target)
                continue
            components = parse_coords(raw)
            if not components:
                failed.append(target)
                continue
            stem = unique_stem(safe_stem(target), used_stems)
            for reflected in variants:
                # One reflection for the whole target, taken before any
                # splitting, so every component of a link is reflected the
                # same way and they stay in register.
                pieces = (mirror_points(components, mirror_axis) if reflected
                          else components)
                # The mirror's stem is claimed too, so a later target
                # literally named <stem>_mirror cannot overwrite it.
                v_stem = (unique_stem(stem + "_mirror", used_stems)
                          if reflected else stem)
                axis = mirror_axis if reflected else ""
                if split and len(pieces) > 1:
                    for c_index, pts in enumerate(pieces):
                        out = os.path.join(
                            outdir,
                            "%s_%s.xyz" % (v_stem, component_label(c_index)))
                        write_xyz(out, [pts], target, precision, comments,
                                  reflected, mirror_axis)
                        written.append(out)
                        files.append({"path": out, "target": target,
                                      "components": component_report([pts]),
                                      "mirrored": reflected,
                                      "mirror_axis": axis})
                else:
                    out = os.path.join(outdir, "%s.xyz" % v_stem)
                    write_xyz(out, pieces, target, precision, comments,
                              reflected, mirror_axis)
                    written.append(out)
                    files.append({"path": out, "target": target,
                                  "components": component_report(pieces),
                                  "mirrored": reflected,
                                  "mirror_axis": axis})

    # Diagnostics only when there is something to diagnose, so a clean run
    # leaves nothing but curve files behind.
    if failed:
        with open(os.path.join(outdir, "_failed.txt"), "w") as fh:
            fh.write("\n".join(failed) + "\n")
        with open(os.path.join(outdir, "_knotplot.log"), "w") as fh:
            fh.write(log)

    return {"written": written, "failed": failed, "log": log, "files": files}


def component_guidance(files):
    """The "USING THIS FILE IN CURVE IT" block, or [] when nothing needs it.

    Curve It concatenates every component of a curve file by default, so a
    link written as one file becomes one long path unless a component is
    selected.  Roughly 200 of the 522 catalogue entries are multi-component
    links, so this is the normal case rather than the exotic one.
    """
    multi = [entry for entry in files if len(entry["components"]) > 1]
    if not multi:
        return []
    entry = multi[0]
    parts = entry["components"]
    name = os.path.basename(entry["path"])
    lines = ["",
             "USING THIS FILE IN CURVE IT",
             "  Curve It concatenates every component by default, which would",
             "  join the %d curves in %s into one path. Select one at a time:"
             % (len(parts), name)]
    for part in parts[:3]:
        lines.append("    python3 curve_it.py in.pdb %s --curve-components %s "
                     "--path-type %s"
                     % (name, part["label"],
                        "closed" if part["closed"] else "open"))
    if len(parts) > 3:
        lines.append("    ... and so on through component %s"
                     % parts[-1]["label"])
    lines.append("  In the GUI, use Select components... beside the curve file.")
    if len({part["closed"] for part in parts}) > 1:
        lines.append("  Path type is one setting for the whole run, and these")
        lines.append("  components are not all closed, so convert them separately.")
    if len(multi) > 1:
        lines.append("  %d other file(s) from this run also hold several "
                     "components." % (len(multi) - 1))
    return lines


def preview(paths, closed=True):
    """Open Curve It's own curve viewer on the files just written.

    view_xyzV3 ends in a blocking plt.show(), so it is run as a separate
    process: the GUI stays usable and several previews can be open at once.
    Returns the paths actually opened.
    """
    viewer = resource_path(os.path.join("curve_it_lib", "view_xyzV3.py"))
    if not os.path.isfile(viewer):
        raise ValueError("the curve viewer is missing: %s" % viewer)
    opened = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        command = [sys.executable, viewer, path]
        if not closed:
            command.append("--open")
        try:
            subprocess.Popen(command)
        except OSError as exc:
            raise ValueError("could not start the curve viewer: %s" % exc)
        opened.append(path)
    if not opened:
        raise ValueError("there is nothing to preview yet - extract first")
    return opened


PREVIEW_LIMIT = 6


def preview_selection(written):
    """Which files a preview should open, and what to say when it trims.

    A whole-catalogue run writes hundreds of files; opening a window for each
    would be unusable, so only the first few are shown.
    """
    chosen = list(written)[:PREVIEW_LIMIT]
    note = ""
    if len(written) > len(chosen):
        note = ("previewing the first %d of %d files; the rest are on disk"
                % (len(chosen), len(written)))
    return chosen, note


def torus_note(targets):
    """Warn about `torus p q`'s index order, which no convention agrees on."""
    used = [t for t in targets if str(t).strip().lower().startswith("torus")]
    if not used:
        return []
    return ["",
            "TORUS INDEX ORDER",
            "  KnotPlot's torus p q winds p times about the axis and q times",
            "  about the tube. Some texts name that same curve (q,p), so a",
            "  (p,q) from a paper may be this tool's torus q p. The knot type",
            "  is the same either way; the shape is not. See the torus help."]


def mirror_flags(result):
    """path -> the axis it was reflected on, for every reflected file.

    Both reports mark their reflected files: a folder holding 8.19.xyz next
    to 8.19_mirror.xyz should say which is which without either header being
    opened.
    """
    return {entry["path"]: entry.get("mirror_axis") or DEFAULT_MIRROR_AXIS
            for entry in result.get("files", []) if entry.get("mirrored")}


def _mirror_tag(axis):
    """What a reflected file's line carries in the list of what was written."""
    if not axis:
        return ""
    return "   <- mirror image, %s -> -%s" % (axis, axis)


def describe(result, targets, outdir, guidance=True):
    """The report both the CLI and the GUI print after a run.

    `guidance` is off only for the GUI, which prints the same block itself
    further down, after the list of files it refers to.
    """
    written = result["written"]
    failed = result["failed"]
    lines = ["%d target(s), %d file(s) written to %s"
             % (len(targets), len(written), os.path.abspath(outdir))]
    reflected = [entry for entry in result.get("files", [])
                 if entry.get("mirrored")]
    if reflected:
        axis = reflected[0].get("mirror_axis") or DEFAULT_MIRROR_AXIS
        # A single-target --mirror run is the common case, and "1 of them
        # are MIRROR IMAGES" is what it used to print.
        count = len(reflected)
        lines.append("%s: named _mirror, reflected %s -> -%s, writhe and "
                     "linking number negated"
                     % ("1 of them is a MIRROR IMAGE" if count == 1
                        else "%d of them are MIRROR IMAGES" % count,
                        axis, axis))
    if failed:
        # Pair each failure with what KnotPlot said about it, by position in
        # the target list, so "nosuchknot" reads as a typo and not a mystery.
        reasons = knotplot_errors(result.get("log", ""))
        outstanding = set(failed)
        detail = []
        for index, target in enumerate(targets):
            if target in outstanding and len(detail) < 12:
                detail.append("  %s  --  %s"
                              % (target, reasons.get(
                                  index, "KnotPlot produced no coordinates")))
                outstanding.discard(target)
        lines.append("no coordinates for %d target(s):" % len(failed))
        lines += detail
        if len(failed) > len(detail):
            lines.append("  ... and %d more, listed in _failed.txt"
                         % (len(failed) - len(detail)))
        lines.append("KnotPlot's own output is in _knotplot.log")
    if guidance:
        lines += component_guidance(result.get("files", []))
    lines += torus_note(targets)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="kp2xyz.py",
        description="Extract knots and links from KnotPlot's shipped "
                    "catalogue as plain XYZ curves.",
        epilog="Run with no arguments for the GUI. The exit status is 0 only "
               "when every target produced a file; a run in which any target "
               "failed exits 1, so a scripted --all can tell a partial run "
               "from a complete one.")
    p.add_argument("targets", nargs="*",
                   help="catalogue names (4.1, 6.3.2) or quoted KnotPlot "
                        "commands ('torus 2 3')")
    p.add_argument("--gui", action="store_true",
                   help="force the graphical interface")

    p.add_argument("--knotplot", metavar="PATH",
                   help="the KnotPlot binary, or the folder or .app holding "
                        "it; overrides $KNOTPLOT and the search")
    p.add_argument("--check", action="store_true",
                   help="report where KnotPlot was found, launch it to "
                        "confirm it really is KnotPlot, and exit")
    p.add_argument("--list", action="store_true", dest="list_names",
                   help="print every catalogue name and exit")

    p.add_argument("--all", action="store_true",
                   help="extract every knot and link in KnotPlot's basic "
                        "catalogue")
    p.add_argument("--no-dedupe", action="store_true",
                   help="with --all or --list, keep every catalogue file "
                        "including the byte-identical `3_1`-style copies; "
                        "the 30 dotted/underscored pairs that differ - 28 of "
                        "them mirror images of each other - are kept either "
                        "way")

    p.add_argument("-o", "--outdir", default=DEFAULT_OUTDIR,
                   help="output directory (default %s)" % DEFAULT_OUTDIR)
    p.add_argument("-n", "--nbeads", type=int,
                   help="resample each curve to this many points before "
                        "dumping; on a link it is the total over components "
                        "(default: the shipped resolution)")
    p.add_argument("--split", action="store_true",
                   help="write one .xyz per component instead of one per link")
    p.add_argument("--mirror", action="store_true",
                   help="write the mirror image instead of the original: "
                        "exactly one coordinate is negated, which negates "
                        "writhe and linking number")
    p.add_argument("--mirror-axis", choices=MIRROR_AXES,
                   default=DEFAULT_MIRROR_AXIS, metavar="{x,y,z}",
                   help="the coordinate to negate (default %s, which leaves "
                        "the x-y projection alone and swaps every crossing)"
                        % DEFAULT_MIRROR_AXIS)
    p.add_argument("--both", action="store_true",
                   help="write the original AND the mirror, the reflected "
                        "one as <name>_mirror.xyz; never implied, since with "
                        "--all it writes 1044 files")
    p.add_argument("--preview", action="store_true",
                   help="open each written curve in Curve It's own 3D viewer "
                        "after extracting (at most %d windows)" % PREVIEW_LIMIT)
    p.add_argument("--precision", type=int, default=DEFAULT_PRECISION,
                   help="decimal places written per coordinate (default 6)")
    p.add_argument("--no-comments", action="store_true",
                   help="omit the '#' header naming the target and each "
                        "component")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="seconds to wait for KnotPlot (default %g)"
                        % DEFAULT_TIMEOUT)
    p.add_argument("--info", action="store_true",
                   help="list what would be extracted and exit without "
                        "writing")
    return p


def resolve_targets(args, install):
    """The full target list for this run, catalogue expansion included."""
    targets = list(args.targets)
    if args.all:
        targets += catalogue_names(install, dedupe=not args.no_dedupe)
    return targets


def run_cli(args):
    # --knotplot pins the run to one binary, so a bad one is fatal instead of
    # a warning: falling through to whatever the search finds next would
    # quietly extract with an install other than the one that was pinned.
    # $KNOTPLOT is a preference rather than a pin, and keeps its
    # warn-and-fall-back behaviour below.
    if args.knotplot and not any(_is_executable(p)
                                 for p in _expand_candidate(args.knotplot)):
        raise ValueError(
            "--knotplot %s holds no usable KnotPlot binary.\n"
            "--knotplot pins this run to that one binary, so the search is "
            "not continued\npast it and no other install is used instead.\n"
            "Give the KnotPlot binary itself, or the folder or .app holding "
            "it, or drop\n--knotplot to let the usual search run.\n"
            "Download KnotPlot from %s" % (args.knotplot, DOWNLOAD_URL))

    install = find_knotplot(args.knotplot)

    # The search falls through past a bad --knotplot or a typo'd $KNOTPLOT,
    # so an install found somewhere else must not look as though the
    # override had been honoured.  --check says the same thing in its own
    # message body, so it is not repeated here.
    if not args.check:
        for problem in override_problems(args.knotplot):
            print("warning: %s; ignored" % problem, file=sys.stderr)

    if args.check:
        probe = probe_knotplot(install) if install else None
        print(installation_message(install, args.knotplot, probe))
        return 0 if (install and probe and probe[0]) else 1
    if not install:
        raise ValueError(installation_message(None))

    if args.list_names:
        names = catalogue_names(install, dedupe=not args.no_dedupe)
        if not args.no_dedupe:
            # Kept off stdout so `--list | xargs` still works.
            print(CHIRALITY_ONE_LINER, file=sys.stderr)
        for name in names:
            print(name)
        return 0

    if args.nbeads is not None and args.nbeads < 3:
        raise ValueError("--nbeads needs at least 3 points")

    targets = resolve_targets(args, install)
    if not targets:
        raise ValueError("give at least one target, or --all")

    print("KnotPlot: %s  (%s)" % (install.executable, install.source))
    print("targets:  %d" % len(targets))
    if args.info:
        for target in targets[:40]:
            print("  %s" % target)
        if len(targets) > 40:
            print("  ... and %d more" % (len(targets) - 40))
        return 0

    result = extract(targets, args.outdir, install=install, nbeads=args.nbeads,
                     split=args.split, precision=args.precision,
                     timeout=args.timeout, comments=not args.no_comments,
                     mirror=args.mirror, mirror_axis=args.mirror_axis,
                     both=args.both)
    flags = mirror_flags(result)
    print("\nWROTE")
    for item in result["written"]:
        print("  %s%s" % (item, _mirror_tag(flags.get(item))))
    print("\n" + describe(result, targets, args.outdir))
    if not result["written"]:
        # Nothing came back at all.  Before that is read as every target
        # being wrong, check the binary: something executable that is not
        # KnotPlot fails in exactly this shape, and the launch costs about
        # ten milliseconds, so it is only paid on a total failure.
        ok, detail = probe_knotplot(install)
        if not ok:
            print("\n%s did not answer as KnotPlot: %s\n"
                  "Being executable does not make a file KnotPlot, which is "
                  "why nothing could be\nextracted. Run --check for the whole "
                  "picture." % (install.executable, detail), file=sys.stderr)
    if args.preview and result["written"]:
        chosen, note = preview_selection(result["written"])
        if note:
            print(note)
        try:
            preview(chosen)
        except ValueError as exc:
            # A failed preview must not fail the extraction that succeeded.
            print("preview unavailable: %s" % exc, file=sys.stderr)
    # A scripted --all has to be able to tell a partial run from a whole one.
    return 0 if result["written"] and not result["failed"] else 1


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def default_gui_outdir():
    """An absolute folder to prefill the GUI's output field with.

    The field used to open empty, and an empty field meant "here", which for
    an app launched from the Finder is whatever working directory Curve It
    happened to inherit - not a folder the user could find again.
    """
    home = os.path.expanduser("~")
    documents = os.path.join(home, "Documents")
    root = documents if os.path.isdir(documents) else home
    return os.path.join(root, DEFAULT_OUTDIR)


def run_gui(initial_outdir=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("kp2xyz  -  KnotPlot knots and links  ->  XYZ curves   v%s"
               % __version__)
    root.minsize(940, 620)
    set_optional_window_icon(
        root, tk, ["kp2xyz_icon.png", "icon.png"], "_kp2xyz_icon_image")

    # "explicit" remembers the path chosen with Locate KnotPlot..., so that
    # Re-check re-runs both the search and the launch check with it instead
    # of throwing it away and finding nothing again; it is forgotten as soon
    # as it stops resolving.  "names" caches catalogue listings keyed on the
    # dedupe checkbox, because the two settings give different listings.
    # "probe" is the (ok, detail) pair from the last launch, and Extract is
    # enabled only while it says the binary answered as KnotPlot.
    state = {"install": None, "names": {}, "explicit": None, "probe": None}
    V = {
        "targets":   tk.StringVar(value=""),
        "all":       tk.BooleanVar(value=False),
        "dedupe":    tk.BooleanVar(value=True),
        "nbeads":    tk.StringVar(value=""),
        "precision": tk.StringVar(value="%d" % DEFAULT_PRECISION),
        "split":     tk.BooleanVar(value=False),
        "mirror":    tk.BooleanVar(value=False),
        "mirror_axis": tk.StringVar(value=DEFAULT_MIRROR_AXIS),
        "both":      tk.BooleanVar(value=False),
        "comments":  tk.BooleanVar(value=True),
        "outdir":    tk.StringVar(value=initial_outdir or default_gui_outdir()),
    }

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
        open_help["win"] = top
        top.title(title)
        top.transient(root)
        top.configure(bg="#f4f9ff")
        frm = tk.Frame(top, bg="#f4f9ff", padx=16, pady=14)
        frm.pack(fill="both", expand=True)
        tk.Label(frm, text=title, font=("Helvetica", 14, "bold"),
                 bg="#f4f9ff", fg="#0b4d80", anchor="w",
                 justify="left").pack(fill="x", pady=(0, 8))
        tk.Label(frm, text=prose, wraplength=470, justify="left",
                 bg="#f4f9ff", fg="#1a1a1a", font=("Helvetica", 12),
                 anchor="w").pack(fill="x")
        if example:
            tk.Label(frm, text="Example", font=("Helvetica", 11, "bold"),
                     bg="#f4f9ff", fg="#0b4d80", anchor="w").pack(fill="x",
                                                                  pady=(12, 3))
            # Height counted in DISPLAY rows, not in newlines: a line
            # longer than the box wraps, and a height counted in newlines
            # then clips the tail of the block off the bottom.  The two
            # export lines in the "install" example are 89 characters, so
            # this is the normal case rather than a corner one.
            width = 62
            rows = sum(max(1, -(-len(line) // width))
                       for line in example.split("\n"))
            box = tk.Text(frm, font=("Menlo", 11), height=rows,
                          width=width, bg="#e8f1fa", relief="flat",
                          padx=8, pady=6, highlightthickness=0)
            box.insert("1.0", example)
            box.configure(state="disabled")
            box.pack(fill="x")
        tk.Button(frm, text="Close", command=top.destroy).pack(pady=(12, 0))
        top.bind("<Escape>", lambda e: top.destroy())
        try:
            x = near.winfo_rootx() + 26
            y = near.winfo_rooty() - 10
            top.geometry("+%d+%d" % (max(x, 0), max(y, 0)))
        except Exception:                   # noqa: BLE001
            pass
        top.focus_set()

    def chip(parent, key):
        """A small light-blue '?' that opens the explanation for `key`.

        Built from a Label rather than a Button because macOS's native button
        ignores background colour.
        """
        c = tk.Label(parent, text="?", font=("Helvetica", 11, "bold"),
                     bg="#bcdcf5", fg="#0b4d80", width=2, relief="raised",
                     borderwidth=1, cursor="hand2")
        c.bind("<Button-1>", lambda e, k=key, w=c: show_help(k, w))
        c.bind("<Enter>", lambda e, w=c: w.configure(bg="#8fc7ee"))
        c.bind("<Leave>", lambda e, w=c: w.configure(bg="#bcdcf5"))
        return c

    # ---- installation banner ---------------------------------------------
    # This sits above everything else because nothing else in the window can
    # do anything until KnotPlot has been found.
    head = ttk.Frame(root, padding=(10, 10, 10, 0))
    head.pack(fill="x")
    head.columnconfigure(0, weight=1)
    inst_lbl = tk.Label(head, text="looking for KnotPlot ...", justify="left",
                        anchor="w", font=("Helvetica", 11), fg="#888",
                        wraplength=700)
    inst_lbl.grid(row=0, column=0, sticky="ew")
    locate_btn = ttk.Button(head, text="Locate KnotPlot...")
    locate_btn.grid(row=0, column=1, padx=(8, 0))
    chip(head, "install").grid(row=0, column=2, padx=(6, 0))

    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(0, weight=1)
    left = ttk.Frame(main)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    right = ttk.Frame(main)
    right.grid(row=0, column=1, sticky="nsew")
    right.rowconfigure(1, weight=1)
    right.columnconfigure(0, weight=1)

    # targets ---------------------------------------------------------------
    ft = ttk.LabelFrame(left, text="What to extract", padding=8)
    ft.pack(fill="x", pady=(0, 8))
    ft.columnconfigure(0, weight=1)
    targets_entry = ttk.Entry(ft, textvariable=V["targets"], width=34)
    targets_entry.grid(row=0, column=0, columnspan=2, sticky="ew")
    chip(ft, "targets").grid(row=0, column=2, padx=(6, 0))
    tk.Label(ft, text="names and quoted commands, space separated",
             font=("Helvetica", 10), fg="#8a929b").grid(
                 row=1, column=0, columnspan=3, sticky="w", pady=(3, 0))
    browse_btn = ttk.Button(ft, text="Browse catalogue...")
    browse_btn.grid(row=2, column=0, sticky="w", pady=(6, 0))
    chip(ft, "catalogue").grid(row=2, column=2, padx=(6, 0), pady=(6, 0))
    all_chk = ttk.Checkbutton(ft, text="Entire catalogue", variable=V["all"])
    all_chk.grid(row=3, column=0, sticky="w", pady=(6, 0))
    chip(ft, "all").grid(row=3, column=2, padx=(6, 0), pady=(6, 0))
    dedupe_chk = ttk.Checkbutton(ft, text="drop byte-identical duplicate files",
                                 variable=V["dedupe"])
    dedupe_chk.grid(row=4, column=0, columnspan=2, sticky="w")
    tk.Label(ft, text="mirror-image pairs such as 8.19 / 8_19 are always kept",
             font=("Helvetica", 10), fg="#8a929b").grid(
                 row=5, column=0, columnspan=3, sticky="w")
    tk.Label(ft, text="torus p q: p turns about the axis, q about the tube",
             font=("Helvetica", 10), fg="#8a929b").grid(
                 row=6, column=0, columnspan=2, sticky="w", pady=(3, 0))
    chip(ft, "torus").grid(row=6, column=2, padx=(6, 0), pady=(3, 0))

    # extraction ------------------------------------------------------------
    fe = ttk.LabelFrame(left, text="Extraction", padding=8)
    fe.pack(fill="x", pady=(0, 8))
    extraction_rows = [("points per curve", "nbeads", "blank = as shipped"),
                       ("precision", "precision", "decimals per coordinate")]
    for r, (label, key, hint) in enumerate(extraction_rows):
        ttk.Label(fe, text=label).grid(row=r, column=0, sticky="w")
        ttk.Entry(fe, textvariable=V[key], width=10).grid(row=r, column=1,
                                                          sticky="w", padx=(4, 0))
        chip(fe, key).grid(row=r, column=2, padx=(6, 0))
        tk.Label(fe, text=hint, font=("Helvetica", 10), fg="#8a929b").grid(
            row=r, column=3, sticky="w", padx=(8, 0))
    ttk.Checkbutton(fe, text="one file per component",
                    variable=V["split"]).grid(row=2, column=0, columnspan=2,
                                              sticky="w", pady=(4, 0))
    chip(fe, "split").grid(row=2, column=2, padx=(6, 0), pady=(4, 0))
    ttk.Checkbutton(fe, text="mirror image", variable=V["mirror"]).grid(
        row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
    chip(fe, "mirror").grid(row=3, column=2, padx=(6, 0), pady=(4, 0))
    mirror_row = ttk.Frame(fe)
    mirror_row.grid(row=4, column=0, columnspan=4, sticky="w")
    ttk.Label(mirror_row, text="axis").pack(side="left")
    ttk.Combobox(mirror_row, textvariable=V["mirror_axis"],
                 values=list(MIRROR_AXES), width=3, state="readonly"
                 ).pack(side="left", padx=(4, 12))
    ttk.Checkbutton(mirror_row, text="write both, original and mirror",
                    variable=V["both"]).pack(side="left")
    tk.Label(fe, text="negates writhe and linking number; z keeps the "
                      "projection you see",
             font=("Helvetica", 10), fg="#8a929b").grid(
                 row=5, column=0, columnspan=4, sticky="w")
    tk.Label(fe, text="shapes are the ones KnotPlot ships, exported unchanged",
             font=("Helvetica", 10), fg="#8a929b").grid(
                 row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))
    chip(fe, "conformations").grid(row=6, column=2, padx=(6, 0), pady=(6, 0))

    # output ----------------------------------------------------------------
    fo = ttk.LabelFrame(left, text="Output", padding=8)
    fo.pack(fill="x")
    fo.columnconfigure(1, weight=1)
    ttk.Label(fo, text="folder").grid(row=0, column=0, sticky="w")
    ttk.Entry(fo, textvariable=V["outdir"], width=24).grid(row=0, column=1,
                                                           sticky="ew")
    ttk.Button(fo, text="...", width=3,
               command=lambda: V["outdir"].set(filedialog.askdirectory()
                                               or V["outdir"].get())
               ).grid(row=0, column=2, padx=(4, 0))
    chip(fo, "output").grid(row=0, column=3, padx=(6, 0))
    ttk.Checkbutton(fo, text="comment header",
                    variable=V["comments"]).grid(row=1, column=0, columnspan=2,
                                                 sticky="w", pady=(4, 0))

    # right pane ------------------------------------------------------------
    bar = ttk.Frame(right)
    bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    recheck_btn = ttk.Button(bar, text="Re-check")
    recheck_btn.pack(side="left")
    list_btn = ttk.Button(bar, text="List catalogue")
    list_btn.pack(side="left", padx=6)
    preview_btn = ttk.Button(bar, text="Preview")
    preview_btn.pack(side="left", padx=(0, 6))
    extract_btn = ttk.Button(bar, text="Extract")
    extract_btn.pack(side="left")
    txt = tk.Text(right, wrap="none", font=("Menlo", 11), height=26)
    txt.grid(row=1, column=0, sticky="nsew")
    sb = ttk.Scrollbar(right, orient="vertical", command=txt.yview)
    sb.grid(row=1, column=1, sticky="ns")
    txt.configure(yscrollcommand=sb.set)
    status = ttk.Label(right, text="", foreground="#0a7")
    status.grid(row=2, column=0, sticky="w", pady=(4, 0))

    def show(s):
        txt.delete("1.0", "end")
        txt.insert("1.0", s)

    def append(s):
        txt.insert("end", s)
        txt.see("end")

    # A complaint about a field names the field as this window labels it,
    # not as the command line spells it: GUI_TERMS is the same table
    # extract() is handed below, so the two never diverge.
    def field(key):
        return GUI_TERMS.get(key, key)

    def fnum(key, default=None):
        text = V[key].get().strip()
        if text.lower() in ("", "auto", "none") and default is not None:
            return default
        try:
            return float(text)
        except ValueError:
            raise ValueError("%s: %r is not a number" % (field(key), text))

    def optional(key):
        text = V[key].get().strip()
        if text.lower() in ("", "auto", "none"):
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError("%s: %r is not a number" % (field(key), text))

    # ---- installation state ----------------------------------------------
    def refresh_install(announce=True):
        """Re-run the search, launch what it finds, and reflect the answer.

        Two separate things have to be true before anything can be
        extracted, so both are settled here rather than only on the command
        line.  First the binary has to be found: the remembered Locate
        choice is fed back in every time, and is forgotten the moment it
        stops resolving, so a path that has since been moved or deleted is
        dropped instead of being named for the rest of the session as though
        it were in force.

        Second, what was found has to BE KnotPlot.  Being executable is no
        evidence of that - /bin/ls is executable - so the binary is
        launched, asked to quit, and its own startup banner is what decides.
        A file that runs but does not answer as KnotPlot is reported as
        exactly that, and leaves Extract disabled.  The launch costs about
        ten milliseconds, so it can run on every Re-check.

        The window always opens, whatever the answer; a missing KnotPlot
        disables the buttons that need it and says why, rather than
        refusing to start.

        The banner, the status label and Extract are all driven from the
        probe result here, and unconditionally, so that the three of them
        cannot disagree.  `announce` only governs the two things a caller
        might want to say differently: the interim "checking ..." wording,
        and whether the report pane is overwritten with `message` -- the
        startup call writes its own longer welcome text there instead, and
        must build it from state["probe"] for the same reason.
        """
        explicit = state["explicit"]
        install = find_knotplot(explicit)
        dropped = ""
        if explicit and (not install or install.source != "explicit"):
            # The located path holds no binary any more.  Forget it and
            # search again without it, rather than go on reporting it.
            dropped = explicit
            state["explicit"] = explicit = None
            install = find_knotplot(None)
        state["install"] = install
        state["names"] = {}
        state["probe"] = None
        if install:
            if announce:
                inst_lbl.configure(
                    text="found %s - launching it to check that it really is "
                         "KnotPlot ..." % install.executable, fg="#888")
                status.configure(text="checking KnotPlot ...",
                                 foreground="#888")
                root.update_idletasks()
            state["probe"] = probe_knotplot(install)
        probe = state["probe"]
        answered = bool(probe and probe[0])
        message = installation_message(install, explicit, probe,
                                       explicit_label=LOCATED_LABEL)
        if dropped:
            message = ("The KnotPlot picked earlier with Locate KnotPlot... "
                       "no longer holds a usable\nbinary, so it has been "
                       "forgotten:\n  %s\n\n%s" % (dropped, message))
        # The listing reads the catalogue folder directly, so it depends on
        # the folder being there and not on the binary answering.
        catalogue_state = ("normal" if install and install.has_catalogue
                           else "disabled")
        list_btn.configure(state=catalogue_state)
        browse_btn.configure(state=catalogue_state)
        all_chk.configure(state=catalogue_state)
        dedupe_chk.configure(state=catalogue_state)
        if catalogue_state == "disabled":
            # A ticked box that cannot be unticked because it is greyed out
            # still reads as ticked, and Extract then raised on it.
            V["all"].set(False)
        if answered:
            inst_lbl.configure(
                text="KnotPlot found at %s  (%s) - it runs, and answers as "
                     "KnotPlot" % (install.executable, install.source),
                fg="#0a7")
            extract_btn.configure(state="normal")
            preview_btn.configure(state="normal")
            status.configure(text="KnotPlot ready", foreground="#0a7")
        elif install:
            # The full reason is in the report pane below; the banner keeps
            # to one wrapped line or two so it cannot squeeze the window.
            detail = probe[1] if probe else "it could not be launched"
            if len(detail) > 96:
                detail = detail[:93] + "..."
            inst_lbl.configure(
                text=("%s was found, but it DID NOT RESPOND AS KNOTPLOT: %s. "
                      "Nothing can be extracted with it - use Locate "
                      "KnotPlot... to point at the real binary."
                      % (install.executable, detail)),
                fg="#c00")
            extract_btn.configure(state="disabled")
            preview_btn.configure(state="disabled")
            status.configure(text="found, but it is not KnotPlot",
                             foreground="#c00")
        else:
            inst_lbl.configure(
                text=("KnotPlot was not found - nothing can be extracted "
                      "until it is. Use Locate KnotPlot... or set $KNOTPLOT. "
                      "Download: %s" % DOWNLOAD_URL),
                fg="#c00")
            extract_btn.configure(state="disabled")
            preview_btn.configure(state="disabled")
            status.configure(text="KnotPlot not found", foreground="#c00")
        if announce:
            show(message + "\n")
        return install

    def do_locate():
        # A macOS chooser hands back KnotPlot.app itself, and people pick the
        # install folder as often as the binary; _expand_candidate sorts it out.
        path = filedialog.askopenfilename(
            title="Locate the KnotPlot application or binary",
            filetypes=[("KnotPlot application", "*.app"),
                       ("Executables", "*.exe"),
                       ("All files", "*.*")])
        if not path:
            path = filedialog.askdirectory(title="...or its install folder")
        if not path:
            return
        previous = state["explicit"]
        state["explicit"] = path
        install = refresh_install()
        # The search falls through to the other locations, so "found" does
        # not by itself mean the chosen path was the thing that was found.
        if not install or install.source != "explicit":
            # Do not keep a path that leads nowhere: Re-check would then go
            # on reporting it forever.  Announce the revert, so the report
            # pane describes the install actually in force.
            state["explicit"] = previous
            install = refresh_install()
            detail = ["No usable KnotPlot binary under:", "  %s" % path, "",
                      "Pick the binary itself (KnotPlot.app on macOS, "
                      "knotplot.exe on Windows)",
                      "or the folder holding it. $KNOTPLOT and $KNOTPLOT_HOME "
                      "do the same thing",
                      "from a shell.", "",
                      "Download KnotPlot from %s" % DOWNLOAD_URL, "",
                      "Looked in:"]
            detail += ["  %s" % item for item in searched_locations()]
            if install:
                detail += ["", "Still using %s  (%s)"
                           % (install.executable, install.source)]
            messagebox.showerror("KnotPlot not found there", "\n".join(detail))
            return
        probe = state["probe"]
        if not (probe and probe[0]):
            # There is an executable file there and it runs, but it is not
            # KnotPlot.  The choice is kept rather than silently swapped for
            # some other install - that silent swap is exactly what this
            # check exists to prevent - and Extract stays disabled until a
            # real KnotPlot is picked.
            messagebox.showerror(
                "That is not KnotPlot",
                "\n".join([
                    "This runs, but it does not answer as KnotPlot:",
                    "  %s" % path, "",
                    "  %s" % (probe[1] if probe else
                             "it could not be launched"), "",
                    "Being executable does not make a file KnotPlot, so "
                    "Extract stays disabled",
                    "until a real one is found. Use Locate KnotPlot... "
                    "again and pick the",
                    "application or binary itself - KnotPlot.app on macOS, "
                    "knotplot.exe on",
                    "Windows - or the folder holding it.", "",
                    "Download KnotPlot from %s" % DOWNLOAD_URL]))

    locate_btn.configure(command=do_locate)
    recheck_btn.configure(command=lambda: refresh_install())

    # ---- catalogue --------------------------------------------------------
    def load_names():
        """The catalogue listing for the dedupe setting in force right now.

        Keyed on the checkbox, because the two settings give genuinely
        different listings - 522 entries against 585 - and one unkeyed cache
        handed the first listing out for the rest of the session, so
        unticking the box looked as though it did nothing.
        """
        key = bool(V["dedupe"].get())
        if key not in state["names"]:
            state["names"][key] = catalogue_names(state["install"],
                                                  dedupe=key)
        return state["names"][key]

    def do_list():
        try:
            names = load_names()
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Catalogue unavailable", str(exc))
            status.configure(text="no catalogue", foreground="#c00")
            return
        show("%d catalogue entries in %s\n\n"
             % (len(names), state["install"].catalogue))
        if V["dedupe"].get():
            append(CHIRALITY_ONE_LINER + "\n\n")
        # Eight to a line keeps a 500-entry catalogue readable in the pane.
        for start in range(0, len(names), 8):
            append("  " + "  ".join("%-9s" % n for n in names[start:start + 8])
                   + "\n")
        status.configure(text="listed %d entries" % len(names), foreground="#0a7")

    def do_browse():
        try:
            names = load_names()
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Catalogue unavailable", str(exc))
            return
        top = tk.Toplevel(root)
        top.title("KnotPlot catalogue")
        top.transient(root)
        top.minsize(320, 460)
        frm = ttk.Frame(top, padding=8)
        frm.pack(fill="both", expand=True)
        frm.rowconfigure(2, weight=1)
        frm.columnconfigure(0, weight=1)
        tk.Label(frm, justify="left", anchor="w", font=("Helvetica", 10),
                 fg="#8a4b00", bg="#fff2e0", padx=6, pady=3, wraplength=300,
                 text=("8.19 and 8_19 are mirror images, not duplicates - "
                       "28 of the 30 differing pairs are. Pick the "
                       "handedness you mean. 9.20 / 9_20 and 9.35 / 9_35 "
                       "are the two exceptions: same handedness, but two "
                       "slightly different conformations, not one file "
                       "under two names."
                       )).grid(row=0, column=0, columnspan=3,
                                      sticky="ew", pady=(0, 6))
        query = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=query).grid(row=1, column=0, sticky="ew")
        tk.Label(frm, text="filter", font=("Helvetica", 10),
                 fg="#8a929b").grid(row=1, column=1, padx=(6, 0))
        box = tk.Listbox(frm, selectmode="extended", font=("Menlo", 11),
                         height=22, activestyle="none")
        box.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        bsb = ttk.Scrollbar(frm, orient="vertical", command=box.yview)
        bsb.grid(row=2, column=2, sticky="ns", pady=(6, 0))
        box.configure(yscrollcommand=bsb.set)

        def repopulate(*_args):
            needle = query.get().strip().lower()
            box.delete(0, "end")
            for name in names:
                if not needle or needle in name.lower():
                    box.insert("end", name)

        query.trace_add("write", repopulate)
        repopulate()

        def add_selected():
            picked = [box.get(i) for i in box.curselection()]
            if not picked:
                return
            current = V["targets"].get().strip()
            V["targets"].set((current + " " + " ".join(picked)).strip())
            top.destroy()

        row = ttk.Frame(frm)
        row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(row, text="Add selected", command=add_selected).pack(side="left")
        ttk.Button(row, text="Cancel", command=top.destroy).pack(side="left",
                                                                 padx=6)
        box.bind("<Double-Button-1>", lambda e: add_selected())
        top.bind("<Escape>", lambda e: top.destroy())
        top.focus_set()

    list_btn.configure(command=do_list)
    browse_btn.configure(command=do_browse)

    # ---- extraction -------------------------------------------------------
    def gui_targets():
        """The target list, parsed the same way the command line parses it."""
        text = V["targets"].get().strip()
        try:
            targets = shlex.split(text) if text else []
        except ValueError as exc:
            raise ValueError("targets: %s (check the quotes)" % exc)
        if V["all"].get():
            targets += catalogue_names(state["install"], dedupe=V["dedupe"].get())
        return targets

    def do_preview():
        """Extract into a scratch folder and open the 3D viewer on it.

        Deliberately does not touch the output folder: this is for looking at
        a knot before committing to it.  The files land in a temporary
        directory that the viewer processes read immediately.
        """
        install = state["install"]
        probe = state["probe"]
        if not install or not (probe and probe[0]):
            messagebox.showerror(
                "KnotPlot not ready",
                "Preview needs a working KnotPlot.\n\n"
                "Use Locate KnotPlot... to point at the binary, or download "
                "it from\n%s" % DOWNLOAD_URL)
            return
        try:
            targets = gui_targets()
            nbeads = optional("nbeads")
            precision = int(fnum("precision", DEFAULT_PRECISION))
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Preview failed", str(exc))
            return
        if not targets:
            messagebox.showwarning(
                "Nothing to preview",
                "No targets given.\n\n"
                "Type a name such as 4.1, or pick some with Browse "
                "catalogue...")
            return
        shown, trimmed = preview_selection(targets)
        status.configure(text="extracting %d target(s) to preview..."
                              % len(shown), foreground="#8a929b")
        root.update_idletasks()
        try:
            scratch = tempfile.mkdtemp(prefix="kp2xyz-preview-")
            result = extract(shown, scratch, install=install,
                             nbeads=int(nbeads) if nbeads is not None else None,
                             split=False, precision=precision,
                             timeout=GUI_TIMEOUT,
                             comments=V["comments"].get(),
                             mirror=V["mirror"].get(),
                             mirror_axis=V["mirror_axis"].get(),
                             both=V["both"].get(),
                             explicit_label=LOCATED_LABEL)
            opened = preview(result["written"])
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Preview failed", str(exc))
            status.configure(text="preview failed", foreground="#c00")
            return
        lines = ["PREVIEW", "",
                 "%d viewer window(s) opening. These files are temporary and "
                 "were NOT" % len(opened),
                 "written to the output folder - use Extract to keep them.",
                 ""]
        lines += ["  " + os.path.basename(f) for f in opened]
        if trimmed:
            lines += ["", trimmed]
        lines += torus_note(shown)
        show("\n".join(lines) + "\n")
        status.configure(text="preview open -- nothing written to the output "
                              "folder", foreground="#0a7")

    def do_extract():
        install = state["install"]
        if not install:
            messagebox.showerror(
                "KnotPlot not found",
                installation_message(None, state["explicit"],
                                     explicit_label=LOCATED_LABEL))
            return
        probe = state["probe"]
        if not (probe and probe[0]):
            # Extract is already disabled in this state; this is the belt to
            # that braces.  A binary that does not answer as KnotPlot would
            # otherwise fail much later, on the extraction, with nothing to
            # explain it.
            messagebox.showerror(
                "That is not KnotPlot",
                "%s runs, but it did not answer as KnotPlot:\n  %s\n\n"
                "Use Locate KnotPlot... to point at the real binary, or "
                "download KnotPlot\nfrom %s"
                % (install.executable,
                   probe[1] if probe else "it could not be launched",
                   DOWNLOAD_URL))
            return
        try:
            targets = gui_targets()
            nbeads = optional("nbeads")
            precision = int(fnum("precision", DEFAULT_PRECISION))
            outdir = V["outdir"].get().strip() or default_gui_outdir()
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Extract failed", str(exc))
            status.configure(text="failed", foreground="#c00")
            return
        if not targets:
            messagebox.showwarning(
                "Nothing to extract",
                "No targets given.\n\n"
                "Type a name such as 4.1, pick some with Browse "
                "catalogue..., or tick Entire catalogue.")
            return
        try:
            # The work blocks the event loop deliberately, the way the other
            # Curve It tools do it, so the timeout doubles as the longest the
            # window can be frozen.  The entire catalogue takes about 0.26 s,
            # so
            # GUI_TIMEOUT only ever fires on a hung KnotPlot.
            axis = V["mirror_axis"].get()
            both = V["both"].get()
            mirror = V["mirror"].get()
            if both:
                note = ("  both forms: each target as shipped and reflected "
                        "%s -> -%s, so %d file(s)\n" % (axis, axis,
                                                        2 * len(targets)))
            elif mirror:
                note = ("  MIRROR IMAGES only, reflected %s -> -%s\n"
                        % (axis, axis))
            else:
                note = ""
            status.configure(
                text="running KnotPlot on %d target(s) - the window is busy "
                     "until it answers" % len(targets), foreground="#888")
            show("running KnotPlot on %d target(s) ...\n"
                 "  %s\n"
                 "  writing to %s\n"
                 "%s"
                 "  the window stays busy until KnotPlot answers, and gives "
                 "up after %gs\n"
                 % (len(targets), install.executable, os.path.abspath(outdir),
                    note, GUI_TIMEOUT))
            root.update_idletasks()
            result = extract(targets, outdir, install=install,
                             nbeads=int(nbeads) if nbeads is not None else None,
                             split=V["split"].get(), precision=precision,
                             timeout=GUI_TIMEOUT,
                             comments=V["comments"].get(),
                             mirror=mirror, mirror_axis=axis, both=both,
                             explicit_label=LOCATED_LABEL)
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("Extract failed", str(exc))
            status.configure(text="failed", foreground="#c00")
            show("extract failed:\n  %s\n" % exc)
            return

        written = result["written"]
        # The guidance block names files, so it is printed after the list of
        # them rather than inside describe()'s own summary.
        report = describe(result, targets, outdir, guidance=False)
        if written:
            shown = written[:40]
            flags = mirror_flags(result)
            report += "\n\nWROTE\n" + "\n".join(
                "  " + w + _mirror_tag(flags.get(w)) for w in shown)
            if len(written) > len(shown):
                report += "\n  ... and %d more" % (len(written) - len(shown))
            report += ("\n\nLoad %s in Curve It as the Curve XYZ/txt input.\n"
                       % os.path.basename(written[0]))

        guidance = component_guidance(result.get("files", []))
        if guidance:
            report += "\n" + "\n".join(guidance)
        note = torus_note(targets)
        if note:
            report += "\n" + "\n".join(note)
        if result["failed"]:
            report += "\n\nFAILED\n" + "\n".join("  " + f
                                                 for f in result["failed"][:40])
        show(report + "\n")
        if result["failed"] and written:
            status.configure(text="done -- %d file(s), %d target(s) failed"
                                  % (len(written), len(result["failed"])),
                             foreground="#c00")
        elif written:
            status.configure(text="done -- %d file(s)" % len(written),
                             foreground="#0a7")
        else:
            status.configure(text="nothing written", foreground="#c00")

    preview_btn.configure(command=do_preview)
    extract_btn.configure(command=do_extract)

    # refresh_install has already set the banner, the status label and
    # Extract from the probe.  What is left is the report pane, and it has
    # to be built from the same probe: `if install:` here used to say
    # "KnotPlot ready" in green, and installation_message(install) used to
    # open with "KnotPlot found at ...", for a binary that had just failed
    # to answer as KnotPlot - because KnotPlotInstall is truthy for any
    # executable file.  KNOTPLOT=/bin/ls reproduced it exactly.
    install = refresh_install(announce=False)
    show(installation_message(install, state["explicit"], state["probe"],
                              explicit_label=LOCATED_LABEL) + "\n\n"
         "Type a knot name such as 4.1, or pick some with Browse catalogue...\n"
         "A target with a space in it is passed to KnotPlot as a command, so\n"
         "\"torus 2 3\" builds the (2,3) torus knot.\n\n"
         "Order of operations:\n"
         "    KnotPlot catalogue entry\n"
         "      -> load 6.3.2         <- or a generator command, torus 2 3\n"
         "      x  refine nbeads N    <- optional; on a link, the total\n"
         "      -> coords dump        <- KnotPlot's own ascii writer\n"
         "      -  components         <- one curve each, closure measured\n"
         "      -> XYZ components     <- x y z rows, blank line between them\n\n"
         "Mirror images: 585 catalogue files hold 522 distinct entries, and\n"
         "the reduction only merges byte-identical files. 30 of the 93\n"
         "dotted/underscored pairs DIFFER and both are kept. 28 of those 30\n"
         "are the two chiral forms of one knot - 8.19 has writhe -8.626478,\n"
         "8_19 +8.626478 - and the other 2, 9.20 / 9_20 (+6.346644 and\n"
         "+6.346636) and 9.35 / 9_35 (+7.539209 and +7.539219), share a\n"
         "handedness. Those two are not one conformation stored twice: the\n"
         "files differ, and so do their writhes, in the fifth decimal.\n"
         "Writhe from " + WRITHE_TOOL + " at its defaults.\n\n"
         "Mirror image reflects whatever you extract: one coordinate is\n"
         "negated, which negates writhe and linking number. z is the default\n"
         "axis because it keeps the projection and swaps every crossing.\n\n"
         "Shapes are the ones KnotPlot ships, exported unchanged.\n\n"
         "Click any light-blue  ?  for an explanation and examples.\n")

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
        return run_gui(args.outdir if args.outdir != DEFAULT_OUTDIR else None)
    if not (args.targets or args.all or args.list_names or args.check):
        parser.error("give at least one target, or --all, --list, --check "
                     "(or use --gui)")
    try:
        return run_cli(args)
    except FileNotFoundError as exc:
        print("error: cannot open %s" % exc.filename, file=sys.stderr)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
