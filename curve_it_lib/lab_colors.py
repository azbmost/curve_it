#!/usr/bin/env python3
"""
lab_colors.py -- the DiLiuLab figure colours, shared by every curve_it tool.

The palette is stored once, in assets/diliulab_colors.json, so that any script
can read it without importing anything; this module is the one Python reader
and the one place the tools ask for it.  It holds the nine DiLiuLab colours
and their tint-matched neutral at the lab's five tint levels, T100, T80, T60,
T40 and T20, exactly as gr_colors V3.3 (https://github.com/DiLiuLab/gr_colors)
generates them:

    hue_i = i * 0.618033988749895 mod 1,  S 0.67,  V 0.90,  H start 0
    indexes 0 1 3 4 5 6 7 10 12  ->  red blue magenta cyan orange purple
                                     green yellow mint green
    tint T mixes toward white:  c_T = round(T * c + (1 - T) * 255)
    neutral: black mixed the same way, so black at T100 and gray below

Every tool keeps its own colours as the default; the lab palette is an
option, named on the command line as

    --palette DiLiuLab          the T100 colours
    --palette DiLiuLab-T80      or T60, T40, T20 ("DiLiuLab T80", "diliulab80"
                                and "diliulab:t80" are read the same way)

One colour can also be named, in the spelling of the lab's own colour lists:
the name in CamelCase and its tint, RedT80, MintGreenT60, GrayT40, BlackT100
(case, spaces, "_" and "-" do not matter; "DiLiuLab red" means RedT100).  The
tint is required because the bare names are ordinary colour names elsewhere,
red being #ff0000 in SVG, Tk and vect_io; resolve_lab_color() turns a name
into its hex code and returns None for anything that is not one.

    python3 lab_colors.py                       # print every tint, with the names
    python3 lab_colors.py --tint T80            # one tint
    python3 lab_colors.py --name MintGreenT80   # one colour's hex code
    python3 lab_colors.py --write-asset         # regenerate the asset from the formula

If the asset is missing, unreadable or not laid out as the formula writes
it, the formula above is used instead, so the option never disappears;
tests/test_lab_colors.py holds the two to the same values.

curve_it does not need gr_colors: nothing here imports, runs or reads it.  The
values live in the asset, the fallback formula is this module's own code, and
the tests compare both against values recorded from gr_colors V3.3, so a
gr_colors checkout is never needed and a later gr_colors cannot change them.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

__version__ = "1.1"

PALETTE_NAME = "DiLiuLab"
ASSET_NAME = "diliulab_colors.json"
TINTS = ("T100", "T80", "T60", "T40", "T20")
DEFAULT_TINT = "T100"

# The generator settings and the curated indexes, from gr_colors V3.3.
GOLDEN_RATIO_CONJUGATE = 0.618033988749895
SATURATION = 0.67
VALUE = 0.90
START_HUE = 0.0
LAB_INDEXES = (
    (0, "red"), (1, "blue"), (3, "magenta"), (4, "cyan"), (5, "orange"),
    (6, "purple"), (7, "green"), (10, "yellow"), (12, "mint green"),
)

_SPEC = re.compile(r"^\s*diliu\s*lab(?:[\s_:@-]*t?\s*(\d{1,3}))?\s*$", re.IGNORECASE)
_DEFAULT_WORDS = {"", "default", "none", "auto"}
_CACHE: Dict[str, dict] = {}


def resource_path(relative_path):
    """Return a resource path that also works from a PyInstaller bundle."""
    source_dir = os.path.dirname(os.path.abspath(__file__))
    source_root = (os.path.dirname(source_dir)
                   if os.path.basename(source_dir) == "curve_it_lib" else source_dir)
    base_dir = getattr(sys, "_MEIPASS", source_root)
    return os.path.join(base_dir, relative_path)


def asset_path() -> str:
    """Where the palette asset lives: <repo>/assets/diliulab_colors.json."""
    return resource_path(os.path.join("assets", ASSET_NAME))


# --------------------------------------------------------------------------
# the formula (what the asset holds)
# --------------------------------------------------------------------------
def _tint_fraction(tint: str) -> float:
    return int(tint[1:]) / 100.0


def _hsv_to_rgb255(hue, saturation, value):
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, saturation, value)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def _apply_tint(rgb, fraction):
    # gr_colors' apply_tint, rounding included, so every value matches it.
    return tuple(int(round(fraction * c + (1.0 - fraction) * 255)) for c in rgb)


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def generate_palette() -> dict:
    """The palette computed from gr_colors' formula, in the asset's layout."""
    colors = []
    for index, name in LAB_INDEXES:
        base = _hsv_to_rgb255(START_HUE + index * GOLDEN_RATIO_CONJUGATE, SATURATION, VALUE)
        entry = {"index": index, "name": name}
        for tint in TINTS:
            entry[tint] = _hex(_apply_tint(base, _tint_fraction(tint)))
        colors.append(entry)
    neutral = {"index": "K", "name": "neutral"}
    for tint in TINTS:
        neutral[tint] = _hex(_apply_tint((0, 0, 0), _tint_fraction(tint)))
    return {
        "name": PALETTE_NAME,
        "description": ("The nine DiLiuLab figure colours and their tint-matched "
                        "neutral at the lab's five tint levels."),
        "source": ("gr_colors V3.3, https://github.com/DiLiuLab/gr_colors: golden-ratio "
                   "HSV, hue_i = i * 0.618033988749895 mod 1, S 0.67, V 0.90, H start 0; "
                   "indexes 0 1 3 4 5 6 7 10 12; tint T mixes toward white as "
                   "round(T * c + (1 - T) * 255); the neutral is black mixed the same way"),
        "tints": list(TINTS),
        "default_tint": DEFAULT_TINT,
        "colors": colors,
        "neutral": neutral,
    }


def palette_json(palette: Optional[dict] = None) -> str:
    """The asset's exact text: two-space indent, one trailing newline."""
    return json.dumps(palette or generate_palette(), indent=2) + "\n"


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def _checked(data: dict, where: str) -> dict:
    """The asset as read, or ValueError for any shape the readers below cannot
    use, so that load_palette falls back to the formula rather than a later
    call failing on it."""
    if not isinstance(data, dict):
        raise ValueError("%s does not hold a JSON object" % where)
    tints, colors, neutral = data.get("tints"), data.get("colors"), data.get("neutral")
    if not tints or not isinstance(tints, list) or not isinstance(colors, list) or not colors:
        raise ValueError("%s has no tints or no colours" % where)
    if DEFAULT_TINT not in tints or not all(
            isinstance(tint, str) and re.match(r"^T(?:100|[1-9][0-9]?)$", tint) for tint in tints):
        raise ValueError("%s: the tints must be labels such as T80 and include %s"
                         % (where, DEFAULT_TINT))
    if not all(isinstance(entry, dict) for entry in colors + [neutral]):
        raise ValueError("%s: every colour and the neutral must be an object" % where)
    if not all(isinstance(entry.get("name"), str) and entry["name"].strip() for entry in colors):
        raise ValueError("%s: every colour needs a name" % where)
    hexcode = re.compile(r"^#[0-9a-fA-F]{6}$")
    for entry in colors + [neutral]:
        for tint in tints:
            if not hexcode.match(str(entry.get(tint, ""))):
                raise ValueError("%s: %s has no #rrggbb value for %s"
                                 % (where, entry.get("name", "?"), tint))
    return data


def load_palette(path: Optional[str] = None) -> dict:
    """The palette from the asset, or from the formula when the asset cannot
    be read.  The result is cached per path; the caller must not modify it."""
    path = path or asset_path()
    if path not in _CACHE:
        try:
            with open(path, encoding="utf-8") as handle:
                data = _checked(json.load(handle), path)
            data = dict(data, loaded_from=path)
        except (OSError, ValueError) as exc:
            data = dict(generate_palette(), loaded_from=None, load_error=str(exc))
        _CACHE[path] = data
    return _CACHE[path]


def available_tints(palette: Optional[dict] = None) -> List[str]:
    return list((palette or load_palette())["tints"])


def normalize_tint(tint) -> str:
    """80, "80", "t80", "T80", 0.8 and "0.8" all mean "T80"."""
    if tint is None or str(tint).strip() == "":
        return DEFAULT_TINT
    text = str(tint).strip().upper().lstrip("T").strip()
    try:
        number = float(text)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):           # "x", "inf" and "nan" alike
        raise ValueError("tint %r is not one of %s" % (tint, ", ".join(available_tints())))
    if 0.0 < number <= 1.0 and "." in text:
        number *= 100.0
    label = "T%d" % int(round(number))
    if label not in available_tints():
        raise ValueError("tint %r is not one of %s" % (tint, ", ".join(available_tints())))
    return label


def lab_named_colors(tint=DEFAULT_TINT, neutral: bool = False) -> List[Tuple[str, str]]:
    """[(name, "#rrggbb"), ...] in the lab's order, optionally ending with the
    neutral, which is named black at T100 and gray below."""
    tint = normalize_tint(tint)
    palette = load_palette()
    rows = [(entry["name"], entry[tint].lower()) for entry in palette["colors"]]
    if neutral:
        rows.append(("black" if tint == "T100" else "gray",
                     palette["neutral"][tint].lower()))
    return rows


def lab_colors(tint=DEFAULT_TINT, neutral: bool = False) -> List[str]:
    """The palette's hex codes in the lab's order."""
    return [code for _name, code in lab_named_colors(tint, neutral)]


def hex_to_rgb01(code: str) -> Tuple[float, float, float]:
    digits = code.strip().lstrip("#")
    return tuple(int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# --------------------------------------------------------------------------
# the --palette option
# --------------------------------------------------------------------------
def palette_label(tint=DEFAULT_TINT) -> str:
    """The spelling the GUIs show: "DiLiuLab T80"."""
    return "%s %s" % (PALETTE_NAME, normalize_tint(tint))


def palette_choices(default_label: str = "default") -> List[str]:
    """What a palette menu offers: the tool's own colours first, then the lab
    palette at every tint."""
    return [default_label] + [palette_label(t) for t in available_tints()]


def lab_tint_of(spec) -> Optional[str]:
    """The tint a palette name asks for, or None when it does not name the
    DiLiuLab palette.  An unknown tint is an error, not a None."""
    if spec is None:
        return None
    match = _SPEC.match(str(spec))
    if not match:
        return None
    return normalize_tint(match.group(1)) if match.group(1) else DEFAULT_TINT


def is_default(spec) -> bool:
    return spec is None or str(spec).strip().lower() in _DEFAULT_WORDS


def check_palette(spec) -> str:
    """Validate a --palette value; returns "default" or "DiLiuLab T..".  For
    argparse's type= as well as for GUI fields."""
    if is_default(spec):
        return "default"
    tint = lab_tint_of(spec)
    if tint is None:
        raise ValueError("unknown palette %r: use default or %s, optionally with a tint "
                         "(%s), e.g. %s-T80" % (spec, PALETTE_NAME, ", ".join(available_tints()),
                                                PALETTE_NAME))
    return palette_label(tint)


def resolve_palette(spec, default: Sequence[str], neutral: bool = False) -> List[str]:
    """The colours a --palette value stands for: the tool's own `default`
    list, or the lab palette at the tint the value names."""
    label = check_palette(spec)
    if label == "default":
        return list(default)
    return lab_colors(label.split()[-1], neutral=neutral)


def argparse_palette(value: str) -> str:
    """argparse type= wrapper: a bad palette becomes a usage error."""
    try:
        return check_palette(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


PALETTE_HELP = ("colour palette: default (this tool's own colours) or %s, the DiLiuLab "
                "figure colours, optionally at a tint: %s-T80, -T60, -T40 or -T20 "
                "(default tint T100)" % (PALETTE_NAME, PALETTE_NAME))


# --------------------------------------------------------------------------
# colour names
# --------------------------------------------------------------------------
# One lab colour by name, spelled the way the lab's own colour lists (the
# gr_colors .clr swatches) spell it: the name in CamelCase followed by its
# tint, RedT80, MintGreenT60, GrayT40, BlackT100.  The tint is required
# because the bare names are ordinary colour names everywhere else -- red is
# #ff0000 in SVG, in Tk and in vect_io -- so "red" keeps meaning what it
# always meant; a DiLiuLab prefix may stand in for the tint instead, and then
# means T100 ("DiLiuLab red" is RedT100).  Case, spaces, "_" and "-" do not
# matter: "mint green t80" and "Mint_Green-T80" are MintGreenT80.
_NEUTRAL_WORDS = ("black", "gray", "grey", "neutral")
_NAME_TINT = re.compile(
    r"^\s*(?:diliu\s*lab[\s_:@-]*)?([a-z][a-z\s_-]*?)[\s_:@-]*t[\s_-]*(\d{1,3})\s*$", re.IGNORECASE)
_NAME_PREFIXED = re.compile(r"^\s*diliu\s*lab[\s_:@-]*([a-z][a-z\s_-]*?)\s*$", re.IGNORECASE)


def _name_key(name: str) -> str:
    return re.sub(r"[\s_-]+", "", str(name).lower())


def _camel(name: str) -> str:
    return "".join(word.capitalize() for word in str(name).split())


def lab_color_name(name: str, tint=DEFAULT_TINT) -> str:
    """The lab's spelling of one colour at one tint, "MintGreenT80".  The
    neutral is BlackT100 at full tint and GrayT80 ... GrayT20 below it, as in
    the lab's colour lists; black, gray, grey and neutral all name it."""
    tint = normalize_tint(tint)
    key = _name_key(name)
    if key in _NEUTRAL_WORDS:
        return ("Black" if tint == DEFAULT_TINT else "Gray") + tint
    for entry in load_palette()["colors"]:
        if _name_key(entry["name"]) == key:
            return _camel(entry["name"]) + tint
    raise ValueError("%r is not a %s colour; the names are %s" % (
        name, PALETTE_NAME, ", ".join(_camel(e["name"]) for e in load_palette()["colors"])
        + " and Black/Gray for the neutral"))


def lab_color_names(tint=DEFAULT_TINT, neutral: bool = False) -> List[str]:
    """Every colour's name at one tint, in the lab's order: RedT80, BlueT80, ..."""
    return [lab_color_name(name, tint) for name, _code in lab_named_colors(tint, neutral)]


def resolve_lab_color(text) -> Optional[str]:
    """The "#rrggbb" a DiLiuLab colour name stands for, or None when `text` is
    not spelled as one -- a hex code, an ordinary colour name such as red,
    numbers, or blank -- so a caller can fall through to its own colour
    parsing.  One of the palette's names with a tint the palette does not
    hold, such as RedT75, is an error rather than a None."""
    if text is None:
        return None
    match = _NAME_TINT.match(str(text))
    if match:
        name, tint = match.group(1), match.group(2)
    else:
        match = _NAME_PREFIXED.match(str(text))
        if not match:
            return None
        name, tint = match.group(1), DEFAULT_TINT
    key = _name_key(name)
    if key == _name_key(PALETTE_NAME):
        return None                         # "DiLiuLab-T80" names the palette, not a colour
    palette = load_palette()
    if key in _NEUTRAL_WORDS:
        entry = palette["neutral"]
    else:
        entry = next((e for e in palette["colors"] if _name_key(e["name"]) == key), None)
        if entry is None:
            if re.match(r"^\s*diliu\s*lab", str(text), re.IGNORECASE):
                lab_color_name(name)        # raises, listing the palette's names
            return None                     # "wheat1" is not ours to judge
    try:
        tint = normalize_tint(tint)
    except ValueError:
        example = next((t for t in available_tints() if t != DEFAULT_TINT), DEFAULT_TINT)
        raise ValueError("%s colour %r: the tint must be one of %s, e.g. %s" % (
            PALETTE_NAME, str(text).strip(), ", ".join(available_tints()),
            lab_color_name(name, example)))
    return entry[tint].lower()


def is_lab_color_name(text) -> bool:
    try:
        return resolve_lab_color(text) is not None
    except ValueError:
        return True                         # spelled as one, with a bad tint


LAB_COLOR_HELP = ("a %s colour may be given by name with its tint, as in the lab's colour "
                  "lists: RedT100, BlueT80, MintGreenT60, GrayT40 (Red, Blue, Magenta, Cyan, "
                  "Orange, Purple, Green, Yellow, MintGreen, and Black/Gray for the neutral; "
                  "tints T100, T80, T60, T40, T20)" % PALETTE_NAME)


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="lab_colors.py",
        description="Print the DiLiuLab figure colours, or regenerate their asset.")
    # one thing at a time: --name, --tint and --write-asset exclude each other
    what = parser.add_mutually_exclusive_group()
    what.add_argument("--tint", default=None,
                      help="one tint only: %s" % ", ".join(TINTS))
    what.add_argument("--write-asset", nargs="?", const=asset_path(), metavar="PATH",
                      help="write the palette computed from gr_colors' formula as JSON "
                           "(default: the package asset)")
    what.add_argument("--name", metavar="NAME",
                      help="print the hex code of one lab colour given by name, e.g. MintGreenT80")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    if args.name is not None:
        try:
            code = resolve_lab_color(args.name)
        except ValueError as exc:
            parser.error(str(exc))
        if code is None:
            parser.error("%r is not a %s colour name; %s" % (args.name, PALETTE_NAME, LAB_COLOR_HELP))
        print(code)
        return 0
    if args.write_asset:
        try:
            with open(args.write_asset, "w", encoding="utf-8") as handle:
                handle.write(palette_json())
        except OSError as exc:
            parser.error("cannot write %s: %s" % (args.write_asset, exc.strerror or exc))
        print("wrote %s" % args.write_asset)
        return 0
    try:                                    # a usage error before any output
        tints = [normalize_tint(args.tint)] if args.tint else available_tints()
    except ValueError as exc:
        parser.error(str(exc))
    palette = load_palette()
    source = palette.get("loaded_from") or "gr_colors formula (asset unreadable: %s)" % (
        palette.get("load_error"))
    print("%s palette, from %s" % (PALETTE_NAME, source))
    for tint in tints:
        print("%-5s " % tint + "  ".join(
            "%s %s" % (code, lab_color_name(name, tint))
            for name, code in lab_named_colors(tint, neutral=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
