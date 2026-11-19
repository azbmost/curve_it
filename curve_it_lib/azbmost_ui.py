#!/usr/bin/env python3
"""AZBMOST UI: the shared look of the AZBMOST launcher, BNP_NA, RE_Helix and Curve_It.

One ttk theme ("azbmost", built on clam so it draws the same on every platform),
the DiLiuLab colours, a type scale, crisp vector icons and a few composite
widgets: window header, section cards, round help buttons, tool lists, a log
view, a footer with the run buttons, and drop-in themed stand-ins for the
classic Tk widgets.

Identical copies of this file live in

    azbmost/azbmost_ui.py
    bnp_na/bnp_na_lib/azbmost_ui.py
    re_helix/re_helix_lib/azbmost_ui.py
    curve_it/curve_it_lib/azbmost_ui.py

Edit one, then copy it over the other three; UI_VERSION says which is newer.

Only the look lives here. Nothing in this file changes what a tool computes,
reads or writes, and every widget it makes keeps the API of the Tk widget it
stands in for.

    python3 azbmost_ui.py --demo            show every widget

The icon artwork is in azbmost_icons.py beside this file.
"""
from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # the artwork lives beside this file
    from . import azbmost_icons as _icons  # type: ignore[no-redef]
except ImportError:
    try:
        import azbmost_icons as _icons  # type: ignore[no-redef]
    except ImportError:
        _icons = None  # icons fall back to plain tiles

UI_VERSION = "1.0.0"
THEME_NAME = "azbmost"

# ---------------------------------------------------------------- colours
# The nine DiLiuLab figure colours at the lab's five tints (gr_colors V3.3,
# assets/diliulab_colors.json in curve_it). Icons draw only with these.
LAB: Dict[str, Dict[str, str]] = {
    "red": {"T100": "#e64c4c", "T80": "#eb7070", "T60": "#f09494", "T40": "#f5b7b7", "T20": "#fadbdb"},
    "blue": {"T100": "#4c79e6", "T80": "#7094eb", "T60": "#94aff0", "T40": "#b7c9f5", "T20": "#dbe4fa"},
    "magenta": {"T100": "#e64cd2", "T80": "#eb70db", "T60": "#f094e4", "T40": "#f5b7ed", "T20": "#fadbf6"},
    "cyan": {"T100": "#4ce6cc", "T80": "#70ebd6", "T60": "#94f0e0", "T40": "#b7f5eb", "T20": "#dbfaf5"},
    "orange": {"T100": "#e69f4c", "T80": "#ebb270", "T60": "#f0c594", "T40": "#f5d9b7", "T20": "#faecdb"},
    "purple": {"T100": "#724ce6", "T80": "#8e70eb", "T60": "#aa94f0", "T40": "#c7b7f5", "T20": "#e3dbfa"},
    "green": {"T100": "#52e64c", "T80": "#75eb70", "T60": "#97f094", "T40": "#baf5b7", "T20": "#dcfadb"},
    "yellow": {"T100": "#d9e64c", "T80": "#e1eb70", "T60": "#e8f094", "T40": "#f0f5b7", "T20": "#f7fadb"},
    "mint": {"T100": "#4ce698", "T80": "#70ebad", "T60": "#94f0c1", "T40": "#b7f5d6", "T20": "#dbfaea"},
    "neutral": {"T100": "#000000", "T80": "#333333", "T60": "#666666", "T40": "#999999", "T20": "#cccccc"},
}


def lab(hue: str, tint: str = "T100") -> str:
    return LAB[hue][tint]


def _rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def mix(color_a: str, color_b: str, t: float) -> str:
    """Blend color_a toward color_b by t (0 keeps a, 1 gives b)."""
    ra, ga, ba = _rgb(color_a)
    rb, gb, bb = _rgb(color_b)
    t = max(0.0, min(1.0, float(t)))
    return "#%02x%02x%02x" % (
        round(ra + (rb - ra) * t),
        round(ga + (gb - ga) * t),
        round(ba + (bb - ba) * t),
    )


class C:
    """Interface colours. Greys are interface chrome, not figure colours."""

    BG = "#ffffff"            # window and card surface
    SUBTLE = "#f6f8fa"        # footer, read-only fields, secondary buttons
    SUBTLE_HOVER = "#eef1f4"
    SUBTLE_PRESSED = "#e5e9ee"
    BORDER = "#d8dee4"        # card outlines, separators
    BORDER_INPUT = "#c9d1d9"  # entries, combo boxes
    BORDER_HOVER = "#aeb7c1"
    TROUGH = "#eaeef2"
    TEXT = "#1f2328"
    TEXT_2 = "#4b535d"        # field labels
    MUTED = "#6e7781"         # hints
    DISABLED = "#a5adb6"
    DISABLED_BG = "#f3f4f6"
    ACCENT = LAB["blue"]["T100"]          # the one interaction colour
    ACCENT_HOVER = mix(LAB["blue"]["T100"], "#000000", 0.10)
    ACCENT_PRESSED = mix(LAB["blue"]["T100"], "#000000", 0.20)
    ACCENT_SOFT = LAB["blue"]["T20"]
    ACCENT_SOFT_2 = LAB["blue"]["T40"]
    OK = LAB["green"]["T100"]
    WARN = LAB["orange"]["T100"]
    ERROR = LAB["red"]["T100"]
    INFO = LAB["blue"]["T100"]
    LOG_BG = "#fbfcfd"
    ICON_TOP = "#333333"      # neutral T80
    ICON_BOTTOM = "#000000"   # neutral T100


# Module identities: name as written, number, hue and a one-line purpose.
MODULES: Dict[str, Dict[str, object]] = {
    "azbmost": {"name": "AZBMOST", "number": 0, "hue": "blue",
                "tagline": "Nucleic-acid model building suite"},
    "bnp_na": {"name": "BNP_NA", "number": 1, "hue": "blue",
               "tagline": "Build and place nucleic-acid helices"},
    "re_helix": {"name": "RE_Helix", "number": 2, "hue": "purple",
                 "tagline": "Align helices and perform reciprocal exchanges"},
    "curve_it": {"name": "Curve_It", "number": 3, "hue": "cyan",
                 "tagline": "Sculpt PDB structures along any 3D curve"},
}


def module_hue(module: Optional[str]) -> str:
    info = MODULES.get(module or "", MODULES["azbmost"])
    return str(info["hue"])


# Every tool window and tool list entry: icon key -> (module, one-line purpose).
# The button labels themselves stay in each module, unchanged.
TOOL_INFO: Dict[str, Tuple[str, str]] = {
    # BNP_NA
    "bz_builder": ("bnp_na", "Join alternating B- and Z-DNA PDBs into one helix."),
    "triplex": ("bnp_na", "Add strand III to a duplex; tune X-disp (e.g. -1.9 Å) and h-Twist (e.g. 30°) first."),
    "measure_angle": ("bnp_na", "Around-axis angle between two atoms or XYZ points, with a .bild drawing."),
    "align_z": ("bnp_na", "Move a helix PDB so its DSSR axis lies along +Z."),
    "axis_info": ("bnp_na", "Axis ends, direction and angle for two chains; optional .bild."),
    "add_phosphates": ("bnp_na", "Report and add terminal 5′ and 3′ phosphates by chain."),
    "xyz_bild": ("bnp_na", "A coordinate-axes .bild helper for Chimera and ChimeraX."),
    "combine_pdb": ("bnp_na", "Merge PDB files or selected chains; IDs, LINKs and REMARKs follow."),
    "regularize_phosphates": ("bnp_na", "Restore the regular backbone repeat after Phenix minimization."),
    # RE_Helix
    "bend_helix": ("re_helix", "Bend a straight two-chain helix at a phosphorus pivot."),
    "do_symmetry": ("re_helix", "Average a pseudosymmetric assembly into an exactly symmetric model."),
    "add_link": ("re_helix", "Cyclize termini or stage LINK records and rebuild chain topology."),
    "insert_virtual_resi": ("re_helix", "Open residue-numbering gaps after chosen residues."),
    "permute_chain": ("re_helix", "Cyclically rotate whole residue blocks within chains."),
    "generate_lattice": ("re_helix", "Write a P1 CRYST1 record from three lattice vectors."),
    "phenix_restraints": ("re_helix", "Turn LINK records into Phenix geometry restraints."),
    "reverse_strand": ("re_helix", "Reverse the residue order of chains without moving atoms."),
    "check_clashes": ("re_helix", "Count close heavy-atom contacts in any PDB."),
    # Curve_It
    "convert_xyz": ("curve_it", "Convert curves between XYZ, molecular XYZ, VECT and fake PDB."),
    "svg2xyz": ("curve_it", "Turn an SVG drawing into planar curve components."),
    "kp2xyz": ("curve_it", "Extract knots and links from KnotPlot's catalogue."),
    "generate_helix": ("curve_it", "Write a circular-helix curve file."),
    "generate_sc": ("curve_it", "Supercoil axes (plectonemic or loose) with exact writhe."),
    "local_curvature": ("curve_it", "Per-point curvature, torsion and writhe density as CSV."),
    "plane_it": ("curve_it", "Project atoms or points onto a 2D SVG."),
    "curved_connector": ("curve_it", "Screen bent connector duplexes between two helix ends."),
    "xyz2model": ("curve_it", "Sweep curves into STL and GLB solids for printing."),
    "multicolor_split": ("curve_it", "One STL per colour for multi-material printing."),
    "reverse_curve_it": ("curve_it", "Recover a centerline XYZ from a curved and a straight PDB."),
    "get_phase": ("curve_it", "Find the helix phase that best matches a target."),
    "view_curve": ("curve_it", "Look at the loaded curve in 3D."),
}


def tool_description(key: str) -> str:
    return TOOL_INFO.get(key, ("", ""))[1]


# ------------------------------------------------------------------ fonts
_FONT_SPECS = (
    # name, size factor, weight, family role
    ("AzbBody", 1.0, "normal", "body"),
    ("AzbBodyBold", 1.0, "bold", "body"),
    ("AzbSmall", 0.85, "normal", "body"),
    ("AzbSmallBold", 0.85, "bold", "body"),
    ("AzbTiny", 0.77, "bold", "body"),
    ("AzbSection", 1.0, "bold", "body"),
    ("AzbTitle", 1.16, "bold", "display"),
    ("AzbDisplay", 1.55, "bold", "display"),
    ("AzbMono", 0.85, "normal", "mono"),
)


def _pick_family(root: tk.Misc, candidates: Sequence[str], fallback: str) -> str:
    try:
        available = set(tkfont.families(root))
    except Exception:
        available = set()
    for family in candidates:
        if family in available:
            return family
    return fallback


def ensure_fonts(root: tk.Misc) -> Dict[str, tkfont.Font]:
    """Create (once per interpreter) the named fonts the styles refer to."""
    cache = getattr(root, "_azbmost_fonts", None)
    if cache:
        return cache
    base = tkfont.nametofont("TkDefaultFont", root=root).actual()
    body_family = base.get("family") or "TkDefaultFont"
    body_size = abs(int(base.get("size") or 13)) or 13
    if sys.platform == "darwin":
        body_size = max(body_size, 13)
    else:
        body_size = max(body_size, 10)
    mono_family = _pick_family(root, ("Menlo", "SF Mono", "Consolas", "DejaVu Sans Mono", "Courier New"),
                               tkfont.nametofont("TkFixedFont", root=root).actual().get("family", "Courier"))
    families = {"body": body_family, "display": body_family, "mono": mono_family}
    fonts: Dict[str, tkfont.Font] = {}
    for name, factor, weight, role in _FONT_SPECS:
        size = max(8, int(round(body_size * factor)))
        if role == "mono":
            size = max(9, int(round(body_size * 0.88)))
        try:
            font = tkfont.nametofont(name, root=root)
            font.configure(family=families[role], size=size, weight=weight)
        except tk.TclError:
            font = tkfont.Font(root=root, name=name, family=families[role], size=size, weight=weight)
        fonts[name] = font
    try:
        root._azbmost_fonts = fonts
    except Exception:
        pass
    return fonts


# ------------------------------------------------------------------ theme
def _style_map(style: ttk.Style, name: str, **kw) -> None:
    style.map(name, **kw)


def _configure_styles(style: ttk.Style) -> None:
    body, bold, small = "AzbBody", "AzbBodyBold", "AzbSmall"
    style.configure(
        ".",
        background=C.BG, foreground=C.TEXT, font=body,
        bordercolor=C.BORDER, darkcolor=C.BG, lightcolor=C.BG,
        troughcolor=C.TROUGH, focuscolor=C.ACCENT, focusthickness=0,
        selectbackground=C.ACCENT_SOFT, selectforeground=C.TEXT,
        insertcolor=C.TEXT, insertwidth=1, fieldbackground=C.BG,
        arrowcolor=C.TEXT_2, relief="flat",
    )
    style.map(".", foreground=[("disabled", C.DISABLED)])

    # Frames and labels -------------------------------------------------
    style.configure("TFrame", background=C.BG)
    style.configure("Subtle.TFrame", background=C.SUBTLE)
    style.configure("Footer.TFrame", background=C.SUBTLE)
    style.configure("Header.TFrame", background=C.BG)
    style.configure("TLabel", background=C.BG, foreground=C.TEXT)
    style.configure("Field.TLabel", foreground=C.TEXT_2)
    style.configure("Hint.TLabel", foreground=C.MUTED, font=small)
    style.configure("Status.TLabel", foreground=C.MUTED, font=small)
    style.configure("Muted.TLabel", foreground=C.MUTED)
    style.configure("Strong.TLabel", font=bold)
    style.configure("Section.TLabel", font="AzbSection", foreground=C.TEXT)
    style.configure("Title.TLabel", font="AzbTitle", foreground=C.TEXT)
    style.configure("Display.TLabel", font="AzbDisplay", foreground=C.TEXT)
    style.configure("Warn.TLabel", foreground=mix(C.WARN, "#000000", 0.35), font="AzbSmallBold")
    style.configure("Error.TLabel", foreground=mix(C.ERROR, "#000000", 0.25), font="AzbSmallBold")
    style.configure("Value.TLabel", font=bold, foreground=C.TEXT)
    style.configure("Mono.TLabel", font="AzbMono", foreground=C.TEXT_2)
    style.configure("Link.TLabel", foreground=C.ACCENT)
    for prefix, bg in (("Footer", C.SUBTLE), ("Subtle", C.SUBTLE)):
        style.configure(f"{prefix}.TLabel", background=bg)
        style.configure(f"{prefix}.Hint.TLabel", background=bg, foreground=C.MUTED, font=small)
        style.configure(f"{prefix}.Status.TLabel", background=bg, foreground=C.TEXT_2, font=small)
        style.configure(f"{prefix}.TCheckbutton", background=bg)

    # Cards (labelframes) -------------------------------------------------
    for name in ("TLabelframe", "Card.TLabelframe"):
        style.configure(name, background=C.BG, bordercolor=C.BORDER, lightcolor=C.BG,
                        darkcolor=C.BG, relief="raised", borderwidth=1, padding=(14, 8, 14, 12),
                        labelmargins=(10, 0, 10, 0))
        style.configure(f"{name}.Label", background=C.BG, foreground=C.TEXT, font="AzbSection")
    style.configure("Subtle.TLabelframe", background=C.SUBTLE)
    style.configure("Subtle.TLabelframe.Label", background=C.SUBTLE)

    # Buttons --------------------------------------------------------------
    style.configure("TButton", background=C.SUBTLE, foreground=C.TEXT, bordercolor=C.BORDER_INPUT,
                    lightcolor=C.SUBTLE, darkcolor=C.SUBTLE, padding=(12, 4), relief="raised",
                    borderwidth=1, anchor="center", font=body, focusthickness=0)
    style.map("TButton",
              background=[("disabled", C.SUBTLE), ("pressed", C.SUBTLE_PRESSED), ("active", C.SUBTLE_HOVER)],
              lightcolor=[("disabled", C.SUBTLE), ("pressed", C.SUBTLE_PRESSED), ("active", C.SUBTLE_HOVER)],
              darkcolor=[("disabled", C.SUBTLE), ("pressed", C.SUBTLE_PRESSED), ("active", C.SUBTLE_HOVER)],
              bordercolor=[("disabled", C.BORDER), ("focus", C.ACCENT), ("active", C.BORDER_HOVER)],
              foreground=[("disabled", C.DISABLED)])
    style.configure("Accent.TButton", background=C.ACCENT, foreground="#ffffff", bordercolor=C.ACCENT_HOVER,
                    lightcolor=C.ACCENT, darkcolor=C.ACCENT, font=bold, padding=(16, 5))
    style.map("Accent.TButton",
              background=[("disabled", C.ACCENT_SOFT_2), ("pressed", C.ACCENT_PRESSED), ("active", C.ACCENT_HOVER)],
              lightcolor=[("disabled", C.ACCENT_SOFT_2), ("pressed", C.ACCENT_PRESSED), ("active", C.ACCENT_HOVER)],
              darkcolor=[("disabled", C.ACCENT_SOFT_2), ("pressed", C.ACCENT_PRESSED), ("active", C.ACCENT_HOVER)],
              bordercolor=[("disabled", C.ACCENT_SOFT_2), ("focus", C.ACCENT_PRESSED), ("active", C.ACCENT_PRESSED)],
              foreground=[("disabled", "#ffffff")])
    style.configure("Small.TButton", padding=(8, 1), font=small)
    style.configure("Ghost.TButton", background=C.BG, bordercolor=C.BG, lightcolor=C.BG, darkcolor=C.BG,
                    foreground=C.ACCENT, padding=(6, 2))
    style.map("Ghost.TButton",
              background=[("pressed", C.ACCENT_SOFT), ("active", C.SUBTLE)],
              lightcolor=[("pressed", C.ACCENT_SOFT), ("active", C.SUBTLE)],
              darkcolor=[("pressed", C.ACCENT_SOFT), ("active", C.SUBTLE)],
              bordercolor=[("focus", C.ACCENT_SOFT_2), ("active", C.SUBTLE)],
              foreground=[("disabled", C.DISABLED)])
    style.configure("Footer.TButton", background=C.BG)
    style.map("Footer.TButton",
              background=[("disabled", C.BG), ("pressed", C.SUBTLE_PRESSED), ("active", C.SUBTLE_HOVER)],
              lightcolor=[("disabled", C.BG), ("pressed", C.SUBTLE_PRESSED), ("active", C.SUBTLE_HOVER)],
              darkcolor=[("disabled", C.BG), ("pressed", C.SUBTLE_PRESSED), ("active", C.SUBTLE_HOVER)])
    style.configure("Footer.TButton", lightcolor=C.BG, darkcolor=C.BG)

    # Check and radio buttons ---------------------------------------------
    for name in ("TCheckbutton", "TRadiobutton"):
        style.configure(name, background=C.BG, foreground=C.TEXT, padding=(0, 2, 4, 2),
                        indicatorbackground=C.BG, indicatorforeground="#ffffff",
                        upperbordercolor=C.BORDER_HOVER, lowerbordercolor=C.BORDER_HOVER,
                        indicatormargin=(0, 0, 6, 0), indicatorsize=15, focusthickness=0)
        style.map(name,
                  indicatorbackground=[("disabled", C.DISABLED_BG), ("selected", C.ACCENT),
                                       ("pressed", C.ACCENT_SOFT), ("active", C.ACCENT_SOFT)],
                  upperbordercolor=[("disabled", C.BORDER), ("selected", C.ACCENT_HOVER), ("active", C.ACCENT)],
                  lowerbordercolor=[("disabled", C.BORDER), ("selected", C.ACCENT_HOVER), ("active", C.ACCENT)],
                  indicatorforeground=[("disabled", C.DISABLED), ("selected", "#ffffff")],
                  foreground=[("disabled", C.DISABLED)],
                  background=[("active", C.BG)])
    # Segmented control: radio buttons without the dot.
    style.layout("Segment.TRadiobutton", [
        ("Radiobutton.border", {"sticky": "nswe", "children": [
            ("Radiobutton.padding", {"sticky": "nswe", "children": [
                ("Radiobutton.label", {"sticky": "nswe"})]})]})])
    style.configure("Segment.TRadiobutton", background=C.BG, foreground=C.TEXT_2, bordercolor=C.BG,
                    lightcolor=C.BG, darkcolor=C.BG, relief="flat", borderwidth=0, padding=(16, 5),
                    anchor="center", font=body)
    style.map("Segment.TRadiobutton",
              background=[("disabled", C.DISABLED_BG), ("selected", C.ACCENT), ("active", C.SUBTLE_HOVER)],
              lightcolor=[("selected", C.ACCENT), ("active", C.SUBTLE_HOVER)],
              darkcolor=[("selected", C.ACCENT), ("active", C.SUBTLE_HOVER)],
              bordercolor=[("selected", C.ACCENT), ("active", C.SUBTLE_HOVER)],
              foreground=[("disabled", C.DISABLED), ("selected", "#ffffff")],
              font=[("selected", bold)])

    # Entry-like fields -------------------------------------------------------
    field_common = dict(fieldbackground=C.BG, foreground=C.TEXT, bordercolor=C.BORDER_INPUT,
                        lightcolor=C.BG, darkcolor=C.BG, padding=(7, 4), insertcolor=C.TEXT,
                        selectbackground=C.ACCENT_SOFT_2, selectforeground=C.TEXT)
    field_map = dict(
        fieldbackground=[("disabled", C.DISABLED_BG), ("readonly", C.SUBTLE)],
        bordercolor=[("disabled", C.BORDER), ("focus", C.ACCENT), ("hover", C.BORDER_HOVER)],
        lightcolor=[("disabled", C.DISABLED_BG), ("focus", C.ACCENT_SOFT), ("readonly", C.SUBTLE)],
        foreground=[("disabled", C.DISABLED)],
    )
    style.configure("TEntry", **field_common)
    style.map("TEntry", **field_map)
    # Read-only values shown as quiet tiles (still selectable for copying).
    style.configure("Stat.TEntry", fieldbackground=C.SUBTLE, bordercolor=C.SUBTLE, lightcolor=C.SUBTLE,
                    darkcolor=C.SUBTLE, font="AzbBodyBold", padding=(9, 5))
    style.map("Stat.TEntry", fieldbackground=[("readonly", C.SUBTLE), ("disabled", C.SUBTLE)],
              bordercolor=[("focus", C.ACCENT_SOFT_2), ("readonly", C.SUBTLE)],
              lightcolor=[("readonly", C.SUBTLE)], foreground=[("disabled", C.MUTED)])
    style.configure("TSpinbox", arrowcolor=C.TEXT_2, arrowsize=10, background=C.SUBTLE, **field_common)
    style.map("TSpinbox", **field_map)
    style.configure("TCombobox", arrowcolor=C.TEXT_2, arrowsize=11, background=C.BG, **field_common)
    style.map("TCombobox",
              fieldbackground=[("disabled", C.DISABLED_BG), ("readonly", C.BG)],
              bordercolor=[("disabled", C.BORDER), ("focus", C.ACCENT), ("hover", C.BORDER_HOVER)],
              lightcolor=[("disabled", C.DISABLED_BG), ("focus", C.ACCENT_SOFT)],
              background=[("disabled", C.DISABLED_BG), ("pressed", C.SUBTLE_PRESSED), ("active", C.SUBTLE)],
              foreground=[("disabled", C.DISABLED)],
              arrowcolor=[("disabled", C.DISABLED)],
              selectbackground=[("readonly", C.BG)], selectforeground=[("readonly", C.TEXT)])
    style.configure("TMenubutton", background=C.BG, foreground=C.TEXT, bordercolor=C.BORDER_INPUT,
                    lightcolor=C.BG, darkcolor=C.BG, arrowcolor=C.TEXT_2, padding=(8, 3), relief="raised",
                    borderwidth=1, arrowsize=4)
    style.map("TMenubutton",
              background=[("disabled", C.DISABLED_BG), ("pressed", C.SUBTLE_PRESSED), ("active", C.SUBTLE)],
              lightcolor=[("disabled", C.DISABLED_BG), ("pressed", C.SUBTLE_PRESSED), ("active", C.SUBTLE)],
              darkcolor=[("disabled", C.DISABLED_BG), ("pressed", C.SUBTLE_PRESSED), ("active", C.SUBTLE)],
              bordercolor=[("disabled", C.BORDER), ("focus", C.ACCENT), ("active", C.BORDER_HOVER)],
              foreground=[("disabled", C.DISABLED)], arrowcolor=[("disabled", C.DISABLED)])

    # Scrollbars without arrows ----------------------------------------------
    for orient, sticky in (("Vertical", "ns"), ("Horizontal", "we")):
        style.layout(f"{orient}.TScrollbar", [
            (f"{orient}.Scrollbar.trough", {"sticky": sticky, "children": [
                (f"{orient}.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})])
        style.configure(f"{orient}.TScrollbar", troughcolor=C.BG, background=C.BORDER_INPUT,
                        bordercolor=C.BG, lightcolor=C.BORDER_INPUT, darkcolor=C.BORDER_INPUT,
                        arrowsize=9, gripcount=0, relief="flat", borderwidth=2)
        style.map(f"{orient}.TScrollbar",
                  background=[("pressed", C.MUTED), ("active", C.BORDER_HOVER)],
                  lightcolor=[("pressed", C.MUTED), ("active", C.BORDER_HOVER)],
                  darkcolor=[("pressed", C.MUTED), ("active", C.BORDER_HOVER)])

    # Notebook ---------------------------------------------------------------
    style.configure("TNotebook", background=C.BG, bordercolor=C.BORDER, lightcolor=C.BG, darkcolor=C.BG,
                    tabmargins=(0, 4, 0, 0), padding=0)
    style.configure("TNotebook.Tab", background=C.SUBTLE, foreground=C.TEXT_2, bordercolor=C.BORDER,
                    lightcolor=C.SUBTLE, darkcolor=C.SUBTLE, padding=(14, 5), font=body)
    style.map("TNotebook.Tab",
              background=[("selected", C.BG), ("active", C.SUBTLE_HOVER)],
              lightcolor=[("selected", C.BG), ("active", C.SUBTLE_HOVER)],
              darkcolor=[("selected", C.BG), ("active", C.SUBTLE_HOVER)],
              foreground=[("selected", C.TEXT), ("disabled", C.DISABLED)],
              font=[("selected", bold)], expand=[("selected", (0, 2, 0, 0))])

    # Tables -------------------------------------------------------------------
    style.configure("Treeview", background=C.BG, fieldbackground=C.BG, foreground=C.TEXT,
                    bordercolor=C.BORDER, lightcolor=C.BG, darkcolor=C.BG, rowheight=24, font=small,
                    borderwidth=0)
    style.map("Treeview", background=[("selected", C.ACCENT_SOFT)], foreground=[("selected", C.TEXT)])
    style.configure("Treeview.Heading", background=C.SUBTLE, foreground=C.TEXT_2, font="AzbSmallBold",
                    bordercolor=C.BORDER, lightcolor=C.SUBTLE, darkcolor=C.SUBTLE, relief="flat",
                    padding=(6, 5))
    style.map("Treeview.Heading", background=[("active", C.SUBTLE_HOVER)])

    # Odds and ends ------------------------------------------------------------
    style.configure("TSeparator", background=C.BORDER)
    style.configure("TProgressbar", troughcolor=C.TROUGH, background=C.ACCENT, bordercolor=C.TROUGH,
                    lightcolor=C.ACCENT, darkcolor=C.ACCENT, thickness=6)
    style.configure("TScale", troughcolor=C.TROUGH, background=C.BG, bordercolor=C.BORDER_HOVER,
                    lightcolor=C.BG, darkcolor=C.BG, sliderlength=18, sliderthickness=18, gripcount=0)
    style.map("TScale", background=[("active", C.ACCENT_SOFT)], bordercolor=[("active", C.ACCENT)])
    style.configure("TPanedwindow", background=C.BORDER)
    style.configure("Sash", sashthickness=6, gripcount=0, background=C.BG, bordercolor=C.BG,
                    lightcolor=C.BG, darkcolor=C.BG)


def _configure_options(root: tk.Misc) -> None:
    """Defaults for the classic Tk widgets, so they match the ttk ones.

    Every pattern names a classic widget class (Label, Button, ...). A bare
    "*background" would also reach ttk widgets through their widget-level
    options and override the theme's styles.
    """
    # Tk's own class names: note "Labelframe" and "Panedwindow" (lower-case f / w).
    classic = ("Toplevel", "Frame", "Label", "Button", "Checkbutton", "Radiobutton", "Entry", "Spinbox",
               "Text", "Listbox", "Canvas", "Labelframe", "Scale", "Message", "Menubutton", "Panedwindow",
               "Scrollbar")
    opts: List[Tuple[str, object]] = []
    for cls in classic:
        opts.append((f"*{cls}.background", C.BG))
        opts.append((f"*{cls}.highlightBackground", C.BG))
        opts.append((f"*{cls}.highlightColor", C.ACCENT))
    for cls in ("Label", "Button", "Checkbutton", "Radiobutton", "Entry", "Spinbox", "Listbox", "Message",
                "Menubutton", "Labelframe", "Scale"):
        opts.append((f"*{cls}.foreground", C.TEXT))
        opts.append((f"*{cls}.font", "AzbBody"))
    for cls in ("Entry", "Spinbox", "Text", "Listbox"):
        opts.append((f"*{cls}.selectBackground", C.ACCENT_SOFT_2))
        opts.append((f"*{cls}.selectForeground", C.TEXT))
        opts.append((f"*{cls}.insertBackground", C.TEXT))
    for cls in ("Button", "Checkbutton", "Radiobutton", "Menubutton", "Scale"):
        opts.append((f"*{cls}.activeBackground", C.SUBTLE_HOVER))
        opts.append((f"*{cls}.activeForeground", C.TEXT))
    for cls in ("Button", "Checkbutton", "Radiobutton", "Label", "Menubutton", "Entry", "Spinbox"):
        opts.append((f"*{cls}.disabledForeground", C.DISABLED))
    opts += [
        ("*Text.inactiveSelectBackground", C.ACCENT_SOFT),
        ("*Scale.troughColor", C.BORDER),
        ("*Scrollbar.troughColor", C.BG),
        # Flat entries: the border is padding, the 1-pt highlight ring is the outline.
        ("*Entry.relief", "flat"),
        ("*Entry.borderWidth", 3),
        ("*Entry.highlightThickness", 1),
        ("*Entry.highlightBackground", C.BORDER_INPUT),
        ("*Entry.disabledBackground", C.DISABLED_BG),
        ("*Entry.readonlyBackground", C.SUBTLE),
        ("*Spinbox.relief", "flat"),
        ("*Spinbox.borderWidth", 3),
        ("*Spinbox.highlightThickness", 1),
        ("*Spinbox.highlightBackground", C.BORDER_INPUT),
        ("*Spinbox.buttonBackground", C.SUBTLE),
        ("*Spinbox.readonlyBackground", C.SUBTLE),
        ("*Spinbox.disabledBackground", C.DISABLED_BG),
        ("*Text.relief", "flat"),
        ("*Text.borderWidth", 0),
        ("*Text.highlightThickness", 1),
        ("*Text.highlightBackground", C.BORDER),
        ("*Text.highlightColor", C.BORDER),
        ("*Listbox.relief", "flat"),
        ("*Listbox.borderWidth", 0),
        ("*Listbox.highlightThickness", 1),
        ("*Listbox.highlightBackground", C.BORDER),
        ("*Listbox.highlightColor", C.ACCENT),
        ("*Listbox.activeStyle", "none"),
        ("*Canvas.highlightThickness", 0),
        ("*Labelframe.relief", "flat"),
        ("*Labelframe.borderWidth", 0),
        ("*Labelframe.highlightThickness", 1),
        ("*Labelframe.highlightBackground", C.BORDER),
        ("*Labelframe.highlightColor", C.BORDER),
        ("*Labelframe.font", "AzbSection"),
        ("*Labelframe.padX", 12),
        ("*Labelframe.padY", 8),
        ("*Scale.highlightThickness", 0),
        ("*Scale.sliderRelief", "raised"),
        ("*Scale.borderWidth", 1),
        ("*Scale.width", 10),
        ("*Scale.font", "AzbSmall"),
        ("*Scale.foreground", C.MUTED),
        ("*Checkbutton.highlightThickness", 0),
        ("*Radiobutton.highlightThickness", 0),
        ("*TCombobox*Listbox.background", C.BG),
        ("*TCombobox*Listbox.foreground", C.TEXT),
        ("*TCombobox*Listbox.selectBackground", C.ACCENT_SOFT),
        ("*TCombobox*Listbox.selectForeground", C.TEXT),
        ("*TCombobox*Listbox.font", "AzbBody"),
        ("*TCombobox*Listbox.highlightThickness", 0),
    ]
    for pattern, value in opts:
        try:
            root.option_add(pattern, value, "widgetDefault")
        except Exception:
            pass


def _force_light(window: tk.Misc) -> None:
    """Keep the macOS title bar light, to match the light interface."""
    if sys.platform != "darwin":
        return
    try:
        window.tk.call("::tk::unsupported::MacWindowStyle", "appearance", window, "aqua")
    except Exception:
        pass


def apply_theme(root: tk.Misc, module: Optional[str] = None) -> ttk.Style:
    """Theme a Tk root (or the root behind any widget) for the whole interpreter.

    Safe to call more than once and from a Toplevel: the theme, its styles and
    the Toplevel hook are set up once per interpreter, so a tool window opened
    inside a module does not re-lay out the module's own window. Toplevels
    created later pick the theme up on their own.
    """
    # winfo_toplevel() is the window itself for a root or Toplevel. (No
    # isinstance(tk.Tk): tests may replace tkinter.Tk with a mock.)
    top = root.winfo_toplevel()
    try:
        tk_root = top._root()
    except Exception:
        tk_root = top
    style = ttk.Style(top)
    already = bool(getattr(tk_root, "_azbmost_themed", False)) and style.theme_use() == THEME_NAME
    if not already:
        ensure_fonts(tk_root)
        if THEME_NAME not in style.theme_names():
            style.theme_create(THEME_NAME, parent="clam")
        style.theme_use(THEME_NAME)
        _configure_styles(style)
        _configure_options(tk_root)
        try:
            tk_root._azbmost_themed = True
        except Exception:
            pass
    style_window(top)
    try:
        top._azbmost_module = module or getattr(top, "_azbmost_module", None)
    except Exception:
        pass
    if not getattr(tk_root, "_azbmost_toplevel_hook", False):
        try:
            tk_root.bind_class("Toplevel", "<Map>", lambda event: _on_toplevel_map(event), add="+")
            tk_root._azbmost_toplevel_hook = True
        except Exception:
            pass
    return style


def _on_toplevel_map(event: tk.Event) -> None:
    widget = event.widget
    if getattr(widget, "_azbmost_styled", False):
        return
    try:
        if widget.winfo_class() == "Toplevel" and str(widget.winfo_toplevel()) == str(widget):
            style_window(widget)
    except Exception:
        pass


def style_window(window: tk.Misc, bg: str = C.BG) -> None:
    """Window background and light title bar for a Tk root or Toplevel.

    macOS can only change the appearance of a window that is on screen, so
    for one not mapped yet the title bar is set when it first maps.
    """
    try:
        window.configure(background=bg)
    except Exception:
        pass
    try:
        mapped = bool(window.winfo_ismapped())
    except Exception:
        mapped = True
    if mapped:
        _force_light(window)
        try:
            window._azbmost_styled = True
        except Exception:
            pass
        return
    if getattr(window, "_azbmost_pending", False):
        return
    try:
        window._azbmost_pending = True
    except Exception:
        pass

    def on_map(event: tk.Event) -> None:
        if str(event.widget) != str(window):
            return
        _force_light(window)
        try:
            window._azbmost_styled = True
            window._azbmost_pending = False
        except Exception:
            pass
        try:
            window.unbind("<Map>", binding)
        except Exception:
            pass

    try:
        binding = window.bind("<Map>", on_map, add="+")
    except Exception:
        _force_light(window)


def is_themed(widget: tk.Misc) -> bool:
    try:
        return ttk.Style(widget).theme_use() == THEME_NAME
    except Exception:
        return False


def widget_background(widget: tk.Misc, default: str = C.BG) -> str:
    """The colour a widget paints behind its children (ttk or classic)."""
    try:
        return str(widget.cget("background"))
    except Exception:
        pass
    try:
        style_name = str(widget.cget("style")) or widget.winfo_class()
        value = ttk.Style(widget).lookup(style_name, "background")
        if value:
            return str(value)
    except Exception:
        pass
    return default


# ----------------------------------------------------------- small widgets
class HelpButton(tk.Canvas):
    """A round "?" button. Same job as the old light-blue ? buttons."""

    def __init__(self, parent: tk.Misc, command: Optional[Callable[[], object]] = None,
                 size: int = 20, bg: Optional[str] = None, **kw) -> None:
        self._bg = bg or widget_background(parent)
        super().__init__(parent, width=size, height=size, background=self._bg, highlightthickness=0,
                         borderwidth=0, cursor="hand2", takefocus=1, **kw)
        self._command = command
        self._size = size
        self._state = "normal"
        pad = 1.5
        self._ring = self.create_oval(pad, pad, size - pad, size - pad, outline=C.BORDER_INPUT,
                                      fill=C.SUBTLE, width=1)
        self._mark = self.create_text(size / 2, size / 2 + 0.5, text="?", fill=C.TEXT_2, font="AzbSmallBold")
        self.bind("<Enter>", lambda _e: self._paint("hover"))
        self.bind("<Leave>", lambda _e: self._paint("normal"))
        self.bind("<ButtonPress-1>", lambda _e: self._paint("pressed"))
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<KeyPress-space>", lambda _e: self.invoke())
        self.bind("<KeyPress-Return>", lambda _e: self.invoke())
        self.bind("<FocusIn>", lambda _e: self._paint("hover"))
        self.bind("<FocusOut>", lambda _e: self._paint("normal"))

    def _paint(self, state: str) -> None:
        if self._state == "disabled":
            state = "disabled"
        fill, ring, mark = {
            "normal": (C.SUBTLE, C.BORDER_INPUT, C.TEXT_2),
            "hover": (C.ACCENT_SOFT, C.ACCENT, mix(C.ACCENT, "#000000", 0.2)),
            "pressed": (C.ACCENT, C.ACCENT, "#ffffff"),
            "disabled": (C.BG, C.BORDER, C.DISABLED),
        }[state]
        self.itemconfigure(self._ring, fill=fill, outline=ring)
        self.itemconfigure(self._mark, fill=mark)

    def _on_release(self, event: tk.Event) -> None:
        inside = 0 <= event.x <= self._size and 0 <= event.y <= self._size
        self._paint("hover" if inside else "normal")
        if inside:
            self.invoke()

    def invoke(self) -> None:
        if self._command is not None and self._state != "disabled":
            self._command()

    def configure(self, cnf=None, **kw):  # type: ignore[override]
        if "command" in kw:
            self._command = kw.pop("command")
        if "state" in kw:
            self._state = "disabled" if str(kw.pop("state")) == "disabled" else "normal"
            self._paint("normal")
        if cnf or kw:
            return super().configure(cnf, **kw)
        return None

    config = configure


def help_button(parent: tk.Misc, command: Callable[[], object], **kw) -> HelpButton:
    return HelpButton(parent, command=command, **kw)


class Pill(tk.Canvas):
    """A rounded label: version badges, counts, states."""

    def __init__(self, parent: tk.Misc, text: str = "", fill: str = C.ACCENT_SOFT, fg: str = C.TEXT,
                 font: str = "AzbTiny", padx: int = 8, height: int = 20, bg: Optional[str] = None,
                 textvariable: Optional[tk.Variable] = None, outline: Optional[str] = None) -> None:
        self._bg = bg or widget_background(parent)
        super().__init__(parent, height=height, width=10, background=self._bg, highlightthickness=0,
                         borderwidth=0)
        self._fill, self._fg, self._font, self._padx, self._h = fill, fg, font, padx, height
        self._outline = outline
        self._var = textvariable
        self._text = text
        if textvariable is not None:
            textvariable.trace_add("write", lambda *_a: self._draw())
        self._draw()

    def set(self, text: str, fill: Optional[str] = None, fg: Optional[str] = None) -> None:
        self._text = text
        if fill:
            self._fill = fill
        if fg:
            self._fg = fg
        self._draw()

    def _draw(self) -> None:
        text = self._var.get() if self._var is not None else self._text
        self.delete("all")
        font = tkfont.nametofont(self._font) if isinstance(self._font, str) else self._font
        width = int(font.measure(text) + 2 * self._padx)
        h = self._h
        self.configure(width=width)
        r = h / 2
        outline = self._outline or self._fill
        rounded_rect(self, 0.5, 0.5, width - 0.5, h - 0.5, r - 0.5, fill=self._fill, outline=outline)
        self.create_text(width / 2, h / 2, text=text, fill=self._fg, font=self._font)


class StatusDot(tk.Canvas):
    """A small coloured dot: ok / warn / error / busy / off."""

    COLORS = {"ok": C.OK, "warn": C.WARN, "error": C.ERROR, "busy": C.INFO, "off": C.BORDER_HOVER}

    def __init__(self, parent: tk.Misc, kind: str = "off", size: int = 10, bg: Optional[str] = None) -> None:
        super().__init__(parent, width=size, height=size, background=bg or widget_background(parent),
                         highlightthickness=0, borderwidth=0)
        self._oval = self.create_oval(1, 1, size - 1, size - 1, outline="", fill=self.COLORS.get(kind, C.BORDER))

    def set(self, kind: str) -> None:
        self.itemconfigure(self._oval, fill=self.COLORS.get(kind, C.BORDER))


class StatusChip(ttk.Frame):
    """Dot plus text, for checks such as "x3dna-dssr found"."""

    def __init__(self, parent: tk.Misc, textvariable: Optional[tk.StringVar] = None, text: str = "",
                 kind: str = "off", style: str = "TFrame", label_style: str = "Status.TLabel") -> None:
        super().__init__(parent, style=style)
        self.dot = StatusDot(self, kind=kind, bg=widget_background(self))
        self.dot.pack(side="left", padx=(0, 6))
        self.label = ttk.Label(self, text=text, textvariable=textvariable, style=label_style)
        self.label.pack(side="left")

    def set(self, kind: str, text: Optional[str] = None) -> None:
        self.dot.set(kind)
        if text is not None:
            self.label.configure(text=text)


def rounded_rect(canvas: tk.Canvas, x0: float, y0: float, x1: float, y1: float, r: float, **kw) -> int:
    """A smooth rounded rectangle on a canvas (crisp at any scale)."""
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = [
        x0 + r, y0, x0 + r, y0, x1 - r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y0 + r,
        x1, y1 - r, x1, y1 - r, x1, y1, x1 - r, y1, x1 - r, y1, x0 + r, y1, x0 + r, y1,
        x0, y1, x0, y1 - r, x0, y1 - r, x0, y0 + r, x0, y0 + r, x0, y0,
    ]
    return canvas.create_polygon(pts, smooth=True, splinesteps=12, **kw)


# ---------------------------------------------------------- composite widgets
class Header(ttk.Frame):
    """Window header: icon, module name, version badge, tagline, status area.

    `right` is a frame on the right for status chips or buttons.
    """

    def __init__(self, parent: tk.Misc, module: str, title: Optional[str] = None,
                 version: Optional[str] = None, subtitle: Optional[str] = None,
                 icon: Optional[str] = None, icon_size: int = 46) -> None:
        super().__init__(parent, style="Header.TFrame", padding=(20, 14, 20, 12))
        info = MODULES.get(module, MODULES["azbmost"])
        self.columnconfigure(1, weight=1)
        self.icon = IconCanvas(self, icon or module, size=icon_size, bg=C.BG)
        self.icon.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 14))
        line = ttk.Frame(self, style="Header.TFrame")
        line.grid(row=0, column=1, sticky="sw")
        self.title_label = ttk.Label(line, text=title or str(info["name"]), style="Display.TLabel")
        self.title_label.pack(side="left")
        self.version_pill: Optional[Pill] = None
        if version:
            self.version_pill = Pill(line, text=version, fill=lab(str(info["hue"]), "T20"), fg=C.TEXT,
                                     bg=C.BG)
            self.version_pill.pack(side="left", padx=(10, 0), pady=(4, 0))
        sub = subtitle if subtitle is not None else str(info["tagline"])
        self.subtitle_label = ttk.Label(self, text=sub, style="Muted.TLabel")
        self.subtitle_label.grid(row=1, column=1, sticky="nw", pady=(1, 0))
        self.right = ttk.Frame(self, style="Header.TFrame")
        self.right.grid(row=0, column=2, rowspan=2, sticky="e")


class ToolHeader(ttk.Frame):
    """Compact header for a tool window: tool icon, name, owning module, purpose.

    Pack or grid it at the top of the window, then `hairline()` under it.
    """

    def __init__(self, parent: tk.Misc, title: str, tool: Optional[str] = None,
                 module: Optional[str] = None, subtitle: Optional[str] = None,
                 version: Optional[str] = None, icon_size: int = 38) -> None:
        super().__init__(parent, style="Header.TFrame", padding=(18, 12, 18, 10))
        owner = module or (TOOL_INFO.get(tool or "", ("azbmost", ""))[0])
        self.columnconfigure(1, weight=1)
        key = f"tool:{tool}" if tool else (owner or "azbmost")
        self.icon = IconCanvas(self, key, size=icon_size, bg=C.BG)
        self.icon.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 12))
        line = ttk.Frame(self, style="Header.TFrame")
        line.grid(row=0, column=1, sticky="sw")
        self.title_label = ttk.Label(line, text=title, style="Title.TLabel")
        self.title_label.pack(side="left")
        info = MODULES.get(owner or "", None)
        if info is not None and owner != "azbmost":
            Pill(line, text=f"{info['name']} tool", fill=lab(str(info["hue"]), "T20"), fg=C.TEXT,
                 bg=C.BG).pack(side="left", padx=(10, 0), pady=(2, 0))
        if version:
            Pill(line, text=version, fill=C.SUBTLE, fg=C.TEXT_2, outline=C.BORDER, bg=C.BG).pack(
                side="left", padx=(6, 0), pady=(2, 0))
        text = subtitle if subtitle is not None else tool_description(tool or "")
        self.subtitle_label = ttk.Label(self, text=text, style="Muted.TLabel")
        if text:
            self.subtitle_label.grid(row=1, column=1, sticky="nw", pady=(1, 0))
        self.right = ttk.Frame(self, style="Header.TFrame")
        self.right.grid(row=0, column=2, rowspan=2, sticky="e")


def hairline(parent: tk.Misc, color: str = C.BORDER, orient: str = "horizontal") -> tk.Frame:
    """A crisp 1-point divider (pack or grid it with fill/sticky)."""
    if orient == "horizontal":
        return tk.Frame(parent, height=1, background=color, highlightthickness=0, borderwidth=0)
    return tk.Frame(parent, width=1, background=color, highlightthickness=0, borderwidth=0)


def tool_window(root: tk.Misc, title: str, tool: Optional[str] = None, module: Optional[str] = None,
                subtitle: Optional[str] = None, version: Optional[str] = None,
                pack: bool = True) -> ToolHeader:
    """Theme a tool's window and put the standard header on top of it.

    Call it right after creating the Tk root (or Toplevel) and before the
    tool builds its own widgets. With pack=True the header and its hairline
    are packed at the top; a tool that grids directly on the root should use
    pack=False and grid the returned header itself.
    """
    apply_theme(root, module or TOOL_INFO.get(tool or "", ("azbmost", ""))[0])
    style_window(root)
    header = ToolHeader(root, title, tool=tool, module=module, subtitle=subtitle, version=version)
    if pack:
        header.pack(side="top", fill="x")
        hairline(root).pack(side="top", fill="x")
    return header


class Footer(ttk.Frame):
    """Bottom bar: a status message on the left, run buttons on the right."""

    def __init__(self, parent: tk.Misc, textvariable: Optional[tk.StringVar] = None, text: str = "") -> None:
        super().__init__(parent, style="Footer.TFrame", padding=(20, 10, 20, 10))
        self.columnconfigure(0, weight=1)
        self.status = ttk.Label(self, text=text, textvariable=textvariable, style="Footer.Status.TLabel",
                                anchor="w")
        self.status.grid(row=0, column=0, sticky="we", padx=(0, 12))
        self.buttons = ttk.Frame(self, style="Footer.TFrame")
        self.buttons.grid(row=0, column=1, sticky="e")

    def add_button(self, text: str, command: Optional[Callable] = None, primary: bool = False,
                   **kw) -> ttk.Button:
        button = ttk.Button(self.buttons, text=text, command=command,
                            style="Accent.TButton" if primary else "Footer.TButton", **kw)
        button.pack(side="left", padx=(8, 0))
        return button


def separator(parent: tk.Misc, orient: str = "horizontal") -> ttk.Separator:
    return ttk.Separator(parent, orient=orient)


class Card(ttk.Labelframe):
    """A section card: hairline outline, bold title, optional ? and actions.

    Children go into the card itself (it is the Labelframe), exactly as they
    went into the plain LabelFrame it replaces.
    """

    def __init__(self, parent: tk.Misc, title: str, help_command: Optional[Callable] = None,
                 subtitle: Optional[str] = None, padding=(14, 6, 14, 12), **kw) -> None:
        self.title_frame = ttk.Frame(parent)
        # text= is kept too (the title widget is what shows), so cget("text") names the card.
        super().__init__(parent, text=title, labelwidget=self.title_frame, style="Card.TLabelframe",
                         padding=padding, **kw)
        self.title_label = ttk.Label(self.title_frame, text=title, style="Section.TLabel")
        self.title_label.pack(side="left", padx=(2, 4))
        if help_command is not None:
            HelpButton(self.title_frame, command=help_command, size=18).pack(side="left", padx=(0, 4))
        if subtitle:
            ttk.Label(self.title_frame, text=subtitle, style="Hint.TLabel").pack(side="left", padx=(2, 4))

    def set_title(self, text: str) -> None:
        self.title_label.configure(text=text)


class ScrollFrame(ttk.Frame):
    """A vertically scrolling area. Put children in `.body`."""

    def __init__(self, parent: tk.Misc, bg: str = C.BG, **kw) -> None:
        super().__init__(parent, **kw)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, background=bg, highlightthickness=0, borderwidth=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.body = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self._shown = False
        self._pending = False
        self.body.bind("<Configure>", lambda _e: self.refresh())
        self.canvas.bind("<Configure>", self._on_canvas)
        # One application-wide wheel binding per interpreter; it scrolls the
        # scroll area under the pointer (children included).
        top = self.winfo_toplevel()
        root = top._root() if hasattr(top, "_root") else top
        if not getattr(root, "_azbmost_wheel", False):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                root.bind_all(sequence, ScrollFrame._dispatch_wheel, add="+")
            root._azbmost_wheel = True

    def _on_canvas(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window, width=max(1, event.width))
        self.refresh()

    def refresh(self) -> None:
        if self._pending:
            return
        self._pending = True
        self.after_idle(self._do_refresh)

    def _do_refresh(self) -> None:
        self._pending = False
        try:
            self.update_idletasks()
        except tk.TclError:
            return
        height = self.body.winfo_reqheight()
        width = max(self.canvas.winfo_width(), self.body.winfo_reqwidth())
        self.canvas.configure(scrollregion=(0, 0, width, height))
        need = height > self.canvas.winfo_height() + 1
        if need and not self._shown:
            self.scrollbar.grid(row=0, column=1, sticky="ns")
            self._shown = True
        elif not need and self._shown:
            self.scrollbar.grid_remove()
            self.canvas.yview_moveto(0)
            self._shown = False

    @staticmethod
    def _dispatch_wheel(event: tk.Event) -> None:
        widget = event.widget
        try:
            target = widget.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return
        area = target
        while area is not None and not isinstance(area, ScrollFrame):
            area = getattr(area, "master", None)
        if area is not None:
            area._on_wheel(event, target)

    def _on_wheel(self, event: tk.Event, target=None) -> None:
        if not self._shown or target is None:
            return
        # Let text boxes, lists and tables that can scroll do so themselves.
        try:
            if target.winfo_class() in ("Text", "Listbox", "Treeview"):
                first, last = target.yview()
                if first > 0.0 or last < 1.0:
                    return
        except Exception:
            pass
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            d = event.delta
            if sys.platform == "darwin":
                delta = -d
            else:
                delta = -int(d / 120) if abs(d) >= 120 else (-1 if d > 0 else 1)
        if delta:
            self.canvas.yview_scroll(int(delta), "units")


class LogView(ttk.Frame):
    """The run log: a quiet monospace text box with thin scrollbars.

    `.text` is the tk.Text widget; use it exactly like the old log widget.
    """

    def __init__(self, parent: tk.Misc, wrap: str = "word", height: int = 12, horizontal: bool = False,
                 **kw) -> None:
        super().__init__(parent, **kw)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.text = tk.Text(self, wrap=wrap, height=height, background=C.LOG_BG, foreground=C.TEXT,
                            font="AzbMono", relief="flat", borderwidth=0, highlightthickness=1,
                            highlightbackground=C.BORDER, highlightcolor=C.BORDER, padx=10, pady=8,
                            insertbackground=C.TEXT, selectbackground=C.ACCENT_SOFT_2, spacing1=1,
                            spacing3=1, undo=False)
        self.text.grid(row=0, column=0, sticky="nsew")
        self.yscroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.yscroll.grid(row=0, column=1, sticky="ns", padx=(2, 0))
        self.text.configure(yscrollcommand=self.yscroll.set)
        self.xscroll: Optional[ttk.Scrollbar] = None
        if horizontal:
            self.xscroll = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
            self.xscroll.grid(row=1, column=0, sticky="we", pady=(2, 0))
            self.text.configure(xscrollcommand=self.xscroll.set)


class Segmented(tk.Frame):
    """Radio buttons drawn as one segmented control (same variable, same command)."""

    def __init__(self, parent: tk.Misc, variable: tk.Variable, values: Sequence[str],
                 command: Optional[Callable] = None, labels: Optional[Sequence[str]] = None) -> None:
        super().__init__(parent, background=C.BORDER_INPUT, highlightthickness=0, borderwidth=0)
        self.buttons: List[ttk.Radiobutton] = []
        for idx, value in enumerate(values):
            text = labels[idx] if labels else value
            button = ttk.Radiobutton(self, text=text, value=value, variable=variable, command=command,
                                     style="Segment.TRadiobutton", takefocus=1)
            button.grid(row=0, column=idx, sticky="nsew", padx=(1 if idx == 0 else 0, 1), pady=1)
            self.buttons.append(button)


# ----------------------------------------------------------------- tools list
class ToolRow(tk.Frame):
    """One entry of an "Other tools" list: icon, name, description, ?."""

    def __init__(self, parent: tk.Misc, label: str, command: Optional[Callable] = None,
                 description: str = "", icon: Optional[str] = None, help_command: Optional[Callable] = None,
                 hue: str = "blue", bg: str = C.BG) -> None:
        # Widget class "ToolButton", so code and tests can find tool entries.
        super().__init__(parent, class_="ToolButton", background=bg, highlightthickness=0, borderwidth=0,
                         cursor="hand2", takefocus=1)
        self._bg = bg
        self._command = command
        self._label = label
        self._enabled = True
        self.columnconfigure(1, weight=1)
        self.icon = IconCanvas(self, icon or "tool:generic", size=28, bg=bg, hue=hue)
        self.icon.grid(row=0, column=0, rowspan=2, sticky="w", padx=(8, 10), pady=6)
        self.name = tk.Label(self, text=label, background=bg, foreground=C.TEXT, font="AzbBodyBold",
                             anchor="w", borderwidth=0)
        self.name.grid(row=0, column=1, sticky="sw", pady=(6, 0))
        self.desc = tk.Label(self, text=description, background=bg, foreground=C.MUTED, font="AzbSmall",
                             anchor="w", justify="left", borderwidth=0)
        self.desc.grid(row=1, column=1, sticky="nw", pady=(0, 6))
        if not description:
            self.desc.grid_remove()
            self.name.grid_configure(rowspan=2, sticky="w", pady=6)
        self.help: Optional[HelpButton] = None
        if help_command is not None:
            self.help = HelpButton(self, command=help_command, size=18, bg=bg)
            self.help.grid(row=0, column=2, rowspan=2, sticky="e", padx=(6, 8))
        for widget in (self, self.icon, self.name, self.desc):
            widget.bind("<Enter>", self._hover_on, add="+")
            widget.bind("<Leave>", self._hover_off, add="+")
            widget.bind("<ButtonRelease-1>", self._click, add="+")
        self.bind("<KeyPress-space>", lambda _e: self.invoke())
        self.bind("<KeyPress-Return>", lambda _e: self.invoke())
        self.bind("<FocusIn>", self._hover_on)
        self.bind("<FocusOut>", self._hover_off)
        self.bind("<Configure>", self._wrap)

    def _wrap(self, event: tk.Event) -> None:
        width = max(80, event.width - 28 - 18 - 50)
        self.desc.configure(wraplength=width)

    def _paint(self, bg: str) -> None:
        for widget in (self, self.name, self.desc):
            widget.configure(background=bg)
        self.icon.configure(background=bg)
        if self.help is not None:
            self.help.configure(background=bg)

    def _hover_on(self, _event=None) -> None:
        if self._enabled:
            self._paint(C.SUBTLE_HOVER)

    def _hover_off(self, event=None) -> None:
        if event is not None and event.type == tk.EventType.Leave:
            x, y = self.winfo_pointerxy()
            if self.winfo_containing(x, y) in (self, self.icon, self.name, self.desc):
                return
        self._paint(self._bg)

    def _click(self, event: tk.Event) -> None:
        x, y = self.winfo_pointerxy()
        if self.winfo_containing(x, y) in (self, self.icon, self.name, self.desc):
            self.invoke()

    def invoke(self) -> None:
        if self._enabled and self._command is not None:
            self._command()

    # Keep the ttk.Button calls that code makes on its tool buttons working.
    def configure(self, cnf=None, **kw):  # type: ignore[override]
        if "command" in kw:
            self._command = kw.pop("command")
        if "text" in kw:
            self.name.configure(text=kw.pop("text"))
        if "state" in kw:
            self._enabled = str(kw.pop("state")) != "disabled"
            self.name.configure(foreground=C.TEXT if self._enabled else C.DISABLED)
        if cnf or kw:
            return super().configure(cnf, **kw)
        return None

    config = configure

    def cget(self, key):  # type: ignore[override]
        if key == "text":
            return self.name.cget("text")
        if key == "state":
            return "normal" if self._enabled else "disabled"
        if key == "command":
            return self._command
        return super().cget(key)

    def state(self, statespec=None):  # ttk-style state(["disabled"]) / state(["!disabled"])
        if statespec:
            for flag in statespec:
                if flag == "disabled":
                    self.configure(state="disabled")
                elif flag == "!disabled":
                    self.configure(state="normal")
        return () if self._enabled else ("disabled",)


class ToolList(ttk.Frame):
    """A titled list of ToolRows separated by hairlines."""

    def __init__(self, parent: tk.Misc, title: Optional[str] = None, hue: str = "blue",
                 columns: int = 1, **kw) -> None:
        super().__init__(parent, **kw)
        self._hue = hue
        self._columns = max(1, columns)
        self._count = 0
        for col in range(self._columns):
            self.columnconfigure(col, weight=1, uniform="tools")
        if title:
            ttk.Label(self, text=title.upper(), style="Hint.TLabel", font="AzbTiny").grid(
                row=0, column=0, columnspan=self._columns, sticky="w", padx=8, pady=(4, 2))

    def add(self, label: str, command: Optional[Callable] = None, description: str = "",
            icon: Optional[str] = None, help_command: Optional[Callable] = None) -> ToolRow:
        row = ToolRow(self, label, command=command, description=description, icon=icon,
                      help_command=help_command, hue=self._hue)
        r, c = divmod(self._count, self._columns)
        row.grid(row=1 + r, column=c, sticky="nsew", padx=0, pady=0)
        self._count += 1
        return row


# ------------------------------------------------------- classic-widget stand-ins
# Themed ttk widgets that accept (and quietly drop) the classic-only options
# such as bg=, relief= or padx=, so a tk.Button(...) call can become a
# Button(...) call without touching anything else it does.
_CLASSIC_ONLY = {
    "bg", "background", "fg", "foreground", "activebackground", "activeforeground", "relief", "bd",
    "borderwidth", "highlightbackground", "highlightcolor", "highlightthickness", "padx", "pady",
    "selectcolor", "indicatoron", "disabledforeground", "overrelief", "offrelief", "anchor",
    "justify", "wraplength", "font", "insertbackground", "selectbackground", "selectforeground",
    "disabledbackground", "readonlybackground", "insertwidth", "selectborderwidth", "buttonbackground",
    "repeatdelay", "repeatinterval", "height", "bitmap",
}
_KEEP: Dict[type, set] = {}


def _ttk_options(cls: type) -> set:
    if cls not in _KEEP:
        probe_root = tk._get_default_root() if hasattr(tk, "_get_default_root") else tk._default_root
        names: set = set()
        try:
            probe = cls(probe_root)
            names = set(probe.keys())
            probe.destroy()
        except Exception:
            names = set()
        _KEEP[cls] = names
    return _KEEP[cls]


def _filter(cls: type, kw: dict) -> dict:
    known = _ttk_options(cls)
    out = {}
    for key, value in kw.items():
        if key in ("command", "variable", "textvariable", "text", "value", "onvalue", "offvalue", "state",
                   "width", "style", "takefocus", "cursor", "image", "compound", "padding", "underline",
                   "validate", "validatecommand", "vcmd", "invalidcommand", "show", "exportselection",
                   "xscrollcommand", "from_", "to", "increment", "values", "format", "wrap", "name",
                   "labelanchor", "labelwidget", "class_", "default", "orient", "length", "mode",
                   "maximum", "postcommand"):
            if key == "vcmd":
                out["validatecommand"] = value
            else:
                out[key] = value
        elif key in known and key not in _CLASSIC_ONLY:
            out[key] = value
        elif key in ("font", "justify", "foreground", "background") and key in known:
            out[key] = value
    return out


class _CompatMixin:
    _ttk_class: type = ttk.Widget

    def configure(self, cnf=None, **kw):  # type: ignore[override]
        if isinstance(cnf, dict):
            kw = {**cnf, **kw}
            cnf = None
        if not kw:
            return super().configure(cnf)  # type: ignore[misc]
        kw = _filter(self._ttk_class, kw)
        if not kw:
            return None
        return super().configure(**kw)  # type: ignore[misc]

    config = configure

    def cget(self, key):  # type: ignore[override]
        try:
            value = super().cget(key)  # type: ignore[misc]
        except tk.TclError:
            if key in ("bg", "background"):
                return widget_background(self)  # type: ignore[arg-type]
            raise
        return str(value) if key == "state" else value

    def __setitem__(self, key, value):
        self.configure(**{key: value})


class Button(_CompatMixin, ttk.Button):
    _ttk_class = ttk.Button

    def __init__(self, master=None, cnf=None, **kw):
        kw = {**(cnf or {}), **kw}
        primary = kw.pop("primary", False)
        kw = _filter(ttk.Button, kw)
        if primary and "style" not in kw:
            kw["style"] = "Accent.TButton"
        ttk.Button.__init__(self, master, **kw)


def _wrap_style(master, base: str, wraplength) -> Optional[str]:
    """A derived style that wraps the label at `wraplength` (ttk check and
    radio buttons have no wraplength option of their own)."""
    try:
        pixels = int(master.winfo_pixels(wraplength))
    except Exception:
        return None
    if pixels <= 0:
        return None
    name = f"Wrap{pixels}.{base}"
    try:
        ttk.Style(master).configure(name, wraplength=pixels, justify="left")
    except Exception:
        return None
    return name


class Checkbutton(_CompatMixin, ttk.Checkbutton):
    _ttk_class = ttk.Checkbutton

    def __init__(self, master=None, cnf=None, **kw):
        kw = {**(cnf or {}), **kw}
        wrap = kw.pop("wraplength", None)
        kw = _filter(ttk.Checkbutton, kw)
        if wrap and "style" not in kw:
            style = _wrap_style(master, "TCheckbutton", wrap)
            if style:
                kw["style"] = style
        ttk.Checkbutton.__init__(self, master, **kw)

    def select(self):
        var = self.cget("variable")
        if var:
            self.setvar(str(var), self.cget("onvalue"))

    def deselect(self):
        var = self.cget("variable")
        if var:
            self.setvar(str(var), self.cget("offvalue"))

    def toggle(self):
        self.invoke()


class Radiobutton(_CompatMixin, ttk.Radiobutton):
    _ttk_class = ttk.Radiobutton

    def __init__(self, master=None, cnf=None, **kw):
        kw = {**(cnf or {}), **kw}
        wrap = kw.pop("wraplength", None)
        kw = _filter(ttk.Radiobutton, kw)
        if wrap and "style" not in kw:
            style = _wrap_style(master, "TRadiobutton", wrap)
            if style:
                kw["style"] = style
        ttk.Radiobutton.__init__(self, master, **kw)

    def select(self):
        var = self.cget("variable")
        if var:
            self.setvar(str(var), self.cget("value"))


class Entry(_CompatMixin, ttk.Entry):
    _ttk_class = ttk.Entry

    def __init__(self, master=None, cnf=None, **kw):
        kw = _filter(ttk.Entry, {**(cnf or {}), **kw})
        ttk.Entry.__init__(self, master, **kw)


class Spinbox(_CompatMixin, ttk.Spinbox):
    _ttk_class = ttk.Spinbox

    def __init__(self, master=None, cnf=None, **kw):
        kw = _filter(ttk.Spinbox, {**(cnf or {}), **kw})
        ttk.Spinbox.__init__(self, master, **kw)


class LabelFrame(_CompatMixin, ttk.Labelframe):
    """tk.LabelFrame stand-in drawn as a card. padx/pady become padding."""

    _ttk_class = ttk.Labelframe

    def __init__(self, master=None, cnf=None, **kw):
        kw = {**(cnf or {}), **kw}
        padx = kw.pop("padx", 12)
        pady = kw.pop("pady", 8)
        kw = _filter(ttk.Labelframe, kw)
        kw.setdefault("padding", (max(int(padx), 12), max(int(pady), 6), max(int(padx), 12), max(int(pady), 10)))
        kw.setdefault("style", "Card.TLabelframe")
        ttk.Labelframe.__init__(self, master, **kw)


class OptionMenu(ttk.OptionMenu):
    """tk.OptionMenu(parent, var, *values, command=...) with the themed look."""

    def __init__(self, master, variable, *values, **kw):
        command = kw.pop("command", None)
        width = kw.pop("width", None)
        current = variable.get()
        ttk.OptionMenu.__init__(self, master, variable, current, *values, command=command)
        if width is not None:
            self.configure(width=width)

    def configure(self, cnf=None, **kw):  # type: ignore[override]
        if isinstance(cnf, dict):
            kw = {**cnf, **kw}
            cnf = None
        if not kw:
            return super().configure(cnf)
        kw = {k: v for k, v in kw.items() if k in ("state", "width", "text", "textvariable", "style",
                                                      "menu", "direction", "cursor", "takefocus")}
        if not kw:
            return None
        return super().configure(**kw)

    config = configure

    def cget(self, key):  # type: ignore[override]
        value = super().cget(key)
        return str(value) if key == "state" else value


class IconCanvas(tk.Canvas):
    """A crisp vector icon at any size: an app icon ("bnp_na") or a tool
    glyph ("tool:bend_helix"), drawn by azbmost_icons."""

    def __init__(self, parent: tk.Misc, key: str, size: int = 32, bg: Optional[str] = None,
                 hue: Optional[str] = None) -> None:
        super().__init__(parent, width=size, height=size, background=bg or widget_background(parent),
                         highlightthickness=0, borderwidth=0)
        self.key = key
        if _icons is not None:
            try:
                _icons.draw(_icons.CanvasPainter(self, size), key)
                return
            except Exception:
                self.delete("all")
        rounded_rect(self, 1, 1, size - 1, size - 1, size * 0.22, fill=lab(hue or module_hue(key), "T100"),
                     outline="")


# ------------------------------------------------------------ icon files
def window_icon(window: tk.Misc, paths: Iterable) -> Optional[tk.PhotoImage]:
    """Set the window/dock icon from the first PNG that loads."""
    for path in paths:
        try:
            image = tk.PhotoImage(master=window, file=str(path))
        except Exception:
            continue
        try:
            window.iconphoto(True, image)
        except Exception:
            pass
        try:
            window._azbmost_icon = image
        except Exception:
            pass
        return image
    return None


# ------------------------------------------------------------------- demo
def _demo() -> None:
    root = tk.Tk()
    root.title("AZBMOST UI demo")
    apply_theme(root, "bnp_na")
    root.geometry("1180x820")
    head = Header(root, "bnp_na", version="V13.21")
    head.pack(fill="x")
    StatusChip(head.right, text="x3dna-dssr found", kind="ok", style="Header.TFrame").pack(anchor="e")
    hairline(root).pack(fill="x")
    th = ToolHeader(root, "Bend Helix", tool="bend_helix", version="V2.9")
    th.pack(fill="x")
    hairline(root).pack(fill="x")
    body = ttk.Frame(root, padding=(20, 14))
    body.pack(fill="both", expand=True)
    body.columnconfigure(0, weight=3)
    body.columnconfigure(1, weight=2)
    left = ttk.Frame(body)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    card = Card(left, "Helix", help_command=lambda: None)
    card.pack(fill="x", pady=(0, 12))
    card.columnconfigure(1, weight=1)
    var = tk.StringVar(value="B-DNA")
    ttk.Label(card, text="Type", style="Field.TLabel").grid(row=0, column=0, sticky="w", pady=4)
    Segmented(card, var, ["B-DNA", "A-DNA", "A-RNA", "Z-DNA"]).grid(row=0, column=1, sticky="w", pady=4)
    ttk.Label(card, text="Sequence (5'->3')", style="Field.TLabel").grid(row=1, column=0, sticky="w", pady=4, padx=(0, 12))
    ttk.Entry(card).grid(row=1, column=1, sticky="we", pady=4)
    ttk.Label(card, text="Output folder", style="Field.TLabel").grid(row=2, column=0, sticky="w", pady=4)
    row = ttk.Frame(card)
    row.grid(row=2, column=1, sticky="we")
    row.columnconfigure(0, weight=1)
    e = ttk.Entry(row)
    e.insert(0, "/Users/me/output")
    e.grid(row=0, column=0, sticky="we")
    ttk.Button(row, text="Browse…").grid(row=0, column=1, padx=(8, 0))
    HelpButton(row, command=lambda: None).grid(row=0, column=2, padx=(8, 0))
    c2 = Card(left, "Options")
    c2.pack(fill="x", pady=(0, 12))
    bv = tk.BooleanVar(value=True)
    ttk.Checkbutton(c2, text="Run phenix.geometry_minimization", variable=bv).grid(row=0, column=0, sticky="w")
    ttk.Checkbutton(c2, text="Delete hydrogens").grid(row=1, column=0, sticky="w")
    rv = tk.StringVar(value="a")
    ttk.Radiobutton(c2, text="Radio A", value="a", variable=rv).grid(row=0, column=1, sticky="w", padx=20)
    ttk.Radiobutton(c2, text="Radio B", value="b", variable=rv).grid(row=1, column=1, sticky="w", padx=20)
    ttk.Combobox(c2, values=["oyz", "oxz"], state="readonly", width=10).grid(row=0, column=2, padx=10)
    ttk.Spinbox(c2, from_=0, to=10, width=6).grid(row=1, column=2, padx=10)
    dis = ttk.Entry(c2, width=10)
    dis.insert(0, "disabled")
    dis.state(["disabled"])
    dis.grid(row=0, column=3)
    ro = ttk.Entry(c2, width=10)
    ro.insert(0, "read-only")
    ro.state(["readonly"])
    ro.grid(row=1, column=3)
    tv = ttk.Treeview(left, columns=("a", "b", "c"), show="headings", height=3)
    for col in ("a", "b", "c"):
        tv.heading(col, text=col.upper())
    for i in range(3):
        tv.insert("", "end", values=(i, i * 2, i * 3))
    tv.pack(fill="x")
    right = ttk.Frame(body)
    right.grid(row=0, column=1, sticky="nsew")
    tools = ToolList(right, title="Other tools", hue="blue")
    tools.pack(fill="x")
    for key, label in (("measure_angle", "Measure angle around axis"), ("align_z", "Align helix to z"),
                       ("combine_pdb", "Combine_PDB")):
        tools.add(label, command=lambda: None, description="One-line description of the tool.",
                  icon="tool:" + key, help_command=lambda: None)
    log = LogView(right, height=8)
    log.pack(fill="both", expand=True, pady=(12, 0))
    log.text.insert("end", "=== bnp_na startup ===\nApp folder: /Users/me/bnp_na\n")
    foot = Footer(root, text="Ready.")
    foot.pack(fill="x", side="bottom")
    foot.add_button("Quit")
    foot.add_button("Generate", primary=True)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--demo":
        _demo()
    else:
        print(__doc__)
