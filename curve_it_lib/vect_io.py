#!/usr/bin/env python3
"""vect_io.py

Geomview VECT reading and writing, shared by every Curve It tool.

VECT is the polyline format Geomview reads and that the knot-theory tools
around it write: ridgerunner's ropelength minimizations, Geomview itself, and
anything that speaks its OOGL family.  It is the one curve format in common
use that states outright whether each component is a closed loop, which is
information a plain XYZ file cannot carry and which every tool here otherwise
has to guess from the gap between the last point and the first.

Layout::

    VECT
    NPolylines NVertices NColors
    Nv[0] ... Nv[NPolylines-1]        vertex count per polyline
    Nc[0] ... Nc[NPolylines-1]        colour count per polyline
    x y z                             NVertices of these, components in order
    ...
    r g b a                           NColors of these, components in order

Three properties of real files drive the parser below, and all three were
measured against the 396 files ridgerunner leaves in ~/.local/share/ridgerunner
rather than assumed:

- **A negative vertex count means the polyline is closed**, and its magnitude
  is the number of vertices.  Every closed knot in that corpus is negative
  (`7.7.vect` is ``-91``); the one file holding open arcs clamped between
  planes, `simple_clasp.vect`, is positive (``33 38``).
- **A closed component does not repeat its first vertex.**  Measured over that
  corpus the gap from last vertex back to first is 0.83 to 1.02 of the median
  step, i.e. one ordinary segment.  This is the same invariant Curve It's own
  curve files already hold, so geometry crosses between the two formats
  without a dedupe pass.
- **The file is a token stream, not a line-oriented format.**  `#` starts a
  comment that runs to end of line and may sit anywhere, including trailing a
  coordinate row (`simple_clasp.vect` marks constrained vertices that way),
  and a count list may wrap across lines.  So the whole file is stripped of
  comments and split on whitespace before anything is interpreted.

Colours are read and reported but do not participate in geometry.  They
survive nothing that passes through XYZ, which has nowhere to put them, so
`write_vect` gives each component one opaque white colour unless a caller
supplies something else through `parse_color_spec`.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__version__ = "1.0"

FORMAT_NAME = "VECT"

# Matches the other writers in this package (generate_sc_xyz, generate_helix_xyz).
DEFAULT_PRECISION = 8

# Opaque white: visible in Geomview against its default background, and a
# neutral placeholder that does not pretend to mean anything about the curve.
DEFAULT_COLOR: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

# Enough names to spell a link's components apart without pulling in a colour
# library.  Anything outside this can still be given as numbers or as hex.
NAMED_COLORS = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 0.5, 0.0),
    "lime": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "orange": (1.0, 0.65, 0.0),
    "purple": (0.5, 0.0, 0.5),
    "pink": (1.0, 0.75, 0.8),
    "brown": (0.55, 0.27, 0.07),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
}

ColorSpec = Union[None, Sequence[float], np.ndarray]


class VectError(ValueError):
    """Raised when a file announces itself as VECT but cannot be read as one.

    It subclasses ValueError so that callers which already funnel read errors
    through `except ValueError` keep working without knowing this type exists.
    """


class VectCurves:
    """Geometry read from a VECT file.

    `components` is one (N,3) float array per polyline in file order, so it
    drops straight into the component machinery the tools already have.
    `closed[i]` is the file's own statement about component i rather than a
    guess, and `colors[i]` is an (M,4) array with M of 0, 1, or N.
    """

    __slots__ = ("components", "closed", "colors")

    def __init__(self,
                 components: List[np.ndarray],
                 closed: List[bool],
                 colors: List[np.ndarray]) -> None:
        self.components = components
        self.closed = closed
        self.colors = colors

    def __len__(self) -> int:
        return len(self.components)

    @property
    def points(self) -> np.ndarray:
        """Every component concatenated in file order, as one (N,3) array."""
        if not self.components:
            raise VectError("VECT file holds no components.")
        return np.vstack(self.components)

    @property
    def all_closed(self) -> bool:
        """True when the file says every component is a closed loop."""
        return bool(self.closed) and all(self.closed)

    @property
    def any_closed(self) -> bool:
        return any(self.closed)

    def n_colors(self) -> int:
        return int(sum(int(c.shape[0]) for c in self.colors))

    def summary(self) -> str:
        """A one-line description for tool reports and GUI status lines."""
        parts = []
        for index, pts in enumerate(self.components):
            parts.append("%s:%d%s" % (component_label(index), pts.shape[0],
                                      " closed" if self.closed[index] else " open"))
        text = "%d component(s), %d point(s): %s" % (
            len(self.components),
            int(sum(p.shape[0] for p in self.components)),
            ", ".join(parts))
        n_colors = self.n_colors()
        if n_colors:
            text += "; %d colour(s)" % n_colors
        return text


def component_label(index: int) -> str:
    """Spreadsheet-style labels A, B, ..., Z, AA, AB, ... as used everywhere here."""
    if index < 0:
        raise ValueError("Component index must be non-negative.")
    label = ""
    n = index
    while True:
        label = chr(ord("A") + (n % 26)) + label
        n = n // 26 - 1
        if n < 0:
            break
    return label


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _tokens(text: str) -> List[Tuple[str, int]]:
    """Split the whole file into (token, line number), dropping `#` comments.

    Comments are stripped before splitting rather than by skipping whole lines,
    because real files trail them on coordinate rows: simple_clasp.vect writes
    `-6.98 12.31 5.01 # Cst: 1` to mark a vertex pinned to a constraint plane.
    """
    out: List[Tuple[str, int]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        hash_at = line.find("#")
        if hash_at >= 0:
            line = line[:hash_at]
        for token in line.split():
            out.append((token, lineno))
    return out


def _magic(text: str) -> Optional[str]:
    """The first non-comment token, or None for an entirely empty file."""
    for token, _lineno in _tokens(text):
        return token
    return None


def _is_vect_variant(token: str) -> bool:
    """True for a Geomview VECT variant such as 4VECT or nVECT.

    Geomview builds these names by prefixing the base word, so the test is on
    the prefix: only the letters Geomview uses for that (C for colour, N for
    normals, n for an arbitrary dimension) and digits may appear there.  A
    word that merely ends in VECT, such as NOTVECT, is not a variant and is
    reported as "not a VECT file" instead.
    """
    upper = token.upper()
    if not upper.endswith("VECT") or upper == "VECT":
        return False
    prefix = token[:-4]
    return bool(prefix) and all(ch in "CN4n" or ch.isdigit() for ch in prefix)


def looks_like_vect(text: str) -> bool:
    """True when this text is a VECT file.

    Detection is by the magic word rather than by filename, matching how every
    `auto` mode in this package already sniffs content: a VECT file named
    `.txt` is still a VECT file.  The check is deliberately narrow, because a
    false positive here would take a readable XYZ file down a parser that then
    rejects it.
    """
    token = _magic(text)
    return token is not None and token.upper() == "VECT"


def looks_like_vect_file(path: str) -> bool:
    """As `looks_like_vect`, reading only enough of the file to decide."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    return looks_like_vect(head)


class _Cursor:
    """A position in the token stream that reports where it ran into trouble."""

    def __init__(self, tokens: List[Tuple[str, int]], source: Optional[str]) -> None:
        self.tokens = tokens
        self.index = 0
        self.source = source

    def _where(self) -> str:
        if self.index < len(self.tokens):
            return " at line %d" % self.tokens[self.index][1]
        if self.tokens:
            return " at end of file (line %d)" % self.tokens[-1][1]
        return ""

    def fail(self, message: str) -> "VectError":
        where = self._where()
        if self.source:
            return VectError("%s%s in %s" % (message, where, self.source))
        return VectError("%s%s" % (message, where))

    def take(self, what: str) -> str:
        if self.index >= len(self.tokens):
            raise self.fail("VECT file ended before %s could be read" % what)
        token = self.tokens[self.index][0]
        self.index += 1
        return token

    def take_int(self, what: str) -> int:
        token = self.take(what)
        try:
            return int(token)
        except ValueError:
            self.index -= 1
            raise self.fail("VECT %s must be a whole number, found %r" % (what, token))

    def take_float(self, what: str) -> float:
        token = self.take(what)
        try:
            return float(token)
        except ValueError:
            self.index -= 1
            raise self.fail("VECT %s must be a number, found %r" % (what, token))


def read_vect_text(text: str, source: Optional[str] = None) -> VectCurves:
    """Parse VECT text into components, per-component closure, and colours."""
    tokens = _tokens(text)
    cursor = _Cursor(tokens, source)

    if not tokens:
        raise cursor.fail("file holds no readable tokens")

    magic = cursor.take("the VECT header word")
    if magic.upper() != "VECT":
        if _is_vect_variant(magic):
            # 4VECT and nVECT carry homogeneous or N-dimensional vertices, so
            # their coordinate rows are not 3 wide and cannot be read here.
            raise VectError(
                "%r is a %s file, a Geomview variant this reader does not "
                "handle; only plain 3D VECT is supported."
                % (source or "input", magic))
        raise VectError(
            "%r is not a VECT file: it begins with %r, not VECT."
            % (source or "input", magic))

    n_polylines = cursor.take_int("polyline count")
    n_vertices = cursor.take_int("vertex count")
    n_colors = cursor.take_int("colour count")

    if n_polylines <= 0:
        raise VectError("VECT file declares %d polylines; at least one is needed."
                        % n_polylines)
    if n_vertices < 0 or n_colors < 0:
        raise VectError("VECT file declares a negative vertex or colour total "
                        "(%d vertices, %d colours)." % (n_vertices, n_colors))

    raw_counts = [cursor.take_int("vertex count for component %s" % component_label(i))
                  for i in range(n_polylines)]
    color_counts = [cursor.take_int("colour count for component %s" % component_label(i))
                    for i in range(n_polylines)]

    # A negative count is the file saying "closed"; the magnitude is the count.
    closed = [count < 0 for count in raw_counts]
    counts = [abs(count) for count in raw_counts]

    for index, count in enumerate(counts):
        if count == 0:
            raise VectError(
                "VECT component %s has no vertices. Dropping it would relabel "
                "every component after it, so the file is rejected instead."
                % component_label(index))
    for index, count in enumerate(color_counts):
        if count < 0:
            raise VectError("VECT component %s declares %d colours."
                            % (component_label(index), count))
        if count not in (0, 1, counts[index]):
            raise VectError(
                "VECT component %s declares %d colours for %d vertices; VECT "
                "allows 0 (inherit), 1 (whole component), or one per vertex."
                % (component_label(index), count, counts[index]))

    if sum(counts) != n_vertices:
        raise VectError(
            "VECT header declares %d vertices but the per-component counts sum "
            "to %d." % (n_vertices, sum(counts)))
    if sum(color_counts) != n_colors:
        raise VectError(
            "VECT header declares %d colours but the per-component counts sum "
            "to %d." % (n_colors, sum(color_counts)))

    components: List[np.ndarray] = []
    for index, count in enumerate(counts):
        label = component_label(index)
        flat = [cursor.take_float("coordinate %d of component %s" % (k + 1, label))
                for k in range(3 * count)]
        components.append(np.asarray(flat, dtype=float).reshape(count, 3))

    colors: List[np.ndarray] = []
    for index, count in enumerate(color_counts):
        label = component_label(index)
        flat = [cursor.take_float("colour channel %d of component %s" % (k + 1, label))
                for k in range(4 * count)]
        colors.append(np.asarray(flat, dtype=float).reshape(count, 4))

    for index, pts in enumerate(components):
        if not np.all(np.isfinite(pts)):
            raise VectError("VECT component %s holds a non-finite coordinate."
                            % component_label(index))

    # Anything past the colours is Geomview trailer material (appearances and
    # the like) that says nothing about this geometry, so it is left alone.
    return VectCurves(components, closed, colors)


def read_vect(path: str) -> VectCurves:
    """Read a VECT file from disk."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return read_vect_text(handle.read(), source=path)


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
def _parse_one_color(entry: str) -> Tuple[float, float, float, float]:
    """Turn one colour entry into RGBA in 0-1."""
    text = entry.strip()
    if not text:
        raise ValueError("empty colour")

    lowered = text.lower()
    if lowered in NAMED_COLORS:
        r, g, b = NAMED_COLORS[lowered]
        return (r, g, b, 1.0)

    if text.startswith("#"):
        digits = text[1:]
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        if len(digits) not in (6, 8):
            raise ValueError("hex colour %r must be #rgb, #rrggbb or #rrggbbaa" % text)
        try:
            values = [int(digits[i:i + 2], 16) / 255.0 for i in range(0, len(digits), 2)]
        except ValueError:
            raise ValueError("hex colour %r holds a non-hex digit" % text)
        if len(values) == 3:
            values.append(1.0)
        return (values[0], values[1], values[2], values[3])

    parts = text.replace(",", " ").split()
    if len(parts) not in (3, 4):
        raise ValueError(
            "colour %r must be a name, a #hex value, or 3 or 4 numbers" % text)
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError("colour %r holds a non-numeric channel" % text)
    # 0-255 is the other convention people reach for, and no 0-1 colour has a
    # channel above 1, so the two cannot be confused.
    if any(v > 1.0 for v in values):
        values = [v / 255.0 for v in values]
    if len(values) == 3:
        values.append(1.0)
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise ValueError("colour %r has a channel outside 0-1" % text)
    return (values[0], values[1], values[2], values[3])


def parse_color_spec(spec: Optional[str], n_components: int) -> List[np.ndarray]:
    """Parse a colour string into one (1,4) RGBA array per component.

    Components are separated by commas and channels within a component by
    whitespace, so ``"1 0 0 1, 0 0 1 1"`` is red then blue.  Names and hex
    values hold no whitespace, so ``"red,blue"`` works too.  Fewer entries than
    components cycle, which is what makes ``--color red`` colour a whole link.

    An empty or missing spec returns one opaque white per component.
    """
    if n_components <= 0:
        raise ValueError("There are no components to colour.")
    text = (spec or "").strip()
    if not text:
        return [np.asarray([DEFAULT_COLOR], dtype=float) for _ in range(n_components)]

    entries = [chunk for chunk in text.split(",") if chunk.strip()]
    if not entries:
        raise ValueError("No colours could be read from %r." % spec)

    parsed: List[Tuple[float, float, float, float]] = []
    for entry in entries:
        try:
            parsed.append(_parse_one_color(entry))
        except ValueError as exc:
            raise ValueError("%s (in --color %r)" % (exc, spec))

    return [np.asarray([parsed[i % len(parsed)]], dtype=float)
            for i in range(n_components)]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _normalize_components(components: Iterable[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for index, block in enumerate(components):
        pts = np.asarray(block, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Component %s must be an N x 3 array, got shape %s."
                             % (component_label(index), (pts.shape,)))
        if pts.shape[0] < 2:
            raise ValueError("Component %s has %d point(s); at least 2 are needed."
                             % (component_label(index), pts.shape[0]))
        if not np.all(np.isfinite(pts)):
            raise ValueError("Component %s holds a non-finite coordinate."
                             % component_label(index))
        out.append(pts)
    if not out:
        raise ValueError("There are no components to write.")
    return out


def _normalize_closed(closed: Union[None, bool, Sequence[bool]],
                      n_components: int) -> List[bool]:
    if closed is None:
        return [False] * n_components
    if isinstance(closed, (bool, np.bool_)):
        return [bool(closed)] * n_components
    flags = [bool(flag) for flag in closed]
    if len(flags) != n_components:
        raise ValueError("Got %d closure flag(s) for %d component(s)."
                         % (len(flags), n_components))
    return flags


def _normalize_colors(colors: Union[None, str, Sequence[ColorSpec]],
                      components: List[np.ndarray]) -> List[np.ndarray]:
    if colors is None:
        return [np.asarray([DEFAULT_COLOR], dtype=float) for _ in components]
    if isinstance(colors, str):
        return parse_color_spec(colors, len(components))

    entries = list(colors)
    if len(entries) != len(components):
        raise ValueError("Got %d colour entr(y/ies) for %d component(s)."
                         % (len(entries), len(components)))
    out: List[np.ndarray] = []
    for index, (entry, pts) in enumerate(zip(entries, components)):
        label = component_label(index)
        if entry is None:
            out.append(np.zeros((0, 4), dtype=float))
            continue
        arr = np.asarray(entry, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] not in (3, 4):
            raise ValueError("Colour for component %s must have 3 or 4 channels."
                             % label)
        if arr.shape[1] == 3:
            arr = np.hstack([arr, np.ones((arr.shape[0], 1), dtype=float)])
        if arr.shape[0] not in (0, 1, pts.shape[0]):
            raise ValueError(
                "Component %s has %d colour(s) for %d vertices; VECT allows 0, "
                "1, or one per vertex." % (label, arr.shape[0], pts.shape[0]))
        out.append(arr)
    return out


def strip_repeated_endpoint(points: np.ndarray, precision: int) -> np.ndarray:
    """Drop a final vertex that repeats the first once both are rounded.

    A closed VECT component is closed by its negative count, not by repeating
    a vertex, so a caller that hands over a curve which already repeats its
    first point would otherwise produce a zero-length segment.  The comparison
    is made on the rounded rows that will actually be written, because that is
    what any reader downstream will see -- a geometric epsilon would leave
    behind pairs that round to the same text.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return pts
    fmt = "%%.%df" % precision
    first = tuple(fmt % v for v in pts[0])
    last = tuple(fmt % v for v in pts[-1])
    if first == last:
        return pts[:-1]
    return pts


def write_vect(path: str,
               components: Iterable[np.ndarray],
               closed: Union[None, bool, Sequence[bool]] = None,
               colors: Union[None, str, Sequence[ColorSpec]] = None,
               comments: bool = True,
               precision: int = DEFAULT_PRECISION,
               source: Optional[str] = None,
               tool: Optional[str] = None) -> str:
    """Write components as a Geomview VECT file and return the path.

    `closed` is one flag per component (or one for all of them) and is written
    as the sign of the vertex count, which is the only place VECT records it.
    A closed component that arrives with its first point repeated at the end
    has the repeat removed, so the file holds the same invariant every other
    curve file in this package does.

    `colors` may be a `--color` string, or one entry per component, each being
    None for no colour, one RGBA, or one RGBA per vertex.  The default is one
    opaque white per component.
    """
    blocks = _normalize_components(components)
    flags = _normalize_closed(closed, len(blocks))

    trimmed: List[np.ndarray] = []
    for pts, is_closed in zip(blocks, flags):
        trimmed.append(strip_repeated_endpoint(pts, precision) if is_closed else pts)
    for index, pts in enumerate(trimmed):
        if pts.shape[0] < 2:
            raise ValueError(
                "Component %s is left with %d point(s) after its repeated "
                "endpoint was removed." % (component_label(index), pts.shape[0]))
    blocks = trimmed

    color_blocks = _normalize_colors(colors, blocks)

    coord_fmt = "%%.%df %%.%df %%.%df\n" % (precision, precision, precision)
    color_fmt = "%%.%df %%.%df %%.%df %%.%df\n" % (6, 6, 6, 6)

    n_vertices = int(sum(p.shape[0] for p in blocks))
    n_colors = int(sum(c.shape[0] for c in color_blocks))

    with open(path, "w") as fh:
        fh.write("VECT\n")
        if comments:
            origin = " from %s" % _flat(source) if source else ""
            fh.write("# %s%s\n" % (tool or "Curve It", origin))
            fh.write("# a negative vertex count marks a closed component; "
                     "a closed component does not repeat its first vertex\n")
        fh.write("%d %d %d" % (len(blocks), n_vertices, n_colors))
        fh.write(" # components vertices colours\n" if comments else "\n")

        fh.write(" ".join(str(-p.shape[0] if flag else p.shape[0])
                          for p, flag in zip(blocks, flags)))
        fh.write(" # vertices per component\n" if comments else "\n")

        fh.write(" ".join(str(c.shape[0]) for c in color_blocks))
        fh.write(" # colours per component\n" if comments else "\n")

        for index, (pts, flag) in enumerate(zip(blocks, flags)):
            if comments:
                fh.write("# component %s: %s points=%d\n"
                         % (component_label(index),
                            "closed" if flag else "open", pts.shape[0]))
            for x, y, z in pts:
                fh.write(coord_fmt % (x, y, z))

        if n_colors and comments:
            fh.write("# colours (red green blue alpha)\n")
        for index, block in enumerate(color_blocks):
            if block.shape[0] == 0:
                continue
            if comments:
                fh.write("# component %s\n" % component_label(index))
            for r, g, b, a in block:
                fh.write(color_fmt % (r, g, b, a))
    return path


def _flat(text: object) -> str:
    """Collapse whitespace so a value can never break out of a comment line."""
    return " ".join(str(text).split())
