# Curve It

`curve_it.py` bends a roughly straight PDB structure so its principal axis follows a user-provided 3D curve. It was originally developed for DNA/RNA helices and now also handles protein PDBs by grouping protein atoms residue-by-residue.

Version: `V3_12`
GUI title: `AZBMOST Package Module #3 - Curve It: Sculpt PDB Structures Along Any 3D Curve`

## What It Does

- Reads a PDB file containing a roughly straight DNA/RNA helix, protein helix, or other filament-like structure.
- Reads an optional curve file: coordinate XYZ/text, molecular XYZ, or Geomview VECT. If no curve is supplied, it uses a default planar ring.
- Maps the PDB onto the curve using a rotation-minimizing frame.
- Preserves local geometry with rigid group mapping:
  - nucleic acids: phosphate, sugar, and base groups
  - proteins and unknown residues: whole-residue groups
- Reports closed-curve writhe from the exact piecewise-linear path used for mapping, using an analytic solid-angle sum over nonadjacent segment pairs. No smoothing or spline substitution is applied to mapped-polyline writhe.
- Supports open and closed curves, curve scaling, path start control, helix phase rotation, extra twist, and optional curve interpolation.

## Requirements

- Python 3.9 or newer
- Required: `numpy`
- Required by **Generate SC** and **Multicolor Split**; otherwise optional but recommended: `scipy` for curvature/writhe reporting and the local curvature/torsion tool
- Optional: `matplotlib` for the GUI curve viewer, local analysis plots, the SVG to XYZ preview, and the Multicolor Split preview image
- **SVG to XYZ** needs nothing beyond `numpy` and the standard library
- Required by **XYZ to 3D Model** and **Multicolor Split**; not needed otherwise: `trimesh` for STL/GLB mesh output
- Required by **KnotPlot to XYZ**; not needed otherwise: a local KnotPlot installation. KnotPlot is third-party software under its own license and is not bundled with this package; download it from https://knotplot.com/download/. The tool finds a normal installation on its own; set the `KNOTPLOT` environment variable to the full path of the executable if yours is elsewhere.
- Required by **Multicolor Split**; not needed otherwise: `manifold3d` for the exact mesh booleans, and a local UCSF ChimeraX installation, which the tool runs headless for `molmap` and its contour surfaces. ChimeraX is third-party software under its own license and is not bundled with this package; download it from https://www.cgl.ucsf.edu/chimerax/download.html. The tool finds a normal installation on its own; if yours is elsewhere, set the `CHIMERAX` environment variable to the full path of the executable, pass `--chimerax` with the executable or the `.app`, or fill in the tool's **ChimeraX** field, which it remembers. Parts written as atom selections in UCSF Chimera's convention also need a local UCSF Chimera 1.x, which reads them (https://www.cgl.ucsf.edu/chimera/download.html); the tool finds it the same way, through the `CHIMERA` environment variable, `--chimera`, or its **Chimera** field. Selections in ChimeraX's convention need nothing more.
- Optional: Tkinter for GUI mode. It is included with many Python installations.

Install the Python packages:

```bash
python3 -m pip install -r requirements.txt
```

## Git Clone And Git Pull

Clone downloads a fresh copy of the repository:

```bash
git clone https://github.com/azbmost/curve_it.git
cd curve_it
```

Use `git pull` later inside an existing clone to bring in new commits from GitHub:

```bash
git pull origin main
```

In short: use `git clone` once to get the repo, then use `git pull` whenever you want to update that local folder.

## Basic Usage

Show the version:

```bash
python3 curve_it.py --version
python3 curve_it.py -v
```

Launch the GUI:

```bash
python3 curve_it.py
```

or:

```bash
python3 curve_it.py --gui
```

Run from the command line with a PDB only, using the default ring curve:

```bash
python3 curve_it.py input.pdb
```

Run with a curve file:

```bash
python3 curve_it.py input.pdb curve.xyz -o output_curved.pdb
```

Use selected components from a blank-line-separated curve file:

```bash
python3 curve_it.py input.pdb br_abz.txt --curve-components B,C -o output_curved.pdb
```

Treat the curve as a closed loop:

```bash
python3 curve_it.py input.pdb curve.xyz --path-type closed -o output_curved.pdb
```

Use the curve without length scaling:

```bash
python3 curve_it.py input.pdb curve.xyz --scale-mode none -o output_curved.pdb
```

With `--scale-mode none`, Curve It does not scale the PDB/helix or the curve. It maps the PDB/helix onto the curve using the PDB/helix's native axial spacing. For open curves, the curve must be at least as long as the PDB/helix principal-axis length; if it is longer, only the needed initial part of the curve is used. For closed curves, periodic wrapping is allowed. Use `--scale-mode helix_to_curve` when you want the unscaled PDB/helix distributed over the full unscaled curve.

Scale the curve to a numeric target length in Angstrom:

```bash
python3 curve_it.py input.pdb curve.xyz --path-type closed --scale-mode 340.0
```

Add phase and twist:

```bash
python3 curve_it.py input.pdb curve.xyz --helix_phase 90 --twist 360
```

Interpolate the curve before fitting:

```bash
python3 curve_it.py input.pdb curve.xyz --interp-mode n --interp-n 400
python3 curve_it.py input.pdb curve.xyz --interp-mode p --interp-p 5
```

## Curve Interpolation

Interpolation under **Curve parameters** changes the curve that Curve It actually uses for the run. It is not only for total curvature and writhe reporting.

When interpolation is enabled, the input curve is first resampled or densified, then that interpolated curve is used for:

- fitting/curving the PDB coordinates
- computing curve length
- computing total curvature and writhe when applicable
- viewing the curve in the GUI
- writing the optional `<curve>_interpolated.<ext>` helper file in GUI mode

Use `--interp-mode n` when you want exactly `--interp-n` evenly arc-length-spaced points. Use `--interp-mode p` when you want to insert `--interp-p` points between each adjacent pair of original curve points.

`--interp-mode n` reproduces exactly every vertex that turns by at least `--min-corner-angle`, 20 degrees by default, and resamples each smooth stretch between two kept corners on its own, so a polygonal curve keeps its corners instead of having them clipped by evenly spaced samples. On smooth, densely sampled curves no vertex reaches 20 degrees and the result is bitwise identical to blind resampling; on a coarse polygon it preserves contour length that blind resampling loses. `--min-corner-angle 0` restores the blind behaviour exactly. The setting is ignored by `--interp-mode none` and `--interp-mode p`, and it is the same threshold, the same default and the same code as the SVG to XYZ tool uses, since both share the resampler in `curve_it_lib/interpolate_xyz.py`.

## Curve File Format

Curve files can be plain whitespace-separated coordinates:

```text
x y z
x y z
x y z
```

They can also be standard XYZ-like files with an atom count/comment header and an element label before each coordinate triplet.

They can also be Geomview **VECT** files, the polyline format Geomview reads and that ridgerunner and the knot-theory tools around it write:

```text
VECT
NPolylines NVertices NColors
Nv[0] ... Nv[NPolylines-1]     vertex count per polyline; negative means closed
Nc[0] ... Nc[NPolylines-1]     colour count per polyline: 0, 1, or one per vertex
x y z                          NVertices of these, polylines in order
r g b a                        NColors of these, polylines in order
```

Every tool in this package reads VECT, and the format is recognised from the file's header word rather than from its extension, so a `.txt` holding VECT still works. `#` starts a comment that runs to the end of the line and may appear anywhere, including trailing a coordinate row; count lists may wrap across lines.

VECT is the only curve format here that **states** whether each component is a closed loop, in the sign of its vertex count. Everything else leaves closure unsaid and is either assumed open or measured from the gap back to the first point. So a VECT file answers the question for the tool that reads it: it sets `--path-type` in Curve It and Get phase, `--closed` in XYZ to 3D Model and Interpolate, the drawn closure in View curve, and `--closed-chains` in Plane It, exactly as `LINK` records do for a PDB. An explicit option always wins over the file. A closed VECT component does not repeat its first vertex, which is the same invariant Curve It's own curve files hold, so geometry crosses between the two formats unchanged.

Colours are read and reported but are not geometry. No other format here can hold them, so they are lost through any conversion that does not end in VECT.

Curve files may contain multiple components: separated by one or more blank lines in a plain coordinate file, and one per polyline in a VECT file. Curve It labels those components `A`, `B`, `C`, and so on in file order. By default, all components are concatenated in file order and used as the curve. Use `--curve-components` in CLI mode, or **Select components...** in the GUI, to choose a subset such as `A`, `B,C`, or `A-C`.

The GUI **View curve** window can show all parsed components or the currently selected components. When the mouse cursor is close to the plotted curve or points, the viewer reports the normalized path location `u` from `0` to `1`. Molecular XYZ files are treated as one component and can also be used directly as the curve input.

## GUI Tools

The GUI has a dedicated **Tools** area for utility tools.

**Get phase** beside the main **Phase (deg)** field opens `curve_it_lib/get_curve_it_phaseV5_1.py`. Its atom-selection area can use a single real PDB atom, the centroid of multiple individually selected real atoms, or a virtual source-space `x,y,z` point that does not exist in the PDB. Choose **Real atom(s)** and set the number of selection rows, or choose **Virtual atom** and enter coordinates in Angstrom. The computed phase transfers back to the main GUI automatically.

The helper can also be run directly. Repeat `--real-atom CHAIN:RESSEQ:ATOM_NAME` to use a centroid; use `_` for a blank chain. Supplying `--virtual-atom` selects virtual-atom mode automatically.

```bash
python3 curve_it_lib/get_curve_it_phaseV5_1.py input.pdb curve.xyz \
    --virtual-atom 10,5,3 --target-mode curvature_angle
python3 curve_it_lib/get_curve_it_phaseV5_1.py input.pdb curve.xyz \
    --real-atom A:12:P --real-atom B:12:P --target-mode curvature_angle
```

**Convert XYZ...** opens a small conversion window for coordinate XYZ/txt, molecular XYZ, Geomview VECT, and fake-PDB output. An optional scale factor multiplies every output coordinate before writing. Fake PDB output is meant for molecular visualization: each point becomes one atom in one residue, using residue `ALA` and atom `CA` by default. Blank-line-separated coordinate components become chains `A`, `B`, `C`, and so on; selected closed chains can be written with `LINK` records.

This window, and its command-line form below, are the only places the package **writes** VECT; every tool reads it, but the generators keep writing coordinate XYZ. Loading a VECT input fills in **Closed components** from the file's own header, and **Colours** takes one entry per component — `red,blue`, `#ff0000`, or `1 0 0 1, 0 0 1 1` — cycling when there are fewer colours than components and defaulting to opaque white. **Colours** also takes the name of a whole palette, `DiLiuLab` or a tint such as `DiLiuLab-T80`, which gives the components the lab's figure colours in turn, and the **Palette** menu below it writes that name in. An entry can also be one lab colour by name, as in `RedT80,BlueT80`; see [DiLiuLab Palette](#diliulab-palette). Only VECT and fake PDB keep components apart on the way out; both XYZ forms join them into one block.

```bash
# Same window, from the command line. The output extension picks the format.
python3 curve_it.py --convert curve.xyz curve.vect --convert-closed all
python3 curve_it.py --convert curve.xyz link.vect --convert-closed A,C --convert-color "red,blue"
python3 curve_it.py --convert knot.vect knot.pdb          # closure becomes LINK records
python3 curve_it.py --convert knot.vect knot.xyz --convert-to coordinate
```

`--convert-closed` takes `all`, `none`, a selection such as `A,C` or `A-C`, or `auto` (the default), which believes a VECT input and otherwise writes components open rather than guessing from the geometry.

**SVG to XYZ...** opens `curve_it_lib/svg2xyz.py`, which turns a 2D vector drawing into space curves. Every drawn `<path>`, `<polyline>`, `<polygon>`, `<line>`, `<rect>`, `<circle>`, and `<ellipse>` becomes one component of a coordinate XYZ/txt file, so a drawing holding several curves produces a curve file holding the same number of space curves, selectable as components `A`, `B`, `C`, and so on. The drawing is flat, so every `z` is the same constant, `0` by default.

Bezier and elliptical-arc geometry is flattened by adaptive subdivision at a chosen tolerance, which keeps corners exact and arcs smooth; elliptical arcs, circles, ellipses, and rounded rectangles are all built from cubic pieces of at most 45 degrees, so their radial error stays near `2e-5` of the radius. Group and element transforms are composed exactly, `<use>` references are expanded in place, and nested `<svg>` and `<symbol>` viewBoxes are applied. Text, images, gradients, clip paths, markers, anything inside `<defs>`, and anything hidden with `display:none`, `visibility:hidden`, or `opacity:0` are all skipped, including a hidden Illustrator layer delivered as a `.st5 { display:none; }` rule in a `<style>` block.

The SVG y axis points down the page, so it is flipped by default and the exported curve comes out with the same orientation as the drawing; `--no-flip-y` keeps the raw SVG coordinates. Flipping is a reflection, so it reverses the sign of any signed quantity computed from the curve later. Closed shapes are written without repeating the first point, which is the convention Curve It, its writhe calculation, and Generate SC all use; load such a curve with **Path type: closed**.

Coordinates are written at the requested precision, and points that would round to the same row are removed before writing, so the file never contains a zero-length segment. That matters because Curve It's writhe calculation rejects those outright and its discrete total curvature silently loses the turning angle there.

Sampling is controlled by `-n/--points`, which takes a whole number, `keep`, or `auto` (the default). `auto` works a point count out from the curve's own geometry: for a chord `h` across a bend of radius `R` the gap to the true curve is about `h^2 / (8R)`, so holding that at the flattening tolerance gives `h = sqrt(8 * R_min * tolerance)` and `n = ceil(contour_length / h)`. `keep` returns the adaptively flattened points untouched, which is what versions before 1.1 did by default; their spacing can vary enormously, by a factor of 420 across the sample star. `--spacing` takes a step in output units or `auto`: `--points auto` resolves a chord per curve so every curve is equally accurate, while `--spacing auto` resolves one chord for the whole drawing from its tightest bend so every curve is equally dense.

**Auto matches the drawing's own accuracy, not any downstream requirement.** At the default tolerance the sample star resolves to 1.542 units, which is 3.62 Angstrom once Curve It scales it to a 168-bp B-DNA contour, coarser than the 3.40 Angstrom helical rise. Pass an explicit count or spacing when something further along the pipeline needs a particular resolution.

Corners survive every setting. Resampling reproduces exactly every vertex that turns by at least `--min-corner-angle`, 20 degrees by default, and resamples each smooth stretch between two kept corners on its own, so a rectangle stays a rectangle: resampling a 60 by 40 rectangle to 137 points clips its corners by 0.52 units at a threshold of 0 and reproduces all four exactly at 20. A vertex only counts as a corner when both of its segments reach a quarter of the median segment length, which keeps a drawing program's closepath stub from being mistaken for one. Resampled points always lie on the drawn path, so the only shape change possible is a chord cutting inward across a corner.

The resolved sampling is reported in `--info` and recorded in the XYZ header, so a file says how it was built.

Use `--info` first to list what the drawing contains, then pick from that listing with `-c/--components`, spelled exactly like Curve It's own `--curve-components`: `A`, `B,C`, `A-C`, or `all`. Selection is applied after every other filter, so the labels are the ones you just read, and after scaling and centring, so a component extracted on its own is bit-identical to the same component of the full file and separately extracted curves stay in register. `--split` writes every component to its own file in one pass instead.

`--include` and `--exclude` select curves by id, class, or enclosing group name, and unlike `--components` they run before scaling, so the curves left over reframe the output. This is how the decoration is dropped from a Plane It projection SVG. `--elements` restricts by shape type.

```bash
python3 curve_it_lib/svg2xyz.py drawing.svg --info
python3 curve_it_lib/svg2xyz.py drawing.svg
python3 curve_it_lib/svg2xyz.py drawing.svg --points keep
python3 curve_it_lib/svg2xyz.py drawing.svg -s 0.25 --points 400
python3 curve_it_lib/svg2xyz.py drawing.svg --spacing auto
python3 curve_it_lib/svg2xyz.py drawing.svg -c B
python3 curve_it_lib/svg2xyz.py drawing.svg --fit-size 340 --split
python3 curve_it_lib/svg2xyz.py projection.svg --elements path --exclude xy-plane
```

Curve It concatenates every component of a curve file by default, so a multi-curve XYZ file should normally be used one component at a time, with **Select components...** in the GUI or `--curve-components A` on the command line. The tool prints the exact commands after a conversion.

**KnotPlot to XYZ...** opens `curve_it_lib/kp2xyz.py`, which turns the knot and link catalogue shipped with a local KnotPlot installation into coordinate XYZ/txt curve files. Name a target the way KnotPlot names it, `4.1` or `6.3.2`, or give a generator command such as `torus 2 3`, and the tool writes one file of plain `x y z` rows per target. Every component of a link becomes one blank-line-separated block of the same file, selectable as components `A`, `B`, `C`, and so on; `--split` writes each component to its own file instead, and `--all` converts the whole catalogue in one pass.

A shipped catalogue entry is KnotPlot's own binary format rather than text, so the coordinates always come out of KnotPlot itself: the tool runs it headlessly with no window and asks it for the coordinates of the loaded conformation. The catalogue files are read directly only to list their names and to hash their contents for deduplication; no coordinate is ever parsed out of them. Nothing is displayed, and no KnotPlot window opens.

**KnotPlot's `torus p q` does not use the index order every text uses.** Measured here, `torus p q` winds the curve `p` times around the `z` axis, the axis of the torus, and `q` times around the tube: `torus 2 3` goes twice around the axis and three times around the tube. The common parametrization, and so most software, names that curve the `(2,3)` torus knot; other sources, including Adams' *The Knot Book*, put the winding around the tube first and call the same curve the `(3,2)` torus knot. A `(p,q)` quoted in a paper may therefore be this tool's `torus q p`. The knot type is unaffected, since `T(p,q)` and `T(q,p)` are the same knot, but the shape is not: `torus 2 3` shows 3 crossings looking down `z`, the familiar minimal trefoil diagram, while `torus 3 2` shows 4, the three-strand braid form of the same trefoil. If you are reproducing a figure and the shape looks wrong, swap the two numbers. The report says so whenever a `torus` target is used, and the tool's `torus` help topic carries the detail.

**Preview** extracts the current targets into a temporary folder and opens them in Curve It's own 3D curve viewer, `curve_it_lib/view_xyzV3.py`, without touching the output folder, so a knot can be looked at before it is kept; **Extract** is what writes files. `--preview` does the same after a command-line run. Each curve opens in its own viewer process, so the tool window stays usable and several can be compared side by side; at most 6 windows open at once, and a larger run says how many it left on disk.

`--nbeads N` sets the number of points. KnotPlot spreads that number over the whole conformation, so on a link it is the total and not a per-component count: `--nbeads 300` on the three-component link `6.3.2` gives components of 107, 97, and 95 points. Leave it unset to export the shipped conformation at its own resolution.

Conformations are exported exactly as KnotPlot ships them, and relaxation is deliberately not offered. The tool does not relax, tighten, or smooth the geometry. Three things change the coordinates, and all three are opt-in: the reflection described below, the `--nbeads` resampling above, and `--precision`, which rounds each value as it is written. Resampling redistributes vertices along the same curve rather than reshaping it, but it does move them, so it is not a no-op even at the shipped point count: `3.1` ships 47 points, and `--nbeads 47` shifts them by up to `0.096853` in a single coordinate and takes the writhe from `+3.3722171577968423` to `+3.3710769510513137`, both measured with `curve_it_lib/cal_xyz_total_curvature_writheV2.py` at its defaults. Leave `--nbeads` unset and the shipped coordinates are what you get. These conformations are correct representatives of their knot and link types, though not tight or ropelength-minimizing ones; relax a knot inside KnotPlot first and export it afterwards if that is what you need.

`--list` prints the catalogue and `--info` shows what a run would extract without writing anything. Duplicates are dropped by comparing file contents rather than names, which reduces the 585 catalogue files to 522 distinct knots and links; `--no-dedupe` keeps all 585. Everything is extracted in one KnotPlot session rather than one session per target, so even the largest run starts KnotPlot once: those 522 entries write 59877 points in about 0.3 seconds here. Files are written to `-o/--outdir`, `kp_xyz` by default.

**The dotted and underscored spellings are not two names for one knot, and the difference is chirality.** 93 names ship in both spellings, but only 63 of those pairs are byte-identical; those 63 collapse onto the dotted form KnotPlot's own `load` syntax uses. The other 30 pairs are genuinely different curves, so both spellings survive deduplication and both appear in `--list` and `--all`. 28 of the 30 are mirror images, the two chiral forms of the same knot type: measured with `curve_it_lib/cal_xyz_total_curvature_writheV2.py` at its defaults, `8.19` has writhe `-8.62647841209301` and `8_19` has `+8.62647841925399`, equal in magnitude to eight significant figures and opposite in sign. Treat `8.19` and `8_19` as opposite-handed knots rather than as a duplicate to discard, and check which one you loaded before reading any signed quantity off the curve.

The two remaining pairs, `9.20`/`9_20` and `9.35`/`9_35`, share a handedness, and they are two slightly different conformations of one knot rather than one conformation stored in a different orientation. Measured with the same tool, `9.20` gives `+6.346643502580359` against `+6.346635731949251` for `9_20`, and `9.35` gives `+7.539209051573574` against `+7.539218781711285` for `9_35`. Rotating or translating a curve leaves its writhe alone, so two files that agree in sign and still disagree in value cannot be one conformation re-oriented; their contents differ as well, which is why the content hash keeps both.

The tool can also write the mirror image of any target, so either handedness is available even where the catalogue ships only one. `--mirror` writes the reflection in place of the original, `--both` writes the original and the reflection, and `--mirror-axis x`, `y`, or `z` chooses which coordinate is negated. Exactly one coordinate is negated, because negating one axis is a genuine reflection while negating two composes into a rotation and leaves the handedness alone. The default axis is `z`, which leaves `x` and `y` untouched: the projection you see looking down the `z` axis is identical, while every crossing swaps over for under, which is the mirror of the textbook knot diagram. Reflection negates the writhe, the linking number, and every other chirality-sensitive invariant; mirroring the shipped `8.19`, writhe `-8.62647841209301` by the same tool, gives `+8.62647841209301` and so matches KnotPlot's own chiral partner `8_19`, `+8.62647841925399`, to eight significant figures. A link is reflected as a whole, so the components of a mirrored link stay in register with one another. For an amphichiral knot such as `4.1`, whose shipped conformation measures `+0.15240866875536058`, near zero beside the `8.6` of `8.19`, the mirror measures `-0.15240866875536058` and is the same knot type in a different conformation, not a second knot.

A reflected file is written as `<stem>_mirror.xyz`, or `<stem>_mirror_A.xyz` per component with `--split`, and its `#` header records that the coordinates were reflected and on which axis, so a file on disk is never ambiguous about its handedness. `--both` is opt-in rather than the default because `--all --both` writes 1044 files instead of 522.

This requires a local KnotPlot installation. The tool searches the `KNOTPLOT` environment variable holding the full path of the executable, then `KNOTPLOT_HOME`, then `PATH`, then the usual install locations for the platform; `--knotplot /path/to/KnotPlot` overrides the search for one run. `--check` prints what it found and exits, and it does more than look for a file: it launches the binary, asks it to quit, and looks for KnotPlot's own startup banner, so something executable that is not KnotPlot is reported as exactly that and `--check` exits nonzero. The tool opens whether or not the search succeeds, and its **Locate KnotPlot...** button points it at an install the search missed, so the Curve It launcher warns about a missing installation and then offers to open the tool anyway. KnotPlot is third-party software under its own license and is not bundled here; download it from https://knotplot.com/download/.

```bash
python3 curve_it_lib/kp2xyz.py --check
python3 curve_it_lib/kp2xyz.py --list
python3 curve_it_lib/kp2xyz.py 4.1 -o out
python3 curve_it_lib/kp2xyz.py 6.3.2 -o out --nbeads 200 --split
python3 curve_it_lib/kp2xyz.py "torus 2 3" -o out
python3 curve_it_lib/kp2xyz.py 8.19 -o out --mirror
python3 curve_it_lib/kp2xyz.py 3.1 -o out --both
python3 curve_it_lib/kp2xyz.py 4.1 -o out --mirror --mirror-axis x
python3 curve_it_lib/kp2xyz.py --all -o catalogue
```

Closure is measured per component rather than assumed. Almost everything KnotPlot ships is a closed curve, written without repeating the first point, so load it with **Path type: closed**; the `#` header records `closed=yes` or `closed=no` for every component, and in the shipped catalogue exactly one component, component A of `n3.1s`, is a genuine open arc. A link file holds several components and Curve It concatenates all of them by default, so use one at a time with **Select components...** in the GUI or `--curve-components A` on the command line. Whenever a written file holds more than one component, the report ends with a `USING THIS FILE IN CURVE IT` block spelling out the exact command for the first three components and naming the last when the file holds more than three; a single-component knot, and every file written by `--split`, leaves nothing to choose between and gets no block.

**Generate helical curve...** opens `curve_it_lib/generate_helix_xyzV2.py`. This tool writes a plain-coordinate XYZ file for a circular helix:

```text
x(t) = R cos(t + phi)
y(t) = +/- R sin(t + phi)
z(t) = z0 + c t
```

The output has one `x y z` point per line and can be loaded directly as a Curve It curve input. The GUI can derive `c` from `R` and pitch angle, derive `R` from `c` and pitch angle, or use `R` and `c` directly.

You can also run it from the command line:

```bash
python3 curve_it_lib/generate_helix_xyzV2.py -R 10 -c 2 -L 200 -n 1000 -o helix.xyz
python3 curve_it_lib/generate_helix_xyzV2.py -R 10 --pitch-angle-deg 20 --derive c-from-R -L 200 -o helix.xyz
```

**Generate SC...** opens `curve_it_lib/generate_sc_xyzV3_8.py`, the package's only supercoil-centerline generator. It writes a closed plectonemic axis as plain `x y z` rows with the requested contour length and signed Gauss writhe. The default contour length is 1071 and the default output density is 2000 unique periodic points. Before writing, V3.8 applies Curve It's periodic smoothing once, resamples, and rescales to the requested closed length. It then verifies the exact decimal coordinates with the same closed-polyline segment-pair writhe calculation Curve It uses for mapping. Load the result as a Curve It curve with **Path type: closed**.

The report distinguishes the nominal scaled superhelix radius from the measured final arm radius. The latter is computed from the serialized coordinates as the median radial distance of central arm points from the translated superhelical axis, excluding the end loops and their smoothing transitions.

The arm-phase revolution count is reported as `plectoneme_phase_turns = |theta_total| / (2*pi)`. This name distinguishes the generator's complete arm-phase revolutions from a generic geometric or topological “number of superhelical turns.”

In the GUI, **Minimum final measured radius** appears before **Qualifying views** and defaults to 13. The qualifying-views field is dynamically disabled and greyed whenever arm-phase trimming is off or a minimum final radius is entered. Clear the radius field to select qualifying-view screening. The radius field is itself disabled when trimming is off.

Light-blue `?` buttons beside every generation argument and opening-angle option open concise explanations and examples without leaving the GUI.

When `--output` is omitted, V3.8 constructs a filename from the defining selections. The default is `sc_L1071_Wm3_R13_ABend.xyz`. `ABend`, `ALocal`, `ATotal`, and `AEq` identify the four automatic opening-angle objectives; a manual angle uses a token such as `A25deg`. Radius screening uses `R13`, qualifying-view screening uses a token such as `Q60`, and untrimmed geometry uses `NoTrim`. The opt-in zero-H0 family appends `H0zero`, for example `sc_L1071_Wm3_R13_ABend_H0zero.xyz`, and the accurate angle search appends `Acc`, for example `sc_L714_Wm1_R13_ABend_Acc.xyz`. The GUI updates this name dynamically while preserving a custom typed or browsed filename.

The default screening mode requires a final measured radius of at least 13. Set another threshold with `--minimum-final-radius LENGTH` (alias `--minimum-final-measured-radius`). Generate SC finds the largest feasible phase trim from 0 through `0.90*pi` whose radius, measured from the central 90% of the arms after smoothing, scaling, centering, and XYZ decimal quantization, remains at or above the threshold. Thus the screened geometry and the reported final radius refer to the same serialized curve that Curve It will map. Within this phase-trim family, decreasing the allowed radius permits a larger trim and generally a higher qualifying-view fraction. Supplying `--qualifying-views PERCENT` explicitly selects percentage screening instead. Radius-constrained screening requires a nonzero integer writhe and enabled trimming.

For nonzero integer writhe, V3.8 shortens the signed arm phase by a fitted trim below `1.0*pi`, refits the end-loop control distance to retain the requested exact writhe, and screens deterministic generic orthographic projections. Trimming is enabled by default; use `--no-trim` to retain the original `pi*W` phase. When percentage screening is selected, the required qualifying-view percentage defaults to 55. V3.8 rotates the complete trimmed curve around z by half of the phase removed from the legacy sweep. This restores the symmetric fixed-`xz` presentation of V2.2: odd integer writhe has reflection symmetry under `z -> -z`, and even integer writhe has central symmetry under `(x,z) -> (-x,-z)`. Because the rotation centers the shortened phase interval, all `|W|` intended xz crossings remain inside it for `phase_trim < 1`; the former `0.5*pi` limit applied to the unrotated interval. The 256 viewing directions are reflection-paired, so mirror-related positive- and negative-writhe curves receive identical finite-sample screening. For bending-energy, max-local, and total-curvature modes, projection screening holds the opening angle found at the default `0.40*pi` trim while testing progressively larger trims through `0.90*pi`. Equal-lobes mode instead resolves its opening angle at each tested trim so the terminal/middle equality remains satisfied. The final report shows whether trimming was enabled, the active radius or percentage criterion, selected phase trim, z-axis symmetry rotation, V2.2-style middle-segment peaks, lobe heights where applicable, projection statistics, and exact writhe. Fractional writhe and `W=0` use untrimmed geometry.

For the bending-energy, max-local and total-curvature objectives, `--angle-search` (**Angle search** in the GUI) chooses how the automatic angle is found. The default, **quick**, is the search described above: its angle is optimal only for the default-trim curve, so the curve written is in general not the objective's optimum. **Accurate** instead gives every candidate angle its own screening, smoothing, scaling and quantization, and ranks the angles by the objective measured on the curve that would be written; at the angle chosen, the curve is the one `-a` writes. Over contour lengths 714 and 1071 and `|W| = 1-5` at the default 13 radius, quick's bending-energy integral came within 0.05-1.0 % of accurate's for `|W| >= 2` but was 8-9 % higher at `|W| = 1`, where the accurate optimum sits at the 85-degree search limit and the log says so; quick's largest local curvature was 8-158 % higher, the gap growing with `|W|`; and quick could not build four total-curvature configurations that accurate built. An accurate max-local optimum can leave far fewer qualifying views, so check the projection report. Accurate samples the same 17 angles and three refinements as quick and takes minutes rather than seconds.

V3.8 supports `|W| <= 10` and four automatic opening-angle objectives over 5-85 degrees. The default `--angle-objective bending-energy` minimizes total bending energy, proportional to `integral kappa(s)^2 ds`; for a homogeneous isotropic rod with constant bending rigidity `A` and no intrinsic curvature, physical bending energy is `A/2` times this integral. `--angle-objective max-local` minimizes the largest local curvature, while `--angle-objective total` minimizes `integral kappa(s) ds`. `--angle-objective equal-lobes` matches terminal/middle fixed-`xz` lobe heights for integer `|W| >= 2` in both trimmed and untrimmed modes. Supply `-a ANGLE` or `--opening-angle ANGLE` to retain a manual angle strictly between 0 and 90 degrees. The nominal input angle satisfies `tan(alpha) = |theta| R / (H-H0)`, where `H` is the full canonical arm height and `H0` is the phase-independent axial allowance. The default remains `H0 = 2R`, for which the realized acute arm-tangent angle is `beta = atan(|theta| R/H)`, slightly smaller than `alpha`. Use `--zero-h0` to set `H0 = 0`, in which case `beta = alpha` for nonzero W. In both nonzero-writhe families, the endpoint directions supplied to the closing Bezier loops are normalized exact derivatives of the parametric arm equations; no tangent is estimated from a plot or sampled polyline. The zero-H0 choice repeats all angle, loop-control, writhe, trim, smoothing, scaling, and serialized-coordinate checks. At exactly `W = 0`, where the arm/loop parameterization collapses, V3.8 instead directly writes a regular planar ring in the xz plane. Its polygon radius is `L/[2N sin(pi/N)]`, so the unrounded N-point closed polygon has contour length L; decimal serialization is then verified exactly. Opening-angle, loop-control, trimming, smoothing, and projection screening are not applied to this ring. Neither angle is the DNA base-pair twist angle or the full angle between the two arms. The older option name `--curvature-objective` remains an alias.

You can also run Generate SC from the command line:

```bash
python3 curve_it_lib/generate_sc_xyzV3_8.py -w -3
python3 curve_it_lib/generate_sc_xyzV3_8.py --angle-objective total
python3 curve_it_lib/generate_sc_xyzV3_8.py --angle-objective max-local
python3 curve_it_lib/generate_sc_xyzV3_8.py --angle-objective equal-lobes
python3 curve_it_lib/generate_sc_xyzV3_8.py --no-trim
python3 curve_it_lib/generate_sc_xyzV3_8.py --qualifying-views 60
python3 curve_it_lib/generate_sc_xyzV3_8.py --minimum-final-radius 13
python3 curve_it_lib/generate_sc_xyzV3_8.py -a 25
python3 curve_it_lib/generate_sc_xyzV3_8.py --zero-h0
python3 curve_it_lib/generate_sc_xyzV3_8.py -w 0 --zero-h0
python3 curve_it_lib/generate_sc_xyzV3_8.py -L 714 -w -1 --angle-search accurate
# Supply -o custom_name.xyz to override the automatic filename.
```

**Local curvature/torsion...** opens `curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py`. This tool writes a CSV table with normalized path position, coordinates, local curvature, regularized local torsion, local writhe density, and diagnostic columns. Its GUI includes a quick-loading test example for a three-lobe trefoil knot, the `(2,3)` torus knot:

```text
x(t) = (2 + cos(3t)) cos(2t)
y(t) = (2 + cos(3t)) sin(2t)
z(t) = sin(3t),   0 <= t < 2*pi
```

You can also run the trefoil example from the command line:

```bash
python3 curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py --example-trefoil --no-plot
```

**Curved Connector...** opens `curve_it_lib/curved_connectorV3_4.py`. This tool screens curved nucleic-acid connectors between two target helical end base-pairs using a straight duplex template. It builds a practical clamped Euler-elastica proxy centerline for each candidate length, ranks candidates by destination-end fit, and writes ranked PDB assemblies plus `connector_summary.tsv`.

```bash
python3 curve_it_lib/curved_connectorV3_4.py target.pdb template.pdb \
    --source-bp A33,B1 --dest-bp E1,F33 --top-k 5
```

V3_4 adds an optional sampled local-curvature constraint. Use `--max-local-curvature` to set a maximum centerline curvature in `A^-1`; the equivalent minimum bend radius is `1 / kappa_max` Angstrom. When this constraint is active, the default screening order is long-to-short, quick feasibility checks skip impossible lengths, and capped optimizer controls such as `--curvature-opt-maxiter`, `--curvature-opt-starts`, `--curvature-constraint-samples`, and `--curvature-opt-timeout-sec` help avoid very long infeasible solves.

```bash
python3 curve_it_lib/curved_connectorV3_4.py target.pdb template.pdb \
    --source-bp A33,B1 --dest-bp E1,F33 --max-local-curvature 0.04
```

The summary's `twist_mismatch_deg` is an endpoint base-pair orientation mismatch, not integrated geometric torsion or material twist energy.

**Plane It...** opens the Plane It companion GUI. Plane It projects selected atoms or 3D points from PDB/XYZ/text files into 2D SVG using PCA or current XY coordinates. The stable launcher is `plane_it.py`; the current versioned implementation is `curve_it_lib/plane_itV3_8.py`.

Launch the Plane It GUI:

```bash
python3 plane_it.py
python3 plane_it.py --gui
```

Basic CLI examples:

```bash
python3 plane_it.py input.pdb --atom-type P
python3 plane_it.py input.pdb --atom-type P --draw-lines
python3 plane_it.py input.pdb --atom-type P --draw-lines --draw-base-pairs
python3 plane_it.py input.pdb --atom-type P --draw-base-pairs --base-pair-atom "C4'"
python3 plane_it.py points.txt --atom-type all
python3 plane_it.py input.pdb --atom-type P --write-projection-basis
python3 plane_it.py input.pdb --atom-type P --depth-order-circles
python3 plane_it.py input.pdb --atom-type P --palette DiLiuLab-T80 --chain-colors A=BlueT80,B=RedT80
python3 plane_it.py input.pdb --atom-types "P,C1'" --style "C1' fill=GrayT40"
```

Plain coordinate text files may contain multiple components separated by blank lines; Plane It treats those components as chains `A`, `B`, `C`, and so on.

`--color-by chain`, the default, gives each chain its own colour, and `--color-by atom-type` gives each atom type its own. Every colour starts at the **Palette**'s assignment and can be changed freely. In the window, the swatches beside the **Palette** menu show the colours a run will use, one per chain of the input or one per atom type, with `+N more` opening the rest; click a swatch to change it, and **Reset** returns to the palette. Colours picked for chains belong to the file they were picked for and are cleared when another file is loaded. Each atom-type row's **Fill** and line **Color** start at the palette's colour, or at `chain` in chain mode, and take any colour: typed as a hex code, an SVG name, or a DiLiuLab name such as `RedT80`, or picked from the square beside the field, whose picker offers the DiLiuLab colours at every tint. A colour set in a row wins for that atom type in either mode; in chain mode a chain's own colour comes next, then the palette. On the command line the same colours are `--chain-colors A=RedT80,B=#4c79e6` and `--style "C1' fill=MintGreenT80 line_stroke=#1f77b4"`.

Plane It includes a finite patch of the projection-basis xy-plane in the SVG by default; use `--no-xy-plane` to omit it. In PCA mode, this is the PC1/PC2 plane through the selected-atom centroid, where projected depth is `0`; in current-XY mode, it is the current coordinate xy-plane after any optional pre-projection transform. The SVG group/layer is named `xy-plane`, and its polygon shape is named `xy-plane-shape`. If SVG depth ordering is enabled for circles, neighbor lines, or base-pair lines, the xy-plane patch is sorted with those items at projected depth `0`.

Plane It SVGs include a projected-length scale bar by default. The top-level projection group stores the conversion factor as `data-scale`, and the scale-bar layer reports the same value visibly as `scale: 1 <unit> = <data-scale> SVG units`. For PDB files, the projected coordinate unit is normally Angstrom. The default scale bar is 10 Angstrom; use `--scale-bar-length`, `--scale-bar-unit-label`, or `--no-scale-bar` to adjust it.

DSSR base-pair lines use the default output path `<input_folder>/tmp_file/<input_filename>.out`. When needed, Plane It may try to run:

```bash
x3dna-dssr -i=<input> --more -o=<default output>
```

When Plane It runs `x3dna-dssr`, it uses that `tmp_file` folder as the working directory so DSSR sidecar files stay with the DSSR output instead of appearing in the folder where Plane It was launched.

This requires `x3dna-dssr` to be installed and available on `PATH`; otherwise, place an existing DSSR output file at the default path before using `--draw-base-pairs`.

Base-pair lines use `--base-pair-atom` as the residue anchor atom. The default is `C3'`, recommended for B-DNA; `C4'` is recommended for A-RNA.

**XYZ to 3D Model...** opens `curve_it_lib/xyz2model.py`, which turns a curve into a printable solid. It sweeps a round rod along every curve in the file and writes a binary STL holding all components as one multi-shell file, plus a binary GLB with one separately colored mesh per component for rendering. Closed loops and open strands are both handled, and open ends receive rounded caps. When the main window already has a curve file loaded, the tool opens with that file selected. This tool requires `trimesh`.

The scale factor is applied to the centerline first and the rod is swept afterwards, so the rod diameter is in final output units and the scale factor does not change it:

```text
input coordinates  x --scale  ->  centre-line  + --diameter  ->  solid
```

`MODEL SIZE` is the printed extent and is the number to compare against a build volume. A rod of radius `r` reaches `r` beyond the centerline in every direction, so the solid is exactly one rod diameter larger than the centerline extent on each of the three axes, rounded end caps included. The measured size read back from the finished mesh is slightly under `MODEL SIZE` because the tube is a 24-sided prism inscribed in the true circle rather than touching it.

The report states the closest approach between every pair of components, and the gap the chosen rod leaves between their surfaces. A rod of radius `r` grows each curve by `r` in every direction, so two centerlines passing `d` apart leave `d - diameter` of air between the solids, and a gap at or below zero means the two components fuse into one piece. That is what you want for a sculpture and not what you want for a link whose parts have to move, so the report also names the thickest rod that keeps everything separate. The centerline distance is exact: it is measured segment to segment rather than sampled at the vertices, so a coarse curve does not overstate its own clearance. A mesh input has no centerline, so its clearance is measured between the nearest vertices of the shells instead, an upper bound on the true surface gap that is tight to about one edge length.

Output names carry the settings that change the solid without changing the file it came from: the rod diameter for a curve input, the scale factor for a mesh. `curve.xyz -d 2.0` writes `curve-D2.0.stl`, so two rod sizes swept from one curve no longer overwrite each other, and re-running on an output's own name replaces that tag rather than stacking a second one. Explicit `--stl` and `--glb` paths are used exactly as given. A run that would write over its own input stops before writing anything and says so.

An existing `STL`, `OBJ`, `PLY`, `GLB`, `3MF`, or `OFF` mesh is also accepted. A mesh is only rescaled, so rod diameter, segments, facets, and the closed-curve setting do not apply. The input mode is taken from the file extension, then from the file's first bytes, and can be forced with `--as`.

```bash
python3 curve_it_lib/xyz2model.py curve.xyz --info          # report only, writes nothing
python3 curve_it_lib/xyz2model.py curve.xyz -d 2.0 -s 0.25  # curve-D2.0.stl, curve-D2.0.glb
python3 curve_it_lib/xyz2model.py curve.xyz -d 1.5 --split --preview
python3 curve_it_lib/xyz2model.py model.stl --as mesh -s 0.5   # model-S0.5.stl
```

**Multicolor Split...** opens `curve_it_lib/multicolor_split.py`, which turns a PDB or mmCIF model into one printable solid per colour for a multi-material printer. Each part is one chain, a group of chains, or any set of atoms written as an atom selection in ChimeraX's or UCSF Chimera's convention, and together the part STLs tile the model's ChimeraX `molmap` surface exactly, with no gap, no overlap, and no wrong-colour slivers. This tool requires a local ChimeraX, which it runs headless, and `scipy`, `trimesh`, and `manifold3d`.

The split is built so that the parts cannot leave a gap between them. One molmap of the whole and one per part are made on the same grid, 0.5 Angstrom by default, which is fine enough to resolve the colour interface. Each part gets an ownership field that equals the whole map wherever that part is densest and drops steeply across the boundary to a neighbour. Two solids contoured independently from such fields leave a void about 0.25 Angstrom wide between them, which prints as a visible groove, so the parts are peeled in order instead: each part is what is left of the whole intersected with its own field, and the last part is the remainder. Every colour boundary therefore has one side built from a field and the other as its exact complement. The contour level is scanned around molmap's own level and accepted only when every check passes: the whole and every part's field solid are closed 2-manifolds with the expected number of pieces, and the parts sum to the whole within 0.001 % with no overlap and no void. `surface dust` removes specks and fills small cavities of the whole without ever removing its largest piece or a piece that carries a part's main piece, and the exported files are read back from disk and checked again.

From Curve It, the tool opens with the curved **Output PDB** when that file exists, and otherwise with the loaded input PDB. Its default output folder, `<structure stem>_molmap<resolution>_split/`, is made in the working directory, and the launcher starts the tool in that PDB's folder, so the results land beside the model. Every file starts with the same tag, `<structure stem>_molmap<resolution>_split` unless `--tag` gives another; without `--out` the tag also names the folder:

```text
<tag>_p1_chainA.stl, <tag>_p2_chainC.stl, ...   the parts, one per colour, all in one frame
                                                (<tag>_p1_bbA.stl for a selection named bbA)
<tag>_pads.stl                                  with --pads: every pad, as one more part to print with the parts
<tag>_whole.stl                                 the whole molmap, for reference; do not print it with the parts
<tag>_report.txt / .json                        level scan, validation, printing notes, REPRODUCE command
<tag>_preview.png / .glb                        colour previews
<tag>_work/                                     ChimeraX jobs and logs; <tag>_work_failed/ if a run did not finish
```

Outputs are staged, and only once the report is written does a re-run with the same tag move that tag's previous set into `_previous_<timestamp>/` and then move the new set in, so the folder holds one complete set per tag and never two runs' files mixed. A run that stops with an error or is interrupted leaves the previous set in place, while one that exits with `3` replaces it like any other, so read its report before printing; other files there are never touched. The exit code is `0` when every check passes, warnings allowed, `3` when the files were exported but a check FAILED, `1` for an error, `2` for bad options, and `130` when stopped.

STL units are Angstrom and slicers read them as millimetres, so the default is 1 Angstrom to 1 mm; `--scale` changes that. Import **all the part STLs at once**: in Bambu Studio this makes one object with one sub-model per part, already in register, so give each part its own filament and never move the parts apart. Models are laid flat by default, the longest principal axis along X and the shortest vertical, since purge scales with the number of layers; `--no-lay-flat` keeps the PDB frame. Hydrogens are left out of the maps by default, and `--keep-hydrogens` puts the file's hydrogens in; a model with none, as most are, is not changed by that. Both defaults are new in 1.7.0: a command or REPRODUCE line from an earlier version, made without `--lay-flat`, builds its set again byte for byte with `--no-lay-flat` added (and `--keep-hydrogens`, if its model had hydrogens). A set laid flat before 1.7.0 may come out turned or upside down, since 1.7.0 fixes the axes' signs, and cannot be rebuilt byte for byte. `--lay-flat` and `--no-hydrogens` are still accepted. Every REPRODUCE line now spells out both. The colour boundary sits about 0.1-0.2 mm into the earlier part of each pair at 1:1, because that part carries the field-built face; reorder the parts to choose which side of a boundary takes it. Where the complement closes to zero thickness a part touches itself along a line; the tool moves each vertex copy there into its own solid by 1/1000 of the median edge (about 0.5 µm at 1:1) before writing, so the STLs carry no non-manifold edges and Bambu Studio and similar slicers take them as they are. `--keep-pinch-edges` writes them joined, as versions before 1.4.2 did.

A part can also be a set of atoms rather than whole chains: give `--part` once per colour, in peeling order, each an atom selection, optionally named as `NAME=selection` for its file name, and the word `rest` for every atom no other part takes. `--part 'bbA=/A & backbone' --part 'bbC=/C & backbone' --part rest` gives each strand's sugar-phosphate backbone its own colour and the bases of both a third. Selections are written in ChimeraX's convention by default; `--selection-syntax chimera` takes UCSF Chimera's instead, which UCSF Chimera itself reads, so `:.A@P,OP1,OP2,O5',C5',C4',O4',C3',O3',C2',C1'` there is the same backbone, and both conventions give byte-identical STLs from the same atoms. An atom two selections share goes to the earlier part, and the log and the report say how many each pair shares; `--overlap error` refuses such a run instead. In the window, **Parts by** switches between chains and atom selections, and **Backbone per chain** fills in each chain's backbone and `rest` in either convention.

Parts that are separate blobs, such as each chain's phosphate groups, can instead be cut as **blob parts** (`--blobs`, or **Blob parts** in the window). Every part but the last is then its own molmap surface, the surface its atoms' density alone has at the contour level, clipped to the whole, and the last part is the whole minus the blobs, so the blobs sit in sockets in it. A blob is rounder than an ownership part, and reaches as far into its neighbours as its own density does: on the switchback666 duplex at 3.7 A, chain A's phosphate groups make 1,474 A^3 as blobs against 1,274 A^3 by ownership. No K is needed, and the level scan stops at the nearest clean level, so a 210 bp supercoil's phosphate split took 184 s as blob parts, against 496 s by ownership with v1.5.0. Where a blob faces outwards its own surface lies a hair inside the whole's, which would leave the last part a near-zero film over it in the files and previews; `--blob-skin` (default 0.2 A) gives such skins to the blob, and `--blob-skin 0` keeps each blob exactly its own molmap surface.

A small blob is held only by the faces it shares with its neighbours, so where the slicer's support touches it, pulling the support off can take the blob along. With `--pads` (**Pads under blobs that touch the support** in the window, with blob parts) the tool also writes `<tag>_pads.stl`: a pad under every blob face the support would touch, so the support meets the pad instead. A blob face is taken to touch the support when it is one of the whole's faces, faces down at a slope from horizontal below `--support-angle` (default 35; the number in Bambu Studio's Support > Threshold angle, to which Bambu adds 1, as the tool does; the slope is taken from the blob's surface averaged over 1 Angstrom) and stands more than 0.2 mm above the bed. Under each such face the pad fills the space straight below it, `--pad-thickness` deep (default 0.8 Angstrom, to the bed at most), and shares the blob's faces exactly. A face whose pad would come within 0.5 Angstrom of anything but its own blob, the other parts or the other blobs of its own part, is left bare, so a pad touches only its blob; the clearance is measured exactly from samples of each pad's column with a 5 % margin, and confirmed on each finished pad with manifold3d's `min_gap`; a pad that fails is rebuilt alone with a wider margin, and pads under 0.5 mm^2 are dropped. A blob's own cavity counts as the blob. With `--blob-skin 0` the last part's film covers most blob faces, so few pads are made, and a note says so. All the pads are one part, one more sub-model in the slicer: give it a filament that does not bond to the parts', PVA or BVOH, which dissolve, or PETG with PLA parts, which breaks away, since a pad in the parts' own filament fuses to its blob. The pads are built for the exported orientation and size, so give the print scale with `--scale` and do not tilt, flip or scale the object in the slicer; turning it about Z or moving it is fine. The report's PADS section gives, per blob part, how many blobs touch the support and how many are padded, the padded area, the closest approach and the layers the pads are in, and its checks confirm that the pads are closed, keep 0.5 Angstrom from the other parts and blobs, and lie outside the model, warn when blobs touch the support but no pad fits, and warn when a pad is under two 0.2 mm layers or its gap is narrower than a 0.4 mm line at the export scale. On a 210 bp supercoil at 3.7 Angstrom, laid flat, 242 of the 337 phosphate blobs the support would touch got pads, 248 in all.

Before opening the window, the Curve It launcher checks for ChimeraX and the three packages, and if something is missing it says so and asks whether to open the tool anyway. Every field and checkbox in the window has a light-blue `?` beside it that opens an explanation, with an example for every entry field.

```bash
python3 curve_it_lib/multicolor_split.py MODEL.pdb                   # every chain its own colour
python3 curve_it_lib/multicolor_split.py MODEL.pdb --parts A C       # choose and order the parts
python3 curve_it_lib/multicolor_split.py MODEL.pdb --parts A,B C     # chains A and B share one colour
python3 curve_it_lib/multicolor_split.py MODEL.pdb --resolution 3.7  # more detail, few holes; K rescaled and checked
python3 curve_it_lib/multicolor_split.py MODEL.pdb --no-lay-flat     # keep the PDB frame (laid flat by default)
python3 curve_it_lib/multicolor_split.py MODEL.pdb --part 'bbA=/A & backbone' --part 'bbB=/B & backbone' --part rest
                                                                     # atom selections as parts, ChimeraX's convention
python3 curve_it_lib/multicolor_split.py MODEL.pdb --selection-syntax chimera \
    --part "bbA=:.A@P,OP1,OP2,O5',C5',C4',O4',C3',O3',C2',C1'" --part rest   # Chimera's convention
python3 curve_it_lib/multicolor_split.py MODEL.pdb --blobs \
    --part "phosA=/A@P,OP1,OP2,O5',O3'" --part "phosB=/B@P,OP1,OP2,O5',O3'" --part rest   # phosphate blobs
python3 curve_it_lib/multicolor_split.py MODEL.pdb --blobs --pads --allow-multi-shell \
    --part "phosA=/A@P,OP1,OP2,O5',O3'" --part "phosB=/B@P,OP1,OP2,O5',O3'" --part rest   # + pads under them
python3 curve_it_lib/multicolor_split.py MODEL.pdb --gui             # the window, pre-filled
```

## DiLiuLab Palette

Every tool that tells components, chains, or parts apart by colour can use the DiLiuLab figure palette instead of its own colours. The palette is the lab's nine figure colours from gr_colors (https://github.com/DiLiuLab/gr_colors), always in the lab's order: red, blue, magenta, cyan, orange, purple, green, yellow, mint green. It comes at five tint levels. T100 is the full colour, and T80, T60, T40, and T20 are the same colours mixed toward white, so T20 is nearly white and T40 and T20 can be hard to see on a white background. Each tint also has a matching neutral, black at T100 and gray below. The tools use only the nine colours: the first item is red, the second blue, and so on, starting again at red after the ninth. None of them uses the neutral.

The palette is stored once, in `assets/diliulab_colors.json`, and read through `curve_it_lib/lab_colors.py`; no tool keeps its own copy of the colours. Any other script can read the JSON directly or import the reader:

```python
from curve_it_lib import lab_colors                 # plain "import lab_colors" inside curve_it_lib/
lab_colors.lab_colors("T80")                        # ['#eb7070', '#7094eb', '#eb70db', ...] in lab order
lab_colors.lab_named_colors("T100", neutral=True)   # [('red', '#e64c4c'), ..., ('black', '#000000')]
lab_colors.resolve_lab_color("MintGreenT80")        # '#70ebad'; None for anything that is not a lab name
```

The values come from gr_colors V3.3's own formula: golden-ratio HSV hues at saturation 0.67 and value 0.90, the lab's indexes 0, 1, 3, 4, 5, 6, 7, 10, and 12, and each tint mixed toward white with gr_colors' rounding. So T100 matches the table in the gr_colors README, and every tint matches gr_colors' shipped `grcT100.clr` to `grcT20.clr` colour lists swatch for swatch. If the asset is missing or cannot be read, the reader falls back to the formula.

Curve It does not depend on gr_colors. Nothing in this package imports, runs, or reads the gr_colors script: the colours live in the asset, the fallback formula is `lab_colors.py`'s own code, and the tests check both against values recorded from gr_colors V3.3. A gr_colors checkout is therefore never needed, and a later gr_colors release cannot change these colours; to adopt new lab colours, edit or regenerate the asset.

```bash
python3 curve_it_lib/lab_colors.py                # print every tint, each colour beside its name
python3 curve_it_lib/lab_colors.py --name MintGreenT80   # one colour's hex code
python3 curve_it_lib/lab_colors.py --write-asset  # regenerate the asset from the formula
```

A single lab colour can also be named wherever a tool takes a colour value, spelled the way the lab's own colour lists spell it: the colour's name followed by its tint, `RedT80`, `MintGreenT60`, `GrayT40`, `BlackT100`. Case, spaces, `_`, and `-` are ignored, so `mint green T80` works too, and `DiLiuLab red` means `RedT100`. The tint is what marks a name as the lab's: `red` on its own keeps its usual meaning, pure red, everywhere it meant that before, so no colour value that worked before changes meaning. A lab name with a tint the palette does not hold, such as `RedT75`, is refused with the tints listed. Names are accepted by **Convert XYZ...** and `--convert-color`, one per component and mixed freely with the other forms (`RedT80,BlueT80` or `RedT80, #00ff00`), by XYZ to 3D Model's `--colors` (`RedT80,#3cb44b,MintGreenT60`), and by every colour Plane It takes: its window's **Fill** and **Color** fields, the `fill`, `stroke`, and `line_stroke` style keys, `--chain-colors`, and the underlay, xy-plane, scale-bar, and base-pair colour options, each resolved to its hex code before the SVG is written. Plane It's `--style` and `--chain-colors` split their values at spaces, so write a name there without them: `MintGreenT80`, `A=RedT100`.

Every tool keeps its own colours as the default, unchanged. `--palette DiLiuLab` picks the T100 colours, and `--palette DiLiuLab-T80`, `-T60`, `-T40`, or `-T20` picks a tint. Case and the separator are ignored, so `"DiLiuLab T80"` and `diliulab80` are read the same way, and `none` or `auto` mean `default`; any other name, or a tint such as `DiLiuLab-T50`, is refused. Each tool's window has a matching palette menu, explained in the window's own help. A tool records the palette where it already records its settings, such as a report line, the JSON, SVG metadata, or a `REPRODUCE` line, but only when it is not `default`. What the palette colours in each tool:

- **XYZ to 3D Model**: the GLB components and the PNG previews; the STL has no colour. `--colors` wins over `--palette`. The window's menu sits with the component swatches and refills them.
- **SVG to XYZ**: the components in the Preview window and the `--preview` PNG. The `.xyz` files carry no colour.
- **KnotPlot to XYZ**: a link's components in the Preview windows. A knot's single component looks the same either way, and the `.xyz` files carry no colour.
- **View curve** (`view_xyzV3.py`): the components of a multi-component file. A single-component curve keeps its gray line with points coloured by position, and the red start and black end markers never change. In Curve It, the menu sits under **View curve** and changes only the view.
- **Convert XYZ...** and `curve_it.py --convert`: the VECT colours. There is no separate option, because the palette's name is itself a colour value, `--convert-color DiLiuLab-T80`. The file's colour comment names the palette.
- **Plane It**: the chain colours with `--color-by chain`, and each atom type's fill and line colour with `--color-by atom-type`. They are the starting colours the window shows, in the swatches beside its menu and in each row's **Fill** and **Color**, and any of them can be changed; `--chain-colors` and `--style` change them on the command line.
- **Multicolor Split**: the part colours of the preview PNG and GLB, the window's swatches, and each part's colour name in the log, report, and JSON. The STLs carry no colour.
- **Local curvature/torsion**: the pop-up plot's three traces, curvature red, torsion blue, and local writhe density magenta. The CSV does not change.

The other tools draw nothing in categorical colours, so they have no palette. Colour maps that run continuously along a curve, and fixed interface colours, are left as they were.

```bash
python3 curve_it_lib/xyz2model.py link.xyz -d 2.0 --preview --palette DiLiuLab-T80   # A light red, B light blue, C light magenta
```

## Outputs

- The curved PDB is written to `-o/--output-pdb`, or to `<input>_curved.pdb` if no output path is given.
- If a user-supplied curve is rescaled, a sibling `<curve>_rescaled.xyz` file is written.
- If interpolation is enabled in the GUI, a sibling `<curve>_interpolated.xyz` file is written; this interpolated curve is also the curve used to fit the output PDB.

## Protein PDB Support

Protein PDBs can be handled when the structure has a meaningful roughly straight principal axis, such as an alpha helix, coiled coil, or elongated filament. Protein residues are mapped as whole rigid residues. This is not a protein-folding tool, and compact globular proteins may not produce a useful result because a single principal axis is a poor description of their shape.

## Make The Script Executable

On macOS/Linux, make the script directly executable:

```bash
chmod +x curve_it.py plane_it.py
./curve_it.py --version
./curve_it.py input.pdb curve.xyz -o output_curved.pdb
./plane_it.py --help
```

To run them from anywhere, place this repo folder on your `PATH`, or create small wrapper scripts that call the full paths to `curve_it.py` and `plane_it.py`.

## Build A Standalone Executable

PyInstaller is one common option:

```bash
python3 -m pip install pyinstaller
python3 -m PyInstaller --onefile --name curve_it --add-data "assets:assets" --add-data "curve_it_lib:curve_it_lib" curve_it.py
python3 -m PyInstaller --onefile --name plane_it --add-data "assets:assets" --add-data "curve_it_lib/plane_itV3_8.py:curve_it_lib" --add-data "curve_it_lib/lab_colors.py:curve_it_lib" --add-data "curve_it_lib/vect_io.py:curve_it_lib" plane_it.py
```

For a GUI-style app bundle, you can add `--windowed`. On macOS, PyInstaller's `--icon` option expects an `.icns` file, so PNG files in `assets/` are included as GUI/task-menu assets but are not required for the scripts to run. The `assets/` folder includes optional task-menu/window icons for Curve It, Plane It, and the helper tools, and `diliulab_colors.json`, the DiLiuLab palette, which `lab_colors.py` recomputes from its formula if the file is missing. If the Plane It implementation file is updated later, replace `plane_itV3_8.py` in the PyInstaller command with the current `plane_itV*.py` file. `plane_it.py` loads that file by path, so PyInstaller cannot see what it imports: `lab_colors.py` and `vect_io.py` are added beside it by hand, and a Plane It built without them offers only the default palette and reads a `.vect` file as plain XYZ.

The scripts check for their icons at runtime and continue normally if an icon is missing.

In a one-file build there is no Python interpreter to hand a script to, so Multicolor Split's window runs each split as a child process through `curve_it --multicolor-split ...`, and Curve It forwards everything after that flag to the tool. ChimeraX stays an external program and is never bundled, and `scipy`, `trimesh`, and `manifold3d` must be importable by the Python that runs PyInstaller.

## Helper Modules

Supporting scripts live in `curve_it_lib/`:

- `interpolate_xyz.py`
- `cal_xyz_total_curvature_writheV2.py`
- `cal_xyz_local_curvature_torsionV3_1.py`
- `generate_helix_xyzV2.py`
- `generate_sc_xyzV3_8.py`
- `get_curve_it_phaseV5_1.py`
- `curved_connectorV3_4.py`
- `plane_itV3_8.py` (versioned Plane It implementation; use `plane_it.py` as the stable launcher)
- `kp2xyz.py`
- `svg2xyz.py`
- `view_xyzV3.py`
- `xyz2model.py`
- `vect_io.py` (Geomview VECT reading and writing, shared by every tool above)
- `multicolor_split.py` (Multicolor Split; reads PDB or mmCIF rather than curve files)
- `lab_colors.py` (the DiLiuLab figure palette, read from `assets/diliulab_colors.json` and shared by every tool that offers the palette)

They can still be run directly, for example:

```bash
python3 curve_it_lib/interpolate_xyz.py curve.xyz --n 400
python3 curve_it_lib/cal_xyz_total_curvature_writheV2.py curve.xyz
python3 curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py curve.xyz --no-plot
python3 curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py --example-trefoil --no-plot
python3 curve_it_lib/generate_helix_xyzV2.py -R 10 -c 2 -L 200 -o helix.xyz
python3 curve_it_lib/generate_sc_xyzV3_8.py -L 1071 -w -3 -n 2000
python3 curve_it_lib/kp2xyz.py 4.1 -o out
python3 curve_it_lib/svg2xyz.py drawing.svg --info
python3 curve_it_lib/view_xyzV3.py curve.xyz
python3 curve_it_lib/view_xyzV3.py multi_component.txt --components A,C
python3 curve_it_lib/xyz2model.py curve.xyz -d 2.0 -s 0.25
python3 curve_it_lib/multicolor_split.py MODEL.pdb --parts A C
python3 curve_it_lib/lab_colors.py --tint T80
```

Every one of these accepts a Geomview VECT file wherever it accepts a curve file, and takes the closure it states rather than measuring or assuming it:

```bash
python3 curve_it.py helix.pdb knot.vect                    # --path-type comes from the file
python3 curve_it_lib/interpolate_xyz.py knot.vect --n 400  # closed loop; writes knot_interpolated.xyz
python3 curve_it_lib/cal_xyz_total_curvature_writheV2.py knot.vect
python3 curve_it_lib/view_xyzV3.py link.vect --components A,C
python3 curve_it_lib/xyz2model.py knot.vect -d 2.0         # no end caps on a closed component
python3 curve_it_lib/plane_itV3_8.py link.vect --atom-types X   # polylines become chains A, B, C
```

## License

MIT License. See `LICENSE`.
