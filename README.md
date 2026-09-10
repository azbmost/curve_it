# Curve It

`curve_it.py` bends a roughly straight PDB structure so its principal axis follows a user-provided 3D curve. It was originally developed for DNA/RNA helices and now also handles protein PDBs by grouping protein atoms residue-by-residue.

Version: `V3_9`
GUI title: `AZBMOST Package Module #3 - Curve It: Sculpt PDB Structures Along Any 3D Curve`

## What It Does

- Reads a PDB file containing a roughly straight DNA/RNA helix, protein helix, or other filament-like structure.
- Reads an optional XYZ/text curve file. If no curve is supplied, it uses a default planar ring.
- Maps the PDB onto the curve using a rotation-minimizing frame.
- Preserves local geometry with rigid group mapping:
  - nucleic acids: phosphate, sugar, and base groups
  - proteins and unknown residues: whole-residue groups
- Reports closed-curve writhe from the exact piecewise-linear path used for mapping, using an analytic solid-angle sum over nonadjacent segment pairs. No smoothing or spline substitution is applied to mapped-polyline writhe.
- Supports open and closed curves, curve scaling, path start control, helix phase rotation, extra twist, and optional curve interpolation.

## Requirements

- Python 3.9 or newer
- Required: `numpy`
- Required by **Generate SC**; otherwise optional but recommended: `scipy` for curvature/writhe reporting and the local curvature/torsion tool
- Optional: `matplotlib` for the GUI curve viewer, local analysis plots, and the SVG to XYZ preview
- **SVG to XYZ** needs nothing beyond `numpy` and the standard library
- Required by **XYZ to 3D Model**; not needed otherwise: `trimesh` for STL/GLB mesh output
- Required by **KnotPlot to XYZ**; not needed otherwise: a local KnotPlot installation. KnotPlot is third-party software under its own license and is not bundled with this package; download it from https://knotplot.com/download/. The tool finds a normal installation on its own; set the `KNOTPLOT` environment variable to the full path of the executable if yours is elsewhere.
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

Plain coordinate files may contain multiple components separated by one or more blank lines. Curve It labels those components `A`, `B`, `C`, and so on in file order. By default, all components are concatenated in file order and used as the curve. Use `--curve-components` in CLI mode, or **Select components...** in the GUI, to choose a subset such as `A`, `B,C`, or `A-C`.

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

**Convert XYZ...** opens a small conversion window for coordinate XYZ/txt, molecular XYZ, and fake-PDB output. An optional scale factor multiplies every output coordinate before writing. Fake PDB output is meant for molecular visualization: each point becomes one atom in one residue, using residue `ALA` and atom `CA` by default. Blank-line-separated coordinate components become chains `A`, `B`, `C`, and so on; selected closed chains can be written with `LINK` records.

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

`--nbeads N` sets the number of points. KnotPlot spreads that number over the whole conformation, so on a link it is the total and not a per-component count: `--nbeads 300` on the three-component link `6.3.2` gives components of 107, 97, and 95 points. Leave it unset to export the shipped conformation at its own resolution.

Conformations are exported exactly as KnotPlot ships them, and relaxation is deliberately not offered. The tool does not relax, tighten, or smooth the geometry. Three things change the coordinates, and all three are opt-in: the reflection described below, the `--nbeads` resampling above, and `--precision`, which rounds each value as it is written. Resampling redistributes vertices along the same curve rather than reshaping it, but it does move them, so it is not a no-op even at the shipped point count: `3.1` ships 47 points, and `--nbeads 47` shifts them by up to `0.096853` in a single coordinate and takes the writhe from `+3.3722171577968423` to `+3.3710769510513137`, both measured with `curve_it_lib/cal_xyz_total_curvature_writheV2.py` at its defaults. Leave `--nbeads` unset and the shipped coordinates are what you get. These conformations are correct representatives of their knot and link types, but they are not ideal or ropelength-minimizing shapes, so do not read their length, curvature, or writhe as ideal-knot values. Relax a knot inside KnotPlot first and export it afterwards if that is what you want.

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

**Generate SC...** opens `curve_it_lib/generate_sc_xyzV3_7.py`, the package's only supercoil-centerline generator. It writes a closed plectonemic axis as plain `x y z` rows with the requested contour length and signed Gauss writhe. The default contour length is 1071 and the default output density is 2000 unique periodic points. Before writing, V3.7 applies Curve It's periodic smoothing once, resamples, and rescales to the requested closed length. It then verifies the exact decimal coordinates with the same closed-polyline segment-pair writhe calculation Curve It uses for mapping. Load the result as a Curve It curve with **Path type: closed**.

The report distinguishes the nominal scaled superhelix radius from the measured final arm radius. The latter is computed from the serialized coordinates as the median radial distance of central arm points from the translated superhelical axis, excluding the end loops and their smoothing transitions.

The arm-phase revolution count is reported as `plectoneme_phase_turns = |theta_total| / (2*pi)`. This name distinguishes the generator's complete arm-phase revolutions from a generic geometric or topological “number of superhelical turns.”

In the GUI, **Minimum final measured radius** appears before **Qualifying views** and defaults to 13. The qualifying-views field is dynamically disabled and greyed whenever arm-phase trimming is off or a minimum final radius is entered. Clear the radius field to select qualifying-view screening. The radius field is itself disabled when trimming is off.

Light-blue `?` buttons beside every generation argument and opening-angle option open concise explanations and examples without leaving the GUI.

When `--output` is omitted, V3.7 constructs a filename from the defining selections. The default is `sc_L1071_Wm3_R13_ABend.xyz`. `ABend`, `ALocal`, `ATotal`, and `AEq` identify the four automatic opening-angle objectives; a manual angle uses a token such as `A25deg`. Radius screening uses `R13`, qualifying-view screening uses a token such as `Q60`, and untrimmed geometry uses `NoTrim`. The opt-in zero-H0 family appends `H0zero`, for example `sc_L1071_Wm3_R13_ABend_H0zero.xyz`. The GUI updates this name dynamically while preserving a custom typed or browsed filename.

The default screening mode requires a final measured radius of at least 13. Set another threshold with `--minimum-final-radius LENGTH` (alias `--minimum-final-measured-radius`). Generate SC finds the largest feasible phase trim from 0 through `0.90*pi` whose radius, measured from the central 90% of the arms after smoothing, scaling, centering, and XYZ decimal quantization, remains at or above the threshold. Thus the screened geometry and the reported final radius refer to the same serialized curve that Curve It will map. Within this phase-trim family, decreasing the allowed radius permits a larger trim and generally a higher qualifying-view fraction. Supplying `--qualifying-views PERCENT` explicitly selects percentage screening instead. Radius-constrained screening requires a nonzero integer writhe and enabled trimming.

For nonzero integer writhe, V3.7 shortens the signed arm phase by a fitted trim below `1.0*pi`, refits the end-loop control distance to retain the requested exact writhe, and screens deterministic generic orthographic projections. Trimming is enabled by default; use `--no-trim` to retain the original `pi*W` phase. When percentage screening is selected, the required qualifying-view percentage defaults to 55. V3.7 rotates the complete trimmed curve around z by half of the phase removed from the legacy sweep. This restores the symmetric fixed-`xz` presentation of V2.2: odd integer writhe has reflection symmetry under `z -> -z`, and even integer writhe has central symmetry under `(x,z) -> (-x,-z)`. Because the rotation centers the shortened phase interval, all `|W|` intended xz crossings remain inside it for `phase_trim < 1`; the former `0.5*pi` limit applied to the unrotated interval. The 256 viewing directions are reflection-paired, so mirror-related positive- and negative-writhe curves receive identical finite-sample screening. For bending-energy, max-local, and total-curvature modes, projection screening holds the opening angle found at the default `0.40*pi` trim while testing progressively larger trims through `0.90*pi`. Equal-lobes mode instead resolves its opening angle at each tested trim so the terminal/middle equality remains satisfied. The final report shows whether trimming was enabled, the active radius or percentage criterion, selected phase trim, z-axis symmetry rotation, V2.2-style middle-segment peaks, lobe heights where applicable, projection statistics, and exact writhe. Fractional writhe and `W=0` use untrimmed geometry.

V3.7 supports `|W| <= 10` and four automatic opening-angle objectives over 5-85 degrees. The default `--angle-objective bending-energy` minimizes total bending energy, proportional to `integral kappa(s)^2 ds`; for a homogeneous isotropic rod with constant bending rigidity `A` and no intrinsic curvature, physical bending energy is `A/2` times this integral. `--angle-objective max-local` minimizes the largest local curvature, while `--angle-objective total` minimizes `integral kappa(s) ds`. `--angle-objective equal-lobes` matches terminal/middle fixed-`xz` lobe heights for integer `|W| >= 2` in both trimmed and untrimmed modes. Supply `-a ANGLE` or `--opening-angle ANGLE` to retain a manual angle strictly between 0 and 90 degrees. The nominal input angle satisfies `tan(alpha) = |theta| R / (H-H0)`, where `H` is the full canonical arm height and `H0` is the phase-independent axial allowance. The default remains `H0 = 2R`, for which the realized acute arm-tangent angle is `beta = atan(|theta| R/H)`, slightly smaller than `alpha`. Use `--zero-h0` to set `H0 = 0`, in which case `beta = alpha` for nonzero W. In both nonzero-writhe families, the endpoint directions supplied to the closing Bezier loops are normalized exact derivatives of the parametric arm equations; no tangent is estimated from a plot or sampled polyline. The zero-H0 choice repeats all angle, loop-control, writhe, trim, smoothing, scaling, and serialized-coordinate checks. At exactly `W = 0`, where the arm/loop parameterization collapses, V3.7 instead directly writes a regular planar ring in the xz plane. Its polygon radius is `L/[2N sin(pi/N)]`, so the unrounded N-point closed polygon has contour length L; decimal serialization is then verified exactly. Opening-angle, loop-control, trimming, smoothing, and projection screening are not applied to this ring. Neither angle is the DNA base-pair twist angle or the full angle between the two arms. The older option name `--curvature-objective` remains an alias.

You can also run Generate SC from the command line:

```bash
python3 curve_it_lib/generate_sc_xyzV3_7.py -w -3
python3 curve_it_lib/generate_sc_xyzV3_7.py --angle-objective total
python3 curve_it_lib/generate_sc_xyzV3_7.py --angle-objective max-local
python3 curve_it_lib/generate_sc_xyzV3_7.py --angle-objective equal-lobes
python3 curve_it_lib/generate_sc_xyzV3_7.py --no-trim
python3 curve_it_lib/generate_sc_xyzV3_7.py --qualifying-views 60
python3 curve_it_lib/generate_sc_xyzV3_7.py --minimum-final-radius 13
python3 curve_it_lib/generate_sc_xyzV3_7.py -a 25
python3 curve_it_lib/generate_sc_xyzV3_7.py --zero-h0
python3 curve_it_lib/generate_sc_xyzV3_7.py -w 0 --zero-h0
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
```

Plain coordinate text files may contain multiple components separated by blank lines; Plane It treats those components as chains `A`, `B`, `C`, and so on.

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

An existing `STL`, `OBJ`, `PLY`, `GLB`, `3MF`, or `OFF` mesh is also accepted. A mesh is only rescaled, so rod diameter, segments, facets, and the closed-curve setting do not apply. The input mode is taken from the file extension, then from the file's first bytes, and can be forced with `--as`.

```bash
python3 curve_it_lib/xyz2model.py curve.xyz --info
python3 curve_it_lib/xyz2model.py curve.xyz -d 2.0 -s 0.25
python3 curve_it_lib/xyz2model.py curve.xyz -d 1.5 --split --preview
python3 curve_it_lib/xyz2model.py model.stl --as mesh -s 0.5
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
python3 -m PyInstaller --onefile --name plane_it --add-data "assets:assets" --add-data "curve_it_lib/plane_itV3_8.py:curve_it_lib" plane_it.py
```

For a GUI-style app bundle, you can add `--windowed`. On macOS, PyInstaller's `--icon` option expects an `.icns` file, so PNG files in `assets/` are included as GUI/task-menu assets but are not required for the scripts to run. The `assets/` folder includes optional task-menu/window icons for Curve It, Plane It, and the helper tools. If the Plane It implementation file is updated later, replace `plane_itV3_8.py` in the PyInstaller command with the current `plane_itV*.py` file.

The scripts check for their icons at runtime and continue normally if an icon is missing.

## Helper Modules

Supporting scripts live in `curve_it_lib/`:

- `interpolate_xyz.py`
- `cal_xyz_total_curvature_writheV2.py`
- `cal_xyz_local_curvature_torsionV3_1.py`
- `generate_helix_xyzV2.py`
- `generate_sc_xyzV3_7.py`
- `get_curve_it_phaseV5_1.py`
- `curved_connectorV3_4.py`
- `plane_itV3_8.py` (versioned Plane It implementation; use `plane_it.py` as the stable launcher)
- `kp2xyz.py`
- `svg2xyz.py`
- `view_xyzV3.py`
- `xyz2model.py`

They can still be run directly, for example:

```bash
python3 curve_it_lib/interpolate_xyz.py curve.xyz --n 400
python3 curve_it_lib/cal_xyz_total_curvature_writheV2.py curve.xyz
python3 curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py curve.xyz --no-plot
python3 curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py --example-trefoil --no-plot
python3 curve_it_lib/generate_helix_xyzV2.py -R 10 -c 2 -L 200 -o helix.xyz
python3 curve_it_lib/generate_sc_xyzV3_7.py -L 1071 -w -3 -n 2000
python3 curve_it_lib/kp2xyz.py 4.1 -o out
python3 curve_it_lib/svg2xyz.py drawing.svg --info
python3 curve_it_lib/view_xyzV3.py curve.xyz
python3 curve_it_lib/view_xyzV3.py multi_component.txt --components A,C
python3 curve_it_lib/xyz2model.py curve.xyz -d 2.0 -s 0.25
```

## License

MIT License. See `LICENSE`.
