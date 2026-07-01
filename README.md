# Curve It

`curve_it.py` bends a roughly straight PDB structure so its principal axis follows a user-provided 3D curve. It was originally developed for DNA/RNA helices and now also handles protein PDBs by grouping protein atoms residue-by-residue.

Version: `V3_3`
GUI title: `AZBMOST Package Module #3 - Curve It: Sculpt PDB Structures Along Any 3D Curve`

## What It Does

- Reads a PDB file containing a roughly straight DNA/RNA helix, protein helix, or other filament-like structure.
- Reads an optional XYZ/text curve file. If no curve is supplied, it uses a default planar ring.
- Maps the PDB onto the curve using a rotation-minimizing frame.
- Preserves local geometry with rigid group mapping:
  - nucleic acids: phosphate, sugar, and base groups
  - proteins and unknown residues: whole-residue groups
- Supports open and closed curves, curve scaling, path start control, helix phase rotation, extra twist, and optional curve interpolation.

## Requirements

- Python 3.9 or newer
- Required: `numpy`
- Optional but recommended: `scipy` for curvature/writhe reporting and the local curvature/torsion tool
- Optional: `matplotlib` for the GUI curve viewer and local analysis plots
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

**Convert XYZ...** opens a small conversion window for coordinate XYZ/txt, molecular XYZ, and fake-PDB output. Fake PDB output is meant for molecular visualization: each point becomes one atom in one residue, using residue `ALA` and atom `CA` by default. Blank-line-separated coordinate components become chains `A`, `B`, `C`, and so on; selected closed chains can be written with `LINK` records.

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

**Curved Connector...** opens `curve_it_lib/curved_connectorV3_0.py`. This tool screens curved nucleic-acid connectors between two target helical end base-pairs using a straight duplex template. It builds a practical clamped Euler-elastica proxy centerline for each candidate length, ranks candidates by destination-end fit, and writes ranked PDB assemblies plus `connector_summary.tsv`.

```bash
python3 curve_it_lib/curved_connectorV3_0.py target.pdb template.pdb \
    --source-bp A33,B1 --dest-bp E1,F33 --top-k 5
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
- `curved_connectorV3_0.py`
- `plane_itV3_8.py` (versioned Plane It implementation; use `plane_it.py` as the stable launcher)
- `view_xyzV3.py`

They can still be run directly, for example:

```bash
python3 curve_it_lib/interpolate_xyz.py curve.xyz --n 400
python3 curve_it_lib/cal_xyz_total_curvature_writheV2.py curve.xyz
python3 curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py curve.xyz --no-plot
python3 curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py --example-trefoil --no-plot
python3 curve_it_lib/generate_helix_xyzV2.py -R 10 -c 2 -L 200 -o helix.xyz
python3 curve_it_lib/view_xyzV3.py curve.xyz
python3 curve_it_lib/view_xyzV3.py multi_component.txt --components A,C
```

## License

MIT License. See `LICENSE`.
