# Curve It

`curve_it.py` bends a roughly straight PDB structure so its principal axis follows a user-provided 3D curve. It was originally developed for DNA/RNA helices and now also handles protein PDBs by grouping protein atoms residue-by-residue.

Version: `V2.4`  
GUI title: `re_helix is AZBMOST Package Module #3 - Fit PDB along Any Curve`

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
- Optional but recommended: `scipy` for curvature/writhe reporting
- Optional: `matplotlib` for the GUI curve viewer
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

Treat the curve as a closed loop:

```bash
python3 curve_it.py input.pdb curve.xyz --path-type closed -o output_curved.pdb
```

Use the curve without length scaling:

```bash
python3 curve_it.py input.pdb curve.xyz --scale-mode none -o output_curved.pdb
```

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

## Outputs

- The curved PDB is written to `-o/--output-pdb`, or to `<input>_curved.pdb` if no output path is given.
- If a user-supplied curve is rescaled, a sibling `<curve>_rescaled.xyz` file is written.
- If interpolation is enabled in the GUI, a sibling `<curve>_interpolated.xyz` file is written; this interpolated curve is also the curve used to fit the output PDB.

## Protein PDB Support

Protein PDBs can be handled when the structure has a meaningful roughly straight principal axis, such as an alpha helix, coiled coil, or elongated filament. Protein residues are mapped as whole rigid residues. This is not a protein-folding tool, and compact globular proteins may not produce a useful result because a single principal axis is a poor description of their shape.

## Make The Script Executable

On macOS/Linux, make the script directly executable:

```bash
chmod +x curve_it.py
./curve_it.py --version
./curve_it.py input.pdb curve.xyz -o output_curved.pdb
```

To run it from anywhere, place this repo folder on your `PATH`, or create a small wrapper script that calls the full path to `curve_it.py`.

## Build A Standalone Executable

PyInstaller is one common option:

```bash
python3 -m pip install pyinstaller
python3 -m PyInstaller --onefile --name curve_it --add-data "assets/icon.png:assets" curve_it.py
```

For a GUI-style app bundle, you can add `--windowed`. On macOS, PyInstaller's `--icon` option expects an `.icns` file, so `assets/icon.png` is included as a GUI/task-menu asset but is not required for the script to run.

The script checks for `assets/icon.png` at runtime and continues normally if it is missing.

## Helper Modules

Supporting scripts live in `curve_it_lib/`:

- `interpolate_xyz.py`
- `cal_xyz_total_curvature_writheV2.py`
- `view_xyzV2.py`

They can still be run directly, for example:

```bash
python3 curve_it_lib/interpolate_xyz.py curve.xyz --n 400
python3 curve_it_lib/cal_xyz_total_curvature_writheV2.py curve.xyz
python3 curve_it_lib/view_xyzV2.py curve.xyz
```

## License

MIT License. See `LICENSE`.
