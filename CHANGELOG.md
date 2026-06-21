# Changelog

## V2.6 - 2026-06-20

- Added Plane It as a companion GUI/CLI tool via stable launcher `plane_it.py` and versioned implementation `curve_it_lib/plane_itV3_6.py`, adapted from the provided V3.6 projection script.
- Updated `plane_it.py` to discover the newest `curve_it_lib/plane_itV*.py` implementation automatically when future versioned files are added.
- Added **Plane It...** to the Curve It GUI **Tools** area without changing existing Curve It processing behavior.
- Added the supplied Plane It icon at `assets/plane_it_icon.png`.
- Updated Plane It branding in user-facing docstrings, CLI help, GUI title/messages, and projection-basis PDB remarks.
- Updated README with Plane It GUI/CLI usage, DSSR default-output behavior, blank-line coordinate component handling, and PyInstaller packaging examples.
- Added `curve_it_lib/cal_xyz_local_curvature_torsionV3_1.py` as a Curve It tool for local curvature, regularized torsion, and local writhe-density CSV output.
- Added a dedicated GUI **Tools** area containing **Convert XYZ...** and **Local curvature/torsion...** for current and future utilities.
- Added a quick-loading and quick-running trefoil example to the local curvature/torsion tool: `x=(2+cos(3t))cos(2t)`, `y=(2+cos(3t))sin(2t)`, `z=sin(3t)`.
- Added cursor hover reporting in **View curve** so the plot footer shows normalized path location `u` from `0` to `1` when the cursor is close to the curve.
- Revised `cal_xyz_total_curvature_writheV2.py` to strip duplicated closed-curve endpoints before smoothing/spline fitting and to use normalized chord-length spline parameterization.
- Updated in-memory closed-curve smoothing in `curve_it.py` to strip duplicated final points before smoothing.

## V2.5 - 2026-06-20

- Added component-aware parsing for plain XYZ/txt curve files separated by blank lines; components are labeled `A`, `B`, `C`, and so on.
- Added CLI `--curve-components` selection with support for values such as `A`, `B,C`, `A-C`, and `all`.
- Added GUI component summary and **Select components...** dialog; all components remain selected by default for backward compatibility.
- Updated **View curve** to use `curve_it_lib/view_xyzV3.py`, including component checkboxes plus **All components** and **Selected** view controls.
- Added selected curve component labels to generated PDB `REMARK 900` provenance lines.
- Confirmed molecular XYZ files remain usable as curve inputs and are treated as a single component.
- Kept optional GUI icon loading from `assets/icon.png`; the app still runs if the icon asset is absent.

## V2.4 - 2026-06-17

- Renamed the main script from `curve_naV2.3.py` to `curve_it.py`.
- Moved helper modules into `curve_it_lib/` and updated imports accordingly, while preserving existing helper version suffixes such as `V2`.
- Added `-v` and `--version` support reporting `curve_it V2.4`.
- Added protein-aware grouping: nucleic acid residues use phosphate/sugar/base groups, while protein and other residues use whole-residue rigid groups.
- Updated the GUI title to `AZBMOST Package Module #3 - Curve It: Sculpt PDB Structures Along Any 3D Curve`.
- Added optional GUI icon loading from `assets/icon.png`; the script continues normally if the icon is absent.
- Added a knot-style project icon at `assets/icon.png`.
- Added provenance `REMARK 900` lines at the start of generated PDB files.
- Added light-blue `?` help buttons with pop-up explanations and examples for GUI arguments.
- Added a scrollable run log window at the bottom of the GUI.
- Added the current run's CLI-equivalent command to the GUI run log.
- Added a GUI `Convert XYZ...` tool for coordinate XYZ <-> molecular XYZ conversion.
- Replaced the converter's Yes/No prompt with explicit conversion-direction buttons.
- Improved molecular XYZ curve parsing by explicitly skipping atom-count and comment header lines.
- Reorganized GUI fields to reduce vertical space and narrowed short input fields.
- Restored compact always-visible GUI hints for curve and mapping parameters.
- Bolded GUI section titles for emphasis.
- Added GitHub-ready repository files: `README.md`, `LICENSE`, `requirements.txt`, `.gitignore`, and this changelog.

## V2.3 and earlier

- Original Curve NA script lineage for fitting straight nucleic acid PDB structures along 3D curves.
