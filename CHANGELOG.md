# Changelog

## V3_3 - 2026-07-01

- Bumped Curve It version metadata to `V3_3`.
- Updated **Convert XYZ...** fake-PDB output so `LINK` records are written before `ATOM`/`TER` records and match the corrected LINK spacing example.
- Changed **Convert XYZ...** fake-PDB options so **Close all chains with LINK records** is enabled by default.
- Prevented in-process helper dialogs from changing Curve It's main window icon.

## V3_2 - 2026-06-26

- Added `curve_it_lib/curved_connectorV3_0.py` as a bundled **Curved Connector...** GUI/CLI tool.
- Added a default Plane It SVG scale bar that reports the projected-coordinate conversion factor (`data-scale`) and draws a 10 Angstrom reference length.
- Added `curve_it_lib/plane_itV3_7.py` with dynamic greying for Plane It's xy-plane and scale-bar styling fields.
- Changed Plane It's xy-plane layer to draw the projection-basis xy-plane at projected depth 0 instead of the original input `z=0` plane.
- Removed the white halo stroke from Plane It's scale-bar SVG text for cleaner Adobe Illustrator import.
- Added `curve_it_lib/plane_itV3_8.py` with xy-plane and line-underlay enabled by default, first atom-type circle opacity defaulting to `0.2`, circle SVG object names such as `A36P`, and depth-ordered segment names such as `A36P_A37P`.
- Refined the Plane It V3.8 GUI width/height labels and made the projection-basis PDB/XYZ checkbox follow the selected projection mode by default.
- Added optional PNG icons for Curve It's helper-tool windows and dialogs.
- Changed **Convert XYZ...** from a step-by-step prompt sequence into a single conversion window, including fake-PDB output with configurable residue/atom names and optional LINK records for closed chains.
- Added a Curve It GUI Tools launcher and help entry for curved connector screening.
- Incorporated the elastica note into the connector tool documentation/help: the centerline is a practical clamped Euler-elastica proxy, and `twist_mismatch_deg` is an endpoint orientation mismatch rather than physical torsion.
- Updated README helper-module documentation and bumped Curve It version metadata to `V3_2`.

## V3_1 - 2026-06-26

- Updated Plane It's **Draw base-pair lines** controls to choose the atom used as each residue's line anchor.
- Changed the default base-pair line atom from `C1'` to `C3'`.
- Updated Plane It help text to recommend `C3'` for B-DNA and `C4'` for A-RNA.
- Updated Curve It version metadata to `V3_1`.

## V3_0 - 2026-06-25

- Added `curve_it_lib/get_curve_it_phaseV5.py` as an integrated **Get phase** helper beside the main GUI **Phase (deg)** field.
- Prefills the phase helper from the main Curve It GUI and transfers the computed phase back automatically.
- Added light-blue `?` help buttons, unit labels, and dynamic enable/disable behavior to the phase helper GUI.
- Renamed the main GUI label from **Helix phase (deg)** to **Phase (deg)**.

## V2.8 - 2026-06-23

- Added `curve_it_lib/generate_helix_xyzV2.py` as a **Generate helical curve...** tool in the Curve It GUI **Tools** area.
- Updated the generator GUI to use Curve It-style bold sections, light-blue `?` help buttons, and optional package icon loading.
- Added a Plane It `--draw-xy-plane` option and GUI control to draw an SVG group/layer named `xy-plane` with a `xy-plane-shape` polygon for the original coordinate plane `z=0`; when SVG depth ordering is enabled, the plane patch is sorted with the same back-to-front rules by mean projected corner depth.
- Updated Curve It version, README, and helper-module documentation for the new circular helix curve generator.

## V2.7 - 2026-06-21

- Changed `--scale-mode none` so it preserves both the PDB/helix length and the curve length, then maps using native PDB/helix axial spacing. Open curves must be at least as long as the PDB/helix principal-axis length; closed curves may wrap periodically.
- Kept `--scale-mode helix_to_curve` as the full-curve distribution mode for an unscaled curve.
- Updated Curve It version, CLI help, GUI help, and README wording for the new scale-mode distinction.

## V2.6 - 2026-06-20

- Added Plane It as a companion GUI/CLI tool via stable launcher `plane_it.py` and versioned implementation `curve_it_lib/plane_itV3_6.py`, adapted from the provided V3.6 projection script.
- Updated `plane_it.py` to discover the newest `curve_it_lib/plane_itV*.py` implementation automatically when future versioned files are added.
- Added **Plane It...** to the Curve It GUI **Tools** area without changing existing Curve It processing behavior.
- Added the supplied Plane It icon at `assets/plane_it_icon.png`.
- Updated Plane It branding in user-facing docstrings, CLI help, GUI title/messages, and projection-basis PDB remarks.
- Updated Plane It's DSSR runner to launch `x3dna-dssr` from the per-input `tmp_file` folder while passing absolute input/output paths, keeping DSSR sidecar files out of the launch directory.
- Updated Plane It defaults: DSSR base-pair line width is now `3.0`, and depth-ordered neighbor-line underlay extra width is now `8.0`.
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
