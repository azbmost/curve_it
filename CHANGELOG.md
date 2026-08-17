# Changelog

## Unreleased

- Replaced the previous Generate SC implementations with standalone **Generate SC V3_1** at `curve_it_lib/generate_sc_xyzV3_1.py`, now the package's only `generate_sc_xyz*` supercoil generator.
- Added projection-robust integer-writhe geometry: shortened arm-phase families, exact end-loop writhe refitting, deterministic generic-view signed-crossing screening, and final phase/fraction/histogram diagnostics while retaining the fixed-`xz` `|W|` crossing check.
- Retained the `max-local`, `total`, `bending-energy`, `equal-lobes`, and manual opening-angle modes; `bending-energy` minimizes `integral kappa(s)^2 ds`, while `equal-lobes` deliberately uses the legacy untrimmed arm phase.
- Added a final exact-polyline loop-control correction so the default 2000-point output remains within writhe tolerance through the supported `|W| <= 10` range.
- Updated the Curve It GUI launcher, frozen-app dispatch, help, README, and CLI examples for V3_1.

## V3_8 - 2026-08-15

- Changed Generate SC to smooth its final centerline once before writing, rescale that smoothed path to the requested contour length, and verify writhe/curvature directly from the serialized geometry without smoothing it again.
- Increased Generate SC's default output density from 1000 to 2000 unique periodic points to reduce Curve It's discrete closed-frame transport error at modest runtime cost.
- Unified Generate SC verification and Curve It CLI/GUI reporting on an exact solid-angle writhe sum for the same closed polyline used by mapping; periodic-spline writhe remains only an internal fast search heuristic.
- Replaced projection-based frame propagation with exact minimal rotations between successive polyline tangents, eliminating accumulated artificial twist and making closed-frame holonomy agree with mapped-polyline writhe.
- Bumped Generate SC to **V2_1** and renamed its implementation to `curve_it_lib/generate_sc_xyzV2_1.py`.
- Added the `equal-lobes` automatic opening-angle objective, which matches terminal- and middle-lobe z-spans in the fixed `xz` projection for integer `|W| >= 2`.
- Defined terminal projected height from each outermost crossing to its z-extreme tip and middle projected height between adjacent crossings; added a serialized-coordinate equality check with 0.2 percent relative tolerance.
- Added top, bottom, mean terminal, middle, absolute-mismatch, relative-mismatch, and equality-check lines to Generate SC reports whenever integer `|W| >= 2`.
- Added `--angle-objective` as the preferred CLI name while retaining `--curvature-objective` as an alias, and updated the GUI, frozen-app dispatch, Curve It help, README, and version metadata to `V3_8`.
- Replaced Generate SC's non-selectable report label with a read-only, scrollable text log that supports mouse selection, Select All, and clipboard copying with Command-C or Control-C.

## V3_7 - 2026-08-15

- Replaced Generate SC V1 with `curve_it_lib/generate_sc_xyzV2.py` and bumped the helper to **Generate SC V2**.
- Added two automatic opening-angle objectives while retaining the user-provided opening-angle option: minimize the largest local curvature (default) or minimize total curvature.
- Added a deterministic coarse-to-fine opening-angle search over the documented 5-85 degree interval while preserving the requested length, fitted Gauss writhe, and integer fixed-`xz` crossing verification.
- Added total curvature, largest local curvature, minimum local bend radius, selected mode, selected opening angle, and angle-search evaluation count to CLI and GUI reports for automatic and manual modes.
- Updated the Generate SC GUI, command-line interface, frozen-app dispatch, Curve It help, and README examples for V2, and bumped Curve It version metadata to `V3_7`.

## V3_6 - 2026-08-14

- Added `curve_it_lib/generate_sc_xyzV1.py` as **Generate SC...** under the Curve It GUI **Other tools** area.
- Added generation of closed plectonemic supercoil-axis XYZ curves fitted to a requested contour length and signed Gauss writhe.
- For integer writhe requests, added verification that the fixed `xz` projection contains exactly `|W|` visible crossings; documented that this count and continuous Gauss writhe are distinct quantities even though Generate SC satisfies both constraints.
- Made the plectoneme opening angle controllable from the Python API, command line, and GUI, with 25 degrees as the default.
- Added separate PCA-plane reporting, including an explicit note when PCA selects `yz` rather than the guaranteed educational `xz` projection.
- Added contour-landmark reporting for both z-extreme termini and both centerline peaks of every interior `xz` lobe, with XYZ row numbers and closed-length percentages measured from the first output row.
- Revalidates the decimal coordinates that will actually be written, rejecting insufficient output precision instead of reporting a false length/writhe/crossing pass.
- Added frozen-app dispatch so **Generate SC...** also launches from the documented PyInstaller one-file build.
- Corrected closed-curve length reporting so the final-to-first seam is included when the input does not repeat its first point.
- Wrapped the **Other tools** launchers into two rows and bumped Curve It version metadata to `V3_6`.

## V3_5 - 2026-07-14

- Added `curve_it_lib/get_curve_it_phaseV5_1.py` for the integrated **Get phase** helper.
- Added a real-atom/virtual-atom selector to Get phase. Virtual atoms are arbitrary source-space `x,y,z` points and do not need to exist in the input PDB.
- Added a real-atom count field with dynamic per-atom selection rows; when more than one real atom is selected, Get phase uses their centroid.
- Added CLI support for `--virtual-atom x,y,z` and repeated `--real-atom CHAIN:RESSEQ:ATOM_NAME` selections.
- Bumped Curve It version metadata to `V3_5`.

## V3_4 - 2026-07-05

- Replaced the bundled **Curved Connector...** implementation with `curve_it_lib/curved_connectorV3_4.py`.
- Added optional sampled maximum local-curvature enforcement for connector centerline optimization via `--max-local-curvature`.
- Added bounded-curvature feasibility checks, long-to-short constrained screening, stop/continue controls for curvature-infeasible shorter candidates, optimizer iteration/start/sample controls, and an optional soft timeout for difficult constrained solves.
- Updated connector outputs so ranked PDB filenames include endpoint twist mismatch as `TwMm`, and `connector_summary.tsv` includes curvature limit and margin columns when a limit is used.
- Improved the connector GUI with V3_4 curvature controls, new default screening values, streamed run-log output from a worker thread, and a responsive window during long optimizer runs.
- Set the connector GUI **Max curvature** default to `0` as the disabled state and noted that an 84 bp dsDNA ring has curvature of about `0.022 A^-1`.
- Bumped Curve It version metadata to `V3_4`.

## V3_3 - 2026-07-01

- Bumped Curve It version metadata to `V3_3`.
- Updated **Convert XYZ...** fake-PDB output so `LINK` records are written before `ATOM`/`TER` records and match the corrected LINK spacing example.
- Changed **Convert XYZ...** fake-PDB options so **Close all chains with LINK records** is enabled by default.
- Prevented in-process helper dialogs from changing Curve It's main window icon.
- Removed older Plane It implementation snapshots, keeping `curve_it_lib/plane_itV3_8.py` as the single bundled Plane It implementation.
- Added a **Convert XYZ...** scale factor field that multiplies all output coordinates before writing.

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
