# Changelog

## V2.4 - 2026-06-17

- Renamed the main script from `curve_naV2.3.py` to `curve_it.py`.
- Moved helper modules into `curve_it_lib/` and updated imports accordingly, while preserving existing helper version suffixes such as `V2`.
- Added `-v` and `--version` support reporting `curve_it V2.4`.
- Added protein-aware grouping: nucleic acid residues use phosphate/sugar/base groups, while protein and other residues use whole-residue rigid groups.
- Updated the GUI title to `re_helix is AZBMOST Package Module #3 - Fit PDB along Any Curve`.
- Added optional GUI icon loading from `assets/icon.png`; the script continues normally if the icon is absent.
- Added a knot-style project icon at `assets/icon.png`.
- Added provenance `REMARK 900` lines at the start of generated PDB files.
- Added light-blue `?` help buttons with pop-up explanations and examples for GUI arguments.
- Added a scrollable run log window at the bottom of the GUI.
- Reorganized GUI fields to reduce vertical space and narrowed short input fields.
- Bolded GUI section titles for emphasis.
- Added GitHub-ready repository files: `README.md`, `LICENSE`, `requirements.txt`, `.gitignore`, and this changelog.

## V2.3 and earlier

- Original Curve NA script lineage for fitting straight nucleic acid PDB structures along 3D curves.
