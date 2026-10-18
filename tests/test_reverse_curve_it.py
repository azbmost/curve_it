"""Forward/reverse geometry round trips and the public file/CLI interface.

Templates are generated here so this test needs no local PDB fixtures.
"""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import curve_it
from curve_it_lib import reverse_curve_it as reverse


REPO = Path(__file__).resolve().parent.parent


def template_pdb(nucleic=True):
    lines = []
    if nucleic:
        # Two strands at opposite phases, numbered antiparallel. Each chemistry
        # group has enough geometry to define a rotation, including planar bases.
        names = ["P", "OP1", "OP2", "C1'", "C2'", "O4'", "C3'", "N1", "C2", "N3"]
        offsets = np.array([[7, 0, -.3], [8, .7, .2], [7.5, -1, .1],
                            [5, 0, -.5], [5.8, 1, .1], [4.5, .6, .7], [5.5, -.5, .4],
                            [1.5, 0, .1], [2.5, 1, .1], [3.5, -.5, .1]])
        chains = ("A", "B")
    else:
        names = ["N", "CA", "C", "O", "CB"]
        offsets = np.array([[2, 0, -.8], [2.5, 1, 0], [1.5, 1.8, .9],
                            [1, 1, 1.2], [3.5, .5, .3]])
        chains = ("",)
    for chain in chains:
        for i in range(40):
            phase = 2 * np.pi * i / 10 + (np.pi if chain == "B" else 0)
            c, s = np.cos(phase), np.sin(phase)
            rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            for name, offset in zip(names, offsets):
                xyz = offset @ rot + [12, -3, 3.4 * i + 7]
                lines.append(curve_it.format_fake_pdb_atom_line(
                    len(lines) + 1, name, "DA" if nucleic else "ALA", chain,
                    40 - i if chain == "B" else i + 1, xyz, name[0]))
    return "\n".join(lines) + "\nEND\n"


def make_curve(path_type):
    if path_type == "closed":
        t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        return np.column_stack([(28 + 4 * np.cos(3*t)) * np.cos(t),
                                (28 + 4 * np.cos(3*t)) * np.sin(t), 4 * np.sin(3*t)])
    t = np.linspace(0, 1, 180)
    return np.column_stack([30 * np.sin(2*t), 15 * np.sin(3*t), 160*t])


def forward(template, path_type="open", twist=0, phase=0, start=0,
            scale_mode="curve_to_helix", curve=None):
    atoms = curve_it.parse_pdb_atoms(template)
    with contextlib.redirect_stdout(io.StringIO()):
        xyz, mapped, _ = curve_it.embed_helix_on_curve(
            atoms, make_curve(path_type) if curve is None else curve,
            path_type=path_type, helix_phase=phase, twist=twist,
            path_start=start, scale_mode=scale_mode)
    remarks = curve_it.build_generation_remarks(
        "template.pdb", "curve.xyz", scale_mode, path_type, phase, twist,
        "centroid", start, "none", 200, 0)
    body = [curve_it.update_pdb_line_coords(atom.line, point)
            for atom, point in zip(atoms, xyz)]
    return "\n".join(remarks + body) + "\nEND\n", mapped


def expected_samples(result, template, mapped, scale_mode="curve_to_helix"):
    arc = curve_it.compute_arc_lengths(mapped)
    if scale_mode in ("curve_to_helix", "helix_to_curve"):
        queries = result.stations / result.axis_length * arc[-1]
    else:
        queries = result.stations.copy()
    if result.path_type == "closed":
        queries %= arc[-1]
    queries = np.clip(queries, 0, arc[-1])
    # Use the exact forward sampler, including its tiny denominator epsilon.
    nf, bf, tf = curve_it.compute_parallel_transport_frames(mapped)
    return np.array([curve_it.sample_curve_with_frame(mapped, arc, nf, bf, tf, q)[0]
                     for q in queries])


class RecoveryTests(unittest.TestCase):
    def assert_samples(self, result, template, mapped, scale_mode="curve_to_helix"):
        expected = expected_samples(result, template, mapped, scale_mode)
        self.assertEqual(result.points.shape, expected.shape)
        # Local axis extrapolation amplifies PDB's 0.001 A coordinate rounding.
        self.assertLess(np.linalg.norm(result.points - expected, axis=1).max(), .02)
        self.assertLess(result.rmsd, .0007)

    def test_open_and_closed_nucleic_acid_and_protein(self):
        for nucleic in (True, False):
            template = template_pdb(nucleic)
            for path_type in ("open", "closed"):
                with self.subTest(nucleic=nucleic, path_type=path_type):
                    curved, mapped = forward(template, path_type, phase=71, start=.37)
                    result = reverse.recover_curve(curved, template)
                    self.assertEqual(result.path_type, path_type)
                    self.assert_samples(result, template, mapped)
                    self.assertTrue(np.all(np.diff(result.stations) >= 0))

    def test_per_atom_extra_twist_from_wrapped_remarks(self):
        template = template_pdb()
        curved, mapped = forward(template, "closed", twist=-720, phase=217, start=.23)
        result = reverse.recover_curve(curved, template)
        self.assertEqual(result.twist, -720)
        self.assert_samples(result, template, mapped)
        wrong = reverse.recover_curve(curved, template, twist=0)
        self.assertGreater(wrong.rmsd, .01)

    def test_native_spacing_and_helix_to_curve_scaling(self):
        template = template_pdb(False)
        for scale in ("none", "helix_to_curve", "200"):
            with self.subTest(scale=scale):
                curved, mapped = forward(template, scale_mode=scale)
                result = reverse.recover_curve(curved, template)
                self.assert_samples(result, template, mapped, scale)

    def test_numeric_closed_multiwrap_keeps_traversals(self):
        template = template_pdb()
        curved, mapped = forward(template, "closed", scale_mode="60", phase=41, twist=360)
        result = reverse.recover_curve(curved, template)
        self.assert_samples(result, template, mapped, "60")
        self.assertTrue(any("Multiple wraps" in note for note in result.notes))

    def test_reordered_renumbered_records_and_old_prime_names(self):
        template = template_pdb()
        curved, mapped = forward(template, phase=39)
        atoms = curve_it.parse_pdb_atoms(curved)
        records = []
        for serial, atom in enumerate(reversed(atoms), 1):
            line = atom.line
            line = line[:6] + f"{serial:5d}" + line[11:12] + line[12:16].replace("'", "*") + line[16:]
            records.append(line)
        result = reverse.recover_curve("\n".join(records), template)
        self.assert_samples(result, template, mapped)

    def test_global_rotation_and_translation_of_curved_model(self):
        template = template_pdb()
        curved, mapped = forward(template, "closed", twist=360)
        theta = .7
        rot = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        shift = np.array([52, -31, 17])
        transformed = "\n".join(curve_it.update_pdb_line_coords(a.line, a.coord @ rot + shift)
                                for a in curve_it.parse_pdb_atoms(curved))
        result = reverse.recover_curve(transformed, template, path_type="closed", twist=360)
        self.assert_samples(result, template, mapped @ rot + shift)

    def test_missing_atoms_keeps_complete_template_axis_and_group_stations(self):
        template = template_pdb()
        curved, mapped = forward(template, twist=123)
        # Remove an entire terminal residue and one atom of a sugar group.
        lines = [a.line for a in curve_it.parse_pdb_atoms(curved)
                 if not (a.chain_id == "A" and a.res_seq == 1)
                 and not (a.chain_id == "B" and a.res_seq == 12 and a.atom_name_norm == "C1'")]
        metadata = [line for line in curved.splitlines() if line.startswith("REMARK")]
        edited = "\n".join(metadata + lines)
        with self.assertRaisesRegex(ValueError, "Atom identities do not match"):
            reverse.recover_curve(edited, template)
        result = reverse.recover_curve(edited, template, allow_missing_atoms=True)
        self.assertEqual(result.missing_atoms, 11)
        self.assertEqual(result.skipped_groups, 3)
        self.assert_samples(result, template, mapped)

    def test_extra_atoms_are_ignored_only_when_requested(self):
        template = template_pdb(False)
        curved, mapped = forward(template)
        water = curve_it.format_fake_pdb_atom_line(9999, "O", "HOH", "W", 1,
                                                   np.array([0, 0, 0]), "O")
        with self.assertRaisesRegex(ValueError, "1 extra"):
            reverse.recover_curve(curved + water, template)
        result = reverse.recover_curve(curved + water, template, allow_missing_atoms=True)
        self.assertEqual(result.extra_atoms, 1)
        self.assert_samples(result, template, mapped)

    def test_noisy_coordinates_report_deformation(self):
        template = template_pdb(False)
        curved, _ = forward(template)
        rng = np.random.default_rng(8)
        noisy = "\n".join(curve_it.update_pdb_line_coords(a.line, a.coord + rng.normal(0, .15, 3))
                          for a in curve_it.parse_pdb_atoms(curved))
        result = reverse.recover_curve(noisy, template)
        self.assertGreater(result.rmsd, .1)
        self.assertTrue(any("0.1 A RMSD" in note for note in result.notes))

    def test_duplicate_and_multiple_models_are_ambiguous(self):
        template = template_pdb(False)
        curved, _ = forward(template)
        first = curve_it.parse_pdb_atoms(curved)[0].line
        for edited, message in ((curved + first, "duplicate atom"),
                                ("MODEL        1\n" + curved + "MODEL        2\n" + curved,
                                 "multiple MODEL")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                reverse.recover_curve(edited, template)

    def test_invalid_coordinates_are_not_silently_skipped(self):
        template = template_pdb(False)
        line = template.splitlines()[0]
        for field in ("     nan", "     inf", " missing"):
            edited = line[:30] + field + line[38:]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "invalid atom"):
                reverse.recover_curve(edited, template)

    def test_insufficient_geometry_is_rejected(self):
        # Points on a single line leave the axis offset/rotation unconstrained.
        line = "\n".join(curve_it.format_fake_pdb_atom_line(
            3*r+i, name, "ALA", "A", r, np.array([0, 0, 5*r+i]), name[0])
            for r in range(1, 10) for i, name in enumerate(("N", "CA", "C")))
        with self.assertRaisesRegex(ValueError, "three noncollinear"):
            reverse.recover_curve(line, line)
        ca_only = "\n".join(row for row in template_pdb(False).splitlines()
                            if row[12:16].strip() == "CA")
        with self.assertRaisesRegex(ValueError, "CA-only"):
            reverse.recover_curve(ca_only, ca_only)
        template = template_pdb(False)
        collapsed = "\n".join(curve_it.update_pdb_line_coords(a.line, np.zeros(3))
                              for a in curve_it.parse_pdb_atoms(template))
        with self.assertRaisesRegex(ValueError, "three noncollinear"):
            reverse.recover_curve(collapsed, template)

    def test_invalid_twist_and_path_type(self):
        template = template_pdb(False)
        curved, _ = forward(template)
        for kwargs, message in (({"twist": float("nan")}, "finite"),
                                ({"path_type": "guess"}, "path_type")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                reverse.recover_curve(curved, template, **kwargs)

    def test_newest_generation_metadata_wins_and_explicit_options_override(self):
        template = template_pdb(False)
        curved, _ = forward(template, "closed", twist=90)
        older = "REMARK 900 Generated by curve_it V3_8.\nREMARK 900 Options: twist=360 deg; path_type=open;\n"
        result = reverse.recover_curve(curved + older, template)
        self.assertEqual((result.twist, result.path_type), (90, "closed"))
        result = reverse.recover_curve(curved + older, template, path_type="open", twist=0)
        self.assertEqual((result.twist, result.path_type), (0, "open"))


class FileAndCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.template = self.folder / "straight template.pdb"
        self.curved = self.folder / "curved model.pdb"
        template = template_pdb(False)
        self.template.write_text(template)
        self.curved.write_text(forward(template)[0])

    def test_default_output_roundtrips_through_curve_it_reader(self):
        result, path, report = reverse.recover_files(str(self.curved), str(self.template))
        self.assertEqual(Path(path).name, "curved model_curve.xyz")
        loaded = curve_it.read_xyz_curve_from_text(Path(path).read_text())
        np.testing.assert_allclose(loaded, result.points, atol=5.1e-7, rtol=0)
        self.assertIn("Group fit RMSD:", report)

    def test_resampling_and_closed_endpoint_convention(self):
        self.curved.write_text(forward(self.template.read_text(), "closed")[0])
        result, path, _ = reverse.recover_files(str(self.curved), str(self.template), points=100)
        points = curve_it.read_xyz_curve_from_text(Path(path).read_text())
        self.assertEqual((len(points), result.path_type), (100, "closed"))
        self.assertGreater(np.linalg.norm(points[0] - points[-1]), .001)

    def test_input_files_and_hardlinks_cannot_be_overwritten(self):
        for source in (self.template, self.curved):
            before = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite"):
                reverse.recover_files(str(self.curved), str(self.template), str(source))
            self.assertEqual(source.read_bytes(), before)
        alias = self.folder / "alias.xyz"
        alias.hardlink_to(self.template)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            reverse.recover_files(str(self.curved), str(self.template), str(alias))

    def test_bad_count_does_not_touch_existing_output(self):
        output = self.folder / "keep.xyz"
        output.write_text("existing file\n")
        for n in (0, 1, -2):
            with self.subTest(n=n), self.assertRaises(ValueError):
                reverse.recover_files(str(self.curved), str(self.template), str(output), points=n)
            self.assertEqual(output.read_text(), "existing file\n")

    def test_all_three_command_line_launchers_work_from_other_directory(self):
        for command in (("reverse_curve_it.py",), ("curve_it_lib/reverse_curve_it.py",),
                        ("curve_it.py", "--reverse-curve-it")):
            with self.subTest(command=command):
                output = self.folder / "result.xyz"
                args = [sys.executable, str(REPO / command[0]), *command[1:],
                        str(self.curved), str(self.template), "-o", str(output), "-n", "100"]
                run = subprocess.run(args, cwd=self.folder, text=True, capture_output=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                self.assertEqual(len(np.loadtxt(output)), 100)

    def test_command_line_failures_have_useful_exit_codes(self):
        for arguments, code, text in ((["missing.pdb", str(self.template)], 1, "error:"),
                                      ([str(self.curved)], 2, "template_pdb")):
            run = subprocess.run([sys.executable, str(REPO / "curve_it.py"),
                                  "--reverse-curve-it", *arguments],
                                 cwd=self.folder, text=True, capture_output=True)
            self.assertEqual(run.returncode, code)
            self.assertIn(text, run.stderr)


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


@unittest.skipUnless(os.environ.get("CURVE_IT_GUI_TESTS") == "1",
                     "Set CURVE_IT_GUI_TESTS=1 with access to a Tk display.")
class GuiTests(unittest.TestCase):
    def test_generation_and_all_field_help_buttons(self):
        import tkinter as tk

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            straight, curved, output = (folder / name for name in
                                        ("template.pdb", "curved.pdb", "curve.xyz"))
            straight.write_text(template_pdb(False))
            curved.write_text(forward(straight.read_text(), "closed", twist=360)[0])
            args = reverse.build_parser().parse_args(
                [str(curved), str(straight), "-o", str(output), "--gui"])
            root = tk.Tk()
            root.withdraw()

            def exercise():
                root.update_idletasks()
                buttons = [w for w in descendants(root) if w.winfo_class() == "TButton"]
                helps = [w for w in buttons if w.cget("text") == "?"]
                self.assertEqual(len(helps), 7)
                for button in helps:
                    button.invoke()
                [w for w in buttons if w.cget("text") == "Recover curve"][0].invoke()
                self.assertTrue(output.is_file(), showerror.call_args)
                points = curve_it.read_xyz_curve_from_text(output.read_text())
                self.assertEqual(len(points), 40)
                self.assertIn("extra twist: 360 deg", output.read_text())

            try:
                with mock.patch("tkinter.Tk", return_value=root), \
                        mock.patch.object(root, "mainloop", side_effect=exercise), \
                        mock.patch("tkinter.messagebox.showerror") as showerror, \
                        mock.patch("tkinter.messagebox.showinfo") as showinfo:
                    reverse.run_gui(args)
                showerror.assert_not_called()
                self.assertEqual(showinfo.call_count, 7)
            finally:
                root.destroy()

    def test_main_gui_source_and_frozen_tool_launchers(self):
        import tkinter as tk

        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                root = tk.Tk()
                root.withdraw()
                try:
                    with mock.patch("tkinter.Tk", return_value=root), \
                            mock.patch.object(root, "mainloop"), \
                            mock.patch("subprocess.Popen") as popen, \
                            mock.patch.object(sys, "frozen", frozen, create=True):
                        curve_it.launch_gui()
                        root.update_idletasks()
                        button = [w for w in descendants(root) if w.winfo_class() == "Button"
                                  and w.cget("text") == "Reverse Curve It..."][0]
                        button.invoke()
                        command = popen.call_args.args[0]
                        self.assertEqual(command[-1], "--gui")
                        self.assertEqual("--reverse-curve-it" in command, frozen)
                finally:
                    root.destroy()


if __name__ == "__main__":
    unittest.main()
