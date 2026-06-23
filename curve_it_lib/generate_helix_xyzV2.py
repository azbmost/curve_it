#!/usr/bin/env python3
"""
generate_helix_xyzV2.py

Generate a circular helical curve and save it as a plain-coordinate XYZ file.

Inputs:
  - R: helix radius
  - c: axial rise per radian in z = c*t; pitch = 2*pi*c
  - optional pitch angle alpha in degrees, where alpha = atan(c/R)
  - total_length: arc length measured along the helix
  - number of points and output file path

Output:
  - A plain-coordinate XYZ file with one point per line: x y z
    No atom names, atom counts, or molecular-XYZ header are written.

Examples:
  GUI mode:
    python generate_helix_xyzV2.py

  Use R and c directly:
    python generate_helix_xyzV2.py -R 10 -c 2 -L 200 -n 1000 -o helix.xyz

  Use R and pitch angle to calculate c:
    python generate_helix_xyzV2.py -R 10 --pitch-angle-deg 20 --derive c-from-R -L 200 -o helix.xyz

  Use c and pitch angle to calculate R:
    python generate_helix_xyzV2.py -c 2 --pitch-angle-deg 20 --derive R-from-c -L 200 -o helix.xyz

If run with no arguments, or with --gui, a Tkinter GUI will open.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

Point3D = Tuple[float, float, float]

DEFAULT_RADIUS = 10.0
DEFAULT_C = 2.0
DEFAULT_TOTAL_LENGTH = 200.0
DEFAULT_NUM_POINTS = 1000
DEFAULT_PRECISION = 8
TOOL_NAME = "Generate Helical Curve"
TOOL_VERSION = "V2"


def resource_path(relative_path: str) -> Path:
    """Return a resource path that also works from a PyInstaller bundle."""
    source_dir = Path(__file__).resolve().parent
    source_root = source_dir.parent if source_dir.name == "curve_it_lib" else source_dir
    base_dir = Path(getattr(sys, "_MEIPASS", source_root))
    return base_dir / relative_path


class ParameterResolutionError(ValueError):
    """Raised when R, c, and pitch angle cannot be resolved consistently."""


def generate_helix_points(
    radius: float,
    c: float,
    total_length: float,
    num_points: int,
    handedness: str = "right",
    phase_deg: float = 0.0,
    z0: float = 0.0,
) -> List[Point3D]:
    """Generate equally arc-length-spaced points on a circular helix.

    The helix is defined as:
        x(t) = R cos(t + phi)
        y(t) = +/- R sin(t + phi)
        z(t) = z0 + c t

    The speed is constant:
        ds/dt = sqrt(R^2 + c^2)

    Therefore, for a requested arc length L:
        t_end = L / sqrt(R^2 + c^2)
    """
    validate_inputs(radius, c, total_length, num_points, handedness)

    speed = math.sqrt(radius * radius + c * c)
    t_end = total_length / speed
    phase = math.radians(phase_deg)
    y_sign = 1.0 if handedness == "right" else -1.0

    points: List[Point3D] = []
    for i in range(num_points):
        if num_points == 1:
            t = 0.0
        else:
            t = t_end * i / (num_points - 1)
        angle = t + phase
        x = radius * math.cos(angle)
        y = y_sign * radius * math.sin(angle)
        z = z0 + c * t
        points.append((x, y, z))

    return points


def validate_inputs(
    radius: float,
    c: float,
    total_length: float,
    num_points: int,
    handedness: str,
) -> None:
    """Validate helix parameters and raise ValueError for invalid input."""
    if radius < 0:
        raise ValueError("R must be non-negative.")
    if total_length <= 0:
        raise ValueError("Total length must be positive.")
    if num_points < 2:
        raise ValueError("Number of points must be at least 2.")
    if handedness not in {"right", "left"}:
        raise ValueError("Handedness must be 'right' or 'left'.")
    if radius == 0 and c == 0:
        raise ValueError("R and c cannot both be zero, because the curve would have zero length.")


def pitch_angle_from_radius_c(radius: float, c: float) -> float:
    """Return pitch angle alpha in degrees, alpha = atan(c/R)."""
    if radius == 0:
        if c > 0:
            return 90.0
        if c < 0:
            return -90.0
        raise ValueError("Pitch angle is undefined when both R and c are zero.")
    return math.degrees(math.atan(c / radius))


def resolve_radius_c(
    radius: Optional[float],
    c: Optional[float],
    pitch_angle_deg: Optional[float],
    derive: str = "auto",
) -> Tuple[float, float, str]:
    """Resolve R and c from optional R, c, and pitch angle.

    alpha is the pitch angle measured from the horizontal circumferential
    direction after unwrapping the cylinder:
        alpha = atan(c/R)
        c = R tan(alpha)
        R = c / tan(alpha)

    derive options:
      - auto: use pitch angle only when exactly one of R or c is missing;
              if both are present, require consistency.
      - c-from-R: calculate c from R and pitch angle.
      - R-from-c: calculate R from c and pitch angle.
      - ignore: ignore pitch angle and use R and c directly.
    """
    if derive not in {"auto", "c-from-R", "R-from-c", "ignore"}:
        raise ParameterResolutionError("derive must be auto, c-from-R, R-from-c, or ignore.")

    if pitch_angle_deg is None or derive == "ignore":
        resolved_radius = DEFAULT_RADIUS if radius is None else radius
        resolved_c = DEFAULT_C if c is None else c
        return resolved_radius, resolved_c, "Using R and c directly."

    # Avoid a numerically infinite tangent at +/- 90 degrees.
    if abs(abs(pitch_angle_deg) - 90.0) < 1.0e-12:
        raise ParameterResolutionError(
            "Pitch angle cannot be exactly +/-90 degrees in this parameterization. "
            "Use R = 0 and specify c directly instead."
        )

    alpha_rad = math.radians(pitch_angle_deg)
    tan_alpha = math.tan(alpha_rad)

    if derive == "c-from-R":
        resolved_radius = DEFAULT_RADIUS if radius is None else radius
        resolved_c = resolved_radius * tan_alpha
        return resolved_radius, resolved_c, "Calculated c from R and pitch angle."

    if derive == "R-from-c":
        resolved_c = DEFAULT_C if c is None else c
        if abs(tan_alpha) < 1.0e-15:
            if abs(resolved_c) > 1.0e-15:
                raise ParameterResolutionError("A pitch angle of 0 degrees requires c = 0.")
            resolved_radius = DEFAULT_RADIUS if radius is None else radius
        else:
            resolved_radius = resolved_c / tan_alpha
        if resolved_radius < 0:
            raise ParameterResolutionError(
                "The signs of c and pitch angle are inconsistent; calculated R would be negative."
            )
        return resolved_radius, resolved_c, "Calculated R from c and pitch angle."

    # derive == "auto"
    if radius is not None and c is None:
        resolved_radius = radius
        resolved_c = radius * tan_alpha
        return resolved_radius, resolved_c, "Auto: calculated c from R and pitch angle."

    if radius is None and c is not None:
        resolved_c = c
        if abs(tan_alpha) < 1.0e-15:
            if abs(c) > 1.0e-15:
                raise ParameterResolutionError("A pitch angle of 0 degrees requires c = 0.")
            resolved_radius = DEFAULT_RADIUS
        else:
            resolved_radius = c / tan_alpha
        if resolved_radius < 0:
            raise ParameterResolutionError(
                "The signs of c and pitch angle are inconsistent; calculated R would be negative."
            )
        return resolved_radius, resolved_c, "Auto: calculated R from c and pitch angle."

    if radius is None and c is None:
        resolved_radius = DEFAULT_RADIUS
        resolved_c = resolved_radius * tan_alpha
        return resolved_radius, resolved_c, "Auto: used default R and calculated c from pitch angle."

    # Both R and c were provided. In auto mode, do not silently overwrite one.
    assert radius is not None and c is not None
    implied_angle = pitch_angle_from_radius_c(radius, c)
    if abs(implied_angle - pitch_angle_deg) > 1.0e-7:
        raise ParameterResolutionError(
            "R, c, and pitch angle are inconsistent. Use --derive c-from-R or --derive R-from-c "
            "if you want the pitch angle to overwrite one of the values."
        )
    return radius, c, "Auto: R, c, and pitch angle are mutually consistent."


def write_plain_xyz(points: List[Point3D], output_path: str, precision: int = DEFAULT_PRECISION) -> None:
    """Write points to a plain-coordinate XYZ file with columns x y z."""
    if precision < 0:
        raise ValueError("Precision must be non-negative.")

    path = Path(output_path)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)

    fmt = f"{{:.{precision}f}} {{:.{precision}f}} {{:.{precision}f}}\n"
    with path.open("w", encoding="utf-8") as f:
        for x, y, z in points:
            f.write(fmt.format(x, y, z))


def helix_summary(radius: float, c: float, total_length: float, note: str = "") -> str:
    """Return a short summary of derived helix quantities."""
    speed = math.sqrt(radius * radius + c * c)
    t_end = total_length / speed
    turns = t_end / (2.0 * math.pi)
    pitch = 2.0 * math.pi * c
    z_height = c * t_end
    alpha = pitch_angle_from_radius_c(radius, c)
    circumference = 2.0 * math.pi * radius

    lines = [
        f"R = {radius:.8g}",
        f"c = {c:.8g}",
        f"pitch P = 2*pi*c = {pitch:.8g}",
        f"circumference 2*pi*R = {circumference:.8g}",
        f"pitch angle alpha = atan(c/R) = {alpha:.8g} deg",
        f"t_end = {t_end:.8g} rad",
        f"turns = {turns:.8g}",
        f"z displacement = {z_height:.8g}",
    ]
    if note:
        lines.append(note)
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a circular helical curve as a plain-coordinate XYZ file."
    )
    parser.add_argument(
        "-R",
        "--radius",
        type=float,
        default=None,
        help=f"Helix radius R. Default when omitted: {DEFAULT_RADIUS}",
    )
    parser.add_argument(
        "-c",
        type=float,
        default=None,
        help=f"Axial rise per radian, z = c*t. Pitch = 2*pi*c. Default when omitted: {DEFAULT_C}",
    )
    parser.add_argument(
        "--pitch-angle-deg",
        type=float,
        default=None,
        help="Pitch angle alpha in degrees, where alpha = atan(c/R). Optional.",
    )
    parser.add_argument(
        "--derive",
        choices=["auto", "c-from-R", "R-from-c", "ignore"],
        default="auto",
        help=(
            "How to use pitch angle. auto derives the missing value only; "
            "c-from-R overwrites c; R-from-c overwrites R; ignore uses R and c directly. Default: auto"
        ),
    )
    parser.add_argument(
        "-L",
        "--total-length",
        type=float,
        default=DEFAULT_TOTAL_LENGTH,
        help=f"Total arc length along the helix. Default: {DEFAULT_TOTAL_LENGTH}",
    )
    parser.add_argument(
        "-n",
        "--num-points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help=f"Number of output points. Default: {DEFAULT_NUM_POINTS}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="helix.xyz",
        help="Output plain-coordinate XYZ file. Default: helix.xyz",
    )
    parser.add_argument(
        "--handedness",
        choices=["right", "left"],
        default="right",
        help="Use y = R*sin(t) for right, y = -R*sin(t) for left. Default: right",
    )
    parser.add_argument("--phase-deg", type=float, default=0.0, help="Initial angular phase in degrees. Default: 0.0")
    parser.add_argument("--z0", type=float, default=0.0, help="Initial z coordinate. Default: 0.0")
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help=f"Decimal places in output. Default: {DEFAULT_PRECISION}",
    )
    parser.add_argument("--gui", action="store_true", help="Open the graphical user interface.")
    return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> None:
    """Generate the helix using command-line arguments."""
    radius, c_val, note = resolve_radius_c(
        radius=args.radius,
        c=args.c,
        pitch_angle_deg=args.pitch_angle_deg,
        derive=args.derive,
    )
    validate_inputs(radius, c_val, args.total_length, args.num_points, args.handedness)
    if args.precision < 0:
        raise ValueError("Precision must be non-negative.")

    points = generate_helix_points(
        radius=radius,
        c=c_val,
        total_length=args.total_length,
        num_points=args.num_points,
        handedness=args.handedness,
        phase_deg=args.phase_deg,
        z0=args.z0,
    )
    write_plain_xyz(points, args.output, args.precision)
    print(f"Wrote {len(points)} points to: {args.output}")
    print(helix_summary(radius, c_val, args.total_length, note))


def run_gui() -> None:
    """Open a Tkinter GUI for helix generation."""
    try:
        import tkinter as tk
        from tkinter import filedialog, font, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError("Tkinter is not available in this Python installation.") from exc

    root = tk.Tk()
    root.title(f"{TOOL_NAME} {TOOL_VERSION}")
    root.geometry("760x680")
    root.minsize(700, 620)

    icon_path = resource_path("assets/icon.png")
    if icon_path.exists():
        try:
            icon_image = tk.PhotoImage(file=str(icon_path))
            root.iconphoto(True, icon_image)
            root._curve_it_icon_image = icon_image
        except Exception:
            pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    title_font = font.Font(root, family="Helvetica", size=16, weight="bold")
    equation_font = font.Font(root, family="Helvetica", size=13)
    mono_font = font.Font(root, family="Menlo", size=11)
    section_font = ("TkDefaultFont", 10, "bold")
    style.configure("Tool.TLabelframe.Label", font=section_font)
    style.configure("Hint.TLabel", foreground="gray35")

    help_button_kwargs = {
        "text": "?",
        "width": 2,
        "bg": "#cfefff",
        "activebackground": "#aee6ff",
        "relief": tk.RAISED,
        "borderwidth": 1,
    }
    help_texts = {
        "overview": (
            "Generate Helical Curve",
            "Generate a plain-coordinate XYZ curve for a circular helix. The output can be used directly as Curve It's curve input."
        ),
        "helix_definition": (
            "Helix Definition",
            "The curve is x(t)=R cos(t+phi), y(t)=+/- R sin(t+phi), z(t)=z0+c t. The total length is measured along the helical path, not only along z."
        ),
        "radius": (
            "Radius R",
            "Helix radius in Angstrom-like coordinate units. R controls the distance from the z axis to the curve."
        ),
        "c": (
            "Rise Per Radian c",
            "Axial z rise per radian. The pitch per full turn is 2*pi*c."
        ),
        "pitch_angle_deg": (
            "Pitch Angle Alpha",
            "Optional pitch angle in degrees, alpha=atan(c/R). Use the derive setting to decide whether alpha calculates c or R."
        ),
        "total_length": (
            "Total Length L",
            "Arc length along the helical curve. Curve It will read the generated points as the target curve."
        ),
        "num_points": (
            "Number Of Points",
            "How many equally arc-length-spaced points to write. More points give a smoother curve."
        ),
        "phase_deg": (
            "Phase Phi",
            "Initial angular phase in degrees. This rotates the starting point around the helix axis."
        ),
        "z0": (
            "Initial z0",
            "Starting z coordinate for the helix."
        ),
        "precision": (
            "Precision",
            "Number of decimal places written to the output XYZ file."
        ),
        "derive": (
            "When Alpha Is Provided",
            "Choose whether the pitch angle alpha should calculate c from R, calculate R from c, or be ignored."
        ),
        "handedness": (
            "Handedness",
            "Right-handed uses y=+R sin(t+phi). Left-handed uses y=-R sin(t+phi)."
        ),
        "output": (
            "Output File",
            "Writes a plain-coordinate XYZ file with one x y z point per line and no molecular-XYZ header."
        ),
    }

    def help_button(parent, title: str, body: str):
        return tk.Button(parent, command=lambda: messagebox.showinfo(title, body), **help_button_kwargs)

    main = ttk.Frame(root, padding=14)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)

    header_row = ttk.Frame(main)
    header_row.grid(row=0, column=0, sticky="ew")
    header = ttk.Label(header_row, text=TOOL_NAME, font=title_font)
    header.grid(row=0, column=0, sticky="w")
    help_button(header_row, *help_texts["overview"]).grid(row=0, column=1, sticky="w", padx=(8, 0))

    equation_frame = ttk.LabelFrame(main, text="Helix definition", style="Tool.TLabelframe")
    equation_frame.grid(row=1, column=0, sticky="ew", pady=(10, 12))
    equation_frame.columnconfigure(0, weight=1)
    equation_frame.columnconfigure(1, weight=0)

    equation_var = tk.StringVar()
    equation_label = ttk.Label(
        equation_frame,
        textvariable=equation_var,
        font=equation_font,
        justify="left",
        padding=(12, 8),
    )
    equation_label.grid(row=0, column=0, sticky="ew")
    help_button(equation_frame, *help_texts["helix_definition"]).grid(row=0, column=1, sticky="ne", padx=(4, 8), pady=8)

    body = ttk.Frame(main)
    body.grid(row=2, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, weight=1)

    param_frame = ttk.LabelFrame(body, text="Inputs", style="Tool.TLabelframe")
    param_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    param_frame.columnconfigure(1, weight=1)

    entries = {}

    def add_row(row: int, label: str, default: str, key: str, tip: str = "") -> None:
        label_text = label if not tip else f"{label}"
        ttk.Label(param_frame, text=label_text, anchor="w").grid(row=row, column=0, sticky="w", pady=4, padx=(8, 4))
        var = tk.StringVar(value=default)
        ent = ttk.Entry(param_frame, textvariable=var, width=22)
        ent.grid(row=row, column=1, sticky="ew", pady=4, padx=(4, 8))
        if key in help_texts:
            help_button(param_frame, *help_texts[key]).grid(row=row, column=2, sticky="w", pady=4, padx=(0, 4))
        if tip:
            ttk.Label(param_frame, text=tip, style="Hint.TLabel").grid(
                row=row, column=3, sticky="w", pady=4, padx=(0, 8)
            )
        entries[key] = var

    add_row(0, "R", str(DEFAULT_RADIUS), "radius", "radius")
    add_row(1, "c", str(DEFAULT_C), "c", "z rise per radian")
    add_row(2, "α", "", "pitch_angle_deg", "pitch angle, degrees")
    add_row(3, "L", str(DEFAULT_TOTAL_LENGTH), "total_length", "arc length")
    add_row(4, "Points", str(DEFAULT_NUM_POINTS), "num_points")
    add_row(5, "φ", "0.0", "phase_deg", "phase, degrees")
    add_row(6, "z₀", "0.0", "z0")
    add_row(7, "Precision", str(DEFAULT_PRECISION), "precision")

    derive_var = tk.StringVar(value="c-from-R")
    derive_frame = ttk.LabelFrame(param_frame, text="When α is provided", style="Tool.TLabelframe")
    derive_frame.grid(row=8, column=0, columnspan=4, sticky="ew", padx=8, pady=(8, 4))
    ttk.Radiobutton(derive_frame, text="calculate c from R and α", variable=derive_var, value="c-from-R").grid(
        row=0, column=0, sticky="w", padx=8, pady=2
    )
    ttk.Radiobutton(derive_frame, text="calculate R from c and α", variable=derive_var, value="R-from-c").grid(
        row=1, column=0, sticky="w", padx=8, pady=2
    )
    ttk.Radiobutton(derive_frame, text="ignore α; use current R and c", variable=derive_var, value="ignore").grid(
        row=2, column=0, sticky="w", padx=8, pady=(2, 8)
    )
    help_button(derive_frame, *help_texts["derive"]).grid(row=0, column=1, sticky="nw", padx=(8, 8), pady=2)

    handedness_var = tk.StringVar(value="right")
    hand_frame = ttk.LabelFrame(param_frame, text="Handedness", style="Tool.TLabelframe")
    hand_frame.grid(row=9, column=0, columnspan=4, sticky="ew", padx=8, pady=(4, 8))
    ttk.Radiobutton(hand_frame, text="right: y = +R sin(t + φ)", variable=handedness_var, value="right").grid(
        row=0, column=0, sticky="w", padx=8, pady=2
    )
    ttk.Radiobutton(hand_frame, text="left: y = −R sin(t + φ)", variable=handedness_var, value="left").grid(
        row=1, column=0, sticky="w", padx=8, pady=(2, 8)
    )
    help_button(hand_frame, *help_texts["handedness"]).grid(row=0, column=1, sticky="nw", padx=(8, 8), pady=2)

    output_frame = ttk.LabelFrame(body, text="Output and derived values", style="Tool.TLabelframe")
    output_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
    output_frame.columnconfigure(0, weight=1)

    output_header = ttk.Frame(output_frame)
    output_header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 2))
    ttk.Label(output_header, text="Output file").pack(side="left")
    help_button(output_header, *help_texts["output"]).pack(side="left", padx=(6, 0))
    output_var = tk.StringVar(value="helix.xyz")
    file_row = ttk.Frame(output_frame)
    file_row.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
    file_row.columnconfigure(0, weight=1)
    ttk.Entry(file_row, textvariable=output_var).grid(row=0, column=0, sticky="ew")

    def browse_output() -> None:
        filename = filedialog.asksaveasfilename(
            title="Save plain-coordinate XYZ file",
            defaultextension=".xyz",
            filetypes=[("XYZ files", "*.xyz"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if filename:
            output_var.set(filename)

    ttk.Button(file_row, text="Browse...", command=browse_output).grid(row=0, column=1, padx=(6, 0))

    summary_var = tk.StringVar(value="Derived values will appear here.")
    summary_label = ttk.Label(output_frame, text="Derived values")
    summary_label.grid(row=2, column=0, sticky="w", padx=8, pady=(4, 2))
    summary_box = tk.Label(
        output_frame,
        textvariable=summary_var,
        anchor="nw",
        justify="left",
        relief="groove",
        padx=8,
        pady=8,
        font=mono_font,
        bg="white",
    )
    summary_box.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
    output_frame.rowconfigure(3, weight=1)

    note_text = (
        "Plain-coordinate XYZ output: one line per point, written as x y z.\n"
        "No atom names and no molecular-XYZ header are included."
    )
    ttk.Label(output_frame, text=note_text, justify="left", style="Hint.TLabel").grid(
        row=4, column=0, sticky="ew", padx=8, pady=(0, 8)
    )

    def parse_optional_float(value: str, field_name: str) -> Optional[float]:
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return float(stripped)
        except ValueError:
            raise ValueError(f"{field_name} must be a number or left blank.")

    def parse_required_float(value: str, field_name: str) -> float:
        stripped = value.strip()
        if stripped == "":
            raise ValueError(f"{field_name} is required.")
        try:
            return float(stripped)
        except ValueError:
            raise ValueError(f"{field_name} must be a number.")

    def parse_required_int(value: str, field_name: str) -> int:
        stripped = value.strip()
        if stripped == "":
            raise ValueError(f"{field_name} is required.")
        try:
            return int(stripped)
        except ValueError:
            raise ValueError(f"{field_name} must be an integer.")

    def get_gui_values() -> Tuple[float, float, float, int, str, float, float, int, str, str]:
        radius_in = parse_optional_float(entries["radius"].get(), "R")
        c_in = parse_optional_float(entries["c"].get(), "c")
        pitch_angle_in = parse_optional_float(entries["pitch_angle_deg"].get(), "Pitch angle alpha")
        total_length = parse_required_float(entries["total_length"].get(), "Total length L")
        num_points = parse_required_int(entries["num_points"].get(), "Number of points")
        phase_deg = parse_required_float(entries["phase_deg"].get(), "Phase phi")
        z0 = parse_required_float(entries["z0"].get(), "z0")
        precision = parse_required_int(entries["precision"].get(), "Precision")
        handedness = handedness_var.get()
        output = output_var.get().strip()
        if not output:
            raise ValueError("Please specify an output file.")

        derive = derive_var.get()
        radius, c_val, note = resolve_radius_c(radius_in, c_in, pitch_angle_in, derive)
        validate_inputs(radius, c_val, total_length, num_points, handedness)
        if precision < 0:
            raise ValueError("Precision must be non-negative.")
        return radius, c_val, total_length, num_points, handedness, phase_deg, z0, precision, output, note

    def set_resolved_entries(radius: float, c_val: float) -> None:
        entries["radius"].set(f"{radius:.10g}")
        entries["c"].set(f"{c_val:.10g}")

    def update_equation_text() -> None:
        sign = "+" if handedness_var.get() == "right" else "−"
        equation_var.set(
            "x(t) = R cos(t + φ)\n"
            f"y(t) = {sign} R sin(t + φ)\n"
            "z(t) = z₀ + c t\n\n"
            "Pitch:  P = 2πc        Pitch angle:  α = atan(c/R)\n"
            "Arc length:  L = √(R² + c²) · tmax"
        )

    def update_summary(write_back: bool = True) -> None:
        try:
            radius, c_val, total_length, num_points, handedness, phase_deg, z0, precision, output, note = get_gui_values()
            if write_back:
                set_resolved_entries(radius, c_val)
            summary_var.set(helix_summary(radius, c_val, total_length, note))
        except Exception as exc:
            summary_var.set(f"Input issue: {exc}")
        update_equation_text()

    def fill_pitch_angle_from_current_rc() -> None:
        try:
            radius = parse_required_float(entries["radius"].get(), "R")
            c_val = parse_required_float(entries["c"].get(), "c")
            validate_inputs(radius, c_val, 1.0, 2, handedness_var.get())
            alpha = pitch_angle_from_radius_c(radius, c_val)
            entries["pitch_angle_deg"].set(f"{alpha:.10g}")
            update_summary(write_back=False)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            summary_var.set(f"Input issue: {exc}")

    def generate_file() -> None:
        try:
            radius, c_val, total_length, num_points, handedness, phase_deg, z0, precision, output, note = get_gui_values()
            set_resolved_entries(radius, c_val)
            points = generate_helix_points(
                radius=radius,
                c=c_val,
                total_length=total_length,
                num_points=num_points,
                handedness=handedness,
                phase_deg=phase_deg,
                z0=z0,
            )
            write_plain_xyz(points, output, precision)
            summary_var.set(helix_summary(radius, c_val, total_length, note))
            messagebox.showinfo("Done", f"Wrote {len(points)} points to:\n{output}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            summary_var.set(f"Input issue: {exc}")
        update_equation_text()

    button_frame = ttk.Frame(main)
    button_frame.grid(row=3, column=0, sticky="e", pady=(14, 0))
    ttk.Button(button_frame, text="Fill α from current R,c", command=fill_pitch_angle_from_current_rc).grid(
        row=0, column=0, padx=(0, 8)
    )
    ttk.Button(button_frame, text="Update derived values", command=lambda: update_summary(True)).grid(
        row=0, column=1, padx=(0, 8)
    )
    ttk.Button(button_frame, text="Generate XYZ", command=generate_file).grid(row=0, column=2)

    for var in list(entries.values()) + [handedness_var, derive_var]:
        var.trace_add("write", lambda *_args: update_equation_text())

    update_equation_text()
    update_summary(write_back=False)
    root.mainloop()


def main() -> None:
    """Entry point for CLI or GUI mode."""
    args = parse_args(sys.argv[1:])

    if args.gui or len(sys.argv) == 1:
        run_gui()
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
