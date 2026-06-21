#!/usr/bin/env python3
"""Stable launcher for the newest versioned Plane It implementation."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


def _repo_or_bundle_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent


def _version_key(path: Path) -> tuple:
    match = re.match(r"plane_itV(.+)\.py$", path.name)
    if not match:
        return ()
    return tuple(int(part) for part in re.findall(r"\d+", match.group(1)))


def _latest_implementation() -> Path:
    lib_dir = _repo_or_bundle_root() / "curve_it_lib"
    candidates = sorted(lib_dir.glob("plane_itV*.py"), key=_version_key)
    if not candidates:
        raise FileNotFoundError(f"No Plane It implementation found in {lib_dir}")
    return candidates[-1]


def main() -> int:
    impl_path = _latest_implementation()
    spec = importlib.util.spec_from_file_location(impl_path.stem, impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Plane It implementation: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[impl_path.stem] = module
    spec.loader.exec_module(module)
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
