#!/usr/bin/env python3
"""Compatibility launcher for Generate SC V2_2.

The implementation moved to :mod:`generate_sc_xyzV2_2`.  This module keeps
existing V2_1 imports and command lines working.
"""

try:
    from curve_it_lib.generate_sc_xyzV2_2 import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - direct execution from curve_it_lib
    from generate_sc_xyzV2_2 import *  # type: ignore  # noqa: F401,F403


if __name__ == "__main__":
    main()
