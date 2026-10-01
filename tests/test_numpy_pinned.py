"""The environment runs the numpy the benchmark is pinned to.

Every instance is drawn from one ``numpy.random.Generator`` stream, and NumPy
guarantees bit-for-bit stream stability across releases only for the legacy
``RandomState``, not for ``Generator``. The frozen instances are therefore
defined by the generator code *and* the numpy version, and an environment on
another version may draw different instances from the same seed.
"""
import tomllib
from pathlib import Path

import numpy

ROOT = Path(__file__).resolve().parents[1]


def _pinned() -> str:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    spec = next(d for d in deps if d.replace(" ", "").startswith("numpy"))
    assert "==" in spec, f"numpy must be pinned exactly, found {spec!r}"
    return spec.split("==", 1)[1].strip()


def test_installed_numpy_matches_the_pin():
    want = _pinned()
    assert numpy.__version__ == want, (
        f"numpy {numpy.__version__} installed, pyproject pins {want}: "
        f"run `pip install numpy=={want}` in this environment")
