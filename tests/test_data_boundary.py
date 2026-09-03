"""The data layer must not know how it will be evaluated.

The dependency runs one way -- `seal`, `budget`, `agent` and `arms` all import
from `glyph.data`, and nothing in it imports back. That was true before the
package existed, but only by habit; this holds the line so that a second task
can take the generator whole without dragging in an evaluation protocol built
for the first one.
"""
import ast
import pathlib

import pytest

DATA = pathlib.Path(__file__).resolve().parents[1] / "src" / "glyph" / "data"
SIBLINGS = {p.stem for p in DATA.glob("*.py")} | {"data"}


def _modules():
    return sorted(p for p in DATA.glob("*.py"))


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_data_layer_is_self_contained(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            # a relative import inside the package: the target must be a sibling
            target = (node.module or "").split(".")[0]
            assert node.level == 1 and target in SIBLINGS, (
                f"{path.name} reaches outside the data layer: "
                f"{'.' * node.level}{node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("glyph."), (
                    f"{path.name} imports {alias.name}")
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("glyph."):
            assert node.module.startswith("glyph.data"), (
                f"{path.name} imports {node.module}")


def test_the_public_surface_is_importable_on_its_own():
    """Someone reusing the generator should need one import, not seven."""
    from glyph.data import PRESETS, GlyphConfig, generate, measure_pi
    inst = generate(7, PRESETS["smoke"])
    assert inst.test and inst.demos
    assert 0.0 <= measure_pi(inst)["pi"] <= 1.0
    assert isinstance(inst.cfg, GlyphConfig)


def test_the_protocol_layer_still_depends_on_the_data_layer():
    """The one-way arrow, asserted from the other side: if this ever stops
    being true the layers have been merged by accident."""
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "glyph"
    users = set()
    for path in src.rglob("*.py"):
        if DATA in path.parents or "__pycache__" in str(path):
            continue
        if "data." in path.read_text():
            users.add(path.name)
    assert {"seal.py", "base.py", "tools.py"} <= users, users
