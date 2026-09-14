"""The data layer: everything that makes an instance, and nothing that scores one.

Isolated here because the dependency runs one way and should keep running one
way. Generating an instance does not need to know how it will be evaluated --
`seal`, `budget`, `agent` and `arms` all import from this package, and nothing
in it imports back. `test_data_layer_is_self_contained` holds that line, which
was previously maintained only by habit.

The point of the boundary is reuse: a second task can take this generator whole
without dragging in an evaluation protocol built for the first one.

    generate(seed, cfg) -> GlyphInstance
        .demos          30 worked examples, free
        .query(expr)    the metered oracle
        .test_set()     iid / comp / depth, fixed at generation
        .derive_tail()  items whose entries this run never bought
        .measured_pi()  which half the difficulty sits in
        .ceilings()     what each crippled oracle scores on given items
"""

from .config import PRESETS, VALUE_FORMS, GlyphConfig
from .grammar import (Expr, check, depth, digits, enabled_ops, op_pairs, parse,
                      parse_value, render, render_list, render_value,
                      syntax_spec, undigits)
from .instance import (SPLITS, GenerationFailed, GlyphInstance, TestItem,
                       generate)
from .interp import Interpreter, LookupLog
from .measure import measure_pi
from .semantics import StructSem, sample_skeleton, trivial_skeleton
from .tables import IdentityTables, Tables

__all__ = [
    "GlyphConfig", "PRESETS", "VALUE_FORMS",
    "generate", "GlyphInstance", "TestItem", "SPLITS", "GenerationFailed",
    "Interpreter", "LookupLog", "measure_pi",
    "Tables", "IdentityTables",
    "StructSem", "sample_skeleton", "trivial_skeleton",
    "Expr", "parse", "check", "render", "render_value", "render_list",
    "parse_value", "digits", "undigits", "depth", "op_pairs", "enabled_ops",
    "syntax_spec",
]
