"""Syntax of the Glyph language.

The syntax is PUBLIC: the agent is told which expressions are well formed,
the arity of every operator, and how values are written.  Only the
*semantics* (what each operator does) is hidden.

Operator names are deliberately opaque -- `s0`/`u2`/`b0` rather than
`map`/`fold`/`add` -- so that naming priors leak nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import GlyphConfig

Type = Literal["LIST", "VAL"]

# ---------------------------------------------------------------------
# Structural operator shapes.  These are fixed across all instances --
# they are part of the public syntax.  Only the semantics vary.
# ---------------------------------------------------------------------
#   UL : (unary, list) -> list      consumes an atomic unary operator
#   LB : (list, binary) -> value    consumes an atomic binary operator
#   L  : (list) -> list             pure structure
#   KL : (int, list) -> list        pure structure
STRUCT_SHAPES: tuple[tuple[str, str], ...] = (
    ("s0", "UL"),
    ("s1", "LB"),
    ("s2", "L"),
    ("s3", "KL"),
    ("s4", "L"),
    ("s5", "UL"),
    ("s6", "KL"),
    ("s7", "LB"),
)
SHAPE_RESULT: dict[str, Type] = {"UL": "LIST", "LB": "VAL", "L": "LIST", "KL": "LIST"}
K_RANGE = (1, 3)


def enabled_ops(cfg: GlyphConfig) -> list[tuple[str, str]]:
    """The first `n_structural` shapes, in order.  Order matters: s0 (map-like)
    and s1 (fold-like) come first because they are the two that consume atomic
    operators, and the pi->0 preset keeps only those."""
    return list(STRUCT_SHAPES[: cfg.n_structural])


def unary_names(cfg: GlyphConfig) -> list[str]:
    return [f"u{i}" for i in range(cfg.n_unary)]


def binary_names(cfg: GlyphConfig) -> list[str]:
    return [f"b{i}" for i in range(cfg.n_binary)]


# ---------------------------------------------------------------------
# Value codec (D8, option A: digits are spelled out in the surface form)
# ---------------------------------------------------------------------
def digits(idx: int, cfg: GlyphConfig) -> tuple[int, ...]:
    """Little-endian digit tuple: d[k] = (idx // base**k) % base."""
    return tuple((idx // cfg.base**k) % cfg.base for k in range(cfg.n_digits))


def undigits(ds: tuple[int, ...], cfg: GlyphConfig) -> int:
    return sum(d * cfg.base**k for k, d in enumerate(ds))


def render_value(idx: int, cfg: GlyphConfig) -> str:
    ds = digits(idx, cfg)
    form = cfg.value_form
    if form == "underscore":
        return "v_" + "_".join(str(d) for d in ds)
    if form == "bracket":
        return "v[" + ",".join(str(d) for d in ds) + "]"
    if form == "flat":
        return f"v{idx}"
    raise ValueError(f"unknown value_form {form!r}")


def parse_value(tok: str, cfg: GlyphConfig) -> int:
    form = cfg.value_form
    if form == "underscore":
        parts = tok.split("_")
        if parts[0] != "v" or len(parts) != cfg.n_digits + 1:
            raise SyntaxError(f"bad value {tok!r}")
        return undigits(tuple(int(p) for p in parts[1:]), cfg)
    if form == "bracket":
        if not (tok.startswith("v[") and tok.endswith("]")):
            raise SyntaxError(f"bad value {tok!r}")
        return undigits(tuple(int(p) for p in tok[2:-1].split(",")), cfg)
    if form == "flat":
        if not tok.startswith("v"):
            raise SyntaxError(f"bad value {tok!r}")
        return int(tok[1:])
    raise ValueError(f"unknown value_form {form!r}")


# ---------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Val:
    """A bare value, e.g. v_3_13_2."""
    idx: int


@dataclass(frozen=True)
class Lit:
    """A list literal, e.g. [v_0_1_0, v_3_13_2]."""
    items: tuple[int, ...]


@dataclass(frozen=True)
class App:
    """A structural-operator application.  `args` mixes operator names (str),
    small integers (int) and sub-expressions, per the operator's shape."""
    op: str
    args: tuple


Expr = Val | Lit | App


def result_type(e: Expr) -> Type:
    if isinstance(e, Val):
        return "VAL"
    if isinstance(e, Lit):
        return "LIST"
    shape = dict(STRUCT_SHAPES)[e.op]
    return SHAPE_RESULT[shape]


def depth(e: Expr) -> int:
    if isinstance(e, (Val, Lit)):
        return 0
    return 1 + max((depth(a) for a in e.args if isinstance(a, (Val, Lit, App))), default=0)


def op_pairs(e: Expr) -> set[tuple[str, str]]:
    """Directly-nested (outer, inner) structural-op pairs.  Used to build the
    `comp` split: some pairs are held out of the demo distribution entirely."""
    out: set[tuple[str, str]] = set()
    if isinstance(e, App):
        for a in e.args:
            if isinstance(a, App):
                out.add((e.op, a.op))
                out |= op_pairs(a)
            elif isinstance(a, (Val, Lit)):
                out |= op_pairs(a)
    return out


# ---------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------
def render(e: Expr, cfg: GlyphConfig) -> str:
    if isinstance(e, Val):
        return render_value(e.idx, cfg)
    if isinstance(e, Lit):
        return "[" + ", ".join(render_value(i, cfg) for i in e.items) + "]"
    parts = []
    for a in e.args:
        if isinstance(a, str):
            parts.append(a)
        elif isinstance(a, int):
            parts.append(str(a))
        else:
            parts.append(render(a, cfg))
    return f"{e.op}({', '.join(parts)})"


def render_list(vals: list[int], cfg: GlyphConfig) -> str:
    return "[" + ", ".join(render_value(i, cfg) for i in vals) + "]"


# ---------------------------------------------------------------------
# Parser (recursive descent over a tiny token stream)
# ---------------------------------------------------------------------
class _P:
    def __init__(self, s: str, cfg: GlyphConfig):
        self.toks = _tokenize(s)
        self.i = 0
        self.cfg = cfg

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, expect: str | None = None) -> str:
        t = self.peek()
        if t is None:
            raise SyntaxError("unexpected end of input")
        if expect is not None and t != expect:
            raise SyntaxError(f"expected {expect!r}, got {t!r}")
        self.i += 1
        return t

    def expr(self) -> Expr:
        t = self.peek()
        if t is None:
            raise SyntaxError("empty expression")
        if t == "[":
            self.take("[")
            items: list[int] = []
            if self.peek() != "]":
                while True:
                    items.append(parse_value(self.take(), self.cfg))
                    if self.peek() == ",":
                        self.take(",")
                    else:
                        break
            self.take("]")
            return Lit(tuple(items))
        if t.startswith("s") and self.i + 1 < len(self.toks) and self.toks[self.i + 1] == "(":
            op = self.take()
            shape = dict(STRUCT_SHAPES).get(op)
            if shape is None:
                raise SyntaxError(f"unknown structural operator {op!r}")
            self.take("(")
            args: list = []
            for slot in shape:
                if slot == "U" or slot == "B":
                    args.append(self.take())
                elif slot == "K":
                    args.append(int(self.take()))
                else:  # "L"
                    args.append(self.expr())
                if self.peek() == ",":
                    self.take(",")
            self.take(")")
            return App(op, tuple(args))
        return Val(parse_value(self.take(), self.cfg))


def _tokenize(s: str) -> list[str]:
    out, buf = [], ""
    for ch in s:
        if ch in "(),[] " or ch == "\n" or ch == "\t":
            if buf:
                out.append(buf)
                buf = ""
            if ch not in " \n\t":
                out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def parse(s: str, cfg: GlyphConfig) -> Expr:
    p = _P(s, cfg)
    e = p.expr()
    if p.peek() is not None:
        raise SyntaxError(f"trailing input at {p.peek()!r}")
    return e


# ---------------------------------------------------------------------
# Well-formedness (used to police agent queries)
# ---------------------------------------------------------------------
def check(e: Expr, cfg: GlyphConfig) -> None:
    """Raise SyntaxError if `e` is not a legal expression under `cfg`."""
    ops = dict(enabled_ops(cfg))
    us, bs = set(unary_names(cfg)), set(binary_names(cfg))
    nv = cfg.n_values

    def walk(node: Expr, d: int) -> None:
        if d > cfg.max_expr_depth:
            raise SyntaxError(f"nesting deeper than {cfg.max_expr_depth}")
        if isinstance(node, Val):
            if not 0 <= node.idx < nv:
                raise SyntaxError(f"value out of range: {node.idx}")
            return
        if isinstance(node, Lit):
            lo, hi = cfg.list_len_range
            if not lo <= len(node.items) <= hi:
                raise SyntaxError(f"list length {len(node.items)} outside {lo}..{hi}")
            for i in node.items:
                if not 0 <= i < nv:
                    raise SyntaxError(f"value out of range: {i}")
            return
        shape = ops.get(node.op)
        if shape is None:
            raise SyntaxError(f"operator {node.op!r} is not enabled")
        if len(node.args) != len(shape):
            raise SyntaxError(f"{node.op} takes {len(shape)} arguments")
        for slot, a in zip(shape, node.args):
            if slot == "U":
                if a not in us:
                    raise SyntaxError(f"{a!r} is not a unary operator")
            elif slot == "B":
                if a not in bs:
                    raise SyntaxError(f"{a!r} is not a binary operator")
            elif slot == "K":
                if not isinstance(a, int) or not K_RANGE[0] <= a <= K_RANGE[1]:
                    raise SyntaxError(f"integer argument must be {K_RANGE[0]}..{K_RANGE[1]}")
            else:  # "L"
                if not isinstance(a, (Val, Lit, App)) or result_type(a) != "LIST":
                    raise SyntaxError(f"{node.op} expects a list argument")
                walk(a, d + 1)

    walk(e, 0)


def syntax_spec(cfg: GlyphConfig) -> str:
    """The public description handed to the agent.  Semantics are absent."""
    lines = [
        "You are facing a small language you have never seen before.",
        "The SYNTAX below is given to you.  The SEMANTICS -- what each",
        "operator actually does -- is hidden and must be discovered.",
        "",
        f"Values:            {render_value(0, cfg)} .. {render_value(cfg.n_values - 1, cfg)}",
        f"                   ({cfg.n_values} of them; the digits carry no arithmetic meaning)",
        f"Unary operators:   {' '.join(unary_names(cfg))}",
        f"Binary operators:  {' '.join(binary_names(cfg))}",
        "Structural operators (argument shapes only):",
    ]
    sig = {
        "UL": "(unary, list) -> list",
        "LB": "(list, binary) -> value",
        "L": "(list) -> list",
        "KL": "(int, list) -> list",
    }
    for name, shape in enabled_ops(cfg):
        lines.append(f"    {name}{sig[shape]}")
    lo, hi = cfg.list_len_range
    lines += [
        f"Integer arguments are {K_RANGE[0]}..{K_RANGE[1]}.",
        f"List literals hold {lo}..{hi} values.",
        f"Nesting depth is at most {cfg.max_expr_depth}.",
    ]
    return "\n".join(lines)
