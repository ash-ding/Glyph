#!/usr/bin/env python3
"""Self-check #6 -- pick the surface notation for values (D8) and the digit
layout that goes with it (D3).

D8 is settled as option A: the surface form spells out a value's digits, so
the digit structure the tables rely on survives tokenisation.  Which exact
notation, and over how many digits of what base, is not settled by argument
-- it is settled by running the student's tokenizer and looking at where the
boundaries actually fall.

**The criterion is that no token may straddle a digit boundary.**

An earlier version of this script asked for something stricter and wrong:
that every digit land inside its own single token.  Current tokenizers --
Qwen3 among them -- split numbers into one token per character, so a base-17
digit like `13` occupies two tokens no matter how it is written.  Under that
criterion every candidate fails at any base above 10, which says nothing
about the notation and everything about the test.

What the design actually needs is weaker and checkable: the student must be
able to tell where one digit ends and the next begins.  A digit spread over
two tokens is fine as long as no token reaches across a boundary and smears
two digit positions together.  That is what gets checked here, against the
real `render_value`, using the tokenizer's own character offsets.

`flat` fails by construction and is kept only to show why: it renders the
decimal index (`v802`), not the digits, so at any base but 10 the characters
have nothing to do with digit positions -- and even at base 10 it drops
leading zeros, so the field count varies with the value.

    python scripts/probe_tokenizer.py Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import sys

from glyph.config import GlyphConfig
from glyph.grammar import digits, render_value

# Layouts worth considering: all land near |V| ~ 5000, and D3 is open
# precisely because which one the student learns most cleanly is empirical.
LAYOUTS = [(17, 3), (10, 4), (8, 4), (7, 4)]
FORMS = ("underscore", "bracket", "flat")

#: Values whose digits exercise one-character and two-character fields, a
#: zero field, and a leading-zero field.
SAMPLES = (802, 0, 1, 4912, 170, 289)


def digit_spans(idx: int, cfg: GlyphConfig) -> list[tuple[int, int]] | None:
    """Character span of each digit field in the rendered value.

    Returns None when the form exposes no digit fields at all.
    """
    ds = digits(idx, cfg)
    form = cfg.value_form
    if form == "flat":
        return None
    if form == "underscore":
        pos, sep, spans = 2, 1, []          # "v_"
    elif form == "bracket":
        pos, sep, spans = 2, 1, []          # "v["
    else:
        raise ValueError(f"unknown value_form {form!r}")
    for k, d in enumerate(ds):
        text = str(d)
        spans.append((pos, pos + len(text)))
        pos += len(text) + (sep if k < len(ds) - 1 else 0)
    return spans


def straddles(tok_span: tuple[int, int], fields: list[tuple[int, int]]) -> bool:
    """True if this token reaches across a digit boundary."""
    a, b = tok_span
    for s, e in fields:
        # overlaps the field but is not contained in it
        if a < e and b > s and not (a >= s and b <= e):
            return True
    return False


def check(tok, cfg: GlyphConfig) -> tuple[bool, str, float]:
    """Returns (passes, reason, mean tokens per value)."""
    total = 0
    for idx in SAMPLES:
        if idx >= cfg.n_values:
            continue
        s = render_value(idx, cfg)
        enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        total += len(offsets)
        fields = digit_spans(idx, cfg)
        if fields is None:
            return False, "form carries no digit fields (renders the decimal index)", 0.0
        for span in offsets:
            if straddles(tuple(span), fields):
                piece = s[span[0]:span[1]]
                return False, f"token {piece!r} straddles a digit boundary in {s!r}", 0.0
    n = sum(1 for i in SAMPLES if i < cfg.n_values)
    return True, "ok", total / n


def main(model: str) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    if not tok.is_fast:
        print(f"{model} has no fast tokenizer; offsets are unavailable.")
        return 2

    print(f"tokenizer: {model}")
    print("criterion: no token may straddle a digit boundary\n")
    print(f"  {'layout':<10} {'|V|':>6}  {'form':<11} {'tok/value':>9}  verdict")
    print(f"  {'-' * 10} {'-' * 6}  {'-' * 11} {'-' * 9}  {'-' * 40}")

    passing: list[tuple[float, str, int, int]] = []
    for base, ndig in LAYOUTS:
        for form in FORMS:
            cfg = GlyphConfig(base=base, n_digits=ndig, value_form=form)
            ok, why, cost = check(tok, cfg)
            layout = f"{base}^{ndig}"
            if ok:
                passing.append((cost, form, base, ndig))
                print(f"  {layout:<10} {cfg.n_values:>6}  {form:<11} {cost:>9.1f}  PASS")
            else:
                print(f"  {layout:<10} {cfg.n_values:>6}  {form:<11} {'-':>9}  fail: {why}")

    print()
    if not passing:
        print("NO CANDIDATE PASSES -- reopen D8.")
        return 1

    cost, form, base, ndig = min(passing)
    print(f"cheapest passing: value_form={form!r}, base={base}, n_digits={ndig} "
          f"({cost:.1f} tokens per value, |V|={base ** ndig})")
    print()
    print("Token cost is not the only axis, and it is the weaker one: it sets")
    print("how much the context arm pays per query, while the digit layout sets")
    print("whether the student can learn the tables at all.  Settle D3 on the")
    print("capacity check (self-check #5, `scripts/capacity_check.py`), which")
    print("sweeps the same layouts, and use this table to break ties on cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"))
