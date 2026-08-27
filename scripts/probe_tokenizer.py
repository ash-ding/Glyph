#!/usr/bin/env python3
"""Self-check #6 -- pick the surface notation for values.

D8 is settled as option A: the surface form spells out a value's digits, so
that the digit structure the tables rely on survives tokenisation.  Which
exact notation to use is NOT settled by argument -- it is settled by running
the student's tokenizer and looking at where the boundaries fall.

Criterion: every digit must land inside its own token, never glued to the
leading `v` or to a neighbouring digit.

    python scripts/probe_tokenizer.py Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import sys

CANDIDATES = [
    ("flat",       "v802"),
    ("underscore", "v_3_13_2"),
    ("bracket",    "v[3,13,2]"),
    ("dot",        "v3.13.2"),
    ("spaced",     "v 3 13 2"),
]
DIGITS = ["3", "13", "2"]


def verdict(tokens: list[str]) -> tuple[bool, str]:
    clean = [t.lstrip("Ġ▁ ") for t in tokens]
    remaining = list(DIGITS)
    for t in clean:
        if remaining and t == remaining[0]:
            remaining.pop(0)
    if remaining:
        return False, f"digits not isolated (missing {remaining})"
    for t in clean:
        if t.startswith("v") and len(t) > 1 and t[1:].isdigit():
            return False, f"digit glued to 'v' in {t!r}"
    return True, "ok"


def main(model: str) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    print(f"tokenizer: {model}\n")
    width = max(len(n) for n, _ in CANDIDATES)
    passed = []
    for name, form in CANDIDATES:
        toks = tok.tokenize(form)
        ok, why = verdict(toks)
        print(f"  {name:<{width}}  {form:<12}  {len(toks):>2} tok  "
              f"{'PASS' if ok else 'fail'}  {toks}")
        if not ok:
            print(f"  {'':<{width}}  -> {why}")
        if ok:
            passed.append((name, len(toks)))
    print()
    if not passed:
        print("NO CANDIDATE PASSES -- reopen D8.")
        return 1
    best = min(passed, key=lambda x: x[1])
    print(f"recommendation: value_form = {best[0]!r}  ({best[1]} tokens per value)")
    print("set it in GlyphConfig.value_form and regenerate.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"))
