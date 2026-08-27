#!/usr/bin/env python3
"""Self-check #6 -- pick the surface notation for values (D8) and the digit
layout it goes with (D3).

D8 is settled as option A: the surface form spells out a value's digits, so
the structure the tables rely on survives tokenisation.  Which notation, over
how many digits of what base, is not settled by argument -- it is settled by
running a tokenizer and looking at where the boundaries actually fall.

**The criterion is that no token may cover two or more digit positions.**

Two earlier versions of this script got that wrong in opposite directions and
both mistakes were in the test, not in the notation:

  * asking that every digit land in its own single token.  Current
    tokenizers split numbers one token per character, so a base-17 digit
    like `13` spans two tokens however it is written; under that rule
    nothing can pass above base 10, which says nothing about the notation.
  * failing a token that swallows a separator along with its digit -- `_a`
    as one token.  That smears nothing.  It is one token per digit, which
    is the cleanest signal available.  What breaks the design is a token
    covering two *digits*, because then there is no boundary left for the
    student to key on.

Coverage is exhaustive by default.  Tokenisation is content-dependent, so a
form that never straddles across a handful of samples can still straddle on
the next one; the earlier six-value spot check also mis-estimated cost by
half a token per value.

`letter_*` forms are proposed, not implemented: `grammar.py` cannot render
them yet.  They are measured here because a digit written as a letter is one
character wide at any base below 27 -- fixed-width and potentially much
cheaper -- and the design already states that digits carry no arithmetic
meaning, so letters cost nothing conceptually.

    python scripts/probe_tokenizer.py Qwen/Qwen3-1.7B
    python scripts/probe_tokenizer.py Qwen/Qwen3-1.7B --teacher
"""

from __future__ import annotations

import argparse

from glyph.config import GlyphConfig

LAYOUTS = [(17, 3), (10, 4), (8, 4)]
IMPLEMENTED = ("underscore", "bracket", "flat")
PROPOSED = ("letter_sep", "letter_flat")


def render(idx: int, base: int, ndig: int, form: str):
    """Render a value; return it with the character span of each digit field.

    A span list of None means the form exposes no digit fields at all.
    """
    ds = [(idx // base ** k) % base for k in range(ndig)]
    if form == "flat":
        return f"v{idx}", None

    letters = form.startswith("letter")
    parts = [chr(ord("a") + d) if letters else str(d) for d in ds]
    head, sep, tail = {
        "underscore": ("v_", "_", ""),
        "letter_sep": ("v_", "_", ""),
        "bracket": ("v[", ",", "]"),
        "letter_flat": ("v", "", ""),
    }[form]

    s, spans, pos = head, [], len(head)
    for k, p in enumerate(parts):
        spans.append((pos, pos + len(p)))
        s += p
        pos += len(p)
        if k < len(parts) - 1:
            s += sep
            pos += len(sep)
    return s + tail, spans


def smears(span, fields) -> bool:
    """True if this token covers two or more digit positions."""
    a, b = span
    return sum(1 for s, e in fields if a < e and b > s) > 1


def audit(tok, base: int, ndig: int, form: str, limit: int | None) -> dict:
    n = base ** ndig
    idxs = range(n) if limit is None or n <= limit else range(0, n, max(1, n // limit))
    strings, spans = [], []
    for i in idxs:
        s, f = render(i, base, ndig, form)
        if f is None:
            return {"ok": False, "why": "renders the decimal index, not the digits",
                    "cost": None, "n": 0}
        strings.append(s)
        spans.append(f)

    offsets = tok(strings, add_special_tokens=False,
                  return_offsets_mapping=True)["offset_mapping"]
    total, bad = 0, None
    for s, fields, offs in zip(strings, spans, offsets):
        total += len(offs)
        if bad is None:
            for o in offs:
                if smears(tuple(o), fields):
                    bad = f"{s!r}: token {s[o[0]:o[1]]!r} covers two digits"
                    break
    return {"ok": bad is None, "why": bad or "ok",
            "cost": total / len(strings), "n": len(strings)}


def teacher_cost(forms: list[str], base: int, ndig: int, n: int = 300) -> dict[str, float]:
    """Tokens per value on the teacher.

    The agent reads demos and pays for every oracle query in the teacher's
    tokens, so the notation is billed on both sides of the ledger.
    `count_tokens` is blocked by org policy on this Vertex project, so the
    number comes from `usage.input_tokens` on a real request instead.
    """
    from glyph.vertex import TEACHER, client

    c = client()

    def count(text: str) -> int:
        return c.messages.create(
            model=TEACHER, max_tokens=16,
            thinking={"type": "disabled"},      # input_tokens is unaffected
            messages=[{"role": "user", "content": text}],
        ).usage.input_tokens

    envelope = count("x") - 1
    out = {}
    for form in forms:
        vals = [render(i * 7 % (base ** ndig), base, ndig, form)[0] for i in range(n)]
        out[form] = (count(" ".join(vals)) - envelope) / n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--limit", type=int, default=None,
                    help="sample this many values instead of checking all of them")
    ap.add_argument("--teacher", action="store_true",
                    help="also bill each notation against the teacher (costs a few API calls)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.is_fast:
        print(f"{args.model} has no fast tokenizer; character offsets are unavailable.")
        return 2

    print(f"student tokenizer: {args.model}")
    print("criterion: no token may cover two or more digit positions")
    print(f"coverage: {'every value' if args.limit is None else f'{args.limit} sampled'}\n")
    print(f"  {'layout':<7} {'form':<12} {'checked':>8} {'tok/val':>8}  verdict")
    print(f"  {'-'*7} {'-'*12} {'-'*8} {'-'*8}  {'-'*46}")

    passing: list[tuple[float, str, int, int]] = []
    for base, ndig in LAYOUTS:
        for form in IMPLEMENTED + PROPOSED:
            if form in PROPOSED and base > 26:
                continue
            r = audit(tok, base, ndig, form, args.limit)
            mark = "" if form in IMPLEMENTED else " *"
            if r["ok"]:
                passing.append((r["cost"], form, base, ndig))
                print(f"  {base}^{ndig:<5} {form + mark:<12} {r['n']:>8} "
                      f"{r['cost']:>8.2f}  PASS")
            else:
                print(f"  {base}^{ndig:<5} {form + mark:<12} {r['n']:>8} "
                      f"{'-':>8}  fail: {r['why'][:46]}")
        print()

    if not passing:
        print("NO CANDIDATE PASSES -- reopen D8.")
        return 1

    print("* proposed only: grammar.py cannot render these yet.\n")
    passing.sort()
    print("cheapest passing on the student:")
    for cost, form, base, ndig in passing[:6]:
        print(f"  {cost:>5.2f} tok/value   {form:<12} {base}^{ndig}")

    if args.teacher:
        cheapest_layout = (passing[0][2], passing[0][3])
        forms = sorted({f for _, f, b, d in passing if (b, d) == cheapest_layout})
        print(f"\nteacher, at {cheapest_layout[0]}^{cheapest_layout[1]}:")
        for form, cost in sorted(teacher_cost(forms, *cheapest_layout).items(),
                                 key=lambda kv: kv[1]):
            print(f"  {cost:>5.2f} tok/value   {form}")

    print()
    print("Token cost is the weaker axis and it is not neutral: the context arm")
    print("re-pays it on every one of 10^4 test queries, so a cheaper notation")
    print("directly weakens the weights arm's economic advantage.  That is the")
    print("honest direction -- the plan already refuses to handicap A2 by")
    print("disabling its prefix cache.  Settle the digit layout on self-check #5")
    print("(scripts/capacity_check.py), which sweeps the same layouts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
