#!/usr/bin/env python3
"""Self-check #4 -- hiddenness.

The teacher is given the syntax spec and the free demos, and nothing else:
no queries, no budget, no oracle.  It then answers held-out expressions.

**The null is not 1/|V|.**  That was the first criterion here and it is the
wrong question: it asks whether the teacher can guess a value uniformly, and
nobody was ever going to.  Two things are legitimately available without
buying anything -- the skeleton, which the design *intends* to be inferable
from demos, and whatever table entries the demos reveal outright.  A teacher
using both is not leaking.

So the null is measured, on the same items with the same metric: an
interpreter with the true skeleton and a table that is real where the demos
exposed it and identity everywhere else.  On `pi_high` that null alone scores
0.89 exact, so a 1/|V| criterion would have screamed leak at a benchmark
behaving exactly as designed.  Scoring meaningfully *above* the null is what
means something leaked.

This is the last of the five checks that gate work on the arms, and it is
the only one that needs a frontier model, which is why it waited for Vertex.

Digit-level accuracy is reported next to exact match because the two fail
differently.  Exact match at chance with digits above chance would mean the
value *space* leaks (the model has learned which digits are plausible)
without the mapping leaking -- a much milder problem, but not nothing.

    python scripts/hiddenness_check.py
    python scripts/hiddenness_check.py --presets pi_mid --n 100
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time

from glyph.config import PRESETS
from glyph.grammar import parse, render_list, render_value
from glyph.instance import generate
from glyph.interp import Interpreter, LookupLog
from glyph.vertex import TEACHER, client, request_kwargs

PROMPT = """You are shown a small language you have never seen before.

{spec}

Here are {n_demo} worked examples:

{demos}

Now evaluate the following {n} expressions. They use the same language.

{items}

Answer with exactly {n} lines, each of the form

  <number>: <result>

where <number> is the expression's number above and <result> is the value
or list the expression evaluates to, written exactly the way results are
written in the examples. Give your single best guess for every one -- do not
skip any and do not explain."""


def ask(c, spec: str, demos: str, n_demo: int, items: list[str],
        model: str, effort_kwargs: dict) -> list[str]:
    numbered = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(items))
    prompt = PROMPT.format(spec=spec, demos=demos, n_demo=n_demo,
                           n=len(items), items=numbered)
    with c.messages.stream(messages=[{"role": "user", "content": prompt}],
                           **effort_kwargs) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if b.type == "text")

    got = [""] * len(items)
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[:.]\s*(.+?)\s*$", line)
        if m:
            k = int(m.group(1)) - 1
            if 0 <= k < len(items):
                got[k] = m.group(2).strip()
    return got, msg.usage, msg.stop_reason


class _DemoTables:
    """True where the demos revealed it, identity everywhere else."""

    def __init__(self, real, log):
        self.real, self.log = real, log

    def apply_unary(self, name, i):
        return self.real.apply_unary(name, i) if (name, i) in self.log.unary else i

    def apply_binary(self, name, i, j):
        return (self.real.apply_binary(name, i, j)
                if (name, i, j) in self.log.binary else i)


def demo_null(inst, items) -> dict:
    """What a teacher is entitled to reach from the free demos alone."""
    log = LookupLog()
    for src, _ in inst.demos:
        _, lg = inst.P.eval_logged(parse(src, inst.cfg))
        log |= lg
    interp = Interpreter(inst.cfg, inst.skeleton, _DemoTables(inst.tables, log))

    hits = dh = dt = 0
    for t in items:
        try:
            out = interp.eval(parse(t.expr_src, inst.cfg))
            got = (render_value(out, inst.cfg) if isinstance(out, int)
                   else render_list(out, inst.cfg))
        except Exception:
            got = ""
        hits += (got.strip() == t.answer_src.strip())
        h, n = digit_score(got, t.answer_src)
        dh += h
        dt += n
    return {"exact": hits / max(1, len(items)), "digit": dh / max(1, dt),
            "revealed_unary": len(log.unary), "revealed_binary": len(log.binary)}


def binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p) -- how surprising this many hits are
    if the teacher were guessing."""
    if k <= 0:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def digit_score(pred: str, gold: str) -> tuple[int, int]:
    """Digit positions agreed on, over digit positions in the gold answer."""
    gp = re.findall(r"v(?:_\d+)+", pred)
    gg = re.findall(r"v(?:_\d+)+", gold)
    hits = total = 0
    for k, g in enumerate(gg):
        gd = g.split("_")[1:]
        pd = gp[k].split("_")[1:] if k < len(gp) else []
        for j, d in enumerate(gd):
            total += 1
            hits += (j < len(pd) and pd[j] == d)
    return hits, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="+",
                    default=["pi_low", "pi_mid", "pi_high"])
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--n", type=int, default=300, help="test items per preset")
    ap.add_argument("--chunk", type=int, default=25, help="items per API call")
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--model", default=TEACHER)
    ap.add_argument("--effort", default="medium",
                    help="the arms fix this at high; #4 is not an arm, and at "
                         "high the teacher thinks past its output budget "
                         "without answering")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    c = client()
    results = {}
    print(f"self-check #4 -- hiddenness   teacher {args.model} "
          f"effort={args.effort}\n")

    for name in args.presets:
        cfg = PRESETS[name].scaled(max(args.n * 3, 300))
        t0 = time.time()
        inst = generate(args.seed, cfg)
        items = inst.test_set()[:args.n]
        chance = 1.0 / cfg.n_values
        null = demo_null(inst, items)

        # An expression that touches no table entry is solvable from the
        # skeleton alone, and the skeleton is meant to be inferable.  Those
        # items cannot test hiddenness, so they are scored separately --
        # mixing them in is what makes a leak look like competence.
        tok_in = tok_out = 0
        answered = truncated = 0
        tab = {"hit": 0, "n": 0, "dh": 0, "dt": 0}      # needs the tables
        strc = {"hit": 0, "n": 0, "dh": 0, "dt": 0}     # skeleton only
        for s in range(0, len(items), args.chunk):
            chunk = items[s:s + args.chunk]
            kwargs = request_kwargs(args.model, max_tokens=args.max_tokens,
                                    effort=args.effort)
            got, usage, stop = ask(c, inst.syntax_spec(), inst.demo_block(),
                                   cfg.n_demos, [t.expr_src for t in chunk],
                                   args.model, kwargs)
            tok_in += usage.input_tokens
            tok_out += usage.output_tokens
            truncated += (stop == "max_tokens")
            answered += sum(1 for g in got if g.strip())
            for t, g in zip(chunk, got):
                bucket = tab if (t.needs_u or t.needs_b) else strc
                bucket["n"] += 1
                bucket["hit"] += (g.strip() == t.answer_src.strip())
                h, n = digit_score(g, t.answer_src)
                bucket["dh"] += h
                bucket["dt"] += n

        exact = tab["hit"] + strc["hit"]
        acc = exact / len(items)
        dacc = (tab["dh"] + strc["dh"]) / max(1, tab["dt"] + strc["dt"])
        coverage = answered / len(items)
        # Hiddenness is only testable on the table-dependent items.  One hit
        # out of thirty at 1/4913 is already p ~ 0.006, so the test is a
        # binomial tail, not a hand-picked cutoff -- the previous threshold
        # (">2 hits") called a 1000x-chance result "hidden".
        # Against the measured null, not against uniform guessing.
        p_tab = binom_tail(tab["hit"], max(1, tab["n"]), max(chance, null["exact"]))
        suspicious = tab["n"] > 0 and p_tab < 0.01
        # A run where the teacher never got its answers out measures the
        # token budget, not hiddenness, and would sail through as "at
        # chance".  Refuse to score it.
        valid = coverage >= 0.9 and truncated == 0
        results[name] = {"preset": name, "n_values": cfg.n_values,
                         "items": len(items), "exact_hits": exact,
                         "exact": acc, "chance": chance,
                         "digit": dacc, "digit_chance": 1 / cfg.base,
                         "null": null,
                         "table_dependent": {**tab, "p_value": p_tab},
                         "skeleton_only": strc,
                         "answer_coverage": coverage, "truncated_calls": truncated,
                         "valid": valid,
                         "effort": args.effort,
                         "input_tokens": tok_in, "output_tokens": tok_out,
                         "leaks": bool(suspicious and valid),
                         "minutes": round((time.time() - t0) / 60, 1)}

        if valid:
            verdict = "LEAK SUSPECTED" if suspicious else "at chance -- hidden"
        else:
            verdict = (f"INVALID -- coverage {coverage:.0%}, "
                       f"{truncated} truncated call(s); raise --max-tokens "
                       f"or lower --chunk")
        print(f"  {name:<8} |V|={cfg.n_values:<5} n={len(items)}")
        print(f"    overall        {exact}/{len(items)} = {acc:.4f}   "
              f"digit {dacc:.4f}")
        print(f"    null (demos)   {null['exact']:.4f}"
              f"{'':<6} digit {null['digit']:.4f}"
              f"   true skeleton + {null['revealed_unary']}u/"
              f"{null['revealed_binary']}b revealed entries")
        print(f"    needs tables   {tab['hit']}/{tab['n']}"
              f"{'':<4} digit {tab['dh'] / max(1, tab['dt']):.4f}"
              f"   vs null {max(chance, null['exact']):.4f}"
              f"   p={p_tab:.2e}   <- the actual test")
        print(f"    skeleton only  {strc['hit']}/{strc['n']}"
              f"{'':<4} digit {strc['dh'] / max(1, strc['dt']):.4f}"
              f"   (inferable by design; not a leak)")
        print(f"    answers parsed {answered}/{len(items)}  truncated {truncated}")
        print(f"    tokens in={tok_in} out={tok_out}   "
              f"{results[name]['minutes']}m   {verdict}\n", flush=True)

    leaked = [k for k, v in results.items() if v["leaks"]]
    invalid = [k for k, v in results.items() if not v["valid"]]
    if invalid:
        print("verdict: INVALID for " + ", ".join(invalid) +
              " -- the teacher never produced answers, so nothing was measured")
    elif leaked:
        print("verdict: LEAK in " + ", ".join(leaked))
    else:
        print("verdict: no preset beats chance -- semantics are hidden")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
    return 2 if invalid else (1 if leaked else 0)


if __name__ == "__main__":
    raise SystemExit(main())
