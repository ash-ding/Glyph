"""E-1 + E-2: what a Glyph instance actually is, before any arm touches it.

Two things this settles, both prerequisites for finalising the data layer:

**E-1, the pi axis.** The presets carry an *intended* pi and the phase diagram
uses the measured one, so what matters is the measured distribution per preset:
its spread, its overlap with neighbours, and how many seeds fail to generate at
all. It also reports pi two ways -- the current definition and a stratified one
-- because `measure_pi` samples `test[:1500]`, and at full size the first 6500
items are *all* `iid`. Under the current definition `comp` and `depth` have
never entered a pi measurement, and those are exactly the splits where the
skeleton carries weight.

**E-2, the ceilings.** An arm's score is uninterpretable without them. On
pi_mid/1001 the true skeleton with an identity table scores 0.222, exactly the
fraction of items needing no table lookup at all, and A0' and A4 both landed
within one standard error of it -- so "the frontier extracted essentially no
table knowledge" and "the weights arm has not learned the skeleton" are the
same measurement read from two sides. Every arm figure needs these lines on it.

Usage:  python scripts/bench_scan.py --seeds 20 --out scan.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time

from glyph.data.config import PRESETS
from glyph.data.grammar import parse, render_list, render_value
from glyph.data.instance import GenerationFailed, generate
from glyph.data.interp import Interpreter
from glyph.data.measure import _score
from glyph.data.semantics import trivial_skeleton
from glyph.data.tables import IdentityTables


def _render(out, cfg) -> str:
    return render_value(out, cfg) if isinstance(out, int) else render_list(out, cfg)


def _pi_from(skel_interp, tab_interp, inst, items) -> dict:
    a_skel = _score(skel_interp, inst, items)
    a_tab = _score(tab_interp, inst, items)
    l_table, l_skel = 1.0 - a_skel, 1.0 - a_tab
    denom = l_table + l_skel
    return {"a_skel": round(a_skel, 4), "a_tab": round(a_tab, 4),
            "pi": round(l_skel / denom, 4) if denom > 0 else None}


def _stratified(items, n: int, rng):
    """Same size as the current sample, but drawn across all three splits."""
    by = collections.defaultdict(list)
    for t in items:
        by[t.split].append(t)
    out = []
    for split, group in sorted(by.items()):
        k = max(1, round(n * len(group) / len(items)))
        idx = rng.permutation(len(group))[:k]
        out += [group[i] for i in idx]
    return out


def scan_one(preset: str, seed: int) -> dict:
    import numpy as np
    cfg = PRESETS[preset]
    t0 = time.time()
    try:
        inst = generate(seed, cfg)
    except GenerationFailed as e:
        return {"preset": preset, "seed": seed, "ok": False,
                "failed_split": e.split, "made": e.made, "want": e.want,
                "diag": e.diag, "seconds": round(time.time() - t0, 1)}

    skel = Interpreter(cfg, inst.skeleton, IdentityTables())
    tab = Interpreter(cfg, trivial_skeleton(cfg), inst.tables)
    rng = np.random.default_rng(seed)

    # -- E-1: pi under the current definition, and stratified ------------
    cur = _pi_from(skel, tab, inst, inst.test[:1500])
    strat = _pi_from(skel, tab, inst, _stratified(inst.test, 1500, rng))
    per_split_pi = {s: _pi_from(skel, tab, inst, inst.test_set(s))["pi"]
                    for s in ("iid", "comp", "depth")}

    # -- E-2: ceilings, exact match, per split ---------------------------
    ceil = {}
    for s in ("iid", "comp", "depth"):
        group = inst.test_set(s)
        no_tab = sum(1 for t in group if not t.needs_u and not t.needs_b)
        ceil[s] = {
            "n": len(group),
            "needs_no_table": round(no_tab / len(group), 4),
            "skeleton_ceiling": round(_score(skel, inst, group, graded=False), 4),
            "table_only_ceiling": round(_score(tab, inst, group, graded=False), 4),
        }
    n_all = len(inst.test)
    no_tab_all = sum(1 for t in inst.test if not t.needs_u and not t.needs_b)

    lookups = collections.Counter(len(t.needs_u) + len(t.needs_b) for t in inst.test)
    ans = collections.Counter(t.answer_src for t in inst.test)
    need_u, need_b = set(), set()
    for t in inst.test:
        need_u |= set(t.needs_u); need_b |= set(t.needs_b)

    return {
        "preset": preset, "seed": seed, "ok": True, "n_test": n_all,
        "pi_current_iid_only": cur["pi"], "pi_stratified": strat["pi"],
        "pi_per_split": per_split_pi,
        "a_skel_graded": cur["a_skel"], "a_tab_graded": cur["a_tab"],
        "ceiling_overall": {
            "needs_no_table": round(no_tab_all / n_all, 4),
            "skeleton_ceiling": round(_score(skel, inst, inst.test, graded=False), 4),
            "table_only_ceiling": round(_score(tab, inst, inst.test, graded=False), 4),
        },
        "ceiling_per_split": ceil,
        "mean_lookups_per_item": round(
            sum(k * c for k, c in lookups.items()) / n_all, 3),
        "distinct_answers": len(ans),
        "mode_answer_frac": round(ans.most_common(1)[0][1] / n_all, 4),
        "distinct_unary_entries_needed": len(need_u),
        "distinct_binary_entries_needed": len(need_b),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="+", default=["pi_low", "pi_mid", "pi_high"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1001)
    ap.add_argument("--out", default="scan.jsonl")
    a = ap.parse_args()

    rows = []
    with open(a.out, "w") as fh:
        for preset in a.presets:
            for seed in range(a.seed0, a.seed0 + a.seeds):
                r = scan_one(preset, seed)
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                tag = "ok  " if r["ok"] else "FAIL"
                extra = (f"pi={r['pi_current_iid_only']} strat={r['pi_stratified']} "
                         f"ceil={r['ceiling_overall']['skeleton_ceiling']}"
                         if r["ok"] else f"split={r['failed_split']}")
                print(f"{tag} {preset:8s} {seed}  {extra}  {r['seconds']}s")
                sys.stdout.flush()

    print("\n" + "=" * 72)
    for preset in a.presets:
        got = [r for r in rows if r["preset"] == preset]
        ok = [r for r in got if r["ok"]]
        bad = [r for r in got if not r["ok"]]
        if not ok:
            print(f"{preset}: 0/{len(got)} generated")
            continue
        cur = sorted(r["pi_current_iid_only"] for r in ok)
        strat = sorted(r["pi_stratified"] for r in ok)
        ceil = sorted(r["ceiling_overall"]["skeleton_ceiling"] for r in ok)
        by_reason = collections.Counter(r["failed_split"] for r in bad)
        print(f"{preset:8s} generated {len(ok)}/{len(got)}   unfillable={dict(by_reason)}")
        print(f"         pi (current, iid only) median={cur[len(cur)//2]:.3f} "
              f"range=[{cur[0]:.3f}, {cur[-1]:.3f}]")
        print(f"         pi (stratified)        median={strat[len(strat)//2]:.3f} "
              f"range=[{strat[0]:.3f}, {strat[-1]:.3f}]")
        print(f"         skeleton ceiling       median={ceil[len(ceil)//2]:.3f} "
              f"range=[{ceil[0]:.3f}, {ceil[-1]:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
