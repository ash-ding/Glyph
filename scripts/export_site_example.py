"""Export one real instance's construction, step by step, for the project page.

    PYTHONPATH=src python export_example.py > site/assets/js/example-data.js

Every number is read from the generator; nothing is typed by hand.  Where the
page shows a re-derivation (sampling decisions, evaluation trace), it is
checked here against the real code path and the script fails on a mismatch.
"""
import collections
import dataclasses
import json
import sys

import numpy as np

from glyph.data import PRESETS, generate
from glyph.data.config import GlyphConfig
from glyph.data.grammar import (App, Lit, SHAPE_RESULT, STRUCT_SHAPES, Val,
                                binary_names, depth, enabled_ops, op_pairs,
                                parse, render, render_list, render_value,
                                syntax_spec, unary_names)
from glyph.data.interp import Interpreter
from glyph.data.semantics import (ARG_K, GEven, GFirstEqLast, GLenGt, TDedup,
                                  TDrop, TIdent, TMapAll, TMapSkip, TReverse,
                                  TRotate, TSeq, TTake, describe,
                                  trivial_skeleton)
from glyph.data.tables import IdentityTables

PRESET, SEED, FROZEN_ID = "pi_mid", 1019, "mid_3"
EXAMPLE_DEMO = 0

cfg = PRESETS[PRESET]
inst = generate(SEED, cfg)
T, P = inst.tables, inst.P
R = lambda v: render_value(int(v), cfg)
rnd = lambda a, n=2: [round(float(x), n) for x in np.asarray(a).ravel()]


# ---------------------------------------------------------------- config
FIELDS = [f.name for f in dataclasses.fields(GlyphConfig)]
presets = {p: {f: getattr(PRESETS[p], f) for f in FIELDS} for p in ("pi_low", "pi_mid", "pi_high")}
for p in presets.values():
    p["list_len_range"] = list(p["list_len_range"])


# --------------------------------------------------------------- skeleton
def t_json(t):
    k = lambda v: "k" if v == ARG_K else v
    if isinstance(t, TSeq):
        return {"seq": [t_json(t.a), t_json(t.b)]}
    if isinstance(t, TIdent): return {"prim": "ident"}
    if isinstance(t, TReverse): return {"prim": "reverse"}
    if isinstance(t, TDedup): return {"prim": "dedup"}
    if isinstance(t, TRotate): return {"prim": f"rotate {k(t.k)}"}
    if isinstance(t, TTake): return {"prim": f"take {k(t.k)}"}
    if isinstance(t, TDrop): return {"prim": f"drop {k(t.k)}"}
    if isinstance(t, TMapAll): return {"prim": "map_all u"}
    if isinstance(t, TMapSkip): return {"prim": f"map_skip u {t.j}"}
    raise TypeError(t)


def g_text(g):
    if isinstance(g, GEven): return "len is even"
    if isinstance(g, GLenGt): return f"len > {g.k}"
    if isinstance(g, GFirstEqLast): return "first == last"
    raise TypeError(g)


def flat(t):
    return flat(t.a) + flat(t.b) if isinstance(t, TSeq) else [t_json(t)["prim"]]


def derivation(sem):
    """Leftmost expansion of the combinator grammar, one choice per step."""
    steps = [{"form": "‹structural›", "note": "start"}]
    if sem.guard is not None:
        cur = ["if", "‹guard›", "then", "‹transform›", "else", "‹transform›"]
        steps.append({"form": " ".join(cur), "note": "guarded form  (probability guard_prob)",
                      "knob": "guard_prob"})
        cur[1] = g_text(sem.guard)
        steps.append({"form": " ".join(cur), "note": "guard drawn uniformly from 3 kinds"})
        slots = [(3, sem.then_t), (5, sem.else_t)]
    else:
        cur = ["‹transform›"]
        steps.append({"form": " ".join(cur), "note": "plain form  (probability 1 − guard_prob)",
                      "knob": "guard_prob"})
        slots = [(0, sem.then_t)]

    def expand(cur, idx, t, steps):
        if isinstance(t, TSeq):
            cur[idx:idx + 1] = ["‹transform›", ";", "‹transform›"]
            steps.append({"form": " ".join(cur), "note": "split into a sequence  (45% while depth remains)",
                          "knob": "max_transform_depth"})
            n_after = expand(cur, idx, t.a, steps)
            return n_after + 1 + expand(cur, idx + n_after + 1, t.b, steps)
        cur[idx] = t_json(t)["prim"]
        steps.append({"form": " ".join(cur), "note": "primitive drawn uniformly"})
        return 1

    offset = 0
    for pos, t in slots:
        offset += expand(cur, pos + offset, t, steps) - 1
    if sem.fold:
        steps.append({"form": " ".join(cur) + f"   ⟹ fold-{'left' if sem.fold == 'L' else 'right'} with b",
                      "note": "fold direction drawn 50/50", "knob": None})
    return steps


skeleton = []
for op, shape in enabled_ops(cfg):
    sem = inst.skeleton[op]
    skeleton.append({
        "op": op, "shape": shape,
        "guard": g_text(sem.guard) if sem.guard else None,
        "then": flat(sem.then_t), "else": flat(sem.else_t) if sem.else_t else None,
        "fold": sem.fold, "describe": describe(sem),
    })


# ------------------------------------------------------- running example
src, gold = inst.demos[EXAMPLE_DEMO]
expr = parse(src, cfg)
assert render(expr, cfg) == src and inst.query(src) == gold


def sampling_decisions(e):
    """Re-derive the choices `_sample.go` made to produce `e`."""
    ops = enabled_ops(cfg)
    out = []

    def go(node, ttype, budget, path):
        cands = [o for o, s in ops if SHAPE_RESULT[s] == ttype]
        atomic = [o for o, s in ops if SHAPE_RESULT[s] == ttype and s in ("UL", "LB")]
        pure = [o for o, s in ops if SHAPE_RESULT[s] == ttype and s in ("L", "KL")]
        if isinstance(node, (Lit, Val)):
            why = "budget exhausted" if budget <= 0 else "stopped early (depth_stop_prob)"
            out.append({"path": path, "type": ttype, "budget": budget, "leaf": True,
                        "why": why, "value": render_list(list(node.items), cfg)
                        if isinstance(node, Lit) else R(node.idx),
                        "len": len(node.items) if isinstance(node, Lit) else None})
            return
        shape = dict(ops)[node.op]
        if not pure:
            rule = "forced: only fold-like ops return a value"
        elif node.op in atomic:
            rule = "coin < atomic_ratio → table-consuming pool"
        else:
            rule = "coin ≥ atomic_ratio → pure-structure pool"
        slots = []
        for slot, a in zip(shape, node.args):
            if slot == "U": slots.append({"slot": "unary", "value": a, "from": unary_names(cfg)})
            elif slot == "B": slots.append({"slot": "binary", "value": a, "from": binary_names(cfg)})
            elif slot == "K": slots.append({"slot": "int", "value": a, "from": [1, 2, 3]})
            else: slots.append({"slot": "list"})
        out.append({"path": path, "type": ttype, "budget": budget, "leaf": False,
                    "cands": cands, "atomic": atomic, "pure": pure, "rule": rule,
                    "op": node.op, "shape": shape, "slots": slots})
        for slot, a in zip(shape, node.args):
            if slot == "L":
                go(a, "LIST", budget - 1, path + [node.op])

    root_type = "VAL" if SHAPE_RESULT[dict(ops)[expr.op]] == "VAL" else "LIST"
    go(e, root_type, cfg.demo_max_depth, [])
    return root_type, out


root_type, decisions = sampling_decisions(expr)


def trace(e):
    """Bottom-up evaluation, mirroring Interpreter._eval, with every step logged."""
    steps = []

    def ev(node):
        if isinstance(node, Lit):
            v = list(node.items)
            steps.append({"kind": "literal", "out": [R(x) for x in v]})
            return v
        shape = dict(enabled_ops(cfg))[node.op]
        sem = inst.skeleton[node.op]
        unary = binary = k = None
        sub = None
        for slot, a in zip(shape, node.args):
            if slot == "U": unary = a
            elif slot == "B": binary = a
            elif slot == "K": k = a
            else: sub = a
        lst = ev(sub)
        rec = {"kind": "op", "op": node.op, "shape": shape, "in": [R(x) for x in lst],
               "unary": unary, "binary": binary, "k": k, "prims": []}
        if sem.guard is not None:
            t = P._pick(sem, lst)
            rec["guard"] = {"text": g_text(sem.guard), "holds": t is sem.then_t,
                            "len": len(lst)}
        else:
            t = sem.then_t
        cur = lst
        for prim_t in (_leaves(t)):
            before = cur
            lk = []
            if isinstance(prim_t, (TMapAll, TMapSkip)):
                j = prim_t.j % len(before) if isinstance(prim_t, TMapSkip) else None
                for idx, x in enumerate(before):
                    if idx != j:
                        lk.append({"op": unary, "args": [R(x)], "out": R(T.apply_unary(unary, x))})
            cur = P._apply(prim_t, before, unary, k, None)
            rec["prims"].append({"prim": t_json(prim_t)["prim"].replace(" k", f" {k}" if k else " k"),
                                 "in": [R(x) for x in before], "out": [R(x) for x in cur],
                                 "lookups": lk})
        if sem.fold:
            folds, acc = [], None
            if len(cur) == 1:
                acc = cur[0]
            elif sem.fold == "L":
                acc = cur[0]
                for x in cur[1:]:
                    y = T.apply_binary(binary, acc, x)
                    folds.append({"op": binary, "args": [R(acc), R(x)], "out": R(y)}); acc = y
            else:
                acc = cur[-1]
                for x in reversed(cur[:-1]):
                    y = T.apply_binary(binary, x, acc)
                    folds.append({"op": binary, "args": [R(x), R(acc)], "out": R(y)}); acc = y
            rec["fold"] = {"dir": sem.fold, "steps": folds, "out": R(acc)}
            rec["out"] = R(acc)
            steps.append(rec)
            return acc
        rec["out"] = [R(x) for x in cur]
        steps.append(rec)
        return cur

    out = ev(e)
    return steps, out


def _leaves(t):
    return _leaves(t.a) + _leaves(t.b) if isinstance(t, TSeq) else [t]


trace_steps, trace_out = trace(expr)
assert (R(trace_out) if isinstance(trace_out, (int, np.integer)) else render_list(trace_out, cfg)) == gold


# ------------------------------------------------------- one table lookup
def unary_internals(name, i):
    op = T._u[name]
    ds = [int(d) for d in __import__("glyph.data.grammar", fromlist=["digits"]).digits(i, cfg)]
    parts = [op.per_digit[k](T.digit_emb[k][ds[k]]) for k in range(cfg.n_digits)]
    y_parts = np.concatenate(parts)
    mix = op.alpha * op.mix(T.embed(i))
    y = y_parts + mix
    assert np.allclose(y, T._raw(op, i))
    z = (y - T._y_mu) / T._y_sd * T._e_sd + T._e_mu
    d2 = ((T.all_emb - z) ** 2).sum(axis=1)
    top = np.argsort(d2)[:4]
    assert int(top[0]) == T.apply_unary(name, i)
    return {
        "op": name, "in": R(i), "digits": ds, "letters": R(i).split("_")[1:],
        "emb": [rnd(T.digit_emb[k][ds[k]]) for k in range(cfg.n_digits)],
        "parts": [rnd(p) for p in parts], "mix": rnd(mix), "alpha": op.alpha,
        "y": rnd(y), "z": rnd(z),
        "nearest": [{"value": R(j), "dist": round(float(np.sqrt(d2[j])), 2)} for j in top],
        "out": R(int(top[0])),
    }


def binary_internals(name, i, j):
    from glyph.data.grammar import digits
    op = T._b[name]
    di, dj = digits(i, cfg), digits(j, cfg)
    parts = [op.per_digit[k](np.concatenate([T.digit_emb[k][di[k]], T.digit_emb[k][dj[k]]]))
             for k in range(cfg.n_digits)]
    mix = op.alpha * op.mix(np.concatenate([T.embed(i), T.embed(j)]))
    y = np.concatenate(parts) + mix
    z = (y - T._y_mu) / T._y_sd * T._e_sd + T._e_mu
    d2 = ((T.all_emb - z) ** 2).sum(axis=1)
    top = np.argsort(d2)[:4]
    assert int(top[0]) == T.apply_binary(name, i, j)
    return {"op": name, "in": [R(i), R(j)], "digit_pairs": [[int(a), int(b)] for a, b in zip(di, dj)],
            "parts": [rnd(p) for p in parts], "mix": rnd(mix), "alpha": op.alpha,
            "nearest": [{"value": R(v), "dist": round(float(np.sqrt(d2[v])), 2)} for v in top],
            "out": R(int(top[0]))}


lit = expr.args[0].args[1]
first_val = lit.items[0]
u_detail = unary_internals(expr.args[0].args[0], first_val)
fold_rec = [s for s in trace_steps if s.get("fold")][0]
fa = fold_rec["fold"]["steps"][0]["args"]
from glyph.data.grammar import parse_value
b_detail = binary_internals(fold_rec["binary"], parse_value(fa[0], cfg), parse_value(fa[1], cfg))

used = {(k, d) for k, d in enumerate(u_detail["digits"])}
digit_bank = [[rnd(T.digit_emb[k][d]) for d in range(cfg.base)] for k in range(cfg.n_digits)]


# ------------------------------------------------------------ held pairs
ops = enabled_ops(cfg)
shape = dict(ops)
realizable = sorted((a, b) for a, _ in ops for b, _ in ops if SHAPE_RESULT[shape[b]] == "LIST")
by_kind = collections.Counter((shape[a], shape[b]) for a, b in realizable)
held_kind = collections.Counter((shape[a], shape[b]) for a, b in inst.held_pairs)
assert sum(held_kind.values()) == len(realizable) // 3


# ---------------------------------------------------------------- splits
def item_view(t):
    return {"expr": t.expr_src, "answer": t.answer_src, "depth": depth(parse(t.expr_src, cfg)),
            "held": [list(p) for p in sorted(op_pairs(parse(t.expr_src, cfg)) & inst.held_pairs)]}


splits = {}
for name in ("iid", "comp", "depth"):
    items = inst.test_set(name)
    hist = collections.Counter(depth(parse(t.expr_src, cfg)) for t in items)
    pick = min(items, key=lambda t: (len(t.expr_src), t.expr_src)) if name != "depth" else \
        min((t for t in items if depth(parse(t.expr_src, cfg)) == max(hist)), key=lambda t: len(t.expr_src))
    splits[name] = {"n": len(items), "depth_hist": dict(sorted(hist.items())), "example": item_view(pick)}
val_hist = collections.Counter(depth(parse(t.expr_src, cfg)) for t in inst.val)
from glyph.data.probe import probe_set
probes = probe_set(inst)
probe_by_op = collections.Counter(t.split for t in probes)
probe_hist = collections.Counter(depth(parse(t.expr_src, cfg)) for t in probes)
probe_kind = {op: ("atomic" if any(t.needs_u or t.needs_b for t in probes if t.split == op) else "structural")
              for op in probe_by_op}
for t in probes:
    assert depth(parse(t.expr_src, cfg)) == (0 if probe_kind[t.split] == "atomic" else 1)
assert {op for op, k in probe_kind.items() if k == "structural"} ==     {op for op, sh in enabled_ops(cfg) if sh in ("L", "KL")}
demo_hist = collections.Counter(depth(parse(e, cfg)) for e, _ in inst.demos)


# -------------------------------------------------------------------- pi
pi = inst.measured_pi()
skel_only = Interpreter(cfg, inst.skeleton, IdentityTables())
tab_only = Interpreter(cfg, trivial_skeleton(cfg), inst.tables)
render_out = lambda o: R(o) if isinstance(o, (int, np.integer)) else render_list(o, cfg)
crippled = {"skeleton_only": render_out(skel_only.eval(expr)),
            "table_only": render_out(tab_only.eval(expr)), "gold": gold}


# ------------------------------------------------- frozen instance battery
with open("docs/benchmark/frozen_instances.json") as f:
    frozen_manifest = json.load(f)["instances"]
with open("docs/benchmark/reference_ceilings.json") as f:
    ceilings = json.load(f)
frozen = []
for entry in frozen_manifest:
    c = ceilings[entry["id"]]
    frozen.append({
        "id": entry["id"], "band": entry["band"], "preset": entry["preset"], "seed": entry["seed"],
        "pi": round(entry["measured_pi"], 4),
        **{o: {"overall": round(c[o]["overall"], 4),
               "by_split": {k: round(v, 4) for k, v in c[o]["by_split"].items()}}
           for o in ("skeleton", "table", "perfect")},
    })
assert any(r["id"] == FROZEN_ID and r["seed"] == SEED and r["preset"] == PRESET for r in frozen)


data = {
    "frozen": frozen,
    "instance": {"preset": PRESET, "seed": SEED, "frozen_id": FROZEN_ID,
                 "n_values": cfg.n_values, "pi": {k: round(float(v), 4) for k, v in pi.items()}},
    "presets": presets,
    "syntax_spec": syntax_spec(cfg),
    "struct_shapes": [list(x) for x in STRUCT_SHAPES],
    "skeleton": skeleton,
    "derivations": {op: derivation(inst.skeleton[op]) for op in ("s0", "s1")},
    "example": {"expr": src, "answer": gold, "root_type": root_type, "depth": depth(expr),
                "pairs": sorted(map(list, op_pairs(expr))), "decisions": decisions,
                "trace": trace_steps},
    "unary_lookup": u_detail, "binary_lookup": b_detail,
    "digit_bank": digit_bank,
    "held": {"ops": [list(x) for x in ops], "realizable": [list(p) for p in realizable],
             "held": sorted(map(list, inst.held_pairs)),
             "classes": [{"outer": a, "inner": b, "n": n, "held": held_kind.get((a, b), 0)}
                         for (a, b), n in sorted(by_kind.items())]},
    "probes": {
        "n": len(probes), "depth_hist": dict(sorted(probe_hist.items())),
        "by_op": [{"op": op, "kind": probe_kind[op], "n": n} for op, n in sorted(probe_by_op.items())],
        "examples": [{"op": op, "kind": probe_kind[op],
                      "expr": next(t for t in probes if t.split == op).expr_src,
                      "answer": next(t for t in probes if t.split == op).answer_src}
                     for op in ("u0", "b0", "s2", "s3")],
    },
    "splits": splits, "n_val": len(inst.val), "val_depth_hist": dict(sorted(val_hist.items())),
    "demos": [{"expr": e, "answer": a} for e, a in inst.demos[:8]], "n_demos": len(inst.demos),
    "demo_depth_hist": dict(sorted(demo_hist.items())),
    "crippled": crippled,
}
sys.stdout.write("/* Generated by export_example.py from the generator itself. Do not edit. */\n")
sys.stdout.write("window.GLYPH_EXAMPLE = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
