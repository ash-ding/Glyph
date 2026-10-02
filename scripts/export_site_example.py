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


# ------------------------------------------------- protocol walkthrough
# The protocol figures replay a short run on this instance through the real
# v2 tool handlers, in a scratch workspace: no SDK, no model, no GPU. The
# answer files are written by the skeleton-only interpreter, so every score
# shown is a real score of a real file. The train arm's student is a stub:
# ids and lineage come from the real StudentPool, GPU-dependent fields are
# elided rather than invented.
import os
import re
import tempfile

from glyph.v2 import tools as V2
from glyph.v2 import workspace as W2
from glyph.v2.ledger import Ledger
from glyph.v2.report import build_report
from glyph.v2.session import Session
from glyph.v2.student import StudentPool

ELIDED = "…"


class _StubBackend:
    def train_fn(self, examples, hp, base_model, out_dir, ledger, init_checkpoint=None):
        return {}

    def make_student(self, base_model, adapter_path, prefix):
        class _S:
            last_truncated = 0

            def answer(self, exprs):
                return ["" for _ in exprs]

            def close(self):
                pass
        return _S()


def _shown(res, elide=()):
    out = {k: v for k, v in res.items() if k != "turns_remaining"}
    if "violations" in out:
        out["violations"] = {k: n for k, n in out["violations"].items() if n}
    for k in elide:
        if k in out:
            out[k] = ELIDED
    return out


def protocol_walkthrough():
    pinst = generate(SEED, cfg)
    skel = Interpreter(cfg, pinst.skeleton, IdentityTables())

    def skel_answer(src):
        o = skel.eval(parse(src, cfg))
        return R(o) if isinstance(o, (int, np.integer)) else render_list(o, cfg)

    def write_answers(path, rows):
        with open(path, "w") as f:
            for i, a in rows:
                f.write(json.dumps({"id": i, "answer": a}) + "\n")

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "run")
        paths, val_id_of, test_id_of = W2.build_workspace(pinst, run_dir)
        os.chdir(paths.root)
        try:
            out = {}
            for arm in ("no_train", "train"):
                pinst.query_log = type(pinst.query_log)()
                pinst.query_count = 0
                open(paths.queries, "w").close()
                s = Session(pinst, Ledger(), run_dir, arm=arm)
                s.val_id_of, s.test_id_of = val_id_of, test_id_of
                s.queries_path, s.work_root = paths.queries, paths.root
                s.probes = probe_set(pinst)
                s.probe_id_of = W2.probe_ids_for(pinst, s.probes)
                calls = []

                def call(tool, args, elide=()):
                    res = getattr(V2, "t_" + tool)(s, **args)
                    calls.append({"tool": tool, "args": args, "result": _shown(res, elide),
                                  "phase": s.phase})
                    return res

                u0_probe = next(t for t in s.probes if t.split == "u0").expr_src
                call("query", {"exprs": [u0_probe, src], "why": "a bare atomic; the running example"})

                comp_src = splits["comp"]["example"]["expr"]
                depth_src = splits["depth"]["example"]["expr"]

                def fresh(e):
                    tok = re.search(r"v_[^\s,()\[\]]+", e).group()
                    for k in range(cfg.n_values):
                        cand = e.replace(tok, R(k), 1)
                        if cand != e and pinst.query_violation(cand) not in ("is_test_item", "is_validation_item"):
                            return cand
                    raise AssertionError(e)

                held_src, deep_src = fresh(comp_src), fresh(depth_src)
                assert pinst.query_violation(held_src) == "contains_held_out_pair"
                assert pinst.query_violation(deep_src) == "deeper_than_demos"
                assert pinst.query_violation(pinst.val[0].expr_src) == "is_validation_item"
                call("query", {"exprs": [pinst.val[0].expr_src, held_src, deep_src, src[:-1]],
                               "why": "edges of the query policy"})

                if arm == "no_train":
                    val_rows = [(val_id_of(t), skel_answer(t.expr_src)) for t in pinst.val]
                    write_answers("val_partial.jsonl", val_rows[:10])
                    call("check_answers", {"path": "val_partial.jsonl", "set": "validation"})
                    write_answers("val_answers.jsonl", val_rows)
                    call("check_answers", {"path": "val_answers.jsonl", "set": "validation"})
                    call("submit_validation_answer", {"path": "val_answers.jsonl"})
                else:
                    s.student = StudentPool("Qwen/Qwen3-0.6B", s.ledger, work_dir=paths.root,
                                            queries_path=paths.queries, backend=_StubBackend())
                    with open(paths.queries) as f:
                        bought = [json.loads(l) for l in f if l.strip()]
                    with open("train.jsonl", "w") as f:
                        for r in bought:
                            f.write(json.dumps({"expr": r["expr"], "answer": r["out"]}) + "\n")
                        for e, a in pinst.demos:
                            f.write(json.dumps({"expr": e, "answer": a}) + "\n")
                    call("build_dataset", {"path": "train.jsonl"})
                    gpu = ("final_loss", "gpu_seconds", "gpu_seconds_remaining")
                    call("train_model", {"dataset_id": "ds1", "epochs": 3, "lr": 1e-5,
                                         "student_id": "a1b2c3d4"}, gpu)
                    call("train_model", {"dataset_id": "ds1", "epochs": 1, "lr": 1e-5,
                                         "student_id": "a1b2c3d4"}, gpu)
                    open("prefix.txt", "w").close()
                    call("infer_model", {"checkpoint": "a1b2c3d4", "input_path": "task/validation.jsonl",
                                         "output_path": "val_student.jsonl", "prefix_path": "prefix.txt"},
                         ("truncated", "gpu_seconds"))
                call("finish_practice", {"reason": "done practising"})

                s.switch_to_final()
                test_path = W2.write_test_file(paths, pinst, test_id_of, probes=s.probes,
                                               probe_id_of=s.probe_id_of)
                with open(test_path) as f:
                    test_rows = [json.loads(l) for l in f if l.strip()]
                gated = call("query", {"exprs": [u0_probe], "why": "is the oracle still open?"})
                if arm == "train":
                    call("infer_model", {"checkpoint": "a1b2c3d4", "input_path": "task/final/test.jsonl",
                                         "output_path": "test_student.jsonl", "prefix_path": "prefix.txt"},
                         ("truncated", "gpu_seconds"))
                write_answers("test_answers.jsonl", [(r["id"], skel_answer(r["expr"])) for r in test_rows])
                call("check_answers", {"path": "test_answers.jsonl", "set": "test"})
                call("submit_final_answer", {"path": "test_answers.jsonl"})

                rep = build_report(s, os.path.abspath("test_answers.jsonl"), test_id_of)
                probe_ops = rep["probe"]["by_op"]
                out[arm] = {
                    "calls": calls,
                    "gate_error": gated["error"],
                    "test_file": {"n": len(test_rows),
                                  "n_test": sum(r["id"].startswith("test_") for r in test_rows),
                                  "n_probe": sum(r["id"].startswith("probe_") for r in test_rows),
                                  "head": test_rows[:2] + [next(r for r in test_rows if r["id"].startswith("probe_"))]},
                    "report": {
                        "overall": round(rep["overall"], 4),
                        "by_split": {k: round(v, 4) for k, v in sorted(rep["by_split"].items())},
                        "tail": round(rep["tail"], 4),
                        "headroom": {k: (None if v is None else round(v, 4)) for k, v in rep["headroom"].items()},
                        "probe": {op: {"kind": d["kind"], "n": d["n"], "overall": round(d["overall"], 4),
                                       "seen": d["seen"] and {"n": d["seen"]["n"], "acc": d["seen"]["acc"] and round(d["seen"]["acc"], 4)},
                                       "unseen": d["unseen"] and {"n": d["unseen"]["n"], "acc": d["unseen"]["acc"] and round(d["unseen"]["acc"], 4)}}
                                  for op, d in sorted(probe_ops.items())},
                        "q_used": rep["covariates"]["q_used"],
                        "submissions": rep["covariates"]["submissions"],
                    },
                }
        finally:
            os.chdir(cwd)
    files = {"syntax.md": None, "demos.jsonl": len(pinst.demos), "validation.jsonl": len(pinst.val)}
    val_head = [{"id": val_id_of(v), "expr": v.expr_src} for v in pinst.val[:2]]
    return {"arms": out, "workspace": files, "n_test": len(pinst.test), "val_head": val_head}


protocol = protocol_walkthrough()


data = {
    "protocol": protocol,
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
