#!/usr/bin/env python3
"""#19 -- the third oracle: a true skeleton over tables a student learned.

pi's two crippled oracles bracket the arms from outside:

    true skeleton + identity tables   the floor: every structural rule, no
                                      table entry.  0.222 on pi_mid/1001
    true skeleton + true tables       1.000 by construction

Between them is the band the whole experiment lives in, and nothing has
measured where the top of it actually is.  This puts a *trained student* in the
table slot and leaves the skeleton perfect, so what comes out is the most a
weights arm could score if its only remaining problem were the table.

Why it needs its own training run rather than a self-check #5 checkpoint: #5
builds its own tables from a fixed seed unrelated to any instance, and trains on
one operator.  An instance has three unary and two binary operators and its own
tables, so a #5 checkpoint has learned a function no instance contains.

Why the prediction is worth checking rather than trusting: from reach 0.710 and
pi_mid's lookup distribution the item-level score should be 0.502, but that
assumes lookup errors are *independent*.  A real student's errors are
correlated -- it gets regions of the table wrong together -- and correlated
errors concentrate failures onto fewer items, which pushes the true number up.

Usage:
  python scripts/table_ceiling.py --preset pi_mid --instance-seed 1001 \
      --steps 6000 --n-test 500
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch

from glyph.data.config import PRESETS
from glyph.data.grammar import (binary_names, parse, render_list, render_value,
                           unary_names)
from glyph.data.instance import generate
from glyph.data.interp import Interpreter
from glyph.data.tables import IdentityTables

SEED = 20260831


# What the student is allowed to train on, as a fraction of the table.
#
# `--holdout-mod` could only express "1 in N held out", so the least it could
# hide was half. The question that matters runs the other way: an agent buys a
# few hundred of 4913 entries, so the arm-relevant ceiling sits at a few percent
# seen, not ninety. Hash-based rather than modular so that the seen set is not
# an arithmetic progression, which would be an easier region than a random one.
def is_seen(key: int, frac: float) -> bool:
    return (key * 2654435761) % 100000 < frac * 100000


def seen_u(i: int, frac: float) -> bool:
    return is_seen(i, frac)


def seen_b(i: int, j: int, frac: float) -> bool:
    return is_seen(i * 7919 + j, frac)


def prompt_unary(op: str, i: int, cfg) -> str:
    return f"{op} {render_value(i, cfg)} ="


def prompt_binary(op: str, i: int, j: int, cfg) -> str:
    return f"{op} {render_value(i, cfg)} {render_value(j, cfg)} ="


def stream(inst, frac, rng, tok, batch):
    """Training examples over every operator the instance has."""
    cfg = inst.cfg
    us, bs = unary_names(cfg), binary_names(cfg)
    n = cfg.n_values
    while True:
        ids, labels = [], []
        while len(ids) < batch:
            if rng.random() < 0.5 and us:
                op = us[int(rng.integers(len(us)))]
                i = int(rng.integers(n))
                if not seen_u(i, frac):
                    continue
                p, a = prompt_unary(op, i, cfg), inst.tables.apply_unary(op, i)
            else:
                op = bs[int(rng.integers(len(bs)))]
                i, j = int(rng.integers(n)), int(rng.integers(n))
                if not seen_b(i, j, frac):
                    continue
                p, a = prompt_binary(op, i, j, cfg), inst.tables.apply_binary(op, i, j)
            pi = tok(p, add_special_tokens=False)["input_ids"]
            ai = tok(f" {render_value(a, cfg)}",
                     add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids.append(pi + ai)
            labels.append([-100] * len(pi) + ai)
        yield ids, labels


def pad(seqs, fill, device):
    m = max(len(s) for s in seqs)
    return torch.tensor([[fill] * (m - len(s)) + s for s in seqs], device=device)


class StudentTables:
    """Answers table queries from a trained model. Precomputed and memoised:
    the interpreter must not stall on a generate() call per lookup."""

    def __init__(self, cfg, u: dict, b: dict):
        self.cfg, self._u, self._b = cfg, u, b
        self.miss = 0

    def apply_unary(self, name, i):
        v = self._u.get((name, i))
        if v is None:
            self.miss += 1
            return i
        return v

    def apply_binary(self, name, i, j):
        v = self._b.get((name, i, j))
        if v is None:
            self.miss += 1
            return i
        return v


def generate_answers(model, tok, cfg, prompts, device, batch=256):
    """One batched pass; returns the decoded value index or -1."""
    from glyph.data.grammar import parse_value
    out = []
    model.eval()
    for s in range(0, len(prompts), batch):
        chunk = prompts[s:s + batch]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  padding_side="left", add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=12, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for k in range(len(chunk)):
            txt = tok.decode(gen[k][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            try:
                out.append(parse_value(txt.split()[0], cfg))
            except Exception:
                out.append(-1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pi_mid")
    ap.add_argument("--instance-seed", type=int, default=1001)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seen-frac", type=float, default=0.9,
                    help="fraction of table entries the student may train on. "
                         "0.9 reproduces the old 1-in-10 holdout; a real agent "
                         "buys a few percent.")
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-model", default=None,
                    help="directory to write the trained model to. The first "
                         "sweep did not, so adding a column meant retraining "
                         "six students to re-score items they had already "
                         "answered once.")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    device = "cuda"
    cfg = PRESETS[a.preset]
    inst = generate(a.instance_seed, cfg)

    import sys
    sys.path.insert(0, "scripts")
    from run_a0p import paired_subset
    items = paired_subset(inst, a.n_test)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    model.train()
    for step, (ids, labels) in enumerate(stream(inst, a.seen_frac, rng, tok, a.batch), 1):
        x = pad(ids, tok.pad_token_id, device)
        y = pad(labels, -100, device)
        loss = model(input_ids=x, attention_mask=(x != tok.pad_token_id).long(),
                     labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 500 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}  "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)
        if step >= a.steps:
            break
    model.config.use_cache = True

    # every entry the scored items need
    need_u, need_b = set(), set()
    for t in items:
        need_u |= set(t.needs_u); need_b |= set(t.needs_b)
    print(f"\n  entries needed by {len(items)} items: "
          f"unary {len(need_u)}  binary {len(need_b)}", flush=True)

    nu, nb = sorted(need_u), sorted(need_b)
    au = generate_answers(model, tok, cfg, [prompt_unary(o, i, cfg) for o, i in nu], device)
    ab = generate_answers(model, tok, cfg, [prompt_binary(o, i, j, cfg) for o, i, j in nb], device)
    u_map = {k: v for k, v in zip(nu, au) if v >= 0}
    b_map = {k: v for k, v in zip(nb, ab) if v >= 0}

    ok_u = lambda k: u_map.get(k, -1) == inst.tables.apply_unary(*k)
    ok_b = lambda k: b_map.get(k, -1) == inst.tables.apply_binary(*k)
    seen = [k for k in nu if seen_u(k[1], a.seen_frac)]
    unseen = [k for k in nu if not seen_u(k[1], a.seen_frac)]
    seen_bb = [k for k in nb if seen_b(k[1], k[2], a.seen_frac)]
    unseen_bb = [k for k in nb if not seen_b(k[1], k[2], a.seen_frac)]
    m = lambda xs, f: float(np.mean([f(k) for k in xs])) if xs else float("nan")
    entry_acc_u, entry_acc_b = m(nu, ok_u), m(nb, ok_b)
    reach_u, fit_u = m(unseen, ok_u), m(seen, ok_u)
    reach_b, fit_b = m(unseen_bb, ok_b), m(seen_bb, ok_b)

    # `tail`, for this measurement path.
    #
    # `inst.is_tail` asks whether an item needs an entry the run never *bought*
    # through the oracle. This student never touched the oracle -- it was
    # trained directly on the tables behind a `seen_frac` mask -- so
    # `query_log` is empty and `is_tail` would call every item tail. The
    # analogue is the same question against the mask: does the item need an
    # entry the student was never shown?
    #
    # The mask is per *value*, shared across operators: if value i is visible
    # then u0(i), u1(i) and u2(i) were all trainable. That mirrors an agent
    # buying a query that mentions i, rather than buying one operator's entry
    # in isolation.
    def item_is_tail(t):
        return (any(not seen_u(i, a.seen_frac) for _, i in t.needs_u)
                or any(not seen_b(i, j, a.seen_frac) for _, i, j in t.needs_b))

    st = StudentTables(cfg, u_map, b_map)
    interp = Interpreter(cfg, inst.skeleton, st)
    skel_only = Interpreter(cfg, inst.skeleton, IdentityTables())
    rd = lambda o: render_value(o, cfg) if isinstance(o, int) else render_list(o, cfg)

    hits = collections.Counter(); tot = collections.Counter()
    floor = 0
    tail_n = tail_hit = tail_floor = 0
    for t in items:
        tot[t.split] += 1
        ok = False
        try:
            ok = rd(interp.eval(parse(t.expr_src, cfg))) == t.answer_src
        except Exception:
            pass
        hits[t.split] += ok
        f = False
        try:
            f = rd(skel_only.eval(parse(t.expr_src, cfg))) == t.answer_src
        except Exception:
            pass
        floor += f
        if item_is_tail(t):
            tail_n += 1
            tail_hit += ok
            tail_floor += f

    overall = sum(hits.values()) / len(items)
    res = {
        "preset": a.preset, "instance_seed": a.instance_seed,
        "steps": a.steps, "lr": a.lr, "seen_frac": a.seen_frac,
        "n_test": len(items),
        "entries_needed": {"unary": len(nu), "binary": len(nb),
                           "unary_seen": len(seen), "unary_unseen": len(unseen),
                           "binary_seen": len(seen_bb), "binary_unseen": len(unseen_bb)},
        "entry_accuracy": {"unary_all": entry_acc_u, "unary_fit": fit_u,
                           "unary_reach": reach_u, "binary_all": entry_acc_b,
                           "binary_fit": fit_b, "binary_reach": reach_b},
        "item_score": {"overall": overall,
                       **{k: hits[k] / tot[k] for k in tot},
                       "tail": (tail_hit / tail_n) if tail_n else None},
        "tail_items": tail_n,
        "skeleton_ceiling": floor / len(items),
        "skeleton_ceiling_tail": (tail_floor / tail_n) if tail_n else None,
        "unparsed_lookups": st.miss,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    if a.save_model:
        d = Path(a.save_model)
        d.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(d); tok.save_pretrained(d)
        (d / "run_config.json").write_text(json.dumps(
            {k: res[k] for k in ("preset", "instance_seed", "seen_frac",
                                 "steps", "lr")}, indent=2))
        res["saved_to"] = str(d)

    print("\n" + json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
