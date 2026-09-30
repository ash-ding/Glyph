"""#19 -- the third oracle: a true skeleton over tables a student learned.

pi's two crippled oracles bracket the arms from outside:

    true skeleton + identity tables   the floor: every structural rule, no
                                      table entry.  0.222 on pi_mid/1001
    true skeleton + true tables       1.000 by construction

Between them is the band the whole experiment lives in, and nothing has
measured where the top of it actually is.  This puts a *trained student* in the
table slot and leaves the skeleton perfect, so what comes out is the most a
weights arm could score if its only remaining problem were the table.

Ported from `scripts/table_ceiling.py`. The only structural change from that
script: `paired_subset` comes from `glyph.reference.subset` rather than
`run_a0p`, and `torch`/`transformers` are imported lazily inside the two
functions that actually need them (`train_student`, `generate_answers`) so
this module -- and the CPU unit test -- import cleanly with no GPU.
"""

from __future__ import annotations

import collections
import time

from ..data.grammar import (AtomApp, binary_names, parse, render, render_list,
                            render_value, unary_names)
from ..data.interp import Interpreter
from ..seal import headroom as _headroom

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


# The training/answering surface form is the PUBLIC syntax (bare atomic
# applications, legal since PR A) -- CLAUDE.md rule 7: one surface form for
# everyone.  The pre-2026-09-29 private format ("u0 v_a_b_c =") makes earlier
# weights numbers (e.g. the published 0.498) incomparable with new ones.
def prompt_unary(op: str, i: int, cfg) -> str:
    return render(AtomApp(op, (i,)), cfg) + " ="


def prompt_binary(op: str, i: int, j: int, cfg) -> str:
    return render(AtomApp(op, (i, j)), cfg) + " ="


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


def _student_tables_from(answer_unary, answer_binary, need_u, need_b, cfg) -> StudentTables:
    """Build a `StudentTables` from two per-entry answer callables.

    `answer_unary(name, i) -> int` and `answer_binary(name, i, j) -> int`
    return the answered value index, or a negative number when no usable
    answer is available (decode failure, unparsed generation, ...) -- such
    entries are simply left out of the table, same as v1's `if v >= 0` filter.

    This is the seam: `score_ceiling` builds these callables from a trained
    model's (batched) generations; a test can build them directly from the
    instance's true tables to get a "perfect student" with no torch involved.
    """
    u_map = {}
    for key in need_u:
        v = answer_unary(*key)
        if v is not None and v >= 0:
            u_map[key] = v
    b_map = {}
    for key in need_b:
        v = answer_binary(*key)
        if v is not None and v >= 0:
            b_map[key] = v
    return StudentTables(cfg, u_map, b_map)


def _score_on(inst, student_tables, items) -> dict:
    """True skeleton over `student_tables`, scored on `items`.

    Same metric shapes as `glyph.reference.subset.ceilings_on`: `overall`,
    `by_split`, `tail` (`None` when no item in `items` is tail), `headroom`
    against `inst.ceilings(items)["skeleton"]["overall"]`. `entries_needed`
    reports how many distinct unary/binary table entries `items` require.
    """
    cfg = inst.cfg
    interp = Interpreter(cfg, inst.skeleton, student_tables)
    rd = lambda o: render_value(o, cfg) if isinstance(o, int) else render_list(o, cfg)

    hits: collections.Counter = collections.Counter()
    tot: collections.Counter = collections.Counter()
    tail_hit = tail_n = 0
    for t in items:
        tot[t.split] += 1
        ok = False
        try:
            ok = rd(interp.eval(parse(t.expr_src, cfg))) == t.answer_src
        except Exception:
            pass
        hits[t.split] += ok
        if inst.is_tail(t):
            tail_n += 1
            tail_hit += ok

    overall = sum(hits.values()) / len(items) if items else 0.0
    by_split = {k: hits[k] / tot[k] for k in tot}
    tail = (tail_hit / tail_n) if tail_n else None

    skeleton_overall = inst.ceilings(items)["skeleton"]["overall"]

    need_u, need_b = set(), set()
    for t in items:
        need_u |= set(t.needs_u)
        need_b |= set(t.needs_b)

    return {
        "overall": overall,
        "by_split": by_split,
        "tail": tail,
        "headroom": _headroom(overall, skeleton_overall),
        "entries_needed": {"unary": len(need_u), "binary": len(need_b)},
    }


def train_student(inst, seen_frac, *, model="Qwen/Qwen3-1.7B", lr=1e-4,
                  steps=6000, batch=128, device="cuda", ops=None):
    """Train one student model on a `seen_frac` slice of `inst`'s tables.

    `ops` restricts the training stream to the given atomic operators (e.g.
    `ops=["u0"]` trains a per-op specialist); None trains on all of them.

    [GPU] The training loop from `table_ceiling.py::main` (its `stream`/`pad`/
    optimizer section). Not exercised by the CPU unit test -- `torch` and
    `transformers` are imported here, not at module scope, so importing this
    module never requires a GPU or those packages to be importable.
    """
    cfg = inst.cfg
    us, bs = unary_names(cfg), binary_names(cfg)
    if ops is not None:
        unknown = [o for o in ops if o not in us + bs]
        if unknown:
            raise ValueError(f"unknown atomic op(s) {unknown!r}")
        us = [o for o in us if o in ops]
        bs = [o for o in bs if o in ops]

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    n = cfg.n_values

    def stream(rng, tok):
        """Training examples over every operator the instance has."""
        while True:
            ids, labels = [], []
            while len(ids) < batch:
                if (rng.random() < 0.5 and us) or not bs:
                    op = us[int(rng.integers(len(us)))]
                    i = int(rng.integers(n))
                    if not seen_u(i, seen_frac):
                        continue
                    p, a = prompt_unary(op, i, cfg), inst.tables.apply_unary(op, i)
                else:
                    op = bs[int(rng.integers(len(bs)))]
                    i, j = int(rng.integers(n)), int(rng.integers(n))
                    if not seen_b(i, j, seen_frac):
                        continue
                    p, a = prompt_binary(op, i, j, cfg), inst.tables.apply_binary(op, i, j)
                pi = tok(p, add_special_tokens=False)["input_ids"]
                ai = tok(f" {render_value(a, cfg)}",
                         add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
                ids.append(pi + ai)
                labels.append([-100] * len(pi) + ai)
            yield ids, labels

    def pad(seqs, fill):
        m = max(len(s) for s in seqs)
        return torch.tensor([[fill] * (m - len(s)) + s for s in seqs], device=device)

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    net = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    net.gradient_checkpointing_enable()
    net.config.use_cache = False
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    net.train()
    for step, (ids, labels) in enumerate(stream(rng, tok), 1):
        x = pad(ids, tok.pad_token_id)
        y = pad(labels, -100)
        loss = net(input_ids=x, attention_mask=(x != tok.pad_token_id).long(),
                   labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 500 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}  "
                  f"{(time.time() - t0) / 60:.1f}m", flush=True)
        if step >= steps:
            break
    net.config.use_cache = True
    return net, tok


def generate_answers(model, tok, cfg, prompts, device, batch=256):
    """One batched pass; returns the decoded value index or -1.

    [GPU] Inference-only; `torch` is imported here, not at module scope.
    """
    import torch

    from ..data.grammar import parse_value
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


def score_ceiling(inst, model, tok, items, *, device="cuda") -> dict:
    """True skeleton over a trained student's tables, scored on `items`.

    Collects every table entry `items` need, answers each with the model (one
    batched `generate_answers` pass per operator kind), builds a
    `StudentTables` from the answers via the `_student_tables_from` seam, and
    scores the true skeleton over it exactly as `_score_on` does for the CPU
    test's fake student.
    """
    cfg = inst.cfg
    need_u, need_b = set(), set()
    for t in items:
        need_u |= set(t.needs_u)
        need_b |= set(t.needs_b)

    nu, nb = sorted(need_u), sorted(need_b)
    au = generate_answers(model, tok, cfg, [prompt_unary(o, i, cfg) for o, i in nu], device)
    ab = generate_answers(model, tok, cfg, [prompt_binary(o, i, j, cfg) for o, i, j in nb], device)
    u_ans = dict(zip(nu, au))
    b_ans = dict(zip(nb, ab))

    student_tables = _student_tables_from(
        lambda name, i: u_ans.get((name, i), -1),
        lambda name, i, j: b_ans.get((name, i, j), -1),
        need_u, need_b, cfg)

    return _score_on(inst, student_tables, items)


def score_probes(inst, probe_items, seen_frac, answer_unary, answer_binary) -> dict:
    """Score a student on bare-atomic probe items, split seen/unseen by the
    TRAINING hash (`seen_u`/`seen_b`) -- the reference-side analogue of the
    agent report's LookupLog split. CPU-pure via the answer-callable seam."""
    cfg = inst.cfg
    need_u, need_b = set(), set()
    for t in probe_items:
        need_u |= set(t.needs_u)
        need_b |= set(t.needs_b)
    tables = _student_tables_from(answer_unary, answer_binary, need_u, need_b, cfg)

    def _is_seen(t) -> bool:
        for (_, i) in t.needs_u:
            return seen_u(i, seen_frac)
        for (_, i, j) in t.needs_b:
            return seen_b(i, j, seen_frac)
        return False

    seen = [t for t in probe_items if _is_seen(t)]
    unseen = [t for t in probe_items if not _is_seen(t)]

    def _acc(sub):
        return _score_on(inst, tables, sub)["overall"] if sub else None

    return {
        "overall": _acc(probe_items),
        "n": len(probe_items),
        "seen": {"n": len(seen), "acc": _acc(seen)},
        "unseen": {"n": len(unseen), "acc": _acc(unseen)},
    }


def score_probes_model(inst, model, tok, probe_items, seen_frac, *,
                       device="cuda") -> dict:
    """[GPU] `score_probes` with the callables built from one batched
    `generate_answers` pass per operator kind."""
    cfg = inst.cfg
    need_u, need_b = set(), set()
    for t in probe_items:
        need_u |= set(t.needs_u)
        need_b |= set(t.needs_b)
    nu, nb = sorted(need_u), sorted(need_b)
    au = generate_answers(model, tok, cfg, [prompt_unary(o, i, cfg) for o, i in nu], device)
    ab = generate_answers(model, tok, cfg, [prompt_binary(o, i, j, cfg) for o, i, j in nb], device)
    u_ans, b_ans = dict(zip(nu, au)), dict(zip(nb, ab))
    return score_probes(inst, probe_items, seen_frac,
                        lambda name, i: u_ans.get((name, i), -1),
                        lambda name, i, j: b_ans.get((name, i, j), -1))
