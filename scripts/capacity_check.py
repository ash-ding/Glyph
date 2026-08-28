#!/usr/bin/env python3
"""Self-check #5 -- the capacity check.

The one check whose failure forces a design change rather than a bug fix.

The question: given the student and *unlimited data*, can it absorb the pi=0
end's tables?  If it cannot, every curve downstream measures the student's
capacity instead of the delegation decision, and the student choice is
mis-calibrated -- four weeks of runs answering the wrong question.

It needs no benchmark instance.  Tables are a standalone function, so this
trains directly on `Tables`, which is why it can start on day one and run in
parallel with everything else.

Two results are reported, because "can it fit" and "can it extrapolate" are
different questions and the design depends on both:

  fit    -- accuracy on entries it was trained on.  Pure capacity: can a
            1.7B model represent this function at all?  The tables have
            ~50k parameters against the student's 1.7 billion, so a low
            number here means something is wrong with the setup, not with
            the student.
  reach  -- accuracy on entries it never saw.  This is the one that matters.
            It is the entire reason the weights arm can beat a lookup table:
            digit-wise embeddings make the table a smooth landscape, and
            gradient descent interpolates it.  If `reach` collapses to
            chance, the pi=0 end has no winner -- the weights arm and the
            code arm lie down together and the phase diagram loses the half
            that carries the paper.

Unary and binary are reported SEPARATELY and never averaged.  Binary is
harder by orders of magnitude -- 4913^2 pairs against 4913 values -- so a
combined number hides exactly the failure D2 is about.

Sweeping `--coupling` is D2's verification:

    0     purely digit-wise.  Learnable, but only base^2 * n_digits distinct
          parts -- an agent can enumerate them into a lookup table, which
          hands the win to the code arm.
    small digit-wise dominant, weak coupling.  The target: learnable, not
          tabulatable.
    large approaching a naive joint MLP.  Nobody can write it down, and the
          student cannot learn it either.

Usage:
    python scripts/capacity_check.py --task unary
    python scripts/capacity_check.py --task binary --coupling 0.25
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from glyph.config import PRESETS
from glyph.grammar import digits, render_value
from glyph.tables import Tables

# The pi=0 end is the only place this question is interesting: it is where
# the tables carry the difficulty.
CFG = PRESETS["pi_low"]
HOLDOUT_MOD = 10          # 1 in 10 entries is never trained on; 0 = none
SEED = 20260827


def mode_baseline(tables, task: str, n: int = 4000, seed: int = 11) -> dict:
    """What "always answer the most common output" scores.

    Uniform chance is the wrong reference when the table's image is skewed:
    a student that has learned only the output distribution already beats it.
    Under the old `nearest` decode this baseline reached 0.187 exact and
    0.582 digit on pi_low -- more than a third of what the trained model got
    -- so a reach number without it beside it overstates what was learned.
    """
    import collections

    cfg = tables.cfg
    rng = np.random.default_rng(seed)
    outs = []
    for _ in range(n):
        i = int(rng.integers(cfg.n_values))
        outs.append(tables.apply_unary("u0", i) if task == "unary"
                    else tables.apply_binary("b0", i, int(rng.integers(cfg.n_values))))
    mode, hits = collections.Counter(outs).most_common(1)[0]
    md = digits(mode, cfg)
    dh = dt = 0
    for o in outs:
        for a, b in zip(digits(o, cfg), md):
            dt += 1
            dh += (a == b)
    return {"exact": hits / len(outs), "digit": dh / max(1, dt),
            "distinct": len(set(outs))}


def build_tables(coupling: float, base: int, n_digits: int,
                 value_form: str = "underscore",
                 unary_coupling: float | None = None) -> Tables:
    """The pi=0 table config, with D3's digit layout left open.

    Self-check #6 turned D3 into a live decision: Qwen3 splits numbers into
    single characters, so a base-17 digit like `13` always spans two tokens
    and the digit positions become variable-width.  At base <= 10 every
    digit is one character and the layout is fixed-width.  Whether that
    actually helps the student is an empirical question -- which is what
    this check answers.
    """
    cfg = CFG.with_(binary_coupling=coupling, base=base, n_digits=n_digits,
                    value_form=value_form, unary_coupling=unary_coupling)
    return Tables(cfg, np.random.default_rng(SEED))


def is_holdout_unary(i: int) -> bool:
    return HOLDOUT_MOD > 0 and i % HOLDOUT_MOD == 0


def is_holdout_binary(i: int, j: int) -> bool:
    # Deterministic, O(1), and independent of the digit structure so the
    # held-out set is not accidentally an easier or harder region.
    return HOLDOUT_MOD > 0 and (i * 7919 + j) % HOLDOUT_MOD == 0


def render_example(task: str, tables: Tables, i: int, j: int | None) -> tuple[str, str]:
    """Returns (prompt, answer) in the one surface form everybody sees."""
    cfg = tables.cfg
    if task == "unary":
        out = tables.apply_unary("u0", i)
        return f"u0 {render_value(i, cfg)} =", f" {render_value(out, cfg)}"
    out = tables.apply_binary("b0", i, j)
    return (f"b0 {render_value(i, cfg)} {render_value(j, cfg)} =",
            f" {render_value(out, cfg)}")


class Stream(IterableDataset):
    """Unlimited training data: entries are sampled fresh, forever, from
    everything outside the held-out set."""

    def __init__(self, task: str, tables: Tables, tok, seed: int):
        self.task, self.tables, self.tok, self.seed = task, tables, tok, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0))
        n = self.tables.cfg.n_values
        while True:
            i = int(rng.integers(n))
            if self.task == "unary":
                if is_holdout_unary(i):
                    continue
                j = None
            else:
                j = int(rng.integers(n))
                if is_holdout_binary(i, j):
                    continue
            yield encode(self.tok, *render_example(self.task, self.tables, i, j))


def encode(tok, prompt: str, answer: str) -> dict:
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = p + a
    # Loss on the answer only -- the prompt is given, not predicted.
    labels = [-100] * len(p) + a
    return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id: int):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), width), -100, dtype=torch.long)
    att = torch.zeros((len(batch), width), dtype=torch.long)
    for k, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[k, :n] = torch.tensor(b["input_ids"])
        lab[k, :n] = torch.tensor(b["labels"])
        att[k, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def eval_split(model, tok, tables: Tables, task: str, held_out: bool,
               n: int, batch: int, device) -> dict:
    """Greedy decode, then exact match plus digit-level partial credit.

    Digit accuracy is the leading indicator: a model that is learning the
    digit structure but has not nailed it shows up here well before exact
    match moves off the floor.
    """
    rng = np.random.default_rng(SEED + (1 if held_out else 2))
    cfg = tables.cfg
    items = []
    while len(items) < n:
        i = int(rng.integers(cfg.n_values))
        if task == "unary":
            if is_holdout_unary(i) != held_out:
                continue
            j = None
        else:
            j = int(rng.integers(cfg.n_values))
            if is_holdout_binary(i, j) != held_out:
                continue
        items.append(render_example(task, tables, i, j))

    tok.padding_side = "left"
    exact = digit_hits = digit_total = 0
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        enc = tok([p for p, _ in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=False).to(device)
        want = max(len(tok(a, add_special_tokens=False)["input_ids"]) for _, a in chunk)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=want + 2, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (_, gold), got in zip(chunk, gen):
            g, w = got.strip(), gold.strip()
            exact += (g == w)
            gd, wd = g.split("_")[1:], w.split("_")[1:]
            for k in range(len(wd)):
                digit_total += 1
                digit_hits += (k < len(gd) and gd[k] == wd[k])
    return {"exact": exact / len(items), "digit": digit_hits / max(1, digit_total)}


def main() -> int:
    global HOLDOUT_MOD
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["unary", "binary"], required=True)
    ap.add_argument("--coupling", type=float, default=CFG.binary_coupling)
    ap.add_argument("--base", type=int, default=CFG.base)
    ap.add_argument("--n-digits", type=int, default=CFG.n_digits)
    ap.add_argument("--value-form", default="underscore")
    ap.add_argument("--unary-coupling", type=float, default=None,
                    help="give the unary tables the digit-wise form binary has")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--holdout-mod", type=int, default=HOLDOUT_MOD,
                    help="1 entry in N is never trained on; 0 trains on "
                         "everything, which answers 'can it hold the whole "
                         "table' rather than 'can it extrapolate'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    HOLDOUT_MOD = args.holdout_mod

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    device = "cuda"
    tables = build_tables(args.coupling, args.base, args.n_digits,
                          args.value_form, args.unary_coupling)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    label = (f"{args.task} coupling={args.coupling:g} "
             f"base={args.base}^{args.n_digits} form={args.value_form}"
             + ("" if args.unary_coupling is None
                else f" unary_dw={args.unary_coupling:g}"))
    print(f"=== capacity check: {label} ===", flush=True)
    print(f"    student {args.model} | table params {tables.param_count():,} "
          f"| |V| {tables.cfg.n_values}", flush=True)

    loader = DataLoader(
        Stream(args.task, tables, tok, SEED), batch_size=args.batch,
        num_workers=4, collate_fn=lambda b: collate(b, tok.pad_token_id))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)

    model.train()
    t0, run = time.time(), 0.0
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run += loss.item()
        if step % 200 == 0:
            print(f"    step {step:>5}/{args.steps}  loss {run / 200:.4f}"
                  f"  {(time.time() - t0) / 60:.1f}m", flush=True)
            run = 0.0
        if step >= args.steps:
            break

    model.eval()
    model.config.use_cache = True
    fit = eval_split(model, tok, tables, args.task, False, args.eval_n, args.eval_batch, device)
    # With no held-out region there is nothing to reach for; reporting a
    # `reach` there would be reporting `fit` twice under two names.
    reach = ({"exact": float("nan"), "digit": float("nan")} if HOLDOUT_MOD == 0
             else eval_split(model, tok, tables, args.task, True, args.eval_n,
                             args.eval_batch, device))
    chance = 1.0 / tables.cfg.n_values
    mode_ref = mode_baseline(tables, args.task)

    print(f"\n    fit    exact {fit['exact']:.3f}   digit {fit['digit']:.3f}")
    print(f"    reach  exact {reach['exact']:.3f}   digit {reach['digit']:.3f}")
    print(f"    mode   exact {mode_ref['exact']:.3f}   digit {mode_ref['digit']:.3f}"
          f"   ({mode_ref['distinct']} distinct outputs seen)")
    print(f"    chance exact {chance:.5f}   digit {1 / tables.cfg.base:.3f}")
    if HOLDOUT_MOD == 0:
        verdict = ("HOLDS THE WHOLE TABLE" if fit["exact"] > 0.99
                   else f"cannot hold the whole table: fit {fit['exact']:.3f}")
    else:
        # Beating uniform chance is not enough: the output distribution is
        # free information. What matters is clearing the mode baseline.
        verdict = ("LEARNABLE"
                   if (reach["exact"] > 10 * chance
                       and reach["exact"] > 1.5 * mode_ref["exact"]
                       and reach["digit"] > mode_ref["digit"])
                   else "NOT REACHED -- design decision required")
    print(f"    verdict: {verdict}\n", flush=True)

    result = {"task": args.task, "coupling": args.coupling,
              "base": args.base, "n_digits": args.n_digits,
              "value_form": args.value_form, "unary_coupling": args.unary_coupling,
              "model": args.model,
              "steps": args.steps, "batch": args.batch, "lr": args.lr,
              "holdout_mod": HOLDOUT_MOD,
              "fit": fit, "reach": reach, "chance": chance,
              "mode_baseline": mode_ref, "decode": tables.cfg.decode,
              "verdict": verdict,
              "minutes": round((time.time() - t0) / 60, 1)}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
