#!/usr/bin/env python3
"""Run the reference oracle battery for a frozen instance -> reference_ceilings.json.

Five oracles bracket/probe an arm's ceiling on a frozen instance's paired
subset: three are CPU-cheap (`skeleton`, `table`, `perfect`, from
`glyph.reference.subset.ceilings_on`), one needs a GPU (`weights`, a trained
student's tables under the true skeleton), and one needs a real frontier
call (`a0prime`, the information-saturated retrieval upper bound). Results
are merged per-oracle into a single JSON file so re-running one oracle for
one instance never clobbers another oracle's results, or another instance's.

Usage:
    python tools/run_reference.py --instance low_1 --only skeleton,table,perfect
    python tools/run_reference.py --instance all --only weights --seen-frac 0.02 0.05 0.10
    python tools/run_reference.py --instance all --only a0prime

The GPU (`weights`) and API (`a0prime`) branches are only entered when
explicitly requested via `--only`; the default `--only` is the three CPU
oracles.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glyph.data import PRESETS, generate  # noqa: E402
from glyph.reference.a0prime import buy_evidence, run_a0prime  # noqa: E402
from glyph.reference.frozen import load_frozen  # noqa: E402
from glyph.reference.subset import ceilings_on, paired_subset  # noqa: E402
from glyph.reference.weights_ceiling import score_ceiling, train_student  # noqa: E402

CHEAP_ORACLES = ("skeleton", "table", "perfect")
ALL_ORACLES = CHEAP_ORACLES + ("weights", "a0prime")


def merge_reference(path, instance_id: str, oracle: str, payload) -> dict:
    """Merge one oracle's result for one instance into the ceilings file at `path`.

    Loads the JSON at `path` (`{}` if it does not exist yet), sets
    `data.setdefault(instance_id, {})[oracle] = payload`, writes the result
    back as pretty JSON, and returns the updated full dict.

    Idempotent per oracle: re-merging a different oracle for the same
    instance keeps what was already there; re-merging the same oracle
    overwrites only that oracle's entry; other instances are untouched.
    """
    p = Path(path)
    if p.exists():
        with p.open() as f:
            data = json.load(f)
    else:
        data = {}

    data.setdefault(instance_id, {})[oracle] = payload

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    return data


def run_cheap(inst, items) -> dict:
    """The three CPU oracles -- skeleton/table/perfect -- for `items`.

    `ceilings_on` already returns exactly this shape
    (`{skeleton, table, perfect}`, each `{overall, by_split, tail,
    headroom}`), so this just forwards to it.
    """
    return ceilings_on(inst, items)


def _evidence_text(facts: list[tuple[str, str]]) -> str:
    """Render bought (expr, answer) facts as an evidence block for the frontier."""
    return "\n".join(f"{src} = {answer}" for src, answer in facts)


_ANSWER_LINE_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$")


def _align_answers(reply: str, n: int) -> list[str]:
    """Parse `reply` into `n` answers, aligned by leading `N.` index -- not
    by line position.

    Each non-blank line of `reply` is matched against `^\\s*(\\d+)\\.\\s*(.*)$`;
    a match places its text at position `idx - 1` (1-based index -> 0-based
    slot). Lines that do not match (a preamble, an acknowledgement, trailing
    commentary, ...) are ignored rather than shifting every later answer.
    Any index in `1..n` that is never matched is left as `""`, so a missing
    or garbled answer scores wrong on its own item instead of misaligning
    the rest.
    """
    out = [""] * n
    for line in reply.splitlines():
        m = _ANSWER_LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= n:
            out[idx - 1] = m.group(2).strip()
    return out


def _frontier_answer_fn(model: str | None = None):
    """[API] Build a real-frontier `answer_fn` for `run_a0prime`.

    Sends `evidence` plus a numbered list of expressions to the teacher
    model (via `glyph.vertex.chat`), requiring each answer on its own line
    prefixed with its 1-based index (`N. <answer>`), and aligns the reply
    back to `expr_srcs` by that index via `_align_answers` -- not by line
    position, so a preamble or acknowledgement line from the model cannot
    silently shift every answer by one. Imports `glyph.vertex` lazily so
    this module -- and the CPU unit test -- import cleanly with no Vertex
    credentials configured. Never called except from the CLI's `a0prime`
    branch, which itself is only entered when `a0prime` is explicitly
    requested via `--only`.
    """
    from glyph import vertex

    def answer_fn(evidence: str, expr_srcs: list[str]) -> list[str]:
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(expr_srcs))
        prompt = (
            "You are given evidence about a hidden symbolic language, then a "
            "numbered list of expressions in that language. Answer each one "
            "on its own line, prefixed with its number exactly like "
            "\"1. <answer>\", with nothing else on that line and no other "
            "text before, between, or after the answers.\n\n"
            f"Evidence:\n{evidence}\n\nExpressions:\n{numbered}"
        )
        text = vertex.chat(prompt, model=model or vertex.TEACHER)
        return _align_answers(text, len(expr_srcs))

    return answer_fn


def _parse_only(only: str) -> list[str]:
    oracles = [o.strip() for o in only.split(",") if o.strip()]
    unknown = [o for o in oracles if o not in ALL_ORACLES]
    if unknown:
        raise ValueError(f"unknown oracle(s) {unknown!r}; choose from {ALL_ORACLES!r}")
    return oracles


def _select_instances(manifest: list[dict], instance: str) -> list[dict]:
    if instance == "all":
        return manifest
    entries = [e for e in manifest if e["id"] == instance]
    if not entries:
        raise ValueError(f"no frozen instance with id {instance!r}")
    return entries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the reference oracle battery and merge results into "
                    "reference_ceilings.json.")
    ap.add_argument("--instance", default="all",
                     help="a frozen instance id (from the manifest), or 'all'")
    ap.add_argument("--only", default="skeleton,table,perfect",
                     help="comma-separated oracles to run: "
                          "skeleton,table,perfect,weights,a0prime")
    ap.add_argument("--seen-frac", type=float, nargs="+", default=[0.02, 0.05, 0.10],
                     help="seen fractions of the table, used by weights/a0prime")
    ap.add_argument("--out", default="docs/benchmark/reference_ceilings.json",
                     help="output reference_ceilings.json path")
    ap.add_argument("--frozen", default="docs/benchmark/frozen_instances.json",
                     help="frozen instance manifest path")
    args = ap.parse_args(argv)

    oracles = _parse_only(args.only)
    manifest = load_frozen(args.frozen)
    try:
        entries = _select_instances(manifest, args.instance)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for entry in entries:
        instance_id = entry["id"]
        inst = generate(entry["seed"], PRESETS[entry["preset"]])
        items = paired_subset(inst, 500)

        cheap_requested = [o for o in CHEAP_ORACLES if o in oracles]
        if cheap_requested:
            cheap = run_cheap(inst, items)
            for name in cheap_requested:
                merge_reference(args.out, instance_id, name, cheap[name])
                print(f"{instance_id}: {name} overall={cheap[name]['overall']:.3f}")

        if "weights" in oracles:
            # [GPU] only entered when explicitly requested via --only.
            by_frac = {}
            for frac in args.seen_frac:
                model, tok = train_student(inst, frac)
                by_frac[str(frac)] = score_ceiling(inst, model, tok, items)
            merge_reference(args.out, instance_id, "weights", by_frac)
            print(f"{instance_id}: weights done for seen_frac={args.seen_frac}")

        if "a0prime" in oracles:
            # [API] only entered when explicitly requested via --only.
            answer_fn = _frontier_answer_fn()
            by_frac = {}
            for frac in args.seen_frac:
                entries_target = max(1, round(frac * inst.cfg.n_values))
                facts, _revealed = buy_evidence(inst, entries_target)
                evidence = _evidence_text(facts)
                by_frac[str(frac)] = run_a0prime(inst, items, evidence, answer_fn)
            merge_reference(args.out, instance_id, "a0prime", by_frac)
            print(f"{instance_id}: a0prime done for seen_frac={args.seen_frac}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
