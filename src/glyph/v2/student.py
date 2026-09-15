"""StudentPool: the train-arm's host-side wrapper around dataset build,
full fine-tune, and inference.

Two things this is careful about, both load-bearing:

  - **Lazy heavy imports.** The default backend wraps the real GPU-backed
    ``glyph.train.sft.train`` and ``glyph.train.infer.Student``. Those modules
    import torch / vLLM at module scope, so importing them eagerly here would
    make merely `import glyph.v2.student` (and every unit test that injects a
    fake backend) drag in torch. All such imports stay inside the default
    backend's methods, never at module level. `Example` and `HParams` are
    likewise NOT imported from `glyph.train.sft` for this reason -- local,
    structurally identical stand-ins are defined here instead; `sft.train`'s
    dataset packer and training loop only ever access their declared
    attributes (plus `HParams.resolved_lr()`), so the real backend accepts
    these objects by duck typing without the fake-backend path ever
    importing torch.

  - **GPU caps enforced before dispatch.** `train()` refuses on the
    cumulative-budget pre-check without ever calling the backend -- a
    refused call must not cost anything, and callers must be able to tell
    "we didn't even try" apart from "the backend ran and used 0 seconds".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from glyph.data import parse, render, check
from glyph.v2.answers import parse_answer


@dataclass
class Example:
    """Local stand-in for glyph.train.sft.Example (see module docstring)."""
    prompt: str
    answer: str
    weight: float = 1.0


@dataclass
class HParams:
    """Local stand-in for glyph.train.sft.HParams, same shape/defaults.

    Kept local for the same reason as Example: `glyph.train.sft` imports
    torch at module scope, and this dataclass must be constructible (in
    train(), on every call including fake-backend tests) without pulling
    that in. The real backend's sft.train() only ever accesses these
    attributes and .resolved_lr(), so a duck-typed instance works there too.
    """
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    full_finetune: bool = False
    lr: float | None = None
    epochs: int = 2
    batch_size: int = 32
    max_len: int = 256
    warmup_frac: float = 0.03
    seed: int = 0

    def resolved_lr(self) -> float:
        if self.lr is not None:
            return self.lr
        return 1e-5 if self.full_finetune else 1e-4


def _canon(expr_src: str, cfg) -> str | None:
    """Parse+render an expression to its canonical source, or None if it
    doesn't parse under `cfg`. Mirrors GlyphInstance._canon."""
    try:
        return render(parse(expr_src, cfg), cfg)
    except Exception:
        return None


def _default_backend():
    """The real GPU backend: glyph.train.sft.train + glyph.train.infer.Student.

    Constructed lazily (only when StudentPool is built with backend=None and
    actually used), so its torch/vLLM imports never happen on the fake-backend
    unit-test path.
    """

    class _RealStudent:
        def __init__(self, base_model, adapter_path, prefix):
            from glyph.train.infer import Student as _Student
            self._student = _Student(base_model, adapter_path=adapter_path,
                                      context=prefix)
            self.last_truncated = 0

        def answer(self, exprs: list[str]) -> list[str]:
            gen = self._student.answer(exprs)
            self.last_truncated = getattr(gen, "truncated", 0)
            return list(gen.answers)

        def close(self) -> None:
            self._student.close()

    class _RealBackend:
        def train_fn(self, examples, hp, base_model, out_dir, ledger):
            import time
            from glyph.train import sft

            t0 = time.monotonic()
            # `ledger` here is a glyph.v2.ledger.Ledger, which does not
            # implement sft.train's expected gpu_timer() context-manager
            # protocol -- pass None and measure wall-clock instead, rather
            # than let an unrelated interface mismatch crash training.
            rec = sft.train(examples, hp, base_model=base_model,
                             out_dir=out_dir, ledger=None)
            rec = dict(rec)
            rec.setdefault("gpu_seconds", time.monotonic() - t0)
            return rec

        def make_student(self, base_model, adapter_path, prefix):
            return _RealStudent(base_model, adapter_path, prefix)

    return _RealBackend()


class StudentPool:
    """Owns dataset construction, full fine-tuning, and inference for the
    train arm's student model, under a cumulative GPU-seconds budget."""

    def __init__(self, base_model, ledger, *, work_dir,
                 queries_path=None,
                 gpu_cap_per_call_s: float = 1800.0,
                 gpu_cap_total_s: float = 7200.0,
                 backend=None):
        self.base_model = base_model
        self.ledger = ledger
        self.work_dir = Path(work_dir)
        self.queries_path = Path(queries_path) if queries_path else None
        self.gpu_cap_per_call_s = gpu_cap_per_call_s
        self.gpu_cap_total_s = gpu_cap_total_s
        self.backend = backend if backend is not None else _default_backend()

        self.gpu_used_s = 0.0
        self.datasets: dict[str, list[Example]] = {}
        self.checkpoints: dict[str, Path] = {}
        self._ds_n = 0
        self._ck_n = 0

    # -- build_dataset ----------------------------------------------------
    def _purchased_set(self, cfg) -> set[str]:
        if self.queries_path is None or not self.queries_path.exists():
            return set()
        out = set()
        for line in self.queries_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            c = _canon(row["expr"], cfg)
            if c is not None:
                out.add(c)
        return out

    def build_dataset(self, path, inst) -> dict:
        cfg = inst.cfg
        purchased = self._purchased_set(cfg)
        demo = {c for c in (_canon(e, cfg) for e, _ in inst.demos) if c is not None}
        validation = {c for c in (_canon(t.expr_src, cfg) for t in inst.val)
                      if c is not None}

        examples: list[Example] = []
        counts = {"purchased": 0, "demo": 0, "validation": 0, "other": 0}
        dropped = 0

        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            e, a = row["expr"], row["answer"]

            try:
                expr = parse(e, cfg)
                check(expr, cfg)
            except Exception:
                dropped += 1
                continue
            try:
                parse_answer(a, cfg)
            except Exception:
                dropped += 1
                continue

            examples.append(Example(prompt=f"{e} =", answer=f" {a}"))

            canon = _canon(e, cfg)
            if canon is not None and canon in purchased:
                counts["purchased"] += 1
            elif canon is not None and canon in demo:
                counts["demo"] += 1
            elif canon is not None and canon in validation:
                counts["validation"] += 1
            else:
                counts["other"] += 1

        self._ds_n += 1
        dataset_id = f"ds{self._ds_n}"
        self.datasets[dataset_id] = examples

        return {
            "dataset_id": dataset_id,
            "size": len(examples),
            "dropped": dropped,
            "provenance": counts,
        }

    # -- train --------------------------------------------------------
    def train(self, dataset_id, epochs, lr) -> dict:
        if self.gpu_used_s >= self.gpu_cap_total_s:
            return {
                "stopped_at_cap": True,
                "error": "cumulative GPU budget exhausted",
                "gpu_seconds": 0.0,
                "gpu_seconds_remaining": 0.0,
                "checkpoint_id": None,
                "final_loss": None,
            }

        examples = self.datasets[dataset_id]
        hp = HParams(full_finetune=True, epochs=epochs, lr=lr)

        self._ck_n += 1
        checkpoint_id = f"ck{self._ck_n}"
        out_dir = self.work_dir / checkpoint_id

        rec = self.backend.train_fn(examples, hp, base_model=self.base_model,
                                     out_dir=out_dir, ledger=self.ledger)

        gpu_seconds = rec.get("gpu_seconds")
        if gpu_seconds is None:
            gpu_seconds = 0.0
        self.gpu_used_s += gpu_seconds
        self.ledger.add_gpu_seconds("train", gpu_seconds)
        self.checkpoints[checkpoint_id] = out_dir

        stopped_at_cap = (self.gpu_used_s >= self.gpu_cap_total_s or
                           gpu_seconds > self.gpu_cap_per_call_s)

        return {
            "checkpoint_id": checkpoint_id,
            "final_loss": rec.get("final_loss"),
            "gpu_seconds": gpu_seconds,
            "gpu_seconds_remaining": max(0.0, self.gpu_cap_total_s - self.gpu_used_s),
            "stopped_at_cap": stopped_at_cap,
        }

    # -- infer --------------------------------------------------------
    def _resolve_checkpoint(self, ck_dir):
        """Map a trained checkpoint dir to (base_model, adapter_path) for infer.

        A full fine-tune saved the whole model, so it is served AS the base
        model (adapter_path=None); only a LoRA checkpoint is loaded through the
        adapter path on top of the original base. Loading a full-finetune dir
        as a LoRA adapter raises LoRAAdapterNotFoundError -- exactly the
        train/infer mismatch this resolves. sft.train records which in
        train_record.json ("adapter": "full" | "lora-rN"); an absent or
        unreadable record is treated as full, since train() only ever does
        full fine-tunes today (LoRA is deferred).
        """
        ck_dir = Path(ck_dir)
        adapter = "full"
        rec_path = ck_dir / "train_record.json"
        if rec_path.exists():
            try:
                adapter = json.loads(rec_path.read_text()).get("adapter", "full")
            except Exception:
                adapter = "full"
        if adapter == "full":
            return str(ck_dir), None
        return self.base_model, str(ck_dir)

    def infer(self, checkpoint, input_path, output_path, prefix_path=None) -> dict:
        import time

        if checkpoint == "base":
            base_model, adapter_path = self.base_model, None
        else:
            base_model, adapter_path = self._resolve_checkpoint(self.checkpoints[checkpoint])

        prefix = None
        if prefix_path is not None:
            prefix = Path(prefix_path).read_text()

        rows = []
        for line in Path(input_path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

        t0 = time.monotonic()
        student = self.backend.make_student(base_model, adapter_path, prefix)
        try:
            answers = student.answer([r["expr"] for r in rows])
            truncated = getattr(student, "last_truncated", 0)
        finally:
            student.close()
        gpu_seconds = time.monotonic() - t0

        out_lines = [json.dumps({"id": r["id"], "answer": ans})
                     for r, ans in zip(rows, answers)]
        Path(output_path).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

        self.ledger.add_gpu_seconds("infer", gpu_seconds)

        return {
            "rows": len(rows),
            "truncated": truncated,
            "gpu_seconds": gpu_seconds,
        }
