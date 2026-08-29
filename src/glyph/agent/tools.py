"""The API the agent acts through, and the only place it can spend.

Every call is metered, traced, and refused once the budget is gone.  The arm
decides which of these exist at all: A2 never sees `train`, A6 never sees
`set_context`.  Enforcing the arm boundary by withholding tools rather than
by instruction is what stops the boundary from depending on compliance.

One thing here is a design decision rather than plumbing.  `evaluate` needs
labelled items, and labelled items are exactly what the agent is paying for.
So dev is carved out of what it has already queried -- a slice held back from
training -- rather than handed over free.  Giving it a free labelled dev set
would quietly refund part of the query budget and make the 10^3 -> 10^4
asymmetry, the whole reason the weights arm exists, smaller than it says it
is.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..budget import BudgetExhausted, Ledger
from ..instance import GlyphInstance
from ..sandbox import run_solver
from ..seal import SealedArtifact
from ..train.sft import Example, HParams
from .schema import Container, DataSpec, Role

DEV_FRACTION = 0.15


@dataclass
class ToolBox:
    inst: GlyphInstance
    ledger: Ledger
    trace: object                       # TraceWriter
    allowed: set
    work_dir: Path
    base_model: str = "Qwen/Qwen3-1.7B"
    arm: str = "unknown"

    queried: list[tuple[str, str]] = field(default_factory=list)
    datasets: dict[str, list[Example]] = field(default_factory=dict)
    checkpoints: dict[str, str] = field(default_factory=dict)
    role: Role | None = None
    context: str | None = None
    program: str | None = None
    summary: str = ""
    sealed: SealedArtifact | None = None
    _n: int = 0

    # -- dev split, carved from what was bought -------------------------
    def _split(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        k = max(1, int(len(self.queried) * DEV_FRACTION)) if self.queried else 0
        return self.queried[:-k] if k else self.queried, self.queried[-k:] if k else []

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    # -- dispatch -------------------------------------------------------
    def dispatch(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"error": f"no such tool {name!r}"}
        t0 = time.time()
        try:
            out = fn(**args)
        except BudgetExhausted:
            raise
        except Exception as exc:                       # a tool error is data
            out = {"error": f"{type(exc).__name__}: {exc}"}
        self.trace.emit("tool", arm=self.arm, tool=name, args=args, result=out,
                        seconds=round(time.time() - t0, 3),
                        spent_h100s=round(self.ledger.spent_h100s, 4))
        return out

    # -- the oracle -----------------------------------------------------
    def _t_query_oracle(self, exprs: list[str], why: str) -> dict:
        self.ledger.charge("oracle_query", len(exprs), why=why[:200])
        rows, bad = [], 0
        for e in exprs:
            try:
                ans = self.inst.query(e)
            except Exception as exc:
                # Syntax errors are billed like any other query: probing the
                # grammar for free would make the syntax spec worth less than
                # it is meant to be.
                ans = f"<error: {type(exc).__name__}>"
                bad += 1
            rows.append({"expr": e, "out": ans})
            if not ans.startswith("<error"):
                self.queried.append((e, ans))
        return {"results": rows, "malformed": bad,
                "queries_so_far": len(self.queried),
                "remaining_h100s": round(self.ledger.remaining, 3)}

    # -- weights path ---------------------------------------------------
    def _t_declare_target(self, role: str, rationale: str) -> dict:
        if self.role is not None:
            return {"error": f"target already declared as {self.role.value}"}
        self.role = Role(role)
        return {"ok": True, "role": role}

    def _t_synthesize_data(self, description: str, n: int, source: str,
                           emphasis: str, include_reasoning: bool) -> dict:
        spec = DataSpec(description, n, source, emphasis, include_reasoning)
        train_rows, _ = self._split()
        if not train_rows:
            return {"error": "nothing to synthesise from: query the oracle first"}

        with self.ledger.gpu_timer("data_synth"):
            rows = list(train_rows)
            if source == "demos":
                rows = [(e, a) for e, a in self.inst.demos]
            elif source == "mixture":
                rows = list(train_rows) + [(e, a) for e, a in self.inst.demos]
            examples = [Example(prompt=f"{e} =", answer=f" {a}")
                        for e, a in rows][:n]
            # Repeat rather than invent: the agent cannot manufacture labels
            # it has not bought, and pretending otherwise would be leakage.
            while len(examples) < n and examples:
                examples.append(examples[len(examples) % len(rows)])

        ds_id = self._id("ds")
        self.datasets[ds_id] = examples
        self.trace.emit("dataspec", arm=self.arm, dataset_id=ds_id,
                        spec=spec.__dict__, size=len(examples))
        return {"dataset_id": ds_id, "size": len(examples),
                "drawn_from": len(rows), "remaining_h100s": round(self.ledger.remaining, 3)}

    def _t_train(self, dataset_id: str, lora_rank: int, epochs: int, lr: float) -> dict:
        if self.role is None:
            return {"error": "declare_target must come before train"}
        ds = self.datasets.get(dataset_id)
        if not ds:
            return {"error": f"no dataset {dataset_id!r}"}

        from ..train.sft import train as run_train
        ck_id = self._id("ck")
        out = self.work_dir / ck_id
        hp = HParams(lora_rank=lora_rank, lora_alpha=lora_rank, epochs=epochs,
                     lr=lr)
        rec = run_train(ds, hp, base_model=self.base_model, out_dir=out,
                        ledger=self.ledger, log_every=0)
        self.checkpoints[ck_id] = str(out)
        self.trace.emit("train", arm=self.arm, checkpoint_id=ck_id,
                        role=self.role.value, record=rec)
        return {"checkpoint_id": ck_id, "examples": rec["examples"],
                "final_loss": round(rec["final_loss"], 4),
                "remaining_h100s": round(self.ledger.remaining, 3)}

    def _t_evaluate(self, checkpoint_id: str, n: int) -> dict:
        path = self.checkpoints.get(checkpoint_id)
        if not path:
            return {"error": f"no checkpoint {checkpoint_id!r}"}
        _, dev = self._split()
        if not dev:
            return {"error": "no dev items yet: query more first"}
        dev = dev[:n]

        from ..train.infer import Student
        student = Student(self.base_model, adapter_path=path)
        got = student.answer([e for e, _ in dev], ledger=self.ledger)
        hits = sum(g.strip() == a.strip() for g, (_, a) in zip(got.answers, dev))
        student.close()
        del student
        return {"dev_accuracy": round(hits / len(dev), 4), "n": len(dev),
                "remaining_h100s": round(self.ledger.remaining, 3)}

    def _t_inspect_failures(self, checkpoint_id: str, k: int) -> dict:
        path = self.checkpoints.get(checkpoint_id)
        if not path:
            return {"error": f"no checkpoint {checkpoint_id!r}"}
        _, dev = self._split()
        from ..train.infer import Student
        student = Student(self.base_model, adapter_path=path)
        got = student.answer([e for e, _ in dev], ledger=self.ledger)
        bad = [{"expr": e, "want": a, "got": g}
               for g, (e, a) in zip(got.answers, dev) if g.strip() != a.strip()]
        student.close()
        del student
        return {"n_inspected": len(dev), "n_wrong": len(bad), "examples": bad[:k]}

    # -- code path ------------------------------------------------------
    def _t_write_code(self, src: str, check_on: int) -> dict:
        self.program = src
        if check_on <= 0:
            return {"stored": True, "bytes": len(src.encode())}
        _, dev = self._split()
        dev = dev[:check_on]
        if not dev:
            return {"stored": True, "note": "no dev items to check against yet"}
        res = run_solver(src, [e for e, _ in dev], ledger=self.ledger)
        if not res.ok:
            return {"stored": True, "solver_error": res.error}
        hits = sum(g.strip() == a.strip() for g, (_, a) in zip(res.answers, dev))
        return {"stored": True, "bytes": len(src.encode()),
                "dev_accuracy": round(hits / len(dev), 4), "n": len(dev),
                "remaining_h100s": round(self.ledger.remaining, 3)}

    # -- context path ---------------------------------------------------
    def _t_set_context(self, text: str) -> dict:
        self.context = text
        return {"stored": True, "chars": len(text)}

    # -- sealing --------------------------------------------------------
    def _t_seal(self, entry: str, checkpoint_id: str, summary: str,
                forced: bool = False) -> dict:
        """Freeze whatever exists.

        `forced` is the harness sealing on the agent's behalf, and it must
        always succeed. An agent whose training all failed still has to
        produce a comparable data point -- the base student answering with
        no adapter and no prompt is a legitimate, and very bad, artifact.
        Refusing to seal it turns a poor result into a missing one, and the
        arms most likely to hit that are the ones whose preparation costs
        most, which is exactly what the comparison is measuring.
        """
        adapter = self.checkpoints.get(checkpoint_id) if checkpoint_id else None
        if not forced:
            if entry == "model" and adapter is None and self.context is None:
                return {"error": "sealing a model entry needs a checkpoint or a context"}
            if entry == "program" and not self.program:
                return {"error": "sealing a program entry needs write_code first"}
        if forced and entry == "program" and not self.program:
            entry = "model"          # nothing to run; fall back to the student
        self.summary = summary
        self.sealed = SealedArtifact(
            arm=self.arm, entry=entry, context=self.context,
            program=self.program if entry == "program" else None,
            adapter_path=adapter, base_model=self.base_model,
            notes={"role": self.role.value if self.role else None,
                   "summary": summary, "queries": len(self.queried)},
        )
        self.trace.emit("seal", arm=self.arm, entry=entry, forced=forced,
                        empty=adapter is None and self.context is None
                        and self.program is None,
                        role=self.role.value if self.role else None,
                        queries=len(self.queried),
                        ledger=self.ledger.summary())
        return {"sealed": True, "digest": self.sealed.digest()}
