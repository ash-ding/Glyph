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
from ..data.instance import GlyphInstance
from ..sandbox import run_solver
from ..seal import SealedArtifact, answer_with
from ..train.sft import Example, HParams
from .schema import Container, Role

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
    # Everything a production tool made, by id. `set_context`, `write_code` and
    # `train` each register one, which is what lets `evaluate` and `inspect`
    # take any of them instead of only an adapter.
    artifacts: dict[str, SealedArtifact] = field(default_factory=dict)
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

    def _register(self, entry: str, **kw) -> str:
        """Record what a production tool just made, so it can be evaluated.

        One artifact is exactly one production call -- a prompt, a program, or
        a checkpoint -- rather than a running snapshot of everything the box
        holds. That keeps `evaluate(artifact_id)` unambiguous about which thing
        it is measuring.
        """
        aid = self._id("art")
        self.artifacts[aid] = SealedArtifact(
            arm=self.arm, entry=entry, base_model=self.base_model, **kw)
        return aid

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
    def _t_query(self, exprs: list[str], why: str) -> dict:
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

    def _t_build_dataset(self, source: str, n: int) -> dict:
        """Repeat purchased rows up to `n`.

        `n` is an epoch count wearing a costume: the agent cannot manufacture
        labels it has not bought, so this repeats rather than invents.
        Curriculum control -- filtering, emphasis by operator or by tail
        coverage -- is deliberately absent until it is implemented. The
        previous version declared `emphasis`, `description` and
        `include_reasoning` and read none of them, so the agent spent every
        call operating a control that was not connected.
        """
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
                        source=source, n=n, size=len(examples))
        return {"dataset_id": ds_id, "size": len(examples),
                "drawn_from": len(rows), "remaining_h100s": round(self.ledger.remaining, 3)}

    def _t_train(self, dataset_id: str, epochs: int, lr: float) -> dict:
        if self.role is None:
            return {"error": "declare_target must come before train"}
        ds = self.datasets.get(dataset_id)
        if not ds:
            return {"error": f"no dataset {dataset_id!r}"}

        from ..train.sft import train as run_train
        ck_id = self._id("ck")
        out = self.work_dir / ck_id
        # Always a full fine-tune. The published weights-arm ceiling was
        # measured that way, so an arm restricted to LoRA would have been
        # compared against a line it could not reach. LoRA is deferred, not
        # rejected -- see docs/tools.md.
        hp = HParams(full_finetune=True, epochs=epochs, lr=lr)
        rec = run_train(ds, hp, base_model=self.base_model, out_dir=out,
                        ledger=self.ledger, log_every=0)
        self.checkpoints[ck_id] = str(out)
        aid = self._register("model", adapter_path=str(out))
        self.trace.emit("train", arm=self.arm, artifact_id=aid,
                        role=self.role.value, record=rec)
        return {"artifact_id": aid, "examples": rec["examples"],
                "final_loss": round(rec["final_loss"], 4),
                "remaining_h100s": round(self.ledger.remaining, 3)}

    def _t_evaluate(self, artifact_id: str, n: int) -> dict:
        """Score any artifact on the dev split.

        Routed through `seal.answer_with`, which is the same path the test
        phase uses. Self-assessment is not a property of a container -- someone
        writing a prompt can try it, someone writing code can run it, someone
        training a model can hold out a validation set -- so issuing it to one
        arm only would have measured the harness rather than the container.
        """
        art = self.artifacts.get(artifact_id)
        if art is None:
            return {"error": f"no artifact {artifact_id!r}"}
        _, dev = self._split()
        if not dev:
            return {"error": "no dev items yet: query more first"}
        dev = dev[:n]
        got = answer_with(art, self.base_model, self.ledger, [e for e, _ in dev])
        hits = sum(g.strip() == a.strip() for g, (_, a) in zip(got, dev))
        return {"dev_accuracy": round(hits / len(dev), 4), "n": len(dev),
                "remaining_h100s": round(self.ledger.remaining, 3)}

    def _t_inspect(self, artifact_id: str, k: int) -> dict:
        """Look at items an artifact got wrong.

        Evaluates `k` items rather than the whole dev split and truncating the
        output, which is what the previous version did.
        """
        art = self.artifacts.get(artifact_id)
        if art is None:
            return {"error": f"no artifact {artifact_id!r}"}
        _, dev = self._split()
        if not dev:
            return {"error": "no dev items yet: query more first"}
        dev = dev[:max(k, 1)]
        got = answer_with(art, self.base_model, self.ledger, [e for e, _ in dev])
        bad = [{"expr": e, "want": a, "got": g}
               for g, (e, a) in zip(got, dev) if g.strip() != a.strip()]
        return {"n_inspected": len(dev), "n_wrong": len(bad), "examples": bad}

    # -- code path ------------------------------------------------------
    def _t_write_code(self, src: str) -> dict:
        """Store a solver. Checking it is `evaluate`'s job, not this one's."""
        self.program = src
        aid = self._register("program", program=src)
        return {"artifact_id": aid, "bytes": len(src.encode())}

    # -- context path ---------------------------------------------------
    def _t_set_context(self, text: str) -> dict:
        self.context = text
        aid = self._register("model", context=text)
        return {"artifact_id": aid, "chars": len(text)}

    # -- sealing --------------------------------------------------------
    def _t_seal(self, artifact_id: str, summary: str,
                forced: bool = False) -> dict:
        """Freeze one artifact. Nothing is reachable afterwards.

        Takes an artifact rather than an entry plus a checkpoint id, so the
        thing sealed is exactly the thing that was evaluated -- the agent can
        no longer measure one object and hand over another.

        `forced` is the harness sealing on the agent's behalf, and it must
        always succeed. An agent whose training all failed still has to produce
        a comparable data point: the base student answering with no adapter and
        no prompt is a legitimate, and very bad, artifact. Refusing to seal it
        turns a poor result into a missing one, and the arms most likely to hit
        that are the ones whose preparation costs most -- which is exactly what
        the comparison is measuring.
        """
        art = self.artifacts.get(artifact_id) if artifact_id else None
        if art is None:
            if not forced:
                return {"error": "seal needs an artifact_id from set_context, "
                                 "write_code or train"}
            # Nothing was produced. Seal the bare student rather than nothing.
            art = SealedArtifact(arm=self.arm, entry="model",
                                 base_model=self.base_model)

        self.summary = summary
        self.sealed = SealedArtifact(
            arm=self.arm, entry=art.entry, context=art.context,
            program=art.program, adapter_path=art.adapter_path,
            base_model=self.base_model,
            notes={"role": self.role.value if self.role else None,
                   "summary": summary, "queries": len(self.queried),
                   "artifact_id": artifact_id or None},
        )
        empty = not (art.context or art.program or art.adapter_path)
        self.trace.emit("seal", arm=self.arm, entry=art.entry, forced=forced,
                        empty=empty, artifact_id=artifact_id or None,
                        role=self.role.value if self.role else None,
                        queries=len(self.queried),
                        ledger=self.ledger.summary())
        return {"sealed": True, "digest": self.sealed.digest()}
