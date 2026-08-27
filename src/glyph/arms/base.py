"""What every arm shares, and nothing about what makes them differ.

Each arm is its own runner with its own module.  They share setup and
scoring, because those must be identical for the numbers to be comparable --
but no arm is a branch inside another arm's code.  A single function with
`if arm == ...` would mean editing one arm can silently change another, and
an ablation you cannot trust is worse than one you did not run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..budget import Ledger
from ..config import PRESETS, GlyphConfig
from ..instance import GlyphInstance, generate
from ..seal import ScoreReport, evaluate
from ..trace import ResponseCache, TraceWriter
from ..vertex import TEACHER


@dataclass
class RunConfig:
    """One point in the (arm, budget, instance) grid.

    The two seeds are separate on purpose.  `instance_seed` picks the hidden
    interpreter and therefore the task's difficulty; `run_seed` picks the
    agent's sampling and the training shuffle.  Collapsing them would put
    "some interpreters are harder" and "this run got lucky" into the same
    error bar, and the crossover's confidence interval is what decides
    whether B* can be located at all.
    """
    arm: str
    preset: str = "pi_mid"
    instance_seed: int = 1001
    run_seed: int = 0
    budget_h100s: float = 2.0
    base_model: str = "Qwen/Qwen3-1.7B"
    teacher: str = TEACHER
    out_root: Path = Path("runs")
    n_test: int | None = None          # trim the test set for pilots
    max_turns: int = 40
    cfg_overrides: dict = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return (f"{self.arm}_{self.preset}_i{self.instance_seed}"
                f"_r{self.run_seed}_b{self.budget_h100s:g}")

    def config(self) -> GlyphConfig:
        cfg = PRESETS[self.preset]
        if self.n_test:
            cfg = cfg.scaled(self.n_test)
        return cfg.with_(**self.cfg_overrides) if self.cfg_overrides else cfg


@dataclass
class Prepared:
    inst: GlyphInstance
    ledger: Ledger
    trace: TraceWriter
    cache: ResponseCache
    work_dir: Path


def prepare(rc: RunConfig) -> Prepared:
    work = Path(rc.out_root) / rc.tag
    work.mkdir(parents=True, exist_ok=True)
    return Prepared(
        inst=generate(rc.instance_seed, rc.config()),
        ledger=Ledger(total_h100s=rc.budget_h100s),
        trace=TraceWriter(work / "trace.jsonl"),
        # The cache is shared across runs by design: an identical prepare
        # phase should cost nothing the second time, which is what makes the
        # oracle sweep affordable.
        cache=ResponseCache(Path(rc.out_root) / "_cache"),
        work_dir=work,
    )


def finish(p: Prepared, rc: RunConfig, artifact, answer_fn) -> ScoreReport:
    """Score the sealed artifact and write the report next to the trace."""
    report = evaluate(p.inst, artifact, p.ledger, answer_fn=answer_fn)
    (p.work_dir / "report.json").write_text(report.to_json())
    p.trace.emit("report", arm=rc.arm, overall=report.overall,
                 by_split=report.by_split, tail=report.tail)
    p.trace.close()
    return report
