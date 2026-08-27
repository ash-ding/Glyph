"""A6 -- capability into weights.

The agent buys what it can, declares what it is teaching, builds a training
set from what it bought, and fine-tunes the student.  At test time the model
answers with no prompt and no program: the cost was paid once, up front.

`declare_target` is mandatory here.  Without it "what did the agent choose"
would be an inference from prose rather than a field in the trace, and Fig. 4
would be our reading of its behaviour instead of a measurement of it.
"""

from __future__ import annotations

from ..agent import Container, run_agent
from ..seal import ScoreReport
from .base import RunConfig, finish, prepare


def run(rc: RunConfig) -> ScoreReport:
    p = prepare(rc)
    box = run_agent(p.inst, p.ledger, p.trace, p.cache, arm="a6",
                    allowed={Container.WEIGHTS}, work_dir=p.work_dir,
                    base_model=rc.base_model, model=rc.teacher,
                    max_turns=rc.max_turns)

    from ..train.infer import Student
    student = Student(rc.base_model, adapter_path=box.sealed.adapter_path)

    def answer(exprs):
        return student.answer(exprs, ledger=p.ledger).answers

    return finish(p, rc, box.sealed, answer)
