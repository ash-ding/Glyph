'''Session state object and the tool-availability matrix for protocol v2.

Session is the shared per-run state object the whole harness (and every MCP
tool handler) reads and mutates: the instance, ledger, caps, counters, phase,
arm. tool_available_for is the single source of truth for which tool is
available in which (arm, phase).
'''

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from glyph.v2.ledger import Ledger

TOOL_MATRIX = {
    'query',
    'submit_validation_answer',
    'check_answers',
    'finish_practice',
    'build_dataset',
    'train_model',
    'infer_model',
    'submit_final_answer',
}

# The availability matrix (spec Sec 6.1), encoded as tool -> set of allowed
# (arm, phase) tuples.
_MATRIX = {
    'query': {('train', 'practice'), ('no_train', 'practice')},
    'submit_validation_answer': {('train', 'practice'), ('no_train', 'practice')},
    'check_answers': {
        ('train', 'practice'), ('no_train', 'practice'),
        ('train', 'final'), ('no_train', 'final'),
    },
    'finish_practice': {('train', 'practice'), ('no_train', 'practice')},
    'build_dataset': {('train', 'practice')},
    'train_model': {('train', 'practice')},
    'infer_model': {('train', 'practice'), ('train', 'final')},
    'submit_final_answer': {('train', 'final'), ('no_train', 'final')},
}


# The free protocol (spec 2026-10-15-free-protocol-design.md, section 4): four
# tools, the same in every arm; availability depends on the phase only.
# check_answers is not among them: with no submission cap, submit_validation_answer
# and submit_final_answer report the same violations, and an illegal final file is
# not committed, so a separate legality check added nothing (2026-10-19).
_FREE_MATRIX = {
    'query': {'practice'},
    'submit_validation_answer': {'practice'},
    'finish_practice': {'practice'},
    'submit_final_answer': {'final'},
}


def tool_available_for(arm: str, phase: str, name: str, protocol: str = 'v2') -> bool:
    '''Pure function encoding the tool-availability matrix.'''
    if protocol == 'free':
        return phase in _FREE_MATRIX.get(name, set())
    return (arm, phase) in _MATRIX.get(name, set())


@dataclass
class Session:
    '''Shared per-run state for the protocol v2 harness.'''

    inst: Any            # glyph.data GlyphInstance
    ledger: 'Ledger'      # glyph.v2.ledger.Ledger
    run_dir: Path
    arm: str              # 'train' | 'no_train'
    policy: str = 'strict'       # query_ood_policy
    q_cap: int | None = 1000      # None = no query cap
    submit_cap: int | None = 20   # None = no submission cap
    tp: int = 100                # practice turn cap
    tf: int = 30                 # final turn cap
    usd_line: float = 300.0
    # mutable counters (defaults):
    q_used: int = 0
    submissions: int = 0
    turns: int = 0
    phase: str = 'practice'      # 'practice' | 'final'
    should_switch: bool = False  # set by the submit/finish handlers; harness reads it
    events: list = field(default_factory=list)  # in-memory event log (trace wiring is Task 12)
    protocol: str = 'v2'         # 'v2' | 'free'
    deadline: float | None = None    # free protocol: end of the current phase, on `clock`
    clock: Callable[[], float] = time.time
    # free protocol: refuse queries that look up a binary entry a validation item needs
    reserve_val_binary: bool = False
    refused_reserved: int = 0
    # free protocol option: u*(value) and b*(value, value) come back as "malformed",
    # uncharged; cells are then reachable only through structural expressions such
    # as s0(u0, [..]) or s1([x, y], b0)
    refuse_bare_atomics: bool = False
    refused_bare: int = 0

    def q_remaining(self) -> int | None:
        if self.q_cap is None:
            return None                 # None = unlimited
        return max(0, self.q_cap - self.q_used)

    def submissions_remaining(self) -> int | None:
        if self.submit_cap is None:
            return None
        return max(0, self.submit_cap - self.submissions)

    def seconds_remaining(self) -> int | None:
        if self.deadline is None:
            return None
        return max(0, int(self.deadline - self.clock()))

    def turns_remaining(self) -> int:
        cap = self.tp if self.phase == 'practice' else self.tf
        return max(0, cap - self.turns)

    def tool_available(self, name: str) -> bool:
        return tool_available_for(self.arm, self.phase, name, self.protocol)

    def switch_to_final(self) -> None:
        self.phase = 'final'
        self.turns = 0
        self.should_switch = False

    def note(self, **kw) -> None:
        rec = dict(kw)
        if self.protocol == 'free':
            rec.setdefault('ts', self.clock())
        self.events.append(rec)
