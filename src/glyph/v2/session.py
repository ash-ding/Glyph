'''Session state object and the tool-availability matrix for protocol v2.

Session is the shared per-run state object the whole harness (and every MCP
tool handler) reads and mutates: the instance, ledger, caps, counters, phase,
arm. tool_available_for is the single source of truth for which tool is
available in which (arm, phase).
'''

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glyph.v2.ledger import Ledger

TOOL_MATRIX = {
    'query',
    'submit',
    'check_answers',
    'finish_practice',
    'build_dataset',
    'train',
    'student_infer',
    'final_answer',
}

# The availability matrix (spec Sec 6.1), encoded as tool -> set of allowed
# (arm, phase) tuples.
_MATRIX = {
    'query': {('train', 'practice'), ('no_train', 'practice')},
    'submit': {('train', 'practice'), ('no_train', 'practice')},
    'check_answers': {
        ('train', 'practice'), ('no_train', 'practice'),
        ('train', 'final'), ('no_train', 'final'),
    },
    'finish_practice': {('train', 'practice'), ('no_train', 'practice')},
    'build_dataset': {('train', 'practice')},
    'train': {('train', 'practice')},
    'student_infer': {('train', 'practice'), ('train', 'final')},
    'final_answer': {('train', 'final'), ('no_train', 'final')},
}


def tool_available_for(arm: str, phase: str, name: str) -> bool:
    '''Pure function encoding the tool-availability matrix.'''
    return (arm, phase) in _MATRIX.get(name, set())


@dataclass
class Session:
    '''Shared per-run state for the protocol v2 harness.'''

    inst: Any            # glyph.data GlyphInstance
    ledger: 'Ledger'      # glyph.v2.ledger.Ledger
    run_dir: Path
    arm: str              # 'train' | 'no_train'
    policy: str = 'strict'       # query_ood_policy
    q_cap: int = 1000
    submit_cap: int = 20
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

    def q_remaining(self) -> int:
        return max(0, self.q_cap - self.q_used)

    def submissions_remaining(self) -> int:
        return max(0, self.submit_cap - self.submissions)

    def turns_remaining(self) -> int:
        cap = self.tp if self.phase == 'practice' else self.tf
        return max(0, cap - self.turns)

    def tool_available(self, name: str) -> bool:
        return tool_available_for(self.arm, self.phase, name)

    def switch_to_final(self) -> None:
        self.phase = 'final'
        self.turns = 0
        self.should_switch = False

    def note(self, **kw) -> None:
        self.events.append(dict(kw))
