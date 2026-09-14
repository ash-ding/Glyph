"""System prompt, phase openers, and task README for protocol v2.

The system prompt states the protocol, the caps, and the arm's tools -- and
nothing about what the right strategy is. The two arms get the SAME text
except one paragraph describing the student tools (train arm only).

This neutrality principle ensures that we are measuring the agent's judgment,
not hints from the prompt about which approach to take.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from glyph.v2.session import Session
    from glyph.v2.workspace import Paths


# Template for system prompt, without the student paragraph
_SYSTEM_COMMON = """You are solving a task in a small formal language whose syntax is public but whose semantics are private.

**The Protocol**

Your run has two phases:

1. **Practice Phase**: You have a metered query oracle and a visible validation set. You may make up to Q queries to learn about the language, and submit answers up to 20 times to check yourself against the validation set. Each submission gives you an aggregate score back. You have a practice turn budget of T_p turns.

2. **Final Phase**: The oracle is gone. You receive a held-out test set with no answers and no further oracle. You have a final turn budget of T_f turns. You submit your answers once at the end.

**Your Capabilities and Constraints**

You have a query budget of Q queries total. Each query tells you the value of an expression in the exchange for consuming one query.

During practice, you may submit to the validation set up to 20 times. Each submission is JSONL of {{"id", "answer"}} rows covering every validation id exactly once.

An answer is either a value written as v_x_y_z or a non-empty list [].

You must not exceed your turn budgets: T_p turns for practice, T_f turns for final phase.

**Available Tools**

This run provides these tools:
- File and Bash tools for reading and writing files
- query: spend a query to evaluate an expression
- submit: submit validation answers during practice
- check_answers: check if your answers are correct (reveals any errors found, returns aggregate score)
- finish_practice: signal that you are done with practice
- final_answer: submit your final test answers and complete the run"""


# Student paragraph: only for the train arm
_STUDENT_PARAGRAPH = """
**Building a Student Model (train arm only)**

During practice, you may also build a student model:
- build_dataset: collect training data from queries you make
- train: fine-tune a student model on your collected data
- student_infer: use your trained student model to generate answers

These tools allow you to build up a small model that learns from the private semantics you discover. Training and inference consume turns but not queries."""


# Build the system prompts by combining common template with/without student paragraph
def system_prompt(arm: str) -> str:
    """Return the system prompt for the given arm.
    
    Args:
        arm: 'train' or 'no_train'
    
    Returns:
        System prompt text, identical except train arm includes student paragraph.
    """
    if arm not in ("train", "no_train"):
        raise ValueError(f"arm must be 'train' or 'no_train', got {arm!r}")
    
    if arm == "train":
        return _SYSTEM_COMMON + _STUDENT_PARAGRAPH
    else:
        return _SYSTEM_COMMON


def practice_opener(paths: Paths) -> str:
    """Return the opening message for practice phase.
    
    Args:
        paths: Workspace Paths object with task file locations
    
    Returns:
        User message pointing to workspace files and instructions.
    """
    return f"""You are ready to begin the practice phase.

The workspace files are at:
- {paths.readme}: Protocol and rules
- {paths.syntax}: Expression grammar
- {paths.demos}: Worked examples
- {paths.validation}: Validation items to submit answers for
- {paths.queries}: Growing log of your queries

Begin solving. Query when you want to learn something. Submit answers to check yourself. Call finish_practice when you are done with the practice phase."""


def final_opener(paths: Paths) -> str:
    """Return the opening message for final phase.
    
    Args:
        paths: Workspace Paths object with task file locations
    
    Returns:
        User message announcing the final phase.
    """
    return f"""The practice phase is over.

The held-out test set is at {paths.final_dir}/test.jsonl. The oracle and validation are now closed. No further queries or submissions are possible.

Produce your answers and call final_answer() with the path to your submission file."""


def task_readme(session: Session) -> str:
    """Return the in-workspace README with concrete session parameters.
    
    Args:
        session: Session object with q_cap, submit_cap, tp, tf parameters
    
    Returns:
        Factual README documenting the task rules and concrete parameters.
    """
    q_cap = session.q_cap
    submit_cap = session.submit_cap
    tp = session.tp
    tf = session.tf
    
    return f"""# Glyph Task Workspace

## Rules

Your run is organized in two phases:

**Practice Phase** (up to {tp} turns):
- You have a query budget of {q_cap} queries total
- You may submit answers to the validation set up to {submit_cap} times
- Each submission gives you an aggregate score
- Call finish_practice when done

**Final Phase** (up to {tf} turns):
- No oracle and no validation submissions
- A held-out test set with no answers provided
- Submit your final answers once with final_answer()

## File Formats

**validation.jsonl**: Practice items (one per line)
- Format: {{"id": "val_XXXXX", "expr": "..."}}
- No answers provided (that is what you are solving for)

**test.jsonl**: Final test items (appears during final phase)
- Format: {{"id": "test_XXXXX", "expr": "..."}}
- No answers, no oracle

**Your submission**: JSONL of {{"id", "answer"}} rows
- Must cover every item id exactly once
- Answer is either v_x_y_z or a non-empty list
- Examples: "answer": "v_1_2_3" or "answer": [1, 2]

**queries.jsonl**: Growing log of queries you have made
- Recorded for your reference and final analysis
- Automatically appended as you query

## Expression Syntax

See syntax.md for the full grammar.

## Examples

See demos.jsonl for worked examples of expressions and their answers.

## Workflow

1. During practice: read syntax.md and demos.jsonl
2. Query to learn about the language semantics
3. Submit validation answers and check your progress
4. Call finish_practice when ready
5. Solve the final test set
6. Call final_answer() with your answers
"""

