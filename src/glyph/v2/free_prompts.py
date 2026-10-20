"""What the agent is told under the free protocol.

Spec: docs/superpowers/specs/2026-10-15-free-protocol-design.md, section 6.
Facts only: the task, the phases and clocks, the tools, the file formats, the
environment, that running processes are stopped at the end of a phase, the
submission rules and two rules. Nothing names a way to solve the task.

The arms differ only in the environment. `train` and `no_train` get one line
describing the container (one GPU, or none). `silent` has a GPU like `train`
but no Environment section at all, so whatever the agent knows about the
machine it has found out itself.
"""

from __future__ import annotations


def _hm(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, m = divmod(seconds // 60, 60)
    if h and m:
        return f"{h} h {m} min"
    if h:
        return f"{h} h"
    return f"{m} min"


ARMS = ("train", "no_train", "silent")
GPU_ARMS = ("train", "silent")


def environment_line(gpu_name: str | None, cpus: int, mem_gb: int, disk_gb: int) -> str:
    gpu = f"one {gpu_name} GPU" if gpu_name else "no GPU"
    return (f"You are root in an Ubuntu 24.04 container with conda and network access. "
            f"This container has {gpu}, {cpus} CPU cores, {mem_gb} GB of memory and "
            f"{disk_gb} GB of disk.")


def environment_for_arm(arm: str, gpu_name: str | None, cpus: int, mem_gb: int,
                        disk_gb: int) -> str | None:
    """The environment line an arm is shown, or None for `silent`, which is shown none."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    return None if arm == "silent" else environment_line(gpu_name, cpus, mem_gb, disk_gb)


RULES = (
    "1. Do not call any hosted language model other than the one you are running as.\n"
    "2. Do not search for or read material about this benchmark (Glyph)."
)


def system_prompt(env_line: str | None, t_p: float, t_f: float) -> str:
    env = f"**Environment**\n\n{env_line}\n\n" if env_line else ""
    return f"""You are solving a task in a small formal language whose syntax is public but whose semantics are private.

**The run**

Your run has two phases.

1. Practice lasts at most {_hm(t_p)}. You have a query tool that evaluates expressions in the language, and a validation set without answers that you can score answer files against. Practice ends when the time is up or when you call finish_practice.
2. Final lasts at most {_hm(t_f)}. A held-out test set without answers is written to task/final/test.jsonl. The query tool and validation scoring are no longer available. You submit one answer file for the test set.

When a phase ends, every process still running in your container is stopped.

**Tools**

- query: evaluate one or more expressions.
- submit_validation_answer: score an answer file against the validation set; returns overall accuracy and accuracy by expression depth.
- finish_practice: end the practice phase.
- submit_final_answer: submit your answers for the test set; a legal file is accepted once and ends the run.

You also have shell, file, web and subagent tools. Every Glyph tool result reports the seconds remaining in the current phase; task/timer.sh prints the same.

{env}**Rules**

{RULES}

The task README is at task/README.md."""


NO_BARE_RULE = (
    " Bare applications of an\n"
    "    atomic operator to value literals, u*(value) and b*(value, value), are\n"
    "    refused too; they are legal expressions and appear in the test set."
)


def task_readme(env_line: str | None, t_p: float, t_f: float, bare_atomics: bool = True) -> str:
    env = f"Environment:\n  {env_line}\n\n" if env_line else ""
    no_bare = "" if bare_atomics else NO_BARE_RULE
    return f"""# Glyph task workspace

Files in this directory:
  syntax.md          the expression grammar you may write
  demos.jsonl        worked examples: {{"expr", "answer"}} per line
  validation.jsonl   {{"id", "expr"}} per line; no answers
  queries.jsonl      every answered query this run, appended automatically
  timer.sh           prints the time remaining in the current phase
  final/             empty until the run switches to the final phase

Phases:
  - Practice lasts at most {_hm(t_p)}. It ends when that time is up or when you
    call finish_practice. During practice you may call query and
    submit_validation_answer as often as you like.
  - When practice ends, every process still running in this container is
    stopped. final/test.jsonl is then written, and query and
    submit_validation_answer are no longer available.
  - The final phase lasts at most {_hm(t_f)}. It ends when you call
    submit_final_answer with a legal file, or when the time is up. When the
    time is up, every running process is stopped and the most recently
    written legal answer file for final/test.jsonl is submitted for you; if
    there is none, the run scores zero.

Tools:
  - query: evaluates expressions. Validation and test items are refused, as
    are expressions containing a held-out operator pair, expressions deeper
    than the demos, and expressions whose evaluation needs a binary-operator
    entry that a validation item needs.{no_bare} Refused expressions are not
    counted.
  - submit_validation_answer: scores an answer file against validation.jsonl
    and returns the overall accuracy and a breakdown by expression depth.
  - finish_practice: ends the practice phase.
  - submit_final_answer: submits your answers for final/test.jsonl. A legal
    file is accepted once and ends the run; an illegal one is rejected with
    the problems found, and you can submit again.
  Every tool result includes the seconds remaining in the current phase.

Answer files:
  JSONL of {{"id", "answer"}} rows covering every id in the relevant file
  exactly once. An answer is a string: a value such as "v_l_m_e" or a
  non-empty list such as "[v_l_m_e, v_k_m_d]". final/test.jsonl also contains
  rows whose ids start with probe_; a legal final file covers them too.
  Answer files must be outside task/.

{env}Rules:
{RULES}
"""


TIMER_SH = """#!/bin/bash
# Prints the time remaining in the current phase.
D=$(cat "$(dirname "$0")/.phase_deadline" 2>/dev/null)
[ -z "$D" ] && { echo "unknown"; exit 0; }
R=$(( ${D%.*} - $(date +%s) ))
[ $R -le 0 ] && { echo "0:00 (time is up)"; exit 0; }
printf "%d:%02d remaining (hours:minutes)\\n" $((R / 3600)) $(((R % 3600) / 60))
"""


def practice_opener(task_dir: str) -> str:
    return (f"The practice phase has started. The task files are in {task_dir}; "
            f"start with {task_dir}/README.md.")


def final_opener(task_dir: str, t_f: float, killed: int) -> str:
    stopped = (f" {killed} process(es) still running in your container were stopped."
               if killed else " No process was running in your container.")
    return (f"The practice phase is over.{stopped} The held-out test set is at "
            f"{task_dir}/final/test.jsonl. query and submit_validation_answer are no longer "
            f"available. The final phase lasts at most {_hm(t_f)}; submit your answers with "
            f"submit_final_answer.")


def practice_nudge(seconds_left: float) -> str:
    return (f"Continue, or call finish_practice if you are done. "
            f"Time remaining in this phase: {_hm(seconds_left)}.")


def final_nudge(seconds_left: float) -> str:
    return (f"Continue, or call submit_final_answer if you are done. "
            f"Time remaining in this phase: {_hm(seconds_left)}.")


# Words the neutral text must not contain outside the Rules section
# (spec section 6). Checked by tests/v2/test_free_harness.py.
FORBIDDEN_WORDS = ("train", "model", "download", "student", "suggest", "consider")
