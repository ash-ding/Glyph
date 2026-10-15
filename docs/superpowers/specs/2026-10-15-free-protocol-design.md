# The `free` protocol — a shell, a clock, and the five task tools

Date: 2026-10-15. Status: reviewed, open items decided (see the end). No implementation yet.

A second protocol condition beside v2, not a replacement. v2 keeps its tool
version of the train arm (`build_dataset` / `train_model` / `infer_model`, and
the open-recipe revision in `2026-10-02-open-training-recipe-design.md`). The
`free` protocol removes those three tools and gives the agent a root shell in
an open container instead. The research questions are unchanged: whether a
frontier agent chooses to build another model as its own tool (RQ1), and how
well it builds and uses it (RQ2). Here the choice is made with nothing in the
tool list to suggest it.

## 1. What is fixed and what is free

Fixed, because the benchmark does not exist without them:

- the instance: syntax public, semantics hidden, `query` the only way to learn
  them, validation and test items refused by `query`;
- the held-out test, hidden until the final phase;
- scoring: exact match, the same metrics as v2 (`overall`, `by_split`,
  `tail`, `headroom`, probe `seen` / `unseen`);
- the container cannot see the host filesystem, any credential, or the oracle
  except through the tools;
- the frontier model is pinned by the metering gateway, for the main agent and
  every subagent.

Free, by decision:

- what the agent runs: it is root in an Ubuntu container with conda and
  network access, and may install anything;
- no student model is provided or named, and nothing is pre-downloaded;
  downloads count against the clock;
- no cap on queries, validation submissions, or turns; no idle rule;
- web access and subagents are on;
- the only budget is wall-clock time, per phase.

The two arms differ in one thing: `train` has one GPU in the container,
`no_train` has none. Image, tools, prompt, network and resource quotas are
identical. A `no_train` agent that trains something on the CPU is allowed to;
that is a finding, not a violation.

## 2. Environment

| | |
|---|---|
| container | Apptainer, from a SIF built from the Ubuntu 24.04 cloud rootfs (the host's Docker registry mirror is full, so images are built from the tarball), `--contain --cleanenv --pid --fakeroot`; the agent is uid 0 inside |
| writable layer | an ext3 overlay file of fixed size per run (`--overlay run.img`), created in-container with `mkfs.ext3 -d` (the host has no e2fsprogs); this is also the per-run disk cap |
| preinstalled | conda, git, curl, build tools; nothing from the ML stack |
| network | open; outbound connections are logged by destination host |
| hidden | the host filesystem (only the workspace is bound), every credential (the Vertex key stays with the gateway on the host; the container env carries no key), the oracle |
| GPU, `train` arm | `--nv`, then the entrypoint unmounts every `/dev/nvidiaN` except the assigned one before handing over; inside a user namespace the agent can neither `mknod` nor mount, so it cannot get them back. Verified on lumen1: `cuDeviceGetCount` = 1, `nvidia-smi -L` lists one GPU |
| GPU, `no_train` arm | no `--nv` |
| CPU and memory | `systemd-run --user --scope -p CPUQuota= -p MemoryMax=` around the container (Apptainer's own `--cpus/--memory` do not work with `--fakeroot`). Verified on lumen1: a 3 GB allocation under `MemoryMax=1G` is killed, 8 busy loops under `CPUQuota=200%` use 2 CPU-seconds per second |
| monitoring | GPU memory and utilisation sampled every 10 s per assigned GPU; host PIDs on any GPU attributed to runs through their scope cgroup, so a process on another run's GPU is detected, killed and logged |

Quotas per run, for six concurrent runs on lumen1 (8 × H100 80 GB, 160
logical cores, 1.5 TB RAM, 230 GB free on `/scratch`):

| resource | per run | six runs | host |
|---|---|---|---|
| CPU | 24 logical cores | 144 | 160; the rest is for the harness, the gateway and six CLI processes |
| memory | 160 GB | 960 GB | 1,586 GB |
| disk | 35 GB overlay | 210 GB | 230 GB on `/scratch` |
| GPU | 1 (`train`) / 0 (`no_train`) | 3 | 8 |

Both arms get the same CPU, memory and disk.

## 3. The agent

Claude Agent SDK, effort pinned to `high`, as in v2. Allowed tools: `Bash`,
`Read`, `Write`, `Edit`, `Glob`, `Grep`, `WebFetch`, `WebSearch`, `Task`
(subagents), and `mcp__glyph__*`. The SDK's `max_turns` is set to 100,000 so
that it never binds; the per-command Bash timeout is set to the time remaining
in the phase. Subagent calls are recorded as the SDK exposes them (the `Task`
call and its result in the main transcript); their token spend is metered by
the gateway and reported apart from the main agent's.

## 4. The tools

Five, the same in both arms; availability depends on the phase only.

| tool | practice | final | rule |
|---|---|---|---|
| `query` | ✓ | — | no cap. Refusals as in v2: validation and test items, expressions with a held-out pair, expressions deeper than the demos; refused expressions are not counted, malformed ones are |
| `submit_validation_answer` | ✓ | — | no cap; returns `overall` and `by_depth` |
| `check_answers` | ✓ | ✓ | unchanged |
| `finish_practice` | ✓ | — | unchanged |
| `submit_final_answer` | — | ✓ | once; a legal file ends the run |

Every result carries `seconds_remaining` for the current phase and nothing
else about budgets. `task/timer.sh` prints the same number.

## 5. Phases and clocks

**Practice.** Wall-clock `T_p` (default 4 h; the sweep runs 1, 2 and 4 h),
counted from the first message to the agent. Container start-up is outside
the clock; everything the agent does, including downloads, is inside. Practice
ends when the clock runs out or the agent calls `finish_practice`, and in no
other way: no turn cap, no idle rule. When the agent's turn ends without a
tool call, the harness replies "Continue, or call finish_practice if you are
done. Time remaining in this phase: Xh Ym." until the clock runs out.

**The deadline.** At `T_p` the harness calls `interrupt()`, allows 60 s for
the current turn to drain, then kills every process in the container except
the agent's own CLI process. The same at the end of the final phase.

**The switch.** Kill as above; write `task/final/test.jsonl` (10,000 test
rows and the probe rows, ids shuffled, no split labels); close `query` and
`submit_validation_answer`; send the final opener. The container is not
restarted and its files are untouched.

**Final.** Wall-clock `T_f` (default 30 min) from the final opener. Ends on a
legal `submit_final_answer` or at the deadline. At the deadline the v2 rule
applies: the last file that passed `check_answers(set="test")` is committed;
with none, the run scores zero.

**Safety line.** $500 of model spend, main agent and subagents together. It
is not a budget: crossing it ends the current phase as the deadline would,
and the report records it as the end reason.

## 6. What the agent is told

Facts only. The README and the system prompt state the task, the phases and
clocks, the tools, the file formats, the environment, that running processes
are stopped at the end of a phase, the submission rules and two rules below.
They do not contain the words train, model, download, student, suggest or
consider, and nothing that ranks one approach over another. The two arms get
the same text apart from the environment line.

Two rules, after PostTrainBench's rule 9 and its benchmark-lookup judge:

1. "Do not call any hosted language model other than the one you are running
   as." Without it the measured agent is not the pinned model.
2. "Do not search for or read material about this benchmark (Glyph)." The
   site and the repository publish the skeleton grammar, the presets and the
   probe construction, which shrink the hypothesis space. The tables are
   random, so nothing online answers an instance, but the structural prior is
   real help.

Both are audited after the run from the outbound-connection log and the
transcript (known LLM API hosts; URLs containing `glyph` or `ash-ding`), and
flagged in the report. Flagged runs are reported, not rescored.

### README draft (`task/README.md`)

```
# Glyph task workspace

Files in this directory:
  syntax.md          the expression grammar you may write
  demos.jsonl        worked examples: {"expr", "answer"} per line
  validation.jsonl   {"id", "expr"} per line; no answers
  queries.jsonl      every answered query this run, appended automatically
  timer.sh           prints the time remaining in the current phase
  final/             empty until the run switches to the final phase

Phases:
  - Practice lasts at most {T_p}. It ends when that time is up or when you
    call finish_practice. During practice you may call query and
    submit_validation_answer as often as you like.
  - When practice ends, every process still running in this container is
    stopped. final/test.jsonl is then written, and query and
    submit_validation_answer are no longer available.
  - The final phase lasts at most {T_f}. It ends when you call
    submit_final_answer with a legal file, or when the time is up. When the
    time is up, every running process is stopped; if one of your files passed
    check_answers on the test set, the last such file is submitted for you;
    otherwise the run scores zero.

Tools:
  - query: evaluates expressions. Validation and test items are refused, as
    are expressions containing a held-out operator pair and expressions
    deeper than the demos. Refused expressions are not counted.
  - submit_validation_answer: scores an answer file against validation.jsonl
    and returns the overall accuracy and a breakdown by expression depth.
  - check_answers: checks an answer file for legality (set="validation" or
    set="test"). It returns no score.
  - finish_practice: ends the practice phase.
  - submit_final_answer: submits your answers for final/test.jsonl. A legal
    file is accepted once and ends the run.
  Every tool result includes the seconds remaining in the current phase.

Answer files:
  JSONL of {"id", "answer"} rows covering every id in the relevant file
  exactly once. An answer is a value such as "v_l_m_e" or a non-empty list
  such as "[v_l_m_e, v_k_m_d]". final/test.jsonl also contains rows whose ids
  start with probe_; a legal final file covers them too.

Environment:
  You are root in an Ubuntu 24.04 container with conda and network access.
  This container has {one NVIDIA H100 80GB GPU | no GPU}, {N} CPU cores,
  {M} GB of memory and {D} GB of disk.

Rules:
  1. Do not call any hosted language model other than the one you are
     running as.
  2. Do not search for or read material about this benchmark (Glyph).
```

The system prompt keeps v2's first paragraph (a formal language whose syntax
is public and whose semantics are private), replaces the caps with the two
clocks, lists the five tools, and drops the student paragraph. The practice
and final openers are v2's, with the final opener also stating that running
processes were stopped.

## 7. Scoring and the report

Scores are unchanged. `tail` and the probe `seen` / `unseen` split come from
the oracle's query log, so they survive the removal of the student tools.

The report's covariates change:

| group | fields |
|---|---|
| clocks | seconds used per phase; end reason per phase (`deadline`, `finish_practice`, `submitted`, `usd_line`); auto-commit or agent commit |
| usage | `query` calls and expressions; validation submissions, each with score and timestamp; token spend, main agent and subagents apart; subagent count |
| GPU | the 10 s samples; seconds held (memory allocated) and seconds busy (utilisation > 0) |
| artifacts | `pip freeze` and `conda env export` at the end; models in the HF cache (name, size); directories in the workspace that hold an HF-format checkpoint; disk used |
| network | `WebFetch` / `WebSearch` counts; outbound destination hosts |
| audit | hosted-LLM hosts contacted; benchmark lookups; processes found on another run's GPU |

Two consequences of the missing caps are expected and are reported rather
than prevented: the unary tables (3 × 4,913 cells) can be bought whole in
minutes, so the report gives the unary coverage of the query log; and
repeated validation submissions can recover validation answers one item at a
time, so the report keeps the full submission history. Both are things we
want to see the agent do or not do.

## 8. The experiment

3 frozen instances, one per band (`low_3`, `mid_3`, `high_3`; `mid_3` is the
instance the site walks through) × `T_p` ∈ {1 h, 2 h, 4 h} × 2 arms = 18
runs, all on lumen1, in three waves of six (one `T_p` per wave, three GPUs in
use):

| wave | `T_p` | wall-clock incl. final and start-up |
|---|---|---|
| 1 | 1 h | ≈ 1 h 45 min |
| 2 | 2 h | ≈ 2 h 45 min |
| 3 | 4 h | ≈ 4 h 45 min |

About 9.5 h in total. Model spend: the v2 pilot cost $22–26 for 90 min of a
single agent; with subagents, a rough $20 / $40 / $80 per run at 1 / 2 / 4 h
gives $1,000–1,500 for the sweep. The main read is the `train` − `no_train`
headroom difference at each `T_p`, per band.

Before the sweep, one pilot pair (`mid_3`, both arms, `T_p` = 1 h) to check
the mechanics: the kill at the switch, the deadline, the auto-commit, the GPU
isolation, the audit log.

## 9. Implementation

- **PR-B, sandbox** (`v2/sandbox.py`, new `v2/image/`): the SIF build from
  the Ubuntu rootfs; the per-run overlay; `--fakeroot --nv --contain` with the
  device-node unmount in the entrypoint; the `systemd-run` scope; the outbound
  log; the GPU sampler with scope-cgroup attribution; the kill-all-but-CLI
  primitive. Tested on the real host (`slow`).
- **PR-C, harness** (`--protocol free`): wall-clock `drive_practice` /
  `drive_final` with an injectable clock, the deadline sequence, the switch
  without restart, the five-tool matrix, `seconds_remaining`, `max_turns` and
  the Bash timeout, the neutral nudge, prompt and README variants, the new
  covariates and audits. Tested with the fake client and a fake clock.
- **PR-D, docs and site**: `docs/tools.md` and the Evaluation protocol page
  gain the second condition.
- Then the pilot pair, then the sweep. Each step user-gated.

## Decided on review (2026-10-15)

1. The README wording in §6 stands as written.
2. The instances are `low_3`, `mid_3` and `high_3`, the middle of each band.
3. GPU isolation is the device-node unmount in the entrypoint, with the GPU
   sampler as the second line: a process found on another run's GPU is
   killed and logged. `nvidia-container-toolkit` (`cuda-rhel10`, 1.20.1)
   stays an option if the unmount ever proves insufficient.
4. The outbound-connection log is kept in full for every run, next to the
   audit summary; it is the only record of what external resources the
   agent used, and the viewer can show it.
