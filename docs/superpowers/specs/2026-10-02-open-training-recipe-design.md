# Open training recipe — agent-authored training scripts

Date: 2026-10-02. Status: spec for review. No implementation yet.
Revised 2026-10-15: training runs as a background job with a live log in the
agent's workspace (§B, §C, §G). The 2026-10-02 text had `train_model` block
until the script exited and return the last ~100 lines of output.

## Motivation

Today the train arm's recipe is almost entirely fixed (the agent chooses only
data, epochs, lr, and lineage; optimizer/schedule/batch/precision/LoRA are
frozen in `sft.py`). The benchmark's most defensible real-world mapping is
**agent-as-AutoML** (`docs/motivation-real-world-tasks.*`), and the decision
space that mapping cares about is the RECIPE itself. This spec opens the whole
training script to the agent, under a harness contract that preserves the
three things Glyph cannot compromise: **ledger integrity** (every GPU second
billed), **hidden semantics** (the dataset the agent built is the only
knowledge channel), and **base-model identity**.

## What the survey says (sources in the 2026-10-02 research notes)

Five harnesses examined: PostTrainBench (arXiv 2603.08640), FT-Bench/TREX
(arXiv 2604.14116), karpathy/autoresearch, MLE-bench (arXiv 2410.07095),
RE-Bench (arXiv 2411.15114). Conventions adopted here:

- **The contract is an artifact at a fixed path, not an entrypoint API**
  (PostTrainBench `final_model/`, MLE-bench `submission.csv`).
- **The grader never trusts agent code**: final evaluation runs harness-side.
- **Budgets are wall-clock on pinned hardware**; overruns are terminated and
  fall back to a defined score, not a crash of the run.
- **Eval assets live behind an immutability boundary** (autoresearch's
  untouchable `prepare.py` is the cleanest version of this).
- **Failed attempts bill their budget** and the run continues.
- **The agent learns the environment from the task prompt**, not by probing.
- Where we deliberately DIVERGE: PostTrainBench/FT-Bench allow open internet
  because data curation IS their task; for Glyph the hidden table makes the
  purchased dataset the only legal knowledge channel, so the training sandbox
  is **air-gapped** — which also enforces base-model pinning physically
  instead of by post-hoc LLM audit.

## A. Naming prerequisites (settled in discussion, do first)

1. The 8-hex entity is called **`model_id`** everywhere (was `student_id`).
2. **The harness mints model_ids**, not the agent: `train_model` takes
   `init: "base" | <model_id>` — `"base"` mints a fresh model_id
   (deterministic: `sha256(instance_seed:model:<ordinal>)[:8]`), an existing
   model_id continues that lineage from its latest checkpoint. A typo in
   `init` is a loud unknown-id error instead of a silent wrong-lineage
   continuation.
3. `infer_model`'s reference parameter is named **`model_id`** and accepts
   `"base" | <model_id> | <checkpoint_id>`; checkpoint format stays
   `<model_id>_ck_<8hex>` (lineage-position hash, PR #75).

## B. The tool surface

`train_model` is re-signed (fixed-recipe args removed — the recipe now lives
in the script) and **returns as soon as the job is launched** (§G):

```
train_model(script_path: str, dataset_id: str, init: str) ->
  {job_id, model_id, continued_from, log_path, status_path, <remaining>}

wait_model(job_id: str, timeout_s: int) ->
  {job_id, state, exit_code, stopped_by, gpu_seconds, gpu_seconds_remaining,
   checkpoint_id, train_record, log_tail, <remaining>}

cancel_model(job_id: str) -> {job_id, state, gpu_seconds, <remaining>}
```

`state` is `running | succeeded | failed | cancelled | killed`. The fields a
finished job produces (`exit_code`, `stopped_by`, `gpu_seconds`,
`checkpoint_id`, `train_record`) live in `status.json` and come back from
`wait_model`; they are not in `train_model`'s result, because the job has
not run yet when it returns.

- `script_path`: a Python file the agent wrote in its workspace. The harness
  snapshots it into the run dir (full text in the trace — auditability).
- `dataset_id`: a `build_dataset` product, unchanged. The script may
  transform/augment/subset it arbitrarily IN the sandbox; it cannot acquire
  new oracle data there.
- `init`: per §A. Continuation mounts the lineage's latest checkpoint as the
  starting weights.
- `epochs` / `lr` are GONE from the tool: they are the script's business.
- `wait_model` blocks until the job leaves `running` or `timeout_s` elapses
  (capped, see §G), then returns the job's status and the last lines of its
  log. `cancel_model` kills a running job (§G).
- `build_dataset`, `infer_model`, and everything else: unchanged. Inference
  stays harness-side vLLM — the agent never writes inference code, so the
  answer-file pipeline, the one-surface-form rule, and infer metering are
  untouched.
- The tool set grows from eight to ten. `wait_model` and `cancel_model` sit
  in the same availability cell as `train_model`: train arm, practice only.

## C. The execution contract (what the script must obey)

The harness executes the script in a **dedicated bwrap sandbox on one pinned
GPU** (`CUDA_VISIBLE_DEVICES` set to a single device). Working directory
layout, all paths fixed:

```
./train.py          the agent's script (snapshot)
./dataset.jsonl     the chosen dataset, read-only
./base_model/       HF-format starting weights, read-only
                    ("base" -> the pinned base model; continuation -> the
                    lineage's latest checkpoint)
./out/              writable, initially empty
./tmp/              writable scratch
```

Run as `python train.py` (no arguments — the layout IS the interface).
Success = exit 0 AND `./out/` contains an HF-format checkpoint that the
harness can load. On success the harness assigns the checkpoint_id, registers
the lineage, and **discards/overwrites any tokenizer files in `out/` with the
base model's** (the tokenizer is part of the task's fixed surface form, not
the recipe).

**Environment**: the frozen training env (same torch/transformers/numpy as
`sft.py` uses; exact versions printed in the task README). **No network.** No
GPU other than the pinned one. Nothing from the instance is visible — the
sandbox contains exactly the four mounts above.

**Budget**: wall-clock from exec to exit, billed to the ledger's `train` line
against the same per-call (1800 s) and cumulative (7200 s) caps as today. At
the per-call cap the process group is killed (`stopped_by: "timeout"`).
**Every attempt bills**, including failures (non-zero exit, timeout,
cancellation, missing/unloadable `out/`) — the job ends in `failed`,
`killed` or `cancelled` with `exit_code` and `stopped_by` in `status.json`,
and the run continues (failed-attempt-bills-baseline is the survey-wide
convention). A cancelled or killed job yields no checkpoint, whatever the
script left in `out/`.

**Feedback**: the job's combined stdout/stderr is streamed, line-buffered, to
`work/jobs/<job_id>/train.log` in the agent's workspace while it runs, and
`work/jobs/<job_id>/status.json` is written at launch (`state: running`) and
rewritten at exit. Both are read-only to the agent, like `task/`. The agent's
training-loss telemetry is whatever its own script prints, and it reads as
much or as little of it as it wants; nothing is truncated for it. If
`./out/train_record.json` exists it is passed through verbatim into
`status.json`, `wait_model`'s result and the trace.

## D. What stays fixed (the immutability boundary)

- **Base model identity**: only the mounted weights exist; no net ⇒ no
  substitution/distillation channel. (PostTrainBench polices this by LLM
  audit; we get it by construction.)
- **Data legality**: `build_dataset` provenance rules unchanged; the sandbox
  sees one JSONL and nothing else.
- **Evaluation**: probes, test scoring, `submit_final_answer` legality,
  report — all harness-side, unreachable from the training sandbox.
- **Meters**: GPU wall-clock billing, USD line, turn caps. Watching a job
  costs the agent its own turns and tokens, as any other activity does.
- **Prompt neutrality**: the README documents the contract (layout, env
  versions, single GPU, timeout, no-net) and nothing about what a good recipe
  looks like.

## E. Consequences

1. **Protocol version bump**: train-arm results are not comparable across
   this change (they already are not, given this week's changes). The tool
   surface changes shape (ten tools, `train_model` no longer blocks), so the
   system prompt, `docs/tools.md` and the site's Evaluation protocol follow.
   The fixed-recipe code path (`sft.train`) remains in-tree as the reference
   oracle's and the capacity check's trainer; the agent-facing fixed-recipe
   path is REMOVED, not kept alongside (one path, no drift).
2. **Issue #73 (agent-path oracle) is superseded**: with no fixed agent
   pipeline there is nothing to align the oracle to. The oracle keeps its
   role as a *reference-recipe ceiling*; close #73 pointing here.
3. The train arm's GPU-second economics change meaning: budget now prices the
   agent's whole experimental loop (failed attempts included), which is
   exactly the agent-as-AutoML quantity. Whether 1800/7200 s are the right
   caps for script mode is an open calibration question — revisit after the
   first pilot, alongside the small-pool early-stop fix queue.

## F. Implementation sketch & tests

- `v2/executor.py` (new): sandbox assembly (mount layout, env, pinned GPU),
  background launch in its own process group, the live log and
  `status.json`, timeout-kill, cancel, billing on exit, out/ validation (HF
  loadability probe without GPU: config/tokenizer/shape checks; full load
  deferred to first `infer_model`), tokenizer overwrite.
- `v2/student.py`: `StudentPool.train` becomes a thin wrapper over the
  executor (lineage bookkeeping, model_id minting per §A, caps unchanged);
  `HParams`/sft path stays for reference use. The pool tracks the one
  running job for the `gpu_busy` rule (§G).
- `v2/tools.py` / `mcp.py` / `prompts.py` / `docs/tools.md`: new schema and
  contract text; `t_wait_model`, `t_cancel_model`.
- `v2/harness.py`: kill the running job at the phase switch (§G).
- CPU tests with a fake executor: contract validation (missing out/, bad
  exit, timeout flag), billing on failure, init semantics, minting
  determinism, snapshot-into-trace; launch returns before the job ends,
  `wait_model` returns on completion and on timeout, `cancel_model` bills
  elapsed time and leaves no checkpoint, `gpu_busy` refusal is uncharged,
  the job is killed and billed at the phase switch, the log cap holds. One
  `slow` GPU test: a 20-line real script fine-tunes the 0.6B for a few steps
  end to end, then `infer_model` consumes the checkpoint. e2e smoke: scripted agent ships a trivial
  `train.py`.
- Rollout: PR-1 naming (§A), PR-2 executor + tools (§B–D, §G), PR-3 docs/
  open-questions disposition (#73) + progress. Each user-gated as usual.

## G. Asynchronous execution and the live log (revision 2026-10-15)

**Why.** Every v2 tool call is synchronous: the SDK's agent loop waits for
the result, so while `train_model` blocked the agent could do nothing, not
even read a log. The 2026-10-02 text therefore had to choose how much output
to hand back afterwards (`stdout_tail`, ~100 lines), and the agent could
only react to a run once it was over. autoresearch's agent, by contrast,
launches `uv run train.py > run.log 2>&1` and reads `run.log` whenever it
likes. This revision gives the agent the same view without giving it the
shell: the harness launches the script and streams its output into the
workspace, and the agent follows, waits for, or cancels the job.

**Job lifecycle.**
- `train_model` snapshots the script, assembles the sandbox and launches
  `python train.py` in its own process group, then returns `job_id`,
  `model_id`, `continued_from`, `log_path` and `status_path`. Launch
  failures that happen before the script starts (unknown `init`, unknown
  `dataset_id`, GPU budget already exhausted, a job already running) are
  structured errors, uncharged, as today.
- While the job runs, `work/jobs/<job_id>/train.log` grows line by line and
  `status.json` says `running`. The agent may read either at any time with
  its file tools.
- On exit the executor bills the wall-clock, validates `out/`, assigns the
  checkpoint on success, and rewrites `status.json` with the final `state`,
  `exit_code`, `stopped_by`, `gpu_seconds`, `checkpoint_id` and
  `train_record`.
- The harness keeps its own copy of the log and `status.json` in the run
  dir, next to the script snapshot, so the trace and the viewer show the
  whole job.

**`wait_model(job_id, timeout_s)`.** Blocks until the job leaves `running`
or `timeout_s` elapses, whichever is first, and returns the current status
plus `log_tail` (the last 50 lines). `timeout_s` is capped at 600 s; a
longer wait is several calls. This is the cheap way to wait: one tool call
and one turn, instead of a `sleep`-and-`tail` loop that spends a turn per
poll against `T_p`. Calling it on a finished job returns immediately.

**`cancel_model(job_id)`.** Kills the process group, bills the elapsed
wall-clock, sets `state: cancelled`, `stopped_by: "cancelled"`, and produces
no checkpoint; `out/` is discarded. This is what live visibility is for: a
diverging loss or a recipe that is obviously too slow costs the agent what
it has run so far, not the whole per-call cap.

**One GPU, one job.** The sandbox pins a single GPU, so at most one job runs
at a time. While a job is `running`, `train_model` and `infer_model` return
`gpu_busy`, uncharged. The agent can query, build datasets, write files and
edit its script meanwhile; those do not touch the GPU.

**Phase switch.** A job still `running` when practice ends — by
`finish_practice`, the submit cap, the turn cap, idle turns or the USD line
— is killed and billed (`state: killed`, `stopped_by: "phase_switch"`), with
no checkpoint. The task README states this, so an agent that wants the
checkpoint waits for the job before ending practice. `finish_practice` is
not refused while a job runs: the other end triggers cannot be, and one
rule is simpler than two.

**Log cap.** `train.log` is capped at 50 MB. Past the cap the executor
appends one marker line and discards further output; the process keeps
running. A script that prints every step is the agent's own cost in tokens
when it reads the file.

**What the agent gains and loses against running the script itself.** It
gains nothing it would not have in a shell, except that failures are cheap
to abandon. It loses interaction with the live process (no stdin, no
debugger, no `nvidia-smi`), which the experiments do not need. Everything
the harness wanted from executing the script — the air-gapped sandbox, the
pinned GPU, exact billing, the kill at the cap, `out/` validation, the
tokenizer overwrite, lineage bookkeeping — is unchanged.

## Open for review

1. Per-call timeout for script mode: keep 1800 s or raise (a from-scratch
   recipe compiles/warms up slower than our tuned loop)? With `cancel_model`
   the agent can cut a bad run short itself, which argues for a higher cap.
2. ~~`stdout_tail` length (default 100 lines)?~~ Settled by §G: the agent
   reads the live log; `wait_model` returns the last 50 lines.
3. Should continuation mount the WHOLE lineage (all checkpoints) read-only,
   or only the latest? (Spec says latest; whole-lineage enables
   model-averaging recipes at the cost of disk.)
4. Keep `scripts/` under the workspace visible to the final phase? (Spec: the
   training tool is practice-only, as today.)
5. `wait_model`'s cap on `timeout_s` (600 s) and the 50 MB log cap: both are
   guesses to be checked against the first pilot.
6. Should a cancelled job keep a checkpoint the script already saved to
   `out/`? (§G says no — a cancel is a discard — but a script that
   checkpoints every N steps might reasonably expect otherwise.)
