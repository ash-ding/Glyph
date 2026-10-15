# Open training recipe — agent-authored training scripts

Date: 2026-10-02. Status: spec for review. No implementation yet.

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
in the script):

```
train_model(script_path: str, dataset_id: str, init: str) ->
  {model_id, checkpoint_id, continued_from, exit_code, gpu_seconds,
   gpu_seconds_remaining, stopped_by, stdout_tail, train_record}
```

- `script_path`: a Python file the agent wrote in its workspace. The harness
  snapshots it into the run dir (full text in the trace — auditability).
- `dataset_id`: a `build_dataset` product, unchanged. The script may
  transform/augment/subset it arbitrarily IN the sandbox; it cannot acquire
  new oracle data there.
- `init`: per §A. Continuation mounts the lineage's latest checkpoint as the
  starting weights.
- `epochs` / `lr` are GONE from the tool: they are the script's business.
- `build_dataset`, `infer_model`, and everything else: unchanged. Inference
  stays harness-side vLLM — the agent never writes inference code, so the
  answer-file pipeline, the one-surface-form rule, and infer metering are
  untouched.

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
the per-call cap the process group is killed. **Every attempt bills**,
including failures (non-zero exit, timeout, missing/unloadable `out/`) —
which return a structured error carrying `exit_code`, `stopped_by`, and
`stdout_tail`, and the run continues (failed-attempt-bills-baseline is the
survey-wide convention).

**Feedback**: `stdout_tail` (last ~100 lines of combined stdout/stderr) comes
back in the tool result — the agent's training-loss telemetry is whatever its
own script prints. If `./out/train_record.json` exists it is passed through
verbatim into the result and the trace.

## D. What stays fixed (the immutability boundary)

- **Base model identity**: only the mounted weights exist; no net ⇒ no
  substitution/distillation channel. (PostTrainBench polices this by LLM
  audit; we get it by construction.)
- **Data legality**: `build_dataset` provenance rules unchanged; the sandbox
  sees one JSONL and nothing else.
- **Evaluation**: probes, test scoring, `submit_final_answer` legality,
  report — all harness-side, unreachable from the training sandbox.
- **Meters**: GPU wall-clock billing, USD line, turn caps.
- **Prompt neutrality**: the README documents the contract (layout, env
  versions, single GPU, timeout, no-net) and nothing about what a good recipe
  looks like.

## E. Consequences

1. **Protocol version bump**: train-arm results are not comparable across
   this change (they already are not, given this week's changes). The
   fixed-recipe code path (`sft.train`) remains in-tree as the reference
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
  exec + timeout-kill + billing, out/ validation (HF loadability probe
  without GPU: config/tokenizer/shape checks; full load deferred to first
  `infer_model`), tokenizer overwrite.
- `v2/student.py`: `StudentPool.train` becomes a thin wrapper over the
  executor (lineage bookkeeping, model_id minting per §A, caps unchanged);
  `HParams`/sft path stays for reference use.
- `v2/tools.py` / `mcp.py` / `prompts.py` / `docs/tools.md`: new schema and
  contract text.
- CPU tests with a fake executor: contract validation (missing out/, bad
  exit, timeout flag), billing on failure, init semantics, minting
  determinism, snapshot-into-trace. One `slow` GPU test: a 20-line real
  script fine-tunes the 0.6B for a few steps end to end, then `infer_model`
  consumes the checkpoint. e2e smoke: scripted agent ships a trivial
  `train.py`.
- Rollout: PR-1 naming (§A), PR-2 executor + tool (§B–D), PR-3 docs/
  open-questions disposition (#73) + progress. Each user-gated as usual.

## Open for review

1. Per-call timeout for script mode: keep 1800 s or raise (a from-scratch
   recipe compiles/warms up slower than our tuned loop)?
2. `stdout_tail` length (default 100 lines)?
3. Should continuation mount the WHOLE lineage (all checkpoints) read-only,
   or only the latest? (Spec says latest; whole-lineage enables
   model-averaging recipes at the cost of disk.)
4. Keep `scripts/` under the workspace visible to the final phase? (Spec: the
   training tool is practice-only, as today.)
