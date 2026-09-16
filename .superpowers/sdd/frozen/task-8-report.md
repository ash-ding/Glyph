# Task 8 report — reference-battery runner

## Files created
- `tools/run_reference.py` — `merge_reference`, `run_cheap`, `_evidence_text`,
  `_frontier_answer_fn`, `_parse_only`, `_select_instances`, `main` (CLI).
- `tests/reference/test_run_reference.py` — `test_merge_per_oracle_idempotent`,
  `test_run_cheap_shape`.

## Commit
`cacfbfd` — "Add reference-battery runner: merge_reference + run_cheap (task 8)"
(branch `spec-frozen-instances`, trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`).

## Test command + final output
    cd ~/code/Glyph-v2 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate glyph && \
    python -m pytest tests/reference/test_run_reference.py -q -p no:cacheprovider

    ..                                                                       [100%]
    2 passed in 0.19s

Confirmed RED first (ModuleNotFoundError: No module named run_reference) before
adding the implementation, then GREEN as above. Also ran `python -m py_compile
tools/run_reference.py` and `python tools/run_reference.py --help` as an import
and argparse sanity check; neither executes the weights or a0prime branches.

## Deviations / assumptions
- `docs/benchmark/frozen_instances.json` does not exist yet on this branch (the
  manifest-writing tools from tasks 1/3/4 exist but no manifest has been
  generated/committed). `load_frozen` already returns `[]` for a missing path, so
  `--instance all` against a missing manifest is a no-op with no crash; `--instance
  <id>` against a missing/empty manifest prints a message to stderr and returns 1.
  This is consistent with the brief (manifest generation is out of scope for task
  8), but it means the CLI's per-instance loop body is not exercised end-to-end in
  this task -- only `merge_reference` and `run_cheap` are unit-tested, per the
  brief's explicit scope.
- `weights` branch: follows the brief -- for each `--seen-frac`, calls
  `train_student` then `score_ceiling` (both from `glyph.reference.weights_ceiling`),
  stores under `{"<frac>": {...}}`. Never invoked by any test.
- `a0prime` branch: the brief names `buy_evidence` + "a real frontier `answer_fn`" +
  `run_a0prime`. There was no existing frontier-call helper already wired for this
  purpose (`a0prime.py`'s own docstring says the real call is a later [API] step), so
  I built `_frontier_answer_fn` on top of the repo's existing `glyph.vertex.chat`
  convenience wrapper (`src/glyph/vertex.py`, used elsewhere for one-turn
  Vertex/Anthropic calls): one prompt per chunk containing the evidence text plus a
  numbered list of expressions, parsed back into one answer line per item.
  `glyph.vertex` (and therefore `anthropic`/`AnthropicVertex`) is imported lazily
  inside `_frontier_answer_fn`, not at module scope, so importing
  `tools/run_reference.py` never requires Vertex credentials or the `anthropic`
  package -- mirrors how `weights_ceiling.py` lazily imports `torch`/`transformers`.
  This is new code (not a straight port -- no prior script implemented this exact
  seam) and is entirely unexercised by any test in this task, by design.
- Coverage-point -> `entries_target` mapping for `buy_evidence`: used
  `max(1, round(seen_frac * inst.cfg.n_values))`, the same `n = cfg.n_values`
  quantity `weights_ceiling.train_student` uses for its own seen-fraction hashing,
  so `--seen-frac` means the same thing (fraction of the table) for both the
  `weights` and `a0prime` branches. Not specified verbatim in the brief; flagged as
  the one interpretive choice in the CLI wiring.
- `merge_reference` writes pretty JSON with `indent=2`, insertion order preserved
  (no `sort_keys`), trailing newline. Not specified beyond "pretty JSON" in the brief.

## Concerns
- The `a0prime` CLI branch (evidence-text formatting + frontier prompt shape) is my
  own construction, not ported from an existing script, and is entirely untested by
  design -- worth a careful look before it is ever run for real against the API in a
  later task.
- No `docs/benchmark/frozen_instances.json` exists on this branch yet, so the full
  `--instance <id>` CLI path (manifest lookup -> generate -> paired_subset -> merge)
  has only been sanity-checked via `--help`/`py_compile`, not run end-to-end.


## Fix round 1 (review finding: Important -- A0' answer alignment by line position)

**Issue:** `_frontier_answer_fn` aligned the frontier's reply to `expr_srcs` by
line position (split non-blank lines, truncate to `len(expr_srcs)`). A leading
preamble/acknowledgement line from the model would shift every answer by one,
and `run_a0prime` only checks the returned count, not identity, so this would
silently misscore.

**Fix applied to `tools/run_reference.py`:**
1. The A0' prompt built in `_frontier_answer_fn` now requires each answer on
   its own line prefixed with its 1-based index, exactly `N. <answer>` (e.g.
   `1. v_a_b`), and explicitly says no other text before/between/after the
   answers.
2. Added a pure helper `_align_answers(reply: str, n: int) -> list[str]` that
   parses each line of `reply` against `^\s*(\d+)\.\s*(.*)$`, places each
   matched answer at position `idx - 1`, ignores lines that do not match
   (preamble, commentary, ...), and leaves any index in `1..n` that was never
   matched as `""` -- so a missing/garbled answer scores wrong on its own item
   rather than shifting the rest. `_frontier_answer_fn` now calls
   `_align_answers(text, len(expr_srcs))` instead of the old line-split/
   truncate logic.

**Covering test** added to `tests/reference/test_run_reference.py`:
`test_align_answers_ignores_preamble` -- one case with a leading preamble line
("Here are the answers:") plus `1. v_a\n2. v_b\n3. v_c`, asserting
`_align_answers(reply, 3) == ["v_a", "v_b", "v_c"]`; one case with a missing
index (`1. v_a\n3. v_c`, n=3), asserting the result is `["v_a", "", "v_c"]`.

**Test command + output:**

    cd ~/code/Glyph-v2 && source ~/miniforge3/etc/profile.d/conda.sh && conda activate glyph && \
    python -m pytest tests/reference/test_run_reference.py -q -p no:cacheprovider

    ...                                                                      [100%]
    3 passed in 0.20s

No API call, no GPU, no full suite run -- only the scoped test above (plus a
`py_compile` import sanity check). `_frontier_answer_fn`/`_align_answers` are
still only reachable from the CLI's `a0prime` branch, itself only entered via
explicit `--only a0prime`.
