# Glyph — Session Handoff (2026-09-29)

A snapshot to let a fresh agent resume work with the user on the **Glyph** research
project. Read this top-to-bottom once, then the doc map in §9 as needed.

---

## 0. TL;DR — how to resume in 2 minutes

- **What it is:** a *controlled synthetic* benchmark that measures **when an agent should train a
  specialist model vs. solve a task in-context** — parameterized by a measured knob **π**.
- **Repo:** `~/code/Glyph` on hosts **lumen1 / lumen2 / lumen3** (SSH); on **penguin** at
  `/export/scratch/large/dimcode/Glyph`. Branch `main` (all four hosts synced). No open PRs.
- **Env:** conda env `glyph` (python at `~/miniforge3/envs/glyph/bin/python`, pytest 9.1.1).
- **Run tests:** `~/miniforge3/envs/glyph/bin/python -m pytest -q -m "not slow"` → ~265 pass, 6 slow deselected (~10 min). The `slow` marker gates GPU/API/sweeps.
- **Viewer:** static server is UP on lumen1:8000; restart the local SSH tunnel to see it (see §2).
- **Reply language:** the user wants **Chinese** replies (see their memory). Repo docs are English/bilingual.
- **Everything since 2026-09-15 landed via PRs #39–#45** (frozen benchmark, reuse-by-id, viewer, thinking capture, docs). Details in §4.

---

## 1. What the project is (the framing)

Glyph generates a small language whose **semantics are hidden**. Each task = a hidden program
`P = skeleton ∘ tables`:
- **skeleton** — a composition of structural operators, sampled from a *finite grammar* → **describable**, the "code/reasoning" half a frontier model is **good at**.
- **tables** — per-operator lookups realized as **frozen random MLPs** → **indescribable** (no rule), the "memorization" half a frontier model is **structurally bad at**.

**π = the skeleton's share of total difficulty** (`π = L_skel/(L_skel+L_table)`). The sharpened
framing (this session, in `docs/motivation-real-world-tasks.*`):

> **π ≈ how much of a task falls in the frontier model's blind spot.** Low π → mostly blind spot
> (a random-table part it can't reason out) → *training a specialist should pay off*. High π →
> mostly reasoning → *just solve in-context*. The benchmark asks whether an agent **recognizes the
> blind spot, decides to train a specialist, and whether that actually helps — and at what π it's worth it.**

Real-world maps: private-KB **RAG vs fine-tuning**; per-tenant personalization; private DSL/API
semantics; and most defensibly, **agent-as-AutoML** (an agent that mid-task trains a specialist).
The honest caveat: a random-MLP table is *worst-case for generalization*; the
`unary/binary_coupling` config knob dials "how learnable the blind spot is" (hard↔easy modality),
turning that caveat into a controllable axis. See `docs/motivation-real-world-tasks.en.md`.

Related-work neighbors (from a web scan this session, **verify before citing**): MLE-bench,
RE-bench, AI4AI-Bench (RSI), **PostTrainBench / FT-Bench** (agent-automates-fine-tuning — closest),
and ICL-vs-FT empirical studies. None occupy our exact intersection (controlled synthetic + π knob
+ reference ceilings + the ICL-vs-weights *decision* + contamination-proof). A differentiation pass
(deep-read AI4AI-Bench / PostTrainBench / FT-Bench) is an open to-do.

---

## 2. Repo, environment, infra

- **Hosts:** the codebase is checked out on **lumen1 / lumen2 / lumen3** at `~/code/Glyph`, and on
  **penguin** (`cs-foundations`) at `/export/scratch/large/dimcode/Glyph` (penguin has 4× L40S; its
  conda env is a prefix env at `/export/scratch/large/dimcode/envs/glyph`, with caches on scratch — its
  `/tmp` is only ~3 G). Primary work host is **lumen1**; the local machine (`/Users/asherding`) is NOT
  the repo — drive everything via `ssh <host> '...'`. All four are on `main` and kept in sync.
- **SSH gotcha (important):** always use `ssh -o ServerAliveInterval=0 -o ConnectTimeout=30 lumen1`.
  `exit 255` on a reachable host = aggressive keepalive under load, just retry. Wholesale
  unreachability = idle GPU/host reclaim (recovers on its own or after a restart) — not hardware failure.
- **conda:** env `glyph`. Prefer the explicit interpreter `~/miniforge3/envs/glyph/bin/python`
  (a plain `python` over SSH resolves to a base conda WITHOUT pytest — a known false start).
- **Heredoc-over-SSH gotcha:** backticks / `$(...)` / apostrophes / brackets in single-quoted SSH
  commands break in the *local* zsh. For file writes and commit messages, **pipe a local file to
  stdin** (`ssh lumen1 'cat > path' < localfile`; `git commit -F -` < msgfile`) instead of heredocs.
- **Detached background jobs over SSH are flaky** (`setsid ... &` often dies with the session).
  Prefer running a long job in the FOREGROUND under the harness's own background mechanism
  (a blocking `ssh` launched with run_in_background), which notifies on completion.
- **Viewer (run browser):**
  - Static server on lumen1: `~/miniforge3/envs/glyph/bin/python -m http.server 8000 --bind 127.0.0.1`
    from `~/code/Glyph` (serves `glyph-viewer/`; it was UP as of this handoff).
  - Alternative that re-collects runs: `tools/serve_viewer.py --port 8000 <run-roots>`.
  - Local tunnel (from the user's laptop): `ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 8000:localhost:8000 lumen1`, then open **http://localhost:8000/glyph-viewer/**.
  - The tunnel drops periodically (lumen connection behavior); "can't see the viewer" almost always = dead tunnel, just restart it. Benchmark JSONs live under `glyph-viewer/runs/`.
- **Model access on lumen:** Claude runs via the Vertex SA key (lightwell-devel/global) through the
  metering gateway — never subscription / personal ADC there.

---

## 3. Architecture — the three stages

Full detail is in the bilingual docs (`docs/data-generation.*`, `docs/data-validation.*`). Summary:

**① Data generation** (`src/glyph/data/`). `generate(seed, cfg)` is deterministic off a single
`np.random.default_rng(seed)` → same seed reproduces the instance bit-for-bit (basis of the
fingerprint + reuse-by-id). Value space `|V| = 17³ = 4913` (values render like `v_a_b_c`). Config
`GlyphConfig` (`config.py`); `atomic_ratio` is the primary π knob. Presets `pi_low / pi_mid /
pi_high` (`PRESETS`). Test splits `iid / comp / depth`. **Atomic ops are only reachable through a
structural op** (bare `u0(x)` is illegal; lists must be length 2–4) — so an agent can only obtain
*expression-level* data, never clean atomic cells.

**② Data validation — reference oracles** (`src/glyph/reference/`, `tools/run_reference.py`). A
per-instance **paired 500-item subset** (`subset.paired_subset`, seed 777). Oracles, each reporting
`{overall, by_split, tail, headroom}` by exact match:
- `skeleton` (true skel + identity tables), `table` (true tables + trivial skel), `perfect` (=1.0) — **CPU, committed**.
- `weights` **[GPU, deferred]** — Qwen3-1.7B fine-tuned on a `seen_frac` slice of **atomic cells** (6000 steps); table-only.
- `a0prime` **[API, deferred]** — a frontier model answers **end-to-end** from a `seen_frac` of evidence in context.
`weights` vs `a0prime` is the core "weights vs context, same coverage" comparison. **No p-value
anywhere; π is the quantity.**

**③ Evaluation protocol v2** (`src/glyph/v2/`). A CA-agent (frontier model, Claude Code CLI) runs
in a metered sandbox: **Practice** (query oracle Q=1000, submit ≤20 to validation, turn cap T_p) →
**Final** (no oracle, held-out test, T_f, one commit). Arms **train / no_train** — train adds
`build_dataset / train / student_infer`. Scored on the **full 10k test** by exact match →
`overall / by_split / by_depth / tail / headroom` + covariates. A **metering gateway** pins the
model and bills by usage; the USD safety line is host-side only (agent never sees dollars).
Cost told to the agent: prompt says only "training/inference cost **turns**, not queries"; the GPU
budget (per-call 1800s / total 7200s) is only discoverable from `gpu_seconds_remaining` in tool
results — and that counter meters **training only, not inference**.

---

## 4. Current state on main (what's built)

Landed via PRs #39–#45 (all merged, branches deleted):
- **Frozen benchmark (#39):** 15 instances (3 π-bands × 5), selected **preset-pure within
  archetype π-windows** — low `pi_low` [0.20,0.30), mid `pi_mid` [0.45,0.53), high `pi_high`
  [0.70,0.80). Committed: `docs/benchmark/{frozen_instances,candidates(210),reference_ceilings}.json`.
  Integrity gate `tests/test_frozen_instances.py` regenerates all 15 and checks fingerprint +
  measured_pi. Cheap ceilings show a clean monotone phase diagram (low table-favored → high
  skeleton-favored; crossover ~π 0.48–0.49).
- **Reuse-by-id (part of the reuse feature):** `load_instance("high_3")` / `frozen_entry(id)` in
  `src/glyph/reference/frozen.py` (regenerate + fingerprint-verify); `RunConfig.instance_id`;
  CLI `python -m glyph.v2.cli run --arm no_train --instance-id high_3` (and `grid --instance-ids`).
- **Viewer (#40, #41):** Runs list + a **band→ID instance catalog**; a preset-scoped instance
  filter (preset dropdown = all/pi_low/pi_mid/pi_high); reference card shows `overall/tail` +
  `by_split (iid·comp·depth)` + `headroom`.
- **Gateway thinking capture (#43):** the CLI transcript redacts extended-thinking text; the
  **gateway** now captures it (SSE `thinking_delta` reassembly + non-streamed) into
  `run_dir/thinking.jsonl`; `export.py` joins it to turns **by tool_use id** (id-only, no positional
  fallback → no misattribution); the viewer expands it in a collapsed "💭 thinking" block. **New
  runs only** — old runs keep the redacted marker. NOTE: **the SSE field names follow the public
  Anthropic Messages spec and are NOT yet validated against real Vertex SSE** — the FIRST real
  train/no_train run validates this; if `thinking.jsonl` is empty, only the field names need adjusting.
- **Docs (#42, #44, #45):** `data-generation.*`, `data-validation.*`, `motivation-real-world-tasks.*`
  (bilingual), plus an open-questions item on train-arm data starvation.
- Also there is a private **overview artifact** (published this session): a one-page
  English intro (data gen / oracles / protocol) at a claude.ai/artifact URL — still framed as
  "structure vs memorization"; re-framing it to "frontier blind-spot → train specialist" is a to-do.

Two flagged follow-up chips (out-of-scope this session): run.json `n_val` metadata on the frozen
path; the viewer "💭 thinking" block reuses the `.think` class so captured text renders slightly muted.

---

## 5. Key findings from this session (the pilot analysis)

Only two real runs exist so far — `pi_mid`/seed 1001, train & no_train (in `glyph-viewer/runs/`).
They are **not** among the 15 frozen instances (different seed). Findings:

- **Both cracked the structure and stalled on the atomic tables.** comp split highest (train 0.51 /
  no_train 0.49 — held-out composition generalized); depth split lowest and collapsing with depth
  (each deeper expr chains more atomic lookups). `test_covered ≈ 0.25` (Q=1000 only covers ~25% of
  needed cells); `tail ≈ 0.20`. Overall train **0.4197** / no_train **0.4015** — nearly identical.
  The frontier is good at the skeleton, stuck on the random-MLP tables.
- **The train arm's student was catastrophically under-trained.** `train` steps =
  `ceil(|dataset|/batch=32) × epochs`; the agent built only expression-level data (160 examples) →
  **ck1 = 3 steps, ck2 = 50 steps** vs the weights oracle's 6000 steps. `student_infer` output was
  mostly truncated (ck1 5/5, ck2 95/100 hit the 96-token cap) → the agent abandoned it for a
  hand-written symbolic solver. **Root cause is training budget (50 steps), not LR** — compounded by
  expression-level data. The arm is *structurally starved* (bare-atomic queries illegal → only
  expression-level data; Q=1000; steps∝data). Recorded in `docs/open_questions.md`.
- **GPU cost of that train run: ~247.6 GPU-s total** (train 35.8 / **inference 211.8**). The visible
  `gpu_seconds_remaining` counter meters training only; inference (the bigger cost) is unmetered there.

---

## 6. Open questions / deferred work / next steps (prioritized)

These are the natural next moves; #1–#2 are the science, #3 gates them.

1. **[GPU] weights-ceiling real runs** — `python tools/run_reference.py --only weights --seen-frac 0.02 0.05 0.10 --instance <id>`. A specific open question raised this session: at low `seen_frac`, does the student produce **valid, terminating** outputs (failure = wrong value) or does low data also break the *format* (truncation)? Report parse-rate / truncation-rate / accuracy separately. Fills the `weights` column.
2. **[API] a0prime real runs** — `--only a0prime`. BUT resolve the design cluster first (from the spec/plan reviews): **N1** route through the metering gateway (currently `_frontier_answer_fn` calls `glyph.vertex.chat` directly); **N2** the ~3× "entries-seen" denominator mismatch (a0prime counts distinct unary ≈ frac×4913 vs weights `seen_u` per-op ≈ frac×3×4913) — align it; **N3** wire `retrieval_split` / the `retrieval_vs_extrapolation` output; **N4** confirm the run-scoring 500-subset == reference `paired_subset(seed=777)`.
3. **Fix the train-arm data starvation** (blocks measuring "when delegating to a specialist pays off"): a min-steps floor / decouple `train` steps from dataset size; a wider training-data budget (larger Q, or a cheaper bulk-cell channel); and/or turn up `coupling` so a specialist can generalize. See the `open_questions.md` item.
4. **External-validity bridge** — the single most important thing to make the benchmark matter: show one Glyph finding predicts/matches behavior on a real structure-vs-memorization (or specialist-training) task. Without it, it's an elegant instrument with an unproven dial.
5. **Related-work differentiation** — deep-read AI4AI-Bench / PostTrainBench / FT-Bench; write a "how we differ" paragraph.
6. Smaller: re-frame the overview artifact to the blind-spot framing; the two follow-up chips (n_val metadata, viewer thinking-muted).

Also flagged for the user earlier (now effectively settled): the selection-by-measured-π-within-windows policy is in place; the "final_from_student" covariate is always null (see open_questions).

---

## 7. Working conventions

- **PR flow (established this session):** every change on its own branch off `main`; push;
  `gh pr create`; `gh pr merge <n> --merge`; then sync local main (`git checkout main && git fetch
  origin main && git merge --ff-only origin/main`) and delete the branch (local `-d` + `push origin
  --delete`). The user says "开 PR 合并" per change — they approve merges explicitly.
- **Commit trailer:** end commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  (subagents sometimes emit their own model name — amend if so). PR descriptions end with the
  "Generated with Claude Code" line.
- **SDD:** larger features were built subagent-per-task with a review after each + a whole-branch
  review before merge, using the superpowers `subagent-driven-development` skill. Ledgers were kept
  under `~/code/Glyph/.superpowers/sdd/**` (git-ignored scratch) — useful history if resuming a
  half-done feature.
- **Determinism is sacred:** data generation must stay single-RNG, no torch/global-random; the
  integrity test + fingerprints enforce it. Don't break reproducibility.
- **Tests:** run scoped tests during dev; run the full `-m "not slow"` suite before a merge that
  touches real code (harness/gateway/data). Never run the full suite carelessly — `slow` gates
  real-API/GPU cost.

---

## 8. Doc map (further reading, all on main)

- `docs/data-generation.{en,zh}.md` — stage ①, full detail + config/preset tables + π formula.
- `docs/data-validation.{en,zh}.md` — stage ② + ③: oracles, metrics, arms, protocol, scoring.
- `docs/motivation-real-world-tasks.{en,zh}.md` — the "frontier blind-spot → train specialist" framing + real-task catalog.
- `docs/open_questions.md` — the standing decision/experiment list (incl. the train-arm starvation item, #15 query cap, D-items). `docs/progress.md` — append-only record. `docs/HANDOFF.md` — older (2026-09-15) general handoff.
- `docs/benchmark/*.json` — the 15 frozen instances, 210 candidates, cheap reference ceilings.
- Code: `src/glyph/data/` (generation), `src/glyph/reference/` (oracles + `frozen.py` reuse API), `src/glyph/v2/` (protocol harness: `harness.py`, `session.py`, `mcp.py`, `prompts.py`, `gateway.py`, `student.py`, `export.py`, `report.py`, `cli.py`), `src/glyph/train/` (`sft.py`, `infer.py`), `tools/run_reference.py`, `glyph-viewer/index.html`.

---

*Handoff written 2026-09-29. Main @ 77ee46e. Ask the user before any merge/push or any [GPU]/[API]
run — those cost money and the user gates them.*
