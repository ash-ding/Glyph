# Pilot calibration notes

Task 20: one full-size run per arm to set the caps for the grid. Each entry
carries the command, the config, the numbers, and where it ran.

## no_train / pi_mid / 1001 — 2026-09-15, lumen1

```
python -m glyph.v2 run --arm no_train --preset pi_mid --seed 1001
```
Config (Global-Constraints defaults): model `claude-opus-4-8`, `effort=high`,
`Q=1000`, `submit_cap=20`, `T_p=100`, `T_f=30`, `n_val=5000`, `usd_line=300`,
`max_turns=200`.

| metric | value |
|---|---|
| overall | **0.4015** |
| by split (iid / comp / depth) | 0.410 / 0.486 / **0.196** |
| headroom (overall / tail) | 0.203 / 0.199 |
| q_used | **1000 (= Q cap)** |
| submissions (legal / attempts) | 7 / 11 |
| turns_practice / turns_final | **421** / 8 |
| spent | **$22.46** (output 10.09, cache_read 8.31, cache_write 4.02, input 0.04) |
| wall time | 90 min (5429 s) |
| final_commit / run_status | agent / completed |

### What bound, and what did not

- **`Q=1000` BOUND.** The agent used every query (37 batched `query` calls →
  1000), then **voluntarily called `finish_practice`** and committed. Q is the
  effective practice budget and the coverage lever.
- **`T_p=100` did NOT bind — and cannot, as implemented.** The real SDK streams
  the whole agent loop as *one* `receive_response`, so `drive_practice`'s
  `turns >= T_p` check runs only once, *after* the entire loop. The trace
  confirms it: every tool call records `session.turns = 0` (turns are added only
  after the loop returns), and `turns_practice = 421` lands in a single step.
  Practice was ~185 model calls in that one loop.
- **The SDK `max_turns=200` was the real turn backstop** (~185 practice model
  calls, just under 200) — but the agent self-terminated first, so it did not
  bind either.
- **`T_f=30`, `submit_cap=20`, `usd_line=300` were all generous** — final used 8
  turns, 7 legal submits, and $22.46 of a $300 line.
- **Gating verified in a live run.** `build_dataset` was correctly refused:
  `"tool 'build_dataset' not available in phase practice for arm no_train"`.

### Cost

A no_train pi_mid run ≈ **$22, 90 min**, no GPU. The final phase is ~$0.4 (just
the commit; `finish_practice` at $22.07 → `final_answer` at $22.46). Output
tokens dominate under `effort=high`; caching keeps input near zero.

### Calibration decisions

1. **Q is the swept coverage axis — keep it as the lever.** no_train wants the
   full 1000 at pi_mid. The grid sweeps Q; the arm comparison is at matched Q.
2. **`T_p` is not a hard turn cap under the single-`receive_response` loop.**
   Two options: **(a)** accept `Q` + the SDK `max_turns` as the real practice
   bounds and treat `T_p` as a coarse between-loop backstop (document it as
   such); **(b)** enforce a hard per-turn practice budget mid-loop via
   `interrupt()` — the mechanism `submit_cap` already uses. **Recommend (a)**:
   `Q` is the meaningful budget and v2 already made it the coverage knob.
   Revisit only if turn-parity between arms turns out to matter. *(deferred
   decision — flagged)*
3. **`max_turns=200` → keep** as the practice backstop (sits just above the ~185
   used). Raise toward ~300 only if a richer instance or the train arm needs it.
4. **`T_f=30` → keep** (8 used; generous but cheap).
5. **`usd_line=300` → keep** as the grid safety line (the train arm and harder
   seeds will cost more; it is a backstop, not a target).
6. **`submit_cap=20` → keep** (7 used).

### Open for the train-arm pilot

GPU cost and time, whether the agent actually trains a student, and whether its
turn/cost profile differs enough to warrant different caps.

## train / pi_mid / 1001 — 2026-09-15/16, lumen1 (gpu0)

```
python -m glyph.v2 run --arm train --preset pi_mid --seed 1001
```
Same config as the no_train run (`effort=high`, `Q=1000`, `T_p=100`, `T_f=30`,
`n_val=5000`, `usd_line=300`, `max_turns=200`), plus `student_model =
Qwen/Qwen3-1.7B` on `CUDA_VISIBLE_DEVICES=0`.

| metric | value |
|---|---|
| overall | **0.4197** |
| by split (iid / comp / depth) | 0.424 / 0.508 / **0.228** |
| headroom (overall / tail) | 0.228 / 0.225 |
| q_used | **1000 (= Q cap)** |
| submissions (legal) | 9 |
| turns_practice / turns_final | 449 / 7 |
| spent | **$25.71** (output 11.10, cache_read 9.55, cache_write 4.89, input 0.18) |
| **gpu_seconds** | **train 35.8 + infer 211.8 = 247.6** (~4 GPU-min) |
| wall time | 96 min (5768 s) |
| final_commit / final_from_student | agent / **null** (see caveat) |

### train vs no_train (pi_mid / 1001, paired on the instance)

| | no_train | train | Δ |
|---|---|---|---|
| overall | 0.4015 | 0.4197 | **+0.018** |
| iid / comp / depth | .41 / .49 / .20 | .42 / .51 / .23 | small, all splits |
| q_used | 1000 | 1000 | Q bound both |
| turns_practice | 421 | 449 | — |
| spent | $22.46 | $25.71 | +$3.25 |
| GPU | none | 247.6 s | — |

**This instance is not a test of whether training helps.** Its measured
**π = 0.44** (skeleton 44% / table 56% of the difficulty; skeleton ceiling
0.249 vs table ceiling 0.305) is near the middle, only marginally table-leaning.
Training's regime is **low π** (table-dominated). At π≈0.44 neither arm is
strongly advantaged, so the **+0.018 near-tie is expected and not diagnostic**;
and it is one seed (single-arm SE is large). The arm comparison belongs on
`pi_low` and swept across π, paired over seeds 1001–1005 (spec §10.4).

### What the agent did with the student (the point of running train)

The train-arm machinery all fired in a real run:

- **`build_dataset` ×5.** The first two pointed at the query log
  (`work/task/queries.jsonl`) and returned **0 examples** — the log is
  `{expr, out}` but `build_dataset` expects `{expr, answer}`, so nothing
  parsed. The agent then built from `demos.jsonl` (ds3, 30 rows) and its own
  assembled `train_data.jsonl` (ds5, 160 rows = 30 demos + 130 reformatted).
  *(Friction worth noting: purchased query results cannot be fed to
  `build_dataset` directly; the key names differ.)*
- **`train` ×2**, iterating on data and hyperparameters:

  | call | dataset | epochs | lr | size | final_loss | gpu_s |
  |---|---|---|---|---|---|---|
  | #1 | ds3 (demos) | 3 | 1e-3 | 30 | 7.16 (bad) | 20.1 |
  | #2 | ds5 (160) | 10 | 5e-4 | 160 | **1.73** | 15.7 |

- **`student_infer` ×4** to probe the checkpoints. Output quality was poor:
  ck1 truncated **5/5** rows; ck2 on a 100-row probe truncated **95/100**. The
  student was rambling past the answer, not producing clean values.
- **`final_answer`** committed `work/test_out.jsonl` — **not** any
  `student_infer` output path — so the committed answers were the agent's own,
  not the student's raw output.

**Reading:** even given the student, the agent could not get a usable model out
of it at this scale — tiny datasets (30–160 rows), few epochs (3–10), and heavily
truncated inference. So the train arm's 0.42 reflects the **frontier's** answers
(having spent some turns and ~4 GPU-min training a student it did not commit
from), not a weights win. Whether a low-π instance, more student data, or better
inference formatting changes this is the real train-arm question — untouched by
this calibration run.

### `final_from_student` caveat

`final_from_student` reads `null` here, but that is **not** evidence the agent
avoided the student: the covariate is never set (unwired since commit
`2fba9aa`) and its verbatim-identity definition would read ~0 anyway when the
student output has to be cleaned. See the open question of the same name.

### Calibration deltas from no_train

- **GPU is cheap and well under caps.** 247.6 GPU-s total (train 35.8 / call,
  infer 211.8) vs the caps of 1800 s/call and 7200 s/practice — no pressure.
  Infer dominates (vLLM engine load), not training.
- **Cost ≈ +15%** over no_train ($25.71 vs $22.46) — the build/train/infer turns
  add frontier calls; GPU itself is a rounding error in USD.
- **All other caps identical to no_train**: Q binds, `T_p` does not, `T_f` /
  `submit_cap` / `usd_line` generous.
