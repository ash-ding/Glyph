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
