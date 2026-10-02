# Depth-0 bare atomics + per-op oracle v2 — design

Date: 2026-10-01. Status: approved in discussion; implementation goes to review
BEFORE any GPU run.

## A. `depth(AtomApp) = 0`

`depth` means "levels of structural-operator nesting". A bare atomic
application contains none, so its depth is 0 (it currently returns 1 via the
`App` fallthrough). Depth 0 then covers `Val`/`Lit` (no computation) and
`AtomApp` (one table lookup); nothing in the code relies on
"depth 0 ⇒ no computation".

- `grammar.depth`: explicit `AtomApp → 0` branch; the PR-A test assertion
  moves from 1 to 0.
- `syntax_spec` gains one form-only line: a bare atomic application has
  nesting depth 0; the nesting cap counts structural operators.
- Docs record two invariants discovered along the way:
  - **root-only**: AtomApp can never appear inside another expression — the
    parser only accepts value literals as its arguments, and every structural
    slot is LIST-typed while AtomApp is VAL-typed.
  - **probes are the language's depth-0 stratum**, administered separately
    (frozen test/val/demos stay structural; depth-0 items would mechanically
    deflate measured π because the table oracle gets them right for free).
- Seen-semantics notes for `data-validation`: the agent-side `seen` split is
  LOG-based and therefore an upper bound on knowledge (a composed query marks
  every touched cell seen although the agent only observes the final output);
  `unseen` is the clean lower-bound generalization read. Demos are computed at
  generation time and never enter `query_log`, so demo-touched cells count as
  unseen.

## B. Per-op oracle v2: two coverage knobs, budget-aware training

The current per-op path (`train_student(ops=[op])`, fixed 6000 steps) has two
problems, both measured on 2026-10-01:

1. **Fixed compute × wildly different data.** At `seen_frac` the eligible pool
   is ~98–491 cells for a unary op but 0.48M–2.4M for a binary op; 768k
   stream samples mean a unary specialist trains ~1,500–7,800 effective epochs
   (over-training that may DEPRESS the unseen ceiling), while a binary
   specialist sees most "eligible" cells zero times (eligibility ≠ exposure).
2. **frac answers only one of the two questions.** Fraction-of-own-table is
   the right knob for "how learnable is this table intrinsically" (and stays
   consistent with the generalist weights oracle and a0prime's coverage axis).
   But the agent-relevant question — "what can a specialist reach within a
   budget an agent could actually buy (Q=1000)" — needs an ABSOLUTE knob: the
   same N cells for every op.

### Decisions

- **Both knobs coexist**: `--seen-frac` (existing semantics, per-op-table
  fraction via the frozen hash) and `--n-seen` (exact N cells per op, same N
  for unary and binary). One `weights-per-op` invocation may sweep both;
  payload keys are `"frac=0.05"` / `"n=1000"`.
- **Eligibility construction**:
  - frac mode: the existing `seen_u`/`seen_b` hash (unchanged, so frac-mode
    eligibility stays comparable with the generalist oracle).
  - n mode: exactly N distinct cells drawn from a dedicated RNG stream
    `default_rng((inst.seed, op_index, N_SEEN_SALT))` — deterministic per
    (instance, op, N), fingerprint-exempt, probe-style.
- **`train_specialist(inst, op, *, seen_frac | n_seen, model, lr, batch,
  max_epochs, max_steps, patience, holdout_frac)`** replaces the
  `train_student(ops=...)` path (which is removed — one path, no drift):
  - materializes the eligible cells (unary: filter/draw over 4913; binary
    frac-mode: vectorized hash over the 4913² key space; binary n-mode: draw);
  - holds out `holdout_frac` (default 0.1, min 8, max 1024) of them as an
    early-stop set, NEVER trained;
  - epoch training over the materialized set (shuffle per epoch), evaluating
    holdout LOSS on a cadence (every epoch for small sets, every
    `eval_every_steps` for large); early-stops on `patience` misses and
    restores the best state; hard caps `max_epochs` and `max_steps`;
  - returns `(model, tok, record)` where record carries `n_eligible, n_train,
    n_holdout, steps, epochs, stopped_by, best_holdout_loss` **and the exact
    train/holdout cell sets**.
- **Exposure-exact scoring**: `score_probes` now takes explicit seen-cell sets
  (the actual TRAIN split) instead of `(seen_frac, hash)`. Cells in the
  early-stop holdout count as unseen (never trained on — honest). The
  eligibility≠exposure caveat disappears by construction: every train-split
  cell is visited every epoch.
- **The generalist weights oracle is untouched** (6000-step protocol keeps
  comparability with the 2026-09-29 re-measurement).
- **No GPU run in this change.** The first v2 per-op sweep is a separate,
  user-gated execution after implementation review.

### Testing (CPU; the training loop itself stays [GPU] like train_student)

- depth: AtomApp depth 0; spec line present; leak gate green; frozen
  integrity untouched.
- eligibility: n-mode returns exactly N distinct in-range cells,
  deterministic per (instance, op, N), different across ops; frac-mode counts
  match the hash within tolerance.
- holdout split: disjoint, sized per the rule, deterministic.
- `score_probes` with explicit sets: perfect-student 1.0; seen/unseen n's
  follow the provided sets; holdout-as-unseen honored.
- fail-fast: unknown op name raises ValueError before any torch import.
- `run_reference` arg plumbing: `--n-seen` parsed, keys rendered as
  `"n=1000"`.

## C. Structural probes (added later the same day)

Pure-structure ops (shapes L and KL) can be isolated: a single application
`op([…])` / `op(k, […])` is a legal depth-1 expression whose answer depends
only on the skeleton. UL/LB ops cannot be isolated — any expression containing
them entangles the tables — and stay diagnosed indirectly via comp/depth.

- `probe_set` gains structural items: for every enabled L/KL op,
  `struct_reps` (default 2) items per BEHAVIOR cell. Values are mere cargo for
  pure-structure ops, so the cells enumerate what transforms/guards actually
  branch on: list length {2,3,4} × variant {all-distinct, first==last,
  internal-duplicate (len ≥ 3)} × k {1..3} (KL only) — 8 cells for an L op,
  24 for a KL op.
- Items colliding with test/val sources are redrawn: unlike bare atomics,
  single-application structural items CAN coincide with depth-1 iid items,
  and test/val must stay sealed.
- `split` = op name; `needs_u`/`needs_b` empty; same probe RNG stream (the
  atomic prefix of the set is unchanged byte-for-byte).
- Report: seen/unseen does not apply (structural knowledge is a rule, not
  cells; `is_tail` would classify everything as seen). `by_op` entries carry
  `kind: "atomic" | "structural"`, with seen/unseen null for structural ops.
  An exposure covariate (how often the op appeared in demos / purchased
  queries) is noted as future work.
- Reference side unchanged: `weights-per-op` filters probes by atomic op
  names, so structural probes never reach the specialists; their reference
  ceiling is trivially 1.0 (the skeleton oracle knows every rule).
- Positioning fix: the probe set is now the SINGLE-OPERATOR ISOLATION
  stratum — bare atomics at depth 0, pure-structure ops at depth 1 — no
  longer only "the depth-0 stratum".
