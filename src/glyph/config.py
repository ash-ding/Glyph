"""Configuration for Glyph instance generation.

Every knob that shapes an instance lives here.  Nothing in this file is
random; the randomness all comes from `instance_seed` at generation time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


# --- D8 (settled, option A): the surface form of a value spells out its
# digits, so the digit structure survives tokenisation.  The exact notation
# is a sub-decision resolved by self-check #6 (scripts/probe_tokenizer.py);
# until that runs, `underscore` is the placeholder default.
VALUE_FORMS = ("underscore", "bracket", "flat")


@dataclass(frozen=True)
class GlyphConfig:
    # ---- value space -------------------------------------------------
    base: int = 17              # digits per position
    n_digits: int = 3           # 17**3 = 4913 values
    d_digit: int = 16           # embedding dims per digit position
    value_form: str = "underscore"

    # ---- skeleton (structural operator semantics) --------------------
    n_structural: int = 5       # how many of s0..s7 are enabled
    max_transform_depth: int = 2   # how deep `<transform> then <transform>` may nest
    guard_prob: float = 0.5     # chance a structural op gets a guard

    # ---- tables (atomic operator semantics) --------------------------
    n_unary: int = 3
    n_binary: int = 2
    mlp_width: int = 64
    mlp_temp: float = 1.0       # pre-activation scale; higher = less smooth
    binary_coupling: float = 0.25   # D2: 0 = purely digit-wise, large = naive 96-dim MLP

    # ---- expression sampling ----------------------------------------
    atomic_ratio: float = 0.5   # primary continuous knob for sweeping pi
    binary_freq: float = 0.35   # how often value-producing (fold) ops are chosen
    max_expr_depth: int = 4
    demo_max_depth: int = 2     # demos (and iid/comp) stay at or below this
    list_len_range: tuple[int, int] = (2, 4)

    # ---- dataset sizes ----------------------------------------------
    n_demos: int = 30
    n_iid: int = 6500
    n_comp: int = 2300
    n_depth: int = 1200

    @property
    def n_values(self) -> int:
        return self.base ** self.n_digits

    @property
    def d_total(self) -> int:
        return self.n_digits * self.d_digit

    @property
    def n_test(self) -> int:
        return self.n_iid + self.n_comp + self.n_depth

    def with_(self, **kw) -> "GlyphConfig":
        return replace(self, **kw)

    def scaled(self, n_test: int) -> "GlyphConfig":
        """Resize the test set, keeping the 65/23/12 split proportions.

        Generation is cheap (10k items lands in 1.5-11 s depending on preset),
        so this exists for fast test-suite feedback rather than out of
        necessity.  Real runs use the full 10k: the statistical resolution it
        buys -- ~0.5% standard error on a proportion -- is what lets arms be
        separated at the 1-2% level.
        """
        return replace(self, n_iid=round(n_test * 0.65),
                       n_comp=round(n_test * 0.23),
                       n_depth=round(n_test * 0.12))


# ---------------------------------------------------------------------
# Presets.  The names carry the *intended* pi; the real value is whatever
# `instance.measured_pi()` reports.  Phase-diagram axes use the measured
# value, never the preset name.
# ---------------------------------------------------------------------
def _preset(atomic_ratio: float, **kw) -> GlyphConfig:
    return GlyphConfig(atomic_ratio=atomic_ratio, **kw)


PRESETS: dict[str, GlyphConfig] = {
    # pi -> 1 : difficulty lives in the skeleton.  Small value space (the
    # tables are almost free), many structural ops, deep nesting, guards.
    "pi_high": _preset(
        atomic_ratio=0.15, base=8, n_digits=1, d_digit=24,
        n_structural=8, max_transform_depth=3, guard_prob=0.9,
        max_expr_depth=5, demo_max_depth=3,
    ),
    "pi_mid": _preset(
        atomic_ratio=0.5, base=17, n_digits=3,
        n_structural=5, max_transform_depth=2, guard_prob=0.5,
    ),
    # pi -> 0 : difficulty lives in the tables.  Large value space, two
    # near-trivial structural ops, no guards, no composition.
    "pi_low": _preset(
        atomic_ratio=0.85, base=17, n_digits=3,
        n_structural=2, max_transform_depth=0, guard_prob=0.0,
        max_expr_depth=3, demo_max_depth=2,
    ),
    # tiny config for fast tests
    "smoke": _preset(
        atomic_ratio=0.5, base=5, n_digits=2, d_digit=8,
        n_structural=4, mlp_width=16,
        n_demos=8, n_iid=60, n_comp=25, n_depth=15,
    ),
}
