"""Configuration for Glyph instance generation.

Every knob that shapes an instance lives here.  Nothing in this file is
random; the randomness all comes from `instance_seed` at generation time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


# --- D8 (settled, option A): the surface form of a value spells out its
# digits, so the digit structure survives tokenisation.  The notation is
# `letter_sep` -- each digit as a letter, `v_k_e_e` -- and the reason is
# neutrality rather than cost.
#
# Measured on the student's tokenizer over 700 values at 17**3:
#
#     underscore  v_10_4_4   8.24 tokens   max 10   variable width
#     bracket     v[10,4,4]  9.24          max 11   variable
#     flat        v1234      4.77          max  5   variable, erases the digits
#     letter_sep  v_k_e_e    4.00          max  4   FIXED
#
# `flat` is out on design grounds: it erases the digit structure that is the
# only reason an unqueried table entry can be extrapolated.  Between the other
# three, much of `underscore`'s 8.24 is a tokenizer accident -- base 17 has
# two-character digits 10..16 -- so a value's token length correlates with its
# digit values, which is a weak leak and an unpredictable output length, and
# output length is the soil the truncation bug grew in.  `letter_sep` tracks
# the information content instead: three digits, three tokens, plus a prefix.
#
# The switch is not neutral between arms and was made knowing that. A2's and
# A0''s costs are token-driven and both roughly halve; A4's and A6's do not
# move.  The argument for taking it anyway is that an arm paying double
# because we chose underscores is a measurement artifact rather than a
# property of holding capability in context.
#
# Constraint: `letter_sep` needs base <= 26 (grammar._need_letters).
VALUE_FORMS = ("underscore", "bracket", "flat", "letter_sep")


@dataclass(frozen=True)
class GlyphConfig:
    # ---- value space -------------------------------------------------
    base: int = 17              # digits per position
    n_digits: int = 3           # 17**3 = 4913 values
    d_digit: int = 16           # embedding dims per digit position
    value_form: str = "letter_sep"   # see the note above VALUE_FORMS

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
    # Unary operators are built the same way as binary ones: one small MLP per
    # digit position, plus a weak global coupling term at this weight.  `None`
    # restores the original single joint MLP over the whole embedding.
    #
    # The joint form was never a decision.  D2 factorised the *binary* tables
    # because 24M entries leave no choice; unary has 4913 and simply never got
    # the same treatment.  Self-check #5 then found that this made unary the
    # **harder** half to extrapolate -- reach 0.158 against binary's 0.734 --
    # inverting what the spec assumed.  So the asymmetry was an omission whose
    # sign turned out to be backwards, not a trade-off anyone chose.
    #
    # Why the value matters, and why 0.25 is provisional.  Coupling sets how
    # far the table sits from being decomposable into per-digit parts, and that
    # one number moves both halves of the design in opposite directions:
    #
    #   too low   the parts can be enumerated into a lookup table, and the code
    #             arm takes the win the weights arm was supposed to have
    #   too high  nobody learns it, and the pi->0 end has no winner at all
    #
    # Measured on a least-squares fit over the parts (an upper bound: it sees
    # the raw output vector and 8000 observations, neither of which an agent
    # gets):
    #
    #     coupling   parts model R^2   unseen-entry accuracy
    #        0.0          1.000              1.000
    #        0.1          0.999              0.922
    #        0.25         0.995              0.821
    #        0.5          0.982              0.680
    #        1.0          0.956              0.524
    #        None         0.911              0.427   <- the joint form
    #
    # 0.25 is carried over from binary and is not yet justified for unary,
    # because the two have very different part counts: binary's parts are
    # per-position *pairs*, 3 x 17^2 = 867, while unary's are single digits,
    # 3 x 17 = 51.  Sixteen times fewer parts means enumeration is that much
    # cheaper, so the same coupling does not buy the same resistance.  Under a
    # realistic threat model -- decoded symbols only, N purchased queries --
    # binary at 0.25 saturates at 0.514 and needs ~2000 queries to get there.
    # The unary equivalent has not been measured.
    unary_coupling: float | None = 0.25

    # How an MLP's real-valued output becomes a legal symbol again.
    #
    # `nearest` was the original and it collapses. tanh holds outputs near the
    # centre of the space while the embeddings are Gaussian, so the few
    # embeddings nearest the centroid win almost every argmin: on pi_low, 4913
    # inputs land on 222 distinct outputs with one taking 18.7%. That makes
    # |V| a fiction, and "the table cannot be bought outright" goes with it.
    #
    # `whiten` puts the MLP output on the embeddings' own per-dimension scale
    # before asking which is nearest. Across pi_low and pi_mid, two seeds
    # each: distinct outputs 160-224 -> 1132-1265, top output 15-30% -> 0.8-2%,
    # and the neighbour-vs-random structure contrast roughly doubles (1.4x ->
    # 3.0x). It works for binary as well, which is what settles it.
    #
    # `per_digit` is kept only to be refuted: with concatenated embeddings and
    # every digit combination legal, squared distance decomposes and global
    # argmin *is* per-digit argmin. It is the same function, and measured
    # identical to `nearest` on every preset, seed and operator.
    #
    # `assign` gives a true bijection but cannot exist for binary -- it needs
    # all 24M outputs materialised -- so it is a bound, not an option.
    decode: str = "whiten"       # whiten | nearest | per_digit | assign

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
    # pi -> 1 : difficulty lives in the skeleton.  Many structural ops, deep
    # nesting, guards, and expressions that rarely reach for the table.
    #
    # The value space is 17**3, the same as the other two presets. It used to
    # be 8**1 = 8 values, on the reasoning that pi -> 1 needs the tables to be
    # nearly free. That reasoning was wrong: what keeps `L_table` small is
    # `atomic_ratio`, which decides how often an expression touches the table
    # at all, not how many entries the table has. Measured at 17**3 with the
    # same skeleton knobs, median pi is 0.723 against 0.716 at 8**1 -- the same
    # -- with 0.67 table lookups per item against 0.69. Dropping atomic_ratio
    # to 0.02 reaches pi 0.92, higher than the old preset ever did.
    #
    # A small value space was not free. It moved four things that have nothing
    # to do with pi, all of them confounded with the axis:
    #
    #   mode-answer baseline   0.099-0.134  ->  0.002   (an arm scoring 0.035
    #                                                    was below chance)
    #   distinct answers              262   ->  ~7500
    #   surface form of a value      `v_3`  ->  `v_1_2_3`  (3x the tokens, so
    #                                                    A2's per-query cost and
    #                                                    the truncation risk
    #                                                    differed by preset)
    #   unary entries the test set needs 24 ->  ~4000-8000
    #
    # The last one changes what the region above the skeleton ceiling means: at
    # 24 entries the whole table is buyable in a few queries, so that region
    # tested whether the agent bothered. At 17**3 it tests generalisation,
    # which is the question.
    "pi_high": _preset(
        atomic_ratio=0.15, base=17, n_digits=3,
        n_structural=8, max_transform_depth=3, guard_prob=0.9,
        max_expr_depth=5, demo_max_depth=3,
    ),
    "pi_mid": _preset(
        atomic_ratio=0.5, base=17, n_digits=3,
        n_structural=5, max_transform_depth=2, guard_prob=0.5,
    ),
    # pi -> 0 : difficulty lives in the tables.  Large value space,
    # near-trivial structural ops, no guards, no composition.
    #
    # `n_structural` is 3 rather than 2, and the reason is generation rather
    # than difficulty.  The `comp` split requires a held-out (outer, inner)
    # operator pair to appear and the `depth` split forbids every held-out
    # pair, so both are spent from the same small vocabulary.  With two
    # operators there are four Cartesian pairs but only two the grammar can
    # actually build -- s1 returns VAL and nothing accepts a VAL argument, so
    # any pair with s1 inside is a phantom -- and `n_hold` takes one of the
    # four at random.  Half the time it takes a phantom and `comp` can never
    # be filled; a quarter of the time it takes (s0, s0), which is the only
    # route to depth 3, and `depth` can never be filled.
    #
    # Measured: 13 of 20 seeds could not generate, and the seven that did all
    # held out the same pair, so they were not a sample of anything. Enumerating
    # the choices gives 75% unfillable at two operators, 1.2% at three, 0% at
    # four or more. Three seeds 20/20 here.
    #
    # The cost is that pi's low end rises -- median measured pi goes from about
    # 0.15 to about 0.33, because three operators are more skeleton to not
    # know. That is the trade: 0.18 of pi range against a 65% generation
    # failure rate and a survivorship-biased sample.
    "pi_low": _preset(
        atomic_ratio=0.85, base=17, n_digits=3,
        n_structural=3, max_transform_depth=0, guard_prob=0.0,
        max_expr_depth=3, demo_max_depth=2,
    ),
    # tiny config for fast tests
    "smoke": _preset(
        atomic_ratio=0.5, base=5, n_digits=2, d_digit=8,
        n_structural=4, mlp_width=16,
        n_demos=8, n_iid=60, n_comp=25, n_depth=15,
    ),
}
