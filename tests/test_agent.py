"""The properties the arms' comparability rests on.

None of these need a GPU or the API. They are the invariants that, if they
broke silently, would make a difference between arms look like a difference
between containers.
"""
import json

import pytest

from glyph.agent.schema import Container, Role, tool_defs, validate
from glyph.agent.tools import ToolBox
from glyph.budget import BudgetExhausted, Ledger
from glyph.config import PRESETS
from glyph.instance import generate
from glyph.seal import SealedArtifact, score_answers
from glyph.trace import TraceWriter

CFG = PRESETS["smoke"]


@pytest.fixture
def box(tmp_path):
    return ToolBox(inst=generate(7, CFG), ledger=Ledger(total_h100s=1e6),
                   trace=TraceWriter(tmp_path / "t.jsonl"),
                   allowed={Container.WEIGHTS, Container.CODE},
                   work_dir=tmp_path, arm="test")


# -- the arm boundary is the tool list, not an instruction ---------------
def test_each_arm_only_gets_its_own_container_tools():
    ctx = {t["name"] for t in tool_defs({Container.CONTEXT})}
    code = {t["name"] for t in tool_defs({Container.CODE})}
    weights = {t["name"] for t in tool_defs({Container.WEIGHTS})}

    assert "train" not in ctx and "write_code" not in ctx
    assert "train" not in code and "set_context" not in code
    assert "write_code" not in weights and "set_context" not in weights


def test_self_assessment_is_universal_not_a_container_property():
    """Someone writing a prompt can try it, someone writing code can run it,
    someone training a model can hold out a validation set. Issuing it to one
    arm only measures the harness rather than the container -- and the dev
    signal is known to shape behaviour: once it stopped reading zero, one agent
    went from buying 187 facts to 565."""
    for containers in ({Container.CONTEXT}, {Container.CODE}, {Container.WEIGHTS}):
        names = {t["name"] for t in tool_defs(containers)}
        assert {"seal", "query", "evaluate", "inspect"} <= names, containers


def test_the_free_choice_arm_is_an_allocation_not_a_module():
    """A7 is every container at once, which under this structure is one row in
    the allocation table rather than a new arm implementation."""
    all_names = {t["name"] for t in tool_defs(set(Container))}
    assert {"set_context", "write_code", "train"} <= all_names


def test_every_declared_parameter_is_read_by_the_implementation():
    """A declared-and-ignored parameter is worse than a missing one: the agent
    operates a control that is not connected, and the trace records the intent
    as though it had an effect. `synthesize_data` accepted three such."""
    import inspect as _inspect
    from glyph.agent.tools import ToolBox
    from glyph.agent.schema import TOOLS
    for name, spec in TOOLS.items():
        fn = getattr(ToolBox, f"_t_{name}", None)
        assert fn is not None, f"{name} is declared but not implemented"
        params = set(_inspect.signature(fn).parameters) - {"self"}
        declared = set(spec["input_schema"]["properties"])
        assert declared <= params, (name, declared - params)


def test_declaring_a_target_is_only_offered_where_weights_are():
    assert "declare_target" in {t["name"] for t in tool_defs({Container.WEIGHTS})}
    assert "declare_target" not in {t["name"] for t in tool_defs({Container.CODE})}


# -- validation stands in for strict: true -------------------------------
def test_validation_rejects_what_strict_would_have():
    tool = [t for t in tool_defs({Container.WEIGHTS})
            if t["name"] == "declare_target"][0]
    assert validate(tool, {"role": "answer", "rationale": "x"}) is None
    assert "not one of" in validate(tool, {"role": "nonsense", "rationale": "x"})
    assert "missing" in validate(tool, {"role": "answer"})
    assert "unknown" in validate(tool, {"role": "answer", "rationale": "x", "z": 1})


def test_a_role_outside_the_enum_cannot_reach_the_trace():
    """The property Fig. 4 depends on."""
    tool = [t for t in tool_defs({Container.WEIGHTS})
            if t["name"] == "declare_target"][0]
    for bad in ("Answer", "r1", "", "proposal_distribution"):
        assert validate(tool, {"role": bad, "rationale": "x"}) is not None


# -- the oracle is metered, errors included ------------------------------
def test_every_query_is_billed_including_malformed_ones(box):
    """Probing the grammar is not free: a syntax error costs a query.

    Otherwise the public syntax spec is worth less than it is meant to be --
    an agent could map the grammar for nothing and only pay for semantics.
    """
    good = box.inst.demos[0][0]          # a real expression, not a hand-written one
    before = box.ledger.spent_h100s
    out = box.dispatch("query", {"exprs": [good, "not an expr"],
                                        "why": "probing"})
    assert box.ledger.spent_h100s > before
    assert out["malformed"] == 1
    # the malformed one is billed but not banked as knowledge
    assert out["queries_so_far"] == 1


def test_the_budget_stops_the_run(box):
    box.ledger.total = box.ledger.spent_h100s + 1e-9
    with pytest.raises(BudgetExhausted):
        box.dispatch("query", {"exprs": [box.inst.demos[0][0]] * 50,
                                      "why": "overspend"})


# -- dev comes out of what was paid for ----------------------------------
def test_dev_is_carved_from_purchased_queries_not_given_free(box):
    exprs = [e for e, _ in box.inst.demos][:6]
    box.dispatch("query", {"exprs": exprs, "why": "buy"})
    train, dev = box._split()
    assert dev, "a dev slice should exist once something has been bought"
    assert len(train) + len(dev) == len(box.queried)
    assert not set(map(id, train)) & set(map(id, dev))


def test_evaluating_before_buying_anything_is_an_error(box):
    out = box.dispatch("evaluate", {"artifact_id": "nope", "n": 16})
    assert "error" in out


# -- sealing has prerequisites -------------------------------------------
def test_training_without_declaring_a_target_is_refused(box):
    box.datasets["ds1"] = []
    out = box.dispatch("train", {"dataset_id": "ds1", "epochs": 1, "lr": 1e-4})
    assert "declare_target" in out["error"]


def test_a_target_cannot_be_redeclared(box):
    assert box.dispatch("declare_target", {"role": "answer", "rationale": "a"})["ok"]
    assert "error" in box.dispatch("declare_target", {"role": "cot", "rationale": "b"})
    assert box.role is Role.ANSWER


def test_sealing_needs_an_artifact_that_was_actually_made(box):
    """Taking an artifact id rather than an entry plus a checkpoint means the
    thing sealed is the thing that was evaluated -- an agent can no longer
    measure one object and hand over another."""
    assert "error" in box.dispatch("seal", {"artifact_id": "", "summary": "s"})
    made = box.dispatch("write_code", {"src": "def solve(e):\n    return e"})
    assert box.dispatch("seal", {"artifact_id": made["artifact_id"],
                                 "summary": "s"})["sealed"]
    assert box.sealed.program and box.sealed.entry == "program"


# -- scoring ------------------------------------------------------------
def test_scores_are_reported_per_split():
    inst = generate(7, CFG)
    items = inst.test_set()[:20]
    perfect = [t.answer_src for t in items]
    total, by_split = score_answers(items, perfect)
    assert total == 1.0
    assert all(v == 1.0 for v in by_split.values())

    wrong = ["nope"] * len(items)
    total, by_split = score_answers(items, wrong)
    assert total == 0.0


def test_a_context_artifact_declares_its_size():
    a = SealedArtifact(arm="a2", entry="model", context="x" * 500)
    assert a.sizes()["context_chars"] == 500
    b = SealedArtifact(arm="a4", entry="program", program="def solve(e): return e")
    assert b.sizes()["program_bytes"] > 0
