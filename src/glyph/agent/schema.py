"""The agent's decisions, as types rather than prose.

Everything the agent decides crosses this boundary as a validated object, and
the analysis reads those objects.  Nothing downstream parses natural
language: "it chose a proposal distribution" has to be a field in a JSONL
row, not an inference from a paragraph, or the target distribution in Fig. 4
is a summary of our reading rather than of its behaviour.

The plan asks for `strict: true` so the API guarantees argument shape and
this side does no defensive parsing.  **That is not available here**: this
Vertex project's org policy refuses `structured_outputs` for partner models
(`constraints/vertexai.allowedPartnerModelFeatures`), and a request carrying
`strict` comes back 400.  So `strict` is off by default and `validate()`
below does the same job on our side.  The guarantee is weaker -- a malformed
call becomes an error the model sees and retries, rather than something the
API made impossible -- but `declare_target` still cannot be recorded with a
role outside the enum, which is the property the analysis depends on.

Turn `strict` back on the day the policy allows it; the schemas already carry
`additionalProperties: false` and `required`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """The seven delegation targets.  `declare_target` is mandatory before
    training, which is what turns "what did it choose" into a statistic."""
    ANSWER = "answer"              # R1  input -> answer
    COT = "cot"                    # R2  input -> reasoning -> answer
    SUBROUTINE = "subroutine"      # R3  one narrow piece, code keeps the rest
    PROPOSAL = "proposal"          # R4  candidates, a verifier filters
    VALUE = "value"                # R5  scores, does not generate
    WORLD_MODEL = "world_model"    # R6  dynamics, so search can go offline
    INTERFACE = "interface"        # R7  protocol and schema only


class Container(str, Enum):
    CONTEXT = "context"
    CODE = "code"
    WEIGHTS = "weights"


@dataclass
class DataSpec:
    """What the agent wants synthesised.  Deliberately coarse: the levers we
    care about are coverage and filtering, not prompt micro-format."""
    description: str
    n: int
    source: str = "queries"        # queries | demos | mixture
    emphasis: str = "uniform"      # uniform | tail | operator-focused
    include_reasoning: bool = False


@dataclass
class FailureReport:
    n_inspected: int
    examples: list[dict]
    by_split: dict[str, float] = field(default_factory=dict)


def validate(tool: dict, args: dict) -> str | None:
    """Check `args` against a tool's schema.  Returns an error string, or None.

    This is what `strict: true` would have done server-side.  It is
    deliberately blunt: required keys, no unknown keys, enums, and coarse
    types.  Anything subtler belongs in the tool itself, where a bad value is
    a result rather than a protocol violation.
    """
    schema = tool["input_schema"]
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    if schema.get("additionalProperties") is False:
        extra = [k for k in args if k not in props]
        if extra:
            return f"unknown argument(s): {', '.join(extra)}"
    for key, value in args.items():
        spec = props.get(key, {})
        if "enum" in spec and value not in spec["enum"]:
            return (f"{key}={value!r} is not one of "
                    f"{', '.join(map(str, spec['enum']))}")
        want = spec.get("type")
        ok = {"string": str, "integer": int, "number": (int, float),
              "boolean": bool, "array": list, "object": dict}.get(want)
        if ok and not isinstance(value, ok):
            if want == "integer" and isinstance(value, bool):
                return f"{key} must be an integer, not a boolean"
            if not (want == "number" and isinstance(value, (int, float))):
                return f"{key} must be of type {want}"
        if want == "integer" and isinstance(value, int):
            if "minimum" in spec and value < spec["minimum"]:
                return f"{key}={value} is below the minimum {spec['minimum']}"
            if "maximum" in spec and value > spec["maximum"]:
                return f"{key}={value} is above the maximum {spec['maximum']}"
    return None


# ---------------------------------------------------------------------
# Tool declarations.
#
# Declared once, here, and allocated to arms further down.  Those used to be
# the same act -- the list was built inside `if Container.X in allowed:`
# blocks -- so a tool belonged to an arm because of where it happened to be
# typed.  `evaluate` was weights-only for exactly that reason; nobody decided
# it.  The arms are meant to differ *only* in which container they may spend
# on, and keeping declaration apart from allocation makes that readable in one
# table instead of spread through a function body.
#
# Every parameter here is read by the implementation.  A declared-and-ignored
# parameter is worse than a missing one: the agent operates a control that is
# not connected and the trace records the intent as though it had an effect.
# ---------------------------------------------------------------------
def _t(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": props,
                             "required": required,
                             "additionalProperties": False}}


TOOLS: dict[str, dict] = {
    "query": _t(
        "query",
        "Ask the hidden interpreter to evaluate expressions. This is the only "
        "way to learn the semantics. Every expression costs budget, and "
        "malformed ones cost the same as well-formed ones.",
        {"exprs": {"type": "array", "items": {"type": "string"}, "maxItems": 256,
                   "description": "Well-formed expressions to evaluate."},
         "why": {"type": "string",
                 "description": "What you expect to learn. Recorded, not acted on."}},
        ["exprs", "why"]),

    "build_dataset": _t(
        "build_dataset",
        "Build a training set from what you have queried. It repeats the rows "
        "you bought -- it cannot invent labels you do not have -- so `n` is "
        "effectively an epoch count.",
        {"source": {"type": "string", "enum": ["queries", "demos", "mixture"]},
         "n": {"type": "integer", "minimum": 1, "maximum": 200000}},
        ["source", "n"]),

    "declare_target": _t(
        "declare_target",
        "State what you are putting into the student's weights. Required "
        "before train. NOTE: this is a declaration only -- every role trains "
        "identically at present.",
        {"role": {"type": "string", "enum": [r.value for r in Role]},
         "rationale": {"type": "string"}},
        ["role", "rationale"]),

    "train": _t(
        "train",
        "Fine-tune the student on a dataset you built. Always a full "
        "fine-tune. Costs GPU seconds, measured not estimated.",
        {"dataset_id": {"type": "string"},
         "epochs": {"type": "integer", "minimum": 1, "maximum": 4},
         "lr": {"type": "number"}},
        ["dataset_id", "epochs", "lr"]),

    "write_code": _t(
        "write_code",
        "Submit a Python solver defining solve(expr) -> str. It runs sandboxed "
        "with no network. Use `evaluate` to check it.",
        {"src": {"type": "string"}},
        ["src"]),

    "set_context": _t(
        "set_context",
        "Fix the prompt prefix the student will carry at test time. Its token "
        "count is recorded and re-paid on every test query.",
        {"text": {"type": "string"}},
        ["text"]),

    "evaluate": _t(
        "evaluate",
        "Score any artifact you have made on your dev split. The test split is "
        "not reachable from here.",
        {"artifact_id": {"type": "string"},
         "n": {"type": "integer", "minimum": 16, "maximum": 2000}},
        ["artifact_id", "n"]),

    "inspect": _t(
        "inspect",
        "Look at items an artifact got wrong on dev.",
        {"artifact_id": {"type": "string"},
         "k": {"type": "integer", "minimum": 1, "maximum": 50}},
        ["artifact_id", "k"]),

    "seal": _t(
        "seal",
        "End the prepare phase and freeze one artifact as what answers the "
        "test set. Nothing is reachable afterwards -- no oracle, no further "
        "training, no you.",
        {"artifact_id": {"type": "string",
                         "description": "From set_context, write_code or train."},
         "summary": {"type": "string",
                     "description": "What you taught it, in your own words."}},
        ["artifact_id", "summary"]),
}

# ---------------------------------------------------------------------
# Allocation.  Which arm gets which tool, as a table rather than as control
# flow inside the declarations.
#
# `evaluate` and `inspect` are universal on purpose.  Self-assessment is not a
# property of a container -- someone writing a prompt can try it, someone
# writing code can run it, someone training a model can hold out a validation
# set.  Issuing it to one arm only measures the harness rather than the
# container, and the dev signal is known to shape behaviour strongly: once it
# stopped reading zero, one agent went from buying 187 facts to 565.
# ---------------------------------------------------------------------
UNIVERSAL: tuple[str, ...] = ("query", "evaluate", "inspect", "seal")

BY_CONTAINER: dict[Container, tuple[str, ...]] = {
    Container.CONTEXT: ("set_context",),
    Container.CODE: ("write_code",),
    Container.WEIGHTS: ("declare_target", "build_dataset", "train"),
}

# Presentation order only; membership is decided above.
ORDER: tuple[str, ...] = (
    "query", "declare_target", "build_dataset", "train",
    "write_code", "set_context", "evaluate", "inspect", "seal")


def allocate(allowed: set[Container], *, can_query: bool = True) -> set[str]:
    """The tool names one arm may call."""
    names = set(UNIVERSAL)
    if not can_query:
        names.discard("query")
    for c in allowed:
        names |= set(BY_CONTAINER[c])
    return names


def tool_defs(allowed: set[Container], *, can_query: bool = True,
              strict: bool = False) -> list[dict]:
    """The tool surface for one arm: declarations filtered by allocation.

    `strict` defaults to False because this Vertex project forbids it; see the
    module docstring.
    """
    names = allocate(allowed, can_query=can_query)
    out = [dict(TOOLS[n]) for n in ORDER if n in names]
    if not strict:
        for t in out:
            t.pop("strict", None)
    return out
