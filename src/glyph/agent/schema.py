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


def tool_defs(allowed: set[Container], *, can_query: bool = True,
              strict: bool = False) -> list[dict]:
    """The tool surface for one arm.

    An arm is defined by which containers it may spend on, and that is
    enforced here rather than by asking the model nicely: A2 is simply not
    given `train`.  Handing over every tool and instructing it to avoid some
    would leave the arm boundary depending on compliance.

    `strict` defaults to False because this Vertex project forbids it; see the
    module docstring.
    """
    tools: list[dict] = []

    if can_query:
        tools.append({
            "name": "query_oracle",
            "description": (
                "Ask the hidden interpreter to evaluate expressions. This is "
                "the only way to learn the semantics. Every expression costs "
                "budget, and malformed ones cost the same as well-formed ones."),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "exprs": {"type": "array", "items": {"type": "string"},
                              "maxItems": 256,
                              "description": "Well-formed expressions to evaluate."},
                    "why": {"type": "string",
                            "description": "What you expect to learn. Recorded, not acted on."},
                },
                "required": ["exprs", "why"],
                "additionalProperties": False,
            },
        })

    if Container.WEIGHTS in allowed:
        tools += [
            {
                "name": "declare_target",
                "description": (
                    "State what you are putting into the student's weights. "
                    "Required before train, and it cannot be changed afterwards."),
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string",
                                 "enum": [r.value for r in Role]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["role", "rationale"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "synthesize_data",
                "description": (
                    "Build a training set from what you have queried. Costs "
                    "budget in proportion to size."),
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "n": {"type": "integer", "minimum": 1, "maximum": 200000},
                        "source": {"type": "string",
                                   "enum": ["queries", "demos", "mixture"]},
                        "emphasis": {"type": "string",
                                     "enum": ["uniform", "tail", "operator-focused"]},
                        "include_reasoning": {"type": "boolean"},
                    },
                    "required": ["description", "n", "source", "emphasis",
                                 "include_reasoning"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "train",
                "description": (
                    "Fine-tune the student on a dataset you built. Costs GPU "
                    "seconds, measured not estimated."),
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "lora_rank": {"type": "integer", "minimum": 4, "maximum": 128},
                        "epochs": {"type": "integer", "minimum": 1, "maximum": 4},
                        "lr": {"type": "number"},
                    },
                    "required": ["dataset_id", "lora_rank", "epochs", "lr"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "evaluate",
                "description": (
                    "Score a checkpoint on the dev split. The test split is "
                    "not reachable from here."),
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {"checkpoint_id": {"type": "string"},
                                   "n": {"type": "integer", "minimum": 16, "maximum": 2000}},
                    "required": ["checkpoint_id", "n"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "inspect_failures",
                "description": "Look at items a checkpoint got wrong on dev.",
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {"checkpoint_id": {"type": "string"},
                                   "k": {"type": "integer", "minimum": 1, "maximum": 50}},
                    "required": ["checkpoint_id", "k"],
                    "additionalProperties": False,
                },
            },
        ]

    if Container.CODE in allowed:
        tools.append({
            "name": "write_code",
            "description": (
                "Submit a Python solver defining solve(expr) -> str. It runs "
                "sandboxed with no network. Costs the same budget pool as "
                "training."),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "check_on": {"type": "integer", "minimum": 0, "maximum": 500,
                                 "description": "Dev items to try it on before sealing."},
                },
                "required": ["src", "check_on"],
                "additionalProperties": False,
            },
        })

    if Container.CONTEXT in allowed:
        tools.append({
            "name": "set_context",
            "description": (
                "Fix the prompt prefix the student will carry at test time. "
                "Its token count is recorded and re-paid on every test query."),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        })

    tools.append({
        "name": "seal",
        "description": (
            "End the prepare phase and freeze what answers the test set. "
            "Nothing is reachable afterwards -- no oracle, no training, no you."),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "entry": {"type": "string", "enum": ["program", "model"],
                          "description": "What drives the test loop."},
                "checkpoint_id": {"type": "string",
                                  "description": "Empty string if none."},
                "summary": {"type": "string",
                            "description": "What you taught it, in your own words. "
                                           "Used by the verbalization-gap arm."},
            },
            "required": ["entry", "checkpoint_id", "summary"],
            "additionalProperties": False,
        },
    })

    if not strict:
        for t in tools:
            t.pop("strict", None)
    return tools
