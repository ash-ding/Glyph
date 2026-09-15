"""Wrap the Task-10 tool handlers (glyph.v2.tools.t_*) as in-process SDK MCP
tools.

register_glyph_server(session, trace) builds a claude_agent_sdk in-process
MCP server whose every tool is a thin async shim around the matching plain
handler t_<name>(session, **args). The handler already does all the state
work (charging, phase/arm gating, counters, echoing remaining fields); this
layer only:
  * adapts the SDK single-args calling convention to t_<name>,
  * emits a "tool" trace event, and
  * packs the handler dict result into the SDK content envelope.

IMPORTANT: claude_agent_sdk is imported INSIDE register_glyph_server so this
module (and harness) is importable without the SDK installed -- the harness
unit test drives the state machine with a fake client and never touches the
SDK.
"""

from __future__ import annotations

import json

from glyph.v2 import tools as _T

# name -> (handler, input-schema). The schema is the SDK @tool dict form
# {arg: python-type}; the handler is called as handler(session, **args) so the
# schema keys must match the handler keyword parameters exactly.
_TOOL_SPECS = {
    "query": (_T.t_query, {"exprs": list, "why": str}),
    "submit": (_T.t_submit, {"path": str}),
    "check_answers": (_T.t_check_answers, {"path": str, "set": str}),
    "finish_practice": (_T.t_finish_practice, {"reason": str}),
    "build_dataset": (_T.t_build_dataset, {"path": str}),
    "train": (_T.t_train, {"dataset_id": str, "epochs": int, "lr": float}),
    "student_infer": (_T.t_student_infer, {
        "checkpoint": str, "input_path": str,
        "output_path": str, "prefix_path": str,
    }),
    "final_answer": (_T.t_final_answer, {"path": str}),
}

_DESCRIPTIONS = {
    "query": "Spend a query to evaluate one or more expressions.",
    "submit": "Submit validation answers (practice phase); returns an aggregate score.",
    "check_answers": "Check an answer file for legality (set='validation' or 'test').",
    "finish_practice": "Signal that the practice phase is done.",
    "build_dataset": "Build a student training dataset from purchased queries (train arm).",
    "train": "Fine-tune the student model on a dataset (train arm).",
    "student_infer": "Run inference with a trained student checkpoint (train arm).",
    "final_answer": "Commit the final held-out-test answers and end the run.",
}


def _make_tool(sdk_tool, name, handler, schema, session, trace):
    """Return an SdkMcpTool wrapping handler for session."""

    async def _shim(args):
        result = handler(session, **args)
        trace.emit(
            "tool", tool=name, args=args, result=result,
            turns=session.turns, spent_usd=session.ledger.spent_usd,
        )
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    _shim.__name__ = f"tool_{name}"
    return sdk_tool(name, _DESCRIPTIONS[name], schema)(_shim)


def register_glyph_server(session, trace):
    """Build and return the in-process "glyph" MCP server for this run.

    Imports claude_agent_sdk lazily so importing this module does not require
    the SDK (the harness unit test runs SDK-free).
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool as sdk_tool

    tools = [
        _make_tool(sdk_tool, name, handler, schema, session, trace)
        for name, (handler, schema) in _TOOL_SPECS.items()
    ]
    return create_sdk_mcp_server(name="glyph", tools=tools)
