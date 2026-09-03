"""The agent loop, written by hand.

The SDK's tool runner would drive this loop, but four things have to happen
on every call and the fourth needs the loop itself: charge the ledger, write
the trace, stop when the budget is gone, and **replay from cache instead of
re-billing**.  The oracle sweep and the regret decomposition re-run the same
(arm, preset, seed, budget) hundreds of times, and they are only affordable
if a repeat costs nothing.  That is a property of the loop, not a hook.

Cost accounting follows the real bill: cached prompt reads at 0.1x, cache
writes at 1.25x, so caching that saves money in practice saves it in the
ledger too, and the crossover is denominated in something that matches
reality.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..budget import BudgetExhausted, Ledger
from ..data.instance import GlyphInstance
from ..trace import ResponseCache, TraceWriter
from ..vertex import EFFORT, TEACHER, client, request_kwargs
from .prompts import opening_message, system_prompt
from .schema import Container, tool_defs, validate
from .tools import ToolBox

MAX_TURNS = 40


def _digest(payload) -> str:
    return ResponseCache.key(payload)


def _charge_usage(ledger: Ledger, usage) -> None:
    """Bill one API round trip, splitting cached reads from fresh input."""
    cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    written = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    fresh = int(usage.input_tokens or 0)
    if fresh:
        ledger.charge("frontier_in", fresh)
    if cached:
        ledger.charge("frontier_in", cached, cached=True)
    if written:
        ledger.charge("frontier_in", written, cache_write=True)
    ledger.charge("frontier_out", int(usage.output_tokens or 0))


def _blocks(msg) -> list[dict]:
    """The assistant turn, as plain dicts we can cache and replay."""
    out = []
    for b in msg.content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "thinking":
            out.append({"type": "thinking", "thinking": b.thinking,
                        "signature": getattr(b, "signature", "")})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name,
                        "input": b.input})
    return out


def run_agent(inst: GlyphInstance, ledger: Ledger, trace: TraceWriter,
              cache: ResponseCache, *, arm: str, allowed: set[Container],
              work_dir: str | Path, base_model: str = "Qwen/Qwen3-1.7B",
              model: str = TEACHER, max_turns: int = MAX_TURNS,
              can_query: bool = True, max_tokens: int = 32000) -> ToolBox:
    """Run one prepare phase and return the toolbox holding what it sealed."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    box = ToolBox(inst=inst, ledger=ledger, trace=trace, allowed=allowed,
                  work_dir=work_dir, base_model=base_model, arm=arm)
    tools = tool_defs(allowed, can_query=can_query)
    q_hint = int(ledger.remaining / max(1e-9, ledger.costs.to_h100s(
        ledger.costs.usd_per_oracle_query))) if ledger.total else 2000

    system = system_prompt(allowed, n_test=inst.cfg.n_test,
                           n_values=inst.cfg.n_values, q_hint=q_hint)
    messages: list[dict] = [{
        "role": "user",
        "content": opening_message(inst.syntax_spec(), inst.demo_block(),
                                   inst.cfg.n_demos),
    }]

    c = client()
    trace.emit("run_start", arm=arm, model=model, effort=EFFORT,
               allowed=sorted(x.value for x in allowed),
               budget_h100s=ledger.total, tools=[t["name"] for t in tools])

    forced_seal = False
    for turn in range(max_turns):
        key = {"model": model, "effort": EFFORT, "tools": _digest(tools),
               "messages": _digest(messages)}
        hit = cache.get(key)
        if hit is not None:
            blocks, stop_reason = hit["blocks"], hit["stop_reason"]
            trace.emit("replay", arm=arm, turn=turn)
        else:
            kwargs = request_kwargs(model, max_tokens=max_tokens)
            try:
                with c.messages.stream(system=system, messages=messages,
                                       tools=tools, **kwargs) as stream:
                    msg = stream.get_final_message()
            except Exception as exc:
                trace.emit("api_error", arm=arm, turn=turn, error=str(exc)[:400])
                raise
            blocks, stop_reason = _blocks(msg), msg.stop_reason
            cache.put(key, {"blocks": blocks, "stop_reason": stop_reason})
            try:
                _charge_usage(ledger, msg.usage)
            except BudgetExhausted as exc:
                # The tokens were spent -- the reply already came back -- so
                # the ledger keeps the overspend. What must not happen is the
                # run dying here: a prepare phase that runs out of budget is
                # supposed to seal what it has, which is a comparable
                # number, rather than leave a hole in the grid.
                trace.emit("budget_exhausted", arm=arm, turn=turn,
                           where="frontier_usage", detail=str(exc))
                messages.append({"role": "assistant", "content": blocks})
                break

        messages.append({"role": "assistant", "content": blocks})
        calls = [b for b in blocks if b["type"] == "tool_use"]

        if stop_reason == "refusal":
            trace.emit("refusal", arm=arm, turn=turn)
            break
        if not calls:
            # Nothing asked for and nothing sealed: nudge once, then stop.
            if forced_seal:
                break
            forced_seal = True
            messages.append({"role": "user", "content":
                             "Continue, or call seal if you are done."})
            continue

        by_name = {t["name"]: t for t in tools}
        results = []
        for call in calls:
            spec = by_name.get(call["name"])
            problem = validate(spec, call["input"]) if spec else \
                f"no such tool {call['name']!r}"
            if problem:
                # What `strict: true` would have prevented server-side. The
                # model sees the problem and can fix it; the trace records
                # that it happened, so protocol noise is visible rather than
                # silently absorbed.
                trace.emit("bad_tool_call", arm=arm, turn=turn,
                           tool=call["name"], problem=problem)
                results.append({"type": "tool_result",
                                "tool_use_id": call["id"],
                                "content": json.dumps({"error": problem})})
                continue
            try:
                out = box.dispatch(call["name"], call["input"])
            except BudgetExhausted as exc:
                trace.emit("budget_exhausted", arm=arm, turn=turn, detail=str(exc))
                out = {"error": f"budget exhausted: {exc}. Seal now."}
                forced_seal = True
            results.append({"type": "tool_result", "tool_use_id": call["id"],
                            "content": json.dumps(out, ensure_ascii=False)[:8000]})

        # Every result goes back in ONE user message: splitting them teaches
        # the model to stop making parallel calls.
        messages.append({"role": "user", "content": results})

        if box.sealed is not None:
            break
        if forced_seal and ledger.remaining <= 0:
            break

    if box.sealed is None:
        # Out of turns or out of budget without sealing. Seal what exists, so
        # the run still produces a comparable number rather than a hole.
        # The last artifact produced, or none at all -- either way the seal
        # must succeed, so that a run out of turns still yields a comparable
        # number rather than a hole in the grid.
        last = next(reversed(box.artifacts), "") if box.artifacts else ""
        box.dispatch("seal", {"artifact_id": last,
                              "summary": "(sealed by the harness: the agent "
                                         "did not seal in time)",
                              "forced": True})
        assert box.sealed is not None, "the harness seal must not be refusable"
        trace.emit("harness_seal", arm=arm, artifacts=len(box.artifacts),
                   had_program=bool(box.program),
                   had_checkpoint=bool(box.checkpoints),
                   had_context=bool(box.context))

    trace.emit("run_end", arm=arm, sealed=box.sealed is not None,
               queries=len(box.queried), ledger=ledger.summary())
    return box
