"""Export a self-contained `run.json` for a completed protocol-v2 run.

The viewer (`viewer/index.html`) reads only this stable schema, never Claude
Code's internal transcript format. `run.json` bundles the config, the prompts
the agent was measured on, the task README it saw, the final report, and a
normalized turn-by-turn transcript.

Note on thinking: the persisted CLI transcript redacts extended-thinking text
(only a signature remains), so the raw chain-of-thought is not recoverable
from the CLI transcript alone. Redacted thinking blocks are counted so the
viewer can show a marker. Separately, the gateway (see T1) can capture the
actual thinking text it observes and write it to `run_dir/thinking.jsonl`;
when present, `build_run_json` joins that captured text onto the matching
turn (`thinking_texts`) so the viewer can show real reasoning instead of just
a redacted count.
"""
from __future__ import annotations

import json
from pathlib import Path


def _blocks(row):
    m = row.get("message", {})
    c = m.get("content", []) if isinstance(m, dict) else []
    return c if isinstance(c, list) else []


def _is_opener(row):
    """A user row whose content is a plain string is an injected prompt/opener,
    not a tool-result turn."""
    m = row.get("message", {})
    return row.get("type") == "user" and isinstance(m, dict) and isinstance(m.get("content"), str)


def _opener_text(row):
    return row.get("message", {}).get("content", "")


def normalize_transcript(rows, final_opener=None):
    """Turn the CLI transcript rows into an ordered list of turns.

    Each turn: {turn, phase, text: [str], actions: [{id, name, input, result,
    is_error}], thinking_redacted: int}. A turn is a run of assistant blocks up
    to the tool-results that answer them; a user tool-result row closes it.
    Phase flips to "final" at the turn following the user opener that matches
    `final_opener` (a stable prefix match); everything before is "practice".
    """
    fo = (final_opener or "").strip()[:60]
    id_to_action = {}
    turns = []
    cur = None
    phase = "practice"

    def flush():
        nonlocal cur
        if cur is not None and (cur["text"] or cur["actions"] or cur["thinking_redacted"]):
            cur["turn"] = len(turns) + 1
            cur.setdefault("kind", "turn")
            turns.append(cur)
        cur = None

    def ensure():
        nonlocal cur
        if cur is None:
            cur = {"turn": 0, "phase": phase, "text": [], "actions": [],
                   "thinking_redacted": 0}

    for row in rows:
        t = row.get("type")
        if t == "user" and _is_opener(row):
            # an injected prompt -- closes any open turn; may switch phase or
            # mark a context compaction (both are meaningful boundaries).
            flush()
            otext = _opener_text(row).strip()
            if fo and otext[:60] == fo:
                phase = "final"
            elif otext.startswith("This session is being continued"):
                turns.append({"turn": len(turns) + 1, "phase": phase,
                              "kind": "compaction", "text": [], "actions": [],
                              "thinking_redacted": 0})
            continue
        if t == "assistant":
            for b in _blocks(row):
                bt = b.get("type")
                if bt == "thinking":
                    ensure(); cur["thinking_redacted"] += 1
                elif bt == "text":
                    txt = (b.get("text") or "").strip()
                    if txt:
                        ensure(); cur["text"].append(txt)
                elif bt == "tool_use":
                    ensure()
                    act = {"id": b.get("id"), "name": b.get("name"),
                           "input": b.get("input"), "result": None,
                           "is_error": False}
                    cur["actions"].append(act)
                    if b.get("id"):
                        id_to_action[b["id"]] = act
        elif t == "user":
            # tool-result row: fill matching actions, then close the turn
            for b in _blocks(row):
                if b.get("type") == "tool_result":
                    act = id_to_action.get(b.get("tool_use_id"))
                    if act is not None:
                        c = b.get("content")
                        act["result"] = c if isinstance(c, str) else json.dumps(c)
                        act["is_error"] = bool(b.get("is_error"))
            flush()
    flush()
    # stamp phase per turn (phase may have advanced after a turn was appended)
    # re-derive: turns keep the phase captured when their first block landed
    return turns


def _read_cli_transcript(run_dir):
    run_dir = Path(run_dir)
    cand = sorted((run_dir / "agent_home" / ".claude" / "projects").rglob("*.jsonl"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        return []
    rows = []
    for line in cand[0].read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _read_thinking_capture(run_dir):
    """Read the gateway-captured thinking records from `run_dir/thinking.jsonl`.

    One JSON object per line: {"thinking": [str,...], "redacted": int,
    "tool_use_ids": [str,...], "text_preview": str}. Missing file, blank
    lines, and malformed lines are all tolerated -- this never raises.
    """
    path = Path(run_dir) / "thinking.jsonl"
    if not path.exists():
        return []
    try:
        text = path.read_text()
    except OSError:
        return []
    captures = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            captures.append(rec)
    return captures


def attach_thinking(turns, captures):
    """Join captured thinking text (from `_read_thinking_capture`) onto turns.

    Mutates and returns `turns`, adding a `thinking_texts: [str]` field to
    every turn. Join rule:
      1. By tool_use id: a capture whose `tool_use_ids` intersects a turn's
         action ids is attached to that turn (its `thinking` strings
         appended, in capture file order). Each capture is consumed at most
         once, even if it shares an id with several turns' actions.
      2. Fallback: any capture left unconsumed (typically one with no
         `tool_use_ids`, e.g. thinking that preceded only a final text
         answer) is assigned, in file order, to the next turn (in turn
         order) that has no `thinking_texts` yet and was not matched in
         step 1.
    Backward compatible: with no captures, every turn just gets
    `thinking_texts: []` and nothing else changes.
    """
    for t in turns:
        t["thinking_texts"] = []
    if not captures:
        return turns

    consumed = [False] * len(captures)
    id_to_idx = {}
    for i, cap in enumerate(captures):
        for tid in (cap.get("tool_use_ids") or []):
            if tid and tid not in id_to_idx:
                id_to_idx[tid] = i

    matched_turn = [False] * len(turns)
    for ti, t in enumerate(turns):
        action_ids = [a.get("id") for a in t.get("actions", []) if a.get("id")]
        cap_idxs = sorted({id_to_idx[aid] for aid in action_ids if aid in id_to_idx})
        for i in cap_idxs:
            if not consumed[i]:
                t["thinking_texts"].extend(captures[i].get("thinking") or [])
                consumed[i] = True
                matched_turn[ti] = True

    # Deterministic sequential fallback for remaining captures (typically
    # ones with no tool_use_ids at all): assign each, in file order, to the
    # next turn (in order) that has no thinking_texts yet and wasn't matched
    # by id above.
    ti = 0
    for i, cap in enumerate(captures):
        if consumed[i]:
            continue
        while ti < len(turns) and (matched_turn[ti] or turns[ti]["thinking_texts"]):
            ti += 1
        if ti >= len(turns):
            break
        turns[ti]["thinking_texts"].extend(cap.get("thinking") or [])
        consumed[i] = True
        ti += 1

    return turns


def build_run_json(*, run_id, created, config, task_readme, prompts, report,
                   run_dir):
    """Assemble the self-contained run.json dict."""
    rows = _read_cli_transcript(run_dir)
    transcript = normalize_transcript(rows, final_opener=prompts.get("final_opener"))
    captures = _read_thinking_capture(run_dir)
    attach_thinking(transcript, captures)
    inst = report.get("instance", {}) if isinstance(report, dict) else {}
    return {
        "id": run_id,
        "created": created,
        "config": config,
        "task": {"readme": task_readme},
        "prompts": prompts,
        "report": report,
        "summary": {
            "arm": config.get("arm"),
            "preset": config.get("preset"),
            "seed": config.get("seed"),
            "instance_id": config.get("instance_id"),
            "pi": (inst.get("pi") or {}).get("pi") if isinstance(inst.get("pi"), dict) else None,
            "overall": report.get("overall") if isinstance(report, dict) else None,
            "spent_usd": (report.get("ledger", {}) or {}).get("spent_usd") if isinstance(report, dict) else None,
            "run_status": (report.get("covariates", {}) or {}).get("run_status") if isinstance(report, dict) else None,
            "n_turns": len(transcript),
        },
        "transcript": transcript,
    }
