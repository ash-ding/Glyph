"""Harness orchestrator for protocol v2: the two-phase run driver.

Two layers live here:

1. The PHASE STATE MACHINE (drive_practice / drive_final / iter_tool_calls /
   ClientProto) -- pure logic over a Session and a client interface. This is
   the part the Task-17 unit test exercises with a fake client, so it MUST be
   importable without claude_agent_sdk (and without torch).

2. run(rc) -- the REAL path that stands up the workspace, gateway, MCP server
   and ClaudeSDKClient and drives a full run. All SDK imports are LOCAL to
   run() (and to mcp.register_glyph_server), so importing this module never
   requires the SDK.

Phase-switch recipe (spec "S4"): the MCP handlers run DURING the SDK's stream
and mutate the session (submissions, should_switch, final_commit_path, ...).
The driver therefore reads those flags AFTER draining each assistant turn.
When the end trigger is a should_switch detected mid-stream (the cap-th submit
or finish_practice), we interrupt() the client and then drain the current
receive_response to its terminal event before returning -- we never break
early and lose the drained result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


# ---------------------------------------------------------------------
# client interface
# ---------------------------------------------------------------------
class ClientProto(Protocol):
    async def query(self, text: str) -> None: ...
    def receive_response(self): ...          # async iterator of "events"
    async def interrupt(self) -> None: ...


_PRACTICE_CONTINUE = "Continue, or call finish_practice if you are done."
_FINAL_CONTINUE = "Continue, or call final_answer with your submission if you are done."
_IDLE_LIMIT = 3


def iter_tool_calls(event) -> list[str]:
    """Return the tool names an assistant event called, [] otherwise.

    Shared by the real SDK adapter and the fake test client:
      * a fake assistant event exposes ``.tool_names`` directly;
      * a real AssistantMessage exposes ``.content`` blocks -- a ToolUseBlock
        is the only block carrying BOTH ``name`` and ``input``;
      * everything else (terminal/result, user/tool-result, text) -> [].
    """
    names = getattr(event, "tool_names", None)
    if names is not None:
        return list(names)
    content = getattr(event, "content", None)
    if not content:
        return []
    out = []
    for block in content:
        if hasattr(block, "name") and hasattr(block, "input"):
            out.append(block.name)
    return out


def _is_assistant_event(event) -> bool:
    """True if ``event`` is one assistant message (real or fake).

    The real ClaudeSDKClient.receive_response() streams the WHOLE agent-loop
    response as a single iteration -- many AssistantMessages, interleaved
    with UserMessage (tool results) and a terminal ResultMessage. A turn
    (spec Sec 4.5) is one assistant message, so we count events identified as
    AssistantMessage here rather than counting receive_response() calls.

      * a fake assistant event exposes ``.tool_names`` directly (even an idle
        one, where the list is empty) -- the fake terminal event does not;
      * a real AssistantMessage is identified by class name (duck-typed, so
        this module never needs to import claude_agent_sdk).
    """
    if hasattr(event, "tool_names"):
        return True
    return type(event).__name__ == "AssistantMessage"


async def _consume_turn(client, should_stop) -> tuple[list[str], int]:
    """Drain ONE receive_response() to its terminal event.

    Returns (tool_names, n_turns): tool_names is every tool name seen across
    the drained response; n_turns is the count of assistant messages drained
    (spec Sec 4.5: a turn = one assistant message). For the real SDK client
    this can be >1 per call (the whole agent loop streams as one iteration);
    for the fake test client it is always 1 (one scripted turn -> one
    assistant event). If ``should_stop()`` becomes true mid-stream (a handler
    set the end flag), interrupt() the client once but keep draining to the
    terminal event.
    """
    tool_names: list[str] = []
    n_turns = 0
    interrupted = False
    async for event in client.receive_response():
        if _is_assistant_event(event):
            n_turns += 1
        tool_names.extend(iter_tool_calls(event))
        if not interrupted and should_stop():
            await client.interrupt()
            interrupted = True
    return tool_names, n_turns


async def drive_practice(session, client: ClientProto, *, opener: str) -> str:
    """Drive the practice phase. Returns the end reason.

    Every assistant message drained increments session.turns (idle turns
    included; at least 1 per cycle even if the drain produced none, so idle
    cycles still advance). An idle turn (no tool call anywhere in the drained
    response) counts consecutively toward _IDLE_LIMIT, ending practice as
    finish_practice ("idle"). Otherwise practice ends on should_switch (cap-th
    submit / finish_practice), the practice turn cap, or the USD safety line.
    """
    await client.query(opener)
    idle = 0
    while True:
        names, n_turns = await _consume_turn(client, lambda: session.should_switch)
        session.turns += max(1, n_turns)

        if names:
            idle = 0
        else:
            idle += 1
            if idle >= _IDLE_LIMIT:
                session.should_switch = True
                session.finished_practice_early = True
                session.note(kind="finish_practice", reason="idle")
                return "idle"
            await client.query(_PRACTICE_CONTINUE)

        if session.should_switch:
            return "should_switch"
        if session.turns >= session.tp:
            return "turn_cap"
        if session.ledger.over_safety_line(session.usd_line):
            return "usd_line"


async def drive_final(session, client: ClientProto, *, opener: str) -> str:
    """Drive the final phase. Returns the end reason.

    Ends when a legal final_answer commits (session.final_commit_path set by
    the handler -> reason "agent"), or on the final turn cap / USD line /
    _IDLE_LIMIT idle turns. On any NON-final_answer end, apply the AUTO-COMMIT
    rule: if a passing check_answers(set="test") recorded last_checked_test_path,
    commit that path (final_commit="auto_checked_path"); else final_commit="none".
    """
    await client.query(opener)
    idle = 0
    reason = None
    while True:
        names, n_turns = await _consume_turn(
            client, lambda: getattr(session, "final_commit_path", None) is not None)
        session.turns += max(1, n_turns)

        if getattr(session, "final_commit_path", None) is not None:
            session.turns_final = session.turns
            return "agent"

        if names:
            idle = 0
        else:
            idle += 1
            if idle >= _IDLE_LIMIT:
                reason = "idle"
                break
            await client.query(_FINAL_CONTINUE)

        if session.turns >= session.tf:
            reason = "turn_cap"
            break
        if session.ledger.over_safety_line(session.usd_line):
            reason = "usd_line"
            break

    # non-final_answer end -> auto-commit rule
    session.turns_final = session.turns
    last_checked = getattr(session, "last_checked_test_path", None)
    if last_checked is not None:
        session.final_commit_path = last_checked
        session.final_commit = "auto_checked_path"
    else:
        session.final_commit = "none"
    return reason


# ---------------------------------------------------------------------
# run config
# ---------------------------------------------------------------------
@dataclass
class RunConfig:
    arm: str
    preset: str = "pi_mid"
    instance_seed: int = 1001
    model: str = "claude-opus-4-8"
    student_model: str = "Qwen/Qwen3-1.7B"
    q_cap: int = 1000
    submit_cap: int = 20
    tp: int = 100
    tf: int = 30
    usd_line: float = 300.0
    n_val: int = 5000
    # max_turns is the SDK's own agent-loop backstop; it must sit ABOVE
    # T_p + T_f (the spec's binding phase caps) or it truncates the run
    # before those caps do.
    max_turns: int = 200
    effort: str = "high"  # pinned for comparability (plan Global Constraints)
    out_root: str | None = None


# ---------------------------------------------------------------------
# real path (SDK imports are LOCAL to run())
# ---------------------------------------------------------------------
def _resolve_cli_path():
    """Path to the SDK's bundled Claude Code CLI (lazy; only called from run()).

    claude_agent_sdk._bundled is a namespace package (its __file__ is None), so
    resolve the CLI relative to the top-level package directory rather than a
    submodule __file__.
    """
    import pathlib, claude_agent_sdk
    p = pathlib.Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    if p.exists():
        return p
    import shutil
    w = shutil.which("claude")
    if w:
        return pathlib.Path(w)
    raise FileNotFoundError("bundled Claude Code CLI not found under claude_agent_sdk._bundled")


def _make_student(rc, ledger, paths):
    """Build the train-arm student pool.

    The student is a small, locally trainable model (Qwen3-1.7B by default,
    ``rc.student_model``) -- deliberately NOT ``rc.model``, which is the
    frontier the agent talks to and the id the gateway pins on. Conflating
    the two would hand a frontier model id to ``sft.train``'s
    ``from_pretrained`` and fail on the first train call.
    """
    from glyph.v2.student import StudentPool

    return StudentPool(
        rc.student_model, ledger, work_dir=paths.root,
        queries_path=paths.queries,
    )


def run(rc: RunConfig) -> dict:
    """Full protocol-v2 run over the real SDK. Returns the ScoreReport dict.

    Every claude_agent_sdk import is local to this function (and to
    register_glyph_server), keeping the state machine above importable
    SDK-free.
    """
    import asyncio
    import json

    from glyph.data import PRESETS, generate
    from glyph.v2 import sandbox, workspace
    from glyph.v2.gateway import Gateway
    from glyph.v2.ledger import Ledger
    from glyph.v2.mcp import register_glyph_server
    from glyph.v2.report import build_report
    from glyph.v2.session import Session
    from glyph.v2.trace import RunTrace
    from glyph.v2 import prompts

    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    # 1. instance
    cfg = PRESETS[rc.preset]
    if rc.n_val is not None:
        try:
            cfg = cfg.__class__(**{**cfg.__dict__, "n_val": rc.n_val})
        except Exception:
            pass
    inst = generate(rc.instance_seed, cfg)

    # 2. run dir + workspace
    out_root = Path(rc.out_root) if rc.out_root else Path.cwd() / "runs"
    run_dir = out_root / f"{rc.preset}_{rc.arm}_{rc.instance_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    paths, val_id_of, test_id_of = workspace.build_workspace(inst, run_dir)

    # 3. ledger + session (README needs the session caps)
    ledger = Ledger()
    session = Session(
        inst, ledger, run_dir, arm=rc.arm,
        q_cap=rc.q_cap, submit_cap=rc.submit_cap,
        tp=rc.tp, tf=rc.tf, usd_line=rc.usd_line,
    )
    session.val_id_of = val_id_of
    session.test_id_of = test_id_of
    session.preset = rc.preset
    if rc.arm == "train":
        session.student = _make_student(rc, ledger, paths)

    # 3b. task README with concrete caps (fills the Task-11 placeholder)
    paths.readme.write_text(prompts.task_readme(session))

    # 4. gateway (S1: gateway.py re-inserts /v1; allowed_models pins the model)
    #    log_path persists a per-request audit log alongside the run's artifacts.
    gw_sock = run_dir / "gw.sock"
    gateway = Gateway(ledger, pinned_model=rc.model, allowed_models={rc.model},
                       log_path=str(run_dir / "gateway_log.jsonl"))

    # 5. trace + MCP server
    trace = RunTrace(run_dir)
    server = register_glyph_server(session, trace)

    # 6. sandbox wrapper
    cli_path = _resolve_cli_path()
    wrapper = sandbox.write_wrapper(run_dir, cli_path=cli_path, gw_sock=gw_sock)

    # 7. SDK options (S1: pin ANTHROPIC_SMALL_FAST_MODEL so background small-model
    #    calls are not 403'd by the gateway allow-list). max_turns bounds the
    #    SDK's own agent loop -- the real runaway-turn guard; the harness
    #    tp/tf caps below are a secondary per-cycle limit.
    # Capture the CLI subprocess's real stderr to a file. The SDK's own
    # ProcessError message is a fixed "Check stderr output for details"
    # placeholder (claude-agent-sdk #256), so without this a CLI crash (e.g.
    # the exit-250 seen on a very large turn) is undiagnosable.
    cli_stderr_path = run_dir / "cli_stderr.log"

    def _capture_stderr(line: str) -> None:
        try:
            with open(cli_stderr_path, "a", encoding="utf-8") as _f:
                _f.write(line.rstrip("\n") + "\n")
        except Exception:
            pass

    options = ClaudeAgentOptions(
        cli_path=str(wrapper),
        stderr=_capture_stderr,
        cwd=str(paths.root),
        model=rc.model,
        system_prompt=prompts.system_prompt(rc.arm),
        setting_sources=[],
        allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep", "mcp__glyph__*"],
        disallowed_tools=["WebFetch", "WebSearch", "Task", "TodoWrite",
                          "NotebookEdit", "Skill"],
        mcp_servers={"glyph": server},
        env={"ANTHROPIC_SMALL_FAST_MODEL": rc.model},
        max_turns=rc.max_turns,
        effort=rc.effort,
    )

    async def _go():
        serve_task = asyncio.create_task(gateway.serve_unix(str(gw_sock)))
        for _ in range(200):                  # up to ~10s for the socket to bind
            if gw_sock.exists():
                break
            await asyncio.sleep(0.05)
        try:
            client = ClaudeSDKClient(options)
            async with client:
                await drive_practice(session, client,
                                     opener=prompts.practice_opener(paths))
                session.turns_practice = session.turns
                session.switch_to_final()
                workspace.write_test_file(paths, inst, test_id_of)
                await drive_final(session, client,
                                  opener=prompts.final_opener(paths))
        finally:
            serve_task.cancel()

    asyncio.run(_go())

    committed = getattr(session, "final_commit_path", None)
    report = build_report(session, committed, test_id_of)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))

    # Self-contained run.json for the run viewer. Never fail the run over it.
    try:
        import time as _t
        from glyph.v2.export import build_run_json
        run_json = build_run_json(
            run_id="%s_%s" % (run_dir.name, _t.strftime("%Y%m%d-%H%M%S")),
            created=_t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()),
            config={
                "arm": rc.arm, "preset": rc.preset, "seed": rc.instance_seed,
                "model": rc.model, "q_cap": rc.q_cap, "submit_cap": rc.submit_cap,
                "tp": rc.tp, "tf": rc.tf, "n_val": rc.n_val, "usd_line": rc.usd_line,
                "max_turns": rc.max_turns, "effort": rc.effort,
                "student_model": rc.student_model,
            },
            task_readme=paths.readme.read_text(),
            prompts={
                "system": prompts.system_prompt(rc.arm),
                "practice_opener": prompts.practice_opener(paths),
                "final_opener": prompts.final_opener(paths),
            },
            report=report,
            run_dir=run_dir,
        )
        (run_dir / "run.json").write_text(json.dumps(run_json, indent=2))
    except Exception as _e:
        import sys as _sys
        _sys.stderr.write("run.json export failed: %r\n" % _e)

    trace.emit("report", overall=report["overall"],
               final_commit=report["covariates"]["final_commit"])
    trace.close()
    return report
