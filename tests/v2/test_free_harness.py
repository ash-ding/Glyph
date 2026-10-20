"""The free protocol's harness: phases on a wall clock, five tools, neutral text."""
from __future__ import annotations

import asyncio
import json
import re
import time

import pytest

from glyph.data import PRESETS, generate
from glyph.v2 import free_prompts as FP
from glyph.v2 import tools as T
from glyph.v2.free_harness import (drive_free_phase, final_usd_line, transcript_audit,
                                     transcript_usage, unary_coverage)
from glyph.v2.ledger import Ledger
from glyph.v2.session import Session, tool_available_for

CFG = PRESETS["smoke"]
ALL = ["query", "submit_validation_answer", "check_answers", "finish_practice",
       "build_dataset", "train_model", "infer_model", "submit_final_answer"]


class FakeAssistant:
    def __init__(self, names):
        self.tool_names = list(names)


class FakeTerminal:
    pass


class FakeClient:
    """turns: list of (tool calls, seconds the turn takes)."""

    def __init__(self, session, turns):
        self.session, self._turns, self._i = session, list(turns), 0
        self.queries, self.interrupted = [], 0

    async def query(self, text):
        self.queries.append(text)

    async def interrupt(self):
        self.interrupted += 1

    def receive_response(self):
        turn = self._turns[self._i] if self._i < len(self._turns) else ([], 0)
        self._i += 1
        return self._play(*turn)

    async def _play(self, calls, delay):
        names = []
        for name, kw in calls:
            getattr(T, "t_" + name)(self.session, **kw)
            names.append(name)
        if delay:
            await asyncio.sleep(delay)
        yield FakeAssistant(names)
        yield FakeTerminal()


@pytest.fixture
def session(tmp_path):
    inst = generate(1001, CFG)
    s = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm="train", protocol="free",
                q_cap=None, submit_cap=None)
    s.val_id_of = T.default_id_of("val", inst.val)
    s.test_id_of = T.default_id_of("test", inst.test)
    return s


def test_free_matrix_is_four_tools_in_every_arm():
    for arm in ("train", "no_train", "silent"):
        practice = [n for n in ALL if tool_available_for(arm, "practice", n, "free")]
        final = [n for n in ALL if tool_available_for(arm, "final", n, "free")]
        assert practice == ["query", "submit_validation_answer", "finish_practice"]
        assert final == ["submit_final_answer"]
    from glyph.v2.free_harness import FREE_TOOLS
    assert "check_answers" not in FREE_TOOLS
    assert tool_available_for("train", "practice", "train_model", "v2")


def test_remaining_reports_seconds_only(session):
    session.deadline = time.time() + 125
    out = T.t_finish_practice(session, reason="x")
    assert set(out) == {"ok", "seconds_remaining"}
    assert 120 <= out["seconds_remaining"] <= 125


def test_no_submission_cap(session, tmp_path):
    f = tmp_path / "work"
    f.mkdir()
    ans = f / "val.jsonl"
    ans.write_text("".join(json.dumps({"id": session.val_id_of(t), "answer": t.answer_src}) + "\n"
                           for t in session.inst.val))
    for _ in range(25):
        r = T.t_submit(session, path=str(ans))
        assert r["overall"] == 1.0
    assert session.submissions == 25 and not session.should_switch
    assert all("ts" in e for e in session.events if e["kind"] == "validation_submit")


def test_phase_ends_on_finish_and_nudges_between_turns(session):
    client = FakeClient(session, [([], 0), ([], 0), ([("finish_practice", {"reason": "done"})], 0)])
    stop = lambda: "finish_practice" if session.should_switch else None
    res = asyncio.run(drive_free_phase(session, client, opener="go", deadline=time.time() + 30,
                                       should_stop=stop, nudge=FP.practice_nudge))
    assert res["reason"] == "finish_practice"
    assert client.queries[0] == "go"
    assert len(client.queries) == 3 and all(q.startswith("Continue, or call finish_practice")
                                            for q in client.queries[1:])
    assert res["tool_counts"] == {"finish_practice": 1}


def test_deadline_mid_turn_interrupts_kills_and_drains(session):
    killed = []

    async def kill():
        killed.append(time.time())

    client = FakeClient(session, [([], 5.0), ([], 0)])
    t0 = time.time()
    res = asyncio.run(drive_free_phase(session, client, opener="go", deadline=t0 + 0.3,
                                       should_stop=lambda: None, nudge=FP.practice_nudge,
                                       on_deadline=kill, grace_s=2.0))
    assert res["reason"] == "deadline" and res["drained"]
    assert client.interrupted == 1 and len(killed) == 1
    assert time.time() - t0 < 3


def test_idle_agent_is_nudged_until_the_deadline(session):
    client = FakeClient(session, [([], 0.05)] * 1000)
    res = asyncio.run(drive_free_phase(session, client, opener="go", deadline=time.time() + 0.4,
                                       should_stop=lambda: None, nudge=FP.practice_nudge))
    assert res["reason"] == "deadline"
    assert len(client.queries) > 3


def _without_rules(text: str) -> str:
    return text.replace(FP.RULES, "")


@pytest.mark.parametrize("gpu", ["NVIDIA H100 80GB HBM3", None])
def test_text_is_neutral(gpu):
    env = FP.environment_line(gpu, 24, 160, 35)
    texts = [FP.system_prompt(env, 3600, 1800), FP.task_readme(env, 3600, 1800),
             FP.practice_opener("/w/task"), FP.final_opener("/w/task", 1800, 2),
             FP.practice_nudge(100), FP.final_nudge(100), FP.TIMER_SH]
    pat = re.compile(r"\b(" + "|".join(FP.FORBIDDEN_WORDS) + r")", re.I)
    for t in texts:
        assert not pat.search(_without_rules(t)), pat.search(_without_rules(t))


def test_arms_differ_only_in_the_environment_line():
    a = FP.task_readme(FP.environment_line("NVIDIA H100 80GB HBM3", 24, 160, 35), 3600, 1800)
    b = FP.task_readme(FP.environment_line(None, 24, 160, 35), 3600, 1800)
    diff = [(x, y) for x, y in zip(a.splitlines(), b.splitlines()) if x != y]
    assert len(diff) == 1 and "GPU" in diff[0][0]


def test_cli_flags_build_a_free_config():
    from glyph.v2.cli import build_parser, build_run_config
    ns = build_parser().parse_args(["run", "--arm", "train", "--protocol", "free", "--t-p", "2",
                                    "--gpu-minor", "3", "--instance-id", "mid_3"])
    rc = build_run_config(ns)
    assert rc.protocol == "free" and rc.t_p_s == 7200 and rc.t_f_s == 1800 and rc.gpu_minor == 3


def test_unary_coverage_and_transcript_usage(session, tmp_path):
    from glyph.data.grammar import unary_names
    assert unary_coverage(session.inst) == 0.0
    op = unary_names(CFG)[0]
    session.inst.query_log.unary |= {(op, 0), (op, 1)}
    assert unary_coverage(session.inst) == 2 / (len(unary_names(CFG)) * CFG.n_values)

    proj = tmp_path / "home" / ".claude" / "projects" / "-w"
    (proj / "s1" / "subagents").mkdir(parents=True)
    (proj / "s1.jsonl").write_text(json.dumps({"message": {"usage": {"input_tokens": 5, "output_tokens": 2}}}) + "\n")
    (proj / "s1" / "subagents" / "agent-a.jsonl").write_text(
        json.dumps({"message": {"usage": {"input_tokens": 7, "output_tokens": 1}}}) + "\n")
    u = transcript_usage(tmp_path / "home")
    assert u["main"]["input_tokens"] == 5 and u["subagents"]["input_tokens"] == 7
    assert u["n_subagent_files"] == 1


def test_transcript_audit_flags_llm_hosts_and_benchmark_lookups(tmp_path):
    proj = tmp_path / ".claude" / "projects" / "-w"
    proj.mkdir(parents=True)
    rows = [
        {"message": {"content": [{"type": "tool_use", "name": "WebFetch",
                                  "input": {"url": "https://ash-ding.github.io/Glyph/"}}]}},
        {"message": {"content": [{"type": "tool_use", "name": "WebSearch",
                                  "input": {"query": "pytorch lora tutorial"}}]}},
        {"message": {"content": [{"type": "tool_use", "name": "Bash",
                                  "input": {"command": "curl -s https://api.openai.com/v1/chat/completions -d x; pip download https://files.pythonhosted.org/x.whl"}}]}},
    ]
    (proj / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    a = transcript_audit(tmp_path)
    assert a["llm_api_hosts"] == ["api.openai.com"]
    assert a["benchmark_lookups"] == ["https://ash-ding.github.io/Glyph/"]
    assert a["websearch_queries"] == ["pytorch lora tutorial"]
    assert "files.pythonhosted.org" in a["hosts"]


def test_final_phase_keeps_an_allowance_after_the_safety_line():
    assert final_usd_line(500, 120) == 500
    assert final_usd_line(500, 450) == 550
    assert final_usd_line(500, 503.5) == 603.5


def test_run_directory_does_not_name_the_arm(tmp_path):
    from glyph.v2.free_harness import neutral_run_dir
    run_dir = neutral_run_dir(tmp_path, "low_3_no_train_tp1h")
    assert re.fullmatch(r"r[0-9a-f]{8}", run_dir.name)
    for word in ("train", "low", "tp1h"):
        assert word not in str(run_dir.relative_to(tmp_path))
    link = tmp_path / "low_3_no_train_tp1h"
    assert link.is_symlink() and link.resolve() == run_dir.resolve()
    assert (run_dir / "run_name").read_text().strip() == "low_3_no_train_tp1h"
    with pytest.raises(FileExistsError):
        neutral_run_dir(tmp_path, "low_3_no_train_tp1h")


ENV_WORDS = ("Environment", "GPU", "H100", "CPU", "cores", "memory", "disk", "Ubuntu", "conda", "network")


def test_silent_text_has_no_environment_section():
    env = FP.environment_for_arm("silent", "NVIDIA H100 80GB HBM3", 24, 160, 35)
    assert env is None
    for t in (FP.system_prompt(env, 3600, 1800), FP.task_readme(env, 3600, 1800)):
        for w in ENV_WORDS:
            assert not re.search(rf"\b{w}\b", t), w


def test_silent_differs_from_train_only_by_the_environment_section():
    line = FP.environment_for_arm("train", "NVIDIA H100 80GB HBM3", 24, 160, 35)
    sp_train, sp_silent = FP.system_prompt(line, 3600, 1800), FP.system_prompt(None, 3600, 1800)
    assert sp_train.replace(f"**Environment**\n\n{line}\n\n", "") == sp_silent
    rd_train, rd_silent = FP.task_readme(line, 3600, 1800), FP.task_readme(None, 3600, 1800)
    assert rd_train.replace(f"Environment:\n  {line}\n\n", "") == rd_silent


def test_environment_for_arm():
    assert "one NVIDIA H100 80GB HBM3 GPU" in FP.environment_for_arm("train", "NVIDIA H100 80GB HBM3", 24, 160, 35)
    assert "no GPU" in FP.environment_for_arm("no_train", None, 24, 160, 35)
    assert FP.GPU_ARMS == ("train", "silent")
    with pytest.raises(ValueError):
        FP.environment_for_arm("trian", None, 24, 160, 35)


def test_cli_accepts_the_silent_arm():
    from glyph.v2.cli import build_parser, build_run_config
    from glyph.v2.free_sandbox import DEFAULT_IMAGE
    ns = build_parser().parse_args(["run", "--arm", "silent", "--protocol", "free", "--t-p", "1",
                                    "--gpu-minor", "2", "--instance-id", "low_3"])
    rc = build_run_config(ns)
    assert rc.arm == "silent" and rc.gpu_minor == 2 and rc.image == DEFAULT_IMAGE


def _cell_expr(inst, op, i, j):
    from glyph.data.grammar import render_value
    return f"{op}({render_value(i, inst.cfg)}, {render_value(j, inst.cfg)})"


def _fold_reaching(inst, cell):
    """A two-element structural expression whose evaluation looks up `cell`."""
    from glyph.data.grammar import parse, check
    op, i, j = cell
    xs = [_cell_expr(inst, op, i, j)[len(op) + 1:-1].split(", ")]
    x, y = xs[0]
    for s in sorted(inst.skeleton):
        for lst in (f"[{x}, {y}]", f"[{y}, {x}]"):
            src = f"{s}({lst}, {op})"
            try:
                e = parse(src, inst.cfg)
                check(e, inst.cfg)
                _, log = inst.P.eval_logged(e)
            except Exception:
                continue
            if cell in log.binary and inst.query_violation(src) is None:
                return src
    return None


def test_free_query_refuses_reserved_validation_entries(session):
    session.reserve_val_binary = True
    inst = session.inst
    reserved = inst.reserved_binary()
    assert reserved
    cell = sorted(reserved)[0]
    free_cell = next((op, i, j) for op in ("b0", "b1") for i in range(inst.cfg.n_values)
                     for j in range(3) if (op, i, j) not in reserved)
    out = T.t_query(session, exprs=[_cell_expr(inst, *cell), _cell_expr(inst, *free_cell)], why="t")
    first, second = out["results"]
    assert first.get("refused") == "needs_validation_entry"
    assert "out" in second
    assert session.q_used == 1 and session.refused_reserved == 1
    assert cell not in inst.query_log.binary


def test_free_query_refuses_a_fold_that_reaches_a_reserved_entry(session):
    session.reserve_val_binary = True
    inst = session.inst
    found = next(((c, src) for c in sorted(inst.reserved_binary())
                  if (src := _fold_reaching(inst, c))), None)
    assert found, "no two-element fold reaches a reserved entry on this instance"
    cell, src = found
    out = T.t_query(session, exprs=[src], why="t")
    assert out["results"][0].get("refused") == "needs_validation_entry"
    assert cell not in inst.query_log.binary


def test_reserved_entries_are_off_outside_the_free_protocol(session):
    assert session.reserve_val_binary is False
    inst = session.inst
    cell = sorted(inst.reserved_binary())[0]
    out = T.t_query(session, exprs=[_cell_expr(inst, *cell)], why="t")
    assert "out" in out["results"][0]


def test_readme_states_the_reserved_entry_rule():
    r = FP.task_readme(FP.environment_line(None, 24, 160, 35), 3600, 1800)
    assert "binary-operator\n    entry that a validation item needs" in r


def _write_answers(path, session, legal=True):
    items, id_of = T._final_items(session)
    rows = [{"id": id_of(t), "answer": t.answer_src} for t in items]
    if not legal:
        rows = rows[:-1]          # a missing id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_deadline_fallback_takes_the_newest_legal_answer_file(session, tmp_path):
    import os
    from glyph.v2.free_harness import latest_legal_answer_file
    session.work_root = tmp_path
    t0 = time.time() - 100
    old = _write_answers(tmp_path / "old.jsonl", session)
    os.utime(old, (t0 - 50, t0 - 50))                       # before the final began
    good = _write_answers(tmp_path / "out" / "good.jsonl", session)
    os.utime(good, (t0 + 10, t0 + 10))
    bad = _write_answers(tmp_path / "bad.jsonl", session, legal=False)
    os.utime(bad, (t0 + 20, t0 + 20))                       # newer, but illegal
    inside_task = _write_answers(tmp_path / "task" / "x.jsonl", session)
    os.utime(inside_task, (t0 + 30, t0 + 30))               # newest, but under task/
    assert latest_legal_answer_file(session, t0) == str(good.resolve())
    assert latest_legal_answer_file(session, t0 + 15) is None


def test_free_text_has_no_check_answers():
    env = FP.environment_line(None, 24, 160, 35)
    for t in (FP.system_prompt(env, 3600, 1800), FP.task_readme(env, 3600, 1800)):
        assert "check_answers" not in t
    assert "most recently\n    written legal answer file" in FP.task_readme(env, 3600, 1800)


def test_bare_atomics_refused_when_the_option_is_on(session):
    session.refuse_bare_atomics = True
    inst = session.inst
    from glyph.data.grammar import render_value
    x, y = render_value(5, inst.cfg), render_value(7, inst.cfg)
    out = T.t_query(session, exprs=[f"u0({x})", f"b0({x}, {y})", f"s0(u0, [{x}, {y}])"], why="t")
    r = out["results"]
    assert r[0].get("refused") == "bare_atomic" and r[1].get("refused") == "bare_atomic"
    assert "out" in r[2], r[2]
    assert session.q_used == 1 and session.refused_bare == 2
    assert ("u0", 5) in inst.query_log.unary          # the map still buys the cell


def test_bare_atomics_allowed_by_default(session):
    from glyph.data.grammar import render_value
    assert session.refuse_bare_atomics is False
    out = T.t_query(session, exprs=[f"u0({render_value(5, session.inst.cfg)})"], why="t")
    assert "out" in out["results"][0]


def test_readme_does_not_mention_the_bare_atomic_rule():
    # The agent is told nothing about it; a refused query (reason "bare_atomic") is the only signal.
    env = FP.environment_line(None, 24, 160, 35)
    r = FP.task_readme(env, 3600, 1800)
    assert "bare" not in r.lower() and "u*(value)" not in r


def test_cli_no_bare_atomics_flag():
    from glyph.v2.cli import build_parser, build_run_config
    ns = build_parser().parse_args(["run", "--arm", "silent", "--protocol", "free", "--t-p", "1",
                                    "--gpu-minor", "0", "--instance-id", "low_3", "--no-bare-atomics"])
    assert build_run_config(ns).bare_atomics is False
    ns = build_parser().parse_args(["run", "--arm", "silent", "--protocol", "free", "--gpu-minor", "0"])
    assert build_run_config(ns).bare_atomics is True
