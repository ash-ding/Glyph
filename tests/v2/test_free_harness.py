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


def test_free_matrix_is_five_tools_in_both_arms():
    for arm in ("train", "no_train"):
        practice = [n for n in ALL if tool_available_for(arm, "practice", n, "free")]
        final = [n for n in ALL if tool_available_for(arm, "final", n, "free")]
        assert practice == ["query", "submit_validation_answer", "check_answers", "finish_practice"]
        assert final == ["check_answers", "submit_final_answer"]
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
