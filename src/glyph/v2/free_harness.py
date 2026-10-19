"""The free protocol's run driver.

Spec: docs/superpowers/specs/2026-10-15-free-protocol-design.md, sections 3-7.

Two phases, each bounded only by wall-clock. The agent's CLI runs in the
container from free_sandbox.py; the five Glyph tools run host-side in the SDK
process, as in v2. At each phase end the harness interrupts the agent, kills
every process in the container except the CLI, and lets the current turn
drain for at most GRACE_S seconds.

The phase driver (`drive_free_phase`) is pure logic over a client and a
clock, so tests drive it with a fake client and short real deadlines.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from glyph.v2.harness import _is_assistant_event, iter_tool_calls

GRACE_S = 60.0
# Model spend the final phase may use even when practice ended at the safety
# line; otherwise a run whose practice crossed the line could not answer.
FINAL_USD_ALLOWANCE = 100.0


def final_usd_line(usd_line: float, spent_at_switch: float) -> float:
    """The final phase's safety line: the run's line, or what practice spent plus
    the allowance, whichever is higher."""
    return max(usd_line, spent_at_switch + FINAL_USD_ALLOWANCE)
FREE_TOOLS = ("query", "submit_validation_answer", "check_answers",
              "finish_practice", "submit_final_answer")
FREE_ALLOWED_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch",
                      "WebSearch", "Task", "TodoWrite", "NotebookEdit", "mcp__glyph__*"]


async def _drain(client, timeout: float) -> bool:
    """Read the rest of the current response; True if it ended in time."""
    async def go():
        async for _ in client.receive_response():
            pass
    try:
        await asyncio.wait_for(go(), timeout)
        return True
    except asyncio.TimeoutError:
        return False
    except Exception:
        return False


async def drive_free_phase(session, client, *, opener: str, deadline: float,
                           should_stop: Callable[[], str | None],
                           nudge: Callable[[float], str],
                           on_deadline: Callable[[], object] | None = None,  # may be async
                           clock: Callable[[], float] = time.time,
                           grace_s: float = GRACE_S) -> dict:
    """Run one phase until should_stop() names a reason or the deadline passes.

    should_stop() returns None to continue, or the end reason ("finish_practice",
    "submitted", "usd_line", ...). At the deadline: interrupt(), then
    on_deadline() (the kill), then drain the turn for at most grace_s.
    Returns {"reason", "tool_counts", "drained"}.
    """
    counts: dict[str, int] = {}

    def tally(names):
        for n in names:
            counts[n] = counts.get(n, 0) + 1

    async def turn():
        # Tally as the turn streams, so a turn cut short by the deadline
        # still counts the calls it made.
        n_turns, interrupted = 0, False
        async for event in client.receive_response():
            if _is_assistant_event(event):
                n_turns += 1
            tally(iter_tool_calls(event))
            if not interrupted and should_stop() is not None:
                await client.interrupt()
                interrupted = True
        return n_turns

    await client.query(opener)
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            reason = "deadline"
            drained = True
            await _call(on_deadline)
            break
        try:
            n = await asyncio.wait_for(turn(), timeout=remaining)
        except asyncio.TimeoutError:
            try:
                await client.interrupt()
            except Exception:
                pass
            await _call(on_deadline)
            drained = await _drain(client, grace_s)
            reason = "deadline"
            break
        session.turns += max(1, n)
        stop = should_stop()
        if stop is not None:
            reason, drained = stop, True
            break
        if clock() >= deadline:
            continue
        await client.query(nudge(deadline - clock()))
    return {"reason": reason, "tool_counts": counts, "drained": drained}


async def _call(fn) -> None:
    if fn is None:
        return
    r = fn()
    if asyncio.iscoroutine(r):
        await r


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def gpu_name_for(uuid: str | None) -> str | None:
    if uuid is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            u, name = [x.strip() for x in line.split(",", 1)]
            if u == uuid:
                return name
    except Exception:
        pass
    return "NVIDIA"


def neutral_run_dir(out_root: Path, name: str) -> Path:
    """Create a run's directory under an opaque id and link `out_root/name` to it.

    The agent sees this directory's path in every message and tool call (its
    working directory, the task paths, the CLI's project folder), so the path
    must not say which arm, instance or practice length the run is: an agent
    that reads "no_train" in its own path may take it as an instruction. The
    readable name exists only as a host-side symlink. Returns the real directory.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    link = out_root / name
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"{link} exists; move it aside before starting this run again")
    while True:
        run_dir = out_root / f"r{secrets.token_hex(4)}"
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            continue
    (run_dir / "run_name").write_text(name + "\n")
    link.symlink_to(run_dir.name)
    return run_dir


def unary_coverage(inst) -> float:
    from glyph.data.grammar import unary_names
    n = len(unary_names(inst.cfg)) * inst.cfg.n_values
    return len(inst.query_log.unary) / n if n else 0.0


def transcript_usage(agent_home: Path) -> dict:
    """Token usage from the CLI's session files: main agent and subagents apart."""
    out = {"main": {}, "subagents": {}, "n_subagent_files": 0}
    root = Path(agent_home) / ".claude" / "projects"
    if not root.exists():
        return out
    for f in root.rglob("*.jsonl"):
        kind = "subagents" if "subagents" in f.parts or f.name.startswith("agent-") else "main"
        if kind == "subagents":
            out["n_subagent_files"] += 1
        for line in f.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = (row.get("message") or {}).get("usage") if isinstance(row, dict) else None
            if not isinstance(u, dict):
                continue
            agg = out[kind]
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                      "cache_creation_input_tokens"):
                agg[k] = agg.get(k, 0) + int(u.get(k, 0) or 0)
    return out


def transcript_audit(agent_home: Path, llm_hosts=None) -> dict:
    """What the agent fetched, searched and pointed commands at, from its session files.

    Flags URLs on known hosted-LLM API hosts (rule 1) and any URL or search
    mentioning the benchmark (rule 2). Main agent and subagents alike.
    """
    import re
    from urllib.parse import urlparse
    from glyph.v2.free_sandbox import LLM_API_HOSTS
    llm_hosts = set(llm_hosts or LLM_API_HOSTS)
    url_re = re.compile(r"https?://[^\s\"'<>)\]]+")
    fetches, searches, bash_urls = [], [], []
    root = Path(agent_home) / ".claude" / "projects"
    for f in (root.rglob("*.jsonl") if root.exists() else []):
        for line in f.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = row.get("message") if isinstance(row, dict) else None
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                name, inp = b.get("name"), b.get("input") or {}
                if name == "WebFetch" and inp.get("url"):
                    fetches.append(inp["url"])
                elif name == "WebSearch" and inp.get("query"):
                    searches.append(inp["query"])
                elif name == "Bash" and inp.get("command"):
                    bash_urls += url_re.findall(inp["command"])
    urls = fetches + bash_urls
    hosts = sorted({urlparse(u).hostname or "" for u in urls} - {""})
    llm_hits = sorted({h for h in hosts if h in llm_hosts or any(h.endswith("." + x) for x in llm_hosts)})
    bench = [x for x in urls + searches if "glyph" in x.lower() or "ash-ding" in x.lower()]
    return {"webfetch_urls": fetches, "websearch_queries": searches, "bash_urls": bash_urls[:500],
            "hosts": hosts, "llm_api_hosts": llm_hits, "benchmark_lookups": bench}


def run_free(rc) -> dict:
    """A full free-protocol run over the real SDK. Returns the report dict."""
    import claude_agent_sdk
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    from glyph.data.probe import probe_set
    from glyph.v2 import free_prompts as FP
    from glyph.v2 import free_sandbox as FS
    from glyph.v2 import workspace
    from glyph.v2.export import build_run_json
    from glyph.v2.gateway import Gateway
    from glyph.v2.harness import resolve_run_instance
    from glyph.v2.ledger import Ledger
    from glyph.v2.mcp import register_glyph_server
    from glyph.v2.report import build_report
    from glyph.v2.session import Session
    from glyph.v2.trace import RunTrace

    inst, preset, seed, instance_id = resolve_run_instance(rc)
    out_root = Path(rc.out_root) if rc.out_root else Path.cwd() / "runs"
    tag = f"tp{rc.t_p_s / 3600:g}h"
    run_name = f"{instance_id or f'{preset}_{seed}'}_{rc.arm}_{tag}"
    run_dir = neutral_run_dir(out_root, run_name)
    paths, val_id_of, test_id_of = workspace.build_workspace(inst, run_dir)

    ledger = Ledger()
    session = Session(inst, ledger, run_dir, arm=rc.arm, protocol="free", reserve_val_binary=True,
                      q_cap=None, submit_cap=None, tp=10**9, tf=10**9, usd_line=rc.usd_line)
    session.val_id_of, session.test_id_of = val_id_of, test_id_of
    session.queries_path, session.work_root = paths.queries, paths.root
    session.preset = preset
    session.probes = probe_set(inst)
    session.probe_id_of = workspace.probe_ids_for(inst, session.probes)

    gpu_uuid = None
    if rc.arm not in FP.ARMS:
        raise ValueError(f"unknown arm {rc.arm!r}; the free protocol has {FP.ARMS}")
    if rc.arm in FP.GPU_ARMS:
        if rc.gpu_minor is None:
            raise ValueError(f"the {rc.arm} arm needs --gpu-minor")
        gpu_uuid = rc.gpu_uuid or FS.probe_gpu_minors(rc.image).get(rc.gpu_minor)
        if gpu_uuid is None:
            raise RuntimeError(f"GPU minor {rc.gpu_minor} not found")
    gpu_name = gpu_name_for(gpu_uuid)
    env_line = FP.environment_for_arm(rc.arm, gpu_name, rc.cpus, rc.mem_gb, rc.disk_gb)
    readme = FP.task_readme(env_line, rc.t_p_s, rc.t_f_s)
    paths.readme.write_text(readme)
    timer = paths.task / "timer.sh"
    timer.write_text(FP.TIMER_SH)
    timer.chmod(0o755)
    deadline_file = paths.task / ".phase_deadline"
    system = FP.system_prompt(env_line, rc.t_p_s, rc.t_f_s)

    port, token = free_port(), secrets.token_hex(16)
    gateway = Gateway(ledger, pinned_model=rc.model, allowed_models={rc.model},
                      path_token=token, log_path=str(run_dir / "gateway_log.jsonl"),
                      thinking_log_path=str(run_dir / "thinking.jsonl"))
    trace = RunTrace(run_dir)
    server = register_glyph_server(session, trace, names=FREE_TOOLS)

    cli_path = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    overlay = FS.make_overlay(Path(rc.scratch_root) / "overlays" / f"{run_dir.name}.img",
                              rc.disk_gb, rc.image)
    scope = f"glyph-free-{run_dir.name}"   # readable in /proc/self/cgroup
    model = rc.model
    env = {
        "CLAUDE_CODE_USE_VERTEX": "1",
        "ANTHROPIC_VERTEX_BASE_URL": f"http://127.0.0.1:{port}/{token}",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH": "1",
        "ANTHROPIC_VERTEX_PROJECT_ID": "lightwell-devel",
        "CLOUD_ML_REGION": "global",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "IS_SANDBOX": "1",
        "DISABLE_AUTOUPDATER": "1",
        "BASH_MAX_TIMEOUT_MS": str(int(max(rc.t_p_s, rc.t_f_s) * 1000)),
    }
    spec = FS.ContainerSpec(run_dir=run_dir, work_dir=paths.root, cli_path=cli_path,
                            overlay=overlay, scope=scope, image=rc.image,
                            gpu_minor=rc.gpu_minor if rc.arm in FP.GPU_ARMS else None,
                            cpus=rc.cpus, mem_gb=rc.mem_gb, env=env)
    wrapper = FS.write_free_wrapper(spec)
    samplers = FS.Samplers(scope, run_dir, gpu_uuid)
    cli_stderr = run_dir / "cli_stderr.log"

    def _stderr(line: str) -> None:
        try:
            with open(cli_stderr, "a") as f:
                f.write(line.rstrip("\n") + "\n")
        except Exception:
            pass

    options = ClaudeAgentOptions(
        cli_path=str(wrapper), stderr=_stderr, cwd=str(paths.root), model=model,
        system_prompt=system, setting_sources=[], allowed_tools=FREE_ALLOWED_TOOLS,
        permission_mode="bypassPermissions", mcp_servers={"glyph": server},
        max_turns=100000, effort=rc.effort)

    phases: dict = {}
    killed: dict = {}
    status = {"run_status": "completed", "error": None, "gpu_check": None}

    def stop_practice():
        if session.should_switch:
            return "finish_practice"
        if ledger.over_safety_line(rc.usd_line):
            return "usd_line"
        return None

    final_line = {"usd": rc.usd_line}

    def stop_final():
        if getattr(session, "final_commit_path", None) is not None:
            return "submitted"
        if ledger.over_safety_line(final_line["usd"]):
            return "usd_line"
        return None

    def set_deadline(seconds: float) -> float:
        session.deadline = time.time() + seconds
        deadline_file.write_text(str(int(session.deadline)))
        return session.deadline

    async def sample_loop():
        i = 0
        while True:
            try:
                await asyncio.to_thread(samplers.sample_net)
                if gpu_uuid and i % 5 == 0:
                    await asyncio.to_thread(samplers.sample_gpu)
            except Exception as exc:
                print("sampler error: %r" % exc, file=sys.stderr)
            i += 1
            await asyncio.sleep(2)

    async def go():
        ready = asyncio.Event()
        gw_task = asyncio.create_task(gateway.serve_tcp("127.0.0.1", port, ready))
        await ready.wait()
        smp_task = asyncio.create_task(sample_loop())
        try:
            client = ClaudeSDKClient(options)
            async with client:
                status["gpu_check"] = await _wait_entry_check(spec, gpu_uuid)
                if not status["gpu_check"]["ok"]:
                    raise RuntimeError("container GPU check failed: %s" % status["gpu_check"])
                # practice
                t0 = time.time()
                deadline = set_deadline(rc.t_p_s)
                trace.phase_event("practice", "start", deadline=deadline)
                async def kill_p():
                    if "practice" not in killed:
                        killed["practice"] = await asyncio.to_thread(FS.kill_container_processes, scope)
                res = await drive_free_phase(
                    session, client, opener=FP.practice_opener(str(paths.task)),
                    deadline=deadline, should_stop=stop_practice, nudge=FP.practice_nudge,
                    on_deadline=kill_p)
                if "practice" not in killed:
                    killed["practice"] = await asyncio.to_thread(FS.kill_container_processes, scope)
                phases["practice"] = {**res, "seconds": round(time.time() - t0, 1)}
                trace.phase_event("practice", res["reason"], seconds=phases["practice"]["seconds"])

                # switch
                session.turns_practice = session.turns
                session.switch_to_final()
                workspace.write_test_file(paths, inst, test_id_of, probes=session.probes,
                                          probe_id_of=session.probe_id_of)
                t1 = time.time()
                final_line["usd"] = final_usd_line(rc.usd_line, ledger.spent_usd)
                deadline = set_deadline(rc.t_f_s)
                async def kill_f():
                    if "final" not in killed:
                        killed["final"] = await asyncio.to_thread(FS.kill_container_processes, scope)
                res = await drive_free_phase(
                    session, client,
                    opener=FP.final_opener(str(paths.task), rc.t_f_s, len(killed["practice"])),
                    deadline=deadline, should_stop=stop_final, nudge=FP.final_nudge,
                    on_deadline=kill_f)
                if "final" not in killed:
                    killed["final"] = await asyncio.to_thread(FS.kill_container_processes, scope)
                phases["final"] = {**res, "seconds": round(time.time() - t1, 1)}
                trace.phase_event("final", res["reason"], seconds=phases["final"]["seconds"])
        except Exception as exc:
            status["run_status"] = "error"
            status["error"] = repr(exc)[:500]
            print("free run error: %r" % exc, file=sys.stderr)
        finally:
            smp_task.cancel()
            gw_task.cancel()

    asyncio.run(go())

    session.turns_final = session.turns
    if getattr(session, "final_commit_path", None) is None:
        last = getattr(session, "last_checked_test_path", None)
        if last is not None:
            session.final_commit_path, session.final_commit = last, "auto_checked_path"
        else:
            session.final_commit = "none"
    session.run_status = status["run_status"]

    agent_home = run_dir / "agent_home"
    _copy_out_agent_home(spec, agent_home)
    artifacts = FS.collect_artifacts(spec)
    audit = samplers.audit()
    usage = transcript_usage(agent_home)
    tx_audit = transcript_audit(agent_home)

    committed = getattr(session, "final_commit_path", None)
    report = build_report(session, committed, test_id_of)
    report["free"] = {
        "protocol": "free",
        "t_p_s": rc.t_p_s, "t_f_s": rc.t_f_s,
        "phases": phases, "killed": {k: len(v) for k, v in killed.items()},
        "killed_detail": killed,
        "status": status,
        "usage": {
            "query_calls": sum(1 for e in session.events if e.get("kind") == "query"),
            "query_expressions": sum(e.get("n", 0) for e in session.events if e.get("kind") == "query"),
            "q_used": session.q_used,
            "reserved_binary_entries": len(inst.reserved_binary()),
            "refused_reserved": session.refused_reserved,
            "validation_submissions": session.submissions,
            "unary_coverage": round(unary_coverage(inst), 4),
            "transcript_tokens": usage,
        },
        "gpu": {"uuid": gpu_uuid, "minor": spec.gpu_minor,
                "seconds_held": audit["gpu_seconds_held"],
                "seconds_busy": audit["gpu_seconds_busy"],
                "foreign_gpu_kills": audit["foreign_gpu_kills"]},
        "network": {"n_endpoints": audit["n_endpoints"], "endpoints": audit["endpoints"],
                    "possible_llm_api": audit["possible_llm_api"], "transcript": tx_audit},
        "audit": {"llm_api_hosts": tx_audit["llm_api_hosts"],
                  "possible_llm_api_sockets": audit["possible_llm_api"],
                  "benchmark_lookups": tx_audit["benchmark_lookups"],
                  "foreign_gpu_kills": audit["foreign_gpu_kills"]},
        "artifacts": artifacts,
        "resources": {"cpus": rc.cpus, "mem_gb": rc.mem_gb, "disk_gb": rc.disk_gb,
                      "gpu_name": gpu_name, "image": rc.image},
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))

    try:
        rj = build_run_json(
            run_id="%s_%s" % (run_name, time.strftime("%Y%m%d-%H%M%S")),
            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            config={"protocol": "free", "arm": rc.arm, "preset": preset, "seed": seed,
                    "instance_id": instance_id, "model": model, "t_p_s": rc.t_p_s,
                    "t_f_s": rc.t_f_s, "usd_line": rc.usd_line, "effort": rc.effort,
                    "cpus": rc.cpus, "mem_gb": rc.mem_gb, "disk_gb": rc.disk_gb,
                    "gpu_uuid": gpu_uuid},
            task_readme=readme,
            prompts={"system": system,
                     "practice_opener": FP.practice_opener(str(paths.task)),
                     "final_opener": FP.final_opener(str(paths.task), rc.t_f_s,
                                                     len(killed.get("practice", [])))},
            report=report, run_dir=run_dir)
        (run_dir / "run.json").write_text(json.dumps(rj, indent=2))
    except Exception as exc:
        print("run.json export failed: %r" % exc, file=sys.stderr)

    trace.emit("report", overall=report["overall"], final_commit=report["covariates"]["final_commit"])
    trace.close()
    if rc.delete_overlay:
        import shutil
        try:
            overlay.unlink()
        except OSError:
            pass
        shutil.rmtree(spec.scratch_dir, ignore_errors=True)
    return report


def _entry_check(spec, gpu_uuid) -> dict:
    """What the container's entry script saw, against what was assigned."""
    try:
        txt = (spec.meta_dir / "entry.txt").read_text()
    except OSError:
        return {"ok": False, "entry": None}
    seen = [t for t in txt.split() if t.startswith("GPU-") or t.startswith("gpu_uuids=GPU-")]
    uuids = [t.split("=")[-1] for t in seen]
    ok = (uuids == [gpu_uuid]) if gpu_uuid else (len(uuids) == 0)
    return {"ok": ok, "assigned": gpu_uuid, "seen": uuids, "entry": txt[:400]}


async def _wait_entry_check(spec, gpu_uuid, timeout_s: float = 120.0) -> dict:
    """Wait for the entry script's report, then check the GPU it left."""
    t_end = time.time() + timeout_s
    while time.time() < t_end and not (spec.meta_dir / "entry.txt").exists():
        await asyncio.sleep(1)
    await asyncio.sleep(1)
    return _entry_check(spec, gpu_uuid)


def _copy_out_agent_home(spec, dest: Path) -> None:
    """Copy the CLI's ~/.claude out of the overlay, for the transcript export."""
    from glyph.v2 import free_sandbox as FS
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["apptainer", "exec", "--fakeroot", "--contain", "--no-home", "--cleanenv",
             "--overlay", f"{spec.overlay}:ro", "--bind", f"{dest}:/mnt/out",
             spec.image, "bash", "-c", "cp -r /root/.claude /mnt/out/ 2>/dev/null; true"],
            capture_output=True, timeout=600, env=FS.apptainer_env(), cwd="/tmp")
    except Exception as exc:
        print("agent_home copy failed: %r" % exc, file=sys.stderr)
