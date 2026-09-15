"""Generates the bwrap sandbox wrapper script for a v2 run.

Productionized from the spike that passed S1 Stage D
(/tmp/glyph_v2_spikes/wrapper.sh + bridge.py): a real CLI turn reached
Vertex only via the bind-mounted gw.sock, with no /home, no /dev/nvidia0,
and no direct network inside the sandbox.

Differences from the spike:
  - work/task is additionally bind-mounted read-only at /work/task (on top
    of the read-write /work bind), so the agent can read task files but
    never edit them in place.
  - the bwrap invocation is prefixed with `taskset -c <cpus> prlimit
    --as=<mem_limit_bytes> --nproc=<cap> --` to cap CPU affinity and
    address-space/process-count for the whole sandboxed run. If taskset or
    prlimit are not present on the host (see `have_resource_limiters`),
    that prefix is omitted entirely and write_wrapper falls back to an
    uncapped bwrap invocation.

HOST NOTE (lumen1, discovered while productionizing this task): the brief's
literal `--nproc=512` is a *fixed* RLIMIT_NPROC, and RLIMIT_NPROC counts
against the real UID's total process/thread count *system-wide*, not just
the processes under this one sandboxed run. On lumen1 the same user already
has on the order of 10-15k threads running system-wide (other sessions,
notebooks, etc.), so a bare `--nproc=512` makes bwrap's own unshare()/clone()
call fail immediately with "Creating new namespace failed: Resource
temporarily unavailable" -- 100% reproducible, not a flake (verified: 0/60
successes at a fixed 512 cap on this host; --as alone, without --nproc, is
fine). This is a property of prlimit --nproc + a busy shared host, not of
the sandbox's isolation. To keep the *spirit* of the brief's cap (a ~512
process/thread budget for the sandboxed run itself) while not immediately
tripping over unrelated host load, `_nproc_limit()` below sizes the cap as
"current system-wide thread count for this uid, plus a fixed per-run
budget" instead of a bare constant. This is still an inherent limitation of
RLIMIT_NPROC on a busy multi-tenant host (see that function's docstring)
and is called out again in task-15-report.md.

stdlib only.
"""
from __future__ import annotations

import os
import shutil
import textwrap
from pathlib import Path

# The default bwrap binary location on the target hosts (lumen1/2/3), a
# homebrew (linuxbrew) install rather than a system package.
DEFAULT_BWRAP = "/home/linuxbrew/.linuxbrew/bin/bwrap"


def have_resource_limiters() -> bool:
    """True iff both `taskset` and `prlimit` are on PATH. write_wrapper uses
    this to decide whether it can emit the CPU/mem-capping prefix."""
    return shutil.which("taskset") is not None and shutil.which("prlimit") is not None


# Per-run process/thread budget added on top of whatever this uid is
# already using system-wide, when sizing the --nproc cap. See the HOST NOTE
# in the module docstring: RLIMIT_NPROC is charged against the real uid's
# *entire* system-wide process/thread count, not just this sandboxed run's,
# so a bare constant cap (as literally written in the task-15 brief) fails
# outright on a busy shared host. This budget is added to a live count of
# the current uid's threads instead, so the run still gets roughly the
# intended headroom without colliding with pre-existing host load.
NPROC_BUDGET = 512

# Fallback --nproc cap used only if the current uid's thread count can't be
# read from /proc (e.g. /proc unavailable). Generous on purpose since it is
# a last resort, not the common path.
_NPROC_FALLBACK_CAP = 8192


def _current_uid_thread_count() -> int | None:
    """Best-effort count of processes+threads already charged against
    RLIMIT_NPROC for this process's real uid, by scanning /proc. Returns
    None if /proc can't be read this way (e.g. non-Linux)."""
    try:
        uid = os.getuid()
        pids = [e for e in os.listdir("/proc") if e.isdigit()]
    except OSError:
        return None

    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("Uid:"):
                        if int(line.split()[1]) == uid:
                            total += len(os.listdir(f"/proc/{pid}/task"))
                        break
        except (OSError, ValueError):
            continue
    return total


def _nproc_limit() -> int:
    """The --nproc cap to pass to prlimit: current system-wide thread count
    for this uid, plus NPROC_BUDGET headroom for the sandboxed run itself.
    Falls back to _NPROC_FALLBACK_CAP if the current count can't be read."""
    current = _current_uid_thread_count()
    if current is None:
        return _NPROC_FALLBACK_CAP
    return current + NPROC_BUDGET


def write_wrapper(
    run_dir: Path,
    cli_path: Path,
    gw_sock: Path,
    *,
    cpus: str = "0",
    mem_limit_bytes: int = 2 * 1024**3,
    bwrap: str = DEFAULT_BWRAP,
    extra_setenv: dict[str, str] | None = None,
) -> Path:
    """Write run_dir/"wrapper.sh" (chmod +x) and return its path.

    Creates run_dir/work, run_dir/work/task, run_dir/agent_home if they do
    not already exist (Task 17 populates work/task's contents via
    workspace.py -- this function only needs the directories to exist so
    the bwrap binds succeed).

    cli_path's parent directory is bind-mounted read-only at /opt/cli_dir
    and cli_path itself is run (by basename) as /opt/cli_dir/<basename>.
    bridge.py (this package's own src/glyph/v2/bridge.py) is bind-mounted
    read-only at /opt/bridge.py and is what the sandboxed process actually
    execs; it proxies 127.0.0.1:8787 -> gw_sock and then os.execv's the CLI.

    If `have_resource_limiters()` is False (taskset/prlimit missing from
    PATH), the `taskset -c <cpus> prlimit --as=... --nproc=...` prefix is
    omitted and the generated script runs bwrap directly, uncapped. See
    `_nproc_limit` for how the --nproc value itself is sized -- it is NOT
    the brief's bare literal 512 (see the module's HOST NOTE).
    """
    run_dir = Path(run_dir)
    cli_path = Path(cli_path)
    gw_sock = Path(gw_sock)

    work_dir = run_dir / "work"
    task_dir = work_dir / "task"
    agent_home_dir = run_dir / "agent_home"
    # A reused run_dir must not hand the CLI a previous run's session state.
    # A SIGKILLed run leaves half-written ~/.claude sessions in agent_home;
    # inheriting them crashed the CLI (exit 250) on a later large turn. Start
    # every run from a clean agent HOME. (work/task is repopulated by
    # workspace.py each run, so only agent_home needs clearing here.)
    if agent_home_dir.exists():
        import shutil
        shutil.rmtree(agent_home_dir, ignore_errors=True)
        if agent_home_dir.exists():
            import sys
            sys.stderr.write(
                "sandbox: warning: could not fully clear %s; stale CLI "
                "session state may remain\n" % agent_home_dir)
    for d in (work_dir, task_dir, agent_home_dir):
        d.mkdir(parents=True, exist_ok=True)

    bridge_path = Path(__file__).resolve().parent / "bridge.py"
    cli_dir = cli_path.resolve().parent
    cli_basename = cli_path.name

    setenv_pairs = [
        ("HOME", "/agent_home"),
        ("PATH", "/usr/bin:/bin"),
        ("CLAUDE_CODE_USE_VERTEX", "1"),
        ("ANTHROPIC_VERTEX_BASE_URL", "http://127.0.0.1:8787"),
        ("CLAUDE_CODE_SKIP_VERTEX_AUTH", "1"),
        ("ANTHROPIC_VERTEX_PROJECT_ID", "lightwell-devel"),
        ("CLOUD_ML_REGION", "global"),
    ]
    if extra_setenv:
        setenv_pairs.extend(extra_setenv.items())
    setenv_flags = " ".join(f"--setenv {k} {_sh_quote(v)}" for k, v in setenv_pairs)

    if have_resource_limiters():
        limit_prefix = (
            f"taskset -c {_sh_quote(cpus)} "
            f"prlimit --as={int(mem_limit_bytes)} --nproc={_nproc_limit()} -- "
        )
    else:
        limit_prefix = ""

    bwrap_cmd = (
        f"{_sh_quote(bwrap)} --unshare-all --die-with-parent --new-session --clearenv "
        f"--ro-bind /usr /usr --symlink usr/lib64 /lib64 --symlink usr/lib /lib --symlink usr/bin /bin "
        f"--proc /proc --dev /dev --tmpfs /tmp "
        f"--ro-bind {_sh_quote(str(cli_dir))} /opt/cli_dir "
        f"--ro-bind {_sh_quote(str(bridge_path))} /opt/bridge.py "
        f"--bind {_sh_quote(str(work_dir))} /work "
        f"--ro-bind {_sh_quote(str(task_dir))} /work/task "
        f"--bind {_sh_quote(str(agent_home_dir))} /agent_home "
        f"--ro-bind {_sh_quote(str(gw_sock))} /run/gw.sock "
        f"{setenv_flags} "
        f"--chdir /work "
        f"/usr/bin/python3 /opt/bridge.py -- /opt/cli_dir/{_sh_quote(cli_basename)} \"$@\""
    )

    script = "#!/bin/bash\n" + f"exec {limit_prefix}{bwrap_cmd}\n"

    wrapper_path = run_dir / "wrapper.sh"
    wrapper_path.write_text(script)
    wrapper_path.chmod(0o755)
    return wrapper_path


def _sh_quote(s: str) -> str:
    """Minimal POSIX shell single-quoting: safe for the fixed set of paths
    and env values this module ever generates (no embedded newlines)."""
    return "'" + str(s).replace("'", "'\\''") + "'"
