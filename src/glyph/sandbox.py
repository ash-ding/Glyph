"""Running a solver the agent wrote, without trusting it.

The code arm's artifact is a Python program the teacher authored, and the
harness has to run it over 10^4 expressions to score it.  Three things have
to hold at once: it must not reach the network (the oracle is closed after
sealing, and a solver that phones home is not a sealed artifact), it must not
outlive its budget (an accidental infinite loop would stall a worker), and it
must not be able to read the answers off disk.

So the solver runs in a fresh subprocess, with a wall clock, an address-space
cap, and a working directory holding nothing.  Everything crosses the boundary
as JSON on stdin and stdout, which also means a crashed solver produces a
diagnosable failure rather than a hung parent.

This is isolation for a research harness, not a security boundary.  The code
comes from our own teacher, not from the internet; the point is to contain
mistakes and stop accidental leakage, not to withstand an adversary who
intends to escape.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache

#: Wrapped around the solver.  It must define `solve(expr: str) -> str`;
#: anything it prints is discarded so it cannot corrupt the result channel.
_HARNESS = r'''
import json, sys, io, contextlib

# Belt to the network namespace's braces, and the whole defence when no
# namespace is available.  A solver that reaches the network after sealing
# is not a sealed artifact.
try:
    import socket as _s
    def _closed(*a, **k):
        raise OSError("network access is closed after sealing")
    _s.socket = _closed
    _s.create_connection = _closed
    _s.socketpair = _closed
    _s.create_server = _closed
except Exception:
    pass

_src = json.loads(sys.stdin.readline())
_items = json.loads(sys.stdin.readline())

_ns = {}
_scratch = io.StringIO()
try:
    with contextlib.redirect_stdout(_scratch), contextlib.redirect_stderr(_scratch):
        exec(_src, _ns)
    fn = _ns.get("solve")
    if not callable(fn):
        raise NameError("the solver must define solve(expr) -> str")
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(0)

_out = []
with contextlib.redirect_stdout(_scratch), contextlib.redirect_stderr(_scratch):
    for _e in _items:
        try:
            _r = fn(_e)
            _out.append("" if _r is None else str(_r))
        except Exception as exc:
            _out.append("")
print(json.dumps({"ok": True, "answers": _out}))
'''


@dataclass
class SandboxResult:
    ok: bool
    answers: list[str] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0

    @property
    def crashed(self) -> bool:
        return not self.ok


@lru_cache(maxsize=1)
def _netns_prefix() -> tuple[str, ...]:
    """`unshare -rn`, if it actually works on this host.

    A user namespace plus a network namespace is real isolation and needs no
    privileges on a modern kernel.  Where it is unavailable -- macOS, a
    locked-down container -- this returns empty and the in-process socket
    block in the harness carries the load on its own.
    """
    exe = shutil.which("unshare")
    if not exe:
        return ()
    try:
        probe = subprocess.run([exe, "-rn", sys.executable, "-c", "pass"],
                               capture_output=True, timeout=20)
        return (exe, "-rn") if probe.returncode == 0 else ()
    except Exception:
        return ()


def isolation_report() -> str:
    return ("network namespace + in-process socket block" if _netns_prefix()
            else "in-process socket block only (no usable unshare)")


def _limits(seconds: int, address_space_mb: int):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
        cap = address_space_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()
    return apply


def run_solver(src: str, exprs: list[str], *, timeout: int = 300,
               address_space_mb: int = 4096, ledger=None) -> SandboxResult:
    """Answer `exprs` with the agent's solver.  One subprocess for the batch.

    A solver that crashes, hangs, or never defines `solve` returns
    `ok=False`; a solver that raises on an individual expression returns an
    empty answer for that one and keeps going, because a single bad input
    should cost one item rather than the whole run.

    Wall time is charged to `ledger` if one is given -- the code arm's test
    phase is not free, and pretending otherwise would flatter it against the
    arms whose inference time is obviously billed.
    """
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        # No proxy vars, no cloud credentials, no HF token: whatever the
        # parent has, the child does not inherit.
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_PROXY": "*",
    }
    payload = json.dumps(src) + "\n" + json.dumps(exprs) + "\n"

    with tempfile.TemporaryDirectory(prefix="glyph-solver-") as cwd:
        timer = ledger.gpu_timer("sandbox") if ledger is not None else None
        if timer is not None:
            timer.__enter__()
        try:
            proc = subprocess.run(
                [*_netns_prefix(), sys.executable, "-I", "-S", "-c", _HARNESS],
                input=payload, capture_output=True, text=True,
                cwd=cwd, env=env, timeout=timeout,
                preexec_fn=_limits(timeout, address_space_mb),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, error=f"timed out after {timeout}s")
        finally:
            if timer is not None:
                timer.__exit__(None, None, None)

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return SandboxResult(False, error=f"exit {proc.returncode}: "
                                          f"{tail[-1] if tail else 'no stderr'}")
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return SandboxResult(False, error="solver produced no output")
    try:
        got = json.loads(line[-1])
    except json.JSONDecodeError:
        return SandboxResult(False, error=f"unparseable output: {line[-1][:120]!r}")

    if not got.get("ok"):
        return SandboxResult(False, error=got.get("error", "unknown failure"))
    answers = got.get("answers", [])
    if len(answers) != len(exprs):
        return SandboxResult(False, error=f"expected {len(exprs)} answers, "
                                          f"got {len(answers)}")
    return SandboxResult(True, answers=answers)
