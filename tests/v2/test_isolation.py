"""Permanent security regression test for the v2 bwrap sandbox.

Host-side only: no gateway, no API key, no GPU. Runs the real wrapper.sh
under bwrap with a STUB "cli" that probes the sandbox from the inside and
writes what it found to /work/out.txt. Asserts the sandbox hides the host
/home (repo, gcloud creds included), the GPU device nodes, the network, and
clears credential env vars -- and that /work/task is read-only.

This is the security contract for protocol v2. Never weaken these
assertions to make the test pass.
"""
import os, subprocess, pathlib, textwrap, pytest

from glyph.v2.sandbox import write_wrapper

BWRAP = "/home/linuxbrew/.linuxbrew/bin/bwrap"


@pytest.mark.skipif(not pathlib.Path(BWRAP).exists(), reason="bwrap not installed")
def test_sandbox_hides_everything(tmp_path):
    run = tmp_path / "run"
    (run / "work" / "task").mkdir(parents=True)
    (run / "agent_home").mkdir(parents=True)
    (run / "work" / "task" / "readme.txt").write_text("ro")

    # a stub 'cli' that probes the sandbox and writes /work/out.txt, then exits 0
    cli_dir = tmp_path / "clid"
    cli_dir.mkdir()
    stub = cli_dir / "claude"
    stub.write_text(textwrap.dedent('''\
        #!/usr/bin/env python3
        import os, socket, pathlib
        out = []
        out.append("home:%s" % ("PRESENT" if os.path.isdir("/home") else "MISSING"))
        out.append("repo:%s" % ("PRESENT" if os.path.exists("/home/asherding/code/Glyph") else "MISSING"))
        out.append("gcloud:%s" % ("PRESENT" if os.path.exists("/home/asherding/.config/gcloud") else "MISSING"))
        out.append("nvidia:%s" % ("PRESENT" if os.path.exists("/dev/nvidia0") else "MISSING"))
        out.append("creds_env:%s" % ("SET" if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") else "UNSET"))
        try:
            s = socket.socket(); s.settimeout(2); s.connect(("1.1.1.1", 443)); out.append("net:OPEN"); s.close()
        except Exception:
            out.append("net:UNREACHABLE")
        try:
            open("/work/task/x", "w").close(); out.append("taskwrite:RW")
        except Exception:
            out.append("taskwrite:READONLY")
        pathlib.Path("/work/out.txt").write_text("\\n".join(out))
    '''))
    stub.chmod(0o755)

    gw = run / "gw.sock"
    gw.write_bytes(b"")  # dummy; bridge won't dial it (stub makes no 8787 connection)

    wrapper = write_wrapper(run, cli_path=stub, gw_sock=gw, cpus="0", mem_limit_bytes=2 * 1024**3, bwrap=BWRAP)
    subprocess.run(["/bin/bash", str(wrapper)], check=True, timeout=60)

    out = (run / "work" / "out.txt").read_text()
    assert "home:MISSING" in out
    assert "repo:MISSING" in out
    assert "gcloud:MISSING" in out
    assert "nvidia:MISSING" in out
    assert "creds_env:UNSET" in out
    assert "net:UNREACHABLE" in out
    assert "taskwrite:READONLY" in out


def test_write_wrapper_clears_stale_agent_home(tmp_path):
    """A reused run_dir must not inherit a prior (killed) run's CLI session
    state: write_wrapper clears agent_home so the CLI always starts fresh.
    Inheriting stale ~/.claude sessions crashed the CLI (exit 250) on a later
    large turn during the train-arm E2E."""
    run = tmp_path / "run"
    home = run / "agent_home"
    home.mkdir(parents=True)
    stale = home / ".claude" / "sessions" / "4.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{stale}")

    cli = tmp_path / "clid" / "claude"
    cli.parent.mkdir(parents=True)
    cli.write_text("#!/bin/sh\n")

    write_wrapper(run, cli, run / "gw.sock")

    assert home.is_dir()           # recreated for the bind mount
    assert not stale.exists()      # stale CLI session state cleared
