"""The free protocol's container (src/glyph/v2/free_sandbox.py) and the gateway's run token."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import anyio
import pytest

from glyph.v2 import free_sandbox as F
from glyph.v2.gateway import Gateway
from glyph.v2.ledger import Ledger

PIN = "claude-opus-4-8"
PATH = "/projects/p/locations/global/publishers/anthropic/models/%s:rawPredict" % PIN


def test_gateway_run_token_is_required_and_stripped():
    seen = []

    async def fwd(method, url, headers, body):
        seen.append(url)
        return 200, {}, json.dumps({"usage": {"input_tokens": 3, "output_tokens": 2}}).encode()

    ledger = Ledger()
    gw = Gateway(ledger, pinned_model=PIN, forward_fn=fwd, path_token="t0k")

    async def go():
        ok = await gw.handle_request("POST", "/t0k" + PATH, {}, b"{}")
        missing = await gw.handle_request("POST", PATH, {}, b"{}")
        wrong = await gw.handle_request("POST", "/nope" + PATH, {}, b"{}")
        prefix_only = await gw.handle_request("POST", "/t0kX" + PATH, {}, b"{}")
        return ok, missing, wrong, prefix_only

    ok, missing, wrong, prefix_only = anyio.run(go)
    assert ok[0] == 200
    assert seen == ["https://aiplatform.googleapis.com/v1" + PATH]
    assert missing[0] == wrong[0] == prefix_only[0] == 401
    assert len(seen) == 1
    assert ledger.spent_usd > 0
    billed = ledger.spent_usd
    assert [r["status"] for r in gw.records] == [200, 401, 401, 401]
    assert ledger.spent_usd == billed


def test_gateway_without_token_is_unchanged():
    async def fwd(method, url, headers, body):
        return 200, {}, b'{"usage": {}}'

    gw = Gateway(Ledger(), pinned_model=PIN, forward_fn=fwd)
    assert anyio.run(gw.handle_request, "POST", PATH, {}, b"{}")[0] == 200


def _spec(tmp_path, gpu=None):
    cli = tmp_path / "cli" / "claude"
    cli.parent.mkdir()
    cli.write_text("")
    (tmp_path / "work" / "task").mkdir(parents=True)
    return F.ContainerSpec(run_dir=tmp_path, work_dir=tmp_path / "work", cli_path=cli,
                           overlay=tmp_path / "run.img", scope="glyph-free-x", gpu_minor=gpu,
                           cpus=24, mem_gb=160, env={"A": "b c"})


def test_wrapper_train_arm(tmp_path):
    w = F.write_free_wrapper(_spec(tmp_path, gpu=5)).read_text()
    assert "systemctl --user reset-failed glyph-free-x.scope" in w
    assert "--unit=glyph-free-x" in w
    assert "-p CPUQuota=2400% -p MemoryMax=160G -p MemorySwapMax=0" in w
    assert "-p OOMPolicy=continue" in w
    assert "apptainer exec --fakeroot --contain --no-home --cleanenv --pid --nv" in w
    assert f"--workdir {(tmp_path / 'ctr_tmp').resolve()} " in w
    work = str((tmp_path / "work").resolve())
    assert f"--bind {work}:{work} " in w
    assert f"--bind {work}/task:{work}/task:ro" in w
    assert "--env 'A=b c'" in w
    assert "/opt/glyph/meta/entry.sh 5 /opt/claude/claude \"$@\"" in w
    entry = (tmp_path / "sandbox_meta" / "entry.sh").read_text()
    assert 'umount "$d"' in entry and "exec \"$@\"" in entry


def test_wrapper_no_train_arm_has_no_gpu(tmp_path):
    w = F.write_free_wrapper(_spec(tmp_path, gpu=None)).read_text()
    assert "--nv" not in w
    assert "entry.sh none /opt/claude/claude" in w


def _p(pid, ppid, exe, ns="ctr"):
    return F.Proc(pid, ppid, os.path.basename(exe), exe, ns, exe)


def test_select_victims_keeps_cli_and_ancestors_only():
    procs = [
        _p(10, 1, "/usr/libexec/apptainer/bin/starter", ns="host"),
        _p(11, 10, "/usr/libexec/apptainer/bin/starter"),            # appinit, in the container
        _p(12, 10, "/usr/bin/fuse-overlayfs", ns="host"),
        _p(20, 11, "/opt/claude/claude"),                            # the CLI
        _p(21, 20, "/opt/claude/claude"),                            # a CLI helper
        _p(30, 20, "/usr/bin/bash"),
        _p(31, 30, "/opt/conda/bin/python"),
        _p(40, 11, "/usr/bin/sleep"),                                # reparented daemon
    ]
    top, victims = F.select_victims(procs, host_pidns="host")
    assert top.pid == 20
    assert sorted(v.pid for v in victims) == [21, 30, 31, 40]


def test_select_victims_without_cli_kills_container_only():
    procs = [_p(10, 1, "/usr/bin/starter", ns="host"), _p(11, 10, "/usr/bin/sleep")]
    top, victims = F.select_victims(procs, host_pidns="host")
    assert top is None and [v.pid for v in victims] == [11]


def test_parse_proc_net():
    v4 = (" sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
          " 0: 0100007F:4E20 0100007F:B5F0 01 00000000:00000000 00:00000000 00000000 1000 0 111 1\n"
          " 1: 0A00000A:D2F4 22D7BA97:01BB 01 00000000:00000000 00:00000000 00000000 1000 0 222 1\n")
    t = F.parse_proc_net(v4, v6=False)
    assert t[111] == ("127.0.0.1", 46576, "ESTABLISHED")
    assert t[222] == ("151.186.215.34", 443, "ESTABLISHED")
    v6 = (" sl local rem st a b c d e f inode\n"
          " 0: 00000000000000000000000000000000:1F90 0000000000000000FFFF00000A0B0C0D:0050 02 0 0 0 0 0 333 1\n")
    t6 = F.parse_proc_net(v6, v6=True)
    assert t6[333] == ("13.12.11.10", 80, "SYN_SENT")


needs_host = pytest.mark.skipif(
    not (shutil.which("apptainer") and Path(F.DEFAULT_IMAGE).exists() and shutil.which("systemd-run")),
    reason="needs apptainer, systemd-run and the glyph-free image")


@pytest.mark.slow
@needs_host
def test_container_isolation_on_host(tmp_path_factory):
    base = Path("/scratch") / ("glyph_free_test_%d" % os.getpid())
    if base.exists():
        shutil.rmtree(base)
    (base / "work" / "task").mkdir(parents=True)
    (base / "cli").mkdir()
    shutil.copy("/usr/bin/bash", base / "cli" / "claude")
    try:
        minors = F.probe_gpu_minors()
        gpu = min(minors) if minors else None
        F.make_overlay(base / "run.img", 2)
        scope = "glyph-free-test-%d" % os.getpid()
        spec = F.ContainerSpec(run_dir=base, work_dir=base / "work", cli_path=base / "cli" / "claude",
                               overlay=base / "run.img", scope=scope, gpu_minor=gpu, cpus=2, mem_gb=1)
        w = F.write_free_wrapper(spec)
        (spec.meta_dir / "inner.sh").write_text(
            "M=/opt/glyph/meta; sleep 1000 &\n"
            "mknod /tmp/x c 195 0 2>/dev/null; echo $? > $M/mknod\n"
            "python3 -c \"x=bytearray(2*1024**3); x[::4096]=b'1'*len(x[::4096])\"; echo $? > $M/mem\n"
            "id -u > $M/uid\n"
            "while true; do sleep 1; done\n")
        p = subprocess.Popen([str(w), "/opt/glyph/meta/inner.sh"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        deadline = time.time() + 120
        while time.time() < deadline and not (spec.meta_dir / "uid").exists():
            time.sleep(1)
        meta = spec.meta_dir
        assert (meta / "uid").read_text().strip() == "0"
        assert (meta / "mknod").read_text().strip() != "0"
        assert (meta / "mem").read_text().strip() == "137"
        entry = (meta / "entry.txt").read_text()
        if gpu is not None:
            assert minors[gpu] in entry and entry.count("GPU-") == 1
        cli = F.cli_pid(scope)
        assert cli is not None
        killed = F.kill_container_processes(scope, grace_s=1)
        assert any("sleep 1000" in k["cmdline"] for k in killed)
        assert F.read_proc(cli) is not None
        os.kill(cli, 9)
        p.wait(timeout=60)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _fake_nodes(root, *cpulists):
    for k, s in enumerate(cpulists):
        d = root / f"node{k}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(s + "\n")
    return str(root)


def test_cpulist_round_trip():
    assert F.parse_cpulist("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert F.format_cpulist([11, 10, 8, 3, 2, 1, 0]) == "0-3,8,10-11"
    assert F.format_cpulist(list(range(24, 48))) == "24-47"


def test_cpu_blocks_stay_inside_a_numa_node(tmp_path):
    nodes = _fake_nodes(tmp_path / "node", "0-79", "80-159")
    blocks = F.cpu_blocks(24, nodes)
    assert [F.format_cpulist(b) for b in blocks] == ["0-23", "24-47", "48-71", "80-103", "104-127", "128-151"]


def test_cpu_leases_are_disjoint_and_released(tmp_path):
    nodes = _fake_nodes(tmp_path / "node", "0-47")
    a = F.CpuLease(tmp_path / "locks", 24, nodes)
    b = F.CpuLease(tmp_path / "locks", 24, nodes)
    assert {a.cpulist, b.cpulist} == {"0-23", "24-47"}
    with pytest.raises(RuntimeError, match="no free block"):
        F.CpuLease(tmp_path / "locks", 24, nodes)
    a.release()
    c = F.CpuLease(tmp_path / "locks", 24, nodes)
    assert c.cpulist == a.cpulist
    b.release(); c.release()


def test_wrapper_pins_the_container_to_its_cpu_block(tmp_path):
    spec = _spec(tmp_path, gpu=5)
    spec.cpu_list = "24-47"
    w = F.write_free_wrapper(spec).read_text()
    assert "taskset -c 24-47 apptainer exec --fakeroot" in w
    (tmp_path / "b").mkdir()
    assert "taskset" not in F.write_free_wrapper(_spec(tmp_path / "b", gpu=None)).read_text()
