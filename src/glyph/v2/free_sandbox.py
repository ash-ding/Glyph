"""The free protocol's container: an open Ubuntu box with a root shell.

Spec: docs/superpowers/specs/2026-10-15-free-protocol-design.md, section 2.

The agent's Claude Code CLI runs inside an Apptainer container started with
`--fakeroot --contain --cleanenv --pid`, so the agent is uid 0 in its own PID
namespace with nothing of the host filesystem but its workspace. The
container shares the host network (the protocol gives the agent the open
internet), which is why the gateway listens on a per-run TCP port guarded by
a per-run path token rather than on a socket.

Host-side mechanisms, each verified on lumen1 (see the spec):

  * CPU and memory are capped by `systemd-run --user --scope` around the
    container; Apptainer's own --cpus/--memory do not work with --fakeroot.
  * Disk is capped by a fixed-size ext3 overlay file, created in-container
    with `mkfs.ext3 -d` because the host has no e2fsprogs.
  * One GPU: `--nv` binds every /dev/nvidiaN; the entry script unmounts all
    but the assigned one before handing over. In a user namespace the agent
    cannot mknod or mount, so the others cannot be restored. Device minor
    numbers do not follow nvidia-smi's index order on lumen1, so the minor
    for a GPU is found by probing (`probe_gpu_minors`).
  * Ending a phase kills every process in the container except the CLI and
    its ancestors (`kill_container_processes`), found through the scope's
    cgroup and the PID namespace.

stdlib only, so the harness can import it without the GPU stack.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_IMAGE = "/scratch/images/glyph-free.sif"
CLI_MOUNT = "/opt/claude"
GLYPH_MOUNT = "/opt/glyph"

# Hosts that serve hosted language models. Used only to flag outbound
# connections after the run (spec section 6, rule 1); a match is "possible",
# because several of these sit behind shared CDN addresses.
LLM_API_HOSTS = (
    "api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com", "openrouter.ai", "api.groq.com",
    "api.together.xyz", "api.together.ai", "api.deepseek.com", "api.mistral.ai",
    "api.cohere.com", "api.cohere.ai", "router.huggingface.co",
    "api-inference.huggingface.co", "api.x.ai", "api.fireworks.ai",
    "api.perplexity.ai", "api.replicate.com", "api.deepinfra.com",
)


# ---------------------------------------------------------------------------
# image and overlay
# ---------------------------------------------------------------------------

def apptainer_env() -> dict:
    env = dict(os.environ)
    env.setdefault("APPTAINER_CACHEDIR", "/scratch/.cache/apptainer")
    return env


def make_overlay(path: Path, size_gb: int, image: str = DEFAULT_IMAGE) -> Path:
    """Create a sparse ext3 overlay of size_gb at path, formatted in-container."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with open(path, "wb") as f:
        f.truncate(int(size_gb) * 1024 ** 3)
    script = ("mkdir -p /tmp/ov/upper /tmp/ov/work && "
              "mkfs.ext3 -q -F -d /tmp/ov /mnt/overlay.img")
    subprocess.run(
        ["apptainer", "exec", "--fakeroot", "--contain",
         "--bind", f"{path}:/mnt/overlay.img", image, "bash", "-c", script],
        check=True, capture_output=True, env=apptainer_env(), cwd="/tmp")
    return path


def probe_gpu_minors(image: str = DEFAULT_IMAGE) -> dict[int, str]:
    """{device minor: GPU UUID}, found by leaving one /dev/nvidiaN in a container."""
    out = subprocess.run(["bash", "-c", "ls /dev/nvidia[0-9]*"], capture_output=True,
                         text=True, check=True).stdout.split()
    minors = sorted(int(re.sub(r"\D", "", Path(d).name)) for d in out)
    found: dict[int, str] = {}
    for m in minors:
        script = (f"for d in /dev/nvidia[0-9]*; do [ $d = /dev/nvidia{m} ] || umount $d; done; "
                  "nvidia-smi --query-gpu=uuid --format=csv,noheader")
        r = subprocess.run(["apptainer", "exec", "--fakeroot", "--contain", "--nv", image,
                            "bash", "-c", script], capture_output=True, text=True,
                           env=apptainer_env(), cwd="/tmp")
        uuids = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("GPU-")]
        if len(uuids) == 1:
            found[m] = uuids[0]
    return found


# ---------------------------------------------------------------------------
# the container launch
# ---------------------------------------------------------------------------

@dataclass
class ContainerSpec:
    run_dir: Path
    work_dir: Path
    cli_path: Path
    overlay: Path
    scope: str                      # systemd scope unit name, without ".scope"
    image: str = DEFAULT_IMAGE
    gpu_minor: int | None = None    # None: no GPU in the container
    cpus: int = 24
    mem_gb: int = 160
    env: dict = field(default_factory=dict)

    @property
    def task_dir(self) -> Path:
        return Path(self.work_dir) / "task"

    @property
    def meta_dir(self) -> Path:
        return Path(self.run_dir) / "sandbox_meta"


ENTRY_SH = r"""#!/bin/bash
# Runs as uid 0 inside the container before the CLI. Leaves one GPU, or none.
GPU_MINOR="$1"; shift
if [ "$GPU_MINOR" != "none" ]; then
    for d in /dev/nvidia[0-9]*; do
        [ "$d" = "/dev/nvidia$GPU_MINOR" ] || umount "$d" 2>/dev/null
    done
fi
unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
{
    echo "uid=$(id -u)"
    echo "gpu_nodes=$(ls /dev/nvidia[0-9]* 2>/dev/null | tr '\n' ' ')"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "gpu_uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
    fi
    echo "nproc=$(nproc)"
} > /opt/glyph/meta/entry.txt 2>&1
cd "$GLYPH_WORKDIR"
exec "$@"
"""


def write_free_wrapper(spec: ContainerSpec) -> Path:
    """Write run_dir/wrapper.sh, the CLI path the SDK spawns, and return it."""
    run_dir = Path(spec.run_dir)
    work = Path(spec.work_dir).resolve()
    cli_path = Path(spec.cli_path).resolve()
    spec.meta_dir.mkdir(parents=True, exist_ok=True)
    entry = spec.meta_dir / "entry.sh"
    entry.write_text(ENTRY_SH)
    entry.chmod(0o755)

    env = {
        "PATH": "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "GLYPH_WORKDIR": str(work),
        **spec.env,
    }
    env_flags = " ".join(f"--env {shlex.quote(f'{k}={v}')}" for k, v in env.items())
    gpu = "none" if spec.gpu_minor is None else str(int(spec.gpu_minor))
    nv = "--nv " if spec.gpu_minor is not None else ""
    script = (
        "#!/bin/bash\n"
        "cd /tmp\n"
        f"systemctl --user reset-failed {shlex.quote(spec.scope)}.scope >/dev/null 2>&1\n"
        f"exec systemd-run --user --scope --quiet --unit={shlex.quote(spec.scope)} "
        f"-p CPUQuota={int(spec.cpus) * 100}% -p MemoryMax={int(spec.mem_gb)}G -p MemorySwapMax=0 "
        # An out-of-memory process is killed on its own; the default policy
        # would stop the whole scope, CLI included, and end the run.
        f"-p OOMPolicy=continue "
        f"apptainer exec --fakeroot --contain --cleanenv --pid {nv}"
        f"--overlay {shlex.quote(str(spec.overlay))} "
        f"--bind {shlex.quote(str(work))}:{shlex.quote(str(work))} "
        f"--bind {shlex.quote(str(spec.task_dir.resolve()))}:{shlex.quote(str(spec.task_dir.resolve()))}:ro "
        f"--bind {shlex.quote(str(cli_path.parent))}:{CLI_MOUNT}:ro "
        f"--bind {shlex.quote(str(spec.meta_dir.resolve()))}:{GLYPH_MOUNT}/meta "
        f"{env_flags} "
        f"{shlex.quote(spec.image)} "
        f"/bin/bash {GLYPH_MOUNT}/meta/entry.sh {gpu} {CLI_MOUNT}/{shlex.quote(cli_path.name)} \"$@\"\n"
    )
    wrapper = run_dir / "wrapper.sh"
    wrapper.write_text(script)
    wrapper.chmod(0o755)
    return wrapper


# ---------------------------------------------------------------------------
# processes in a run's scope
# ---------------------------------------------------------------------------

def scope_cgroup(scope: str) -> Path:
    uid = os.getuid()
    return Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{scope}.scope")


def scope_pids(scope: str) -> list[int]:
    root = scope_cgroup(scope)
    pids: list[int] = []
    if not root.exists():
        return pids
    for procs in root.rglob("cgroup.procs"):
        try:
            pids += [int(x) for x in procs.read_text().split()]
        except OSError:
            continue
    return sorted(set(pids))


@dataclass
class Proc:
    pid: int
    ppid: int
    comm: str
    exe: str
    pidns: str
    cmdline: str


def read_proc(pid: int) -> Proc | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        ppid = int(stat[stat.rindex(")") + 2:].split()[1])
        comm = stat[stat.index("(") + 1:stat.rindex(")")]
        try:
            exe = os.readlink(f"/proc/{pid}/exe")
        except OSError:
            exe = ""
        try:
            pidns = os.readlink(f"/proc/{pid}/ns/pid")
        except OSError:
            pidns = ""
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        return Proc(pid, ppid, comm, exe, pidns, cmd.strip()[:300])
    except (OSError, ValueError):
        return None


def select_victims(procs: list[Proc], host_pidns: str, cli_basename: str = "claude") -> tuple[Proc | None, list[Proc]]:
    """(the CLI process, the processes to kill) for a run's scope.

    Kept: the topmost process whose executable is the CLI, and its ancestors
    (systemd-run's exec chain, Apptainer's starter, the entry script). Killed:
    every other process inside the container's PID namespace. Processes in
    the host PID namespace (Apptainer's helpers, e.g. the overlay mount) are
    never touched.
    """
    by_pid = {p.pid: p for p in procs}
    clis = [p for p in procs if os.path.basename(p.exe.split(" (deleted)")[0]) == cli_basename]
    top = None
    for p in sorted(clis, key=lambda q: q.pid):
        parent = by_pid.get(p.ppid)
        if parent is None or os.path.basename(parent.exe.split(" (deleted)")[0]) != cli_basename:
            top = p
            break
    keep: set[int] = set()
    if top is not None:
        keep.add(top.pid)
        q = by_pid.get(top.ppid)
        while q is not None and q.pid not in keep:
            keep.add(q.pid)
            q = by_pid.get(q.ppid)
    victims = [p for p in procs if p.pid not in keep and p.pidns and p.pidns != host_pidns]
    return top, victims


def kill_container_processes(scope: str, grace_s: float = 5.0, cli_basename: str = "claude") -> list[dict]:
    """Kill every process in the container except the CLI; return what was killed."""
    host_pidns = os.readlink("/proc/self/ns/pid")
    killed: dict[int, dict] = {}
    for round_ in range(3):
        procs = [p for p in (read_proc(pid) for pid in scope_pids(scope)) if p is not None]
        _, victims = select_victims(procs, host_pidns, cli_basename)
        if not victims:
            break
        sig = signal.SIGTERM if round_ == 0 else signal.SIGKILL
        for v in victims:
            killed.setdefault(v.pid, {"pid": v.pid, "comm": v.comm, "cmdline": v.cmdline})
            try:
                os.kill(v.pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(grace_s if round_ == 0 else 1.0)
    return list(killed.values())


def cli_pid(scope: str, cli_basename: str = "claude") -> int | None:
    host_pidns = os.readlink("/proc/self/ns/pid")
    procs = [p for p in (read_proc(pid) for pid in scope_pids(scope)) if p is not None]
    top, _ = select_victims(procs, host_pidns, cli_basename)
    return top.pid if top else None


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------

def parse_proc_net(text: str, v6: bool) -> dict[int, tuple[str, int, str]]:
    """{socket inode: (remote ip, remote port, state)} from /proc/net/tcp{,6}."""
    states = {"01": "ESTABLISHED", "02": "SYN_SENT", "06": "TIME_WAIT", "08": "CLOSE_WAIT"}
    out: dict[int, tuple[str, int, str]] = {}
    for line in text.splitlines()[1:]:
        f = line.split()
        if len(f) < 10:
            continue
        rem, st, inode = f[2], f[3], int(f[9])
        ip_hex, port_hex = rem.split(":")
        if v6:
            b = bytes.fromhex(ip_hex)
            b = b"".join(b[i:i + 4][::-1] for i in range(0, 16, 4))
            ip = socket.inet_ntop(socket.AF_INET6, b)
            if ip.startswith("::ffff:"):
                ip = ip[7:]
        else:
            ip = socket.inet_ntop(socket.AF_INET, bytes.fromhex(ip_hex)[::-1])
        out[inode] = (ip, int(port_hex, 16), states.get(st, st))
    return out


def _is_local(ip: str) -> bool:
    return ip.startswith("127.") or ip in ("::1", "0.0.0.0", "::") or ip.startswith("::ffff:127.")


class Samplers:
    """GPU and network sampling for one run, driven by the harness's event loop."""

    def __init__(self, scope: str, run_dir: Path, gpu_uuid: str | None):
        self.scope = scope
        self.run_dir = Path(run_dir)
        self.gpu_uuid = gpu_uuid
        self.endpoints: dict[tuple[str, int], dict] = {}
        self.foreign_gpu_kills: list[dict] = []
        self.gpu_seconds_held = 0.0
        self.gpu_seconds_busy = 0.0
        self._last_gpu_t: float | None = None

    def sample_net(self) -> None:
        pids = scope_pids(self.scope)
        inodes: dict[int, int] = {}
        for pid in pids:
            try:
                for fd in os.listdir(f"/proc/{pid}/fd"):
                    try:
                        t = os.readlink(f"/proc/{pid}/fd/{fd}")
                    except OSError:
                        continue
                    if t.startswith("socket:["):
                        inodes[int(t[8:-1])] = pid
            except OSError:
                continue
        if not inodes:
            return
        table: dict[int, tuple[str, int, str]] = {}
        for name, v6 in (("tcp", False), ("tcp6", True)):
            try:
                table.update(parse_proc_net(Path(f"/proc/net/{name}").read_text(), v6))
            except OSError:
                pass
        now = time.time()
        for ino, pid in inodes.items():
            if ino not in table:
                continue
            ip, port, state = table[ino]
            if _is_local(ip) or port == 0:
                continue
            key = (ip, port)
            if key not in self.endpoints:
                p = read_proc(pid)
                rec = {"ip": ip, "port": port, "first_seen": now, "last_seen": now,
                       "comm": p.comm if p else None,
                       "cmdline": p.cmdline[:160] if p else None}
                self.endpoints[key] = rec
                with open(self.run_dir / "net_connections.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
            else:
                self.endpoints[key]["last_seen"] = now

    def sample_gpu(self) -> None:
        now = time.time()
        try:
            g = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
            a = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.TimeoutExpired):
            return
        mine = set(scope_pids(self.scope))
        apps = []
        for line in a.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) == 3 and f[1].isdigit():
                apps.append((f[0], int(f[1]), f[2]))
        rec = {"ts": now, "gpu": None, "apps": []}
        for line in g.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) == 3 and f[0] == self.gpu_uuid:
                rec["gpu"] = {"uuid": f[0], "mem_mib": _num(f[1]), "util": _num(f[2])}
        for uuid, pid, mem in apps:
            if pid not in mine:
                continue
            rec["apps"].append({"uuid": uuid, "pid": pid, "mem_mib": _num(mem)})
            if uuid != self.gpu_uuid:
                p = read_proc(pid)
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                ev = {"ts": now, "pid": pid, "gpu": uuid, "comm": p.comm if p else None}
                self.foreign_gpu_kills.append(ev)
        if self._last_gpu_t is not None and rec["gpu"] is not None:
            dt = now - self._last_gpu_t
            if any(x["uuid"] == self.gpu_uuid for x in rec["apps"]) or (rec["gpu"]["mem_mib"] or 0) > 0:
                self.gpu_seconds_held += dt
            if (rec["gpu"]["util"] or 0) > 0:
                self.gpu_seconds_busy += dt
        self._last_gpu_t = now
        with open(self.run_dir / "gpu_samples.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def audit(self) -> dict:
        """Resolve the endpoints seen against known LLM API hosts, after the run."""
        llm_ips: dict[str, str] = {}
        for host in LLM_API_HOSTS:
            try:
                for info in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP):
                    llm_ips[info[4][0]] = host
            except OSError:
                continue
        hits = []
        hosts = {}
        socket.setdefaulttimeout(2.0)
        for (ip, port), rec in self.endpoints.items():
            try:
                name = socket.gethostbyaddr(ip)[0]
            except OSError:
                name = None
            hosts[f"{ip}:{port}"] = name
            if ip in llm_ips:
                hits.append({"ip": ip, "port": port, "possible_host": llm_ips[ip], "rdns": name,
                             "first_seen": rec["first_seen"]})
        socket.setdefaulttimeout(None)
        return {"n_endpoints": len(self.endpoints), "endpoints": hosts,
                "possible_llm_api": hits, "foreign_gpu_kills": self.foreign_gpu_kills,
                "gpu_seconds_held": round(self.gpu_seconds_held, 1),
                "gpu_seconds_busy": round(self.gpu_seconds_busy, 1)}


def _num(x: str):
    try:
        return float(x)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# after the run
# ---------------------------------------------------------------------------

ARTIFACTS_SH = r"""
set +e
echo '{'
echo '"pip_freeze": '; (python -m pip freeze 2>/dev/null || true) | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().splitlines()))'
echo ', "conda_envs": '; (conda env list --json 2>/dev/null || echo '{}')
echo ', "hf_models": '; python3 - <<'PY'
import json, os
out = []
for root in ("/root/.cache/huggingface/hub", os.environ.get("HF_HOME", "") + "/hub"):
    if root and os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            if d.startswith("models--"):
                p = os.path.join(root, d)
                size = 0
                for dp, _, fs in os.walk(p):
                    for f in fs:
                        try: size += os.path.getsize(os.path.join(dp, f))
                        except OSError: pass
                out.append({"name": d[8:].replace("--", "/"), "bytes": size})
print(json.dumps(out))
PY
echo ', "checkpoint_dirs": '; python3 - <<'PY'
import json, os, sys
hits = []
for base in ("/root", sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GLYPH_WORKDIR", "")):
    if not base or not os.path.isdir(base):
        continue
    for dp, dns, fs in os.walk(base):
        if "/.cache/huggingface/hub" in dp or "/site-packages" in dp or "/pkgs/" in dp:
            continue
        if "config.json" in fs and any(f.endswith((".safetensors", ".bin")) for f in fs):
            try:
                cfg = json.load(open(os.path.join(dp, "config.json")))
                if "model_type" in cfg or "architectures" in cfg:
                    hits.append({"path": dp, "model_type": cfg.get("model_type")})
            except Exception:
                pass
print(json.dumps(hits[:200]))
PY
echo ', "overlay_df": '; df -B1 --output=used,size / 2>/dev/null | tail -1 | python3 -c 'import sys,json; v=sys.stdin.read().split(); print(json.dumps({"used": int(v[0]), "size": int(v[1])} if len(v)==2 else {}))'
echo '}'
"""


def collect_artifacts(spec: ContainerSpec, timeout_s: int = 600) -> dict:
    """Inventory the container after the run: environments, models, checkpoints, disk."""
    work = str(Path(spec.work_dir).resolve())
    try:
        r = subprocess.run(
            ["apptainer", "exec", "--fakeroot", "--contain", "--cleanenv",
             "--overlay", f"{spec.overlay}:ro", "--bind", f"{work}:{work}:ro",
             "--env", f"GLYPH_WORKDIR={work}", spec.image, "bash", "-c", ARTIFACTS_SH],
            capture_output=True, text=True, timeout=timeout_s, env=apptainer_env(), cwd="/tmp")
        data = json.loads(r.stdout)
    except Exception as exc:  # the inventory must never fail the run
        data = {"error": repr(exc)[:300]}
    try:
        data["workspace_bytes"] = int(subprocess.run(
            ["du", "-sb", work], capture_output=True, text=True, timeout=300).stdout.split()[0])
    except Exception:
        data["workspace_bytes"] = None
    return data
