#!/usr/bin/env python3
"""Run the free-protocol sweep in waves (spec 2026-10-15-free-protocol-design.md, section 8).

One wave per practice length; every (instance, arm) pair of a wave runs at
once, each train run on its own GPU. A wave starts when the previous one has
finished. Progress goes to <out-root>/<tag>_status.json; each run's output to
<out-root>/logs/<run>.log. Re-running skips runs that already have a report.

    python tools/run_free_sweep.py --out-root /scratch/glyph_free/runs
    python tools/run_free_sweep.py --t-p 1 --instances mid_3      # the pilot pair
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def run_name(inst: str, arm: str, t_p: float) -> str:
    return f"{inst}_{arm}_tp{t_p:g}h"


def report_state(run_dir: Path) -> str | None:
    """'done' for a usable report, 'error' for a report of a run that failed, None if none."""
    p = run_dir / "report.json"
    if not p.exists():
        return None
    try:
        r = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return "error"
    return "error" if r.get("covariates", {}).get("run_status") == "error" else "done"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["low_3", "mid_3", "high_3"])
    ap.add_argument("--arms", nargs="+", default=["train", "no_train"])
    ap.add_argument("--t-p", nargs="+", type=float, default=[1.0, 2.0, 4.0], help="hours, one wave each")
    ap.add_argument("--t-f", type=float, default=30.0, help="minutes")
    ap.add_argument("--out-root", default="/scratch/glyph_free/runs")
    ap.add_argument("--usd-line", type=float, default=500.0)
    ap.add_argument("--cpus", type=int, default=24)
    ap.add_argument("--mem-gb", type=int, default=160)
    ap.add_argument("--disk-gb", type=int, default=35)
    ap.add_argument("--gpu-minors", nargs="+", type=int, default=None,
                    help="device minors to use for train runs (default: every GPU found)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tag", default="sweep", help="names the status file: <out-root>/<tag>_status.json")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="run names to leave alone (e.g. ones already running elsewhere)")
    a = ap.parse_args(argv)

    from glyph.v2.free_sandbox import probe_gpu_minors
    out = Path(a.out_root)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    status_path = out / f"{a.tag}_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"runs": {}}
    minors = a.gpu_minors if a.gpu_minors is not None else sorted(probe_gpu_minors())
    status["gpu_minors"] = minors
    status["started"] = status.get("started") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def save():
        status["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(status_path)

    for t_p in a.t_p:
        procs = []
        free_minors = list(minors)
        for inst in a.instances:
            for arm in a.arms:
                name = run_name(inst, arm, t_p)
                st = report_state(out / name)
                if st == "done":
                    status["runs"].setdefault(name, {})["state"] = "done"
                    continue
                if name in a.skip:
                    status["runs"].setdefault(name, {})["state"] = "skipped"
                    continue
                if st == "error" or (out / name).exists():
                    aside = out / f"{name}.failed-{time.strftime('%H%M%S')}"
                    (out / name).rename(aside)
                    status.setdefault("set_aside", []).append(str(aside))
                cmd = [a.python, "-m", "glyph.v2", "run", "--protocol", "free",
                       "--instance-id", inst, "--arm", arm, "--t-p", str(t_p), "--t-f", str(a.t_f),
                       "--usd-line", str(a.usd_line), "--cpus", str(a.cpus), "--mem-gb", str(a.mem_gb),
                       "--disk-gb", str(a.disk_gb), "--out-root", str(out)]
                minor = None
                if arm in ("train", "silent"):
                    if not free_minors:
                        raise SystemExit("not enough GPUs for wave t_p=%g" % t_p)
                    minor = free_minors.pop(0)
                    cmd += ["--gpu-minor", str(minor)]
                log = open(out / "logs" / f"{name}.log", "a")
                env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONUNBUFFERED="1")
                env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
                env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={env['XDG_RUNTIME_DIR']}/bus")
                p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO), env=env)
                procs.append((name, p, log))
                status["runs"][name] = {"state": "running", "pid": p.pid, "gpu_minor": minor,
                                        "t_p": t_p, "started": time.time()}
                save()
                time.sleep(5)
        while any(p.poll() is None for _, p, _ in procs):
            time.sleep(30)
            save()
        for name, p, log in procs:
            log.close()
            r = status["runs"][name]
            r["rc"] = p.returncode
            r["ended"] = time.time()
            r["state"] = "done" if report_state(out / name) == "done" else "failed"
        save()
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
