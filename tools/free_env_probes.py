#!/usr/bin/env python3
"""How a free-protocol run explored its own machine: what it checked, when, and what it learned.

Reads the CLI session files a run copied out (main agent and subagents) and
reports, per kind of probe, the minute of the first probe after practice
started and how many there were. It also reports when a shell command first
printed evidence of a GPU (text the agent read, such as the README, does not
count), and how long the GPU was busy (from report.json). Built for the `silent` arm, which is told nothing about the
machine, but it works on any run.

    python tools/free_env_probes.py /scratch/glyph_free/runs6/low_3_silent_tp1h [...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROBES = {
    "gpu": r"nvidia-smi|torch\.cuda|cuda\.is_available|/dev/nvidia|nvcc|CUDA_VISIBLE_DEVICES",
    "cpu": r"\bnproc\b|lscpu|/proc/cpuinfo|os\.cpu_count|multiprocessing\.cpu_count|cpu\.max",
    "memory": r"\bfree\s+-|/proc/meminfo|memory\.max|psutil\.virtual_memory",
    "disk": r"\bdf\s+-|\bdu\s+-sh\s+/\s*$|statvfs",
    "os": r"/etc/os-release|\buname\b|lsb_release",
    "python_env": r"python3?\s+--version|sys\.version|\bconda\s+(env\s+list|info|list)\b|\bpip3?\s+(list|freeze)\b|which\s+python",
    "network": r"\bping\b|curl\s+-?I?\s*https?://(pypi|download\.pytorch|www\.google)|pip3?\s+download",
}
GPU_SEEN = re.compile(r"NVIDIA H100|Tesla|A100|cuda[^\n]{0,20}True|is_available\(\)[^\n]{0,10}True|GPU-[0-9a-f]{8}", re.I)
# The environment line the train and no_train arms are shown; printing it back is not discovery.
OUR_LINE = re.compile(r"This container has [^.]*\.")


def session_files(run_dir: Path) -> tuple[Path, list[Path]]:
    files = list((run_dir / "agent_home/.claude/projects").rglob("*.jsonl"))
    mains = [f for f in files if "subagents" not in f.parts and not f.name.startswith("agent-")]
    if not mains:
        raise SystemExit(f"{run_dir}: no session files (has the run finished?)")
    main = max(mains, key=lambda p: p.stat().st_size)
    return main, [f for f in files if f != main]


def start_time(main: Path) -> datetime:
    for line in main.read_text(errors="replace").splitlines():
        try:
            ts = json.loads(line).get("timestamp")
        except json.JSONDecodeError:
            continue
        if ts:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    raise SystemExit(f"{main}: no timestamps")


def scan(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    main, subs = session_files(run_dir)
    t0 = start_time(main)
    first: dict[str, float] = {}
    count = {k: 0 for k in PROBES}
    examples: dict[str, str] = {}
    gpu_seen = None
    bash_ids: set[str] = set()
    for f in [main, *subs]:
        for line in f.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = row.get("timestamp")
            content = (row.get("message") or {}).get("content")
            if not ts or not isinstance(content, list):
                continue
            minute = (datetime.fromisoformat(ts.replace("Z", "+00:00")) - t0).total_seconds() / 60
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use" and b.get("name") == "Bash":
                    cmd = str((b.get("input") or {}).get("command", ""))
                    bash_ids.add(b.get("id"))
                    for k, pat in PROBES.items():
                        if re.search(pat, cmd, re.I | re.M):
                            count[k] += 1
                            if k not in first or minute < first[k]:
                                first[k] = minute
                                examples[k] = cmd.replace("\n", " / ")[:120]
                elif b.get("type") == "tool_result" and b.get("tool_use_id") in bash_ids:
                    out = b.get("content")
                    if isinstance(out, list):
                        out = " ".join(x.get("text", "") for x in out if isinstance(x, dict))
                    if GPU_SEEN.search(OUR_LINE.sub("", str(out))) and (gpu_seen is None or minute < gpu_seen):
                        gpu_seen = minute
    report = run_dir / "report.json"
    gpu_busy = None
    if report.exists():
        g = json.loads(report.read_text()).get("free", {}).get("gpu", {})
        gpu_busy = round((g.get("seconds_busy") or 0) / 60, 1)
    return {"run_dir": str(run_dir), "first_min": {k: round(v, 1) for k, v in first.items()},
            "count": count, "first_example": examples,
            "gpu_first_seen_min": None if gpu_seen is None else round(gpu_seen, 1),
            "gpu_busy_min": gpu_busy}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="one JSON object per run")
    a = ap.parse_args(argv)
    for r in a.runs:
        res = scan(r)
        if a.json:
            print(json.dumps({"run": str(r), **res}))
            continue
        print(f"== {r}  (gpu first seen: {res['gpu_first_seen_min']} min, gpu busy: {res['gpu_busy_min']} min)")
        for k in PROBES:
            m = res["first_min"].get(k)
            print(f"  {k:10} first {('%.1f min' % m) if m is not None else 'never':10} n={res['count'][k]:3}  "
                  f"{res['first_example'].get(k, '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
