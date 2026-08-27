"""Scheduling runs across the eight GPUs, grouped by instance.

Phase 1 is 600-1000 independent small runs, not one big job, so eight H100s
are eight workers rather than one colocated cluster.  One run, one GPU.

**Grouping by `instance_seed` is the whole point of this file.**  The plan's
statistics are paired: at a given budget point, every arm must be measured on
the *same* hidden interpreter, because some interpreters are simply harder
than others and an unpaired comparison buries the effect under that variance.
The crossover's confidence interval is what decides whether B* can be located
at all, so this is not a refinement.

The rule is easy to state and easy to lose. Written only in the method
section it survives until the first time someone reruns a single arm to fill
a gap and silently pairs it against a different instance. Here it is a
property of the scheduler: jobs are emitted instance-major, and a group is
not reported until every arm in it has finished.

Each run goes to its own subprocess. vLLM and torch do not survive being
set up twice in one process, and a crashed run should cost one cell of the
grid rather than the sweep.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ARMS = ("a2", "a4", "a6")


@dataclass
class Job:
    arm: str
    preset: str
    instance_seed: int
    run_seed: int
    budget_h100s: float
    n_test: int | None = None
    out_root: str = "runs"
    base_model: str = "Qwen/Qwen3-1.7B"

    @property
    def tag(self) -> str:
        return (f"{self.arm}_{self.preset}_i{self.instance_seed}"
                f"_r{self.run_seed}_b{self.budget_h100s:g}")

    def argv(self) -> list[str]:
        argv = [sys.executable, "-m", "glyph.cli", "run",
                "--arm", self.arm, "--preset", self.preset,
                "--instance-seed", str(self.instance_seed),
                "--run-seed", str(self.run_seed),
                "--budget", str(self.budget_h100s),
                "--out-root", self.out_root,
                "--base-model", self.base_model]
        if self.n_test:
            argv += ["--n-test", str(self.n_test)]
        return argv


def grid(*, arms=ARMS, presets=("pi_mid",), instance_seeds=(1001,),
         budgets=(15000.0,), run_seeds=(0,), n_test=None,
         out_root="runs") -> list[Job]:
    """Every cell, emitted instance-major so pairs stay together."""
    jobs: list[Job] = []
    for inst in instance_seeds:
        for preset in presets:
            for budget in budgets:
                for run_seed in run_seeds:
                    for arm in arms:
                        jobs.append(Job(arm=arm, preset=preset,
                                        instance_seed=inst, run_seed=run_seed,
                                        budget_h100s=budget, n_test=n_test,
                                        out_root=out_root))
    return jobs


@dataclass
class Slot:
    gpu: int
    proc: subprocess.Popen | None = None
    job: Job | None = None
    started: float = 0.0


def run_grid(jobs: list[Job], *, gpus: list[int] | None = None,
             timeout: float = 7200, log_root: str = "runs/_logs") -> dict:
    """Run every job, at most one per GPU, and report what happened.

    Nothing here retries. A run that fails is a cell with a reason attached,
    and re-running it deliberately is better than a scheduler that quietly
    papers over a systematic failure by trying again.
    """
    gpus = gpus if gpus is not None else list(range(_visible_gpus()))
    logs = Path(log_root)
    logs.mkdir(parents=True, exist_ok=True)

    pending = list(jobs)
    slots = [Slot(gpu=g) for g in gpus]
    done: dict[str, dict] = {}
    t0 = time.time()

    while pending or any(s.proc for s in slots):
        for s in slots:
            if s.proc is None and pending:
                s.job = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(s.gpu))
                fh = (logs / f"{s.job.tag}.log").open("w")
                s.proc = subprocess.Popen(s.job.argv(), env=env,
                                          stdout=fh, stderr=subprocess.STDOUT)
                s.started = time.time()
                print(f"  [gpu{s.gpu}] start {s.job.tag}", flush=True)

            elif s.proc is not None:
                code = s.proc.poll()
                if code is None and time.time() - s.started > timeout:
                    s.proc.kill()
                    code = -9
                if code is not None:
                    mins = (time.time() - s.started) / 60
                    done[s.job.tag] = {"job": s.job.__dict__, "returncode": code,
                                       "minutes": round(mins, 1)}
                    print(f"  [gpu{s.gpu}] {'ok  ' if code == 0 else 'FAIL'} "
                          f"{s.job.tag}  {mins:.1f}m", flush=True)
                    s.proc, s.job = None, None
        time.sleep(2)

    by_instance = defaultdict(list)
    for tag, rec in done.items():
        by_instance[rec["job"]["instance_seed"]].append(rec["returncode"] == 0)
    incomplete = [k for k, v in by_instance.items() if not all(v)]

    summary = {"jobs": len(jobs), "ok": sum(1 for r in done.values()
                                            if r["returncode"] == 0),
               "minutes": round((time.time() - t0) / 60, 1),
               "instances_with_a_failed_arm": sorted(incomplete),
               "runs": done}
    (logs / "grid_summary.json").write_text(json.dumps(summary, indent=2))
    if incomplete:
        # Said loudly, because a partially-finished instance cannot be used in
        # a paired comparison and silently averaging what is left would put
        # instance difficulty back into the error bar.
        print(f"\n  {len(incomplete)} instance(s) have a failed arm and must "
              f"be excluded or re-run as a group: {sorted(incomplete)}",
              flush=True)
    return summary


def _visible_gpus() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return len([x for x in vis.split(",") if x.strip()])
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"],
                             capture_output=True, text=True, timeout=30)
        return max(1, len(out.stdout.strip().splitlines()))
    except Exception:
        return 1
