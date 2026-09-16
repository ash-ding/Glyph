#!/usr/bin/env python3
"""Collect run.json files into the viewer's data directory.

Each protocol-v2 run writes a self-contained `run.json` into its run dir. This
gathers them for the browser viewer:

    python tools/collect_runs.py --out viewer/runs /tmp/glyph_pilot_train /path/to/runs

It copies every `run.json` found under the given roots to `<out>/<id>.json` and
writes `<out>/index.json` (one summary row per run, newest first). Idempotent —
re-run it whenever new runs finish. Then serve the repo and open the viewer:

    python -m http.server 8000        # from the repo root
    # open http://localhost:8000/viewer/
"""
import argparse
import json
import sys
from pathlib import Path


def collect(roots, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    seen = set()
    for root in roots:
        for rj_path in sorted(Path(root).rglob("run.json")):
            try:
                rj = json.loads(rj_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                sys.stderr.write("skip %s: %r\n" % (rj_path, e))
                continue
            rid = rj.get("id")
            if not rid or rid in seen:
                # de-dupe by id; keep the first seen (roots are ordered)
                if rid in seen:
                    continue
            seen.add(rid)
            (out_dir / ("%s.json" % rid)).write_text(json.dumps(rj))
            summ = dict(rj.get("summary", {}))
            summ["id"] = rid
            summ["created"] = rj.get("created")
            index.append(summ)
    index.sort(key=lambda r: (r.get("created") or ""), reverse=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description="Collect run.json files for the viewer.")
    ap.add_argument("roots", nargs="+", help="directories to scan for run.json (recursively)")
    ap.add_argument("--out", default="viewer/runs", help="output dir (default: viewer/runs)")
    args = ap.parse_args(argv)
    index = collect(args.roots, args.out)
    print("collected %d run(s) into %s" % (len(index), args.out))
    for r in index:
        print("  %-40s %-9s %-8s seed=%s  overall=%s  $%s" % (
            r.get("id"), r.get("arm"), r.get("preset"), r.get("seed"),
            r.get("overall"), r.get("spent_usd")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
