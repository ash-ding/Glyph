#!/usr/bin/env python3
"""Select the frozen benchmark instances from a scanned candidate pool.

Filters candidates down to those usable for the benchmark (binary-table
support, measured pi inside a defined band), then spreads a fixed number of
picks across each band's pi range rather than clustering near one edge:

    python tools/select_instances.py --candidates docs/benchmark/candidates.json \
        --out docs/benchmark/frozen_instances.json --n-per-band 5

See ``tools/scan_instances.py`` for how ``candidates.json`` is produced.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glyph.reference.frozen import DEFAULT_CUTOFFS, band_of  # noqa: E402


def select(candidates: list[dict], cutoffs=DEFAULT_CUTOFFS, n_per_band: int = 5) -> list[dict]:
    """Filter, band, and spread-select candidates into frozen manifest entries."""
    survivors = [c for c in candidates if c.get("uses_binary_tables")]

    by_band: dict[str, list[dict]] = {name: [] for name in cutoffs}
    for c in survivors:
        band = band_of(c["pi"], cutoffs)
        if band is None:
            continue
        by_band[band].append(c)

    chosen = []
    for band in cutoffs:
        group = sorted(by_band[band], key=lambda c: c["pi"])
        n = len(group)
        if n == 0:
            continue
        if n <= n_per_band:
            picks = group
        else:
            if n_per_band <= 1:
                idxs = [0]
            else:
                idxs = sorted({round(k * (n - 1) / (n_per_band - 1)) for k in range(n_per_band)})
            picks = [group[i] for i in idxs]

        for k, c in enumerate(picks, start=1):
            chosen.append({
                "id": f"{band}_{k}",
                "band": band,
                "preset": c["preset"],
                "seed": c["seed"],
                "measured_pi": c["pi"],
                "uses_binary_tables": True,
                "fingerprint_sha256": c["fingerprint"],
            })
    return chosen


def write_manifest(chosen: list[dict], cutoffs, n_per_band: int, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created": datetime.date.today().isoformat(),
        "band_cutoffs": {band: list(bounds) for band, bounds in cutoffs.items()},
        "n_per_band": n_per_band,
        "instances": chosen,
    }
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Select the frozen benchmark instances from a scanned candidate pool.")
    ap.add_argument("--candidates", default="docs/benchmark/candidates.json",
                     help="input candidates manifest (default: docs/benchmark/candidates.json)")
    ap.add_argument("--out", default="docs/benchmark/frozen_instances.json",
                     help="output manifest path (default: docs/benchmark/frozen_instances.json)")
    ap.add_argument("--n-per-band", type=int, default=5,
                     help="number of instances to pick per band (default: 5)")
    args = ap.parse_args(argv)

    with open(args.candidates) as f:
        manifest = json.load(f)
    candidates = manifest["candidates"]

    chosen = select(candidates, DEFAULT_CUTOFFS, args.n_per_band)
    write_manifest(chosen, DEFAULT_CUTOFFS, args.n_per_band, args.out)

    by_band: dict[str, list[dict]] = {}
    for inst in chosen:
        by_band.setdefault(inst["band"], []).append(inst)

    for band in DEFAULT_CUTOFFS:
        group = by_band.get(band, [])
        print("%s: %d instance(s)" % (band, len(group)))
        for inst in group:
            print("  %-12s pi=%.3f seed=%d preset=%s" % (
                inst["id"], inst["measured_pi"], inst["seed"], inst["preset"]))
        if len(group) < args.n_per_band:
            print("  ! shortfall: only %d/%d in band %s" % (len(group), args.n_per_band, band))

    print("selected %d instance(s) -> %s" % (len(chosen), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
