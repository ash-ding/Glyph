#!/usr/bin/env python3
"""Scan a pool of candidate seeds per preset and record every candidate's metrics.

The frozen benchmark instance set is later chosen from the whole distribution
of a preset's seeds, not just the first few. This tool generates each
candidate instance, measures it, and writes the results to a JSON manifest so
that selection can happen offline against the full pool:

    python tools/scan_instances.py --presets pi_low pi_mid pi_high --n-seeds 60

See ``docs/benchmark`` for how the resulting ``candidates.json`` is consumed.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from glyph.data import PRESETS, generate  # noqa: E402
from glyph.reference.frozen import fingerprint  # noqa: E402
from glyph.reference.subset import ceilings_on, paired_subset  # noqa: E402


def scan(presets: list[str], n_seeds: int, start_seed: int = 1001) -> list[dict]:
    """Generate every (preset, seed) candidate and record its metrics."""
    candidates = []
    for preset in presets:
        cfg = PRESETS[preset]
        for seed in range(start_seed, start_seed + n_seeds):
            inst = generate(seed, cfg)
            pi_components = inst.measured_pi()
            subset = paired_subset(inst, 500)
            candidates.append({
                "seed": seed,
                "preset": preset,
                "pi": pi_components["pi"],
                "pi_components": pi_components,
                "uses_binary_tables": any(t.needs_b for t in inst.test),
                "fingerprint": fingerprint(inst),
                "skeleton_ceiling": ceilings_on(inst, subset)["skeleton"]["overall"],
            })
    return candidates


def write_candidates(candidates: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created": datetime.date.today().isoformat(),
        "candidates": candidates,
    }
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)


def histogram(candidates: list[dict]) -> str:
    """A text histogram of measured pi per preset, 10 bins over [0, 1]."""
    by_preset: dict[str, list[float]] = {}
    for c in candidates:
        by_preset.setdefault(c["preset"], []).append(c["pi"])

    n_bins = 10
    lines = []
    for preset in sorted(by_preset):
        pis = by_preset[preset]
        lines.append("%s (n=%d)" % (preset, len(pis)))
        counts = [0] * n_bins
        for pi in pis:
            idx = min(int(pi * n_bins), n_bins - 1)
            counts[idx] += 1
        for i, count in enumerate(counts):
            lo, hi = i / n_bins, (i + 1) / n_bins
            lines.append("  [%.1f, %.1f) %s %d" % (lo, hi, "#" * count, count))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Scan a pool of candidate seeds per preset and record their metrics.")
    ap.add_argument("--presets", nargs="+", default=["pi_low", "pi_mid", "pi_high"],
                     help="preset names to scan (default: pi_low pi_mid pi_high)")
    ap.add_argument("--n-seeds", type=int, default=60,
                     help="number of seeds per preset (default: 60)")
    ap.add_argument("--start-seed", type=int, default=1001,
                     help="first seed in the range (default: 1001)")
    ap.add_argument("--out", default="docs/benchmark/candidates.json",
                     help="output path (default: docs/benchmark/candidates.json)")
    args = ap.parse_args(argv)

    candidates = scan(args.presets, args.n_seeds, args.start_seed)
    write_candidates(candidates, args.out)
    print(histogram(candidates))
    print("scanned %d candidate(s) across %d preset(s) -> %s" % (
        len(candidates), len(args.presets), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
