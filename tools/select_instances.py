#!/usr/bin/env python3
"""Select the frozen benchmark instances from a scanned candidate pool.

Each band is selected **preset-pure** and **within a narrow archetype
window** of measured pi (a strict subset of the band's regime cutoffs from
``glyph.reference.frozen.DEFAULT_CUTOFFS``), so that:

  - every instance in a band comes from that band's matching preset (no
    cross-preset drift, e.g. a ``pi_mid`` seed landing in the ``high`` band),
  - every instance sits near the archetype center of its regime rather than
    at a band edge, so the three bands are maximally separated in pi-space.

For each band, candidates are filtered to ``preset == <band's preset>``,
``uses_binary_tables is True``, and measured pi inside the band's window.
Five equidistant target positions are placed across the window and the
nearest not-yet-chosen candidate is picked for each target (ties broken by
seed ascending), so picks are spread across the window rather than
clustered, deterministically and without reuse.

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

# Per-band selection spec: required source preset + narrow archetype window
# (measured pi, half-open [lo, hi)). Each window is a strict subset of the
# corresponding regime cutoff in DEFAULT_CUTOFFS, chosen to sit near the
# center of the band so selected instances are clean, well-separated
# exemplars of their regime rather than borderline cases at a band edge.
DEFAULT_SELECTION = {
    "low": {"preset": "pi_low", "window": (0.20, 0.30)},
    "mid": {"preset": "pi_mid", "window": (0.45, 0.53)},
    "high": {"preset": "pi_high", "window": (0.70, 0.80)},
}


def _spread_pick(group: list[dict], n: int, win_lo: float, win_hi: float) -> list[dict]:
    """Pick n candidates from group spread evenly across [win_lo, win_hi), without reuse.

    ``group`` must already be sorted deterministically (by pi, ties by seed).
    Places n equidistant target pi values across the DECLARED WINDOW
    ``[win_lo, win_hi)`` -- not the observed min/max of ``group`` -- and
    greedily assigns each target the nearest not-yet-chosen candidate,
    breaking distance ties by seed ascending. Anchoring to the window (rather
    than the candidate pool's own span) is what keeps picks tied to the
    archetype window regardless of where the surviving candidates happen to
    cluster; the top target equals win_hi (the exclusive bound), which is
    fine since the nearest-candidate rule picks the closest survivor
    strictly below it.
    """
    if not group:
        return []
    if len(group) <= n:
        return list(group)

    if n <= 1:
        targets = [win_lo]
    else:
        targets = [win_lo + (win_hi - win_lo) * k / (n - 1) for k in range(n)]

    chosen_idxs: list[int] = []
    used: set[int] = set()
    for target in targets:
        best_idx = None
        best_key = None
        for i, c in enumerate(group):
            if i in used:
                continue
            key = (abs(c["pi"] - target), c["seed"])
            if best_key is None or key < best_key:
                best_key = key
                best_idx = i
        used.add(best_idx)
        chosen_idxs.append(best_idx)

    return [group[i] for i in sorted(chosen_idxs)]


def select(candidates: list[dict], cutoffs=DEFAULT_CUTOFFS, n_per_band: int = 5,
           selection: dict = DEFAULT_SELECTION) -> list[dict]:
    """Filter, band, and window-select candidates into frozen manifest entries.

    ``selection`` maps band name -> {"preset": str, "window": (lo, hi)}.
    Each band's candidates are restricted to that band's required preset and
    to the band's archetype window (a strict subset of that band's regime
    cutoff in ``cutoffs``), then ``n_per_band`` are picked spread across the
    window, deterministically and without reuse.
    """
    chosen = []
    for band in cutoffs:
        spec = selection[band]
        required_preset = spec["preset"]
        win_lo, win_hi = spec["window"]

        survivors = [
            c for c in candidates
            if c.get("uses_binary_tables")
            and c.get("preset") == required_preset
            and win_lo <= c["pi"] < win_hi
        ]
        # Deterministic order: sort by pi, ties by seed ascending.
        group = sorted(survivors, key=lambda c: (c["pi"], c["seed"]))

        picks = _spread_pick(group, n_per_band, win_lo, win_hi)

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

    chosen = select(candidates, DEFAULT_CUTOFFS, args.n_per_band, DEFAULT_SELECTION)
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
