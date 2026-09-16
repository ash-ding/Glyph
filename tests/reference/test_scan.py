import json
import sys

sys.path.insert(0, "tools")
from scan_instances import histogram, scan, write_candidates  # noqa: E402


def test_scan_smoke_small():
    candidates = scan(["smoke"], n_seeds=3, start_seed=1001)
    assert isinstance(candidates, list)
    assert len(candidates) == 3

    expected_keys = {
        "seed", "preset", "pi", "pi_components", "uses_binary_tables",
        "fingerprint", "skeleton_ceiling",
    }
    for d in candidates:
        assert set(d.keys()) == expected_keys
        assert 0.0 <= d["pi"] <= 1.0
        assert len(d["fingerprint"]) == 64
        assert isinstance(d["uses_binary_tables"], bool)

    # deterministic: a second scan gives the same pi and fingerprint per seed
    again = scan(["smoke"], n_seeds=3, start_seed=1001)
    for a, b in zip(candidates, again):
        assert a["seed"] == b["seed"]
        assert a["pi"] == b["pi"]
        assert a["fingerprint"] == b["fingerprint"]


def test_write_and_histogram(tmp_path):
    candidates = scan(["smoke"], 2)
    path = tmp_path / "c.json"
    write_candidates(candidates, path)

    with open(path) as f:
        manifest = json.load(f)
    assert isinstance(manifest, dict)
    assert "created" in manifest
    assert isinstance(manifest["candidates"], list)
    assert len(manifest["candidates"]) == 2

    text = histogram(candidates)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "smoke" in text
