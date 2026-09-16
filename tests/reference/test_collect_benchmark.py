import json
import sys

sys.path.insert(0, "tools")
import collect_runs
from collect_runs import copy_benchmark


def _write_benchmark(bench_dir, with_ceilings=True):
    bench_dir.mkdir(parents=True)
    (bench_dir / "frozen_instances.json").write_text(json.dumps({
        "instances": [{"id": "low_1", "band": "low", "seed": 1068, "preset": "pi_low"}]
    }))
    if with_ceilings:
        (bench_dir / "reference_ceilings.json").write_text(json.dumps({
            "low_1": {"skeleton": {"overall": 0.034, "tail": 0.0}}
        }))


def test_copy_benchmark_copies_both_files(tmp_path, monkeypatch):
    bench = tmp_path / "docs" / "benchmark"
    _write_benchmark(bench)
    monkeypatch.setattr(collect_runs, "BENCHMARK_DIR", bench)

    out = tmp_path / "viewer_runs"
    copy_benchmark(out)

    assert (out / "frozen_instances.json").exists()
    assert (out / "reference_ceilings.json").exists()
    assert json.loads((out / "frozen_instances.json").read_text())["instances"][0]["id"] == "low_1"


def test_copy_benchmark_skips_missing_reference_ceilings(tmp_path, monkeypatch):
    bench = tmp_path / "docs" / "benchmark"
    _write_benchmark(bench, with_ceilings=False)
    monkeypatch.setattr(collect_runs, "BENCHMARK_DIR", bench)

    out = tmp_path / "viewer_runs"
    copy_benchmark(out)  # must not raise even though reference_ceilings.json is absent

    assert (out / "frozen_instances.json").exists()
    assert not (out / "reference_ceilings.json").exists()


def test_copy_benchmark_skips_everything_when_dir_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_runs, "BENCHMARK_DIR", tmp_path / "docs" / "benchmark")

    out = tmp_path / "viewer_runs"
    copy_benchmark(out)  # neither file exists; must not raise

    assert out.exists()
    assert not (out / "frozen_instances.json").exists()
    assert not (out / "reference_ceilings.json").exists()
