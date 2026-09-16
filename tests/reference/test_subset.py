import math

from glyph.data import PRESETS, generate
from glyph.reference.subset import ceilings_on, paired_subset


def test_paired_subset_fixed_and_sized():
    inst = generate(1001, PRESETS['smoke'])
    items1 = paired_subset(inst, 40)
    items2 = paired_subset(inst, 40)
    assert len(items1) <= len(inst.test)
    assert [id(t) for t in items1] == [id(t) for t in items2]
    for t in items1:
        assert t in inst.test


def test_paired_subset_stratified():
    inst = generate(1001, PRESETS['smoke'])
    n = 40
    items = paired_subset(inst, n)
    by_split_test = {}
    for t in inst.test:
        by_split_test.setdefault(t.split, 0)
        by_split_test[t.split] += 1
    by_split_picked = {}
    for t in items:
        by_split_picked.setdefault(t.split, 0)
        by_split_picked[t.split] += 1
    for split, count in by_split_test.items():
        expected = round(n * count / len(inst.test))
        got = by_split_picked.get(split, 0)
        assert abs(got - expected) <= 1


def test_cpu_ceilings_shape_and_order():
    inst = generate(1001, PRESETS['smoke'])
    items = paired_subset(inst, 40)
    out = ceilings_on(inst, items)

    assert set(out.keys()) == {'skeleton', 'table', 'perfect'}
    assert out['skeleton']['overall'] <= out['table']['overall']
    assert out['perfect']['overall'] == 1.0
    assert out['skeleton']['tail'] in (0.0, None)

    for oracle in ('skeleton', 'table', 'perfect'):
        for key in ('overall', 'by_split', 'tail', 'headroom'):
            assert key in out[oracle]
        h = out[oracle]['headroom']
        assert h is None or -1e-9 <= h <= 1.0 + 1e-9
