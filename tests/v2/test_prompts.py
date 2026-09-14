from glyph.v2.prompts import system_prompt, task_readme
from glyph.v2.session import Session
from glyph.v2.ledger import Ledger
import pathlib

def test_arms_differ_only_by_student_paragraph():
    tr = system_prompt("train"); nt = system_prompt("no_train")
    assert tr != nt
    # removing the student paragraph(s) from train makes it equal to no_train
    # (implement so that no_train is train minus exactly one contiguous block)
    # Heuristic check: every line of no_train appears in train, and the extra
    # lines in train all mention the student/training.
    nt_lines = [l for l in nt.splitlines() if l.strip()]
    tr_lines = [l for l in tr.splitlines() if l.strip()]
    for l in nt_lines:
        assert l in tr_lines, f"no_train line missing from train: {l!r}"
    extra = [l for l in tr_lines if l not in nt_lines]
    assert extra, "train must have an extra student paragraph"
    assert all(any(w in l.lower() for w in ("train","student","fine-tune","weights","model"))
               for l in extra)

def test_no_strategy_hints():
    for arm in ("train","no_train"):
        s = system_prompt(arm).lower()
        for banned in ("should win","better to","best strategy","we recommend","optimal","you should train","prefer training"):
            assert banned not in s

def test_caps_named_in_system_prompt():
    s = system_prompt("train").lower()
    assert "20" in s                     # submissions
    assert "query" in s and ("budget" in s or "cap" in s or "limit" in s)
    assert "final" in s and "practice" in s

def test_task_readme_has_concrete_caps(tmp_path):
    inst = None
    sess = Session(inst=inst, ledger=Ledger(), run_dir=tmp_path, arm="train",
                   q_cap=1000, submit_cap=20, tp=100, tf=30)
    r = task_readme(sess)
    assert "1000" in r and "20" in r and "100" in r and "30" in r
    assert "test" in r.lower() and "no" in r.lower()  # mentions no-oracle final phase
    assert "final_answer" in r or "final answer" in r.lower()
