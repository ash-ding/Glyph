import pytest
from glyph.v2.session import TOOL_MATRIX, tool_available_for

EXPECTED = {
  ('train','practice'):   {'query','submit','check_answers','finish_practice','build_dataset','train','student_infer'},
  ('no_train','practice'):{'query','submit','check_answers','finish_practice'},
  ('train','final'):      {'check_answers','student_infer','final_answer'},
  ('no_train','final'):   {'check_answers','final_answer'},
}

@pytest.mark.parametrize('key,expected', list(EXPECTED.items()))
def test_tool_matrix(key, expected):
    arm, phase = key
    got = {n for n in TOOL_MATRIX if tool_available_for(arm, phase, n)}
    assert got == expected

def test_all_matrix_names_known():
    assert TOOL_MATRIX == {'query','submit','check_answers','finish_practice',
                           'build_dataset','train','student_infer','final_answer'}

def _session(**kw):
    from glyph.v2.session import Session
    from glyph.v2.ledger import Ledger
    # inst/run_dir not needed for counter logic; pass simple placeholders
    import pathlib
    kw.setdefault('arm', 'train')
    return Session(inst=None, ledger=Ledger(), run_dir=pathlib.Path('/tmp'), **kw)

def test_counters_and_remaining():
    s = _session(q_cap=5, submit_cap=2, tp=3, tf=2)
    assert s.q_remaining()==5 and s.submissions_remaining()==2 and s.turns_remaining()==3
    s.q_used=5; s.submissions=2; s.turns=3
    assert s.q_remaining()==0 and s.submissions_remaining()==0 and s.turns_remaining()==0
    s.q_used=99  # never negative
    assert s.q_remaining()==0

def test_switch_to_final_resets_turns_and_uses_tf():
    s = _session(tp=3, tf=2); s.turns=3
    s.switch_to_final()
    assert s.phase=='final' and s.turns==0 and s.should_switch is False
    assert s.turns_remaining()==2

def test_tool_available_uses_arm_and_phase():
    s = _session(arm='no_train')
    assert s.tool_available('query') and not s.tool_available('train')
    s.switch_to_final()
    assert s.tool_available('final_answer') and not s.tool_available('query')

def test_note_appends_event():
    s = _session(); s.note(kind='x', v=1)
    assert s.events[-1] == {'kind':'x','v':1}
