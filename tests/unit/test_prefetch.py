"""dialogue.prefetch (docs/ARCH_REVIEW.md section 32): while the agent speaks, the start of the next likely prompt is sent to the model server, so the next reply reads
less. It is only worth anything if what is sent is exactly how the next prompt starts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_model_speaks import ENGINE, engine  # noqa: E402


class Recorder:
    def __init__(self, model):
        self.model, self.sent = model, []

    def lead_in_stream(self, *a, **k):
        return self.model.lead_in_stream(*a, **k)

    def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        self.sent.append(user)
        return "x"


def _engine():
    e, model = engine(ENGINE)
    e.prefetch_on = True
    e.reply_writer.backend = rec = Recorder(model)
    return e, model, rec


def _settle(e):
    e._prefetcher.submit(lambda: None).result(timeout=5)                 # the prefetch thread has done what was queued


def test_after_a_list_each_car_s_prompt_start_is_sent_and_the_next_reply_starts_with_it():
    e, model, rec = _engine()
    s = e.new_session()
    list(e.handle_stream(s, "Porsche."))                                  # one Porsche listed: the Macan
    _settle(e)
    assert len(rec.sent) == 1 and "Everything on the spec sheet of the 2016 Porsche Macan" in rec.sent[0]
    list(e.handle_stream(s, "Yes, tell me about the Macan."))
    assert model.asked[-1].startswith(rec.sent[0])                       # the whole prefix the server has already read


def test_a_follow_up_about_the_same_car_starts_with_what_was_sent():
    e, model, rec = _engine()
    s = e.new_session()
    for t in ("Porsche.", "Yes, tell me about the Macan."):
        list(e.handle_stream(s, t))
    _settle(e)
    sent = rec.sent[-1]
    list(e.handle_stream(s, "And the boot?"))
    assert model.asked[-1].startswith(sent) and sent.endswith("CUSTOMER (just now): ")


def test_nothing_is_sent_while_a_booking_is_being_filled_in_or_when_switched_off():
    e, _, rec = _engine()
    s = e.new_session()
    for t in ("Porsche.", "Yes, tell me about the Macan.", "I'd like a test drive."):
        list(e.handle_stream(s, t))
    _settle(e)
    before = len(rec.sent)
    assert s.state.value in ("slot", "contact") and e.prefetch(s) == 0 and len(rec.sent) == before
    e.prefetch_on = False
    s2 = e.new_session()
    list(e.handle_stream(s2, "Porsche."))
    _settle(e)
    assert len(rec.sent) == before


def test_a_new_turn_stops_what_was_queued_for_the_old_one():
    e, _, rec = _engine()
    s = e.new_session()
    gate = __import__("threading").Event()
    e._prefetcher.submit(gate.wait)                                       # the prefetch thread is busy
    list(e.handle_stream(s, "Porsche."))                                  # queues the Macan
    e._prefetch_turn.pop(id(s), None)                                     # the caller speaks before it runs (handle_stream does this first)
    gate.set()
    _settle(e)
    assert rec.sent == []
