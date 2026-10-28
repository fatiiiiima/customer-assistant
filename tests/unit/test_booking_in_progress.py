"""28 Oct, 13:30, a booking being arranged (nothing booked yet):
 * the number "050, 0790, 054, 056" (for 0507932546) was not a valid number, was dropped, and the agent asked "Anything else you'd like to know?" (the model, after 2.3 s);
 * "Do." (for "two") and "I want to change the time", said while times were on offer, were sent to the tools for bookings that already exist and answered
   "I don't see any test drives booked to move."."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dealer_core.agent.engine import AgentEngine
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.dialogue.session import KnownBooking
from dealer_core.dialogue.states import Intent, State
from dealer_core.nlu.backend import IntentResult

TZ = ZoneInfo("Asia/Dubai")
NOW = datetime(2026, 10, 28, 11, 0, tzinfo=TZ)


class Scripted:
    """The model: the tool for each decision comes from a list (anything not listed is "none"); replies are a fixed sentence. `asked` counts decisions made by the model."""
    name = "scripted"

    def __init__(self, decisions=()):
        self.decisions, self.asked = list(decisions), 0

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, system, user, *a, **k):
        if "TOOL:" in user:
            self.asked += 1
            return self.decisions.pop(0) if self.decisions else "none"
        return "ok"

    def lead_in_stream(self, *a, **k):
        yield "Here you go."


def engine(decisions=()):
    cfg = load_config(environ={})
    app = build_app(cfg, classifier=Scripted(), reply_mode="conversational", clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Scripted(), inventory="demo")
    model = Scripted(decisions)
    return AgentEngine(app.service, model, app.engine.reply_writer), model


def say(e, s, text):
    out = list(e.handle_stream(s, text))
    return out[-1].debug.intent, " ".join(x for x in out if isinstance(x, str))


def booking_at_the_contact_step(e):
    s = e.new_session()
    say(e, s, "I'm looking for a Toyota")
    s.car, s.state = s.shown[0].stock_id, State.ANSWERING
    say(e, s, "I want to come and see it")
    say(e, s, "Tomorrow")
    assert len(s.offered_slots) == 2
    say(e, s, "9")
    assert s.draft_slot is not None and s.state is State.CONTACT
    return s


# ------------------------------------------------------------------------------------------------ a number that is not a whole number
def test_a_number_that_is_not_a_whole_number_is_asked_for_again_by_the_code():
    e, model = engine()
    s = booking_at_the_contact_step(e)
    slot = s.draft_slot
    tool, said = say(e, s, "050, 0790, 054, 056")
    assert tool == "prepare_booking" and model.asked == 0                                   # (no 2.3 s wait for the model)
    assert "didn't catch the whole number" in said and "Anything else" not in said
    assert s.draft_slot == slot and s.state is State.CONTACT and s.draft_phone is None      # the time chosen is kept, still waiting for the number


def test_the_whole_number_said_next_is_taken():
    e, model = engine()
    s = booking_at_the_contact_step(e)
    say(e, s, "050, 0790, 054, 056")
    tool, said = say(e, s, "Omar, 050 793 2546")
    assert s.draft_phone and s.draft_phone.endswith("507932546") and s.draft_name


# ------------------------------------------------------------------------------------------------ nothing booked yet, so nothing to move
def test_a_word_that_is_not_a_time_while_times_are_on_offer_offers_them_again_not_a_move():
    e, model = engine(decisions=["reschedule_booking"])                                     # what the model said for "Do."
    s = e.new_session()
    say(e, s, "I'm looking for a Toyota")
    s.car, s.state = s.shown[0].stock_id, State.ANSWERING
    say(e, s, "I want to come and see it")
    say(e, s, "Tomorrow")
    tool, said = say(e, s, "Do.")
    assert tool == "prepare_booking" and "booked to move" not in said and len(s.offered_slots) == 2


def test_change_the_time_with_a_time_already_chosen_lets_it_go_and_offers_times():
    e, model = engine(decisions=["reschedule_booking"])
    s = booking_at_the_contact_step(e)
    first = s.draft_slot
    tool, said = say(e, s, "I want to change the time")
    assert tool == "prepare_booking" and "booked to move" not in said
    assert s.draft_slot is None and s.state is State.SLOT


def test_cancelling_while_only_a_draft_exists_drops_the_draft_not_a_booking():
    e, model = engine(decisions=["cancel_booking"])
    s = booking_at_the_contact_step(e)
    tool, said = say(e, s, "forget it, cancel that")
    assert tool == "cancel_draft" and s.draft_slot is None and "don't see" not in said


def test_with_a_real_booking_the_move_still_goes_to_the_booking_tools():
    e, model = engine(decisions=["reschedule_booking"])
    s = e.new_session()
    say(e, s, "I'm looking for a Toyota")
    s.car, s.state = s.shown[0].stock_id, State.ANSWERING
    car = e.svc.get_car(s.car)
    s.known_bookings = [KnownBooking(car.stock_id, datetime(2026, 10, 29, 14, 0, tzinfo=TZ), "Omar", "0507932546", booking_id="B1")]
    tool, _ = say(e, s, "hmm change that please")
    assert tool == "reschedule_booking"
