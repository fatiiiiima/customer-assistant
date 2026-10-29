"""What the model is given when it writes a reply, laid out so the model server can reuse what it read the turn before (docs/ARCH_REVIEW.md section 30).
29 Oct trace on the Mac: reading the prompt was 1.47 s of a 1.85 s first sentence; the car in focus was given three times and the conversation window moved every
turn, so nothing after the instructions could be reused. Facts that hold while one car is discussed now come first, each fact once, then a conversation window
that moves only every few lines, then this turn's own facts."""
import sys
from pathlib import Path

from dealer_core.dialogue.session import Session
from dealer_core.dialogue.writer import ReplyWriter, WriterInput

sys.path.insert(0, str(Path(__file__).parent))

from test_model_speaks import ENGINE, engine  # noqa: E402


def _common(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def test_the_car_in_focus_is_given_once_and_before_the_conversation():
    e, model = engine(ENGINE)
    s = e.new_session()
    for t in ("Porsche.", "Yes, tell me about the Macan.", "Tell me more about the engine."):
        list(e.handle_stream(s, t))
    prompt = model.asked[-1]
    stable, rest = prompt.split("CONVERSATION SO FAR:")
    assert stable.startswith("FACTS THAT HOLD FOR THIS PART OF THE CALL:")
    assert "Everything on the spec sheet of the 2016 Porsche Macan" in stable and "Prices are fixed" in stable and "all-wheel drive" in stable
    assert prompt.count("- 2016 Porsche Macan:") == 1                       # was three times: the fact line, the cars in the call, the spec sheet
    assert "Prices are fixed" not in rest and "Everything on the spec sheet" not in rest


def test_two_turns_about_the_same_car_share_the_facts_and_the_conversation_so_far():
    e, model = engine(ENGINE)
    s = e.new_session()
    for t in ("Porsche.", "Yes, tell me about the Macan.", "Tell me more about the engine.", "And the boot?"):
        list(e.handle_stream(s, t))
    before, now = model.asked[-2], model.asked[-1]
    shared = _common(before, now)
    convo_start = before.index("CONVERSATION SO FAR:")
    assert shared > before.index("CUSTOMER (just now)") - 40 > convo_start   # everything up to this turn's own words is the same as last turn's


def test_other_cars_in_the_call_are_named_with_their_colour_and_price_only():
    e, model = engine(ENGINE)
    s = e.new_session()
    for t in ("I'm looking for a Toyota.", "Any model and any budget is fine.", "Tell me about the Corolla.", "Is it a good car?"):
        list(e.handle_stream(s, t))
    prompt = model.asked[-1]
    other = next(line for line in prompt.splitlines() if line.startswith("- 2020 Toyota Camry:"))
    assert other.endswith(" dirhams") and "seats" not in other


def test_the_spec_sheet_answer_is_not_given_twice():
    e, model = engine(ENGINE)
    s = e.new_session()
    for t in ("Porsche.", "Yes, tell me about the Macan.", "Does it have any accidents?"):
        list(e.handle_stream(s, t))
    assert "Answered from the spec sheet" not in model.asked[-1]


def test_a_fact_from_the_stable_part_is_not_rejected_as_invented():
    e, model = engine("It's all-wheel drive, with a 500-litre boot.")
    s = e.new_session()
    for t in ("Porsche.", "Yes, tell me about the Macan.", "Is it all-wheel drive?"):
        out = [x for x in e.handle_stream(s, t) if isinstance(x, str)]
    assert out[-1] == "It's all-wheel drive, with a 500-litre boot."


def test_the_window_moves_only_every_few_lines():
    w = ReplyWriter(None, temperature=None)
    w.history_lines = 6
    lines = [("customer" if i % 2 == 0 else "agent", f"line {i}") for i in range(30)]
    first_shown = []
    for n in range(1, 31):
        kept = lines[:n][-16:]                                              # the call keeps its last 16 lines
        shown = w.window(WriterInput(caller="x", core="y", risk="low", history=kept, history_base=n - len(kept)))
        assert 6 <= len(shown) <= 11 or len(shown) == n
        first_shown.append(shown[0][1])
    moves = sum(1 for a, b in zip(first_shown, first_shown[1:]) if a != b)
    assert moves <= 4                                                       # 30 lines: the start moved 4 times, not 24


def test_the_lines_forgotten_are_counted_and_kept_across_a_reconnect():
    from dealer_core.shared.state import restore, snapshot

    e, _ = engine(ENGINE)
    s = e.new_session()
    for i in range(10):
        s.remember("customer", f"hi {i}")
        s.remember("agent", f"hello {i}")
    assert len(s.history) == 16 and s.forgotten == 4
    fresh = Session()
    restore(snapshot(s), fresh, e.svc)
    assert fresh.forgotten == 4
