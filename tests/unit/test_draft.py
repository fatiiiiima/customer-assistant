import pytest

from dealer_core.dialogue.draft import DraftWriter, verify_draft

CORE = "I have a 2021 Toyota Camry with 58,000 kilometres at 78,500 dirhams. Would you like to hear more, or book a test drive?"
USER = "do you have a toyota camry"


def test_a_faithful_natural_rewrite_passes():
    draft = "Yes, I do have a 2021 Toyota Camry with 58,000 kilometres, priced at 78,500 dirhams. Want to hear more, or shall we book a test drive?"
    assert verify_draft(draft, USER, CORE) == []


@pytest.mark.parametrize("draft,reason", [
    ("I have a 2021 Toyota Camry with 58,000 kilometres at 75,000 dirhams. Would you like to hear more, or book a test drive?", "new number"),
    ("I have a 2021 Toyota Camry at 78,500 dirhams. Would you like to hear more, or book a test drive?", "dropped number"),
    ("I have a 2021 Toyota Camry with 58,000 kilometres at 78,500 dirhams, and a Lexus too. Want to hear more, or book a test drive?", "new name"),
    ("I have a 2021 Toyota Camry with 58,000 kilometres at 78,500 dirhams. Tell me more.", "question was dropped"),
    ("I have a 2021 Toyota Camry with 58,000 kilometres at 78,500 dirhams, a great deal. Want to hear more, or book a test drive?", "commercial"),
    ("I will make sure you get the Camry at 78,500 dirhams with 58,000 kilometres. Shall we book a test drive for 2021?", "promise"),
    ("I have a 2021 Toyota Camry with 58,000 kilometres at 78,500 dirhams – lovely. Want to hear more, or book a test drive?", "characters"),
    ("Sure. I have it. It is a 2021 Toyota Camry. It has 58,000 kilometres. It is 78,500 dirhams. Test drive?", "sentences"),
])
def test_unfaithful_drafts_are_rejected(draft, reason):
    problems = " | ".join(verify_draft(draft, USER, CORE))
    assert reason in problems, f"{draft!r} -> {problems!r}"


def test_a_flipped_negation_is_caught():
    core = "I don't have any Bentley cars right now. Would you like to change the budget or the type of car?"
    assert "negation" in " ".join(verify_draft("We have Bentley cars right now. Would you like to change the budget or the type of car?",
                                               "do you have a bently", core))
    ok = "Sorry, no Bentleys on my side. Is there another budget or type of car you'd like to try?"
    assert "negation" not in " ".join(verify_draft(ok, "do you have a bently", core))        # the same meaning, reworded


def test_quoted_phrases_must_survive():
    core = "That's still 6 SUVs. Tell me a make, or say 'just show me' and I'll read out the first three."
    bad = verify_draft("There are still 6 SUVs. Pick a make, or ask me to list them and I'll read out three.", "hmm", core)
    assert any("quoted phrase" in p for p in bad)


def test_identical_to_the_template_is_not_a_draft():
    assert "identical" in " ".join(verify_draft(CORE, USER, CORE))


class FakeBackend:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.seen = reply, error, []

    def lead_in(self, system, user, max_tokens=36):
        self.seen.append((system, user, max_tokens))
        if self.error:
            raise self.error
        return self.reply


def test_writer_gives_the_model_only_the_caller_and_the_facts_and_falls_back_on_any_failure():
    fb = FakeBackend("Yes, I do have a 2021 Toyota Camry with 58,000 kilometres, priced at 78,500 dirhams. Want to hear more, or book a test drive?")
    out = DraftWriter(fb, {"search_cars"}).write(USER, CORE)
    assert out.text and fb.seen[0][1] == f"CALLER: {USER}\nREPLY: {CORE}" and fb.seen[0][2] > 36
    assert DraftWriter(FakeBackend("We have a Bentley."), {"search_cars"}).write(USER, CORE).text is None
    assert DraftWriter(FakeBackend(error=TimeoutError()), {"search_cars"}).write(USER, CORE).text is None


def test_engine_draft_replaces_only_eligible_turns():
    from scenarios.conftest import Call

    c = Call({"how much is it": "ask_lowest_price"})
    fb = FakeBackend("Yes, I do have a 2021 Toyota Camry with 58,000 kilometres, priced at 78,500 dirhams. Want to hear more, or book a test drive?")
    c.app.engine.drafter = DraftWriter(fb, {"search_cars", "ask_car_info", "out_of_scope"})
    r = c.say("do you have a toyota camry")
    assert r.reply.startswith("Yes, I do have") and r.debug.draft and not r.debug.violations
    n = len(fb.seen)
    r = c.say("how much is it")                                       # a price turn: never drafted, exact template
    assert len(fb.seen) == n and not r.debug.draft and "subject to confirmation" in r.reply


def test_dropping_the_options_offered_is_caught_and_long_replies_are_not_drafted():
    core = "Which one do you mean: the first, the second or the third?"
    assert any("options" in p for p in verify_draft("Could you clarify which one you mean?", "is it red", core))
    assert verify_draft("Which do you mean, the first, the second or the third?", "is it red", core) == []
    long_core = "I have a 2019 Nissan Altima at 52,500 dirhams, a 2020 Honda Accord at 69,000 dirhams and a 2021 Toyota Camry at 78,500 dirhams. Which one would you like to hear more about?"
    fb = FakeBackend("anything")
    out = DraftWriter(fb, {"search_cars"}).write("a sedan", long_core)
    assert out.text is None and fb.seen == [] and "template" in out.why_not          # the model is not even called


def test_apostrophes_are_not_mistaken_for_quoted_phrases():
    core = "Yes, it's automatic. Anything else you'd like to know?"
    assert verify_draft("Yes, the car's automatic. Anything else you'd like to know?", "is it automatic", core) == []


def test_i_contractions_are_not_names():
    core = "I have 8 SUVs. Would you pick a make or a mileage limit?"
    assert verify_draft("Sure, I've got 8 SUVs. Would you pick a make or a mileage limit?", "an suv", core) == []
