import random

import pytest
import yaml

from dealer_core.dialogue.phrasebook import SITUATIONS, Phrasebook, check_variant, load_variants


def sit(sid):
    return SITUATIONS[sid]


def test_every_base_wording_passes_its_own_voice_rules_and_has_consistent_placeholders():
    for s in SITUATIONS.values():
        sample = s.base.format_map({f: "item" for f in s.fields})
        assert sample                                   # formats without error
        assert all("voice rule" not in p for p in check_variant(s, s.base + " "))      # the base wording obeys the voice rules


def test_good_variant_passes():
    assert check_variant(sit("no_match"), "Sorry, I have no {desc} at the moment. Shall we try another budget or type of car?") == []


@pytest.mark.parametrize("sid,text,reason", [
    ("no_match", "I have none right now. Want to change the budget?", "placeholders"),                       # lost {desc}
    ("no_match", "I don't have any {desc} or {total} right now.", "placeholders"),                          # extra placeholder
    ("narrow_again", "That's still {total} {desc}. Tell me a make or a model.", "required phrase"),
    ("no_match", "I don't have any {desc}, but I can offer you a discount on another car.", "commercial"),
    ("opening", "Hello, welcome. I promise to find you the perfect car, what are you after?", "commitment"),
    ("no_match", "We have plenty of {desc} in stock. Would you like to change the budget?", "negation"),       # flipped meaning
    ("cannot_filter", "I can search on anything you like. What would you like to change?", "negation"),
    ("goodbye", "Thank you so very much for calling us today and we hope that you have a truly wonderful rest of your day ahead of you.", "longer"),
    ("didnt_catch", "Sorry. Could you say that again? I missed it. Please repeat it.", "voice rule"),
    ("didnt_catch", "Sorry, could you say that again?", "identical"),
    ("opening", "Hello {name}, welcome!", "placeholders"),
])
def test_unsafe_variants_are_rejected(sid, text, reason):
    problems = " | ".join(check_variant(sit(sid), text))
    assert reason in problems, f"{text!r}: expected '{reason}' in {problems!r}"


def test_only_approved_entries_are_loaded_and_invalid_approved_entries_stop_startup(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(yaml.safe_dump({"goodbye": [{"text": "Thanks for calling, take care!", "status": "approved"},
                                             {"text": "Bye for now!", "status": "proposed"},
                                             {"text": "Never call again.", "status": "rejected"}]}), encoding="utf-8")
    assert load_variants(p) == {"goodbye": ["Thanks for calling, take care!"]}
    p.write_text(yaml.safe_dump({"no_match": [{"text": "Nothing here.", "status": "approved"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="not valid"):
        load_variants(p)
    p.write_text(yaml.safe_dump({"made_up": [{"text": "x", "status": "approved"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown situation"):
        load_variants(p)


def test_without_approved_variants_the_base_wording_is_always_used():
    pb, rng, last = Phrasebook(), random.Random(1), {}
    assert {pb.say(rng, last, "goodbye") for _ in range(20)} == {sit("goodbye").base}


def test_variants_are_picked_by_seed_never_twice_in_a_row_and_values_are_filled_in():
    pb = Phrasebook({"narrow_budget": ["There are {total} {desc} in stock. What budget do you have in mind?"]})

    def run(seed):
        rng, last, out = random.Random(seed), {}, []
        for _ in range(8):
            out.append(pb.say(rng, last, "narrow_budget", total=6, desc="SUVs"))
        return out

    a = run(3)
    assert a == run(3)                                              # same seed -> same call
    assert all(x != y for x, y in zip(a, a[1:]))                    # never the same wording twice in a row
    assert set(a) == {"I have 6 SUVs. What's your budget?", "There are 6 SUVs in stock. What budget do you have in mind?"}


def test_shipped_variants_file_is_valid_and_nothing_is_approved_without_a_human():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "config" / "phrase_variants.yaml"
    if not path.exists():
        pytest.skip("no variants file yet")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for sid, entries in data.items():
        assert sid in SITUATIONS
        for e in entries:
            assert e["status"] in ("proposed", "approved", "rejected")
            if e["status"] == "proposed":                           # proposed entries must at least be valid, so review is meaningful
                assert check_variant(SITUATIONS[sid], e["text"]) == [], e["text"]
    load_variants(path)                                             # whatever is approved must validate


def test_engine_uses_variants_only_when_enabled():
    from scenarios.conftest import Call

    off = Call({})
    assert {off.app.engine.opening(off.app.engine.new_session()) for _ in range(5)} == {sit("opening").base}
    on = Call({}, variants=True)
    lines = set()
    for seed in range(40):
        on.app.engine.variant_seed = seed
        lines.add(on.app.engine.opening(on.app.engine.new_session()))
    assert len(lines) > 1 and all("AGMC dealership" in x for x in lines)      # varied, and always names the dealership


def test_a_whole_call_with_variants_on_stays_valid():
    from scenarios.conftest import Call

    c = Call({"just show me": "search_cars"}, variants=True)
    for said in ("SUV under 150 thousand", "SUV under 150 thousand", "just show me", "do you have a bently car", "thanks bye"):
        r = c.say(said)                              # say() also asserts the voice rules on every reply
        assert r.reply
