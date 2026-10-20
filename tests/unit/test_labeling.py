import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from dealer_core.dialogue.states import STATE_SPEC, Intent, State
from dealer_core.eval.datasets import load_csv_set
from dealer_core.labeling import core
from dealer_core.labeling.server import App, make_handler, schema_json


@pytest.fixture
def items(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("# comment\ncan you do forty eight thousand\n\nwhat colour is it\nu9\tmy name is Aisha\tspoken\n", encoding="utf-8")
    return core.load_items(f)


def test_load_txt_skips_comments_and_blank_lines_and_reads_optional_id_and_source(items):
    assert [i.utterance for i in items] == ["can you do forty eight thousand", "what colour is it", "my name is Aisha"]
    assert items[2].id == "u9" and items[2].source == "spoken" and items[0].source == "typed"


def test_load_csv_and_duplicate_ids_rejected(tmp_path):
    f = tmp_path / "u.csv"
    f.write_text("id,utterance,source\na,hello there,typed\nb,book it,spoken\n", encoding="utf-8")
    assert [i.id for i in core.load_items(f)] == ["a", "b"]
    f.write_text("id,utterance\na,x\na,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        core.load_items(f)


def test_audio_is_attached_when_a_wav_with_the_id_exists(tmp_path):
    (tmp_path / "w.wav").write_bytes(b"RIFF")
    f = tmp_path / "u.csv"
    f.write_text("id,utterance\nw,hello\nz,bye\n", encoding="utf-8")
    its = core.load_items(f, tmp_path)
    assert its[0].audio and its[0].source == "spoken" and its[1].audio is None


def test_clean_slots_parses_and_validates():
    assert core.clean_slots({"offer_amount": "48,000", "phone": "050 123-4567", "body_type": "SUV", "make": " Toyota ", "model": ""}) == {
        "offer_amount": 48000, "phone": "0501234567", "body_type": "suv", "make": "toyota"}
    for bad in ({"offer_amount": "forty"}, {"phone": "12"}, {"body_type": "van"}, {"colour": "red"}):
        with pytest.raises(ValueError):
            core.clean_slots(bad)


def test_validate_label_demands_an_explicit_state_and_intent(items):
    ids = {i.id for i in items}
    for raw in ({"id": items[0].id}, {"id": items[0].id, "state": "negotiating"}, {"id": items[0].id, "intent": "make_offer"}):
        with pytest.raises(ValueError, match="nothing is pre-filled"):
            core.validate_label(raw, ids)
    with pytest.raises(ValueError, match="unknown utterance id"):
        core.validate_label({"id": "nope", "state": "greeting", "intent": "greet"}, ids)
    with pytest.raises(ValueError, match="terminal"):
        core.validate_label({"id": items[0].id, "state": "goodbye", "intent": "goodbye"}, ids)
    ok = core.validate_label({"id": items[0].id, "state": "negotiating", "intent": "make_offer", "car": " S1042 ",
                              "slots": {"offer_amount": "48000"}}, ids)
    assert ok.intent == "make_offer" and ok.car == "S1042" and ok.slots == {"offer_amount": 48000}
    assert core.validate_label({"id": items[0].id, "discard": True}, ids).discard


def test_store_saves_resumes_and_exports_a_csv_the_evaluator_can_load(items, tmp_path):
    store = core.LabelStore(items, tmp_path / "l.jsonl", tmp_path / "mine.csv")
    ids = {i.id for i in items}
    store.save(core.validate_label({"id": items[0].id, "state": "negotiating", "car": "S1042", "intent": "make_offer",
                                    "slots": {"offer_amount": "48,000"}}, ids))
    store.save(core.validate_label({"id": items[1].id, "state": "answering", "car": "S1042", "intent": "ask_car_info"}, ids))
    store.save(core.validate_label({"id": items[2].id, "discard": True, "notes": "unclear"}, ids))
    store.save(core.validate_label({"id": items[1].id, "state": "answering", "car": "S1042", "intent": "ask_policy",
                                    "slots": {"topic": "warranty"}}, ids))                         # relabel: last one wins
    resumed = core.LabelStore(items, tmp_path / "l.jsonl", tmp_path / "mine.csv")
    assert resumed.labels[items[1].id].intent == "ask_policy" and resumed.first_unlabeled() == len(items)
    rows = load_csv_set(tmp_path / "mine.csv")
    assert [(r.utterance, r.state, r.car, r.gold_intent, r.gold_slots) for r in rows] == [
        ("can you do forty eight thousand", State.NEGOTIATING, "S1042", Intent.MAKE_OFFER, {"offer_amount": 48000}),
        ("what colour is it", State.ANSWERING, "S1042", Intent.ASK_POLICY, {"topic": "warranty"})]      # discarded one omitted
    assert store.coverage()["make_offer"] == 1 and store.coverage()["ask_policy"] == 1


def test_existing_utterances_flags_leakage(tmp_path):
    (tmp_path / "dev.json").write_text(json.dumps([{"state": "greeting", "utterance": "Hello, hi!", "intent": "greet", "slots": {}}]), encoding="utf-8")
    assert core.existing_utterances(tmp_path) == {"hello hi": "dev"}


def test_schema_exposes_the_same_tables_as_the_nlu():
    s = schema_json()
    assert {x["value"] for x in s["states"]} == {"greeting", "interest", "answering", "negotiating", "slot", "contact", "confirm"}
    assert len(s["intents"]) == len(Intent) and s["slots"]["offer_amount"]["type"] == "int"
    assert "ask_car_info" not in next(x for x in s["states"] if x["value"] == "interest")["valid"]


def test_http_api_roundtrip(items, tmp_path):
    store = core.LabelStore(items, tmp_path / "l.jsonl", tmp_path / "mine.csv")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(store, ["S1042"], {})))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        def post(body):
            req = urllib.request.Request(base + "/api/label", json.dumps(body).encode(), {"Content-Type": "application/json"})
            try:
                return json.load(urllib.request.urlopen(req))
            except urllib.error.HTTPError as e:
                return json.load(e)

        assert "<title>Utterance labeling</title>" in urllib.request.urlopen(base + "/").read().decode()
        assert json.load(urllib.request.urlopen(base + "/api/item/0"))["utterance"] == items[0].utterance
        bad = post({"id": items[0].id, "state": "negotiating"})
        assert bad["ok"] is False and "choose a state and an intent" in bad["error"]
        good = post({"id": items[0].id, "state": "negotiating", "intent": "make_offer", "slots": {"offer_amount": "48000"}})
        assert good["ok"] and good["labeled"] == 1 and good["coverage"]["make_offer"] == 1
        assert (tmp_path / "mine.csv").exists()
    finally:
        httpd.shutdown()


# ---- folder loading, ambiguity, guide (added with the batch-1 spontaneous set) --------------------------------------------
def test_load_items_from_dir_is_verbatim_and_never_touches_the_source_files(tmp_path):
    raw = b"i didn't say that can you read my question properly\r\n"
    (tmp_path / "spont001.txt").write_bytes(raw)
    (tmp_path / "spont002.txt").write_bytes(b"  the weather is really hot today  \r\n")
    (tmp_path / "spont002.wav").write_bytes(b"RIFF")
    (tmp_path / "spont003.txt").write_bytes(b"\r\n")                      # empty -> skipped
    (tmp_path / "README.txt").write_text("not an utterance")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    items = core.load_items_from_dir(tmp_path)
    assert [(i.id, i.utterance, i.source) for i in items] == [
        ("spont001", "i didn't say that can you read my question properly", "typed"),
        ("spont002", "  the weather is really hot today  ", "spoken")]       # inner and outer text untouched, only the line break goes
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before      # source files byte-identical afterwards
    with pytest.raises(ValueError, match="no utterances matched"):
        core.load_items_from_dir(tmp_path, "nothing*.txt")


def test_ambiguous_label_rules(items):
    ids = {i.id for i in items}
    base = {"id": items[0].id, "state": "answering", "intent": "ask_policy", "ambiguous": True}
    with pytest.raises(ValueError, match="ambiguous needs"):
        core.validate_label(base, ids)                                          # a silent "ambiguous" flag is refused
    with pytest.raises(ValueError, match="different from the primary"):
        core.validate_label({**base, "also_ok": ["ask_policy"]}, ids)
    with pytest.raises(ValueError, match="unknown intent"):
        core.validate_label({**base, "also_ok": ["nonsense"]}, ids)
    a = core.validate_label({**base, "also_ok": ["ask_car_info", "ask_car_info", "send_details"]}, ids)
    assert a.ambiguous and a.also_ok == ["ask_car_info", "send_details"]       # de-duplicated, order kept
    assert core.validate_label({**base, "notes": "asks two things"}, ids).ambiguous          # a note alone is enough
    plain = core.validate_label({"id": items[0].id, "state": "answering", "intent": "ask_policy", "also_ok": ["send_details"]}, ids)
    assert not plain.ambiguous and plain.also_ok == []                          # alternates are ignored unless ambiguous is ticked


def test_ambiguity_survives_csv_round_trip_and_lenient_intent_accepts_alternates(items, tmp_path):
    store = core.LabelStore(items, tmp_path / "l.jsonl", tmp_path / "mine.csv")
    ids = {i.id for i in items}
    store.save(core.validate_label({"id": items[0].id, "state": "negotiating", "intent": "make_offer", "ambiguous": True,
                                    "also_ok": ["ask_lowest_price"], "notes": "offers a price and asks for the best one",
                                    "slots": {"offer_amount": "48000"}}, ids))
    store.save(core.validate_label({"id": items[1].id, "state": "answering", "intent": "ask_car_info"}, ids))
    header = (tmp_path / "mine.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.endswith("notes,ambiguous,also_ok_intents,state_guess,other_plausible_states")
    rows = load_csv_set(tmp_path / "mine.csv")
    assert (rows[0].ambiguous, rows[0].also_ok) == (True, ("ask_lowest_price",))
    assert (rows[1].ambiguous, rows[1].also_ok) == (False, ())


def test_old_label_files_without_the_new_fields_still_load(items, tmp_path):
    old = {"id": items[0].id, "state": "greeting", "car": "", "intent": "greet", "slots": {}, "notes": "", "discard": False, "labeled_at": "x"}
    (tmp_path / "l.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    store = core.LabelStore(items, tmp_path / "l.jsonl", tmp_path / "mine.csv")
    assert store.labels[items[0].id].ambiguous is False and store.labels[items[0].id].also_ok == []


def test_guide_covers_every_state_and_intent_and_its_examples_are_invented(tmp_path):
    from pathlib import Path
    import re

    assert set(core.STATE_GUIDE) == {s.value for s in STATE_SPEC} and set(core.INTENT_GUIDE) == {i.value for i in Intent}
    norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
    in_sets = set()
    for p in (Path(__file__).resolve().parents[2] / "data" / "nlu_sets").glob("*.json"):
        in_sets |= {norm(r["utterance"]) for r in json.loads(p.read_text(encoding="utf-8"))}
    examples = [e for _, e in core.STATE_GUIDE.values()] + [e for _, e in core.INTENT_GUIDE.values()]
    assert all(norm(e) not in in_sets for e in examples), "a guide example duplicates a test-set utterance"
    s = schema_json()
    assert all(x["example"] and x["doc"] for x in s["intents"]) and all(x["example"] and x["help"] for x in s["states"])


def test_http_accepts_an_ambiguous_label(items, tmp_path):
    store = core.LabelStore(items, tmp_path / "l.jsonl", tmp_path / "mine.csv")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(store, ["S1042"], {})))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        def post(body):
            req = urllib.request.Request(base + "/api/label", json.dumps(body).encode(), {"Content-Type": "application/json"})
            try:
                return json.load(urllib.request.urlopen(req))
            except urllib.error.HTTPError as e:
                return json.load(e)

        refused = post({"id": items[0].id, "state": "answering", "intent": "ask_car_info", "ambiguous": True})
        assert refused["ok"] is False and "ambiguous needs" in refused["error"]
        ok = post({"id": items[0].id, "state": "answering", "intent": "ask_car_info", "ambiguous": True, "also_ok": ["send_details"]})
        assert ok["ok"] and json.load(urllib.request.urlopen(base + "/api/item/0"))["label"]["also_ok"] == ["send_details"]
    finally:
        httpd.shutdown()


def test_state_guess_flag_and_other_plausible_states(items, tmp_path):
    ids = {i.id for i in items}
    base = {"id": items[0].id, "state": "interest", "intent": "out_of_scope"}
    plain = core.validate_label({**base, "other_states": ["answering"]}, ids)
    assert not plain.state_guess and plain.other_states == []                       # ignored unless the flag is ticked
    g = core.validate_label({**base, "state_guess": True, "other_states": ["answering", "interest", "answering", "slot"]}, ids)
    assert g.state_guess and g.other_states == ["answering", "slot"]                # de-duplicated, primary state excluded
    assert core.validate_label({**base, "state_guess": True}, ids).other_states == []     # the flag alone is allowed
    with pytest.raises(ValueError, match="unknown state"):
        core.validate_label({**base, "state_guess": True, "other_states": ["lobby"]}, ids)
    with pytest.raises(ValueError, match="terminal"):
        core.validate_label({**base, "state_guess": True, "other_states": ["goodbye"]}, ids)
    store = core.LabelStore(items, tmp_path / "l.jsonl", tmp_path / "mine.csv")
    store.save(g)
    assert (tmp_path / "mine.csv").read_text(encoding="utf-8-sig").splitlines()[0].endswith("state_guess,other_plausible_states")
    row = load_csv_set(tmp_path / "mine.csv")[0]
    assert row.state_guess and row.other_states == ("answering", "slot")


def test_labels_saved_before_the_state_guess_field_still_load(items, tmp_path):
    old = {"id": items[0].id, "state": "greeting", "car": "", "intent": "greet", "slots": {}, "notes": "", "discard": False,
           "labeled_at": "x", "ambiguous": False, "also_ok": []}
    (tmp_path / "l.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    lb = core.LabelStore(items, tmp_path / "l.jsonl", tmp_path / "mine.csv").labels[items[0].id]
    assert lb.state_guess is False and lb.other_states == []
