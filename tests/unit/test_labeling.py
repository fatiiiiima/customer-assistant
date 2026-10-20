import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from dealer_core.dialogue.states import Intent, State
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
