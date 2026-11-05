"""Recognition with the stock's car names as the model's hot-word list instead of the hint (voice.stt_hotwords_field, ARCH_REVIEW section 82): what RemoteWhisper sends, where the names
come from, and what the echo guard does when the recogniser never saw a hint."""
import re

import httpx
import pytest

from dealer_core.config import load_config
from dealer_core.dialogue.states import State
from dealer_core.voice.pipeline import attach_stock_hotwords
from dealer_core.voice.prompts import CONTACT, stock_names, stt_prompt
from dealer_core.voice.services import UNCLEAR, EchoGuard
from dealer_core.voice.stt import HOTWORDS_SENT, RemoteWhisper

VOCAB = {"toyota": ["land cruiser", "rav4", "camry"], "porsche": ["macan", "cayenne"], "bmw": ["m3"]}


def server(text="Porsche."):
    """A transcription server that keeps every form it was sent."""
    forms = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("latin-1")
        forms.append({n: v for n, v in re.findall(r'name="(\w+)"\r\n\r\n([^\r]*)\r\n', body)})
        return httpx.Response(200, json={"text": text})

    return forms, httpx.Client(transport=httpx.MockTransport(handle))


def remote(client, **kw):
    return RemoteWhisper("http://srv/v1", "qwen3-asr", 3.0, kw.pop("hotwords", ["Toyota", "Nissan"]), client=client, **kw)


PCM = bytes(2 * 16000)


# ------------------------------------------------------------------------------------------------ the stock's names
def test_the_stock_names_are_makes_and_models_in_capitals_where_a_model_has_a_digit_with_no_repeats():
    assert stock_names(VOCAB) == ["Toyota", "Land Cruiser", "RAV4", "Camry", "Porsche", "Macan", "Cayenne", "Bmw", "M3"]
    assert stock_names(VOCAB, 3) == ["Toyota", "Land Cruiser", "RAV4"] and stock_names(None) == [] and stock_names({}) == []
    assert stock_names({"kia": ["kia"]}) == ["Kia"]


def test_the_hint_the_whisper_servers_get_is_unchanged_by_sharing_the_names():
    """The hint used to build the names inline; it must come out the same."""
    hint = stt_prompt(State.ANSWERING, "", {"makes": ["Toyota"], "models": ["Camry"]}, VOCAB)
    assert hint.startswith("Toyota, Land Cruiser, RAV4, Camry, Porsche, Macan, Cayenne, Bmw, M3, ") and "white, black, grey, silver, red, blue" in hint


# ------------------------------------------------------------------------------------------------ what is sent
def test_by_default_the_hint_is_sent_as_the_prompt_and_nothing_else_changes():
    forms, client = server()
    r = remote(client)
    assert r.transcribe(PCM, "Toyota, Macan. Which one?")[0] == "Porsche."
    assert forms[-1]["prompt"] == "Toyota, Macan. Which one?" and "hotwords" not in forms[-1]
    r.transcribe(PCM, None)
    assert forms[-1]["prompt"] == "Toyota, Nissan" and not r.ignores_prompt                 # the configured list stands in, as before


def test_with_the_setting_the_names_go_as_hotwords_and_the_hint_is_never_sent():
    forms, client = server()
    r = remote(client, hotwords_field=True)
    r.hotword_source = lambda: ["Toyota", "Land Cruiser", "Macan"]
    r.transcribe(PCM, "Toyota, Macan. Which one?")
    assert forms[-1]["hotwords"] == "Toyota, Land Cruiser, Macan" and "prompt" not in forms[-1] and r.ignores_prompt
    r.transcribe(PCM, "")
    r.transcribe(PCM, None)
    assert all("prompt" not in f for f in forms) and len(forms) == 3


def test_the_names_are_read_afresh_each_turn_so_a_car_added_to_the_stock_is_heard():
    forms, client = server()
    r = remote(client, hotwords_field=True)
    stock = ["Toyota"]
    r.hotword_source = lambda: list(stock)
    r.transcribe(PCM, None)
    stock.append("Macan")
    r.transcribe(PCM, None)
    assert [f["hotwords"] for f in forms] == ["Toyota", "Toyota, Macan"]


def test_no_names_means_no_field_not_an_empty_one_and_a_failing_source_falls_back_to_the_configured_list():
    forms, client = server()
    r = remote(client, hotwords_field=True, hotwords=[])
    r.hotword_source = list
    r.transcribe(PCM, None)
    assert "hotwords" not in forms[-1] and "prompt" not in forms[-1]
    r2 = remote(client, hotwords_field=True, hotwords=["Toyota", "Nissan"])

    def broken():
        raise RuntimeError("database locked")

    r2.hotword_source = broken
    assert r2.transcribe(PCM, None)[0] == "Porsche." and forms[-1]["hotwords"] == "Toyota, Nissan" and r2.errors == 0


def test_no_more_than_the_set_number_of_names_are_sent_and_the_default_covers_a_95_name_stock():
    forms, client = server()
    r = remote(client, hotwords_field=True)
    r.hotword_source = lambda: [f"Car{i}" for i in range(300)]
    r.transcribe(PCM, None)
    assert len(forms[-1]["hotwords"].split(", ")) == HOTWORDS_SENT == 200           # (95 names in the real stock) 2 Nov: at 45, Toyota and its models were cut from a 95-name stock; 3 Nov: 200, room for twice the stock
    small = remote(client, hotwords_field=True, hotwords_max=40)
    small.hotword_source = lambda: [f"Car{i}" for i in range(200)]
    small.transcribe(PCM, None)
    assert len(forms[-1]["hotwords"].split(", ")) == 40


def test_the_makes_in_play_and_their_models_come_first_whatever_the_cut():
    from dealer_core.voice.prompts import hotword_names

    vocab = {"audi": ["a4", "q7"], "bmw": ["m3"], "toyota": ["camry", "yaris", "rav4"], "volvo": ["xc90"]}
    assert hotword_names(vocab)[:4] == ["Audi", "A4", "Q7", "BMW"]                                        # nothing in play: the stock's own order
    assert hotword_names(vocab, ["toyota"])[:4] == ["Toyota", "Camry", "Yaris", "RAV4"]
    assert hotword_names(vocab, ["toyota", "volvo", "nothing"], 6) == ["Toyota", "Camry", "Yaris", "RAV4", "Volvo", "XC90"]       # a cut keeps what the call is about
    assert sorted(hotword_names(vocab, ["toyota"])) == sorted(hotword_names(vocab))                       # the same names, only first or later
    assert hotword_names(None, ["toyota"]) == []


def test_the_makes_in_play_are_the_search_the_cars_shown_and_the_car_in_focus():
    from types import SimpleNamespace as N

    from dealer_core.voice.pipeline import makes_in_play

    service = N(get_car=lambda stock_id: N(make="Tesla") if stock_id == "AG-164" else None)
    s = N(filters={"make": "toyota"}, shown=[N(make="Toyota"), N(make="Kia")], car="AG-164")
    assert makes_in_play(s, service) == ["toyota", "kia", "tesla"]
    assert makes_in_play(N(filters={}, shown=[], car=None), service) == []


def test_the_pipeline_asks_for_the_names_with_the_makes_in_play_first():
    class Inventory:
        def vocabulary(self):
            return {"audi": ["a4"], "toyota": ["camry"]}

    _, client = server()
    hot = remote(client, hotwords_field=True)
    attach_stock_hotwords(hot, Inventory(), lambda: ["toyota"])
    assert hot.hotword_source() == ["Toyota", "Camry", "Audi", "A4"]


def test_the_start_up_check_works_with_hotwords_and_a_dead_server_still_fails_it_loudly():
    forms, client = server()
    remote(client, hotwords_field=True).warm_up()
    assert len(forms) == 1
    down = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        remote(down, hotwords_field=True).warm_up()


# ------------------------------------------------------------------------------------------------ wiring
def test_the_setting_is_off_by_default_and_the_environment_turns_it_on():
    assert load_config(environ={}).voice.stt_hotwords_field is False and load_config(environ={}).voice.stt_hotwords_max == 200
    assert load_config(environ={"VA__VOICE__STT_HOTWORDS_FIELD": "true"}).voice.stt_hotwords_field is True


def test_the_pipeline_gives_only_a_hotwords_recogniser_the_stock_and_reads_it_from_the_inventory_each_time():
    class Inventory:
        calls = 0

        def vocabulary(self):
            Inventory.calls += 1
            return VOCAB

    _, client = server()
    plain, hot = remote(client), remote(client, hotwords_field=True)
    attach_stock_hotwords(plain, Inventory())
    attach_stock_hotwords(hot, Inventory())
    assert plain.hotword_source is None
    assert hot.hotword_source()[:3] == ["Toyota", "Land Cruiser", "RAV4"] and Inventory.calls == 1                   # nothing in play: the stock's own order
    hot.hotword_source()
    assert Inventory.calls == 2


# ------------------------------------------------------------------------------------------------ the echo guard
class Engine:
    def __init__(self, texts, ignores_prompt):
        self.texts, self.ignores_prompt, self.prompts = list(texts), ignores_prompt, []

    def transcribe(self, pcm, prompt=None):
        self.prompts.append(prompt)
        return self.texts.pop(0), 5.0


AGENT = "Which day and time would you like to come in for the test drive?"


def test_a_recogniser_that_was_never_given_the_hint_is_not_asked_again_with_another():
    # a short run of digits after the digits hint used to mean "the hint was written down: listen again". With hot-words there was no hint.
    eng = Engine(["050, 056"], ignores_prompt=True)
    assert EchoGuard(eng, lambda: AGENT, lambda: "plain").transcribe(PCM, CONTACT)[0] == "050, 056" and len(eng.prompts) == 1
    old = Engine(["050, 056", "no"], ignores_prompt=False)
    assert EchoGuard(old, lambda: AGENT, lambda: "plain").transcribe(PCM, CONTACT)[0] == "no" and len(old.prompts) == 2           # unchanged for a Whisper server


def test_the_agents_own_question_coming_back_is_still_not_an_answer_with_no_second_try():
    eng = Engine(["Which day and time would you like to come in"], ignores_prompt=True)
    guard = EchoGuard(eng, lambda: AGENT, lambda: "plain")
    assert guard.transcribe(PCM, "x")[0] == UNCLEAR and len(eng.prompts) == 1 and guard.retries == 1
    eng2 = Engine(["tomorrow at nine"], ignores_prompt=True)
    assert EchoGuard(eng2, lambda: AGENT, lambda: "plain").transcribe(PCM, "x")[0] == "tomorrow at nine"


# ------------------------------------------------------------------------------------------------ the list written back
def test_a_transcript_that_is_the_hot_word_list_written_back_is_no_speech():
    """2 Nov live call: Qwen3-ASR wrote the list it was given for clips with no speech; it reached the agent as "um" and was answered."""
    names = ["Toyota", "Land Cruiser", "Macan", "Camry", "Yaris", "RAV4"]
    _, client = server("Toyota, Land Cruiser, Macan, Camry, Yaris, RAV4")
    r = remote(client, hotwords_field=True)
    r.hotword_source = lambda: names
    assert r.transcribe(PCM, None)[0] == "" and r.word_list_echoes == 1 and r.errors == 0


def test_a_caller_naming_two_cars_or_a_sentence_with_a_name_is_not_an_echo():
    from dealer_core.voice.stt import is_word_list_echo

    names = ["Toyota", "Land Cruiser", "Macan", "Camry", "Yaris", "RAV4"]
    assert not is_word_list_echo("Camry, Yaris.", names) and not is_word_list_echo("Yes, talk about the Macan.", names)
    assert not is_word_list_echo("Camry, Yaris, or the Macan, whichever is cheaper", names)
    assert is_word_list_echo("toyota, land cruiser, macan, camry.", names) and not is_word_list_echo("", names)
