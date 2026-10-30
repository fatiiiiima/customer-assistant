"""voice/remote_tts.py: Chatterbox Turbo from its streaming server, Kokoro as the fallback (owner, 29 Oct: "Chatterbox sounds better"; docs/ARCH_REVIEW.md section 39)."""
from contextlib import contextmanager

import numpy as np
import pytest

from dealer_core.voice.remote_tts import StreamingVoice


class Kokoro:
    sample_rate, name = 24000, "kokoro"

    def __init__(self):
        self.said = []

    def stream(self, text):
        self.said.append(text)
        yield np.full(100, 0.5, dtype=np.float32)

    def warm_up(self):
        pass

    @staticmethod
    def to_pcm16(x):
        return (x * 32767).astype(np.int16).tobytes()


class Response:
    def __init__(self, parts, fail_after=None):
        self.parts, self.fail_after = parts, fail_after

    def raise_for_status(self):
        pass

    def iter_bytes(self):
        for i, p in enumerate(self.parts):
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionError("server went away")
            yield p


class Server:
    """Stands in for httpx: records each request and answers with the given byte pieces (or fails)."""

    def __init__(self, parts=(), fail=False, fail_after=None):
        self.parts, self.fail, self.fail_after, self.asked = list(parts), fail, fail_after, []

    @contextmanager
    def stream(self, method, url, json=None):
        self.asked.append(json)
        if self.fail:
            raise ConnectionError("no server")
        yield Response(self.parts, self.fail_after)


def pcm(values):
    return np.asarray(values, dtype=np.int16).tobytes()


def test_the_server_s_audio_is_passed_on_as_it_comes_even_when_a_sample_is_split():
    whole = pcm([1000, -2000, 3000, -4000])
    server = Server([whole[:3], whole[3:]])                                  # a 16-bit sample split across two chunks
    v = StreamingVoice("http://x/v1", Kokoro(), client=server)
    got = np.concatenate(list(v.stream("There's the Macan at 149,000 dirhams.")))
    assert np.allclose(got * 32768, [1000, -2000, 3000, -4000])
    assert server.asked[0]["stream"] is True and "one hundred and forty nine thousand" in server.asked[0]["input"]     # numbers as words, as Kokoro gets them


def test_no_server_and_kokoro_says_the_sentence():
    k = Kokoro()
    v = StreamingVoice("http://x/v1", k, client=Server(fail=True))
    out = list(v.stream("Which one would you like?"))
    assert k.said == ["Which one would you like?"] and len(out) == 1 and v.fallbacks == 1


def test_a_server_that_stops_mid_sentence_is_not_followed_by_kokoro_saying_it_again():
    k = Kokoro()
    v = StreamingVoice("http://x/v1", k, client=Server([pcm([1, 2]), pcm([3, 4])], fail_after=1))
    out = list(v.stream("It has a 3.0-litre engine."))
    assert len(out) == 1 and k.said == []                                     # what was said stands; no repeat in another voice


def test_a_short_sentence_is_kept_and_played_at_once_the_next_time():
    server = Server([pcm([5, 6, 7, 8])])
    v = StreamingVoice("http://x/v1", Kokoro(), client=server)
    list(v.stream("Shall I book it?"))
    again = list(v.stream("Shall I book it?"))
    assert len(server.asked) == 1 and len(again) == 1


def test_a_worker_whose_voice_server_is_down_still_starts():
    StreamingVoice("http://x/v1", Kokoro(), client=Server(fail=True)).warm_up()


def test_the_setting_chooses_it():
    from dealer_core.config import load_config

    cfg = load_config(environ={"VA__TTS__BACKEND": "chatterbox"})
    assert cfg.tts.backend == "chatterbox" and cfg.tts.url.endswith("/v1") and cfg.tts.first_timeout_s > 0
    with pytest.raises(Exception):
        load_config(environ={"VA__TTS__BACKEND": "fish"})


def test_phrases_made_ahead_of_time_ask_for_low_priority_and_live_sentences_do_not():
    """The server stops a low-priority sentence for a live one (scripts/chatterbox_server.py): 29 Oct, the start-up phrases held the caller's sentences back."""
    server = Server([pcm([1, 2])])
    v = StreamingVoice("http://x/v1", Kokoro(), client=server)
    list(v._remote("Shall I book it?", background=True))
    list(v.stream("There's the Macan."))
    assert server.asked[0]["priority"] == "low" and "priority" not in server.asked[1]


def test_eve_is_asked_for_by_name():
    """Section 42: Pocket-TTS's servers take the voice by name (eve from scripts/pocket_server.py, alba from the speech toolkit's)."""
    server = Server([pcm([1, 2])])
    list(StreamingVoice("http://x/v1", Kokoro(), model="pocket", voice="eve", client=server).stream("Hello there."))
    assert server.asked[0]["voice"] == "eve" and server.asked[0]["model"] == "pocket"


def test_eve_down_and_alba_says_the_sentence_not_kokoro():
    """Owner, 30 Oct: "make alba the fallback cause kokoro sucks"."""
    k, alba_server = Kokoro(), Server([pcm([7, 8])])
    alba = StreamingVoice("http://alba/v1", k, model="pocket", voice="alba", client=alba_server)
    eve = StreamingVoice("http://eve/v1", alba, model="pocket", voice="eve", client=Server(fail=True))
    out = np.concatenate(list(eve.stream("Which one would you like?")))
    assert np.allclose(out * 32768, [7, 8]) and alba_server.asked[0]["voice"] == "alba" and k.said == [] and eve.fallbacks == 1


def test_both_pocket_servers_down_and_kokoro_still_speaks():
    k = Kokoro()
    alba = StreamingVoice("http://alba/v1", k, model="pocket", voice="alba", client=Server(fail=True))
    eve = StreamingVoice("http://eve/v1", alba, model="pocket", voice="eve", client=Server(fail=True))
    assert len(list(eve.stream("Which one would you like?"))) == 1 and k.said == ["Which one would you like?"]
    eve.warm_up()                                                              # a worker with neither server still starts


def test_pocket_is_the_default_voice():
    from dealer_core.config import load_config

    t = load_config(environ={}).tts
    assert t.backend == "pocket" and t.pocket_voice == "eve" and t.pocket_fallback_voice == "alba"
    assert t.pocket_url != t.pocket_fallback_url and 0 < t.pocket_first_timeout_s <= 1.0


def test_pocket_gets_agmc_as_written_and_kokoro_keeps_the_spelling():
    """Owner, 30 Oct: "when the bot says A G M C, can it say it faster". Spelled out, eve stops after every letter (section 43)."""
    server = Server([pcm([1, 2])])
    list(StreamingVoice("http://x/v1", Kokoro(), model="pocket", voice="eve", spell_letters=False, client=server).stream("Welcome to AGMC, a GCC SUV."))
    assert server.asked[0]["input"] == "Welcome to AGMC, a GCC SUV."
    from dealer_core.voice.tts import pronounce

    assert "eigh gee em see" in pronounce("Welcome to AGMC.")


def test_the_worker_sends_pocket_the_letters_as_written():
    import inspect

    from dealer_core.voice import pipeline

    assert inspect.getsource(pipeline.load_engines).count("spell_letters=False") == 2
