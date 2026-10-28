import asyncio
import csv

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame

from dealer_core.voice.audiotap import AudioTap


def test_the_tap_writes_the_audio_and_a_clock_for_every_chunk(tmp_path):
    async def go():
        tap = AudioTap(tmp_path, "in", InputAudioRawFrame)
        tap.record(b"\x01\x00" * 320, 16000)
        tap.record(b"\x02\x00" * 320, 16000)
        await tap.cleanup()

    asyncio.run(go())
    pcm = next(tmp_path.glob("*_in.pcm")).read_bytes()
    rows = list(csv.DictReader(next(tmp_path.glob("*_in.csv")).open()))
    assert len(pcm) == 1280 and [r["bytes"] for r in rows] == ["640", "640"] and rows[0]["sample_rate"] == "16000"


def test_the_pipeline_has_no_recorder_unless_asked(monkeypatch):
    import inspect

    from dealer_core.voice import pipeline

    monkeypatch.delenv("VA_DUMP_AUDIO_DIR", raising=False)
    src = inspect.getsource(pipeline.build_pipeline)
    assert 'os.environ.get("VA_DUMP_AUDIO_DIR")' in src and "if dump" in src


def test_the_browser_is_named_from_the_user_agent():
    from dealer_core.server.app import browser_of

    chrome = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    safari = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15"
    edge = chrome + " Edg/130.0.0.0"
    assert (browser_of(chrome), browser_of(safari), browser_of(edge), browser_of(""), browser_of("curl/8")) == ("Chrome", "Safari", "Edge", "unknown", "unknown")


def test_the_words_are_kept_beside_the_audio_so_a_bad_call_can_be_scored(tmp_path):
    """28 Oct: "it understood me at first and then not" could not be checked, because only the audio was kept and the transcripts were not logged."""
    async def go():
        tap = AudioTap(tmp_path, "in", InputAudioRawFrame)
        tap.note("stt", "is it accident free")
        tap.note("heard", "is it accident free")
        tap.note("said", "It's accident-free.")
        await tap.cleanup()

    asyncio.run(go())
    rows = [line.split("\t") for line in next(tmp_path.glob("*_in.txt")).read_text().splitlines()]
    assert [(r[1], r[2]) for r in rows] == [("stt", "is it accident free"), ("heard", "is it accident free"), ("said", "It's accident-free.")]
    assert all(float(r[0]) > 1e9 for r in rows)                      # the same clock as the audio's csv


def test_the_recogniser_reports_its_words_only_to_a_diagnostic_listener():
    import numpy as np

    from dealer_core.voice.services import OpenVinoWhisperSTTService

    class Engine:
        name = "fake"

        def transcribe(self, pcm, prompt=None):
            return "tell me about the macan", 1.0

    tone = (np.sin(np.arange(16000) / 5) * 0.2 * 32768).astype(np.int16).tobytes()
    seen = []

    async def run(listener):
        stt = OpenVinoWhisperSTTService(engine=Engine())
        stt.on_text = listener
        return [f.text async for f in stt.run_stt(tone)]

    assert asyncio.run(run(None)) == ["tell me about the macan"]            # no listener: nothing extra happens
    assert asyncio.run(run(seen.append)) == ["tell me about the macan"] and seen == ["tell me about the macan"]


def test_the_event_loop_never_touches_the_files(tmp_path):
    """B7: record() and note() run on the loop that carries every call's audio. They used to write and flush four times per 20 ms frame; now a thread of the tap's own does."""
    import threading
    import time

    seen = {"threads": set(), "flushes": 0}

    class Spy:
        def __init__(self, real):
            self.real = real

        def write(self, data):
            seen["threads"].add(threading.current_thread().name)
            return self.real.write(data)

        def flush(self):
            seen["threads"].add(threading.current_thread().name)
            seen["flushes"] += 1
            self.real.flush()

        def close(self):
            self.real.close()

    async def go():
        tap = AudioTap(tmp_path, "in", InputAudioRawFrame)
        tap._pcm, tap._idx, tap._notes = Spy(tap._pcm), Spy(tap._idx), Spy(tap._notes)
        before = time.time()
        for i in range(300):                                          # six seconds of 20 ms frames, arriving at once
            tap.record(bytes([i % 256, 0]) * 320, 16000)
        tap.note("said", "hello")
        await tap.cleanup()
        return before

    before = asyncio.run(go())
    assert threading.current_thread().name not in seen["threads"] and seen["threads"] == {"audiotap-in"}
    assert seen["flushes"] <= 6                                         # about once a second at most, not 3 per frame (900)
    rows = list(csv.DictReader(next(tmp_path.glob("*_in.csv")).open()))
    pcm = next(tmp_path.glob("*_in.pcm")).read_bytes()
    assert len(rows) == 300 and len(pcm) == 300 * 640 and pcm[640:642] == bytes([1, 0]) and pcm[299 * 640] == 299 % 256      # nothing lost, in order
    assert all(float(r["unix_time"]) >= before - 0.001 for r in rows) and float(rows[-1]["unix_time"]) - float(rows[0]["unix_time"]) < 1.0   # the time each frame ARRIVED, not when it was written
    assert "\tsaid\thello\n" in next(tmp_path.glob("*_in.txt")).read_text()
