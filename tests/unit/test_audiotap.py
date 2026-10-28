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
