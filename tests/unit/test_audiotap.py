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
