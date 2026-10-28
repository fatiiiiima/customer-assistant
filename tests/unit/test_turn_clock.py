"""28 Oct review (B1): the turn clock was built with Smart Turn's 0.2 s, but tap mode runs the voice detector at 0.6 s (0.9 s after a phone-number question). The clock finds the
real end of speech by subtracting the silence the detector waited, so every time shown on the page, in the metrics and in the admission controller was 0.4 s short, and the
worker's call cap held replies to about 1.4 s while believing it was 1 s."""
import inspect

from pipecat.frames.frames import BotStartedSpeakingFrame, VADUserStoppedSpeakingFrame

from dealer_core.voice.status import TurnClock


def one_turn(stop_secs, detector_wait_changes_to=None):
    now = [100.0]
    got = []
    state = {"wait": 0.6}
    clock = TurnClock(stop_secs(state) if callable(stop_secs) else stop_secs, lambda total, parts: got.append((total, parts)), clock=lambda: now[0])
    clock.see(VADUserStoppedSpeakingFrame(stop_secs=state["wait"]), "vad")            # the detector reports the end 0.6 s after the caller really stopped
    if detector_wait_changes_to is not None:
        state["wait"] = detector_wait_changes_to                                       # (the pipeline re-arms for the next turn as soon as this one is answered)
    now[0] += 1.5
    clock.see(BotStartedSpeakingFrame(), "tts")
    return got[0]


def test_the_time_includes_the_silence_the_detector_really_waited():
    total, parts = one_turn(lambda state: (lambda: state["wait"]))
    assert round(total, 2) == 2.1 and parts["waiting for you to finish"] == 0.6      # 0.6 s of waiting + 1.5 s to the first sound
    assert round(one_turn(0.2)[0], 2) == 1.7                                           # what it said before: 0.4 s short


def test_a_change_of_the_wait_after_the_turn_began_does_not_move_this_turns_time():
    total, parts = one_turn(lambda state: (lambda: state["wait"]), detector_wait_changes_to=0.9)
    assert round(total, 2) == 2.1 and parts["waiting for you to finish"] == 0.6


def test_a_plain_number_still_works_as_before():
    assert round(one_turn(0.6)[0], 2) == 2.1


def test_the_server_gives_the_clock_the_pipelines_own_wait_and_the_pipeline_exposes_it():
    from dealer_core.server import app
    from dealer_core.voice import pipeline

    assert 'getattr(brain_ref.get("brain"), "vad_stop_secs"' in inspect.getsource(app)
    assert "brain.vad_stop_secs = stop_now" in inspect.getsource(pipeline.build_pipeline) and "current_stop[0]" in inspect.getsource(pipeline.build_pipeline)
