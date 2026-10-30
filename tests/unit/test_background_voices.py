"""Hands-free: voices across the room are not answered (owner, 30 Oct: "it keeps hearing people in the background, so it catches some random stuff and keeps
replying"; docs/ARCH_REVIEW.md section 43). A turn under `voice.background_below` of the loudest of the caller's recent turns is ignored: not answered, not
remembered, no "speak up"."""
import asyncio
import sys
from pathlib import Path

from pipecat.frames.frames import TTSSpeakFrame

sys.path.insert(0, str(Path(__file__).parent))

from test_speak_up import SUSPECT, VERY, Engine, hear  # noqa: E402

from dealer_core.config import load_config  # noqa: E402
from dealer_core.voice.brain import DealerBrain  # noqa: E402
from dealer_core.voice.services import SPEAK_UP  # noqa: E402


def make(background_below=0.25):
    levels = {"now": 0.15}
    said, pushed = [], []
    brain = DealerBrain(Engine(), speak_opening=False, on_say=said.append, level_of=lambda: levels["now"], speak_up=(VERY, SUSPECT), background_below=background_below)

    async def push(frame, direction=None):
        pushed.append(frame)
    brain.push_frame = push
    return brain, levels, said, pushed


def run(background_below, turns):
    async def go():
        brain, levels, said, pushed = make(background_below)
        for level, text in turns:
            levels["now"] = level
            await hear(brain, text)
        return brain, said, pushed
    return asyncio.run(go())


def test_a_voice_across_the_room_after_the_caller_has_spoken_is_ignored():
    brain, said, pushed = run(0.25, [(0.15, "I'm looking for a Toyota."), (0.03, "I'm going to drive a little bit.")])
    assert brain.engine.seen == ["I'm looking for a Toyota."] and brain.background == 1
    assert SPEAK_UP not in said and not any(isinstance(f, TTSSpeakFrame) and f.text == SPEAK_UP for f in pushed)


def test_the_caller_s_own_quieter_turns_are_still_answered():
    """The owner's recorded call (27 Oct): turns from 0.103 to 0.273, the quietest 0.37 of the loudest."""
    turns = [(0.221, "I'm looking for a Toyota."), (0.273, "What do you have?"), (0.113, "What are the other makes?"), (0.103, "Yes, talk about the Macan.")]
    brain, said, pushed = run(0.25, turns)
    assert brain.engine.seen == [t for _, t in turns] and brain.background == 0


def test_room_voices_heard_before_the_caller_do_not_set_the_bar():
    """30 Oct: the room was heard before the caller said anything; once the caller speaks, the room is under the bar."""
    brain, said, pushed = run(0.25, [(0.035, "I'm going to see you there."), (0.15, "I'm looking for a Toyota."), (0.035, "I didn't know.")])
    assert brain.engine.seen == ["I'm going to see you there.", "I'm looking for a Toyota."] and brain.background == 1


def test_off_means_every_turn_is_answered_as_before():
    brain, said, pushed = run(0.0, [(0.15, "I'm looking for a Toyota."), (0.03, "Tell me about the Camry.")])
    assert len(brain.engine.seen) == 2 and brain.background == 0


def test_it_is_on_for_hands_free_and_off_in_tap_mode():
    import inspect

    from dealer_core.voice import pipeline

    assert load_config(environ={}).voice.background_below == 0.25
    assert "background_below=vcfg.background_below if mic_gate is None else 0.0" in inspect.getsource(pipeline.build_pipeline)
