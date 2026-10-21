"""Voice path: microphone -> VAD / turn detection -> speech-to-text -> dialogue engine -> text-to-speech -> speaker.

`stt.py` and `tts.py` are plain classes (no framework) so they can be timed and tested alone. `services.py` wraps them as Pipecat services,
`brain.py` is the Pipecat processor around `DialogueEngine`, and `pipeline.py` builds and runs the whole thing. Everything runs locally.
"""
