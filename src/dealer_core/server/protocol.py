"""The wire format between the website and the voice server (one WebSocket per call).

browser -> server   binary: 16 kHz mono signed 16-bit PCM, any chunk size (the page sends ~20 ms)
                    text:   {"type": "mic", "on": true | false}   tap-to-talk: the visitor opened / closed the microphone (a tap while the agent speaks interrupts it)
                    text:   {"type": "hello", "car": "AG-101" | null, "mode": "tap" | "handsfree"}  first message; the car page the visitor is on and how they talk
                            {"type": "bye"}
server  -> browser  binary: signed 16-bit PCM mono at the rate in "ready" (the voice's own rate, 22.05 kHz)
                    text:   {"type": "ready", "sample_rate": 22050}
                            {"type": "line", "who": "you" | "agent", "turn": 3, "text": "..."}   live transcript (agent lines come sentence by sentence)
                            {"type": "cars", "ids": ["AG-101", ...]}                              cars to show as cards
                            {"type": "booking", "car_id", "name", "phone", "date": "YYYY-MM-DD", "time": "HH:MM"}   the page books it in its own store
                            {"type": "timing", "seconds": 1.4}                                      end of caller's speech -> first sound
                            {"type": "mic", "on": false, "reason": "turn"}                      tap-to-talk: your turn is over, the microphone is closed (tap to reply)
                            {"type": "stop"}                                                        the caller spoke over the agent: stop playing what is queued
                            {"type": "busy"} | {"type": "error", "message": "..."}                  then the server closes the socket
"""
from __future__ import annotations

import json

from pipecat.frames.frames import Frame, InputAudioRawFrame, InputTransportMessageFrame, InterruptionFrame, OutputAudioRawFrame, OutputTransportMessageFrame, OutputTransportMessageUrgentFrame
from pipecat.serializers.base_serializer import FrameSerializer

IN_RATE = 16000


class PortalSerializer(FrameSerializer):
    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        if isinstance(frame, InterruptionFrame):
            return json.dumps({"type": "stop"})                 # barge-in: the page drops the audio it has queued
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, (bytes, bytearray)):
            return InputAudioRawFrame(audio=bytes(data), sample_rate=IN_RATE, num_channels=1) if len(data) else None
        try:
            msg = json.loads(data)
        except (TypeError, ValueError):
            return None
        if isinstance(msg, dict) and msg.get("type") == "mic":      # tap-to-talk: the microphone gate in the pipeline opens and closes on these
            return InputTransportMessageFrame(message=msg)
        return None                                              # other text messages (hello, bye) are handled by the endpoint, not the pipeline
