"""Numbers a dashboard (Prometheus + Grafana) collects from every voice worker at GET /metrics. No transcripts, no phone numbers.

  voice_calls_active             calls on this worker now
  voice_calls_total{result}      calls served / refused (busy)
  voice_reply_seconds            end of the caller's speech -> first sound of the reply (histogram: median, 90th percentile, slow tail)
  voice_turn_level_total{level}  turns answered at full / prefix / code quality
  voice_model_in_flight          turns waiting on the model right now
  voice_model_first_ms           the governor's estimate of the model's time to a first sentence

Alert on: the 90th percentile of voice_reply_seconds above ~2 s, the share of `code` turns rising, refused calls appearing.
"""
from __future__ import annotations

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
    from prometheus_client import CONTENT_TYPE_LATEST
except ImportError:                                            # metrics are optional: the server runs without the package
    CollectorRegistry = None


class Metrics:
    def __init__(self) -> None:
        self.enabled = CollectorRegistry is not None
        if not self.enabled:
            return
        self.registry = CollectorRegistry()
        self.active = Gauge("voice_calls_active", "calls on this worker now", registry=self.registry)
        self.calls = Counter("voice_calls_total", "calls by result", ["result"], registry=self.registry)
        self.reply = Histogram("voice_reply_seconds", "end of the caller's speech to the first sound of the reply",
                               buckets=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0), registry=self.registry)
        self.level = Counter("voice_turn_level_total", "turns by quality level", ["level"], registry=self.registry)
        self.in_flight = Gauge("voice_model_in_flight", "turns waiting on the model", registry=self.registry)
        self.model_ms = Gauge("voice_model_first_ms", "model's estimated time to a first sentence (governor)", registry=self.registry)
        self._levels_seen: dict[str, int] = {}

    def call(self, result: str) -> None:
        if self.enabled:
            self.calls.labels(result).inc()

    def reply_time(self, seconds: float) -> None:
        if self.enabled and seconds > 0:
            self.reply.observe(seconds)

    def refresh(self, active: int, governor) -> None:
        if not self.enabled:
            return
        self.active.set(active)
        if governor is not None:
            snap = governor.snapshot()
            self.in_flight.set(snap["in_flight"])
            self.model_ms.set(snap["full_first_ms"])
            for level, n in snap["turns"].items():                 # the governor counts; publish the increase since last time
                new = n - self._levels_seen.get(level, 0)
                if new > 0:
                    self.level.labels(level).inc(new)
                self._levels_seen[level] = n

    def render(self) -> tuple[bytes, str]:
        if not self.enabled:
            return b"# prometheus_client not installed\n", "text/plain"
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
