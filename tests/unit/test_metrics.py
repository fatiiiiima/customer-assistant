"""The worker's /metrics numbers: calls, reply times, quality levels."""
import pytest

pytest.importorskip("prometheus_client")
from dealer_core.agent.governor import QualityGovernor   # noqa: E402
from dealer_core.server.metrics import Metrics          # noqa: E402


def test_metrics_render_calls_reply_times_and_levels():
    m, g = Metrics(), QualityGovernor("auto")
    m.call("served"), m.call("refused")
    m.reply_time(0.8), m.reply_time(2.6)
    g.choose(), g.choose()
    m.refresh(active=3, governor=g)
    body = m.render()[0].decode()
    assert 'voice_calls_total{result="served"} 1.0' in body and "voice_calls_active 3.0" in body
    assert 'voice_reply_seconds_bucket{le="1.0"} 1.0' in body and 'voice_turn_level_total{level="full"} 2.0' in body
    m.refresh(active=3, governor=g)                       # refreshing again does not count the same turns twice
    assert 'voice_turn_level_total{level="full"} 2.0' in m.render()[0].decode()
