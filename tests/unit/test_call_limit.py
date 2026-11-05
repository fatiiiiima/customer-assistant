"""How long one call may last (5 Nov, the owner: calls go on until the caller stops them). The old fixed 900 s is gone; a quiet call ends by itself."""
import inspect
import types

from dealer_core.config import VoiceCfg
from dealer_core.server import __main__ as server_main
from dealer_core.server.app import ServerState


def test_a_call_may_last_two_hours_not_fifteen_minutes():
    cfg = types.SimpleNamespace(voice=VoiceCfg())
    assert cfg.voice.max_call_secs == 7200.0
    assert ServerState.__dataclass_fields__["max_call_secs"].default == 7200.0


def test_the_quiet_call_still_ends_by_itself_long_before_the_limit():
    v = VoiceCfg()
    assert 0 < v.still_there_secs + v.hang_up_secs < 60           # "Are you still there?", then the goodbye: a forgotten open tab does not hold a place
    assert v.calls_kept_secs >= 86400.0                           # and the conversation is kept, so coming back carries on


def test_the_limit_comes_from_the_config_and_zero_means_none():
    assert "max_call_secs=cfg.voice.max_call_secs" in inspect.getsource(server_main)
    from dealer_core.server import app as server_app

    src = inspect.getsource(server_app.run_call)
    assert "if state.max_call_secs > 0 else None" in src and "if limit is not None" in src
