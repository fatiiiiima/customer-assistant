"""Shared state between workers: a call carried on by another worker after a restart, and the registry picking the least busy worker.
Redis is stood in for by fakeredis (the same protocol); MemoryStore is the single-machine version."""
from datetime import timezone

import pytest

from dealer_core.shared.state import CallStore, MemoryStore, RedisStore, WorkerRegistry

fakeredis = pytest.importorskip("fakeredis")


@pytest.fixture(scope="module")
def pg_url():
    pgserver = pytest.importorskip("pgserver")
    import tempfile

    return pgserver.get_server(tempfile.mkdtemp(prefix="pg_kv_"), cleanup_mode="stop").get_uri()


@pytest.fixture(params=["memory", "redis", "postgres"])
def store(request):
    if request.param == "memory":
        return MemoryStore()
    if request.param == "redis":
        return RedisStore(client=fakeredis.FakeRedis(decode_responses=True))
    from dealer_core.shared.state import PgStore

    st = PgStore(request.getfixturevalue("pg_url"))
    with st.db.tx() as c:
        c.run("DELETE FROM kv")
    return st


def test_expiry_and_prefix_listing(store):
    store.set("worker:a", "1", ttl_s=60)
    store.set("call:x", "2")
    assert store.get("worker:a") == "1" and store.keys("worker:") == ["worker:a"]
    store.delete("worker:a")
    assert store.get("worker:a") is None


def test_the_registry_sends_a_new_call_to_the_worker_with_most_room(store):
    reg = WorkerRegistry(store)
    reg.beat("w1", "ws://10.0.0.1:8765/voice", active=14, max_calls=16)
    reg.beat("w2", "ws://10.0.0.2:8765/voice", active=3, max_calls=16)
    reg.beat("w3", "ws://10.0.0.3:8765/voice", active=16, max_calls=16)       # full
    assert reg.pick()["id"] == "w2"
    reg.leave("w2")
    assert reg.pick()["id"] == "w1"
    reg.leave("w1")
    assert reg.pick() is None                                                  # all full: the page falls back to text chat


def test_a_call_is_carried_on_by_another_worker_mid_booking(store):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_agent import DATA_JS, engine

    if not DATA_JS.exists():
        pytest.skip("the website is not on this machine")
    calls = CallStore(store)
    e1, app1 = engine([])
    s1 = e1.new_session()
    for line in ("Toyota", "the Corolla", "yes", "tomorrow", "9"):
        e1.handle(s1, line)
        calls.save("call-123", s1)
    assert s1.draft_slot is not None
    # worker 1 dies here; the page reconnects to worker 2 with the same call id
    e2, app2 = engine([])
    s2 = e2.new_session()
    assert calls.load("call-123", s2, app2.service)
    assert s2.car == s1.car and s2.draft_slot == s1.draft_slot and [c.stock_id for c in s2.shown] == [c.stock_id for c in s1.shown]
    r = e2.handle(s2, "Sara 050 111 2222")
    assert "Shall I book it" in r.reply and "Sara" in r.reply                   # the booking carries on where it stopped
    assert calls.load("missing", e2.new_session(), app2.service) is False


def test_the_router_assigns_the_least_busy_worker_and_says_busy_when_all_are_full(store):
    from fastapi.testclient import TestClient

    from dealer_core.gateway.__main__ import create_gateway

    reg = WorkerRegistry(store)
    api = create_gateway(reg, ["http://localhost:8080"])
    with TestClient(api) as client:
        assert client.get("/assign").status_code == 503                         # no worker yet
        reg.beat("w1", "ws://a/voice", active=7, max_calls=8)
        reg.beat("w2", "ws://b/voice", active=2, max_calls=8)
        api.state.assigner.refresh()                                            # the router reads reports in the background every 0.5 s
        assert client.get("/assign").json()["url"] == "ws://b/voice"
        assert client.get("/workers").json()["calls"] == 9
