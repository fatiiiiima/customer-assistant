"""A worker told to stop (SIGTERM, in Kubernetes a 900 s grace period for the calls in progress) must stop being handed new callers at once. Before, its registry entry went on
saying "room for 16 more" until the worker was gone, and the router sent callers to an address that no longer accepted them."""
import json
import random
import time
import types

from fastapi.testclient import TestClient

from dealer_core.agent.governor import QualityGovernor
from dealer_core.gateway.__main__ import Assigner
from dealer_core.server.__main__ import drain_on_stop
from dealer_core.server.app import ServerState, create_server
from dealer_core.shared.state import MemoryStore, WorkerRegistry


def state_with(reg, worker_id="w1", max_calls=16):
    return ServerState(app=types.SimpleNamespace(engine=types.SimpleNamespace(governor=QualityGovernor("auto"))), parts=None, cfg=None,
                       max_calls=max_calls, registry=reg, worker_id=worker_id, public_url=f"ws://{worker_id}/voice")


def wait_for(reg, predicate, secs=3.0):
    end = time.time() + secs
    while time.time() < end:
        ws = reg.workers()
        if ws and predicate(ws[0]):
            return ws[0]
        time.sleep(0.02)
    raise AssertionError(f"the registry never showed it: {reg.workers()}")


def test_a_draining_worker_reports_no_room_at_once_and_says_so():
    reg = WorkerRegistry(MemoryStore())
    state = state_with(reg)
    with TestClient(create_server(state)) as client:
        assert wait_for(reg, lambda w: w["max"] == 16)["draining"] is False
        state.drain()
        w = wait_for(reg, lambda w: w.get("draining"), secs=1.2)            # the beat is every 2 s: this is the report made when it happened
        assert w["max"] == 0
        assert client.get("/health").json()["draining"] is True


def test_a_draining_worker_turns_a_late_caller_away():
    reg = WorkerRegistry(MemoryStore())
    state = state_with(reg)
    state.drain()
    with TestClient(create_server(state)) as client:
        with client.websocket_connect("/voice", headers={"origin": "http://localhost:8080"}) as ws:
            assert json.loads(ws.receive_text()) == {"type": "busy"}


def test_the_registry_never_picks_a_draining_worker():
    reg = WorkerRegistry(MemoryStore())
    reg.beat("old", "ws://old/voice", 1, 16, {"draining": True})
    reg.beat("new", "ws://new/voice", 9, 16)
    assert reg.pick()["id"] == "new"                                         # the old one has far more room on paper
    reg.beat("new", "ws://new/voice", 9, 16, {"draining": True})
    assert reg.pick() is None


def test_the_router_never_assigns_to_a_draining_worker():
    reg = WorkerRegistry(MemoryStore())
    reg.beat("old", "ws://old/voice", 0, 16, {"draining": True})
    reg.beat("new", "ws://new/voice", 2, 16)
    a = Assigner(reg, rng=random.Random(3))
    a.refresh()
    assert {a.pick()["id"] for _ in range(10)} == {"new"}
    reg.beat("new", "ws://new/voice", 2, 16, {"draining": True})
    a.refresh()
    assert a.pick() is None


def test_the_first_stop_signal_marks_the_worker_before_uvicorn_begins_to_stop():
    seen = []

    class Server:                                                            # stands in for uvicorn.Server
        def __init__(self, config):
            self.config = config

        def handle_exit(self, sig, frame):
            seen.append(("uvicorn", state.draining))

    state = state_with(None)
    Draining = drain_on_stop(Server)
    Draining(object(), state).handle_exit(15, None)
    assert seen == [("uvicorn", True)]                                       # already draining when uvicorn's own handling starts
