"""The simulated cluster (src/dealer_core/sim/scale.py): the real router, governor and call cap against stand-in hardware. Smaller and
shorter than scripts/simulate_scale.py so it runs in a few seconds; the claims are the same."""
from dealer_core.sim.scale import FAST, SLOW, Scenario, run

RAMP_200 = [(0, 80), (30, 200)]


def sc(name, hw, **kw):
    return Scenario(name, hw, RAMP_200, duration_s=360, measure_from_s=180, min_workers=16, **kw)


def test_healthy_hardware_answers_within_a_second_and_nobody_is_turned_away():
    r = run(sc("healthy", FAST))
    assert r["answer p90"] < 1.0 and r["busy %"] == 0 and r["answers > 2 s %"] == 0


def test_when_most_gpu_capacity_fails_protection_keeps_answers_fast_and_without_it_they_are_not():
    on = run(sc("gpu loss", FAST, model_slots_at=[(150, 2)], adaptive=True))
    off = run(sc("gpu loss", FAST, model_slots_at=[(150, 2)], adaptive=False))
    assert off["answer p90"] > 3.0 and off["answers > 2 s %"] > 20                       # without it: everyone waits in the model's queue
    assert on["answer p90"] < 1.1 and on["answers > 2 s %"] < 3 and on["shed %"] > 10   # with it: the code answers part of the turns
    assert on["busy %"] < 2


def test_a_slow_gpu_makes_replies_simpler_not_slower():
    r = run(sc("slow gpu", SLOW))
    full, prefix, code = (int(x) for x in r["model full/prefix/code %"].split("/"))
    assert prefix > full and r["answer p90"] < 1.0


def test_a_cold_start_is_caught_up_when_the_autoscaler_counts_the_callers_turned_away():
    on = run(Scenario("cold", FAST, RAMP_200, duration_s=360, measure_from_s=120, min_workers=16, cold_start=True, cold_min=4, adaptive=True))
    off = run(Scenario("cold", FAST, RAMP_200, duration_s=360, measure_from_s=120, min_workers=16, cold_start=True, cold_min=4, adaptive=False,
                       scale_on_demand=False))
    assert on["busy %"] < off["busy %"]


def test_the_router_reports_callers_turned_away_for_the_autoscaler():
    from fastapi.testclient import TestClient

    from dealer_core.gateway.__main__ import create_gateway
    from dealer_core.shared.state import MemoryStore, WorkerRegistry

    with TestClient(create_gateway(WorkerRegistry(MemoryStore()), ["http://localhost:8080"])) as client:
        assert client.get("/assign").status_code == 503
        body = client.get("/metrics").text
    assert "voice_router_turned_away_30s 1" in body and "voice_router_capacity 0" in body
