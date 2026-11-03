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


# ------------------------------------------------------------------------------------------------ B5: the probe rule, measured in the cluster simulation
def probes(sc_, **kw):
    s = Scenario(sc_, FAST, RAMP_200, duration_s=360, measure_from_s=180, min_workers=16, **kw)
    return run(s)


def test_a_healthy_cluster_does_not_probe_workers_whose_own_turns_are_timing_the_model():
    new = probes("healthy", probe_skip_fresh=True)
    old = probes("healthy", probe_skip_fresh=False)
    assert old["probe requests"] > 5 * new["probe requests"]                   # measured 1,257 against 68 over 10 simulated minutes at 500 callers
    assert abs(new["answer p90"] - old["answer p90"]) < 0.05 and new["answers > 2 s %"] == 0


def test_workers_return_to_the_full_level_soon_after_the_model_server_recovers_even_when_skipping_fresh_probes():
    r = probes("outage", model_slots_at=[(180, 2), (240, 160)], probe_skip_fresh=True)
    assert r["seconds to recover"] is not None and r["seconds to recover"] <= 120      # 60 s measured at 500 callers


# ------------------------------------------------------------------------------------------------ 1,000 and 1,500 callers (docs/SCALE_MODEL.md)
def test_the_manifests_72_workers_hold_1000_callers_with_the_answer_time_under_a_second():
    from dealer_core.sim.scale import RAMP_1000

    r = run(Scenario("1000", FAST, RAMP_1000, max_workers=72, adaptive=True))
    assert r["busy %"] < 1 and r["answer p90"] < 1.0 and r["answers > 2 s %"] == 0 and r["workers max"] <= 72


def test_more_callers_need_more_workers_and_more_model_in_flight_and_the_workers_follow_the_formula():
    from dealer_core.sim.scale import model_capacity_sweep

    rows = {n: model_capacity_sweep(n, [160], 180)[0] for n in (500, 1000, 1500)}
    assert rows[500]["model peak in flight"] < rows[1000]["model peak in flight"] < rows[1500]["model peak in flight"]
    for n, r in rows.items():
        assert abs(r["workers max"] - n / (16 * 0.7)) <= 4                  # SCALE_MODEL.md: callers / (16 x 0.7): 45, 90, 134 (the simulation reached 45, 89, 132)


def test_with_too_little_model_capacity_callers_keep_getting_through_and_answers_stay_near_a_second():
    from dealer_core.sim.scale import model_capacity_sweep

    r = model_capacity_sweep(1000, [10], 120)[0]
    assert r["busy %"] < 1 and r["answer p90"] < 1.3 and r["shed %"] > 20          # wording and shedding give way first, not the callers


# ------------------------------------------------------------------------------------------------ the measured stack (ARCH_REVIEW section 103)
def test_a_serial_server_makes_each_request_wait_for_the_ones_ahead_of_it_and_copies_share_the_work():
    from dealer_core.sim.scale import Serial, Sim

    sim = Sim(1)
    one = Serial(sim, 1)
    c0, w0 = one.ask(1.0)
    c1, w1 = one.ask(1.0)
    assert w0 == 0 and 0.4 < w1 < 2.5                                     # the second starts when the first is done (the service time varies by about 20%)
    two = Serial(sim, 2)
    assert two.ask(1.0)[1] == 0 and two.ask(1.0)[1] == 0                  # two copies: both start at once
    solo = Serial(sim, 1)
    copy, _ = solo.ask(2.0)
    solo.give_up(copy)                                                    # the caller closed it: the copy is free again
    assert solo.ask(1.0)[1] == 0


def short(sc, seconds=300.0):
    sc.duration_s, sc.measure_from_s = seconds, 150.0
    return sc


def test_the_stack_sized_as_measured_holds_the_answer_time_and_as_built_it_does_not():
    from dealer_core.sim.scale import measured_scenarios, run

    by = {s.name: s for s in measured_scenarios(True)}
    sized = run(short(by["100 callers, 8 voice servers, 5 Qwen3 (sized)"]))
    built = run(short(by["100 callers, 1 voice server, 1 Qwen3 (as built)"]))
    assert sized["answer p90"] < 1.5 and sized["answers > 2 s %"] == 0 and sized["fallback voice %"] == 0 and sized["voice server busy %"] < 75 and sized["recogniser busy %"] < 75
    assert built["answer p90"] > 10 and built["recogniser busy %"] > 90                  # one recogniser copy cannot hear 100 callers: the queue grows


def test_the_calibration_agrees_with_the_8_caller_run_of_the_laptop():
    """Measured (section 98): 8 callers, median 0.94 to 1.05 s, p90 1.5 to 2.2 s, 4 to 11% over 2 s, no fallback voice to speak of."""
    from dealer_core.sim.scale import measured_scenarios, run

    sc = next(s for s in measured_scenarios(False) if s.name == "calibration: 8 callers, as the laptop")
    r = run(sc)
    assert 0.7 <= r["answer p50"] <= 1.2 and 1.2 <= r["answer p90"] <= 2.4 and r["answers > 2 s %"] <= 12 and r["fallback voice %"] <= 3


def test_a_voice_server_that_cannot_answer_in_time_hands_the_sentence_to_the_fallback_voice():
    from dealer_core.sim.scale import Hardware, Scenario, run

    hw = Hardware("one voice copy", model_slots=64, decide_ms=230, first_ms=650, code_share=0.0, tts_copies=1, endpoint_s=0.24)
    r = run(Scenario("flooded voice server", hw, [(0, 60)], adaptive=False, min_workers=4, max_workers=4, duration_s=300, measure_from_s=100, think_s=(0.1, 0.3)))
    assert r["fallback voice %"] > 20 and r["voice server busy %"] > 80


def test_the_new_fields_are_off_by_default_so_the_older_scenarios_are_what_they_were():
    from dealer_core.sim.scale import Hardware

    hw = Hardware("x", model_slots=1, decide_ms=1, first_ms=1)
    assert (hw.decide_share, hw.ahead_share, hw.stt_copies, hw.tts_copies) == (1.0, 0.0, 0, 0)
