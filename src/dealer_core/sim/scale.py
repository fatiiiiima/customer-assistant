"""A simulated production cluster: hundreds of callers, many voice workers, a shared model server and recognizer, on a simulated clock.

What is REAL here (the production code, unchanged): the router's choice of worker (gateway Assigner), each worker's quality governor
(full / prefix / code and model load shedding, agent/governor.py) and its measured call cap (server/admission.py). The decision deadline and the
code's own decision when the model is skipped are the engine's rules (agent/engine.py), modelled as their timings.
What is a STAND-IN: the model server, the speech recognizer and each worker's CPU. Each has a speed and a capacity; past its capacity every
request on it slows in proportion (they share it), and a worker's own CPU work (speech detection, voice) slows sharply once it carries more
calls than its cores can keep up with. Change those numbers to stand for any hardware: the point is to see what the mechanisms do with it.

Not covered: real audio, real networks, the real model's answers. This shows that the control mechanisms keep the answer time where it should be
(and what they give up to do it: quality level, more workers, or "busy" -> text chat), not that a particular GPU is fast enough. The capacity
probe and the load test on the real hardware answer that.

    python scripts/simulate_scale.py                     # every scenario, protection on vs off
"""
from __future__ import annotations

import heapq
import math
import random
import statistics
from dataclasses import dataclass, field

from ..agent.governor import QualityGovernor
from ..gateway.__main__ import Assigner
from ..server.admission import AdmissionController


# ------------------------------------------------------------------------------------------------ the simulated clock
class Sim:
    """Processes are generators that yield how long to wait (seconds of simulated time)."""

    def __init__(self, seed: int = 1) -> None:
        self.t = 0.0
        self.rng = random.Random(seed)
        self._q: list = []
        self._n = 0

    def now(self) -> float:
        return self.t

    def spawn(self, gen, delay: float = 0.0) -> None:
        heapq.heappush(self._q, (self.t + delay, self._n, gen))
        self._n += 1

    def run(self, until: float) -> None:
        while self._q and self._q[0][0] <= until:
            at, _, gen = heapq.heappop(self._q)
            self.t = at
            try:
                wait = next(gen)
            except StopIteration:
                continue
            heapq.heappush(self._q, (self.t + max(0.0, wait), self._n, gen))
            self._n += 1


# ------------------------------------------------------------------------------------------------ hardware stand-ins
@dataclass
class Hardware:
    name: str
    model_slots: int              # requests the model server(s) serve together at full speed (all GPUs together)
    decide_ms: float              # the tool decision, unloaded
    first_ms: float               # the model's first sentence (full quality), unloaded
    stt_slots: int = 64           # recognitions served together at full speed
    stt_ms: float = 90.0
    worker_cpu_calls: int = 16    # calls one worker's CPU keeps up with (measured on the laptop: ~16 per process)
    tts_ms: float = 60.0          # the voice's first audio, unloaded
    endpoint_s: float = 0.25      # waiting for the caller to finish (Smart Turn)
    code_share: float = 0.6       # turns the code answers without the model (measured in the live calls: most of them)


class Shared:
    """A shared server (model or recognizer): full speed up to `slots` requests at once, then everyone slows in proportion."""

    def __init__(self, sim: Sim, slots: int, base_ms: float) -> None:
        self.sim, self.slots, self.base_ms = sim, slots, base_ms
        self.in_flight = self.peak = 0

    def begin(self, base_ms: float | None = None) -> float:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        jitter = math.exp(self.sim.rng.gauss(0, 0.2))
        return (base_ms or self.base_ms) / 1000 * max(1.0, self.in_flight / self.slots) * jitter

    def end(self) -> None:
        self.in_flight -= 1


# ------------------------------------------------------------------------------------------------ the cluster
@dataclass
class Worker:
    id: str
    ceiling: int
    governor: QualityGovernor
    admission: AdmissionController | None
    ready_at: float
    active: int = 0
    draining: bool = False

    def cap(self) -> int:
        return self.admission.adjust(self.active) if self.admission is not None else self.ceiling

    def accepts(self) -> bool:
        return self.active < self.cap()


@dataclass
class Stats:
    first: list[float] = field(default_factory=list)       # end of speech -> first sound
    answer: list[float] = field(default_factory=list)      # end of speech -> the answer itself
    levels: dict[str, int] = field(default_factory=lambda: {"code-routed": 0, "full": 0, "prefix": 0, "code": 0})
    bridged: int = 0
    shed: int = 0
    calls: int = 0
    refused: int = 0
    timeline: list[tuple] = field(default_factory=list)    # (t, callers wanting a call, in calls, workers ready, answer p90 last 30 s)


@dataclass
class Scenario:
    name: str
    hardware: Hardware
    callers: list[tuple[float, int]]                       # (from t, callers wanting to talk) steps
    duration_s: float = 600.0
    measure_from_s: float = 120.0                          # after the ramp
    model_slots_at: list[tuple[float, int]] = field(default_factory=list)   # (t, slots): a GPU fails / comes back
    adaptive: bool = True                                  # protection on: the measured call cap, model load shedding, the 1.2 s decision deadline.
                                                           # False = the code before it: a fixed cap, no shedding, a 5 s deadline
    ceiling: int = 16
    min_workers: int = 36                                  # what is running when the calls start (deploy/k8s: 36 replicas)
    max_workers: int = 72                                  # deploy/k8s: maxReplicas
    worker_start_s: float = 45.0                           # a new worker: scheduled, started, speech models loaded
    target_s: float = 1.0
    scale_on_demand: bool = True                           # the autoscaler also counts callers just turned away (not only those in calls)
    cold_start: bool = False                               # start from `cold_min` workers (everything scaled down overnight)
    cold_min: int = 8
    seed: int = 7


def callers_wanted(sc: Scenario, t: float) -> int:
    n = 0
    for at, k in sc.callers:
        if t >= at:
            n = k
    return n


def run(sc: Scenario) -> dict:
    sim = Sim(sc.seed)
    hw = sc.hardware
    model = Shared(sim, hw.model_slots, hw.decide_ms)
    stt = Shared(sim, hw.stt_slots, hw.stt_ms)
    router = Assigner(None, rng=random.Random(sc.seed), clock=sim.now)
    workers: list[Worker] = []
    st = Stats()

    def new_worker(start_delay: float) -> Worker:
        gov = QualityGovernor("auto", budget_ms=700, shed_after_ms=450 if sc.adaptive else 0, rng=random.Random(sim.rng.random()))
        w = Worker(f"w{len(workers) + 1}", sc.ceiling, gov,
                   AdmissionController(sc.ceiling, target_s=sc.target_s, clock=sim.now) if sc.adaptive else None, sim.t + start_delay)
        workers.append(w)
        return w

    for _ in range(sc.cold_min if sc.cold_start else sc.min_workers):
        new_worker(0.0)

    def ready() -> list[Worker]:
        return [w for w in workers if w.ready_at <= sim.t and not w.draining]

    def beats():                                            # workers report; the router reads the reports (both ~every second)
        while True:
            router.workers = [{"id": w.id, "url": w.id, "active": w.active, "max": w.cap()} for w in ready()]
            yield 1.0

    refused_at: list[float] = []

    def autoscaler():                                       # every 15 s, on DEMAND: callers in calls + callers just told "busy", at 70% of the caps
        while True:
            yield 15.0
            live = ready()
            starting = [w for w in workers if w.ready_at > sim.t]
            cap = sum(w.cap() for w in live) or 1
            demand = sum(w.active for w in live) + (sum(1 for t in refused_at if t >= sim.t - 30) if sc.scale_on_demand else 0)
            per_worker = cap / max(1, len(live))
            desired = max(sc.min_workers if not sc.cold_start else sc.cold_min, min(sc.max_workers, math.ceil(demand / (per_worker * 0.7))))
            for _ in range(desired - len(live) - len(starting)):
                new_worker(sc.worker_start_s)
            if desired < len(live) - 2:                     # scale down slowly: drain the emptiest
                emptiest = min(live, key=lambda w: w.active)
                emptiest.draining = True

    def gpu_events():
        for at, slots in sorted(sc.model_slots_at):
            yield max(0.0, at - sim.t)
            model.slots = slots

    def timeline():
        while True:
            recent = sorted(a for t, a in recent_answers if t >= sim.t - 30)
            p90 = recent[int(0.9 * (len(recent) - 1))] if recent else 0.0
            st.timeline.append((round(sim.t), callers_wanted(sc, sim.t), sum(w.active for w in workers), len(ready()), round(p90, 2)))
            yield 30.0

    recent_answers: list[tuple[float, float]] = []

    def turn(w: Worker):
        t0 = sim.t
        rng = sim.rng
        cpu = max(1.0, w.active / hw.worker_cpu_calls) ** 3      # a worker past what its CPU keeps up with: everything on it lags
        lag = 0.04 * (cpu - 1)
        yield hw.endpoint_s * math.exp(rng.gauss(0, 0.1)) + lag
        lat = stt.begin()
        yield lat
        stt.end()
        heard = sim.t - t0
        bridged = False
        deadline = 1.2 if sc.adaptive else 5.0
        if rng.random() < hw.code_share:
            st.levels["code-routed"] += 1 if t0 >= sc.measure_from_s else 0
            yield 0.02
        elif not w.governor.model_allowed():                       # the model is overwhelmed: shed, the code decides and speaks at once
            if t0 >= sc.measure_from_s:
                st.levels["code"] += 1
                st.shed += 1
            yield 0.02
        else:
            wait = w.governor.expected_wait_ms()
            bridged = wait is not None and wait >= 450 and rng.random() < 0.7      # "Let me check." (lookup-like requests only)
            lat = model.begin(hw.decide_ms)
            yield min(lat, deadline)                               # past the deadline the request is dropped (the server stops it)
            model.end()
            w.governor.record_decide(min(lat, deadline) * 1000)
            level = "code" if lat > deadline else w.governor.choose()   # too slow: the code's own decision and wording
            if t0 >= sc.measure_from_s:
                st.levels[level] += 1
            if level == "full":
                lat = model.begin(hw.first_ms)
                yield lat
                model.end()
                w.governor.record_full(lat * 1000)
            else:
                yield 0.01                                         # the code's facts, at once
                if level == "prefix":
                    sim.spawn(continuation())                       # the model adds the question while the facts are spoken
        yield hw.tts_ms / 1000 * cpu
        answer = sim.t - t0
        first = min(answer, heard + 0.03 + lag) if bridged else answer
        if w.admission is not None:
            w.admission.record(answer)
        recent_answers.append((sim.t, answer))
        if len(recent_answers) > 20000:
            del recent_answers[:10000]
        if t0 >= sc.measure_from_s:
            st.first.append(first)
            st.answer.append(answer)
            st.bridged += bridged

    def continuation():
        lat = model.begin(hw.first_ms * 0.6)
        yield lat
        model.end()

    def caller(slot: int):
        rng = sim.rng
        yield rng.uniform(0, 30)
        while True:
            if slot >= callers_wanted(sc, sim.t):
                yield 5.0
                continue
            pick = router.pick()
            w = next((x for x in workers if pick and x.id == pick["id"]), None)
            if w is None or not w.accepts():                      # every worker full, or this one has just filled: "busy" -> text chat
                refused_at.append(sim.t)
                if len(refused_at) > 5000:
                    del refused_at[:2500]
                if sim.t >= sc.measure_from_s:
                    st.refused += 1
                yield rng.uniform(20, 40)                          # this person goes to text chat; someone else calls later
                continue
            w.active += 1
            if sim.t >= sc.measure_from_s:
                st.calls += 1
            yield rng.uniform(4.0, 7.0)                            # the greeting
            for _ in range(rng.randint(6, 12)):
                yield rng.uniform(1.5, 4.0)                        # the caller speaks
                yield from turn(w)
                yield rng.uniform(2.5, 6.0)                        # the agent speaks
            w.active -= 1
            if w.draining and w.active == 0:
                workers.remove(w)
            yield rng.uniform(1, 5)

    sim.spawn(beats())
    sim.spawn(autoscaler())
    sim.spawn(gpu_events())
    sim.spawn(timeline())
    for slot in range(max(k for _, k in sc.callers)):
        sim.spawn(caller(slot))
    sim.run(sc.duration_s)
    return summarize(sc, st, model)


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(sc: Scenario, st: Stats, model: Shared) -> dict:
    model_turns = st.levels["full"] + st.levels["prefix"] + st.levels["code"]
    turns = model_turns + st.levels["code-routed"]
    return {
        "scenario": sc.name, "protection": "on" if sc.adaptive else "off",
        "shed %": round(100 * st.shed / max(1, model_turns)),
        "calls": st.calls, "busy %": round(100 * st.refused / max(1, st.calls + st.refused), 1),
        "first p50": round(pct(st.first, 0.5), 2), "first p90": round(pct(st.first, 0.9), 2),
        "answer p50": round(pct(st.answer, 0.5), 2), "answer p90": round(pct(st.answer, 0.9), 2), "answer p99": round(pct(st.answer, 0.99), 2),
        "answers > 1 s %": round(100 * sum(a > 1.0 for a in st.answer) / max(1, len(st.answer)), 1),
        "answers > 2 s %": round(100 * sum(a > 2.0 for a in st.answer) / max(1, len(st.answer)), 1),
        "by code %": round(100 * st.levels["code-routed"] / max(1, turns)),
        "model full/prefix/code %": "/".join(str(round(100 * st.levels[k] / max(1, model_turns))) for k in ("full", "prefix", "code")),
        "turns": turns, "model peak in flight": model.peak,
        "workers max": max((r[3] for r in st.timeline), default=0), "timeline": st.timeline,
        "answer mean": round(statistics.fmean(st.answer), 2) if st.answer else 0.0,
    }


# ------------------------------------------------------------------------------------------------ the scenarios
FAST = Hardware("fast GPUs", model_slots=160, decide_ms=120, first_ms=300)
SLOW = Hardware("slow GPU", model_slots=24, decide_ms=300, first_ms=900)
WEAK_CPU = Hardware("fast GPUs, weak worker CPUs", model_slots=160, decide_ms=120, first_ms=300, worker_cpu_calls=9)

RAMP_500 = [(0, 100), (30, 250), (60, 400), (90, 500)]


def scenarios(adaptive: bool) -> list[Scenario]:
    """Each scenario with protection on (measured cap, shedding, deadline, autoscaling on demand) or off (the code before them)."""
    return [sc for sc in _scenarios(adaptive) if not setattr(sc, "scale_on_demand", adaptive)]


def _scenarios(adaptive: bool) -> list[Scenario]:
    return [
        Scenario("500 callers, fast GPUs", FAST, RAMP_500, adaptive=adaptive),
        Scenario("500 callers, slow GPU", SLOW, RAMP_500, adaptive=adaptive),
        Scenario("500 callers, weak worker CPUs", WEAK_CPU, RAMP_500, adaptive=adaptive),
        Scenario("500 callers, 95% of GPU capacity lost at 5 min", FAST, RAMP_500, model_slots_at=[(300, 6)], adaptive=adaptive),
        Scenario("spike: 500 -> 800 callers at 5 min", FAST, RAMP_500 + [(300, 800), (420, 500)], adaptive=adaptive),
        Scenario("cold start: 8 workers when 500 callers arrive", FAST, RAMP_500, cold_start=True, adaptive=adaptive),
    ]
