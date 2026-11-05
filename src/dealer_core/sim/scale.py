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
    # The measured stack (ARCH_REVIEW sections 101 and 102): servers that do ONE request at a time per copy (our recogniser and voice server), what a turn asks of them, and
    # the reply written ahead. All off (0) by default: the scenarios above are unchanged.
    decide_share: float = 1.0     # of the turns that use the model, those that need it for the decision too (measured: 0.2; the code's rules decide the rest)
    ahead_share: float = 0.0      # replies already being written when the words arrive (measured: 0.72 to 0.85) ...
    ahead_s: float = 0.0          # ... and how long before (measured: 0.2 s)
    stt_copies: int = 0           # recogniser copies, each one clip at a time (0: the shared server above)
    stt_service_ms: float = 167.0  # one clip (measured: Qwen3 6.0 a second, Parakeet 23)
    stt_clips: float = 2.7        # clips a turn: the early readings and the final one (measured 2.5 to 2.85)
    tts_copies: int = 0           # voice server copies, each one sentence at a time (0: the voice above, unshared)
    tts_first_ms: float = 70.0    # the first byte of a sentence from an idle server (measured)
    tts_first_service_ms: float = 170.0   # the server's time for the first clause of a reply ...
    tts_rest_service_ms: float = 620.0    # ... and for the rest of it (5.5 s of speech a turn at 7 s of speech a second: 0.79 s)
    tts_timeout_s: float = 2.0    # a voice server that has not answered by then is given up on (voice/remote_tts.py first_timeout_s)
    tts_fallback_ms: float = 300.0  # the fallback voice's first audio (Kokoro, on the worker)
    extra_model_requests: float = 0.0   # model requests a turn makes that nobody waits for (a reply written ahead for a lookup that was not picked, one written during a hold): measured 1.0 of the 2.8 to 3.3 a model-decides turn makes


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


class Serial:
    """A server whose every copy does one request at a time (our Qwen3 recogniser, our Pocket voice server): a request goes to the copy that is free soonest and waits for its
    turn. MEASURED the same way (section 101): the wait is the number ahead of it times the service time."""

    def __init__(self, sim: Sim, copies: int) -> None:
        self.sim, self.free = sim, [0.0] * max(1, copies)
        self.requests = 0
        self.busy_s = 0.0

    def ask(self, service_s: float) -> tuple[int, float]:
        """(the copy, the time before this request starts)."""
        service_s *= math.exp(self.sim.rng.gauss(0, 0.2))
        i = min(range(len(self.free)), key=self.free.__getitem__)
        start = max(self.sim.t, self.free[i])
        self.free[i] = start + service_s
        self.requests += 1
        self.busy_s += service_s
        self._last = service_s
        return i, start - self.sim.t

    def give_up(self, copy: int) -> None:
        """The caller closed the request (the voice's 2 s timeout): the copy does not make that speech."""
        self.free[copy] = max(self.sim.t, self.free[copy] - self._last)
        self.busy_s -= self._last


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
    probes: int = 0                                        # background probe requests sent to the model server
    probes_overloaded: int = 0                             # ...of which sent while the model server already had more requests than slots
    prefix_idle: int = 0                                   # model turns spoken at the prefix level while the model server was less than half busy
    stuck: int = 0                                         # samples (every 5 s) of a worker with calls whose governor is above its budget although the model server is less than half busy
    worker_samples: int = 0
    voice_sentences: int = 0                               # first sentences asked of the voice server ...
    voice_fallbacks: int = 0                               # ... and the ones its 2 s timeout gave to the fallback voice
    timeline: list[tuple] = field(default_factory=list)    # (t, callers wanting a call, in calls, workers ready, answer p90 last 30 s, workers above their budget, model load 0-1+)


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
    model_only: bool = False                               # the boss's rule: the model decides and writes EVERY turn, no code or prefix answers, nothing shed, no deadline after which the code answers.
                                                           # The governor only observes; what protects callers is the call cap (callers turned away) and more model capacity.
    ceiling: int = 16
    min_workers: int = 36                                  # what is running when the calls start (deploy/k8s: 36 replicas)
    max_workers: int = 72                                  # deploy/k8s: maxReplicas
    worker_start_s: float = 45.0                           # a new worker: scheduled, started, speech models loaded
    target_s: float = 1.0
    scale_on_demand: bool = True                           # the autoscaler also counts callers just turned away (not only those in calls)
    probe_every_s: float = 20.0                            # dialogue.probe_every_s
    probe_max_in_flight: int = 0                           # dialogue.probe_max_in_flight (0: only when no turn at all is using the model)
    probe_skip_fresh: bool = True                          # skip a round when a caller's model-written turn timed the model within the interval (B5; False = the rule before it)
    think_s: tuple[float, float] = (0.0, 0.0)              # the caller's pause after a reply, before the next question (the recorded calls: a turn every 15 s in all)
    agent_speaks_s: tuple[float, float] = (2.5, 6.0)       # how long a reply is spoken (measured, section 102: 5 s at the median, 6.4 on average)
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
    stt_serial = Serial(sim, hw.stt_copies) if hw.stt_copies else None
    tts_serial = Serial(sim, hw.tts_copies) if hw.tts_copies else None
    router = Assigner(None, rng=random.Random(sc.seed), clock=sim.now)
    workers: list[Worker] = []
    st = Stats()

    def new_worker(start_delay: float) -> Worker:
        gov = QualityGovernor("observe" if sc.model_only else "auto", budget_ms=700, shed_after_ms=450 if sc.adaptive and not sc.model_only else 0, rng=random.Random(sim.rng.random()), clock=sim.now)
        gov.probed = True                                   # the voice server always runs the probe (server/app.py): an unmeasured model is played safe
        w = Worker(f"w{len(workers) + 1}", sc.ceiling, gov,
                   AdmissionController(sc.ceiling, target_s=sc.target_s, clock=sim.now) if sc.adaptive else None, sim.t + start_delay)
        workers.append(w)
        sim.spawn(prober(w), start_delay)
        return w

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

    def probe_once(w: Worker):
        st.probes += 1
        st.probes_overloaded += model.in_flight >= model.slots
        lat = model.begin(hw.first_ms)                      # the probe has the shape of a model-written turn: it costs what one costs
        yield lat
        model.end()
        w.governor.record_probe(lat * 1000)

    def prober(w: Worker):
        """server/app.py probe_loop: a start-up probe, then every `probe_every_s` at a moment the rule allows (looked for for 10 s, 1 s apart)."""
        yield from probe_once(w)
        while True:
            yield sc.probe_every_s
            if w not in workers:
                return
            if sc.probe_skip_fresh and w.governor.measured_within(sc.probe_every_s):
                continue
            for _ in range(10):
                if w.governor.snapshot()["in_flight"] <= sc.probe_max_in_flight:
                    yield from probe_once(w)
                    break
                yield 1.0

    def sampler():
        """Every 5 s: of the workers with calls, how many are held below the full level by an old measure while the model server is not busy."""
        while True:
            yield 5.0
            if sim.t < sc.measure_from_s or model.in_flight / model.slots >= 0.5:
                continue
            for w in ready():
                if w.active:
                    st.worker_samples += 1
                    st.stuck += (w.governor.stats.full_first_ms or 0) > w.governor.budget_ms

    def gpu_events():
        for at, slots in sorted(sc.model_slots_at):
            yield max(0.0, at - sim.t)
            model.slots = slots

    def timeline():
        while True:
            recent = sorted(a for t, a in recent_answers if t >= sim.t - 30)
            p90 = recent[int(0.9 * (len(recent) - 1))] if recent else 0.0
            down = sum(1 for w in ready() if (w.governor.stats.full_first_ms or 0) > w.governor.budget_ms)
            st.timeline.append((round(sim.t), callers_wanted(sc, sim.t), sum(w.active for w in workers), len(ready()), round(p90, 2), down, round(model.in_flight / model.slots, 2)))
            yield 30.0

    recent_answers: list[tuple[float, float]] = []

    def turn(w: Worker):
        t0 = sim.t
        rng = sim.rng
        cpu = max(1.0, w.active / hw.worker_cpu_calls) ** 3      # a worker past what its CPU keeps up with: everything on it lags
        lag = 0.04 * (cpu - 1)
        yield hw.endpoint_s * math.exp(rng.gauss(0, 0.1)) + lag
        if stt_serial is not None:                              # the final clip waits behind the early readings of every caller
            _, wait = stt_serial.ask(hw.stt_service_ms / 1000)
            yield wait + hw.stt_service_ms / 1000
        else:
            lat = stt.begin()
            yield lat
            stt.end()
        heard = sim.t - t0
        bridged = False
        deadline = 1e9 if sc.model_only else 1.2 if sc.adaptive else 5.0         # model_only: no request is dropped for the code to answer; the caller waits for the model
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
            bridged = wait is not None and wait >= 450 and rng.random() < 0.7 and not sc.model_only      # (model_only: the engine says no "Let me check.")      # "Let me check." (lookup-like requests only)
            if rng.random() < hw.decide_share:
                lat = model.begin(hw.decide_ms)
                w.governor.stats.in_flight += 1                    # engine: `with governor.using_model()` around the decision and the writing
                yield min(lat, deadline)                           # past the deadline the request is dropped (the server stops it)
                model.end()
                w.governor.stats.in_flight -= 1
                w.governor.record_decide(min(lat, deadline) * 1000)
            else:
                lat = 0.0                                          # the code's rules chose the look-up: no decision call
            level = "code" if lat > deadline else w.governor.choose()   # too slow: the code's own decision and wording
            if t0 >= sc.measure_from_s:
                st.levels[level] += 1
                st.prefix_idle += level == "prefix" and model.in_flight / model.slots < 0.5
            if hw.extra_model_requests and level == "full":
                for _ in range(int(hw.extra_model_requests) + (rng.random() < hw.extra_model_requests % 1)):
                    sim.spawn(continuation(w, 1.0))                 # work the model does that nobody waits for: it still takes a place in the queue
            if level == "full":
                lat = model.begin(hw.first_ms)
                w.governor.stats.in_flight += 1
                ahead = rng.random() < hw.ahead_share              # the request went out during the pause: part of the wait is already over
                yield max(0.0, lat - hw.ahead_s) if ahead else lat
                model.end()
                w.governor.stats.in_flight -= 1
                w.governor.record_full(lat * 1000)
            else:
                yield 0.01                                         # the code's facts, at once
                if level == "prefix":
                    sim.spawn(continuation(w))                      # the model adds the question while the facts are spoken
        if tts_serial is not None:
            copy, wait = tts_serial.ask(hw.tts_first_service_ms / 1000)
            delay = wait + hw.tts_first_ms / 1000
            if t0 >= sc.measure_from_s:
                st.voice_sentences += 1
            if delay > hw.tts_timeout_s:                            # no first byte in time: the fallback voice says it, on the worker's CPU
                tts_serial.give_up(copy)
                delay = hw.tts_timeout_s + hw.tts_fallback_ms / 1000 * cpu
                if t0 >= sc.measure_from_s:
                    st.voice_fallbacks += 1
            else:
                sim.spawn(voice_rest(), hw.tts_first_service_ms / 1000)
            yield delay
        else:
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

    def early_clip():
        stt_serial.ask(hw.stt_service_ms / 1000)
        yield 0.0

    def voice_rest():                                       # the rest of the reply's speech: not waited for, but it holds a voice server copy
        tts_serial.ask(hw.tts_rest_service_ms / 1000)
        yield 0.0

    def continuation(w: Worker, share: float = 0.6):
        lat = model.begin(hw.first_ms * share)
        w.governor.stats.in_flight += 1
        yield lat
        model.end()
        w.governor.stats.in_flight -= 1

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
                speaking = rng.uniform(1.5, 4.0)
                if stt_serial is not None:                          # the early start's readings, while the caller is still speaking or pausing
                    for _k in range(max(0, round(hw.stt_clips - 1))):
                        sim.spawn(early_clip(), rng.uniform(0.2, speaking + 0.2))
                yield speaking                                      # the caller speaks
                yield from turn(w)
                yield rng.uniform(*sc.agent_speaks_s)              # the agent speaks
                if sc.think_s != (0.0, 0.0):
                    yield rng.uniform(*sc.think_s)                 # the caller thinks
            w.active -= 1
            if w.draining and w.active == 0:
                workers.remove(w)
            yield rng.uniform(1, 5)

    for _ in range(sc.cold_min if sc.cold_start else sc.min_workers):
        new_worker(0.0)                                     # (after the probe is defined: each worker starts one)
    sim.spawn(beats())
    sim.spawn(autoscaler())
    sim.spawn(gpu_events())
    sim.spawn(timeline())
    sim.spawn(sampler())
    for slot in range(max(k for _, k in sc.callers)):
        sim.spawn(caller(slot))
    sim.run(sc.duration_s)
    out = summarize(sc, st, model)
    window = max(1.0, sc.duration_s)
    out["recogniser busy %"] = round(100 * stt_serial.busy_s / (len(stt_serial.free) * window)) if stt_serial else None
    out["voice server busy %"] = round(100 * tts_serial.busy_s / (len(tts_serial.free) * window)) if tts_serial else None
    out["fallback voice %"] = round(100 * st.voice_fallbacks / max(1, st.voice_sentences)) if tts_serial else None
    return out


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def seconds_to_recover(sc: Scenario, st: Stats) -> float | None:
    """After the last time model capacity was restored: seconds until no worker was above its model budget (30 s resolution). None if capacity never came back."""
    restores, last = [], None
    for at, slots in sorted(sc.model_slots_at):
        if last is not None and slots > last:
            restores.append(at)
        last = slots
    if not restores:
        return None
    t0 = restores[-1]
    return next((row[0] - t0 for row in st.timeline if row[0] >= t0 and row[5] == 0), None)


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
        "prefix while model idle %": round(100 * st.prefix_idle / max(1, model_turns), 1),
        "workers held down while model idle %": round(100 * st.stuck / max(1, st.worker_samples), 1), "probe requests": st.probes, "probes while model overloaded": st.probes_overloaded,
        "seconds to recover": seconds_to_recover(sc, st),
        "workers max": max((r[3] for r in st.timeline), default=0), "timeline": st.timeline,
        "answer mean": round(statistics.fmean(st.answer), 2) if st.answer else 0.0,
    }


# ------------------------------------------------------------------------------------------------ the scenarios
FAST = Hardware("fast GPUs", model_slots=160, decide_ms=120, first_ms=300)
BUSY = Hardware("every turn on the model", model_slots=160, decide_ms=300, first_ms=500, code_share=0.0)     # the manifest's 10-20% of callers waiting on the model at any moment
SLOW = Hardware("slow GPU", model_slots=24, decide_ms=300, first_ms=900)
WEAK_CPU = Hardware("fast GPUs, weak worker CPUs", model_slots=160, decide_ms=120, first_ms=300, worker_cpu_calls=9)

RAMP_500 = [(0, 100), (30, 250), (60, 400), (90, 500)]
RAMP_1000 = [(0, 200), (30, 500), (60, 800), (90, 1000)]
RAMP_1500 = [(0, 300), (30, 750), (60, 1200), (90, 1500)]


def model_capacity_sweep(callers: int, slots: list[int], max_workers: int) -> list[dict]:
    """How much model capacity a given number of callers needs, with the stand-in speeds of FAST (decision 120 ms, first sentence 300 ms): the same cluster run with
    fewer and fewer `slots` (requests the model server(s) serve together at full speed) until answers slow down or the mechanisms start to give up quality."""
    ramp = {500: RAMP_500, 1000: RAMP_1000, 1500: RAMP_1500}[callers]
    out = []
    for n in slots:
        hw = Hardware(f"{n} slots", model_slots=n, decide_ms=FAST.decide_ms, first_ms=FAST.first_ms, stt_slots=max(64, callers // 4))
        out.append({"model slots": n, **{k: v for k, v in run(Scenario(f"{callers} callers, {n} slots", hw, ramp, max_workers=max_workers)).items() if k != "timeline"}})
    return out


def measured_hardware(name: str, voice_copies: int, stt_copies: int, stt_service_ms: float = 167.0, model_slots: int = 64) -> Hardware:
    """The stack as measured on 3 November (ARCH_REVIEW sections 98 to 102): the model writes every reply and is asked for the decision in 1 turn in 5, the reply is already
    being written in 3 turns in 4, the recogniser and the voice server do one request at a time per copy. `model_slots`: the model server was measured linear to 48 requests at once."""
    return Hardware(name, model_slots=model_slots, decide_ms=230, first_ms=650, code_share=0.0, decide_share=0.2, ahead_share=0.78, ahead_s=0.22, endpoint_s=0.24,
                    stt_copies=stt_copies, stt_service_ms=stt_service_ms, stt_clips=2.7, tts_copies=voice_copies, tts_first_ms=70.0)


def measured_scenarios(adaptive: bool = True) -> list[Scenario]:
    """Calibration first: the 8 and 16 caller runs of section 98 (one worker, one recogniser, one voice server, the call cap off), then 100 and 200 callers, then 1,000 (4 Nov)."""
    rhythm = {"think_s": (3.0, 6.0), "agent_speaks_s": (4.0, 9.0)}                     # a real caller: a turn about every 15 s
    as_tested = {"think_s": (0.2, 0.6), "agent_speaks_s": (4.0, 9.0)}                  # the load test's callers asked again at once: a turn every 11 or 12 s
    one = measured_hardware("1 worker, 1 voice server, 1 Qwen3", 1, 1)
    ramp100 = [(0, 25), (30, 50), (60, 80), (90, 100)]
    ramp200 = [(0, 50), (30, 100), (60, 150), (90, 200)]
    ramp1000 = [(0, 250), (30, 500), (60, 750), (90, 1000)]
    return [
        Scenario("calibration: 8 callers, as the laptop", one, [(0, 8)], adaptive=False, min_workers=1, max_workers=1, duration_s=400, measure_from_s=60, **as_tested),
        Scenario("calibration: 16 callers, as the laptop", one, [(0, 16)], adaptive=False, min_workers=1, max_workers=1, duration_s=400, measure_from_s=60, **as_tested),
        Scenario("calibration: 16 callers, as the laptop, real-call rhythm", one, [(0, 16)], adaptive=False, min_workers=1, max_workers=1, duration_s=400, measure_from_s=60, **rhythm),
        Scenario("100 callers, 1 voice server, 1 Qwen3 (as built)", one, ramp100, adaptive=adaptive, min_workers=8, **rhythm),
        Scenario("100 callers, 4 voice servers, 2 Qwen3", measured_hardware("4 voice, 2 Qwen3", 4, 2), ramp100, adaptive=adaptive, min_workers=8, **rhythm),
        Scenario("100 callers, 8 voice servers, 5 Qwen3 (sized)", measured_hardware("8 voice, 5 Qwen3", 8, 5), ramp100, adaptive=adaptive, min_workers=8, **rhythm),
        Scenario("100 callers, 8 voice servers, 2 Parakeet", measured_hardware("8 voice, 2 Parakeet", 8, 2, stt_service_ms=43.0), ramp100, adaptive=adaptive, min_workers=8, **rhythm),
        Scenario("200 callers, 8 voice servers, 5 Qwen3 (half of what it needs)", measured_hardware("8 voice, 5 Qwen3", 8, 5), ramp200, adaptive=adaptive, min_workers=8, **rhythm),
        Scenario("200 callers, 16 voice servers, 9 Qwen3 (sized)", measured_hardware("16 voice, 9 Qwen3", 16, 9), ramp200, adaptive=adaptive, min_workers=8, **rhythm),
        Scenario("200 callers, 16 voice servers, 3 Parakeet", measured_hardware("16 voice, 3 Parakeet", 16, 3, stt_service_ms=43.0), ramp200, adaptive=adaptive, min_workers=8, **rhythm),
        Scenario("200 callers, 20 voice servers, 9 Qwen3, a surge to 300 at 5 min", measured_hardware("20 voice, 9 Qwen3", 20, 9), ramp200 + [(300, 300), (420, 200)], adaptive=adaptive, min_workers=8, **rhythm),
        Scenario("1000 callers, 80 voice servers, 45 Qwen3 (sized)", measured_hardware("80 voice, 45 Qwen3", 80, 45), ramp1000, adaptive=adaptive, min_workers=8, max_workers=72, **rhythm),
        Scenario("1000 callers, 80 voice servers, 15 Parakeet", measured_hardware("80 voice, 15 Parakeet", 80, 15, stt_service_ms=43.0), ramp1000, adaptive=adaptive, min_workers=8, max_workers=72, **rhythm),
        Scenario("1000 callers, 40 voice servers, 22 Qwen3 (half of what it needs)", measured_hardware("40 voice, 22 Qwen3", 40, 22), ramp1000, adaptive=adaptive, min_workers=8, max_workers=72, **rhythm),
    ]


def model_only_hardware(name: str, voice_copies: int, stt_copies: int, model_slots: int, stt_service_ms: float = 167.0) -> Hardware:
    """measured_hardware, but with the model deciding every turn (not 1 in 5) and one more request a turn that nobody waits for (ARCH_REVIEW section 108: 2.8 to 3.3 requests a turn, 1.25 s median)."""
    from dataclasses import replace

    return replace(measured_hardware(name, voice_copies, stt_copies, stt_service_ms, model_slots), decide_share=1.0, extra_model_requests=1.0, ahead_share=0.5)


def model_only_scenarios() -> list[Scenario]:
    """1,000 callers with the boss's rule, at 64 model requests at once per GPU (assumed): how many GPUs does it take, and what happens with fewer."""
    rhythm = {"think_s": (3.0, 6.0), "agent_speaks_s": (4.0, 9.0)}
    ramp1000 = [(0, 250), (30, 500), (60, 750), (90, 1000)]
    out = []
    for gpus in (1, 2, 3, 4, 8):
        hw = model_only_hardware(f"model-only, {gpus} GPUs", 80, 45, model_slots=64 * gpus)
        out.append(Scenario(f"1000 callers, model-only, {gpus} GPUs ({64 * gpus} slots), cap target 2 s", hw, ramp1000, adaptive=True, model_only=True, target_s=2.0, min_workers=8, max_workers=72, **rhythm))
    hw = model_only_hardware("model-only, 4 GPUs", 80, 45, model_slots=256)
    out.append(Scenario("1000 callers, model-only, 4 GPUs (256 slots), cap target 1 s (as built)", hw, ramp1000, adaptive=True, model_only=True, target_s=1.0, min_workers=8, max_workers=72, **rhythm))
    return out


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
        Scenario("1,000 callers, fast GPUs, the manifest's 72 workers", FAST, RAMP_1000, adaptive=adaptive),
        Scenario("1,000 callers, fast GPUs, 120 workers allowed", FAST, RAMP_1000, max_workers=120, adaptive=adaptive),
        Scenario("1,500 callers, fast GPUs, 180 workers allowed", FAST, RAMP_1500, max_workers=180, adaptive=adaptive),
        Scenario("500 callers, GPU capacity lost at 5 min, back at 7 min", FAST, RAMP_500, model_slots_at=[(300, 6), (420, 160)], adaptive=adaptive),
        Scenario("500 callers, every turn on the model, GPU lost at 5 min, back at 7", BUSY, RAMP_500, model_slots_at=[(300, 6), (420, 160)], adaptive=adaptive),
    ]
