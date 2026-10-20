import pytest

from dealer_core.config import BackendCfg
from dealer_core.dialogue.states import STATE_SPEC, Intent, State
from dealer_core.nlu import prompts
from dealer_core.nlu.backend import BackendError, BackendHealth, BackendTimeout, IntentResult, NluBackend, resolve_prompt_mode
from dealer_core.nlu.chain import BackendChain


class Fake(NluBackend):
    def __init__(self, name, healthy=True, fail=False):
        self.name, self.healthy, self.fail, self.calls = name, healthy, fail, 0

    def health(self):
        return BackendHealth(self.healthy, "fake")

    def classify(self, state, car, text):
        self.calls += 1
        if self.fail:
            raise BackendError("boom")
        return IntentResult(Intent.GREET, 1.0, self.name, "{}")

    def complete_json(self, state, car, text):
        return {"intent": "greet", "slots": {}}

    def answer(self, question, context, max_sentences=2):
        yield "ok."


def test_start_picks_first_healthy_and_reports_skips():
    chain = BackendChain([Fake("a", healthy=False), Fake("b"), Fake("c")])
    msg = chain.start()
    assert chain.name == "b" and "skipped" in msg and "a" in msg


def test_start_raises_when_nothing_healthy():
    with pytest.raises(BackendError, match="no healthy"):
        BackendChain([Fake("a", healthy=False)]).start()


def test_call_failure_falls_through_and_sticks():
    a, b = Fake("a", fail=True), Fake("b")
    chain = BackendChain([a, b])
    assert chain.classify(State.GREETING, None, "hi").backend == "b"
    chain.classify(State.GREETING, None, "hi")
    assert a.calls == 1 and chain.name == "b"        # does not retry the broken backend on every turn


def test_all_failing_raises():
    with pytest.raises(BackendError, match="all NLU backends failed"):
        BackendChain([Fake("a", fail=True), Fake("b", fail=True)]).classify(State.GREETING, None, "x")


@pytest.mark.parametrize("runtime,device,expected", [("openvino", "GPU", "state"), ("openvino", "CPU", "static"),
                                                      ("ollama", None, "static")])
def test_auto_prompt_mode(runtime, device, expected):
    assert resolve_prompt_mode(BackendCfg(runtime=runtime, model="m", device=device)) == expected


def test_explicit_prompt_mode_wins():
    assert resolve_prompt_mode(BackendCfg(runtime="openvino", model="m", device="GPU", prompt_mode="static")) == "static"


def test_grammar_only_allows_state_valid_intents():
    for state, (intents, _) in STATE_SPEC.items():
        enum = prompts.intent_only_schema(state)["properties"]["intent"]["enum"]
        assert enum == [i.value for i in intents]
    assert "ask_car_info" not in prompts.intent_only_schema(State.INTEREST)["properties"]["intent"]["enum"]
    assert "ask_car_info" in prompts.intent_only_schema(State.ANSWERING)["properties"]["intent"]["enum"]


def test_static_prompt_is_identical_for_every_state():
    """The point of the static prompt is that the KV-cache prefix never changes between states."""
    assert prompts.system_prompt(State.GREETING, "static") == prompts.system_prompt(State.CONFIRM, "static")
    assert prompts.system_prompt(State.GREETING, "state") != prompts.system_prompt(State.CONFIRM, "state")


class SlowOnce(Fake):
    """Times out on its first call, fine afterwards (a cold start)."""

    def classify(self, state, car, text):
        self.calls += 1
        if self.calls == 1:
            raise BackendTimeout("cold")
        return IntentResult(Intent.GREET, 1.0, self.name, "{}")


def test_a_timeout_uses_the_fallback_for_that_call_only_and_keeps_the_primary():
    a, b = SlowOnce("a"), Fake("b")
    chain = BackendChain([a, b])
    assert chain.classify(State.GREETING, None, "x").backend == "b"       # a timed out: b answered this one call
    assert chain.name == "a"                                             # ...but a is still the active backend
    assert chain.classify(State.GREETING, None, "x").backend == "a"       # and answers the next call


def test_start_warms_only_the_active_backend():
    warmed = []

    class W(Fake):
        def warm_up(self):
            warmed.append(self.name)

    BackendChain([W("a"), W("b")]).start()
    assert warmed == ["a"]
    warmed.clear()
    BackendChain([W("a")]).start(warm=False)
    assert warmed == []
